// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// LM Studio decides whether a reasoning model thinks from a per-model custom
// field, not from the request: reasoning_effort, enable_thinking and
// chat_template_kwargs in a chat request leave the reply unchanged. A model
// declares the field in its hub model.yaml (a boolean custom field whose effect
// sets the enable_thinking template variable), and the value the person chose
// lives in the model's default configuration file:
//
//	~/.lmstudio/.internal/user-concrete-model-default-config/<model>.json
//
// under operation.fields as ext.virtualModel.customField.<owner>.<name>.<field>.
// LM Studio reads it on the next request, with no reload. This file edits only
// that one entry and keeps every other field, key and value as it found them.

var thinkingModelRe = regexp.MustCompile(`^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?$`)

// thinkingState is what PAIR reports for one model's thinking setting.
type thinkingState struct {
	// Supported is false for a model whose hub definition has no thinking field.
	Supported bool `json:"supported"`
	// Reason says why it is not supported.
	Reason string `json:"reason,omitempty"`
	// Default is what the model does when nothing is set.
	Default bool `json:"default"`
	// Value is the setting in force: the person's choice, else the default.
	Value bool `json:"value"`
	// Explicit is true when the person has set it, false when the model default applies.
	Explicit bool `json:"explicit"`
}

func lmStudioHome() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".lmstudio"), nil
}

func validThinkingModel(model string) error {
	if !thinkingModelRe.MatchString(model) || strings.Contains(model, "..") {
		return errors.New("model is not a plain LM Studio model key")
	}
	return nil
}

// thinkingFieldOf reads a model.yaml and returns the custom field that sets the
// enable_thinking template variable, and its default.
func thinkingFieldOf(modelYAML []byte) (field string, def bool, ok bool) {
	var (
		inFields bool
		current  string
		curDef   = true
		isThink  bool
	)
	keyRe := regexp.MustCompile(`^\s+-\s+key:\s*(\S+)\s*$`)
	flush := func() {
		if isThink && current != "" && !ok {
			field, def, ok = current, curDef, true
		}
	}
	sc := bufio.NewScanner(bytes.NewReader(modelYAML))
	for sc.Scan() {
		line := sc.Text()
		trimmed := strings.TrimSpace(line)
		if !inFields {
			inFields = strings.TrimRight(line, " ") == "customFields:"
			continue
		}
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		if line[0] != ' ' && line[0] != '\t' {
			break // the next top-level key ends the block
		}
		if m := keyRe.FindStringSubmatch(line); m != nil {
			flush()
			current, curDef, isThink = m[1], true, false
			continue
		}
		switch {
		case strings.HasPrefix(trimmed, "defaultValue:"):
			curDef = strings.TrimSpace(strings.TrimPrefix(trimmed, "defaultValue:")) != "false"
		case strings.HasPrefix(trimmed, "variable:"):
			if strings.TrimSpace(strings.TrimPrefix(trimmed, "variable:")) == "enable_thinking" {
				isThink = true
			}
		}
	}
	flush()
	return field, def, ok
}

// thinkingConfigKey is the key LM Studio stores a custom field under:
// ext.virtualModel.customField.<owner>.<name>.<field>, where each part of the
// model key loses its hyphens (qwen/qwen3.8-27b is qwen.qwen3.827b).
func thinkingConfigKey(model, field string) string {
	parts := strings.Split(model, "/")
	for i, p := range parts {
		parts[i] = strings.ReplaceAll(p, "-", "")
	}
	return "ext.virtualModel.customField." + strings.Join(parts, ".") + "." + field
}

type thinkingStore struct {
	home string // the .lmstudio directory
}

func (t thinkingStore) configPath(model string) string {
	return filepath.Join(t.home, ".internal", "user-concrete-model-default-config", filepath.FromSlash(model)+".json")
}

func (t thinkingStore) field(model string) (string, bool, error) {
	if err := validThinkingModel(model); err != nil {
		return "", false, err
	}
	raw, err := os.ReadFile(filepath.Join(t.home, "hub", "models", filepath.FromSlash(model), "model.yaml"))
	if err != nil {
		return "", false, errors.New("LM Studio has no hub definition for this model, so it has no thinking setting")
	}
	field, def, ok := thinkingFieldOf(raw)
	if !ok {
		return "", false, errors.New("this model has no thinking setting in LM Studio")
	}
	return field, def, nil
}

// read reports a model's thinking setting.
func (t thinkingStore) read(model string) (thinkingState, error) {
	field, def, err := t.field(model)
	if err != nil {
		if verr := validThinkingModel(model); verr != nil {
			return thinkingState{}, verr
		}
		return thinkingState{Supported: false, Reason: err.Error()}, nil
	}
	st := thinkingState{Supported: true, Default: def, Value: def}
	cfg, err := t.load(model)
	if err != nil {
		return thinkingState{}, err
	}
	key := thinkingConfigKey(model, field)
	for _, f := range cfg.fields() {
		if f.key == key {
			if v, ok := f.value.(bool); ok {
				st.Value, st.Explicit = v, true
			}
		}
	}
	return st, nil
}

// write sets a model's thinking field, or removes it for the model's default
// when value is nil. Every other entry of the file is kept as it was.
func (t thinkingStore) write(model string, value *bool) (thinkingState, error) {
	field, _, err := t.field(model)
	if err != nil {
		return thinkingState{}, err
	}
	cfg, err := t.load(model)
	if err != nil {
		return thinkingState{}, err
	}
	if err := cfg.setField(thinkingConfigKey(model, field), value); err != nil {
		return thinkingState{}, err
	}
	if err := cfg.save(t.configPath(model)); err != nil {
		return thinkingState{}, err
	}
	return t.read(model)
}

// lmConfig is a model default configuration file with the parts PAIR does not
// own kept as raw JSON.
type lmConfig struct {
	top       map[string]json.RawMessage
	operation map[string]json.RawMessage
	opFields  []json.RawMessage
}

type lmField struct {
	key   string
	value any
}

func (t thinkingStore) load(model string) (*lmConfig, error) {
	cfg := &lmConfig{top: map[string]json.RawMessage{}, operation: map[string]json.RawMessage{}}
	raw, err := os.ReadFile(t.configPath(model))
	if errors.Is(err, os.ErrNotExist) {
		return cfg, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &cfg.top); err != nil {
		return nil, fmt.Errorf("LM Studio's configuration for this model is not valid JSON; PAIR left it alone: %w", err)
	}
	if op, ok := cfg.top["operation"]; ok {
		if err := json.Unmarshal(op, &cfg.operation); err != nil {
			return nil, fmt.Errorf("LM Studio's configuration for this model has an unexpected operation block; PAIR left it alone: %w", err)
		}
	}
	if fields, ok := cfg.operation["fields"]; ok {
		if err := json.Unmarshal(fields, &cfg.opFields); err != nil {
			return nil, fmt.Errorf("LM Studio's configuration for this model has unexpected fields; PAIR left it alone: %w", err)
		}
	}
	return cfg, nil
}

func (c *lmConfig) fields() []lmField {
	var out []lmField
	for _, raw := range c.opFields {
		var f struct {
			Key   string `json:"key"`
			Value any    `json:"value"`
		}
		if json.Unmarshal(raw, &f) == nil {
			out = append(out, lmField{key: f.Key, value: f.Value})
		}
	}
	return out
}

func (c *lmConfig) setField(key string, value *bool) error {
	kept := make([]json.RawMessage, 0, len(c.opFields)+1)
	for _, raw := range c.opFields {
		var f struct {
			Key string `json:"key"`
		}
		if json.Unmarshal(raw, &f) == nil && f.Key == key {
			continue
		}
		kept = append(kept, raw)
	}
	if value != nil {
		entry, err := json.Marshal(map[string]any{"key": key, "value": *value})
		if err != nil {
			return err
		}
		kept = append(kept, entry)
	}
	c.opFields = kept
	return nil
}

func (c *lmConfig) save(path string) error {
	fields := c.opFields
	if fields == nil {
		fields = []json.RawMessage{}
	}
	rawFields, err := json.Marshal(fields)
	if err != nil {
		return err
	}
	c.operation["fields"] = rawFields
	rawOp, err := json.Marshal(c.operation)
	if err != nil {
		return err
	}
	c.top["operation"] = rawOp
	if _, ok := c.top["preset"]; !ok {
		c.top["preset"] = json.RawMessage(`""`)
	}
	var out bytes.Buffer
	body, err := json.Marshal(c.top)
	if err != nil {
		return err
	}
	if err := json.Indent(&out, body, "", "  "); err != nil {
		return err
	}
	out.WriteByte('\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".pair-thinking-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(out.Bytes()); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

type thinkingParams struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
	// Set turns thinking on or off; Clear returns the model to its default.
	// Neither reads the setting.
	Set   *bool `json:"set,omitempty"`
	Clear bool  `json:"clear,omitempty"`
}

// thinking runs capacity/thinking: read, set or clear one LM Studio model's
// thinking setting on this node.
func (p *Proxy) thinking(raw json.RawMessage) (thinkingState, error) {
	var params thinkingParams
	if err := json.Unmarshal(raw, &params); err != nil {
		return thinkingState{}, errors.New(`expected {"engine":"lmstudio","model","set"?,"clear"?}`)
	}
	if params.Engine != "lmstudio" {
		return thinkingState{}, errors.New("only LM Studio has a per-model thinking setting")
	}
	if params.Set != nil && params.Clear {
		return thinkingState{}, errors.New("set and clear cannot be combined")
	}
	home, err := lmStudioHome()
	if err != nil {
		return thinkingState{}, err
	}
	store := thinkingStore{home: home}
	switch {
	case params.Set != nil:
		return store.write(params.Model, params.Set)
	case params.Clear:
		return store.write(params.Model, nil)
	}
	return store.read(params.Model)
}
