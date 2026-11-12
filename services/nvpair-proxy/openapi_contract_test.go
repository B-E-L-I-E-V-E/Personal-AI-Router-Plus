// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// The OpenAPI reference (fern/openapi.yml) is what client authors build
// against. These checks fail when the API returns or accepts a field the
// reference does not describe, so the two cannot drift apart silently.

func openAPIText(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "fern", "openapi.yml"))
	if err != nil {
		t.Fatalf("read the OpenAPI reference: %v", err)
	}
	return string(b)
}

// section returns the reference text from the first line matching start up
// to the next line at the same or a shallower indent.
func section(t *testing.T, doc, start string) string {
	t.Helper()
	i := strings.Index(doc, "\n"+start+"\n")
	if i < 0 {
		t.Fatalf("the OpenAPI reference has no %q", strings.TrimSpace(start))
	}
	indent := len(start) - len(strings.TrimLeft(start, " "))
	rest := doc[i+len(start)+2:]
	end := regexp.MustCompile(`\n {0,` + strconv.Itoa(indent) + `}[^ \n]`).FindStringIndex(rest)
	if end == nil {
		return rest
	}
	return rest[:end[0]]
}

// jsonFields lists v's JSON field names.
func jsonFields(v any) []string {
	var out []string
	t := reflect.TypeOf(v)
	for i := 0; i < t.NumField(); i++ {
		name := strings.Split(t.Field(i).Tag.Get("json"), ",")[0]
		if name != "" && name != "-" {
			out = append(out, name)
		}
	}
	return out
}

func wantDocumented(t *testing.T, where, text string, fields []string) {
	t.Helper()
	for _, f := range fields {
		if !regexp.MustCompile(`\n\s+` + regexp.QuoteMeta(f) + `:`).MatchString(text) {
			t.Errorf("%s: field %q is not in the OpenAPI reference", where, f)
		}
	}
}

func TestOpenAPI_DocumentsTheCapacityRow(t *testing.T) {
	doc := openAPIText(t)
	wantDocumented(t, "ModelCapacity", section(t, doc, "    ModelCapacity:"), jsonFields(modelCapacity{}))
	wantDocumented(t, "ModelCapacity.configured", section(t, doc, "        configured:"), jsonFields(capacitySetting{}))
}

func TestOpenAPI_DocumentsTheCapacitySetting(t *testing.T) {
	put := section(t, openAPIText(t), "    put:")
	wantDocumented(t, "PUT /pair/v1/capacity", put, jsonFields(capacitySetParams{}))
}

func TestOpenAPI_DocumentsTheSession(t *testing.T) {
	doc := openAPIText(t)
	session := section(t, doc, "    Session:")
	wantDocumented(t, "Session", session, jsonFields(sessionJSON{}))
	wantDocumented(t, "Session.context", section(t, session, "        context:"), jsonFields(sessionContextJSON{}))
}

func TestOpenAPI_DocumentsTheContextHeaders(t *testing.T) {
	doc := openAPIText(t)
	for _, h := range []string{contextTokensHeader, contextLimitHeader, contextReplyHeader, contextMessageHeader, queueWaitHeader} {
		if !strings.Contains(doc, h) {
			t.Errorf("header %s is not in the OpenAPI reference", h)
		}
	}
}

func TestOpenAPI_DocumentsThePerformanceReport(t *testing.T) {
	doc := openAPIText(t)
	cases := []struct {
		schema string
		fields []string
	}{
		{"PerformanceReport", jsonFields(perfReport{})},
		{"PerfModelSummary", jsonFields(perfModelSummary{})},
		{"PerfRecord", jsonFields(perfRecord{})},
		{"PerfStat", jsonFields(perfStat{})},
		{"PerfSources", jsonFields(perfSources{})},
		{"BenchmarkRun", jsonFields(benchmarkRun{})},
		{"BenchmarkStep", jsonFields(benchmarkStep{})},
		{"BenchmarkPrompt", jsonFields(benchmarkPrompt{})},
	}
	for _, tc := range cases {
		t.Run(tc.schema, func(t *testing.T) {
			wantDocumented(t, tc.schema, section(t, doc, "    "+tc.schema+":"), tc.fields)
		})
	}
	wantDocumented(t, "PerfModelSummary.trend", section(t, section(t, doc, "    PerfModelSummary:"), "        trend:"), jsonFields(perfTrendPoint{}))
}
