// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// The capacity planner reads a model's architecture from its GGUF header: the
// layer count, key/value heads and head sizes decide how much memory each
// token of context costs. The tensor table that follows is read too, for the
// size of each tensor (mixture-of-experts weights can live in system memory);
// tensor data is never read.

const (
	ggufMaxKeys       = 1 << 16
	ggufMaxStringLen  = 1 << 20
	ggufMaxKeptArray  = 4096
	ggufMaxArrayItems = 1 << 24
	ggufMaxTensors    = 1 << 20
	ggufMaxDims       = 8
)

// ggufTensor is one entry of the tensor table: its name, ggml type and
// element count.
type ggufTensor struct {
	Name     string
	Type     uint32
	Elements uint64
}

// ggufValue is one decoded metadata value: an integer, float, bool, string,
// or a slice of integers. Arrays of other kinds, and long arrays, are skipped
// and recorded as nil.
type ggufValue any

type ggufReader struct {
	r   *bufio.Reader
	err error
}

func (g *ggufReader) read(n int) []byte {
	if g.err != nil {
		return nil
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(g.r, buf); err != nil {
		g.err = err
		return nil
	}
	return buf
}

func (g *ggufReader) skip(n uint64) {
	if g.err != nil {
		return
	}
	if _, err := io.CopyN(io.Discard, g.r, int64(n)); err != nil {
		g.err = err
	}
}

func (g *ggufReader) u32() uint32 {
	b := g.read(4)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(b)
}

func (g *ggufReader) u64() uint64 {
	b := g.read(8)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(b)
}

func (g *ggufReader) str() string {
	n := g.u64()
	if g.err != nil {
		return ""
	}
	if n > ggufMaxStringLen {
		g.err = fmt.Errorf("gguf string of %d bytes is too long", n)
		return ""
	}
	return string(g.read(int(n)))
}

// ggufScalarSize is the byte size of each fixed-size GGUF value type.
var ggufScalarSize = map[uint32]uint64{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

const (
	ggufTypeString = 8
	ggufTypeArray  = 9
)

func (g *ggufReader) scalar(t uint32) ggufValue {
	switch t {
	case 0:
		b := g.read(1)
		if b == nil {
			return nil
		}
		return int64(b[0])
	case 1:
		b := g.read(1)
		if b == nil {
			return nil
		}
		return int64(int8(b[0]))
	case 2:
		b := g.read(2)
		if b == nil {
			return nil
		}
		return int64(binary.LittleEndian.Uint16(b))
	case 3:
		b := g.read(2)
		if b == nil {
			return nil
		}
		return int64(int16(binary.LittleEndian.Uint16(b)))
	case 4:
		return int64(g.u32())
	case 5:
		return int64(int32(g.u32()))
	case 6:
		return float64(math.Float32frombits(g.u32()))
	case 7:
		b := g.read(1)
		if b == nil {
			return nil
		}
		return b[0] != 0
	case 10:
		v := g.u64()
		if v > math.MaxInt64 {
			return int64(math.MaxInt64)
		}
		return int64(v)
	case 11:
		return int64(g.u64())
	case 12:
		return math.Float64frombits(g.u64())
	case ggufTypeString:
		return g.str()
	}
	g.err = fmt.Errorf("unknown gguf value type %d", t)
	return nil
}

func (g *ggufReader) value(t uint32) ggufValue {
	if t != ggufTypeArray {
		return g.scalar(t)
	}
	et := g.u32()
	n := g.u64()
	if g.err != nil {
		return nil
	}
	if n > ggufMaxArrayItems {
		g.err = fmt.Errorf("gguf array of %d items is too long", n)
		return nil
	}
	size, fixed := ggufScalarSize[et]
	isInt := fixed && et != 6 && et != 7 && et != 12
	if isInt && n <= ggufMaxKeptArray {
		out := make([]int64, 0, n)
		for i := uint64(0); i < n; i++ {
			v, _ := g.scalar(et).(int64)
			out = append(out, v)
		}
		return out
	}
	if fixed {
		g.skip(size * n)
		return nil
	}
	if et == ggufTypeString {
		for i := uint64(0); i < n && g.err == nil; i++ {
			l := g.u64()
			if l > ggufMaxStringLen {
				g.err = fmt.Errorf("gguf string of %d bytes is too long", l)
				return nil
			}
			g.skip(l)
		}
		return nil
	}
	g.err = fmt.Errorf("unsupported gguf array element type %d", et)
	return nil
}

// readGGUFMetadata returns the metadata keys of a GGUF file (version 2 or 3).
func readGGUFMetadata(r io.Reader) (map[string]ggufValue, error) {
	meta, _, err := readGGUF(r, false)
	return meta, err
}

// readGGUF reads the metadata keys and, when tensors is set, the tensor table.
func readGGUF(r io.Reader, tensors bool) (map[string]ggufValue, []ggufTensor, error) {
	g := &ggufReader{r: bufio.NewReaderSize(r, 1<<16)}
	if magic := g.read(4); g.err == nil && string(magic) != "GGUF" {
		return nil, nil, errors.New("not a GGUF file")
	}
	version := g.u32()
	if g.err == nil && (version < 2 || version > 3) {
		return nil, nil, fmt.Errorf("unsupported GGUF version %d", version)
	}
	tensorCount := g.u64()
	count := g.u64()
	if g.err != nil {
		return nil, nil, g.err
	}
	if count > ggufMaxKeys {
		return nil, nil, fmt.Errorf("gguf header declares %d keys", count)
	}
	meta := make(map[string]ggufValue, count)
	for i := uint64(0); i < count; i++ {
		key := g.str()
		t := g.u32()
		v := g.value(t)
		if g.err != nil {
			return nil, nil, fmt.Errorf("gguf metadata: %w", g.err)
		}
		meta[key] = v
	}
	if !tensors {
		return meta, nil, nil
	}
	if tensorCount > ggufMaxTensors {
		return nil, nil, fmt.Errorf("gguf header declares %d tensors", tensorCount)
	}
	out := make([]ggufTensor, 0, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name := g.str()
		dims := g.u32()
		if g.err == nil && dims > ggufMaxDims {
			return nil, nil, fmt.Errorf("gguf tensor %q has %d dimensions", name, dims)
		}
		elements := uint64(1)
		for d := uint32(0); d < dims; d++ {
			n := g.u64()
			if n != 0 && elements > math.MaxUint64/n {
				return nil, nil, fmt.Errorf("gguf tensor %q is too large", name)
			}
			elements *= n
		}
		typ := g.u32()
		_ = g.u64() // data offset
		if g.err != nil {
			return nil, nil, fmt.Errorf("gguf tensors: %w", g.err)
		}
		out = append(out, ggufTensor{Name: name, Type: typ, Elements: elements})
	}
	return meta, out, nil
}

func readGGUFFile(path string) (map[string]ggufValue, []ggufTensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	return readGGUF(f, true)
}
