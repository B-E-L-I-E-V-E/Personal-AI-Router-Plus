// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"nvpair-shared/noderec"
)

// TestWriteClusterIdentityFrame pins the wire form of the push node-info decodes.
// node-info reads its stdin as newline-delimited JSON-RPC, so a missing newline or
// a renamed method silently stops membership reaching it — and the only symptom
// would be peers going on suppressing an invite for a node that has left.
func TestWriteClusterIdentityFrame(t *testing.T) {
	for _, tc := range []struct {
		name        string
		clusterUUID string
	}{
		{"a principal", "our-principal"},
		// A departure is the value peers are waiting for, so it is sent like any
		// other rather than skipped as empty.
		{"a departure", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			var mu sync.Mutex
			if err := writeClusterIdentityFrame(&mu, &buf, tc.clusterUUID); err != nil {
				t.Fatalf("write: %v", err)
			}

			line := buf.String()
			if !strings.HasSuffix(line, "\n") {
				t.Error("frame is not newline-terminated; node-info reads line-delimited frames")
			}
			if strings.Count(line, "\n") != 1 {
				t.Errorf("frame contains %d newlines, want exactly one", strings.Count(line, "\n"))
			}

			var frame struct {
				JSONRPC string                        `json:"jsonrpc"`
				Method  string                        `json:"method"`
				ID      json.RawMessage               `json:"id"`
				Params  noderec.ClusterIdentityParams `json:"params"`
			}
			if err := json.Unmarshal([]byte(line), &frame); err != nil {
				t.Fatalf("decode frame %q: %v", line, err)
			}
			if frame.JSONRPC != "2.0" {
				t.Errorf("jsonrpc = %q, want 2.0", frame.JSONRPC)
			}
			if frame.Method != noderec.MethodSetClusterIdentity {
				t.Errorf("method = %q, want %q", frame.Method, noderec.MethodSetClusterIdentity)
			}
			// A notification, not a request: node-info's stdout is drained to
			// io.Discard, so an id-bearing frame would strand a reply.
			if len(frame.ID) != 0 {
				t.Errorf("frame carries an id (%s); the push must be a notification", frame.ID)
			}
			if frame.Params.ClusterUUID != tc.clusterUUID {
				t.Errorf("clusterUuid = %q, want %q", frame.Params.ClusterUUID, tc.clusterUUID)
			}
			// The field must be on the wire even when empty, since that is how a
			// departure is expressed.
			if !strings.Contains(line, `"clusterUuid"`) {
				t.Errorf("frame omitted clusterUuid: %s", line)
			}
		})
	}
}

// The service-port frame node-info reads is a complete map, so a push after
// every service unregistered still sends an (empty) object, never null.
func TestWriteServicesFrame(t *testing.T) {
	var mu sync.Mutex
	var buf bytes.Buffer
	regs := []noderec.RegisterParams{
		{Service: noderec.ServiceNodeInfo, Port: 14318},
		{Service: noderec.ServiceCluster, Port: 14321},
		{Service: "", Port: 1},
		{Service: noderec.ServiceOllama, Port: 0},
	}
	if err := writeServicesFrame(&mu, &buf, servicesFromRegistrations(regs)); err != nil {
		t.Fatal(err)
	}
	var frame struct {
		Method string                 `json:"method"`
		Params noderec.ServicesParams `json:"params"`
	}
	if err := json.Unmarshal(buf.Bytes(), &frame); err != nil {
		t.Fatal(err)
	}
	if frame.Method != noderec.MethodSetServices || len(frame.Params.Services) != 2 || frame.Params.Services[noderec.ServiceCluster] != 14321 {
		t.Fatalf("frame = %+v", frame)
	}

	buf.Reset()
	if err := writeServicesFrame(&mu, &buf, nil); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(buf.Bytes(), []byte(`"services":{}`)) {
		t.Fatalf("empty push = %s, want an empty services object", buf.Bytes())
	}
}
