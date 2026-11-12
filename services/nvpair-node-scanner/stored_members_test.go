// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/noderec"
)

// storedMemberFixture is a clustered daemon with one stored member at a fake
// node-info on loopback, plus whatever that node-info answers.
type storedMemberFixture struct {
	d    *daemon
	info NodeInfoResponse
}

func newStoredMemberFixture(t *testing.T, members []storedMember, info func() NodeInfoResponse) *storedMemberFixture {
	t.Helper()
	clusterDir := filepath.Join(t.TempDir(), "cluster")
	clustertrusttest.Join(t, clusterDir, "cluster-vpn", "principal-self", "principal-peer")
	data, err := json.Marshal(members)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(clusterDir, storedMembersFile), data, 0o600); err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/node-info" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(info())
	}))
	t.Cleanup(srv.Close)
	_, portStr, _ := net.SplitHostPort(srv.Listener.Addr().String())
	port, _ := strconv.Atoi(portStr)
	prev := storedMemberNodeInfoPort
	storedMemberNodeInfoPort = port
	t.Cleanup(func() { storedMemberNodeInfoPort = prev })

	d := newMembershipTestDaemon(clusterDir)
	d.mesh.Refresh()
	d.http = &http.Client{}
	d.browser = NewDiscovery(noderec.ServiceType, noderec.Domain, WithKeyFunc(func(n RawNode) string { return UUIDFromTXT(n.TXT) }))
	return &storedMemberFixture{d: d}
}

func peerInfo(hostUUID, principal string) NodeInfoResponse {
	return NodeInfoResponse{
		HostUUID:    hostUUID,
		ClusterUUID: &principal,
		Services:    map[string]int{"ni": 14318, "ol": 11434, "cl": 14321, "em": 14322},
	}
}

// A member mDNS never reported is reached at its stored address and folded in
// with the record its node-info describes — the routed-VPN case.
func TestStoredMemberReachedOutsideMDNS(t *testing.T) {
	members := []storedMember{
		{NodeUUID: "principal-self", Name: "self", IPAddress: "127.0.0.1", State: "member"},
		{NodeUUID: "principal-peer", Name: "workstation-02", IPAddress: "127.0.0.1", State: "member"},
	}
	f := newStoredMemberFixture(t, members, func() NodeInfoResponse { return peerInfo("principal-peer", "principal-peer") })

	events := f.d.storedMemberEvents()
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	ev := events[0]
	if ev.Type != "discovered" || ev.Node.ID != "workstation-02" {
		t.Fatalf("event = %+v", ev)
	}
	rec := noderec.ParseTXT(ev.Node.TXT)
	if rec.HostUUID != "principal-peer" || rec.ClusterUUID != "principal-peer" || rec.IP != "127.0.0.1" {
		t.Fatalf("record = %+v", rec)
	}
	if port, _ := rec.Port(noderec.ServiceCluster); port != 14321 {
		t.Fatalf("cl port = %d, want 14321", port)
	}
	if len(f.d.browser.Nodes()) != 1 {
		t.Fatal("the reached member must be seeded so liveness tracks it")
	}
	// Already known: the next pass leaves it to the browser.
	if again := f.d.storedMemberEvents(); len(again) != 0 {
		t.Fatalf("second pass produced %d events, want 0", len(again))
	}
}

// The machine at a stored address must prove it is the member and hold a
// pinned principal; anything else is ignored.
func TestStoredMemberRejectsWrongOrUntrustedAnswers(t *testing.T) {
	cases := map[string]NodeInfoResponse{
		"different node":       peerInfo("someone-else", "someone-else"),
		"unpinned principal":   peerInfo("principal-peer", "principal-stranger"),
		"unclustered answer":   peerInfo("principal-peer", ""),
		"no cluster field":     {HostUUID: "principal-peer", Services: map[string]int{"ni": 14318}},
	}
	for name, info := range cases {
		t.Run(name, func(t *testing.T) {
			members := []storedMember{{NodeUUID: "principal-peer", Name: "workstation-02", IPAddress: "127.0.0.1", State: "member"}}
			f := newStoredMemberFixture(t, members, func() NodeInfoResponse { return info })
			if events := f.d.storedMemberEvents(); len(events) != 0 {
				t.Fatalf("got %d events, want 0", len(events))
			}
		})
	}
}

func TestLoadStoredMembersFiltersEntries(t *testing.T) {
	dir := t.TempDir()
	members := []storedMember{
		{NodeUUID: "a", IPAddress: "10.10.8.192", State: "member"},
		{NodeUUID: "b", IPAddress: "192.168.1.5", State: "pending"},
		{NodeUUID: "", IPAddress: "192.168.1.6", State: "member"},
		{NodeUUID: "c", IPAddress: "", State: "member"},
		{NodeUUID: "d", IPAddress: "0.0.0.0", State: "member"},
	}
	data, _ := json.Marshal(members)
	if err := os.WriteFile(filepath.Join(dir, storedMembersFile), data, 0o600); err != nil {
		t.Fatal(err)
	}
	got := loadStoredMembers(dir)
	if len(got) != 1 || got[0].NodeUUID != "a" {
		t.Fatalf("got %+v, want only member a", got)
	}
	if loadStoredMembers("") != nil || loadStoredMembers(filepath.Join(dir, "missing")) != nil {
		t.Fatal("no cluster dir or no file must yield nothing")
	}
}

func strPtr(s string) *string { return &s }

// A member on an older version reports no service ports; it is still reached,
// with PAIR's fixed ports assumed, so one machine can upgrade on its own.
func TestStoredMemberOlderPeerGetsFixedPorts(t *testing.T) {
	members := []storedMember{{NodeUUID: "principal-peer", Name: "workstation-02", IPAddress: "127.0.0.1", State: "member"}}
	f := newStoredMemberFixture(t, members, func() NodeInfoResponse {
		return NodeInfoResponse{HostUUID: "principal-peer", ClusterUUID: strPtr("principal-peer")}
	})
	events := f.d.storedMemberEvents()
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	rec := noderec.ParseTXT(events[0].Node.TXT)
	for k, want := range legacyServicePorts {
		if got, _ := rec.Port(k); got != want {
			t.Fatalf("%s port = %d, want %d", k, got, want)
		}
	}
}
