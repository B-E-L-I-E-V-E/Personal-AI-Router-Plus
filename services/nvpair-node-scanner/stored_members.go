// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// stored_members.go reaches cluster members that mDNS cannot see.
//
// mDNS is link-local multicast: it never crosses a router, so a member on the
// far side of a routed VPN (OpenVPN, WireGuard, Tailscale) or on another subnet
// of the same site never appears in the browse map, even though unicast traffic
// to it works. cluster-manager already stores every member's last reachable
// address in members.json, so this loop asks node-info at that address directly
// and, when the machine there proves to be the member, folds it into the
// directory exactly as an mDNS sighting would be.
//
// What it trusts, and why that is no weaker than mDNS:
//   - It only dials addresses cluster-manager wrote after pairing, never anything
//     learned from the network on its own.
//   - The answering node-info must name the member's own nodeUuid, and must
//     report a cluster principal this node holds a pin for. A member too old to
//     report its service ports is assumed to use PAIR's fixed ones, so a node
//     that upgrades alone still reaches the rest. An mDNS record is
//     accepted on its uuid= and cluster-uuid= alone, with no answer required.
//   - Everything that matters afterwards (models, peer calls, inference) is
//     cluster mTLS pinned to that principal, so an impostor at the address gets
//     no further than a forged mDNS record would.
//
// A node seeded here is kept alive by the browser's ordinary liveness probe
// (reachable), which already asks node-info directly, and evicted by it when the
// member stops answering; the next pass re-adds it when it comes back. A member
// that mDNS later does see is simply updated in place, because the browser is
// keyed by uuid=.

import (
	"context"
	"encoding/json"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"time"

	"nvpair-shared/discovery"
	"nvpair-shared/noderec"
)

// storedMembersFile is cluster-manager's durable member list (membersFile in
// nvpair-cluster-manager/membership_store.go).
const storedMembersFile = "members.json"

// storedMembersInterval is how often members missing from the browse map are
// tried. The first pass runs after storedMembersFirstDelay so mDNS has a chance
// to find everyone on the local link first.
const (
	storedMembersInterval   = 15 * time.Second
	storedMembersFirstDelay = 8 * time.Second
)

// storedMemberNodeInfoPort is where node-info is asked. node-info binds the
// fixed SRVPort under the broker; tests point it at their own server.
var storedMemberNodeInfoPort = noderec.SRVPort

// legacyServicePorts are the fixed ports PAIR binds, assumed for a member too
// old to report its own (node-info's services field arrived with this
// fallback). They let a node that upgrades alone reach members that have not
// upgraded yet. The engine proxies are included because the proxy listens on
// them whether or not the engine behind it runs; which models a node actually
// serves still comes from its engine-manager, so a missing engine contributes
// no models and attracts no routing. Ports that turn out not to answer are
// handled like any other unreachable service.
var legacyServicePorts = map[noderec.ServiceKey]int{
	noderec.ServiceNodeInfo:      noderec.SRVPort,
	noderec.ServiceErrors:        14319,
	noderec.ServiceWorkload:      14320,
	noderec.ServiceCluster:       14321,
	noderec.ServiceEngineManager: 14322,
	noderec.ServiceEngineControl: 14323,
	noderec.ServiceOllama:        11434,
	noderec.ServiceLMStudio:      1234,
	noderec.ServiceLlamaCPP:      8080,
}

// storedMember is the subset of cluster-manager's ClusterNode this loop reads.
type storedMember struct {
	NodeUUID  string `json:"nodeUuid"`
	Name      string `json:"name"`
	IPAddress string `json:"ipAddress"`
	State     string `json:"state"`
}

// loadStoredMembers reads members.json from the cluster dir, keeping confirmed
// members with a dialable address. This node's own entry is skipped by
// identity in storedMemberEvents, not by address here. A missing or unreadable file
// yields nothing: this is a fallback and must never fail the daemon.
func loadStoredMembers(clusterDir string) []storedMember {
	if clusterDir == "" {
		return nil
	}
	data, err := os.ReadFile(filepath.Join(clusterDir, storedMembersFile))
	if err != nil {
		return nil
	}
	var all []storedMember
	if err := json.Unmarshal(data, &all); err != nil {
		slog.Debug("stored members unreadable", "err", err)
		return nil
	}
	out := make([]storedMember, 0, len(all))
	for _, m := range all {
		if m.State != "member" || m.NodeUUID == "" {
			continue
		}
		ip := net.ParseIP(m.IPAddress)
		if ip == nil || ip.IsUnspecified() || ip.IsMulticast() {
			continue
		}
		out = append(out, m)
	}
	return out
}

func (d *daemon) storedMembersLoop(ctx context.Context) {
	if d.clusterDir == "" {
		return
	}
	timer := time.NewTimer(storedMembersFirstDelay)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
		}
		for _, ev := range d.storedMemberEvents() {
			select {
			case d.directEvents <- ev:
			case <-ctx.Done():
				return
			}
		}
		timer.Reset(storedMembersInterval)
	}
}

// storedMemberEvents returns a Discovered event for every stored member that is
// missing from the browse map and proves to be itself at its stored address.
// Each such node is also seeded into the browser so its liveness and eviction
// follow the same path as any mDNS node.
func (d *daemon) storedMemberEvents() []DiscoveryEvent {
	members := loadStoredMembers(d.clusterDir)
	if len(members) == 0 {
		return nil
	}
	known := make(map[string]bool)
	for _, n := range d.browser.Nodes() {
		if u := UUIDFromTXT(n.TXT); u != "" {
			known[u] = true
		}
	}
	self := ""
	if d.reg != nil {
		self = d.reg.record().HostUUID
	}
	d.mesh.Refresh()
	selfPrincipal := d.mesh.NodeUUID()

	var events []DiscoveryEvent
	for _, m := range members {
		if m.NodeUUID == self || m.NodeUUID == selfPrincipal || known[m.NodeUUID] {
			continue
		}
		raw, ok := d.dialStoredMember(m)
		if !ok {
			continue
		}
		if u := UUIDFromTXT(raw.TXT); known[u] {
			continue
		}
		d.browser.Seed(raw)
		slog.Info("reached a cluster member outside mDNS range at its stored address",
			"name", m.Name, "ip", m.IPAddress, "hostUuid", UUIDFromTXT(raw.TXT))
		events = append(events, DiscoveryEvent{Type: discovery.Discovered, Node: raw})
	}
	return events
}

// dialStoredMember asks node-info at the member's stored address who it is and
// rebuilds its _nvpair-node record from the answer. It succeeds only when the
// machine names the member (by host uuid or cluster principal), holds a cluster
// principal this node has pinned. A member that does not report its service
// ports (an older version) is assumed to use PAIR's fixed ones.
func (d *daemon) dialStoredMember(m storedMember) (RawNode, bool) {
	info, ok := d.probeIdentity(m.IPAddress, storedMemberNodeInfoPort)
	if !ok || info.HostUUID == "" || info.ClusterUUID == nil {
		return RawNode{}, false
	}
	principal := *info.ClusterUUID
	if info.HostUUID != m.NodeUUID && principal != m.NodeUUID {
		slog.Debug("stored member address answered for a different node",
			"ip", m.IPAddress, "expected", m.NodeUUID, "reported", info.HostUUID)
		return RawNode{}, false
	}
	if principal == "" || !d.mesh.HasPin(principal) {
		return RawNode{}, false
	}
	rec := noderec.NodeRecord{
		HostUUID:    info.HostUUID,
		ClusterUUID: principal,
		IP:          m.IPAddress,
		Services:    make(map[noderec.ServiceKey]int, len(info.Services)),
	}
	for k, port := range info.Services {
		if k != "" && port > 0 {
			rec.Services[noderec.ServiceKey(k)] = port
		}
	}
	if len(rec.Services) == 0 {
		for k, port := range legacyServicePorts {
			rec.Services[k] = port
		}
	}
	if _, ok := rec.Port(noderec.ServiceNodeInfo); !ok {
		rec.Services[noderec.ServiceNodeInfo] = noderec.SRVPort
	}
	name := m.Name
	if name == "" {
		name = m.IPAddress
	}
	return RawNode{
		ID:        name,
		Host:      name,
		Port:      noderec.SRVPort,
		Addresses: []string{m.IPAddress},
		TXT:       rec.TXT(),
	}, true
}
