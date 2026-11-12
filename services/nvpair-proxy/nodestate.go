// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Node state: whether this node's engine takes requests.
//
//   - running: the default. The node serves requests as usual.
//   - paused: the node takes no new requests, from its own clients or from
//     other nodes. Requests it is already serving finish normally. A new
//     request routes to another node that can serve it, or is refused with
//     503 when none can.
//   - stopped: as paused, and every request this node is serving is cancelled
//     at once, along with any request waiting here for a context slot. The
//     slots sessions hold here are freed.
//
// The state is about serving, not routing. A paused or stopped node still
// routes its own clients' requests to other nodes; it just never sends one to
// its own engine, and its cluster ingress refuses other nodes' requests with
// 503 and the X-PAIR-Node-State header, which a calling node treats as "try
// the next node". Sessions pinned here move: their next turn goes to another
// node, and moving releases the slot they held here.
//
// Other nodes learn of a change at once: this node announces it to every
// pinned peer, and repeats the announcement while it is not running so a peer
// that missed it, or joined since, learns it too. A peer forgets an
// announcement after peerStateTTL, so a lost "running" announcement costs at
// most that long; a request refused in the meantime is retried elsewhere and
// renews the mark.
//
// The state survives a restart. Pausing or stopping a node is a deliberate
// choice to take it out of service, and an app update or a reboot should not
// put it back silently. After a restart nothing is in flight, so a stopped
// node behaves as a paused one until it is resumed.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

type nodeState string

const (
	nodeRunning nodeState = "running"
	nodePaused  nodeState = "paused"
	nodeStopped nodeState = "stopped"
)

const (
	nodeStateFileName = "node-state.json"
	// nodeStateHeader names the state of a node that refused a request.
	nodeStateHeader = "X-PAIR-Node-State"
	// nodeStateAPIPath is this node's state, on the loopback personality.
	//
	//	GET /pair/v1/node/state   {"state","since","serving"}
	//	PUT /pair/v1/node/state   {"state":"running"|"paused"|"stopped"}
	nodeStateAPIPath = "/pair/v1/node/state"
	// peerNodeStatePath lets a cluster peer read or set this node's state, on
	// the mTLS ingress only: any member of the cluster may manage any node.
	peerNodeStatePath = "/pair/v1/peer/node/state"
	// peerNodeNoticePath receives another node's announcement of its state.
	peerNodeNoticePath = "/pair/v1/peer/node/notice"
)

// peerStateTTL is how long an announcement from a peer holds. The peer repeats
// it every nodeNoticeInterval while it is not running.
var (
	peerStateTTL        = 60 * time.Second
	nodeNoticeInterval  = 20 * time.Second
	peerNodeCallTimeout = 10 * time.Second
)

var (
	errNodePaused  = errors.New("this node is paused and is not taking new requests")
	errNodeStopped = errors.New("this node was stopped")
	// errNodesHeld refuses a request every capable node is paused or stopped for.
	errNodesHeld = errors.New("every node that can serve this request is paused or stopped")
)

func parseNodeState(s string) (nodeState, error) {
	switch nodeState(s) {
	case nodeRunning, nodePaused, nodeStopped:
		return nodeState(s), nil
	}
	return "", fmt.Errorf("unknown node state %q: expected running, paused or stopped", s)
}

// refusal is the error a request is refused with in a state.
func (s nodeState) refusal() error {
	if s == nodeStopped {
		return errNodeStopped
	}
	return errNodePaused
}

// nodeStateJSON is the wire form of a node's state.
type nodeStateJSON struct {
	// NodeID is set when the state is another node's.
	NodeID string `json:"nodeId,omitempty"`
	State  string `json:"state"`
	// Since is when the node entered this state (RFC 3339); empty for a node
	// that has run since it started and was never paused.
	Since string `json:"since,omitempty"`
	// Serving is how many requests the node's engine is serving now.
	Serving int `json:"serving"`
	// Cancelled is how many requests a stop cancelled.
	Cancelled int `json:"cancelled,omitempty"`
}

// nodeStateFile is the saved state.
type nodeStateFile struct {
	State string `json:"state"`
	Since string `json:"since,omitempty"`
}

type peerStateMark struct {
	state nodeState
	until time.Time
}

// nodeStateStore holds this node's state, the requests its engine is serving
// so a stop can cancel them, and the states other nodes have announced.
type nodeStateStore struct {
	mu      sync.Mutex
	state   nodeState
	since   time.Time
	serving map[uint64]func()
	next    uint64
	// peers is keyed by the peer's cluster UUID, the identity its certificate
	// proves and its discovery record carries.
	peers map[string]peerStateMark
	now   func() time.Time
	// path saves the state; empty keeps it in memory only (tests).
	path string
}

func newNodeStateStore() *nodeStateStore {
	return &nodeStateStore{
		state:   nodeRunning,
		serving: make(map[uint64]func()),
		peers:   make(map[string]peerStateMark),
		now:     time.Now,
	}
}

// accepting reports whether this node takes new requests.
func (s *nodeStateStore) accepting() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state == nodeRunning
}

func (s *nodeStateStore) current() nodeState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state
}

func (s *nodeStateStore) view() nodeStateJSON {
	s.mu.Lock()
	defer s.mu.Unlock()
	v := nodeStateJSON{State: string(s.state), Serving: len(s.serving)}
	if !s.since.IsZero() {
		v.Since = s.since.UTC().Format(time.RFC3339)
	}
	return v
}

// begin records a request this node's engine is about to serve. stop is how a
// stop cancels it. The check and the record are one step, so a request racing
// a pause is either refused or counted as already running, never lost between.
// The returned end is safe to call more than once.
func (s *nodeStateStore) begin(stop func()) (end func(), err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.state != nodeRunning {
		return func() {}, s.state.refusal()
	}
	s.next++
	token := s.next
	s.serving[token] = stop
	var once sync.Once
	return func() {
		once.Do(func() {
			s.mu.Lock()
			delete(s.serving, token)
			s.mu.Unlock()
		})
	}, nil
}

// set changes the state. Entering stopped cancels every request being served
// and reports how many. changed is false when the state was already in force.
func (s *nodeStateStore) set(next nodeState) (changed bool, cancelled int, err error) {
	s.mu.Lock()
	prev := s.state
	var stops []func()
	if next == nodeStopped {
		for _, stop := range s.serving {
			stops = append(stops, stop)
		}
	}
	if prev != next {
		s.state = next
		s.since = s.now()
	}
	file := nodeStateFile{State: string(s.state), Since: s.since.UTC().Format(time.RFC3339)}
	path := s.path
	s.mu.Unlock()

	for _, stop := range stops {
		stop()
	}
	if prev != next && path != "" {
		err = saveNodeState(path, file)
	}
	return prev != next, len(stops), err
}

// load restores a saved state. A missing file leaves the node running.
func (s *nodeStateStore) load() error {
	data, err := os.ReadFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	var f nodeStateFile
	if err := json.Unmarshal(data, &f); err != nil {
		return fmt.Errorf("parse %s: %w", s.path, err)
	}
	st, err := parseNodeState(f.State)
	if err != nil {
		return err
	}
	since, _ := time.Parse(time.RFC3339, f.Since)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state, s.since = st, since
	return nil
}

func saveNodeState(path string, f nodeStateFile) error {
	data, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// markPeer records a peer's announced or observed state. Running clears it.
func (s *nodeStateStore) markPeer(clusterUUID string, st nodeState) {
	if clusterUUID == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if st == nodeRunning {
		delete(s.peers, clusterUUID)
		return
	}
	s.peers[clusterUUID] = peerStateMark{state: st, until: s.now().Add(peerStateTTL)}
}

// peerState is a peer's last known state; running when nothing current is known.
func (s *nodeStateStore) peerState(clusterUUID string) nodeState {
	if clusterUUID == "" {
		return nodeRunning
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	m, ok := s.peers[clusterUUID]
	if !ok {
		return nodeRunning
	}
	if !s.now().Before(m.until) {
		delete(s.peers, clusterUUID)
		return nodeRunning
	}
	return m.state
}

// setLocalNodeState applies a state to this node, with everything that goes with
// it: waiters for a slot here give up, a stop frees the slots sessions hold
// here, and other nodes are told.
func (p *Proxy) setLocalNodeState(st nodeState) (nodeStateJSON, error) {
	changed, cancelled, err := p.node.set(st)
	if err != nil {
		slog.Warn("node state will not persist", "err", err)
	}
	if st != nodeRunning {
		p.sessions.refuseWaiters()
	}
	if st == nodeStopped {
		p.sessions.releaseAllSlots()
	}
	if changed {
		slog.Info("node state changed", "state", st, "cancelled", cancelled)
		go p.announceNodeState()
	}
	v := p.node.view()
	v.Cancelled = cancelled
	return v, nil
}

// peerTarget is a pinned cluster peer and the ingress address to reach it.
type peerTarget struct {
	nodeID      string
	clusterUUID string
	host        string
}

// clusterPeers lists the pinned peers this node can reach over mTLS, from
// every enabled facade's discovery, one per cluster UUID. self reports whether
// nodeID (when given) resolved to this node.
func (p *Proxy) clusterPeers(nodeID string) (peers []peerTarget, self bool) {
	if p.mesh == nil {
		return nil, false
	}
	p.mesh.Refresh()
	seen := make(map[string]bool)
	for _, f := range p.enabledFacades() {
		selfPort, alias := f.selfAddresses()
		for _, n := range f.discovery.Nodes() {
			if nodeID != "" && n.ID != nodeID {
				continue
			}
			u := f.targetURL(n)
			if u == nil {
				continue
			}
			if isSelfTarget(u, selfPort) || isAnyAliasSelfTarget(u, alias) {
				if nodeID != "" {
					self = true
				}
				continue
			}
			if !p.mesh.HasPin(n.ClusterUUID) || seen[n.ClusterUUID] {
				continue
			}
			seen[n.ClusterUUID] = true
			peers = append(peers, peerTarget{nodeID: n.ID, clusterUUID: n.ClusterUUID, host: u.Host})
		}
	}
	sort.Slice(peers, func(i, j int) bool { return peers[i].nodeID < peers[j].nodeID })
	return peers, self
}

// peerCall sends one request to a peer's ingress and decodes a JSON answer.
func (p *Proxy) peerCall(ctx context.Context, t peerTarget, method, path string, body, out any) error {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rd = bytes.NewReader(b)
	}
	u := url.URL{Scheme: "https", Host: t.host, Path: path}
	req, err := http.NewRequestWithContext(ctx, method, u.String(), rd)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	client := &http.Client{Transport: p.peerHTTPTransport(t.clusterUUID)}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if resp.StatusCode/100 != 2 {
		var e struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(raw, &e) == nil && e.Error != "" {
			return fmt.Errorf("node %s refused: %s", t.nodeID, e.Error)
		}
		return fmt.Errorf("node %s answered %d", t.nodeID, resp.StatusCode)
	}
	if out == nil || len(raw) == 0 {
		return nil
	}
	return json.Unmarshal(raw, out)
}

// announceNodeState tells every pinned peer this node's state. Best effort: a
// peer that misses it learns on its next refused request or the next notice.
func (p *Proxy) announceNodeState() {
	st := p.node.current()
	peers, _ := p.clusterPeers("")
	var wg sync.WaitGroup
	for _, t := range peers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(p.lifetime(), peerNodeCallTimeout)
			defer cancel()
			if err := p.peerCall(ctx, t, http.MethodPost, peerNodeNoticePath, map[string]string{"state": string(st)}, nil); err != nil {
				slog.Debug("node state notice failed", "node_id", t.nodeID, "err", err)
			}
		}()
	}
	wg.Wait()
}

// runNodeNotices repeats the announcement while this node is not running.
func (p *Proxy) runNodeNotices(ctx context.Context) {
	if p.node.current() != nodeRunning {
		p.announceNodeState()
	}
	t := time.NewTicker(nodeNoticeInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if p.node.current() != nodeRunning {
				p.announceNodeState()
			}
		}
	}
}

// lifetime is the process context, or Background before Run.
func (p *Proxy) lifetime() context.Context {
	if p.serveCtx != nil {
		return p.serveCtx
	}
	return context.Background()
}

// nodeStateParams is node/state's params: no state reads it; no nodeId means
// this node.
type nodeStateParams struct {
	NodeID string `json:"nodeId"`
	State  string `json:"state"`
}

// nodeStateCall reads or sets this node's state or, through its mTLS ingress,
// another cluster member's.
func (p *Proxy) nodeStateCall(ctx context.Context, params nodeStateParams) (nodeStateJSON, error) {
	var st nodeState
	if params.State != "" {
		var err error
		if st, err = parseNodeState(params.State); err != nil {
			return nodeStateJSON{}, err
		}
	}
	if params.NodeID != "" {
		peers, self := p.clusterPeers(params.NodeID)
		if !self {
			if len(peers) == 0 {
				return nodeStateJSON{}, fmt.Errorf("node %s is not a reachable member of this node's cluster", params.NodeID)
			}
			t := peers[0]
			var out nodeStateJSON
			var err error
			if st == "" {
				err = p.peerCall(ctx, t, http.MethodGet, peerNodeStatePath, nil, &out)
			} else {
				err = p.peerCall(ctx, t, http.MethodPut, peerNodeStatePath, map[string]string{"state": string(st)}, &out)
			}
			if err != nil {
				return nodeStateJSON{}, err
			}
			if got, perr := parseNodeState(out.State); perr == nil {
				p.node.markPeer(t.clusterUUID, got)
			}
			out.NodeID = t.nodeID
			return out, nil
		}
	}
	if st == "" {
		return p.node.view(), nil
	}
	return p.setLocalNodeState(st)
}

// nodeStatesResult is this node's state and every other node's known one.
type nodeStatesResult struct {
	Self  nodeStateJSON   `json:"self"`
	Peers []nodeStateJSON `json:"peers"`
}

func (p *Proxy) nodeStates() nodeStatesResult {
	res := nodeStatesResult{Self: p.node.view(), Peers: []nodeStateJSON{}}
	peers, _ := p.clusterPeers("")
	for _, t := range peers {
		if st := p.node.peerState(t.clusterUUID); st != nodeRunning {
			res.Peers = append(res.Peers, nodeStateJSON{NodeID: t.nodeID, State: string(st)})
		}
	}
	return res
}

// serveNodeStateAPI answers this node's state on the loopback personality.
func (f *facade) serveNodeStateAPI(w http.ResponseWriter, r *http.Request) bool {
	if r.URL.Path != nodeStateAPIPath {
		return false
	}
	w.Header().Set(requestIDHeader, resolveRequestID(r.Header))
	f.host.serveNodeState(w, r)
	return true
}

func (p *Proxy) serveNodeState(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		writeSessionJSON(w, http.StatusOK, p.node.view())
	case http.MethodPut:
		var body struct {
			State string `json:"state"`
		}
		raw, err := io.ReadAll(io.LimitReader(r.Body, maxSessionAPIBody))
		if err == nil {
			err = json.Unmarshal(raw, &body)
		}
		var st nodeState
		if err == nil {
			st, err = parseNodeState(body.State)
		}
		if err != nil {
			writeSessionJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		v, _ := p.setLocalNodeState(st)
		writeSessionJSON(w, http.StatusOK, v)
	default:
		methodNotAllowed(w, "GET, PUT")
	}
}

// servePeerNodeState answers the node-state routes of the mTLS ingress.
func (f *facade) servePeerNodeState(w http.ResponseWriter, r *http.Request, peer string) bool {
	p := f.host
	switch r.URL.Path {
	case peerNodeStatePath:
		slog.Info("cluster peer node state request", "peer", peer, "method", r.Method)
		p.serveNodeState(w, r)
		return true
	case peerNodeNoticePath:
		if r.Method != http.MethodPost {
			methodNotAllowed(w, "POST")
			return true
		}
		var body struct {
			State string `json:"state"`
		}
		raw, err := io.ReadAll(io.LimitReader(r.Body, 4096))
		if err == nil {
			err = json.Unmarshal(raw, &body)
		}
		st, perr := parseNodeState(body.State)
		if err != nil || perr != nil {
			writeIngressError(w, http.StatusBadRequest, "node-notice", `expected {"state":"running"|"paused"|"stopped"}`)
			return true
		}
		p.node.markPeer(peer, st)
		w.WriteHeader(http.StatusNoContent)
		return true
	}
	return false
}

// writeNodeRefusal answers a request this node will not take in its state.
func writeNodeRefusal(w http.ResponseWriter, st nodeState) {
	w.Header().Set(nodeStateHeader, string(st))
	w.Header().Set("Retry-After", "5")
	writeIngressError(w, http.StatusServiceUnavailable, "node-"+string(st), st.refusal().Error())
}

// trackLocal records an attempt on this node's engine so a stop can cancel
// it. stopped reports whether a stop did.
func (p *Proxy) trackLocal(cancel context.CancelFunc) (end func(), stopped *atomic.Bool, err error) {
	stopped = &atomic.Bool{}
	end, err = p.node.begin(func() {
		stopped.Store(true)
		cancel()
	})
	return end, stopped, err
}
