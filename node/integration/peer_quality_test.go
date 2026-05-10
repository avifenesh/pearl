//go:build rpctest
// +build rpctest

// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Regression test for the per-peer trust gate in netsync.handleInvMsg
// (node/netsync/manager.go). A low-quality peer's block inv must be
// answered with `getheaders` first; once the peer has supplied a
// tip-extending block (peerQualityCounter resets to 0), subsequent
// block invs must be answered with `getdata` directly.

package integration

import (
	"net"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/integration/rpctest"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

const (
	// pqHandshakeTimeout bounds how long the scripted peer waits for
	// the version/verack handshake with the victim node.
	pqHandshakeTimeout = 15 * time.Second

	// pqMessageBudget bounds the wait for a "must arrive" wire message
	// from the victim. Generous enough to absorb single-message round
	// trips on a loaded CI machine.
	pqMessageBudget = 5 * time.Second

	// pqDrainBudget is how long we wait while asserting that a wire
	// message must NOT arrive. Short enough to keep the test snappy.
	pqDrainBudget = 1 * time.Second

	// pqAcceptWait bounds how long the victim has to accept a block
	// pushed by the scripted peer.
	pqAcceptWait = 30 * time.Second

	// pqCurrentWait bounds how long we wait for the victim to mine its
	// initial bootstrap block via the local RPC.
	pqCurrentWait = 30 * time.Second
)

// scriptedPeer is a fake p2p peer used to observe wire-level replies
// from a victim pearld instance. Its OnGetHeaders / OnGetData callbacks
// push received messages onto buffered channels so the test driver can
// assert what the victim sent.
type scriptedPeer struct {
	*peer.Peer
	getHeadersCh chan *wire.MsgGetHeaders
	getDataCh    chan *wire.MsgGetData
}

// Close disconnects the underlying peer and waits for the read loop to
// finish.
func (sp *scriptedPeer) Close() {
	sp.Disconnect()
	sp.WaitForDisconnect()
}

// drain empties any pending messages from the observation channels.
func (sp *scriptedPeer) drain() {
	for {
		select {
		case <-sp.getHeadersCh:
		case <-sp.getDataCh:
		default:
			return
		}
	}
}

// expectGetHeaders blocks for up to pqMessageBudget for a getheaders
// message from the victim.
func (sp *scriptedPeer) expectGetHeaders(t *testing.T) *wire.MsgGetHeaders {
	t.Helper()
	select {
	case msg := <-sp.getHeadersCh:
		return msg
	case <-time.After(pqMessageBudget):
		t.Fatal("scripted: timed out waiting for getheaders")
		return nil
	}
}

// expectGetDataFor blocks for up to pqMessageBudget for a getdata
// message from the victim that includes hash and returns it.
func (sp *scriptedPeer) expectGetDataFor(t *testing.T,
	hash *chainhash.Hash) *wire.MsgGetData {

	t.Helper()
	deadline := time.After(pqMessageBudget)
	for {
		select {
		case msg := <-sp.getDataCh:
			for _, iv := range msg.InvList {
				if iv.Hash.IsEqual(hash) {
					return msg
				}
			}
		case <-deadline:
			t.Fatalf("scripted: timed out waiting for getdata "+
				"for %s", hash)
			return nil
		}
	}
}

// assertNoGetHeaders waits pqDrainBudget and fails if any getheaders
// message arrives in that window.
func (sp *scriptedPeer) assertNoGetHeaders(t *testing.T) {
	t.Helper()
	select {
	case msg := <-sp.getHeadersCh:
		t.Fatalf("scripted: unexpected getheaders: %+v", msg)
	case <-time.After(pqDrainBudget):
	}
}

// assertNoGetData waits pqDrainBudget and fails if any getdata message
// arrives in that window.
func (sp *scriptedPeer) assertNoGetData(t *testing.T) {
	t.Helper()
	select {
	case msg := <-sp.getDataCh:
		t.Fatalf("scripted: unexpected getdata: %+v", msg)
	case <-time.After(pqDrainBudget):
	}
}

// newScriptedPeer dials the victim, completes the v2 handshake, and
// returns a peer that records every getheaders / getdata it receives.
// The peer is inbound from the victim's POV so pickSyncCandidate
// excludes it; it advertises LastBlock=0 so the victim never demotes
// itself out of `current()` on its account.
func newScriptedPeer(t *testing.T, nodeAddr string) *scriptedPeer {
	t.Helper()

	conn, err := net.DialTimeout("tcp", nodeAddr, 5*time.Second)
	require.NoError(t, err, "scripted: dial victim")

	sp := &scriptedPeer{
		getHeadersCh: make(chan *wire.MsgGetHeaders, 16),
		getDataCh:    make(chan *wire.MsgGetData, 16),
	}

	verackCh := make(chan struct{})
	cfg := &peer.Config{
		Listeners: peer.MessageListeners{
			OnVerAck: func(_ *peer.Peer, _ *wire.MsgVerAck) {
				close(verackCh)
			},
			OnGetHeaders: func(_ *peer.Peer, msg *wire.MsgGetHeaders) {
				select {
				case sp.getHeadersCh <- msg:
				default:
				}
			},
			OnGetData: func(_ *peer.Peer, msg *wire.MsgGetData) {
				select {
				case sp.getDataCh <- msg:
				default:
				}
			},
		},
		// Report LastBlock=0 in our outgoing version handshake.
		NewestBlock: func() (*chainhash.Hash, int32, error) {
			return &chainhash.Hash{}, 0, nil
		},
		UserAgentName:       "scripted-peer",
		UserAgentVersion:    "1.0.0",
		Services:            wire.SFNodeNetwork | wire.SFNodeWitness | wire.SFNodeP2PV2,
		ChainParams:         &chaincfg.SimNetParams,
		DisableStallHandler: true,
	}

	p, err := peer.NewOutboundPeer(cfg, nodeAddr)
	if err != nil {
		conn.Close()
		t.Fatalf("scripted: NewOutboundPeer: %v", err)
	}
	p.AssociateConnection(conn)

	select {
	case <-verackCh:
		sp.Peer = p
		return sp
	case <-time.After(pqHandshakeTimeout):
		p.Disconnect()
		p.WaitForDisconnect()
		t.Fatal("scripted: timed out waiting for verack")
		return nil
	}
}

// startCurrentVictim returns a fresh simnet rpctest harness whose tip
// has a recent timestamp (so chain.IsCurrent reports true). One block
// is mined on the victim itself via the local RPC; SimNet's PoW is a
// dummy certificate so this is effectively instant. The harness is
// registered for teardown via t.Cleanup.
func startCurrentVictim(t *testing.T) *rpctest.Harness {
	t.Helper()

	victim, err := rpctest.New(&chaincfg.SimNetParams, nil, nil, "")
	require.NoError(t, err)
	require.NoError(t, victim.SetUp(true, 0))
	t.Cleanup(func() { require.NoError(t, victim.TearDown()) })

	if _, err := victim.Client.Generate(1); err != nil {
		t.Fatalf("startCurrentVictim: Generate(1): %v", err)
	}
	require.Eventually(t, func() bool {
		_, h, err := victim.Client.GetBestBlock()
		return err == nil && h >= 1
	}, pqCurrentWait, 100*time.Millisecond,
		"startCurrentVictim: tip didn't advance to >=1")

	return victim
}

// blockOnVictimTip constructs a simnet child block building on the
// victim's current tip. The block timestamp is one second past the
// parent's; this satisfies the "strictly increasing timestamp" rule
// regardless of how quickly the test produces blocks. SimNet PoW is a
// dummy certificate so block creation is effectively instant.
func blockOnVictimTip(t *testing.T, victim *rpctest.Harness) *btcutil.Block {
	t.Helper()

	prevHash, prevHeight, err := victim.Client.GetBestBlock()
	require.NoError(t, err)
	prevMsg, err := victim.Client.GetBlock(prevHash)
	require.NoError(t, err)
	prevBlock := btcutil.NewBlock(prevMsg)
	prevBlock.SetHeight(prevHeight)

	addr, err := victim.NewAddress()
	require.NoError(t, err)

	// Pass the zero time.Time so CreateBlock derives the timestamp as
	// prevBlockTime + 1s -- guaranteed strictly after the parent.
	blk, err := rpctest.CreateBlock(prevBlock, nil, rpctest.BlockVersion,
		time.Time{}, addr, nil, &chaincfg.SimNetParams)
	require.NoError(t, err)
	return blk
}

// TestPeerQualityInvGating exercises the per-peer trust gate in
// netsync.handleInvMsg (node/netsync/manager.go) and the corresponding
// counter-reset path in handleBlockMsg.
//
// Sub-tests:
//
//   - low_quality_peer_inv_triggers_getheaders: a freshly-handshaked
//     peer is low-quality (peerQualityCounter == peerQualityThreshold).
//     Its block inv must be answered with `getheaders` first; no
//     `getdata` is sent yet.
//
//   - high_quality_peer_inv_triggers_getdata: after the same peer
//     supplies a tip-extending block (counter resets to 0), a follow-up
//     block inv must be answered with `getdata` directly, without a
//     preceding `getheaders`.
//
// Both sub-tests rely on chain.IsCurrent reporting true at the victim,
// which requires the tip's timestamp to be within the last 24h. The
// shared startCurrentVictim helper bootstraps that condition by mining
// a single block on the victim itself.
func TestPeerQualityInvGating(t *testing.T) {
	t.Run("low_quality_peer_inv_triggers_getheaders", func(t *testing.T) {
		victim := startCurrentVictim(t)

		sp := newScriptedPeer(t, victim.P2PAddress())
		defer sp.Close()
		sp.drain()

		// A bogus block hash the victim cannot have. The choice of
		// hash is irrelevant; we just need an unknown InvTypeBlock so
		// haveInventory returns false.
		bogus := chainhash.Hash{0xab, 0xcd, 0xef}
		inv := wire.NewMsgInv()
		require.NoError(t, inv.AddInvVect(
			wire.NewInvVect(wire.InvTypeBlock, &bogus)))
		sp.QueueMessage(inv, nil)

		// !isPeerHighQuality branch: getheaders fires; no getdata.
		sp.expectGetHeaders(t)
		sp.assertNoGetData(t)
	})

	t.Run("high_quality_peer_inv_triggers_getdata", func(t *testing.T) {
		victim := startCurrentVictim(t)

		sp := newScriptedPeer(t, victim.P2PAddress())
		defer sp.Close()
		sp.drain()

		// Construct a child block in-process building on the victim's
		// current tip. The victim does not yet know about this block.
		blockA := blockOnVictimTip(t, victim)
		blockAHash := blockA.MsgBlock().BlockHeader().BlockHash()

		// Phase A: deliver blockA via inv -> getheaders -> getdata to
		// flip the peer quality counter to 0 (high-quality). The peer
		// starts low-quality so the inv path here MUST go through
		// getheaders before getdata.
		invA := wire.NewMsgInv()
		require.NoError(t, invA.AddInvVect(
			wire.NewInvVect(wire.InvTypeBlock, &blockAHash)))
		sp.QueueMessage(invA, nil)

		sp.expectGetHeaders(t)

		hdrs := wire.NewMsgHeaders()
		require.NoError(t, hdrs.AddBlockHeader(
			*blockA.MsgBlock().BlockHeader(),
			blockA.MsgBlock().BlockCertificate()))
		sp.QueueMessage(hdrs, nil)

		sp.expectGetDataFor(t, &blockAHash)
		sp.QueueMessage(blockA.MsgBlock(), nil)

		// Wait for the victim to accept blockA as its new tip. This
		// is the synchronization point that guarantees handleBlockMsg
		// has run and reset peerQualityCounter to 0.
		require.Eventually(t, func() bool {
			_, h, err := victim.Client.GetBestBlock()
			return err == nil && h >= 2
		}, pqAcceptWait, 100*time.Millisecond,
			"victim failed to accept blockA from scripted peer")

		// Drain any messages queued by the relay/trickle path before
		// asserting Phase B's quietness.
		sp.drain()

		// Construct blockB on top of (now-accepted) blockA.
		blockB := blockOnVictimTip(t, victim)
		blockBHash := blockB.MsgBlock().BlockHeader().BlockHash()

		// Phase B: announce blockB. As a high-quality peer we MUST
		// skip the inv -> getheaders gate and receive a direct
		// getdata for blockB.
		invB := wire.NewMsgInv()
		require.NoError(t, invB.AddInvVect(
			wire.NewInvVect(wire.InvTypeBlock, &blockBHash)))
		sp.QueueMessage(invB, nil)

		sp.expectGetDataFor(t, &blockBHash)
		sp.assertNoGetHeaders(t)
	})
}
