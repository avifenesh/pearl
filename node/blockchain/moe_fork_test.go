// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcutil"
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// These tests exercise the MoE certificate version cutover on SimNet. PoW
// verification is auto-skipped there (BFNoPoWCheck), but the version policy is
// not, so it is fully exercised despite the fail-closed placeholder verifier.

const moeForkTestHeight = int32(4)

func moeSimNetParams() chaincfg.Params {
	params := chaincfg.SimNetParams
	params.ReduceMinDifficulty = false
	params.MoEForkHeight = moeForkTestHeight
	return params
}

// newBlockForcedCert builds (but does not process) the next block on top of
// prev and replaces its certificate with the provided one, regardless of the
// version SolveBlock would have chosen for the height. Used to drive
// wrong-version rejection cases.
func newBlockForcedCert(t *testing.T, chain *BlockChain, prev *btcutil.Block,
	cert wire.BlockCertificate) *btcutil.Block {

	t.Helper()
	block, _, err := newBlock(chain, prev, nil)
	require.NoError(t, err)
	block.MsgBlock().MsgHeader.MsgCertificate = wire.MsgCertificate{Certificate: cert}
	return block
}

func requireRuleError(t *testing.T, err error, code ErrorCode) {
	t.Helper()
	require.Error(t, err)
	var ruleErr RuleError
	require.ErrorAs(t, err, &ruleErr)
	require.Equalf(t, code, ruleErr.ErrorCode,
		"expected rule error %v, got %v", code, ruleErr.ErrorCode)
}

// TestMoEForkActivationAcceptsRequiredVersion builds a chain across the
// activation boundary and asserts each block is accepted with the version
// required at its height (ZK before the fork, MoE at and after it). The blocks
// are produced by the height-aware SolveBlock, so this also covers mining-side
// version selection on SimNet.
func TestMoEForkActivationAcceptsRequiredVersion(t *testing.T) {
	params := moeSimNetParams()
	chain, teardown, err := chainSetup("moe_fork_accept", &params)
	require.NoError(t, err)
	defer teardown()

	tip := btcutil.NewBlock(chain.chainParams.GenesisBlock)
	tip.SetHeight(0)

	for h := int32(1); h <= moeForkTestHeight+2; h++ {
		block, _, err := addBlock(chain, tip, nil)
		require.NoErrorf(t, err, "block at height %d should be accepted", h)

		want := wire.CertificateVersionV1
		if h >= moeForkTestHeight {
			want = wire.CertificateVersionV2
		}
		require.Equalf(t, want, block.MsgBlock().BlockCertificate().Version(),
			"unexpected certificate version at height %d", h)

		tip = block
	}

	require.Equal(t, moeForkTestHeight+2, chain.BestSnapshot().Height)
}

// TestMoEForkActivationRejectsMoEBeforeFork asserts an MoE certificate is
// rejected for a block below the activation height.
func TestMoEForkActivationRejectsMoEBeforeFork(t *testing.T) {
	params := moeSimNetParams()
	chain, teardown, err := chainSetup("moe_fork_reject_early", &params)
	require.NoError(t, err)
	defer teardown()

	tip := btcutil.NewBlock(chain.chainParams.GenesisBlock)
	tip.SetHeight(0)

	// Extend with valid ZK blocks until the next block sits at
	// moeForkTestHeight-1 (still pre-fork).
	for h := int32(1); h <= moeForkTestHeight-2; h++ {
		block, _, err := addBlock(chain, tip, nil)
		require.NoError(t, err)
		tip = block
	}

	bad := newBlockForcedCert(t, chain, tip,
		&wire.CertificateV2{ProofData: []byte{0x00}})
	_, _, err = chain.ProcessBlock(bad, BFNone)
	requireRuleError(t, err, ErrDisallowedCertVersion)

	// The tip must be unchanged by the rejected block.
	require.Equal(t, moeForkTestHeight-2, chain.BestSnapshot().Height)
}

// TestMoEForkActivationRejectsZKAtFork asserts the legacy ZK certificate is
// rejected at and after the activation height (strict cutover).
func TestMoEForkActivationRejectsZKAtFork(t *testing.T) {
	params := moeSimNetParams()
	chain, teardown, err := chainSetup("moe_fork_reject_late", &params)
	require.NoError(t, err)
	defer teardown()

	tip := btcutil.NewBlock(chain.chainParams.GenesisBlock)
	tip.SetHeight(0)

	// Extend with valid ZK blocks until the next block sits exactly at the
	// activation height.
	for h := int32(1); h <= moeForkTestHeight-1; h++ {
		block, _, err := addBlock(chain, tip, nil)
		require.NoError(t, err)
		tip = block
	}

	bad := newBlockForcedCert(t, chain, tip,
		&wire.CertificateV1{ProofData: []byte{0x00}})
	_, _, err = chain.ProcessBlock(bad, BFNone)
	requireRuleError(t, err, ErrDisallowedCertVersion)

	require.Equal(t, moeForkTestHeight-1, chain.BestSnapshot().Height)
}

// TestSolveBlockSelectsVersionSimNet asserts the height-aware SolveBlock picks
// the certificate version required by the MoE cutover on SimNet (where it
// returns a lightweight dummy certificate of the correct version).
func TestSolveBlockSelectsVersionSimNet(t *testing.T) {
	params := moeSimNetParams()
	header := &wire.BlockHeader{}

	cert, err := SolveBlock(header, &params, moeForkTestHeight-1)
	require.NoError(t, err)
	require.Equal(t, wire.CertificateVersionV1, cert.Version(),
		"pre-fork height must use the ZK certificate")

	cert, err = SolveBlock(header, &params, moeForkTestHeight)
	require.NoError(t, err)
	require.Equal(t, wire.CertificateVersionV2, cert.Version(),
		"at/after fork height must use the MoE certificate")
}

// TestCheckCertificateVersion exercises the policy helper directly, including
// the disabled-fork case and the nil-certificate guard.
func TestCheckCertificateVersion(t *testing.T) {
	enabled := &chaincfg.Params{MoEForkHeight: moeForkTestHeight}
	disabled := &chaincfg.Params{MoEForkHeight: 0}

	zk := &wire.CertificateV1{}
	moe := &wire.CertificateV2{}

	// Disabled fork: ZK always valid, MoE never valid.
	require.NoError(t, CheckCertificateVersion(zk, 1_000_000, disabled))
	requireRuleError(t, CheckCertificateVersion(moe, 1_000_000, disabled),
		ErrDisallowedCertVersion)

	// Enabled fork: strict cutover at the activation height.
	require.NoError(t, CheckCertificateVersion(zk, moeForkTestHeight-1, enabled))
	require.NoError(t, CheckCertificateVersion(moe, moeForkTestHeight, enabled))
	requireRuleError(t, CheckCertificateVersion(moe, moeForkTestHeight-1, enabled),
		ErrDisallowedCertVersion)
	requireRuleError(t, CheckCertificateVersion(zk, moeForkTestHeight, enabled),
		ErrDisallowedCertVersion)

	// Missing certificate.
	requireRuleError(t, CheckCertificateVersion(nil, 1, enabled),
		ErrCertificateMissing)
}
