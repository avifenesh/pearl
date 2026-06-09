// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package blockchain

import (
	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/node/zkpow"
)

// SolveBlock mines a block certificate for the given header at the given height,
// selecting the version required by the MoE hardfork cutover
// (params.RequiredCertVersion).
//
// On SimNet it returns a lightweight dummy certificate of the required version
// (no actual mining), since PoW verification is skipped there. For real mining
// it modifies header.ProofCommitment to match the mined certificate.
func SolveBlock(header *wire.BlockHeader, params *chaincfg.Params, height int32) (wire.BlockCertificate, error) {
	moeActive := params.IsMoEForkActive(height)

	if params.Net == wire.SimNet {
		if moeActive {
			return &wire.CertificateV2{ProofData: []byte{0x00}}, nil
		}
		return &wire.CertificateV1{ProofData: []byte{0x00}}, nil
	}

	if moeActive {
		return zkpow.MineV2(header)
	}
	return zkpow.MineV1(header)
}
