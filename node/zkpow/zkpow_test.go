//go:build zkpow

// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package zkpow

import (
	"fmt"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// Test block header values from mainnet genesis block (chaincfg/genesis.go)
var (
	testPrevBlock  = chainhash.Hash{}
	testMerkleRoot = chainhash.Hash([chainhash.HashSize]byte{
		0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2,
		0x7a, 0xc7, 0x2c, 0x3e, 0x67, 0x76, 0x8f, 0x61,
		0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32,
		0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a,
	})
	testTimestamp = time.Unix(1231006505, 0)
)

func testBlockHeader(nbits ...uint32) *wire.BlockHeader {
	bits := uint32(DefaultNBits)
	if len(nbits) > 0 {
		bits = nbits[0]
	}
	return &wire.BlockHeader{
		Version:    0,
		PrevBlock:  testPrevBlock,
		MerkleRoot: testMerkleRoot,
		Timestamp:  testTimestamp,
		Bits:       bits,
	}
}

// mineCertificateV1 mines a block and returns the header and CertificateV1.
func mineCertificateV1(t *testing.T) (*wire.BlockHeader, *wire.CertificateV1) {
	t.Helper()
	header := testBlockHeader()

	cert, err := MineV1(header)
	require.NoError(t, err, "mining should succeed")

	return header, cert
}

// copyBlockHeader creates a copy of BlockHeader for tampering tests
func copyBlockHeader(h *wire.BlockHeader) *wire.BlockHeader {
	return &wire.BlockHeader{
		Version:         h.Version,
		PrevBlock:       h.PrevBlock,
		MerkleRoot:      h.MerkleRoot,
		Timestamp:       h.Timestamp,
		Bits:            h.Bits,
		ProofCommitment: h.ProofCommitment,
	}
}

// copyCertificateV1 creates a deep copy of CertificateV1 for tampering tests
func copyCertificateV1(c *wire.CertificateV1) *wire.CertificateV1 {
	cp := &wire.CertificateV1{
		Hash:       c.Hash,
		PublicData: c.PublicData,
		ProofData:  make([]byte, len(c.ProofData)),
	}
	copy(cp.ProofData, c.ProofData)
	return cp
}

// TestMineAndVerifyProof tests the full mining and verification flow
func TestMineAndVerifyProof(t *testing.T) {
	header := testBlockHeader()

	t.Logf("Mining block: M=%d, N=%d, nbits=0x%08X",
		DefaultM, DefaultN, header.Bits)

	startMine := time.Now()
	header, cert := mineCertificateV1(t)
	t.Logf("Mining completed in %v, proof size: %d bytes", time.Since(startMine), len(cert.ProofData))

	startVerify := time.Now()
	err := VerifyCertificate(header, cert)
	require.NoError(t, err, "VerifyProof should succeed for valid proof")
	t.Logf("Verification completed in %v", time.Since(startVerify))
}

// TestTamperedParams tests that tampering any header or certificate field causes rejection.
// PublicData layout: config(0..52) | hash_a(52..84) | hash_b(84..116) | hash_jackpot(116..148) |
// m,n,t_rows,t_cols(148..164)
func TestTamperedParams(t *testing.T) {
	header, cert := mineCertificateV1(t)

	// Block header field tampering.
	headerTampers := []struct {
		name   string
		tamper func(h *wire.BlockHeader)
	}{
		{"Version", func(h *wire.BlockHeader) { h.Version = 1 }},
		{"PrevBlock", func(h *wire.BlockHeader) { h.PrevBlock[0] ^= 0xFF }},
		{"MerkleRoot", func(h *wire.BlockHeader) { h.MerkleRoot[0] ^= 0xFF }},
		{"Timestamp", func(h *wire.BlockHeader) { h.Timestamp = h.Timestamp.Add(time.Second) }},
	}
	for _, tc := range headerTampers {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			tamperedHeader := copyBlockHeader(header)
			tc.tamper(tamperedHeader)
			err := VerifyCertificate(tamperedHeader, cert)
			require.Error(t, err, "proof should be rejected when %s is tampered", tc.name)
			t.Logf("%s tampered: %v", tc.name, err)
		})
	}

	// Every byte of PublicData must individually cause rejection when flipped.
	for i := 0; i < wire.PublicDataSizeV1; i++ {
		i := i
		t.Run(fmt.Sprintf("PublicData[%d]", i), func(t *testing.T) {
			tamperedCert := copyCertificateV1(cert)
			tamperedCert.PublicData[i] ^= 0xFF
			err := VerifyCertificate(header, tamperedCert)
			require.Error(t, err, "proof should be rejected when PublicData[%d] is tampered", i)
			t.Logf("PublicData[%d] tampered: %v", i, err)
		})
	}

	// ProofData tampering.
	t.Run("ProofData", func(t *testing.T) {
		tamperedCert := copyCertificateV1(cert)
		tamperedCert.ProofData[20] ^= 0xFF
		err := VerifyCertificate(header, tamperedCert)
		require.Error(t, err, "proof should be rejected when ProofData is tampered")
		t.Logf("ProofData tampered: %v", err)
	})
}

// TestTamperedProof verifies that overwriting the metadata fields in proof_data is rejected.
func TestTamperedProof(t *testing.T) {
	header, cert := mineCertificateV1(t)

	tamperedCert := copyCertificateV1(cert)
	for i := 0; i < 50; i++ {
		tamperedCert.ProofData[i] ^= 0xFF
	}

	err := VerifyCertificate(header, tamperedCert)
	require.Error(t, err, "verification should fail with tampered proof metadata")
	t.Logf("Tampered proof metadata result: %v", err)
}

// TestVerifyProof_InvalidInput tests edge cases for invalid inputs
func TestVerifyProof_InvalidInput(t *testing.T) {
	header := testBlockHeader()

	// Generate a random 70400-byte proof (the native size of a valid CertificateV1)
	randomProof := make([]byte, 70400)
	for i := range randomProof {
		randomProof[i] = byte(i % 256)
	}

	testCases := []struct {
		name   string
		header *wire.BlockHeader
		cert   *wire.CertificateV1
	}{
		{"EmptyProofData", header, &wire.CertificateV1{ProofData: nil}},
		{"ZeroLengthProofData", header, &wire.CertificateV1{ProofData: []byte{}}},
		{"Random70400ByteProof", header, &wire.CertificateV1{ProofData: randomProof}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := VerifyCertificate(tc.header, tc.cert)
			require.Error(t, err)
			t.Logf("%s: %v", tc.name, err)
		})
	}
}

// ================================================================================
// MOE CERTIFICATE VERIFICATION
// ================================================================================

// TestVerifyV2CertificateFailClosed asserts the MoE verifier is fail-closed:
// a well-formed MoE certificate (header bindings satisfied) is still rejected
// because the cryptographic verifier is not yet implemented.
func TestVerifyV2CertificateFailClosed(t *testing.T) {
	header := testBlockHeader()
	cert := &wire.CertificateV2{ProofData: []byte{0x01, 0x02, 0x03}}

	// Satisfy the header-binding checks so verification reaches the
	// not-implemented (fail-closed) path rather than a binding mismatch.
	header.ProofCommitment = cert.ProofCommitment()
	cert.Hash = header.BlockHash()

	err := VerifyCertificate(header, cert)
	require.Error(t, err, "MoE certificate must be rejected (fail-closed)")
	require.Contains(t, err.Error(), "not yet implemented")
}

// TestVerifyV2CertificateBindingChecks asserts the binding checks reject a
// certificate whose hash/commitment do not match the header, and one with an
// empty proof.
func TestVerifyV2CertificateBindingChecks(t *testing.T) {
	header := testBlockHeader()

	// Mismatched block hash (Hash left zero, header has a real hash).
	err := VerifyCertificate(header, &wire.CertificateV2{ProofData: []byte{0x01}})
	require.Error(t, err)

	// Matching bindings but empty proof data is also rejected.
	matching := &wire.CertificateV2{}
	header.ProofCommitment = matching.ProofCommitment()
	matching.Hash = header.BlockHash()
	require.Error(t, VerifyCertificate(header, matching),
		"empty proof data must be rejected")
}

// TestMineV2FailClosed asserts the MoE miner is fail-closed.
func TestMineV2FailClosed(t *testing.T) {
	_, err := MineV2(testBlockHeader())
	require.Error(t, err, "MoE mining must be fail-closed")
	require.Contains(t, err.Error(), "not yet implemented")
}

// ================================================================================
// BENCHMARKS
// ================================================================================

// BenchmarkMine benchmarks the full mining + ZK proof generation.
func BenchmarkMine(b *testing.B) {
	// Test different difficulty levels (higher bits = easier, lower bits = harder)
	difficulties := []struct {
		name string
		bits uint32
	}{
		{"VeryEasy", 0x1F00FFFF},
		{"Easy", 0x1E7FFFFF},
		{"Medium", 0x1E01FFFF},
		{"Hard", 0x1D0FFFFF},
		{"VeryHard", 0x1C3FFFFF},
	}

	for _, diff := range difficulties {
		b.Run(diff.name, func(b *testing.B) {
			header := testBlockHeader(diff.bits)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cert, err := MineV1(header)
				if err != nil {
					b.Fatalf("mining failed: %v", err)
				}
				if cert == nil {
					b.Fatal("nil certificate")
				}
			}
			b.StopTimer()
			b.ReportMetric(b.Elapsed().Seconds()/float64(b.N), "sec/op")
		})
	}
}

// BenchmarkVerifyProof benchmarks the ZK proof verification phase.
// This measures the time to verify a ZK proof.
func BenchmarkVerifyProof(b *testing.B) {
	header := testBlockHeader()

	cert, err := MineV1(header)
	if err != nil {
		b.Fatalf("mining failed during setup: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := VerifyCertificate(header, cert)
		if err != nil {
			b.Fatalf("verification failed: %v", err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(b.Elapsed().Milliseconds())/float64(b.N), "ms/op")
}
