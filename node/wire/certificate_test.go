// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire_test

import (
	"bytes"
	"encoding/binary"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/node/zkpow"
	"github.com/stretchr/testify/require"
)

// Genesis block values (from mainnet genesis)
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

// testBlockHeader creates a test block header
func testBlockHeader(nbits ...uint32) wire.BlockHeader {
	bits := uint32(zkpow.DefaultNBits)
	if len(nbits) > 0 {
		bits = nbits[0]
	}
	return wire.BlockHeader{
		Version:    0,
		PrevBlock:  testPrevBlock,
		MerkleRoot: testMerkleRoot,
		Timestamp:  testTimestamp,
		Bits:       bits,
	}
}

// ============================================================================
// CertificateV1 Tests
// ============================================================================

func TestCertificateV1_SerializeDeserialize(t *testing.T) {
	header := testBlockHeader()

	cert, err := zkpow.MineV1(&header)
	require.NoError(t, err, "mining should succeed")

	var buf bytes.Buffer
	err = cert.Serialize(&buf)
	require.NoError(t, err, "serialization should succeed")

	serialized := buf.Bytes()
	require.NotEmpty(t, serialized, "serialized data should not be empty")
	t.Logf("Serialized size: %d bytes", len(serialized))

	deserialized := &wire.CertificateV1{}
	err = deserialized.Deserialize(bytes.NewReader(serialized))
	require.NoError(t, err, "deserialization should succeed")

	require.Equal(t, cert.Hash, deserialized.Hash)
	require.Equal(t, cert.PublicData, deserialized.PublicData)
	require.Equal(t, cert.ProofData, deserialized.ProofData)
}

func TestCertificateV1_Verify(t *testing.T) {
	header := testBlockHeader()

	cert, err := zkpow.MineV1(&header)
	require.NoError(t, err, "mining should succeed")

	err = zkpow.VerifyCertificate(&header, cert)
	require.NoError(t, err, "valid CertificateV1 should verify")
}

func TestCertificateV1_VerifyErrors(t *testing.T) {
	header := testBlockHeader()

	origCert, err := zkpow.MineV1(&header)
	require.NoError(t, err, "mining should succeed")

	createCert := func() *wire.CertificateV1 {
		proofDataCopy := make([]byte, len(origCert.ProofData))
		copy(proofDataCopy, origCert.ProofData)
		return &wire.CertificateV1{
			Hash:       origCert.Hash,
			PublicData: origCert.PublicData,
			ProofData:  proofDataCopy,
		}
	}

	// Test certificate-level validation only (not underlying verifier logic)
	tests := []struct {
		name   string
		modify func(*wire.CertificateV1)
	}{
		{
			name: "empty proof data",
			modify: func(c *wire.CertificateV1) {
				c.ProofData = nil
			},
		},
		{
			name: "corrupted config",
			modify: func(c *wire.CertificateV1) {
				// Flip a random byte to corrupt it
				c.PublicData[wire.PublicDataSizeV1/2] ^= 0xFF
			},
		},
		{
			name: "block hash mismatch",
			modify: func(c *wire.CertificateV1) {
				c.Hash[0] ^= 0xFF
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cert := createCert()
			tt.modify(cert)

			err := zkpow.VerifyCertificate(&header, cert)
			require.Error(t, err, "invalid certificate should fail verification")
		})
	}
}

func TestCertificateV1_Version(t *testing.T) {
	cert := &wire.CertificateV1{}
	require.Equal(t, wire.CertificateVersionV1, cert.Version())
}

func TestCertificateV1_BlockHash(t *testing.T) {
	expectedHash := chainhash.Hash{1, 2, 3, 4}
	cert := &wire.CertificateV1{Hash: expectedHash}
	require.Equal(t, expectedHash, cert.BlockHash())
}

// ============================================================================
// MsgCertificate Tests
// ============================================================================

func TestMsgCertificate_ZK_RoundTrip(t *testing.T) {
	header := testBlockHeader()

	cert, err := zkpow.MineV1(&header)
	require.NoError(t, err, "mining should succeed")

	msg := &wire.MsgCertificate{Certificate: cert}
	require.NotNil(t, msg)
	require.Equal(t, wire.CertificateVersionV1, msg.Certificate.Version())

	var buf bytes.Buffer
	err = msg.PrlEncode(&buf, wire.ProtocolVersion)
	require.NoError(t, err, "encoding should succeed")

	decoded := &wire.MsgCertificate{}
	err = decoded.PrlDecode(bytes.NewReader(buf.Bytes()), wire.ProtocolVersion)
	require.NoError(t, err, "decoding should succeed")

	require.Equal(t, wire.CertificateVersionV1, decoded.Certificate.Version())
	decodedZK, ok := decoded.Certificate.(*wire.CertificateV1)
	require.True(t, ok, "decoded certificate should be CertificateV1")
	require.Equal(t, cert.Hash, decodedZK.Hash)
}

// ============================================================================
// CertificateV2 Tests
// ============================================================================

// newTestMoECert builds an CertificateV2 with deterministic placeholder
// contents. It does not call the (fail-closed) MoE miner, so these tests
// exercise the wire layer independently of the verifier implementation.
func newTestMoECert() *wire.CertificateV2 {
	cert := &wire.CertificateV2{
		Hash:      chainhash.Hash{0x11, 0x22, 0x33, 0x44},
		ProofData: []byte{0xde, 0xad, 0xbe, 0xef, 0x00, 0x01, 0x02},
	}
	for i := range cert.PublicData {
		cert.PublicData[i] = byte(i)
	}
	return cert
}

func TestCertificateV2_Version(t *testing.T) {
	cert := &wire.CertificateV2{}
	require.Equal(t, wire.CertificateVersionV2, cert.Version())
}

func TestCertificateV2_BlockHash(t *testing.T) {
	expectedHash := chainhash.Hash{1, 2, 3, 4}
	cert := &wire.CertificateV2{Hash: expectedHash}
	require.Equal(t, expectedHash, cert.BlockHash())
}

func TestCertificateV2_SerializeDeserialize(t *testing.T) {
	cert := newTestMoECert()

	var buf bytes.Buffer
	require.NoError(t, cert.Serialize(&buf), "serialization should succeed")

	wantSize := 32 + wire.PublicDataSizeV2 + 4 + len(cert.ProofData)
	require.Equal(t, wantSize, buf.Len())
	require.Equal(t, wantSize, cert.SerializedSize())

	deserialized := &wire.CertificateV2{}
	require.NoError(t, deserialized.Deserialize(bytes.NewReader(buf.Bytes())),
		"deserialization should succeed")

	require.Equal(t, cert.Hash, deserialized.Hash)
	require.Equal(t, cert.PublicData, deserialized.PublicData)
	require.Equal(t, cert.ProofData, deserialized.ProofData)
}

func TestCertificateV2_ProofCommitmentDeterministic(t *testing.T) {
	cert := newTestMoECert()
	require.Equal(t, cert.ProofCommitment(), cert.ProofCommitment(),
		"commitment must be deterministic")
}

// TestCertificateV2_ProofCommitmentVersioned ensures the commitment binds to
// the certificate version: an MoE certificate and a ZK certificate with
// identical PublicData must produce different commitments.
func TestCertificateV2_ProofCommitmentVersioned(t *testing.T) {
	moe := newTestMoECert()

	var zk wire.CertificateV1
	zk.Hash = moe.Hash
	copy(zk.PublicData[:], moe.PublicData[:])

	require.NotEqual(t, zk.ProofCommitment(), moe.ProofCommitment(),
		"ZK and MoE commitments over identical PublicData must differ by version")
}

func TestCertificateV2_DeserializeProofTooLarge(t *testing.T) {
	var buf bytes.Buffer
	var hash chainhash.Hash
	_, _ = buf.Write(hash[:])
	var pub [wire.PublicDataSizeV2]byte
	_, _ = buf.Write(pub[:])
	// Claim a proof length above the maximum without writing the bytes.
	require.NoError(t, binary.Write(&buf, binary.LittleEndian, uint32(wire.MaxProofSizeV2+1)))

	cert := &wire.CertificateV2{}
	require.Error(t, cert.Deserialize(bytes.NewReader(buf.Bytes())),
		"oversized proof must be rejected")
}

func TestMsgCertificate_MoE_RoundTrip(t *testing.T) {
	cert := newTestMoECert()

	msg := &wire.MsgCertificate{Certificate: cert}
	require.Equal(t, wire.CertificateVersionV2, msg.Certificate.Version())

	var buf bytes.Buffer
	require.NoError(t, msg.PrlEncode(&buf, wire.ProtocolVersion), "encoding should succeed")

	// MsgCertificate adds a 4-byte version prefix.
	require.Equal(t, 4+cert.SerializedSize(), msg.SerializeSize())

	decoded := &wire.MsgCertificate{}
	require.NoError(t, decoded.PrlDecode(bytes.NewReader(buf.Bytes()), wire.ProtocolVersion),
		"decoding should succeed")

	require.Equal(t, wire.CertificateVersionV2, decoded.Certificate.Version())
	decodedMoE, ok := decoded.Certificate.(*wire.CertificateV2)
	require.True(t, ok, "decoded certificate should be *CertificateV2")
	require.Equal(t, cert.Hash, decodedMoE.Hash)
	require.Equal(t, cert.PublicData, decodedMoE.PublicData)
	require.Equal(t, cert.ProofData, decodedMoE.ProofData)
}

func TestMsgCertificate_UnknownVersion(t *testing.T) {
	// A version that is neither Null, ZK, nor MoE must fail to decode.
	var buf bytes.Buffer
	require.NoError(t, binary.Write(&buf, binary.LittleEndian, uint32(wire.CertificateVersionV2+1)))

	decoded := &wire.MsgCertificate{}
	require.Error(t, decoded.PrlDecode(bytes.NewReader(buf.Bytes()), wire.ProtocolVersion),
		"unknown certificate version must be rejected")
}

func TestMsgCertificate_MoE_TooLargeEncode(t *testing.T) {
	cert := newTestMoECert()
	// Make the serialized MsgCertificate exceed CertificateMaxSize.
	cert.ProofData = make([]byte, wire.CertificateMaxSize)

	msg := &wire.MsgCertificate{Certificate: cert}
	var buf bytes.Buffer
	require.Error(t, msg.PrlEncode(&buf, wire.ProtocolVersion),
		"encoding an oversized certificate must be rejected")
}

func TestIsCertVersionAllowed(t *testing.T) {
	tests := []struct {
		version wire.CertificateVersion
		allowed bool
	}{
		{wire.CertificateVersionNull, false},
		{wire.CertificateVersionV1, true},
		{wire.CertificateVersionV2, true},
		{wire.CertificateVersionV2 + 1, false},
	}
	for _, tt := range tests {
		require.Equalf(t, tt.allowed, wire.IsCertVersionAllowed(tt.version),
			"IsCertVersionAllowed(%d)", tt.version)
	}
}
