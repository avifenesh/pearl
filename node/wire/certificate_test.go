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
// ZKCertificate Tests
// ============================================================================

func TestZKCertificate_SerializeDeserialize(t *testing.T) {
	header := testBlockHeader()

	cert, err := zkpow.Mine(&header)
	require.NoError(t, err, "mining should succeed")

	var buf bytes.Buffer
	err = cert.Serialize(&buf)
	require.NoError(t, err, "serialization should succeed")

	serialized := buf.Bytes()
	require.NotEmpty(t, serialized, "serialized data should not be empty")
	t.Logf("Serialized size: %d bytes", len(serialized))

	deserialized := &wire.ZKCertificate{}
	err = deserialized.Deserialize(bytes.NewReader(serialized))
	require.NoError(t, err, "deserialization should succeed")

	require.Equal(t, cert.Hash, deserialized.Hash)
	require.Equal(t, cert.PublicData, deserialized.PublicData)
	require.Equal(t, cert.ProofData, deserialized.ProofData)
}

func TestZKCertificate_Verify(t *testing.T) {
	header := testBlockHeader()

	cert, err := zkpow.Mine(&header)
	require.NoError(t, err, "mining should succeed")

	err = zkpow.VerifyCertificate(&header, cert)
	require.NoError(t, err, "valid ZKCertificate should verify")
}

func TestZKCertificate_VerifyErrors(t *testing.T) {
	header := testBlockHeader()

	origCert, err := zkpow.Mine(&header)
	require.NoError(t, err, "mining should succeed")

	createCert := func() *wire.ZKCertificate {
		proofDataCopy := make([]byte, len(origCert.ProofData))
		copy(proofDataCopy, origCert.ProofData)
		return &wire.ZKCertificate{
			Hash:       origCert.Hash,
			PublicData: origCert.PublicData,
			ProofData:  proofDataCopy,
		}
	}

	// Test certificate-level validation only (not underlying verifier logic)
	tests := []struct {
		name   string
		modify func(*wire.ZKCertificate)
	}{
		{
			name: "empty proof data",
			modify: func(c *wire.ZKCertificate) {
				c.ProofData = nil
			},
		},
		{
			name: "corrupted config",
			modify: func(c *wire.ZKCertificate) {
				// Flip a random byte to corrupt it
				c.PublicData[wire.PublicDataSize/2] ^= 0xFF
			},
		},
		{
			name: "block hash mismatch",
			modify: func(c *wire.ZKCertificate) {
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

func TestZKCertificate_Version(t *testing.T) {
	cert := &wire.ZKCertificate{}
	require.Equal(t, wire.CertificateVersionZK, cert.Version())
}

func TestZKCertificate_BlockHash(t *testing.T) {
	expectedHash := chainhash.Hash{1, 2, 3, 4}
	cert := &wire.ZKCertificate{Hash: expectedHash}
	require.Equal(t, expectedHash, cert.BlockHash())
}

// ============================================================================
// MsgCertificate Tests
// ============================================================================

func TestMsgCertificate_ZK_RoundTrip(t *testing.T) {
	header := testBlockHeader()

	cert, err := zkpow.Mine(&header)
	require.NoError(t, err, "mining should succeed")

	msg := &wire.MsgCertificate{Certificate: cert}
	require.NotNil(t, msg)
	require.Equal(t, wire.CertificateVersionZK, msg.Certificate.Version())

	var buf bytes.Buffer
	err = msg.PrlEncode(&buf, wire.ProtocolVersion)
	require.NoError(t, err, "encoding should succeed")

	decoded := &wire.MsgCertificate{}
	err = decoded.PrlDecode(bytes.NewReader(buf.Bytes()), wire.ProtocolVersion)
	require.NoError(t, err, "decoding should succeed")

	require.Equal(t, wire.CertificateVersionZK, decoded.Certificate.Version())
	decodedZK, ok := decoded.Certificate.(*wire.ZKCertificate)
	require.True(t, ok, "decoded certificate should be ZKCertificate")
	require.Equal(t, cert.Hash, decodedZK.Hash)
}

// ============================================================================
// MoECertificate Tests
// ============================================================================

// newTestMoECert builds an MoECertificate with deterministic placeholder
// contents. It does not call the (fail-closed) MoE miner, so these tests
// exercise the wire layer independently of the verifier implementation.
func newTestMoECert() *wire.MoECertificate {
	cert := &wire.MoECertificate{
		Hash:      chainhash.Hash{0x11, 0x22, 0x33, 0x44},
		ProofData: []byte{0xde, 0xad, 0xbe, 0xef, 0x00, 0x01, 0x02},
	}
	for i := range cert.PublicData {
		cert.PublicData[i] = byte(i)
	}
	return cert
}

func TestMoECertificate_Version(t *testing.T) {
	cert := &wire.MoECertificate{}
	require.Equal(t, wire.CertificateVersionMoE, cert.Version())
}

func TestMoECertificate_BlockHash(t *testing.T) {
	expectedHash := chainhash.Hash{1, 2, 3, 4}
	cert := &wire.MoECertificate{Hash: expectedHash}
	require.Equal(t, expectedHash, cert.BlockHash())
}

func TestMoECertificate_SerializeDeserialize(t *testing.T) {
	cert := newTestMoECert()

	var buf bytes.Buffer
	require.NoError(t, cert.Serialize(&buf), "serialization should succeed")

	wantSize := 32 + wire.MoEPublicDataSize + 4 + len(cert.ProofData)
	require.Equal(t, wantSize, buf.Len())
	require.Equal(t, wantSize, cert.SerializedSize())

	deserialized := &wire.MoECertificate{}
	require.NoError(t, deserialized.Deserialize(bytes.NewReader(buf.Bytes())),
		"deserialization should succeed")

	require.Equal(t, cert.Hash, deserialized.Hash)
	require.Equal(t, cert.PublicData, deserialized.PublicData)
	require.Equal(t, cert.ProofData, deserialized.ProofData)
}

func TestMoECertificate_ProofCommitmentDeterministic(t *testing.T) {
	cert := newTestMoECert()
	require.Equal(t, cert.ProofCommitment(), cert.ProofCommitment(),
		"commitment must be deterministic")
}

// TestMoECertificate_ProofCommitmentVersioned ensures the commitment binds to
// the certificate version: an MoE certificate and a ZK certificate with
// identical PublicData must produce different commitments.
func TestMoECertificate_ProofCommitmentVersioned(t *testing.T) {
	moe := newTestMoECert()

	var zk wire.ZKCertificate
	zk.Hash = moe.Hash
	copy(zk.PublicData[:], moe.PublicData[:])

	require.NotEqual(t, zk.ProofCommitment(), moe.ProofCommitment(),
		"ZK and MoE commitments over identical PublicData must differ by version")
}

func TestMoECertificate_DeserializeProofTooLarge(t *testing.T) {
	var buf bytes.Buffer
	var hash chainhash.Hash
	_, _ = buf.Write(hash[:])
	var pub [wire.MoEPublicDataSize]byte
	_, _ = buf.Write(pub[:])
	// Claim a proof length above the maximum without writing the bytes.
	require.NoError(t, binary.Write(&buf, binary.LittleEndian, uint32(wire.MaxMoEProofSize+1)))

	cert := &wire.MoECertificate{}
	require.Error(t, cert.Deserialize(bytes.NewReader(buf.Bytes())),
		"oversized proof must be rejected")
}

func TestMsgCertificate_MoE_RoundTrip(t *testing.T) {
	cert := newTestMoECert()

	msg := &wire.MsgCertificate{Certificate: cert}
	require.Equal(t, wire.CertificateVersionMoE, msg.Certificate.Version())

	var buf bytes.Buffer
	require.NoError(t, msg.PrlEncode(&buf, wire.ProtocolVersion), "encoding should succeed")

	// MsgCertificate adds a 4-byte version prefix.
	require.Equal(t, 4+cert.SerializedSize(), msg.SerializeSize())

	decoded := &wire.MsgCertificate{}
	require.NoError(t, decoded.PrlDecode(bytes.NewReader(buf.Bytes()), wire.ProtocolVersion),
		"decoding should succeed")

	require.Equal(t, wire.CertificateVersionMoE, decoded.Certificate.Version())
	decodedMoE, ok := decoded.Certificate.(*wire.MoECertificate)
	require.True(t, ok, "decoded certificate should be *MoECertificate")
	require.Equal(t, cert.Hash, decodedMoE.Hash)
	require.Equal(t, cert.PublicData, decodedMoE.PublicData)
	require.Equal(t, cert.ProofData, decodedMoE.ProofData)
}

func TestMsgCertificate_UnknownVersion(t *testing.T) {
	// A version that is neither Null, ZK, nor MoE must fail to decode.
	var buf bytes.Buffer
	require.NoError(t, binary.Write(&buf, binary.LittleEndian, uint32(wire.CertificateVersionMoE+1)))

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
		{wire.CertificateVersionZK, true},
		{wire.CertificateVersionMoE, true},
		{wire.CertificateVersionMoE + 1, false},
	}
	for _, tt := range tests {
		require.Equalf(t, tt.allowed, wire.IsCertVersionAllowed(tt.version),
			"IsCertVersionAllowed(%d)", tt.version)
	}
}
