// Copyright (c) 2025-2026 The Pearl Research Labs developers
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package wire

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
)

// PublicDataSizeV2 is the size of the committed PublicData prefix for a V2
// certificate. It mirrors CertificateV1 as a placeholder.
//
// TODO Or: update when the V2 certificate format is finalized.
const PublicDataSizeV2 = 164

// MaxProofSizeV2 is the maximum size of a serialized MoE proof blob.
//
// TODO Or: update when the MoE proof format is finalized.
const MaxProofSizeV2 = 60000

// CertificateV2 is the Block Certificate introduced by the MoE hardfork. It is
// a distinct type from CertificateV1 so the finalized format can diverge; the
// current fields are placeholders.
//
// TODO Or: update when the V2 certificate format and verifier are finalized.
type CertificateV2 struct {
	// Hash is the block hash this certificate is bound to.
	Hash chainhash.Hash

	// PublicData contains the committed public fields. Placeholder layout.
	PublicData [PublicDataSizeV2]byte

	// ProofData contains the MoE proof. Placeholder format.
	ProofData []byte
}

func (c *CertificateV2) Version() CertificateVersion {
	return CertificateVersionV2
}

func (c *CertificateV2) BlockHash() chainhash.Hash {
	return c.Hash
}

// ProofCommitment computes SHA256d(CertificateVersion_LE(4) || PublicData). The
// version is included so the commitment binds to the certificate version.
func (c *CertificateV2) ProofCommitment() chainhash.Hash {
	var buf [4 + PublicDataSizeV2]byte
	binary.LittleEndian.PutUint32(buf[:4], uint32(c.Version()))
	copy(buf[4:], c.PublicData[:])
	return chainhash.DoubleHashH(buf[:])
}

// Serialize: BlockHash(32) + PublicData(PublicDataSizeV2) + ProofLen(4) + ProofData
// Version excluded - handled by MsgCertificate.
func (c *CertificateV2) Serialize(w io.Writer) error {
	if _, err := w.Write(c.Hash[:]); err != nil {
		return err
	}
	if _, err := w.Write(c.PublicData[:]); err != nil {
		return err
	}

	if err := binary.Write(w, binary.LittleEndian, uint32(len(c.ProofData))); err != nil {
		return err
	}
	if _, err := w.Write(c.ProofData); err != nil {
		return err
	}

	return nil
}

// Deserialize: BlockHash(32) + PublicData(PublicDataSizeV2) + ProofLen(4) + ProofData
// Version excluded - handled by MsgCertificate.
func (c *CertificateV2) Deserialize(r io.Reader) error {
	if _, err := io.ReadFull(r, c.Hash[:]); err != nil {
		return err
	}
	if _, err := io.ReadFull(r, c.PublicData[:]); err != nil {
		return err
	}

	var proofLen uint32
	if err := binary.Read(r, binary.LittleEndian, &proofLen); err != nil {
		return err
	}
	if proofLen > MaxProofSizeV2 {
		return fmt.Errorf("proof data too large: %d bytes (max %d)", proofLen, MaxProofSizeV2)
	}

	c.ProofData = make([]byte, proofLen)
	if _, err := io.ReadFull(r, c.ProofData); err != nil {
		return err
	}

	return nil
}

// SerializedSize returns the number of bytes needed to serialize the certificate
// fields. Format: BlockHash(32) + PublicData(PublicDataSizeV2) + ProofLen(4) + ProofData
// Note: Does NOT include version (4 bytes) - that's handled by MsgCertificate.
func (c *CertificateV2) SerializedSize() int {
	return 32 + PublicDataSizeV2 + 4 + len(c.ProofData)
}
