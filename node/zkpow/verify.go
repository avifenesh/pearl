//go:build zkpow

// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

// Package zkpow provides ZK proof verification via Rust FFI.
package zkpow

/*
#cgo linux LDFLAGS: ${SRCDIR}/../../zk-pow/bindings/go/target/release/libzk_pow_ffi.a -ldl -lpthread -lm -lgcc_s
#cgo darwin LDFLAGS: ${SRCDIR}/../../zk-pow/bindings/go/target/release/libzk_pow_ffi.a -framework Security -lpthread -lm
#cgo windows LDFLAGS: ${SRCDIR}/../../zk-pow/bindings/go/target/x86_64-pc-windows-gnu/release/libzk_pow_ffi.a -lws2_32 -luserenv -lbcrypt -lntdll
#include "../../zk-pow/bindings/go/zk_pow_ffi.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/pearl-research-labs/pearl/node/wire"
)

// ================================================================================
// CERTIFICATE VERIFICATION
// ================================================================================

// VerifyCertificate performs sanity checks followed by cryptographic proof verification.
// It returns an error if the certificate is invalid or does not match the header.
func VerifyCertificate(header *wire.BlockHeader, cert wire.BlockCertificate) error {
	switch c := cert.(type) {
	case *wire.CertificateV1:
		return verifyV1Certificate(header, c)
	case *wire.CertificateV2:
		return verifyV2Certificate(header, c)
	default:
		return fmt.Errorf("unknown certificate type: %T", cert)
	}
}

// ================================================================================
// V1 CERTIFICATE VERIFICATION
// ================================================================================

func verifyV1Certificate(header *wire.BlockHeader, c *wire.CertificateV1) error {
	return verifyV1CertificateInner(header, c, nil)
}

// VerifyCertificateV1WithNbits verifies a V1 certificate using nbitsOverride
// as the difficulty target instead of the block header's nbits field.
//
// WARNING: This bypasses the header's embedded difficulty, Do not use it in block acceptance or relay paths.
func VerifyCertificateV1WithNbits(header *wire.BlockHeader, c *wire.CertificateV1, nbitsOverride uint32) error {
	return verifyV1CertificateInner(header, c, &nbitsOverride)
}

func verifyV1CertificateInner(header *wire.BlockHeader, c *wire.CertificateV1, nbitsOverride *uint32) error {
	blockHash := header.BlockHash()
	if !c.Hash.IsEqual(&blockHash) {
		return fmt.Errorf("block hash mismatch: certificate has %s, header has %s",
			c.Hash, blockHash)
	}

	certCommitment := c.ProofCommitment()
	if header.ProofCommitment != certCommitment {
		return fmt.Errorf("proof commitment mismatch: header has %s, certificate has %s",
			header.ProofCommitment, certCommitment)
	}

	if len(c.ProofData) == 0 { // avoid this case because of c.ProofData[0] access below
		return fmt.Errorf("empty proof data")
	}

	cBlockHeader := blockHeaderToC(header)

	var cZKProof C.CZKProof
	C.memcpy(unsafe.Pointer(&cZKProof.public_data[0]), unsafe.Pointer(&c.PublicData[0]), C.size_t(wire.PublicDataSizeV1))

	// Pin the ProofData memory to prevent GC from moving it during the C call
	var pinner runtime.Pinner
	pinner.Pin(&c.ProofData[0])
	defer pinner.Unpin()

	proofBlobPtr := (*C.uint8_t)(unsafe.Pointer(&c.ProofData[0]))
	cZKProof.proof_blob_len = C.uintptr_t(len(c.ProofData))
	cZKProof.proof_blob = proofBlobPtr

	// Call Rust FFI
	var errorBuf [C.ERROR_MSG_MAX_SIZE]C.char
	var result C.int32_t
	if nbitsOverride != nil {
		result = C.verify_zk_proof_with_nbits(&cBlockHeader, &cZKProof, C.uint32_t(*nbitsOverride), &errorBuf[0])
	} else {
		result = C.verify_zk_proof(&cBlockHeader, &cZKProof, &errorBuf[0])
	}
	msg := C.GoString(&errorBuf[0])

	switch result {
	case 0:
		return nil
	case 1:
		return fmt.Errorf("proof rejected: %s", msg)
	case 2:
		return fmt.Errorf("verification system error: %s", msg)
	default:
		return fmt.Errorf("unknown verification result %d: %s", result, msg)
	}
}

// ================================================================================
// V2 CERTIFICATE VERIFICATION
// ================================================================================

// verifyV2Certificate verifies a V2 certificate against the header. The
// header-binding checks are format-independent, but the cryptographic proof
// verification is not yet implemented, so this is fail-closed (always rejects).
//
// TODO Or: implement when the V2 verifier is finalized. The intended shape
// mirrors verifyV1CertificateInner: convert the header to C form, copy the
// PublicData, pin the ProofData, then call a Rust FFI entry point (e.g.
// C.verify_moe_proof) and translate its result code.
func verifyV2Certificate(header *wire.BlockHeader, c *wire.CertificateV2) error {
	blockHash := header.BlockHash()
	if !c.Hash.IsEqual(&blockHash) {
		return fmt.Errorf("block hash mismatch: certificate has %s, header has %s",
			c.Hash, blockHash)
	}

	certCommitment := c.ProofCommitment()
	if header.ProofCommitment != certCommitment {
		return fmt.Errorf("proof commitment mismatch: header has %s, certificate has %s",
			header.ProofCommitment, certCommitment)
	}

	if len(c.ProofData) == 0 {
		return fmt.Errorf("empty proof data")
	}

	// TODO Or: call the real V2 proof verifier (Rust FFI) when finalized.
	return fmt.Errorf("V2 certificate verifier not yet implemented")
}

// ================================================================================
// FFI CONVERSION HELPERS
// ================================================================================

// blockHeaderToC converts a Go BlockHeader to C.IncompleteBlockHeader.
// Note: PrevBlock and MerkleRoot are reversed from wire order to display order
func blockHeaderToC(header *wire.BlockHeader) C.IncompleteBlockHeader {
	cHeader := C.IncompleteBlockHeader{
		version:   C.uint32_t(header.Version),
		timestamp: C.uint32_t(header.Timestamp.Unix()),
		nbits:     C.uint32_t(header.Bits),
	}
	// Reverse hashes from wire order (internal) to display order
	hashLen := len(header.PrevBlock)
	for i := range cHeader.prev_block {
		cHeader.prev_block[i] = C.uint8_t(header.PrevBlock[hashLen-1-i])
		cHeader.merkle_root[i] = C.uint8_t(header.MerkleRoot[hashLen-1-i])
	}
	return cHeader
}
