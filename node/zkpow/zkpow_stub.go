//go:build !zkpow

package zkpow

import (
	"fmt"

	"github.com/pearl-research-labs/pearl/node/wire"
)

const (
	DefaultNBits     = 0x1E01FFFF
	DefaultM         = 256
	DefaultN         = 512
	DefaultNoiseRank = 32
	DefaultMMAType   = 0
)

func VerifyCertificate(header *wire.BlockHeader, cert wire.BlockCertificate) error {
	return fmt.Errorf("zkpow: build with -tags zkpow to enable proof verification")
}

func VerifyCertificateV1WithNbits(header *wire.BlockHeader, c *wire.CertificateV1, nbitsOverride uint32) error {
	return fmt.Errorf("zkpow: build with -tags zkpow to enable proof verification with nbits override")
}

func MineV1(header *wire.BlockHeader) (*wire.CertificateV1, error) {
	return nil, fmt.Errorf("zkpow: build with -tags zkpow to enable mining")
}

// MineV2 is the no-zkpow stub for MoE mining; the real implementation lives in
// miner.go under the zkpow build tag.
func MineV2(header *wire.BlockHeader) (*wire.CertificateV2, error) {
	return nil, fmt.Errorf("zkpow: build with -tags zkpow to enable mining")
}
