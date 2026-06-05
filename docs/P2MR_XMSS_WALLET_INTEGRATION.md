# Pearl P2MR + XMSS — Wallet Integration Guide

A reference for external wallet integrators adding support for Pearl's
post-quantum address type: **P2MR** (Pay-to-Merkle-Root, BIP 360) spent with
**XMSS** signatures.

All links below pin to either:

- **master** @ [`f8804af`](https://github.com/pearl-research-labs/pearl/commit/f8804af6d0a4d951f0e8576e170c0ca28f304d9d) — the merged consensus / script / address primitives.
- **PR #111** [`feat(xmss): add P2MR post-quantum transaction proof`](https://github.com/pearl-research-labs/pearl/pull/111) (branch `feat/post_quantum_sig`) — the wallet-facing transaction helpers, end-to-end tests, and an on-chain proof.

---

## TL;DR

A P2MR output commits to a Taproot-style script tree whose single leaf is
`<xmss_pubkey> OP_CHECKXMSSSIG`. Unlike Taproot, **there is no internal key
and no key-path spend** — the witness program *is* the script merkle root, so
the output can only be spent by revealing the XMSS leaf and a valid XMSS
signature. This makes a P2MR UTXO fully post-quantum.

```
xmss keypair ──> leaf: <xmss_pk(64B)> OP_CHECKXMSSSIG ──> taproot merkle root (32B)
                                                                  │
                  bech32m(witver=2, root)  <───  pkScript: OP_2 <32B root>
                       = P2MR address
```

| Property | Value |
|---|---|
| Output script | `OP_2 <32-byte merkle root>` (witness v2) |
| Address encoding | bech32m, witness version **2** (mainnet HRP `prl`, e.g. `prl1z…`) |
| Witness program | 32-byte tapscript merkle root (must be exactly 32 bytes) |
| Spend type | **script-path only** (no Schnorr key-path) |
| Signature scheme | XMSS `XMSS-SHAKE256_5_256` |
| Leaf opcode | `OP_CHECKXMSSSIG` = `0xde` (222), tapscript-only |
| pubkey / sig / msg size | 64 B / 2340 B / 32 B |
| **Max signatures per key** | **32** (stateful — see warning below) |

> ### ⚠️ XMSS is stateful — read this first
> Each key can sign **at most 32 times** (`full_height = 5`), and every
> `msgUID` in `[0, 31]` must be used **exactly once**. Signing two different
> messages with the same `msgUID` lets an attacker forge signatures. A wallet
> **must persist a monotonic per-key signature counter** and never reuse an
> index, even across restarts/devices. Treat each P2MR address as having a
> 32-signature budget. Keygen is also comparatively slow.

---

## 1. The XMSS scheme

`XMSS-SHAKE256_5_256` (not in RFC 8391, ~256-bit security). C/C++ reference
behind a small Go FFI.

- Parameter set: [`xmss/src/xmss.cpp` L12–L31](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/xmss/src/xmss.cpp#L12-L31)
- C ABI + sizes: [`xmss/xmss.h` L4–L35](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/xmss/xmss.h#L4-L35)
- Go bindings (`Keygen` / `Sign` / `Verify` + constants): [`xmss/xmss.go`](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/xmss/xmss.go#L18-L75)

```go
// xmss/xmss.go
PrivateSeedLen = 64; PublicSeedLen = 32; PKLen = 64; SKLen = 128
MsgLen = 32; SignatureLen = 2340; MaxSigns = 32

func Keygen(privateSeed [64]byte, publicSeed [32]byte) (pk [64]byte, sk [128]byte, err error)
func Sign(msgUID uint32, sk [128]byte, msg [32]byte) ([2340]byte, error) // msgUID in [0,31], once
func Verify(pk [64]byte, msg [32]byte, sig [2340]byte) bool
```

The message signed is always the 32-byte BIP 341/342 **tapscript sighash**
(`SigHashDefault`) of the spending input.

---

## 2. Address derivation (XMSS pubkey → P2MR address)

PR #111 provides the canonical, dependency-light routine.
[`wallet/wallet/xmsstxn/xmsstxn.go` `DeriveDescriptor` L56–L101](https://github.com/pearl-research-labs/pearl/blob/feat/post_quantum_sig/wallet/wallet/xmsstxn/xmsstxn.go#L56-L101):

```go
// 1. XMSS keypair from seeds (64B priv seed, 32B pub seed)
xmssPK, xmssSK, _ := xmss.Keygen(privSeed, pubSeed)

// 2. Leaf script: <xmss_pk> OP_CHECKXMSSSIG
leafScript, _ := txscript.XMSSLeafScript(xmssPK)

// 3. Single-leaf taproot tree → 32-byte merkle root (BIP 341 tagged hashing)
tree := txscript.AssembleTaprootScriptTree(txscript.NewBaseTapLeaf(leafScript))
merkleRoot := tree.RootNode.TapHash()

// 4. P2MR address (bech32m, witness v2) and pkScript (OP_2 <root>)
addr, _ := btcutil.NewAddressMerkleRoot(merkleRoot[:], net)
pkScript, _ := txscript.PayToAddrScript(addr)

// 5. Control block needed later to spend (parity bit set in ToBytes)
controlBlock, _ := (&txscript.MerkleRootControlBlock{
    LeafVersion: txscript.BaseLeafVersion, // 0xc0
}).ToBytes()
```

Supporting primitives on **master**:

- Leaf builder & witness layout (PR #111): [`node/txscript/xmss.go`](https://github.com/pearl-research-labs/pearl/blob/feat/post_quantum_sig/node/txscript/xmss.go#L16-L46) — `XMSSLeafScript`, `XMSSScriptPathWitness`, `XMSSSigChunks=5`, `XMSSSigChunkSize=468`.
- pkScript form `OP_2 <32B>`: [`node/txscript/standard.go` L161–L173](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/node/txscript/standard.go#L161-L173) (extract) and [L255–L260](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/node/txscript/standard.go#L255-L260) (`payToMerkleRootScript`).
- Address type (bech32m, witver 2): [`node/btcutil/address.go` L243–L261](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/node/btcutil/address.go#L243-L261); encoding [L17–L50](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/node/btcutil/address.go#L17-L50).
- Control block format (1 version/parity byte + 32·N proof; parity bit must be 1): [`node/txscript/taproot.go` L400–L495](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/node/txscript/taproot.go#L400-L495).

> **Note on seeds:** Pearl's own wallet derives the XMSS seeds deterministically
> from a BIP32 path under **purpose `222'`** (= the opcode number) and expands
> them with HKDF — see [`wallet/waddrmgr/xmss_keys.go`](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/wallet/waddrmgr/xmss_keys.go#L17-L94)
> and [`wallet/waddrmgr/scoped_manager.go` L507–L536](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/wallet/waddrmgr/scoped_manager.go#L507-L536).
> An external wallet may use any 64-byte private seed + 32-byte public seed; the
> address only depends on the resulting XMSS public key.

---

## 3. Spending a P2MR-XMSS output

Canonical 1-in / 1-out builder (PR #111):
[`wallet/wallet/xmsstxn/xmsstxn.go` `BuildSpend` L104–L180](https://github.com/pearl-research-labs/pearl/blob/feat/post_quantum_sig/wallet/wallet/xmsstxn/xmsstxn.go#L104-L180):

```go
// tapscript sighash over the revealed leaf (SigHashDefault)
tapLeaf := txscript.NewBaseTapLeaf(desc.LeafScript)
sigHash, _ := txscript.CalcTapscriptSignaturehash(
    sigHashes, txscript.SigHashDefault, tx, 0, prevFetcher, tapLeaf,
)

var msg [xmss.MsgLen]byte
copy(msg[:], sigHash)
sig, _ := xmss.Sign(req.MsgUID, desc.SecretKey, msg) // MsgUID used once!

// witness = [sig[0:468], sig[468:936], sig[936:1404], sig[1404:1872],
//            sig[1872:2340], leafScript, controlBlock]
tx.TxIn[0].Witness = txscript.XMSSScriptPathWitness(
    sig, desc.LeafScript, desc.ControlBlock,
)
```

The 2340-byte signature is split into **5 chunks of 468 bytes** so each witness
element stays under the 520-byte BIP 342 stack limit. Consensus side:

- Reassembly + verification (`opcodeCheckXmssSig`): [`node/txscript/opcode.go` L2020–L2104](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/node/txscript/opcode.go#L2020-L2104); opcode constant [L264](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/node/txscript/opcode.go#L264). Invalid non-empty sig → `NullFail`.
- P2MR spend path (rejects key-path, verifies leaf commitment): [`node/txscript/engine.go` `verifyMerkleRootSpend` L525–L561](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/node/txscript/engine.go#L525-L561).
- An alternative spend helper already on master: [`wallet/wallet/txauthor/author.go` `spendMerkleRoot` L285–L341](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/wallet/wallet/txauthor/author.go#L285-L341).
- Output/witness sizing constants: [`wallet/wallet/txsizes/size.go` L55–L62](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/wallet/wallet/txsizes/size.go#L55-L62).

---

## 4. Consensus rules to respect

- Every tx output must be **P2TR, P2MR, or OP_RETURN** — any other script type
  is rejected: [`node/blockchain/validate.go` L221–L234](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/node/blockchain/validate.go#L221-L234).
- P2MR script class is `WitnessV2MerkleRootTy`: [`node/txscript/standard.go` L32–L48](https://github.com/pearl-research-labs/pearl/blob/f8804af6d0a4d951f0e8576e170c0ca28f304d9d/node/txscript/standard.go#L32-L48).
- P2MR is **script-path only**; a single-element witness is rejected.

---

## 5. Reference implementation & on-chain proof (PR #111)

- Wallet-facing API: [`wallet/wallet/xmsstxn/xmsstxn.go`](https://github.com/pearl-research-labs/pearl/blob/feat/post_quantum_sig/wallet/wallet/xmsstxn/xmsstxn.go) — `Descriptor`, `UTXO`, `SpendRequest`, `DeriveDescriptor`, `BuildSpend`.
- Script helpers: [`node/txscript/xmss.go`](https://github.com/pearl-research-labs/pearl/blob/feat/post_quantum_sig/node/txscript/xmss.go).
- Simnet round-trip (fund P2MR_A → spend to P2MR_B with XMSS → spend P2MR_B onward): [`node/integration/p2mr_xmss_test.go`](https://github.com/pearl-research-labs/pearl/blob/feat/post_quantum_sig/node/integration/p2mr_xmss_test.go).
- Reproducible mainnet proof (committed seeds, addresses, txid, raw hex): [`wallet/wallet/xmss_p2mr_mainnet_test.go`](https://github.com/pearl-research-labs/pearl/blob/feat/post_quantum_sig/wallet/wallet/xmss_p2mr_mainnet_test.go).

**Verifiable fully-post-quantum P2MR → P2MR transaction:**

- tx: <https://blockbook.pearlresearch.ai/tx/1fc2168472af90dfbcd9cd6392b914c40150ce5eedaefa5d0bd60a02fbaa2b53>
- in `P2MR_A`: `prl1zjrmn44f7n9cqekf9h4f2wl5hgp3apn0a2t7l09gcyc580gqsxm3ss67e3p`
- out `P2MR_B`: `prl1zxxlkld5f4c723urfqemt3mzv9h8z6kf8yrxd8txa3fv2l0r6hkhqhryssk`

(The `prl1z…` prefix is bech32m witness **v2** = P2MR; P2TR addresses begin `prl1p…`.)

Build/test:

```bash
go test -tags xmss -count=1 ./wallet/wallet/xmsstxn/ ./node/txscript/
go test -tags xmss -count=1 -run 'TestP2MRXMSS|TestXMSS' -v ./wallet/wallet/
go test -tags 'rpctest xmss' -run TestP2MRXMSSRoundtripSimnet -count=1 -timeout=180s -v ./node/integration/
```

---

## File index

| Area | File | Source |
|---|---|---|
| XMSS params | `xmss/src/xmss.cpp`, `xmss/xmss.h` | master |
| XMSS Go FFI | `xmss/xmss.go` | master |
| `OP_CHECKXMSSSIG` | `node/txscript/opcode.go` | master |
| P2MR control block / merkle commitment | `node/txscript/taproot.go` | master |
| P2MR spend verification | `node/txscript/engine.go` | master |
| P2MR script class / pkScript | `node/txscript/standard.go` | master |
| P2MR address (bech32m v2) | `node/btcutil/address.go` | master |
| Output standardness | `node/blockchain/validate.go` | master |
| Seed derivation (purpose 222') | `wallet/waddrmgr/xmss_keys.go`, `scoped_manager.go` | master |
| Leaf script + witness chunking | `node/txscript/xmss.go` | PR #111 |
| Derive descriptor + build spend | `wallet/wallet/xmsstxn/xmsstxn.go` | PR #111 |
| Integration / mainnet proof tests | `node/integration/p2mr_xmss_test.go`, `wallet/wallet/xmss_p2mr_mainnet_test.go` | PR #111 |

