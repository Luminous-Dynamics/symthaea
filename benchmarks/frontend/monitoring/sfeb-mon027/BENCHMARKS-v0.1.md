# MON-027 Benchmark Plan v0.1

## Correctness targets (prospective)

| Dimension | Target |
|---|---:|
| Missing allowlisted files | 100% blocked |
| Unlisted regular files / symlinks / special files | 100% blocked |
| Path traversal / absolute / drive-qualified paths | 100% blocked |
| File byte or size substitution between assembly and attach | 100% blocked |
| Wrong run ID / sequence / previous root | 100% blocked or indeterminate as specified |
| Key ID or algorithm substitution | 100% blocked |
| Valid positive bundle | 100% accepted |
| False-qualified bundles | 0 in 10,000 mutation cases |
| Private-key exposure in collector/assembler | 0 private-key reads or copies |

## Performance targets (not measured)

- Sequential assembly hashing ≥250 MiB/s on documented modern CPU/SSD.
- Manifest and signing-request construction ≤100 ms for 1,000 metadata entries, excluding file hashing.
- Signature attach/verification ≤50 ms for Ed25519 on a pinned reference CPU, excluding artifact re-hashing.
- Peak memory ≤64 MiB when hashing large artifacts; artifact bytes must be streamed, not loaded whole.

## Fair measurements

Report file sizes/counts, storage medium, CPU, OS, Go version, SHA-256 implementation, signer type, signing network RTT and whether attach re-hashes the entire bundle. Separate local Ed25519, HSM/KMS and remote signer latency.

Do not claim MON-027 is faster or safer than a vendor signing service based only on local software-signature benchmarks. Compare complete custody guarantees and operational custody model.
