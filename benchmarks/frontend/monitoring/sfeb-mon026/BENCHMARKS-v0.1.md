# MON-026 Evidence Custody Benchmark Plan v0.1

## 1. Correctness targets (prospective)

| Dimension | Target |
|---|---:|
| Artifact corruption/substitution detection | 100% of injected byte changes |
| Missing declared artifact detection | 100% |
| Path traversal and symlink rejection | 100% |
| Root/signature mutation detection | 100% |
| Unknown signer behavior | 100% indeterminate; never qualified |
| Invalid signature behavior | 100% blocked |
| Predecessor/sequence mismatch | 100% blocked |
| Missing predecessor trust state | 100% indeterminate |
| Clock uncertainty over policy bound | 100% indeterminate |
| Positive-control valid bundles | 100% qualify |
| False-qualified bundles | 0 in 10,000 seeded/generated cases |

Targets are not measured results.

## 2. Performance targets

Measure on a pinned reference machine and storage device:
- verification throughput in MiB/s by file size and artifact count;
- p50/p95/p99 full-bundle verification latency;
- CPU time and peak resident memory;
- number of open file descriptors and metadata operations;
- overhead of signing/manifest creation separately from verification.

Prospective goals:
- ≥ 300 MiB/s sequential SHA-256 throughput on a documented modern CPU/storage baseline;
- ≤ 100 ms manifest/signature overhead for 1,000 metadata entries excluding artifact hashing;
- ≤ 1% added storage for manifest/signature metadata (artifact bytes are not duplicated).

## 3. Adversarial matrix

Mutate each field and file independently:
- artifact byte, file length, path, media type, evidence role, producer, command ID and collection clock;
- manifest profile, version, run ID, source identity, system ID, timeline, snapshot, target LSN, scope/inventory digest;
- root digest, signature algorithm, key ID and signature;
- sequence number, previous root and expected chain head;
- capture time, clock domain and uncertainty;
- add unlisted extra files and omit listed files.

The v1 verifier checks listed artifacts. It does not yet reject extra unlisted files in the bundle directory; a future strict-tree mode should compare the actual regular-file inventory with the signed list.

## 4. Fair comparisons

Do not compare raw SHA-256 microbenchmarks to full vendor KMS signing cost. Compare the same bundle, file inventory, key algorithm, storage and machine across:
- local SHA-256 plus Ed25519 verification;
- object-store checksum + KMS signature validation;
- artifact registry/attestation products where supported.

Measure custody guarantees, independent verification effort, detection latency, storage overhead and throughput separately.

## 5. Superiority criteria

MON-026 should only claim an advantage if it provides:
- reproducible offline verification without a vendor control plane;
- exact artifact-level and manifest-level tamper detection;
- clear unknown vs invalid trust outcomes;
- cross-run chain verification;
- less operator work and lower total custody cost than matched external attestations.

A valid signature must never be described as proof that the database or recovery outcome is true.
