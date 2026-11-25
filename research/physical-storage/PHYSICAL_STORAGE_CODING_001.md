# Physical Storage Coding and Layout Co-Design — Research Protocol v0.1

**Programme:** PHYSICAL-STORAGE-CODING-001  
**Owner:** Symthaea search, representation and optimization  
**Physical twin:** Symtropy [PHYSICAL-STORAGE-SYSTEMS-001](https://github.com/Luminous-Dynamics/symtropy/issues/7447)  
**Tracking:** [Symthaea #13630](https://github.com/Luminous-Dynamics/symthaea/issues/13630)  
**Snapshot date:** 2026-11-25

## 1. Thesis

The problem is not to make a more impressive embedding. It is to reconstruct the exact original byte sequence after the physical medium has introduced noise, missing cells, bursts, spatially correlated damage, calibration drift or layer loss.

Optimize the **recoverability frontier**:

- exact payload recovery under a declared channel model;
- net payload density after all redundancy and calibration overhead;
- encode/decode time and resource use;
- local recovery and damage containment;
- energy where measured with a declared system boundary;
- format longevity and reader replaceability.

Every decoded payload is verified against an independent cryptographic digest. A result that looks semantically similar is not a recovered payload.

## 2. Relation to current Symthaea memory

Symthaea already has a digital memory stack with SQLite for durable metadata/records, optional LanceDB for persistent vector/HDC memory, optional mmap-backed BinaryHV storage and DuckDB for offline telemetry. Those are software storage backends. They are not physical recording media and do not establish physical density, retention or device reliability.

The physical-storage programme is a separate research track. It may reuse the existing storage abstractions for experiment records and HDC indexes, but must not conflate vector retrieval with exact archival recovery.

## 3. Paradigm shifts

### 3.1 Code and physical layout are one design problem

A conventional code is often selected before the physical geometry is fixed. Here the optimizer must jointly consider:
- logical payload block and symbol boundaries;
- physical cell/tile/layer placement;
- codeword size and parity allocation;
- interleave depth and spatial orientation;
- local calibration/reference placement;
- metadata and format-header replication;
- physical-region isolation and reconstruction locality.

### 3.2 Optimize damage containment, not only average BER

Two media with the same average raw bit error rate can have very different recoverability when one produces isolated errors and the other loses an entire adjacent region. Evaluate spatial correlation length, burst direction, layer dropout, cross-talk and contiguous damage.

### 3.3 Make archive formats self-describing

The physical payload must be accompanied by redundant format/version identifiers, calibration references, decoder test vectors and integrity metadata. Keep decoder source and known-answer vectors in a portable archive package. Reader replacement is part of the benchmark, not an operational afterthought.

### 3.4 Separate exact data from semantic navigation

HDC may be useful for approximate content indexing, similarity search or finding candidate records. It is not a substitute for BCH/RS/LDPC/erasure codes, exact checksums or cryptographic authentication. HDC similarity must never authorize accepting approximate bytes as a valid archive.

## 4. Baseline tournament

The first benchmark compares methods on identical payloads, physical layout, damage maps, error budgets and compute budgets.

| Baseline | Primary fit | Required controls |
|---|---|---|
| Uncoded payload | Establish raw channel behaviour | Exact digest must fail on altered bytes |
| BCH | Bounded symbol errors in a block | Report code parameters and decoder failure boundary |
| Reed–Solomon | Symbol errors and erasures / burst handling | Use explicit symbolization and erasure metadata |
| LDPC | Large block codes with soft/hard read information | Pin block length, rate, quantization and decoder iterations |
| Interleaving + parity | Convert bursts into distributed errors | Freeze physical layout and interleave depth |
| Erasure codes | Missing tiles/regions and multi-fragment recovery | Declare erasure identification and reconstruction threshold |
| Hybrid code + placement | Joint search over ECC and physical mapping | Equal search/evaluation budget; publish ablations |
| HDC-assisted indexing | Optional retrieval/navigation only | Must not affect exact recovery acceptance |

Use established implementations as pinned comparators before considering custom coding kernels. Keep code-theoretic guarantees separate from empirical results under a particular synthetic or measured channel.

## 5. Fault model strata

Every channel case identifies whether it is a deterministic sensitivity probe, a source-derived measurement, or a fitted distribution supported by repeated physical observations.

Required classes:
1. independent bit/symbol errors;
2. contiguous bursts of varying length and direction;
3. missing or weak cells/voxels;
4. spatial clusters of adjacent missing/corrupt tiles;
5. complete layer or sub-volume dropout;
6. read noise and spatial cross-talk;
7. position/registration drift between layers;
8. spatially varying write threshold and read sensitivity;
9. calibration-marker corruption;
10. metadata/header loss;
11. adversarially selected damage patterns;
12. media handling/environmental damage once measured data exists.

Synthetic fault seeds are not physical failure probabilities. Do not infer reliability distributions from invented or tiny samples.

## 6. Metrics

Report per fault class and per case:
- raw BER, symbol error rate and erasure fraction;
- post-ECC BER and exact decoded-block status;
- exact whole-payload recovery and digest equality;
- uncorrectable tiles/regions and unaffected-region recovery;
- usable payload density and ECC/calibration/metadata/spare overhead;
- encode/decode throughput, p50/p95/p99 latency and peak memory;
- code parameters, decoder iterations and compute budget;
- recovery locality and the amount of data lost per damaged region;
- measured energy only where the complete hardware boundary is declared;
- sample counts, confidence limits and the distinction between synthetic, measured and analytical evidence.

A small no-error test does not prove a BER target such as 10^-15. Publish the actual number of tested bits, failures and the statistical bound.

## 7. Search and optimizer protocol

A candidate is a tuple:

`candidate = (code, rate, block_geometry, interleave, physical_placement, calibration_layout, metadata_replication)`

The evaluator applies a frozen channel/fault map and performs byte-exact decode. The optimizer proposes candidates but cannot modify the evaluation corpus or held-out fault seeds.

- Separate training, validation and held-out spatial fault maps.
- Use equal objective-evaluation budgets for exhaustive search where feasible, greedy/local methods, Bayesian optimization and Symthaea-guided search.
- Report best-found score, wall-clock, memory, number of physical/channel evaluations, and variance across held-out maps.
- Preserve all hard constraints as hard constraints; no weighted scalar may hide a failed exact-recovery criterion.
- Ablate the optimizer, HDC/CfC, placement strategy, interleaver, calibration and ECC independently.
- Test worst-case adversarial masks in addition to random seeds.
- Recompute final payload hashes independently of the optimizer and decoder.

HDC/CfC contribution is a research hypothesis. It must beat an equal-budget classical baseline on a held-out metric to be retained in the design.

## 8. Initial challenge targets

1. All pristine and known-answer fixtures decode byte-for-byte exactly.
2. Corrupted payloads never pass final integrity validation when the digest does not match.
3. At equal net density and identical channel conditions, the proposed method must not regress exact recovery against the strongest classical baseline.
4. Demonstrate local recovery of all unaffected regions after bounded spatial damage; publish every unrecoverable region.
5. Improve at least one Pareto axis (recovery under correlated damage, net density, decode latency or compute budget) without violating the other frozen hard constraints.
6. Provide portable metadata and decoder vectors that allow an independent implementation to reproduce a successful decode.

These are software benchmark gates. They do not establish physical device performance, measured retention or industrial superiority.

## 9. Deterministic starter corpus

The companion fixture `fixtures/physical_storage_channel_cases_v0_1.json` contains one 224-byte known payload (28 payload tiles × 8 bytes), fixed SHA-256 digest, a 4 × 4 × 2 logical tile grid and four explicitly reserved control tiles for headers/calibration. It is a protocol seed only. It does not model a particular glass, magnetic or NAND device and does not include ECC outputs.

The starter corpus deliberately includes pristine, clustered, directional burst, layer-dropout and calibration/metadata damage cases. Tests should reject changed coordinates, changed payloads, mismatched layout identities, wrong format versions, invalid hashes and unsupported evidence claims.

## 10. Receipt contract

Each experiment receipt binds:
- payload byte length and SHA-256 digest;
- corpus, channel case and fault-mask digests;
- logical tile and physical layout version;
- code family, rate, symbols, parity, interleave and decoder configuration;
- physical medium specimen and measurement provenance when available;
- raw read observations and evidence class;
- recovered payload digest and exact-recovery result;
- uncorrectable region list and recovery coverage;
- optimizer identity, evaluation budget and held-out case identities.

Do not treat a hash as proof of who produced the data. Authentication and source provenance require a separate trust mechanism.

## 11. Evidence boundary

The first tranche is an algorithmic benchmark over deterministic byte arrays and synthetic fault masks. It is not a physical glass, tape, HDD, MRAM or flash experiment. No HDC coding advantage, physical density improvement, retention result, or 10^-15 reliability result is claimed. Actual media error maps must come from Symtropy/partner-lab metrology or a clearly identified public dataset.

## 12. References

- Symtropy physical storage twin: https://github.com/Luminous-Dynamics/symtropy/issues/7447
- Symtropy magnetic solver qualification (useful evidence pattern, not a recording model): https://github.com/Luminous-Dynamics/symtropy/issues/7444
- Microsoft Research Project Silica, *Nature* 650, 606–612 (2026): https://www.nature.com/articles/s41586-025-10042-w
- LTO-10 native capacity and throughput: https://www.lto.org/lto-10/
- Micron 6600 ION QLC SSD product information: https://www.micron.com/products/storage/ssd/data-center-ssd/6600-ion
- Spin-current noise and skyrmion racetrack dynamics, *Physical Review B* 113, L180407 (2026): https://journals.aps.org/prb/abstract/10.1103/73z4-ymkp
