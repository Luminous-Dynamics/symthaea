# Classical ECC Baseline Tournament — v0.1

**Programme:** PHYSICAL-STORAGE-CLASSICAL-ECC-TOURNAMENT-001  
**Tracking:** [Symthaea #13750](https://github.com/Luminous-Dynamics/symthaea/issues/13750)  
**Parent coding programme:** [Symthaea #13630](https://github.com/Luminous-Dynamics/symthaea/issues/13630)  
**Custom decoder:** [PR #13713](https://github.com/Luminous-Dynamics/symthaea/pull/13713)

## 1. Purpose

This register makes the existing classical-code comparison requirement executable and reviewable. It does not replace the custom MDS reference or assume that an external crate is automatically equivalent because it is labelled Reed–Solomon.

The custom reference is a systematic [6,4,3] linear MDS array code over GF(2^8), using full primitive polynomial 0x11d. Its guaranteed correction condition is 2e+s<3 per byte lane. The independent whole-payload digest remains the acceptance authority.

The machine-readable candidate register is `classical-ecc-baselines-v0_1.json`. All external entries are documentation/source-reviewed candidates. No external crate has yet been built or run from this branch.

## 2. Important candidate findings

### Pure-Rust `reedsolomon` 0.2.0

The crate documents a configurable systematic RS(n,k) over GF(2^8), up to 255 symbols, with `ReedSolomon::new(parity_len, data_len)`. The docs describe a syndrome → inversionless Berlekamp–Massey → Chien → Forney decoder and mandatory post-correction syndrome verification. The source documents FCR=112 (CCSDS convention). This makes `new(2,4)` a useful candidate for a short [6,4] geometry and `new(85,170)` a candidate long block at 2/3 code rate.

Parameter matching is not yet established: the field polynomial and generator/coordinate convention need source audit and cross-vectors. The crate documents only error correction, not an erasure-location API. Its framed 17-byte CRC header is not FEC-corrected; the raw API leaves code parameters and original length to the caller. The docs explicitly disclose a same-codeword-length/same-FCR parity mismatch that may silently expose parity bytes as data, and an incorrect in-range original length can silently truncate output. These are required negative controls, not reasons to accept the library without testing.

References:
- https://docs.rs/reedsolomon/0.2.0/reedsolomon/struct.ReedSolomon.html
- https://docs.rs/reedsolomon/0.2.0/src/reedsolomon/lib.rs.html

### `fec` 0.2.2

This crate documents a GF(2^8) Reed–Solomon encoder/decoder with configurable primitive polynomial, first consecutive root, generator root gap and parity-root count. It exposes unknown-error decode and `decode_with_erasures`. A 255/170 profile is a candidate equal-rate comparison against the custom 6/4 code, but it operates at different codeword granularity.

There are two points that require executable clarification before use:
1. The encoder documentation says short messages can be encoded with virtual zero padding that is not emitted, but the codeword/block-length descriptions retain a 255-symbol geometry. Do not infer that this produces a valid [6,4] shortened code until the encoded length, decoder input contract and known-answer vectors are tested.
2. The decoder docs describe `min_distance` as the number of parity symbols and say errors up to `min_distance/2` can be corrected, but its erasure method documents `2*errors + erasures < num_roots`. That boundary wording appears inconsistent with the conventional bounded-distance rule when `num_roots` is the parity count. Explicit boundary tests must settle the executable behavior; do not silently normalize the documentation.

The same docs warn that beyond-radius patterns can miscorrect and return a wrong message successfully. The external payload digest is therefore still mandatory.

References:
- https://docs.rs/fec/0.2.2/fec/reed_solomon/struct.Encoder.html
- https://docs.rs/fec/0.2.2/fec/reed_solomon/struct.Decoder.html

### Erasure-only performance references

Intel ISA-L and `reed-solomon-erasure` are useful performance baselines for whole-shard erasure reconstruction, but neither should be scored as an unknown-symbol-error corrector. ISA-L documents the GF(2^8) polynomial as reduction byte 0x1d, corresponding to full polynomial 0x11d; the exact release/commit and matrix convention must be pinned. The Rust erasure crate explicitly requires the caller to detect corruption and mark the bad shards missing.

References:
- https://github.com/intel/isa-l
- https://docs.rs/reed-solomon-erasure/6.0.0/

## 3. Three separate competitions

Do not collapse unlike profiles into one leaderboard.

**Short-code geometry:** custom MDS [6,4,3] versus RS(6,4) only if a baseline's shortening, roots, polynomial, basis, coordinate order and parity layout are verified. Exhaust all six one-symbol error positions, all 15 erasure pairs, and all patterns satisfying 2e+s<3. Include all first-outside-radius cases and known miscorrections.

**Equal-rate longer code:** compare RS(255,170) and any matched custom 2/3-rate construction using the same logical payload, normalized error/erasure fractions and physical mapping. Keep block length, burst localization and decode latency visible; equal rate does not imply equal spatial granularity.

**Erasure throughput:** compare ISA-L and a Rust erasure implementation at identical k/m, shard size, matrix, memory layout, CPU features and resource budgets. This lane measures erasure performance only.

CCSDS RS(255,223) is a separate interoperability profile, not a substitute for RS(255,170) or the custom [6,4,3] reference. Preserve the documented polynomial, first-root and conventional/dual-basis conventions, and qualify standards conformity separately.

## 4. Required test matrix

- pristine round trip for every profile;
- one error at every symbol coordinate;
- all erasure pairs for [6,4,3];
- every mixed error/erasure combination inside the stated correction radius;
- first cases outside the radius, including multi-error and error-plus-erasure;
- wrong field polynomial, root, root gap and symbol basis;
- wrong parity/data lengths, wrong block boundaries, incorrect shortening/padding;
- wrong original length and altered framing/header;
- parity corruption, metadata corruption and valid-codeword miscorrection controls;
- byte-exact equality with an independent finite-field/generator-matrix oracle;
- final out-of-band payload digest comparison.

A decoder may report its own correction count and syndrome result, but it cannot certify the archive. A valid codeword can still be the wrong codeword.

## 5. Fair performance measurement

For each profile, report:
- payload/parity bytes and net code rate including framing;
- encode/decode throughput and p50/p95/p99 latency;
- peak memory and allocation behavior;
- errors, erasures and mixed patterns corrected per codeword;
- uncorrectable and miscorrected outputs;
- CPU model, target features, compiler, build flags and thread count;
- exact input sizes, batch sizes, warm-up, number of repetitions and benchmark harness SHA.

Use the same hardware and resource budgets. Report short-code and 255-symbol profiles separately. No benchmark is comparable until test coverage and output correctness are complete.

## 6. Evidence and qualification status

Current status:
- Custom MDS source parameter 0x11d inspected from exact Symthaea branch.
- Candidate crate docs reviewed and version numbers recorded.
- `fec` error/erasure boundary documentation discrepancy recorded for resolution.
- `reedsolomon` frame/header and raw-parameter mismatch hazards recorded.
- Rust dependency lock, source checksum, full implementation audit, executable cross-vectors, Cargo tests and performance runs remain pending.

No Rust build, codeword comparison, crate correction result, performance result, standard conformity or SOTA improvement is claimed in v0.1.

## 7. Research direction beyond the baseline

The eventual result should be a **decoder-independent channel contract** that allows code families to compete on the same observations and physical geometry. The optimizer may select a code/layout candidate, but:
- the physical owner supplies calibrated raw observations and provenance;
- the decoder emits candidate reconstructed bytes plus a detailed correction receipt;
- an independent scorer verifies byte equality and digest;
- Mycelix assesses operational/common-cause independence;
- the preservation capsule records format parameters, reader contract, decoder source/build identity and migration lineage.

This is how the programme moves beyond “our code beats our old code” toward a reusable qualification system for archival media.
