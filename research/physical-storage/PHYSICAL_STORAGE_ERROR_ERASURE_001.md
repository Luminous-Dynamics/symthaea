# Physical Storage Error-and-Erasure Decoder — v0.1

**Programme:** PHYSICAL-STORAGE-ERROR-ERASURE-001  
**Tracking:** [Symthaea #13712](https://github.com/Luminous-Dynamics/symthaea/issues/13712)  
**Parent:** [PHYSICAL-STORAGE-CODING-001 #13630](https://github.com/Luminous-Dynamics/symthaea/issues/13630)  
**Reference implementation:** [standalone Rust MDS reference](https://github.com/Luminous-Dynamics/symthaea/tree/research/physical-storage-error-erasure-001/research/physical-storage/mds-reference)  
**Physical-channel owner:** [Symtropy #7477](https://github.com/Luminous-Dynamics/symtropy/issues/7477)

## 1. The gap this closes

The v0.1 reader reconstructed tiles explicitly marked unavailable. It detected an unflagged byte substitution only at the end-to-end SHA-256 check; it did not correct noisy symbols. This increment adds a bounded hard-decision error-and-erasure decoder to the existing synthetic [6,4,3] MDS array code.

This matters because a physical reader may return a plausible but incorrect symbol rather than a clear erasure. Glass phase-state confusion, low contrast, read noise, registration error and threshold drift can all produce uncertain or wrong observations. The current code does not model those physical causes; it now tests the narrow mathematical decoding contract for unknown symbol errors.

## 2. Exact coding guarantee

The code has parameters n=6, k=4, minimum distance d=n-k+1=3, over GF(256) using polynomial 0x11d.

For each byte offset independently, the six shard bytes form one codeword. The guaranteed bounded decoding region is:

**2e + s < d = 3**

where:
- e is the number of observed symbols that disagree with the original codeword but were not identified as erasures;
- s is the number of known missing symbols;
- the counts apply **per byte-offset lane**, not to the whole 8-byte physical tile.

Consequences:
- e=0, s≤2: guaranteed;
- e=1, s=0: guaranteed;
- e=1, s=1: outside the guarantee;
- e≥2, s=0: outside the guarantee.

A complete 8-byte tile substitution is eight symbol errors across eight parallel byte lanes. With no erasures, one error in each independent lane remains correctable. That does not imply that arbitrary eight-symbol corruption in one lane is correctable.

## 3. Decoder algorithm

For each byte offset in a stripe:

1. Enumerate every four-shard subset among observed symbols (at most C(6,4)=15 subsets).
2. Invert the selected four generator rows and recover the four data symbols.
3. Re-encode that candidate into all six symbols.
4. Count disagreements across all observed symbols.
5. Retain candidates satisfying 2e+s<3.
6. Require one unique data/codeword candidate. No candidate or ambiguity is an uncorrectable result.
7. Assemble the recovered bytes and compare the complete payload against an out-of-band SHA-256 digest.

Because the code is MDS, every four-row generator submatrix is nonsingular. The enumeration is a deliberately small correctness reference. It is not a throughput implementation; a later optimized decoder should use a pinned, independently tested coding library or a standard decoder.

## 4. Why the digest remains indispensable

A bounded decoder cannot promise to reject every pattern outside its guaranteed radius. With minimum distance three, two unknown errors can move a received vector to distance one from a different valid codeword. The decoder can then produce a mathematically valid but incorrect reconstruction.

The negative-control test deliberately creates this case. The decoder returns a nearby codeword, records one corrected symbol, and the independent whole-payload digest rejects the output.

Therefore:
- ECC establishes whether a candidate lies within the stated decoding radius;
- SHA-256 checks whether the entire reconstructed payload equals the expected payload;
- neither a valid codeword nor a digest alone authenticates who supplied the payload or manifest;
- exact archive acceptance requires both successful decoding and digest equality, with authenticated manifest provenance handled separately.

## 5. Deterministic test matrix

The integration suite covers:
- all 48 single-error positions: six shard positions × eight byte offsets;
- all 15 pairs of known erasures in a stripe;
- one full eight-byte tile substitution, corrected as eight independent byte-lane errors;
- one unknown error plus one known erasure, rejected for the selected over-budget fixture;
- a two-error miscorrection negative control, rejected by the external payload digest.

The machine-readable parameterized fixture is mds-reference/error_erasure_cases_v0_1.json. The tests use deterministic data and no physical probability model.

## 6. Benchmark and industry comparison

This decoder is a correctness increment, not a claim to beat SOTA. The next fair tournament should compare:

1. the existing erasure-only reference;
2. this bounded enumerative MDS decoder;
3. a pinned standards Reed–Solomon/GRS implementation;
4. a pinned erasure-code implementation such as Intel ISA-L;
5. BCH / LDPC where the optical reader exposes a suitable binary or soft-information channel;
6. interleaving and code/layout co-design variants under identical net payload, fault masks and evaluation budgets.

Freeze code parameters, symbolization, geometry, payload, erasure/error locations and compute budget. Report:
- exact payload acceptance rate and wrong-payload rejection;
- corrected symbol count, erasures and uncorrectable stripes;
- net payload efficiency after parity, controls, calibration and spares;
- throughput, tail latency, memory and energy only with a declared hardware boundary;
- worst-case and held-out spatial fault-set coverage;
- performance by evidence class: synthetic, measured or modelled.

Relevant external anchors are the CCSDS TM Synchronization and Channel Coding standard's use of Reed–Solomon and LDPC options, and Intel ISA-L's published GF(2^8) Reed–Solomon-type erasure-code routines. These are baseline families to pin and benchmark, not evidence that this custom array code is standards-conformant or faster.

## 7. Next paradigm shift: channel-aware soft information

The most valuable path beyond hard-decision decoding is to preserve the reader's uncertainty. Symtropy #7477 should export spatially registered per-symbol observations with:
- raw read values and calibration identity;
- symbol confusion matrices by material, depth, batch and reader;
- calibrated confidence or likelihood values;
- coordinate-registration uncertainty and missingness;
- instrument, specimen and process lineage.

Symthaea can then compare hard-decision ECC with soft-decision decoding and generalized-minimum-distance style strategies. Probabilities must be estimated from repeated measured observations; synthetic masks remain deterministic sensitivity probes. The optimizer must not fit and evaluate on the same specimens.

## 8. Evidence boundaries

This v0.1 increment establishes only a deterministic algorithmic property of the custom [6,4,3] linear MDS code. It does not establish:
- physical optical or magnetic channel error rates;
- measured glass symbol confusion or soft-decision calibration;
- physical storage density, retention, endurance or manufacturing yield;
- reader interoperability or archive restore SLA;
- a performance or reliability advantage over established codes.

The code is not Reed–Solomon/GRS. Hashes are not signatures. Any future physical writing/reading experiment requires qualified laboratory operators, specialist instruments and their hardware safety interlocks.

Rust formatting, strict Clippy, Cargo tests and executable runner output must be recorded against the exact branch head. They are not claimed by independent algebra checks.
