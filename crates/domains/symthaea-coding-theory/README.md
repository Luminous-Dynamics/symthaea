# symthaea-coding-theory

Small, auditable error-correcting-code foundations for Symthaea.

## Implemented

- Shared block-code parameter metadata with explicit distance and errata bounds
- Caller-defined fail-closed Reed-Solomon decode policies and correction accounting
- Hamming weight and distance
- Hamming(7,4) single-error correction with explicit correction reports
- Hamming(8,4) SECDED with exhaustive single- and double-error tests
- Canonical byte-packed Hamming(7,4) and Hamming(8,4) APIs
- Strict odd repetition codes plus compatibility wrappers and caller-owned buffers
- Reed-Solomon over the AES GF(2^8) field with an explicitly validated
  primitive element (`0x03`), systematic encoding, syndrome detection,
  bounded-distance unknown-error correction, full-budget known-erasure
  recovery, mixed error/erasure recovery under `2e + s <= nsym`, and
  allocation-reusing systematic encoding, and streaming parity accumulation, and fixed `k/n` frame contracts
- Explicit Reed-Solomon interoperability profiles and independent golden vectors
- Explicit shortened Reed-Solomon parent/transmitted frame contracts
- Checked rectangular block interleaving with frame helpers and coordinate maps
- Seeded binary, byte-symbol, independent/fixed-count erasure, exact mixed-errata, and burst channels
- Reproducible end-to-end Hamming(8,4), Reed-Solomon erasure, and mixed-errata experiments
- Analytical independent-error/erasure reliability from the exact `2e + s < d_min` guarantee
- Stable preregistration manifests for exact-count Reed-Solomon evidence campaigns

## Reed-Solomon convention

The sibling finite-field crate uses the AES polynomial `0x11B`. In that field,
`0x02` has multiplicative order 51 and must not be used as the primitive element
for a length-255 Reed-Solomon code. This crate uses `0x03`, validates order 255,
and records the first consecutive root in `ReedSolomonConfig`.

The default generator roots are `alpha^0 .. alpha^(nsym-1)`. Codewords are
most-significant coefficient first and systematic (`message || parity`). See
`docs/REED_SOLOMON_INTEROPERABILITY.md` and `src/interoperability.rs` for the
complete profile and fixed cross-language vectors.

## Failure semantics

Checked APIs reject malformed parameters and non-binary symbols. Reed-Solomon
decoding returns data only after post-correction syndrome verification. As with
all bounded-distance decoders, corruption beyond the advertised correction
radius can be ambiguous; callers needing authenticated integrity must combine
error correction with a cryptographic authenticator.

## Evidence philosophy

Tests emphasize algebraic properties rather than isolated examples:

- all 16 Hamming payloads under every single-bit corruption;
- all 16 SECDED payloads under every pair of flipped bits;
- all Reed-Solomon symbol positions and all 255 non-zero single-error
  magnitudes;
- seeded multi-error trials through the Reed-Solomon correction radius;
- seeded known-erasure trials through the full parity-symbol budget;
- every mixed error/erasure capacity partition, including nonzero first roots;
- deterministic channel experiments with exact rational probabilities;
- exact-weight Reed-Solomon erasure and mixed-errata campaigns at, below, and above capacity;
- parity equivalence across every chunk split and multiple root conventions.

## Burst resilience

`BlockInterleaver` lays equal-length codewords out as rows and emits columns. A
contiguous burst no longer lands entirely inside one frame: a burst no longer
than the row count contributes at most one corrupted symbol to each component
codeword. Interleaving changes error geometry, not code distance, and must be
reversed before component decoding.

## Analytical reliability

`IndependentErrataModel` keeps unknown-error and known-erasure probabilities as
exact rational inputs. `ErrataWeightDistribution` then computes the full
probability mass over `2e+s`, and `estimate_block_code_reliability` reports the
probability covered by the code's minimum-distance guarantee. This is a lower
bound on successful decoding, not a claim that every outside-radius frame must
fail.
