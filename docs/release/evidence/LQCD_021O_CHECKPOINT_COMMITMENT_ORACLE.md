# LQCD-021O — checkpoint commitment oracle

Independent standard-library execution for the checkpoint commitment semantics required by #2816/#2819.

Exact executed subject SHA-256:

`5b94dc418e5ab641f67e303ee8490cf657e40304984b5b8ce58f6cc6a38fd244`

Canonical result SHA-256:

`291c6b60c1ddb7568a19a6cb9010ea41120546811a134e3c9ab167a7a4157155`

## Frozen domains

- record: `symthaea.lqcd.checkpoint.record.v1\0`
- commitment: `symthaea.lqcd.checkpoint.commitment.v1\0`
- seed commitment: `symthaea.lqcd.seed.commitment.v1\0`

The record uses explicit big-endian integer widths and length-prefixed UTF-8 identifiers. It binds campaign subject, chain ID, checkpoint/update ordinals, phase/purpose stream coordinates, exact stream ID, 128-bit ChaCha word position, seed commitment, canonical gauge-field encoding identity and digest, sampler identity/revision, numerical profile, environment profile, and predecessor/genesis state.

## Frozen vectors

- final gauge-transition stream ID: `0x3100021d00070000`
- seed commitment: `48d69193d3fa4b0e9ba5044e097074d06f95889e217e4920dae10badb06c19c4`
- genesis record length: `431` bytes
- genesis commitment: `e5b439c47911a67ecb9d89cc43debb9d5fe9656c4767b9007a5f1b1ec98c5f3c`
- successor record length: `463` bytes
- successor commitment: `84d2328883ae397e179697f364db000180975946aa4b0f38a88d6a6e65b3a6b8`
- bound canonical gauge-field digest: `7bd62ca29833be0171a4f3232c03a3edca125aa50b33aeb187e702fb1c8c996b`

## Adversarial theorem

All 18 authoritative fields are independently mutated and every mutation changes the checkpoint commitment. The oracle also rejects truncated/trailing/wrong-tag records, stream-coordinate mismatch, a genesis checkpoint carrying a predecessor, and a non-genesis checkpoint lacking one.

The exact seed preimage matches its stored commitment; a different 32-byte seed does not. A one-bit field-digest substitution does not match. A child pointing at a non-current predecessor is classified `StalePredecessor`. Two distinct children sharing the same predecessor and checkpoint index produce distinct commitments and are detectable as a sibling fork.

## Scientific boundary

This subject freezes commitment/canonicality semantics only. It does not establish Rust parity, Cargo.lock qualification, filesystem durability, equilibrium, final-sample admissibility, or any physics result.
