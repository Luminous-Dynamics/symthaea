# LQCD-015 ChaCha8 lattice stream qualification

## Executed independent subject

A standard-library Python implementation of the ChaCha8 block function and the lattice stream packing/open-interval conversion was executed independently before commit. It imports no Symthaea code and no Rust RNG implementation.

The checked-in oracle is:

`scripts/lqcd-chacha8-stream-oracle.py --self-test`

## Executed results

```text
ok
zero_vector_sha256=a5744c70fa6c816613f354433a8b2c1195ff42447f56d2bd261b043c522bca05
packed_stream_id=0x01123456789abcde
qualification_block=138cca6940325d71f9ab52ed1f9a97283e01c5f699dc6a43545012c55c4e2237197bb7880bd322dc5f207b916e1b5009ead716b18bcd82a5259290fd33db2f9d
qualification_block_sha256=72704a85e0549ad635445eb05a8c406848f03ea421a3980d79eafc94bb6233db
open01_min=2.2204460492503126e-16
open01_max=0.99999999999999978
```

## Reference cross-check

The oracle first reproduces the published ChaCha8 256-bit all-zero-key / 64-bit all-zero-IV block vector:

`3e00ef2f895f40d67f5bb8e81f09a5a12c840ec3ce9a7f3b181be188ef711a1e984ce172b9216f419f445367456d5619314a42a3da86b001387bfdb80e0cfe42`

Only after that reference check does it derive the Symthaea lattice qualification vector.

## Stream semantics frozen

The 64-bit stream identifier is injectively packed as:

`[domain:8 | ensemble_slot:24 | replica:16 | rank:16]`

The qualification coordinates are:

- domain: gauge transition = `1`
- ensemble slot: `0x123456`
- replica: `0x789a`
- rank: `0xbcde`
- packed stream: `0x01123456789abcde`

The Rust side pins `rand_chacha = 0.3.1` and tests its first 64 output bytes against the independent qualification block above.

## U(0,1) mapping

Rust and the oracle map a random `u64` to an endpoint-free 52-bit symmetric grid:

`k = (x >> 12) + 1`

`u = k / (2^52 + 1)`

This guarantees `0 < u < 1`. Complementary 52-bit grid indices map to complementary `u` values, preserving the `u -> 1-u` inverse-measure symmetry used by LQCD-013.

## Non-claims

This establishes deterministic cross-language parity, stream-coordinate separation, replay semantics, and an endpoint-free conversion. It does not prove ideal randomness, independence between arbitrarily selected streams, sampler ergodicity, equilibration, or correct QCD observables. Empirical/random-stream qualification and ensemble convergence remain separate evidence gates.
