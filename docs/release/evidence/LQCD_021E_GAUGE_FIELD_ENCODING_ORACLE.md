# LQCD-021E — canonical Wilson gauge-field encoding oracle

Independent standard-library Python subject freezing the byte representation used for persistent gauge-configuration identity and restart checkpoints.

Exact executed subject SHA-256:

`f4b035a98c7a9d058b19fdd7207bfc916bfea70f2c3788cd964b6120377f90be`

Canonical result SHA-256:

`8c368f435da7bf9d1e0bf3d8aff2772081ba0f6dc32222578656edf8b842f0c9`

## Canonical bytes

Stable encoding ID: `wilson_gauge_field_be_f64_v1`.

Bytes begin with the exact domain tag `symthaea.lqcd.wilson-gauge-field.v1\0`, followed by four positive lattice extents as big-endian `u32`, followed by every link in this exact order:

`x -> y -> z -> t -> mu -> row -> col -> real -> imaginary`.

Each real/imaginary scalar is the raw IEEE-754 binary64 bit pattern encoded as big-endian `u64`. No decimal formatting, locale, JSON floating rendering, or host endianness participates in configuration identity.

The decoder requires the exact byte length implied by the dimensions, rejects zero extents and non-finite payload scalars, and rejects trailing/truncated data or the wrong domain tag.

## Executed fixtures

The frozen `2x1x1x1` field has 8 links and therefore exactly 1204 bytes under this encoding.

- identity field SHA-256: `bda8ba8c4b0e8a0c6c906edddb327be77ed37bfc1b3a4d196e1a5990429cbefa`;
- nontrivial exact-SU(3) fixture SHA-256: `7bd62ca29833be0171a4f3232c03a3edca125aa50b33aeb187e702fb1c8c996b`;
- same exact link-value multiset moved to different canonical link positions: `e30e407745ad2f06dde4d73a8f8e71c2a1a00ca93d8c0e46c4ecaeafaf98f8b4`.

The nontrivial fixture uses only exact `{0,+/-1,+/-i}` SU(3) entries, avoiding a cross-language `sin/cos` or libm dependency. Encode -> decode -> encode reproduces exact bytes, and every decoded scalar reproduces its original binary64 bits. A dedicated fixture proves negative zero remains `0x8000000000000000` rather than being normalized to positive zero.

## Authority boundary

The canonical SHA-256 is over the complete encoded field bytes. This is suitable as the `gauge_field_digest` committed by #2784 and later `ConfigId` work.

This oracle freezes persistence representation only. It does not establish that the Rust production decoder accepts the bytes, does not prove actual disk crash durability, does not establish chain equilibrium, and does not itself authorize a retained configuration.
