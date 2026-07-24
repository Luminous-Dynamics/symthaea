# Reed-Solomon interoperability contract

The bytes in a Reed-Solomon codeword are not interoperable unless every
algebraic and serialization convention agrees.

This crate's canonical profile is:

| Parameter | Value |
|---|---|
| Field | GF(2^8) |
| Irreducible polynomial | `0x11B` (`x^8 + x^4 + x^3 + x + 1`) |
| Primitive element | `0x03` |
| Primitive-element order | 255 |
| First consecutive root | configurable, default 0 |
| Generator roots | `alpha^fcr .. alpha^(fcr + nsym - 1)` |
| Polynomial coefficient order | most-significant coefficient first |
| Codeword layout | systematic `message || parity` |
| Maximum codeword length | 255 symbols |

`src/interoperability.rs` exposes a complete `ReedSolomonProfile`, a stable
profile identifier, and four fixed golden vectors. Implementations in another
language should reproduce the generator and codeword bytes exactly before any
cross-system compatibility claim is made.

The profile identifier intentionally names all choices that commonly cause
silent incompatibility. For example:

`rs-gf256-p11b-g03-fcr00-msb-systematic-nsym8`

A receiver must not infer these values from `GF(256)` or `RS(255, k)` alone.
