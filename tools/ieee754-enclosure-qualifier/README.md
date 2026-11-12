# IEEE-754 Enclosure Qualifier

Qualification-only reference for simple binary64 arithmetic enclosures.

Production remains Rust-first. The Python verifier in `verify.py` is invoked
only in a dedicated qualification workflow; it is not a Rust dependency, runtime
module, service or aircraft application component.

## First release scope

The Rust library evaluates one basic operation over two finite binary64 scalar
operands:

- addition;
- subtraction;
- multiplication;
- division when the denominator is nonzero.

It expands the nearest-rounded result by one adjacent representable value in
each direction using bit-level `next_up` and `next_down`. Overflows yield
one-sided unbounded intervals. Non-finite inputs and division by zero are
rejected. The binary emitter produces deterministic raw input and output
`u64` bit patterns.

The Python qualification oracle reconstructs each operand using its raw bits,
then uses `fractions.Fraction.from_float` to calculate the exact rational result.
It verifies exact containment of the Rust interval endpoints. Decimal JSON
serialization is deliberately not used for f64 evidence.

## Important limitation

This first release qualifies **single basic operations only** under the
toolchain/target used by the workflow. It does not yet qualify interval
composition, bilinear map interpolation, motor power, DC demand, trigonometric
constants, transcendental operations, fused multiply-add, compiler
reassociation or the full map/drive implementation. A multi-step expression
must be composed operation by operation with interval operands; adding one ULP
only to the final ordinary f64 result is not an accepted substitute.

The argument for one-neighbour expansion is conditional on the target's basic
binary64 operation being correctly rounded as expected by the implementation.
The exact-rational test is intended to detect enclosure defects over a frozen
bit-pattern corpus; it is not a universal proof of every CPU, compiler, target,
or floating-point environment. Toolchain, target triple, vector identity and
output digest must be recorded by the qualification job.

## Local qualification

```bash
cargo fmt --manifest-path tools/ieee754-enclosure-qualifier/Cargo.toml -- --check
cargo test --locked --manifest-path tools/ieee754-enclosure-qualifier/Cargo.toml
cargo clippy --locked --manifest-path tools/ieee754-enclosure-qualifier/Cargo.toml --all-targets -- -D warnings
cargo run --locked --release --manifest-path tools/ieee754-enclosure-qualifier/Cargo.toml --bin emit-vectors > /tmp/ieee754-vectors.txt
python3 tools/ieee754-enclosure-qualifier/verify.py /tmp/ieee754-vectors.txt
```

All output values and inputs are transmitted as 16-digit hexadecimal binary64
bit patterns. The verifier reports the SHA-256 digest of the exact vector file.

## Integration path

The next stages belong to
[AIRROAD-INTERVAL-QUALIFICATION-001 #11703](https://github.com/Luminous-Dynamics/symthaea/issues/11703):

1. Add interval-operand composition and exact-rational checking.
2. Add map-corner extrema and coordinate-rectangle vectors for #11697.
3. Add motor-shaft, motor-efficiency, inverter, bus-current and energy propagation
   vectors for #11699.
4. Pin and qualify the complete end-to-end integration in #11702.
5. Keep all physical measurement, calibration, aircraft performance, safety and
   airworthiness claims outside the scope of this software-only qualifier.

## Research references

- NIST Digital Library of Mathematical Functions, §3.1(ii), interval arithmetic:
  https://dlmf.nist.gov/3.1.ii
- #2701, internal precedent for a Python `Fraction` exact-rational qualification
  oracle independent of Rust interval arithmetic.
