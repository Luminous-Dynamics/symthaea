# ASSB practical-cell translation audit v0.1

Dependency-free Rust 1.96 read-only audit for complete-cell solid-state battery translation comparisons.

Checks:
- separate fabrication and operating pressure;
- pressure-only comparison requires identical instrument and measurement method, with the operating-pressure contrast strictly greater than combined measurement uncertainty;
- reports the exact pressure instrument and combined uncertainty;
- declared pressure measurement method, instrument and uncertainty;
- complete-cell mass closure including inactive electrode fractions, plus a separately digested external-volume boundary;
- same design revision, batch, manufacturing line and process revision across pressure-only comparisons;
- fabrication-pressure matching within combined measurement uncertainty and identical pressure measurement method;
- evidence rights reference;
- matched conditions for pressure comparisons;
- technology-lane and full/symmetric/half-cell separation;
- evidence admission, physical vs synthetic and right-censored outcome semantics;
- deterministic reports.

The synthetic example compares two complete-cell sulfide designs at 5.0 MPa and 0.3 MPa operating pressure while holding the other declared conditions constant. It does not represent measured battery performance.

## Commands

    cargo metadata --manifest-path tools/assb-practical-translation-audit/Cargo.toml --locked --no-deps
    cargo fmt --manifest-path tools/assb-practical-translation-audit/Cargo.toml -- --check
    cargo test --manifest-path tools/assb-practical-translation-audit/Cargo.toml --locked
    cargo clippy --manifest-path tools/assb-practical-translation-audit/Cargo.toml --all-targets --locked -- -D warnings
