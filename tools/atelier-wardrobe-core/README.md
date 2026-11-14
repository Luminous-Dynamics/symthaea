# atelier-wardrobe-core

A dependency-free exact small-instance oracle for user-declared wardrobe needs.

It enumerates subsets of up to 18 optional choices, always preserves protected (must_keep) items, respects a no-additions policy, and returns a Pareto frontier across new acquisitions, newly manufactured items, user cost, effort and selected actions. It does not impose a universal wardrobe score.

Important boundaries:

- Needs are declared by the user; the crate does not infer social or cultural requirements.
- Compatibility is supplied by the caller; the crate does not judge fit, cleanliness, safety, weather protection or style.
- Owned items, repairs and alterations remain available under a no-additions policy. Borrowed, rented, second-hand, purchased and made-to-order choices are additions.
- Protected items remain in every plan even when they are unusable, but an unusable item cannot satisfy a need.
- Made-to-order production has a separate policy flag; a retail purchase is counted as a new acquisition but is not assumed to have just been manufactured.
- Search beyond 18 optional choices fails explicitly. A future approximate solver must be named and evaluated separately; partial exhaustive work is never returned as an exact answer.
- The Pareto set preserves trade-offs. The user or a separately defined preference layer selects among plans.

Run in a Rust 1.96.0 environment:

```bash
cargo test --manifest-path tools/atelier-wardrobe-core/Cargo.toml
cargo clippy --manifest-path tools/atelier-wardrobe-core/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path tools/atelier-wardrobe-core/Cargo.toml -- --check
```

The tests are authored but have not been compiled or executed in the current authoring environment because the Rust toolchain is absent.
