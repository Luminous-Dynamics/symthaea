# atelier-wardrobe-core

A dependency-free exact small-instance oracle for user-declared wardrobe needs.

It enumerates subsets of up to 18 optional choices, always preserves protected (`must_keep`) items, respects independent policies for borrowing, rental, second-hand, retail purchase and made-to-order, and returns a Pareto frontier across new acquisitions, newly manufactured items, user cost, effort and selected actions. It does not impose a universal wardrobe score.

## Cost and unit rules

- All choice costs use integer `cost_minor_units` and an explicit three-letter uppercase `currency_code`.
- The `WardrobePolicy` declares the planning currency.
- Every choice in one search must use that exact currency code; mixed currencies are rejected rather than compared or silently converted.
- The crate validates the code's three-letter uppercase shape only. The caller must verify it against the appropriate currency registry and supply any conversion evidence before creating a single-currency planning corpus.
- Effort is expressed in minutes. Do not mix estimated, observed or differently scoped effort without recording the source upstream.

## Important boundaries

- Needs are declared by the user; the crate does not infer social or cultural requirements.
- Compatibility is supplied by the caller; the crate does not judge fit, cleanliness, safety, weather protection or style.
- Owned items, repairs and alterations are always eligible. Borrowing, renting, second-hand acquisition, retail purchase and made-to-order are separately enabled or disabled by policy.
- A no-purchase policy can still permit borrowing, rental or second-hand use. A strict no-acquisition policy disables all acquisition routes while retaining owned items, repair and alteration.
- Protected items remain in every plan even when they are unusable, but an unusable item cannot satisfy a need.
- Made-to-order production has a separate policy flag; a retail purchase is counted as a new acquisition but is not assumed to have just been manufactured.
- Search beyond 18 optional choices fails explicitly. A future approximate solver must be named and evaluated separately; partial exhaustive work is never returned as an exact answer.
- The Pareto set preserves trade-offs. The user or a separately defined preference layer selects among plans.

## Run

In a Rust 1.96.0 environment:

```bash
cargo test --manifest-path tools/atelier-wardrobe-core/Cargo.toml --locked
cargo clippy --manifest-path tools/atelier-wardrobe-core/Cargo.toml --locked --all-targets -- -D warnings
cargo fmt --manifest-path tools/atelier-wardrobe-core/Cargo.toml -- --check
```

Twelve unit tests are authored, including mixed-currency and malformed-currency-code rejection. They have not been compiled or executed in the current authoring environment because the Rust toolchain is absent.
