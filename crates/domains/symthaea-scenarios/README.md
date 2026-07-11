# symthaea-scenarios

**Cross-domain composition** for Symthaea — worked models that combine several
standalone domain crates, proving they compose into more than the sum of their
parts (the antidote to "isolated islands"). Plus invariant/property tests that
harden the numeric crates past hand-picked ground truth.

Transitively pure `std` (all deps are pure-std domain crates), no
`symthaea-core` link.

## Compositions

- `epidemic_cost` — **epidemiology** (SIR peak/final-size) + **economics** (NPV of
  lost output) + **geodesy** (inter-city distance): the cost of an outbreak.
- `controlled_epidemic` — **control theory** (a PID intervention) +
  **epidemiology**: demonstrates the controller flattens the infection curve.

## Property tests (`tests/invariants.rs`)

Laws checked over thousands of randomly-sampled inputs:
- Gini coefficient always in `[0,1]`; zero for identical values.
- SIR conserves `S+I+R` for all parameters; final size in `[0,1]` and zero iff
  `R₀ ≤ 1`.
- NPV is monotonically non-increasing in the discount rate.

```bash
cargo test -p symthaea-scenarios
```
