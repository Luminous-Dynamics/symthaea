# ROB-BENCHMARK-VALIDITY-001 — paired independent-cluster comparison reference

Status: first executable benchmark-validity component. This is a deterministic statistical-accounting reference over caller-supplied cluster summaries. It is not a general statistical package, benchmark certification, robot controller or safety authority.

Parent: [Symthaea #11775](https://github.com/Luminous-Dynamics/symthaea/issues/11775). Composes with matched cross-vendor comparisons [#11774](https://github.com/Luminous-Dynamics/symthaea/issues/11774), accepted-work accounting [#11607](https://github.com/Luminous-Dynamics/symthaea/issues/11607), Symtropy future-option evaluation [#4588], option-loss ledger [#4908], and single-joint metrology [#4783].

## Research question

Can a benchmark comparison prevent correlated ticks, repeated samples from one robot, or multiple episodes from one collection session from being counted as independent evidence of superiority?

The reference uses one matched, predeclared **independent cluster** as the statistical unit. Each cluster may contain many episodes and many control ticks; those are retained as denominators and descriptive counts, but they do not increase the statistical sample size.

## Implemented first endpoint

The initial primary endpoint is **trajectory-safe success rate**:

`safe_successes / all_episodes`

The denominator includes task failures, timeouts, Unknown outcomes and censored episodes. Unknown and censored counts remain visible and prevent a confirmatory directional claim until adjudicated. Task success that violates an applicable safety constraint is not a safe success. Critical safety violations are reported separately. Any per-cluster candidate regression blocks a directional claim, and any observed candidate critical violation blocks a positive safe-improvement disposition even when the baseline has more violations.

The paired effect is calculated once per independent cluster. The report gives the lower/upper median paired safe-success delta (PPM) over qualified independent clusters and an exact one-sided sign test on non-tied paired cluster differences. The test evaluates a pre-registered directional hypothesis that the candidate improves the median cluster-level safe-success rate. Ties are reported and excluded from the sign-test denominator. This is not a mean-effect test, a confidence interval, a non-inferiority test, or a universal safety proof.

The p-value is reported as an exact numerator/denominator pair. A Bonferroni adjustment is applied using the declared number of confirmatory candidate comparisons on the hidden split.

## Frozen comparison scope

The input binds the task, scenario, robot article/configuration, initial-state, evaluator, resource budget and held-out split manifests. The split must be the hidden confirmatory split for a confirmatory result.

Two comparison modes are defined:
- `SameModelCausalAblation`: baseline and candidate must share the exact model digest; this isolates a system/policy component under matched model identity.
- `BestAvailableProduct`: model identities may differ, but the declared shared task, robot article, initial state, evaluator and resource budget must still match. This is not a claim that dissimilar vendors or embodiments are comparable.

Each cluster carries one paired episode-set digest and an independence receipt from the benchmark design owner. The evaluator checks the receipt is present and its declared disposition is Qualified. It cannot prove that clusters are truly independent; that remains an experimental-design and adjudication responsibility.

## Denominator and anti-pseudoreplication rules

For each arm and cluster, episode outcomes form mutually exclusive categories:
- task successes;
- task failures;
- timeouts;
- Unknown outcomes;
- censored outcomes.

These must sum exactly to the total episode count. Safe successes are a subset of task successes. Critical safety violations are tracked separately because they can coexist with task completion or failure.

The comparator requires baseline and candidate to share the exact episode count and paired episode manifest within each cluster. Multiple episodes or millions of control ticks inside one cluster never turn into extra independent clusters. The exact sign-test implementation is explicitly bounded to at most 100 independent clusters; excess input is rejected rather than truncated.

## Claim ceiling

A QualifiedDirectionalImprovement result means only that the supplied independent-cluster receipts and matched summaries pass the deterministic contract, and the exact sign test crosses the declared multiplicity-adjusted alpha. It does not prove the independence receipts are true, the evaluator is valid, the safety specifications are complete, or a robot is physically safe.

The reference should be used as one gate in #11775, alongside probe/shortcut controls, hidden-split and repeated-selection accounting, provenance-group holdouts, trajectory-level safety evaluation, and physical qualification. No benchmark-wide SOTA claim follows from this crate alone.

## Running the reference

The crate is dependency-free and isolated from the application workspace:

```sh
cargo fmt --manifest-path research/rob-benchmark-validity-001/reference/Cargo.toml -- --check
cargo test --locked --manifest-path research/rob-benchmark-validity-001/reference/Cargo.toml
cargo clippy --locked --manifest-path research/rob-benchmark-validity-001/reference/Cargo.toml --all-targets -- -D warnings
```

Exact-head Rust 1.96 GitHub Actions is the qualification gate. Authored tests are not passes until that workflow succeeds. The current source contains 20 authored deterministic tests.
