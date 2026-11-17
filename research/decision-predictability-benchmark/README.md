# Decision-Relevant Predictability Metrics

Rust metric contracts, a paired-world evaluator fixture, a transparent forecasting tournament adapter and a strict split-manifest validator for [issue #12546](https://github.com/Luminous-Dynamics/symthaea/issues/12546).

## Included

- Qualified Lyapunov-time normalization that rejects any reference exponent not marked `Reliable`.
- Observed forecast-horizon extraction with finite-value and monotonic-time validation.
- Information-set Bayes risk, candidate policy risk, a separately labelled perfect-information lower bound, and regret.
- Expected value of information from expected posterior Bayes risk and acquisition cost in task-loss units.
- Componentwise Pareto dominance over sensing, acquisition, compute, p99 latency, peak/persistent memory, training, update, preprocessing, energy and communication.
- A paired-world fixture in which a physical hazard and sensor-offset-only world generate the same passive primary-sensor observations.
- Separate candidate-visible episode/observation types and evaluator-only mechanism/loss records.
- Probe permission, remaining-budget and decision-deadline admission checks.
- A **Decision-Relevant Predictability Horizon (DRPH)** contract that consumes independently qualified simultaneous upper bounds on decision regret and critical-event risk. It does not construct or certify those bounds.
- Transparent univariate forecasting baselines: persistence, training-window mean, and recursive linear AR(p) fitted with Householder QR and optional ridge regularization.
- A declared uniform-delay quadratic NG-RC reference arm with equally spaced taps and a QR-fitted polynomial readout; not a claim of NGRC-MDS reproduction.
- A bounded train-reference state library that selects the maximum raw dot-product reservoir-state match and returns the selected reference-minus-query correction vector; this is the lookup primitive only, not a complete ESN/DB-RC implementation.
- A CSV adapter/runner that evaluates those baselines against the output contract of the independent chaos-reference generator.

## Run the focused crate

```bash
cargo fmt --manifest-path research/decision-predictability-benchmark/Cargo.toml -- --check
cargo clippy --locked --manifest-path research/decision-predictability-benchmark/Cargo.toml --all-targets -- -D warnings
cargo test --locked --manifest-path research/decision-predictability-benchmark/Cargo.toml --all-targets
```

The standalone workspace commits `Cargo.lock`; all build/test/CLI examples use `--locked` so dependency resolution cannot drift silently. The lockfile was captured from the pinned Rust 1.96.0 focused workflow and checked into the repository.

The package is an isolated Cargo workspace. Metric and baseline code is standard-library-only; the split-manifest module uses exact-pinned Serde/serde_json and SHA-256 dependencies. Its focused GitHub Actions workflow is pinned to Rust 1.96.0.

The uniform-delay NG-RC implementation is a transparent local reference variant: it fixes the delay count/spacing and includes all linear taps plus pairwise quadratic terms. It is not a faithful reproduction of NGRC-MDS and does not replace the separate paper-parity work in #12624.

The chronological tournament selects the declared NG-RC arm on the validation prefix only:
- delay count ∈ {1, 2, 3, 4};
- uniform spacing ∈ {1, 2};
- ridge penalty ∈ {0, 10⁻⁸, 10⁻⁵}.

It minimizes full validation free-running NRMSE with deterministic tie-breaking (score, delay count, spacing, penalty), then refits the selected feature/readout configuration on the 70% development prefix and forecasts the 30% test suffix recursively from one fixed origin. No test-time observations are assimilated. The CSV reports the selected configuration in the model name and the validation NRMSE; the `ar_order` column remains empty for NG-RC. This is a small reproducible baseline search, not a broad hyperparameter sweep or an assertion that this configuration is optimal for every dynamical system.

## Held-out split-manifest V1

The manifest module is the first executable R0/R1 handoff for [issue #12608](https://github.com/Luminous-Dynamics/symthaea/issues/12608).

```bash
cargo run --locked --manifest-path research/decision-predictability-benchmark/Cargo.toml --bin validate-split-manifest -- research/decision-predictability-benchmark/fixtures/split-manifest-v1.json
```

The validator rejects unsupported schemas, unknown/duplicate fields, malformed SHA-256 values or commit identifiers, non-opaque IDs, split-group overlap, inconsistent scenario-to-role assignments, missing axis memberships, unreferenced declared groups, and insufficient primary independent units. It enforces a 16 MiB input limit and 100,000 scenario/group ceilings. It also requires every scenario to carry every axis in the ordered `sampling.nesting` path and ensures each child group maps to exactly one parent group across all scenarios; a parameter-cell ID cannot silently migrate between equation families.

The `sampling.nesting` array is an outer-to-inner path (for example, equation family → parameter cell → trajectory). Every scenario supplies one opaque group ID for every path axis. A child group ID is stable within its parent and cannot be reused under a different parent. Context axes may be shared across split roles; their parent/child hierarchy must still remain coherent.

Canonicalization sorts JSON object keys through the typed Serde representation and preserves array order. The resulting SHA-256 is an integrity identifier, not an authenticity signature or proof that source claims are true. The digest is emitted in a detached receipt: a manifest must never contain its own digest. Seed commitments should use a domain-separated SHA-256 commitment over the seed and a high-entropy secret salt, with seed and salt revealed only after the confirmatory run is sealed. A bare hash of a small integer seed is brute-forceable. Commitments bind a seed choice but do not prove that a generator or solver is correct.

Scenario and group IDs are 32 lowercase hexadecimal characters. The evaluator keeps the mapping from opaque IDs to hidden truth outside the candidate process. Only axes listed in `split_axes` impose cross-role disjointness; other axes in `sampling.nesting` can be attached to each scenario as shared context (for example, the same equation family across train/validation/confirmatory parameter regions). This allows parameter transfer within one known family without accidentally requiring equation-family holdout too. A syntactically valid manifest does not itself establish correct holdout assignment; the manifest must be generated by a trusted evaluator and independently reviewed. The parser validates declared group disjointness and scenario coverage, not the scientific validity of parameter-region boundaries.

The split manifest is deliberately not yet wired to the forecast tournament. That integration follows after the schema and primary independent-unit contract have been independently reviewed.

## Dynamic structural-bias reference primitive

The standalone crate contains `structural_bias::TrainingReferenceLibrary`, a bounded row-major library built from a caller-declared training reference window. For each query state it checks the query dimension and finite values, selects the maximum **raw dot-product** reference (not cosine or Euclidean distance), breaks ties by the earliest index, and returns the reference index, raw similarity and correction vector (reference state minus query state).

The library caps reference count, state dimension and total stored cells, offers an allocation-free output-buffer path and intentionally has no Serde serialization. The caller must enforce that states came only from the frozen training window; this primitive cannot establish provenance by itself.

This is only the retrieval/correction primitive from the 2026 dynamic-structural-bias reservoir method. It does not implement the recurrent reservoir, learned bias matrix, readout, autonomous rollout or full paper training protocol. See #12650 for the full method-parity work.

## Candidate exposure envelope

The candidate-view module provides an explicit, serializable `CandidatePacketV1` and a separate, non-serializable `EvaluatorTicketBook`.

- Candidate packets contain only a per-run opaque ticket, task-contract digest, time unit, as-of time, channel names, allowed action IDs and causal observation frames.
- The evaluator binds a packet only through a `ValidatedManifest`, requires the packet's task-contract digest to match that manifest exactly, and rejects scenario IDs absent from the validated manifest.
- The packet schema has no fields for scenario ID, split role, group memberships, seed commitment, reference-data digest, solver qualification digest, latent state, intervention label or hidden truth. Unknown fields are rejected when deserializing.
- The evaluator ticket book retains the mapping from one-use ticket to the hidden scenario record. Ticket reuse in one book is rejected.
- Observation timestamps must be monotonic and no later than the declared as-of time. Values must be finite when present; missing channels remain explicit.
- Tickets must be fresh uniformly random 128-bit values from an OS CSPRNG per scenario per evaluator run. The library validates ticket shape and collisions; it does not generate randomness or prove entropy.

This is a serialization/API boundary and a defense against accidental metadata leakage, **not an operating-system sandbox**. The evaluator must send only the packet bytes to an isolated candidate process; it must never link, expose, or serialize the ticket book into the candidate environment. Process isolation, filesystem/network restrictions, anti-covert-channel controls and independently generated hidden-truth storage remain separate work (coordinate with #12567). The packet schema does not establish statistical validity or causal identifiability.

The module tests check candidate JSON for forbidden evaluator fields, reject unknown fields, reject observations after as-of time, and reject reused tickets. These are structural conformance tests, not an adversarial process-isolation proof.

## Run the transparent forecasting tournament

The runner consumes the comment/header/CSV output emitted by `research/chaos-reference-suite`. It does not link against that separate crate: this keeps the fixture generator and candidate evaluator independently buildable while the two pull requests are reviewed.

After the chaos-reference-suite package is present in the same checkout:

```bash
cargo run --manifest-path research/chaos-reference-suite/Cargo.toml --bin chaos-fixtures -- 2048 20261116 > chaos-fixtures.csv
cargo run --locked --manifest-path research/decision-predictability-benchmark/Cargo.toml --bin forecast-fixtures -- chaos-fixtures.csv > forecast-report.csv
```

The smoke fixture at `fixtures/forecast-smoke.csv` lets CI test the adapter without depending on the separate generator pull request:

```bash
cargo run --locked --manifest-path research/decision-predictability-benchmark/Cargo.toml --bin forecast-fixtures -- research/decision-predictability-benchmark/fixtures/forecast-smoke.csv > forecast-report.csv
```

### Evaluation protocol v0

- Split each individual trajectory chronologically: 60% training, next 10% validation, final 30% test.
- Forecast the validation and test segments recursively/free-running from one fixed as-of origin. Test observations are never assimilated after forecasting begins.
- Compare persistence, mean, AR(1), AR(2), AR(4), AR(8), and validation-selected AR(order).
- Select the AR order only on the validation continuation; choose the lowest full-validation rollout NRMSE, with lower order winning exact ties.
- Refit each test arm on the first 70% and forecast the final 30% once.
- Report MAE, RMSE, NRMSE (RMSE divided by development-segment population standard deviation), and skill relative to persistence in fixed lead-time bins. Persistence skill is left undefined when the reference RMSE is no greater than 1e-12 times the development signal scale, avoiding unstable ratios when the baseline is already effectively exact.
- Preserve fit/forecast failures and constant channels as explicit status rows rather than silently dropping them. Channels whose population standard deviation is no greater than 16 machine epsilons times their maximum absolute value are treated as numerically unresolved; no NRMSE is reported for those channels.

**Claim boundary:** this is a within-trajectory, same-regime continuation diagnostic. It is not a held-out equation/parameter-family generalization test, it has one trajectory per generator, and it reports no confidence intervals because adjacent forecast leads are correlated. A chronological split prevents future-to-past leakage but does not create independent episodes. Do not treat this report as an SOTA comparison.

The runner reports an FNV-1a input digest labelled `non_security`; it is only a reproducibility convenience, never an authentication mechanism.

## Paired-world evaluator boundary

Candidate-facing code should receive only `CandidateEpisode` and authorized `PrimarySensorObservation` values. Evaluator-only `EvaluatorWorld` records contain latent mechanism and action losses. The Rust type split makes this boundary explicit, but this package is **not a sandbox or process-security boundary**. Confirmatory evaluations must keep evaluator truth in a separate process or permissioned environment and release only authorized observations to candidates.

The canonical fixture currently tests a single pair of worlds and exact deterministic probe outcomes. Its step-based deadline and budget checks are local benchmark controls, not a replacement for capture/arrival clock domains or full observation-event semantics. It does not yet implement noisy/missing/delayed probe channels or a broad world generator.

## Scope and limits

The crate contains reference metrics and transparent diagnostic arms, not a complete benchmark tournament, a forecasting foundation model, a physical sensor system, or a safety controller. Remaining work includes:

- Held-out equation and parameter families, independent seeds and paired uncertainty analysis.
- ESN, NGRC and NGRC-MDS matched baselines, followed by HDC/CfC/LTC and external foundation models.
- Probabilistic forecasts and independently qualified coverage/confidence bounds.
- Irregular timestamp, missingness, delayed observation and event-time adapters through the existing #9700/#12407 owners.
- Noisy active-probe and decision-focused EVSI tournament through #9817.
- Candidate/evaluator process isolation and independent fixture review through #12567.

Resource vectors are comparable only when units, measurement boundaries, hardware and workloads match. This crate does not measure physical energy or hardware performance.

No output from this crate establishes SOTA, physical reliability, causal reconstruction, consciousness or real-world risk guarantees.
