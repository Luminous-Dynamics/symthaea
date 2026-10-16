# SYM-FIN-SYS-016 — Benchmark targets and qualification gates

**Status:** Proposed benchmark contract. No performance result or economic finding is claimed.  
**Applies to:** socioeconomic interoperability laboratory, run receipt validator, and future simulator.  
**Principle:** qualify correctness before throughput; qualify simulation behavior before comparative claims about real economies.

This document defines target thresholds, not achievements. Numeric performance targets are initial hypotheses to be ratified or revised after a pinned reference-runner baseline exists. Changing a target must be versioned and justified, never silently relaxed to make a run pass.

## 1. Qualification ladder

A higher tier cannot compensate for a lower-tier failure.

| Tier | Question | Proposed target | Required evidence |
|---|---|---|---|
| Q0 — Receipt integrity | Is the receipt structurally and semantically valid? | Positive fixture accepted; **26/26** pinned negative controls rejected by their intended layer; **11/11** authored unit tests pass when actually executed | Exact source SHA, dependency lock, validator log, control-by-control result |
| Q1 — Artifact identity | Do declared input digests resolve to the bytes used? | **100%** of required input digests resolve to exact bytes or are explicitly classified as unavailable/unverified; **0** unlabelled placeholders accepted as verified | Artifact manifest, digest algorithm, exact-byte hash report, missing-artifact register |
| Q2 — Deterministic replay | Does identical input reproduce identical behavior? | **100/100** same-runner replays per frozen benchmark produce the same ordered event-log digest and final-state digest | Seed, runner fingerprint, source revision, environment and event-log digests |
| Q3 — Mechanism invariants | Can transitions violate conservation or authority boundaries? | **0** invariant violations in 100,000 generated transition sequences; **0** duplicate settlement successes; **0** capacity overcommit; **0** synthetic/unknown evidence promotions | Property-based test report, minimized counterexamples, independent accounting and power oracles |
| Q4 — Stress and comparison | Are effects robust across regimes and shocks? | Execute the staged common-seed experiment matrix in §4; publish every run, failure and exclusion | Frozen preregistration, all seed outcomes, uncertainty intervals, counterexample register |
| Q5 — Performance | Is the implementation useful at declared scales? | Meet the provisional latency and memory targets in §5 on a pinned runner | Raw timing samples, environment fingerprint, p50/p95, peak RSS and run manifest |

A pending, queued, skipped, cancelled or unobserved job does not satisfy a gate. A source-level test count is not an execution result. Q0 should be rerun after every relevant schema, validator, fixture, corpus, or dependency change.

## 2. Receipt and provenance benchmarks

### Q0 — Fail-closed receipt checking

The current contract pins 26 negative controls: 10 JSON-Schema failures and 16 semantic failures. On a real validator run:

- All 10 schema-level cases must be rejected by JSON Schema.
- Each of the 14 semantic cases must remain schema-valid, then be rejected by semantic validation.
- The positive fixture must pass both layers.
- Negative-control IDs, layer assignments, count, metadata, and entry shape must match the versioned corpus exactly.
- Duplicate JSON keys, non-standard numeric constants, floating-point overflow, and non-zero underflow must be rejected before semantic evaluation.

Report false acceptance of invalid fixtures and false rejection of valid fixtures separately. Target both at **0%** over the frozen suite. Do not add a control that is already invalid at an earlier layer when the test is intended to prove a later-layer guard.

### Q1 — Hashes identify artifacts, not merely plausible strings

A digest-shaped string is not proof that the referenced artifact exists. The next provenance increment should introduce a versioned manifest that distinguishes:

1. **Repository source artifacts:** repository path, source revision, hash algorithm, digest of exact checked-out bytes.
2. **Run-generated artifacts:** content-addressed payload or immutable run-bundle entry, exact byte digest, producer and run identity.
3. **External or unavailable artifacts:** explicitly missing, out of scope, or unverified; never silently treated as verified evidence.

Acceptance target: every digest-bearing field that claims to identify an input maps to exactly one declared artifact; every required artifact exists and hash-matches; paths and artifact IDs are unique; undeclared or unexpected bindings fail closed. Keep evidence-integrity status separate from origin authenticity and real-world truth.

Avoid circular hashing: an artifact manifest may hash source inputs while the run receipt records the manifest digest; the manifest must not include a digest of the receipt that itself includes the manifest digest. Define whether hashes apply to exact bytes or canonical JSON. If canonical JSON is used, select and test a named algorithm such as [RFC 8785 (JCS)](https://www.rfc-editor.org/rfc/rfc8785.html) rather than inventing serialization rules.

Adopt established provenance patterns where useful instead of implying a bespoke format is independently authenticated. [in-toto attestations](https://github.com/in-toto/attestation/blob/main/spec/v1/README.md) separate statement, subject, predicate and envelope; [SLSA provenance](https://slsa.dev/spec/v1.0/provenance) describes how an artifact was produced. Their existence does not automatically authenticate this research receipt or prove an economic claim.

### Provenance and trust must remain different

A hash can show that bytes match a claimed digest. A signature can help establish who signed a statement under a specific key policy. Neither alone proves that a physical delivery occurred, an economic claim is true, or a policy is legitimate. Preserve distinct fields and gates for byte integrity, origin, issuer authentication, freshness/revocation, independent observation, evidence status, policy authority and settlement finality.

## 3. State-machine and accounting invariants

The future simulator and Mycelix adapter should expose independent oracles for these invariants:

- **Double-entry balance:** for each instrument and posting, total debits equal total credits in the declared atomic unit.
- **No replay settlement:** an idempotency/transaction key produces at most one successful posting; retries return the prior result or an explicit indeterminate state.
- **Capacity conservation:** available + reserved + consumed equals declared capacity after accounting for explicit release/adjustment events; no transition makes consumed capacity available again.
- **Claim conservation:** in a closed conversion cycle with no declared subsidy, net new claims equal zero. With a subsidy, net creation cannot exceed that declared subsidy under the declared units and finality profile.
- **No semantic promotion:** a synthetic receipt, unknown-origin evidence, a recommendation, a reputation value, or external recognition alone cannot produce real authority, entitlement, capacity, observed delivery, or settlement.
- **Finality monotonicity:** observing or transmitting one leg of a cross-domain transfer must not silently make the other leg final. A timeout yields indeterminate unless the named finality profile proves otherwise.
- **Stock/flow reconciliation:** each ending stock equals starting stock plus declared inflows minus outflows and write-offs in the same unit.
- **No hidden metric substitution:** metric ID, dimension, unit, population scope, period, measurement status and missingness must match the registry/receipt contract. Incompatible measures must not be silently compared.

Target: **zero invariant violations** across 100,000 generated transition sequences and 10,000 adversarial settlement attempts. These counts are proposed campaign sizes, not current results. Every failure must retain the smallest reproducible sequence, seed, source revision and before/after state.

## 4. Socioeconomic mechanism benchmark matrix

Do not crown one economic regime from a weighted scalar. Treat rights and ecological limits as hard, preregistered constraints; report outcomes as a vector with distributions, uncertainty, missingness and trade-offs.

### Stage A — Baseline and shock calibration

Use all **8** registry regime profiles and all **15** declared shock families. Run **100 seeds per regime/shock pair**: 8 × 15 × 100 = **12,000 runs**. Use common random numbers for paired comparisons wherever the same event sequence is meaningful. Reserve at least 20% of seeds as a held-out set and do not tune mechanisms against held-out results.

### Stage B — Directed interoperability comparisons

For every ordered pair of distinct regimes, test the three bridge profiles that alter exchange/conversion/finality most directly: trade-only, explicit conversion, and synchronized settlement. With eight regimes this is 8 × 7 × 3 × 100 = **16,800 runs**. Keep no-bridge controls and report bridge removal as a paired counterfactual. Test mutual recognition, shared commons and transition corridors in separate profile-specific campaigns rather than pretending they have identical semantics.

### Stage C — Power and recovery attacks

Cover every one of the **8** declared power attacks, at minimum across the baseline mixed world and the directed-bridge configurations most exposed to the attack. Include verifier cartel, bridge monopoly/capture, dominant supplier, housing allocator capture, information monopoly, governance agenda capture, infrastructure control and emergency-authority persistence.

Measure not just whether the system “recovers,” but:
- essential-service degradation and worst-decile access;
- single points of failure, bridge concentration and substitute depth;
- time to detect, time to bounded containment, time to restore safe operation;
- unresolved/indeterminate transactions and permanent losses;
- monitoring burden, privacy costs, and who can appeal or exit.

Fault recovery must never count as successful if it restores throughput by violating a rights floor, ecological bound, or authority boundary.

### Stage D — Cross-checks and reporting

At minimum:
- rerun with mutation-order permutations and repeated seeds to identify order-dependent outcomes;
- compare the primary accounting oracle to a separately implemented reconciliation oracle;
- compare primary power/concentration calculations to an independent oracle;
- include fixed-institution, no-bridge, proposal-ablation, adoption-ablation, random-mutation and bridge-removal controls;
- publish all failed, unstable, censored and excluded runs with reasons.

For comparative claims, preregister the primary estimands, exclusion rules and analysis. Report effect sizes and confidence intervals, multiple-comparison handling, seed-level distributions, sensitivity to model parameters and the held-out result. A simulation establishes behavior within the model; it does **not** by itself establish real-world welfare, incentive compatibility, ecological sufficiency, political legitimacy or causal effects in an actual economy.

## 5. Provisional performance targets

Performance gates apply only after behavior and workload are frozen. Report the exact hardware, OS, compiler/runtime, dependencies, seed set and workload size. Use at least 100 measured repetitions per workload after warm-up; retain raw timings and peak memory. Proposed targets on a pinned reference runner with **4 CPU cores and 8 GiB RAM**:

| Workload | Provisional target |
|---|---:|
| Receipt validation: positive fixture + 26 controls | p95 ≤ 2 seconds |
| Baseline world: 100 people, 20 projects, 4 communities, 8 scarce-resource classes, one scenario horizon | p95 ≤ 2 seconds |
| Medium world: 1,000 people, 200 projects, 8 communities, 100 ticks | p95 ≤ 10 seconds |
| Stress world: 10,000 people, 2,000 projects, 32 communities, 1,000 ticks | p95 ≤ 120 seconds and peak RSS ≤ 2 GiB |

These are starting targets, not measured claims. If the first reproducible baseline makes one unrealistic, record the baseline and propose a versioned target change with evidence; do not alter correctness gates or hide regressions to satisfy timing goals. Scale runs should also report cost per run and throughput under parallelism, but parallel execution must not change deterministic outputs.

## 6. Reporting contract

Every published benchmark report must include:

- repository, exact source revision and dirty-tree status;
- receipt, scenario registry, artifact manifest, model, agent, world seed, shock schedule, policy and bridge-profile digests;
- runner fingerprint and dependency/toolchain lock;
- test/negative-control totals, observed outcomes and unexecuted/queued/skipped categories;
- random seeds, raw output artifacts, failed runs and minimized counterexamples;
- p50/p95 latency, peak memory and repetitions for performance;
- metrics with stable identifiers, unit, population scope, period, measurement status and uncertainty;
- preregistered estimands, controls, held-out split, effect sizes and confidence intervals;
- explicit claim ceiling: synthetic-model evidence only, unless independent external evidence is separately collected and verified.

Use consistent metric semantics and units; [OpenTelemetry metric conventions](https://opentelemetry.io/docs/specs/semconv/general/metrics/) provide prior art on units and avoiding ambiguous metric names. The socioeconomic registry can adopt compatible naming/unit conventions without pretending to be an OpenTelemetry implementation.

## 7. Current baseline status

As of the current design revision:

- The source contains 26 negative-control definitions and 11 authored Python unit tests. Those counts are not evidence that the tests executed.
- A structural mirror checked the fixture's registry digest and rejected the 16 semantic mutations; this is not execution of the Python validator or JSON Schema engine.
- The run-receipt validator now checks an exact-byte artifact manifest: it re-hashes a pinned set of repository source files, binds every digest-shaped receipt field to exactly one JSON-pointer binding, verifies the receipt’s manifest digest, and explicitly labels unavailable run-generated payloads unverified. Of the 19 digest-shaped fixture fields, the scenario-registry digest is the only one tied to repository bytes; the other 18 are illustrative values and remain unverified.
- No 12,000-run or 16,800-run simulator campaign has been executed by this document.
- The performance targets are proposed and unmeasured.

The success criterion is not simply more tests or faster simulations. It is a falsifiable, replayable mechanism map with auditable inputs, explicit failures, invariant-preserving coordination, and honest uncertainty.
