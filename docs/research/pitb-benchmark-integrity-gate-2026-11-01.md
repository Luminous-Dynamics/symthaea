# PITB research track: benchmark target integrity and semantic validity (2026-11-01)

**Status:** target-integrity receipt schema, Rust verifier and synthetic no-run fixture added. No public benchmark has yet been audited by PITB, and no task is marked eligible from measured evidence.

## Why this is a first-class capability

A Lean kernel proves that a proof term inhabits a formal type. It does not prove that the formal type faithfully represents the intended mathematical or software requirement. A benchmark can therefore be machine-checkable and still measure the wrong thing.

The ICML 2026 audit *Faults in Our Formal Benchmarking: Dataset Defects and Evaluation Failures in Lean Theorem Proving* makes this concrete. Its authors audit five widely used Lean theorem-proving benchmarks and forks, reporting 4,833 findings, including 398 mechanically certified defects such as counterexamples, vacuous theorems and unsound axioms. It also identifies semantic defects including missing hypotheses, incorrect translations and benchmark-specific specification hazards. Source: https://proceedings.mlr.press/v306/ammanamanchi26a.html.

PITB should not optimize only the prover against a benchmark and assume the benchmark is correct. It should measure a second capability: **can the system establish that the evaluation target itself is faithful, well-formed, non-vacuous under registered probes, and appropriately reviewed?**

This is separate from proving the target:
- **Proof validity:** a kernel-checked term establishes the target type.
- **Target integrity:** the exact target came from the pinned source and toolchain, has no unresolved metavariables or forbidden trust assumptions, and survives registered defect probes.
- **Semantic validity:** a qualified reviewer or downstream semantic suite confirms that the formal target corresponds to the intended problem.
- **Benchmark eligibility:** the target meets the preregistered requirements for inclusion in a particular score denominator.

Do not collapse these into one green checkmark.

## PITB integrity receipt v0

The new contract is:

- data/benchmarks/proof-insight-transfer-v0/benchmark-integrity/benchmark-integrity-receipt-v0.schema.json
- data/benchmarks/proof-insight-transfer-v0/benchmark-integrity/fixtures/benchmark-integrity.synthetic.json
- crates/bridges/symthaea-proof-insights/src/benchmark_integrity.rs
- crates/bridges/symthaea-proof-insights/src/bin/verify_benchmark_integrity.rs
- .github/workflows/pitb-sorrydb-contract.yml (focused path triggers, schema checks and verifier self-test)

The receipt pins:
1. Dataset cohort, snapshot, task ID and task-manifest digest.
2. Repository remote, immutable commit, relative source path, file digest, source coordinates and target statement digest.
3. Lean toolchain, executable, Lake manifest, root module and environment identities.
4. Elaboration status, exact checked-statement digest, unresolved metavariable/universe counts and sorryAx presence.
5. Sorted transitive axiom-use list, its content digest, the sorted allowlist and a recomputed difference count.
6. Counterexample and vacuity probe status/method/receipt identities.
7. Semantic-review status, scope, reviewer identity and attestation digest.
8. A derived integrity status (not_attempted, pending, eligible or ineligible).

The Rust verifier recomputes the sorted axiom-list digests and unexpected-axiom count; it derives the integrity status rather than trusting the claimed status; checks source-path safety and exact target identity; and can verify the receipt's exact byte SHA-256.

### What eligibility means—and does not mean

A measured target can become **eligible** only when:
- baseline elaboration passes against the pinned environment;
- the exact elaborated target statement digest matches the pinned source target;
- unresolved metavariables and universe metavariables are zero;
- no sorryAx is present;
- all used axioms are in the registered allowlist;
- the counterexample probe reports no counterexample;
- the vacuity probe reports no counterexample found;
- an independent semantic review is approved and carries its reviewer attestation digest.

A counterexample, target mismatch, unresolved metavariable, sorryAx, unapproved axiom, or rejected semantic review makes the task ineligible. A missing test, pending review, inconclusive probe or absent attestation keeps it pending. Pending and ineligible tasks remain visible in the cohort manifest; they are not silently deleted or counted as prover failures.

**No counterexample found is not a proof of non-vacuity.** It only records the result of a named, bounded probe. The independent semantic review is mandatory for eligibility. The review must consider the original source/problem statement and intended assumptions, not merely the generated Lean declaration. A source-to-coordinate check alone does not qualify: the approved scope must include informal-to-formal review with a pinned intended-statement digest, domain-expert review, or combined review. The combined scope requires the intended informal-statement digest. The current receipt hash binds a reviewer-attestation artifact but does not cryptographically authenticate reviewer identity; a signed-attestation/trusted-reviewer-key layer remains future work.

Likewise, a sorted axiom allowlist is a policy boundary, not a universal claim that every allowed axiom is appropriate for every benchmark. Each cohort must pin and justify its allowlist, and task-level reports must expose the actual axiom closure.

## Paradigm shift: evaluate the benchmark and solver together

The field should move from “proof rate on a frozen task list” toward a two-sided audit:

**Task-side integrity**
- Is the task source pinned and reconstructible?
- Does the extracted target match the source bytes and Lean elaboration?
- Are assumptions and imported axioms visible?
- Does the formal statement preserve the intended claim?
- Are there certified counterexamples or vacuity defects?
- Has an independent semantic reviewer approved the task?

**Solver-side integrity**
- Does the proof term pass the kernel?
- Does it preserve the exact target type?
- Does it avoid sorryAx and unapproved axioms?
- Does an independent clean replay verify the exact candidate bytes?
- Do applicable successor/invariant tests pass?
- Are resources and non-attempts fully accounted for?

This lets PITB distinguish at least four situations: a valid task the system solved, a valid task the system failed, a defective task that must not count as a prover miss, and an ambiguous task held pending review. That is a stronger scientific measurement than treating every compilable benchmark item as ground truth.

## Adversarial integrity suite

Every cohort should include registered mutations and independent checks:

| Mutation or defect | Required behavior |
|---|---|
| Change the target statement after source extraction | Reject exact target/source identity |
| Add a hypothesis that makes the theorem trivial or inconsistent | Flag semantic/assumption drift for review |
| Replace a target with True or an equivalent vacuous statement | Reject or hold pending semantic review |
| Add an unapproved axiom or sorryAx | Mark ineligible |
| Introduce unresolved expression/universe metavariables | Mark ineligible |
| Swap repository commit, source file or toolchain | Reject environment identity |
| Alter the allowlist without updating the registered policy digest | Reject axiom manifest identity |
| Change a reviewer/probe receipt after aggregation | Reject content digest |
| Produce a proof for a neighboring or weakened statement | Reject exact checked-target digest |

No single syntactic test can identify all vacuity or informal-to-formalization errors. Use several independent tools, counterexample search where feasible, successor-theorem tests, property-based checks and expert review. Report each probe's scope and limits; do not say “proved non-vacuous” merely because bounded search found no model.

## Benchmark portfolio integration

The integrity gate is orthogonal to the performance cohorts:

| Cohort | Performance endpoint | Integrity gate |
|---|---|---|
| Research-family transfer | Held-out family proof completion and strategy transfer | Pinned source theorem, fixed/mutable context, exact target type, axiom closure and reviewed theorem mapping |
| SorryDB_2601 | Open sorry completion under the original project/toolchain | Exact SorryDB source coordinates, repository commit, Lean toolchain, target type, proof candidate/replay receipts, trust-tier/axiom audit |
| LeanLean | Repository compression with protected declaration preservation | Pinned repository/comparator, protected declaration manifest, full rebuild, target declarations, axiom/trust changes and reviewer scope |
| PutnamBench routing subset | Cost-quality curve and success by fixed budget | Task statement source and commit, theorem type integrity, hidden test/counterexample audit and exact proof verification |
| VeriSoftBench / Aegis | Repository software-verification obligations | Source implementation identity, specification provenance, model/translator identity, differential tests, proof/refutation receipts and trust policy |

Do not combine these cohorts into a single integrity or success percentage. Their tasks, target definitions, proof obligations and acceptance protocols differ.

## Qualification targets

These are programme goals, not measured results:

| Endpoint | Target |
|---|---:|
| Source and environment identity | 100% of scored tasks have immutable source, target statement and Lean-environment digests |
| Elaboration | 100% of scored tasks elaborate with zero unresolved metavariables/universes |
| Trust | Zero unapproved axioms, zero sorryAx, zero unreviewed trust changes |
| Semantic approval | 100% of eligible scored tasks have an independent semantic review attestation |
| Probe coverage | Counterexample/vacuity probe status and method reported for every task; inconclusive remains pending |
| Audit reproducibility | Exact receipt bytes and manifest reproduce the same derived status |
| Denominator transparency | Every task is reported as eligible, pending or ineligible; no silent filtering |
| Regression | All target, source, axiom and reviewer-receipt mutation controls fail closed |

The strongest defensible initial result is not necessarily a higher solve rate. It may be a better-audited denominator and a clear explanation of how many benchmark tasks actually measure the intended capability.

## Remaining blockers

1. The current receipt contains reviewer/probe digests but does not yet authenticate those external artifact bytes or reviewer signatures.
2. The verifier checks source/target identities and the claimed sorted axiom closure, but the measured capture pipeline that obtains the full transitive axiom closure must still be implemented and independently tested.
3. Counterexample and vacuity probes are evidence, not formal completeness proofs. Their algorithms, bounds and failure modes must be pinned.
4. A complete source-to-informal-statement audit needs reviewer guidance, adjudication and disagreement handling.
5. No benchmark cohort has been audited by PITB yet. The fixture and workflow are contract infrastructure only.

## Immediate work sequence

1. Compile and run verify_benchmark_integrity in the isolated PITB crate workflow.
2. Add a Lean capture adapter that records the target type, environment, metavariable state and transitive axiom closure at the exact task source coordinates.
3. Pin and replay counterexample/vacuity probes on a small smoke cohort from at least two distinct Lean toolchains.
4. Prepare an independent semantic-review protocol and signed reviewer-attestation format.
5. Audit a stratified smoke sample from SorryDB_2601, report eligible/pending/ineligible counts, and retain all evidence.
6. Only after that audit should PITB compare proof-system performance on the eligible denominator; publish pending/ineligible strata separately.
