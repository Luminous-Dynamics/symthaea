# PITB Aegis integrity contract

Rust-first intake validator for the Aegis Rust-to-Lean verification benchmark in arXiv:2610.38174v1 (submitted 28 October 2026).

## Intake state

As of 5 November 2026, the public arXiv paper record inspected for this work does not link a task archive or Aegis repository. Searches of public GitHub repositories did not locate an official Aegis data release. The archive license is therefore not confirmed. The provenance file records dataset_artifact_status=not_located and is a paper-metadata pin only. It is not an Aegis task dataset; this PR makes no measured Aegis result claim.

## Paper baseline

The paper reports 2,356 source problems attempted; 2,045 Rust-Lean program pairs validated (86.8%); 11,533 candidate specifications; and a stratified 2,000-statement sample across 1,257 programs. EasyProver attempted 1,997 sample tasks under a 30-minute per-task wall budget, proved 1,054 original statements (52.7% of selected tasks), and left 943 unresolved. The paper reports 84.1% proof rate for edge-case specs versus 31.0% for explicit contracts, and 66.7% for Easy vs 37.6% for Hard.

The sample excludes tasks already refuted by Plausible and the category requested to fail. Therefore the reported 52.7% is proof of the original statement over a hold-only sample, not a polarity-balanced proof/refutation resolution score.

## Assurance model

The crate validates five separate ledgers:

1. Program-pair identity and admission: Rust source/toolchain, Charon extraction, Aeneas translation, Lean environment, Rust test receipt and differential-test receipt. All retained differential cases must match.
2. Specification identity: theorem statement digest, frozen theorem-prefix digest, specification ID, category, difficulty and requested polarity.
3. Sample construction: explicit selected/excluded status and reason. The paper-v1 protocol excludes requested-to-fail specs and specs already refuted by Plausible.
4. Prover outcomes: proved P, proved not-P, unresolved, resource exhaustion or harness error. A resolved result requires unchanged target/prefix, kernel acceptance, allowed axiom closure, no sorry, proof artifact, patch and independent Comparator receipt.
5. Specification intent: proof of P confirms a requested-to-hold property; proof of not-P confirms a requested-to-fail property. Proving the opposite polarity still resolves the formal proposition, but indicates the generated specification intent was wrong. These scores are reported separately.

Differential testing on retained reference inputs is finite evidence, not universal Rust-Lean equivalence. A kernel proof establishes a proposition about the translated Lean program; it does not alone establish that the translation matches Rust for all inputs or that the generated property captures original problem intent. This validator checks digest syntax and ledger joins; it does not open receipt bytes or rerun Rust, Charon, Aeneas or Lean Comparator.

## Proposed experiments

- Matched harness ablation: same model, task list, Lean environment, wall/token budgets. Compare plain Lean tool use, retrieval-only, tactic probing, recursive delegation and full harness. The paper explicitly did not run the same Fable-5 model without the EasyProver harness or ablate components.
- Balanced proof/refutation set: measure proof of P, proof of not-P, unresolved, timeouts and requested-polarity accuracy separately.
- Mutation generalization: vary bounds, integer widths, casts, loop boundaries, mutation order, preconditions, failure branches and collection invariants. The paper says mutation-generated tasks were created but not evaluated.
- Counterexample-assisted routing: Plausible hits are candidate counterexamples, not proof labels; no counterexample is not a proof.
- Specification quality: track vacuity, omitted preconditions, over-strong/under-strong claims, target mismatch, source/translation mismatch and polarity errors separately.
- Cost: report selected and attempted denominators, p50/p95, tokens, API cost, CPU/GPU time and cost per kernel-accepted result. Use program-clustered uncertainty because 2,000 statements come from 1,257 programs.

## Proposed qualification targets

- Reproduce the published paper-v1 cohort and baseline before claiming a comparison.
- Improve original-statement proof rate by at least 5 percentage points at the same 30-minute budget, with program-clustered confidence intervals.
- Preserve or improve proof/refutation resolution and requested-polarity accuracy without task exclusion or theorem mutation.
- Aim for at least 20% lower cost at no more than 1 percentage point proof-rate loss, or improve proof rate at matched cost.
- On mutation holdouts, require zero false claims that finite differential tests prove source equivalence.

The strongest initial contribution is not a claim to beat EasyProver; it is a compute-matched ablation and a source-to-Lean assurance protocol that shows what was proved, which source program it refers to, and whether the requested behavior was established.

## Sources

- Paper: https://arxiv.org/abs/2610.38174
- Aeneas: https://github.com/AeneasVerif/aeneas
- Charon: https://github.com/AeneasVerif/charon
- Lean Comparator: https://github.com/leanprover/comparator
- Plausible: https://github.com/leanprover-community/plausible
