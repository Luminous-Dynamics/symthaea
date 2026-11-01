# PITB external benchmark track: Aegis program verification (2026-11-01)

**Priority:** P0 external challenge after the real Lean target-integrity adapter exists.  
**Status:** design only; PITB has not obtained, reconstructed, or run Aegis tasks.

## Why this matters

The 28 October 2026 preprint *Scaling Lean Theorem Proving for Real-World Code Verification* reports EasyProver at 669/672 PutnamBench, all six IMO 2026 formalizations, 597/600 NTP4VC verification conditions, and all 189 VERINA tasks resolved by proof or refutation. The paper introduces Aegis, a pipeline that generates safe Rust implementations, translates them through Charon/Aeneas into Lean, creates candidate specifications, and freezes each proposition with a kernel-checked proof or refutation label.

It reports 2,045 differentially validated Rust–Lean program pairs and 11,533 candidate specifications. On a stratified sample of 2,000 statements over 1,257 translated programs, EasyProver attempted 1,997, proved 1,054 (52.7% of the sample), and left 943 unresolved within 30 minutes per statement. The reported proof rate is 84.1% for edge-case specifications, 31.0% for explicit contracts, 66.7% on Easy problems, and 37.6% on Hard problems. Sources: https://arxiv.org/abs/2610.38174 (submitted 28 October 2026); https://arxiv.org/html/2610.38174.

This is an unusually valuable PITB test because its obligations include machine arithmetic, overflow, explicit failure, loop termination, generated control flow, casts and functionalized mutation—not only textbook theorem statements.

## What the paper did not establish

- 52.7% is from a stratified 2,000-statement sample, not all 11,533 candidate specifications.
- The sample excludes specifications already refuted by the Plausible screen and the category requested to fail.
- Evaluating other provers on Aegis and ablation of harness components remain future work.
- The authors do not compare the same base model using Lean without the EasyProver harness.
- A kernel-checked proof validates the frozen Lean proposition; it does not by itself prove that an auto-generated specification matches the intended programming requirement.

PITB must reproduce the published sample/eligibility protocol for any direct comparison and maintain separate results for the broader candidate pool, refutation-capable tasks, mutation variants and integrity-pending tasks.

## Evaluation cohorts

**A. Published 2,000-statement sample.** Use the exact sample, specification bytes, translated program identity, Lean environment, polarity, difficulty/category strata and source manifest. The 30-minute per-statement budget is the primary published budget. Reconstruct the baseline harness and reported model where lawful and feasible. If that cannot be reproduced, report a PITB-local baseline and do not call it a head-to-head.

**B. Full validated program/specification pool.** Evaluate the full accessible candidate pool with a frozen sample manifest. Preserve 2,045 program-pair identities and 11,533 candidate-specification identities as the paper reports, but distinguish validated program count from candidate specification count. Candidate specifications from one translated program are not independent; resample by program component.

**C. Hard implementation obligations.** Stratify by fixed-width arithmetic/range constraints, checked arithmetic/overflow, failure and totality, bounds-checked indexing, generated loop invariants/termination, casts, mutation, explicit contracts/postconditions, difficulty and source-program family. Do not report only aggregate performance; explicit contracts are a key hard stratum.

**D. Refutation and mutation cohort.** Keep proof of the frozen positive proposition, proof of its negation, invalid/mismatched generated specification, and abstention due to ambiguity as separate outcomes. Do not count a counterexample as a proof or a proof of the wrong polarity as success. The Plausible screen is a negative-only prefilter in the paper, not a source of positive labels. For mutation variants, validate the changed Rust program and translated Lean model first; the agent must not prove original behavior against a mutated implementation or vice versa.

## PITB capability arms

| Arm | Method | Purpose |
|---|---|---|
| A0 | Fixed-step baseline | Conventional budgeted proof agent |
| A1 | PITB proof-state/context retrieval | Premise and local-instance coverage |
| A2 | Support-antichain-guided context | Value of learned minimal support vs full/raw context |
| A3 | Typed blueprint repair | Structural repair of loops, casts and invariants |
| A4 | Adaptive action routing | Proof-state-conditioned intervention selection |
| A5 | Full PITB | Combined retrieval, support, repair and routing |

Oracle-minimal-support and size-matched random-support arms belong in a small diagnostic experiment. Oracle support is an upper bound, not a deployable competitor. Ablate one capability at a time before combining them.

## Metrics

Primary endpoints: exact-target kernel proof rate at fixed 30-minute budget; exact-target semantic/successor-gate success; intention-to-treat success over every assigned statement (non-attempts and unavailable tools remain failures); cost per independently verified proof; cost-quality AUC.

Secondary endpoints: success by specification category/difficulty/project; proof/refutation/ambiguous-label/abstention rates separately; tokens, model/tool calls, Lean CPU, wall time, peak memory and dollars; proof artifact size/helper-lemma count; time to kernel-valid proof and semantic approval; integrity-pending rate; calibration and interval coverage.

Resample at translated-program component level, not statement-row level.

## Targets (research goals, not results)

- ≥60% exact-target semantic-gated proof rate on the same 2,000-task sample within 30 minutes, versus EasyProver's reported 52.7%.
- ≥40% on explicit-contract statements, versus 31.0%.
- ≥45% on Hard tasks, versus 37.6%.
- ≥30% lower cost per semantic-gated proof at non-inferior success versus a faithful matched baseline.
- Zero false proof/refutation accepts under statement, polarity, implementation, axiom and receipt mutation.
- Same-direction gains on a later or independently generated program cohort.

## Integrity and anti-gaming

1. Require the PITB target-integrity gate for every scored specification.
2. Pin Rust source, Cargo lock, compiler, Charon/Aeneas, generated Lean code and exact proposition bytes.
3. Preserve requested polarity and proof/refutation labels.
4. Require independent replay in a clean process, no sorryAx and approved axiom closure.
5. Differentially check Rust–Lean behavior on reference and held-out/generated tests.
6. Use Plausible/property tests as screening evidence only; a negative finding is not a positive proof.
7. Retain every pipeline denominator: attempted program, validated program, candidate spec, screen-rejected spec, selected sample, attempted proof, accepted proof/refutation.
8. Keep training, calibration, validation and external challenge program IDs disjoint.
9. Report all strata and cluster uncertainty by translated program.

## Immediate execution

1. Locate the exact public Aegis code/data release and license; verify if the 2,000 sample can be reconstructed.
2. Materialize and hash a small stratified smoke cohort without treating it as a performance result.
3. Audit source → Rust → Aeneas Lean → frozen target → proof/refutation receipt with PITB integrity verifier.
4. Reproduce one A0 baseline task per obligation class and confirm budget accounting.
5. Preregister A0–A5 and paired-support experiment on a validation subset.
6. Scale to the 2,000-task sample only after independent receipt replay.
