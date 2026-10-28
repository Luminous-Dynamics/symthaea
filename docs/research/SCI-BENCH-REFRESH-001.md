# SCI-BENCH-REFRESH-001 — Adaptive Scientific-Law Benchmark Refresh

**Status:** protocol proposal; not a qualified implementation or empirical result.

**Parent tracking:** [Issue #8771](https://github.com/Luminous-Dynamics/symthaea/issues/8771).

## 1. Objective

Maintain a fixed reference point for longitudinal comparison while continuously testing whether scientific-discovery policies generalize beyond familiar task templates. A benchmark refresh must change the challenge set without silently changing what an existing score means.

This work is lifecycle/evaluation protocol only. It composes with:

- [#8143 — OPEN-WORLD-PHYSICS-001](https://github.com/Luminous-Dynamics/symthaea/issues/8143) for hidden-law evaluation and observational-equivalence semantics;
- [#8588 — SCI-REGIME-001](https://github.com/Luminous-Dynamics/symthaea/issues/8588) for regime-transition discovery;
- [#6283 — TRANSPORT-BENCH-CAL-003](https://github.com/Luminous-Dynamics/symthaea/issues/6283) for measurement validity and benchmark exposure history;
- [#8756 / PR #8757](https://github.com/Luminous-Dynamics/symthaea/pull/8757) for importing and pairing native NewtonBench trial artifacts.

It does not replace any of those evaluators or create another law-discovery solver.

## 2. Evaluation lanes

| Lane | Task exposure | Intended inference | Longitudinal rule |
|---|---|---|---|
| Public development | Tasks, answers and evaluator are visible by design | Debugging, smoke tests, hypothesis generation | Never counts as sealed confirmation |
| Immutable anchors | Frozen task bytes, evaluator version and scoring semantics | Within-anchor comparison across policies and revisions | New versions are append-only; never rewrite old score meaning |
| Sealed refresh | Public construction recipe; evaluator-held seed/target unavailable to policy until submission | Generalization to a fresh task draw from the declared task-family mixture | Report separately by generation; reveal task/seed after scoring for audit |
| External scientific tasks | Independent published data, experiment or domain-specific reference | Transfer beyond synthetic task construction | Separate from all synthetic results and judged symbolic accuracy |

A benchmark can be reproducible without being sealed. If a runner cannot keep the seed and oracle fields inaccessible to the policy, label the result `reproducible_open`, not `sealed_refresh`.

Never average these lanes into a single trendline. Report anchor and refresh results side-by-side because they answer different questions.

## 3. Threat model

Assume the system being evaluated can exploit:

- familiar templates, constants, task names, and evaluator quirks;
- contamination from public examples, prior refresh generations or model/tool pretraining;
- direct or indirect access to target labels or evaluation-only observations;
- near-duplicate task leakage across development, anchor and refresh sets;
- repeated benchmark feedback used to tune prompts, tools, policy thresholds or task generation;
- evaluator failure on invalid, non-finite, not-identifiable or deliberately adversarial cases;
- missing or failed trials that disappear from denominators;
- selective reporting of easy task families or favorable budgets.

The protocol does not claim to defeat all pretraining contamination. It makes exposure explicit, tests for observable leakage, and limits claims to the declared source and task-generation process.

## 4. Refresh task construction

Publish the constructor recipe and version its source code. Freeze the family mixture and task acceptance rules before scoring any candidate policy. Keep task-generation logic independent from both the evaluated policy and the evaluator that holds ground truth.

Initial constructor families:

1. **Semantic invariance:** symbol renaming, variable reparameterization, dimensionally valid unit and scale changes. Outputs should remain equivalent after the declared inverse mapping.
2. **Structural generalization:** held-out operator combinations or expression structures; explicit unit/dimensional checks prevent malformed expressions from becoming easy false positives.
3. **Sampling and noise shifts:** smaller/larger observation budgets, heteroscedastic or correlated noise when specified, and shifted sampling distributions.
4. **Domain extrapolation:** held-out input regions and joint shifts not used for candidate selection or fitting.
5. **Confounding and null worlds:** nuisance variables, spurious correlations, and no-effect controls to measure false discovery.
6. **Identifiability limits:** worlds where several laws are observationally equivalent under available measurements, or where the target is not identifiable. The correct response is a bounded equivalence class, calibrated uncertainty, or abstention—not forced exact recovery.
7. **Regime controls:** smooth laws that resemble a threshold over a narrow range, alongside real transitions. Compose with #8588 rather than duplicating its specialist evaluator.
8. **Evaluator canaries:** frozen known-answer positive, negative, malformed, ambiguous and not-evaluable cases. The canary suite is a gate on the evaluator and generator, not a model leaderboard.

Each generated task records its family and provenance without exposing the hidden target to the policy. Use family-level splits, not row-level random splits, where generated tasks share a parent law, generator lineage, or transformation seed.

## 5. Commitment and reveal protocol

Before executing a refresh generation, create and preserve a commitment to:

- exact generator source revision and binary/build identity;
- canonical generator configuration and task-family mixture;
- evaluator source/version and hidden-target artifact;
- seed bytes, task bytes, and expected ordered task IDs;
- policy/model/tool/prompt/configuration identity and resource limits;
- analysis plan, metrics, exclusions, pairing keys and stopping rules.

Use domain-separated SHA-256 commitments over canonical byte encodings. Publish the commitment before a run, but keep the seed and evaluator-only bytes in a separate trusted evaluation boundary. The evaluated policy must receive only the declared observation interface. Bind the final result receipt to the pre-run commitment and every raw task/trial digest.

After submissions are locked, reveal the seed and generated task/evaluator bytes when licensing, privacy and security constraints permit. An independent process should regenerate the task set from the revealed inputs and compare exact task digests. Retain failed, missing and invalid attempts as outcomes rather than deleting them.

A hash proves byte identity, not secrecy, correct execution, scientific validity or freedom from common-mode bugs. Execution provenance and score correctness remain separate gates.

### 5.1 Executable commit–reveal contract

The Rust validator implements a versioned, domain-separated commitment profile:

1. `protocol_commitment_sha256` commits to every frozen non-artifact manifest field: generation/lane/purpose, generator identity and configuration digest, evaluator identity and oracle-separation declaration, family mixture and expected denominator, policy/tool/prompt/configuration identities, resource budget, analysis-plan digest, and reveal policy.
2. `seed_commitment_sha256` commits to the exact seed-file bytes plus the protocol digest. A seed must contain at least 32 bytes; a real run must use a cryptographically secure generator and keep the bytes outside the policy-visible environment.
3. `task_set_commitment_sha256` commits to the exact bytes of the complete task-set artifact plus the same protocol digest. Whitespace or ordering changes intentionally change the committed task artifact.

All three use SHA-256 with a fixed protocol prefix, a distinct domain for `protocol`, `seed`, or `task-set`, and explicit unsigned 64-bit big-endian length framing. The protocol context uses a versioned `serde_json` struct projection that excludes the three commitment fields to avoid a circular digest. Artifact commitments bind the protocol digest, so altering a frozen budget, task family, generator/evaluator revision, policy identity, or reveal rule invalidates the context and both artifact commitments.

The task-set artifact follows `research/science-bench-refresh/schema/task-set.schema.json`. Each row binds a unique task ID, a declared family ID, a policy-visible `policy_input` object, and a separate evaluator-only object. At reveal time, the validator checks that the generation ID matches, row count equals the frozen expected denominator, IDs are unique, per-family counts equal the manifest, and policy inputs do not contain known oracle-field names (including in nested structures). The runner must physically provide only `policy_input` to the evaluated policy; this structural check does not prove runtime isolation.

CLI entry points:

- `protocol-commit MANIFEST.json` computes the frozen non-artifact context commitment.
- `artifact-commit MANIFEST.json seed SEED.bin` and `artifact-commit MANIFEST.json task-set TASK_SET.json` compute context-bound artifact commitments.
- `verify-reveal MANIFEST.json SEED.bin TASK_SET.json` recomputes protocol/seed/task commitments against the exact revealed bytes and reports match/mismatch without printing seed contents.

The verification command admits only a structurally valid `qualified_run` manifest in the `sealed_refresh` lane. This gate verifies byte consistency after reveal; it is not a secure enclave, signature, trusted timestamp, or remote-attestation mechanism.

The next required layer is an independently implemented task regeneration check: regenerate the task set using the revealed seed and pinned generator revision, then compare exact task bytes/digests. Byte-level commit–reveal without regeneration cannot establish that the advertised generator created the committed tasks. Runner isolation, task identifiability, duplicate/family leakage checks, evaluator positive/negative controls, and execution receipts remain separate gates.

## 6. Frozen policy and fair comparison

Freeze before opening the refresh seed:

- model ID/revision, system prompt, policy source SHA and relevant build artifact;
- tool availability, tool versions, external data access and network policy;
- task budget, observation/experiment caps, token/compute caps, wall-clock limit and stopping criteria;
- evaluator revision, symbolic checker or judge identity, judge prompt and metric implementation;
- paired seeds/tasks, exclusions, analysis plan and uncertainty method.

Compare at least:

- random or fixed feasible query policy;
- the benchmark's native/reference baseline, where available;
- the strongest eligible symbolic-regression baseline for tasks it can represent;
- the current frozen Symthaea policy;
- preregistered ablations for mechanisms actually claimed to help.

All arms use the same initial task information and declared resource caps. Record tuning exposure and revision history. Do not tune on the sealed results and then describe the same generation as confirmatory; a result used to adapt a policy moves into development history.

## 7. Metrics — keep the dimensions separate

### Discovery quality

- Exact symbolic recovery only when a formal or independently checkable equivalence procedure supports it.
- Judge-derived exactness as a separate metric with judge identity and limitations.
- Prediction error on held-out observations and held-out domains, including extrapolation error.
- Validated discovery rate by total observation, experiment, reference-call, token and wall-clock budgets.
- Cost to first independently validated law, counting failed experiments and invalid candidates.

### Epistemic behavior

- False-discovery rate on null/confounded worlds.
- Confidently-wrong rate when the policy makes a high-confidence unsupported claim.
- Calibration and interval coverage where probabilities/intervals are emitted and an appropriate target is defined.
- Correct abstention, unsupported abstention, unresolved, and not-evaluable rates as distinct outcomes.

### Robustness and generalization

- Paired anchor deltas and refresh-only scores, separately by task family, noise, sample budget and shift.
- Public-development to sealed-refresh degradation with uncertainty, without assuming that every difference is contamination.
- Sensitivity to equivalent reparameterizations, scale/unit transformations, seed changes and generator-family holdouts.
- Near-duplicate and family leakage rate across all splits.

### Cost and auditability

- Total and available-field counts for tokens, experiments, observations, tool calls, reference evaluations, runtime and compute cost.
- Expected, observed, completed, failed, invalid, missing and not-evaluable denominators.
- Source/task/config/evaluator digests, runner execution receipts, and exact policy identity.
- Generator defect rate, evaluator control pass rate, and proportion of generated cases judged identifiable by a separate eligibility rule.

Do not create a single benchmark-quality or science-capability scalar that allows high symbolic accuracy to compensate for false discoveries, leakage, or missing provenance.

## 8. Task admission and evaluator gates

Before comparing policies:

1. Rebuild the generated task set from its committed recipe/seed in an independent process and compare task digests.
2. Validate dimensional/semantic invariants and ensure transformation pairs preserve the intended target.
3. Run positive and negative evaluator controls, including failures, malformed values, non-finite observations, no-effect controls and ambiguous cases.
4. Check generated-family solvability by eligible reference baselines; report floor/ceiling saturation and retain hard-but-solvable cases.
5. Audit family-level and near-duplicate leakage across development, anchor and refresh sets.
6. Compute identifiability eligibility independently of the tested policy and report both the eligible fraction and conditional performance.
7. Confirm no oracle-only field is accepted by the policy input schema and no post-outcome data are used to select, fit, stop or tune a policy.
8. Freeze the denominator and analysis before scoring the tested policies.

If a gate fails, the generation is a generator/evaluator diagnostic, not a valid model comparison.

## 9. Proposed go/no-go targets

These are prospective targets, not established results:

- 100% agreement with frozen known-answer evaluator controls before policy metrics are admitted.
- Zero accepted oracle-only fields in the policy input contract; adversarial field insertion and task/seed/config mutation must be rejected or change the commitment.
- 100% expected-task accounting in the report, including failed, missing, invalid, and not-evaluable states.
- No anchor/refresh score conflation; task and scoring revisions create new generation IDs.
- Independent audits find no known family or near-duplicate leakage across the declared splits, or the affected cases are explicitly quarantined before unblinding.
- Report generation quality and baseline-only performance before claiming any Symthaea improvement.
- A superiority claim requires a preregistered paired comparison, matched resource caps, held-out families, multiplicity-aware analysis where multiple confirmatory comparisons are made, and uncertainty that supports the stated effect.
- Synthetic results remain labelled synthetic. A claim of real scientific discovery requires independent real-data or experimental validation appropriate to the specific domain.

Do not use a universal fixed percentage gain as the only success criterion. First establish valid task generation and eligible baselines; then power the prospective comparison around a domain-relevant minimum effect.

## 10. External benchmark context

- **AutoSciBench** (arXiv:2610.05140, 4 Oct 2026) explores iterative generation and refinement of scientific-agent tasks to reduce benchmark saturation. Its reported effects are specific to its tested settings and are not presumed to transfer to Symthaea: https://arxiv.org/abs/2610.05140
- **NewtonBench** evaluates interactive scientific-law discovery across 324 tasks and 12 physics domains, with metaphysical shifts and increasing difficulty/noise: https://arxiv.org/abs/2510.07172
- **SRBench** demonstrates the value of standardized reproducible symbolic-regression comparisons and emphasizes that no single method dominates every dataset; its updated benchmark discussion includes accuracy/complexity/resource trade-offs: https://doi.org/10.1145/3712255.3734309
- **ScienceAgentBench** provides a complementary set of real scientific tasks sourced from peer-reviewed publications, with expert validation; synthetic law recovery is not a substitute for it: https://hal.cs.princeton.edu/scienceagentbench
- **SCI-AGENT-EXAM** (arXiv:2610.32441, 23 Oct 2026) covers 120 expert-curated tasks across 11 fields and seven workflow task types, requiring structured answers, executable artifacts, and written reports. It also evaluates execution trajectories and stability. The paper reports an 80.9 overall score for its top agent but only 26.0% of runs at or above its 0.95 success threshold; this illustrates why mean score alone is insufficient, but is not directly comparable to law-discovery metrics: https://arxiv.org/abs/2610.32441
- **GravityBench** is a complementary interactive physics-discovery environment and demonstrates the need to evaluate observation planning as well as equation output: https://gravitybench.github.io/

These benchmarks are complementary, not directly commensurate. Do not compare scores across them without matching task definitions, scoring semantics, budgets and evaluation units.

## 11. Competitive baseline plan and stretch targets

Do not publish one cross-benchmark leaderboard number. These benchmarks measure different tasks and score semantics; use a portfolio and state which inference each result supports.

| Evidence lane | Baseline portfolio | What it can establish | What it cannot establish by itself |
|---|---|---|---|
| Interactive law discovery | Pinned NewtonBench native baseline; paired no-tool/tool policy; same task/seed and total resource caps | Whether a policy improves interactive discovery and native equation scores | Physical discovery or transfer to unrelated task families |
| Autonomous experiment design | Science-Gym and/or GravityBench policy baselines, including random/fixed observation selection and strongest available native baseline | Whether the policy chooses informative observations under a controlled budget | Superiority to offline symbolic-regression solvers on non-comparable tasks |
| Offline symbolic regression | A frozen snapshot of eligible top methods from SRBench/SRBench++, with method revisions, tuning budget and resource envelope captured | Accuracy/complexity/resource trade-offs on compatible equation-recovery tasks | That an agent can independently design experiments or manage scientific workflow |
| Real scientific workflows | ScienceAgentBench tasks or another independently sourced, rights-cleared task set with expert/domain validation | A bounded check of transfer into real data-driven scientific tasks | Exact law-discovery performance on synthetic physics; its metric must not be pooled with the others |
| Scientific-workflow trajectories | SCI-AGENT-EXAM: 120 expert-curated tasks across 11 fields and 7 workflow types, with executable + text deliverables and trajectory analysis | Stability, execution efficiency, task-type strengths and result quality across scientific workflows | A direct law-discovery score; its composite is not commensurate with NewtonBench or SRBench |

Selection of the strongest comparator must be reproducible from a dated source snapshot. Do not select only the baseline that loses: keep complementary methods when their strengths differ, and document hyperparameter/tuning effort. SRBench's own motivation is that no single symbolic-regression method dominates across all datasets, and it explicitly considers accuracy, complexity and energy/cost trade-offs.

### Prospective stretch targets — not current results

Use the baseline-only pilot to estimate variance and define the practical minimum effect before powering the confirmatory study. The following are proposed research ambitions, not universal benchmark standards:

1. **Discovery efficiency:** at matched total budget, target at least 20% lower median total cost to the first independently validated law, without worse validated-hit rate or higher confidently-wrong rate. Count failed experiments, reference/oracle calls, tokens, runtime and rejected candidates; if costs cannot be converted to a defensible common unit, publish separate budget curves rather than a synthetic total.
2. **Held-out symbolic recovery:** target at least a 10 percentage-point absolute increase in exact, independently checkable recovery over the strongest eligible baseline at matched budget, with a preregistered paired uncertainty interval excluding no improvement. If exact recovery is not formally checkable, do not replace it with a judge score without relabelling the claim.
3. **False discovery:** target a one-sided 95% upper confidence bound no greater than 5% for confidently wrong law claims on independently generated null/confounded worlds. The number of worlds must be selected prospectively; for intuition, zero failures in 60 independent Bernoulli null worlds gives an upper bound of about 5% by the rule-of-three approximation, but task-family dependence invalidates treating correlated cases as 60 independent trials.
4. **Invariant behavior:** require 100% pass on frozen evaluator/generator positive and negative controls, no leaked oracle-only inputs accepted by the policy interface, and 100% expected-task denominator accounting.
5. **Generalization:** demonstrate gains on held-out task families and refresh generations while preserving anchor scores; report per-family deltas and the public-to-refresh gap instead of hiding regressions in a pooled mean.
6. **Uncertainty:** whenever confidence/probability outputs are used to justify a scientific claim, report grouped calibration and coverage against the matching target population. Do not tune a threshold on the sealed generation and reuse it as confirmatory evidence.

To claim a win over the state of the art, the candidate must beat the strongest eligible comparator on a predeclared decision-relevant metric at matched budgets, report uncertainty and per-family failures, and retain acceptable false-discovery/calibration behavior. Pareto improvements are preferable to a single composite: validated discovery rate, total cost, predictive extrapolation, complexity and epistemic reliability should remain separately visible.

These ambitions should be narrowed or revised only using baseline-only pilot evidence before unblinding—not after seeing the candidate's confirmatory result.

## 12. Initial delivery sequence

1. **Protocol (this document):** freeze lanes, threat model, generation and scoring rules.
2. **Manifest/schema + synthetic fixture:** canonical example with pre-run commitments, hidden-field boundaries and expected denominator.
3. **Manifest validator (implemented in draft PR #8773):** strict Rust manifest parsing, unknown-field rejection, digest-format checks, pinned-revision checks for qualified runs, task-denominator closure, family-weight closure, duplicate-family rejection, reveal-policy/lane consistency, and separation between declared eligibility and proof of execution isolation. It does not recompute the hidden seed/task commitments or authenticate a runner.
4. **Generator qualification:** positive/negative controls, duplicate/family leakage, identifiability, baseline solvability and seed replay.
5. **Baseline-only pilot:** use pinned NewtonBench or a separately declared small synthetic profile; no candidate-policy superiority claim.
6. **Confirmatory study:** paired frozen policies, predeclared effect and sample size, complete cost accounting, held-out generation(s), and independent review.

## 13. Claim ceiling

This document is a protocol proposal. It does not show that a benchmark has been generated, that an evaluator-held secret can be safely maintained by the current runner, that task constructors are qualified, or that Symthaea exceeds any state-of-the-art baseline. The first valid result is successful generator/evaluator qualification; a model win is a later and separate claim.