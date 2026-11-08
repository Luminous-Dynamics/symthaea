# CEB Review Preservation-Aware Repair Verification Protocol V0 — 2026-11-08

**Status:** Public design-only protocol and synthetic development challenge. No admitted tasks, candidate execution, hidden labels, benchmark result or new metric lane.
**Composes with:** causal diagnosis, causal topology, causal identifiability, counterfactual-control qualification, finding adjudication and the three existing CEB-REVIEW primary measures.
**Execution boundary:** candidate-controlled execution remains blocked by #10064; independent pristine regrade remains blocked by #10369.

## 1. The missing half of a correct review

A review finding is not trustworthy merely because it points to a plausible defect. A suggested correction can be unnecessary, over-broad, or behavior-changing outside the target invariant. CEB must establish two separate claims:

1. **Target claim:** the reviewed implementation violates a stated requirement or invariant under a defined input/state.
2. **Preservation claim:** the proposed correction does not break valid behavior in the relevant neighboring cases and preserves unrelated invariants.

This protocol makes the second claim explicit. It does not require every review finding to include a patch. A finding without a proposed patch can still be well-grounded, but it cannot claim repair verification.

## 2. Paradigm shift: dual-witness review

For every finding that proposes a correction, require two independently grounded witnesses.

### 2.1 Target witness

A target witness identifies the exact requirement, source path, condition and observable consequence that make the original implementation defective. It must state whether the witness is:

- source-structural possibility;
- build-world selection;
- execution realization;
- actual-cause attribution;
- interventional-effect claim; or
- incident attribution.

Use the causal-identifiability protocol's claim-kind discipline. A source-level counterexample is not a production trace.

### 2.2 Preservation witness

A preservation witness identifies a nearby valid case that the correction must not break. It must be derived from the same normative requirement and exact scope, not invented from a generic best-practice preference.

A preservation witness should answer:
- Which valid behavior is at risk?
- What input/state exercises it?
- What does the contract require before and after the proposed correction?
- Which unrelated invariant must remain unchanged?
- What evidence would reveal an over-repair?

A finding with a target witness but no preservation analysis may remain actionable as a diagnosis; it must not be described as a verified repair.

### 2.3 Independent oracle

The requirement and oracle must be frozen independently of the model's rationale and proposed patch. A test generated only from the candidate's own interpretation cannot validate that interpretation. Use contract-authored examples, existing authoritative tests, independently adjudicated properties, or reviewer-approved metamorphic relations. Record any gaps or ambiguous contract semantics.

## 3. Four-way original/patch outcome matrix

Let C be the original candidate, C' the proposed correction, T the target oracle, and P the preservation oracle. The oracle definitions and environment are fixed before inspecting the proposed patch.

| Original C | Proposed C' | Interpretation | Review disposition |
|---|---|---|---|
| T fails; P passes | T passes; P passes | Target repair with preservation evidence | Repair-supported within the tested scope |
| T passes; P passes | T passes; P passes, and behavior is equivalent on the relevant domain | Proposed change is unnecessary for this requirement | Over-correction / no-fix finding |
| T passes; P passes | P fails or valid behavior changes | Proposed correction is regressive or over-broad | Reject the patch; preserve the original behavior |
| T fails; P fails | C' still fails T, or P is already violated | Inadequate repair, invalid oracle, or multiple interacting defects | Unresolved; refine the contract/evidence before attribution |

Important qualifications:

- A test pass is evidence only for its declared domain and oracle. It does not prove universal equivalence.
- If T and P disagree, the oracle is not qualified; do not select the model-favorable oracle.
- If both versions pass a shallow suite, a bug may remain undetected. Preserve an unresolved oracle-coverage state.
- A repair can fix the target and still regress a preservation invariant. Do not call it a successful repair by target-test pass alone.
- A patch that is behaviorally equivalent under the qualified scope may demonstrate that the original negative verdict was unsupported; it does not establish correctness outside that scope.

## 4. Evidence contract for the two witnesses

Each repair-bearing finding records:

| Field | Required content |
|---|---|
| Frozen requirement | Exact contract clause and authoritative source |
| Original revision | Repository, commit/tree, relevant file/hunk identity |
| Target witness | Input/state, predicted violation, source anchors and claim kind |
| Preservation witness | Closest valid neighboring input/state and expected behavior |
| Proposed correction | Exact patch identity, changed paths and intended causal effect |
| Target oracle | Independently authored test/property and expected result |
| Preservation oracle | Independent test/property for valid neighboring behavior |
| Other invariants | Security, compatibility, ordering, idempotency, error behavior, resource bounds, or documented side effects |
| Environment | Build/features/target/dependencies/runtime/data/seed as applicable |
| Oracle authority | Existing CEB evidence-authority class; never silently upgrade |
| Residual uncertainty | Uncovered inputs, latent context, equivalence gap and alternate paths |
| Disposition | Repair-supported / unnecessary / regressive / unresolved / not executed |

## 5. Oracle construction rules

1. **Specification-first:** derive tests and properties from the frozen requirement, not from the proposed patch text.
2. **Paired boundary coverage:** include the target-triggering input and at least one nearest valid neighbor.
3. **Discriminating power:** target and preservation oracles must distinguish the candidate defect from the closest plausible correct implementation.
4. **No requirement expansion:** do not encode extra performance, style, security or edge-case expectations unless the contract requires them.
5. **No self-grading:** the model that proposed the patch cannot be the sole author or adjudicator of its oracle.
6. **Independent mutation control:** a useful negative control should alter the target mechanism while preserving unrelated semantics; equivalent or collateral mutants remain explicitly labelled.
7. **Oracle weakness disclosure:** report test count, property scope, generated-input strategy, seed, timeouts and known blind spots.
8. **Multi-oracle disagreement:** if contract, tests, static analysis and runtime traces conflict, preserve the disagreement and escalate rather than voting it away.

## 6. Relationship to existing CEB controls

The new dual-witness requirement composes with the existing P/I/F/N/M/IRR controls:

- **P — Parent:** establish pre-introduction behavior and the preserved contract.
- **I — Introducing:** establish the smallest source/configuration condition that violates the target.
- **F — Fixed companion:** establish both target removal and preservation of valid neighbors.
- **N — Neighboring mechanism:** prove the test does not merely detect a nearby but non-target defect.
- **M — Minimal intervention:** show the proposed causal change is minimal within the declared model/scope.
- **IRR — Irrelevant intervention:** confirm the oracle does not react to a semantically irrelevant edit.

Do not create another control family, scheduler, receipt or task schema. The existing control-qualification Q1–Q6 process remains authoritative.

## 7. Over-correction and rationale integrity

Separate four things:

- the verdict (is the implementation defective?);
- the causal explanation (what mechanism violates the contract?);
- the patch (what source change is proposed?);
- the verification result (what the independent target and preservation oracles establish).

A correct verdict does not prove the rationale names the correct cause. A plausible rationale does not prove the patch is safe. A patch passing its generated tests does not prove the generated tests are complete.

Record rationale consistency as an adjudication observation under existing CEB rules. Do not add a composite explanation score or treat longer rationales as better.

## 8. Development experiment: paired preservation challenge

Use the [six-case public development challenge](CEB_REVIEW_PRESERVATION_AWARE_REPAIR_VERIFICATION_CHALLENGE_V0_2026-11-08.md) and separate [answer key](CEB_REVIEW_PRESERVATION_AWARE_REPAIR_VERIFICATION_KEY_V0_2026-11-08.md).

Each case contains a normative contract, a canonical implementation, a single-mutant implementation, the nearest preservation boundary, and the exact target/preservation oracles expected from an expert reviewer. The cases are synthetic, public and permanently ineligible as a sealed holdout.

Compare three development conditions on identical paired cases and matched output/cost budgets:

- **P0 — verdict only:** determine whether each variant conforms.
- **P1 — verdict + rationale + patch:** existing explanation/repair style.
- **P2 — dual-witness:** P1 plus target witness, preservation witness, independently grounded target/preservation oracle plans and claim ceiling.

P2 is supported only if it improves both defect discrimination and preservation reasoning without increasing unsupported claims, unnecessary changes, false acceptance or reviewer burden beyond the existing registered constraints. This is exploratory development analysis; it is not a new confirmatory Phase A hypothesis.

## 9. Diagnostic annotations, not new primary metrics

For the development challenge, report separately:

- correct acceptance of canonical variants;
- correct rejection of mutated variants;
- target-witness correctness;
- preservation-witness correctness;
- unnecessary-patch detection;
- regression/over-repair detection;
- oracle independence and authority errors;
- rationale-to-cause alignment;
- human minutes and all-in evidence cost.

These are diagnostic annotations. CEB-REVIEW remains exactly three primary measures: defect_recall_at_5, actionable_precision, reviewer_burden. Do not add a preservation metric lane or blend the annotations into one score.

## 10. Prospective benchmark targets

These are proposed aspirations for a future preregistered, repository-/defect-family-disjoint evaluation, not observed results:

- **Zero critical over-repair events** in the adjudicated critical-security slice.
- **At least 95%** canonical-implementation acceptance on the paired preservation slice, with a confidence interval and independently checked oracle.
- **At least 95%** target-mutant detection on the paired defect slice, with no inflation from duplicate prompt variants.
- **At least 90%** of proposed patches have both a valid target witness and valid preservation witness before being described as repair-supported.
- No material reduction in the three existing CEB-REVIEW primary measures.
- Report the FPR/FNR trade-off, including absolute counts, rather than optimizing either in isolation.

Do not treat these thresholds as achieved, universal or sufficient for safety certification. Freeze sample size, cluster structure, adjudication procedure and thresholds before confirmatory holdout access.

## 11. Relationship to current research

Jin and Chen (Automated Software Engineering, 2026) report that richer explanation/repair prompts can sharply increase false rejection of correct implementations. Their paired HumanEval/MBPP/QuixBugs evaluation contains over 700 tasks and over 1,400 canonical/buggy instances. Their Fix-guided Verification Filter compares original and proposed implementations on benchmark and specification-constrained tests; averaged FNR falls from 54.8% to 16.3% on HumanEval, 69.0% to 28.9% on MBPP and 51.0% to 24.0% on QuixBugs under Full prompting. The authors also report small FPR increases in some settings and warn that weak oracle coverage can allow buggy implementations to pass. These figures are cross-benchmark conformance results, not CEB-review scores.

**CEB extension:** add a required preservation witness and independent oracle contract, then test the filter's failure mode explicitly: passing both original and patch on a shallow suite cannot prove the patch is unnecessary or safe outside the covered domain.

ReviewGrounder (ACL 2026) separates review drafting from evidence grounding with rubric-guided tool use and reports gains over larger backbones on academic-paper review quality across eight rubric dimensions. It is an adjacent evidence-grounding method, not a software code-review benchmark. CEB can borrow the draft-then-ground pipeline while grounding each software claim in exact source, contract and oracle evidence.

SCOPE (ACL 2026) uses bidirectional preference entropy and conformal calibration to control error among accepted pairwise LLM judgments under exchangeability, reporting empirical FDR around 0.097–0.099 at a 0.10 target and up to 2.4× as many accepted judgments as vanilla baselines. This is a judge-calibration method, not code review. Its useful CEB lesson is to calibrate accepted claims on an appropriate disjoint calibration population, not to treat verbal confidence or judge agreement as a correctness guarantee.

## 12. Integration and stop/go boundaries

- No new primary metric, task schema, receipt, grader, oracle authority, task-admission state or runner is created.
- The new challenge is public development material, not a sealed set.
- No candidate code, Cargo command, build, test, PoC, augmented test, hidden grader or production change is executed by this protocol.
- Executable target/preservation checks remain blocked until #10064 qualification.
- Independent pristine regrade remains blocked until #10369 qualification.
- If specification semantics are ambiguous, do not manufacture a test to force one interpretation. Preserve the ambiguity and ask for an authoritative contract clarification.
- If a target witness exists but preservation scope is unknown, report diagnosis-supported / repair-unverified.

## 13. Runner-independent work sequence

1. Have two independent reviewers complete the six paired cases without seeing the public key.
2. Lock canonical/mutant verdicts, target/preservation witnesses and oracle plans separately.
3. Adjudicate whether the target and preservation oracles are both contract-grounded and genuinely discriminating.
4. Compare P0/P1/P2 development outputs on matched budgets; retain each diagnostic annotation independently.
5. Revise ambiguous cases without overwriting first-pass annotations.
6. Expand only to repository-/defect-family-disjoint admitted material after oracle, privacy, licensing and contamination review.
7. Run executable verification only after #10064 and #10369 are separately qualified.

## 14. Claim boundary

This is a research protocol and synthetic authoring design. It reports no CEB candidate evaluation, no human reliability estimate, no repair safety result and no SOTA achievement. The purpose is to prevent a persuasive defect claim or proposed fix from silently overriding valid behavior.
