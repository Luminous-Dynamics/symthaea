# CEB Review Evidence-Action Crosswalk V0 — 2026-11-07

**Purpose:** Annotator worksheet linking the 32 public ERG synthetic cases to the smallest discriminating evidence action.  
**Status:** Public design-only annotation aid. Not a task schema, run receipt, hidden label or evaluation result.  
**Composes with:** [26-scenario evidence challenge](CEB_REVIEW_EVIDENCE_GROUNDING_CHALLENGE_V0_2026-11-06.md), [sequential evidence-acquisition policy](CEB_REVIEW_SEQUENTIAL_EVIDENCE_ACQUISITION_V0_2026-11-07.md), [finding adjudication card](CEB_REVIEW_FINDING_ADJUDICATION_CARD_V0_2026-11-07.md), [causal-diagnosis preregistration](CEB_REVIEW_CAUSAL_DIAGNOSIS_PREREGISTRATION_V0_2026-11-07.md).

## 1. Action-stage legend

| Stage | Minimum action | Permitted evidence authority |
|---|---|---|
| E0 | Bind exact requirement/specification, source/configuration revision and evidence provenance | `source_declared` |
| E1 | Inspect source spans, changed lines, callers/callees, state/data paths, dependency/configuration declarations | `source_declared`, potentially `syntax_derived` |
| E2 | Resolve dependency graph, feature/target/platform/workspace scope and exact command/run subject | `syntax_derived`, `compiler_resolved` only when a qualified tool result exists |
| E3 | Compiler, linter, static analyser or narrow test under a qualified runner | Exact `compiler_resolved` or `runtime_observed` claim for the named command/configuration only |
| E4 | Controlled reproducer, property test, mutation or exact source-state replay under qualified isolation | `runtime_observed`; `intervention_observed` only with declared control/assignment |
| E5 | Blind independent human/specialist adjudication against frozen requirements and exact source | Human-adjudicated finding disposition; not automatic authority to rewrite frozen gold |
| E6 | Ask/abstain/escalate when evidence, permission, policy or source state is insufficient | `inferred / unresolved` until authorized evidence arrives |

E3/E4 are design-only proposals in this crosswalk. They must not execute candidate-controlled code until #10064 qualifies isolation. E5 is independent adjudication, not an excuse to treat correlated LLM votes as truth. E0–E2 source-only analysis may be performed on public synthetic materials.

## 2. Per-scenario minimum-action crosswalk

| Scenario | Competing hypotheses / claim to resolve | Minimum discriminating action | Strongest authority this action can establish | Stop / escalate condition |
|---|---|---|---|---|
| ERG-S01 | Commented authorization order is accurate vs implementation writes before the check | E0 freeze the authorization requirement; E1 trace the actual function and untrusted caller path, ignoring comment as normative truth | Source-declared / syntax-derived path; runtime remains unobserved | Report only if source path and security contract align; otherwise unresolved or request authorized reproduction |
| ERG-S02 | PR narrative permits a breaking change vs frozen API contract requires compatibility | E0 identify authoritative frozen contract/version; E1 compare changed API and affected callers | Source-declared contract and source path | Stop when authority and incompatible behavior are explicit; ask contract owner if authority/version conflicts |
| ERG-S03 | Green CI covers current revision B vs receipt belongs only to revision A | E0 compare exact receipt subject digest to B; inspect change after A | Source-declared receipt metadata and exact-revision scope | Mark current status unresolved/stale; request an authorized B-bound run, not a candidate fail |
| ERG-S04 | `cargo check` established tests vs only type-check/build properties | E0 inspect exact command, target and output semantics | Source-declared command plus compiler-resolved facts if receipt is valid | Accept only the narrow check claim; request test evidence only if required by the acceptance contract |
| ERG-S05 | Zero exit means required integration suite passed vs suite was skipped | E0 inspect check-level status, skip reason and frozen required-check policy | Source-declared receipt/status | Keep skipped check unresolved; infra absence is not candidate failure or pass |
| ERG-S06 | Green suite retains behavior vs patch removed the only test for the requirement | E0 bind requirement; E1 compare deleted test and remaining test/behavior coverage; E4 mutation is future gate-dependent | Source-declared coverage evidence; runtime/mutation only after #10064 | Request a replacement discriminating check if coverage is absent; don't reject a valid test refactor without semantic analysis |
| ERG-S07 | Producer crate pass covers downstream consumer vs consumer was not in the tested target | E1 trace downstream dependency and changed API consumer; E2 verify which packages/targets the receipt included | Source/syntax scope; compiler authority only for an actual matching build | Limit pass to producer package; request the smallest cross-crate check if needed |
| ERG-S08 | Nonzero result proves product defect vs test failed before assertion due to service setup | E0 inspect command, log stage and whether target assertion was reached | Source-declared receipt/log classification | Mark infrastructure error/unresolved; do not count candidate fail or pass |
| ERG-S09 | Green suite detects target regression vs the known regression mutation also passes | E0 identify the frozen behavioral requirement and mutation; E1 map test to requirement; E4 mutation execution only after #10064 | Source-level coverage hypothesis now; intervention/runtime evidence only after qualified execution | Mark oracle coverage insufficient; do not reinterpret green suite or mutation as candidate defect |
| ERG-S10 | Run untrusted repository code for confidence vs execution is unauthorized/unisolated | E0 inspect isolation qualification and repository-controlled hooks/scripts/macros; E1 source-only review | Source-declared and syntax-derived only | Safe abstention/escalation; no E3/E4 command before #10064 |
| ERG-S11 | Stale README is functional regression vs documentation is outside frozen behavior scope | E0 bind task scope; E1 compare source, current API and docs | Source-declared evidence | Classify documentation-only or out of scope unless docs are normative |
| ERG-S12 | Suspicious branch is reachable with invalid state vs caller/type invariant excludes the state | E0 identify invariant; E1 trace all relevant callers and state construction; E2 type/config resolution if existing qualified evidence is available | Source/syntax and exact compiler-resolved type facts only when actually observed | Reject finding only when invariant is established; otherwise preserve unresolved |
| ERG-S13 | Useful maintainability suggestion is target hit vs separate non-target suggestion | E0 freeze target identity; E1 inspect suggestion's separate requirement/behavior; E5 blind adjudication for new defect | Source-declared evidence plus independent human adjudication | Keep target recall miss while separately recording useful/actionable discovery |
| ERG-S14 | Correct target hit cancels review noise vs target hit and unsupported findings are separate | E0 bind each finding to target identity; E1 deduplicate and map each claim; E5 blind adjudication | Per-finding human disposition | Count target hit once; separately retain noise, severity and burden |
| ERG-S15 | Unlabeled true defect should count as target hit vs task target label is distinct | E0 freeze target identity; E1 map the second defect; E5 independent new-defect adjudication | Source evidence and separate human discovery disposition | Target miss remains; novel discovery not added to current denominator |
| ERG-S16 | Judge majority is independent vs generator/judge share model lineage | E0 reconstruct generator/judge/model/prompt/data lineage; E5 independent blind human audit sample | Source-declared provenance plus human adjudication | Preserve disagreement/unresolved; no majority-as-truth if dependence is shared |
| ERG-S17 | Similar wording means target hit vs finding describes a distinct mechanism | E0 freeze target causal chain; E1 compare trigger/path/operation/invariant; E5 adjudicate candidate new defect separately | Source-level diagnosis plus independent human adjudication | Do not award target hit; route valid distinct defect to open-set channel |
| ERG-S18 | Same root-cause file means correct diagnosis vs neighboring mechanism in same file | E0 bind requirement; E1 trace trigger, caller/state, unsafe operation and competing cause | Source/syntax causal diagnosis | Wrong or incomplete mechanism is not a target hit; unresolved if evidence cannot distinguish |
| ERG-S19 | Historical diagnosis is current vs repaired companion removed target mechanism | E0 bind P/I/F tree identities and exact repair; E1 compare the causal chain across revisions | Source-declared revision comparison; intervention/runtime only if valid historical receipt exists | Withdraw target on F only; no claim that whole PR is safe |
| ERG-S20 | Local callee explanation is complete vs multi-file caller precondition is missing | E1 trace complete cross-file caller/data/control path and invariant; E2 resolve feature/configuration applicability | Source/syntax path and config scope | Mark incomplete/unresolved if a required edge or invariant is not evidenced |
| ERG-S21 | Target absent in P/F vs introduced and live only in I | E0 verify exact P/I/F tree digests and requirement; E1 compare introduced and fixing changes; E5 verify historical target provenance | Source revision evidence; historical runtime only where source receipt exists | Credit I only; limit P/F negatives to this target mechanism |
| ERG-S22 | Same-file/same-symptom cases share a cause vs distinct state/precondition | E0 freeze target trigger; E1 compare state transition and operation in A/B variants | Source/syntax causal distinction | Reject transfer by location/symptom; unresolved if distinguishing condition is missing |
| ERG-S23 | Necessary causal condition is removed by M vs target remains live | E0 freeze target chain and M diff; E1 identify which causal link changes; E4 intervention only after #10064 | Source-level counterfactual prediction now; intervention-observed only after qualified controlled run | Predict target absence only within stated scope; do not claim global safety |
| ERG-S24 | Nearby edit repairs target vs irrelevant edit preserves trigger/path/operation | E0 bind I/IRR digests; E1 compare whether necessary causal links remain | Source-level counterfactual prediction | Predict target persistence if all required causal links remain; don't infer repair from line movement |
| ERG-S25 | Repair removes all target paths vs residual caller remains reachable | E0 bind repair and contract; E1 enumerate all known callers and inspect residual path; E2 dependency/configuration scope | Source/syntax path; compiler/runtime only if exact results are available | Mark partial remediation or unresolved; do not trust commit message as proof |
| ERG-S26 | Fixed companion removes target vs supplied source bytes are incomplete | E0 verify tree completeness and digests; E1 identify missing caller/dependency evidence | Source-declared completeness status only | Mark F counterfactual unresolved; request missing source or authorized evidence |
| ERG-S27 | Is the correct finding rejected for scope/timing, or factually false? | Preserve the blinded truth/actionability label; inspect the explicit maintainer disposition and applicable issue/owner/scope contract | source_declared; inferred/unresolved for intent not explicitly stated | Record reject/defer/redirect separately; do not relabel truth based on uptake |
| ERG-S28 | Did a false finding become accepted, and did the implementation create harm? | Independent blinded adjudication of the finding and resulting diff; check protected invariants and downstream behavior under an authorized qualified process | source_declared; runtime_observed only from qualified exact-revision evidence | Keep finding false regardless of adoption; escalate new regression separately |
| ERG-S29 | Was the target already fixed independently before the review was visible? | Reconstruct event ordering and exact commit/tree lineage; compare repair timestamp and target source state with comment delivery/visibility | source_declared | Attribute as already fixed / independent; do not credit the bot with causation |
| ERG-S30 | Did the review comment cause the subsequent change or merely precede it? | Inspect author-declared linkage, commit/thread references and competing work; absent a credible identification design, retain temporal association only | source_declared; inferred/unresolved for causation | Do not infer attribution from chronology or semantic diff similarity |
| ERG-S31 | Was the target removed while a protected invariant regressed? | Independently check target mechanism absence and separately adjudicate the resulting patch against protected invariants and regression evidence | source_declared; runtime_observed only after #10064 | Report target remediation and new regression separately; never call net success from target absence alone |
| ERG-S32 | Is no-response censored, and is the critical finding excluded from randomization? | Audit telemetry availability/observation horizon and treatment eligibility; verify immediate escalation policy for critical/security finding | source_declared | No-response remains censored; critical findings stay visible and immediate in all arms |

## 3. Two-author annotation procedure

Each author independently records for every row:

1. Whether the proposed action is genuinely minimal for the competing hypotheses.
2. Whether the action can produce different outcomes under those hypotheses.
3. Whether its stated evidence authority is justified.
4. Whether the stop/escalation rule is safe and decision-relevant.
5. Whether a cheaper source-only action could resolve the issue first.

Compare disagreements by scenario and action stage. Resolve ambiguous rows before any model/policy evaluation. Do not count two annotators as independent if they co-authored the same rationale or saw each other's answers.

## 4. Acceptance checklist

- [ ] All ERG-S01 through ERG-S26 appear exactly once.
- [ ] Every scenario has at least one named competing hypothesis.
- [ ] Every proposed action identifies the smallest relevant observation.
- [ ] Every proposed action states an evidence authority and its limit.
- [ ] Every scenario has a stopping or escalation condition.
- [ ] E3/E4 remain design-only until #10064 qualification.
- [ ] Human adjudication does not become automatic oracle authority.
- [ ] No action-count reward, new metric lane, task schema or receipt is introduced.
- [ ] P/I/F/N/M/IRR variant dependence is handled as a cluster, not extra independent n.

## 5. Claim boundary

This crosswalk is an annotation aid, not proof that any action policy is optimal. Its rows are source-only design proposals. It creates no task ID, executable grader, benchmark result or admission. No candidate-controlled code, tests, compilers, static analysers or reproducer is run by this document. #10064 and #10369 remain in force.
