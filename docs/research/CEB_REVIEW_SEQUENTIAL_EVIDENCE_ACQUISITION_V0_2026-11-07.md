# CEB Review Sequential Evidence Acquisition V0 — 2026-11-07

**Status:** Public policy/experiment design. No candidate execution or benchmark result is claimed.  
**Owner:** Existing CEB-REVIEW / CEB-HUMAN / causal-diagnosis work; #10723, #10367, #10419.  
**Primary measures:** Existing CEB-REVIEW only — `defect_recall_at_5`, `actionable_precision`, `reviewer_burden`.  
**Gates:** #10064 candidate-controlled execution; #10369 independent regrade.

## 1. Research question

Can a reviewer spend a bounded evidence budget more intelligently by selecting the next observation that best separates its leading causal hypotheses, stopping when the decision is sufficiently supported, and escalating when further evidence is unsafe, unavailable or low-value?

This moves beyond "use tools" versus "do not use tools." It evaluates **sequential evidence policy**: what should be inspected next, why that observation is discriminating, which claim it can license, what it costs and when the reviewer should stop.

A command count is not an evidence-quality metric. A tool call earns no credit because it ran. A static source trace may be stronger than a test that exercises the wrong path. A test may be highly discriminating for one claim but irrelevant to another.

## 2. The paradigm shift: active evidence acquisition as experiment design

Treat each consequential diagnosis as a small scientific inquiry:

1. State the target proposition and its scope.
2. List the two or three nearest alternative explanations that could change the review disposition.
3. Identify which observation would distinguish those explanations.
4. Check whether acquiring that evidence is authorized and safe.
5. Choose the smallest applicable observation with the best expected decision value per total cost.
6. Bind the result to its exact source/configuration/revision and evidence authority.
7. Update the disposition; stop, escalate, or choose a new discriminating observation.

No numerical posterior or expected-information-gain score should be invented unless the relevant probability estimates are calibrated on a separate development set. Before calibration, use a transparent ordinal rationale (high/medium/low discrimination, applicability, safety and cost) and preserve uncertainty.

## 3. Evidence action ladder

Actions are ordered by typical invasiveness, not as a mandatory fixed sequence. The policy may skip any inapplicable action.

| Stage | Evidence action | Can support | Cannot establish by itself |
|---|---|---|---|
| E0 | Frozen requirement, source/config and revision identity | What the requirement and inspected bytes say | Runtime behavior |
| E1 | Local source/symbol/call-site/data-flow inspection | Declared control/data paths and source-level causal candidates | Dynamic reachability or observed behavior |
| E2 | Dependency, feature, target, platform and workspace resolution | Which components/configurations participate | Tests passed or production behavior |
| E3 | Parser/type/compiler/linter/static-analysis result | The exact syntax/type/config/property checked by that tool | Unchecked semantics or full behavioral correctness |
| E4 | Narrow unit/integration/property test | Behavior for the tested input/path/environment | Untested paths, other configurations or all consumers |
| E5 | Controlled reproducer, mutation or minimal intervention | Whether the named causal hypothesis survives the declared intervention | Whole-PR safety or untested side effects |
| E6 | Independent reviewer / specialist adjudication | Independent challenge under a declared evidence scope | Ground truth if reviewer/judge dependence is unqualified |

Stages E3–E5 require qualified candidate-controlled execution. They must not run until #10064 has qualified OS isolation and the applicable environment, output custody and side-effect policy. Existing historical evidence may be reviewed under its own provenance but must not be described as candidate-generated.

## 4. Required action proposal

Before each evidence action, record a short proposal:

- **Claim:** Which exact proposition is unresolved?
- **Alternatives:** Which plausible explanations or dispositions are competing?
- **Observation:** What precise source span, graph edge, compiler configuration, test, mutation or expert review will be requested?
- **Discrimination:** What result would favor each alternative? Could the result be inconclusive?
- **Applicability:** Is the observation bound to the exact source revision, feature set, platform, dependency and environment needed for the claim?
- **Authority:** Which existing evidence class can this action establish?
- **Safety:** Is execution/inspection authorized? What could the action read, write, invoke or expose?
- **Cost:** Expected model/tool/runtime/human/queue cost; actual cost after execution.
- **Stopping implication:** What result would make the next action unnecessary, and what would trigger escalation?

If the reviewer cannot identify a plausible result that changes its disposition or licenses a more precise claim, the action is probably non-discriminating and should not be taken.

## 5. Sequential policy arms

Use paired frozen tasks and equal total budget, where feasible:

### S0 — Static-first fixed sequence

A fixed, transparent source/context sequence. This is a reproducible baseline and may waste effort on irrelevant actions.

### S1 — Unconstrained model-selected tools

The model chooses tools without an explicit evidence/discrimination contract. This captures common agentic practice but must be constrained by the same safety permissions and cost cap.

### S2 — Evidence-directed heuristic

A transparent rule chooses the smallest authorized observation with high applicability, high discrimination and low cost; it stops when the evidence supports a disposition or no safe useful action remains.

### S3 — Calibrated sequential value-of-information policy

After a development-only calibration phase, estimate which observation is likely to change a consequential decision and at what cost. Choose actions by a predeclared utility policy. Calibration inputs and thresholds are frozen before holdout access. Do not call this Bayesian or information-theoretic unless the quantities actually satisfy those semantics.

### S4 — Ask/abstain/escalate policy

The reviewer stops acquiring evidence and requests a specific human/authority input when evidence is not obtainable safely, a normative ambiguity requires an authorized decision, or additional checks are unlikely to change the justified disposition.

The S0–S4 arms compare policies, not raw tool volume. Include a source-only arm to measure whether any evidence acquisition adds value at all.

## 6. Matched-budget accounting

For each paired task and arm, account for:

- model input/output/reasoning/cache tokens and provider cost;
- retrieval and index/search cost where measurable;
- compiler, static-analysis, test, reproducer or environment compute;
- setup, infrastructure wait, retries and failed evidence acquisition;
- evidence artifact storage and verification;
- independent judge and human-adjudication cost;
- second reviewer and disconfirm-first effort;
- engineer time to reproduce, dismiss or act on findings.

Report at least:
1. quality at fixed all-in budget;
2. all-in cost at non-inferior quality;
3. the quality/cost Pareto frontier;
4. marginal cost and decision effect of each selected observation;
5. wasted/non-discriminating action cost.

Do not compare model token cost alone against another arm's all-in cost. Do not turn wall-clock duration into human reviewer burden.

## 7. Stopping and escalation policy

A reviewer should stop when one of the following holds:

- **Supported:** evidence resolves the material alternatives and supports a claim at a bounded authority/scope.
- **Refuted:** evidence contradicts the target mechanism; withdraw or revise the finding.
- **No useful safe action:** all remaining observations are unauthorized, unsafe, inapplicable or unlikely to discriminate; preserve unresolved and request the minimal human input.
- **Budget exhausted:** do not continue beyond the declared cap; report unresolved rather than overstate.
- **Human authority needed:** specification, policy, risk tolerance or permission cannot be inferred by an agent; ask the named decision authority.

Never stop solely because the model "feels confident." Never continue merely because budget remains. Never treat no result, timeout, skip or infrastructure failure as a candidate pass/fail.

## 8. Counterfactual and causal-diagnosis integration

For P/I/F/N/M/IRR states in the causal preregistration, the action policy must identify which observation can distinguish:

- target absent versus present;
- correct cause versus neighboring mechanism;
- necessary condition removed versus irrelevant change;
- full repair versus residual path;
- stale evidence versus current revision evidence.

Prefer source-level counterfactual reasoning before execution is qualified. After #10064, a candidate may request only the smallest test or intervention that discriminates the named contrast. A positive/negative test result is scoped to its exact inputs, revision, configuration and environment.

## 9. Failure and gaming controls

| Failure mode | Required control |
|---|---|
| Tool-call inflation | No activity credit; count only decision-relevant, admissible evidence |
| Test suite run without target coverage | Require an applicability/coverage rationale |
| Compile treated as behavior pass | Preserve compiler versus runtime authority |
| Random tool choice | Compare with source-only and transparent heuristic arms |
| Excessive escalation | Track whether the requested human input was necessary and actionable |
| Premature stop | Include counterfactual cases where one more observation changes the correct disposition |
| Endless investigation | Fixed budget, stop rule and unresolved state |
| Wrong-revision result | Exact subject digest/source/configuration binding |
| Hidden side effect | #10064 isolation and command policy |
| Selective reporting of successful actions | Retain failed, skipped, redundant and inconclusive actions and their cost |
| Judge-created evidence | The judge may not invent a run or promote inference to observation |

## 10. Outcome mapping — existing CEB metrics only

The primary outcomes remain:
- `defect_recall_at_5`;
- `actionable_precision`;
- `reviewer_burden`.

Diagnostic annotations include evidence-action relevance, claim-authority fidelity, stale result use, proportion of decisions changed by a targeted observation, correct stop/ask/abstain, infrastructure-error handling, cost of non-discriminating actions, and residual uncertainty. These do not create new primary metrics or a new lane.

A new action is valuable only if it improves a justified engineering decision, supports a more precise claim or reduces human work without increasing missed defects and false alarms.

## 11. Runner-independent authoring sequence

1. Use the 26 public ERG scenarios to identify the minimum observation that separates each leading diagnosis from its nearest alternative.
2. Have two authors independently label action applicability, discrimination and authority; adjudicate disagreement before coding.
3. Write source-only E0–E2 example traces; include cases where static evidence is sufficient and execution would be wasteful.
4. Add design-only E3–E5 action proposals without running candidate-controlled tools.
5. Preregister equal-cost S0–S4 arms, budget accounting and stopping rules.
6. Review the policy with #10064 isolation/security and #10619/#10624 statistical claim owners.
7. Only after all gates, decide whether the evidence policy merits an admitted pilot.

## 12. Claim boundary

This document specifies a sequential evidence policy and experiment. It does not show that one policy outperforms another, that more tools improve review, or that expected-information-gain estimates are calibrated. No candidate-controlled code, builds, tests, static analyzers, reproducer, PoCs, graders or benchmark tasks were executed. Execution remains blocked until #10064; independent pristine regrade remains blocked until #10369.
