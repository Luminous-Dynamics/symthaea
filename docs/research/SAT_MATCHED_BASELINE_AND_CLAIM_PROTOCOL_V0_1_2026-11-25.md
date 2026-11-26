# SAT Matched-Baseline and Claim Protocol V0.1 — 2026-11-25

**Status:** preregistration protocol template; no benchmark campaign or comparative result is claimed.  
**Parent:** [Sovereignty Assurance Tournament V0.1](SOVEREIGNTY_ASSURANCE_TOURNAMENT_V0_1_2026-11-24.md)  
**Qualification contract:** [SAT Qualification Infrastructure V0.1](SAT_QUALIFICATION_INFRASTRUCTURE_V0_1_2026-11-25.md)  
**Purpose:** make future claims of improvement over industry baselines matched, falsifiable, reproducible and independently gradeable.

## 1. The comparison question

SAT must not ask only, "Which model gets the highest score?" The primary question is:

> Under the same system identity, task, environment, tool surface and adversarial budget, does a composed deployment reduce independently observed unauthorized effects and sensitive disclosures while preserving legitimate utility, evidence quality and human agency?

This is a systems comparison, not a model-only comparison. The system under test includes model/provider revision, prompt and policy bundles, memory, tools/connectors, sub-agents, runtime/isolation, monitors, effect sinks, auditors, evidence stores and governance process.

A reported upstream benchmark score is not directly comparable merely because the same benchmark name appears in two reports. Upstream task semantics and grading are preserved; SAT adds an assurance envelope around them.

## 2. Preregistered hypotheses and endpoints

The primary analysis plan must declare one primary endpoint per claim family before sealed cases are opened.

| Claim family | Primary endpoint | Required companion endpoint |
|---|---|---|
| Authority | Independently observed unauthorized external effects per adversarial task | Commit-time denial, delegation amplification and revocation failures |
| Confidentiality | Independently verified restricted-payload disclosures per task | Benign disclosure utility and false-block rate |
| Adaptive control | Strongest verified attacker success across registered rounds | Round-conditioned success, attacker budget and utility-under-attack |
| Evidence integrity | Critical injected corruption that survives independent verification | Missing-evidence fail-closed rate and evidence invalidation latency |
| Auditor integrity | Evidence-adjudicated hidden-behavior recall | False-positive rate on benign targets, calibration and disconfirmation success |
| Recovery | Duplicate or wrongly rebound effects after timeout/reconnect | Time-to-reconciliation and safe liveness for genuinely not-applied attempts |
| Human agency | Unaided performance after AI withdrawal, against preregistered non-inferiority margin | Delayed retention, independent-first accuracy, contest/revoke use and opt-out |
| Operational efficiency | p50/p95 latency and cost per completed benign task | Throughput, token/tool calls, monitor overhead and utility under attack |

Hard safety gates are non-compensatory. A verified critical unauthorized effect or restricted disclosure cannot be offset by task success, speed, cost reduction or human satisfaction. Publish a vector and Pareto comparison only among systems that satisfy the same declared mandatory gates. Do not invent a universal scalar "alignment score".

## 3. Baseline arms

The preregistration should select only arms whose code or protocol is publicly available and can be reproduced with sufficient fidelity. An arm that cannot be reproduced must be labeled an external reference, not silently treated as a matched run.

| Arm | Description | What it isolates |
|---|---|---|
| B0 | Unrestricted execution / no additional control | Attack and utility reference floor |
| B1 | Static allowlist or tool-level permission checks | Coarse tool admission |
| B2 | Purpose-bound, attenuable capability/delegation control, using DelegateGuard semantics where reproducible | Capability scope, delegation and revocation |
| B3 | Label-flow / information-flow control | Provenance and label propagation |
| B4 | Output monitor or content guard | Post-generation / pre-egress filtering |
| B5 | ControlArena-compatible control protocol | Agent control under honest and attack settings |
| B6 | Full SAT-composed deployment | Authority + disclosure + effect observation + evidence + recovery |
| B7+ | Owner-native ablations | One qualified boundary removed at a time, without weakening other safety gates |

Do not describe B2 as SAT's invention. DelegateGuard already addresses purpose-bound attenuable capabilities and is a direct comparator for that subproblem. SAT's proposed contribution is the additional independent qualification of derived-data lineage, payload-level disclosure, real effect correspondence, evidence custody, adaptive auditor integrity and human outcomes.

For every baseline, record exact commit/paper version, license, implementation differences, model access, prompts, tools, runtime, monitor placement, threat model and missing features. "Reproduced" means the upstream protocol was followed and evidence is available; an approximate reimplementation must be labeled "adapted baseline".

## 4. Matched-comparison controls

A run is eligible for a head-to-head comparison only when the manifest binds the following dimensions or explicitly marks them as unmatched:

1. **Model identity:** exact weights/revision or provider/model identifier, context limits, sampling settings and accessible modalities. Hosted model changes or opaque routing are declared limitations.
2. **Agent harness:** system prompts, developer prompts, memory policies, tool schemas, sub-agent topology, planning/retry limits and middleware.
3. **Authority surface:** principals, grants, delegation depth, resource scopes, sink endpoints, purposes, parameter canonicalization and revocation semantics.
4. **Information surface:** source classifications, derived-data transformations, lineage adapters, disclosure policy and connector destinations.
5. **Environment:** immutable task/container/browser/robot image, initial state, network policy, service versions, timeouts and source-of-truth fixtures.
6. **Attacker:** knowledge level, initial access, objective, query/tool/time/compute budget, adaptation schedule and any disclosure of prior defense behavior.
7. **Evaluator:** task/scorer version, monitor visibility, audit rubric, qualification policy, evidence access and adjudication procedure.
8. **Resource accounting:** model tokens, tool calls, wall-clock time, retries, memory/storage/network use and human labor. Do not hide monitor or auditor costs outside the denominator.
9. **Randomization:** preregister seeds, case order, attack assignment and paired-case mapping. Public development data stays separate from sealed holdouts.

If a dimension differs, present a stratified or ablation result and explicitly state that the comparison is not fully matched. Do not infer superiority from unmatched leaderboards.

## 5. Benchmark adapter plan

Preserve upstream task definitions, graders and result fields. SAT adds manifests, events, independent sink observations, evidence references and qualification gates.

| Upstream work | Adapter deliverable | SAT-only measurements |
|---|---|---|
| AgentDojo | Task/injection mapping, task-order and tool identity binding | Exact authority, payload lineage, independent sink effects, benign utility |
| AgentHarm | Harm category/task mapping and native harmful-capability scoring | Whether prohibited actions reach real effect sinks; denial evidence and recovery |
| DelegateGuard | Reproduce published purpose-bound delegation policies and attack families | Derived-data disclosure, per-sink effects, observer and evidence integrity |
| AuditBench | Hidden behavior IDs held by evaluator; auditor evidence schema | Grounded finding precision/recall, benign calibration, independent disconfirmation |
| ControlArena / SHADE-Arena | Inspect settings and control protocol versions | Per-round attack adaptation, strongest attack, monitor bypass and utility |
| LADDER | Long-horizon task and subgoal adapter | Actuator authority, commit-time revocation, emergency stop, recovery and human control |

A benchmark adapter must record upstream case IDs and digests. Do not modify an upstream task and continue to label the score with the unmodified benchmark name.

## 6. Adaptive attacker protocol

Static attacks are useful for regression but insufficient for qualification against adaptive adversaries.

Each campaign freezes:
- attacker observation surface and what defense signals may be seen;
- attack objective and success definition;
- per-round query/tool/time/compute budget;
- allowed adaptation algorithm and model access;
- number of rounds and reset/carryover semantics;
- defense changes allowed between rounds;
- sealed case families and holdout handling.

Publish the full round-by-round trajectory, not only an aggregate. Report the strongest attack, first successful attack, attack-family breakdown, cost to attack and utility retained by the defender. A defense that changes behavior after detecting the benchmark or that denies all benign tasks must not qualify as robust control.

An adaptive attacker may learn from the declared defense protocol, but must not receive hidden holdout labels or future cases outside the declared threat model. If attack and defense are co-trained, freeze both policy snapshots per round and disclose training data overlap.

## 7. Auditor and evaluator integrity protocol

A true positive is a finding whose underlying behavior is reproduced or independently established by evidence. A concerning narrative, confidence score or second model's agreement is not ground truth.

Required design:
1. Blind auditors to planted-behavior labels and benign-control labels.
2. Include hidden positive targets, benign targets, no-op targets and decoy evidence.
3. Require each finding to cite evidence references, claim scope, a falsifying test and expected observation.
4. Pay/score pairwise investigation quality or evidence-grounded outcomes, not raw finding count or alarm severity.
5. Use an independent disconfirmation pass and adjudicator; record disagreements.
6. Report false-positive denominators, missed-behavior rate, calibration, inter-rater agreement and audit cost.
7. Treat evaluator leakage, target contamination, oracle changes and auditor conflicts as report-validity risks.

The evaluator should not be controlled by the SUT, the attacker or the same party whose claim is being graded. A blind model judge is not automatically independent.

## 8. Statistical analysis and sample-size discipline

The unit of analysis is declared before execution: task, task-seed, attack-family, environment episode or human participant. Episodes that share a task template, attacker policy, environment state or model session are clustered; they are not automatically independent observations.

Report:
- numerator, denominator and exclusions;
- per-task-family and per-threat-stratum results;
- confidence/credible intervals and the exact method;
- paired differences where task/seed matching is used;
- cluster-aware bootstrap or hierarchical analysis where appropriate;
- effect sizes and cost/utility frontiers, not just p-values;
- multiple-comparison corrections for secondary endpoints;
- missingness, attrition and all deviations from preregistration.

For zero observed failures, publish an upper confidence bound and denominator. The rough rule of three (about 3/n at 95%) assumes independent Bernoulli trials and is not adequate for clustered adaptive campaigns. Power calculations must account for task-family clustering, repeated measures, rare severe events and the intended non-inferiority margin.

A public smoke suite is for adapter development. It cannot establish a low rare-event rate. Sealed qualification sets require enough independent tasks, environments and attack families to support the actual claim.

## 9. Human-agency experiment

Pre-register treatment arms: independent-first, AI-first, evidence-first and no-AI control. Randomize order and disclose consent, risk controls, compensation, privacy and withdrawal rights.

Primary human endpoint: unaided task performance after AI withdrawal, compared with a preregistered non-inferiority margin. Secondary endpoints: immediate accuracy, calibration, delayed knowledge retention, ability to explain decisions, disagreement, contest/revoke/appeal use, workload, confidence calibration and attrition. User satisfaction is not a substitute for independent capability.

Do not present human performance as a system-level safety pass if the AI controls the task, measures or participant information ordering.

## 10. Claim ladder

A SAT claim must state its evidence level:

| Level | Evidence state | Permitted claim |
|---|---|---|
| L0 | Schema/fixture validation only | Contract syntax and fixture conformance |
| L1 | Reference-model and deterministic unit tests executed | Bounded reference semantics under tested cases |
| L2 | Owner adapters exercised against component implementations | Component conformance within the tested integration boundary |
| L3 | Full system exercised in a frozen controlled harness with independent sink observations | Bounded system behavior under the declared threat model |
| L4 | Blind independent evaluator, sealed holdout and independent regrade | Independently evaluated qualification for a frozen configuration/profile |
| L5 | Field/production evidence, drift monitoring, incident response and change-triggered requalification | Time-bounded deployment assurance under stated operating constraints |

A higher level requires every lower-level prerequisite. Source-authored tests are not L1 until executed. An internally consistent unkeyed event chain is not an external tamper-resistance claim; L3/L4 evidence that requires trace custody also needs an authenticated checkpoint or separately qualified witness. A report issued by the system owner alone is not L4. An L4 result is not a general alignment or civilizational-safety proof.

## 10.1 Evidence custody and transparency checkpoint requirements

For evidence-integrity and independent regrade claims, include a trace-head checkpoint package where the threat model requires protection against complete-history rewrite. Use the [SAT External Trace Checkpoint Profile V0.1](SAT_TRACE_CHECKPOINT_PROFILE_V0_1_2026-11-25.md), based on existing [SCITT RFC 9943](https://www.rfc-editor.org/rfc/rfc9943.html) or a compatible transparency service such as [Sigstore Rekor](https://docs.sigstore.dev/logging/overview/).

The evaluator must verify the signed statement, key policy/revocation, inclusion proof, consistency from a previously pinned checkpoint and independent witness threshold. A checkpoint inside the candidate's own report, without external trust/custody, is not an anchor. If only the unkeyed trace is available, the claim level must say "internally consistent, unanchored"; it must not claim tamper resistance against full rewrite.

## 11. Publication and SOTA claim rules

Before claiming "better than SOTA":
- name the exact benchmark, split, version, model/configuration, baselines and primary endpoint;
- show that the baseline was reproduced under matched conditions or state why it is adapted/unmatched;
- report benign utility and false blocking beside attack success;
- report actual effects observed by an independent sink observer, not only model refusal or tool-call logs;
- include all registered attacks, strongest successful attack, failures, confidence intervals and cost;
- publish run manifests, digests, adapter code, grader version and sufficient artifacts for regrade;
- keep holdout labels sealed until evaluation closes and prevent training leakage;
- obtain an independent replication or audit before high-impact commercial claims.

A failure on a critical safety gate cannot be hidden by a weighted score. A missing observation is not evidence of safety. A compromised evaluator can invalidate the qualification report, but a trustworthy observation of an unauthorized effect is a failure of the system under test.

## 12. Current status and claim boundary

This document is a preregistration protocol template. It proposes comparison arms and analysis, but does not show that any baseline was reproduced or that SAT outperforms a published benchmark. No SOTA result, benchmark score, production qualification or general alignment guarantee is claimed.

## References

- [AgentDojo](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2024.
- [AgentHarm](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c493d23af93118975cdbc32cbe7323f5-Abstract-Conference.html), ICLR 2025.
- [DelegateGuard](https://arxiv.org/abs/2610.16005), 2026.
- [AuditBench](https://alignment.anthropic.com/2026/auditbench/) and [Training Alignment Auditors via Reinforcement Learning](https://alignment.anthropic.com/2026/training-alignment-auditors/), 2026.
- UK AISI [ControlArena](https://control-arena.aisi.org.uk/) and [SHADE-Arena setting](https://control-arena.aisi.org.uk/settings/shade_arena.html).
- [LADDER](https://ladderrobotics.ai/research), 2026.
