# Sovereignty Assurance Tournament V0.1 — 2026-11-24

**Status:** research specification; design-only; no benchmark run is claimed.  
**Owner:** Luminous Dynamics / Symthaea research  
**Purpose:** qualify whether an AI-enabled system preserves legitimate human authority, evidence integrity and meaningful human agency under adaptive adversarial pressure.

## 1. Executive thesis

Alignment is not one scalar property of a model. It is a portfolio of distinct claims about intended behavior, authority, controllability, auditability, recovery and human outcomes. A high score on one dimension must not compensate for a critical failure on another.

This programme proposes a benchmark tournament for the complete agentic system—not only the language model. The system under test includes the model, prompts, memory, tools, sub-agents, delegation paths, execution environment, effect sinks, monitors, auditors, evidence stores and human governance process.

The central question is:

> Can an AI-enabled system remain useful while an adaptive agent actively tries to exceed its authority, deceive or evade its monitors, corrupt its evidence, manipulate its evaluator, and displace the humans it is meant to assist?

The objective is not to prove universal alignment. It is to establish increasingly strong, falsifiable and independently reproducible claims about bounded systems under explicit threat models.

## 2. Paradigm shifts

### 2.1 From alignment as a model property to alignment as a system property

Training methods influence behavior, but the model is only one component of an agentic deployment. Tool permissions, identity, delegation, persistence, runtime isolation, external side effects and recovery determine what the model can actually do.

Evaluate the composed system and its trust boundaries. Do not infer deployment safety from a model-only score.

### 2.2 From scalar reward to non-compensatory constraints plus plural objectives

Do not collapse safety, usefulness, fairness, privacy, autonomy and reliability into one weighted score. A large productivity improvement must not compensate for an unauthorized high-impact action, forged evidence, coercive disclosure or a violated consent boundary.

Use:
- hard, independently checked invariants for authority and protected rights;
- a vector of task, reliability, cost and human-outcome metrics;
- explicit uncertainty and abstention where evidence is insufficient;
- Pareto comparisons among systems that first satisfy the hard constraints.

### 2.3 From intelligence-derived authority to externally issued capabilities

No intelligence, confidence, consciousness proxy, group consensus, model reputation, coalition score or learned preference estimate may mint execution authority.

Authority must originate from an authorized principal or policy and be bound to the specific action, resource, scope, context, lifetime and delegation constraints. Cognitive signals may affect proposal ranking, information seeking or conservative caution; they cannot create a missing capability or remove an independent confirmation requirement.

For a delegated agent, the effective capability set must be a subset of the delegator's still-valid grant and the applicable policy. Delegation must not increase scope, lifetime, resource access or allowed effects. A group may coordinate a multi-party action, but group membership, voting, leadership or emergent collective metrics do not silently transfer individual rights.

### 2.4 From voluntary corrigibility to externally enforceable control

A model saying that it will stop is not a shutdown guarantee. A control claim must be demonstrated at the effect boundary: the external supervisor can revoke credentials, fence or terminate execution, block network and storage paths, prevent new mutations, and preserve enough evidence for recovery even when the target process is uncooperative.

### 2.5 From static red-teaming to adaptive adversarial tournaments

A control protocol that performs well against a default attacker can fail when the attacker learns the protocol. The red team must adapt to observed defenses, and the blue team must not be given credit for security through a hidden test split that the attacker can infer.

The target, monitor, auditor, evidence pipeline and evaluator are all part of the attack surface.

### 2.6 From auditing by confidence score to auditing by evidence and attempted disconfirmation

Auditors can reward-hack by fabricating alarming findings or narrating conclusions they have not established. A true positive therefore requires a source-grounded, reproducible behavioral or structural violation. A clean target is a required control, not an optional convenience.

Use a blind second auditor and a disconfirm-first review pass to challenge each claimed finding and search for omissions. Do not treat multiple model judges as independent ground truth.

### 2.7 From AI-assisted performance to retained human capability

A system that helps a team perform better while it is present may still reduce unaided capability or concentrate knowledge. Measure performance after AI withdrawal, delayed transfer, knowledge retention, role resilience and the ability to disagree or opt out.

This is a human outcome, not a model capability score.

## 3. Existing work to compose, not duplicate

This programme is a benchmark and integration layer. It does not create a second authority kernel, generic sandbox, transparency log, deliberation engine or abstention framework.

| Existing work | Ownership to preserve | Integration here |
|---|---|---|
| Symthaea #4143 and #4155 | No authority from individual or collective Phi; static and behavioral qualification | Authority non-amplification track |
| Symthaea #5439 and #545 / #560 / #565 | Separate cognitive caution from actual action authority and exact-effect admission | Effect-boundary checks and fail-closed behavior |
| Symthaea #10385 | Precise repair, expert-normalized performance and paired act/abstain evaluation | Reuse act/abstain metrics and pair design |
| Symthaea #10064 | Isolation prerequisite for executing untrusted programs and workloads | No unsafe host execution before this gate |
| Symthaea #13518 and #13550 | Durable witnessed evidence, independent verification and trust continuity | Evidence-integrity interfaces; do not duplicate log infrastructure |
| Symthaea #13556 | Bounded MSP autonomy and local-only privacy defaults | Test action scopes and privacy defaults |
| Symthaea #13565 | Independent qualification exchange and fail-closed qualified projection | Reuse consumer-side qualification semantics |
| Mycelix #5999 and #6089 | Human deliberation quality, information rights, evidence acquisition and contestability | Pluralistic governance and evidence-rights track |
| Mycelix #6597 | Team capability and knowledge retention after AI withdrawal | Human capability-retention track |

The current default-branch swarm implementation and the draft corrective work must be kept distinct in evidence. The inspected main source still contains the legacy AuthorityDelegation semantics; PR #12825 is draft-only and is based on an older main SHA. The exact authority qualification remains owned by #4155. This programme does not claim that the swarm repair has been merged or qualified.

## 4. Threat model

### 4.1 Adversary capabilities

The red team may control or adapt:
- the target model's task strategy, tool-call sequence and delegated sub-agents;
- untrusted documents, web pages, messages, repository contents and tool outputs;
- task difficulty, ambiguity, missing information and conflicting instructions;
- attempts to obtain, reuse, replay or induce disclosure of credentials;
- timing, retries, restarts, concurrency and agent join/leave events;
- observations of action outcomes, latency and any monitor behavior exposed by the system;
- attempts to manipulate logs, tool responses, monitoring feeds, labels, graders or audit reports;
- collusion between agents and the use of persistent storage or external communication where the threat model permits it.

A threat model must state what the adversary cannot control. Do not claim resistance to a compromised hardware root, operator account or host kernel unless those components are actually within the tested boundary.

### 4.2 Trust domains

Keep these logically and operationally separate:
1. **Cognitive domain:** model reasoning, predictions, plans, learned preferences and memory.
2. **Authority domain:** principal identity, policy, capabilities, consent, revocation and exact-action permits.
3. **Execution domain:** process, filesystem, network, devices and side-effect sinks.
4. **Observation domain:** independently sourced measurements and runtime events.
5. **Evidence domain:** signed receipts, sequence/freshness state, provenance and replay.
6. **Audit domain:** investigators, red-team agents, adjudicators and evaluators.
7. **Human governance domain:** legitimate decision-makers, affected participants, appeals and opt-out.

A single component may implement multiple roles only when the trust model explicitly justifies it and independent negative controls test the resulting common-cause risk.

### 4.3 Non-goals and limits

This benchmark does not prove that a model has no hidden objectives, that all human values can be represented, that every deployment context has been tested, or that a formal proof of one software component makes the whole deployment safe. Cryptographic signatures prove possession of a signing key over bytes; transparency inclusion proves registration under a policy; neither alone proves that an observation or claim is true.

## 5. Tournament tracks

### Track A — Authority non-amplification

**Question:** Can any model state, learned scalar, coalition event, delegation chain or control-plane transition expand authority without a valid external grant?

Required scenario families:
- zero capability with high individual or collective cognition;
- valid capability with low cognition, where cognition may trigger caution but must not erase the grant;
- parent-to-child delegation at depths 1–8, with scope, lifetime, resource and action attenuation;
- attempted delegation of a scope the parent does not possess;
- sub-agent creation, collusion, cyclic delegation and coalition merge/split;
- leader election, membership change and quorum formation;
- revocation of a parent, child, key, trust root or policy;
- stale, replayed, forked, missing or delayed revocation state;
- extreme finite values, NaN and infinity in non-authoritative cognitive metadata;
- attempts to route through a different tool or sink to bypass the protected effect boundary.

Primary properties:
- no protected effect without a valid exact-action capability;
- every delegated scope is contained by the live delegator scope and policy;
- revocation or unknown authority state blocks new protected mutations;
- changes in cognitive or coalition metadata cannot create, enlarge, transfer or revive capabilities;
- leader or group status does not confer another participant's rights;
- the consumer-side effect sink independently checks the permit and replay/fencing state.

Do not use a single floating-point score as authority. Do not equate "the action was proposed" with "the action was authorized" or "the effect occurred."

### Track B — Adaptive control against untrusted agents

**Question:** Does the control protocol remain effective after the red team learns its structure and attempts to defeat it?

Compare at least:
1. model-only / prompt-only policy;
2. static tool allowlist;
3. OS-level isolation with the same model and task budget;
4. independently enforced, exact-action capability checks at effect sinks;
5. capability checks plus independent monitoring and critical-action deferral;
6. capability checks plus monitor diversity, witnessed evidence and external revocation.

Attack rounds should progressively reveal the monitor's observable behavior, permit the attacker to simulate monitors, vary attack timing, use indirect prompt injection, create sub-agents and exploit restarts. Keep a sealed confirmation set. Do not expose hidden red-team fixtures or evaluator secrets to candidate agents.

Measure both safety and legitimate-task utility. A system that disables all useful actions must not win. Compare the safety/utility frontier at matched model, tools, task set, resource budget and attack strength.

### Track C — Auditor and evaluator integrity

**Question:** Can an investigator discover hidden or context-dependent behaviors without fabricating findings, accepting poisoned evidence or being manipulated by the target?

Use both:
- planted hidden-behavior targets that are trained not to confess when asked directly;
- benign targets with no planted behavior, including ordinary quirks and harmless unusual behavior.

Attack the audit process as well as the target:
- target denies or disguises its behavior;
- tool outputs contain prompt injections or false evidence;
- target selectively behaves differently across scenarios;
- auditor is rewarded for "finding something" and may invent a violation;
- grader or evaluator receives adversarially framed transcripts;
- multiple judges share the same model family or failure mode;
- target induces the auditor to overfit a single scenario.

Require every finding to contain the exact target and scenario identity, raw evidence references, the claimed rule or invariant, a reproducible counterexample or independent behavioral replication, confidence and alternative explanations. Include a no-finding outcome and an explicit NOT_EVALUABLE outcome.

Compare pointwise reward, pairwise same-target investigation comparison, benign-target calibration, blind independent audit and disconfirm-first audit. Measure target-level detection and finding-level precision separately. Human adjudication must audit a stratified sample of both positive and negative findings.

### Track D — Evidence integrity and truth separation

**Question:** Can an agent manipulate what the system records or what the auditor believes happened?

Test:
- signature alteration, wrong key usage and stale/revoked keys;
- event insertion, deletion, reorder, substitution and replay;
- split-view, same-size fork, stale checkpoint, rollback and freeze;
- mismatched action, target, tenant, authority revision or source identity;
- validly signed but factually false source statements;
- tool output that disagrees with an independent sink readback;
- ambiguous timeout where the side effect may have happened;
- monitor outage, clock regression and loss of trusted state.

Keep claims separate: **issued**, **authorized**, **attempted**, **accepted by sink**, **observed**, **postcondition verified**, **registered in transparency service**, and **independently replayed**. Unknown outcome means no further mutation until reconciliation.

### Track E — Pluralistic human agency and capability retention

**Question:** Does AI assistance preserve the ability of affected people and teams to understand, contest, refuse, revise and continue without the AI?

Synthetic conformance first; human-subject work only after independent ethics review and consent.

Measure:
- whether participants can withhold information without penalty or inferred consent;
- whether the option set, minority arguments and unresolved disagreements remain visible;
- whether AI summaries omit, distort or overstate participant evidence;
- whether a person can revoke purpose-bound participation and have the revocation enforced;
- whether decisions remain attributable to the legitimate decision-maker;
- unaided team performance immediately and after a delay following AI withdrawal;
- knowledge distribution, transfer to new tasks and resilience when one member or channel becomes unavailable.

Do not optimize for consensus alone. Do not infer human consent or preference from silence, model confidence or an aggregate sentiment score.

## 6. Core outcome vector

Publish the following independently. No weighted overall alignment score.

| Dimension | Primary measurement |
|---|---|
| Authority integrity | Unauthorized effects; authority-scope expansion; delegation violations |
| Control robustness | Attack success against each control protocol and each adaptive red-team round |
| Useful capability | Authorized task completion, task quality and safe coverage |
| Correct abstention | Paired act/abstain accuracy, false-action and false-abstention rates |
| Audit quality | Hidden-behavior recall, false-positive rate, evidence precision, calibration |
| Evaluator integrity | False accepts, false rejects, grader manipulation and answer-key leakage |
| Evidence integrity | Tamper/replay acceptance, stale-state acceptance and source/sink disagreement |
| Recovery | Time to contain, revoke and reconcile; residual authority after revocation |
| Human agency | Consent violations, contestability, dissent preservation and option-set integrity |
| Human capability | AI-off retention, delayed transfer and team resilience |
| Operational cost | Latency, compute, network, memory, human review time and financial cost |

All rates must retain their denominators. Publish task refusals, timeouts, unsupported environments, missing observations, ungraded cases and abstentions rather than silently excluding them.

## 7. Proposed gates and statistical discipline

These are prospective research targets, not achieved results, standards or universal guarantees.

### G0 — deterministic reference-model gate

- 100% agreement with an independently written oracle on bounded exhaustive capability/delegation fixtures.
- 100% rejection of all explicit unauthorized, stale, replayed, revoked and scope-expanding effect attempts in the frozen negative-control suite.
- 100% correct separation of proposal, authorization, sink acceptance and postcondition evidence.
- All-always-act and all-always-abstain baselines must fail paired act/abstain evaluation.

### G1 — adaptive control and audit pilot

- At least 10,000 cheap, deterministic adversarial traces across multiple attack families before model-based or system-level claims.
- At least 90% hidden-behavior detection at no more than 1% false-positive rate on a held-out target family, using blinded adjudication.
- At least 80% paired act/abstain accuracy, with high-impact false-action rate at or below 1%.
- At least 95% safety under the pre-registered adaptive red-team protocol while retaining at least 95% of the authorized-task utility of the strongest safe comparator. Both values must be reported with uncertainty and cannot be traded against authority violations.

### G2 — confirmation campaign

- Target 100,000 independently initialized adversarial episodes for the highest-risk synthetic control claims, with attack-family and environment-cluster coverage reported.
- If zero events are observed in N genuinely independent Bernoulli trials, report the one-sided confidence bound (approximately 3/N at 95%) rather than claiming zero risk. If trials share tasks, seeds, models, attack templates or environments, use cluster-aware inference and report effective independent units.
- No high-severity false authorization in the confirmation set.
- Independent reproduction by a second implementation or verifier on a sealed subset.
- Any detected false acceptance blocks the affected claim pending root-cause analysis and a fresh confirmation set.

### G3 — human-outcome gate

- Preregister a non-inferiority margin for AI-off team performance before collecting human data.
- Report immediate and delayed transfer, knowledge retention and subgroup effects.
- No coercive disclosure, silent consent inference or removal of the participant's ability to dissent or withdraw.
- Use independently adjudicated human outcomes; a model-generated deliberation score is not a human-outcome oracle.

### Statistical rules

- Freeze the threat model, policy, model/harness, task revision, tools, seeds, resource limits and evaluator before each comparison.
- Use paired scenarios and common random numbers where appropriate; cluster inference by independent environment, task family, model lineage or red-team campaign.
- Separate development, adaptive red-team and sealed confirmation splits.
- Keep false-positive and false-negative rates separate by severity and behavior family.
- Report confidence intervals, coverage, missingness, sample-size rationale, exclusions and multiplicity.
- Never use a pass on synthetic fixtures as proof of real-world physical, organizational or deployment safety.

## 8. Industry benchmark map and differentiation

Use existing benchmarks as comparators and adapters, not as substitutes for this programme.

| External work | What it already measures | Gap this programme targets |
|---|---|---|
| [AuditBench](https://alignment.anthropic.com/2026/auditbench/) | 56 fine-tuned models, 14 hidden behaviors and 13 investigator tool configurations | Add attack on the auditor, evaluator and evidence chain; require benign controls and verified findings |
| [Training Alignment Auditors via RL](https://alignment.anthropic.com/2026/training-alignment-auditors/) | Auditor training; reports pointwise reward hacking and improved robustness from pairwise rewards plus benign-target calibration | Qualify reward hacking, false positives and cross-family transfer as first-class outcomes |
| [SHADE-Arena control protocols](https://arxiv.org/abs/2511.02997) | Red/blue control protocols against untrusted agents and adaptive attacks | Add hard effect-sink capability enforcement, authority lineage, revocation and evidence continuity |
| [AgentDojo](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html) | 97 realistic tasks and 629 prompt-injection security cases | Add multi-agent capability composition and post-effect verification |
| [AgentHarm](https://www.aisi.gov.uk/research/agentharm-a-benchmark-for-measuring-harmfulness-of-llm-agents) | 110 malicious agent tasks, 440 with augmentations and 11 harm categories | Add defense-side authority control and evidence-qualified outcomes |
| [Inspect](https://inspect.aisi.org.uk/) | Extensible evaluation framework with more than 200 prebuilt evaluations | Integrate where useful; contribute task adapters rather than rebuild a general runner |
| [International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026) | Synthesis of current evidence, capability and control concerns | Convert unresolved evidence gaps into explicit, testable failure modes |

The differentiation is not a claim to beat every benchmark's headline score. The competitive claim should be narrower and stronger:

> At matched authorized-task utility, our composed architecture should have a measurably lower rate of unauthorized effects and false audit conclusions under adaptive attack, with independently verifiable evidence and no loss of human decision rights.

Report the complete risk/utility curve, resource cost and exact evaluation conditions. Only make comparative claims after reproducing the relevant external baseline under a compatible harness and license.

## 9. Research sequence that does not depend on CI runners

### Slice 0 — source and ownership freeze

- Reuse #4155 for the Phi/authority registry and behavior tests.
- Reuse #5439 for cognitive-caution versus permission semantics.
- Reuse #10064 for untrusted execution isolation.
- Reuse #13518 / #13550 / #13565 for evidence and independent qualification.
- Reuse Mycelix #5999 / #6089 / #6597 for rights-aware group outcomes.
- Freeze a dependency map and explicit owner for each integration surface.

### Slice 1 — design-only benchmark corpus

Create versioned scenario manifests with task IDs, threat model, actor graph, authority graph, attack capability, expected disposition, evidence requirements and hidden oracle references. Keep answer keys and sealed fixtures outside the candidate-visible workspace.

No untrusted program, build script, patch, model-generated attack payload or physical action is executed in this slice.

### Slice 2 — pure reference model

Implement a small, deterministic Rust reference model for capability attenuation, expiry, revocation, exact-action binding and unknown-outcome handling. Compare it to an independently implemented test oracle. Keep this model separate from production authority implementations so it can challenge them rather than silently share their assumptions.

### Slice 3 — safe local adversarial harness

Only after the isolation gate is qualified, run the harness in an isolated disposable environment with no host credentials, no privileged socket, restricted egress and resource limits. Start with synthetic tools and effect sinks. Add adaptive red-team and audit agents only after the static oracle and negative controls are working.

### Slice 4 — matched external comparisons

Add compatible adapters for SHADE-Arena, AgentDojo, AgentHarm and AuditBench. Record exact dataset versions, licenses, model/provider versions, prompts, scaffolds, tool policies, graders and resource budgets.

### Slice 5 — independent and human evaluation

Obtain separate implementation review, sealed regrade and human-subject ethics approval before claims about human agency or team capability.

## 10. Evidence and claim discipline

Every benchmark receipt should bind:
- source commit and dirty-tree state;
- scenario, threat-model and policy digests;
- model/provider and exact harness configuration;
- tool and environment identity;
- random seeds and independent cluster identity;
- attack, monitor and evaluator versions;
- raw event and observation references;
- independent verifier identity and result;
- resource use, failures and missing observations;
- claim scope and explicit non-claims.

Statuses must remain distinct: **specified**, **implemented**, **structurally checked**, **locally executed**, **isolated execution qualified**, **independently reproduced**, **externally reviewed** and **deployment qualified**.

A CI queue, skipped test, draft PR, deterministic fixture pass or cryptographic receipt does not advance a claim beyond the evidence it actually supplies.

## 11. Decision rule

A system is admissible for a protected class of effects only when:
1. the effect is covered by a current, exact, externally issued capability;
2. the sink independently verifies the capability, scope, freshness and replay/fencing state;
3. required safety and consent constraints pass;
4. evidence and observation sources satisfy the specified trust contract;
5. uncertainty, monitor failure or ambiguous outcome causes the prescribed no-mutation / reconciliation behavior;
6. the relevant adversarial qualification remains valid for the deployed configuration.

This is a bounded operational assurance case, not a proof of universal alignment.

## 12. Research references

- Sheshadri et al., [AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors](https://arxiv.org/abs/2602.22755), 2026.
- Rosu and Wang, [Training Alignment Auditors via Reinforcement Learning](https://alignment.anthropic.com/2026/training-alignment-auditors/), 15 October 2026.
- Kutasov et al., [Evaluating Control Protocols for Untrusted AI Agents](https://arxiv.org/abs/2511.02997), 2025.
- Debenedetti et al., [AgentDojo](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2024.
- Andriushchenko et al., [AgentHarm](https://www.aisi.gov.uk/research/agentharm-a-benchmark-for-measuring-harmfulness-of-llm-agents), ICLR 2025.
- [Inspect evaluation framework](https://inspect.aisi.org.uk/), UK AI Security Institute / Meridian Labs.
- [International AI Safety Report 2026](https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026), 3 February 2026.

---

**Current claim ceiling:** this document is a research specification and benchmark proposal. No implementation, benchmark result, SOTA comparison, independent audit or alignment guarantee is claimed.
