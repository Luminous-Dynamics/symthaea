# Industry Stack Comparison — 2026-11-12

**Status:** research snapshot / proposed comparison programme  
**Owner:** Symthaea, composing existing Luminous subsystem owners  
**Tracking:** [INDUSTRY-STACK-COMP-001 / #11774](https://github.com/Luminous-Dynamics/symthaea/issues/11774)  
**Claim class:** source-qualified industry map; no comparative Luminous performance result is claimed.

## Executive conclusion

Luminous should not try to reproduce six corporations. The highest-leverage position is a vendor-neutral **engineering, intelligence and execution-assurance layer** that can operate above and across incumbent compute, cloud, enterprise, industrial and physical systems.

The potential product is not another generic agent framework. It is the composition of:

- **Symthaea:** reasoning, planning, causal diagnosis, prediction and proposals.
- **Mycelix:** identity, authority, delegation, coordination and governance.
- **Symtropy:** simulation, counterfactual evaluation, physical-state reconstruction and operational feasibility.
- **Sol-Atlas:** spatial/geometry intelligence, coordinate and scene semantics, spatial provenance.
- **Luminous:** engineering and operational digital thread, lifecycle state, change impact, evidence and accountability.

The differentiator must be demonstrated at the boundary where a recommendation becomes an effect: *who authorized it, what exact operation was dispatched, what actually changed, what was observed, what independent check ran, and what remains unknown.*

That does not automatically beat a commercial platform. AWS and Microsoft already provide meaningful agent identity, policy, evaluation and observability capabilities; NVIDIA has reorganized its agent-tooling portfolio; Google is pushing into physical intelligence. Luminous must outperform on a clearly measured residual gap: cross-provider semantics, independent effect verification, lifecycle applicability, fail-closed recovery and total cost per verified outcome.

## 1. Competitor map: what the incumbents actually offer

| Company | Existing strength / evidence | Where Luminous can add value | What not to claim |
|---|---|---|---|
| **NVIDIA** | Accelerated computing, GPUs, networking and a large AI/robotics software ecosystem. NVIDIA's NeMo Agent Toolkit was deprecated in October 2026 and is scheduled for end-of-life at year-end; NVIDIA directs users to NeMo Relay (observability/middleware/governance), NeMo Helix (evaluation/optimization/hardening) and NeMo Fabric (agent harness execution). | Hardware-agnostic orchestration, independent simulation/qualification, lifecycle evidence, physical-operations assurance and multi-vendor workload/energy optimization. | Do not claim to replace GPU silicon, CUDA, semiconductor design/manufacturing or NVIDIA's full simulation/robotics stack. |
| **Google / DeepMind** | Frontier models, search/cloud ecosystems and embodied AI research. Gemini Robotics 2.5 was announced 2026-11-10. Google reports 200+ task evaluations and a 96% result in a specific Apollo 2 automotive-kitting proof of concept. | Domain-specific engineering knowledge, spatial provenance, evidence-qualified decisions and a model/provider-independent control/evidence plane. | Treat the 96% figure as Google-reported pilot evidence, not a matched external benchmark or a general robot reliability rate. Do not try to recreate Search, Android or YouTube. |
| **Microsoft** | Enterprise applications, Azure, GitHub and Foundry agent infrastructure. Microsoft describes end-to-end tracing, evaluation, monitoring and optimization across agents built on different frameworks and OpenTelemetry. | Cross-vendor execution semantics, verified change impact, independent effect truth, configuration/lifecycle traceability and a portable evidence layer. | Generic agent telemetry, enterprise search or a Copilot clone is not sufficient differentiation. |
| **Amazon / AWS** | Cloud scale, logistics and physical operations. Bedrock AgentCore Evaluations became generally available in March 2026 with built-in evaluators, ground-truth assertions and online/on-demand evaluation; AgentCore also offers identity, gateway, policy and observability functions. | Independent verification of actual effects, cross-provider portability, whole-facility/whole-workflow optimization and evidence that can be reconstructed outside the producing runtime. | Do not claim AWS lacks agent governance, policy enforcement or evaluations. |
| **Broadcom** | Semiconductor solutions (including custom AI accelerators/networking) and infrastructure software. Q3 FY2026 reported revenue of $29.6B, up 86% year-on-year. | Infrastructure observability, capacity/thermal/energy optimization, private-cloud lifecycle assurance, hardware-independent scheduling and engineering change impact. | Do not claim to replace advanced semiconductor fabrication, custom silicon, VMware or Broadcom's IP portfolio. |
| **SpaceX** | Vertically integrated launch, connectivity and increasingly large AI infrastructure. Its Q3 2026 SEC-filed results describe Space, Connectivity and Super Intelligence segments and 1.7 GW nameplate compute as of 2026-09-30. | Engineering digital thread, configuration/change impact, test/mission assurance, fleet digital twins, infrastructure optimization and reproducible operational evidence. | SpaceX is not a plug-compatible enterprise-agent product. Physical launch, satellite and manufacturing capabilities require capital, facilities and flight/field evidence. |

### Competitive implications

1. **The agent-platform layer is crowded.** Observability, trace collection, tool gateways, policy engines, model evaluation and human approval are already offered by major vendors. A generic "agent control plane" is not enough.
2. **The strongest wedge is evidence across boundaries.** A provider's trace says what its runtime recorded. Luminous should independently reconstruct whether the intended effect occurred and whether the evidence is sufficient for the downstream engineering claim.
3. **Avoid vendor lock-in in both directions.** Treat provider adapters as replaceable, pin exact capability/version profiles, and fail visibly when a provider cannot preserve required semantics.
4. **Physical-world claims need a higher evidence tier.** A simulator result, an HIL run, an instrumented laboratory trial and field operation are distinct claim classes.
5. **The product must save money or time while improving assurance.** Assurance that is too expensive, slow or cumbersome will not displace incumbent workflows.

## 2. Paradigm shifts to pursue

### Shift A — from generated answers to verified effects

A completion message is not proof of completion. Define a transaction boundary:

`intent \u2192 authority \u2192 prepare \u2192 dispatch \u2192 observe \u2192 independently verify \u2192 commit/reconcile`

Record a dispatch before execution. An ambiguous timeout becomes `OUTCOME_UNKNOWN`; no new mutation is allowed until a content-bound reconciliation proves the prior outcome. Do not claim distributed exactly-once execution unless the external effect system actually provides the necessary idempotency and durable transaction semantics.

### Shift B — from one platform to capability federation

Represent providers through versioned capability profiles: identity, delegated authority, tool semantics, output guarantees, isolation, evidence export, data residency, cost and failure behavior. Negotiate only capabilities that are actually supported. A provider migration must preserve declared invariants or return `Unsupported`/`Indeterminate`, never silently degrade.

### Shift C — from benchmark scores to claim-bearing measurements

A result is a claim over a named system, dataset, task universe, environment, budget, grader and version. Preserve refused, abstained, unsupported, timed-out, tool-failed and ungraded tasks. Separate:
- task truth (did the task succeed?);
- control truth (was the action permitted and within limits?);
- effect truth (what changed?);
- evidence truth (is the support complete and independently reconstructable?);
- measurement truth (is the grader/calibration itself qualified?).

No weighted average can offset a critical unauthorized action, stale-evidence false pass, physical safety violation or SLA breach.

### Shift D — from simulation pass to model discrepancy management

Every digital twin must carry assumptions, applicability domain, calibration evidence, uncertainty, discrepancy and the evidence tier of each input. The system must detect when reality falls outside the model's supported envelope. Symtropy may compare scenarios; it must not transform a synthetic/simulation pass into physical qualification.

### Shift E — from local optimization to whole-system service

Optimize useful delivered service, not proxies. For compute, that means completed tasks at the required quality/QoS, not just GPU watts. For robotics, it means safe completed work and recovery, not just manipulation success. For engineering change, it means correct impact/revalidation decisions without either laundering stale evidence or forcing every item through full requalification.

### Shift F — from confidence to calibrated abstention

Model confidence, Phi, reputation, committee size and self-reported certainty are not authority. Use explicit unknown/insufficient states, calibrated intervals where appropriate, and an escalation path whose cost is measured. The system should be rewarded for detecting when its evidence is insufficient.

## 3. Benchmark portfolio and current comparison anchors

| Workload | External / industry anchors | Luminous comparison surface |
|---|---|---|
| Agentic software engineering | SWE-bench Pro / Pro V2, Terminal-Bench, OSWorld-Verified / OSWorld 2.x, CRBugBench, Rust-specific repository tasks. Use each benchmark's official denominator and exact revision. | CEB #10419, #10637; repository twin and independent task/evidence contracts. |
| Agent security and authority | NIST AI Agent Standards Initiative; OWASP agent security work; AgentDojo / InjecAgent-style indirect prompt-injection tasks; provider policy/evaluation capabilities (AWS, Microsoft, NVIDIA). | XSTACK-AGENT-RESILIENCE #10771; one-shot authorization and unknown-outcome reconciliation #11540; qualified judge/evidence oracle #11772. |
| Engineering lifecycle | Commercial PLM/MBSE/change-control workflows; NASA systems-engineering change-impact guidance; NIST digital-thread work. | ENG-DESIGN-CHANGE-QUAL #10543 and its fixture/adjudication work #10546 / #10581. |
| Datacentre compute/energy | MLPerf Power for workload efficiency; Nature Energy 2025 field reference (25% power reduction for 3 hours on a 256-GPU cluster while preserving declared QoS); 2026 Alibaba trace-based task flexibility result (up to 22% in its studied scenario). These are different evidence classes and must not be conflated. | Symtropy #3435 and #3776; compare useful-work equivalence, grid-facing response, whole-facility energy, rebound, storage/cooling constraints and compute opportunity cost. |
| Robotics / physical AI | LIBERO, CALVIN, RoboCasa, RoboTwin, ManiSkill and relevant real-robot task suites. Audit task shortcuts, statistical power, overfitting and data-source dependence before accepting a leaderboard claim. | Sol-Atlas spatial/geometry evidence + Symtropy simulation and fault/recovery qualification; physical results only after instrumented trials. |

**Benchmark integrity warning:** a June 2026 manipulation-benchmark audit identified shortcut solvability, weak statistical significance, overfitting and data-source dependence in commonly used LIBERO/CALVIN evaluations. Do not optimize for a single benchmark score or assume simulation task success transfers to the real world. Reference: https://arxiv.org/abs/2606.04233.

### Baselines to run

For each workload profile, register eligible arms only:

1. Qualified human/expert workflow.
2. Conventional deterministic workflow or established engineering toolchain.
3. Vendor-native workflow (only where that product is actually relevant and available).
4. Best-of-breed open system with pinned model, harness and tools.
5. Luminous component ablations.
6. Full Luminous composition.
7. Same-model matched-harness comparison where possible to isolate architecture from foundation-model capability.

Run two comparisons separately:
- **Causal ablation:** same model, task, tools, budget and environment; change one Luminous component.
- **Best available product:** each vendor uses its strongest documented supported configuration; record every difference in models, tools, hardware, budgets and access.

## 4. Metrics and target-setting

Report a vector, not one grand score:

- task correctness at the official metric definition;
- eligible/attempted/completed/graded/refused/abstained/unsupported/timeout/tool-failure accounting;
- critical constraint violations and unauthorized semantic effects;
- authority, delegation, revocation and scope correctness;
- stale-evidence acceptance and evidence-applicability errors;
- independent reconstruction completeness and receipt/evidence integrity;
- recovery after crash, timeout and ambiguous outcome; duplicate effects;
- uncertainty calibration, selective risk and out-of-domain abstention;
- p50/p95/p99 latency where sample sizes justify;
- cost per independently verified successful task, human minutes and rework;
- CPU/GPU time, energy per useful work, water and facility resources where relevant;
- adapter effort, portability and semantic/evidence loss across vendor changes.

### Proposed targets — hypotheses only

- First reproduce the external/owner baseline on the same eligible tasks and denominator.
- In deterministic adversarial controls: zero unauthorized effects; publish the suite size and a confidence bound rather than claiming zero real-world risk.
- On engineering change controls: zero critical-impact misses and zero stale-evidence false-current decisions in the frozen pilot corpus.
- Require 100% task accounting and valid artifact/evaluator bindings before aggregation.
- Establish non-inferiority on task correctness and every hard constraint before claiming efficiency gains.
- After non-inferiority, target at least 10% lower cost per verified successful task **or** 10% lower human review time on a preregistered profile, with no material regression in energy, tail latency, critical recall, abstention or safety. These are pilot hypotheses, not SOTA results.
- Provider/runtime migration must preserve required authority/evidence semantics or visibly fail closed.

All numeric targets require power analysis, paired tasks, cluster-aware uncertainty, independent adjudication and held-out data. Do not convert a synthetic pass into a field result.

## 5. Recommended execution order

### P0 — runner-independent (now)

- Maintain the dated vendor/product capability register.
- Freeze eligible profile-to-vendor applicability matrix.
- Reuse current run receipts, judge qualification, XSTACK and CEB contracts.
- Define the thin adapter profile, hard-gate metric vector and claim labels.
- Audit licenses, API terms, data residency, pricing and what can actually be tested.
- Prepare the first blinded engineering-change pilot and its sample-size plan.

### P1 — first empirical pilot

Start with engineering change impact, because Luminous already has a 24-case synthetic-first corpus and adjudication protocol. Use a conventional rules/graph baseline, expert reviewers, a strong vendor-native coding/agent workflow where it can access the same permitted inputs, and Luminous ablations. Measure critical-impact recall, stale-evidence false-current, unnecessary requalification, reviewer time and cost.

### P2 — cross-stack adversarial composition

Use #10771 to test whether the integrated composition prevents unauthorized effects and independently reconstructs why an effect was allowed, denied or left indeterminate. Keep all attack/effect fixtures synthetic and isolated.

### P3 — physical operations and datacentre

Use Symtropy #3776 and Sol-Atlas owners. Match hardware, workload, QoS, site boundary and energy measurement. Only advance from trace replay to simulation, HIL, lab and field claims when that evidence exists.

### P4 — external validation

Seek an independent evaluator and one design partner with a bounded engineering workflow. Publish negative results and limitations. Do not make "better than NVIDIA/Google/Microsoft/Amazon" claims from internal-only tests.

## 6. Decision

**Build the vendor-neutral evidence and execution layer; integrate with incumbent stacks first.** Compete directly only at the specific functions where matched, independently verifiable evidence demonstrates better outcomes.

The initial wedge is engineering change assurance and verified agent execution. It uses existing public research and software, has a plausible narrow buyer, and creates a path toward datacentre, robotics and aerospace workflows without pretending to own their physical infrastructure.

## Sources

- NVIDIA NeMo Agent Toolkit migration guide: https://docs.nvidia.com/nemo/agent-toolkit/latest/resources/migration-guide.html
- Microsoft Foundry, Build 2026 agent observability/evaluation: https://devblogs.microsoft.com/foundry/build-2026-from-observability-to-roi-for-ai-agents-on-any-framework/
- AWS AgentCore Evaluations GA (2026-03-31): https://aws.amazon.com/about-aws/whats-new/2026/03/agentcore-evaluations-generally-available/
- AWS AgentCore policy Guardrails (2026-06-17): https://aws.amazon.com/about-aws/whats-new/2026/06/amazon-bedrock-agentcore-policy-guardrails-generally-available/
- Google DeepMind Gemini Robotics 2.5 (2026-11-10): https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/
- NIST AI Agent Standards Initiative (2026-02-17): https://www.nist.gov/news-events/news/2026/02/announcing-ai-agent-standards-initiative-interoperable-and-secure
- Broadcom FY2026 Q3 results (2026-09-02): https://investors.broadcom.com/news-releases/news-release-details/broadcom-inc-announces-third-quarter-fiscal-year-2026-financial
- SpaceX Q3 2026 SEC-filed results: https://www.sec.gov/Archives/edgar/data/1181412/000162828026066482/spaceexplorationtechnologi.htm
- Robotics benchmark audit (2026-06-02): https://arxiv.org/abs/2606.04233
- Symtropy AI datacentre flexibility baseline programme: https://github.com/Luminous-Dynamics/symtropy/issues/3776
- Symthaea engineering-change qualification: https://github.com/Luminous-Dynamics/symthaea/issues/10543
- Symthaea cross-stack agent resilience: https://github.com/Luminous-Dynamics/symthaea/issues/10771


## 7. Benchmark frontier refresh — 2026-11-13

The benchmark portfolio should be treated as a versioned, adversarially audited measurement instrument. Do not optimize for a leaderboard row without qualifying the benchmark revision, harness, oracle, task validity, task split, confidence interval, budget and execution environment.

### Software engineering

**SWE-bench Pro V2** (arXiv:2610.24206, 2026-10-16) is a material benchmark-maintenance development. Its authors report that agentic audits, expert adjudication and blind re-implementation repaired or retired 480 of the original 731 tasks, with 23 professional engineers spending 404 hours. The hardened protocol restricts network access to the model endpoint and re-grades patches in clean containers. The paper reports up to 99.4% resolution on 642 V2 tasks, while a 51-task Hard split separates systems from 52.9% to 98.0%. These are paper-reported results, not an independent replication by Luminous.

**Implication:** the large V2 public set can be useful for replication but may be close to saturation for frontier agents. The Hard split, fresh tasks, unseen repositories, security-sensitive changes and engineering lifecycle tasks are more discriminating. A benchmark maintenance and anti-leakage process is a first-class capability, not clerical work.

### Terminal and computer-use agents

The official Terminal-Bench site now shows a 4.1 board. Its displayed leading configurations include Opus 5.5 + Claude Code at 65.8% ± 3.1%, Sonnet 5.5 + Claude Code at 63.9% ± 3.0%, GPT-6.1 Sol + Codex at 60.9% ± 3.0%, and GPT-6 Astra + Codex at 60.3% ± 2.8%. These are harness-plus-model results; they are not model-only results.

Terminal-Bench 4.0's release notes explain that the maintainers recalibrated task resources, fixed tasks and removed saturated tasks. A separate Epoch AI review of 4.0.0 labeled that revision flawed and noted scoring defects in 30 of 66 tasks. Do not automatically carry that 4.0 audit finding onto 4.1; re-audit the current revision independently.

**Implication:** publish paired task results, uncertainty intervals, task-level cost/time/token/compute use and benchmark revision hashes. Do not report only mean resolution rate.

### Robotics benchmark validity

Jiang et al., *What Are We Actually Benchmarking in Robot Manipulation?* (arXiv:2606.04233, 2026-06-02; CoRL 2026 oral listing) diagnose shortcut solvability, weak statistical significance, creeping overfitting and data-source dependence across common manipulation benchmarks. The paper reports that LIBERO and CALVIN fail multiple diagnostics, while RoboCasa and RoboTwin 2.0 fail fewer under the authors' analysis.

**Implication:** each Luminous robotics evaluation should include benchmark-shortcut probes, randomized within-distribution perturbations, held-out data sources, adequately powered repeated trials and real-hardware evidence tiers. Simulation success and physical reliability remain separate outcomes.

### Datacentre operations

The strongest public field reference remains the Nature Energy field demonstration: a 256-GPU cluster reduced power by 25% for three hours during peak demand while meeting its declared AI QoS, without hardware modification or energy storage. This is a specific demonstrated operating envelope, not a universal data-centre flexibility guarantee.

**Implication:** Symtropy #3435 / #3776 should benchmark useful completed work, whole-facility electricity per work unit, QoS tails, rebound, storage/cooling state and grid delivery—not just GPU power reduction.

### Manufacturing and engineering traceability

NIST finalized IR 8536 on 2026-09-09. It describes an industry-neutral manufacturing traceability meta-framework for linking, querying and verifying provenance across ecosystems, with cryptographic links and selective disclosure. The published abstract emphasizes retaining existing industry standards rather than replacing them. W3C Verifiable Credentials Data Model 2.0 and Data Integrity 1.0 became Recommendations on 2025-05-15. IETF RFC 9943 (June 2026) defines a proposed-standard architecture for trustworthy and transparent digital supply chains; related SCITT API and CCF-profile documents remain in the standards pipeline and must not be presented as finalized RFCs.

**Implication:** Mycelix should prove interoperability with existing manufacturing event and credential standards. It should not invent another product-passport ontology. The next work item is a standards mapping and semantic round-trip/adversarial conformance benchmark for Mycelix SupplyChain, composing NIST IR 8536, GS1 EPCIS 2.0/2.0.1, W3C VC 2.0, and IETF SCITT/RFC 9943.

### Updated external references

- SWE-Bench Pro V2 paper: https://arxiv.org/abs/2610.24206
- Terminal-Bench 4.1 leaderboard and resource/versioning notes: https://www.tbench.ai/
- Terminal-Bench 4.0 release notes: https://www.tbench.ai/news/terminal-bench-4-0
- Epoch AI review of Terminal-Bench 4.0.0: https://epoch.ai/benchmarks/terminal-bench-4/review
- Robotics manipulation benchmark audit: https://arxiv.org/abs/2606.04233
- Nature Energy datacentre field demonstration: https://doi.org/10.1038/s41560-025-01927-1
- NIST IR 8536: https://csrc.nist.gov/pubs/ir/8536/final
- GS1 EPCIS 2.0.1 normative artefacts: https://ref.gs1.org/standards/epcis/artefacts
- W3C VC 2.0 Recommendations: https://www.w3.org/news/2025/the-verifiable-credentials-2-0-family-of-specifications-is-now-a-w3c-recommendation/
- IETF RFC 9943: https://www.rfc-editor.org/info/rfc9943/


## 8. Benchmark admission and validity rules — 2026-11-13

A leaderboard value is admissible only with its exact benchmark revision, task universe, system (model + harness + tools), budgets, grader version and evidence class. The benchmark itself must be audited before its score is used as a proxy for capability.

### Current anchors (not a single shared leaderboard)

- **SWE-bench Pro V2:** the 2026-10-16 paper reports up to 99.4% on 642 V2 tasks and a smaller 51-task Hard split spanning 52.9%–98.0%. The paper describes 480 of 731 tasks repaired/retired after agentic audit, expert review and blind re-implementation. This is a paper-reported result and a benchmark revision, not an independent Luminous run. Use exact revision and Hard/general split labels.
- **Terminal-Bench 4.1:** the official leaderboard snapshot accessed 2026-11-13 lists Opus 5.5 + Claude Code at 65.8% ± 3.1%, Sonnet 5.5 + Claude Code at 63.9% ± 3.0%, GPT-6.1 Sol + Codex at 60.9% ± 3.0%, and GPT-6 Astra + Codex at 60.3% ± 2.8%. These are complete system configurations, not model-only scores.
- **Terminal-Bench 4.0.0 review:** Epoch AI's independent review labelled that specific revision flawed and recorded noted scoring defects in 30/66 tasks. Terminal-Bench 4.0 release notes describe resource recalibration, task fixes and saturation removal; the current 4.1 revision must be reviewed independently. Do not automatically transfer the 4.0.0 defect count to 4.1.
- **Robot manipulation:** Jiang et al. (2026-06-02) report shortcut-solvability, weak statistical significance, overfitting and data-source dependence in common manipulation benchmark practice; their audit identifies specific weaknesses in LIBERO/CALVIN and fewer diagnosed issues in RoboCasa/RoboTwin 2.0. Treat this as a reason to run validity diagnostics and held-out perturbations, not as an automatic invalidation of every later benchmark version.
- **Datacentre flexibility:** Nature Energy's field demonstration (DOI 10.1038/s41560-025-01927-1) reports 25% reduction for 3 hours on a 256-GPU cluster while meeting its declared AI QoS. This is a field anchor with a specific workload, facility and measurement boundary, not a universal target.
- **Manufacturing traceability:** NIST IR 8536 is final (2026-09-09); GS1 EPCIS/CBV 2.0.1 provides current event interoperability artefacts; W3C VC DM 2.0 and Data Integrity 1.0 are Recommendations; IETF RFC 9943 is a Proposed Standard. These provide conformance targets, not performance leaderboard scores.

### Version semantics

Use the existing CEB benchmark lifecycle and receipt owners. Adopt the following operational distinction when planning comparisons:

- **Rerun:** task universe, instructions, task data, environment, tool access, budgets or agent execution semantics change.
- **Regrade:** oracle/verifier/judge semantics are corrected while the exact candidate artifacts and execution evidence remain reusable.
- **Reuse:** only metadata, documentation or non-semantic presentation changes, with unchanged task, environment, grader and run evidence.

This mirrors the Terminal-Bench maintainers' published continuous-benchmark/versioning approach. Exact CEB schema/receipt owners remain authoritative for implementation; this note does not create another receipt or benchmark lifecycle system.

### Admission gate before a benchmark can support an industry claim

1. Pin benchmark revision, task IDs, task source, licenses and eligible denominator.
2. Audit task clarity, oracle correctness, reference-patch validity, hidden-data custody and contamination.
3. Test shortcut solvability and negative controls.
4. Record model, harness, agent tools, system prompts, context and budgets separately.
5. Use repeated runs and cluster-aware confidence intervals; report uncertainty and cost.
6. Reproduce the published baseline in the same configuration before asserting an improvement.
7. Keep synthetic, replay, simulator, HIL, laboratory and field evidence classes separate.
8. Retire or down-rank saturated/invalid tasks; do not silently combine benchmark versions.

The machine-readable evidence register in `INDUSTRY_BENCHMARK_VALIDITY_REGISTER_2026-11-13.json` captures the sources and claim limitations for these anchors.
