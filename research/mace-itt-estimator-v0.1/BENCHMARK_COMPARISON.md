# MACE-CoS benchmark positioning (2026-11-12)

This is a research-positioning snapshot, not an independent reproduction or a claim that MACE currently outperforms any listed system. Published results belong to their authors' methods. Comparisons must match task, model, harness, permissions, environment, judge and scoring protocol.

| Benchmark | Published scope / result | MACE comparison target | Caveat |
|---|---|---|---|
| TraceTrial-Bench | 95 mechanism-focused tasks; 4 harnesses; 9 models; 3,420 Docker runs. 31.1% task-pass runs fail at least one harness criterion; 8.4% pass every criterion across both layers. | Four-cell task × controls outcomes, coverage-qualified evidence, repeated runs, recovery lineage and failure attribution. | Logical run counts are not automatically independent repeats. Preserve the original unit and denominator. |
| SkillsBench v1.1 | Official board: 87 tasks, 8 domains, up to 3 trials/task, 25 model-harness configurations; 24 paired with/without-skill configurations, with 95% CIs. | Randomized ITT skill-availability effect, task-macro aggregation, uncertainty and rescued/harmed/unchanged tasks. | Compare to the official release snapshot, not a third-party leaderboard scrape. |
| SRA-Bench | 5,400 instances, 6 domains, 636 gold skills, 26,262-skill corpus; separates retrieval, incorporation and application. | Keep non-retrieval tasks in ITT; treat retrieval/load/invocation as mechanism diagnostics. | Retrieval is not proof of causal benefit. |
| NVIDIA SkillEvaluator | Static screening, semantic distinctiveness and isolated live with/without-skill evaluation; correctness, discoverability, effectiveness, efficiency and security; pass@k support. | Add judge calibration, independent gold validation, four-cell classification and runtime/effect provenance. | Reported judge provider/model provenance is not proof of judge validity. Docs note no published inter-rater agreement, calibration or human-gold comparison. |
| CoSA-Bench | 288 tasks across recruiting, finance, IT and business. A 5%-capped skill-pool contamination condition raises violations from 35.02% to 58.45%. Its guard reduced violation rates in its tested OpenClaw recruiting condition. | Reproduce the published guard; compare multi-skill composition controls on task utility and unauthorized effects. | Guard figures are condition-specific, not universal rates. |
| CoRedSkill | 150 cases / 50 tasks / 3 attack objectives; main setting reports 98.0% attack success, 95.3% task success, 3.3% audit detection and 90.7% joint attack/task/audit-evasion success. | Joint harmful success plus task utility in the same run; distinguish detection from prevention. | Benchmark attack rates are not real-world attack prevalence. |
| SkillLifeBench | 388 paired source scenarios; 14 models; runtime and autonomous lifecycle tasks across implementation, interface, runtime, composition and version changes. | Identity continuity from resolve through execute; stale/substituted implementation rejection; recovery and independent effect receipts. | Keep recovery success separate from first-attempt success. |

## NIST alignment

NIST's 2026-10-27 agent identity design tenets are initial guidance, not a finalized standard. They cover distinct agent identity, cryptographically linked and attenuating delegation, least/zero-standing privilege, short-lived revocable authorization, signed mandates, policy/reasoning separation, deterministic access decisions, continuous monitoring and independent evidence.

NIST feedback favours evolving existing IAM and identity standards rather than creating a parallel identity stack. MACE should profile and verify established identity primitives and add composition, evidence and conformance semantics above them—not build a competing IAM system.

## MACE's differentiation hypothesis

MACE should not claim novelty for scanning, agent identity, delegation, composition graphs, append-only logs or policy engines in isolation. The differentiating conjunction is:

1. Pre-execution compilation of a typed skill DAG with data lineage, scope, purpose, audience, recipient and capabilities.
2. Portable, attenuating authority mandates bound to the actual agent/runtime, task, action and budget, using existing identity primitives.
3. Independent observation of actual resolver, loader, invocation, sandbox/process and completion paths.
4. Exact joins between execution identity, authorization and authenticated downstream effect receipts.
5. Independent freshness anchoring and reconciliation after unknown outcomes.
6. Causal ITT task utility, task × control clean-success classification and joint harmful-success evaluation with clustered uncertainty.
7. Blinded and calibrated adjudication, validated against deterministic or human gold oracles.

## Proposed qualification targets (not results)

| Track | Research target | Required baseline/evidence |
|---|---|---|
| Outcome classifier | 100% golden-fixture agreement; unknown evidence never becomes clean success | Independent oracle and mutation corpus |
| Skill utility | ITT task-macro lift ≥ +5 pp overall and positive in each reported domain | Official SkillsBench/SRA-Bench semantics and matched configurations |
| Benign utility | Regression ≤ 1.5 pp | Strongest reproduced guard, not only unguarded baseline |
| Distributed attacks | Joint harmful success ≤ 5% on held-out preregistered cases | CoRedSkill + CoSA guard + benign paired controls |
| Composition | 100% agreement on bounded finite-state oracle cases | Exhaustive small compositions and authority schedules |
| Lifecycle | 100% rejection of registered identity mismatches before invocation/effect | SkillLifeBench-like mutation corpus |
| Observer | Zero silent event loss/reorder/duplication in registered fault schedules | Independent OS/process/sandbox trace comparison |
| Judge quality | Pre-registered sensitivity/specificity and calibration thresholds with CIs | Deterministic gold + blinded human double annotation |
| Cost | p50/p95/p99 overhead, memory, storage, tokens and human burden reported | Matched runtime-off/runtime-on baselines |

No weighted score should allow utility to compensate for unauthorized effects or unproven controls. Publish a Pareto frontier over utility, security, evidence completeness, latency and cost.

## Sources

- TraceTrial-Bench: https://arxiv.org/abs/2610.32285 and https://github.com/TraceLite-AI/TraceTrial-Bench
- SkillsBench official leaderboard: https://www.skillsbench.ai/leaderboard
- SRA-Bench: https://aclanthology.org/2026.findings-emnlp.1955/ and https://arxiv.org/abs/2604.24594
- NVIDIA SkillEvaluator: https://docs.nvidia.com/skills/skillevaluator and https://docs.nvidia.com/skills/skillevaluator/reports
- CoSA: https://arxiv.org/abs/2610.35082
- CoRedSkill: https://arxiv.org/abs/2610.20049
- SkillLifeBench: https://arxiv.org/abs/2610.35215
- NIST initial design tenets: https://www.nist.gov/blogs/cybersecurity-insights/design-tenets-agent-identity-management-systems
- NIST NCCoE feedback summary: https://pages.nist.gov/nccoe-ai-identity/summary-of-comments.html

## 2026-11-13 update: repeated-run reliability is two-sided

NVIDIA SkillEvaluator's current Tier 3 documentation supports `pass@k`: with `--n-attempts k`, a case passes when at least one of its k attempts clears the threshold, and per-arm case-level 95% Wilson intervals are reported. This is a useful baseline, but pass@k is an optimistic measure of finding one successful attempt. Princeton's agent-evaluation guidance and the recent *On Randomness in Agentic Evals* study recommend reporting a complementary all-runs-success measure (pass^k / pass∧k) and repeated trajectories rather than treating pass@k as reliability by itself.

MACE's repeated-run report should therefore publish, for task-pass and clean-success separately:

- **pass@k**: at least one of k sampled first-attempt runs succeeds;
- **pass^k**: all k sampled first-attempt runs succeed;
- per-arm rates and the paired ITT contrast;
- task-clustered uncertainty and partial-identification bounds for unknown outcomes;
- the number of tasks with enough preregistered seeds for each k, plus excluded/insufficient task counts.

For n observed preregistered first attempts and c successes, the finite-sample subset estimates are:

\[\mathrm{pass@k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},\qquad
\mathrm{pass^k}=\frac{\binom{c}{k}}{\binom{n}{k}}.\]

This estimates the fraction of size-k subsets of the registered first-attempt runs that satisfy the respective event. It is not a guarantee of future deployment reliability. Retries/recovery attempts stay in the event ledger but do not count as independent seeds or increase k. Tasks with fewer than k preregistered seeds are explicitly marked insufficient, not treated as failures.

Sources:
- NVIDIA SkillEvaluator live evaluation and reports: https://docs.nvidia.com/skills/skillevaluator and https://docs.nvidia.com/skills/skillevaluator/reports
- Princeton, *Emerging Trends in AI Agent Evaluation*: https://sage.cs.princeton.edu/trends/
- Bjarnason et al., *On Randomness in Agentic Evals* (60,000 SWE-Bench-Verified trajectories): https://arxiv.org/abs/2602.07150
- Current official SkillsBench v1.1 release export (87-task roster; current board): https://huggingface.co/datasets/benchflow/skillsbench-leaderboard

This is a measurement-method update only. No external MACE benchmark reproduction or performance improvement is claimed.


## 2026-11-13 update: evaluate composition interactions and effect-evidence joins

The IETF individual Internet-Draft on agent security evaluation profiles (draft-hcf-bpm-agent-security-benchmark-profiles-00, 23 October 2026) adds two important requirements that MACE should adopt as research guidance, not claim as an adopted standard:

1. **Composition interaction effects (§7.3):** assess ordering effects (one layer masks another), state effects (one component changes state affecting later components, such as memory writes), and non-additivity (the composite score is not the sum of component scores). MACE should report all three and avoid assuming independently passing skills compose safely.
2. **Independent evidence comparison (§8.5):** include observed system calls, API calls, file/network operations, identity and permissions at the time of the operation, and compare system audit records against independently collected harness events.

MACE's evidence architecture should build on existing interoperable formats rather than create parallel telemetry, identity or provenance standards:

- **OpenTelemetry GenAI conventions:** correlate agent and tool spans, including failed tool calls; treat the conventions as evolving telemetry semantics, not proof of a complete or truthful observer.
- **in-toto Attestation Framework:** use subjects/statements/envelopes/bundles to bind immutable implementation artifacts and authenticated statements. An attestation does not by itself prove that a live runtime loaded or invoked the subject.
- **SPIFFE Workload API / SPIRE:** obtain portable workload identity material and bind observer/broker/gateway/effect-source roles to configured trust domains. A workload ID alone does not prove administrative independence or process behavior.
- **External witness/transparency service:** anchor bundle heads and freshness independently; a local hash chain or local signed checkpoint does not establish anti-rollback against an actor controlling the complete local store.

This motivates a new MACE verifier track: exact causal joins from authorization receipt → observed invocation and implementation → gateway dispatch → effect-origin receipt → independently witnessed bundle digest. The verifier checks that every receipt refers to the same task, execution, principal, target, action, arguments, capability, policy generation and idempotency key. It does not replace the #9490 effect-authority kernel or #11741 runtime observer.

References (current snapshots; standards status matters):
- IETF agent security evaluation profiles, individual Internet-Draft (work in progress; not an RFC): https://datatracker.ietf.org/doc/html/draft-hcf-bpm-agent-security-benchmark-profiles-00
- OpenTelemetry GenAI semantic conventions (development): https://github.com/open-telemetry/semantic-conventions-genai
- in-toto Attestation Framework v1.2: https://github.com/in-toto/attestation
- SPIFFE Workload API: https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE_Workload_API.md
- NIST agent identity design tenets (27 October 2026): https://www.nist.gov/blogs/cybersecurity-insights/design-tenets-agent-identity-management-systems

No external MACE benchmark reproduction or runtime qualification is claimed by this documentation update.
