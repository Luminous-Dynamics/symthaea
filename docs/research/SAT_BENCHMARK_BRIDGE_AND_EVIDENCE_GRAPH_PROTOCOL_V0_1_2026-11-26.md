# SAT Benchmark Bridge and Evidence-Graph Qualification Protocol v0.1

**Date:** 2026-11-26  
**Status:** research protocol and contract proposal; adapter execution pending.  
**Purpose:** add independently verifiable system-level evidence to existing agent-security, AI-control, auditor and robotics evaluations without changing their native task protocols or score definitions.

## 1. Executive decision

SAT should not become another closed task suite or a single safety leaderboard. It should become a **qualification envelope around native evaluations**.

A benchmark owner remains authoritative for its tasks, environment, scorer and native results. SAT adds a second, separately reported layer that asks:

1. Was the exact native experiment frozen and reproduced?
2. What authority and disclosure boundaries applied?
3. Which effects were actually observed at their sinks?
4. Can each material claim be reconstructed from independently held evidence?
5. Which claims remain valid after system, tool, policy, source or trust changes?
6. Can an evaluator challenge or falsify the evidence without trusting the subject?

The architecture has three distinct records:

- **Native run receipt:** the benchmark's own result and accounting record. Reuse the repository's canonical benchmark receipt owner (#10314); do not create a competing general run-receipt system.
- **SAT augmentation record:** binds the native run to SAT observations, claim/evidence links, sink observations, threat model, independence evaluation and limitations.
- **Qualification report:** applies SAT's non-compensatory gates and returns bounded dispositions. It must not overwrite or relabel the benchmark's native metric.

A benchmark score, an authenticated statement, a transparency receipt, an observed external outcome and a qualification decision are different things. No one is a substitute for the others.

## 2. Paradigm shift: benchmark scores become evidence-bearing experiments

Most benchmark reports summarize task outcomes. SAT should additionally preserve the chain from a claim to the experiment and evidence that supports or refutes it.

Example:

- Native claim: the agent completed the requested banking task.
- Authority claim: the user authorized this exact payment, sink, recipient, purpose and amount.
- Disclosure claim: the payload and labels were permitted to leave the source boundary.
- Effect claim: the payment sink applied the exact authorized transfer.
- Continuity claim: the evidence remains applicable to the exact model, prompt, connector, policy, key and runner revision.
- Qualification claim: the required claims have sufficient, authentic, sufficiently independent evidence under the frozen policy.

These are separate claim nodes. A successful benchmark task cannot compensate for a critical unauthorized transfer. A valid receipt cannot compensate for a false statement. A truthful source statement cannot establish independent observation if the source and evaluator share a common administrator.

## 3. Evidence graph model

Represent qualification as a typed directed acyclic graph (DAG), not a weighted scalar and not a bag of untyped links.

### Node types

| Node | Required semantics |
|---|---|
| Claim | Falsifiable proposition, subject, scope, time interval and consequence if false |
| Evidence artifact | Content digest, custody/source reference, creation time, classification and access limitations |
| Observation | Source, observed subject/effect, projection version, attempt identity, sequence/version and uncertainty |
| Verifier | Exact verifier build, version, policy and inputs; result and limitations |
| Component | Model, prompt, policy, connector, tool, runner, sandbox, sink, grader, trust root or witness |
| Decision | Bounded disposition, applicable hard gates, evaluator identity and report digest |

### Edge types

- `supports`: evidence supports a claim under a declared inference rule.
- `refutes`: evidence contradicts a claim.
- `derived_from`: a derived artifact has explicit parent artifacts and transformation identity.
- `verified_by`: a verifier checked an artifact or claim.
- `depends_on`: a claim relies on another claim or component.
- `observes`: an observation concerns an exact attempt, component or sink.
- `invalidated_by`: a change or incident invalidates a claim or evidence item.
- `supersedes`: a new artifact/report replaces a prior one without deleting history.

No implicit support edges are permitted. A digest establishes byte identity, not truth, authority, freshness, independence or semantic correctness.

### Evidence-graph root commitment

Define the canonical `graph_payload` as exactly:
`{schema_version, augmentation_id, native_benchmark, subject, run_context, observations, claims, claim_edges}`.
Before encoding, sort observations by `observation_id`, claims by `claim_id`, and claim edges by the tuple `(from_claim_id, to_claim_id, relation, rule_ref)` using UTF-16 code-unit ordering. Sort and require uniqueness for each claim's `evidence_refs`, `depends_on` and `invalidation_triggers` arrays. Preserve all other array order only where the schema declares it semantically meaningful; otherwise define and version a deterministic ordering in the adapter profile.

The commitment is `SHA-256(UTF8("SAT-EVIDENCE-GRAPH-v0.1\\n") || canonical_json(graph_payload))`, encoded as lowercase `sha256:` hex. `canonical_json` uses the restricted SAT v0.1 JSON profile: reject duplicate object keys, non-integer JSON numbers and integers outside the safe-integer range; sort object keys by UTF-16 code units; retain array order after the graph-specific normalization above; do not Unicode-normalize strings. This is a versioned restricted canonicalization profile, not a claim of full RFC 8785 conformance. The root excludes the result/report and SAT metrics so that the evaluator can bind the graph before rendering or grading it.


For claim edges, `from_claim_id` and `to_claim_id` have fixed direction: `depends_on` means the source/from claim depends on the target/to premise; `supports` or `refutes` means the source/from claim is the explicit argument that supports or refutes the target/to claim; `supersedes` points from the new claim to the historical claim; `invalidated_by` points from the invalidating claim/change record to the affected claim. Every `evidence_refs` entry must resolve to exactly one observation ID in the same augmentation record. Evidence references are not prose labels. A semantic validator must reject dangling references, duplicate IDs, unknown edge endpoints and cycles in the dependency subgraph.

### Claim states

Use orthogonal fields rather than a single `verified` boolean:

- `supported`: declared evidence requirements were satisfied under the frozen rule.
- `refuted`: evidence establishes a counterexample or violation.
- `inconclusive`: evidence is missing, unavailable, contradictory or insufficient.
- `stale`: a required premise or validity interval no longer holds.
- `invalid_evidence`: custody, identity, format or verifier integrity is broken.
- `withdrawn`: the issuer/evaluator explicitly withdraws the claim or source.

A claim may be supported at time t for a bounded scope and stale at t+1. A supported claim also requires admissible evidence under its declared trust policy; an observation with unknown authenticity cannot silently support a critical claim. Historical reports remain immutable; new reports supersede them.

### Provenance closure

Every public qualification claim must close over:

1. subject and exact run identity;
2. source native receipt and native metric definition;
3. frozen manifest and component digests;
4. relevant event, attempt and external-effect identifiers;
5. evidence artifacts and transformation lineage;
6. verifier build, policy, trust roots and prior checkpoint pins;
7. independence dependencies and common-cause analysis;
8. limitations, unavailable evidence and human adjudications.

If any mandatory dependency is missing, the claim is inconclusive rather than supported.

## 4. Continuous qualification and change impact

A qualification result is a time-bounded evidence state, not a permanent certificate.

Changes should invalidate only claims that depend on the changed component, but only where the dependency graph is complete enough to justify selective reuse.

| Change | Default disposition |
|---|---|
| Model weights, serving backend or model identity | Requalify model-behaviour and dependent claims |
| System/developer prompt, memory, retrieval corpus or policy | Requalify claims whose behavior or authority premises depend on them |
| Connector, tool schema, permissions, API behavior or sink projection | Requalify authority, disclosure, effect and recovery claims |
| Trust root, signing key, revocation snapshot or transparency policy | Reverify authenticity and continuity claims; preserve substantive claims only if their evidence dependencies remain valid |
| Runner, sandbox, isolation, network or host policy | Requalify execution isolation and any claim dependent on containment |
| Grader or verifier code | Regrade retained immutable artifacts under a new verifier version; preserve old report |
| Task source/environment or attacker budget | New native run; no reuse as a matched comparison unless the benchmark owner explicitly permits it |
| Report formatting only | Re-render retained results; do not rerun or change claim semantics |

### Selective revalidation metrics

Measure:
- **Invalidation recall:** fraction of claims that truly require revalidation and are correctly flagged.
- **Invalidation precision:** fraction of flagged claims that genuinely depend on the changed component.
- **Stale-claim false carry rate:** critical rate at which invalidated claims remain supported.
- **Unnecessary revalidation rate:** evidence discarded and rerun despite unchanged, still-valid premises.
- **Revalidation cost:** time, compute, money, human review and external-sink operations required to restore the evidence state.
- **Time-to-detect drift:** time from relevant change or incident to stale/inconclusive disposition.

The target is not to minimize revalidation at all costs. It is to minimize unnecessary work subject to a hard ceiling on stale evidence being treated as valid.

## 5. Evidence debt and assurance fragility

Do not compress unresolved assurance into a single score. Track an **evidence-debt vector**:

- count of stale, inconclusive, unavailable and invalid claims by consequence class;
- age of each unresolved claim and time remaining to its declared expiry;
- number of downstream claims blocked by each unresolved dependency;
- expected revalidation effort and required external/human evidence;
- whether the debt is caused by missing evidence, failed authenticity, source disagreement, change drift or evaluator uncertainty.

Critical claims remain hard gates regardless of the aggregate size of the debt. A dashboard may show trends, but it must not let many low-risk supported claims compensate for one unresolved critical effect claim.

Track **assurance fragility** separately: the disclosed minimum common-cause cut set that can invalidate a claim's supporting evidence, and the fraction of its provenance that depends on one operator, one service, one time source or one custody domain. If graph completeness is unknown, fragility is unbounded/unknown rather than zero.

A useful counterfactual test is to remove or mutate one evidence premise and recompute the affected claim closure. Claims that remain supported despite removal of a mandatory premise reveal a verifier defect; claims that all collapse after removal of an irrelevant artifact reveal over-coupling. These are graph-semantic tests, not model-behaviour scores.

## 6. Common-cause independence as a graph property

Two signatures are not two independent witnesses merely because the key IDs differ.

The dependency graph should include, when applicable:
- legal/operator control and beneficial ownership;
- identity provider, key custodian, signing service and administrator;
- cloud tenant, hosting provider, region and control plane;
- source application, database and observability pipeline;
- network path and time source;
- physical sensor, calibration chain and controller;
- transparency service, witness governance and log replication;
- evaluator, grader and subject/vendor conflicts.

For a declared threshold/quorum, calculate the smallest disclosed common-cause cut set that can defeat the quorum. If dependencies are unknown, independence remains `unverified`. Do not assume missing edges mean independent operators.

Report the graph and cut sets, not just the witness count. An independence policy must define which dependencies are disqualifying, correlated, tolerated or unknown.

## 7. Native benchmark adapter rules

### 6.1 Preserve native semantics

An adapter MUST preserve:
- upstream benchmark name, version, task IDs and environment version;
- native attacker/defender protocol and budget;
- model, prompt, tool, harness and grader versions;
- official task denominator, retries, refusals, abstentions, timeouts and exclusions;
- official scoring code and metric labels;
- raw native output references and original run receipt.

SAT MUST NOT:
- replace an upstream task with a Luminous-authored lookalike and call it a reproduction;
- change the grader while preserving the upstream leaderboard label;
- infer a native pass from a SAT effect check, or vice versa;
- report only answered/completed episodes while omitting refused, skipped, failed or ungraded instances;
- aggregate distinct benchmark scores into a single universal SOTA score.

### 6.2 Adapter envelope

The SAT augmentation record references the native receipt by immutable ID and digest. Its `claims` array is the claim-node set, `observations` are evidence/observation nodes, and `claim_edges` are typed graph links. The schema checks local shape; a semantic graph validator must enforce global uniqueness, endpoint resolution, evidence-reference closure, cycle detection, pinned graph-root recomputation and policy-specific claim/evidence sufficiency. It adds:
- frozen SAT evaluation policy and threat model;
- exact subject component digests and execution environment;
- SAT trace and checkpoint references;
- authority and disclosure observations;
- attempt-bound effect receipts, where the native environment has external effects;
- evidence graph root and independent evaluator identity;
- SAT disposition and claim limitations.

For benchmarks with no real external effect (for example, pure hidden-behavior probes), effect evidence can be `not_applicable` only under an explicit adapter policy. Do not invent an external sink to satisfy the schema.

### 6.3 Adapter families

| Benchmark | Native contract to preserve | SAT extension |
|---|---|---|
| AgentDojo | 97 user tasks, 629 security test cases, native utility-under-attack and targeted ASR definitions | Independently record unauthorized tool effects, disclosure boundaries, exact task/attack identity and evidence custody |
| AgentHarm | 110 malicious tasks, 440 augmented variants, 11 harm categories and native harmful-task scoring | Separate harmful request compliance, harmful tool attempt, sink effect, and independently observed impact |
| DelegateGuard | Purpose-bound capability/delegation semantics, attenuable grants and revocation protocol | Independently verify effect boundary, revocation timing, source truth and evidence continuity; do not claim to invent capability attenuation |
| AuditBench | 56 hidden-behavior target models, 14 behavior categories and its investigation protocol | Claim/evidence graph for auditor findings, blind disconfirmation, benign-target calibration and provenance of judge decisions |
| ControlArena | Inspect AI tasks, settings, micro-protocols and honest/attack modes | Portable assurance envelope around trajectory, monitor intervention, trusted-delegate boundary and evidence/decision continuity |
| LADDER | 145 tasks across three suites, 1–10 subgoals, two embodiments and official success/progress metrics | Separate planning, authorization, actuator command, calibrated physical observation and task completion; report sensor uncertainty and embodiment |
| Inspect AI | Native task, solver, scorer, log and evaluation configuration | Store canonical references to native logs and separately produce SAT claims/evidence edges |

### 6.4 Why these are complementary

- AgentDojo is explicitly a dynamic tool-use environment for indirect prompt injection, not a complete authority/effect qualification protocol.
- AgentHarm measures harmful agent capabilities and malicious-task completion, not the full evidence chain for a particular external side effect.
- DelegateGuard moves authority enforcement outside the LLM and studies attenuable, purpose-bound delegation; SAT should compose with it, not claim that mechanism as novel.
- AuditBench studies hidden-behavior auditing and exposes a tool-to-agent gap: tools that surface evidence do not guarantee the investigator uses it correctly.
- ControlArena already provides a compositional AI-control framework on Inspect AI; SAT should be an assurance adapter, not another competing micro-protocol framework.
- LADDER shows why high-level orchestration and physical execution must be scored separately. SAT can add actuator authority, effect observation and physical evidence without changing LADDER's official success metric.

## 8. SAT research benchmark: EVIDENCE-CONTINUITY

Create a new cross-benchmark benchmark family that tests qualification itself. It uses synthetic, deterministic evidence graphs first; native benchmark adapters follow.

### Track A — Evidence graph correctness

- missing supporting artifact;
- wrong artifact digest;
- wrong run/attempt binding;
- derived artifact with missing parent;
- cycle in a claimed DAG;
- verifier/policy substitution;
- contradictory claims;
- duplicated or re-ordered evidence;
- refuted claim mislabeled supported.

### Track B — Change-impact precision

- model/prompt/tool/connector/grader/trust-root/sink projection changes;
- benign metadata-only changes;
- unrelated component changes;
- transitive dependency changes;
- unknown or incomplete dependency edges.

### Track C — Witness/common-cause analysis

- distinct keys, same custodian;
- distinct organizations, common cloud control plane;
- same source data and mirrored telemetry;
- shared time source;
- one witness equivocates across views;
- missing dependency disclosures.

### Track D — Adversarial evaluator integrity

- attacker changes subject and report together;
- evaluator uses stale native receipt;
- hidden target labels leak into the auditor;
- grader changes after seeing candidate outputs;
- benign targets are falsely flagged;
- auditor fabricates unsupported findings;
- candidate withholds difficult tasks to inflate conditional score.

### Track E — Evidence survivability

- original service unavailable;
- evidence bundle offline;
- missing artifact;
- expired/revoked key;
- unavailable transparency proof;
- checkpoint fork;
- verifier binary or policy version missing;
- independent replay on a second machine.

### Initial public corpus

Start with 50 public deterministic cases across the five tracks (10 each). Keep a separately generated, sealed holdout, never derive it from the public fixtures, and do not publish its expected labels. Report case-family denominators and every disposition. These fixtures test the qualification evaluator, not agent intelligence.

## 9. Metrics and initial research targets

All figures below are proposed targets, not current results or universal standards.

| Metric | Initial target | Interpretation |
|---|---:|---|
| Critical stale/invalid claim falsely supported | 0 in finite negative controls | Non-compensatory hard gate |
| Wrong-run/attempt evidence accepted | 0 in finite negative controls | Identity binding |
| Refuted claim reported supported | 0 in finite negative controls | Claim logic |
| Evidence graph cycles accepted | 0 | DAG correctness |
| Missing mandatory provenance silently accepted | 0 | Provenance closure |
| Invalidation recall | >=99% on defined mutations | Avoid stale assurance |
| Invalidation precision | >=95% on declared dependency cases | Avoid needless requalification |
| Independent replayer reproduces dispositions | 100% of public deterministic cases | Portability |
| Benign false-block rate | <=1%, with confidence interval | Avoid utility collapse |
| Native benchmark utility under attack | Report official metric unchanged | Comparability |
| SAT evidence overhead | Measure p50/p95, cost and bytes; set target after pilot | Avoid invented budget |
| Critical witness common-cause undisclosed and treated independent | 0 in finite controls | Independence integrity |
| Human unaided performance after withdrawal | Non-inferior within preregistered 5% margin | Human capability preservation |

Zero observed failures do not establish zero risk. Publish numerator, denominator, confidence interval, exclusions and clustering unit. Do not treat multiple actions inside one episode as independent Bernoulli trials.

## 10. Baseline design

For each native benchmark, compare on exactly the same tasks, attack budget, seeds, candidate build, tools and graders:

- **B0:** native system/control with no added SAT instrumentation (official baseline).
- **B1:** static allowlist or native reference monitor, where applicable.
- **B2:** benchmark-native strongest published control/micro-protocol.
- **B3:** full SAT evidence envelope.
- **B4+:** SAT ablations (remove sink read-back, evaluator pins, source-independence graph, checkpoint continuity or change-impact engine one at a time).

Use paired task/seed comparisons. Keep performance and assurance axes separate. A defense that blocks all tool use cannot win by safety alone. A fast agent that performs unauthorized critical actions cannot win by utility alone.

## 11. Statistical and adjudication requirements

Before confirmatory runs:
- freeze primary estimands, practical effect margins, exclusions and missingness;
- freeze candidate model, prompts, memory, tools, permissions, connector/sink versions, runner, network and budgets;
- use multiple independent seeds and estimate variance from a pilot;
- cluster uncertainty by task family, scenario template, environment and episode;
- report paired effects, confidence intervals and per-family disagreement;
- separate discovery, development, calibration, public test and sealed holdout;
- keep native graders and SAT evaluators separate;
- audit evaluator/LLM-judge agreement against blinded human adjudication;
- publish raw-log and evidence artifact digests, not sensitive payloads;
- preserve refusals, abstentions, timeouts, unavailable evidence and ungraded episodes in accounting.

Use NIST AI 800-2's evaluation-reporting guidance as a baseline for preserving full logs, exact system versions, commit identifiers, metadata and comparison groups. NIST AI RMF / GenAI Profile and OWASP Top 10 for Agentic Applications are governance crosswalks, not certification shortcuts.

## 12. Integration with existing Luminous infrastructure

Reuse authoritative owners:
- **#10314:** canonical benchmark run/accounting receipt; SAT augmentation references it.
- **#10436:** task custody and task identity.
- **#10064:** execution isolation and runner boundary.
- **#10369:** trust roots and trust policy.
- **#10173:** repository/system dependency graph and change-impact facts.
- **#9112:** decision-use admissibility semantics.
- **#10366/#10387:** causal diagnosis, interventions and next-experiment selection.
- **#13602:** independent authority oracle and exact-effect reference semantics.
- **#13636:** SAT qualification contract, native manifest/event/report contract and matched-baseline claim protocol.
- **#13667:** read-only trace/checkpoint verifier.
- **#13715:** external effect receipt and attempt-binding extension.

SAT is the composition and qualification layer. It must not create a competing task registry, repository twin, authority owner, payment state machine, generic causal scheduler, evidence store, trust-root registry or benchmark run receipt.

## 13. Release/evidence levels

For every adapter and corpus:
1. **Specified:** protocol/schema authored.
2. **Contract-checked:** standards-compliant schema validation, examples, unique IDs and negative contract controls.
3. **Locally executed:** exact commit compiled and tests/fixtures executed; logs retained.
4. **Independently replayed:** a second evaluator reconstructs results from immutable artifacts.
5. **Externally reproduced:** independent party reproduces the claimed result under the preregistered protocol.

Do not call a design benchmark a run. Do not call schema validity an implementation pass. Do not claim SOTA from internal fixtures or a smoke test.

## 14. Immediate work order

1. Publish adapter-envelope schema and an explicit synthetic example.
2. Validate all public examples/fixtures with a standards-compliant JSON Schema 2020-12 validator.
3. Add 50 public evidence-continuity cases, then independent negative controls.
4. Implement a deterministic, offline Rust graph verifier with cycle detection, provenance closure, evaluator-owned root pins and change-impact dispositions.
5. Add a read-only bridge to #10314 native benchmark receipts.
6. Reproduce AgentDojo and AgentHarm native task/score protocols in their original environments; add SAT augmentation separately.
7. Add ControlArena/Inspect and AuditBench adapters; keep hidden labels in evaluator custody.
8. Add LADDER physical observation adapter only after sensor calibration, uncertainty and embodiment boundaries are declared.
9. Run adaptive attacks and blind independent adjudication.
10. Publish a first external replay before making comparative superiority claims.

## 15. Research references and exact scope

- AgentDojo, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html
- AgentDojo official project: https://agentdojo.spylab.ai/
- AgentHarm, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/hash/c493d23af93118975cdbc32cbe7323f5-Abstract-Conference.html
- DelegateGuard, arXiv preprint submitted 12 October 2026: https://arxiv.org/abs/2610.16005 (preprint; not described as peer-reviewed)
- AuditBench, Anthropic, 10 March 2026: https://alignment.anthropic.com/2026/auditbench/
- ControlArena documentation: https://control-arena.aisi.org.uk/
- Inspect AI documentation: https://inspect.aisi.org.uk/
- LADDER, arXiv:2610.18053, 2026: https://arxiv.org/abs/2610.18053 and https://ladderrobotics.ai/research
- OWASP Top 10 for Agentic Applications 2026: https://genai.owasp.org/top-10-for-agentic-applications-2026/
- NIST AI 600-1, Generative AI Profile: https://doi.org/10.6028/NIST.AI.600-1
- NIST AI 800-2, Practices for Automated Benchmark Evaluations of Language Models, January 2026: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-2.ipd.pdf
- SCITT RFC 9943: https://www.rfc-editor.org/rfc/rfc9943.html

**No benchmark scores, native benchmark reproductions, external cryptographic proofs, human studies or production qualification are claimed by this protocol.**