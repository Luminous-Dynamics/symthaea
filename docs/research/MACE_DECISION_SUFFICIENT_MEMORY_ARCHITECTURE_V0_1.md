# MACE Decision-Sufficient Memory Architecture v0.1

**Programme:** Broca / Symthaea MACE  
**Status:** Research architecture and qualification protocol; no production implementation or measured improvement is claimed.  
**Core thesis:** Memory should preserve the information and evidence needed to make future decisions correctly—not necessarily preserve the original information as a stored fact.

## 1. The paradigm shift

Most memory systems optimize a chain resembling:

conversation → summary/chunks → index → retrieval → model answer

MACE should qualify a different chain:

raw events → evidence graph → decision capsule → representation choice → query-conditioned evidence retrieval → use-time revalidation → decision or safe information acquisition → reconciled outcome

The objective is not maximum recall. It is **decision sufficiency under changing conditions**.

A stored item is valuable when it changes a future decision for the better, remains applicable at the time of use, and can be justified from admissible evidence. A memory that is retrieved but does not improve the decision is not automatically useful. A memory that once was correct but is stale at consumption is actively harmful.

This architecture composes existing MACE, Mycelix, Symtropy, HDC-LTC, bitemporal and selective-action work. It does not propose a second memory store, evidence authority, timestamp authority, source verifier or evaluator.

## 2. Decision capsule: the unit of reusable memory

A **Decision Capsule** is a typed, versioned candidate representation of a reusable decision-relevant relationship. It is not an authority token and never replaces its raw evidence.

A v0.1 capsule should bind:

| Field group | Required semantics |
|---|---|
| Identity | Capsule ID, schema/version, exact source event/revision IDs and canonical content digest |
| Decision question | Objective, decision family, action alternatives, outcome definition/unit and required horizon |
| Applicability | Preconditions, operating regime, scope, population/target class, excluded conditions and known mechanism boundaries |
| Learned relationship | Action-to-outcome or state-transition relation, direction, estimated effect distribution, uncertainty and measurement-error bounds |
| Causal qualification | Identification method, canonical estimand descriptor/digest, design receipt, assignment overlap, covariate balance, interference controls and delayed-horizon completion |
| Evidence closure | Raw payload/frame/action references, canonical source IDs, independent source families/roots, evidence DAG and exact receipt references |
| Epistemic status | Observed, inferred, hypothesized, disputed, superseded, retracted or unknown; never collapse these to a single similarity/confidence score |
| Temporal state | valid_at, known_at, event time, observation time, source revision, supersedes/retracts graph and current-use validity |
| Authority and rights | Principal, capability, tenant/resource, audience, purpose, policy revision, consent/retention class and erasure obligations |
| Reconstructibility | Durable rule/procedure, mutable state, externally re-observable fact, bounded snapshot, ephemeral context or non-retainable item |
| Target applicability | Target-agent/model fingerprint and calibrated future-use prediction when available |
| Resource profile | Bytes, context tokens, index/codebook/sidecar, verification work, re-observation calls, latency, compute/cost and privacy exposure |

The exact semantic field names should be aligned with existing MACE and Mycelix schemas before implementation. This table is a contract inventory, not a new competing source-of-truth schema.

### 2.1 Preserve proof links, not all raw text in the prompt

A capsule can be compact, but its evidence closure cannot be silently discarded. It should retain stable pointers to original text, screenshots, video frames, spatial observations, tool/action traces and reconciled outcomes. Captions and summaries are useful indexes; they are not replacements for original evidence.

### 2.2 Keep different knowledge kinds separate

Do not represent all memory as generic text or generic facts.

- **State:** what is true in a mutable world at a particular time.
- **Procedure:** how to accomplish a task, including preconditions and failure cases.
- **Causal effect:** what changed after an intervention, under a declared estimand.
- **Preference / intent:** what a person wants, with scope, uncertainty and confirmation status.
- **Observation:** what a source or sensor reported, not necessarily what is true.
- **Hypothesis:** an unconfirmed relationship that must not be promoted to an established rule.

The same string or embedding can express different knowledge kinds; type and provenance must govern interpretation.

## 3. Representation is a decision, not a storage default

For each candidate capsule, choose one of four representations:

1. **Persist knowledge** — preserve a durable rule, procedure, preference or causal relationship that cannot safely be reconstructed from current observation.
2. **Store locator** — preserve a verified way to re-observe mutable state at decision time.
3. **Cache snapshot** — retain a source-verified snapshot only through its verified validity interval.
4. **Discard** — allocate neither durable memory nor a re-observation locator when expected decision value is non-positive or constraints make retention inadmissible.

This is the MACE re-observation factor already prototyped in #12743. The selector must compare alternatives using calibrated future-use probability, source availability, query cost, expected failure consequence, context/storage costs and deadline feasibility.

A locator is not a successful re-observation until the source returns a canonical current-state receipt. A cached freshness flag is not enough. A snapshot expiring at the decision deadline is not usable at that deadline.

### 3.1 Hard constraints are not weighted preferences

Optimize quality only over feasible candidates. Do not let a high utility estimate compensate for a privacy, authorization, tenant, source-integrity, deadline or lifecycle violation.

Keep a vector of budgets:
- persistent bytes and evidence-sidecar bytes;
- context tokens and model input;
- CPU/GPU and latency;
- tool / re-observation calls;
- verification work;
- monetary cost;
- privacy exposure and retention obligations;
- decision deadline and failure-loss ceiling.

A scalarized cost may be useful for ranking already-feasible alternatives, but cannot convert a hard violation into a tradeable penalty. Report the Pareto frontier across resource dimensions.

### 3.2 Preserve procedural knowledge when the world can be re-observed

A current price, machine state or appointment slot can often be re-observed. A learned repair procedure, diagnostic rule or action-effect relationship generally cannot be reconstructed from the current state alone. Benchmarks must include both task types; otherwise the locator or persistence policy wins by construction.

## 4. Query-time decision support is a proof-producing path

Retrieval must not authorize use. The candidate may return event IDs, capsule IDs, ranked candidates, similarity and retrieval diagnostics only.

Before memory can support a decision:

1. Resolve every returned ID to exactly one canonical record.
2. Verify the exact content/revision digest and evidence references.
3. Re-evaluate provenance, independent-root, causal-identification and estimand gates.
4. Re-evaluate objective, regime, population, scope and applicability conditions.
5. Query canonical source lifecycle at the current bitemporal knowledge snapshot.
6. Check age at actual decision consumption, not only age at retrieval.
7. Check authority, consent, audience, purpose and privacy-erasure state.
8. Check deadline and component-wise resource feasibility.
9. Construct a decision-support set with an explicit claim ceiling and evidence trace.
10. Let the separate policy choose use, verify, clarify, abstain or bounded probe.

If any mandatory predicate is unknown, invalid or unavailable, fail closed. Historical as-known replay and current-state use are different queries and must remain distinguishable.

An explanation written by the model is not a receipt. A similarity score is not corroboration. A successful tool invocation is not a reconciled world outcome.

## 5. HDC-LTC: qualify the representation before attributing decision value

The current public HDC candidate index in the MACE workspace is a legitimate initial candidate-index experiment, with a narrower claim boundary than semantic retrieval.

Current implementation characteristics:
- candidate event codes are seeded from event IDs rather than experience content;
- records are pre-bucketed by regime, objective and action;
- retrieval ranks candidates within the selected scope/action bucket;
- the LTC arm evolves the bundled bucket state for a configured elapsed interval using zero input;
- the current API does not encode a decision query into an HDC query vector or replay each event's irregular timestamped input stream.

Therefore current results can qualify deterministic index behavior, candidate-ID recovery, payload size and the effect of the zero-input LTC profile. They do **not** yet establish semantic retrieval, query-conditioned temporal reasoning or HDC-LTC decision superiority.

### 5.1 Next representation tranche

Before a semantic HDC claim:
- Encode a typed, versioned ExperienceView using role/value binding over decision-relevant content and conditions, while retaining exact evidence references outside the lossy vector.
- Encode the actual query: current observation, target objective, candidate actions, constraints, required horizon and required evidence kind.
- Compare static HDC, zero-input LTC and input-driven HDC-LTC with event-time/elapsed-time receipts and irregular observation gaps.
- Keep HDC similarity strictly as candidate-ranking metadata; canonical evidence and MACE qualification remain authoritative.

Required metamorphic controls:

| Intervention | Required behavior |
|---|---|
| Permute event IDs while preserving content and canonical links | Semantic ranking remains invariant |
| Change event content while preserving its ID | Candidate ranking changes when the changed field is query-relevant |
| Swap query objective, action or required outcome | Retrieval responds to the query change |
| Insert semantically confusable but irrelevant same-scope events | Relevant evidence remains retrievable and decision utility does not regress |
| Add irregular gaps, late arrivals or duplicate timestamps | Temporal behavior follows canonical time semantics |
| Correct or retract evidence after retrieval but before use | Current-use decision rejects stale evidence after canonical revalidation |
| Change HDC dimension, codebook, profile or event encoder | Profile/model digest changes and replay records the change |

Run exact-scope scan, deterministic KV/episodic, BM25, dense and hybrid baselines under matched full budgets. Retrieval Recall@k, nDCG, MRR, event-ID recovery and latency are separate from MACE utility/regret.

## 6. A causal experiment, not a correlation between retrieval and success

The paired intervention runner tracked by #13141 provides the initial causal-attribution lane. Keep these contrasts distinct:

- Memory value: candidate memory vs true empty memory.
- Content specificity: relevant evidence vs a genuinely irrelevant, same-scope, token/byte-matched placebo.
- Retrieval-substrate value: HDC-LTC vs the strongest reproduced conventional retrieval method.
- Temporal contribution: input-driven HDC-LTC vs static HDC and zero-input LTC.
- Lifecycle value: post-correction vs pre-correction snapshot with canonical source/revision revalidation at use.
- Representation value: persist vs locator vs bounded snapshot vs discard.
- Policy value: cost-sensitive use/verify/clarify/abstain/probe vs fixed-threshold and prompt-only policies.

Within each world-cluster pair, freeze candidate model, initial state, current observation, task, evaluator, permissions, resource profile and deadline. Only the assigned memory intervention may differ. Use independent world-cluster means for paired uncertainty; do not inflate sample size with correlated query rows.

The evaluator must score downstream outcomes and independently inspect evidence support. A change in a decision is not justified merely because the model cites a retrieved item.

## 7. Benchmark ladder and external qualification

No single leaderboard measures the whole architecture. Preserve each benchmark's official protocol and report results separately.

| Lane | What it qualifies | Required controls |
|---|---|---|
| Agent Memory Challenge (AML) Cycle 2 and future eligible cycles | Standardized Add/Search capability across textual, coding and multimodal tracks | Separate track/division scores; public hosted Add/Search endpoint, Smoke then Full evaluation; preserve capability vectors and resource rules |
| LoCoMo / LongMemEval | Long-horizon retrieval and question answering | Same backbone, judge, prompt, retrieval depth, memory budget and token accounting; reproduce Microsoft Memora and other vendor setups independently |
| MemoryArena | Interdependent multi-session memory-agent-environment task completion | Same agent, tasks, tools and budgets; memory must change later actions, not just answer recall questions |
| AMA-Bench | Long-horizon machine-generated trajectories and state/causal information | Official trajectory splits and evaluator; preserve causal/objective information |
| WorldMemArena | Multimodal write, maintain, retrieve and use lifecycle | Original frames/screenshots/action/feedback and stage-specific gold evidence; captions are not ground truth |
| CL-Bench | Genuine experience gain over sequential tasks | Strong naive in-context-learning baseline; separate new experience gain from base-model capability |
| MCB (Remember, Verify, or Ask?) | Write/reuse decision boundary and clarification-vs-verification | Official held-out and evidence-flip contrasts; score declared label and actual tool choice |
| Agentic Abstention | Timely stopping under interactive budgets | Report timely abstention, eventual abstention, excess actions after infeasibility is knowable and task success |
| NeuroBench (where relevant) | Algorithmic correctness/complexity vs deployment-aware system efficiency | Separate hardware-independent algorithm track from measured system timing/efficiency |

### 7.1 Current external anchors

As of 2026-11-22, the official Agent Memory Challenge Cycle 2 results (published 2026-11-19) report open-method leaders:
- Textual: Resurface 61.22/100.
- Coding: InsightMem 63.33% resolved.
- Multimodal: Khipu 43.26/100.

Commercial/closed-source leaders are MemoraX Textual 69.26, Coding 68.00% resolved and Multimodal 53.68. These are separate official tracks/divisions, not interchangeable values.

Microsoft Research Memora reports 86.3% LoCoMo and 87.4% LongMemEval in its own 2026 evaluation setup, plus up to 98% fewer context tokens than full-context inference. Reproduce that exact setup before claiming a delta. Do not pool its results with AML, Redis or third-party benchmark repositories.

MemoryArena reports that near-saturation on long-context memory QA does not ensure success on interdependent agentic tasks. WorldMemArena has 400 multi-session multimodal tasks and exposes write/maintain/retrieve/use failures. CL-Bench reports that dedicated memory systems do not automatically improve experience gain and that naive ICL can outperform them. AMA-Bench reports 57.22% average accuracy for AMA-Agent and an 11.16 percentage-point gain over its strongest baseline within that paper's protocol. These results remain task/model/protocol-specific.

### 7.2 Programme targets (aspirational)

- AML next-cycle open-method stretch: +5 absolute points over the relevant Cycle 2 track leaders—Textual 66.22, Coding 68.33% resolved, Multimodal 48.26. Preserve per-capability leaders, division and protocol; no cross-track pooling.
- Microsoft Memora: first reproduce 86.3 LoCoMo and 87.4 LongMemEval within 1 point under the same setup; only then target +2 points on a held-out extension at no higher context/memory token budget.
- MemoryArena / WorldMemArena / AMA-Bench: preregister at least +5 percentage points downstream task success over the strongest reproduced same-backbone baseline, with the paired 95% interval excluding zero and no family regression worse than 2 points. This is a proposed decision rule, not a universal standard.
- CL-Bench: require positive experience gain over naive ICL and no catastrophic negative transfer on held-out task families.
- MCB: target at least +5 absolute points in held-out action macro-F1 and clarification recall over the strongest reproduced same-backbone baseline, while not increasing false durable commitments.
- Agentic Abstention: target at least 25% fewer excess tool calls after a stop condition is knowable, without task-success regression greater than 1 point versus the same-backbone prompt-only baseline.
- Internal MACE: zero known invalid/stale/wrong-scope/wrong-estimand evidence use on deterministic controls; positive held-out paired decision-utility contrast with cluster-bootstrap lower bound above zero.
- HDC-LTC superiority: positive matched contrast against the strongest reproduced conventional retrieval baseline at equal all-in memory, context, verification, compute and latency budgets. A positive contrast against no-memory alone is not enough.

A target is not a prediction or result. Hard evidence, privacy, authority and lifecycle gates remain separate zero-tolerance outcomes.

## 8. Resource frontier

Every baseline reports:
- persistent memory bytes, index/codebook bytes, metadata, raw/evidence sidecars, journals and tombstones;
- bytes read/written/moved per retrieval and update;
- ingestion and consolidation latency/work;
- retrieval and end-to-end p50/p95/p99;
- context tokens and model calls;
- verification and re-observation calls;
- CPU/GPU time and allocations;
- monetary cost;
- physical joules only when measured on a declared hardware boundary.

Publish quality-vs-cost, risk-vs-coverage, and freshness-vs-latency frontiers. A method is not more efficient because its vector payload is small if canonical evidence resolution, verification, updates or background consolidation dominate total cost.

## 9. Composition and ownership

This document is an integration contract, not a new implementation owner.

- #12743: MACE causal memory, experience ingress/consolidation, re-observation selector and initial generated-world evidence.
- #13128 / #13127: HDC-LTC candidate-retrieval protocol and representation qualification.
- #13141: paired memory-to-decision intervention runner.
- #13153 / #13154: selective ingestion and query-time action policy.
- #8427: external benchmark and cost-quality frontier.
- #11816: independent evaluator and label custody.
- #12846 / #12731: source-bound promotion receipts and governed consolidation lifecycle.
- #9954 / #12401 / #9075 / #13091: canonical time and freshness at decision consumption.
- Mycelix: source, authority, consent, revocation and capability receipts.
- Symtropy: execution, observation and reconciled outcome receipts.

No ownership is moved by this architecture. Reuse each existing verifier, lifecycle owner and evidence schema; do not create parallel authorities.

## 10. Qualification gates

A decision-sufficient memory claim requires:
1. A defined target decision family and estimand.
2. Exact source/evidence resolution and current-at-use revalidation.
3. Zero hard provenance, authority, privacy, scope or lifecycle violations on deterministic adversarial controls.
4. A matched true no-memory / ICL baseline and a matched relevant-content placebo.
5. Strong lexical/dense/hybrid retrieval baselines.
6. Paired held-out world-cluster statistics with all failures and invalid blocks reported.
7. Worst-family and subgroup outcomes; no mean-score masking.
8. Full resource accounting.
9. Official external benchmark adapters and results where the claim says it applies.
10. Independent evaluator custody before challenge-grade or SOTA claims.

A runner passing its own synthetic fixtures qualifies implementation behavior only. It does not establish real-world causal understanding, production readiness, or leaderboard superiority.

## References

- Agent Memory Challenge Cycle 2 official results (2026-11-19): https://agentmemoryleaderboard.ai/results/cycle-2/
- Microsoft Research Memora (ICML 2026): https://www.microsoft.com/en-us/research/blog/memora-a-harmonic-memory-representation-balancing-abstraction-and-specificity/
- MemoryArena (ICML 2026): https://proceedings.mlr.press/v306/he26am.html
- AMA-Bench (ICML 2026): https://proceedings.mlr.press/v306/zhao26bs.html
- WorldMemArena (2026): https://arxiv.org/abs/2605.29341
- CL-Bench (2026): https://arxiv.org/abs/2606.05661
- Memory–Clarification Boundary: https://arxiv.org/abs/2608.19564
- Agentic Abstention (NeurIPS 2026): https://lhannnn.github.io/agentic-abstention/
- NeuroBench: https://neurobench.ai/
