# Retrieval Capability Decomposition Protocol v1

Status: research protocol and synthetic fixture; no model runs or external benchmark reproduction claimed.

## Why end-to-end accuracy is not enough

A search agent can fail at several distinct stages:

1. Search failure: required evidence was never found.
2. Reading failure: required evidence was retrieved but not extracted or used correctly.
3. Grounding failure: the answer may be correct, but its cited support is absent, irrelevant or insufficient.
4. Distractor failure: truthful evidence about a nearby answer-changing query overrides the evidence for the original question.
5. Stopping failure: the agent continues searching after the required claims are already supported, increasing cost and possibly adding harmful context.

Recent Findings papers motivate this decomposition:
- Retrieval Myopia reports that up to 46.2% of examples were answered incorrectly even when a gold-bearing document was present in retrieved material; repackaging the same documents as plain QA recovered up to 48.9% of cases.
- Lazy Grounding reports that factual evidence for nearby answer-changing rewrites can reduce accuracy by 5.9 points on average and up to 17.3 points.
- HALT frames stopping as claim–evidence coverage rather than generator confidence, reducing redundant search while largely preserving exact match.

The final answer score alone conflates all five stages.

## Six paired experimental conditions

Each task is evaluated in six arms. The model checkpoint, task wording, response budget, decoding settings and task-level evidence contents are pinned. Arm order is randomized within each independent run replicate.

| Arm | Candidate input | What it identifies |
|---|---|---|
| no_evidence | Question only; no retrieval corpus or evidence packet | Parametric/prior baseline |
| native_search | Question plus sandbox search tool and controlled corpus | End-to-end search + reading + grounding |
| gold_evidence_injection | Question plus exact supporting documents in a structured evidence bundle | Ability to use evidence when retrieval is removed |
| gold_plus_hard_distractors | Same gold evidence plus factual evidence about nearby answer-changing rewrites | Distractor sensitivity / lazy grounding |
| evidence_complete_stop | Native search; evidence sufficiency becomes true before additional redundant/distracting results are available | Stopping discipline and marginal retrieval utility |
| plain_qa_repackaging | Exact same gold-document bytes and question, repackaged as a plain QA context | Interaction/format effect, following the Retrieval Myopia control |

The gold-injection and plain-QA arms must contain byte-identical evidence bodies. Only the presentation wrapper may differ. The gold+ distractor arm must preserve every gold document and add only the preregistered distractor documents. No arm may silently change the question or oracle.

## Primary contrasts

1. Evidence-use lift: gold_evidence_injection minus no_evidence. This is not a pure reading skill estimate if the model already knows the answer; report task-level prior difficulty and both arms.
2. Search gap: gold_evidence_injection minus native_search. Large positive values indicate that evidence use is stronger than the agent's end-to-end retrieval pipeline.
3. Distractor penalty: gold_evidence_injection minus gold_plus_hard_distractors. A drop with unchanged gold availability is evidence of distractor susceptibility.
4. Format effect: plain_qa_repackaging minus gold_evidence_injection. If the documents are byte-identical, the contrast isolates representation/presentation differences.
5. Stopping utility: evidence_complete_stop against a preregistered continuation policy, on accuracy, redundant calls, tokens, cost and latency.

Report the contrast sign convention in every release. Do not infer a pure direct effect under arbitrary interference between tasks or persistent agent memory.

## Gold evidence and distractor graph

Each sealed task oracle defines:
- atomic claims needed to solve the task;
- one or more exact supporting document spans per claim;
- allowed alternative evidence paths;
- nearby answer-changing rewrites;
- factual distractor documents tied to those rewrites;
- expected answer and answer canonicalization policy;
- explicit ambiguity or conflicting-source state.

A required claim is covered only when its support is present in the evidence actually exposed to the candidate. Lexical overlap alone does not count as entailment. The first v1 synthetic fixture uses exact spans and arithmetic, so its support graph can be checked without an LLM judge.

A distractor is not false misinformation. It is true information about a neighboring scenario that does not answer the original question. This distinction isolates lazy grounding from ordinary fact-checking.

## Metrics

### Search
- Required-evidence Recall@k and claim coverage.
- MRR/NDCG over relevant documents.
- Source-family coverage and duplicate-source concentration.
- Search query count, unique documents and retrieval cost.

### Reading
- Exact/partial answer accuracy with gold evidence injected.
- Correct use of each atomic claim.
- Evidence-use lift relative to no-evidence baseline.
- Contradiction and omission rates.

### Grounding
- Claim-level citation precision/recall.
- Citation span exactness and citation coverage.
- Unsupported-claim rate.
- Source provenance and source reliability strata.

### Distractor resistance
- Nearby-answer adoption rate.
- Robust accuracy delta: gold + distractors versus gold only.
- Incorrect answer selection when the gold evidence remains present.
- Confidence and abstention calibration under evidence conflict.

### Stopping / efficiency
- Queries after all required claims are supported.
- Redundant retrieval fraction.
- Answer accuracy and evidence coverage at stop time.
- Tokens, calls, wall time, tool cost and compute cost.
- Net decision utility per retrieval-cost unit.

Always report denominators, missingness, invalid outputs and abstentions. Do not silently drop failures or charge only the model while ignoring allocator/search overhead.

## Statistical design

- The task is the paired unit for within-task arm contrasts.
- Independent run seeds are repeated measurements, not new tasks.
- Source family and task family are clusters; questions copied from one family are not independent.
- Use paired run-level differences and a preregistered hierarchical/GLMM analysis for generalized accuracy.
- Separate finite-benchmark accuracy from generalized accuracy to the task-family population.
- Determine replicate counts by simulation with task-family variance, source-family dependence, arm interactions, stochastic runs, and missingness. The initial target is at least 90% power for the smallest decision-relevant contrast, not a fixed item count.
- Predeclare primary contrasts, multiplicity treatment, outcome parser, citation scorer, missingness policy, and early-stop rule.

## Controls and qualification gates

- 100% of gold evidence must be present in gold-injection and plain-QA arms, checked by the evaluator before a run starts.
- 100% of gold document content digests must match across the paired gold-format arms.
- 100% of distractor-arm gold documents must match the gold-only arm exactly.
- 0 answer labels, hidden role labels, root seeds or sealed oracle fields in candidate-visible artifacts.
- 0 deterministic synthetic control cases may score as grounded when the cited span does not contain the required support.
- 100% arm runs must bind to the same checkpoint, task, prompt/decoding and declared budget.
- No retrieval_clean or grounding_verified claim when the search-time trace is invalid or channel coverage is partial.
- The evaluator must retain enough evidence to adjudicate semantic matching. A digest alone cannot substantiate a semantic support claim.

These are software/protocol gates. They are not measured SOTA targets and do not certify real-web retrieval performance.

## Industry interoperability

Inspect provides reusable datasets, solvers, tools, scorers, agent execution, sandbox policies and trajectory review. Lighteval provides multi-backend model execution and sample-level results. Use those systems as execution adapters.

Symthaea should specialize in:
- deterministic, answer-changing distractor construction;
- sealed claim/evidence support graphs;
- gold-evidence interventions that isolate reading from search;
- paired format controls;
- trace-linked search and stopping metrics;
- source-family-aware causal analysis and evidence integrity.

The benchmark artifact should export to Inspect and Lighteval rather than require a proprietary runner.

## Synthetic pilot fixture

fixtures/retrieval-decomposition-pilot-v1.json demonstrates one evidence task:
- Delta inspection: 3 billable days at $180/day and 2 overnight stays at $75/night; correct total $690.
- Echo is a nearby, factually true scenario with 3 overnight stays; its total is $765.
- The Echo information is a distractor, not a false fact about Delta.

This fixture proves only that the protocol can represent gold facts and a nearby answer-changing distractor. It is not a reasoning benchmark result.

## Claim boundary

A high gold-injection score does not prove good search. High retrieval recall does not prove good reading. A correct answer does not prove grounded reasoning. A retrieval event does not prove influence. A synthetic distractor test does not establish broad-world robustness. Keep those claims separate in every report.


## Executable artifact qualification

The Rust module src/retrieval_decomposition.rs validates:
- canonical corpus and sealed-oracle SHA-256 digests (each computed with its own digest field cleared);
- every document content hash and UTF-8 byte length;
- exact byte spans and per-span SHA-256, with claim text equal to the cited source span for directly supported claims;
- unique claim/document IDs and acyclic derived-claim dependencies;
- exact arithmetic for the primary answer and each nearby distractor answer;
- allowed evidence path document IDs;
- experiment-to-corpus/oracle/benchmark bindings;
- all six required experimental arms, unique task/replicate IDs, unique paired seed commitments, and per-replicate arm-order permutations;
- candidate/oracle separation declaration.

Audit the synthetic pilot:

~~~sh
cargo run --locked \
  --manifest-path research/contamination-panel-001/Cargo.toml \
  -- audit-retrieval-decomposition \
  research/contamination-panel-001/fixtures/retrieval-decomposition-pilot-v1.corpus.json \
  research/contamination-panel-001/fixtures/retrieval-decomposition-pilot-v1.oracle.json \
  research/contamination-panel-001/fixtures/retrieval-decomposition-pilot-v1.experiment.json \
  /secure/evaluator/retrieval-decomposition-audit.json
~~~

The report is an artifact-contract audit only. The included manifest deliberately has experiment_status=protocol_only; its model checkpoint, inference harness, prompt/decoding, budget, sample-size and seed bindings are fixture digests, not measurements or an executed run. The verifier does not run a search agent, score citations, establish semantic entailment, or estimate causal effects.

The initial span gate is intentionally exact and conservative. Later real-web evaluation needs a separate blinded semantic-adjudication protocol, with exact/fuzzy/semantic matching methods, human reliability estimates and held-out adjudication items. Do not silently replace these exact synthetic controls with an LLM judge.

