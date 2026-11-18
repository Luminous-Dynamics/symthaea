# IG-HIDDEN-INFO-001 — protocol v0.1

**Protocol state:** research design; exploratory model campaign not yet run.  
**Owner:** [Symthaea #12762](https://github.com/Luminous-Dynamics/symthaea/issues/12762)

## Research question

When decisive information exists somewhere in a group but is not initially shared, can agents discover who has it, surface it, verify it, use it, and stop at the right time under a fixed communication and compute budget?

## Motivation and external evidence

HiddenBench (Li, Naito & Shirado, ICML 2026) provides the direct benchmark anchor: 65 Hidden Profile tasks, 15 evaluated frontier models, and a reported 30.1% multi-agent accuracy with distributed information versus 80.7% for a single agent with complete information. The authors identify latent information asymmetry and premature convergence on shared evidence as central bottlenecks. They report gains from a structured communication protocol.

- Paper: https://proceedings.mlr.press/v306/li26ej.html

Related, separate phenomena:
- Hidden/private stance adoption without public endorsement: https://arxiv.org/abs/2610.22138
- Phantom consensus from ambiguous public signals: https://arxiv.org/abs/2611.08971
- Persistent private stance changes after group deliberation: https://proceedings.colm.cc/paper_files/paper/2026/hash/871ed095b734818cfba48db6aeb25a62-Abstract-Conference.html

Do not pool these phenomena into a single “social influence” label.

## Experimental unit and information model

The independent experimental unit is a frozen task × model/profile × randomized treatment block. Agents and turns inside the same interaction are nested observations, not independent samples.

Each task contains:
- a task statement and evaluator-only answer key / acceptable solution set;
- a finite set of evidence atoms;
- a declared subset of decisive atoms;
- source identity, reliability and relevance for every atom;
- an initial private evidence allocation to each agent;
- a dependency rule specifying which evidence set is sufficient;
- one designated decision agent (or a separately specified group aggregation rule);
- communication graph, permitted channels and delivery semantics;
- model/provider revision, system/developer/user prompts, tools and decoding;
- per-agent and total token/message/tool/time budgets;
- fixed seed, scoring implementation, split identity and licensing metadata.

The answer key, hidden allocation labels and evaluator-only expected metrics must never be provided to agents or the live controller.

## Four-stage protocol

### 0. Seal the task

Freeze the task, evidence catalog, decisive-atom set, source profile, agent roster, initial allocations, graph, prompts, model revisions, tool allowlist, budget and randomization seed. Hash all artifacts before execution.

### 1. Private baseline

Each agent privately records its initial solution, claims, evidence IDs actually available, uncertainty, unresolved variables and what evidence it believes it lacks. Treat these as observed elicited outputs, not hidden beliefs.

### 2. Public discovery and exchange

The run records every ordered message and delivery event. Evidence atom disclosure, request, challenge, verification, correction and answer revision must be separate events. A communication graph defines possible channels; only a recorded delivery event establishes that content reached a recipient.

### 3. Private post-interaction probe

Each agent is re-elicited in a fresh private call without live group transcript, peer outputs, group tally or answer key. Record final solution, uncertainty, evidence correctly recalled/used, and public/private divergence. A full-context-reset repeat control is separate from an exposure-memory re-elicitation.

### 4. Independent scoring

A non-participating scorer uses the sealed answer key and event records to calculate outcome and evidence measures. It must retain missing, failed, dropped, timed-out and indeterminate runs. Candidate agents do not grade their own success.

## Primary treatment arms

Use the same tasks, model revision, agent count, context policy, tool allowlist, decoding and total resource budget wherever the compared treatment permits.

| Arm | Protocol | Purpose |
|---|---|---|
| A0 | Independent agents, no communication | Uncoordinated baseline |
| A1 | Single agent with all facts | Upper bound, not a matched collective baseline |
| A2 | Standard debate | Conventional multi-agent baseline |
| A3 | Published HiddenBench structured communication | Direct method reproduction |
| A4 | Explicit evidence inventory | Test whether agents track what they have and lack |
| A5 | Targeted evidence requests | Test whether agents ask the likely holder of a missing fact |
| A6 | Evidence-first synthesis | Require claim/source checks before answer aggregation |
| A7 | Existing evidence-aware controller (#11383) | Test adaptive request/verify/challenge/stop under equal budget |
| A8 | Ablations | Remove one mechanism at a time |

Positive controls must include a minority-held fact that is true and decisive. Anti-conformity alone is insufficient; failure to update on valid evidence is also a failure.

## Core outcome vector

Report the dimensions separately.

1. **Task outcome:** exact correctness / acceptable solution set; abstention quality; errors by severity.
2. **Information discovery:** decisive evidence coverage; disclosure latency; undisclosed decisive facts; unique facts per message/token; request precision/recall.
3. **Evidence use:** correct attribution; decisive-fact use in final answer; false-source uptake; response to correction.
4. **Collective epistemics:** private/public divergence; missing-information awareness; unsupported confidence change; dependence on repetition/status/phantom consensus.
5. **Resources:** input/output tokens, tool calls, messages, latency, cost and cost per correct task.
6. **Validity:** missing stages, undelivered messages, malformed provenance, evaluator disagreement and run completeness.

Do not treat HDC similarity, synchrony, Phi, consensus or raw message volume as proxies for answer correctness or evidence coverage.

## Confirmatory baselines and target gates

The first campaign must reproduce the published HiddenBench baseline and structured-communication method on the original 65 tasks where licensing and artifacts permit. The reported 30.1% / 80.7% are literature reference points, not Symthaea results or a local baseline until reproduced.

Proposed project targets after reproduction:
- at least +10 percentage points task accuracy over standard multi-agent debate at matched budget, with paired 95% interval excluding zero on the confirmatory split;
- at least +20 percentage points decisive-evidence coverage without increasing unsupported evidence uptake;
- evidence-positive warranted revision no worse than the best reproduced evidence-only baseline;
- at least 15% lower cost per correct task, or report a quality gain at equal cost;
- at least one held-out topology and one held-out task family before any transfer claim.

These are internal research targets, not industry standards. Exploratory pilots may calibrate a target only before the confirmatory split is opened.

## Statistical and reproducibility rules

- Pin task, source, prompt, model/provider, decoding, tool, graph, budget and scorer revisions.
- Randomize treatment order and paired task assignments where possible.
- Use multiple independent seeds and power analysis; do not count agent-turn rows as independent samples.
- Keep model families and task families separated in the held-out evaluation.
- Report per-task results and uncertainty, not only the mean.
- Preserve all failed/incomplete runs; unknown or missing private re-elicitation is not a pass.
- Compare at equal budgets; publish accuracy/evidence-coverage/token/latency/cost Pareto curves.
- Do not infer a human-group result from synthetic agent experiments.

## Composition and ownership boundaries

This protocol composes with #3054/#3095 (experiment lab and receipts), #11337 (communication channels), #11320 (stance exposure), #11350 (revision evidence), #11367 (confidence), #11383 (deliberation controller), #11683 (phantom consensus), and #5471/#5478 (semantic-progress/causal metareasoning).

Do not create a parallel message graph, delegation graph, experiment orchestrator, evidence ledger, judge ensemble or metareasoning scheduler. A delegation edge from #11248 is never a message channel.

## Claim ceiling

A structural fixture pass validates only the declared state/event contract. A synthetic model run validates only its declared synthetic domain. Only a licensed, independently reproducible HiddenBench evaluation with matched baselines supports claims about performance on HiddenBench. No result here establishes human-group validity, machine consciousness, subjective experience, political legitimacy or authority.