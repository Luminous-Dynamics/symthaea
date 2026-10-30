# AR-5B — Verifier-grounded direction board v0.1

**Status:** deterministic allocation primitive and unit tests authored; not compiled or executed in this authoring environment.  
**Parent:** AR-5 #9197  
**Research issue:** #9230  
**Dependency:** architecture-engineering foundation #8890  
**Implementation module:** `crates/domains/symthaea-architecture-engineering/src/direction_board.rs`

## Research thesis

Score-only search systems tend to concentrate compute on the currently best-scoring method. That can cause a form of search monoculture: every worker follows the leading candidate family, while structurally different hypotheses are abandoned before their delayed payoff can be measured.

AR-5B introduces a verifier-grounded allocation layer that treats **research directions** as distinct arms. A direction is not a code branch, agent identity, prompt or random seed. It is defined by four canonical dimensions:

1. target capability;
2. solution object / decision unit;
3. construction or search principle;
4. scope.

A method is one concrete candidate delta within a direction. The verifier assigns or confirms direction identity; candidate generators may propose a new direction, but cannot self-certify that it is novel.

This is inspired by the direction-board idea in *Coral Discovery: The Division of Cognitive Labor in Autonomous Research* (arXiv:2610.33281, submitted 23 October 2026). That preprint reports better results on six of nine matched tasks versus the strongest open CORAL comparison, one tie and two losses; it attributes the gains to allocation across directions rather than merely adding agents. These are paper-reported results, not a Symthaea replication or a universal theorem.

## Scope and trust boundary

The board is an advisory scheduler. It:
- consumes records produced by an independent verifier;
- checks schema, digest shape, canonical direction identity and frozen profile consistency;
- records verified, rejected and inconclusive attempts;
- estimates direction-local improvement yield and exploration potential;
- returns a deterministic allocation plan for a finite evaluation-count budget.

It does **not**:
- execute candidate code or perform external actions;
- authenticate verifier receipts or recompute candidate artifacts from their digests;
- access hidden benchmark labels, evaluator truth or qualification keys;
- decide that a candidate is scientifically correct;
- promote a candidate, change acceptance thresholds, modify production cognition or merge code.

The caller must place candidate execution and the evaluator in a separately controlled process boundary. A SHA-256-shaped receipt field is a provenance reference, not a cryptographic signature or proof of verifier honesty.

## Direction identity and method lineage

A direction digest is domain-separated SHA-256 over the canonical descriptor. The current canonicalizer trims, lowercases ASCII, and collapses repeated whitespace. Thus lexical case/whitespace aliases collapse, but semantic equivalence beyond this canonical form remains the verifier's responsibility.

A method attempt binds:
- opaque attempt ID;
- canonical direction digest;
- method ID;
- parent candidate digest;
- candidate digest;
- semantic-delta digest;
- comparison-profile digest;
- metric ID;
- evaluation cost units;
- disposition (verified / rejected / inconclusive);
- normalized score in [0,1] only for verified results;
- verifier receipt digest.

A new-direction proposal binds its descriptor, proposal digest, cost, disposition and verifier receipt. A proposal marked duplicate must map to a direction already on the board; an admitted proposal must not already exist. Duplicate, rejected and inconclusive proposals remain in the new-direction cost denominator.

A direction admitted from the NEW arm starts with its own empty local record. The prior NEW-arm discovery credit is not copied into the new direction's performance yield.

## Allocation rule

The v0.1 board uses a deterministic UCB-style potential:

[
P_i = Y_i + c sqrt{rac{ln(N+2)}{n_i+1}}
]

where:
- (Y_i) is the direction's local-record improvement divided by all evaluation cost spent on that direction;
- (c) is a frozen exploration coefficient;
- (N) is the total number of scored-direction attempts plus NEW-direction proposals;
- (n_i) is the number of attempts/proposals on arm (i).

For the NEW arm, the estimated yield is a declared prior until its first proposal; afterward it is the configured verifier-admission credit divided by the **total** cost of all NEW proposals, including duplicates/rejections/inconclusive outcomes.

The first `minimum_initial_evaluations` on an untried direction and on NEW are protected whenever the requested budget can cover all such arms. Remaining slots are allocated proportionally to potential using deterministic largest-remainder rounding. The plan always allocates exactly the requested number of evaluation slots.

This is a transparent policy baseline, not an assertion that UCB is globally optimal. Future variants should compare Thompson sampling, discounted UCB, contextual bandits, fixed allocations, quality-diversity archives and a Coral Discovery-style board under matched synthetic landscapes.

## Yield semantics

For a direction:
- every attempt increments attempted count and evaluation cost;
- only independently verified attempts can carry a normalized score;
- rejected, inconclusive and non-improving attempts still add cost;
- local best is the maximum verified score observed in that direction;
- local-record improvement is the sum of positive improvements above the previous local best;
- estimated yield is cumulative local-record improvement / cumulative cost.

For NEW:
- every proposal adds its cost, whether admitted, duplicate, rejected or inconclusive;
- verifier-admitted novel directions earn the frozen admission credit;
- new-direction yield is accumulated admission credit / all proposal cost.

This is deliberately not a global score leaderboard. Local yield asks whether a direction is producing new records relative to its own best, while the exploration term preserves opportunities for under-sampled directions.

## Evidence and determinism

The board uses ordered maps/sets for stable serialization. Before a recommendation or digest can be exported, `validate_integrity` now reconstructs every direction's counters, local-best score, total record improvement, cost and yield from the full attempt records. It independently reconstructs NEW-arm proposal counts, admission credit, cost and yield from every proposal, including rejected, duplicate and inconclusive proposals. It rejects noncanonical registered descriptors and multiple admissions of the same canonical direction. This prevents a deserialized board from presenting altered summary counters while retaining apparently well-shaped receipt references.

The full board digest binds:
- schema/version and frozen config;
- canonical direction descriptors and verifier receipt references;
- all full attempt records;
- all full new-direction proposal records;
- derived counters and local records.

The allocation plan binds the source board digest, fixed comparison profile, metric, requested evaluation budget, each arm's prior attempts, estimated yield, exploration bonus, potential and suggested share. The plan hash is domain-separated.

The plan is advisory only. The runtime must separately log the actual arm chosen, actual work performed, overrides and reasons. Recommended allocations must never be reported as executed evaluations.

## Unit tests authored

The module includes tests for:
- direction canonicalization and duplicate rejection;
- local-best credit only on new verified records;
- rejected/inconclusive trials remaining in cost denominators;
- new-direction admission, duplicate and rejection accounting;
- exact budget conservation and deterministic plans;
- protected initial exploration for all untried arms when budget permits;
- a delayed-payoff direction retaining a nonzero allocation after early low yields;
- comparison-profile and metric mismatch rejection;
- duplicate attempt and zero-cost rejection;
- inconsistent disposition/score rejection;
- board digest changes after evidence is recorded;
- board digest binding the semantic-delta receipt, not only summary aggregates.

The follow-up integrity tests also require direction aggregates to reconcile with all attempt receipts, NEW-arm aggregates to reconcile with all proposal receipts, duplicate admissions to fail, and registered descriptors to be canonical. These tests validate deterministic board semantics and serialized-evidence integrity, not the superiority of this allocation policy. They have not been executed yet.

## Next benchmark tranche

The first benchmark should be a synthetic direction landscape with evaluator-hidden payoff schedules and an exact oracle. It should contain:
- an early high-reward direction that saturates;
- a delayed-payoff direction with poor early scores;
- a moderate steady direction;
- misleading early proxy ranking / rank reversal;
- duplicate source refactors that are not distinct directions;
- a genuinely new operator family that is not representable in the frozen grammar;
- a new-direction proposal queue with rejected, duplicate and inconclusive proposals;
- directions that improve one task while regressing on another.

Compare at identical total evaluation and cost budgets:
1. global score-only winner-take-all;
2. fixed round-robin / fixed direction share;
3. random arm selection;
4. UCB over global score;
5. local-record direction board (this module);
6. quality-diversity / MAP-Elites when behavior descriptors are valid;
7. an oracle arm selector as a non-deployable synthetic upper bound.

Primary outputs: best-so-far and regret over budget, time-to-first independently verified improvement, delayed-direction recovery, local records per direction, direction admission rate, unused/wasted budget, coverage of distinct validated directions, allocation entropy/HHI, and cost per validated discovery. Use paired seeds and confidence intervals clustered by generated landscape, not by individual evaluation.

The allocator should only be integrated with live architecture search after this synthetic direction benchmark has passed construct-validity controls. Then connect it to the query-firewalled NAS bridge (#9164), SYM-ARCH-002 (#55), CMAQ-005/006 and operator-held-out scientific discovery (#8536/#8705).

## External comparison

- **ShinkaEvolve (ICLR 2026):** parent sampling, code novelty rejection and bandit selection of LLMs; the official paper reports a 150-sample circle-packing result. https://proceedings.iclr.cc/paper_files/paper/2026/hash/7886b9bafe76c52fd568db10ff9772df-Abstract-Conference.html
- **CodeEvolve (Findings of EMNLP 2026):** island-based evolution, inspiration crossover, meta-prompting, depth refinement, CVT-MAP-Elites and weighted LLM ensemble. Its paper reports matches/surpasses on selected AlphaEvolve-suite tasks; use its released code/data for a pinned reproduction, not the paper numbers as Symthaea results. https://aclanthology.org/2026.findings-emnlp.1345/
- **DeltaEvolve (ICML 2026):** semantic deltas and progressive disclosure to transfer meaningful changes rather than replaying full code history. https://proceedings.mlr.press/v306/jiang26g.html
- **CORAL (2026):** persistent shared memory, long-running asynchronous agents and reflection/intervention. https://arxiv.org/abs/2604.01658
- **Coral Discovery (23 October 2026 preprint):** direction labels, local records, exploration potential, NEW-direction option and advisory allocation. https://arxiv.org/abs/2610.33281

AR-5B should not copy any one method wholesale. Its differentiator is to make direction-level allocation **causal-evidence aware**: every method delta remains attached to a frozen parent, candidate, semantic change, verifier receipt, matched evaluation profile and transfer evidence.

## Claim ceiling

A successful v0.1 test establishes only that the direction-board data contract and allocator behave as specified. A successful synthetic benchmark could establish improved sample allocation on that declared landscape. Neither result establishes a better Symthaea architecture, autonomous scientific discovery, AGI, consciousness, universal SOTA, or production-safe self-modification.
