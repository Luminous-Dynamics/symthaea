# Adaptive transmission programs: CivBench benchmark protocol v0.1

**Status:** synthetic, executable mechanism benchmark.  
**Code:** src/adaptive_transmission.rs and src/bin/adaptive_transmission_sweep.rs.  
**Not a reproduction:** built-in task families contain 14 recipes each, not the 720-element Collective LittleAlchemy task in Colas et al. (2026). The Rust policy is a deterministic hand-coded heuristic, not an LLM-evolved transmission program.

## 1. Research anchor and SOTA comparison

Colas, C., Perez, J., Nisioti, E., Mocherla, A., Oudeyer, P.-Y., Moulin-Frier, C. & Derex, M. (2026). *Discovering Adaptive Transmission Programs for Collective Innovation*. arXiv:2608.24545, submitted 25 August 2026; presented at CogSci 2026.

- Paper: https://arxiv.org/abs/2608.24545
- Author publication record: https://www.cedriccolas.com/publications/

The paper's core distinction is between **state-agnostic networks** (who can communicate and when) and **state-aware transmission programs** that choose who sends what to whom based on agents' current inventories/memories and collective progress. It uses LLM-guided evolutionary search to discover programs in Collective LittleAlchemy. The abstract reports up to 37% higher collective performance over standard literature baselines; ablations report that shuffling content while preserving the transmission schedule removes the gain. The paper also reports transfer across domain variations and population sizes.

The paper is an in-silico result in a deterministic combinatorial task, not a demonstrated 37% improvement in human organizations. Its centralized protocol has access to agent and collective state, which is a significant privacy, autonomy and scalability assumption. CivBench therefore separates:
- **source reproduction:** the authors' exact Collective LittleAlchemy environment and LLM-evolved programs, once artifacts and licenses are pinned;
- **mechanism evaluation:** whether transparent state-aware routing improves over topology/timing controls in a reproducible smaller DAG task;
- **deployment evaluation:** whether useful state information can be shared with consent, minimization and local computation, without central access to private inventories.

No score from this synthetic crate is directly comparable to the paper's 37% result.


### Source-specific reference details (for a future reproduction adapter)

The arXiv HTML methods describe Collective LittleAlchemy as 720 deterministic elements with four base elements (water, fire, earth, air). Each agent has a permanent inventory and a memory of attempted combinations represented as triples (input element 1, input element 2, result), where result records success or failure. A learner attempts one combination per synchronous step. After that attempt, a transmission protocol shares memories. Received memories make their involved elements available in a temporary social inventory for that step; only elements discovered by the recipient's own successful combination become permanent. Thus a recipient can use socially available elements without the shared result automatically becoming an individual discovery. The group objective is the number of unique outputs discovered by any agent, and each agent can discover at most one new element per step.

The primary empowerment-agent model uses a softmax over action empowerment (temperature beta=0.05), an agent-specific noisy item-empowerment model and a fixed social bias: when social inventory is non-empty, probability 0.5 of restricting the candidate combination set to combinations involving at least one socially received element. Alternative agents are uniform stochastic combiners or Gemini 3.0 Flash LLM agents. The attention cap is 20 received memories per agent per step, with uniform subsampling when more are sent.

Five baseline protocols are defined:
- Asocial: no sharing.
- Paired: fixed dyads, up to 10 random memories with 0.5 probability per step.
- Dynamic: paired sharing plus periodic visits to other dyads.
- Graph: sharing along an Erdős–Rényi graph with p=0.2.
- Stochastic: each agent receives 20 randomly sampled memories from other agents per step, saturating the stated attention cap.

The protocol interface is a stateful Python class whose share method observes agent states and emits sharing decisions. The stated design constraint is that protocols can observe state but cannot directly modify agent inventories and do not see the ground-truth recipe graph.

The reported search uses five independent evolutionary runs. Each runs 200 iterations; the first 50 are warm-up candidate generation, then Thompson-sampled parent selection and prompts to improve or diversify. The model is Gemini 3.0 Flash. Candidate programs are executed in a short sandbox with up to three repair attempts. Fitness is mean collective performance across 30 simulations, with 10 empowerment agents over 150 steps. The authors also analyze 1,000 independently generated protocols.

The main population generalization settings include 6 and 20 agents, stochastic and LLM agents, mixed populations, stronger empowerment bias and stronger social bias. Domain generalization includes depth-limited tasks (86 reachable elements), three-item starts (587–684 reachable), heterogeneous starts and item dropout (fewer than 300 reachable elements). The paper reports +33% average and +37% top-protocol gain at 150 steps, +32% average and +36% top-protocol gain at 300 steps, +49% for population size 20, and gains across the listed domain variants. These are paper-reported simulation results; not reproduced here.

A faithful reproduction must recover the exact supplementary task generator, agent learning/update routines, schedule/budget semantics, baseline tie-breaking, memory lifecycle, program API and evaluator. The anonymized author-code URL is not yet machine-readable in the current environment. Do not recreate missing rules from a secondary summary and label them the official implementation.

## 2. Scientific question

Under the same social graph, starting knowledge, individual-learning opportunities, transmission count and recipient attention budget:

> Does routing the right knowledge item to a recipient who can use it improve collective discovery more than choosing only the sender, receiver or time?

A message may be delivered but irrelevant. A recipe may be acquired but dormant because its prerequisites are missing. A dormant recipe may become executable when prerequisites arrive later. The task therefore separates:
1. possession of a recipe/rule;
2. possession of its input elements;
3. execution of the rule to discover an output;
4. contribution to new collective coverage.

This is intentionally different from a one-dimensional adoption rate.

## 3. Synthetic task families

Each built-in task is a topologically ordered directed acyclic graph (DAG) of recipe rules:
ingredients -> output element.

Every family has six base elements and 14 recipes, with outputs 6–19:
- Layered: mostly sequential layers, with a narrower progression.
- Branching: several partially independent branches that can be specialized.
- Convergent: multiple branches share prerequisites and converge toward later results.

A recipe can be learned before it can be executed. The agent stores the rule; when all input elements later become available, deterministic forward closure unlocks its output. This allows us to measure useful latent knowledge and delayed complementarity.

The DAGs are synthetic fixtures, not extracted from the authors' 720-element task. A future source adapter must pin the original task file, rule semantics and scoring before any paper replication claim.

## 4. Policies and controls

The current Rust benchmark includes seven policies:
1. NoSocial — individual innovation only.
2. FixedRing — deterministic neighbour schedule; send the first available rule.
3. FixedRingContentShuffled — preserve that ring's sender/receiver schedule but select a random available rule.
4. RandomNetworkContent — randomize sender/receiver pairs over the same generated graph and choose random sender-held rules.
5. StateAwareFrontier — select messages using recipient prerequisite coverage, immediate executability, novelty to collective knowledge and downstream recipe unlock potential.
6. StateAwareContentShuffled — first compute the state-aware sender/receiver schedule and each sender's selected recipe payload multiset, then shuffle those payloads only among that sender's scheduled recipients. This preserves who communicates with whom, when, how many messages are sent, which sender-owned recipes are sent, and each sender's selected content multiset.

The two controls ask different causal questions. StateAwareRandomContent is the closer analogue of the paper's ablation: preserve who shares with whom, timing and message count, but randomize which sender-held memories are selected. StateAwareRecipientShuffled is stricter: also preserve each sender's selected payload multiset, changing only which scheduled recipient receives which selected payload. The first tests state-dependent content selection; the second tests recipient-specific matching conditional on a sender's selected content.

The heuristic score is transparent and deterministic:
- high priority when the recipient has all prerequisites and can execute the rule immediately;
- partial credit for how many prerequisites are already present;
- a novelty bonus when the rule's output is not yet known to the collective;
- a bounded structural bonus for how many downstream rules depend on the output.

This is a hand-written heuristic, not a discovered program or an LLM-generated policy.

## 5. Matched-budget experiment

The adaptive-transmission-sweep binary runs a predeclared grid:
- task family: Layered, Branching, Convergent;
- population: 8, 16, 32;
- graph shortcut probability: 0.0, 0.2 (ring baseline plus optional undirected shortcuts);
- attention capacity per recipient per step: 1, 2, 4;
- seven policies above;
- 128 seeds per condition;
- 12 steps;
- one seed agent with three initial rules;
- two individual-discovery attempts per agent per step;
- individual-discovery probability 0.04;
- per-step transmission budget equal to the population size;
- successful acquisition probability 0.9.

This is 378 conditions and 48,384 total simulation runs. All policies in a given task/population/topology/attention condition use the same seed IDs and configuration. The simulator now uses domain-separated streams and identity-keyed random draws. Graph generation and schedule randomness have separate streams. Individual discovery priority and success are keyed by (master seed, step, agent ID, recipe ID), so reordering or adding other candidates cannot reassign a draw to a shared recipe. Message-acquisition success is keyed by (master seed, step, sender, receiver, route-slot), deliberately excluding payload identity; a content shuffle that preserves sender/receiver slots therefore preserves acquisition luck for that route. This is stronger common-random-number control than same-seed reuse. Remaining limitation: policy-dependent inventories change which recipes are eligible and which enter the top-k attempt budget; different route sets cannot be paired. The held-out paired-difference estimator remains the correct basis for comparison.

Outputs include:
- final collective result coverage and a 95% normal-approximation interval over simulation seeds;
- maximum discovered depth;
- mean results and recipes per agent;
- individual discoveries;
- social recipe acquisitions;
- new collective results;
- transmissions and redundant transmissions;
- total recipient-attention uses;
- new results per transmission.

The experiment controls message budget and attention, but the current task is not a model of human learning time or cognitive effort. Do not equate one rule transfer with one human demonstration. Individual-learning draws are identity-keyed by agent/recipe, and acquisition draws by sender/receiver/route-slot (not payload), to preserve the random opportunity assigned to a shared experimental unit across matched policy runs. Policies with different eligible recipe sets or route sets are not perfectly paired; this limitation remains part of the interpretation.

## 6. Proposed paradigm shifts

### 6.1 From graph topology to executable transmission programs
Make policies first-class, versioned objects with observable inputs and bounded outputs. A policy should declare which state fields it uses, its message/attention limits, deterministic or stochastic semantics, and its objective. Compare topology-only schedules with content-aware programs under exact matched budgets.

### 6.2 From global optimality to recipient-state complementarity
The value of a knowledge item is conditional on a learner's inventory. A rule that is useless to one recipient can be immediately executable for another. The next extension should represent item value as a function of receiver state and downstream dependencies, rather than a single global score.

### 6.3 From redundancy minimization to useful redundancy
Repeated messages waste attention, but redundant independent paths protect against dropout, corruption and malicious senders. Add controlled sender failure, message corruption, retry and alternate-route experiments. Score both efficiency and recovery.

### 6.4 From centralized omniscience to privacy-preserving local routing
The paper's centralized protocol sees all agent states. CivBench should compare:
- centralized state-aware routing (upper-bound reference);
- consented feature sharing (only declared missing prerequisites or task-family needs);
- receiver-local pull (the receiver asks for a rule);
- decentralized peer negotiation;
- no state disclosure.

The key measure is the performance/privacy/autonomy frontier, not whether centralized routing wins a toy task.

### 6.5 From one-shot routing to adaptive phases
The paper describes strategies that specialize groups early and cross-pollinate later. Test an isolate-then-share phase policy against always-global, always-local, and fixed schedules. A phase transition can be based on collective novelty saturation, branch imbalance, redundancy, or unmet prerequisites.

### 6.6 From hand-coded policy to program discovery
After baselines are reliable, introduce a safe, bounded transmission-program DSL. Search it with evolutionary strategies or LLM-generated candidates in an offline sandbox. Every candidate should be tested on training DAGs, selected on validation DAGs, and reported once on frozen held-out DAG families. The search process is not allowed to see evaluation outcomes.

## 7. Evaluation targets

These are proposed engineering gates, not achieved results:
- On held-out DAG families and population sizes, state-aware routing should beat both fixed-ring and random-network content baselines under the exact same message and attention budget, with paired-seed uncertainty intervals.
- The content-shuffled ablation should remove a substantial share of any state-aware benefit. If it does not, the claimed benefit may be schedule/topology rather than content matching.
- State-aware routing should improve result coverage and depth without relying on more messages, more individual attempts, or lower acquisition thresholds.
- Report redundant transmissions, dormant acquired rules, time-to-execution, lineage concentration and branch coverage alongside final coverage.
- Under sender dropout/corruption, measure recovery and the benefit of diverse alternate routes.
- Under partial state disclosure, quantify capability loss and privacy cost.
- A program is promoted to a stronger benchmark claim only if it transfers to unseen recipe DAG families and agent populations, not merely new seeds from the same tiny task.

A reasonable first target is a statistically reliable positive gain over the strongest matched static baseline on held-out families. A larger numeric target (e.g. 10% relative coverage improvement) should be frozen only after a pilot establishes the attainable scale for these 14-recipe DAGs. Do not set 37% as a target by copying the paper's result; the tasks and scores are different.

## 8. Security, agency and civilizational relevance

A transmission program is a mediator of access to knowledge. It can amplify some voices, starve other lineages, create dependencies or make recipients legible to a central authority. CivBench should therefore test:
- sender and lineage concentration;
- manipulation of claimed usefulness or rarity;
- Sybil amplification;
- strategic withholding of prerequisite rules;
- learner right to reject, fork and request alternatives;
- auditability of why a rule was sent to a particular recipient;
- deletion/minimization of private inventory details after routing.

For Mycelix, the relevant architecture is consentful, federated discovery rather than centralized surveillance. For Symthaea, cognitive mechanisms may improve state representation and recipient modeling. For Symtropy, the DAG/network experiments can be stress-tested at larger population scales. Sol-Atlas can visualize transmission routes and collective coverage, but visualization is not validation.

## 9. Reproducibility status

- The current source is a new synthetic Rust mechanism model and sweep.
- The built-in task DAGs are generated in code and are not the paper's Collective LittleAlchemy data.
- The current algorithm is a fixed heuristic, not an LLM-evolved transmission program.
- No simulation output or benchmark score is claimed in this document.
- Exact-head cargo check and cargo test are required before treating the module as executable evidence.
- No head-to-head paper reproduction or state-of-the-art superiority result is claimed.


## 10. Bounded program-grid search and small-sample uncertainty

The companion binary `src/bin/adaptive_program_grid_search.rs` evaluates 108 bounded `TransmissionProgram` candidates. It selects on Layered + Branching training task families (16 seeds per family), re-ranks the top eight on a disjoint validation seed range (12 seeds per family), and evaluates the selected program only once on the held-out Convergent task family (64 seeds). The same master seed is used for the selected program and each baseline within a held-out comparison.

The program-grid runner reports conservative two-sided 95% Student-t intervals, not a fixed 1.96 normal interval. Its critical-value lookup rounds upward within degrees-of-freedom buckets, so it does not understate the t multiplier for the declared sample sizes. In particular, validation has 24 observations (two task families × 12 seeds), training has 32, and held-out test has 64 per candidate/baseline. The broader 378-condition adaptive-transmission mechanism sweep above remains a separate analysis and continues to report a normal-approximation interval over its 128 seeds.

The reported paired delta is computed per seed as selected-program coverage minus baseline coverage. It is a paired-seed difference under shared graph/discovery/schedule/acquisition stream conventions, not an exact per-action random-number match: endogenous task inventories and action lists may differ between policies. A future stronger design should key individual discovery draws by (master seed, step, agent ID, recipe ID) and acquisition draws by (master seed, step, sender, receiver, message slot), then publish the pairing contract and test it independently.

These confidence intervals are descriptive simulation-replicate intervals, not uncertainty intervals for human outcomes. Because the top-eight validation selection is data-dependent, the held-out test is the only stage that can support a generalization claim. Do not promote a candidate based on its training/validation interval alone.
