# Subterra active perception and hazardous exploration — related-work update

**Date:** 2026-11-16  
**Status:** Literature update and design consequences; no new experimental result claimed.

## 1. Two 2026 papers sharpen the problem definition

### 1.1 Active Visual Perception: Opportunities and Challenges

The October 15, 2026 review in *Sensors* (26(20), 6544; DOI 10.3390/s26206544) proposes that active perception should be analysed as a closed loop linking:

1. what perceptual evidence is currently missing;
2. which controllable sensing action can acquire it;
3. the expected value of the resulting observation;
4. the selection/execution mechanism and subsequent state update.

The review also warns that reported evaluations often fail to distinguish direct improvement in acquired evidence from downstream task improvement and from the resource cost of acquiring the evidence. It lists matched passive/fixed, random/uniform and greedy/immediate-utility baselines, and calls for measurement of before/after evidence quality, downstream outcome, energy, time, computation, communication, sensor noise, actuation delay, packet loss, dynamic occlusion and stopping behaviour.

**Subterra consequence:** benchmark the whole observation-acquisition-decision cycle, not merely a scalar map-uncertainty reduction. Each proposal must state which uncertainty it is expected to resolve, which decisions could change, how the observation will be verified, the resource budget, the risk envelope and the stopping condition.

Source: https://doi.org/10.3390/s26206544  
Open article: https://pmc.ncbi.nlm.nih.gov/articles/PMC13708967/

### 1.2 When Information is Worth the Risk: Behavioral Valuation for Hazardous Robotic Exploration

Srivastava, Suresh, Nieto-Granda and Dames (arXiv:2609.10726, 9 September 2026) study hazardous exploration where acquiring information can itself cause robot failure, truncating all future sensing. Their design isolates the valuation layer: belief update, sensor model, physical risk model and candidate finite-horizon paths are held fixed while path objectives are changed. They compare Shannon information gain, Shannon information with linear risk penalties, and chance-constrained baselines against a risk-augmented behavioral-information objective using Prelec probability weighting.

This is a research result in failure-truncated grid-world environments, not planetary cave validation. It is nevertheless a directly relevant contemporary baseline and a useful warning: a path's expected information gain can be high while its probability of terminating the mission destroys future observation value.

**Subterra consequence:** do not equate EVSI with a complete mission planner. The current single-action EVSI kernel is a mathematical reference for decision value only. The next planner must evaluate a sequence of observations with survival-conditioned continuation value and hard reviewed constraints.

Source: https://arxiv.org/abs/2609.10726  
Full text: https://arxiv.org/html/2609.10726

## 2. Refine the planner into three layers

Keep three distinct mathematical responsibilities:

### Layer A — admissibility and hard constraints

External reviewed gates decide whether a measurement/trajectory is admissible at all. Examples include:
- robot safety, return reserve and a demonstrated retreat route;
- mission resource hard limits;
- planetary-protection restrictions;
- approved survey and sample permissions;
- sensor/platform operating envelope;
- data-custody and communications constraints.

A hard gate cannot be overridden by high information value, expected scientific reward, learned preference or probability weighting. This is a constraint boundary, not a scalar objective.

### Layer B — physical and epistemic prediction

Predict:
- outcome likelihoods and their calibration/domain validity;
- observation lineage and dependence;
- robot state, path feasibility and probability of failure;
- which future sensing opportunities remain available after each outcome;
- the posterior update, including unknown and contradictory hypotheses.

A failed or censored mission must not be treated as if all future observations were still obtainable.

### Layer C — decision valuation and search

Only after the first two layers have produced admissible candidates should the planner compare:
- expected value of sample information (EVSI);
- Shannon expected information gain;
- information gain per energy/time/data;
- risk-penalized information gain;
- chance-constrained planning;
- CVaR / tail-risk-aware planning;
- failure-truncated finite-horizon value;
- optional behavioural probability-weighting objectives as research baselines, never as safety probabilities.

The valuation layer may rank already-admissible candidates. It may not change the underlying physical hazard probability or turn a denied action into an allowed action.

## 3. Research architecture: survival-conditioned scientific planning

The next-generation planner should evolve from one-step EVSI into a bounded receding-horizon partially observed decision process:

`belief_t + mission context → candidate survey sequences → predicted observations/failures → posterior beliefs and future opportunities → expected decision loss → admissible sequence`

For a candidate sequence, model both informative outcomes and absorbing mission interruption. If the robot is lost, immobilized or forced to retreat before a later observation, all downstream sensing is unavailable. The expected value must include:
- value of measurements actually acquired before interruption;
- probability and consequence of failure;
- lost future evidence opportunities;
- mission data exfiltration and store-and-forward survival;
- cost of preserving a return route and relay network;
- decision quality after partial, censored or interrupted evidence.

Initially, do not solve an unrestricted POMDP. Use a small finite scenario tree, bounded action horizon, explicit terminal states, deterministic pruning and a replayable reference implementation. Compare it against the one-step EVSI kernel, greedy EVSI, entropy-greedy, frontier/coverage, risk-penalized information gain and a small exact dynamic-programming oracle on tiny problems.

## 4. Required benchmark extension

Add these comparisons to Subterra Intelligence Benchmarks:

| Baseline / condition | What it isolates |
|---|---|
| Passive / fixed observation | Whether active selection adds value |
| Random / uniform action | Whether the policy beats unstructured sensing |
| Greedy frontier / coverage | Whether the policy improves over geometrical exploration |
| Greedy Shannon information gain | Whether less uncertainty is actually decision-useful |
| Shannon information gain per resource cost | Whether simple efficiency explains the result |
| Shannon information gain minus expected risk | Risk-penalty baseline |
| Chance-constrained information planning | Hard probability-of-failure baseline |
| CVaR-constrained / tail-risk baseline | Worst-case loss sensitivity |
| One-step EVSI | Current reference kernel |
| Greedy EVSI | Repeated one-step replanning baseline |
| Finite-horizon survival-conditioned planner | Proposed research successor |
| Small exact dynamic-programming oracle | Upper/lower bound on tiny frozen cases only |

Measure:
- mission/decision regret after complete and interrupted episodes;
- time and distance to useful, verified evidence;
- probability of reaching the next planned measurement;
- evidence acquired before termination;
- data actually returned to the surface;
- risk exposure and robot loss;
- return-route / reserve integrity;
- energy, time, data, compute and communication;
- calibration and abstention;
- quality of the final scientific decision.

For fair attribution, keep the same belief update, sensor model, physical risk model and feasible candidate set while changing only the valuation policy in the first experiment. In a second experiment, allow the full system to vary its belief and model; report that as integrated performance, not a clean valuation ablation.

## 5. Immediate implementation sequence

1. Preserve the current EVSI crate as the single-step mathematical reference, after Rust compilation and adversarial test execution.
2. Add a separate finite survival-tree crate or module; do not mix finite-horizon policy search into the one-step oracle.
3. Define explicit terminal outcomes: observation obtained, safe retreat, relay loss with store-and-forward, immobilized, mission aborted, robot lost, and out-of-domain.
4. Add exact tiny-case oracle and hand-computable cases before heuristics.
5. Add paired-seed, same-scenario comparisons against entropy, frontier, risk-penalized, chance-constrained and greedy-EVSI baselines.
6. Test common-mode sensor failure, false confidence, correlated sensors, delayed/censored observations, relay loss, stale maps, blocked retreat and loss of future sensing.
7. Integrate proposals with the existing Symthaea mission executive only through bounded work-order requests; the independent safety/authority chain remains unchanged.

## 6. Evidence ceiling

The related-work synthesis is not a novelty, safety, or superiority proof. Expected value of information, active perception, risk-aware exploration, chance constraints, CVaR and behavioural risk valuation all have prior work. Subterra's defensible contribution must be shown by a reproducible integration across planetary scientific hypotheses, provenance-aware evidence, mission constraints, survival-conditioned measurement planning, cross-site transfer and verifiable data return.

The current EVSI source is AUTHORED and its worked mathematical example was independently recalculated. Rust compilation, Rust tests, CI, robot-in-the-loop, analogue and planetary validation remain unperformed.
