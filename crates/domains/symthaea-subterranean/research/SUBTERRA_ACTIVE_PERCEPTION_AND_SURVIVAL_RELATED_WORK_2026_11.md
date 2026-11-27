# Subterra active-perception and hazardous-exploration literature update

**Cut-off:** 2026-11-16  
**Status:** Focused related-work update; not a systematic review

## 1. Why a closed-loop benchmark is required

Li et al., *Active Visual Perception: Opportunities and Challenges*, Sensors 26(20), 6544 (published 15 October 2026), define active visual perception around the connection between task-related evidence needs, actions that change subsequent observations, the value of those observations, selection/execution, and state update.

The review emphasizes that a fair active-perception evaluation should compare:
- passive or fixed observation;
- random and uniform observation selection;
- greedy/immediate utility;
- oracle or privileged-information bounds where feasible;
- observation-level improvement before/after an action;
- downstream task performance;
- motion, computation, communication, energy, time and safety costs;
- calibration, reachability, execution failures and stopping behaviour.

The review warns that final task scores alone do not isolate the effect of the sensing policy. A stronger vision backbone, different candidate viewpoints, more computation or privileged information can masquerade as an active-perception gain.

**Subterra consequence:** use a matched perception backbone, initial belief, candidate action set, likelihood model, loss matrix, resource budget and stopping condition whenever the experiment is meant to compare decision policies. Report the observation evidence change and the downstream mission decision separately.

Source: https://doi.org/10.3390/s26206544  
Open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC13708967/

## 2. Hazard truncates future information

Srivastava, Suresh, Nieto-Granda and Dames, *When Information is Worth the Risk: Behavioral Valuation for Hazardous Robotic Exploration* (arXiv:2609.10726, 9 September 2026), isolate the valuation layer in a finite-horizon hazardous-exploration setting. They hold the belief update, sensor model, physical risk model and candidate planner fixed while changing how information and risk are valued. Their abstract reports that information-only policies can expose robots to failures that truncate future sensing, and compares Shannon information against risk-aware and behavioral valuation objectives.

Important boundary: this paper's grid-world experiments are not evidence of planetary cave performance. Its methodological contribution is directly relevant, but our claims must remain at the tested evidence tier.

**Subterra consequence:** future measurements cannot retain expected value after an absorbing interruption outcome. A planner must explicitly model which future observations remain reachable after each outcome.

Source: https://arxiv.org/abs/2609.10726

## 3. Why we implemented two reference layers

### Layer 1 — One-step EVSI

`subterra-voi-kernel` estimates the expected reduction in terminal decision loss after one measurement, then subtracts declared resource shadow prices. It distinguishes decision value from entropy reduction. It is the small, hand-checkable mathematical reference.

### Layer 2 — Bounded survival-conditioned policy tree

`subterra-survival-planner` adds:
- observation branches with posterior updates and contingent continuation;
- safe-retreat terminal branches;
- mission-interruption terminal branches;
- robot-loss terminal branches;
- relay-loss / store-and-forward terminal branches;
- branch-local remaining resource budgets;
- separate hard limits on mission interruption and robot loss probability;
- stop-now decisions at every node;
- deterministic action order and bounded search-node expansion.

The survival planner is not simply EVSI with an added risk penalty. It changes the future opportunity structure: an absorbing event terminates the branch, and no subsequent measurement can be counted.

It is still a finite, single-agent reference. It does not yet model continuous geometry, route feasibility, actual sensor calibration, common-mode failure automatically, multi-agent coordination, dynamic relay-buffer state, or real physical failure rates. The current static hypothesis / repeated-likelihood formulation assumes conditional independence unless the caller explicitly expands the hypothesis state to represent common causes.

## 4. Competitive evaluation plan

The initial controlled experiment should hold belief update, sensor outcome model, terminal loss, feasible action set, budget and risk limits fixed while comparing:
1. stop-now;
2. passive/fixed sequence;
3. random;
4. frontier / coverage greedy;
5. Shannon entropy greedy;
6. information gain per resource;
7. risk-penalized information gain;
8. chance-constrained planning;
9. CVaR / tail-risk planning;
10. one-step EVSI;
11. receding-horizon greedy EVSI;
12. survival-conditioned finite-horizon planning;
13. exact dynamic programming on tiny cases only.

Primary metrics:
- terminal decision loss and realised regret;
- verified, decision-relevant evidence acquired;
- verified evidence returned to the surface;
- conditional probability of reaching the next measurement;
- mission-interruption / robot-loss probability;
- science opportunities lost to terminal events;
- energy, time, data, exposure, compute and communication;
- likelihood calibration and open-world abstention.

A proposed target is at least 15% more verified, decision-relevant evidence returned under equal expected resources, without relaxing the frozen risk ceilings, and at least 20% lower terminal decision regret than matched passive / entropy / frontier baselines. These are programme hypotheses, not current results or universal standards.

## 5. Further paradigm shift: plan evidence return, not just evidence acquisition

Active-perception evaluations should distinguish:
- measurement proposed;
- measurement action executed;
- observation acquired;
- observation integrity and calibration verified;
- observation persisted locally;
- observation delivered to the surface or an independently verified store-and-forward custodian;
- observation admitted to the scientific belief update.

These are distinct events. Counting a command as a completed observation, or a buffered packet as delivered data, would overstate scientific return.

Subterra should therefore evolve toward a **science-return contract**: a policy only receives full returned-evidence credit after a verifier-bound observation receipt and data-delivery disposition are present. This should compose with the existing Symthaea audit/evidence interfaces, Symtropy replay, and Mycelix custody records, while keeping the hard real-time robot control path independent from distributed consensus.

## 6. References

1. Li, Y. et al. (2026), *Active Visual Perception: Opportunities and Challenges*, Sensors 26(20), 6544. https://doi.org/10.3390/s26206544
2. Srivastava, A. K. et al. (2026), *When Information is Worth the Risk: Behavioral Valuation for Hazardous Robotic Exploration*. https://arxiv.org/abs/2609.10726
3. Kim, H. et al. (2026), *SKiD-SLAM: Resource-Aware Distributed Multi-Robot LiDAR SLAM for Planetary Missions with Field Deployments at Analog Sites*. https://arxiv.org/abs/2610.38141
4. Swinton, S. et al. (2026), *The Evolution of Autonomous Systems for Planetary Cave Exploration: A Review*. https://doi.org/10.1002/rob.70233

## 7. Evidence state

This is a focused literature synthesis. The two Rust reference planners are AUTHORED. Their Rust tests are AUTHORED but NOT EXECUTED. No benchmark result, robot-in-the-loop validation, terrestrial analogue result, flight result or planetary deployment is claimed.
