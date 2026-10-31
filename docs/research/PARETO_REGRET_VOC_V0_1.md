# Pareto-Regret Value of Computation v0.1

**Status:** Exact 2-objective finite-posterior reference oracle and Rust test declarations are on branch research/pareto-regret-voc-v0.1; compile, formatting, execution, independent verifier agreement and benchmark replay remain outstanding.

**Tracking:** [#9389](https://github.com/Luminous-Dynamics/symthaea/issues/9389). Extends rather than replaces scalar decision-focused VoC [#9350](https://github.com/Luminous-Dynamics/symthaea/pull/9350).

## 1. Why decouple objectives

A NAS architecture is not a single score. It can trade validation quality against inference latency, energy, memory, throughput, robustness and hardware compatibility. A weighted sum makes the answer depend on preference weights and can conceal infeasible candidates. Some objective measurements are cheap while others require training or deployment on a particular device.

The next decision is therefore not just *which architecture should we evaluate?* It is *which architecture, which objective, on which device, at which fidelity, and at what marginal cost should we measure next?*

Version 0.1 implements a small exact reference kernel for two normalized maximize-is-better objectives. The calling evaluator must transform direction and normalization using a pre-registered scale.

## 2. Bayes Pareto hypervolume regret

A posterior particle contains a joint vector for every candidate. Joint particles preserve dependencies such as accuracy/latency anti-correlation.

For a frozen lower-left reference point r and a set S of recommended candidates, let HV(S; theta, r) be the dominated 2-D hypervolume of the feasible points in S under posterior state theta.

For recommendation budget K:

R_HV,K(p) = E[HV(all candidates; theta, r)] - max over S with 1 <= |S| <= K of E[HV(S; theta, r)]

The first term is the oracle frontier value after learning the true state; the second is the best fixed recommendation set available now. This distinction matters: choosing a Pareto set based on the posterior-mean vector is not generally equivalent to maximizing expected set hypervolume.

For measurement action a with outcomes o:

VOI_HV,K(a) = R_HV,K(p) - sum over o of P(o|a,p) * R_HV,K(p|o,a)

Action value per unit is the expected regret reduction divided by the declared incremental resource cost. This module returns the value and resource-normalized value; it does not yet select across heterogeneous actions, protect reserves, or enforce the run-wide pruning-risk ledger.

## 3. Hard feasibility, not penalty weights

The reference point and hard minimum-utility constraints are frozen separately. A posterior realization that violates either minimum utility is infeasible and contributes no dominated hypervolume. It is not assigned a soft penalty that could be traded for quality on another axis.

This is a two-objective utility-space representation. Production systems must bind the objective IDs, original direction, normalization transforms, units, hardware/device profile, measurement protocol and hard constraints in an outer manifest. A latency utility on one device cannot be compared with another device's latency utility unless the normalization and device context are explicitly compatible.

## 4. Exact 2-D oracle

**hypervolume_2d** computes exact rectangle-union area up to floating point arithmetic, tolerating duplicate/dominated points and ignoring points that do not dominate the reference point.

**nondominated_indices_2d** exposes the Pareto frontier with stable duplicate handling.

**evaluate_pareto_decision** enumerates every non-empty recommendation subset of size at most K and returns the exact finite-posterior Bayes decision. The exhaustive oracle deliberately refuses more than 16 candidates. It is a correctness target for approximations, not a scalable NAS algorithm.

**expected_pareto_value_of_information** updates posterior particle weights with a normalized discrete action-outcome likelihood, computes exact posterior regret for every non-zero-probability outcome, and reports expected regret reduction per cost unit. Invalid likelihood masses or schema dimensions fail closed.

## 5. Small exact example

Three architectures trade off two objectives. In posterior particle p1, architecture A is best and dominates B/C; in p2, architecture B is best and dominates A/C. Both particles have probability 0.5. With K=1:

- Prior oracle expected hypervolume: 0.475.
- Best fixed recommendation (A or B): 0.2975 expected hypervolume.
- Prior Bayes Pareto regret: 0.1775.
- An observation that perfectly reveals the particle leaves zero posterior regret.
- Expected HV-regret reduction is therefore 0.1775; with action cost 2 units, value is 0.08875 regret units per resource unit.

An independent Python mathematical mirror reproduced those values and the exact 2-D area fixtures. This is a reference calculation, not a Rust build or benchmark result.

## 6. Test inventory and limits

The source has ten Rust test declarations covering:

1. single-point rectangle area;
2. overlap union without double counting;
3. duplicate/dominated point invariance;
4. deterministic non-dominated set;
5. complementary Pareto recommendation portfolio;
6. hard feasibility filtering;
7. zero value from an uninformative action;
8. positive value from a state-revealing action and cost normalization;
9. invalid action likelihood mass rejection;
10. refusal to approximate a cohort larger than the exact 16-candidate cap.

The tests have not been compiled or run in the current environment; cargo, rustc and rustfmt are unavailable. The Python mirror only validates the mathematical fixtures.

## 7. Next implementation layers

1. **Objective registry:** typed metric IDs, units, maximize/minimize transform, normalization source, objective measurement protocol and device identity.
2. **Partial observation state:** track objective-by-candidate missingness and prohibit missing-as-zero; allow distinct costs and fidelities for training, latency, energy, memory and robustness.
3. **Scalable acquisition:** candidate-set generation/pruning followed by approximate qEHVI, Pareto-front Monte Carlo or other bounded estimator. Compare against exact oracle on small random fixtures, and report Monte Carlo seed, standard error and ESS.
4. **Planner integration:** action-level objective selection, hard budget/reserve accounting and mandatory receipt qualification. No unqualified measurement model may drive operational pruning.
5. **Objective-specific qualification:** held-out calibration for each metric/device, joint constraint-feasibility calibration, Pareto-front coverage, hypervolume regret and resource-cost error.
6. **Independent verifier:** implement 2-D HV, posterior update and receipt verification separately from the producer and require matching receipts.

## 8. Competitive comparison protocol

Compare at equal total measured cost, and also publish each objective-specific measurement count.

Baselines:
- random, NSGA-II, SMS-EMOA and MO-CMA-ES;
- qEHVI/EHVI and LaMOO+qEHVI;
- FlexiBO / objective-decoupled cost-aware acquisition;
- preference-conditioned multi-objective differentiable NAS;
- scalarized VoC (diagnostic), versus Pareto-regret VoC;
- objective-coupled versus objective-decoupled action scheduling.

Research references:
- [LaMOO: Multi-Objective NAS by Learning Search Space Partitions, JMLR 2024](https://www.jmlr.org/papers/v25/23-1013.html)
- [Multi-objective Differentiable NAS, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/43d7bc009cf5171e7af77a91ee4bb890-Abstract-Conference.html)
- [FlexiBO, AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/30600)
- [On the Complexity of Expected Hypervolume Improvement, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33781)
- [Pareto-NRPA, 2026](https://journals.sagepub.com/doi/10.3233/FAIA251394)

LaMOO+qEHVI reports 225% sample-efficiency improvement over qEHVI alone on NAS-Bench-201; FlexiBO reports 4.8–12.4% lower hypervolume error than compared baselines at matched total budget. The multi-objective differentiable NAS paper studies 3 objectives across up to 19 devices in one conditional search. These are published results under their respective protocols, not directly interchangeable scores. Hypervolume also depends on objective normalization and the reference point; a numerical HV target copied across papers is not meaningful without those definitions.

## 9. Initial target

For the first reproducible 2-objective benchmark campaign:

- At matched total objective-measurement cost, target at least 10% lower normalized hypervolume regret than the strongest reproducible eligible baseline, with paired-search-seed confidence interval excluding zero.
- Reach a fixed hypervolume/IGD+ threshold using at least 10% fewer expensive objective measurements.
- Maintain or improve hard feasibility rate; target at least 95% feasibility under every predeclared deployment constraint.
- Report Pareto coverage, spread, IGD+, epsilon indicator, hypervolume, objective-specific action counts, elapsed cost and worst-task/worst-seed performance.
- Use at least 30 independent search seeds for an initial estimate, then determine final sample size by power analysis and account for task/device clustering.

These are ambitious research targets, not results from the current code.

## 10. Claim ceiling

The module is an exact two-objective, finite-posterior reference oracle for at most 16 candidates. It does not demonstrate scalable multi-objective NAS, calibrated objective forecasts, broad hardware transfer or better-than-SOTA results. It does not include an independent proof of data lineage or a mandatory qualification token. Those require the issue #9388 and benchmark/data work #9385, plus objective registry and independent-verifier follow-ups.
