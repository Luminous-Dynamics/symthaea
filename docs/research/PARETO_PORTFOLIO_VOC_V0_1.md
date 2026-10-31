# Portfolio-Class Pareto Regret and Objective-Decoupled VoC v0.1

**Status:** Bounded portfolio evaluator, action-level objective VoC, common-random-number QMC cross-fitting and provenance digests are committed on `research/portfolio-class-pareto-voc-v0.1`. Rust compilation and test execution, independent verifier agreement, and real benchmark replay remain outstanding.

**Tracking:** [#9415](https://github.com/Luminous-Dynamics/symthaea/issues/9415). Uses the exact 2-D oracle [#9391](https://github.com/Luminous-Dynamics/symthaea/pull/9391), the randomized QMC kernel [#9414](https://github.com/Luminous-Dynamics/symthaea/pull/9414), and hardware objective registry [#9412](https://github.com/Luminous-Dynamics/symthaea/pull/9412).

## 1. Why portfolio-class regret

Enumerating every candidate subset is exponential and unsuitable for large NAS spaces. Instead, an external search policy provides a bounded family of candidate portfolios, such as:
- a non-dominated archive from NSGA-II/SMS-EMOA;
- a qEHVI/HVKG acquisition sequence;
- preference-conditioned candidate sets;
- a diversity-preserving Pareto archive.

The evaluator computes the posterior expected hypervolume for each supplied portfolio, selects the best expected-hypervolume portfolio, and measures its regret against the all-feasible-candidate hypervolume oracle.

This quantity is explicitly **portfolio-class regret**. It is exact only relative to the supplied portfolio family. It equals unrestricted fixed-size-K Bayes regret only when the family includes every allowed subset. If the generator omits a useful set, the regret may be higher. This is an intentional and explicit trade-off: bounded decision class rather than combinatorial enumeration.

## 2. Joint posterior and feasibility

The posterior stores:
- sorted candidate IDs;
- sorted objective IDs defining the exact coordinate order;
- objective-registry SHA-256 binding value units, directions, normalization, protocols and acquisition-cost models;
- exact per-candidate architecture configuration SHA-256 digests;
- hard device/profile feasibility flags;
- joint posterior particles, each containing a probability and one optional objective vector per candidate.

Every feasible candidate must have a complete normalized objective vector in every particle. A hard-infeasible candidate may have no vector; no numeric latency, energy, accuracy or memory value is fabricated to stand in for infeasibility.

Every vector uses the same frozen objective order and normalized maximize-is-better [0,1] scale. Hard minimum utility constraints are applied before hypervolume. A candidate below any hard floor is infeasible for this deployment profile, rather than receiving a soft penalty that might be traded against another objective.

## 3. QMC sample splitting and common random numbers

A single immutable `HypervolumeQmcDesign` supplies independently shifted Halton scrambles. Portfolio selection uses the first half; held-out validation uses the second half. At least 16 total scrambles are required, providing at least eight independent replicates for each side.

This is a cross-fitting step:
- choose the best supplied portfolio using only selection scrambles;
- estimate the all-candidate oracle and selected portfolio on held-out validation scrambles;
- calculate held-out regret on paired replicate values.

This avoids evaluating the selected winner on the same numerical samples used to choose it. It does not eliminate all adaptive-selection bias from repeated external candidate-family tuning; the portfolio family, objective normalization, reference point, K and policy generator must be frozen before the validation split is inspected.

For every possible action outcome, the same immutable design and held-out replicates are reused. The expected posterior-regret value is computed per replicate across outcomes, then subtracted from the current regret replicate. This preserves common-random-number covariance rather than adding independent estimator errors.

The reported standard error/normal interval describes between-scramble numerical variability. It is not posterior uncertainty, hardware measurement noise, task-transfer uncertainty, or a formal finite-sample frequentist guarantee. A production policy must keep those error sources separate and abstain when action ranking is not robust to QMC error.

## 4. Objective-decoupled measurement actions

A measurement action binds:
- action and candidate IDs;
- exact candidate configuration SHA-256;
- exact objective-registry SHA-256, which binds objective semantics and acquisition-cost model;
- objective ID;
- optional hardware profile SHA-256;
- measurement-protocol SHA-256;
- acquisition resource unit and positive incremental cost;
- outcome-likelihood-model digest;
- mutually exclusive labelled outcomes and one likelihood per posterior particle.

Outcome likelihood arrays must normalize to one for each particle. Outcome IDs are unique. The action outcome taxonomy is canonicalized by label when computing the digest. Registry, sorted objective-coordinate order, objective, protocol, hardware profile, resource unit, candidate configuration and exact likelihood model must all match the current posterior/registry. The posterior cannot be reused under a changed unit, normalization or acquisition-cost definition.

For each non-zero-probability outcome, particle weights are updated by Bayes' rule and the portfolio-class decision is recomputed. The evaluator returns:
- current held-out portfolio-class regret;
- expected posterior regret;
- estimated regret reduction (which can be negative due to numerical error);
- between-scramble standard error and normal interval for the regret-reduction estimate;
- regret reduction per declared acquisition-cost unit;
- exact action likelihood and registry digests.

This version is a diagnostic action evaluator, not an operational scheduler: it does not choose between many actions, enforce hard total budgets or protected reserves, or enforce the run-wide pruning risk ledger. Action likelihoods must separately pass empirical qualification, and #9388 still needs to make verified qualification a mandatory capability.

## 5. Replay identity

Decision outputs bind:
- exact joint posterior digest (including objective-coordinate order, objective registry and candidate configuration digests);
- canonical portfolio-family/config digest;
- immutable full QMC design digest;
- exact disjoint selection and validation scramble subset digests.

Action VoC outputs additionally expose the exact action likelihood-model digest and objective-registry digest. These identities make it possible for a later independent receipt verifier to reconstruct exactly which posterior, portfolio family, numerical design, objective semantics and action model produced the reported value.

Digests establish byte identity, not truthful data acquisition. Source artifact, candidate mapping, task split, device identity and raw measurement provenance still require independent verification.

## 6. Small example and reference fixture

Two posterior states each have probability 0.5. In state p1, candidate A dominates the other candidate choices; in p2, candidate B dominates. The frozen portfolio family is {A-only, B-only}, K=1.

Each state has oracle HV 0.9025. Each fixed portfolio has expected HV 0.47125. Therefore prior portfolio-class regret is 0.43125. A deterministic measurement that reveals the state allows the correct portfolio to be selected and reduces posterior regret to zero. This gives expected regret reduction 0.43125 before acquisition cost.

The code tests this as a QMC approximate comparison with tolerances; it does not claim a Rust execution. The same fixture should also be checked by an independent exact 2-D implementation to separate posterior/decision math from QMC numerical error.

## 7. Test inventory

Sixteen authored Rust tests currently cover:
1. portfolio regret against all-candidate oracle;
2. explicit hard infeasibility without fabricated objective vector;
3. hard minimum utility filtering;
4. disjoint selection and validation scrambles;
5. canonical portfolio and candidate-index ordering;
6. informative objective measurement reduces portfolio regret;
7. uninformative measurement has near-zero regret reduction;
8. action-likelihood digest tampering;
9. wrong acquisition-cost unit;
10. invalid portfolio indices/dimensions;
11. candidate-configuration mismatch;
12. objective-registry mismatch;
13. replay identity binding for posterior/portfolio/QMC splits;
14. rejection of too-small selection/validation scramble split;
15. rejection of candidate-configuration and objective-registry replay mismatch;
16. rejection of objective-column order mismatch and digest-bound replay.

The environment has no `cargo`, `rustc` or `rustfmt`; these are authored test declarations only until compiled, formatted and executed.

## 8. Competitive baselines and target metrics

Baselines under equal objective-query and acquisition-cost budgets:
- random and fixed objective schedules;
- NSGA-II/SMS-EMOA Pareto archives;
- qEHVI/qNEHVI;
- HVKG with decoupled objective evaluation and heterogeneous costs;
- LaMOO+qEHVI;
- FlexiBO-style objective-cost decoupling;
- scalarized VoC;
- all-subset exact Bayes portfolio evaluation on small candidate sets.

BoTorch's current multi-objective Bayesian optimization documentation explicitly includes cost-aware decoupled evaluation and HVKG, which is a meaningful industry-grade baseline for objective-specific measurements. Do not compare only the scalar acquisition score: compare final portfolio regret, hypervolume/IGD+, objective query counts, cost, estimator uncertainty and feasibility.

Initial research targets (not results):
- at least 10% lower normalized portfolio-class regret than the strongest reproducible baseline at matched total acquisition cost, with paired-seed confidence interval excluding zero;
- at least 10% fewer expensive objective measurements to reach a pre-registered hypervolume/IGD+ threshold;
- no increase in hard infeasibility violations;
- measured quality, energy, latency and memory trade-offs reproduced across at least two hardware profiles before device-transfer claims;
- independent replay of posterior, portfolio-family, QMC-design, action-model and objective-registry digests.

## 9. Claim ceiling

This version is a bounded portfolio-family regret evaluator plus a diagnostic objective-measurement VOI computation. It does not prove unrestricted Bayes optimality, global Pareto optimality, calibrated likelihoods, robust task/device transfer or better-than-SOTA performance. It does not yet enforce mandatory qualification receipts, total budget/reserves, or run-wide pruning confidence. Those are explicit downstream requirements, not implied by the current code.
