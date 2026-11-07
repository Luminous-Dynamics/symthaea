# Active model-discrepancy probe benchmark V1

**Tracking:** #10880 — SCI-CONNECTOMICS-097  
**Composes:** #10670, #10779, #10781, #10762, #10831, #10752, #10754 and #10490.  
**Status:** protocol, finite synthetic contract, and independent standard-library oracle; no biological result is claimed.

## 1. Objective

Choose a feasible measurement or intervention that can reveal a response-model failure which the currently admitted models do not explain, while preserving the ability to resolve the frozen scientific query.

This is not another query-value optimizer. The benchmark treats two questions separately:

- Does the experiment help decide the scientific query?
- Does the experiment distinguish the baseline response-model family from independently generated discrepancy challengers?

An action may be excellent for one and useless for the other. The policy must expose that trade-off rather than hiding it in a weighted score.

## 2. Ownership boundaries

- #10670 owns query-decision value and feasible intervention selection.
- #10779 owns append-only response-model challenge, robust-loss monotonicity and support-based identifiability withdrawal.
- #10781 owns held-out trajectory calibration and discrepancy diagnostics.
- #10762 / #10831 own coherent static-model sequential and stochastic risk-bounded policy search.
- #10752 / #10754 own cross-animal transport and C. elegans data custody.
- #10490 owns the general model-discrimination, representativeness and randomized audit-floor architecture.

This benchmark owns the missing acquisition question: which action should be run to expose a discrepancy, given the current model family and an independently declared challenge family?

## 3. Separate objective vector

For each feasible action a, report:

1. **Robust query value per cost.** Let R0 be the frozen prior Bayes query risk, R(a,m) the expected posterior query loss under model m, and M the union of baseline and admitted challenge models:

       Q(a) = max(0, R0 - max_m R(a,m)) / cost(a)

2. **Robust model-discrimination separation per cost.** For the baseline family M0 and challenge family C, marginalize each model's outcome distribution over the same frozen biological-world prior. Define:

       D(a) = min over m0 in M0, m1 in C of TV(P_m0(Y|a), P_m1(Y|a))
       S(a) = D(a) / cost(a)

   This is a finite-family lower bound on pairwise predictive separation, not a guarantee against unenumerated mechanisms.

3. **Model mutual information / EDI-style diagnostics.** Report these only with an explicit model-prior or design assumption. They are diagnostic comparators, not interchangeable with minimum pairwise total variation or query decision value.

4. **Representativeness and audit coverage.** Report selected and unselected units by animal, target neuron, region, state/context and source-quality stratum. A policy must retain the randomized audit floor and mandatory sentinel coverage defined by its protocol.

5. **All-in cost and feasibility.** Include acquisition, preparation, stimulation, imaging, inference, compute, reviewer, adjudication, failed assay and rework costs. Disallowed, inaccessible or unassessed actions are excluded before optimization.

Do not scalarize Q, S, representativeness and cost by default. Return the Pareto frontier. A later operational policy may use lexicographic hard gates, but all thresholds and orderings must be preregistered.

V1's executable synthetic oracle computes Q, minimum pairwise TV, model-mutual-information diagnostic and the Q/TV/cost Pareto frontier. It does not yet estimate population representativeness, calibrate a sequential false-alarm process, or establish challenge-family recall. Those require source-bound independent-unit data and later integrated tranches; their absence cannot be interpreted as a pass.

## 4. Finite query-risk and discrimination semantics

For binary query q(w), world prior p(w), and 0–1 loss, the expected posterior query loss under a model is:

       R(a,m) = sum_y min(P(q=0,y|a,m), P(q=1,y|a,m))

For asymmetric loss, use the exact frozen loss table rather than 0–1 loss. Model discrimination uses the same world prior across baseline/challenge models; do not alter the world set or prior to make a probe look more discriminative.

A baseline/challenger pair's marginal predictive distributions are compared over a common declared outcome alphabet. Total variation is exact on finite categorical outcomes:

       TV(P,Q) = 0.5 * sum_y abs(P(y)-Q(y))

For multiple models, use the minimum TV across every baseline × challenger pair. This intentionally conservative score is zero whenever at least one declared pair is observationally indistinguishable.

A positive D(a) establishes separation only for the supplied finite challenge family. It does not prove model adequacy or completeness. A zero D(a) means that action cannot distinguish at least one declared baseline/challenge pair under this metric; it does not prove the mechanisms are identical in every possible experiment.

## 5. Selection and stopping

- Query-only choice: maximize Q(a) over feasible actions.
- Discrepancy-only choice: maximize S(a) over feasible actions.
- Joint policy: retain the non-dominated feasible actions in the (Q, S, cost, audit-coverage) frontier. Do not discard a discrepancy probe merely because its immediate query value is low.
- If every feasible action has zero robust discrepancy separation, return NO_FEASIBLE_DISCREPANCY_PROBE / NOT_IDENTIFIABLE. Do not report model adequacy.
- If no query-useful action exists, preserve that result separately from discrepancy probe availability.
- An unmodeled observation outside all admitted support is MODEL_DISCREPANCY_DETECTED / NOT_IDENTIFIABLE; do not force a zero-likelihood Bayesian update under the old family.
- For sequential policies, preserve the one-static-model-per-campaign semantics from #10762/#10831. The hidden model ID may never be observed by the deployed policy or switched opportunistically between branches.

## 6. Discovery and confirmation separation

Use separate partitions for:
- baseline model fitting;
- independent challenger generation / challenge discovery;
- final confirmation of the selected discrepancy probe or sequential policy.

If a policy repeatedly inspects confirmation outcomes, use a theorem-matched sequential procedure (e-process, confidence sequence, or predeclared alpha spending) and publish the exact assumptions. A fixed-look p-value is not automatically valid after adaptive action selection or optional stopping.

The policy must preserve a randomized audit floor across the frozen target frame. Report action propensities, selected and unselected unit counts, stratum coverage, and all failed/inaccessible measurements. The discovery process must not silently shrink the population to only regions where the current models already have support.

## 7. Synthetic known-answer fixture

The fixture has two biological worlds, one baseline response model, one admitted challenge model and three candidate actions.

- Action A resolves the query perfectly under both response models, but the baseline and challenger have identical marginal outcome distributions. It has positive Q and zero D.
- Action B has zero robust query value because the baseline model maps both query worlds to the same outcome. The challenger changes the predicted outcome law, so B has positive D and positive model-discrimination mutual information.
- Action C is a perfect query probe but explicitly disallowed. It must not enter either selection or the Pareto frontier.

The expected result is:
- query-only policy selects A;
- discrepancy-only policy selects B;
- Pareto frontier contains both A and B;
- disallowed C is never selected;
- no model-adequacy claim is produced from the finite fixture.

This proves that query resolution and model-discrepancy detection are not the same optimization objective.

## 8. Required adversarial extensions

1. Query-useful / discrepancy-blind action.
2. Query-irrelevant discrepancy that changes response trajectories but not the frozen query.
3. Decision-critical discrepancy that invalidates the former query-optimal action.
4. No feasible discrepancy-discriminating action: explicit non-identifiability.
5. A model family with shared raw-volume, simulator, training, preprocessing or reviewer ancestry.
6. Adaptive selection that starves a low-frequency animal/region stratum.
7. Repeated-look false alarms under adaptive probe selection.
8. Discovery/confirmation leakage.
9. A rectangular adversary that switches model identity across stages despite a static biological mechanism.
10. An observation outside every declared outcome support.
11. An infeasible or unsafe action with attractive predicted value.
12. Missing, failed, inaccessible and non-response outcomes silently treated as negatives.
13. A probe that is discriminative only because its predictions were calibrated or tuned on the confirmation partition.
14. Prior/model weights changed after outcome inspection.
15. Model discrimination driven by a nuisance feature that is irrelevant to the biological query; report it as a discrepancy signal without claiming biological mechanism impact.

## 9. Comparators

Under identical data, world prior, challenge family, feasible actions, costs and budgets:

- #10670 expected / robust query-decision VOI;
- nominal EIG / mutual information;
- expected discriminatory information (EDI) and expected general information gain (EGIG);
- robust EIG under a preregistered prior ambiguity set (REIG);
- representativeness-aware acquisition inspired by R-IDeA;
- expert-ranked and random feasible probes;
- fixed randomized sentinel/audit allocation;
- exhaustive exact oracle on bounded finite cases.

Important literature boundaries:
- Catanach & Das introduce EDI / EGIG as model-discrepancy-aware experiment-design criteria, not as proof of biological model completeness: https://arxiv.org/abs/2304.07949
- Go & Isaac's REIG addresses prior ambiguity sets; it does not automatically cover omitted observation kernels: https://proceedings.mlr.press/v180/go22a.html
- Callahan & Catanach (AISTATS 2026) show that informative experiments can amplify inferential error under model and inferential misspecification: https://proceedings.mlr.press/v300/callahan26a.html
- Tang, Sloman & Kaski (AISTATS 2026) motivate representativeness and error de-amplification under shift: https://proceedings.mlr.press/v300/tang26d.html

These methods optimize different objectives. Report separate scorecards and claim boundaries.

## 10. Metrics and promotion targets

Report:
- Q(a), D(a), S(a), model-mutual-information diagnostic and cost for every action;
- exact Pareto frontier and regret against its finite oracle;
- false model-adequacy certification rate;
- challenge-family detection power and false-alarm rate at independent-unit level;
- power-versus-cost and detection-delay curves;
- query expected/worst-case loss and query regret per all-in cost;
- audit coverage and worst-stratum omission rate;
- calibration, proper predictive scores, sharpness and out-of-support frequency;
- adaptive-test type-I error / anytime-valid coverage;
- action propensities, selected/unselected counts, abstentions and missingness;
- all acquisition, preparation, compute, reviewer and adjudication costs.

Synthetic targets:
- 100% exact metric and Pareto-frontier agreement with independent finite oracle;
- zero false model-adequacy passes after a valid known-answer challenge;
- zero false query-resolution certificates while an opposing query world remains compatible;
- zero confirmation leakage and zero hidden-model branches;
- zero infeasible actions in returned selections;
- 100% correct no-feasible-probe / not-identifiable behavior when all feasible actions have zero D.

Do not set biological superiority percentages from toy examples. Determine those only after a source-bound pilot estimates independent-unit variance.

## 11. Biological staging

1. Synthetic finite exact oracle and mutation arena.
2. C. elegans only after #10754 reconciles source custody and denominators; split at animal level. The Randi et al. atlas covers 23,433 stimulated/recorded neuron pairs across 113 animals and captures effective pathways, including extrasynaptic signalling, not direct synaptic ground truth: https://www.nature.com/articles/s41586-023-06683-4
3. Drosophila walking CPG only with exact source/model/behavior custody from #9934; keep computational predictions and optogenetic measurements distinct: https://www.nature.com/articles/s41586-026-11195-y
4. MICrONS for registered structure/function and stimulus-conditioned probes only; do not re-label passive/stimulus data as optogenetic intervention.
5. Prospective work only with qualified collaborators, approvals and independent confirmation cohorts.

## 12. Claim ceiling

A passing synthetic campaign establishes only that the finite probe metrics, exact ranking and Pareto frontier are recomputed correctly for the supplied challenge family and action universe. It does not prove global model completeness, biological truth of a challenger, independence from opaque ancestry, direct synaptic causation or in-vivo efficacy.
