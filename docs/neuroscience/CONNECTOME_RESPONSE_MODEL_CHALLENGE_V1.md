# Connectome response-model adequacy and expansion benchmark V1

**Tracking:** #10774 — SCI-CONNECTOMICS-094
**Composes:** #10675, #10682, #10603, #10656, #10519, #9255
**Status:** synthetic adversarial response-kernel challenge and independent expansion verifier; no biological model-adequacy result is claimed.

## 1. The unresolved assumption

The query-discriminating intervention planner selects an intervention under a finite declared set of nuisance / observation models. Its result is exact for that set. Exact optimization cannot rescue an incomplete model family.

Three uncertainty objects must remain distinct:

1. **Parameter uncertainty:** unknown parameters inside one declared model family.
2. **Model-class uncertainty:** multiple independently specified observation kernels or simulator families.
3. **Model discrepancy:** actual responses that cannot be adequately represented by the currently admitted model family.

A prior ambiguity set addresses uncertainty over model weights or priors; it does not automatically cover a missing observation mechanism. A set-valued support envelope can state that an outcome is possible without inventing a probability. These modes must not be conflated.

## 2. Paradigm shift

Move from 'robust across the nuisance models we wrote down' to 'robust under an auditable, adversarially challenged model envelope, with explicit abstention when the envelope defeats identification.'

The new layer is additive:
- freeze the query, worlds, prior/loss, intervention universe, costs, feasibility and original prediction matrix;
- admit a provenance-bound challenger model only as a declared sensitivity case;
- append its full intervention × world prediction matrix without rewriting base rows;
- recompute the independent decision-value oracle over the expanded nuisance family;
- require worst-case query loss to be non-decreasing and robust decision value per cost to be non-increasing;
- independently union outcome support across the admitted models for each world and test whether opposite-query worlds remain observationally separable;
- withdraw prior resolution claims when a new admissible response model collapses the discrimination.

The finite challenged model set is still not a proof of global model completeness.

## 3. Synthetic known-answer experiment

The base fixture has three reconstruction worlds, two nuisance models and four candidate interventions. Its fixed-prior expected query loss is 0.10. The feasible `orthogonal-robust` intervention uniquely separates the opposing query worlds under both declared models and reduces worst-case posterior query loss to zero. Entropy-greedy instead selects `high-eig-irrelevant`, which does not resolve the decision.

The challenge fixture appends exactly one new nuisance model, `challenger-unmodelled-overlap`, with a complete 4 × 3 prediction matrix. The query, worlds, priors, interventions, costs, feasibility, split manifest and all 24 base predictions are unchanged. The added kernel maps the opposing query worlds to a shared outcome for the former robust intervention.

Known-answer consequence:
- base: `orthogonal-robust` selected; worst-case query loss 0.00;
- challenged: no feasible intervention has positive robust decision value; worst-case query loss is 0.10 for every feasible action;
- the disallowed perfect intervention remains ineligible even though its nominal value is high;
- union-support analysis has no robustly separating feasible intervention;
- report `NO_FEASIBLE_DISCRIMINATING_INTERVENTION` and `ROBUST_NOT_IDENTIFIABLE`.

This is a software counterexample to overconfidence under model-class omission, not a claim that the challenger is a validated biological mechanism.

## 4. Required monotonicity

Let M be the baseline model family and M' = M ∪ {m*} an append-only extension. For each fixed action a:

- max_{m ∈ M'} R(a,m) ≥ max_{m ∈ M} R(a,m), where R is expected posterior query loss;
- max(0, R0 − max_{m ∈ M'} R(a,m)) / cost(a) ≤ max(0, R0 − max_{m ∈ M} R(a,m)) / cost(a).

Here R0 is the fixed prior baseline Bayes risk. Any strict decrease in worst-case loss or increase in robust value after adding a model is a verifier failure.

For support-only identification, take the union of possible outcomes over all admitted nuisance models for each world. A feasible action can establish robust query identifiability only when every pair of worlds with opposite query decisions has disjoint union support. When all feasible actions retain at least one cross-decision overlap, report `ROBUST_NOT_IDENTIFIABLE`.

## 5. Evidence and independence contract

Every challenge binds the challenger model digest, observation-kernel digest, challenger evidence receipt, adjudication disposition and declared ancestry groups. Different model names do not establish independent evidence. Shared training data, preprocessing, simulator code, source volumes, measurement sessions or reviewer ancestry must remain visible.

`ADMITTED_FOR_SENSITIVITY` means the model is included in the robustness envelope for this audit. It does not mean an independent biological adjudicator has established the mechanism as true.

## 6. Industry and research comparators

- Nominal expected information gain: useful information baseline, but it may select experiments that distinguish query-irrelevant worlds.
- Go & Isaac REIG (UAI/PMLR 2022): robust EIG over a KL ambiguity set of prior distributions. This addresses prior sensitivity, not arbitrary omitted observation kernels. https://proceedings.mlr.press/v180/go22a.html
- Catanach & Das (2023): Expected General Information Gain and Expected Discriminatory Information explicitly address model discrepancy in optimal design. https://arxiv.org/abs/2304.07949
- Abdulsamad et al. (2026 preprint): maximin robust Bayesian experimental design with information-theoretic constraints and PAC-Bayes bounds. https://arxiv.org/abs/2603.14094
- Doumbouya et al. (2026 preprint): certified EIG bounds related to signal-to-noise and Fisher information. Useful for scaling, not proof of model-family adequacy. https://arxiv.org/abs/2609.32282
- Tang, Sloman & Kaski (AISTATS 2026): representative, informative and de-amplifying acquisition under misspecification. https://proceedings.mlr.press/v300/tang26d.html

These methods optimize different objects. Robust EIG, model-discrepancy detection, posterior query loss and support-based identifiability must have separate scorecards. Our opportunity is to connect them to query-critical decision guarantees, not to claim that any one acquisition function solves biological model misspecification.

## 7. Proposed promotion targets

These are aspirations, not achieved results:
- 100% exact agreement with finite-vertex minimax oracles on bounded fixtures;
- 0 false robust query-resolution certificates after a valid overlap challenger is admitted;
- 0 decreases in worst-case posterior query loss after additive model expansion;
- 0 increases in robust decision value per cost after additive model expansion;
- 100% correct abstention on zero-likelihood / out-of-envelope observations;
- 0 independence promotions on shared-ancestry challenge controls;
- ≥20% lower query-regret per total experiment cost than the strongest frozen baseline on held-out prospective cases, with uncertainty supporting improvement.

Biological promotion additionally requires independent calibration, blinded target/animal/session lockboxes, source-bound intervention data, explicit treatment of failed/inaccessible measurements, and external replication.

## 8. Claim ceiling

A passing audit establishes software semantics over the exact finite model family and challenger set supplied. It does not prove the biological response family is complete, establish that the challenger is a true mechanism, or make an in-silico intervention safe or effective in vivo.