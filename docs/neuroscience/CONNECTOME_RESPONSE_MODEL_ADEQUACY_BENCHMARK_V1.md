# Connectome response-model adequacy challenge benchmark V1

**Tracking:** #10774 — response-model adequacy in query-directed intervention planning  
**Depends on:** #10670 / #10675 (query-discriminating intervention objective), #10682 (byte-bound reconstruction challenge replay), #10519 (observation-support challenge), #10752 (cross-animal transport).  
**Status:** protocol, synthetic known-answer fixture and independent reference oracle; no biological result is claimed.

## 1. Research question

Can a query-directed connectome intervention policy detect when its observation/response model is incomplete, admit independently supported model challenges, and withdraw or revise an intervention recommendation before falsely declaring a scientific question resolved?

The existing intervention evaluator operates over a finite declared nuisance-model family. Robustness over that family is conditional. A perfectly replayed optimum can still be wrong if every supplied response model omits the same biologically plausible pathway, indicator effect, state transition or measurement failure.

This benchmark adds a challenge boundary around that response-model family. It does not replace the reconstruction-world planner, query-loss evaluator, sampling design, or intervention scorecard.

## 2. Keep three uncertainty classes separate

1. **Parameter uncertainty:** uncertain parameters within one frozen response model.
2. **Model-class uncertainty:** several independently specified response/observation models, each with its own version and provenance.
3. **Model discrepancy:** evidence that observed responses are not adequately represented by the currently admitted family, including held-out residual structure, orthogonal measurements, or an independently developed mechanistic challenger.

A wider confidence interval on a parameter is not automatically a new model. A new model name is not evidence of independence. A zero-likelihood observation is not a reason to force a posterior update under the old family.

## 3. Frozen experiment and challenge envelope

Before intervention selection, freeze:

- scientific query, decision rule and asymmetric loss table;
- biologically admissible reconstruction worlds and their exact source digests;
- feasible intervention universe, safety/feasibility receipts and all-in costs;
- baseline observation kernels, response models and nuisance family;
- model-fit, challenge/calibration and final policy-evaluation units;
- challenge-channel identities, lineage, evidence bytes and adjudication rules;
- the baseline decision-loss evaluator and exact tie-breaking rule.

A model challenge may propose either:

- a new response/observation model for one or more world × intervention combinations; or
- an explicit widening of outcome support where calibrated probabilities are not defensible.

A challenge must not modify the frozen query, reconstruction-world set, intervention universe, cost table or loss function. Those are different scientific changes and require a separately versioned run.

Each admitted challenger binds the exact model and observation-kernel digests, raw evidence digest, challenge-report digest, adjudication receipt, held-out unit IDs, source lineage and model ancestry. Retain rejected, disputed, inaccessible and unresolved challenges; never silently convert them into negative findings.

## 4. Exact finite probability semantics

For a frozen query with decision q(w), prior p(w), loss L(d,q), intervention a, and response model m, define the model-conditional expected posterior decision loss:

    R(a,m) = sum_y min_d [
        L(d,0) P(q=0,y | a,m) + L(d,1) P(q=1,y | a,m)
    ]

This expression is the expected Bayes decision loss after observing the intervention outcome; it avoids treating a time-bin, cell or trial as an independent biological replicate. For the finite declared model envelope M:

    R_robust(a,M) = max_m R(a,m)
    V(a,M) = max(0, R_0 - R_robust(a,M)) / c(a)

where R_0 is the frozen pre-intervention Bayes risk and c(a)>0 is the all-in intervention cost. The selected intervention maximizes V over feasible actions, with deterministic tie-breaking. If the maximum value is zero, return NO_FEASIBLE_DISCRIMINATING_INTERVENTION.

These are finite-envelope guarantees only. A fixed prior and finite set of model likelihoods do not prove that the real response mechanism is in the envelope.

## 5. Model-expansion monotonicity

For an admitted challenge m* not already in M:

    R_robust(a, M union {m*}) >= R_robust(a, M)
    V(a, M union {m*}) <= V(a, M)

for every unchanged intervention a. The best achievable robust value over the same feasible action set cannot increase after a model is added. The selected action may change; the robust objective must not become more optimistic.

The exact model identity must remain fixed across all outcomes of one experiment. For adaptive multi-step policies, do not let the evaluator choose a different nuisance-model identity after each observed branch unless the declared biological model explicitly permits such time-varying dynamics. Otherwise an adversary could switch models opportunistically between steps, producing an incoherent bound. Carry one model identity (or a declared transition model) through the complete policy tree.

## 6. Challenge evidence and independence

Admissible challenge channels include:

- held-out measured response trajectories with preregistered residual/coverage diagnostics;
- orthogonal measurement modalities or indicator controls;
- an independently implemented mechanistic response model;
- blinded expert/model-discrepancy review under a frozen protocol;
- a probability-sampled challenge cohort, when the estimand concerns a declared population.

For every channel preserve data, training, simulator, code, reviewer, acquisition and referral ancestry separately. Shared ancestry is reported as overlap, not promoted to independent corroboration. An independently measured held-out response can be novel evidence even when a challenger model shares training ancestry; those are different independence questions.

A challenger is ADMITTED, REJECTED, DISPUTED, INDETERMINATE or INACCESSIBLE, each with a reason and adjudication receipt. Only admitted models enter the robust envelope. Any unresolved challenge in the declared promotion set blocks promotion; it is not equivalent to rejection.

A held-out observation outside every admitted model's declared outcome support produces MODEL_DISCREPANCY_DETECTED / NOT_IDENTIFIABLE. It must not be assigned zero-probability Bayesian weight and then discarded, nor should the model invent a probability distribution without evidence.

## 7. Known-answer adversarial fixture

The checked-in synthetic fixture has three reconstruction worlds with priors 0.10 / 0.10 / 0.80 and query values true / false / true.

- Before challenge, a cheap intervention A distinguishes the query classes under the nominal model and is optimal.
- An independently held-out challenge admits a saturation model under which A maps all three worlds to the same observed outcome.
- Intervention B still separates the query classes under both models, but costs more.
- After model expansion the exact robust optimum changes from A to B.
- An artificially perfect but disallowed intervention remains ineligible.
- The oracle proves that the worst-case loss for each unchanged intervention does not decrease and the value-per-cost does not increase after expansion.

The fixture also deliberately gives the baseline and challenger a shared model-training ancestry group. Their model agreement or disagreement therefore receives no independence credit. Its held-out challenge measurement uses a separate challenge unit, which is reported separately from model-development lineage.

## 8. Required adversarial extensions

1. **Nuisance collapse:** a challenger makes the previously selected intervention observationally equivalent across opposing query worlds.
2. **No remaining discriminator:** expand the envelope until every feasible action has zero robust value; return no-feasible-discriminator, not a nominal recommendation.
3. **Out-of-support observation:** report model discrepancy and abstain instead of applying a zero-likelihood posterior update.
4. **Decision-irrelevant discrepancy:** a challenger changes trajectories but leaves the frozen query loss unchanged; detect discrepancy without inventing query impact.
5. **Shared ancestry:** multiple challengers share training data, simulator code or measurement lineage; do not count them as independent channels.
6. **Challenge leakage:** challenge or final lockbox units enter model fitting, tuning, intervention selection or loss selection; reject the run.
7. **Unresolved adjudication:** retain an ambiguous challenge and block promotion.
8. **Parameter/model distinction:** a parameter-range expansion remains within its original model identity; a new mechanism or observation kernel requires a new model identity.
9. **Adaptive policy consistency:** one fixed model identity is carried through every outcome branch unless a dynamic model transition is explicitly declared.
10. **False rejection:** a valid held-out residual challenge cannot be dismissed without a blinded adjudication receipt.
11. **Non-probabilistic support:** when calibrated probabilities are unavailable, use support-based semantics and explicit non-identifiability; do not manufacture normalized probabilities.
12. **Prior sensitivity:** repeat decision-value calculations over a declared prior set where prior robustness is claimed; do not describe fixed-prior results as prior-robust.

## 9. Comparators and measurements

Compare under identical world sets, response data, intervention feasibility, budgets and compute:

- nominal #10675 decision-value selection;
- finite-envelope robust #10675 selection;
- entropy / mutual-information and Bayesian EIG selection;
- expected general information gain and expected discriminatory information as model-discrepancy criteria;
- robust EIG under a declared prior ambiguity set;
- expert-ranked and random feasible intervention;
- held-out posterior-predictive checks and preregistered residual diagnostics;
- the exact finite oracle on small known-answer cases.

Report separately:

- out-of-envelope predictive failure and challenge recall at the independent unit level;
- false query-resolution rate after a valid admitted challenge (target: zero);
- worst-case posterior query loss before and after each challenge;
- per-action and optimal value-per-cost before/after envelope expansion;
- model-envelope expansion monotonicity violations (target: zero);
- false challenge admission/rejection and unresolved-challenge rates;
- calibration, proper predictive scores, interval coverage and sharpness;
- target/animal/session lockbox leakage (target: zero);
- acquisition, compute, analysis and adjudication cost;
- abstention and NOT_IDENTIFIABLE rates.

Never collapse calibration, discrepancy detection, query decision accuracy and cost into one composite score.

## 10. Promotion targets

These are proposed gates, not achieved results:

- 100% exact agreement with an independent oracle on bounded synthetic cases;
- zero robust-loss decreases and zero robust-value increases after admitted model-family expansion;
- zero false query-resolution certificates after an opposing-world saturation challenge;
- 100% explicit handling of out-of-support observations;
- zero infeasible interventions selected;
- zero independence promotions when declared ancestry overlaps;
- zero lockbox leakage;
- every unresolved challenge blocks promotion;
- on biological lockboxes, demonstrate improved decision regret per all-in cost over the strongest frozen baseline with animal/target/session-clustered uncertainty, while preserving calibration and abstention.

The numerical biological target must be set after a source-bound pilot estimates independent-unit variance; do not pick a headline percentage from the synthetic fixture.

## 11. Integration and claim ceiling

- #10675 owns query-decision value and feasible intervention selection.
- #10682 owns byte-bound replay of the finite reconstruction-world challenge.
- #10519 owns support-expansion challenges.
- #9204 / #9248 own discovery sampling and all-channel-missed sensitivity.
- #10752 owns cross-animal transport and response calibration.
- #9255 owns query-scoped certificates and invalidation.

A passing synthetic oracle establishes only the specified finite probability arithmetic and expansion monotonicity. It does not prove source authenticity, biological admissibility, channel independence, response-model completeness, in-vivo intervention feasibility or causal synaptic mechanism.