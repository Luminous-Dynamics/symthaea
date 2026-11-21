# Decision Observability and Counterfactual Ambiguity Bounds (v0.1)

**Tracking:** NEURO-SILICON-012 / #9563  
**Evaluator implementation:** research/temporal-evaluator/src/decision_observability.rs  
**Composes with:** #9637 evaluator-side transient observability and alias-pair controls; #9616 candidate-side active temporal sensing; #9954 common temporal decision benchmark  
**Status:** source-authored evaluator primitive; tests not executed

## 1. Why this is a separate measurement

The existing alias-pair primitive proves that two distinct latent signals can agree at every sampled timestamp and diverge between samples. That establishes schedule-specific signal ambiguity.

But signal ambiguity is not automatically decision ambiguity:
- Two latent worlds can be different but still require the same optimal action.
- Two latent worlds can induce different optimal actions even though the candidate receives exactly the same evidence.
- A candidate can make an avoidable error even when some uncertainty is unavoidable.

A benchmark should not call all three cases the same thing.

This module adds a decision-relative lower bound over an **observational equivalence class**: multiple evaluator-owned hidden worlds that produce identical candidate-visible history, candidate recurrent state, model/configuration and frozen policy RNG seed.

## 2. Candidate-view equivalence

Every world in one class must bind exactly the same:
- SHA-256 of the canonical candidate-visible observation transcript (event time, availability time, values, masks, modality and encoding);
- SHA-256 of candidate recurrent/internal state before the decision;
- SHA-256 of model and configuration;
- deterministic policy seed.

The candidate action must also be identical across paired worlds. A mismatch is rejected as a candidate-view, determinism or evaluator-construction failure; it is not credited as hidden-world inference.

The evaluator-only side additionally binds a world ID, hidden-world digest and an action-loss vector. None of these fields is exposed to the candidate.

## 3. Decision-relative quantities

For world h, admissible action a and evaluator loss L(h,a) normalized to parts per million:

**Per-world full-information optimum**

\[
L^*(h)=\min_a L(h,a)
\]

**Candidate worst-case regret**

\[
R_{\text{cand}}^{\max}=\max_h[L(h,a_{\text{cand}})-L^*(h)]
\]

**Deterministic minimax-regret lower bound**

\[
R_{\text{minimax}}=\min_a\max_h[L(h,a)-L^*(h)]
\]

**Avoidable candidate regret above the deterministic lower bound**

\[
R_{\text{avoidable}}=\max(0,R_{\text{cand}}^{\max}-R_{\text{minimax}})
\]

The certificate also reports:
- decision_identifiable: whether at least one common action is optimal in every indistinguishable world;
- common optimal actions;
- deterministic minimax-loss action and its worst-case raw loss;
- deterministic minimax-regret action;
- candidate mean/worst-case loss and regret;
- irreducible worst-case and mean regret for the minimax-regret action.

Ties are resolved lexicographically by action ID. All input loss values are integer PPM; aggregation uses checked integer arithmetic and a documented floor for mean values.

## 4. Interpretation

- **Different worlds, common optimal action:** hidden state is ambiguous, but the decision is identifiable. Do not penalize a candidate for failing to distinguish a decision-irrelevant latent difference.
- **No common optimal action:** the same candidate view cannot determine one action that is optimal in every possible world. The minimax-regret value quantifies the deterministic-policy ambiguity floor for the declared finite alternative set.
- **Candidate regret above the floor:** the excess is avoidable under the declared action/loss matrix. This is the part that model/policy improvement can target.
- **Candidate action differs across supposedly identical views:** reject the equivalence class. Investigate hidden inputs, state mismatch, RNG mismatch or nondeterministic execution.

A single candidate action across the worlds is required in v0.1. Randomized or mixed strategies need an explicit probability-kernel contract and a separate minimax mixed-strategy solver; the deterministic lower bound must not be mislabeled as a universal lower bound for randomized policies.

## 5. Relationship to signal observability

Compose the new decision certificate with #9637's existing verify_alias_pair and score_sampling_observability:

1. Verify that two evaluator-owned latent traces agree on all candidate observations.
2. Bind their canonical sampled histories and candidate recurrent states to the same candidate-view identity.
3. Produce a hidden outcome/action-loss vector from a preregistered decision contract.
4. Evaluate whether the hidden-world differences change the optimal action.
5. Report unavoidable ambiguity separately from candidate error.

The module does not decide whether a transient was physically detectable; #9637 still owns event observability, sensor amplitude floors, transient detection and alias-pair signal controls.

## 6. Adversarial controls

Ten authored tests cover:
- a shared optimal action across distinct worlds;
- worlds with no common optimal action;
- separation of unavoidable ambiguity from avoidable candidate regret;
- minimax raw loss vs minimax regret objectives;
- mismatched observation/state/configuration/seed identity;
- candidate-action mismatch under the same frozen view;
- inconsistent admissible-action sets;
- duplicate world and action IDs;
- empty equivalence classes;
- invalid loss range;
- a single-world identifiable control.

The source module is evaluator-only and must remain outside candidate linking and candidate-visible payloads. Package separation is not an OS isolation boundary; #9437/#9621 remain required for confirmatory use.

## 7. Claim ceiling

This is an exact finite-set bound for the supplied hidden-world alternatives, action set, loss matrix and deterministic policy seed. It is not a universal impossibility theorem for all latent processes, all sensors or randomized policies. The hidden-world set must be preregistered and independently generated; candidate actions cannot be selected after seeing hidden labels.

No active sensing policy, real sensor, model quality, energy result, SOTA result or hardware advantage is claimed by this evaluator primitive.
