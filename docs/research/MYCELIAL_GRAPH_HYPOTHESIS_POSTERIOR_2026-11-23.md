# Feasible Mycelial Graph Hypothesis Posterior v0.1
**Date:** 2026-11-23
**Status:** Exact small-set research implementation / proposed calibration programme

## 1. Objective

A list of candidate fusion probabilities is not a posterior over possible mycelial graphs. Candidates can compete for the same physical contact, share a source tip, or claim a mutually exclusive endpoint. Treating those Bernoulli outputs as independent graph choices assigns probability to impossible combinations.

This protocol defines a bounded posterior over *feasible combinations* while preserving an explicit distinction between classifier probabilities and biological acceptance.

## 2. v1 statistical contract

The v1 implementation makes one deliberately narrow prior assumption:

`P(S) = ∏_i p_i^{z_i}(1-p_i)^{1-z_i}`

where `z_i` indicates whether candidate `i` is present in an analysis-only graph hypothesis and `p_i` is its calibrated classifier output.

Let `F` be the set of combinations that satisfy every pairwise incompatibility supplied by the canonical fusion-counterfactual conflict detector. The posterior is:

`P(S | S ∈ F) = P(S) / Σ_{T ∈ F} P(T)`

for `S ∈ F`, and zero for infeasible combinations.

This is a **conditional-independent Bernoulli prior conditioned on feasibility**. It is not a learned joint CRF, not a causal model and not evidence that the candidate events are truly independent. The assumption is recorded in the serialized report so downstream work cannot mistake it for a general joint model.

## 3. Feasibility contract

The implementation delegates candidate validation and pairwise conflict discovery to the existing `evaluate_mycelial_fusion_counterfactuals` path, so the posterior does not duplicate those rules.

Current conflict conditions include:
- explicit shared conflict-group IDs;
- same source tip;
- shared tip endpoints;
- identical target segment and arc-length fraction for tip-to-side contacts.

The exact posterior module:
- sorts candidate IDs deterministically;
- builds a canonical conflict-pair inventory;
- enumerates all Boolean assignments for up to 16 candidates (at most 65,536);
- excludes combinations that select both members of any conflict pair;
- calculates Bernoulli log prior weights, using log-space normalization;
- preserves feasible hypotheses with exactly zero prior weight as zero-probability rows;
- fails closed when all feasible assignments have zero prior mass;
- emits posterior marginals, entropy and feasible/infeasible counts;
- independently recomputes the report during validation.

No candidate edge is written to the source graph. Posterior inclusion is a graph-analysis hypothesis only.

## 4. Numerical and resource limits

- Exact candidate bound: 16; maximum enumeration: 65,536 assignments.
- Probabilities are validated by the upstream counterfactual input contract.
- Log probabilities prevent ordinary product underflow during normalization.
- Exact zero and one probabilities are represented as negative-infinity log terms where the corresponding Bernoulli outcome is impossible.
- If the feasible prior normalizer is zero, return an explicit error. Never smooth probabilities silently or inject fallback mass.
- Deterministic sorted candidate IDs and ascending bit-mask enumeration define reproducible output order.

For larger candidate inventories, do not quietly expand exact enumeration. Use a separately versioned approximate posterior (e.g. constrained log-linear/CRF or sequential Monte Carlo) and report approximation error, effective sample size, convergence diagnostics, and proper scores against exact small instances.

## 5. v1 limitations

1. **Independence assumption.** Candidate probabilities are treated as independent before feasibility conditioning. Correlated image evidence, shared acquisition artifacts, common illumination errors and a shared classifier can produce dependence not represented here.
2. **Pairwise constraints.** Higher-order constraints are not expressible unless they can be reduced to the explicit conflict inventory. Future constraints should be first-class versioned objects, not hidden code branches.
3. **Only discrete presence/absence.** The posterior does not yet include alternative endpoint identities, uncertain side-contact locations, topology-changing branch hypotheses, or continuous geometric uncertainty.
4. **No biological utility prior.** Demand reachability, topology deltas and resilience are outcomes to report, not priors that should bias the fusion posterior.
5. **No automatic acceptance.** Posterior inclusion means "this is one feasible graph hypothesis", not "accept this fusion as biologically real".

## 6. Required calibration and benchmark work

Before claiming a useful posterior:
- evaluate the upstream candidate probabilities with Brier score, log loss and calibration curves on replicate-held-out annotations;
- compare independent Bernoulli conditioning to joint-label / structured-prediction baselines on small fully adjudicated contact groups;
- use joint log score for exact graph subsets and event Brier scores for predeclared graph events (e.g. "at least one candidate in this group is real");
- use proper scoring rules for derived quantities and uncertainty (e.g. CRPS for path-length or transport outcome distributions);
- report posterior credible-set coverage and size at biological-replicate level;
- test groupwise calibration under device, organism, species, annotation protocol and occlusion shifts;
- use synthetic exhaustive fixtures to verify zero mass on infeasible assignments and exact agreement with independent enumeration;
- preserve a hard negative-control path for shared source tips, shared endpoints, same-side claims and deliberately contradictory probability-one candidates.

Proposed target hypotheses (not achieved results):
- 100% exact agreement with an independent exhaustive oracle on all tested instances up to 16 candidates;
- zero posterior mass on every infeasible combination;
- nominal 90% graph-event credible-set coverage between 85% and 95% on a held-out replicate cohort, with uncertainty intervals;
- better joint log score and event Brier score than independent unconditioned Bernoulli choices on datasets containing actual contact conflicts;
- no performance claim until calibration and replicate-held-out evaluation data exist.

## 7. Next model evolution

After exact v1 is validated:
1. represent conflict and exclusivity constraints as typed, versioned constraints with provenance;
2. add joint calibration or learn a constrained log-linear potential from adjudicated multi-contact groups;
3. include shared-cause latent variables for illumination/occlusion and common acquisition artifacts;
4. add alternative side-contact positions and uncertainty over physical coordinates when measured centerlines exist;
5. compare exact inference, branch-and-bound, constrained sampling and variational approximations against the exact enumerator on small cases;
6. couple posterior graph hypotheses to counterfactual topology/resilience scores and active-sensing value-of-information, without using those downstream scores as hidden priors.

## 8. Claim boundary

The current implementation is an exact, bounded, conditioned-Bernoulli posterior over candidate subsets subject to explicit pairwise conflicts. It is not a general joint biological posterior, it does not simulate flow or viability, and it does not authorize edge acceptance.
