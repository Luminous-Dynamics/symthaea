# Connectome support-expansion monotonicity benchmark V1

**Tracking issue:** [#10588 — adaptive-policy monotonicity under support expansion](https://github.com/Luminous-Dynamics/symthaea/issues/10588)  
**Implementation branch:** \`research/connectome-support-expansion-monotonicity-v1\`  
**Parent adaptive-policy work:** [#10568](https://github.com/Luminous-Dynamics/symthaea/pull/10568)  
**Correlated support model:** [#10528](https://github.com/Luminous-Dynamics/symthaea/pull/10528)

## 1. Objective

The central research question is no longer just whether a policy is optimal over a declared set of possible observations. It is whether that result is *stable under an evidence-backed enlargement of the possible-outcome set*.

A policy can be exactly optimal and still be scientifically fragile if the support relation omitted a physically possible outcome. The benchmark makes that weakness observable by asking a challenger to add outcomes, then recomputing the exact policy and fixed-plan optima.

This is an assurance and sensitivity benchmark, not a segmentation/proofreading classifier benchmark. It complements ConnectomeBench2 and predictive model evaluations; it does not claim to beat their split/merge balanced accuracy.

## 2. Formal monotonicity property

Let a frozen query be \(q:W\rightarrow\{0,1\}\), with fixed world set \(W\), action universe \(A\), positive action costs \(c(a)\), and finite joint observation support \(J(w)\) for each world.

A challenge creates \(J'(w)\) such that for every world:

\`J'(w) = J(w) \\cup \\Delta(w), \quad \\Delta(w) \\cap J(w)=\\varnothing\`

All worlds, query decisions, actions, costs and provenance for the base support remain fixed. The only operation is adding previously absent, evidence-bound joint tuples.

Any policy that resolves all outcomes in \(J'\) also resolves the subset of outcomes in \(J\). Since \(J'\) can introduce additional branches but cannot remove old branches, the exact minimum worst-case adaptive effort cannot decrease:

\`C^*_{adapt}(J') \\ge C^*_{adapt}(J)\`

The same argument applies to a fixed observation bundle:

\`C^*_{fixed}(J') \\ge C^*_{fixed}(J)\`

If no policy exists over \(J\), none can become feasible over the superset \(J'\). A support expansion can preserve cost, increase cost, or destroy feasibility. It cannot create a cheaper or newly feasible robust plan.

This is a finite-set theorem over the supplied relations. It does not say that the relation is complete or that the new tuples are physically possible; those are evidence and measurement-model questions outside the optimizer.

## 3. Challenge contract

Every challenge is evaluated independently against the same base support digest. This prevents accidental cumulative application and makes each sensitivity result reproducible.

A challenge binds:
- unique challenge identifier;
- challenger evidence and observation-model SHA-256 digests;
- each affected scenario ID;
- per-world source evidence and joint-observation-model digests;
- dependence ancestry groups;
- complete joint tuples containing exactly one outcome per frozen action.

The challenge digest binds the base support digest, challenge identity, source/model digests, ancestry labels and canonicalized joint tuples. A derived expanded-support digest binds the base digest to that challenge digest.

The verifier rejects stale base bindings, unknown worlds/actions, repeated worlds within a challenge, duplicate tuples, attempted duplicate additions, malformed provenance, empty supports and per-world tuple counts above the 64-row exact mask limit. It never deletes, relabels or replaces a base tuple.

The report includes before/after adaptive cost and feasibility, before/after fixed cost and feasibility, added tuple count, challenge digest, expanded-support digest, and independent monotonicity booleans.

Any monotonicity violation changes the overall status to \`SUPPORT_EXPANSION_MONOTONICITY_VIOLATION\` and returns non-zero from the CLI after writing the counterexample report. It is not a warning-only outcome.

## 4. Current known-answer cases

### 4.1 Cost sensitivity

Base support:
- exact adaptive worst-case effort: 4;
- exact fixed-bundle effort: 5;
- adaptive plan is cheaper by 1 effort unit.

Challenge:
- add one new joint tuple to the refuting \`world-d\`;
- keep the base tuples and all action/query/world definitions unchanged;
- recompute both exact plans.

Expected result:
- adaptive optimum increases from 4 to 5;
- fixed optimum remains 5;
- adaptive and fixed monotonicity pass;
- the previous 1-unit adaptive advantage disappears.

This case proves that the challenge mechanism can surface a real policy-cost sensitivity rather than merely accepting arbitrary additions.

### 4.2 No-policy preservation

The base fixture contains one joint tuple shared by opposing-decision worlds, so no robust policy exists. The challenger adds an additional outcome to one world while preserving the original shared tuple.

Expected result:
- no adaptive policy before or after;
- no fixed bundle before or after;
- monotonicity passes;
- no-policy cannot be resurrected by a support superset.

### 4.3 Invalid challenge controls

Additional test controls reject:
- a tuple already present in base support;
- duplicated challenge IDs;
- repeated scenario additions;
- malformed provenance and wrong action coverage;
- stale base joint-support digest;
- expanded support above the exact mask capacity.

## 5. Benchmark matrix and promotion targets

| Metric | Target |
|---|---:|
| Adaptive exact-cost monotonicity under exhaustive small support expansions | 100% |
| Fixed-plan exact-cost monotonicity | 100% |
| Previously impossible policy becomes feasible after expansion | 0 |
| Duplicate/stale support additions accepted | 0 |
| Expanded exact optimum vs independent oracle | 100% |
| Canonical challenge digest drift under ordering permutations | 0 |
| Counterexample reports emitted on deliberately injected monotonicity violation | 100% |
| Real-world challenge additions independently adjudicated | 100% before claims of scientific robustness |

For small universes, enumerate the full support lattice: construct every subset of a bounded candidate-tuple universe, use inclusion pairs \(J\subseteq J'\), and check the theorem for every pair. The initial harness supports at most 12 actions, 12 worlds and 64 joint tuples per world; an exhaustive lattice corpus should be generated at much smaller bounds (e.g. 3 actions, 4 worlds, a small tuple alphabet) to keep complete enumeration tractable.

For realistic challenge campaigns, report the cost delta distribution, feasibility-loss rate, new branch count, worst-case path expansion, which additions triggered a changed policy tree, and the evidence provenance for each addition. Do not report only mean cost: tail and maximum review burden are operationally important.

## 6. How this compares with the field

### Predictive reconstruction and proofreading

[ConnectomeBench2 (2026)](https://arxiv.org/abs/2606.21116) is the direct predictive benchmark. Its public release reports more than 716,000 expert-labelled proofreading decisions and over 4.5 million associated images across mouse, human, zebrafish and fly connectomes. Its multimodal ViT reaches human-level performance on the benchmark's split and merge proofreading tasks and evaluates calibration and distribution shift. We should reproduce this benchmark exactly before making predictive-performance comparisons.

A [September 2026 VLM preprint](https://arxiv.org/abs/2609.36492) evaluates open and closed vision-language models on synapse detection, polarity and proofreading, including cross-species transfer and LoRA adaptation. It shows that adaptation and transfer must be measured explicitly; zero-shot performance is not a sufficient baseline. Treat its numbers as preprint results until independently reproduced.

### Reconstruction uncertainty

[Funke, Nature Methods (2025)](https://www.nature.com/articles/s41592-025-02945-3) argues that scalable connectomics requires quantifying both data and model uncertainty, especially as human supervision decreases. Support expansion is a complementary operationalization: instead of presenting a confidence scalar alone, it tests whether a downstream decision survives an explicit enlargement of the admissible joint outcomes.

### Robust set prediction

[Robust Yet Efficient Conformal Prediction Sets (ICML 2024)](https://proceedings.mlr.press/v235/h-zargarbashi24a.html) studies robustness of prediction sets to adversarial test examples and perturbed calibration data. This is a useful methodological comparator, but the finite support sets used here are not conformal sets and carry no marginal coverage guarantee. The assurance contract must not borrow that guarantee without a valid calibration construction.

## 7. Where we can outperform current practice

The opportunity is not to claim that a planner beats the best segmentation network at detecting a split or merge. It is to make downstream scientific decisions harder to overstate.

The industry baseline is strong at candidate generation, proofreading classification, calibration and active-label selection. Our differentiator should be an independently replayable chain:

1. reproduce the predictive model and its benchmark split;
2. bind predictions to exact data, spatial frames, model weights and calibration;
3. enumerate query-relevant competing reconstruction worlds;
4. preserve correlated joint observation supports;
5. compute exact fixed and adaptive review strategies;
6. adversarially expand possible outcomes and check monotonicity;
7. stop only when every surviving world gives the same frozen scientific answer;
8. issue a versioned query certificate with limitations and independent verification receipts.

Measure quality at the **query outcome**: wrong scientific decisions per accepted certificate, selective risk at coverage, calibration under spatial/species shift, support-expansion fragility, review effort at matched adjudicated quality, and post-review regret. Keep split/merge bAcc as an input-model metric, not the final scientific-reliability metric.

Prospective targets should be: zero false robust certifications in the exhaustive synthetic challenge lattice; exact agreement with independent oracles on bounded cases; at least 25% fewer expert minutes at matched query-critical error rate; and cross-species/spatial reporting with confidence intervals and no hidden denominator changes. The last two require real prospective data and are not achieved here.

## 8. Claim boundary

The tool proves only a computational monotonicity property of finite declared support supersets. It does not verify that a challenger tuple is physically possible, that all possible outcomes are represented, that sources are authentic or independent, or that the query worlds capture biological reality. It does not compute probabilities, likelihoods, posterior distributions, entropy or expected value of information.

The strongest scientifically honest statement is: **the decision policy's worst-case effort did not improve when the declared admissible joint-outcome set was enlarged by these evidence-bound additions.** That is a useful assurance result, not proof of biological truth.
