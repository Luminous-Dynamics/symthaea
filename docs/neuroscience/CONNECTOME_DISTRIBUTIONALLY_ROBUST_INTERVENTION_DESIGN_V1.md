# Distributionally robust connectome intervention design V1

**Tracking:** #10757 — SCI-CONNECTOMICS-094
**Parent:** #10682 — byte-bound independent replay of #10603 hypothesis-space challenges
**Composes:** #10603, #10656, #10661, #10670, #10675, #9522, #9587, #9204, #9248, #9255
**Status:** finite synthetic model-ambiguity contract; no biological intervention or performance result is claimed.

## 1. Why V1 is not enough

The V1 intervention planner evaluates posterior query loss separately inside each declared nuisance model, then takes the worst result. That is a useful sensitivity score, but it can implicitly let the decision rule know which nuisance/model vertex generated the observation. In a real experiment, the true causal, prior and measurement model is not observed.

The deployable decision rule must be frozen before the model vertex is known. It may condition on the observed outcome category, but not on the hidden model-vertex identity.

## 2. Paradigm shift: partial identification and one shared policy

Represent the declared ambiguity set as the convex hull of an enumerated set of joint-distribution vertices. Each vertex binds:
- the prior mass over admissible reconstruction worlds;
- the intervention × world predictive outcome distributions;
- the causal/measurement model digests;
- the dependence and provenance ancestry of the model construction.

For a fixed deterministic outcome-to-decision policy, expected decision loss is linear in the joint distribution. Its maximum over a convex hull is therefore attained at one of the supplied extreme vertices. The exact bounded problem is:

\[ R^*(a) = \min_{\delta \in \{\mathrm{SUPPORT},\mathrm{REFUTE}\}^{Y_a}} \max_{m \in V} R(a,\delta,m) \]

where `Y_a` is the canonical observed outcome alphabet for intervention `a`, and `V` is the supplied finite vertex set.

The policy \(\delta\) is shared across all vertices. It cannot inspect `model_vertex_id`.

### Distinguish the deployable policy from the clairvoyant bound

Also calculate:

\[ R_{\mathrm{clair}}(a) = \max_{m \in V} \min_{\delta_m} R(a,\delta_m,m) \]

This grants the decision-maker knowledge of the true model vertex before interpreting the outcome. It is an optimistic lower bound, not an executable policy.

The model-uncertainty penalty is:

\[ U(a) = R^*(a) - R_{\mathrm{clair}}(a) \geq 0 \]

A large gap is evidence that apparently strong per-model discrimination is not robust to uncertainty about which model generated the response.

## 3. Intervention selection and certification

Compute the no-observation baseline using one shared support/refute decision across every model vertex:

\[ R_0 = \min_{d \in \{\mathrm{SUPPORT},\mathrm{REFUTE}\}} \max_{m \in V} R(0,d,m) \]

For each feasible intervention:

\[ \mathrm{RVOI}(a) = \frac{\max(0,R_0-R^*(a))}{c(a)} \]

where \(c(a)\) is the frozen total experimental-effort cost. Acquisition, assay, reviewer, compute, analysis and rework costs must be included when the biological benchmark is run.

A robust scientific certificate is allowed only when:

\[ R^*(a) \leq \epsilon_{\mathrm{cert}} \]

where the tolerance is preregistered. Otherwise the correct status is `NOT_CERTIFIABLE` / `NOT_IDENTIFIABLE`, even if an optimistic per-model analysis looks perfect.

V1 uses deterministic binary decision maps because a support/refute scientific claim should not be randomized. A no-certificate state is an evaluator disposition, not a random third scientific answer.

## 4. Information-gain comparison: an important caveat

Report information gain for every supplied vertex separately and publish its minimum/maximum across those vertices as a **vertex sensitivity range**. Do not call the minimum vertex EIG a guaranteed lower bound over the convex hull: mutual information is nonlinear in the joint distribution, and an interior mixture can have less information than any listed endpoint.

The benchmark includes pairwise 50/50 joint-distribution mixtures as diagnostics, but these are not a global optimization certificate. In the reversal fixture, the endpoint minimum EIG is approximately 0.921928 bits, while the midpoint-mixture EIG is approximately 0.818540 bits. This directly demonstrates why the minimum EIG across supplied vertices is not a guaranteed lower bound over the convex hull. The primary selection criterion remains robust decision loss under a shared policy.

## 5. Synthetic known-answer fixture

Three admissible worlds are supplied. Their query decisions are true / false / true. The two joint-model vertices use priors:

| World | Vertex M1 | Vertex M2 |
|---|---:|---:|
| `H-TRUE-A` | 0.10 | 0.15 |
| `H-FALSE` | 0.10 | 0.15 |
| `H-TRUE-B` | 0.80 | 0.70 |

Five interventions are declared:

| Intervention | Model-vertex behavior | Feasibility |
|---|---|---|
| `reversal-high-eig` | Each vertex has a perfect per-model decoder, but the observable X/Y meanings reverse between vertices | Feasible |
| `high-eig-irrelevant` | Separates the high-prior true world from a group containing both query decisions | Feasible |
| `cheap-confounded` | Separates decisions under M1, but M2 collapses all outcomes into Q | Feasible |
| `orthogonal-robust` | Outcome labels differ between vertices, but within each vertex true and false query classes remain separable by one shared map | Feasible |
| `infeasible-perfect` | Perfectly separates the query in every vertex | Disallowed; excluded before optimization |

Expected exact results:

| Quantity | Expected value |
|---|---:|
| No-observation robust risk | 0.15 |
| `reversal-high-eig` clairvoyant lower bound | 0 |
| `reversal-high-eig` shared-policy robust risk | 0.15 |
| `reversal-high-eig` model-uncertainty penalty | 0.15 |
| `high-eig-irrelevant` shared-policy robust risk | 0.15 |
| `cheap-confounded` shared-policy robust risk | 0.15 |
| `orthogonal-robust` shared-policy robust risk | 0 |
| `orthogonal-robust` value per cost (cost = 2) | 0.075 |
| Selected feasible action by robust value per cost | `orthogonal-robust` |
| Selected action by minimum vertex EIG among feasible actions (diagnostic only) | `reversal-high-eig` |
| Robust certificate tolerance | 0.01 |

The crucial result is the gap between `reversal-high-eig`'s zero clairvoyant bound and 0.15 robust policy loss. A per-model optimiser can decode each model perfectly; one policy that does not know the model cannot. The infeasible perfect action receives no policy, risk, EIG or value-per-cost result because it is removed before optimization.

## 6. Required invariants

- Every model vertex contains a normalized prior over the exact frozen world set.
- Every model vertex contains exactly one predictive record for every intervention × world pair.
- Every predictive outcome distribution is normalized and uses canonical, frozen observation-category meanings.
- Every decision policy is a single mapping from observed category to SUPPORT/REFUTE, shared across every vertex.
- The model-vertex ID, model digest, or ancestry group is never an input to the deployable outcome-to-decision mapping.
- The exact robust maximum is computed over every supplied vertex, not only a selected or average vertex.
- The clairvoyant bound is calculated separately and is never returned as a deployable policy.
- Infeasible, disallowed, inaccessible and unassessed interventions are excluded before optimisation and ranking.
- Duplicate, missing, stale or unknown world/action/outcome references fail closed.
- Split units are disjoint; animal/session/target grouping is respected for biological runs.
- Shared model ancestry is reported and never promoted to statistical independence.
- Population sampling, all-list-missed estimates, upstream perturbation likelihood calibration and certificate ownership remain with their existing issues.

## 7. Baselines

Compare under the same worlds, vertex set, feasible action set, observation budget and costs:

- V1 per-vertex posterior Bayes risk (clairvoyant diagnostic; not a deployable competitor);
- minimum vertex EIG heuristic (diagnostic only, not a convex-hull information guarantee);
- fixed baseline decision with no observation;
- expert-selected intervention;
- random feasible intervention;
- graph centrality / path heuristics;
- exact exhaustive common-policy optimum on bounded cases.

Do not average away a worst-case model. Do not use the model vertex as an unobserved covariate. Do not rank infeasible interventions.

## 8. Adversarial known-answer extensions

1. Reverse the observable category semantics across two model vertices: reject a clairvoyant certificate.
2. Add a third vertex that collapses outcomes: robust risk must not improve.
3. Duplicate a model vertex under a different ID but identical content: the result must be invariant.
4. Delete one intervention × world prediction: reject the incomplete matrix.
5. Change a prior or outcome probability without updating its digest: reject the stale input.
6. Add a new opposing reconstruction world and require the #10603 challenge/replay chain to bind it.
7. Add shared acquisition/calibration ancestry across vertices: report overlap, no independence credit.
8. Mark the best-looking intervention disallowed: it must not enter either optimizer.
9. Change the vertex set so the deployable policy risk rises while every per-vertex optimum stays low: expose the uncertainty penalty.
10. Insert a convex-mixture interior check with lower EIG than either endpoint: do not falsely report minimum vertex EIG as a convex-hull guarantee.

## 9. Promotion targets

These are targets, not results:

| Evaluation | Target |
|---|---:|
| Exact shared-policy optimum on bounded fixtures (≤8 outcomes/action) | 100% |
| False promotion of clairvoyant policy as deployable | 0 |
| False robust certificates across supplied vertices | 0 |
| Infeasible/disallowed action selections | 0 |
| Fail-closed complete-matrix / stale-digest controls | 100% |
| Robust decision-loss reduction per total experimental cost vs strongest frozen feasible baseline | ≥20% |
| Model-vertex ancestry overlap reporting | 100% |
| Separate reporting of robust risk, clairvoyant bound, uncertainty penalty and vertex EIG range | 100% |

Biological promotion requires paired, held-out animal/target/session uncertainty, predeclared loss/cost, blinded adjudication and a source-pinned ambiguity-set envelope.

## 10. Industry and scientific comparator strategy

Do not position this as a replacement for proofreading classifiers. ConnectomeBench2 V2 supplies the task-level split/merge predictive baseline; adapted VLMs provide another candidate-generation comparator; NEURD and RoboEM remain reconstruction/proofreading systems. Our new endpoint is whether a *single deployable decision policy* survives a declared uncertainty set over candidate causal and measurement models.

Two biological evidence classes are especially useful:

- The C. elegans signal-propagation atlas directly perturbs neurons and records population calcium responses across 23,433 neuron pairs and 113 animals. It also shows that anatomy misses extrasynaptic signalling; this is an explicit reason to maintain causal/measurement-model alternatives rather than equating anatomical edges with functional effects.
- The 2026 Drosophila walking CPG study identifies DNg100 and DNb08 pathways through connectome simulation and optogenetic experiments. Its reported sensitivity to gain and other model parameters makes it a useful model-uncertainty benchmark; its specific outcome and data versions must be pinned before reproduction.

Global graph-invariant convergence remains a global proofreading effort baseline, not a query-specific causal certificate.

Sources:
- ConnectomeBench2 paper: https://synthneuro.org/publications/connectomebench2-a-unified-benchmark-for-automated-connectomic-proofreading/
- ConnectomeBench2 V2 dataset card: https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2
- 2026 VLM benchmark preprint: https://arxiv.org/abs/2609.36492
- NEURD (Nature 2025): https://www.nature.com/articles/s41586-025-08660-5
- RoboEM (Nature Methods): https://www.nature.com/articles/s41592-024-02226-5
- C. elegans signal-propagation atlas (Nature 2023): https://www.nature.com/articles/s41586-023-06683-4
- Drosophila walking CPG (Nature, 28 October 2026): https://www.nature.com/articles/s41586-026-11195-y
- Global graph-invariant stopping preprint: https://pmc.ncbi.nlm.nih.gov/articles/PMC13345220/

## 11. Claim ceiling

A passing verifier establishes exact minimax decision semantics conditional on the supplied finite joint-distribution vertices and declared convex-hull scope. It does not prove that those vertices span biological or causal uncertainty, that source evidence is authentic, that an intervention is feasible in vivo, that a reconstruction is biologically correct, or that a connectome is globally complete.