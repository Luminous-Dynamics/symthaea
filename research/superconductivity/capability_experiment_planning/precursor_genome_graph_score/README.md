# SC-PRECURSOR-GENOME-GRAPH-SCORE-0.1

**Status:** synthetic cold-start scoring contract; no real predictor is trained or scored yet.  
**Research issue:** [#13039](https://github.com/Luminous-Dynamics/symthaea/issues/13039)  
**Source graph audit:** [PR #13027](https://github.com/Luminous-Dynamics/symthaea/pull/13027)  
**Censoring-aware onset evaluator:** [PR #13013](https://github.com/Luminous-Dynamics/symthaea/pull/13013)

## Why a graph-specific scorer is necessary

The released Precursor Genome contains 1,035 unique unordered precursor pairs over 46 precursor formulae: a complete graph K46. Every precursor has 45 incident reactions. After the paper-aligned exclusions (26 physical powder-recovery failures and 9 poor XRD refinements), the analyzable cohort has 1,000 outcomes: 253 cross-precursor products and 747 non-products. A transformed-only product is negative only for the endpoint “a product incorporating material from both precursors”; it is not chemical inertness.

A random edge split is not an out-of-distribution test in this graph. The same two precursor identities can appear in train and test. The graph's natural transfer tests are:

- **LOPO (46 folds):** hold out one precursor and every edge incident to it. Each fold tests 45 edges and trains on 990 edges.
- **Double-node holdout (1,035 folds):** hold out two precursor nodes and all 89 edges incident to either one. Each fold trains only on the 946 edges among the other 44 nodes.

These fold geometries are fixed by graph structure. They test the ability to predict reactions involving unseen precursor identities rather than memorizing node frequencies.

## Implemented contract

The Rust auditor accepts a complete undirected graph with one row per unordered edge, explicit included/excluded state, and predictions for every eligible edge in every required fold. For every fold it verifies:

1. The held-out node or node pair is exactly the frozen fold identity.
2. The training node set is exactly the complement of the held-out nodes; no test precursor can occur in training.
3. The declared training record count matches all eligible edges between training nodes.
4. Every eligible held-out edge is predicted exactly once, and no training or excluded edge is scored as a test prediction.
5. Every probability is finite and in [0,1], and the decision threshold is fixed in the model manifest.

Each receipt includes deterministic SHA-256 identities for the training and test edge ID sets. Those hashes make fold membership reproducible; they are not proof that an opaque model was honestly trained on those rows. A production runner should additionally emit a signed training trace or a verifiable artifact manifest.

Metrics per fold:
- Brier score and log loss for probability quality;
- ROC-AUC and average precision for ranking under class imbalance;
- balanced accuracy, precision, recall, specificity and MCC at the predeclared threshold.

The primary summary is the unweighted mean across folds plus worst-fold Brier and log loss. **Folds overlap and are not independent samples.** Do not construct confidence intervals by treating 46 LOPO or 1,035 double-node folds as independent experiments. Use node-grouped bootstrap or a predeclared hierarchical uncertainty procedure on the underlying predictions and confirm prospective performance on new pairs.

## Reference baseline and stretch targets

For the 1,000-record public cohort, the positive prevalence is 0.253. A constant-prevalence classifier has:

| Metric | No-skill reference |
|---|---:|
| Brier score | 0.188991 |
| Log loss | 0.565607 |
| ROC-AUC | 0.500 |
| Average precision | 0.253 |

These are arithmetic reference floors, not model performance. They are recalculated from the evaluated fixture/cohort by the auditor.

Stretch goals (not achieved):
- Beat the constant-prevalence baseline on Brier and log loss and improve ROC-AUC/average precision under **both** LOPO and double-node holdout.
- Compare with a simple additive precursor main-effect baseline before claiming a pair-interaction model adds value.
- Preserve calibration and not just ranking; report reliability diagrams and confidence intervals with node-aware resampling.
- Report precision@K and verified cross-precursor products per experiment under an equal discovery budget.
- Validate on prospective precursor pairs, then connect outcomes to the onset model, thermochemical network and spatial transport models without treating a single-temperature screen as a complete onset curve.

## Relationship to broader superconductivity ML

The generic superconductivity OOD benchmark [PR #12871](https://github.com/Luminous-Dynamics/symthaea/pull/12871) evaluates family/prototype/source/condition/time holdouts for material-property observations, including its separate tri-state superconductivity feasibility gate. This contract is deliberately narrower: it scores the binary outcome of a precursor-pair synthesis experiment on the complete Precursor Genome graph. Its test geometry is defined by precursor nodes, not Tc-property families.

The onset-temperature benchmark is another distinct task: it models a distribution over event time with left-, right- and interval-censored evidence. Do not merge binary one-temperature synthesis outcomes with in-situ onset regression or superconductivity-property labels.

## Validation and limitations

The bundled K4 fixture is synthetic and exists only to exercise fold generation, leakage rejection, prediction completeness and metric arithmetic. The full K46 cohort must be bound to the source-pinned graph audit receipt before any model result is accepted. A future mode should read that receipt directly rather than hand-duplicating source labels.

No new model, real-cohort score, synthesis recommendation, phase yield, or superconducting material is established here.
