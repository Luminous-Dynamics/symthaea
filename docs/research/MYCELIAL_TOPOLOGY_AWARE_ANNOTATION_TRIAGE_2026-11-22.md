# Mycelial Topology-Aware Annotation Triage

**Date:** 2026-11-22  
**Status:** First renderer-neutral junction triage implementation committed; exact-head Rust/Clippy validation pending.  
**Code:** `crates/domains/symthaea-canvas/src/mycelial_annotation_triage.rs`

## 1. Paradigm shift

Confidence-only review asks: *Which prediction is least certain?*

Topology-aware review asks two separate questions:

1. **Epistemic uncertainty:** how uncertain is the predicted junction class?
2. **Structural consequence:** how much does the current network's topology depend on this junction?

A low-confidence leaf and a moderately uncertain articulation point should not necessarily consume equal human-review effort. A predicted anastomosis can also warrant special attention because a false fusion may add a cycle or merge previously disconnected regions.

The system therefore emits a vector of evidence, not a universal scalar "importance" score. A named review policy determines queue ordering; the reviewer remains the authority.

## 2. First implementation

Module:

`crates/domains/symthaea-canvas/src/mycelial_annotation_triage.rs`

Public builder:

`build_mycelial_annotation_triage(prediction, calibration, config)`

Inputs:
- an annotated prediction graph;
- its calibrated seven-class probability report (six junction classes plus NoObject);
- an explicitly named queue ordering policy and thresholds.

The builder checks that the calibration receipt and prediction graph share the same immutable source-observation and prediction-graph digests. It does not consume ground-truth class labels to rank items; it uses the retained probabilities. The calibration report's validation recomputes aggregate calibration metrics from its evidence, but the queue ordering itself only reads per-junction probabilities and the prediction graph.

### Separate uncertainty signals

For each predicted junction:
- predicted class;
- top-label confidence;
- second-highest probability;
- top-two margin;
- Shannon entropy in nats;
- anastomosis probability;
- projected-crossing probability;
- uncertain-class probability.

### Separate topology signals

For each junction, the builder removes that junction and its incident segments from a temporary graph projection, then recomputes:
- degree and incident segment IDs;
- connected components before/after removal;
- signed component-count change;
- cycle rank before/after removal;
- cycle-rank loss;
- articulation-critical flag under a caller-declared component-delta threshold.

This is a **junction-deletion sensitivity proxy**. It does not yet compute the exact counterfactual of changing one junction's class from crossing to anastomosis, nor does it calculate changes to measured transport or demand-pair delivery. Those require explicit edge-acceptance and downstream-effect models.

### Named orderings

Three deterministic policies are available:

1. `StructuralImpactThenUncertainty`: articulation criticality, cycle loss, component delta and degree first, then entropy/margin.
2. `UncertaintyThenStructuralImpact`: entropy/margin first, then structural impact.
3. `AnastomosisProbabilityThenImpact`: anastomosis probability first, then structural impact and uncertainty.

The policy name and thresholds are bound into the report digest. The full vector is retained so downstream users can compare or re-order items without re-running graph analysis.

No policy converts a candidate into an accepted fusion or mutates the prediction graph.

## 3. Validation

The report includes:
- source observation digest;
- prediction graph digest;
- calibration report digest and policy ID;
- ordering policy and thresholds;
- baseline topology summary;
- complete junction inventory and one-based review ranks;
- counts of uncertainty-review, articulation-critical and anastomosis-sensitive junctions.

It validates:
- digest and threshold metadata;
- unique junction identities;
- bounded finite probabilities, confidence, entropy and margins;
- incident segment IDs are sorted and agree with degree;
- component and cycle deltas are internally consistent;
- derived flags agree with their configured thresholds;
- queue ordering agrees with the named deterministic comparator;
- summary counts match the complete item inventory.

`validate_against(prediction, calibration)` recomputes the triage report from the original graph and calibration evidence and requires exact report equality. The canonical digest binds probabilities, structural effects, all configuration, ordering, segment identities and review rank.

## 4. Benchmark integration

The synthetic reconstruction CLI now evaluates all three review policies on its deterministic graph fixtures and emits:
- uncertainty / articulation / anastomosis-sensitive counts;
- the top five review items, including entropy, margin, degree, component delta and cycle loss;
- a canonical triage receipt digest.

One fixture deliberately predicts an anastomosis with 0.999 probability when the annotation is a projected crossing. This exposes the failure mode of confidence-only review: the erroneous prediction can look maximally certain. The anastomosis-probability-first policy and structural impact remain independently inspectable.

These are synthetic conformance fixtures, not an actual human-review study or measured improvement in annotation efficiency.

## 5. Proposed benchmark protocol

All targets are proposals, not achieved results.

### Review-policy baselines

Compare:
- random selection;
- entropy-first;
- minimum top-two margin;
- confidence-first;
- anastomosis-probability-first;
- structural-only (articulation / cycle participation / degree);
- structural-impact-then-uncertainty;
- uncertainty-then-structural-impact;
- an explicit cost-sensitive policy whose cost matrix is preregistered.

### Fixed review budgets

For each policy, simulate fixed budgets of 5%, 10%, 20% and 30% of predicted junctions. Use hidden annotations only *after* selection to evaluate the errors that the chosen review set would expose.

Report:
- detection/classification error remaining after simulated review;
- edge F1 and false-anastomosis count/rate remaining;
- component and cycle-rank error remaining;
- root reachability and delivery error only where demand/root labels are actually measured;
- reviewed junction count and actual review time once a human study exists;
- error reduction per reviewed item and per reviewer minute;
- AURC/excess AURC and risk at equal automation coverage;
- confidence intervals by biological replicate.

### Split discipline

- Split by organism/replicate, never adjacent image frames.
- Freeze predictions and calibrators before review-policy comparison.
- Keep calibration, policy tuning and final test sets separate.
- Evaluate image-quality, overlap-density, species and acquisition-device strata.
- Include high-confidence false fusions, missed branches, low-impact uncertain tips, true cycles and disconnected components.

### Proposed targets

- At a fixed 10% review budget, reduce residual false-fusion errors by ≥50% relative to random review and ≥20% relative to confidence-only review.
- At equal review time, reduce graph-edge and cycle-rank error by ≥20% relative to uncertainty-only ranking.
- Demonstrate improvements on replicate-held-out microscopy annotations with confidence intervals.
- Zero topology mutations and 100% source/calibration receipt completeness during triage.

These targets remain aspirational until a licensed annotated dataset and actual annotation-time study are available.

## 6. Important limitations

- Node deletion is a stress-test proxy, not the exact effect of changing a candidate contact's class.
- Cycle-rank loss does not imply a lost functional transport route.
- Articulation status depends on the supplied graph reconstruction; false or missing edges can change it.
- The current module does not calculate demand-weighted root reachability, shortest-path rerouting or failure-delivery loss after a class decision.
- Probability calibration may fail under species/device shift; no guarantee transfers automatically.
- A high anastomosis probability is not itself a false-fusion probability.
- The review queue never authorizes a connection.

## 7. Next implementation extensions

1. Add **decision-specific edge toggles**: for each candidate fusion, compute the exact graph delta for accept/reject without accepting it.
2. Extend the structural impact vector with declared root sets, terminal sets and demand-pair reachability.
3. Estimate change in failure-delivery / resilience profiles under each decision, preserving the explicit demand model.
4. Add calibration-aware, cost-sensitive Pareto ranking; retain all unweighted dimensions beside any scalar utility.
5. Run a blinded reviewer study with annotation time, agreement, and corrected graph error at fixed budgets.
6. Compare active-selection with random, uncertainty-only, topology-only and human-expert selection.

The longer-term shift is **consequence-aware evidence acquisition**: the system identifies what it does not know and measures which unknowns materially affect downstream graph claims, while leaving final judgments to independent evidence and review.
