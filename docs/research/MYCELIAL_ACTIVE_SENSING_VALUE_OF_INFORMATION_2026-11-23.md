# Mycelial Active Sensing: Decision Value, Not Entropy
**Date:** 2026-11-23
**Status:** Research protocol and software contract; no microscope actuation or prospective performance claimed

## 1. Paradigm shift

The next step after uncertainty-aware reconstruction is not simply "take another image where entropy is high". It is to ask:

**Which next observation is expected to change the best scientific or annotation decision enough to justify its acquisition cost and sample impact?**

This turns annotation triage into a sequential experimental-design problem. The objective is decision value, not uncertainty reduction for its own sake.

A contact may be highly uncertain but irrelevant to the question being asked. Another contact may have lower entropy but determine whether two demand regions are connected, whether a critical component survives a disruption, or whether a temporal event is real. A useful acquisition policy should rank those cases according to the decision's declared loss.

## 2. Exact Bayes-risk formulation

Let:
- (h) be one feasible graph hypothesis;
- (P(h)) be its posterior probability;
- (d) be a permitted decision (e.g. accept/reject a candidate for *analysis*, request adjudication, or leave unresolved);
- (L(d,h)) be the loss of decision (d) if (h) is the true graph;
- (a) be a candidate acquisition action;
- (o) be a possible outcome of that action;
- (P(o mid h,a)) be a calibrated action-response likelihood;
- (C(a)) be the acquisition cost expressed on the same loss scale.

Baseline Bayes risk:

[
R_0 = min_d sum_h P(h)L(d,h)
]

Predictive probability of an outcome:

[
P(o mid a) = sum_h P(h)P(o mid h,a)
]

Expected Bayes risk after acquisition:

[
R_a = sum_o min_d sum_h P(h)P(o mid h,a)L(d,h)
]

Gross value of information:

[
operatorname{VOI}(a) = R_0 - R_a
]

Net decision value:

[
operatorname{NVOI}(a) = R_0 - R_a - C(a)
]

The no-acquisition option is always available and has net value zero. The selector should recommend an action only when its net value is strictly positive.

This matters because entropy reduction is only a proxy. It can spend acquisition time resolving graph details that do not affect any decision, while overlooking a lower-entropy observation that determines a consequential choice.

## 3. Implemented Rust contract

The companion `mycelial_active_sensing.rs` module implements the bounded evaluator over the exact graph-hypothesis posterior.

### Inputs
- The original annotated skeleton and candidate inventory.
- A posterior that is independently recomputed and checked against those exact inputs.
- A finite set of decisions, each with one loss for every feasible posterior hypothesis.
- Candidate acquisition actions, each with:
  - typed acquisition kind (additional z-plane, high-resolution crop, alternate illumination, new timepoint, human annotation or other);
  - acquisition-response model ID, version and digest;
  - cost evidence digest and scalar cost;
  - a complete list of possible outcomes;
  - (P(omid h,a)) for every posterior hypothesis and outcome.

### Validation
- The posterior must match independent recomputation from the frozen skeleton and candidate inventory.
- Decision rows must match the exact hypothesis order and candidate set.
- Every decision must use one common loss-scale ID.
- Every action must use that same cost scale.
- Likelihoods must be finite, in [0,1], and sum to one across the action's outcomes for *each* hypothesis within a declared tolerance.
- Duplicate IDs, malformed digests, missing rows, invalid costs and over-budget evaluations fail closed.
- Action/model/cost metadata and the complete posterior constraint inventory are committed into canonical digests.
- The report is independently recomputable and explicitly advisory.

No likelihood vector is silently renormalized. A model that is not a valid conditional distribution is rejected rather than repaired by the ranking code.

### Current claim boundary
This is a decision-value evaluator for declared likelihoods and losses. It does **not** yet learn the acquisition-response model, estimate the cost model, operate a microscope or establish that a suggested acquisition is physically safe. Action feasibility and hard constraints must be enforced upstream by a verified device adapter before any future actuation.

## 4. Acquisition-response models: what must be calibrated

For every action class, estimate (P(omid h,a)) using frozen retrospective replay before prospective use.

Examples:
- **Additional z-plane / z-range:** hide known planes from a complete stack; ask whether the model predicts which hidden slice resolves contact identity or crossing ambiguity.
- **High-resolution crop:** downsample or crop known observations; test whether the proposed crop would resolve tip position, overlap or anastomosis.
- **Alternate illumination / focal strategy:** use paired acquisitions with known correspondence; model which uncertainty classes are resolved and which artifacts are introduced.
- **New timepoint:** mask a known timepoint in a longitudinal sequence; measure event identity, persistence and growth-front updates.
- **Human annotation:** use blinded replicate annotations; model inter-reviewer disagreement, time cost and adjudication outcomes.

The likelihood model must be trained and calibrated on biological replicates disjoint from evaluation replicates. The replay must preserve acquisition batches, focal-plane ordering, spatial context and time. Randomly splitting neighbouring crops or frames is leakage.

For categorical action outcomes, report:
- joint negative log-likelihood;
- multiclass Brier score;
- classwise / expected calibration error with confidence intervals;
- reliability diagrams by organism, device, assay, uncertainty class and action type;
- calibration drift under illumination, focus, depth and signal-to-noise changes.

For continuous action outcomes (e.g. expected localization improvement or time saved), use proper distributional scores such as CRPS and interval coverage; do not discretize only to make the action model look calibrated.

## 5. Benchmark comparison

Compare policies at equal acquisition budgets and with identical held-out specimens:

1. Random action selection.
2. Highest candidate entropy.
3. Highest expected entropy reduction.
4. Confidence-only review queue.
5. Largest topology-change score (e.g. component / cycle impact).
6. Expected Bayes decision-risk reduction without cost.
7. Net value of information, including acquisition cost.
8. Perfect-information upper bound (diagnostic ceiling, not an achievable policy).

Report:
- realized decision loss after replay outcome;
- regret relative to the best decision with the revealed ground truth;
- realized net value per minute and per photon-dose unit;
- number of false fusions and missed fusions after a fixed review budget;
- percentage of acquisitions that actually change the optimal decision;
- calibration of predicted versus realized value;
- sample-health constraints and acquisition failures;
- wall-clock, memory and operator minutes.

A policy must outperform entropy and random baselines under a fixed budget, with cluster-bootstrap intervals over biological replicates. An improvement on the same images used to tune the likelihood model does not count.

## 6. Proposed targets (not measured results)

After the first retrospective benchmark and sample-size analysis, target hypotheses are:
- At 10% reviewer/acquisition budget, reduce residual false-fusion decisions by at least 50% relative to random review.
- Improve realized decision loss by at least 20% relative to entropy-only selection at the same acquisition cost.
- At least 70% of selected acquisitions should either change the best decision or materially narrow its confidence interval; publish both counts.
- Predicted action value should have calibration slope 0.9–1.1 on held-out replicates, with interval coverage reported.
- No increase in acquisition-related sample damage or predeclared dose-limit violations.
- At equal net decision loss, reduce acquisition minutes / photon dose by at least 20% relative to a fixed repeated-imaging protocol.

These are aspirational benchmark targets. They are not current performance claims and should be frozen only after baseline reproduction and statistical power analysis.

## 7. Safety and device-boundary contract

A positive NVOI is not a command to actuate. Before any microscope adapter is considered:
- verify device-specific z-range, stage limits, objective working distance, illumination wavelengths and intensity limits;
- enforce phototoxicity / photobleaching budgets and sample viability constraints as hard constraints, not merely as scalar costs;
- require an independent acquisition receipt and exact source-image digest for every new observation;
- preserve a pre-acquisition state and recover cleanly from unknown device outcomes;
- require human approval for any action outside a pre-authorized, validated envelope;
- keep a stop / abort path that does not depend on posterior correctness.

The current module is advisory and contains no actuator, device API or mutation path.

## 8. Current research landscape and industry benchmark boundary

### Academic direction

The March 2026 *npj Imaging* smart-microscopy review defines smart microscopy as real-time analysis, feedback control and automated actuation, and organizes strategies around quality, event, target, information and outcome objectives. It also emphasizes competing constraints among resolution, speed, signal-to-noise, field of view and sample health. This proposal is intentionally the decision-theoretic layer upstream of any future verified actuation; it does not claim to have completed a smart microscope.

The January 2026 multi-agent smart-microscopy framework argues for integrating observation and hypothesis revision rather than running AI only after acquisition. Our contribution should be measurable: decision regret, net decision value, calibration and provenance over a frozen acquisition replay.

### Commercial / open tooling snapshot (checked 23 November 2026)

| Platform | Verified current strengths | What Symthaea should do |
|---|---|---|
| ZEISS arivis Pro 4.6 (released 2 Oct 2026) | AI training workflows, cloud training hand-off, external 3D ONNX semantic models, 2D/4D interaction and a microscopy copilot | Import/export and compare segmentation outputs; add explicit graph hypotheses, posterior constraints, audit digests and decision-value acquisition policy |
| Imaris 11.1 (released 8 Sep 2026) | Native CZI reader, expanded file recognition and Imaris Arena workflow improvements; its product family includes filament tracing and quantitative 3D visualization | Treat Imaris as a practical reconstruction / visualization comparator; test round-trip of graph and physical-unit metadata |
| Fiji SNT | Open multidimensional tracing and editing, automated/semi-automated routines, morphology analysis, scripting and experimental TB-scale data support; explicitly usable for filamentous structures beyond neurons | Build an open adapter and reproduce equivalent tracing/graph metrics rather than requiring users to abandon a mature tool |

We should not claim current product gaps simply because a feature is absent from a public release-note page. The differentiator is the **assurance layer across tools**: the same image/graph digest, explicit uncertainty set, feasible graph posterior, causal topology changes, measured function, and reproducible decision policy can be reviewed independently of which software produced the initial segmentation.

## 9. Data and validation sequence

1. Finish exact-head formatting, compile, tests and Clippy on the renderer-neutral contract.
2. Implement action-response retrospective replay on known z-stack / time-series data; publish held-out replicate splits.
3. Compare entropy, confidence, topology and Bayes-risk policies at fixed cost.
4. Calibrate cost evidence using acquisition time, photon dose, sample impact and human annotation time.
5. Publish negative controls, posterior-predictive checks and independent review before integrating any device adapter.
6. Only after all preceding evidence is qualified, consider a read-only acquisition recommendation bridge into a separately permissioned microscope control system.

## 10. Scientific integration

The value of an acquisition should be conditioned on the *question*:
- reconstruction: resolve a false crossing versus true junction;
- topology: change in β₀ / β₁, cycle witnesses or component attribution;
- resilience: restore or break a critical demand path under a frozen disruption;
- temporal inference: distinguish persistent contacts from transient proximity;
- functional prediction: resolve the graph feature most predictive of independently measured flow, tracer or microbial dispersal.

Topology and geometric directness alone do not prove biological transport. The functional-observation contract and the functional validation benchmark must remain the independent outcome source.

## 11. References

- Rates et al. (2026), *Smart microscopy: adaptive microscope control to improve the way we see life*, npj Imaging 4, 14. https://doi.org/10.1038/s44303-026-00145-y
- Kesavan & Nordenfelt (2026), *From observation to understanding: A multi-agent framework for smart microscopy*, Journal of Microscopy. https://doi.org/10.1111/jmi.70063
- Sten et al. (2025), *Three-dimensional graph reconstruction of filamentous structures from z-stack images*, Ecological Informatics 90, 103317. https://doi.org/10.1016/j.ecoinf.2025.103317
- ZEISS arivis Pro 4.6 release notes (2 October 2026): https://knowledge.zeiss.com/rms/en/arivis-pro/release-notes/arivis-pro-4-6
- Imaris release notes (Imaris 11.1, 8 September 2026): https://imaris.oxinst.com/support/imaris-release-notes/
- ImageJ SNT documentation: https://imagej.net/plugins/snt/index
