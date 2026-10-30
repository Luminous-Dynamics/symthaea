# Hidden co-linear merge boundary detection benchmark V1

**Status:** research protocol; no detector or biological result is implemented.
**Tracking:** [SCI-CONNECTOMICS-014 / #9265](https://github.com/Luminous-Dynamics/symthaea/issues/9265).
**Independent evaluation:** #9204 spatial PSU probability sample, gated by #9260.

## 1. The topological blind spot

NEURD's published discussion identifies two limitations: its current rules cannot detect some merge errors that create a co-linear skeleton segment represented as one non-branching segment, and its current merge-focused pipeline does not repair incomplete processes. This protocol isolates the first limitation.

Reference: NEURD, Nature 2025, PMC full text, discussion: https://pmc.ncbi.nlm.nih.gov/articles/PMC11981913/ (co-linear-segment limitation).

A branch-level detector cannot find an internal junction that is absent from the graph representation. The search unit must move inside the compressed segment.

## 2. Candidate detector design

1. Resample each mesh/skeleton segment by geodesic arclength using a fixed extraction profile; preserve voxel-to-world transforms and original source coordinates.
2. Extract local, multiscale evidence: diameter and cross-sectional shape, curvature/torsion, mesh surface descriptors, local synapse density/type, compartment probabilities, and EM evidence around the segment.
3. Use independent change-point families: robust CUSUM / likelihood ratio, multiscale segmentation, Bayesian change-point models, and 3D EM boundary specialists.
4. Model expected biological variation explicitly: normal axonal taper, bouton transitions, dendritic spine zones, branchpoint-adjacent changes, genuine compartment transitions, and imaging artifacts.
5. Emit one or more split-location hypotheses with a location interval, uncertainty/abstention, feature provenance, and inspectable evidence overlay. Include a no-boundary hypothesis.
6. Keep all proposals review-only. Do not alter the graph until an expert accepts the correction and an independent validator approves the resulting branch.

A single sharp caliber change is not enough evidence for a merge. The method must distinguish an identity boundary from legitimate within-neuron morphology.

## 3. Benchmark data and sampling

- Obtain source 3D EM, mesh, skeleton, and corrected annotation revisions from an immutable, provenance-audited volume.
- Include independently adjudicated co-linear merge boundaries and matched negative cases representing real biological transitions.
- Probability-sample long non-branching segments inside PSUs selected by the independently verified spatial frame (#9260). Include segments where all methods emit no proposal.
- Search all selected segments using a frozen, blinded protocol; do not sample only detector-proposed cut locations.
- Split by volume, neuron, region, and source operation; adjacent arclength windows from one segment must not leak across train/test.
- Keep naturally occurring hidden merges separate from semi-synthetic inserted cut boundaries. Synthetic cases test exact localization mechanics, not natural prevalence.

## 4. Comparator matrix

1. Existing graph-node/subgraph morphology rules (NEURD-compatible baseline).
2. Skeleton-only sliding-window change-point detector.
3. Mesh-only multiscale geometric change-point detector.
4. EM-only local 3D boundary model.
5. EM + mesh + synapse/compartment fusion.
6. Expert search with blinded independent adjudication.
7. Fused detector with calibrated abstention, where calibration is demonstrable.

Record unavailability or incompatible input formats as NOT_COMPARABLE, not as a failed method.

## 5. Metrics

Primary:
- Probability-sampled candidate-generation recall for hidden co-linear merge families and impact-weighted recall.
- Boundary localization error in nanometres: median, 90th and 95th percentiles, plus preregistered tolerance-hit rates.
- False boundary proposals per kilometre of reconstructed cable and per neuron.
- Precision and verified hidden-merge yield at matched expert-minute budgets.
- Reviewer minutes per verified hidden merge and false-proposal review burden.

Secondary:
- Family/compartment/region calibration and coverage-risk under abstention.
- False alarms on normal taper, boutons, spines, and real compartment transitions.
- Compute, memory, image I/O, and latency per kilometre of cable scanned.
- Cross-volume/species generalization and robustness to mesh/skeleton quality shifts.

## 6. Adversarial controls

- Gradual normal axonal taper and abrupt but genuine bouton changes.
- Dendritic spine-rich / spine-poor transitions.
- True axon-to-dendrite transitions within one neuron.
- Two distinct co-linear neurons with similar caliber/orientation.
- Low-contrast and anisotropic EM, missing sections, folds and tears.
- Mesh decimation, smoothing, skeleton simplification and stale coordinate transforms.
- Missing or shifted synapse annotations; wrong compartment labels.
- Semi-synthetic hidden boundaries with exact localization, reported separately from natural cases.

## 7. Proposed targets (not results)

- Hidden co-linear merge recall ≥95% point estimate and design-adjusted lower 95% bound ≥90% on the declared search frame.
- At least 90% of adjudicated cut locations within a preregistered 500 nm tolerance and median localization error ≤200 nm on compatible challenge data; refine these thresholds against voxel resolution and adjudication uncertainty before lockbox use.
- No unacceptable increase in false accepted corrections or biologically legitimate branch removal.
- At least 20% less expert time for this error family at independently matched quality and coverage.

These goals are not directly comparable to NEURD's high-confidence rule-subset agreement; report task and error-family boundaries explicitly.

## 8. Promotion boundary

No hidden-boundary detector is promoted on synthetic localization alone. Biological recall uses the #9204 design-based PSU sample after the #9260 frame audit. Audit outcomes remain evaluator-only and must not tune detector thresholds on the lockbox.