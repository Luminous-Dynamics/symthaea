# Subterra External Benchmark Matrix v0.1

**Research cut:** 17 November 2026  
**Status:** benchmark design and literature synthesis; no external system reproduced and no superiority result claimed.

## 1. Competitive positioning

Subterra should not attempt to out-SLAM JPL NeBula or SKiD-SLAM by building a parallel mapping stack. The testable contribution is the decision/evidence layer *above* strong localization, mapping and perception systems:

> Given several plausible interpretations of the same imperfect observations, which measurement or mission policy is worth executing, remains admissible under model disagreement, and still produces a useful and recoverable result when sensing, communications or the robot itself fails?

This divides the research into orthogonal layers:

1. **Pose / map production:** established SLAM and mapping engines, including CAVERS benchmarked odometry/SLAM methods, SKiD-SLAM, and NeBula-compatible inputs.
2. **Evidence semantics:** raw/derived lineage, observed/inferred/unknown geometry, calibration and correlated source dependencies.
3. **Measurement valuation:** frontier, entropy, information-per-cost, EVSI and FEP policy proposals.
4. **Survival-conditioned policy evaluation:** one shared contingent policy evaluated across a finite set of models; no model clairvoyance; terminal outcomes truncate future value.
5. **Operational admission:** existing Subterra mission executive, field envelope, relay, rescue, operator and preservation authorities. The referee is read-only and cannot override them.

The differentiator is not a new SLAM algorithm. It is a reproducible test of whether downstream scientific decisions become better while upstream mapping quality, mission completion, hard constraints and evidence integrity remain non-inferior.

## 2. External systems and datasets

### 2.1 JPL NeBula / Team CoSTAR — primary autonomy architecture reference

JPL describes NeBula as a modular networked belief-aware autonomy solution for uncertain extreme environments. Public capability descriptions include risk-aware behavior abstraction, heterogeneous wheeled/legged/tracked/flying platforms, GPS-denied navigation with degeneracy-aware multimodal fusion, multi-robot SLAM and dense 3-D mapping, traversability, mesh communication and dynamic capability-to-role assignment.

**Comparison policy:** treat NeBula as a primary system-level reference architecture and a strong external comparator where a compatible dataset, replay interface or reproducible test is available. Do not treat the public capabilities page as a numerical benchmark result. Request exact model, version, robot, sensors, scenario, trial counts and outcome traces before a quantitative comparison.

Sources:
- https://www.jpl.nasa.gov/capabilities/robotics/nebula/
- https://www-robotics.jpl.nasa.gov/what-we-do/research-tasks/nebula-autonomy/

### 2.2 SKiD-SLAM — primary low-bandwidth distributed mapping baseline

The 2026 IEEE Transactions on Field Robotics report describes:
- distributed multi-robot LiDAR SLAM without a central server;
- three wheeled rovers in the Majang lava-tube cave and a separate two-quadruped planetary-analogue deployment;
- a surveyed 250 kbps inter-robot communication budget;
- a 320-byte SOLiD descriptor payload;
- field evidence that SKiD-SLAM merged maps under the imposed budget where the compared systems did not.

These are paper-specific results and trial conditions, not universal guarantees. The cave and surface trials are distinct deployments and must never be represented as a single five-robot test.

**Comparison policy:** use SKiD-SLAM as the required distributed-map baseline. Do not modify its score or substitute a synthetic link model for its reported field result. Where its public data/outputs permit, compare Subterra's map/evidence intake and science-task policy downstream of the same map product.

Source:
- https://sparolab.github.io/research/skid_slam/
- https://github.com/sparolab/SKiD-SLAM
- https://doi.org/10.1109/TFR.2026.3721725

### 2.3 CAVERS — primary localization and multimodal observability dataset

CAVERS provides 24 sequences (approximately 335 GB) from two structurally distinct rooms in Cueva de la Victoria, Spain. The dataset includes RGB-D/IMU, rotating 3-D LiDAR and near-IR thermal sensing, with handheld and wheeled-rover captures in darkness and artificial illumination. Most localization-targeted sequences include 6-DoF pose and velocity ground truth from an in-cave OptiTrack system at 120 Hz, described by the dataset paper as millimetre-accurate.

**Use it for:**
- trajectory/pose error and drift;
- modality and lighting degradation;
- thermal/visual/LiDAR complementarity;
- confidence calibration and degeneracy detection;
- sensor dropout, delayed observation and temporal alignment;
- validating that downstream active sensing uses the correct pose/measurement uncertainty.

**Do not use it for:** multi-robot loop closure, 250 kbps distributed communications, cave-wide topology, planetary geology, geotechnical stability, structural safety, or survival policy success. It is an indoor cave dataset with motion-capture instrumentation and bounded sequences.

The dataset is large. Start with manifests, metadata and a small rights-checked subset rather than downloading all 335 GB onto a development laptop.

Sources:
- https://doi.org/10.1109/LRA.2026.3726391
- https://doi.org/10.5281/zenodo.19367714
- https://github.com/spaceuma/cavers

### 2.4 Hazardous-exploration valuation — direct objective comparator

Srivastava et al. (9 September 2026) isolate the valuation layer in hazardous robotic exploration. Their paper compares information valuation objectives while holding the belief update, sensor, physical risk model and finite-horizon planner fixed. It reports that Shannon information remains a strong raw-information baseline and that risk-aware / behavioral valuation can reshape the information-risk frontier when observations can be truncated by failure.

**Comparison policy:** include Shannon information, risk-aware information objectives and EVSI as separate policy proposals. Report each objective's own mathematical terms. Do not adopt probability weighting as a hard safety gate, call it EVSI, or compare raw objective values with incompatible units. Hard retreat, preservation, operator and authorization constraints remain independent.

Source:
- https://arxiv.org/abs/2609.10726

## 3. Matched benchmark ladder

### B0 — independent referee conformance

No external dataset is required. Use hand-calculated fixtures and the current reference planner output.

- one model, one action and a one-step terminal policy;
- two models with different likelihoods and priors;
- outcome possible in one model but zero-probability in another;
- model-specific gate denial;
- model-specific interruption and robot-loss risk;
- path-wise resource overflow hidden by a low expected cost;
- tampered cached probabilities/posteriors;
- incomplete policy branches and schema mismatches;
- model ordering and branch ordering permutation.

Targets (proposed): exact agreement on hand-computable cases; zero accepted policies with any supplied-model gate/risk/resource violation; deterministic report under input ordering; no model-specific policy substitution.

### B1 — CAVERS localization-to-decision replay

Freeze the exact dataset version, selected sequences, download manifest, sensor calibration/metadata, frame conventions, timestamp synchronization, ground-truth availability and evaluation code. Use a sequence-level split, not random frame split.

Run upstream localization/odometry arms using the paper's reported methods where practicable. Keep each SLAM arm separate. Feed the resulting pose, covariance/quality, visibility and sensor provenance into a fixed synthetic decision case. Vary only the active-measurement policy.

Report:
- ATE/RPE and drift by sequence and motion regime;
- valid ground-truth duration/coverage;
- calibration/uncertainty coverage, not just median pose error;
- failure and abstention rates;
- downstream decision regret and useful information per joule/second/byte;
- whether degraded pose quality correctly reduces the admissible measurement set.

Do not call a SLAM pose output an independently surveyed cave map.

### B2 — SKiD-SLAM communication-constrained team replay

Freeze the exact Majang data and SKiD-SLAM implementation/version, network emulator configuration, compute hardware, robot count and resource budgets. Reproduce the paper's 250 kbps inter-robot budget and its descriptor payload accounting. Record packet loss, latency, outages, map merge status, transmitted bytes, CPU and memory.

Only after reproducing the upstream map result, compare:
- fixed science-task priority;
- frontier/coverage task allocation;
- entropy-greedy allocation;
- EVSI;
- survival-conditioned policy;
- robust shared-policy referee;
- operator-defined plan.

Report mission decision quality, unique science coverage, duplicated effort, interruption/robot-loss exposure, return/recovery and evidence delivery. The policy layer must receive the same upstream mapping products and observation catalogue for each method.

### B3 — model disagreement and domain transfer

Use CAVERS only for the modality/pose aspects it actually observes. Add independently authored synthetic geological/sensor models and site-disjoint public cave analogues only when their rights, calibration and ground-truth metadata are established.

Perturb:
- dust/low light and specular/wet surfaces;
- sensor calibration and range bias;
- common pose/frame error shared by multiple derived products;
- correlated acquisitions and processing duplicates;
- topology disagreement and incomplete branches;
- relays, partitions, stale messages, clock skew and store-and-forward;
- model family shift and an explicit open-world unknown class.

Primary outputs: Brier score, log loss, classwise calibration, risk–coverage, selective risk, abstention quality, worst-case policy value, model-wise interruption and robot-loss risk, and false-clear count.

### B4 — analogue field campaign

Only after B0–B3 and site-specific ODD/engineering gates. Start with reversible, uncrewed, non-destructive mapping and observation. Use independent metrology, surveyed control/checkpoints, time-synchronized sensor records and pre-registered hypotheses. Never infer human-entry, ground support, mine safety or planetary protection from a robot benchmark.

## 4. Matched comparator matrix

| Arm | Role | What it can establish | What it cannot establish |
|---|---|---|---|
| NeBula / CoSTAR | Strong system architecture and analogue field reference | Integrated risk-aware autonomy and multi-robot deployment when exact data are available | Our benchmark score without matched trials |
| SKiD-SLAM | Strong 250 kbps distributed LiDAR SLAM comparator | Distributed map merge under its documented mission conditions | Science-decision value or geotechnical safety |
| CAVERS benchmarked SLAM/odometry | Ground-truth localization and multimodal comparison | Pose error and observability under its measured cave sequences | Multi-robot comms or mission survival |
| Frontier / nearest-frontier | Minimal exploration baseline | Coverage-efficiency baseline | Decision quality under mission-specific loss |
| Shannon information gain | Information-theoretic baseline | Expected entropy reduction | Whether acquired information changes the decision enough to justify hazard |
| Risk-aware information valuation | Recent hazardous-exploration objective | Information/risk trade-off under its assumptions | External authority or calibrated real-world safety |
| Single-model EVSI | Current Subterra value-of-information baseline | Decision-relevant measurement value under one supplied model | Robustness to model misspecification |
| Survival-conditioned planner | Current finite-horizon baseline | Terminal-outcome truncation and resource-aware contingent plans under one model | Validity across plausible model families |
| Robust shared-policy referee | New work item | Independent evaluation of one unchanged policy across a finite model set | Proof that the true model is in that set or a universally safe policy |

## 5. Primary success criterion

The initial superiority hypothesis is:

> At equal upstream map quality and resource budget, Subterra's evidence-aware, survival-conditioned decision layer reduces held-out mission decision regret or increases accepted decision value, while remaining non-inferior on mapping quality and mission completion and improving calibration, abstention or recovery under model disagreement.

Pre-register a minimum 20% relative reduction in mean decision regret versus the strongest matched non-Subterra policy as a **stretch research target**, but only report it if:
- the baseline is faithfully reproduced;
- sites/sequences are held out at the correct independent unit;
- the loss matrix is frozen before test evaluation;
- all arms share the same candidate actions, observations, resource/compute budgets and hard gates;
- paired confidence intervals and per-site/per-condition results are shown;
- no hard safety, evidence or preserve invariant regresses.

If the target is not met, report the result and the Pareto frontier. Non-inferiority on upstream mapping and safety invariants is not tradeable for a higher science score.

## 6. Statistical and reproducibility rules

- Split by site/sequence/acquisition lineage, not frames or derived files.
- Use separate Training, Calibration and Evaluation partitions.
- Keep repeated measurements from one physical acquisition in one dependence component.
- Report per-sequence results, aggregate uncertainty and failure denominators.
- Use paired comparisons on the same cases; bootstrap independent sites/sequences, not individual frames.
- Freeze all exclusions and missing-data rules before Evaluation.
- Preserve all failed, interrupted, incomplete, and abstained runs.
- Record exact model, dataset, code, configuration, hardware, toolchain, policy and score digests.
- Do not compare raw scores across different loss matrices or incompatible physical tasks.

## 7. New research hypotheses

1. **Map quality is not decision quality.** A less complete map can be more useful if it resolves the uncertainty that flips a critical decision.
2. **Model disagreement is a resource signal.** The system should seek discriminating measurements or abstain when plausible models imply different actions.
3. **Evidence lineage is a physical property.** Three derived maps from one scan do not constitute three independent witnesses.
4. **Robust policies must be non-clairvoyant.** The same contingent policy is evaluated across models; only observable outcomes select branches.
5. **Survival truncates value.** Information available only after a likely terminal failure is not useful information.
6. **Communications are a sensing resource.** Relay placement and evidence delivery should be valued jointly with new measurements, not as separate infrastructure housekeeping.
7. **Infrastructure may sense itself.** Where lawful fibre/DAS already exists, use it as a wide-area screening layer, then direct robots only to unresolved/high-value segments. DAS remains a separate calibrated sensing method; it is not a universal detector or an actuation authority.

## Claim ceiling

This matrix is a research and benchmark plan. It does not report reproduced external scores, CAVERS ingestion, SKiD-SLAM execution, NeBula integration, a field campaign, or a SOTA result. A finite synthetic PASS is not physical, mining, cave, planetary, safety or regulatory qualification.
