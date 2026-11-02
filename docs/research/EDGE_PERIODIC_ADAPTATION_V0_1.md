# Periodic edge adaptation benchmark fixture v0.1

Tracking: [#9920](https://github.com/Luminous-Dynamics/symthaea/issues/9920)  
Implementation branch: `research/edge-periodic-adaptation-v0.1`  
Parent stack: [HARTH LOSO receipt PR #9918](https://github.com/Luminous-Dynamics/symthaea/pull/9918)

## Research motivation

On-device adaptation is not automatically a win. **BoTTA: Benchmarking On-device Test Time Adaptation** (SenSys 2026, DOI [10.1145/3774906.3800498](https://doi.org/10.1145/3774906.3800498)) evaluates adaptation under:
- very few target samples;
- incomplete category exposure;
- diverse distribution shifts;
- compound shifts within the same sample.

It also profiles real Raspberry Pi 4B and Jetson Orin Nano systems. In its SHOT example, 512 adaptation samples produce a reported 2.25× accuracy gain in that experiment, while requiring 3× the latency and energy of inference alone. The heavier SURGEON method is reported to be dramatically more expensive. Those values are from BoTTA's models, tasks and devices; they are not transferable to Symthaea or this synthetic fixture.

BoTTA's move toward **periodic adaptation** is particularly relevant for resource-constrained devices. This fixture tests a related decision: when a device should modify persistent state based on unlabeled target evidence, and when apparent shift is only a change in class composition.

MLPerf Tiny v1.4 remains the complementary fixed-task reference for matched static inference quality, latency and energy. This benchmark is an additional temporal/adaptation track, not a replacement for MLPerf or a claim of direct MLPerf equivalence.

## Candidate-facing API and temporal order

The candidate implements `UnsupervisedEdgeModel`:

- `predict(&PublicTargetSample)` receives sample ID and Q15 feature pair only;
- `adapt_after_batch(&PublicTargetBatch)` receives the just-scored batch of **unlabeled** features;
- `state_sha256()` reports the current candidate model-state commitment.

No labels, class-coverage configuration, target prior, scenario seed, transformation parameters or evaluator score are present in these public types.

For every batch (t):
1. The evaluator generates public features and private labels.
2. The model predicts every public sample using the current state.
3. The evaluator scores those predictions against hidden labels.
4. Only after the complete batch is scored does the model receive the unlabeled batch for an optional adaptation update.
5. The update receipt commits the prior and next model-state digests; only future batches may use the updated state.

This avoids a common temporal leak: adapting on a batch and then reporting predictions on that same batch as though they were out-of-sample.

## Synthetic data and shift controls

Source classes are two symmetric Q15 prototypes on the x-axis, with evaluator-only binary labels. The target generator can apply, after a frozen shift-start batch:
- no shift;
- additive translation;
- right-angle rotation;
- scale change;
- compound translation + rotation + scale + bounded noise.

The target class-0 prior is explicit and may represent balanced, skewed or single-class exposure. The generator is deterministic and keyed by scenario seed and sample identity, so different candidate update cadences receive identical counterfactual features.

Only the right-angle rotations are implemented in v0.1; this is deliberate to keep the synthetic transformations integer/fixed-point and reproducible. This fixture does not claim to model the full geometry of real wearable sensors.

## Included controls

The first model is deliberately simple: a fixed linear decision boundary on x, with an optional two-component mean offset.

- **Frozen/no adaptation:** never changes its state.
- **Periodic mean-centering:** after a configured number of completed batches, estimates the unlabeled batch mean and blends it into the persistent offset.
- **Shift-triggered mean-centering:** computes the current unlabeled batch mean and updates only when its Manhattan distance from the current offset exceeds a threshold.

These are transparent diagnostic controls, not implementations of SHOT, TENT, SAR, NOTE, T3A or SURGEON. They test the benchmark's causal and cost accounting, and demonstrate a key identifiability issue: an unlabeled class-prior shift can move the batch mean even when the sensor/domain transform is unchanged. Blind centering may then damage future predictions. The class-zero-only negative control demonstrates this failure mode.

The simple x-axis classifier is not intended to compete with HART, SelfPAB, TFLite Micro or TinyML models.

## Metrics and update accounting

The result retains:
- overall accuracy;
- class-0 and class-1 support and recall;
- balanced accuracy averaged over classes present in that batch/episode;
- worst present-class recall;
- all per-batch confusion matrices and quality metrics;
- number of update decisions considered and updates applied;
- declared compute-operation estimate;
- declared persistent state-write bytes;
- initial, previous, next and final state digests;
- deterministic public trace and canonical result receipt digests.

The operation estimate and state-write counters are synthetic/declared algorithm-track values. They are not measured cycles, wall-clock latency, SRAM use, flash writes or energy. The System track must profile the compiled candidate and complete application event on a named physical target.

## Recommended confirmatory matrix

For later model adapters, freeze and report a complete Cartesian or preregistered fractional matrix across:
- target adaptation samples: 16, 32, 64, 128, 256, 512;
- category exposure: one class, two classes, all classes;
- class prior: balanced, skewed, extreme skew;
- shift type: no shift, sensor translation, scale/rotation, missingness, noise, compound;
- shift severity: low, medium and high;
- adaptation cadence: no update, every batch, every 4 batches, every 16 batches, trigger-based;
- online update budget and persistent write budget.

Keep every cell's data and per-class support visible. A single mean can hide class-collapse or harm under compound shift.

Required future baselines include pinned, independently reproducible adapters for no adaptation, BN adaptation, T3A, TENT, SHOT and other BoTTA methods where their assumptions fit. Do not label our simple centering control as one of those methods. For every external baseline pin code revision, weights, preprocessing, target buffer policy, update cadence, optimizer/state, precision and training/adaptation sample budget.

## Aspirational performance targets (not achieved)

The benchmark should aim for one of two meaningful wins:
1. **Adaptation quality:** improve worst-class or subject-group recall by at least 2 absolute points over the strongest matched periodic TTA baseline, with a paired 95% confidence interval excluding zero and no unreported regression under single-class exposure.
2. **Resource-constrained quality:** match the strongest baseline within 1 absolute quality point while reducing measured complete adaptation-event energy by at least 2× on the same target, with no worse worst-class recall and no p95 deadline violation.

A third research target is to avoid harmful updates: under a fixed update budget, achieve no worse than frozen inference when the shift is not identifiable from unlabeled target data. That is as important as a large score on a favorable shift.

These are preregistered aspirations. No result from this fixture can establish them by itself.

## Trust, status and claim ceiling

- The public candidate API omits evaluator labels and shift configuration, but a Rust trait is **not** a security sandbox. Confirmatory evaluation requires separate candidate/evaluator processes with OS-level permissions and independently replayable receipts.
- The synthetic data is a controlled algorithm fixture, not a substitute for HARTH, MIMII, SHD/SSC or real sensor-stream evaluation.
- The three included controls do not reproduce published TTA implementations.
- Operation/write figures are declarations or estimates, not physical energy/latency.
- Twelve synthetic tests are authored but have not been compiled or run. No HAR score, adaptation win, hardware result or SOTA improvement is claimed.
