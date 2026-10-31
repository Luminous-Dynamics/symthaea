# STDL Spatiotemporal-Training Baseline Audit (v0.1)

**Status:** pinned public-code audit; no training or score reproduction  
**Tracking:** NEURO-SILICON-007 / #9394; temporal suite #9395  
**Audit date:** 2026-10-31

## 1. Why STDL is a required comparator

Ma, Chen, Tan, and Wu, *Spatiotemporal Decoupled Learning for Spiking Neural Networks*, IEEE Transactions on Neural Networks and Learning Systems, 37(9), 4597-4611 (September 2026), DOI [10.1109/TNNLS.2026.3671461](https://doi.org/10.1109/TNNLS.2026.3671461), propose STDL to decouple both spatial and temporal dependencies during SNN training.

The paper abstract describes:
- spatial partitioning into smaller subnetworks, each trained with an auxiliary network;
- selecting the largest useful subset of subsequent-network layers under an auxiliary-memory constraint to preserve inter-subnetwork synergy;
- temporal decoupling for online learning;
- evaluations on seven static and event-based vision datasets;
- comparable accuracy to BPTT with reduced GPU memory, including a reported 4x reduction on ImageNet.

These are paper-reported claims, not results reproduced by Symthaea. STDL is complementary to BrainTrace: STDL combines spatial local subnetworks and truncated temporal credit; BrainTrace provides model-agnostic eligibility-trace online learning.

## 2. Pinned public repository and blobs

- Repository: [ChenxiangMA/STDL](https://github.com/ChenxiangMA/STDL)
- Inspected main commit: `e5e483332ade02c94d8354583ab46aa63e99ecf1` (2025-06-27; README update)
- The public code commit predates the September 2026 journal publication. The exact correspondence between this public snapshot and the final paper's experimental runner is **not established**.
- README: blob `af0b4b84da93abce506f04b88f12434b8fc641c7`
- Experiment script matrix: `run.sh`, blob `87fd14f6db82bbe577ad6642f33306d74620183c`
- Main trainer: `main_train.py`, blob `e461fc6b0ae9c9bf090d9cace25ef815f8f4eef0`
- Spiking wrapper: `models/spk_model.py`, blob `c30b88ffcd764ec0a38620c45d7238f53c70cfa0`
- Local ResNet / auxiliary heads: `models/spk_cifar_resnet_local.py`, blob `0d5a0d42c33f45dfa61e1518722dd8e277097c9e`
- Subnetwork and auxiliary-memory topology: `models/configs.py`, blob `47222fbb04391e30a7f6a33b58d179fdfa00ebd6`
- LIF state and detach semantics: `spk_neuron/neuron.py`, blob `fd941ee6c66c4bf0ab8ab3c4ba20072b45298dcb`
- Dataset split mapping: `load_dataset.py`, blob `99a25f5c94ee772d7a70d45bbf3c6bc4c8777234`
- Seed, reset and utility code: `tools/utils.py`, blob `e873effc8c7bb0e74886b2821ca381c5a56975aa`

The repository README lists PyTorch but does not pin a full dependency lock in the inspected tree. A reproduction must lock Python, PyTorch, torchvision, CUDA, cuDNN, data transforms and hardware explicitly.

## 3. What the public implementation actually does

### Temporal path

`models/spk_model.py` resets the network's state at the start of each model forward, then iterates through the configured time window. It accumulates detached per-timestep logits. The neuron implementation exposes an explicit `detach_mem` switch; with it enabled, the prior membrane value is detached before the next charge update. The example `run.sh` STDL configurations enable `--detach_mem` and `--detach_reset`.

This is a concrete temporal-gradient boundary. It is not identical to BrainTrace eligibility traces or to a continuous streaming model that carries state across independent host calls: the wrapper resets state at the beginning of each forward call.

### Spatial path

`LocalResNet.forward` traverses the backbone and applies an auxiliary classifier at configured local-module boundaries. During training it backpropagates the local loss and detaches the intermediate backbone activation before continuing through later modules. The final classifier also backpropagates its final loss. The local auxiliary head can include selected layers from subsequent network stages.

For example, the inspected ResNet18 `mem_1800` config uses local boundaries at (stage 1, block 0) and (stage 2, block 0); its auxiliary-network map selects later layer indices `[6,7,8]` for the first boundary and `[6,8]` for the second. ResNet19 `mem_4400` uses its own selected later-layer subsets. These are code-defined topology choices; their labels are not a general byte budget until all tensors, optimizer state, activations and runtime allocations are measured.

### Training and baseline controls

The public scripts show:
- STDL's memory-constrained auxiliary configuration (`mem_1800`, `mem_4400`, `mem_5001`, `mem_2600`, depending on architecture/dataset);
- ELL and DECOLLE local-learning controls;
- BPTT remains the full-gradient reference described by the paper; confirm the exact paper runner's matched BPTT config before comparison.

The README/run scripts list CIFAR-10, CIFAR-100, SVHN, ImageNet, and CIFAR10DVS examples. CIFAR10DVS uses a VGG11 example with a 10-step input window; ImageNet uses ResNet34 pretraining at one step followed by a four-step fine-tune example. These are candidate public-code workloads, not evidence that every listed dataset was part of the paper's final seven-task protocol.

## 4. Public-code reproduction caveats

1. **Split contamination / model selection.** In the inspected `load_dataset.py`, CIFAR-10, CIFAR-100 and SVHN use their official test split to populate `val_loader`; CIFAR10DVS and HARDVS similarly use `train=False` for the loader named validation. `main_train.py` evaluates this loader every epoch and selects/saves the best checkpoint by its score. For these code paths, that dataset split is used for iterative model selection and must not also be reported as a one-shot confirmatory test. Reproduction should create a separate development split and preserve a locked confirmatory split.
2. **ImageNet naming.** ImageNet's `valdir` is used for `val_loader` and checkpoint selection. Treat it as development/model-selection data in a strict replication unless a separate test source is supplied.
3. **CUDA-only.** The main training entry asserts that CUDA is available and wraps the model in `DataParallel().cuda()`. This is not an immediately executable CPU baseline on a small laptop. A CPU port would be a separate implementation and must not be conflated with the pinned baseline.
4. **Memory reporting.** The trainer logs PyTorch allocated and peak allocated CUDA memory, but this does not by itself account for CPU RSS, all reserved/cached memory, data-loader buffers, host preprocessing, all auxiliary optimizer state, energy, or end-to-end deployment cost.
5. **Seed reporting.** The inspected code defaults to seed 9 and calls a helper that seeds Python, NumPy, PyTorch and CUDA and enables deterministic cuDNN. The published experiment matrix still requires multiple independent seeds, their exact seed list, and aggregation methodology; a default seed is not a multi-seed result.
6. **Runner lineage.** The inspected repository head predates the journal publication. Do not assume it is the exact version used to produce every published figure or the final accepted paper.

These are source-specific reproducibility caveats. They do not show that the paper's results are invalid.

## 5. How to compare STDL fairly against Symthaea

### Task applicability

- **Direct first parity:** event-based vision, especially CIFAR10DVS and other pinned event-vision datasets where the paper's model/input pipeline can be recreated.
- **Static vision parity:** CIFAR-10, CIFAR-100, ImageNet, SVHN where task/dataset licences, transforms and split semantics are locked.
- **Not a drop-in control for every temporal synthetic task:** scalar sparse-event, DMS and cue accumulation streams require a common input/task adapter and matched training semantics before STDL is a meaningful comparison.
- Keep BrainTrace as the named online-learning / long-horizon credit comparator. Keep GRU/LSTM/TCN and BPTT as compatible recurrent/gradient controls.

### Metrics

Report a Pareto table rather than a single blended score:
- confirmatory accuracy / task quality and worst-class performance;
- training memory by category: parameters, auxiliary-subnetwork weights, optimizer state, activations, neuron state, temporary buffers, peak allocated and peak reserved device memory, CPU RSS;
- local partition count, boundary positions, auxiliary layer subsets and their measured bytes;
- training throughput, wall-clock-to-fixed-confirmatory-quality, total training time and failure/convergence rate;
- energy per training sample / correct update and full-run joules with a calibrated boundary;
- temporal-length scaling of peak memory and update time;
- event-inference batch-1 p50/p95/p99 and continuous-stream idle energy only after a separate streaming-preserving wrapper is specified and validated.

STDL's own reported 4x ImageNet GPU-memory reduction is an external target/reference, not an expected Symthaea result and not comparable until the model, precision, training batch, optimizer, hardware and memory boundary are matched.

### Mandatory controls

1. Pin the exact public STDL and BrainTrace commits and dependency environments.
2. Recreate the same train/development/confirmatory partitions before augmentation/windowing.
3. Use separate fixed development and one-shot confirmatory splits; never select epochs on the confirmatory set.
4. Run at least 10 paired pilot seeds and at least 30 paired confirmatory seeds, or a preregistered power-based alternative.
5. Match data, input time windows, precision, training examples, optimizer update budget, batch size and hardware.
6. Compare STDL `mem` topology variants against ELL, DECOLLE and BPTT controls. Record exact auxiliary layer maps, not just the config string.
7. Profile peak memory, throughput and energy over the complete training/system boundary. Do not infer energy efficiency from spike sparsity or GPU memory alone.
8. Retain failed/non-converged runs and report quality at fixed resource budgets as well as resources at fixed quality.

## 6. Implication for Symthaea research

STDL suggests a promising third design axis beyond fast/slow state and time-aware input integration: **budgeted spatial decomposition with adaptive auxiliary context**.

Candidate research hypotheses:
- use HDC projections to decide which downstream subnetwork context is worth retaining in an auxiliary learning path;
- choose local-module boundaries according to an explicit memory/compute/gradient-synergy budget;
- use causal fast/slow state summaries to pass cross-boundary information without retaining full activation history;
- compare fixed STDL-style local boundaries with event/novelty-triggered boundaries;
- use Symthaea as the online temporal state system while preserving an STDL-like local learning control as a separate optimizer/training arm.

These are hypotheses for experiments, not implemented Symthaea capabilities.

## 7. Reproduction status and claim ceiling

- Public source files inspected at exact commit/blob pins above.
- No STDL package installed or executed.
- No public STDL benchmark run reproduced.
- No dataset downloaded, training, GPU-memory measurement, energy measurement or head-to-head score performed.
- No paper metric imported as a Symthaea score.
- The STDL runner's CUDA requirement prevents a direct CPU-only local run.
- Public-code model-selection use of dataset test splits must be addressed in a confirmatory reproduction.

A successful comparison supports conclusions only for the pinned tasks, model/training implementations, dataset splits, precision and hardware. It does not establish universal SNN superiority or deployed neuromorphic efficiency.
