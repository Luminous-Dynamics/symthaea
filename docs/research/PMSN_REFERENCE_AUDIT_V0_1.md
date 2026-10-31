# PMSN Multi-Compartment SNN Reference Audit (v0.1)

**Status:** pinned public-code source audit; no code execution or score reproduction  
**Tracking:** NEURO-SILICON-009 / #9459 and NEURO-SILICON-007 / #9394  
**Audit commit:** `6828f8f20a93ea7eff513fc6286f981152d2ae17`

## 1. Why PMSN is a required comparator

Chen et al., *PMSN: A Parallel Multi-Compartment Spiking Neuron for Multiscale Temporal Processing*, IEEE Transactions on Neural Networks and Learning Systems, published 13 July 2026, DOI [10.1109/TNNLS.2026.3708510](https://doi.org/10.1109/TNNLS.2026.3708510), introduce a multi-compartment spiking neuron with configurable internal state and parallelization across time.

The abstract reports:
- interacting substructures for multiple timescales;
- flexible compartment/state size;
- two parallelization methods that decouple temporal update dependencies;
- more than 10× acceleration and 30% accuracy improvement over LIF on the paper's Sequential CIFAR-10 experiment at comparable computational cost;
- a neuromorphic-hardware implementation.

These are paper-reported results for the authors' tasks, implementation, hardware and measurement boundary. They are literature anchors only, not Symthaea results.

## 2. Pinned repository and blobs

- Repository: [xychen-comp/Parallel-Multi-compartment-Spiking-Neuron-PMSN](https://github.com/xychen-comp/Parallel-Multi-compartment-Spiking-Neuron-PMSN)
- Inspected commit: `6828f8f20a93ea7eff513fc6286f981152d2ae17`
- README blob: `9900849ef5303159b73f67c61a9dbee862f6c31c`
- Sequential CIFAR-10 (1024 steps) trainer blob: `seqcifar1024/CIFAR_PMSN.py`, `6e4e4a1fb4bfd61ef456df96f9cc170e7d061ad3`
- PMSN neuron/kernel blob: `seqcifar1024/PMSN_neuron.py`, `bcc4005c306554e5c6cc8329d75b5d6b34b26ffd`
- Training utilities blob: `seqcifar1024/lib.py`, `35b5232de2a2300748ca657b016175fb91e5c079`
- S-MNIST / PS-MNIST trainer blob: `S-MNIST/MNIST_PMSN.py`, `4e7524996adb695c143154816b6ac70c7279e105`
- Sequential CIFAR run script blob: `seqcifar1024/run.sh`, `36ea463690fd80c4aa69977a49eb4adf0804c345`
- License blob: `117ef6de35f929665099d9d3ba5e75a6abc90b42` (MIT)

The inspected public repository's top-level README lists PyTorch, torchvision, einops, tqdm and h5py, plus SpikingJelly for ImageNet. No top-level `requirements.txt` was present at the pinned commit. The example `run.sh` uses a placeholder `/path/` dataset path.

## 3. Public implementation details

### Sequential CIFAR-10 (1024-step)

The pinned trainer:
- reshapes each CIFAR-10 image into a sequence of length 1024 with 3 input channels;
- defaults to batch size 64, 200 epochs, 3 PMSN layers, model width 128, state dimension 4, dropout 0.1 and `mode=parallel`;
- supports `mode=serial` and `mode=parallel`;
- defaults to random seed 1111;
- uses AdamW plus parameter-specific optimizer groups attached to PMSN parameters;
- falls back to CPU if CUDA is unavailable;
- records inference/train timing and peak allocated CUDA memory for batches; it does not report calibrated energy or complete host/system memory.

The neuron implementation constructs a diagonal state-space kernel and uses an FFT convolution path for parallel temporal processing. The serial path executes a recurrent per-timestep update and surrogate-gradient threshold/reset. The `d_state` argument controls the kernel state dimension in this code; do not equate it with a count of biologically faithful compartments without mapping the paper's definitions to the implementation.

### Test-split checkpoint-selection issue

The inspected Sequential CIFAR-10 trainer creates torchvision's `train=False` split as `testloader`, evaluates it after every training epoch, and uses that same accuracy to update `best_acc` and save the best checkpoint. The S-MNIST/PS-MNIST trainer follows the same visible pattern.

For a leakage-resistant reproduction:
- split original training data into train and development subsets before any augmentation/windowing;
- use development-only accuracy for checkpoint selection and hyperparameter decisions;
- preserve the official test split for one-shot confirmatory scoring;
- include dataset manifest, partition and preprocessing hashes in the receipt.

This is a caveat in the inspected public code snapshot, not a claim that the published paper is invalid.

### Seed / determinism caveat

The inspected utilities seed Python, NumPy, PyTorch and CUDA with 1111 and set `cudnn.benchmark=False`; the Sequential CIFAR trainer later sets `cudnn.benchmark=True` on CUDA. A single fixed seed is not a multi-seed estimate, and this configuration should not be assumed to provide fully deterministic CUDA execution. Confirmatory work must record deterministic settings and exact seeds.

### Measurement caveat

The scripts measure epoch/batch time and peak allocated CUDA memory in their training/inference paths. They do not establish:
- peak reserved memory;
- host RSS and data-loader memory;
- end-to-end memory traffic;
- calibrated sensor/device energy;
- joules per correct decision;
- continuous-stream idle cost.

Do not compare a paper-reported acceleration or GPU allocated-memory figure with Symthaea unless task, sequence length, input representation, model quality, training budget, precision, hardware, runtime and measurement boundary are matched.

## 4. Required parity protocol

First reproduce the pinned code's task semantics and reference configuration:
- Sequential CIFAR-10 sequence length 1024, with the official transform and input shape pinned;
- S-MNIST and PS-MNIST;
- SHD after pinning the exact event encoding and split;
- parallel and serial PMSN modes;
- compartment/state-size sweep exposed by the implementation.

Use the exact pinned commit and environment. Create a clean development split and one-shot confirmatory test. Pilot with at least 10 paired seeds and confirm with at least 30 paired seeds or a preregistered power-based alternative.

Compare against:
- LIF/ALIF and a compatible SNN baseline;
- BrainTrace methods on compatible online-learning tasks;
- STDL on compatible spatial/temporal training tasks;
- HDC-LTC fast-only, slow-only and fast+slow configurations;
- matched recurrent GRU/LSTM controls where the task semantics permit.

Report accuracy, delay/temporal-dependency curves, update throughput, parameters, persistent state, activations, temporary buffers, peak allocated/reserved GPU memory, CPU RSS, bytes moved, p50/p95/p99 event latency, time-to-quality and calibrated energy. Keep algorithm-track and deployment-track results separate.

## 5. Implication for Symthaea

PMSN is a direct competitor on multi-compartment temporal dynamics and temporal parallelization, while BrainTrace targets online credit assignment and STDL targets spatial/temporal learning decoupling. These are separate axes.

Do not compare them using one generic SNN score. Test each mechanism individually, then run a factorial study to determine whether configurable multi-compartment state, fast/slow HDC summaries, eligibility traces and spatial auxiliary boundaries are complementary or redundant.

A potential Symthaea advantage must be demonstrated at equal temporal task quality and full resource budget. It is not enough to report that a sparse or parallel implementation uses fewer operations.

## 6. Reproduction status and claim ceiling

- Pinned code and source blobs inspected.
- MIT repository license identified.
- No PMSN code, dependency environment or dataset was installed or executed.
- No paper score was reproduced.
- No PMSN-vs-Symthaea comparison, hardware result or energy result is claimed.
- Public code's test-split checkpoint-selection behavior must be corrected for confirmatory reproduction.

A successful comparison supports conclusions only for the declared task, data split, code commit, model configuration, training protocol, precision and hardware. It does not establish universal superiority or neuromorphic efficiency on unmeasured devices.
