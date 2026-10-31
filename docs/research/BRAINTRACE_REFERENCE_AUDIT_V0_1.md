# BrainTrace Online-Learning Baseline and Reference-Task Audit (v0.1)

**Status:** static source audit; no baseline reproduction or score imported  
**Tracking:** NEURO-SILICON-007 / issue #9394  
**Related protocol:** `HDC_LTC_TEMPORAL_COMPETENCE_V0_1.md`  
**Audit date:** 2026-10-31

## 1. Purpose and claim boundary

BrainTrace is a necessary named competitor for Symthaea's online adaptation and long-horizon credit-assignment claims. It must not be reduced to a generic "SNN baseline".

Wang et al., *Model-agnostic linear-memory online learning in spiking neural networks*, Nature Communications 17, 1745 (published 2026-01-19), DOI [10.1038/s41467-026-68453-w](https://doi.org/10.1038/s41467-026-68453-w), introduce a compiler-assisted eligibility-trace framework for online learning in recurrent and spiking neural networks. The paper discusses D-RTRL and pp-prop (also referred to as ES-D-RTRL / input-output-factorized traces) and reports low-memory, high-throughput learning results.

This audit does not reproduce the paper, verify every published experiment, or import its performance numbers into Symthaea. It pins public code and identifies issues that a fair confirmatory replication must handle.

## 2. Pinned artifacts

### BrainTrace library

- Repository: [chaobrain/braintrace](https://github.com/chaobrain/braintrace)
- Inspected commit: `4fd426e1ec58a61a11d47a8d95c32346778e8b11`
- Version at that commit: `0.2.8`
- Version file blob: `braintrace/_version.py`, blob `2adac603279ae3551ac62473cdeac5ed6ecc4088`
- Project manifest blob: `pyproject.toml`, blob `bc04d9db29e17e82be00291c98b248a305184103`
- Requires Python >=3.12 and JAX >=0.9.0; it depends on BrainState and other BrainX ecosystem packages.
- Public API exposes D-RTRL, pp-prop, EProp, OSTL variants, SnAp, UORO, ThreeFactor, and DNI. The README's online sequence interface is `compile(...)` plus `etrace_grad` / `etrace_evolve`.

The current package code is actively maintained; the pinned commit on 2026-10-31 includes fixes around trace sizing, scan handling, and gradient propagation. A reproduction must pin this commit (or an explicit later commit) and record the exact dependency lock.

### Published experiment repository

- Repository: [chaobrain/braintrace-snn-experiments](https://github.com/chaobrain/braintrace-snn-experiments)
- Inspected main head: `81332436b3a0167bc83059878e43a0f0e2eb62c7`, committed 2026-06-28
- README blob: `7c5fe3b5c6bc2a058f58bac41e4dd7e994d1c54f`
- DMS script: `task-rsnn-long-term-dependency.py`, blob `f219f53953cdca8465b00e962c7380ff025e0a0b`
- Evidence-accumulation generator: `ei_coba_net_decision_making/data.py`, blob `2de7e924b458f51c2939653cf5e88fdff5fae349`
- Evidence-accumulation trainer: `ei_coba_net_decision_making/training.py`, blob `a89a32bf18b520dba595d232dbbb153a4de64527`
- Evidence-accumulation model: `ei_coba_net_decision_making/model.py`, blob `5f12e956afa64d3169430291e7e233c0cbf34a57`
- Argument parser: `ei_coba_net_decision_making/args.py`, blob `a41c3fe1dcb80630ae4a84b9ff10aee35df7796a`

The paper-to-exact-public-code-revision correspondence is **not independently verified**. The experiment repository was inspected at the public head above; it is not asserted to be a frozen final-paper release.

## 3. Reference task A: delayed match-to-sample (DMS)

The pinned `task-rsnn-long-term-dependency.py` implements a delayed match-to-sample task:

- eight possible motion directions;
- sample and test stimuli are generated from a direction-tuning population;
- the test direction is a rotated match or a random non-match;
- the stimulus is converted to stochastic spikes with a Bernoulli draw;
- the example entry point uses dt=1 ms, fixation=10 ms, sample=500 ms, delay=1,000 ms, and test=500 ms;
- 100 input channels and a 200-neuron recurrent network;
- the entry-point GIF neuron uses approximately 100 ms neuronal/synaptic time constants and 1,500 ms adaptation-current time constant.

The task is a good long-delay memory/decision reference. It should be added to the suite as an **external parity task**, while keeping Symthaea's synthetic delayed-cue task because the latter has a simpler exact oracle and controlled Hamming interference.

### Code-level caveats

1. The inspected script's `_dms` function calls NumPy's global `np.random` for trial labels, directions, and spike sampling. The inspected script entry path does not visibly establish/reset an independent seed per replicate.
2. The README shows command-line examples for BPTT, ES-D-RTRL, and D-RTRL, but the inspected Python script's entry point directly constructs and runs `OnlineTrainer` with `braintrace.IODimVjpAlgorithm`; it does not parse those README command-line flags in the inspected file. The `BPTTTrainer` class is present in the source, but the shown entry point does not invoke it.
3. The shown DMS training loop repeatedly generates batches until the training-accuracy threshold is reached. The inspected source does not expose a distinct validation or confirmatory test split.

These are limits of this public script snapshot, not evidence that the published paper's experiments are invalid. For a fair replication, write a small pinned adapter around the library API or obtain the authors' exact final-paper runner, explicitly seed the task generator and model, and add a separate locked evaluation set.

## 4. Reference task B: sequential evidence accumulation

The pinned `EvidenceAccumulation` generator offers a second valuable task:

- seven cue periods;
- 50 ms inter-cue interval and 100 ms cue duration;
- a 1,000 ms delay and 150 ms recall interval;
- 25 neurons per left, right, recall, and noise group;
- cue and noise firing rates 40 Hz, recall 40 Hz, noise 10 Hz;
- binary ground truth chosen at trial generation, then seven cue assignments sampled conditionally;
- binary output label determined by whether the count of cue assignments exceeds half the number of cues.

The training script supports BPTT, I/O-factorized trace learning, and parameter-dimensional trace learning through the BrainTrace API. The model seeds BrainState's RNG using `args.seed`; the CLI argument parser creates a random seed when none is supplied. Therefore confirmatory runs must pass a fixed explicit seed and preserve it in the receipt.

### Code-level caveats

1. The data generator is an infinite stream (`while True`) using split random keys. This is useful for online training, but a separate immutable confirmatory evaluation stream still needs to be constructed and locked.
2. The trainer stops when a smoothed training-accuracy threshold is reached or the epoch limit is exceeded. The inspected `f_train` path reports training metrics and does not show held-out validation/test evaluation.
3. The reported memory is taken from JAX device memory statistics where supported; CPU returns zero in the helper. A zero CPU value is not evidence of zero memory use. Use platform-appropriate peak RSS/allocator/counter instrumentation for a comparable system measurement.

These observations are specific to the inspected public script. Do not generalize them to every BrainTrace example or paper result.

## 5. Fair comparison protocol

For DMS and sequential evidence accumulation:

1. Reproduce task semantics from the pinned public generator first, with one exact seed per trial block.
2. Split trial-generation seeds into development, calibration, and sealed confirmatory families before hyperparameter search.
3. Report training accuracy separately from one-shot held-out confirmatory performance.
4. Run D-RTRL and pp-prop with explicit trace configuration/decay; use BPTT as a small-scale gradient reference and EProp where its assumptions match the model.
5. Compare equal observation streams, target timing, update opportunities, training examples, parameter counts, and compute budgets.
6. Record eligibility-trace memory separately from model hidden state, optimizer state, and temporary buffers.
7. Measure both per-step learning throughput and end-to-end time-to-target-quality; include compilation/warm-up separately and then amortized.
8. Run temporal lengths and delay horizons at 1, 4, 16, 64, 256, 1,000, and 1,024 ms/steps where the original task semantics permit. Preserve the original 1,000 ms DMS delay as a named anchor.
9. Keep all failed and non-converged seeds in the denominator.
10. Label results "reference implementation reproduction" unless the exact paper revision and split protocol are confirmed.

## 6. What this means for Symthaea

Do not try to beat BrainTrace by using a narrower or easier task. Match its long-horizon DMS and evidence accumulation tasks, then add conditions where Symthaea's HDC/time representation may have a principled advantage:

- stronger distractor interference and controlled cue similarity;
- irregular intervals and missing observations;
- hidden regime change between cue and recall;
- task-invalidating or retracted evidence;
- time-scale shifts without resetting the recurrent state.

The strongest claim would be a reproducible Pareto improvement in held-out task quality, time-to-recovery, eligibility/state memory, memory traffic, and joules per correct decision. No such improvement has yet been measured.

## 7. Reproducibility status

- Static source inspection only.
- No BrainTrace dependency was added to Symthaea's Rust workspace.
- No Python environment was installed or executed.
- No BrainTrace baseline was run.
- No scores or energy values were imported.
- Exact commit/blob identifiers are recorded above to enable a later reproduction.


## 8. Published complexity/performance anchors (literature only)

The Nature Communications article's full text provides useful, explicitly scoped reference points:

- The paper states pp-prop stores two traces representing pre- and postsynaptic activity and gives memory complexity `O(BN)`, with batch size `B` and neuron count `N`.
- It describes D-RTRL memory as `O(BP)`, where `P` is parameter count; in a dense recurrent network this can grow quadratically with neuron count.
- In its LIF Gesture comparison, the paper reports 0.5 GB for pp-prop versus 17.5 GB for D-RTRL (35× lower memory) and about a 10× compute-time advantage over D-RTRL. It reports BPTT GPU memory increasing with sequence length and a crash at 400 steps in that particular experiment.
- The article reports pp-prop matching BPTT in most settings and beating the strongest stated baseline by more than 10 percentage points on SHD in its experiment table.

These are **paper-reported results for the paper's model, tasks, training code, hardware and memory boundary**. They are not measurements of Symthaea, not a fair direct comparison to its HDC state footprint, and not a guarantee that the public experiment scripts exactly reproduce each paper result.

The article also explicitly notes that pp-prop's approximation quality can depend on firing statistics and network depth. The method builds its low-memory trace approximation with exponential smoothing and a low-pass filter. Therefore a matched replication must sweep/preregister trace decay and report approximation drift/gradient fidelity against D-RTRL or BPTT on tractable settings; it must not present pp-prop as an exact gradient.

### Required head-to-head measurements

For each method, report separately:

1. persistent model parameters and hidden state;
2. eligibility-trace/optimizer memory, including peak temporary buffers;
3. growth with neuron count and sequence length;
4. learning throughput and time-to-target-quality;
5. held-out accuracy by delay and after regime change;
6. energy per correctly learned decision, including compilation/warmup and full update work.

Do not compare pp-prop's eligibility memory alone with HDC-LTC's whole neuron memory, or vice versa. Normalize both to a full system boundary and add a component-level breakdown.
