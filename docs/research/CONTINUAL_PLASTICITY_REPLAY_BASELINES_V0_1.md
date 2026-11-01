# Continual Plasticity: Replay Baseline Contract (v0.1)

**Tracking:** NEURO-SILICON-010  
**Implementation:** `research/continual-plasticity/`  
**Companion joint controller:** #9666 / PR #9667  
**Status:** FIFO and reservoir data structures authored; no Rust execution

## 1. Falsifiable research question

Does Symthaea's temporal-value memory, active sensing and joint resource
controller improve stability/plasticity over simple replay when storage and
adaptation compute are varied independently?

Do not assume dedicated temporal memory is superior. Compare against context
only, no replay, FIFO and reservoir admission at matched retained bytes,
retained byte-time and replay compute. Test whether the performance bottleneck
moves from storage to update/plasticity when memory becomes abundant.

## 2. Candidate-side replay contracts

### FIFO

- bounded capacity in number of items;
- retain new items until full;
- then evict the oldest retained item and admit the new one;
- insertion order is the candidate-visible contiguous sequence index;
- no random draw is accepted or consumed.

### Reservoir

- first `capacity` valid items are retained;
- for the `n`th item after capacity fills, the harness supplies an unbiased
  bounded draw `j in [0,n)`;
- if `j < capacity`, replace reservoir slot `j`; otherwise do not retain the
  new item;
- the crate validates the draw range and never manufactures bounded values by
  modulo reduction;
- publish the seed, PRNG algorithm/version and every bounded draw in the
  benchmark manifest. The policy code receives only the current draw.

### Replay sampling

- partial Fisher-Yates selection without replacement;
- caller supplies one bounded draw for each requested selection;
- every draw is checked against its shrinking range;
- selected-item replay work is summed before state mutation;
- a per-update or per-episode work ceiling is enforced.

The draw stream is a benchmark reproducibility input, not a secret or a
cryptographic randomness guarantee. The evaluator should derive paired arms
from precommitted seeds and record the exact bounded-draw trace.

## 3. Resource measurement

The crate tracks:
- retained item count;
- retained payload bytes;
- time-integrated retained byte-time (byte-microseconds);
- evictions and items not retained;
- selected replay count;
- logical replay payload bytes selected (not physical device I/O);
- declared replay work units.

The evaluator should additionally measure the complete deployed path:
encoding, decoding, indexing, retrieval, memory transfers, CPU/GPU compute,
idle resources, end-to-end latency and physical energy. Do not label declared
replay work or model-estimated energy as measured hardware cost.

## 4. Benchmark arms

1. Online-only / no replay.
2. Finite context-only baseline where compatible with the task.
3. FIFO replay at fixed capacity.
4. Reservoir replay at the same capacity.
5. Task-balanced replay with predeclared task registry and quotas.
6. Random replay sampling without replacement.
7. Temporal-value memory retention.
8. Active refresh / sensing plus retention.
9. Joint controller from #9666.

The first crate implements the FIFO and reservoir data structures and
reproducible sampling. The remaining methods are comparison arms to add under
the same evaluator contract.

## 5. Crossover design

Independent factors:
- retained-memory capacity, from zero through constrained to effectively
  abundant;
- replay work budget per update;
- adaptation compute budget;
- task sequence length/order and task recurrence;
- abrupt versus gradual distribution/regime shifts;
- distraction, retrieval interference and evidence retraction.

At each point use identical task/stream/model/initial-state commitments across
paired arms. Hold out task orders, seeds and regime transitions. Include both
task-visible and task-free conditions, and report task-order variance.

## 6. Metrics

- forward transfer and backward transfer;
- forgetting after each transition and after distraction/retraction;
- recovery time after regime shift;
- adaptation samples and measured compute/energy per new task;
- retained byte-time, peak/steady-state memory, replay I/O and replay work;
- per-task and worst-task performance, not just aggregate averages;
- confidence/calibration and abstention quality when required by task contract.

The evaluator must keep future labels and forgetting/transfer outcomes private.
Candidate buffers receive only current observations and metadata authorized by
the benchmark task.

## 7. Qualification gates (targets)

- exact deterministic admission and sample traces for fixed draw sequences;
- zero retained capacity violations and zero duplicate active IDs;
- invalid draws, sequence jumps, timestamp regressions and arithmetic overflow
  fail without state mutation;
- replay work never exceeds the declared cap for an accepted sample;
- byte-time accounting agrees with an independent hand calculation;
- any policy claimed superior to FIFO/reservoir/context-only must win at
  matched retained bytes, byte-time and replay compute, without material
  regression in forward transfer, backward transfer or worst-task performance.

Targets are not results. Freeze task families, thresholds and primary metrics
before confirmatory evaluations.

## 8. External benchmark mapping

- CL-Bench 1.0 and MemoryBench 2026 provide sequential experience/feedback
  settings, but their LLM agent scores are not direct HDC/LTC results. Use
  compatible task adapters and report architecture/harness differences.
- NeuroBench algorithm and system tracks can separate model-level quality and
  deployed efficiency when the workload matches.
- MLPerf Tiny applies only to compatible edge inference workloads under its
  quality and energy protocol.

Keep these as distinct comparison lanes. Do not collapse heterogeneous task
semantics or resource boundaries into one leaderboard.

## 9. Current implementation and claim ceiling

The Rust crate contains 14 source-authored test functions. No Cargo build,
rustfmt, Clippy, tests or benchmark has run in this environment. No physical
energy or latency has been measured. This package is a pair of replay data
structure baselines, not evidence that replay or any Symthaea policy beats
state-of-the-art continual learning.
