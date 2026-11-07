# Physical Compute Frontier v1 — 2026-11-07

**Status:** evidence map and benchmark design; not a leaderboard  
**Owner:** QARCH-019 #9236  
**Composed owners:** NEURO-SILICON #5804, PHOTONIC-MEMORY-001 #10932, PHYS-COMPUTE-ADAPT-001 #10933, MEM-SUBSTRATE #7919, FPGA qualification #10901.

## Executive finding

The 2026 evidence supports three different research directions, not one generic “non-von-Neumann” architecture:

1. **Integrated photonic memory and transient dynamics** (FLARE).
2. **Free-space optical matrix engines** (FAST-ONN).
3. **Memristive compute-in-memory**, including HDC co-design and fault-adaptive matrix representation.

These systems use different physical mechanisms, operations, timescales and measurement boundaries. Neurons, optical samples, matrix operations, TOPS/W, aJ/op, frames/s and useful task completions cannot be ranked as if they were the same unit.

A fourth cross-cutting idea is especially promising: **adapt mathematical representation to the substrate's structure and imperfections**, then evaluate it against target matrix rank, fault topology and total physical resources.

## 1. Integrated photonic memory — FLARE

**Source:** Zhou, Wu & Fang, “Fully in-memory photonic computing,” Nature Sensors 1, 998–1010 (published 2026-08-27).  
- https://www.nature.com/articles/s44460-026-00123-2
- https://doi.org/10.1038/s44460-026-00123-2

### Paper-reported facts
- Monolithic multicore chip with 7,378 photonic neurons.
- Reported long-term memory retention: 7.45 s.
- Reported short-term dynamics: GHz-scale.
- A ten-core multilayer FLARE system was used for sensing, exploration and adaptation in autonomous racing-drone navigation.
- The paper abstract reports 61.87 aJ/operation at a system level.
- Extended data describes a four-layer network with 2+2+2+4 cores, optical fiber/microlens coupling, a controller board with 256 DAC channels and an FPGA controller. Calibration data also describes a 72-channel programmable controller array.

### Interpretation and caveats
- 7,378 neurons is not a count of general-purpose Boolean operations or HDC dimensions.
- 7.45 s is a measured temporal retention primitive, not durable storage comparable with MRAM, NAND or persistent logs.
- GHz short-term dynamics do not by themselves establish end-to-end task latency.
- The 61.87 aJ/op figure must be mapped to its exact operation denominator and system boundary from the paper/supplement before comparison. Do not assume laser, optical coupling, DAC/ADC, controller, acquisition and host costs are all included just because the abstract says system-level.
- Source data are stated as provided; no reusable open source code was identified in the paper search.

### Luminous experiments
- Reproduce the exact published task/protocol if data and methods permit.
- Build Symthaea-specific sparse-event and CfC workloads with independent oracles.
- Characterize memory retention/error versus elapsed time, read/write count, temperature and calibration.
- Test whether short-term photonic state is useful for temporal inference while digital memory holds exact identity and durable state.

## 2. Free-space optical in-memory computing — FAST-ONN

**Source:** Liang et al., “High-clockrate free-space optical in-memory computing,” Light: Science & Applications 15, 115 (published 2026-02-13).  
- https://www.nature.com/articles/s41377-026-02206-8
- https://doi.org/10.1038/s41377-026-02206-8

### Paper-reported facts
- Uses VCSEL arrays for input modulation and spatial light modulators for optical weights.
- Demonstrates YOLO convolution feature extraction at a reported 100 million frames per second.
- Describes an operating point of 25 GS/s at 7–8-bit compute precision.
- The authors state that approximately 1 mW optical power per detector supports the cited signal-to-noise target; about 1 W total optical power is stated for a 32×32 parallel-channel configuration.
- The paper discusses optical parallelism, electronic digitization/nonlinear processing between layers and options for optical cascading.

### Interpretation and caveats
- Optical sample rate, channel count, convolution rate, frame rate and end-to-end inference throughput are different metrics.
- The stated 1 W figure is an optical power budget, not necessarily complete system wall power.
- Include VCSEL drive, SLM programming/control, laser efficiency, photodetection, conversion, synchronization, cooling and host processing in a system comparison.
- FAST-ONN is a free-space optical architecture and must not be conflated with integrated waveguide photonics or FLARE's photonic memory neurons.

### Luminous experiments
- Reproduce a single convolution / matrix-vector primitive under an explicit precision contract.
- Add full input/output conversion and control latency.
- Compare against optimized CPU, GPU and FPGA versions of the same operator and same model/data.
- Test where optical MVM is useful and where nonlinear activation, memory or inter-layer conversion erases the gain.

## 3. Memristive in-memory HDC

**Source:** Huang et al., “Hyperdimensional in-memory computing with analogue memristive crossbar arrays,” Nature Communications 17, 9162 (published 2026-07-28).  
- https://www.nature.com/articles/s41467-026-76067-5
- https://doi.org/10.1038/s41467-026-76067-5
- https://pubmed.ncbi.nlm.nih.gov/42660943/

### Paper-reported facts
- Implements a hardware/algorithm co-design on a TetraMem MX100 memristive SoC.
- The system uses ten memristive computing cores, each with a 248×256 1T1R crossbar, plus CPU and on-chip peripheral circuits.
- Task: 21-language identification.
- Training samples are sourced from Wortschatz; testing samples from an independent Europarl parallel corpus, with approximately 10,000 training sentences and 1,000 test sentences per language.
- The hardware profile uses 1,024-dimensional hypervectors.
- Reported simulated accuracy: 98.87%; physical SoC accuracy: 95.24%.
- The paper reports 2.58 TOPS/W and 302.5 μs in its system comparison.
- The method uses analog VMM and device randomness/multistate conductance for feature mapping and classification.

### Interpretation and caveats
- This is language classification, not exact 16,384-bit BinaryHV top-1 retrieval.
- 1,024-dimensional hypervectors and 16,384-bit binary vectors have different representations and operations.
- Treat 2.58 TOPS/W and 302.5 μs as paper-specific; confirm included peripheral and energy boundaries before comparison.
- The authors disclose that two authors are TetraMem co-founders and paid consultants. This is relevant context for independent replication, not grounds to dismiss the result.
- The paper is licensed CC BY-NC-ND 4.0; article text/figures are not a source for derivative commercial artifacts without permissions. Dataset and code rights are separate.

### Luminous experiments
- Reproduce the 21-language task exactly where dataset and code rights allow.
- Keep the South African multilingual extension (isiZulu, isiXhosa, Sesotho, Afrikaans, English) as a separate preregistered task with source/speaker-disjoint splits.
- Compare digital HDC, analog VMM/HDC and hybrid HDC with per-language metrics, macro-F1, calibration, open-set rejection, robustness and whole-system joules per accepted classification.
- Keep exact identity/provenance and final decision verification digital unless analog exactness is separately demonstrated.

## 4. Fault-adaptive matrix representation on imperfect hardware

**Source:** Xu et al., “Fault-free analogue computing with imperfect hardware,” Nature Electronics 9, 811–824 (version of record 2026-06-12).  
- https://www.nature.com/articles/s41928-026-01638-9
- https://doi.org/10.1038/s41928-026-01638-9
- Source data: https://doi.org/10.6084/m9.figshare.31813117
- Author code: https://github.com/ZCXU0421/Fault_free_analog_computing

### Paper-reported facts
- Represents a target matrix as a product of adjustable submatrices mapped to imperfect analog hardware, with compensation layers.
- Reports >99.999% cosine similarity for a DFT matrix under a 39% device fault rate.
- Reports a 56-fold bit-error-rate reduction in a wireless prototype.
- Reports density and energy-efficiency improvements in cross-domain comparisons.
- The method includes selective updating for drift mitigation.

### Source and legal audit
- The author repository README describes FFAC.py, SimChip.py, multilayer compensation and a Variation-Enhanced Training (VET) addition dated 2026-08-11.
- The public repository LICENSE declares AGPLv3 plus Commons Clause. This imposes a sell restriction on the software and derivatives/services. Do not copy or embed that code in a Luminous commercial product without explicit additional permission and legal review.
- A clean-room mathematical reimplementation still requires separate patent/freedom-to-operate and data-rights review before commercialization.

### Luminous research hypothesis
The benefit of factorization depends on target rank/structure, latent dimension, fault topology and total resource budget. A low-rank transform may be represented efficiently; a full-rank arbitrary matrix may require much larger k and additional devices. Therefore benchmark:
- DFT/cosine/Hadamard/permutation and other structured matrices;
- known low-rank matrices with controlled singular spectra;
- sparse/banded/Toeplitz/circulant matrices;
- random dense full-rank negative controls;
- independent random, clustered, row/column and correlated defects;
- stuck-off/stuck-on, write error, read noise, drift, temperature and aging.

Compare direct mapping, differential pair, redundancy/compensation, adaptive factor representation, variation-aware training and a digital reference under both equal device count and equal achieved task quality.

## 5. New cross-paradigm thesis — timescale- and structure-matched computing

A fixed CPU/GPU/photonic/memristive ranking is not useful. Architecture selection should be conditioned on:

1. Mathematical structure: rank, sparsity, locality, algebra and conditioning.
2. Temporal profile: state lifetime, update rate, event rate, deadlines and duty cycle.
3. Numerical contract: bit exactness, permitted approximation, noise tolerance and calibration drift.
4. Data movement: bytes between memory, compute, sensors, hosts and accelerators.
5. Physical limits: device variation, thermal behavior, endurance, retention, optical/electrical conversion and fault topology.
6. Product constraints: cost, area, repairability, service life and manufacturing/supply availability.

The system should route each subproblem to an appropriate physical substrate, while using digital verification and provenance where exact identity, auditability and durable state are required.

## 6. Shared benchmark protocol

Keep two comparison tracks separate:

- **Track A — substrate isolation:** same algorithm, same representation, same data, same quality contract; change only the execution substrate.
- **Track B — architecture co-design:** best competent algorithm-plus-substrate design under matched information, development and resource budgets.

Required report vector:
- exact task quality / independent oracle result;
- p50/p95/p99 latency and jitter;
- accepted tasks/s;
- joules per accepted task at declared component and wall-input boundaries;
- bytes moved per accepted task by tier;
- active and idle power;
- memory capacity, retention, write/read cost and update frequency;
- precision/error, drift, calibration and recovery;
- physical device/cell count, active area, channels, package and optical/electrical I/O;
- fault rate and topology, plus confidence intervals;
- bill-of-materials, lifecycle, repairability and supply dependencies.

Do not combine aJ/op, TOPS/W, FPS, optical sample rate, neuron count or accuracy into one leaderboard unless each is mapped to the same useful task and full system boundary.

## 7. Near-term execution order

1. Reproduce the paper-level memristive HDC task on software before seeking physical MX100 access.
2. Complete the independently implemented rank/fault-topology matrix pilot; port the mathematical reference to Rust before it enters production.
3. Audit FLARE and FAST-ONN supplements for exact denominators, conversion/control overhead and accessible data.
4. Choose one Symthaea temporal workload and one exactness-sensitive negative control.
5. Seek physical photonic/memristive collaboration only after the software/task contract and measurement protocol are frozen.
6. Compose results into Symtropy's architecture Pareto search and Mycelix's component/evidence provenance.

## Claim ceiling

Paper result != independent reproduction. Mathematical pilot != paper reproduction. Simulation != physical device. Core/array energy != full system energy. Neuron count != operations. Optical samples/s != useful task throughput. DFT fault tolerance != arbitrary-matrix fault tolerance. Public code != commercial-use permission. Physical device demonstration != local fabrication capability.


## 8. Timescale-matched memory and computation — architectural decision matrix

This is a design hypothesis for workload placement, not a claim that a photonic device replaces a memory product.

| Required state / response regime | Candidate physical substrate | Best-fit role to test | Key failure mode / measurement |
|---|---|---|---|
| Sub-ns to ns optical propagation and GHz-scale dynamics | Integrated photonic / electro-optic networks | Fast transient transformations, wave propagation, selected reservoir/temporal compute | Optical loss, coupling, laser and driver power, phase stability, conversion and calibration |
| ns–µs exact local state and control | SRAM, FPGA BRAM/URAM, register files | Deterministic pipelines, exact HDC reducers, buffering and control | Capacity/ports, timing closure, leakage/clock power, ECC and upset behavior |
| µs–ms burst/event windows | SRAM/DRAM + event-driven or neuromorphic compute | Sensor fusion, sparse updates, temporal event processing | Burst buffering, queue tail latency, scheduler/host overhead, timestamp fidelity |
| ms–seconds adaptive temporal state | Mixed-signal, photonic transient-memory, memristive or digital recurrent state | CfC-like state evolution, temporal context, short-term adaptive memory | Retention curve, state decay, read disturbance, temperature, calibration and update energy |
| Minutes–hours working set | DDR/LPDDR/HBM and CXL only on compatible real systems | Large models, codebooks, working memory and shared accelerator state | Bandwidth/latency/NUMA, refresh, transfer cost, capacity and energy |
| Power-cycle persistent configuration / journal state | MRAM / qualified embedded nonvolatile memory | Recovery metadata, configuration, small write-intensive journals | Endurance, retention, read/write latency, density, part qualification and cost |
| Long-lived exact records and large corpora | NVMe NAND / managed persistent storage | Durable HDC stores, datasets, immutable segments and audit records | Flush/durability semantics, wear, power-loss behavior, recovery, write amplification and provenance |

### Placement rules

1. Match physical state lifetime to the required retention horizon. A reported 7.45-second photonic memory state is potentially useful for temporal state, but it does not replace durable storage.
2. Keep exact identity, provenance, cryptographic integrity, deterministic tie-breaking and durable commit state in digital systems unless a separate exact physical guarantee exists.
3. Use analog or photonic dynamics where approximation/noise is acceptable and where the operation naturally benefits from continuous-time or optical computation.
4. Place sparse hot state close to the compute that consumes it; measure bytes moved and queueing before selecting a new memory technology.
5. Include state refresh, calibration, re-encoding, migration, write/erase and recovery in lifecycle energy. A low-energy cell is not necessarily a low-energy memory system.
6. Allow Symthaea to select multiple substrates within one task, but require explicit conversion/verification boundaries at every handoff.

### New benchmark: temporal-memory contract

For every candidate physical memory, define a retention profile rather than a single retention number:
- required hold time and update frequency;
- probability of exact/acceptable read after elapsed time;
- error versus read count and write count;
- state drift versus temperature and supply;
- energy to write, read, refresh, calibrate and recover;
- latency distribution and throughput under concurrent state access;
- whether read is destructive or disturbs the stored state;
- persistence across power loss and reset;
- remapping/retraining cost after aging or device failure.

The useful metric is not “memory seconds” in isolation. It is the Pareto frontier of **correct state delivered over the required horizon per total system joule, with declared reliability and update semantics**.


## 9. FAST-ONN measurement audit — demonstrated prototype versus scale-up projection

The initial summary intentionally kept 25 GS/s and the ~1 W / 32×32-channel optical budget separate from the prototype, but the quantitative comparison needs a more explicit split.

**Paper:** Liang et al., “High-clockrate free-space optical in-memory computing,” Light: Science & Applications 15, 115 (2026-02-13). https://www.nature.com/articles/s41377-026-02206-8

### Experimental configuration and reported results
- VCSEL device: eight arrays, each 5×5 VCSELs.
- The detailed optical experiment uses 3×3 VCSEL inputs and 3×3 output kernels; the 9-channel balanced photodetector outputs are digitized by a 9-channel FPGA acquisition board at 100 MS/s.
- Demonstrated VCSEL data rate: 100 MS/s.
- Demonstrated arithmetic throughput in the described configuration: 45 GOPS.
- Random signed MVM average computation-error standard deviation: 3.27% across nine channels; the authors describe precision as over 6 bits including sign.
- The small MNIST task achieved 93.75% optical-system accuracy against 95.75% digital; Fashion-MNIST achieved 80.75% against 84.88%.
- The paper describes a YOLO convolution feature-extraction benchmark at 100 million frames/s, but this is not equivalent to 100 million complete end-to-end object-detection decisions/s.
- The paper estimates a CNN input-to-decision path of ~1.4 μs for its small 28×28 example, decomposed into ~1.0 μs optical streaming and ~0.4 μs digital-head work.
- The authors estimate ~370 fJ/op (~3 TOPS/W) for the demonstrated system, with the stated electrical boundary including VCSEL drivers/bias and DAC drivers, SLM weighting, and readout electronics (photodetector/TIA/ADC path).

### Projected operating point — not demonstrated
- 32×32 VCSEL input array with 32×32 optical fanout;
- 25 GS/s VCSEL-rate scenario;
- ~2 fJ/op / 500 TOPS/W estimate;
- >50,000 TOPS derived throughput scenario;
- ~1 W optical budget for the large 32×32 channel configuration.

These are authors' scale-up projections, not prototype measurements. The 100 MS/s/45 GOPS prototype and the 25 GS/s/50,000-TOPS scale-up must never be combined into one achieved result.

### Benchmark consequence
Report separate measures for optical arithmetic, optical convolution, input-to-decision task, and wall-input system energy. Include VCSEL/driver, DAC, SLM update/control, optical coupling, photodetection/TIA, ADC, FPGA acquisition, host classification and calibration. The paper's ~370 fJ/op estimate is not automatically equivalent to joules per accepted Symthaea task; the projected ~2 fJ/op figure is not an achieved benchmark.

Source audit: full open-access paper, including Results, Discussion, Methods and system-energy description. https://www.nature.com/articles/s41377-026-02206-8
