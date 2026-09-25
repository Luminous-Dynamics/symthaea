# NEURO-SILICON-001A — HDC/CfC benchmark contract

Status: source contract only; no benchmark PASS is claimed.

Tracks: #5804, #5805, #5808, #5800.

## Purpose

Freeze the smallest reproducible conventional-compute benchmark surface that later neuromorphic/FPGA/ASIC work must beat or complement without changing the workload after seeing accelerator results.

This contract deliberately reuses existing Symthaea implementations and benchmark code. It does not create a second HDC or CfC algorithm implementation.

## Core theorem

```text
same source subject
+ same frozen input fixture
+ same declared operation/workload
+ same correctness rule
+ explicit environment/measurement boundary
-> comparable execution evidence
```

And:

```text
historical source comment
!= benchmark evidence

documentation timing estimate
!= benchmark evidence

micro-kernel acceleration
!= end-to-end workload acceleration

lower accelerator-core energy
!= lower whole-system energy
```

## Canonical software owners

### Binary HDC

Canonical implementation:

- `crates/core/symthaea-core/src/hdc/binary_hv.rs`

The first benchmark/RTL train treats the existing 16,384-bit `BinaryHV` semantics as authoritative for exact digital HDC.

Required primitive families:

- deterministic fixture generation where applicable;
- XOR binding;
- bind chains;
- majority bundling;
- Hamming/similarity operations;
- permutation/sequence operations;
- temporal binding;
- associative-memory/search primitives only where the existing canonical implementation is used directly.

Existing benchmark surfaces include:

- `crates/core/symthaea-core/benches/hdc_benchmarks.rs`;
- `crates/core/symthaea-core/benches/hdc_simd_compare.rs`;
- `crates/core/symthaea-core/benches/hdc_improvements.rs`;
- `benches/hdc_benchmarks.rs`;
- `benches/quick.rs`;
- `benches/standard.rs`.

These are measurement candidates, not automatically qualified benchmark subjects. Each executed tranche must bind the exact source SHA and selected bench/function set.

### CfC

Canonical implementation family:

- `src/dynamics/cfc/mod.rs`;
- implementation modules below `src/dynamics/cfc/`.

The current declared continuous-time state form is:

```text
h(t) = h_inf + (h_0 - h_inf) * exp(-dt/tau)
```

Existing benchmark surfaces include:

- `benches/cfc_gpu.rs`;
- `benches/dynamics_cfc.rs`.

A neuromorphic mapping must preserve the selected CfC semantics or explicitly declare an approximation profile. A spiking-network translation is not assumed to be equivalent.

## V1 workload registry

Every workload receives a stable workload ID and a frozen source/fixture binding before accelerator comparison.

### HDC-P1 — exact bind

Operation: canonical `BinaryHV::bind`.

Correctness: bit-identical 16,384-bit output.

Required fixture classes:

- zero/zero;
- zero/random;
- ones/random;
- self-bind;
- deterministic random seed pairs;
- structured alternating-byte patterns.

### HDC-P2 — bind chain

Operation: canonical repeated bind / bind-chain semantics.

Correctness: bit-identical output.

Sweep dimensions:

- 1, 2, 4, 8, 16, 64 input vectors;
- cold vs warm-cache profile where measurement method supports it.

### HDC-P3 — bundle

Operation: canonical majority bundle.

Correctness: bit-identical output.

Initial exact set:

- 3, 5, 9, 33 vectors.

Even-cardinality/tie behavior is retained as a separate semantic fixture and must not be redefined by hardware convenience.

### HDC-P4 — permutation / temporal bind

Operations:

- canonical permutation;
- canonical temporal bind.

Correctness: bit-identical output.

Required cases include wraparound and order reversal proving temporal non-commutativity where expected.

### HDC-P5 — similarity / Hamming

Operation: canonical similarity/Hamming implementation.

Preferred hardware comparison surface: exact integer match/distance count before floating-point normalization.

Required fixtures:

- identical;
- complete opposite;
- one-bit difference;
- deterministic near-half random pairs.

### HDC-W1 — streaming HDC composition

A bounded workload built only from canonical HDC primitives, with fixed sequence length and vector corpus.

Purpose: test whether primitive acceleration survives command/transfer/memory overhead.

### HDC-W2 — associative search

Use an existing canonical search implementation or benchmark path; do not implement a hardware-specific search algorithm for the baseline.

Sweep candidate memory sizes only after exact owner/function is frozen.

### CFC-P1 — one CfC state update

Use canonical CfC cell/network path under an exact frozen configuration.

Correctness tolerance must be numeric and preregistered because CfC uses floating-point computation.

### CFC-W1 — recurrent sequence

Run a fixed recurrent sequence through a selected canonical CfC network/configuration.

Measure:

- final/output correctness;
- per-step and sequence latency;
- state/update throughput;
- memory behavior where measurable.

### EDGE-W1 — sensor-to-state bounded edge workload

A later profile composed from existing Symthaea perception/state primitives and one HDC or CfC stage.

It must not be introduced until its exact canonical source path is audited.

### NEG-W1 — dense conventional-compute control

Freeze at least one workload expected to favor CPU SIMD/GPU rather than neuromorphic specialization.

Negative controls are mandatory: a neuromorphic program that cannot preserve conventional wins is not evidence-driven.

## Frozen fixture identity

Every executed benchmark receipt must bind:

- exact Symthaea commit SHA;
- exact workload ID/version;
- exact source file/function or benchmark target;
- exact input fixture hash;
- exact HDC dimension/state sizes;
- exact CfC configuration, dt/tau-related parameters and activation profile where applicable;
- exact random seeds;
- exact sample/warmup counts;
- exact compiler/toolchain and feature flags;
- exact hardware/OS profile;
- exact command line.

Changing any of these creates a new benchmark subject unless the field is explicitly declared non-semantic for that workload.

## Measurement planes

Do not collapse measurements into one efficiency score.

Record independently when available:

- correctness/task utility;
- median latency;
- p95/p99 latency where sample structure permits;
- throughput;
- CPU/GPU/device utilization;
- bytes transferred between host/device;
- working-set size;
- wall energy;
- device-only energy;
- peak power;
- build/synthesis resource use for hardware candidates;
- initialization/setup latency;
- adaptation/write cost for online-learning workloads.

## Energy boundary

Energy claims require an explicit measurement boundary.

Valid distinct statements may include:

```text
accelerator-core energy/op
host+accelerator energy/op
whole-board energy/workload
wall energy/workload
```

These are not interchangeable.

If only estimated synthesis power is available, label it as estimated. Do not compare it directly to measured wall energy without an explicit model and uncertainty statement.

## CPU/GPU baseline policy

The baseline should use the strongest ordinary execution path already available in the selected source subject, including existing SIMD/GPU implementations where appropriate.

Do not intentionally disable SIMD, use debug mode, or compare against a weaker algorithm in order to create an accelerator win.

For HDC, current SIMD paths are part of the incumbent implementation and therefore part of the baseline unless a workload profile explicitly targets a constrained device without them.

For CfC, reuse the existing CPU/GPU benchmark infrastructure before creating a new baseline harness.

## Historical performance claims

Performance figures embedded in source comments, old research notes or architecture documents are hypotheses/context only until rerun under the frozen subject.

Examples include comments in `BinaryHV` describing ns-level operation costs and older HDC standard documents containing projected or historical timings.

These values must not populate comparison tables as measurements unless accompanied by a fresh exact receipt.

## Correctness before performance

Candidate hardware must pass the declared semantic comparison before its speed/energy result can be placed beside the canonical baseline.

For exact digital HDC:

```text
candidate output != canonical output
-> no equivalent-accelerator performance claim
```

For explicitly approximate or analogue implementations:

```text
representation/operator differs
-> new approximation profile
-> measure primitive error
-> measure task-level effect
-> compare under that named profile
```

Approximate HDC may still be useful, but it is not silently equivalent to `BinaryHV`.

## Accelerator comparison classes

Keep comparison classes distinct:

1. CPU scalar/reference where meaningful;
2. CPU incumbent SIMD path;
3. GPU incumbent path where one exists and fits the workload;
4. ordinary FPGA implementation;
5. event-driven/digital neuromorphic FPGA profile;
6. mature-node ASIC estimate/candidate;
7. fabricated ASIC observation;
8. analogue/memristive simulation;
9. physical analogue/memristive device observation.

No result from one class automatically establishes another.

## External research anchors

External systems are architecture/reference anchors only.

### Large-scale digital neuromorphic

Intel Hala Point / Loihi 2 demonstrates that large event-driven neuromorphic systems can be built at substantial scale. This does not establish that Symthaea HDC/CfC workloads benefit from that architecture.

### HDC analogue in-memory computing

Huang et al., *Hyperdimensional in-memory computing with analogue memristive crossbar arrays*, Nature Communications 17, 9162 (2026), demonstrates that HDC can be mapped to analogue memristive in-memory hardware.

This supports investigating hardware-aware HDC, but:

```text
published device/system result
!= Symthaea workload advantage
!= local device reproducibility
!= manufacturable Symthaea accelerator
```

The first executable Symthaea path remains exact digital HDC -> RTL -> FPGA before any claim about memristive hardware.

## Advancement gates

### Gate A — baseline admissible

Requires:

- exact workload registry entry;
- source/fixture identity;
- correctness checks;
- reproducible benchmark command;
- negative controls included.

### Gate B — RTL equivalence admissible

Owned by #5808.

Requires exact digital HDC primitive equivalence to the frozen oracle corpus.

### Gate C — FPGA comparison admissible

Requires:

- Gate A and B;
- exact FPGA target/toolchain;
- synthesis/implementation evidence;
- physical-board observation before measured board-performance claims;
- host/transfer overhead included for end-to-end claims.

### Gate D — ASIC candidate admissible

Requires #5800 design-to-MPW evidence stages. FPGA success alone is not ASIC qualification.

### Gate E — analogue/memristive candidate admissible

Requires separate device/array/non-ideality profile and full peripheral-energy accounting.

## Stop conditions

Pause or demote a candidate architecture if:

- it cannot preserve required task correctness;
- host/data-conversion overhead erases the primitive gain;
- a conventional CPU/GPU/FPGA remains preferable under the target profile;
- device variability requires unrealistic calibration;
- online adaptation/update cost dominates;
- toolchain or hardware dependency burden exceeds the claimed deployment profile;
- energy improvement exists only under incomparable boundaries.

A negative result is a successful research outcome when it prevents a false hardware direction.

## Nonclaims

This document does not establish a performance baseline, neuromorphic advantage, FPGA implementation, fabricated chip, memristor capability, or hardware manufacturing closure.

It freezes the comparison contract that later executable evidence must satisfy.