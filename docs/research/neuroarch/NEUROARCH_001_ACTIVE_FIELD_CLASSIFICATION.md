# NEUROARCH-001 — Active HDC/LTC identity-field classification

Status: preregistered classification draft; qualification pending  
Owner implementation: `symthaea-core::hdc::hdc_ltc_unified`

This document prevents the active adapter from deciding after results which configuration fields count as topology, subject identity, learned parameters, or runtime state.

## Structural topology identity

These fields affect `TopologyCommitment` in V1:

| Active field / property | V1 topology meaning |
|---|---|
| `UnifiedNetworkConfig.layer_sizes` | circuit count + `unit_count` multiplicity for each layer |
| `UnifiedConfig.dimension` | `state_dimension` **per implementation unit** |
| `UnifiedConfig.tau_base` | declared circuit timescale, rounded to nearest nanosecond |
| `UnifiedNetworkConfig.use_layer_binding` | inter-layer route transform (`Bind` vs `Direct`) |
| `UnifiedNetworkConfig.skip_connections` | explicit skip routes + target `BundleAll` merge policy |

For V1 resource accounting:

```text
total_logical_state_dimensions = state_dimension * unit_count
```

with checked multiplication. The adapter MUST NOT pre-multiply layer size into `state_dimension`; doing so would make downstream resource receipts double-count state.

The active implementation has no explicit route communication budget. Its routes are therefore labeled `LegacyUnbounded`, meaning **no declared bounded budget exists in the incumbent API**. The label must not be interpreted as evidence of infinite physical/runtime capacity. NEUROARCH-002 accounts observed packet/byte/operation use separately.

## Subject/model identity — not topology V1

These fields can materially change behavior and therefore MUST be bound by the exact subject/model configuration receipt, but they do not change the static graph in topology schema V1:

- exact raw `UnifiedConfig.tau_base` value/bits in addition to the topology's nanosecond declaration;
- `UnifiedConfig.backbone_tau`;
- `UnifiedConfig.activation`;
- `UnifiedConfig.learning_rate`;
- `UnifiedConfig.momentum`;
- `UnifiedConfig.weight_decay`;
- `UnifiedConfig.gating_steepness`;
- `UnifiedConfig.interp_bias`;
- `UnifiedConfig.fourier_frequencies`;
- `UnifiedConfig.fourier_amplitude`;
- initialization source/profile (`seed` vs Genesis identity and labels);
- realized initial weight/input/tau/gate hypervectors where exact-subject replay requires them.

The `tau_base` dual binding is deliberate: topology V1 records the declared mesoscale timescale at 1 ns resolution, while the subject receipt preserves exact floating-point dynamics. Therefore equal `TopologyCommitment` values do not imply bit-identical temporal parameters.

This classification does **not** say subject fields are unimportant. It says they identify the exact model/subject rather than the declared graph topology. A later topology schema may elevate a behaviorally structural field only through an explicit schema/version change.

## Learned parameter identity

Parameter values changed by learning/training belong to parameter/checkpoint identity rather than structural topology, including at least:

- weight hypervectors;
- input masks;
- tau modulators;
- gate weights/biases where trainable or externally changed;
- momentum/optimizer state where required for exact training continuation.

A trained and untrained subject may share one `TopologyCommitment` while having different checkpoint/subject identities.

## Runtime/evolution state

These values must never enter static topology identity:

- neuron state hypervectors;
- cached layer outputs;
- per-neuron evolution clocks (`total_time`, `update_count`);
- current observations/inputs;
- temporary prediction/snapshot buffers;
- runtime statistics.

The active implementation's `NetworkStateSnapshot` establishes that evolution clocks are genuine mutable evolution state when Fourier dynamics are enabled. They are therefore protected by inspection-purity tests but excluded from topology identity.

## Trial/run profile — not topology V1

The active engine exposes multiple evolution/execution methods. The exact method used by a benchmark must be committed in the trial/run profile, for example:

- Euler `evolve`;
- `evolve_closed_form`;
- fused/SIMD closed-form path;
- exact/iterative closed-form variants where used;
- target architecture / SIMD feature profile where execution semantics or numerical tolerance differ.

A trial must not claim replay parity for an execution method it did not exercise.

## Measurement surfaces — not identity or promotion authority

Existing persistent-homology/TDA outputs and Φ-like/consciousness estimators may be attached to a trial as explicitly identified measurements. They do not define `TopologyCommitment`, do not establish consciousness, and may not select or promote an architecture in the NEUROARCH tournament.

This prevents circular evidence in which an architecture is optimized for a proxy and then the same proxy is cited as independent evidence of superiority.

## Archived-standalone distinction

The standalone `symthaea-hdc-ltc` extraction has its own API, including irregular wall-clock timestamp stepping. That property belongs only to that implementation lineage unless separately implemented and qualified in active core. It is not part of the active-core qualification gate, and #3325 leaves the archived crate unchanged.

## Change rule

After held-out architecture evidence begins, changing any field's identity class requires a new topology/receipt schema or evidence lineage. Reclassification may not be used to rescue a comparison after observing results.
