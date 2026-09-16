# symthaea-neuroarch-adapters

Read-only adapters from active Symthaea engines into the neutral `symthaea-neuroarch-types` contract.

The primary NEUROARCH-001 incumbent is `symthaea-core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork`, because that is the implementation re-exported by the active Symthaea HDC surface and consumed across active domain/controller crates.

This crate intentionally wraps active engines rather than modifying them. A topology adapter may inspect static configuration; it may not mutate runtime state, upgrade evidence, or imply that a structural edge has measured causal effect.

## V1 state accounting

For an adapted circuit, `state_dimension` is the number of logical state dimensions **per implementation unit** and `unit_count` is the number of such units represented by the circuit. A resource receipt derives total logical state dimensions with a checked multiplication:

```text
total_logical_state_dimensions = state_dimension * unit_count
```

Adapters must not pre-multiply multiplicity into `state_dimension`, because NEUROARCH-002 needs the raw factors for matched-budget accounting and would otherwise be vulnerable to double-counting.

The external-input descriptor is structural interface metadata, not evidence that its full HDC input is persistently stored. Persistent-state accounting must therefore distinguish computational circuit state from transient input/message buffers.

## Identity boundary

Execution method, exact model parameters, build/toolchain identity, initialization seed, activation/learning configuration, Fourier configuration, and runtime state belong to subject/trial/checkpoint receipts rather than `TopologyCommitment` unless a future schema explicitly promotes a field into structural identity.

Existing consciousness-analysis surfaces such as persistent-homology/TDA metrics or Φ-like estimators are optional experimental observables. They are not part of static topology identity and have no promotion authority over the preregistered NEUROARCH tournament.
