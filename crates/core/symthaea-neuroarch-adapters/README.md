# symthaea-neuroarch-adapters

Read-only adapters from active Symthaea engines into the neutral `symthaea-neuroarch-types` contract.

The primary NEUROARCH-001 incumbent is `symthaea-core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork`, because that is the implementation re-exported by the active Symthaea HDC surface and consumed across active domain/controller crates.

This crate intentionally wraps active engines rather than modifying them. A topology adapter may inspect static configuration; it may not mutate runtime state, upgrade evidence, or imply that a structural edge has measured causal effect.

Execution method, exact model parameters, build/toolchain identity, initialization seed, and runtime state belong to subject/trial receipts rather than `TopologyCommitment`.
