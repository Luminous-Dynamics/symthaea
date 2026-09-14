# symthaea-assurance-trust-store

Hardware-agnostic assurance contract for the external monotonic trust store that protects policy-lineage anchors from rollback.

This crate does **not** claim TPM, secure-element, HSM, or other hardware support by itself. Instead it defines the invariants any concrete implementation must evidence:

- immutable logical store identity,
- strictly monotonic checkpoint counters within one counter epoch,
- exact binding to a `PolicyLineageAnchor`,
- tamper-evident checkpoint chaining,
- backup identity bound to an exact accepted checkpoint,
- restore floors that cannot move backward,
- explicit reviewed authorization for hardware/store replacement,
- a new counter epoch after replacement,
- continuity from the last accepted checkpoint into the replacement store,
- independently evidenced attestation/verification references.

A future TPM/HSM/secure-element adapter should produce these records from actual hardware evidence; it must not bypass this contract.

This crate only evaluates assurance provenance. It never grants physical authority.
