# symthaea-evidence-verifier-runtime-continuity

Bounded verifier-execution continuity for the ASSURE program.

This crate separates three claims that must not be conflated:

- an approved measured launch identity;
- an observed, signed, hash-linked runtime measurement lineage; and
- one exact verifier computation whose input/output is bound to that lineage.

The first version deliberately forbids in-line executable/configuration/dependency transitions. Any such transition requires a new launch lineage.

`ContinuousVerifierExecution` is non-serializable. Persisted attestations and reports are evidence only and grant no physical authority.

This crate does not claim that software-visible measurements prove TPM/TEE certification, that the attestation authority is uncompromised, or that caller-supplied timestamps are intrinsically trustworthy. Those remain separate deployment theorems.
