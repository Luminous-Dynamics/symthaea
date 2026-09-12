# Simulation Host Conformance Harness

This standalone host fixture proves the executable boundary of Symthaea's public
`simulation-provider-v1` ABI without depending on Symthaea's internal runtime.

It uses Wasmtime 44.0.1 with generated Component Model bindings and an **empty
linker**. The harness therefore fails if the built guest imports WASI,
filesystem, network, clocks, randomness, or any other host capability.

On success it also verifies:

- the guest's control identity and ABI version;
- SHA-256 binding to the exact presented manifest bytes;
- ready health;
- typed `SimulationRequest` transport;
- typed `SimulationOutput` transport;
- exact request-ID correspondence;
- the deterministic synthetic parameter-sum fixture result;
- zero-confidence / maximal-epistemic-uncertainty fixture semantics;
- preservation of the explicit `not engineering evidence` warning.

This is a conformance test, not the production extension executor. It deliberately
contains no signer trust, admission, routing, currentness, or provenance policy.
Those layers remain separate and must wrap the production invocation path.
