# symthaea-sim-extension-routing

Lazy extension-aware dispatch for `symthaea-sim-bridge`.

The existing simulation bridge remains the numerical/request/result contract.
This crate changes only **when** a backend is constructed and **how** an
already-admitted provider is selected.

## Data flow

```text
SimulationRequest
      |
      v
solver capability id
      |
      v
ExtensionRegistry (cheap manifests only)
      |
      v
ExtensionRouter (hard gates + deterministic quality ranking)
      |
      v
selected SimulationBackendFactory
      |
      v
instantiate ONE backend
      |
      v
legacy SimulationRegistry (request/result/evidence invariants)
```

## Lean property

Registering a provider does not construct a solver backend. Heavy native
libraries, processes, remote clients, or future WASM components can remain cold
until their provider wins a routing decision.

## Migration property

This crate does not modify or replace `SimulationBackend` or
`SimulationRegistry`. Existing adapters continue to work. An adapter can gain a
cheap descriptor/factory wrapper incrementally, one provider at a time.

## Policy property

The factory does not decide whether it is trusted. Admission, trust, evidence,
reliability, and readiness are host observations consumed by the generic
extension router. The descriptor is checked again against the instantiated
backend name and solver support before execution.
