# Symthaea Simulation Extension ABI v1

This document defines the first capability-specific public data-plane ABI for
Symthaea extensions. It complements `extension-control-v1`; it does not replace
or widen the control plane.

The WIT source is:

`crates/core/symthaea-extension-core/wit/simulation-provider.wit`

## Why simulation first

Simulation is already a real typed consumer in Symthaea:

- `SimulationRequest` is validated before dispatch;
- providers are selected by the extension capability router;
- heavy native solver factories are instantiated only after selection;
- process-local admission authority is checked separately from routing quality;
- the first real native extension adapter exists for ngspice.

That makes simulation a better first public data plane than a speculative generic
plugin call.

There is intentionally no `invoke(string, bytes) -> bytes` ABI.

## World

`simulation-provider-v1` exports:

- `control` — the v1 identity/health interface;
- `simulation-types` — the named type interface shared by the provider contract;
- `simulation-provider` — one typed `simulate` function.

It imports nothing.

The explicit `simulation-types` export is security-relevant, not cosmetic. WIT
retains type ownership across `use` statements. If an exported interface uses
another interface and that dependency is not explicitly exported, the dependency
is treated as a transitive component import. Exporting `simulation-types` keeps
this pure world genuinely zero-import while preserving reusable named types.

A v1 pure simulation component therefore receives no ambient filesystem,
network, wall clock, randomness, sensor, actuator, or generic WASI authority
through this world. Any future provider that genuinely requires host effects must
use a different world whose imports correspond to explicit manifest permissions.

## Host mapping

The wire types deliberately mirror the stable, normalized subset of
`symthaea-sim-bridge`:

- `simulation-request` -> `SimulationRequest`;
- `model-parameter` -> `ModelParameter`;
- `uncertainty` / `interval` -> normalized uncertainty types;
- `simulation-output` -> the guest-owned numerical portion of
  `SimulationResult`;
- `simulation-metric` -> `SimulationMetric`.

The WIT ABI is still an independent compatibility boundary. Rust struct layout,
Serde representation, and private/internal fields are not public ABI.

## Guest output is not provenance

The guest cannot provide `SimulationEvidence`.

That omission is deliberate. A component must not be able to turn an assertion
about itself into trusted execution provenance. The host constructs evidence from
facts it controls or observes, including:

- admitted extension ID and version;
- exact component digest;
- exact normalized request digest;
- exact normalized output digest;
- WIT/adapter version;
- Wasmtime execution path/profile;
- any future independently resolved validation/evidence grade.

A numerically valid guest output is therefore not automatically engineering
truth or solver-backed evidence.

## Validation before invocation

Before a host crosses the component boundary it must reject requests that violate
the native `SimulationRequest` invariants and additional public-boundary limits.
The v1 host should enforce at least:

- non-empty canonical request ID and objective;
- bounded request/objective/string lengths;
- bounded parameter and requested-metric counts;
- finite parameter values;
- finite uncertainty values in `[0, 1]`;
- finite ordered uncertainty intervals;
- capability/solver agreement with the admitted manifest;
- manifest and host resource ceilings;
- exact selected-provider authority at point of use.

No guest is trusted to validate its own input authority.

## Validation after invocation

Before releasing a result downstream, the host must reject:

- a returned `request-id` different from the request ID;
- non-finite or out-of-range confidence/uncertainty values;
- non-finite metric values;
- empty or overlong metric names/units;
- output collections or strings above host/manifest limits;
- malformed intervals;
- Wasmtime trap/fuel/deadline/memory failures;
- selected-provider authority that became stale/revoked during execution.

Host-side validation should produce a normal `SimulationResult` only after these
checks succeed.

## Authority sequence

The intended execution sequence composes the existing router/authority work:

```text
ScopedAdmissionSet
    -> pre-check all current candidates
    -> deterministic capability routing
    -> post-check all routing candidates
    -> selected extension ID
    -> pre-check selected admission only
    -> instantiate/call simulation-provider-v1
    -> validate and normalize output
    -> post-check selected admission only
    -> release SimulationResult + RoutingDecision
```

A losing candidate that changes after routing has completed must not invalidate
the selected provider's completed result. A selected provider that changes during
execution causes the result to be withheld. Because execution may already have
occurred, that post-use error is not automatically retry-safe.

## Error boundary

`simulation-provider-error` describes only guest-domain computation failures.
The following remain host-side errors and cannot be forged into success by a
guest:

- technical/component incompatibility;
- signer/admission/currentness failure;
- process-local authority mismatch;
- missing/ambiguous selected admission;
- resource or sandbox failure;
- malformed guest output;
- host provenance/evidence validation failure.

## Versioning

Changing field meaning, weakening validation, adding ambient imports, or changing
provenance ownership requires an explicit ABI/profile revision. New optional
behavior should not silently widen `simulation-provider-v1`.
