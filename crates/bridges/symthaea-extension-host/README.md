# symthaea-extension-host

Fail-closed WebAssembly Component control-plane host for public Symthaea
extensions.

## What success means

`ControlPlaneHost::inspect()` proves **technical compatibility only**:

- manifest JSON parses and passes structural validation;
- runtime/ABI/resource requests fit local host policy;
- the base control component asks for no host imports;
- Wasmtime can instantiate it within memory/fuel/time limits;
- the guest's ID, version, ABI and SHA-256 manifest digest match the exact
  manifest bytes presented to the host;
- the guest can report health.

It does **not** prove:

- who signed the component;
- whether that signer is trusted;
- whether the extension is admitted to the capability registry;
- whether an invocation is authorized;
- whether a domain result is scientifically correct.

Those are separate layers by design.

## Zero ambient authority

The initial host constructs an empty Component Model `Linker`. It does not add
WASI, filesystem, network, clock, randomness, sensor, actuator, solver, or other
Symthaea imports. Pre-instantiation therefore fails if a component requires any
host capability.

Later capability hosts should add only the narrow WIT imports corresponding to
permissions explicitly granted by policy.

## Resource containment

The host combines:

- component/manifest byte-size ceilings before compilation;
- Wasmtime `StoreLimits` for guest linear memory and instance/table/memory
  counts;
- fuel metering for deterministic compute exhaustion;
- epoch interruption for an independent wall-time deadline;
- output-size checks for the control-plane response.

The extension-requested envelope must also fit within an independent local
`ControlHostPolicy` ceiling.

## Placement

This crate intentionally lives under `crates/bridges`, not `crates/core`.
Wasmtime is a runtime cost and should not become part of Symthaea's default core
substrate merely because public extensions exist.

## Next layer

The next security layer is package/signer admission:

```text
component + exact manifest bytes
        |
        v
technical control inspection  <-- this crate
        |
        v
signature validity
        |
        v
signer authorization
        |
        v
capability admission
        |
        v
per-invocation routing + authorization
```

Do not collapse these states into a single `trusted: bool`.
