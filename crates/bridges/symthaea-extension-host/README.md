# symthaea-extension-host

Fail-closed WebAssembly Component control-plane host for public Symthaea
extensions.

## What success means

`ControlPlaneHost::inspect()` proves **technical compatibility only**:

- manifest JSON parses and passes structural validation;
- runtime/ABI/resource requests fit local host policy;
- the base control component asks for no host imports;
- Wasmtime can compile and instantiate the component;
- guest execution is bounded by configured store/fuel/epoch limits;
- the guest's ID, version, ABI and SHA-256 manifest digest match the exact
  manifest bytes presented to the host;
- the guest can report health.

It does **not** prove:

- who signed the component;
- whether that signer is trusted;
- whether the extension is admitted to the capability registry;
- whether an invocation is authorized;
- whether a domain result is scientifically correct;
- that JIT compilation itself is CPU/time/memory-contained.

Those are separate layers by design.

## Zero ambient authority

The initial host constructs an empty Component Model `Linker`. It does not add
WASI, filesystem, network, clock, randomness, sensor, actuator, solver, or other
Symthaea imports. Pre-instantiation therefore fails if a component requires any
host capability.

Later capability hosts should add only the narrow WIT imports corresponding to
permissions explicitly granted by policy.

## Lean Wasmtime build

The host disables Wasmtime's broad default Cargo feature set and opts into only:

- `std`;
- `runtime`;
- `cranelift`;
- `component-model`.

Async execution, cache, GC, threads, profiling, coredumps, WAT parsing, pooling,
and other unrelated embedding surfaces are therefore not pulled into this
control host merely because Wasmtime supports them.

## Execution containment

After compilation the host combines:

- `StoreLimits` for guest linear memory and instance/table/memory counts;
- fuel metering for deterministic compute exhaustion;
- epoch interruption for an independent execution wall-time deadline;
- output-size checks for the control-plane response.

Before compilation it applies manifest/component byte-size ceilings and validates
the manifest/resource request against an independent local `ControlHostPolicy`.

### Important compilation boundary

`Component::new` performs compilation **before** the store exists. Store memory
limits, fuel, and epoch deadlines therefore do not constrain compiler CPU,
memory, or wall time.

For curated/development plugins, the current byte-size ceiling is useful defense
in depth. It is not sufficient to claim hostile-input compilation containment.
Before this host is promoted for arbitrary public packages, compilation should
move behind a separately supervised worker/process with explicit CPU, memory,
wall-time, and crash containment. The resulting compiled artifact/receipt should
then be bound to the exact component digest and Wasmtime/compiler profile.

This distinction is intentional:

```text
PackageSizeBounded
    != CompilationResourceBounded
    != GuestExecutionBounded
```

## Placement

This crate intentionally lives under `crates/bridges`, not `crates/core`.
Wasmtime is a runtime cost and should not become part of Symthaea's default core
substrate merely because public extensions exist.

## Admission layer

Technical compatibility remains separate from trust and authority:

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
signer authorization/currentness
        |
        v
local capability admission
        |
        v
point-of-use routing + authorization
```

Do not collapse these states into a single `trusted: bool`.
