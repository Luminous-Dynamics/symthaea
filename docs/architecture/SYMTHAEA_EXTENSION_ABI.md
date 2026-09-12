# Symthaea Extension ABI

## Status

Initial architecture boundary for public/community extensions.

The public plugin ABI should use the WebAssembly Component Model and WIT rather
than exposing Symthaea's internal Rust structs as a permanent foreign-function
interface.

## Why

Symthaea already has a useful native `CognitiveSubsystem` interface whose
`CycleSnapshot`/`SubsystemOutput` structures are `#[repr(C)]` and allocation-free.
That remains a good internal/native fast path, but it should not become the
long-lived public mod ABI:

- internal cognitive fields must remain free to evolve;
- a raw C-layout ABI is easy to accidentally couple to Rust implementation detail;
- third-party plugins should not need to be written in Rust;
- permissions and host imports need an explicit authority boundary;
- typed WIT interfaces can be versioned independently by capability family.

Wasmtime 44, already used by Symthaea, supports the Component Model APIs needed
to bind WIT worlds. No Wasmtime upgrade is required to establish this boundary.

## Layering

```text
signed package manifest
        |
        v
extension-control-v1        stable control plane
        |
        +-- identity()
        +-- health()
        |
        v
permission-scoped host imports
        |
        +-- clock (optional)
        +-- randomness (optional)
        +-- filesystem (optional, scoped)
        +-- network (optional, allowlisted)
        +-- sensors / actuators (optional, typed)
        |
        v
capability-specific WIT worlds
        |
        +-- domain provider
        +-- simulation provider
        +-- tool provider
        +-- perception provider
        +-- embodiment provider
        +-- ...
```

The control plane deliberately imports **nothing**. Ambient WASI authority is
not part of the base world.

## Manifest is authoritative

A package is admitted from its signed `ExtensionManifest`, not from claims made
by executable guest code. The guest's `identity()` export is a consistency
check and must match the admitted package manifest:

- extension ID
- package version
- ABI major/minor
- manifest digest

Signature validity is not sufficient authorization. A signature checked only
against a public key carried inside the same artifact proves self-consistency,
not that the signer is trusted. The future extension host must validate signer
identity against installation policy / a trust store before execution.

## Control plane vs data plane

Do **not** add a universal `invoke(string, bytes) -> bytes` API to the control
world merely for convenience. That would recreate an untyped RPC bus and make
capability contracts difficult to audit.

Instead, add separate WIT packages for real capability families once they have
real consumers. Examples:

- simulation requests/results with explicit uncertainty and provenance;
- cognitive proposals that expose only a deliberately public subset of state;
- sensor streams with explicit units and timestamps;
- tool calls with typed argument/result records.

A capability-specific interface may still carry an opaque payload where the
domain genuinely requires it, but opacity should be local and justified rather
than the universal extension boundary.

## Native extensions

Trusted native Rust extensions may continue to use narrow Rust traits such as
`DomainPlugin`, `CognitiveSubsystem`, `SimulationBackend`, and
`EmbodimentBridge`. They should register the same `ExtensionManifest` metadata
so discovery/routing is shared with sandboxed providers.

Native and WASM providers therefore converge at the **catalog and capability**
layer, not at an identical calling convention.

## Sandboxing rules

The future WASM host should enforce, at minimum:

1. manifest/guest identity agreement before activation;
2. explicit permission-scoped imports only;
3. memory, fuel, wall-time, output-size, and concurrency budgets;
4. no ambient filesystem/network/environment inheritance;
5. signer authorization separate from signature verification;
6. crash/trap isolation so a guest cannot take down the cognitive loop;
7. per-capability effect classification and safety policy;
8. evidence/provenance on externally consequential results.

The existing `WasmArchitect` already provides useful fuel and memory controls,
but public extension admission should be centralized in one hardened host rather
than duplicating Wasmtime setup across action/tool paths.

## Versioning

- `ExtensionManifest.abi.major` breaks only for incompatible host/guest changes.
- additive compatible changes increment the minor version where feasible.
- WIT package/world versions are explicit and pinned by the host.
- capability-specific worlds version independently of the control world.
- the host may support more than one ABI major during a migration window.

## Kill conditions

Keep this layer small. Do not add a field, permission, lifecycle method, or host
import without a concrete extension/host consumer in the same change or an
explicit interoperability requirement.

The goal is not a universal framework. The goal is a stable, auditable doorway
through which many specialized extensions can pass.
