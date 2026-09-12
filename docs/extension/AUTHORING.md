# Authoring Symthaea Extensions

## Current status

The public extension ABI is being introduced incrementally. Native Rust seams
already exist; the community/sandboxed path is converging on WebAssembly
Components + WIT, while declarative contributors should use the lower-authority
`data_only` path whenever executable code is unnecessary.

This guide describes the intended author workflow so the host and SDK can be
judged against a concrete usability target.

## Extension classes

Use the least-privileged form that can express the contribution:

1. **data-only pack** — knowledge, ontology, evidence, curriculum, assets;
2. **pure compute component** — deterministic transformation/model;
3. **read-only provider** — permitted observations or external reads;
4. **side-effecting tool** — explicit scoped effects;
5. **safety-critical provider** — actuation or consequential control, subject to
   much stronger host policy.

Do not write executable code when a data-only extension is sufficient.

## Manifest v1

The portable manifest contract is defined by `symthaea-extension-core`. A
machine-readable companion schema lives at:

```text
docs/extension/manifest-v1.schema.json
```

The JSON Schema provides language/editor/tooling validation for the public wire
shape. `ExtensionManifest::validate()` remains authoritative for semantic
structural invariants that JSON Schema cannot express cleanly, such as duplicate
capability IDs with differing descriptions or provided/required overlap.

Manifest v1 is intentionally strict: unknown object fields are rejected by the
Rust contract. Typos should fail rather than be ignored. Permissions are
requests, never grants.

## Data-only / no-code baseline

If the contribution is declarative, use `runtime: "data_only"`. The complete
standalone example is:

```text
examples/extensions/hello-knowledge-pack/
```

A data-only extension contains no native library, Wasm component, process, or
script. The v1 contract requires zero ambient permissions and pure capabilities.
Its invocation resource values may be zero because no guest code is invoked.

Do **not** put an arbitrary universal data blob behind `data_only`. Each semantic
capability should select a versioned payload schema/parser. The example therefore
ships a tiny `hello-records-v1.schema.json` beside `records.json`; that schema is
local to the example capability, not a universal Symthaea knowledge format.

A data-only package still needs size/count limits, exact-byte digests, schema
validation, provenance/signature policy, and domain-specific epistemic/safety
checks. Removing executable code removes one attack surface; it does not turn
untrusted content into truth.

## Rust guest baseline

Modern Rust supports the `wasm32-wasip2` target directly. Guest bindings can be
generated with `wit-bindgen` from the Symthaea control world.

Typical setup:

```bash
rustup target add wasm32-wasip2
cargo add wit-bindgen
```

A guest library uses `crate-type = ["cdylib"]`, places its WIT files under a
`wit/` directory, generates bindings for `extension-control-v1`, implements the
exported control interface, and builds with:

```bash
cargo build --release --target wasm32-wasip2
```

The resulting `.wasm` is a standard WebAssembly Component and should remain
inspectable with ordinary Component Model tools such as:

```bash
wasm-tools component wit target/wasm32-wasip2/release/<name>.wasm
```

## Minimal guest shape

Conceptually:

```rust
wit_bindgen::generate!({
    path: "wit",
    world: "extension-control-v1",
});

struct Extension;

impl exports::luminous::symthaea_extension::control::Guest for Extension {
    fn identity() -> exports::luminous::symthaea_extension::control::ExtensionIdentity {
        // Return values bound to the release manifest.
        todo!()
    }

    fn health() -> exports::luminous::symthaea_extension::control::HealthReport {
        // Report current readiness; this does not grant authority.
        todo!()
    }
}

export!(Extension);
```

The precise generated Rust module paths are a guest-language tooling detail; the
WIT world is the normative contract.

## Manifest binding

Every executable release has a manifest describing identity, ABI version,
capabilities, requested permissions, runtime kind, and resource budgets. The
guest control identity must agree with that admitted manifest.

The control world includes a manifest digest so a host can detect a component
built against a different declaration than the one presented at installation.
For the first host implementation, use SHA-256 of the **exact manifest bytes** as
the 32-byte digest. Distribution layers may also carry their own OCI/content
digests; those serve a different purpose and must not be conflated.

The future SDK/pack command should calculate and inject this digest automatically.
Extension authors should not hand-maintain it.

## Capabilities, not product names

Declare semantic capabilities describing what the extension provides. Examples:

```text
science.astronomy.orbit_propagation
engineering.cfd.steady_state
language.translation.afrikaans
media.audio.transcription
```

Avoid capability IDs named after a provider or implementation. The router must
be able to replace one implementation with another without changing the caller.

## Permissions

Permissions are requests, not grants.

A manifest asking for network access does not receive network access until the
host explicitly admits and grants it. The base control world imports no ambient
filesystem, network, clock, random, environment, sensor, or actuator authority.

Prefer narrow imports over broad WASI worlds. A weather extension that needs one
allowlisted HTTPS endpoint should not inherit a general filesystem and process
environment merely because those APIs are convenient.

## Resource budgets

Executable authors should request realistic upper bounds for memory, compute/fuel,
wall time, output size, and concurrency. Smaller truthful budgets improve routing
and make denial-of-service containment easier.

A host may impose a stricter budget than the manifest requests. A component must
handle interruption/traps as a normal failure mode rather than assuming unlimited
execution.

Data-only packages are never guest-invoked and may use zero invocation budgets.

## Trust and signing

Signing proves continuity/integrity for a signer; it does not make the signer
trusted. Host policy decides which signer identities may provide which extension
classes and capabilities.

Do not design plugins around possession of a global Symthaea signing key. Local
operators, organizations, and curated distribution channels should be able to
maintain independent trust roots.

## Distribution

For development, a manifest + component/data directory is sufficient. For public
publishing, prefer standard Component Model/OCI tooling (`wkg` /
`wasm-pkg-tools`) for executable components rather than a custom Symthaea archive
or registry.

The host should record resolved immutable digests for reproducibility. A future
packager can use ordinary OCI artifacts for data-only packs as well; the package
format must not imply authority.

## What not to depend on

Public extensions should not depend directly on:

- `CognitiveLoopService` fields;
- private HDC/CfC implementation layout;
- arbitrary root-crate internals;
- registration order;
- unrestricted host filesystem paths;
- a particular LLM/model provider;
- a particular OCI registry.

Use versioned WIT/contracts and semantic capabilities instead.

## Developer-experience target

Executable path:

```text
symthaea extension new my-plugin
        ↓
implement typed WIT capability
        ↓
symthaea extension check
        ↓
symthaea extension test
        ↓
symthaea extension pack/sign
        ↓
symthaea extension install ./component.wasm --manifest manifest.json
        ↓
(optional) publish with standard OCI/wkg tooling
```

No-code path:

```text
symthaea extension new --kind knowledge-pack my-pack
        ↓
add data matching capability-specific schema
        ↓
symthaea extension check
        ↓
symthaea extension pack/sign
        ↓
symthaea extension install ./my-pack
```

`check` should validate the manifest, reject unknown fields, validate all
capability-specific payload schemas, inspect component imports/exports when code
exists, verify ABI compatibility, flag excessive permissions, and fail before
installation if the package cannot satisfy its declared contracts.

## Definition of easy

A competent developer who has never opened the Symthaea monorepo should be able
to implement a pure executable extension using only:

- the public WIT package;
- the small extension manifest schema/SDK;
- a template/example;
- ordinary Component Model tooling.

A domain expert whose contribution is declarative should be able to make a
useful data-only extension **without installing Rust or Wasmtime**.

Needing to edit Symthaea's root `Cargo.toml`, root `lib.rs`, cognitive loop, or a
feature list is a failure of the public plugin experience.
