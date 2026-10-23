# symthaea-extension-core

Stable, dependency-light contracts for Symthaea extensions.

This crate defines the shared vocabulary used before any extension is loaded or
trusted. It contains no cognitive implementation, Wasmtime runtime, networking,
solver integration, signature verifier, or trust database.

## Contract layers

```text
ExtensionManifest
    -> discovery / structural validation
    -> registry
    -> host-specific technical inspection
    -> signer + admission policy
    -> point-of-use authorization
    -> invocation
```

A valid manifest is only a structurally coherent declaration. It is not proof
that an extension is trusted, installed, executable, scientifically correct, or
authorized for an invocation.

## Strict v1 parsing

Named manifest objects reject unknown fields. This is deliberate: a typo such as
`permisisons` must fail parsing rather than silently falling back to the
deny-by-default permission set and leaving authors with a misleading declaration.

Structural validation also rejects:

- malformed non-namespaced extension/capability IDs;
- blank or non-canonical name/version strings;
- duplicate provided/required capabilities;
- a capability that is simultaneously provided and required by the same
  extension;
- blank, non-canonical, control-character, or duplicate permission entries;
- zero invocation budgets for executable/native/remote providers.

Hosts remain responsible for semantic interpretation of filesystem paths,
network destinations, hardware capability names, trust, and platform policy.

## Data-only extensions

`RuntimeKind::DataOnly` is a real zero-executable-authority class for knowledge,
evidence, ontology, curriculum, grammar, schema, and asset packs.

A data-only manifest:

- may use zero invocation budgets because it is never executed;
- cannot request network, filesystem, GPU, clock, randomness, sensor, or
  actuator authority;
- may expose only `EffectClass::Pure` capabilities.

The package loader still needs to validate the content format, schema, digest,
provenance, size, and domain-specific safety rules. `DataOnly` means "no guest
code executes"; it does not mean "all contained data is automatically safe or
true."

## Example

```rust
use symthaea_extension_core::{
    AbiVersion, CapabilityDescriptor, CapabilityId, EffectClass, ExtensionId,
    ExtensionKind, ExtensionManifest, PermissionSet, ResourceBudget, RuntimeKind,
};

let manifest = ExtensionManifest {
    id: ExtensionId::new("org.example.astronomy"),
    name: "Example Astronomy".into(),
    version: "1.0.0".into(),
    abi: AbiVersion::V1,
    kind: ExtensionKind::Domain,
    runtime: RuntimeKind::Wasm,
    description: "Orbit and photometry helpers".into(),
    provides: vec![CapabilityDescriptor {
        id: CapabilityId::new("science.astronomy.orbit_propagation"),
        description: "Propagate an orbit from an initial state".into(),
        effect: EffectClass::Pure,
    }],
    requires: vec![],
    permissions: PermissionSet::default(),
    resources: ResourceBudget::default(),
};

manifest.validate()?;
# Ok::<(), Vec<symthaea_extension_core::ManifestProblem>>(())
```

## Lean boundary

Keep this crate boring. New fields or abstractions should land only with a real
consumer. Runtime lifecycle, Wasm loading, package distribution, trust stores,
routing quality, and cognitive policy belong in higher layers.
