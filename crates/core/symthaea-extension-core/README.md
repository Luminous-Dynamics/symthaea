# symthaea-extension-core

Dependency-light contracts for describing Symthaea extensions.

This crate is intentionally **not** an extension loader. It gives native, WASM,
remote, and data-only extensions one stable vocabulary for identity, capability
discovery, permissions, and resource budgets without depending on the cognitive
loop or any domain implementation.

## Design rules

- The cognitive core does not depend on third-party extension implementations.
- Capabilities describe **what** can be done; providers remain replaceable.
- Permissions deny ambient authority by default.
- Resource budgets are explicit and host-enforced.
- Runtime trust policy belongs to the host, not this data contract.
- New manifest fields should be added only when a real host/provider consumer needs them.

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
