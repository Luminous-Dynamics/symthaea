# symthaea-extension-admission-bridge

Thin adapter between the public WebAssembly Component control host and the
runtime-neutral extension admission-policy layer.

The bridge intentionally does only this:

```text
manifest bytes + component bytes
        -> ControlPlaneHost
        -> ControlInspection
        -> TechnicalInspection
```

It does **not** verify signatures, resolve signer trust, choose permissions,
issue admission policy, route capabilities, or execute a useful plugin
capability. Those remain separate layers.

A successful `TechnicalInspection` therefore means only that the exact Wasm
component passed the configured zero-authority technical host and was bound to
the exact manifest/component digests. It is not an admission decision.

## Dependency direction

```text
extension-core
      ^
      |
admission <- admission-policy
                     ^
                     |
extension-host <- admission-bridge
```

Wasmtime stays in `crates/bridges`; no core admission crate depends on the Wasm
runtime.

## Fail-closed properties

- host size/format/ABI/resource failures propagate unchanged;
- non-Wasm/data-only packages cannot pass through the Component inspector;
- the adapter does not manufacture or modify hashes;
- signer authorization and local policy still run after this bridge;
- data-only packs use a separate typed data-loader path and never become Wasm
  invocation authority.

The first positive end-to-end test should use the real `hello-control` Component
fixture from the authoring stack rather than a handcrafted mock component.
