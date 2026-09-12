# Hello Control Extension

Minimal standalone Rust guest for Symthaea's public extension control world.

This directory intentionally declares its own Cargo workspace so it does not
inherit Symthaea's monorepo dependency graph. It should be copyable into an
independent repository.

## Build

```bash
rustup target add wasm32-wasip2
cargo build --release --target wasm32-wasip2
```

The component will be emitted under:

```text
target/wasm32-wasip2/release/symthaea_hello_control_extension.wasm
```

Inspect its public Component Model surface with:

```bash
wasm-tools component wit \
  target/wasm32-wasip2/release/symthaea_hello_control_extension.wasm
```

## What this example demonstrates

- no dependency on Symthaea's Rust implementation crates;
- WIT as the public guest contract;
- no ambient host imports in the baseline control world;
- build-time SHA-256 binding to the exact `manifest.json` bytes;
- explicit identity/ABI reporting;
- explicit health reporting;
- standard `wasm32-wasip2` + `wit-bindgen` tooling.

## What it deliberately does not demonstrate

This guest provides no useful domain capability yet. It is only the smallest
control-plane compatibility fixture. Capability-specific examples should be
added when the corresponding typed WIT worlds and host consumers exist.

It also does not sign or publish itself. Signing/admission belongs to the future
Symthaea extension host/tooling; public transport should use standard
Component/OCI tooling where possible.

## Security property

Changing `manifest.json` changes the digest embedded into the component on the
next build. A host can therefore reject a component presented alongside a
different declaration even when the human-readable ID/version happen to match.
