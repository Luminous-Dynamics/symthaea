# Hello Simulation Extension

This is a standalone third-party authoring fixture for Symthaea's typed
`simulation-provider-v1` Component Model ABI.

It deliberately implements only a synthetic `custom` solver. The returned metric
is the sum of the supplied parameter values, with zero confidence and maximal
epistemic uncertainty. It is useful for proving the typed request/response path;
it is not an engineering solver and its output must never be promoted to
engineering evidence by itself.

The manifest advertises `engineering.simulation.custom`, requests no filesystem,
network, clock, randomness, sensor, actuator, or GPU authority, and binds guest
identity to the exact manifest bytes at build time.

The copied WIT files are required to remain byte-for-byte identical to the
canonical public WIT under `crates/core/symthaea-extension-core/wit/`; the focused
Extension Authoring workflow enforces that relationship.

Build it independently from the Symthaea workspace:

```bash
cargo build --release --target wasm32-wasip2
```

The expected artifact is:

`target/wasm32-wasip2/release/symthaea_hello_simulation_extension.wasm`
