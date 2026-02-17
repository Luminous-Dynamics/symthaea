# mycelix-hyperfeel

Minimal **Rust-first HyperFeel core** for Mycelix.

This crate provides:

- A backend-agnostic `Hypervector` type (fixed 16,384 dimension)
- A `HyperGradient` struct for hypervector-encoded gradients
- A `ModelAdapter` trait for connecting arbitrary models to HyperFeel
- A small CLI (`hyperfeel_cli`) for use from Python or shell scripts
- An optional Symthaea-backed HDC backend (`symthaea-hv` feature)

It is deliberately small and self-contained so it can:

- Be integrated into `Mycelix-Core/0TML` experiments
- Later swap its internal `Hypervector` representation for Symthaea's HV16
- Compile to WASM or be embedded in Holochain zomes

The implementation here is a starting point, not the final design:

- Encoding from dense gradients is simple and lossy but safe.
- Aggregation uses straightforward bundling + normalization.
- Cryptographic signatures and advanced HDC backends are intentionally left
  as future integrations.

## Symthaea integration (optional)

When built with the `symthaea-hv` feature, this crate depends on the
`symthaea-hlb` repository and exposes conversion helpers between the local
`Hypervector` type and Symthaea's `BinaryHV`:

- See `symthaea_backend::{to_symthaea_binary, from_symthaea_binary}`.
- This makes it possible to reuse Symthaea's HDC operations and Φ tooling
  without changing the HyperFeel public API.

Example (from `mycelix-hyperfeel` directory):

```bash
cargo test --features symthaea-hv
```

## CLI usage

Build and run (once Rust is available on your system):

```bash
cargo build --release

# Encode a gradient provided as JSON
echo '{"gradient":[0.1,-0.3,0.7]}' \
  | target/release/hyperfeel_cli encode

# Aggregate a list of hypergradients
echo '[{"vector":{"data":[0,0,0]}, "layer_info":[], "timestamp":0, "signature":{"bytes":[]}}]' \
  | target/release/hyperfeel_cli aggregate
```
