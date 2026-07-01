# API Inventory and Alpha Stability

Alpha.9 adds a dependency-free API inventory so local scripts, notebooks, and downstream wrappers can inspect the crate surface without scraping docs.

The inventory includes:

- crate version
- schema labels
- run presets
- fixture names
- replay scopes
- public-surface stability records
- global caveats

The stability catalog is not a SemVer guarantee. It is a release-note surface that separates stable-alpha utilities from experimental probes and future integration boundaries.

## CLI

Run:

`cargo run --bin symthaea-quantum-comp -- inventory`

## Example

Run:

`cargo run --example api_inventory`

## Caveats

The inventory does not imply quantum backend execution, quantum advantage, or Mycelix attestation. It is a local alpha documentation aid.
