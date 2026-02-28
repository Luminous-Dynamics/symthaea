# Contributing to Fabrication hApp

## Prerequisites

- Rust 1.92+ with `wasm32-unknown-unknown` target
- [Holochain CLI](https://developer.holochain.org) (`hc`)
- NixOS users: `nix develop` from the repo root

## Project Structure

```
fabrication/
├── crates/fabrication_common/   # Shared types, validation, config
├── zomes/
│   ├── designs/                 # Parametric design CRUD + versioning
│   ├── printers/                # Printer registry + spatial queries
│   ├── prints/                  # Print jobs + Cincinnati monitoring
│   ├── materials/               # Material catalog + certifications
│   ├── verification/            # Safety claims + design verification
│   ├── bridge/                  # Cross-hApp integration + audit trail
│   └── symthaea/                # HDC semantic search + AI generation
├── dna/dna.yaml                 # DNA manifest (7 integrity + 7 coordinator)
├── happ.yaml                    # hApp bundle manifest
└── tests/sweettest/             # Sweettest integration tests (standalone crate)
```

Each zome has an `integrity/` (validation rules, entry/link types) and `coordinator/` (business logic, CRUD).

## Build & Test

```bash
# Unit tests (native)
cargo test --target x86_64-unknown-linux-gnu

# WASM compilation (all 14 zomes)
cargo build --release --target wasm32-unknown-unknown

# Clippy
cargo clippy --workspace --no-deps --lib --target wasm32-unknown-unknown -- -D warnings

# Format
cargo fmt --all --check

# Pack DNA + hApp
hc dna pack dna/ -o workdir/fabrication.dna
hc app pack . -o fabrication.happ
```

## Validation Conventions

All integrity zomes follow these patterns (aligned with mycelix-commons/civic clusters):

1. **Float fields**: `is_finite()` guard via `validation::require_in_range()` — rejects NaN/Inf
2. **String fields**: `require_non_empty()` (trims whitespace) + `require_max_len()`
3. **Vec fields**: `require_max_vec_len()` on the collection, `require_max_len()` on each item
4. **Link tags**: `require_max_tag_len()` in `FlatOp::RegisterCreateLink` (default 256 bytes)
5. **Delete links**: Author check in `FlatOp::RegisterDeleteLink` via `must_get_action()`
6. **Coordinator auth**: Update/delete operations verify `agent_info()` matches the original author

Use the `check!()` macro from `fabrication_common` to short-circuit on validation failure.

## Adding a New Entry Type

1. Define the struct in the relevant `integrity/src/lib.rs` with `#[hdk_entry_helper]`
2. Add it to the `EntryTypes` enum
3. Write a `validate_*()` function covering all fields per the conventions above
4. Add unit tests for valid input, boundary values, and rejection cases
5. Wire CRUD in the matching `coordinator/src/lib.rs`
6. Add coordinator auth checks on update/delete operations

## Shared Types

All types shared across zomes live in `crates/fabrication_common/src/lib.rs`:
- `FabricationConfig` — loaded from DNA properties, cached per-thread
- `FabricationError` — custom error enum (replaces ad-hoc `wasm_error!` strings)
- `TypedFabricationSignal` — domain + event_type + payload
- `PaginationInput` / `PaginatedResponse<T>` — shared pagination
- `validation` module — reusable validation helpers

## CI Pipeline

The GitHub Actions workflow (`fabrication-ci.yml`) runs:
- **fmt** — `cargo fmt --check`
- **clippy** — WASM-target lint with `-D warnings`
- **test** — All unit tests (native target)
- **wasm** — Compile all 14 zomes, verify count
- **happ-pack** — DNA + hApp bundle generation
- **kernel** — Fabrication kernel tests (in symthaea workspace)
