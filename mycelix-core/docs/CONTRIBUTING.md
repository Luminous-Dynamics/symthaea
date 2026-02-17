# Contributing

This is currently a solo project, but the repo is organized as a small workspace
so future collaborators (or future-you) can find things quickly.

## Repository Layout

- `src/` — Rust crate for the Mycelix core coordinator and WASM modules
- `bindings/python/` — Python bridges that talk to the Rust coordinator
- `0TML/` — ZeroTrustML Python package and documentation
- `tools/scripts/` — shell helpers for deployment, builds, and experiments
- `tools/python/` — Python demos, experiments, and one-off utilities
- `docs/` — curated documentation (`docs/root-notes/` holds archived status logs)

## Common Tasks

Use the `justfile` (see below) to keep commands consistent.

```bash
just help            # list available recipes
just check-rust      # run cargo check
just py-install      # install Poetry deps
just py-test         # run pytest via Poetry
```

## Coding Standards

- Prefer Rust 2021 edition conventions for the crate
- Python code should use `ruff` and `black` (both configured via Poetry)
- Keep documentation in `docs/` or `0TML/docs/`; archive older status notes into
  the corresponding `root-notes/` directory instead of leaving files at the repo
  root

Feel free to adapt the structure as the project evolves—update this file and the
`README.md` when you do.
