# Contributing to Mycelix Mail

Thanks for helping evolve Mycelix Mail! This repository is self-contained, so
you no longer need the giant `Mycelix-Core` workspace just to land a fix.
Everything required to build, test, and lint the DNA, CLI, DID registry, and
MATL bridge lives here.

## Ways to help

| Area | Tasks |
| --- | --- |
| DNA (`dna/`) | New zome functionality, validation rules, spam-report flows |
| CLI (`cli/`) | Wire Holochain/MATL/DID APIs, UX polish, packaging |
| DID Registry (`did-registry/`) | FastAPI + Postgres service, observability |
| MATL Bridge (`matl-bridge/`) | Websocket integration with Holochain + Postgres |
| Docs & Tests | Keep our guides, diagrams, and integration suite current |

## Development workflow

1. **Fork & clone** this repo or create a fresh branch locally.
2. **Install prerequisites** (Rust stable + wasm32 target). The simplest path is
   via `nix develop` using the provided `shell.nix`.
3. **Build DNA**:
   ```bash
   cd dna
   cargo build --release --target wasm32-unknown-unknown
   hc dna pack .
   ```
4. **Run CLI checks**:
   ```bash
   cd cli
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test
   ```
5. **Run integration tests**:
   ```bash
   python3 tests/integration_test_suite.py
   ```
6. **Keep commits focused**. Small, readable commits speed up reviews.
7. **Document changes**. Update READMEs or markdown guides when behaviour
   changes, and highlight new commands or environment requirements.

## Pull request expectations

- All tests that previously passed should still pass.
- New features must include tests (Rust, Python, or both) and documentation.
- Explain *why* the change is needed and reference any relevant issues.
- Screenshots or CLI transcripts are invaluable for UX‑level work.

## Code of conduct

Be kind. Assume good intent. Degenerate behaviour, harassment, or other
unprofessional conduct is not tolerated. If you experience issues, reach out to
the maintainers privately.

## Questions?

Open a GitHub Discussion or start an issue if you are unsure how to implement an
idea. Happy hacking! 🍄
