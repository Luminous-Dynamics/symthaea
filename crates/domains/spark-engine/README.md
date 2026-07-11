# spark-engine — Lattice Confinement Fusion physics (research-only, standalone)

**Scope note (2026-07-06):** despite living under `crates/domains/`, this crate
is **not a Symthaea domain plugin and not part of the application build**:

- It is a standalone nuclear-fusion (LCF) physics simulator — Gamow integration,
  Q-factor analysis, neutron-source design, NASA anomaly investigation.
- It is explicitly `exclude`d from the workspace (`symthaea/Cargo.toml`) and has
  **zero consumers** in `src/` or any other crate.
- Its `bridge.rs` Symthaea-integration hooks are placeholders, not real wiring.
- It has nothing to do with the creative/art crates that share this directory
  (`symthaea-muse`, `symthaea-atelier`, `symthaea-aesthetic`, `symthaea-canvas`).

Treat it as an independent research project that happens to be vendored here.
Do not cite it in Symthaea capability inventories, and do not add it to art or
domain-plugin reviews. A relocation out of `crates/domains/` is the eventual
clean fix (see `ART_CULTURE_REVIEW_AND_PLAN_2026-07-06.md` Phase 0).

Build/test it directly:

```bash
cd crates/domains/spark-engine
cargo test
cargo run --bin spark -- analyze
```

**Published** (2026-07-07): standalone repo at
[github.com/Luminous-Dynamics/spark-engine](https://github.com/Luminous-Dynamics/spark-engine)
(with its own public-facing README), crate on
[crates.io](https://crates.io/crates/spark-engine). `docs/TECHNICAL_REPORT.pdf`
(regenerate with `docs/render_pdf.sh`) is the citable technical report.
