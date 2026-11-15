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


## Numerical authority and stale-snapshot warning (2026-11-14)

<blockquote>
**This vendored directory is a historical, excluded snapshot. Do not use it for current quantitative results.**

The authoritative implementation and numerical benchmark work live in the standalone
[ Luminous-Dynamics/spark-engine repository ](https://github.com/Luminous-Dynamics/spark-engine),
not in this copy. The standalone project has identified and is correcting a Maxwellian
reactivity prefactor/unit-conversion defect and a D-D branch S-factor normalization defect.
Its correction is tracked in [draft PR #7](https://github.com/Luminous-Dynamics/spark-engine/pull/7),
with a new [NRL reference benchmark](https://github.com/Luminous-Dynamics/spark-engine/blob/fix/dd-reactivity-unit-baseline-2026/docs/DD_MAXWELLIAN_REACTIVITY_REFERENCE_BENCHMARK.md).

The correction is not yet merged or CI-verified. The 300 K lattice/beam regime remains outside
the 1–20 keV Maxwellian reference benchmark. Neither this snapshot nor the standalone draft
establishes a validated lattice reaction rate, a net-energy reactor or a qualified neutron source.
</blockquote>

This directory remains excluded from the Symthaea workspace and has no application consumers.
It is not a second physics authority. Do not independently patch or consume its numerical outputs.
Any future integration must use a versioned standalone Spark dependency or an explicit evidence
adapter with pinned revision and model-validity metadata. See
[SPARK-VENDORED-SNAPSHOT-DRIFT-001](https://github.com/Luminous-Dynamics/symthaea/issues/12088).
