# Crates

Sub-crates for specific functionality, organised by tier:

- `core/` — substrate and shared contracts
- `domains/` — domain-specific capability crates
- `bridges/` — integration boundaries to external tools

## There is deliberately no crate list here

An earlier version of this file hand-documented two crates, `sophia-gym` and
`symthaea-gym`, with usage examples and `cargo build -p …` instructions. **Neither exists
in this workspace.** They were removed at some point and this file was never updated, so
its build commands had been wrong long enough that nobody noticed — the predictable fate
of any hand-maintained list of 200+ packages.

The inventory is therefore **generated**, never written down:

```bash
cargo xtask crate-status --report     # full inventory, derived from `cargo metadata`
cargo xtask crate-status              # integrity check + classification coverage
```

## A crate's existence does not imply endorsement

The generated inventory is joined with `docs/crate-status.toml`, a registry of the
judgements that cannot be derived from code: lifecycle, evidence level (E0–E5), what that
evidence actually rests on, the commit it was last confirmed at, safety criticality, and
whether the crate may appear on a production path.

Most crates are currently **unclassified**, and the tooling prints that count on every run
rather than hiding it. Unclassified means nobody has assessed the crate — a different and
more useful statement than a confident-looking label nobody earned. Do not read a crate's
presence in this directory as a claim that it is current, validated, or safe to depend on.

See `xtask/src/crate_status.rs` for the schema and the integrity rules it enforces.

## Engineering crates

The engineering track is intentionally split into lightweight crates that keep external
CAD/solver dependencies out of default builds. All eight were verified present at the time
of writing; check the generated inventory rather than trusting this list if it matters.

- `symthaea-sim-bridge`: normalized simulation request/result types and backend traits for
  FEA, CFD, multibody, circuit, and process tools.
- `symthaea-digital-twin`: engineered asset telemetry, health, and free-energy trend
  tracking.
- `symthaea-formal-safety`: safety cases, proof obligations, and evidence records.
- `symthaea-engineering`: facade tying requirements, concepts, simulations, twins, and
  safety gates together.
- `symthaea-mujoco-bridge`: dry-run generic MuJoCo backend boundary.
- `symthaea-opensees-bridge`: dry-run OpenSees structural backend boundary.
- `symthaea-ngspice-bridge`: dry-run ngspice circuit backend boundary.
- `symthaea-openfoam-bridge`: dry-run OpenFOAM CFD backend boundary.

See `docs/engineering/SYMTHAEA_ENGINEERING_ROADMAP.md` for the 18-month plan.
