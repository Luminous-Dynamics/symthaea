# Symthaea Landscape Review

A read-only, deterministic evidence coverage and cross-artifact binding checker for landscape alternatives. It is not a simulation, metric calculator, recommendation ranker, causal inference engine, consent validator, or source downloader.

See [LANDSCAPE_REVIEW_PROFILE_V0.md](LANDSCAPE_REVIEW_PROFILE_V0.md) for the contract, upstream bindings and benchmark limits.

Run:

    cargo fmt --check --package symthaea-landscape-review
    cargo test -p symthaea-landscape-review

The v0 output hard-disables ranking, environmental impact claims and causal claims because the accepted Sol-Atlas project snapshots use schematic local drawing units.
