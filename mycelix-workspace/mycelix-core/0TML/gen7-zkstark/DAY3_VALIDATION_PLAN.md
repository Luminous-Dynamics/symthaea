Mycelix Core — Day 3 Validation Sprint
=====================================

🎯 Objectives
-------------

Lock in the secure coordinator, reproducible weights, and reproducible paper build before starting zkVM guest integration.

🧩 Tasks
--------

| Area | Action | Acceptance Criteria |
| --- | --- | --- |
| Coordinator readiness | Implement `start_with_ready()` (oneshot channel) and remove sleeps; add ping/pong and heartbeat timeout tests | All integration tests pass deterministically < 2 s; logs show `Client authenticated`, `Ping`, `Close: Policy Violation` lines |
| Criterion summary | Extend CI job to upload `target/criterion/loss/*/estimates.json` as artifact; print mean/stddev in summary | CI output includes perf numbers; skipped run exits 0 cleanly |
| Model determinism | Freeze `0TML/models/canary_cnn.pt` using `torch.manual_seed(42)` + Xavier init; document in `scripts/export_weights.py`; compute and store SHA256 | Re-export yields identical SHA (`205e7f35bac9dd2322d5eb0fcdce638c3e65711d205e78e499cfc210e766eb5b`); CI fails if mismatch |
| Paper reproducibility | Run `paper-submission/scripts/build_results.py` in CI; generate `generated/stats.tex` + plots with ± σ; attach compiled PDF | PDF artifact matches regenerated figures/tables |
| Smoke → integration path | Add minimal end-to-end test: detector ↔ coordinator ↔ agent ↔ DHT stub | Test passes locally; ready for zkVM guest wiring |

🧪 Verification
---------------

- `cargo test -p mycelix-core -- --nocapture`
- `poetry run pytest -q 0TML/tests`
- `cargo bench -p vsv-core --features bench-fixtures -- --noplot`

🏁 Deliverables
---------------

- ✅ `src/coordinator.rs` – ready-signaling + ping/pong tests
- ✅ `.github/workflows/ci.yml` – Criterion artifact + summary
- ✅ `scripts/export_weights.py` – seeded init + SHA check
- ✅ `paper-submission/generated/*` – `stats.tex`, updated plots
- ✅ CI PDF artifact showing mean ± σ
