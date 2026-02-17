# Next Cleanup Targets

While pytest runs, here is a short roadmap for tidying the remaining high-noise
areas.

## production-fl-system/
- [x] Move documentation/config/results/scripts into dedicated subfolders
- [x] Identify outdated Python demos and relocate into an `archive/`
- [ ] Review `archive/` inventory, promote any living scripts into `tools/` or `zerotrustml`, and log decisions
- [ ] Extract reusable code into the ZeroTrustML package or `tools/python/`
- [x] Replace legacy `run-mycelix` demo with Phase 10 modular demo (`just run-demo`)
- [ ] Decide which archived PoGQ/aggregation harnesses become part of the new edge SDK and either delete or document the remainder

## scripts/
- [x] Consolidate remaining helpers into `tools/python/` / `tools/scripts/`
- [ ] Tag actively-maintained scripts with usage comments and migrate the rest into `just` recipes *(conductor launchers + installers/monitoring archived; review build/deploy next)*
- [ ] Remove the empty directory from git history once committed
- [ ] Update `run_edge_proof_integration.py` consumers after the EdgeClient SDK stabilises

## Results directories
- Normalize naming in `results/` (`results/<date>/<scenario>.json`) and link the
  important ones from documentation
- Delete or archive stale `.db` files after capturing summaries
- [ ] Store BFT matrix outputs (`cifar10`, `emnist_balanced`, `breast_cancer`) as JSON and link in docs
- [ ] Track label-skew failures in matrix results, run `scripts/sweep_label_skew.py` sweeps until all attacks go green (cover 30% and 40% ratios)

## Testing doc sync
- Once the background pytest run finishes, capture key findings and record them
  in `0TML/docs/testing/` so the roadmap reflects the latest execution status
- [ ] Add Holonix sandbox instructions once conductor automation is available
