# Winterfell AIR Status (December 2025)

**Last updated:** 12 Nov 2025  
**Owner:** Core ZK/PoGQ group  

## TL;DR
The repository no longer contains the `vsv-stark/winterfell-pogq/src/` crate that all documentation references. Only `DEBUGGING_GUIDE.md` and an empty `benches/` directory remain (`ls vsv-stark/winterfell-pogq`), which means:

- `cargo build -p winterfell-pogq` and `cargo test -p winterfell-pogq` cannot run.
- Table VII-bis/dual-backend numbers cannot be regenerated.
- Paper sections describing the dual-backend implementation are currently unverifiable.

Until the Rust sources are recovered, we are freezing Winterfell-related instructions and positioning the zkVM (RISC Zero) prover as the only supported backend.

## Evidence

| Proof point | Command / file |
| --- | --- |
| Workspace references missing crate | `README_START_HERE.md`, `NEXT_SESSION_PRIORITIES.md`, `WINTERFELL_IMPLEMENTATION_SUMMARY.md` |
| Directory exists but lacks source | `ls vsv-stark/winterfell-pogq` → only `benches/`, `DEBUGGING_GUIDE.md` |
| Git history empty for crate | `git log -- vsv-stark/winterfell-pogq` → no commits |

Artifacts such as `benchmark_outputs/winterfell/*.json` and `DEBUGGING_GUIDE.md` confirm the crate once existed, but no buildable code is available in this snapshot.

## Decision
1. **Short term (this repo snapshot):** Update README/quick-start docs to state that Winterfell AIR is temporarily unavailable and that all instructions presently target the RISC Zero backend.
2. **Medium term:** Attempt recovery from prior archives (e.g., `docs/archive_20251111/` references) or upstream repositories. If recovery fails, re-scope the paper/results to zkVM-only and remove Winterfell promises entirely.

## Next Actions
- [ ] Inventory off-repo backups or branches that might contain the crate (`gen7-zkstark` does not).
- [ ] Replace Winterfell commands in onboarding docs with links to this status note.
- [ ] Track recovery progress in `STATUS_DASHBOARD.md` and tie into the improvement plan’s WS-A.

Please keep this file updated as soon as the crate is recovered or permanently de-scoped.

---

### Update — 12 Nov 2025
- Searched the entire workspace (including `gen7-zkstark/`) for Rust sources such as `air.rs`, `trace.rs`, or `Cargo.toml` entries referencing `winterfell-pogq`; none exist beyond `DEBUGGING_GUIDE.md`.
- `git log -- vsv-stark/winterfell-pogq` shows no tracked history, confirming the crate was never committed in this checkout.
- Decision: proceed under the assumption that Winterfell cannot be rebuilt without external backups. Focus near-term efforts on (a) aligning documentation with the zkVM-only reality and (b) contacting whoever has archival artifacts. Update this page and `IMPROVEMENT_PLAN_DEC2025.md` once recovery efforts produce news.
