# mycelix_fl_core Zome (Source Placeholder)

**Status**: Compiled WASM present, source layout to be re-materialised  
**Scope**: Canonical Holochain FL coordinator/integrity zome for Mycelix FL

This directory currently contains:

- `target/` – compiled artifacts for the canonical Mycelix FL zomes.
- `tests/` – test harness and fixtures used when the zome sources were last compiled.
- `src/` – empty, to avoid guessing at the original crate structure.

The intended *source* for these zomes is described, not reconstructed, in:

- `Mycelix-Core/docs/HOLOCHAIN_FL_AGGREGATION_DESIGN.md`

That design specifies three key externs that should live in the core coordinator zome:

- `aggregate_gradients` – wraps the `fl-aggregator` Rust crate for Krum/Median/TrimmedMean/FedAvg.
- `classify_round_participants` – per-node PoGQ/MATL classification (Honest vs Byzantine).
- `evaluate_round_security` – round-level decision (`Ok` / `Escalate` / `Halt`) using AdaptiveByzantineThreshold + CartelDetector.

## How to Recreate the Zome Source

1. Create a new Rust crate in `src/` (e.g. `lib.rs`) following the standard Holochain 0.6/0.7 patterns (`hdk::prelude::*`, `#[hdk_extern]`).
2. Implement the three externs exactly as specified in the design doc, wiring:
   - `get_round_updates(round_id)` from the Agents/FL zome,
   - `fetch_gradient_bytes` / `bytes_to_gradient` bridge into 0TML or another storage backend,
   - `fl_aggregator` and `mycelix_sdk::matl` for detection and aggregation.
3. Update the DNA manifest to point at the new zome crate and re-run:
   - `hc dna pack` and `hc app pack` under `0TML/mycelix_fl/holochain/`.
4. Unskip the FL BFT tests in:
   - `mycelix-workspace/tests/ecosystem/tests/fl-bft.test.ts`
   and iterate until all three “rings” pass.

Until those steps are taken, this directory should be treated as a compiled, historical artifact, with the design doc as the canonical source of truth for future Rust/Holochain implementations.

