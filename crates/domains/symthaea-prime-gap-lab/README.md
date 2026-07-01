# Symthaea Prime Gap Lab (v1.0)

A hardened, automated research workbench for exploring k-tuple prime gap candidates.

## Features
- **Autonomous Search:** Iterative generation and ranking of admissible k-tuples.
- **Adaptive Repair:** Diagnostic loop identifying parity-barrier failures and shifting candidates.
- **Epistemic Discipline:** Claim ledger enforcing clear boundaries between heuristic models and formal proofs.
- **Formal Pipeline:** Direct bridge to Lean 4 for formal verification of admissible candidates.

## Usage
The workbench is orchestrated via `crates/symthaea-prime-gap-lab`.

```bash
# Run the integrated research pipeline
cargo run -p symthaea-prime-gap-lab --example demo
```

## Evidence Discipline
Every claim must be categorized:
- **Heuristic:** Supported by singular series approximations.
- **Computational:** Verified by simulation.
- **Proven:** Formally verified via the Lean 4 proof bridge.

All research progress is tracked in `docs/experiments/prime-gap-lab/latest-claim-ledger.md`.
