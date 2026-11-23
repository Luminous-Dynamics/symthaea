# ProofPlanV1 standalone contract checker

This crate validates the **structure and references** of ProofPlanV1 documents. It does not check mathematical inference, run Lean, certify the theorem, or replace expert review.

## Run

From the repository root:

```bash
cargo test --manifest-path scripts/research/proof-plan-contract/Cargo.toml --offline
cargo run --manifest-path scripts/research/proof-plan-contract/Cargo.toml --offline -- docs/research/PROOF_PLAN_AND_COMM_V1.json
```

The crate has its own workspace boundary. It does not add the large Symthaea workspace to the test graph.

## What is checked

- typed JSON deserialization with unknown-field rejection;
- unique node and evidence IDs;
- root goal and alternative references;
- edge endpoints and evidence references;
- action input/output goals, justifications and side conditions;
- required evidence for formal facts and mathematical interpretations;
- explicit boundaries for analogies and reasons for unknown steps;
- Lean kernel and independent-review status cannot be promoted without matching evidence.

## What is not checked

- whether a proposed mathematical inference is sound;
- whether the Lean proof compiles or the kernel accepts it;
- whether a declaration is equivalent to a paper claim;
- whether the reconstructed plan is expert-approved;
- whether a reader understands or transfers the mathematics.

A successful output is **structural validation only**. Kernel checking, mathematical review, and human-study outcomes remain separate evidence.
