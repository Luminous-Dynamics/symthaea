# AR-5B direction-board reference

Standalone Rust reference for the verifier-grounded direction allocator specified in #9230.

This tool is deliberately advisory-only. It computes a deterministic allocation suggestion over a fixed next-budget envelope. It consumes a separately referenced active-worker allocation snapshot and never confuses already-active work with new recommendations. It does not launch workers, mint direction identities, execute candidates, approve promotions, or act as a scheduler or authority kernel.

## Run

```sh
cargo fmt --manifest-path tools/ar5b-direction-board-reference/Cargo.toml -- --check
cargo clippy --manifest-path tools/ar5b-direction-board-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/ar5b-direction-board-reference/Cargo.toml
cargo run --manifest-path tools/ar5b-direction-board-reference/Cargo.toml -- docs/research/fixtures/AR5B_DIRECTION_BOARD_SYNTHETIC_CAMPAIGN_V0_2026-11-07.json
```

## Allocation rule

For each verifier-qualified direction, the reference:

1. Retains that direction's own local best; it never substitutes the global best.
2. Computes observed gain per total cost, including the cost of rejected, inconclusive and invalid attempts. Inconclusive/invalid attempts contribute zero observed effect, not zero cost.
3. Includes negative paired deltas from rejected attempts.
4. Shrinks sparse-direction gain estimates toward the campaign-wide pooled gain-per-cost prior (or the frozen empty-history prior when no outcomes exist).
5. Adds a cost-adjusted exploration bonus that decreases as the direction receives more evaluations.
6. Applies an explicit per-direction maximum share of the next budget.
7. Offers a separate, priced `NEW` option with a maximum share and verifier-intake reference. `NEW` is a budget reservation for verifier intake; it is not a candidate-minted direction or an admission decision.

The greedy allocation loop uses deterministic lexical tie-breaking and never spends more than the declared available budget or any candidate's share cap. The total campaign envelope must reconcile **completed historical attempt costs + active-worker costs + next-decision available costs + reserved/unavailable costs**. The report distinguishes all four categories plus new recommendations and the unallocated remainder of available budget. Completed history is charged even when an attempt was invalid or inconclusive. Active allocations are sourced from the referenced external snapshot; the board does not mint or mutate that registry.

## Identity and evidence requirements

Every existing direction must include:
- verifier-assigned ID and canonical descriptor fingerprint;
- target capability, solution-object, construction-principle and scope/assumption digests;
- verifier assignment and protected-capability audit references;
- a qualified status and passing protected-capability floor;
- active evaluation count and claimed cost from the independent worker-allocation snapshot;
- paired observations bound to task-family digests, local-best deltas, dispositions and costs.

The validator rejects repeated IDs/fingerprints, unqualified directions, missing verifier references, failed capability floors, invalid costs, and unsupported positive outcomes on inconclusive/invalid observations.

## Limitations

- Fingerprint uniqueness checks only the verifier-supplied canonical fingerprints. It cannot determine whether two descriptions are semantic aliases; the independent verifier must canonicalize and audit them.
- The prior/shrinkage/UCB-like score is a transparent heuristic, not a calibrated posterior, confidence interval, causal effect or optimal bandit policy.
- Expected evaluation costs and protected-floor declarations are inputs; they require independent measurement/audit.
- Active worker allocation counts/costs are trusted only as far as the referenced snapshot; this crate does not authenticate that registry.
- The total-budget invariant includes historical attempts, active work and remaining availability; otherwise search costs from prior generations could be silently externalized.
- The reference does not perform task-family transfer testing or prove a direction's scientific novelty.
- A valid allocation report is a recommendation under the declared input and budget, not evidence that the board improves research outcomes or that the outer-loop improver has improved.

The crate is outside the root Cargo workspace and does not add production dependencies or services.
