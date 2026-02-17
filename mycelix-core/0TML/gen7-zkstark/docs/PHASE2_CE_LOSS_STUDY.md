# Phase 2 Study: Cross-Entropy Approximation Costing

## Objective
Quantify the constraint and proving-time overhead of introducing cross-entropy-like behaviour after the Phase 1 zkVM baseline (MSE-on-logits) is stable.

## Questions to Answer
1. What is the AIR constraint increase for:
   - Direct exp/log implementations?
   - Polynomial (Taylor/Chebyshev) approximations?
   - Lookup-table hybrids?
2. How does proving time change relative to the MSE baseline on:
   - 16-sample batches
   - 32-sample batches
3. What accuracy drift is introduced by each approximation path (vs. true cross-entropy)?

## Proposed Methodology
1. Start from the existing fixed-point pipeline and add a feature-flagged cross-entropy routine.
2. Implement three approximation strategies:
   - 3rd-order Taylor
   - Minimax polynomial obtained via Remez
   - Piece-wise linear with 8 segments
3. Benchmark using the same MNIST canary batches:
   - Record constraint count (via zkVM trace length)
   - Measure proving/verification time
   - Compare loss deltas to float32 ground truth
4. Summarise in a cost table and decide go/no-go for productionisation.

## Deliverables
- Benchmark notebook + CSV of timings and errors
- Updated implementation guide appendix with cost summary
- Recommendation memo (keep MSE, adopt approximation, or hybrid)

## Dependencies
- Requires Phase 1 artifacts: deterministic weights, loss delta pipeline, and proof harness.
- Needs Criterion benchmarking harness and guest integration (target: Day 7 of sprint).
