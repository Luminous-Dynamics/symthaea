# Active Inference Benchmark: Symthaea vs pymdp

**Generated**: 2026-02-03T19:51:55.503311
**pymdp Available**: True

## Summary

- **Inference Speedup**: 1.9× faster than pymdp
- **Action Selection Speedup**: 15.8× faster
- **Total Speedup**: 7.9× faster
- **Quality Ratio**: 3.10 (>1 means Symthaea achieves lower FE)

## Detailed Results

| Task | Method | Inference (ms) | Action (ms) | Free Energy | Success Rate |
|------|--------|----------------|-------------|-------------|--------------|
| t_maze | symthaea | 0.084 | 0.149 | -2.305 | 100.00% |
| grid_3 | symthaea | 0.093 | 0.135 | -2.350 | 92.68% |
| grid_3 | pymdp | 0.318 | 1.812 | -9.421 | 16.00% |
| grid_5 | symthaea | 0.191 | 0.148 | -2.975 | 88.03% |
| grid_5 | pymdp | 0.356 | 2.338 | -9.238 | 10.00% |

## Methodology

### Tasks

1. **T-Maze**: Classic active inference benchmark with context inference
2. **Grid World 3×3**: Simple navigation task
3. **Grid World 5×5**: Larger navigation task testing scalability

### Metrics

- **Inference Time**: Time to update beliefs given observations
- **Action Time**: Time to compute expected free energy and select action
- **Free Energy**: Mean expected free energy (lower = better)
- **Success Rate**: Fraction of trials reaching goal state

### Implementation Notes

- **Symthaea**: Uses HDC-based representations for beliefs
  - Precision-weighted binding for confidence modulation
  - 16,384-dimensional hypervectors
  - Cosine similarity for belief comparison

- **pymdp**: Standard matrix-based active inference
  - Categorical distributions for beliefs
  - Matrix operations for inference
