# Extended Benchmark Results

**Generated**: 2026-02-04 08:44:32

## Results Summary

| Task | Inference (ms) | Action (ms) | Success Rate | Episodes |
|------|----------------|-------------|--------------|----------|
| tiger | 0.118 | 1.226 | 98.0% | 100 |
| grid_10x10 | 0.136 | 1.808 | 0.0% | 50 |
| grid_20x20 | 0.131 | 1.753 | 0.0% | 30 |
| multiagent_coord | 0.120 | 1.361 | 0.0% | 50 |

## Task Descriptions

### 1. Tiger Problem
Classic POMDP with partial observability. Agent must infer tiger location from noisy observations.

### 2. Grid World 10×10
Navigation in 100-state grid. Tests scaling to larger state spaces.

### 3. Grid World 20×20
Navigation in 400-state grid. Tests scaling limits.

### 4. Multi-Agent Coordination
Two HAI agents must coordinate to reach a common target. Tests decentralized coordination.

## Key Findings

1. **Best Task Performance**: tiger (98.0% success)
2. **Scaling**: grid_20x20 → grid_10x10: 1.0× inference time increase
3. **Multi-Agent**: 0.0% coordination success rate

---
*All benchmarks use HAI (Python reference implementation)*
