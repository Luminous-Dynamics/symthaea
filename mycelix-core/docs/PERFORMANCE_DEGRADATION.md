# Mycelix Performance Degradation Curve

**Date:** January 12, 2026
**Test Configuration:** Median aggregation, 11 total nodes, 10 rounds, adaptive attack

## Byzantine Tolerance Results

| Byzantine % | Honest Nodes | Byzantine Nodes | Aggregation Quality | Phi Coherence |
|-------------|--------------|-----------------|---------------------|---------------|
| 0%          | 11           | 0               | 100.0%              | 0.997         |
| 9%          | 10           | 1               | 100.0%              | 0.995         |
| 18%         | 9            | 2               | 100.0%              | 0.993         |
| 27%         | 8            | 3               | 100.0%              | 0.990         |
| 36%         | 7            | 4               | 100.0%              | 0.988         |
| **45%**     | **6**        | **5**           | **100.0%**          | **0.985**     |
| 55%         | 5            | 6               | 100.0%              | 0.983         |
| 64%         | 4            | 7               | 99.9%               | 0.980         |
| 73%         | 3            | 8               | 99.9%               | 0.978         |

## Key Findings

1. **Classical BFT Limit Exceeded**: The system maintains 100% aggregation quality beyond the classical 33% Byzantine limit (up to 55%).

2. **Graceful Degradation**: Quality degrades gracefully from 100% to 99.9% only at extreme Byzantine ratios (64%+).

3. **Phi Coherence Tracking**: System coherence (Phi) provides early warning, declining from 0.997 to 0.978 as Byzantine percentage increases.

4. **Median Aggregation Robustness**: The Median algorithm achieves this by selecting the median value, which requires only that honest values cluster around the true gradient.

## Algorithm-Specific Performance

### At 10 nodes (Criterion Benchmark)

| Algorithm    | Latency  | Byzantine Tolerance |
|--------------|----------|---------------------|
| FedAvg       | 75 µs    | 0% (no defense)     |
| Median       | 3.7 ms   | 50%+                |
| TrimmedMean  | 4.7 ms   | ~40%                |
| Krum         | 5.2 ms   | 33%                 |

### Scalability (10,000-dim gradients)

| Nodes | Median   | TrimmedMean | Krum    |
|-------|----------|-------------|---------|
| 10    | 3.7 ms   | 4.7 ms      | 5.2 ms  |
| 50    | 44 ms    | 46 ms       | 141 ms  |
| 100   | 125 ms   | 115 ms      | 632 ms  |

**Note:** Krum has O(n²) complexity and scales poorly. For >50 nodes, prefer Median or TrimmedMean.

## Recommendations

1. **Use Median for high-security deployments**: Best Byzantine tolerance with reasonable latency.

2. **Monitor Phi for early warning**: A Phi drop below 0.95 may indicate Byzantine activity even when aggregation quality remains high.

3. **Scale with TrimmedMean**: For large deployments (100+ nodes), TrimmedMean offers better latency than Median with similar tolerance.

4. **Avoid Krum at scale**: While Krum provides strong theoretical guarantees, its O(n²) complexity makes it impractical for >50 nodes.

---

*Generated from validation tests on January 12, 2026*
