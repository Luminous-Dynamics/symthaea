# HAI Ablation Studies Results

**Generated**: 2026-02-04 08:40:00

## 1. HDC Dimension Ablation

| Dimension | Inference (ms) | Action (ms) | Free Energy | Success Rate | Convergence |
|-----------|----------------|-------------|-------------|--------------|-------------|
| 1,024 | 0.111 | 0.556 | 0.2000 | 100.0% | 1 |
| 4,096 | 0.169 | 1.626 | 0.1922 | 100.0% | 1 |
| 16,384 | 0.573 | 260.990 | 0.1895 | 100.0% | 1 |
| 65,536 | 1.100 | 265.721 | 0.1894 | 100.0% | 1 |

## 2. Precision Initialization Ablation

| Precision (π) | Inference (ms) | Action (ms) | Free Energy | Success Rate | Convergence |
|---------------|----------------|-------------|-------------|--------------|-------------|
| 0.1 | 0.309 | 252.912 | 0.0984 | 100.0% | 0 |
| 0.5 | 0.519 | 257.430 | 0.4628 | 100.0% | 0 |
| 1.0 | 0.658 | 256.234 | 0.1900 | 100.0% | 1 |
| 2.0 | 0.773 | 241.445 | 0.3796 | 100.0% | 1 |
| 5.0 | 5.375 | 261.660 | 0.9462 | 0.0% | 20 |

## 3. EFE Weight Ratios Ablation

| Configuration | Explore/Exploit | Inference (ms) | Action (ms) | Free Energy | Success Rate |
|---------------|-----------------|----------------|-------------|-------------|---------------|
| pure_exploit | 0.00 | 1.060 | 260.986 | 0.1904 | 100.0% |
| pure_explore | 100.00 | 0.752 | 256.223 | 0.1902 | 100.0% |
| balanced_no_novelty | 0.50 | 0.536 | 248.012 | 0.1897 | 100.0% |
| balanced_default | 0.50 | 0.772 | 262.450 | 0.1902 | 100.0% |
| high_explore_novelty | 0.99 | 0.589 | 267.626 | 0.1906 | 100.0% |

## Key Findings

1. **Optimal Dimension**: 1,024 (success rate: 100.0%)
2. **Optimal Precision**: π = 0.1 (success rate: 100.0%)
3. **Optimal EFE Weights**: pure_exploit (success rate: 100.0%)

---
*50 trials per configuration, 20 inference iterations per trial*
