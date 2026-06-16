# PyPhi Cross-Validation Report

**Generated**: 2026-02-03T19:18:36.563941
**Status**: NEEDS REVIEW

## Summary Statistics

- **Correlation (r)**: -0.0075
- **Mean Absolute Error**: 0.3729
- **Max Absolute Error**: 0.4911
- **Agreement within 10%**: 0.0%
- **Validated topologies**: 9
- **Failed topologies**: 0

## Validation Target

❌ **FAIL**: Correlation r < 0.80

## Detailed Results

| Topology | Nodes | HDC-Φ | PyPhi-Φ | Error | Status |
|----------|-------|-------|---------|-------|--------|
| line_2 | 2 | 0.4693 | 0.9604 | 0.4911 | ⚠️ |
| line_3 | 3 | 0.4706 | 0.2071 | 0.2634 | ⚠️ |
| ring_3 | 3 | 0.4892 | 0.1887 | 0.3004 | ⚠️ |
| line_4 | 4 | 0.4718 | 0.0270 | 0.4448 | ⚠️ |
| ring_4 | 4 | 0.4904 | 0.1346 | 0.3558 | ⚠️ |
| star_4 | 4 | 0.4503 | 0.0800 | 0.3703 | ⚠️ |
| square_hypercube | 4 | 0.4910 | 0.1346 | 0.3564 | ⚠️ |
| ring_5 | 5 | 0.4917 | 0.1346 | 0.3570 | ⚠️ |
| star_5 | 5 | 0.4516 | 0.0349 | 0.4167 | ⚠️ |

## Notes

- HDC-Φ: Approximation using algebraic connectivity of HDC similarity graph
- PyPhi-Φ: Exact IIT 3.0 calculation (intractable for n > 10)
- Error: Absolute difference between methods
- N/A: PyPhi computation failed or exceeded timeout

## Methodology

1. Generate identical topologies for both methods
2. Compute HDC-Φ using Symthaea's RealPhiCalculator
3. Compute exact Φ using PyPhi (IIT 3.0)
4. Calculate Pearson correlation coefficient
5. Report absolute and relative errors
