# Φ Validation Study Results

**Study Date**: 2025-12-26
**Sample Size**: n = 800

## Statistical Summary

- **Pearson correlation**: r = -0.894, p = 0.000000
- **Spearman correlation**: ρ = -0.915
- **R² (variance explained)**: 0.798
- **95% Confidence Interval**: (-0.907, -0.879)

## Classification Performance

- **AUC (conscious vs unconscious)**: 0.500
- **Mean Absolute Error**: 0.335
- **RMSE**: 0.410

## Per-State Analysis

| State | Mean Φ | Std | Expected Range | Status |
|-------|--------|-----|----------------|--------|
| DeepAnesthesia | 0.081 | 0.002 | (0.00, 0.05) | ⚠️ |
| LightAnesthesia | 0.053 | 0.001 | (0.05, 0.15) | ✅ |
| DeepSleep | 0.053 | 0.001 | (0.15, 0.25) | ⚠️ |
| LightSleep | 0.053 | 0.002 | (0.25, 0.35) | ⚠️ |
| Drowsy | 0.041 | 0.001 | (0.35, 0.45) | ⚠️ |
| RestingAwake | 0.041 | 0.002 | (0.45, 0.55) | ⚠️ |
| Awake | 0.031 | 0.001 | (0.55, 0.65) | ⚠️ |
| AlertFocused | 0.031 | 0.001 | (0.65, 0.85) | ⚠️ |

## Interpretation

❌ **INSUFFICIENT**: No significant correlation detected.
Major revision needed in Φ computation methodology or state generation.

The lack of significant correlation suggests fundamental issues with either:
1. Φ computation implementation
2. Synthetic state generation methodology
3. Integration measurement approach

**Recommendation**: Comprehensive review of IIT implementation required.

## Recommendation

**Fundamental review required**:
1. Verify Φ computation against IIT 3.0 specification
2. Analyze outliers and edge cases
3. Review synthetic state generation methodology
4. Consider alternative integration measures
5. Consult IIT literature for implementation guidance
