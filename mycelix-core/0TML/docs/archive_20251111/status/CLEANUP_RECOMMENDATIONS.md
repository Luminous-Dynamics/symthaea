# Code Cleanup Recommendations

**Date**: October 21, 2025
**Context**: After investigating PoGQ implementations and 40% BFT testing
**Status**: ✅ Step 1 Complete - Old implementations archived

---

## What We Discovered

1. **REAL PoGQ**: `src/zerotrustml/experimental/trust_layer.py` - THIS WORKS
2. **Broken PoGQ**: `baselines/pogq.py` - imports non-existent `pogq_system`
3. **Simplified PoGQ**: `baselines/pogq_real.py` - mean-based, fails at 40% BFT
4. **Baseline that worked**: `baselines/multikrum.py` - 30% BFT validated

## Files to Archive (Move to `.archive-2025-10-21/`)

### 1. Broken Implementations
```
baselines/pogq.py  # Imports non-existent pogq_system module - cannot run
```

### 2. Superseded by Better Implementations
```
baselines/pogq_real.py  # Simplified mean-based PoGQ
  - Reason: Superseded by trust_layer.py ProofOfGradientQuality
  - Issue: Fails at 40% BFT (contamination problem)
  - Better: Use src/zerotrustml/experimental/trust_layer.py
```

### 3. Old Test Files That Failed
```
tests/test_40_50_bft_breakthrough.py (old version with simplified PoGQ)
  - Keep updated version with REAL PoGQ
  - Archive old version that used baselines/pogq_real.py
```

### 4. Documentation to Update
```
POGQ_INVESTIGATION_FINDINGS.md  # Created during investigation
  - Update with final 40% test results
  - Document that 40% BFT failed (37.5% detection)
  - Recommend 30% BFT validation approach
```

## Files to KEEP and USE

### 1. Working Implementations ✅
```
src/zerotrustml/experimental/trust_layer.py
  - ProofOfGradientQuality class (REAL PoGQ)
  - ZeroTrustML class (complete trust layer)
  - AnomalyDetector class
  - PeerReputation tracking
```

### 2. Validated Baselines ✅
```
baselines/multikrum.py
  - Proven to work at 30% BFT
  - Reference implementation for comparison
```

### 3. Current Tests ✅
```
tests/test_40_50_bft_breakthrough.py (updated version)
  - Now uses REAL PoGQ from trust_layer.py
  - Validates at 40% (failed - 37.5% detection)
  - Can be modified for 30% BFT validation
```

### 4. Findings Documents ✅
```
REALISTIC_PATH_FORWARD.md
FINDINGS_40_50_BFT_TESTING.md
POGQ_INVESTIGATION_FINDINGS.md
```

---

## Recommended Cleanup Actions

### Step 1: Archive Broken/Superseded Code ✅ COMPLETE
```bash
mkdir -p .archive-2025-10-21
mv baselines/pogq.py .archive-2025-10-21/
mv baselines/pogq_real.py .archive-2025-10-21/

# Create archive log
cat > .archive-2025-10-21/ARCHIVE_LOG.md << 'EOF'
# Archive Log - October 21, 2025

## Why These Files Were Archived

### baselines/pogq.py
- **Issue**: Imports non-existent `pogq_system` module
- **Status**: Cannot run, broken implementation
- **Replacement**: `src/zerotrustml/experimental/trust_layer.py`

### baselines/pogq_real.py
- **Issue**: Simplified mean-based PoGQ fails at 40% BFT (contamination)
- **Status**: Superseded by better implementation
- **Replacement**: `src/zerotrustml/experimental/trust_layer.py` ProofOfGradientQuality

## Investigation Summary

After thorough testing, discovered:
1. Real PoGQ exists in `trust_layer.py` (validates against test set)
2. Baseline tests used Multi-KRUM (not PoGQ) at 30% BFT
3. 40% BFT test failed with 37.5% detection rate
4. 30% BFT is the validated, honest approach for grant submission

## Next Steps

1. Use `trust_layer.py` for all PoGQ implementations
2. Test at 30% BFT to validate matching baseline results
3. Submit grant with validated 30% BFT claims
4. Document 40-50% as future work
EOF
```

### Step 2: Update Documentation
```bash
# Update investigation findings with final results
echo "
## 40% BFT Test Results (Final)

**Test Date**: October 21, 2025
**Configuration**: 20 nodes (12 honest + 8 Byzantine)
**PoGQ**: REAL implementation from trust_layer.py

**Results**:
- PoGQ Detection: 3/8 Byzantine nodes (37.5%)
- False Positives: 0/12 honest nodes (0%)
- Final Byzantine Reputation: 0.625 (FAILED - threshold 0.4)

**Conclusion**: 40% BFT NOT VALIDATED
- Only 37.5% detection rate
- 62.5% of Byzantine nodes undetected
- Recommend testing at 30% BFT instead

**Recommendation**: Submit grant with 30% BFT validation
" >> POGQ_INVESTIGATION_FINDINGS.md
```

### Step 3: Create Clean 30% BFT Test
```bash
# Option: Modify existing test for 30% BFT
# Change: 20 nodes → 14 honest + 6 Byzantine (30%)
# Expected: Match baseline 68-95% detection rates
```

---

## Impact Assessment

**Code Reduction**:
- Remove 2 broken/superseded files
- ~800 lines of old code archived
- Clearer codebase focused on working implementations

**Clarity Gain**:
- One clear PoGQ implementation (trust_layer.py)
- Working baseline (multikrum.py)
- Honest test results documented

**Next Actions**:
1. ✅ Archive old implementations - COMPLETE (2025-10-21)
2. Test at 30% BFT with REAL PoGQ
3. Update grant materials with validated results
4. Submit with honest, proven claims

---

*"Clean code is honest code - remove what doesn't work, keep what does."*
