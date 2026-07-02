# ✈️ AEGIS Gen-5 Pre-Flight Checklist

**Purpose**: Verify environment before launching 300-run validation
**Time**: 2 minutes
**Status**: Execute before dry-run

---

## ✅ Step 1: Environment Verification

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Verify nix develop works and numpy loads
nix develop -c python -c "import numpy; print('NumPy:', numpy.__version__)"
# Expected: NumPy: 2.3.4
```

---

## ✅ Step 2: Clear Previous Runs

```bash
# Archive or remove previous validation results
if [ -d "validation_results" ]; then
  mv validation_results validation_results.backup-$(date +%Y%m%d-%H%M%S)
fi

# Clear figures directory
if [ -d "figures" ]; then
  mv figures figures.backup-$(date +%Y%m%d-%H%M%S)
fi

# Verify clean state
ls -la | grep -E "(validation_results|figures)"
# Expected: No output (directories don't exist)
```

---

## ✅ Step 3: Lock Reproducibility

### Commit Current State
```bash
# Get current git commit
git log -1 --oneline
# Record this commit hash - it will be in the manifest

# Get current state hash (simulated - actual hash created by runner)
find src/gen5 -name "*.py" | sort | xargs cat | sha256sum | cut -d' ' -f1
# Record this for verification later
```

### Set Environment Variables
```bash
# Reduce threading variance
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Deterministic dict ordering
export PYTHONHASHSEED=0

# Verify
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "PYTHONHASHSEED=$PYTHONHASHSEED"
```

---

## ✅ Step 4: Update README with Validation Manifest

Add this section to the main README:

```markdown
### Validation Manifest

**Seeds**: [101, 202, 303, 404, 505]
**Git Commit**: [Will be filled by first run]
**State Hash**: [Will be filled by first run]
**Validation ID**: [Will be filled by first run]

Reproducibility: All validation runs use the above fixed seeds and immutable
code state. State hash is verified every 5 runs to detect drift.
```

---

## ✅ Step 5: Verify Test Success

```bash
# Run full test suite to confirm 147/147
pytest tests/ -v --tb=short | tail -20

# Expected output:
# ==================== 147 passed in X.XXs ====================
```

---

## ✅ Step 6: Pre-Flight Complete

### Final Checklist

- [ ] `nix develop` opens successfully
- [ ] `import numpy` works in nix develop
- [ ] `validation_results/` and `figures/` cleared or archived
- [ ] Environment variables set (OMP_NUM_THREADS=2, PYTHONHASHSEED=0)
- [ ] Git commit hash recorded
- [ ] All 147 tests passing
- [ ] README ready for manifest update

### Ready to Launch

```bash
# Dry-run validation (~30 minutes)
nix develop -c python experiments/run_validation.py --mode dry-run

# Expected first output:
# ✅ Validation manifest created: AEGIS-dry-run-2025-11-XX-XXXXXX
#    State hash: 8a3f9c2d...
#    Git commit: 09762061...
#    Output dir: validation_results
```

---

## 🔍 During Dry-Run: What to Watch

### Positive Indicators ✅

1. **Manifest Created**
   ```
   ✅ Validation manifest created: AEGIS-dry-run-2025-11-12-235900
      State hash: 8a3f9c2d4e5b6f7a...
      Git commit: 09762061abcdef12...
   ```

2. **State Hash Stable**
   ```
   [5/10] ... ✅ Checkpoint saved (ETA: 15.3 min)
   # State hash should match manifest every checkpoint
   ```

3. **Progress Tracking**
   ```
   [1/10] E1_byzantine_tolerance config 0...
      Runtime: 12.34s
   # Consistent runtimes (~10-20s per run)
   ```

### Red Flags ❌

1. **Drift Detected**
   ```
   ❌ DRIFT DETECTED! Code changed during validation.
      Aborting to preserve reproducibility.
   ```
   - **Action**: Stop, commit changes, restart

2. **ETA Creeping**
   ```
   ETA: 15 min → 20 min → 30 min (increasing)
   ```
   - **Action**: Reduce thread count (OMP_NUM_THREADS=1)

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'numpy'
   ```
   - **Action**: Run via `nix develop -c` not bare `python`

---

## 📋 Post Dry-Run Verification

### Check Outputs Created

```bash
# Manifest exists with correct structure
cat validation_results/manifest.json | python -m json.tool | head -20

# Per-run artifacts exist
find validation_results -name "config.json" | wc -l
# Expected: 10 (one per dry-run experiment)

find validation_results -name "metrics.json" | wc -l
# Expected: 10

# Figures generated
ls -lh figures/
# Expected: F1-F9 (18 files: 9 PDF + 9 PNG)
```

### Verify Manifest Consistency

```bash
# Extract and compare hashes
MANIFEST_HASH=$(jq -r '.state_hash' validation_results/manifest.json)
RESULTS_HASH=$(jq -r '.manifest.state_hash' validation_results/results_complete.json)

if [ "$MANIFEST_HASH" = "$RESULTS_HASH" ]; then
  echo "✅ State hash consistent"
else
  echo "❌ State hash mismatch!"
fi
```

### Inspect Figures

```bash
# View figure list
ls -1 figures/*.pdf

# Expected:
# F1_byzantine_tolerance.pdf
# F2_sleeper_detection.pdf
# F3_coordination_detection.pdf
# F4_active_learning.pdf
# F5_federated_convergence.pdf
# F6_privacy_utility.pdf
# F7_validator_overhead.pdf
# F8_self_healing.pdf
# F9_secret_sharing.pdf

# Quick sanity check (file sizes should be reasonable)
du -sh figures/*.pdf
# Expected: Each file 50-500 KB
```

---

## 🚀 Launch Full Validation

### If Dry-Run Succeeded

```bash
# Update README with manifest values from dry-run
MANIFEST_ID=$(jq -r '.validation_id' validation_results/manifest.json)
GIT_COMMIT=$(jq -r '.git_commit' validation_results/manifest.json)
STATE_HASH=$(jq -r '.state_hash' validation_results/manifest.json)

echo "Validation ID: $MANIFEST_ID"
echo "Git Commit: $GIT_COMMIT"
echo "State Hash: $STATE_HASH"

# Archive dry-run results
mv validation_results validation_results_dry_run
mv figures figures_dry_run

# Launch full validation (300 runs, ~8 hours)
nix develop -c python experiments/run_validation.py --mode full

# Expected duration: ~8 hours (300 runs × ~1.5 min/run)
```

### Monitor Progress

```bash
# Check progress periodically
tail -f validation_results/checkpoint.json

# Or monitor ETA
grep -E "ETA:" validation_results/*.log | tail -1
```

---

## 📊 Final QA (After Full Run)

### Manifest Verification
- [ ] Git commit, state hash, seeds match README
- [ ] `results_complete.json` contains exactly 300 runs
- [ ] Each experiment (E1-E9) has expected run count

### Figure Verification
- [ ] All 9 figures (F1-F9) exist in PDF and PNG
- [ ] F1: TPR degrades gracefully with Byzantine rate
- [ ] F4: Speedup ≥ 6× at budget 0.15-0.25
- [ ] F7: Total overhead < 20ms
- [ ] F8: MTTR distributions < 20 rounds

### Statistical Outputs
- [ ] `summary.txt` exists with bootstrap CIs
- [ ] No drift detected during entire run
- [ ] Checkpoint files saved every 5 runs

---

## ✅ Sign-Off

- [ ] **Pre-flight complete**: All 6 steps verified
- [ ] **Dry-run successful**: 10 experiments, all outputs correct
- [ ] **Full run launched**: 300 experiments in progress
- [ ] **Monitoring active**: ETA tracked, checkpoints verified
- [ ] **README updated**: Manifest values committed

**Status**: CLEARED FOR VALIDATION ✈️

---

**Last Updated**: November 12, 2025
**Purpose**: Ensure reproducible, high-quality validation results
**Target**: MLSys / ICML 2026 submission

🎯 **Pre-flight checklist complete - ready for launch!** 🎯
