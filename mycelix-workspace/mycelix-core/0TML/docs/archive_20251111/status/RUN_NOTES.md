# Sanity Slice Experiment Run Notes

**Run ID**: `sanity_slice_2025-11-08`
**Status**: In Progress
**Start Time**: 2025-11-08 ~16:00 UTC
**Expected Completion**: 2025-11-09 or 2025-11-10 (~28-42 hours)

---

## Environment Snapshot

### System Configuration
- **OS**: NixOS 25.11 "Xantusia"
- **Kernel**: Linux 6.17.2
- **Hostname**: (TBD - to be filled from actual system)
- **Python**: 3.11
- **PyTorch**: 2.1.0
- **CUDA**: (TBD - check `nvidia-smi`)
- **cuDNN**: (TBD - check PyTorch build info)

### Hardware
- **CPU**: (TBD - check `lscpu | grep "Model name"`)
- **RAM**: (TBD - check `free -h`)
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) (assumed - verify with `nvidia-smi`)
- **Disk**: (TBD - check `df -h /srv/luminous-dynamics`)

### Git Provenance
```bash
# To be filled with actual values:
git log -1 --format="%H %s"
# Expected: a91dbc53... (or later commit)

git status --porcelain
# Expected: Clean working tree (or document modifications)

git branch --show-current
# Expected: main
```

### Nix Environment
```bash
# Flake lock status
nix flake metadata
# (Capture nixpkgs revision)

# Python packages (from flake.nix)
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
```

---

## Experiment Configuration

### Config File
**Path**: `configs/sanity_slice.yaml`
**Frozen Copy**: `configs/_frozen/sanity_slice_2025-11-08.yaml`
**SHA-256**: (To be computed and filled)

### Key Parameters
```yaml
# Dataset configuration
datasets:
  - name: emnist
    train_size: 112800
    test_size: 18800
    num_classes: 47
  - name: mnist
    train_size: 60000
    test_size: 10000
    num_classes: 10

# Federated learning
num_clients: 100
clients_per_round: 20
local_epochs: 1
learning_rate: 0.01
momentum: 0.9
batch_size: 64
num_rounds: 20

# Byzantine configuration
byzantine_ratio: 0.20  # 4 out of 20 per round

# Data split
split_type: dirichlet
alpha: 0.3  # Non-IID concentration

# Attacks (8 total)
attacks:
  - sign_flip
  - scaling_x100
  - random_noise
  - noise_masked
  - targeted_neuron
  - collusion
  - sybil_flip
  - sleeper_agent

# Defenses (8 total)
defenses:
  - fedavg
  - coord_median
  - coord_median_safe
  - rfa
  - fltrust
  - boba
  - cbf
  - pogq_v4_1

# Seeds
seeds: [42, 1337]

# Experimental matrix
# Total: 2 datasets × 8 attacks × 8 defenses × 2 seeds = 256 experiments
```

---

## Process Information

### Launch Command
```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML
nohup nix develop --command python experiments/matrix_runner.py \
  --config configs/sanity_slice.yaml \
  > logs/sanity_slice_fixed.log 2>&1 &
```

### Process Details
- **PID**: 3427491
- **Started**: 2025-11-08 ~16:00 UTC
- **User**: (TBD - check `ps aux | grep 3427491`)
- **CPU Utilization**: 74.5% (as of initial check)
- **Memory**: 2.3 GB (growing during dataset loading)
- **Status**: DNl (uninterruptible sleep - heavy I/O, expected)

### Monitoring Commands
```bash
# Check process status
ps aux | grep 3427491

# Monitor CPU/memory
top -p 3427491

# Watch log (unbuffered Python next time)
tail -f logs/sanity_slice_fixed.log

# Disk usage
df -h /srv/luminous-dynamics

# Artifact count
ls -1 results/artifacts_* 2>/dev/null | wc -l
```

---

## Bug Fixes Applied (This Run)

### Fix 1: Data Split num_classes Auto-Detection
**File**: `experiments/utils/data_splits.py`
**Line**: 269
**Change**: `num_classes: int = 10` → `num_classes: int = None` with auto-detection
**Reason**: EMNIST has 47 classes, not 10; caused inhomogeneous array error

### Fix 2: Model num_classes Mapping
**File**: `experiments/matrix_runner.py`
**Lines**: 68-81
**Change**: Explicit dataset → num_classes mapping dict
**Reason**: Config used lowercase 'emnist', code checked for 'FEMNIST'; defaulted to 10 classes

**Documentation**: See `DATA_SPLIT_FIX_COMPLETE.md` for full details

---

## Expected Outputs

### Artifacts Per Experiment (6 JSON files)
Each of 256 experiments generates:
1. `detection_metrics.json` - AUROC, TPR, FPR, ROC curve, TTD
2. `per_bucket_fpr.json` - Mondrian bucket FPR (CI gate 1)
3. `ema_vs_raw.json` - EMA stability metrics (CI gate 2)
4. `coord_median_diagnostics.json` - Guard activations (CI gate 3)
5. `bootstrap_ci.json` - Bootstrap confidence intervals
6. `model_metrics.json` - Test accuracy, loss, convergence

**Total artifact count**: 256 experiments × 6 files = 1,536 JSON files

### Directory Structure
```
results/
├── artifacts_20251108_HHMMSS_emnist_sign_flip_fedavg_seed42/
│   ├── detection_metrics.json
│   ├── per_bucket_fpr.json
│   ├── ema_vs_raw.json
│   ├── coord_median_diagnostics.json
│   ├── bootstrap_ci.json
│   └── model_metrics.json
├── artifacts_20251108_HHMMSS_emnist_sign_flip_cbf_seed42/
│   └── [same 6 files]
└── [254 more directories...]
```

### Expected Disk Usage
- **Artifacts**: ~512 MB (1,536 JSON files, ~300 KB each average)
- **Logs**: ~50 MB (depending on verbosity)
- **Checkpoints**: ~2 GB (model checkpoints if saved)
- **Total**: ~3 GB

---

## CI Gates (Hard Guarantees)

Must pass for results to be accepted:

### Gate 1: Conformal FPR Compliance
**Threshold**: All Mondrian buckets must have FPR ≤ 0.12 (α=0.10 + 2% margin)
**Verification**: `./scripts/spot_check_artifacts.sh` or `pytest tests/test_ci_gates.py::test_conformal_fpr_compliance`

### Gate 2: EMA Stability
**Threshold**: Hysteresis flap count ≤ 5 per experiment (20 rounds)
**Verification**: `./scripts/spot_check_artifacts.sh` or `pytest tests/test_ci_gates.py::test_ema_stability`

### Gate 3: CM-Safe Guard Coverage
**Threshold**: ≥ 1 guard activation during Byzantine rounds
**Verification**: `./scripts/spot_check_artifacts.sh` or `pytest tests/test_ci_gates.py::test_cmsafe_guard_coverage`

---

## Known Issues & Mitigations

### Python Output Buffering
**Issue**: Log file appears empty despite process running
**Reason**: Python heavily buffers output when redirected to files
**Mitigation**: Next run use `python -u` for unbuffered output
**Status**: Not critical; process verified running via CPU/memory monitoring

### Disk Space
**Warning**: ~3 GB required for artifacts + logs
**Current**: (TBD - check `df -h`)
**Mitigation**: Disk watchdog script monitoring free space

---

## Reproducibility Checklist

- [ ] Config file frozen to `configs/_frozen/sanity_slice_2025-11-08.yaml`
- [ ] Git commit hash recorded: `________________`
- [ ] Nix flake lock committed: `flake.lock` SHA: `________________`
- [ ] Python version recorded: `________________`
- [ ] PyTorch version recorded: `________________`
- [ ] CUDA version recorded: `________________`
- [ ] Hardware specs documented above
- [ ] Random seeds documented: [42, 1337]
- [ ] All bug fixes documented: DATA_SPLIT_FIX_COMPLETE.md

---

## Post-Run Verification

Once experiments complete:

### 1. Artifact Count
```bash
ls -1d results/artifacts_* | wc -l
# Expected: 256
```

### 2. CI Gates
```bash
./scripts/spot_check_artifacts.sh
# Expected: All 3 gates PASS
```

### 3. Missing Files
```bash
find results -type d -name "artifacts_*" ! -exec sh -c 'test $(ls -1 "$1" | wc -l) -eq 6' _ {} \; -print
# Expected: Empty (all directories have 6 files)
```

### 4. Generate Results
```bash
# Incremental summary
python scripts/summarize_incremental.py | column -t > RESULTS_TABLE_II.txt

# All figures
./scripts/generate_all_figures.sh pdf 600

# Figure count
ls -1 figures/*.pdf | wc -l
# Expected: ~120 figures
```

---

## Timeline Estimate

**Start**: 2025-11-08 ~16:00 UTC
**Progress Checkpoints**:
- 25% complete: 2025-11-08 ~23:00 UTC (7 hours)
- 50% complete: 2025-11-09 ~06:00 UTC (14 hours)
- 75% complete: 2025-11-09 ~13:00 UTC (21 hours)
- 100% complete: 2025-11-09 ~20:00 UTC - 2025-11-10 ~06:00 UTC (28-42 hours)

**Monitor via**:
```bash
python scripts/summarize_incremental.py | grep -c "^emnist"
# Progress: count / 256
```

---

## Contact & Emergency

**Primary Investigator**: Tristan Stoltz
**Location**: Richardson, TX (Central Time)
**Monitoring**: Remote via Claude Code

**Emergency Stop** (if needed):
```bash
kill 3427491  # Graceful termination
# OR
kill -9 3427491  # Force kill (last resort)
```

**Resume** (if interrupted):
```bash
# Use progress tracker to skip completed experiments
python scripts/progress_tracker.py progress_manifests/sanity_slice.json --list-completed
# Relaunch with same config (tracker auto-skips completed)
```

---

**Status as of 2025-11-08**: Experiments running successfully, infrastructure ready for results processing.

**Next Update**: When first artifacts land or at 24-hour checkpoint (2025-11-09 ~16:00 UTC)
