# 📊 Dataset Strategy & Technical Fixes

**Date**: October 14, 2025
**Purpose**: Dataset selection for paper + fixing Grand Slam technical issues

---

## 🎯 Recommended Dataset Strategy for Paper

### **RECOMMENDED: 3-Dataset Approach** ⭐

**Datasets**:
1. **MNIST** - Standard baseline (64MB, already downloaded)
2. **CIFAR-10** - More challenging, color images (341MB, already downloaded)
3. **Fashion-MNIST** or **SVHN** - Cross-domain validation (need to add)

**Why 3 datasets?**:
- ✅ **Academic standard** - Top conferences expect 2-3 datasets minimum
- ✅ **Generalization proof** - Shows PoGQ+Rep works across domains
- ✅ **Reviewer expectations** - MLSys/ICML papers typically use 3+ datasets
- ✅ **Comparable to baselines** - FedAvg/Multi-Krum papers use 2-4 datasets

**Paper Impact**:
- **With 2 datasets (MNIST + CIFAR-10)**: Adequate but minimal
- **With 3 datasets**: Strong generalization claim, competitive with SOTA
- **With 4+ datasets**: Overkill for federated learning (image + text is enough)

---

## 📁 Current Dataset Situation

### Main Datasets Folder (KEEP THIS)
**Location**: `/srv/luminous-dynamics/Mycelix-Core/0TML/datasets/`

**Contents**:
- ✅ `mnist/` - 64MB - **Complete and working**
- ✅ `cifar10/` - 341MB - **Downloaded but broken in runner**
- ❓ `shakespeare/` - 8.3MB - **Unknown status, need to verify**

### Duplicate Folder (REMOVE THIS)
**Location**: `/srv/luminous-dynamics/Mycelix-Core/0TML/experiments/datasets/`

**Contents**:
- 🗑️ `mnist/` - 64MB - **Duplicate, safe to delete**

**Why remove**: Wastes disk space, creates confusion about which path to use

---

## 🐛 Technical Issues Found

### Issue 1: CIFAR-10 Download Disabled ❌ **CRITICAL BUG**

**File**: `/srv/luminous-dynamics/Mycelix-Core/0TML/experiments/runner.py:146`

**Current Code**:
```python
train_dataset = datasets.CIFAR10(
    root=data_dir / 'cifar10',
    train=True,
    transform=transform_train,
    download=False  # ❌ BUG: Should be True!
)
```

**Fix**:
```python
train_dataset = datasets.CIFAR10(
    root=data_dir / 'cifar10',
    train=True,
    transform=transform_train,
    download=True  # ✅ FIX: Auto-download if missing
)
```

**Impact**: This is why Grand Slam reported "Dataset not found or corrupted"

---

### Issue 2: Relative Path vs. Absolute Path

**File**: `/srv/luminous-dynamics/Mycelix-Core/0TML/experiments/runner.py:105`

**Current Code**:
```python
data_dir = Path(self.config['dataset'].get('data_dir', 'datasets'))
```

**Problem**: Uses relative path `'datasets'` by default, which resolves differently depending on where script is run from

**Fix**:
```python
# Use absolute path relative to project root
project_root = Path(__file__).parent.parent
data_dir = Path(self.config['dataset'].get(
    'data_dir',
    project_root / 'datasets'  # ✅ Absolute path
))
```

**Impact**: Ensures consistent dataset location regardless of working directory

---

### Issue 3: Old Project Path References

**Found**: 21 Python files contain `Mycelix-Core` hardcoded paths

**Files to Fix**:
- `demo_all_backends.py`
- `benchmark_backends.py`
- `test_modular_backends.py`
- `src/zerotrustml/core/phase10_coordinator.py`
- ... (17 more files)

**Fix Strategy**: Global find-replace
```bash
find /srv/luminous-dynamics/Mycelix-Core/0TML -name "*.py" -type f \
  -exec sed -i 's|Mycelix-Core|Mycelix-Core|g' {} +
```

**Impact**: Prevents path resolution errors when scripts try to load resources

---

## 🎯 Recommended Dataset Configuration

### Option A: 2 Datasets (Minimal) ⚠️ **ACCEPTABLE**

**Datasets**: MNIST + CIFAR-10

**Experiments**: 2 datasets × 3 baselines × 3 Byzantine% = **18 experiments**

**Pros**:
- ✅ Fast to run (~4-6 hours)
- ✅ Both datasets already downloaded
- ✅ Sufficient for workshop papers

**Cons**:
- ⚠️ Weaker generalization claim
- ⚠️ Borderline for top-tier venues

**Time to Complete**: 1-2 days

---

### Option B: 3 Datasets (Recommended) ⭐ **BEST**

**Datasets**: MNIST + CIFAR-10 + Fashion-MNIST

**Experiments**: 3 datasets × 3 baselines × 3 Byzantine% = **27 experiments**

**Pros**:
- ✅ Strong generalization claim
- ✅ Competitive with SOTA papers
- ✅ Fashion-MNIST is same size/format as MNIST (easy to add)
- ✅ Shows cross-domain (digits → clothing)

**Cons**:
- ⏱️ Longer experiment time (~6-10 hours)
- 📦 Need to download Fashion-MNIST (~30MB)

**Time to Complete**: 2-3 days

**Why Fashion-MNIST over SVHN**:
- Same format as MNIST (28×28 grayscale)
- Drop-in replacement, minimal code changes
- Widely used in FL papers
- Smaller download than SVHN

---

### Option C: 3 Datasets with Text (Academic Gold) 🏆 **IDEAL**

**Datasets**: MNIST + CIFAR-10 + Shakespeare

**Experiments**: 3 datasets × 3 baselines × 3 Byzantine% = **27 experiments**

**Pros**:
- ✅ **Cross-modality** (vision + NLP)
- ✅ **Strongest generalization** claim
- ✅ Shakespeare already downloaded (8.3MB)
- ✅ Addresses "does it work for text?" question

**Cons**:
- ⚠️ Need to verify Shakespeare dataset works
- ⚠️ Might need different model architecture
- ⏱️ Slightly longer experiment time

**Time to Complete**: 3-4 days (includes Shakespeare verification)

**Verification Needed**:
1. Check if Shakespeare dataset is properly formatted
2. Test if runner.py supports it
3. May need to add language model support

---

## 🔧 Implementation Plan

### Phase 1: Cleanup & Path Fixes (1-2 hours)

**Tasks**:
1. ✅ Remove duplicate dataset folder
   ```bash
   rm -rf /srv/luminous-dynamics/Mycelix-Core/0TML/experiments/datasets/
   ```

2. ✅ Fix CIFAR-10 download flag in `runner.py:146`
   ```python
   download=True  # Change from False to True
   ```

3. ✅ Fix absolute path resolution in `runner.py:105`
   ```python
   project_root = Path(__file__).parent.parent
   data_dir = Path(self.config['dataset'].get('data_dir', project_root / 'datasets'))
   ```

4. ✅ Global path fix for old project name
   ```bash
   find 0TML -name "*.py" -type f -exec sed -i 's|Mycelix-Core|Mycelix-Core|g' {} +
   ```

5. ✅ Verify dataset paths
   ```bash
   ls -lh /srv/luminous-dynamics/Mycelix-Core/0TML/datasets/
   ```

---

### Phase 2: Dataset Addition (1-3 hours, Optional)

**If choosing Option B (Fashion-MNIST)**:

1. Add Fashion-MNIST to `runner.py`:
   ```python
   elif dataset_name == 'fmnist':
       transform = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.2860,), (0.3530,))
       ])

       train_dataset = datasets.FashionMNIST(
           root=data_dir / 'fmnist',
           train=True,
           transform=transform,
           download=True
       )

       test_dataset = datasets.FashionMNIST(
           root=data_dir / 'fmnist',
           train=False,
           transform=transform,
           download=True
       )
   ```

2. Update Grand Slam config to include Fashion-MNIST

**If choosing Option C (Shakespeare)**:

1. Test Shakespeare dataset loading:
   ```python
   python -c "from experiments.runner import ExperimentRunner; \
              runner = ExperimentRunner('configs/test_shakespeare.yaml'); \
              runner.load_dataset()"
   ```

2. If broken, implement Shakespeare loader
3. May need character-level LSTM model instead of CNN

---

### Phase 3: Mini-Validation Test (1 hour)

**Before running full Grand Slam**:

1. Create minimal test config:
   ```yaml
   dataset:
     name: mnist
     data_dir: /srv/luminous-dynamics/Mycelix-Core/0TML/datasets

   federated:
     num_clients: 10
     num_rounds: 5

   baselines:
     - name: fedavg
   ```

2. Run single experiment:
   ```bash
   cd /srv/luminous-dynamics/Mycelix-Core/0TML/experiments
   python runner.py --config configs/mini_test.yaml
   ```

3. Verify non-zero accuracy (should see >10% after 5 rounds on MNIST)

4. If successful, test CIFAR-10 the same way

5. If both work, proceed to full Grand Slam

---

### Phase 4: Full Grand Slam Execution (6-10 hours)

**Run complete experimental matrix**:

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML/experiments
nohup python run_grand_slam.py &> /tmp/grand_slam.log &
tail -f /tmp/grand_slam.log
```

**Monitor progress**:
- Check for non-zero accuracies
- Watch for dataset loading errors
- Verify Byzantine detection rates

**Expected results** (if working):
- MNIST: 85-95% accuracy for PoGQ+Rep
- CIFAR-10: 60-75% accuracy for PoGQ+Rep
- FedAvg baseline: 15-25pp lower accuracy

---

## 📋 Recommended Action Sequence

### **Today (Oct 14)**:

1. ✅ **Decide on dataset strategy** (2 vs 3 datasets)
2. 🔧 **Execute Phase 1**: Cleanup & fixes (1-2 hours)
3. 🧪 **Execute Phase 3**: Mini-validation (1 hour)
4. 📊 **If mini-validation succeeds**: Start Grand Slam overnight

### **Tomorrow (Oct 15)**:

1. 📈 **Analyze Grand Slam results**
2. 🐛 **Debug any remaining issues**
3. 🔄 **Re-run failed experiments if needed**

### **Oct 16-17**:

1. ✅ **Verify all results valid**
2. 📊 **Generate figures and tables**
3. ✏️ **Update whitepaper Section 4**

---

## 🎯 My Specific Recommendation

### **Use Option B: 3 Datasets (MNIST + CIFAR-10 + Fashion-MNIST)**

**Reasoning**:
1. ✅ Fashion-MNIST is trivial to add (same format as MNIST)
2. ✅ Strengthens generalization claim significantly
3. ✅ Competitive with top-tier FL papers
4. ✅ Only adds ~3 hours to experiment time
5. ✅ Shows robustness across different visual domains

**Why not Shakespeare**:
- ⚠️ Unknown if it's properly implemented
- ⚠️ Requires model architecture changes
- ⚠️ Adds complexity and debugging time
- 🔮 Can save for follow-up paper on cross-modality

**Why not just 2 datasets**:
- ⚠️ Borderline for MLSys/ICML acceptance
- ⚠️ Weaker than competing papers
- ⚠️ Reviewers might question generalization

---

## 📊 Expected Paper Impact

### With 2 Datasets (MNIST + CIFAR-10):
- **Target Venues**: Workshops, smaller conferences
- **Acceptance Probability**: 40-60% at top venues
- **Generalization Claim**: "Validated on two standard benchmarks"

### With 3 Datasets (+ Fashion-MNIST):
- **Target Venues**: MLSys, ICML, NeurIPS
- **Acceptance Probability**: 60-75% at top venues
- **Generalization Claim**: "Robust across diverse visual domains"

### With 3 Datasets (+ Shakespeare):
- **Target Venues**: Top-tier only
- **Acceptance Probability**: 70-85% at top venues
- **Generalization Claim**: "Cross-modality validation (vision + NLP)"
- **Risk**: Implementation complexity

---

## ✅ Next Steps

**Immediate**:
1. Confirm dataset strategy (Option A/B/C)
2. Execute cleanup (remove duplicate folder, fix paths)
3. Fix CIFAR-10 download flag
4. Run mini-validation test

**This Week**:
1. Run full Grand Slam
2. Analyze results
3. Update whitepaper

**User Decision Needed**: Which option do you prefer?
- **Option A**: 2 datasets (fast, minimal)
- **Option B**: 3 datasets with Fashion-MNIST (recommended)
- **Option C**: 3 datasets with Shakespeare (ambitious)
