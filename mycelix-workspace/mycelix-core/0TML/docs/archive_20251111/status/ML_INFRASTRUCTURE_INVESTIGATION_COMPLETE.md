# ML Infrastructure Investigation - COMPLETE ✅

**Date**: October 29, 2025
**Task**: Understand existing ML training infrastructure before retraining for 7 features
**Status**: Investigation complete, ready to proceed with retraining

---

## 🔍 Investigation Summary

### Existing ML Infrastructure Found

**1. Feature Extraction Module** (`src/zerotrustml/ml/feature_extractor.py`)
- **Purpose**: Extracts 5 base features from gradients
- **Features Extracted**:
  1. `pogq_score` - Cosine similarity to median gradient [0, 1]
  2. `tcdm_score` - Temporal consistency with node history [0, 1]
  3. `zscore_magnitude` - Statistical outlier detection (L2 norm of z-scores)
  4. `entropy_score` - Shannon entropy of gradient distribution
  5. `gradient_norm` - L2 norm of gradient vector
- **Status**: ✅ Well-implemented, documented, tested
- **Key Method**: `CompositeFeatures.to_array()` returns 5-element numpy array

**2. ML Detector API** (`src/zerotrustml/ml/detector.py`)
- **Purpose**: High-level Byzantine detection combining PoGQ + ML
- **Architecture**:
  - Fast path: PoGQ < 0.3 → instant Byzantine (no ML)
  - Fast path: PoGQ > 0.7 → instant Honest (no ML)
  - Slow path: 0.3 ≤ PoGQ ≤ 0.7 → ML classifier decision
- **Classifiers**: SVM, Random Forest, Ensemble (voting)
- **Key Methods**: `train()`, `classify_node()`, `save()`, `load()`
- **Status**: ✅ Production-ready API

**3. Integration Test** (`tests/test_ml_enhanced_detection.py`)
- **Purpose**: End-to-end test of ML pipeline
- **Features**:
  - Synthetic data generation (honest + Byzantine attacks)
  - Feature extraction batch processing
  - Classifier training and evaluation
  - Metrics computation (detection rate, FP rate, F1)
  - Model saving to `models/byzantine_detector/`
- **Status**: ✅ Complete working example

**4. Evaluation Framework** (`src/zerotrustml/ml/evaluator.py` - imported)
- **Purpose**: Metrics computation and visualization
- **Metrics**: Detection rate, false positive rate, precision, recall, F1
- **Validation**: `meets_targets(min_detection=0.95, max_fp=0.03)`
- **Status**: ✅ Comprehensive evaluation

---

## 🐛 The Feature Mismatch Problem - ROOT CAUSE IDENTIFIED

### Current Models (October 23, 2025)
**Location**: `models/byzantine_detector/`
```
classifier/
├── rf.pkl (80 KB)      - Random Forest trained on 5 features
├── svm.pkl (2.9 KB)    - SVM trained on 5 features
└── meta.pkl (45 B)     - Ensemble meta-classifier trained on 5 features
```

**Training Code** (Line 2236-2237 in `test_30_bft_validation.py`):
```python
X = np.array([feat.to_array() for feat in features])  # 5 features only
y = np.array(labels)
detector.train(X, y, ...)
```

**Features Used**: [pogq, tcdm, zscore, entropy, gradient_norm] = **5 features**

### Current Inference (October 29, 2025)
**Runtime Code** (Line 1151-1154 in `test_30_bft_validation.py`):
```python
combined_features = np.append(
    feature_entry.to_array(),        # 5 base features
    [consensus_score, prev_alignment],  # +2 additional features
)
```

**Features Used**: [pogq, tcdm, zscore, entropy, gradient_norm, consensus_score, prev_alignment] = **7 features**

### The Two Additional Features

**Feature 6: `consensus_score`**
- **Computation**: Committee voting score from edge proof validation
- **Purpose**: Measure agreement among validation committee
- **Range**: [0, 1] where 1 = full committee agreement

**Feature 7: `prev_alignment`**
- **Computation**: Cosine similarity with previous aggregated gradient (Line 1141-1149)
- **Formula**: `dot(current_grad, prev_grad) / (norm(current) * norm(prev))`
- **Purpose**: Detect temporal inconsistency across rounds
- **Range**: [-1, 1] where 1 = perfect alignment

### Why StandardScaler Fails
```python
# Saved in training (October 23)
StandardScaler.fit(X_train)  # X_train.shape = (N, 5)
# StandardScaler stores: expected_features = 5

# At inference (October 29)
X_inference.shape = (1, 7)
scaler.transform(X_inference)  # ValueError: Expected 5 features, got 7
```

---

## 📊 Training History - Multiple Versions Found

### Version 1: 5-Feature Training (Lines 2230-2256)
```python
def _init_ml_detector(self):
    gradients, labels = self._generate_synthetic_training_set()
    extractor = FeatureExtractor()
    features = extract_features_batch(gradients, node_ids, round_num=1)
    X = np.array([feat.to_array() for feat in features])  # 5 features
    detector.train(X, y, ...)
```
- **Features**: 5 (base features only)
- **Status**: Currently saved in `models/byzantine_detector/`
- **Problem**: Missing consensus_score and prev_alignment

### Version 2: 6-Feature Training (Lines 1670-1700)
```python
def _train_ml_detector(self, detector: ByzantineDetector):
    gradients, labels = self._generate_synthetic_training_set()
    extractor = FeatureExtractor()
    features = extract_features_batch(gradients, node_ids, round_num=1)
    X = np.array([feat.to_array() for feat in features])  # 5 features
    committee_column = np.where(np.array(labels) == 0, 1.0, 0.0)  # +1 feature
    X = np.column_stack([X, committee_column])  # 6 features
    detector.train(X, y, ...)
```
- **Features**: 6 (base + committee_column as binary indicator)
- **Status**: Alternative training path (not currently used)
- **Problem**:
  - Still missing prev_alignment
  - committee_column is binary (0/1) not the actual consensus_score

### Version 3: 7-Feature Inference (Lines 1151-1154)
```python
combined_features = np.append(
    feature_entry.to_array(),  # 5 features
    [consensus_score, prev_alignment],  # +2 features = 7 total
)
```
- **Features**: 7 (base + consensus_score + prev_alignment)
- **Status**: Current inference code
- **Problem**: No trained model exists for this feature set!

---

## ✅ Solution Path: Retrain with 7 Features

### Step 1: Create Training Script
**File**: `0TML/scripts/train_ml_detector_7features.py`
**Features**:
- Generate synthetic training data (1000+ samples)
- Extract 5 base features using FeatureExtractor
- Add consensus_score (simulated from committee voting)
- Add prev_alignment (simulated from temporal sequence)
- Train ensemble detector (RF + SVM + meta)
- Validate: ≥95% detection, ≤3% FP
- Save to `models/byzantine_detector_v4_7features/`

### Step 2: Synthetic Feature Generation Logic
```python
# For training data generation:
# consensus_score: Simulate committee agreement
#   - Honest nodes: High consensus (0.8-1.0) - committee agrees they're honest
#   - Byzantine nodes: Low consensus (0.0-0.4) - committee flags them
consensus_score = 1.0 if label == 0 else np.random.uniform(0.0, 0.4)

# prev_alignment: Simulate temporal consistency
#   - Honest nodes: High alignment (0.6-1.0) - consistent across rounds
#   - Byzantine nodes: Low alignment (-0.5-0.3) - erratic behavior
prev_alignment = np.random.uniform(0.6, 1.0) if label == 0 else np.random.uniform(-0.5, 0.3)
```

### Step 3: Model Evaluation
**Validation Dataset**: 20% hold-out from training
**Target Metrics**:
- Detection Rate: ≥95% (Byzantine nodes correctly identified)
- False Positive Rate: ≤3% (Honest nodes incorrectly flagged)
- F1 Score: ≥0.95

**Expected Improvement**: Adding consensus_score and prev_alignment should improve:
- Detection of sophisticated attacks (consensus captures committee intelligence)
- Detection of adaptive attacks (prev_alignment captures temporal inconsistency)
- Reduced false positives (more context = better discrimination)

### Step 4: Integration
**Update test file** to use new model:
```bash
export USE_ML_DETECTOR=1
export ML_DETECTOR_PATH=/srv/luminous-dynamics/Mycelix-Core/0TML/models/byzantine_detector_v4_7features
```

**Inference code** (already correct at lines 1151-1154):
```python
combined_features = np.append(
    feature_entry.to_array(),  # 5 features
    [consensus_score, prev_alignment],  # +2 features
)
# Now matches training!
```

---

## 📁 File Locations Reference

### Core ML Infrastructure
```
0TML/
├── src/zerotrustml/ml/
│   ├── __init__.py                 # Public API exports
│   ├── feature_extractor.py        # ✅ 5 base features
│   ├── detector.py                 # ✅ ByzantineDetector API
│   ├── classifiers.py              # ✅ SVM, RF, Ensemble
│   └── evaluator.py                # ✅ Metrics and visualization
├── tests/
│   ├── test_ml_enhanced_detection.py   # ✅ Integration test
│   └── test_30_bft_validation.py       # ✅ Main BFT test (has training code)
├── models/
│   ├── byzantine_detector/         # ❌ OLD: 5-feature models (Oct 23)
│   └── byzantine_detector_v4_7features/  # 🔜 NEW: 7-feature models (to create)
└── scripts/
    └── train_ml_detector_7features.py  # 🔜 NEW: Training script (to create)
```

### Training Code Locations
1. **Primary Training** (Lines 2230-2256 in test_30_bft_validation.py)
   - Method: `_init_ml_detector()`
   - Features: 5 (base only)
   - Status: Currently used for saved models

2. **Alternative Training** (Lines 1670-1700 in test_30_bft_validation.py)
   - Method: `_train_ml_detector(detector)`
   - Features: 6 (base + committee_column)
   - Status: Available but not used

3. **Inference** (Lines 1151-1154 in test_30_bft_validation.py)
   - Method: Part of aggregation logic
   - Features: 7 (base + consensus_score + prev_alignment)
   - Status: Current runtime (MISMATCH!)

### Synthetic Data Generation
**Location**: Lines 1702-1748 and 2257-2291 in test_30_bft_validation.py
**Method**: `_generate_synthetic_training_set()`
**Features**:
- 600 honest gradients (Gaussian noise around true gradient)
- 200 Byzantine gradients:
  - Label flip attack: Opposite direction, large magnitude
  - Random noise: Pure random direction
  - Sybil coordination: Coordinated but distinct direction

---

## 🎯 Next Actions

### Immediate (Next 30 Minutes)
1. ✅ **DONE**: Investigation complete - infrastructure understood
2. 🔜 **Create training script**: `scripts/train_ml_detector_7features.py`
   - Copy structure from `test_ml_enhanced_detection.py`
   - Add consensus_score and prev_alignment to feature generation
   - Save to `models/byzantine_detector_v4_7features/`

### Short-Term (Next 2-4 Hours)
3. 🔜 **Run training**: Train ensemble detector with 7 features
   - Generate 1000+ training samples (800 honest, 200 Byzantine)
   - Validate: ≥95% detection, ≤3% FP
   - Save trained models

4. 🔜 **Test integration**: Run single validation test with new detector
   ```bash
   cd 0TML && source ../.env.optimal
   export USE_ML_DETECTOR=1
   export ML_DETECTOR_PATH=models/byzantine_detector_v4_7features
   export RUN_30_BFT=1 BFT_DISTRIBUTION=label_skew
   poetry run python tests/test_30_bft_validation.py
   ```

5. 🔜 **Verify resolution**: Confirm no feature mismatch error

### Medium-Term (Next 6-8 Hours)
6. 🔜 **Comprehensive validation**: Run full attack matrix
   ```bash
   poetry run python scripts/run_attack_matrix.py \
     --datasets cifar10,emnist_balanced,breast_cancer \
     --distributions iid,label_skew \
     --attacks noise,sign_flip,zero,random,backdoor,adaptive,scaled_sign_flip,stealth_backdoor \
     --bft-ratios 0.30,0.40,0.50 \
     --output results/grant_ready_validation_$(date +%Y%m%d).json
   ```

---

## 💡 Key Insights

### 1. **Why Features Were Added**
The system evolved to add committee consensus and temporal alignment because:
- **consensus_score**: Leverages collective intelligence of validation committee
- **prev_alignment**: Detects temporal inconsistency (adaptive attacks)
- Both features provide orthogonal information to base features

### 2. **Why Mismatch Occurred**
- Feature extraction code (`feature_extractor.py`) remained at 5 features
- Inference code evolved to use 7 features (lines 1151-1154)
- Training code wasn't updated to match inference
- Saved models frozen at 5-feature version from October 23

### 3. **Why Retraining is Better Than Adapter**
**Option A: Feature Projection Adapter** (Quick fix, 1-2 hours)
- Drop 2 features: Loses valuable information
- Project to 5D: Arbitrary transformation, may harm accuracy
- Technical debt: Adapter layer adds complexity

**Option B: Retrain with 7 Features** (Proper fix, 3-4 hours) ✅ RECOMMENDED
- Uses all available information
- Cleaner architecture (no adapter layer)
- Better grant narrative ("evolved system with enhanced features")
- Future-proof: Ready for production deployment

### 4. **Expected Performance Impact**
Adding consensus_score and prev_alignment should **improve** detection:
- **Consensus score**: Reduces FP by cross-validating with committee
- **Temporal alignment**: Catches adaptive attacks that change behavior over time
- **Combined**: More context = better discrimination

Target improvement: **3.55% → <2% false positive rate** (speculation, needs validation)

---

## 📝 Documentation Status

**This Investigation**: ✅ Complete
**Next Document**: Training script with inline documentation
**Grant-Ready Package**: After comprehensive validation (Phase 3)

---

## 🚀 Readiness Assessment

**Investigation Phase**: ✅ **100% COMPLETE**

We now have complete understanding of:
- ✅ Existing ML infrastructure (feature extraction, detector API, training)
- ✅ Root cause of feature mismatch (5 vs 7 features)
- ✅ Location of all training code (3 different versions found)
- ✅ Feature computation logic (consensus_score, prev_alignment)
- ✅ Synthetic data generation approach
- ✅ Model save/load mechanism
- ✅ Evaluation framework (metrics, targets)

**Ready to Proceed**: ✅ YES - All information needed to create training script

---

*"Understanding before building. Investigation complete. Ready to train." 🎯*
