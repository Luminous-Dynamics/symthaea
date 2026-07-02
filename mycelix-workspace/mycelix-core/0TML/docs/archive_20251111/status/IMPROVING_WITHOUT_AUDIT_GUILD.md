# Improving Byzantine Detection Without an Audit Guild

**Date**: 2025-10-28  
**Context**: No audit guild available yet, need alternative approaches  
**Goal**: Continue improving from 42.9% → <5% FP without human-in-the-loop review

## 📊 Current Situation

### What We Have
- ✅ **Automated CI/CD**: Nightly regression tests (`matl-regression.yml`)
- ✅ **Prometheus Metrics**: Detection rates, FP rates, success flags
- ✅ **Comprehensive Tracing**: JSONL logs with per-node behavior data
- ✅ **Behavior-Based Recovery**: 42.9% FP (major improvement from 92.9%)

### What We're Missing
- ❌ **Human Review**: No "audit guild" to manually review edge cases
- ❌ **Grafana Dashboards**: Metrics exported but not visualized yet
- ❌ **Automated Alerts**: No proactive detection of regressions
- ❌ **Feedback Loop**: No way to incorporate manual findings back into system

## 🎯 Three-Track Improvement Strategy

### Track 1: Automated Improvement (NO HUMANS NEEDED) 🤖

**Goal**: Use data-driven optimization to reach <5% FP automatically.

#### 1.1 Parameter Sweep with Grid Search
Instead of manual tuning, automate exploration of parameter space:

```python
#!/usr/bin/env python3
"""Grid search for optimal label skew parameters."""

import subprocess
import json
import itertools

# Parameter ranges to test
param_grid = {
    'BEHAVIOR_RECOVERY_THRESHOLD': [2, 3, 4, 5],
    'BEHAVIOR_RECOVERY_BONUS': [0.08, 0.10, 0.12, 0.15],
    'LABEL_SKEW_COS_MIN': [-0.3, -0.4, -0.5],
    'LABEL_SKEW_COS_MAX': [0.93, 0.95, 0.97],
}

results = []

for params in itertools.product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), params))
    
    # Run test
    env = {
        'RUN_30_BFT': '1',
        'BFT_DISTRIBUTION': 'label_skew',
        **param_dict
    }
    
    result = subprocess.run(
        ['python', 'tests/test_30_bft_validation.py'],
        env={**os.environ, **env},
        capture_output=True,
        timeout=600
    )
    
    # Extract metrics from output
    # ... parse and record results ...
    
    results.append({
        'params': param_dict,
        'fp_rate': fp_rate,
        'detection_rate': detection_rate,
        'honest_rep': honest_rep
    })

# Find best parameters
best = min(results, key=lambda x: x['fp_rate'] if x['detection_rate'] >= 0.95 else 999)
print(f"Best params: {best['params']}")
print(f"FP rate: {best['fp_rate']:.1%}")
```

**Expected outcome**: Automatically find optimal parameters in 2-6 hours of compute time.

#### 1.2 Regression-Based Threshold Learning
Use linear/logistic regression to learn adaptive thresholds:

```python
from sklearn.linear_model import LogisticRegression

# Collect features from trace data
features = []  # [pogq_score, cos_anchor, committee_score, reputation, ...]
labels = []     # [0 = honest, 1 = byzantine]

# Train classifier on high-confidence cases
high_confidence = (reputation > 0.8) | (consecutive_byzantine > 5)
X_train = features[high_confidence]
y_train = labels[high_confidence]

clf = LogisticRegression().fit(X_train, y_train)

# Use learned model to predict on uncertain cases
uncertain = ~high_confidence
y_pred = clf.predict(features[uncertain])
```

**Expected outcome**: Learn decision boundaries from data, not manual tuning.

#### 1.3 Bayesian Optimization
Use Bayesian optimization for faster convergence than grid search:

```python
from skopt import gp_minimize

def objective(params):
    """Run test and return FP rate (minimize this)."""
    threshold, bonus, cos_min, cos_max = params
    # ... run test with these params ...
    return fp_rate if detection_rate >= 0.95 else 1.0

# Define parameter space
space = [
    (2, 6),         # BEHAVIOR_RECOVERY_THRESHOLD
    (0.05, 0.20),   # BEHAVIOR_RECOVERY_BONUS
    (-0.6, -0.2),   # LABEL_SKEW_COS_MIN
    (0.90, 0.99),   # LABEL_SKEW_COS_MAX
]

# Run optimization (much faster than grid search)
result = gp_minimize(
    objective,
    space,
    n_calls=50,      # 50 evaluations instead of 256 for grid search
    random_state=42
)

print(f"Best params: {result.x}")
print(f"Best FP rate: {result.fun:.1%}")
```

**Expected outcome**: Find near-optimal parameters in 1-2 hours instead of 6+ hours.

### Track 2: Self-Monitoring & Alerting (AUTONOMOUS QUALITY ASSURANCE) 📊

**Goal**: Let the system monitor itself and alert when things go wrong.

#### 2.1 Implement P3: Prometheus + Grafana Monitoring

**Step 1**: Set up Grafana dashboard (already have Prometheus export)

```bash
# 1. Install Grafana
docker run -d -p 3000:3000 grafana/grafana

# 2. Add Prometheus data source (point to metrics files)

# 3. Create dashboard with these panels:
# - FP rate over time (by distribution)
# - Detection rate over time
# - Honest reputation histogram
# - Alert if FP > 10% or detection < 90%
```

**Step 2**: Add automated alerts

```yaml
# grafana_alerts.yml
alerts:
  - name: High False Positive Rate
    condition: matl_false_positive_rate_percent{distribution="label_skew"} > 10
    for: 5m
    action: email/slack/pagerduty
    
  - name: Low Detection Rate
    condition: matl_detection_rate_percent < 90
    for: 5m
    action: email/slack/pagerduty
    
  - name: Regression Failure
    condition: matl_regression_success == 0
    for: 1m
    action: email/slack/pagerduty
```

**Expected outcome**: Immediate notification of regressions, no manual checking needed.

#### 2.2 Add Statistical Process Control (SPC)

Implement control charts to detect anomalies:

```python
import numpy as np

class SPCMonitor:
    """Statistical Process Control for BFT metrics."""
    
    def __init__(self, window_size=20):
        self.history = []
        self.window_size = window_size
    
    def check(self, value, metric_name):
        """Check if value is within control limits."""
        self.history.append(value)
        
        if len(self.history) < self.window_size:
            return "insufficient_data"
        
        recent = self.history[-self.window_size:]
        mean = np.mean(recent)
        std = np.std(recent)
        
        # 3-sigma rule: 99.7% of values should be within ±3σ
        ucl = mean + 3 * std  # Upper Control Limit
        lcl = mean - 3 * std  # Lower Control Limit
        
        if value > ucl or value < lcl:
            return f"ALERT: {metric_name} = {value:.3f} outside control limits [{lcl:.3f}, {ucl:.3f}]"
        
        return "ok"

# Usage
fp_monitor = SPCMonitor()
for round_fp_rate in test_results:
    status = fp_monitor.check(round_fp_rate, "FP Rate")
    if status != "ok":
        send_alert(status)
```

**Expected outcome**: Detect regressions automatically, even if metrics are "passing" but trending wrong direction.

### Track 3: Data-Driven Insights (LEARN FROM FAILURES) 📈

**Goal**: Extract patterns from failures to inform improvements without human labeling.

#### 3.1 Cluster Analysis of False Positives

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Collect features from all false positive nodes
fp_features = []
for trace in label_skew_traces:
    for node in trace['nodes']:
        if node['ground_truth'] == 'honest' and not node['correct']:
            fp_features.append([
                node['scores']['pogq'],
                node['scores']['cos_anchor'],
                node['scores']['cos_median'],
                node['reputation'],
                node['committee_score'],
            ])

# Find clusters of similar FP patterns
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(fp_features)

# Analyze each cluster
for cluster_id in range(3):
    cluster_nodes = fp_features[clusters == cluster_id]
    print(f"Cluster {cluster_id}:")
    print(f"  Avg PoGQ: {np.mean(cluster_nodes[:, 0]):.3f}")
    print(f"  Avg cos_anchor: {np.mean(cluster_nodes[:, 1]):.3f}")
    print(f"  Count: {len(cluster_nodes)}")
    
    # Actionable insight: If cluster has cos_anchor in specific range,
    # we know to widen that range in the cosine guard
```

**Expected outcome**: Identify 2-3 distinct FP patterns, target fixes for each pattern.

#### 3.2 Decision Tree for Interpretable Rules

```python
from sklearn.tree import DecisionTreeClassifier, export_text

# Train decision tree on trace data
X = features  # [pogq, cos_anchor, committee_score, ...]
y = labels    # [0=honest, 1=byzantine]

tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20)
tree.fit(X, y)

# Extract human-readable rules
rules = export_text(tree, feature_names=['pogq', 'cos_anchor', 'committee_score'])
print(rules)

# Example output:
# |--- cos_anchor <= -0.35
# |   |--- pogq <= 0.30
# |   |   |--- class: 1 (Byzantine)
# |   |--- pogq > 0.30
# |   |   |--- class: 0 (Honest)  <-- Insight: Need to protect pogq > 0.3 even if cos_anchor low
```

**Expected outcome**: Automatically discover rules that humans would take weeks to find.

#### 3.3 Automated A/B Testing

```python
# Test two approaches simultaneously
configs = {
    'config_a': {'BEHAVIOR_RECOVERY_THRESHOLD': 3, 'LABEL_SKEW_COS_MIN': -0.3},
    'config_b': {'BEHAVIOR_RECOVERY_THRESHOLD': 4, 'LABEL_SKEW_COS_MIN': -0.4},
}

results = {}
for name, config in configs.items():
    result = run_test(config)
    results[name] = {
        'fp_rate': result['fp_rate'],
        'detection_rate': result['detection_rate'],
        'p_value': ttest(result, baseline)  # Statistical significance
    }

# Report winner
best = min(results.items(), key=lambda x: x[1]['fp_rate'])
print(f"Winner: {best[0]} with FP={best[1]['fp_rate']:.1%} (p<{best[1]['p_value']:.3f})")
```

**Expected outcome**: Objectively compare approaches with statistical rigor.

## 📋 Recommended Implementation Plan

### Phase 1: Quick Automated Wins (1-2 days)
1. ✅ **Run grid search** on parameters (Option 1.1)
   - Explore 64-256 combinations
   - Find best performing config
   - Document which parameters matter most

2. ✅ **Implement SPC monitoring** (Option 2.2)
   - Add to CI/CD pipeline
   - Alert if any metric > 3σ from mean

3. ✅ **Cluster analysis of FPs** (Option 3.1)
   - Identify 2-3 FP patterns
   - Target fixes for each pattern

### Phase 2: Infrastructure (2-4 days)
1. ✅ **Set up Grafana dashboard** (Option 2.1)
   - Visualize all Prometheus metrics
   - Add alert rules
   - Share dashboard with team

2. ✅ **Implement Bayesian optimization** (Option 1.3)
   - Faster than grid search
   - Use for ongoing tuning

3. ✅ **Add automated A/B testing** (Option 3.3)
   - Compare approaches objectively
   - Build confidence in changes

### Phase 3: Advanced ML (4-7 days)
1. ⏳ **Train regression model** (Option 1.2)
   - Learn thresholds from data
   - Deploy as "learned detector"

2. ⏳ **Extract decision tree rules** (Option 3.2)
   - Interpret ML decisions
   - Document learned heuristics

3. ⏳ **Build feedback loop**
   - Automatically retrain on new data
   - Continuously improve

## 🎯 Success Metrics (Without Human Review)

| Metric | Current | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|--------|---------|----------------|----------------|----------------|
| **FP Rate** | 42.9% | 25% (grid search) | 10% (Bayesian opt) | **<5%** (learned) |
| **Detection Rate** | 100% | ≥95% | ≥95% | ≥95% |
| **Automation** | Manual tests | Grid search | Grafana alerts | Full feedback loop |
| **Time to Improvement** | Days-weeks | Hours | Minutes | Continuous |

## 💡 Key Insight: You Don't Need Humans Yet!

**Why this works without an audit guild:**

1. **Clear Ground Truth**: You have Byzantine flags in test data (known honest/Byzantine)
2. **Rich Trace Data**: JSONL logs have everything needed for analysis
3. **Objective Metrics**: FP rate, detection rate are unambiguous
4. **Large Search Space**: 1000s of parameter combinations to explore

**When you'll need human review:**
- Production deployment (review real-world edge cases)
- Adversarial attacks not in test suite
- Policy decisions (e.g., "is 5% FP acceptable?")
- Trust building with users

**But for now**: Automated optimization can get you from 42.9% → <5% FP without any human intervention!

## 🚀 Next Steps (Choose Your Path)

### Path A: Fast Results (Recommended for immediate progress)
```bash
# 1. Run grid search (4-6 hours)
python scripts/grid_search_label_skew.py

# 2. Apply best params
export BEHAVIOR_RECOVERY_THRESHOLD=3  # From grid search
export BEHAVIOR_RECOVERY_BONUS=0.12
export LABEL_SKEW_COS_MIN=-0.4

# 3. Test and document
python tests/test_30_bft_validation.py
```

### Path B: Build Infrastructure (Recommended for long-term)
```bash
# 1. Set up Grafana
docker-compose up grafana

# 2. Import dashboard
grafana-cli dashboards import bft_monitoring.json

# 3. Configure alerts
# ... (see section 2.1)
```

### Path C: ML-Driven (Recommended for research)
```bash
# 1. Extract training data
python scripts/extract_training_data_from_traces.py

# 2. Train model
python scripts/train_adaptive_detector.py

# 3. Evaluate
python scripts/evaluate_learned_detector.py
```

---

**Bottom Line**: You have everything you need to continue improving WITHOUT an audit guild. Focus on automated parameter search (Track 1) and self-monitoring (Track 2) to reach <5% FP target. Save human review for production deployment later.

**Immediate Action**: Choose Path A (grid search) to find optimal parameters in next 4-6 hours of compute time. Expected improvement: 42.9% → 15-25% FP.
