# Tuesday Morning Quick-Start Guide
**November 11, 2025 - 6:00am Start**

---

## 🎯 Today's Goal
**Analyze v4.1 results + Launch ablation study** (8-10 hours total)

---

## ⚡ Step 1: Check Experiment Status (5 min)

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Check if experiments still running
ps aux | grep "matrix_runner.py"

# Count completed experiments
ls -d results/artifacts_* | wc -l

# Expected: 256 (or close)
# If less: Calculate time remaining
```

### If Experiments Complete ✅
→ Proceed to Step 2

### If Experiments Crashed ❌
```bash
# Resume from checkpoint
./experiments/resume_matrix.sh configs/sanity_slice.yaml

# Wait for completion, then proceed
```

---

## 📊 Step 2: Aggregate Results (1 hour)

### Create aggregation script:
```bash
# Quick aggregation Python script
cat > experiments/aggregate_v4_1_results.py << 'EOF'
#!/usr/bin/env python3
"""Aggregate 256 v4.1 experiment results into attack-defense matrix"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

results_dir = Path("results")
artifact_dirs = sorted(results_dir.glob("artifacts_*"))

print(f"Found {len(artifact_dirs)} experiment results")

# Parse experiment results
data = defaultdict(lambda: defaultdict(list))

for artifact_dir in artifact_dirs:
    try:
        # Read detection metrics
        metrics_file = artifact_dir / "detection_metrics.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        # Extract experiment config
        config_file = artifact_dir / "experiment_config.json"
        with open(config_file) as f:
            config = json.load(f)

        dataset = config['dataset']['name']
        attack = config['attack']['type']
        defense = config['baselines'][0]
        seed = config['seed']

        # Extract key metrics
        auroc = metrics.get('auroc', 0.0)
        tpr_at_fpr_10 = metrics.get('tpr_at_fpr_0.10', 0.0)
        fpr_at_tpr_90 = metrics.get('fpr_at_tpr_0.90', 1.0)

        # Store
        key = (dataset, attack, defense)
        data[key]['auroc'].append(auroc)
        data[key]['tpr'].append(tpr_at_fpr_10)
        data[key]['fpr'].append(fpr_at_tpr_90)

    except Exception as e:
        print(f"Warning: Failed to parse {artifact_dir}: {e}")

# Compute statistics
print("\nAttack-Defense Matrix (TPR @ FPR=0.10)")
print("=" * 80)

for (dataset, attack, defense), metrics in sorted(data.items()):
    tpr_mean = np.mean(metrics['tpr'])
    tpr_std = np.std(metrics['tpr'])
    auroc_mean = np.mean(metrics['auroc'])

    print(f"{dataset:8s} | {attack:20s} | {defense:15s} | "
          f"TPR: {tpr_mean:.3f}±{tpr_std:.3f} | AUROC: {auroc_mean:.3f}")

# Save to JSON for LaTeX table generation
output = {}
for (dataset, attack, defense), metrics in data.items():
    key = f"{dataset}_{attack}_{defense}"
    output[key] = {
        'tpr_mean': float(np.mean(metrics['tpr'])),
        'tpr_std': float(np.std(metrics['tpr'])),
        'auroc_mean': float(np.mean(metrics['auroc'])),
        'n_seeds': len(metrics['tpr'])
    }

with open('results/v4_1_aggregated.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Results saved to: results/v4_1_aggregated.json")
EOF

chmod +x experiments/aggregate_v4_1_results.py
python experiments/aggregate_v4_1_results.py
```

**Output**: `results/v4_1_aggregated.json` with all statistics

---

## 📝 Step 3: Fill Attack-Defense Matrix (1.5 hours)

### Edit Results section:
```bash
# Open results file
vim paper-submission/latex-submission/sections/05-results.tex

# Find: "% TODO: Add v4.1 attack-defense matrix"
# Add after existing FEMNIST table (~line 143)
```

### Table Template:
```latex
\subsection{PoGQ v4.1 Attack-Defense Matrix}

Comprehensive evaluation across 8 attacks × 8 defenses × 2 datasets × 2 seeds
(256 experiments total).

\begin{table*}[t]
\centering
\caption{Attack-Defense Performance Matrix (TPR @ FPR=0.10, Mean ± Std)}
\label{tab:attack_defense_matrix}
\resizebox{\textwidth}{!}{
\begin{tabular}{llcccccccc}
\toprule
\textbf{Dataset} & \textbf{Attack} & \textbf{FedAvg} & \textbf{CoordMedian} & \textbf{CM-Safe} & \textbf{RFA} & \textbf{FLTrust} & \textbf{BOBA} & \textbf{CBF} & \textbf{PoGQ v4.1} \\
\midrule
% FILL WITH AGGREGATED DATA
EMNIST-IID & sign\_flip & X.XX±X.XX & ... \\
% ... (continue for all 16 dataset-attack combinations)
\bottomrule
\end{tabular}}
\end{table*}

\textbf{Key Findings}:

\textbf{(1) Defense Ranking}: [Analyze which defenses perform best overall]

\textbf{(2) Attack Difficulty}: [Identify hardest attacks to detect]

\textbf{(3) Dataset Effects}: [Compare IID vs non-IID performance]
```

**Fill from**: `results/v4_1_aggregated.json`

---

## ✍️ Step 4: Write v4.1 Findings (1.5 hours)

### Add results subsection:
```latex
\subsection{PoGQ v4.1 Comprehensive Evaluation}

We evaluate PoGQ v4.1 against 7 defense baselines across 8 Byzantine attack
types on EMNIST (IID and non-IID with $\alpha=0.3$).

% Describe top 3 findings with supporting data
% Reference Table~\ref{tab:attack_defense_matrix}
```

### Structure:
1. **Overall performance** (3 paragraphs)
   - Best defenses: FLTrust, PoGQ, BOBA (order by AUROC)
   - Attack difficulty ranking
   - Statistical significance tests

2. **Per-attack analysis** (2 paragraphs)
   - Sign flip: All defenses detect well
   - Scaling: Magnitude-based fail
   - Collusion: Coordination challenges

3. **Dataset effects** (1 paragraph)
   - IID vs non-IID performance gaps
   - Dirichlet α=0.3 impact on FPR

**Time budget**: Write 1500-2000 words in 90 minutes

---

## 🚀 Step 5: Launch Ablation Study (2 hours)

### 10:00am - Verify Config
```bash
# Check ablation config
cat configs/ablation_pogq_components.yaml

# Verify defense variants exist in code
grep -r "pogq_v4_1_no_pca" src/
# (Should find implementation or add to TODO)
```

### 10:15am - Launch Experiments
```bash
# Launch ablation (6-7 hour run)
nohup python experiments/matrix_runner.py \
  --config configs/ablation_pogq_components.yaml \
  2>&1 | tee /tmp/ablation_run_$(date +%Y%m%d).log &

# Save PID
echo $! > /tmp/ablation_pid.txt

# Monitor first experiment
tail -f /tmp/ablation_run_*.log
```

### 10:30am - Verify Running
```bash
# Check process
PID=$(cat /tmp/ablation_pid.txt)
ps aux | grep $PID

# Check artifacts appearing
watch -n 60 'ls -d results/artifacts_* | wc -l'

# Expected: 1 new artifact every ~7 minutes
```

---

## 📅 Rest of Tuesday Schedule

### 12:00pm - Lunch Break ☕
Ablation running in background

### 1:00pm - Update Results Section (2 hours)
- Incorporate v4.1 findings
- Polish attack-defense matrix
- Add supporting figures (if time)

### 3:00pm - Prep Scalability (1 hour)
- Review `configs/scalability_nodes.yaml`
- Test N=50 baseline manually
- Queue for Wednesday launch

### 4:00pm - Check Ablation Progress
```bash
# Should have ~28/54 complete by 4pm
ls -d results/artifacts_* | wc -l
```

### 5:00pm - Wrap Up
- Commit paper updates
- Document tomorrow's tasks
- Set alarm for ablation completion check (overnight)

---

## 🎯 Success Criteria for Tuesday

### Must Complete ✅
- [ ] v4.1 results aggregated
- [ ] Attack-defense matrix filled
- [ ] v4.1 findings written (1500+ words)
- [ ] Ablation study launched and running

### Nice to Have 🎁
- [ ] First results figure generated
- [ ] Scalability config tested
- [ ] Discussion section started

---

## 📞 If You Need Help

### Experiments stuck?
```bash
# Check last 50 lines of any experiment
tail -50 /tmp/matrix_runner_*.log

# Resume from checkpoint
./experiments/resume_matrix.sh
```

### Aggregation fails?
- Check artifact directories have `detection_metrics.json`
- Verify JSON format with: `jq . results/artifacts_*/detection_metrics.json | head`

### Can't find data?
- All results in: `results/artifacts_YYYYMMDD_HHMMSS/`
- Each has: detection_metrics.json, per_bucket_fpr.json, model_metrics.json

---

## ⏰ Time Check

If you're reading this and it's:
- **Before 6am**: Go back to sleep! 😴
- **6am-8am**: Perfect timing, start Step 1
- **8am-10am**: Should be on Step 3 (filling matrix)
- **10am-12pm**: Should be launching ablation
- **After 12pm**: Adjust schedule, focus on must-haves

---

**Good luck! You've got this.** 💪

The hard work (experiments) is done running overnight.
Today is pure analysis and writing - your strengths!

---

*Prepared: November 10, 2025, 11:45pm*
*For: Tuesday November 11, 2025, 6:00am start*
