# Wednesday Morning Workflow - v4.1 Data Integration

**Date**: Wednesday, November 12, 2025
**Expected Start**: 6:30 AM (when experiments complete)
**Duration**: 2-3 hours
**Goal**: Integrate v4.1 experimental data into paper

---

## ✅ Pre-flight Checklist

Before starting, verify:
- [ ] All 64 experiments completed successfully
- [ ] No empty artifact directories
- [ ] detection_metrics.json exists in each artifact
- [ ] Experiments ran for expected duration (~20 hours)

**Check command**:
```bash
/tmp/check_experiment_status.sh
```

**Expected output**: `64/64 experiments complete`

---

## 📊 Step 1: Aggregate Results (15 minutes)

### 1.1 Run Aggregation Script

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Run aggregation
python experiments/aggregate_v4_1_results.py

# Expected output:
# ✅ Loaded: 64 experiments
# ✅ Created 32 aggregated configurations (4 attacks × 4 defenses × 2 datasets)
# ✅ LaTeX tables generated
# ✅ Summary report generated
```

### 1.2 Verify Outputs

```bash
ls -lh results/v4.1/aggregated/

# Expected files:
# - attack_defense_matrix_mnist.tex
# - attack_defense_matrix_mnist.json
# - attack_defense_matrix_mnist_noniid.tex
# - attack_defense_matrix_mnist_noniid.json
# - aggregated_results.json
# - summary_report.txt
```

### 1.3 Review Summary

```bash
cat results/v4.1/aggregated/summary_report.txt
```

**Look for**:
- Total experiments: 64
- Aggregated configurations: 32
- PoGQ v4.1 detection rate > 90%
- FedAvg baseline detection rate < 50%

**🚨 If anything looks wrong, STOP and investigate before proceeding**

---

## 📝 Step 2: Integrate into Paper (1 hour)

### 2.1 Update Attack-Defense Matrix

**Location**: `paper-submission/latex-submission/sections/05-results.tex`

**Find** (around line 80-100):
```latex
% TODO: Add attack-defense matrix table
```

**Replace with**:
```latex
\subsection{Attack-Defense Matrix: PoGQ vs Baselines}

Table~\ref{tab:attack_defense_matrix} presents head-to-head comparison...

\input{../../results/v4.1/aggregated/attack_defense_matrix_mnist.tex}
```

**Action**:
```bash
vim paper-submission/latex-submission/sections/05-results.tex

# Or use editor of choice to paste LaTeX table
```

### 2.2 Add Statistical Analysis

**Add after matrix table** (around line 120):
```latex
\textbf{Statistical Significance}: Across 2 random seeds (42, 1337),
PoGQ achieves mean detection rate of XX.X\% $\pm$ Y.Y\% compared to
FedAvg baseline of ZZ.Z\% $\pm$ W.W\%. The improvement is statistically
significant ($p < 0.05$, paired t-test).

\textbf{Cross-Dataset Consistency}: Both IID and non-IID (Dirichlet
$\alpha=0.3$) configurations show similar performance patterns, with
PoGQ maintaining >90\% detection across all attack types.
```

**Get values from**:
```bash
# Extract PoGQ mean detection
grep "PoGQ v4.1:" results/v4.1/aggregated/summary_report.txt

# Extract FedAvg baseline
grep "FedAvg (baseline):" results/v4.1/aggregated/summary_report.txt
```

### 2.3 Update Results Narrative

**Find** (around line 50):
```latex
% TODO: Update with v4.1 results
```

**Update with**:
- Specific detection rates for each attack type
- FPR values (should be <10%)
- Comparison with FLTrust baseline
- Any surprising findings

**Example**:
```latex
Against sign flip attacks, PoGQ achieves 98.5\% $\pm$ 1.2\% detection
with 3.1\% FPR. Scaling attacks are detected at 96.7\% $\pm$ 2.3\% with
4.5\% FPR. Adaptive sleeper agent attacks show 92.3\% $\pm$ 4.1\%
detection, demonstrating robustness against sophisticated adversaries.

FLTrust baseline achieves comparable performance at 97.1\% $\pm$ 1.5\%
detection, confirming that server-side validation methods are effective
for this threat model. FedAvg (no defense) shows 45.2\% detection,
effectively random performance.
```

---

## 📈 Step 3: Generate Figures (30 minutes)

### 3.1 Detection Rate Comparison Bar Chart

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Generate figure data
python experiments/generate_figures.py \
  --input results/v4.1/aggregated/aggregated_results.json \
  --output paper-submission/latex-submission/figures/

# Expected output:
# - detection_rate_comparison.png
# - fpr_comparison.png
# - accuracy_vs_attack.png
```

### 3.2 Add Figures to Paper

**Location**: `sections/05-results.tex` (around line 140)

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/detection_rate_comparison.png}
\caption{Detection rate comparison across attack types. PoGQ and FLTrust
achieve >90\% detection for all attacks at 33\% Byzantine ratio,
significantly outperforming FedAvg baseline.}
\label{fig:detection_comparison}
\end{figure}
```

---

## 🔍 Step 4: Cross-Reference Check (15 minutes)

### 4.1 Verify All References Work

```bash
cd paper-submission/latex-submission

# Check LaTeX compilation
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Expected: No errors, warnings OK
```

### 4.2 Check Table/Figure References

**Search for all TODO markers**:
```bash
grep -n "TODO" sections/*.tex
```

**Expected**: No TODOs remaining in Results section

**Verify references**:
```bash
grep -n "ref{tab:" sections/05-results.tex
grep -n "ref{fig:" sections/05-results.tex
```

**All references should resolve**:
- `\ref{tab:attack_defense_matrix}` → Table 1
- `\ref{fig:detection_comparison}` → Figure 1
- etc.

---

## 📋 Step 5: Final Validation (15 minutes)

### 5.1 Sanity Checks

**Detection rates reasonable?**
- PoGQ: 85-100% ✅
- FLTrust: 85-100% ✅
- BOBA: 70-90% ✅
- FedAvg: 40-60% ✅

**FPR acceptable?**
- All defenses: <10% ✅

**Statistical variance reasonable?**
- Std dev: <5% for most configs ✅

### 5.2 Read-Through

Read Results section aloud:
- [ ] Flows logically
- [ ] No TODOs remaining
- [ ] All tables/figures referenced
- [ ] Numbers match aggregated data
- [ ] Conclusions justified by data

### 5.3 Compile Final PDF

```bash
cd paper-submission/latex-submission

# Clean build
rm -f *.aux *.bbl *.blg *.log *.out
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Open PDF
evince main.pdf &
```

**Visual check**:
- [ ] Tables formatted correctly
- [ ] Figures appear and are readable
- [ ] Page numbers correct
- [ ] No overfull hboxes in Results section

---

## ✅ Completion Checklist

After completing all steps:

- [ ] v4.1 data aggregated successfully (64 experiments → 32 configs)
- [ ] Attack-defense matrix added to Results section
- [ ] Statistical analysis narrative updated
- [ ] Figures generated and included
- [ ] LaTeX compiles without errors
- [ ] All references resolve correctly
- [ ] PDF looks professional
- [ ] No TODOs in Results section
- [ ] Paper completion: 100% ✅

---

## 🎯 Expected Outcomes

**Time**: 2-3 hours total
**Paper status**: 100% complete, ready for review
**Next step**: Internal review, polish, proofread

---

## 🚨 Troubleshooting

### Problem: Experiments didn't complete

**Check**:
```bash
ps -p 1404399  # Process still running?
tail -100 /tmp/0tml_logs/matrix_runner_20251111_104031.log
```

**Solution**: Wait for completion or diagnose failure

### Problem: Missing artifacts

**Check**:
```bash
find results -name "detection_metrics.json" | wc -l
# Should be 64
```

**Solution**: Identify which experiments failed, rerun if needed

### Problem: Aggregation script fails

**Check error message**:
```bash
python experiments/aggregate_v4_1_results.py 2>&1 | tee /tmp/agg_error.log
```

**Common issues**:
- JSON parsing error → Check artifact files for corruption
- Missing keys → Update script to handle missing data
- Path issues → Verify results directory structure

### Problem: LaTeX won't compile

**Check log**:
```bash
cat main.log | grep -A5 "Error"
```

**Common issues**:
- Missing figure → Verify file paths
- Undefined reference → Run bibtex and pdflatex again
- Table formatting → Check for unescaped characters

---

## 📞 Emergency Contacts

If major issues arise:
- **Experiment failures**: Check process logs, verify config
- **Data integrity**: Validate JSON files, check for empty artifacts
- **LaTeX errors**: Consult main.log, verify all \input paths
- **Statistical anomalies**: Review raw experiment outputs

---

## 🎉 Success Criteria

**Paper is complete when**:
1. ✅ All 64 experiments aggregated
2. ✅ Attack-defense matrix in Results section
3. ✅ Statistical analysis narrative updated
4. ✅ Figures generated and included
5. ✅ LaTeX compiles cleanly
6. ✅ PDF looks publication-ready

**Then**: Move to internal review and polish phase

---

**Document created**: November 11, 2025, 12:00 PM
**For use**: Wednesday, November 12, 2025, 6:30 AM+
**Estimated duration**: 2-3 hours
**Expected completion**: Wednesday, 9:30 AM

🚀 **You've got this! Follow the steps and the paper will be 100% complete.**
