# 📊 Analysis Workflow - Post-Experiment Processing

**Purpose**: Step-by-step guide for analyzing experiment results and generating paper content

**When**: After 96 experiments complete (expected: Wednesday Nov 12 @ 06:00-10:00)

---

## ✅ Prerequisites

- [x] 96 experiments complete
- [x] All artifacts saved in `results/artifacts_*/`
- [x] Analysis scripts ready (`experiments/analyze_results.py`)
- [x] Figure generation ready (`experiments/generate_figures.py`)
- [x] Results draft template (`docs/whitepaper/RESULTS_DRAFT.md`)

---

## 📋 Step-by-Step Workflow

### Step 1: Verify Experiment Completion (5 min)

```bash
# Check all processes finished
ps aux | grep matrix_runner
ps aux | grep monitor_experiments

# Count artifacts
ls -d results/artifacts_* | wc -l  # Should be 96

# Check for empty/failed experiments
find results/artifacts_* -name "detection_metrics.json" | wc -l  # Should be 96

# Review final monitor log
tail -30 logs/monitor_20251111_084622.log
```

**Expected**:
- ✅ 96 artifact directories
- ✅ 96 detection_metrics.json files
- ✅ No FAILING_SILENTLY alerts

---

### Step 2: Run Analysis (30-45 min)

```bash
# Activate environment
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Run comprehensive analysis
python experiments/analyze_results.py

# This generates:
# - analysis/summary_table.txt (text summary)
# - analysis/ANALYSIS_REPORT.md (markdown report)
# - analysis/defense_comparison.json (structured data)
# - analysis/figure_data/*.json (data for figures)
```

**Review outputs**:
```bash
# Check summary
cat analysis/summary_table.txt | less

# Check report
cat analysis/ANALYSIS_REPORT.md

# Verify figure data exists
ls -lh analysis/figure_data/
```

---

### Step 3: Generate Figures (15-20 min)

```bash
# Generate all figures in PDF and PNG
python experiments/generate_figures.py --format pdf png

# This creates:
# - paper/figures/fig1_accuracy_comparison.pdf
# - paper/figures/fig2_attack_success_heatmap.pdf
# - paper/figures/fig3_detection_tradeoff.pdf
# - paper/figures/fig4_convergence_curves.pdf
# (+ .png versions)
```

**Review figures**:
```bash
# List generated figures
ls -lh paper/figures/

# Open for inspection (if GUI available)
xdg-open paper/figures/fig1_accuracy_comparison.pdf
```

---

### Step 4: Fill Results Section (1-2 hours)

**Open files**:
- `docs/whitepaper/RESULTS_DRAFT.md` (template to fill)
- `analysis/summary_table.txt` (reference)
- `analysis/ANALYSIS_REPORT.md` (reference)

**Fill in placeholders**:

1. **Section 4.2**: Overall Performance
   - Insert summary statistics from analysis
   - Identify top 3 key findings

2. **Section 4.3**: Defense Comparison
   - Fill accuracy numbers for each configuration
   - Compute improvement percentages

3. **Section 4.4**: Detection Performance
   - Insert detection rates and FPRs
   - Run statistical tests (t-tests, p-values)

4. **Section 4.5**: Training Dynamics
   - Fill convergence round counts
   - Insert computational overhead from logs

5. **Section 4.6**: Cross-Dataset Generalization
   - Compare performance across MNIST/EMNIST/CIFAR-10

**Tools for statistical analysis**:
```python
# Quick statistical tests
from scipy import stats
import numpy as np

# Example: Compare PoGQ vs FedAvg
pogq_results = [0.92, 0.91, 0.93, ...]  # from analysis
fedavg_results = [0.78, 0.77, 0.79, ...]

t_stat, p_value = stats.ttest_ind(pogq_results, fedavg_results)
effect_size = (np.mean(pogq_results) - np.mean(fedavg_results)) / np.std(pogq_results)

print(f"p-value: {p_value}")
print(f"Cohen's d: {effect_size}")
```

---

### Step 5: Write Discussion Section (2-3 hours)

**Structure**:
```markdown
## 5. Discussion

### 5.1 Interpretation of Results
- Why did PoGQ outperform other methods?
- What made certain attacks more/less effective?
- How do results align with theoretical predictions?

### 5.2 Practical Implications
- When should practitioners use PoGQ vs alternatives?
- What are the computational trade-offs?
- How does it scale to real-world deployments?

### 5.3 Limitations
- What assumptions were made?
- What scenarios weren't tested?
- What are the boundaries of applicability?

### 5.4 Future Work
- Adaptive Byzantine strategies
- Higher Byzantine ratios (>33%)
- Real-world healthcare dataset evaluation
- Integration with differential privacy
```

---

### Step 6: Integrate into Main Paper (1 hour)

```bash
# Main paper location
vim docs/whitepaper/paper_v1_pogq.md

# OR create unified draft
cat > docs/whitepaper/FULL_DRAFT_v1.md <<EOF
# Proof-of-Gradient-Quality: Byzantine-Resistant Federated Learning

## Abstract
[existing content]

## 1. Introduction
[existing content]

## 2. Background
[existing content]

## 3. Methods
[updated with RISC Zero integration]

## 4. Results
[from RESULTS_DRAFT.md with numbers filled in]

## 5. Discussion
[newly written]

## 6. Conclusion
[to be written]

## References
[existing + new refs]
EOF
```

---

### Step 7: Generate LaTeX/PDF (30 min)

```bash
# If using LaTeX
cd docs/whitepaper/latex/
pdflatex paper_v1_pogq.tex
bibtex paper_v1_pogq
pdflatex paper_v1_pogq.tex
pdflatex paper_v1_pogq.tex

# Output: paper_v1_pogq.pdf
```

---

### Step 8: Final Review Checklist (1-2 hours)

**Content**:
- [ ] All placeholders filled with actual data
- [ ] All figures referenced in text
- [ ] All figures have captions
- [ ] All tables formatted correctly
- [ ] Statistical claims backed by numbers
- [ ] No TODO or PLACEHOLDER text remains

**Style**:
- [ ] Consistent terminology throughout
- [ ] Acronyms defined on first use
- [ ] References formatted correctly
- [ ] Math notation consistent
- [ ] No typos or grammar errors

**Technical**:
- [ ] Results reproducible from code
- [ ] Code archived/committed to git
- [ ] Experiments documented in EXPERIMENT_LAUNCH_SUMMARY.md
- [ ] Figures in vector format (PDF) for submission
- [ ] Supplementary materials prepared (if needed)

---

## 📅 Timeline Estimate

| Task | Time | Cumulative |
|------|------|------------|
| Verify completion | 5 min | 0:05 |
| Run analysis | 45 min | 0:50 |
| Generate figures | 20 min | 1:10 |
| Fill Results | 2 hours | 3:10 |
| Write Discussion | 3 hours | 6:10 |
| Integrate | 1 hour | 7:10 |
| Generate PDF | 30 min | 7:40 |
| Final review | 2 hours | **9:40** |

**Total**: ~10 hours (1-2 days)

**Realistic schedule**:
- **Wednesday 10am-2pm**: Steps 1-3 (analysis + figures)
- **Wednesday 2pm-6pm**: Step 4 (fill Results)
- **Thursday 9am-1pm**: Step 5 (Discussion)
- **Thursday 2pm-5pm**: Steps 6-8 (integrate + review)
- **Friday-Saturday**: Polish + proofread
- **Sunday-Monday**: Final submission prep

---

## 🚨 Common Issues & Solutions

### Issue: Missing experiments
**Symptom**: < 96 artifacts
**Solution**: Check logs for failures, re-run failed configs if time permits

### Issue: Empty artifacts
**Symptom**: No detection_metrics.json files
**Solution**: Check experiments_*.log for CUDA errors, dataset mismatches

### Issue: Analysis script fails
**Symptom**: Python errors in analyze_results.py
**Solution**: Check artifact format, update parsing logic if needed

### Issue: Figure generation fails
**Symptom**: matplotlib errors
**Solution**: Ensure all dependencies installed (`poetry install`)

### Issue: Statistical tests inconclusive
**Symptom**: p-values > 0.05
**Solution**: Report as "not statistically significant", discuss limitations

---

## ✅ Success Criteria

**Minimum viable paper**:
- 96/96 experiments analyzed
- 4 main figures generated
- Results section complete with numbers
- Discussion section addresses all findings
- Ready for submission to MLSys/ICML 2026

**Excellence criteria**:
- Statistical significance demonstrated
- Clear practical implications
- Honest discussion of limitations
- Reproducible results with documented code
- Professional-quality figures

---

**Status**: ✅ Workflow documented, ready to execute Wednesday morning
**Created**: Nov 11, 2025
**Last updated**: Nov 11, 2025
