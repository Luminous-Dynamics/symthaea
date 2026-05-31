# F1: Aggregation Effectiveness Study Report

**Experiment ID**: F1_Aggregation_Effectiveness
**Date**: 2026-01-30
**Duration**: 109,943 ms (~1.8 minutes)
**Research Infrastructure**: Epistemic Markets Research Module v0.1.0

---

## Executive Summary

This study investigated the foundational research question: **"Under what conditions does crowd aggregation outperform expert prediction?"**

### Key Findings

1. **Median aggregation emerged as the most effective method** with an average Brier score of 0.2021, slightly outperforming SimpleMean (0.2070), Extremized (0.2044), and BrierWeighted (0.2070).

2. **Aggregation consistently outperforms individual predictions** with wisdom ratios ranging from 1.09 to 1.35 across all conditions tested. This demonstrates a robust "wisdom of crowds" effect.

3. **Experts outperformed crowd aggregates** in this simulation, achieving Brier scores around 0.197 compared to crowd aggregates around 0.202. This suggests the expert model's calibration (10% noise) was superior to crowd conditions tested.

4. **Correlation is the strongest predictor of aggregation degradation**: As prediction correlation increased from 0.0 to 0.6, the wisdom ratio dropped from 1.35 to 1.09.

5. **Scale effects are modest**: Increasing crowd size from 10 to 500 improved Brier scores from 0.2075 to 0.1994 (only 4% improvement), suggesting diminishing returns beyond ~50 agents.

---

## Methodology

### Experimental Design

| Parameter | Values Tested |
|-----------|---------------|
| Agent Counts | 10, 50, 100, 500 |
| Diversity Levels | 0.2 (Low), 0.5 (Medium), 0.8 (High) |
| Correlation Levels | 0.0 (Independent), 0.3 (Moderate), 0.6 (High) |
| Aggregation Methods | SimpleMean, Median, BrierWeighted, Extremized |

### Simulation Parameters

- **Iterations per condition**: 100
- **Questions per iteration**: 50
- **Total predictions analyzed**: 180,000 per condition
- **Expert baseline**: 5 experts with 10% prediction noise
- **Random seed**: 42 (for reproducibility)

### Metrics

| Metric | Description |
|--------|-------------|
| **Brier Score** | Mean squared error: (prediction - outcome)^2. Lower is better (0 = perfect). |
| **Calibration** | How well predicted probabilities match actual frequencies. |
| **Resolution** | Ability to discriminate between outcomes. |
| **Wisdom Ratio** | Individual Brier / Aggregate Brier. Values >1.0 indicate aggregation benefit. |
| **Beat Rate** | Percentage of trials where aggregate outperformed expert. |

---

## Results

### 1. Aggregation Method Performance

| Method | Avg Brier Score | Avg Wisdom Ratio | Avg Beat Rate |
|--------|-----------------|------------------|---------------|
| **Median** | 0.2021 | 1.21 | 36.3% |
| **Extremized** | 0.2044 | 1.20 | 38.2% |
| **SimpleMean** | 0.2070 | 1.18 | 33.6% |
| **BrierWeighted** | 0.2070 | 1.18 | 33.6% |

**Analysis**: Median aggregation's robustness to outliers provides a consistent edge. The Extremized method shows the highest beat rate (38.2%), suggesting it performs better in head-to-head comparisons even if its average Brier is slightly higher.

### 2. Effect of Crowd Size

| Agent Count | Avg Brier (Median) | Wisdom Ratio | Improvement vs. 10 agents |
|-------------|--------------------|--------------|-----------------------------|
| 10 | 0.2075 | 1.18 | Baseline |
| 50 | 0.2003 | 1.22 | +3.5% |
| 100 | 0.2013 | 1.22 | +3.0% |
| 500 | 0.1994 | 1.23 | +3.9% |

**Analysis**: Diminishing returns are evident beyond 50 agents. The marginal improvement from 100 to 500 agents (0.0019 Brier reduction) is minimal. This suggests **50-100 agents is the optimal range** for practical deployments balancing accuracy with operational complexity.

### 3. Effect of Diversity

| Diversity Level | Avg Brier (Median) | Wisdom Ratio |
|-----------------|--------------------|--------------|
| 0.2 (Low) | 0.2010 | 1.20 |
| 0.5 (Medium) | 0.2020 | 1.21 |
| 0.8 (High) | 0.2033 | 1.22 |

**Analysis**: Counter-intuitively, lower diversity produced slightly better Brier scores in this simulation. This occurs because:
1. High diversity introduces more extreme opinions, which can pull aggregates away from truth
2. The simulation's uniform weighting doesn't penalize poorly-calibrated diverse agents
3. When agents have similar calibration quality, less diversity means more consistent predictions

**Implication**: Diversity is beneficial only when it brings genuinely independent information. "Diversity for diversity's sake" can be counterproductive.

### 4. Effect of Correlation (Critical Finding)

| Correlation Level | Crowd Brier | Expert Brier | Wisdom Ratio | Crowd vs Expert |
|-------------------|-------------|--------------|--------------|-----------------|
| 0.0 (Independent) | 0.2020 | 0.1961 | **1.35** | Expert wins |
| 0.3 (Moderate) | 0.2018 | 0.1976 | 1.19 | Expert wins |
| 0.6 (High) | 0.2026 | 0.1977 | **1.09** | Expert wins |

**Analysis**: This is the study's most significant finding. **Correlation is the primary threat to aggregation effectiveness.** When predictions become correlated (common information sources, herding behavior, shared biases), the wisdom ratio drops precipitously.

At correlation=0.6, the wisdom ratio of 1.09 means aggregation provides only 9% improvement over individual predictions, compared to 35% at independence.

### 5. Expert vs. Aggregate Comparison

| Metric | Value |
|--------|-------|
| Expert win rate | 100% (36/36 conditions) |
| Aggregate win rate | 0% |
| Average expert advantage | 0.005 Brier points |

**Why Experts Won**: In this simulation, experts were modeled with 10% noise on true probability, while crowd agents had:
- Variable information quality (10-90%)
- Calibration errors (up to 35% in high-diversity conditions)
- Correlation-induced bias

This represents a scenario where experts have access to better base information. The study confirms that **crowds cannot overcome fundamental information asymmetry** through aggregation alone.

---

## Detailed Condition Analysis

### Best Performing Conditions (Lowest Aggregate Brier)

| Rank | Agents | Diversity | Correlation | Brier (Median) | Wisdom Ratio |
|------|--------|-----------|-------------|----------------|--------------|
| 1 | 500 | 0.2 | 0.0 | 0.1958 | 1.35 |
| 2 | 500 | 0.2 | 0.3 | 0.1982 | 1.20 |
| 3 | 100 | 0.2 | 0.0 | 0.1989 | 1.34 |
| 4 | 500 | 0.5 | 0.0 | 0.1991 | 1.35 |
| 5 | 50 | 0.2 | 0.0 | 0.1994 | 1.33 |

**Pattern**: Large crowds + Low diversity + Independent predictions = Best performance

### Worst Performing Conditions (Highest Aggregate Brier)

| Rank | Agents | Diversity | Correlation | Brier (Median) | Wisdom Ratio |
|------|--------|-----------|-------------|----------------|--------------|
| 1 | 10 | 0.8 | 0.6 | 0.2115 | 1.07 |
| 2 | 10 | 0.5 | 0.6 | 0.2087 | 1.08 |
| 3 | 10 | 0.2 | 0.6 | 0.2064 | 1.08 |
| 4 | 10 | 0.8 | 0.3 | 0.2063 | 1.17 |
| 5 | 50 | 0.8 | 0.6 | 0.2059 | 1.08 |

**Pattern**: Small crowds + High correlation = Worst performance

---

## Method-Specific Insights

### Median
- **Strengths**: Most robust to outliers; consistently lowest Brier across conditions
- **Weaknesses**: May discard valuable extreme information when calibrated forecasters are confident
- **Best Use Case**: Unknown forecaster quality, potential for extreme outliers

### Extremized
- **Strengths**: Highest beat rate (38.2%); counteracts regression to mean
- **Weaknesses**: Can amplify errors when crowd consensus is wrong
- **Best Use Case**: High-diversity crowds with demonstrated calibration

### SimpleMean / BrierWeighted
- **Strengths**: Simple, interpretable, efficient
- **Weaknesses**: Vulnerable to extreme predictions
- **Best Use Case**: Pre-screened forecasters with similar track records

---

## Implications for System Design

### Recommended Parameters

Based on this study, the following parameters are recommended for the Epistemic Markets system:

| Parameter | Recommendation | Rationale |
|-----------|----------------|-----------|
| **Minimum crowd size** | 50 agents | Captures most aggregation benefit with minimal overhead |
| **Target diversity** | 0.3-0.5 | Balanced information diversity without excessive noise |
| **Correlation monitoring** | Alert at 0.3+ | Early warning of degraded aggregation effectiveness |
| **Default aggregation** | Median | Most robust across conditions |
| **High-diversity override** | Extremized | When diversity >0.6 and forecasters are calibrated |

### Warning Conditions

The system should flag or adjust behavior when:

1. **Correlation > 0.5**: Consider falling back to expert judgment or applying correlation correction
2. **Crowd size < 20**: Aggregate predictions have high variance; increase uncertainty bounds
3. **Diversity > 0.8 with unknown calibration**: Risk of extreme outliers; use Median

### Hybrid Expert-Crowd Strategy

Given that experts outperformed crowds in this study, a hybrid approach is recommended:

1. Use crowd aggregation as the **primary signal** when correlation < 0.3 and crowd size >= 50
2. Weight expert opinion more heavily when correlation > 0.5 or domain is narrow
3. Track expert vs. crowd performance over time to calibrate the blend ratio

---

## Limitations and Future Work

### Study Limitations

1. **Simulated agents**: Real human forecasters have more complex behavior patterns
2. **Fixed expert quality**: Experts modeled with uniform 10% noise; real expert quality varies
3. **Binary outcomes**: Study used probabilistic binary outcomes; multi-outcome questions may differ
4. **No learning**: Agents did not improve over time; real systems show adaptation

### Recommended Follow-up Studies

1. **F1b: Real forecaster data validation** - Apply these methods to historical forecasting tournament data
2. **F1c: Dynamic correlation detection** - Test algorithms for detecting emerging correlation in real-time
3. **F1d: Expert identification** - Develop methods to identify and weight domain experts within crowds

---

## Conclusion

This study establishes that crowd aggregation provides meaningful improvement over individual predictions (wisdom ratios of 1.09-1.35), but with important caveats:

1. **Independence is essential**: The aggregation benefit degrades rapidly with correlation
2. **Diminishing returns from scale**: 50-100 forecasters captures most benefit
3. **Method matters**: Median outperforms mean-based methods
4. **Experts can win**: When experts have superior information, crowds cannot bridge the gap through aggregation alone

For the Epistemic Markets system, this suggests a **conditional aggregation strategy**: use crowd wisdom when conditions are favorable (low correlation, sufficient diversity of information sources), but maintain expert input channels for scenarios where crowd conditions deteriorate.

---

## Appendix: Raw Data Summary

```
Total conditions tested: 36
Total predictions per method: 180,000 per condition
Total simulation runs: 3,600

Parameter ranges:
  Agent counts: [10, 50, 100, 500]
  Diversity: [0.2, 0.5, 0.8]
  Correlation: [0.0, 0.3, 0.6]

Output files:
  - f1_aggregation_results.json (detailed results)
  - f1_aggregation_report.md (this report)
```

---

*Report generated by Epistemic Markets Research Infrastructure*
*Experiment timestamp: 2026-01-30T09:50:01Z*
