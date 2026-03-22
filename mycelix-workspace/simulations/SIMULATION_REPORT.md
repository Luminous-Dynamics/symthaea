# Simulation Suite Report

Generated from parameter sweep results across 10 seeds.

## Commons Resource Sustainability

Tests Ostrom's design principles for commons governance with consciousness-gated access.
Calibrated against Spanish Huerta irrigation system (Ostrom 1990, Ch. 3).

### Model
- 100 agents, 10 resources, 365 days
- 10 agent strategies: Steward, Moderate, Cooperator, Altruist, Seasonal, StrategicCooperator, CaptureSeeker, Defector, FreeRider, Overextractor
- Sustainable yield extraction: quotas computed from regeneration budget
- Small-world network topology (4 nearest + 2 random neighbors)
- 5 resource types with agent specialization

### Key Finding 1: Governance Mode Comparison

| Defector % | Consciousness | Flat | Plutocratic | Random | No Gov |
|-----------|--------------|------|-------------|--------|--------|
| 10% | 0.904 +/- 0.014 | 0.941 +/- 0.010 | 0.941 +/- 0.010 | 0.941 +/- 0.010 | 0.000 +/- 0.000 |
| 20% | 0.917 +/- 0.017 | 0.949 +/- 0.011 | 0.949 +/- 0.011 | 0.949 +/- 0.011 | 0.000 +/- 0.000 |
| 30% | 0.928 +/- 0.015 | 0.941 +/- 0.035 | 0.958 +/- 0.009 | 0.958 +/- 0.009 | 0.000 +/- 0.000 |
| 40% | 0.937 +/- 0.016 | 0.917 +/- 0.060 | 0.963 +/- 0.009 | 0.963 +/- 0.009 | 0.000 +/- 0.000 |
| 50% | 0.949 +/- 0.017 | 0.895 +/- 0.057 | 0.972 +/- 0.006 | 0.973 +/- 0.006 | 0.000 +/- 0.000 |
| 60% | 0.964 +/- 0.008 | 0.881 +/- 0.004 | 0.970 +/- 0.029 | 0.979 +/- 0.005 | 0.000 +/- 0.000 |
| 70% | 0.968 +/- 0.011 | 0.875 +/- 0.029 | 0.943 +/- 0.049 | 0.982 +/- 0.005 | 0.000 +/- 0.000 |
| 80% | 0.928 +/- 0.049 | 0.879 +/- 0.028 | 0.898 +/- 0.052 | 0.987 +/- 0.003 | 0.000 +/- 0.000 |

![Voting Comparison](figures/fig_commons_voting_comparison.png)

### Key Finding 2: Cross-Simulation Consciousness Degradation

- Degradation 0.0/day: sustainability = 0.928 +/- 0.015 (n=10)
- Degradation 0.001/day: sustainability = 0.941 +/- 0.013 (n=10)
- Degradation 0.002/day: sustainability = 0.960 +/- 0.008 (n=10)
- Degradation 0.005/day: sustainability = 0.979 +/- 0.004 (n=10)
- Degradation 0.01/day: sustainability = 0.980 +/- 0.004 (n=10)

![Degradation Impact](figures/fig_commons_degradation.png)

## Multi-Currency Macro Economy

Triple-currency system (SAP/TEND/MYCEL) with demurrage, counter-cyclical credit, and consciousness-gated tiers.

### Key Finding 3: Demurrage Rate Has No Significant Effect on Velocity

Within the constitutional bounds (1-5%), demurrage rate does not significantly
affect SAP transaction velocity. The exempt floor (1,000 SAP) dominates — most
agents hold near or below the floor, making the rate irrelevant for circulation.

- Demurrage 1%: velocity = 1.6568 +/- 0.0552 tx/agent/day
- Demurrage 2%: velocity = 1.6568 +/- 0.0552 tx/agent/day
- Demurrage 3%: velocity = 1.6504 +/- 0.0594 tx/agent/day
- Demurrage 4%: velocity = 1.6504 +/- 0.0594 tx/agent/day
- Demurrage 5%: velocity = 1.6504 +/- 0.0594 tx/agent/day

**Implication**: The demurrage rate within constitutional bounds is a political parameter,
not an economic lever. The exempt floor is the binding constraint.

![Demurrage vs Velocity](figures/fig_macro_demurrage.png)

### Key Finding 4: Jubilee Frequency and Reputation Inequality

- Never jubilee: MYCEL Gini = 0.0155 +/- 0.0002
- 2-year jubilee: MYCEL Gini = 0.0155 +/- 0.0002
- 4-year jubilee: MYCEL Gini = 0.0155 +/- 0.0002
- 8-year jubilee: MYCEL Gini = 0.0155 +/- 0.0002

![Jubilee Impact](figures/fig_macro_jubilee.png)

### Key Finding 5: Inactivity and Wealth Concentration

- Inactivity 0.0x: SAP Gini = 0.6992 +/- 0.0073
- Inactivity 0.5x: SAP Gini = 0.6975 +/- 0.0080
- Inactivity 1.0x: SAP Gini = 0.6967 +/- 0.0078
- Inactivity 1.5x: SAP Gini = 0.6950 +/- 0.0075
- Inactivity 2.0x: SAP Gini = 0.6938 +/- 0.0074
- Inactivity 3.0x: SAP Gini = 0.6921 +/- 0.0082

![Inactivity vs Gini](figures/fig_macro_sweep.png)

## Methodology

- **Seeds**: 10 independent runs per configuration (42, 123, 789, 1337, 2024, 3141, 4242, 5555, 6789, 9876)
- **Error bars**: Mean +/- 1 standard deviation
- **Calibration**: Commons parameters calibrated against Ostrom (1990) Spanish Huerta system
- **Extraction model**: Sustainable yield — cooperative quotas derived from logistic regeneration
- **Network**: Watts-Strogatz small-world (k=4 nearest + 2 random long-range)
- **Voting modes**: Consciousness-gated (sigmoid weight), Flat (equal), Plutocratic (extraction-weighted), Random (probabilistic exclusion)

## Data

- **623** sweep CSV files
- **14** publication figures (PNG + PDF)
- **10** random seeds for statistical robustness
