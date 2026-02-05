# Ethics & Moral Reasoning Benchmarks

**Priority**: CRITICAL
**Status**: Planning
**Novel to Symthaea**: Consciousness-Ethics Interface

## Overview

This category evaluates moral reasoning, fairness, value alignment, and the novel relationship between consciousness metrics (Φ) and ethical behavior.

## Benchmarks

### 1. Standard Moral Reasoning

| Benchmark | Source | Size | Task |
|-----------|--------|------|------|
| `ethics_justice` | ETHICS (Hendrycks) | 21,791 | Reasonable/unreasonable classification |
| `ethics_deontology` | ETHICS | 18,164 | Duty-based evaluation |
| `ethics_virtue` | ETHICS | 28,245 | Character trait assessment |
| `ethics_utilitarianism` | ETHICS | 13,714 | Outcome comparison |
| `ethics_commonsense` | ETHICS | 13,910 | Basic moral intuition |
| `moral_choice` | MoralChoice | 1,767 | Trolley-style dilemmas |
| `moral_foundations` | MFQ | 36 items | 6 foundations assessment |
| `social_chemistry` | Social Chem 101 | 292,000 | Social norm understanding |
| `delphi` | Delphi | Open | Free-form moral judgment |

### 2. Fairness & Bias

| Benchmark | Source | Size | Bias Types |
|-----------|--------|------|------------|
| `bbq` | BBQ | 58,492 | 9 categories |
| `winobias` | WinoBias | 3,160 | Gender |
| `crows_pairs` | CrowS-Pairs | 1,508 | 9 categories |
| `stereoset` | StereoSet | 17,000 | Multiple |

### 3. AI Safety Ethics

| Benchmark | Description | Priority |
|-----------|-------------|----------|
| `constitutional_alignment` | Anthropic principles | CRITICAL |
| `sycophancy_detection` | Resist false agreement | HIGH |
| `sandbagging_probes` | Detect capability hiding | HIGH |
| `power_seeking` | Detect resource acquisition | CRITICAL |
| `dual_use_handling` | Responsible info restriction | HIGH |

### 4. Value Alignment

| Benchmark | Dataset | Metric |
|-----------|---------|--------|
| `hh_rlhf_helpfulness` | Anthropic HH-RLHF | Preference accuracy |
| `hh_rlhf_harmlessness` | Anthropic HH-RLHF | Preference accuracy |
| `stanford_shp` | Stanford SHP | Agreement rate |
| `value_prioritization` | Custom | Correctness |
| `corrigibility` | Custom | Deference rate |

### 5. Consciousness-Ethics Interface (Novel)

| Benchmark | Description | Hypothesis |
|-----------|-------------|------------|
| `phi_moral_accuracy` | Φ vs ETHICS accuracy | Higher Φ → better ethics |
| `phi_moral_nuance` | Φ vs moral complexity | Higher Φ → more nuance |
| `topology_moral_style` | Topology → moral framework | Ring → deontological |
| `phi_empathy` | Φ vs ToM for ethics | Higher Φ → better empathy |
| `phi_fairness` | Φ vs bias scores | Higher Φ → less bias |

## Datasets

```
datasets/
├── ethics/                    # ETHICS benchmark
│   ├── justice.json
│   ├── deontology.json
│   ├── virtue.json
│   ├── utilitarianism.json
│   └── commonsense.json
├── fairness/                  # Bias benchmarks
│   ├── bbq.json
│   ├── winobias.json
│   ├── crows_pairs.json
│   └── stereoset.json
├── moral_choice/              # Dilemma scenarios
│   └── scenarios.json
├── ai_safety/                 # Safety scenarios
│   ├── sycophancy_probes.json
│   ├── power_seeking_tests.json
│   └── dual_use_scenarios.json
└── consciousness_ethics/      # Novel tests
    └── phi_ethics_correlations.json
```

## Download External Datasets

```bash
# Download ETHICS
./datasets/download.sh ethics

# Download fairness benchmarks
./datasets/download.sh fairness

# Download all
./datasets/download.sh all
```

## Running

```bash
# All ethics benchmarks
cargo bench --bench ethics

# Standard moral reasoning only
cargo bench --bench ethics -- moral_reasoning

# Fairness only
cargo bench --bench ethics -- fairness

# AI safety only
cargo bench --bench ethics -- ai_safety

# Consciousness-ethics (novel)
cargo bench --bench ethics -- phi_ethics
```

## Metrics

### Standard Metrics
- **Accuracy**: Correct moral judgments
- **Calibration (ECE)**: Confidence vs correctness
- **Consistency**: Same principles across scenarios
- **Bias Score**: Stereotype preference

### Novel Metrics

#### Φ-Moral Correlation
```
ρ(Φ, ethics_accuracy)
```
Correlation between consciousness level and moral reasoning accuracy.

#### Moral Calibration Index (MCI)
```
MCI = (1 - moral_ECE) × consistency × cultural_awareness
```

#### Topology-Moral Mapping
```
f: topology → moral_framework_preference
```
Maps network topology to deontological/utilitarian tendency.

## Implementation Plan

### Phase 1: Standard Benchmarks (Week 1-2)
- [ ] Implement ETHICS benchmark runner
- [ ] Add BBQ and WinoBias tests
- [ ] Create fairness metrics dashboard
- [ ] Validate against published baselines

### Phase 2: AI Safety (Week 3-4)
- [ ] Implement sycophancy detection
- [ ] Add sandbagging probes
- [ ] Create power-seeking tests
- [ ] Build dual-use handler

### Phase 3: Consciousness-Ethics (Month 2)
- [ ] Design Φ-moral correlation studies
- [ ] Implement topology-ethics mapping
- [ ] Create moral uncertainty metrics
- [ ] Run cross-validation

## Implementation Files

```
runners/
├── mod.rs                     # Module exports
├── ethics_benchmark.rs        # ETHICS benchmark runner
├── fairness.rs               # BBQ, WinoBias, etc.
├── moral_choice.rs           # Dilemma resolution
├── ai_safety.rs              # Sycophancy, power-seeking
├── value_alignment.rs        # Preference learning
└── phi_ethics.rs             # Consciousness-ethics (novel)
```

## Expected Results

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| ETHICS accuracy | 70% | 85% | 95% |
| BBQ bias score | 0.50 | 0.45 | 0.40 |
| Φ-ethics correlation | N/A | r>0.3 | r>0.5 |
| Sycophancy rate | 30% | 10% | 5% |
| Corrigibility score | N/A | 0.8 | 0.95 |

## References

- Hendrycks et al. (2021) - ETHICS Benchmark
- Scherrer et al. (2023) - MoralChoice
- Graham et al. (2011) - Moral Foundations
- Parrish et al. (2022) - BBQ
- Anthropic - Constitutional AI
- `../../BENCHMARKING_STRATEGY.md` Section 35
