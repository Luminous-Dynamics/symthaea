# Psych-Bench Calibration Transparency Report

**Date**: 2026-03-17
**Grand mean z**: +1.044 (22 domains)
**Purpose**: Document which benchmark parameters were set a priori vs. tuned post-hoc to match human baselines.

## Classification System

| Label | Definition |
|-------|-----------|
| **A PRIORI** | Value derived from task structure, literature, or mathematical principle *before* seeing benchmark output |
| **LITERATURE** | Value from a cited paper, though adapted to HDC implementation |
| **POST-HOC** | Value was iterated/calibrated specifically to produce output matching the human baseline target |
| **THEORETICAL** | No human baseline exists; comparison target is a theoretical model prediction |

## Key Finding

**~60% of benchmarks contain at least one post-hoc calibrated parameter.** The grand mean z-score is substantially dependent on these calibration choices. Parameters marked POST-HOC could be set to different values that would produce different z-scores while remaining equally "principled."

This does NOT mean the benchmarks are invalid — calibrating a computational model to reproduce human behavior is standard practice in computational cognitive science (e.g., ACT-R parameter fitting). But it means z-scores should be interpreted as "the model CAN reproduce human-like behavior with appropriate parameters" rather than "the model INHERENTLY behaves like humans."

## Per-Benchmark Audit

### Executive Function

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| Stroop | `base_automaticity` | 0.35 | **POST-HOC** | Comment: "calibrated to produce ~10% Stroop effect" |
| Stroop | `temperature` | 0.25 | **POST-HOC** | "base 0.25 matches ~10% interference" |
| Flanker | `attention_leak` | 0.35 | AMBIGUOUS | Encodes flanker influence; comment references 10% effect |
| Flanker | `temperature` | 0.25 | **POST-HOC** | "base 0.25 matches ~10% flanker interference" |
| WCST | — | — | A PRIORI | Rule-switching is structurally determined |
| IGT | — | — | LITERATURE | Deck payoffs from Bechara (1994) |
| TowerOfLondon | — | — | A PRIORI | Planning depth is structural |
| Ravens | — | — | A PRIORI | Pattern matching via HDC similarity |
| DualTask | — | — | A PRIORI | Resource sharing model |

### Inhibition

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| GoNoGo | `go_threshold` | 0.12 | **POST-HOC** | No citation for these values |
| GoNoGo | `nogo_threshold` | 0.32 | **POST-HOC** | Asymmetric thresholds produce target accuracy split |
| StopSignal | `stop_effectiveness` | 0.70 | **POST-HOC** | Chosen to yield 50% stop accuracy |

### Attention

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| AttentionalBlink | `temperature` | 0.30 | **POST-HOC** | "base 0.30 matches ~50% T2 accuracy at lag-2" |
| AttentionalBlink | `t1_cost` | 0.75 | **POST-HOC** | Tuned for blink magnitude |
| VisualSearch | serial cost | 0.75 | **POST-HOC** | Cites Wolfe 1994 (~0.5) but uses 0.75 |
| MismatchNegativity | `detection_threshold` | 0.11 | **POST-HOC** | Threshold for deviant detection |

### Affect

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| EmotionalStroop | `emotional_weight` | 0.28 | **POST-HOC** | Tuned Mar 17 to reduce interference toward 0.070 |
| EmotionalStroop | `temperature` | 0.25 | **POST-HOC** | "base 0.25 yields ~10% interference" |
| MoodCongruent | `valence_w` | 0.15 | LITERATURE | Cites Blaney (1986) |
| ValenceClassification | weights | 0.4/0.6 | A PRIORI | 60/40 split, no calibration comments |

### Binding

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| TemporalOrder | normalization | 0.14 | **POST-HOC** | "Normalizing by 0.14 maps this to ~0.71, matching human baseline (0.70)" |
| CrossModal | `noise_frac` | 0.01 | A PRIORI | No explicit calibration |

### Consciousness

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| BinocularRivalry | `adaptation_rate` | 0.02 | **POST-HOC** | Tuned Mar 17 for longer dominance bouts |
| BinocularRivalry | switch threshold | 0.07 | **POST-HOC** | Tuned Mar 17 for dominance_ratio ~0.55 |
| Blindsight | `gw_threshold` | 0.45 | **POST-HOC** | |

### Metacognition

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| Calibration | `noise_range` | 0.22 | **POST-HOC** | "calibrated to ECE ~0.14" |
| Calibration | cue weights | [0.36,0.46,0.08,0.10] | **POST-HOC** | "Very low gap weight prevents boosting gamma above human range" |
| FeelingOfKnowing | — | — | A PRIORI | |
| ChangeBlindness | — | — | A PRIORI | |

### WorM (Working Memory)

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| SerialRecall | `rehearsal_boost` | 1.12 | **POST-HOC** | "calibrated so items at positions 0-1 produce primacy near 0.15" |
| N-back | `base_threshold` | 0.50 | A PRIORI | Midpoint of similarity range |
| DigitSpan | thresholds | 0.5/0.6 | A PRIORI | Forward < backward structurally |
| ChangeDetection | encoding noise | 0.03*set_size | LITERATURE | Cites Bays & Husain 2008 |
| SpatialUpdating | threshold | 0.3 | AMBIGUOUS | "base 0.3 yields ~75% recall" |
| Binding | — | — | A PRIORI | |

### CogBench

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| Probabilistic | `lr` | 0.15 | LITERATURE | Cites Behrens et al. 2007 |
| RestlessBandit | `ucb_scale` | 0.15 | A PRIORI | Standard UCB |
| BART | — | — | LITERATURE | Lejuez 2002 |
| HorizonTask | — | — | LITERATURE | Wilson 2014 |
| Instrumental | — | — | LITERATURE | Daw 2011 |
| ReversalLearning | `learning_rate` | 0.40 | AMBIGUOUS | Asymmetric learning is literature-supported |
| TwoStep | mb_weight ramp | 0.3→0.95 | **POST-HOC** | Ramp to 0.95 by episode 15 |
| TemporalDiscounting | — | — | LITERATURE | Kirby MCQ stimuli |

### Reasoning (ARC)

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| ArcFluid | `noise_weight` base | 0.008 | **POST-HOC** | Tuned Mar 17 (was 0.015) |
| ArcFluid | training pairs | 4 | **POST-HOC** | Increased Mar 17 (was 2) for higher transfer accuracy |
| ArcAbductive | `noise_weight` base | 0.05 | **POST-HOC** | Higher base for harder task |
| All ARC | BinaryGridEncoder | — | A PRIORI | XOR self-inverse is mathematically principled |

### Mathematics

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| ArithmeticWordProblems | similarity offset | +0.76 | **POST-HOC** | Magic number shifting raw HDC similarity |
| BayesianReasoning | blend weights | [0.5,0.3,0.2] | **POST-HOC** | |
| Others | — | — | MIXED | HDC encodes, thresholds tuned to match |

### Social

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| RME | `social_bonus` | 0.25 | **POST-HOC** | |
| RME | `signal_weight` | 0.65/0.40 | **POST-HOC** | Easy/hard tuned for target accuracies |
| UltimatumGame | `social_bonus` | 0.20 | **POST-HOC** | |
| All game theory | — | — | MIXED | Social bonus pattern throughout |

### Motor

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| SRTT | `pred_speedup` | ×3.0 | **POST-HOC** | Multiplier for measurable learning |
| Fitts | `slope` | 1.2 | **POST-HOC** | Determines R² |

### Sustained Attention

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| PVT | `noise_base` | 0.8 | **POST-HOC** | Chosen to produce lapse_rate ~0.05 |
| CPT | `threshold_increment` | 0.005 | **POST-HOC** | Chosen for vigilance_decrement ~0.03 |

### Speech & Language

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| PhonemeDiscrimination | categorical_diff | 0.3 | **POST-HOC** | For cross-boundary=0.90 |
| SemanticPriming | decay | 0.85 | **POST-HOC** | |
| GardenPath | reparse_cost_scale | 0.8 | AMBIGUOUS | Config parameter |

### Substrate

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| Transfer | — | — | **THEORETICAL** | No human data; IIT/GWT derived |
| Degradation | — | — | **THEORETICAL** | No human data; IIT/GWT derived |

### Institutional Reasoning

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| All | — | — | **THEORETICAL** | "Baselines" are system's own behavior, not human data |

### Creativity

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| AlternateUses | `max_attempts` | 60 | **POST-HOC** | "~60 attempts yields fluency near human mean" |
| RemoteAssociates | — | — | A PRIORI | Accuracy falls where HDC places it |

### ToMBench

| Benchmark | Parameter | Value | Classification | Notes |
|-----------|-----------|-------|---------------|-------|
| FalseBelief | blend | 0.8/0.2 | A PRIORI | Structural belief tracking dominant |
| FauxPas | blend | 0.8/0.2 | A PRIORI | Keyword dominance structural |
| Hinting | — | — | A PRIORI | |
| Persuasion | — | — | A PRIORI | |
| StrangeStory | threshold | 0.5 | **POST-HOC** | |

## Summary Statistics

- **Benchmarks with POST-HOC parameters**: ~45 of ~76 (~60%)
- **Benchmarks with purely A PRIORI / LITERATURE parameters**: ~20 of ~76 (~26%)
- **Benchmarks with THEORETICAL baselines** (no human data): ~9 of ~76 (~12%)

## Interpretation Guide

When citing psych-bench results:

1. **Do say**: "With calibrated parameters, the HDC-IIT-CfC architecture reproduces human-like cognitive profiles across 22 domains"
2. **Don't say**: "The architecture inherently produces human-like behavior without parameter fitting"
3. **The honest claim**: The architecture has sufficient expressiveness to capture human cognitive patterns when appropriately parameterized — a necessary but not sufficient condition for cognitive plausibility
4. **The strongest honest claim**: Benchmarks where z > 0 with purely A PRIORI parameters (RemoteAssociates, N-back, DigitSpan, IGT, etc.) demonstrate genuine structural alignment with human cognition

## Frozen Benchmark Protocol

To distinguish structural from calibrated performance:

1. **Frozen run**: Fix all benchmark parameters at their current values, then measure z-scores on NEW baselines (different effect sizes, different tasks) without any parameter adjustment
2. **A priori run**: Run only benchmarks classified as A PRIORI or LITERATURE above
3. **Generalization test**: Add new benchmarks in existing domains and report z-scores without any parameter tuning

The `BenchmarkConfig.frozen_params` flag (added Mar 17) prevents post-hoc recalibration by logging a warning when benchmark-internal parameters differ from their snapshot values.
