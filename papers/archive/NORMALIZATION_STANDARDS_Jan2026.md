# Component Normalization Standards

**Version**: 1.0
**Created**: 2026-01-16
**Purpose**: Standardize mapping from raw neural measurements to [0,1] component values

---

## Overview

The Five-Component Model requires all components (Φ, B, W, A, R) to be on a [0,1] scale. However, different neural measurements have different natural ranges:

| Component | Primary Metric | Natural Range | Issue |
|-----------|---------------|---------------|-------|
| Φ (Integration) | PCI | ~0.15-0.70 | Never reaches 0 or 1 |
| B (Binding) | Gamma PLV | 0-1 | Already normalized |
| W (Workspace) | P300 amplitude | 0-20+ μV | No natural upper bound |
| A (Attention) | Alpha suppression | -100% to +50% | Negative values possible |
| R (Recursion) | Meta-d' | -1 to +3 | Unbounded positive |

This document specifies standardized normalization procedures.

---

## General Principles

### 1. Reference Population

All normalizations are relative to a **healthy adult reference population**:
- Age: 18-65 years
- No neurological or psychiatric diagnoses
- Eyes-closed resting state + task conditions as appropriate
- N ≥ 50 for robust percentile estimation

### 2. Normalization Methods

We use **percentile-based normalization** rather than min-max scaling:

$$x_{norm} = \Phi^{-1}(F(x_{raw}))$$

where:
- F(x) is the cumulative distribution function for the metric in the reference population
- Φ^{-1} is the standard normal quantile function
- The result is then scaled to [0,1] via: $x_{[0,1]} = \Phi(x_{norm})$

This approach:
- Handles outliers gracefully
- Preserves rank ordering
- Maps population median to 0.5
- Allows meaningful cross-metric comparison

### 3. State-Specific Baselines

Some components require state-specific baselines:
- **Wake baseline**: For A (Attention), W (Workspace)
- **Task baseline**: For R (meta-cognitive tasks)
- **State-independent**: Φ, B (anatomical/oscillatory properties)

---

## Component-Specific Procedures

### Φ (Integration): Perturbational Complexity Index

**Raw metric**: PCI (Perturbational Complexity Index) from TMS-EEG

**Natural range**: ~0.15-0.70 in healthy subjects; <0.31 suggests unconsciousness

**Reference distribution** (healthy adults, eyes-open):
- 5th percentile: 0.31
- Median: 0.45
- 95th percentile: 0.65

**Normalization procedure**:

```python
def normalize_phi(pci_raw, reference_distribution):
    """
    Normalize PCI to [0,1] using reference population percentiles.

    Args:
        pci_raw: Raw PCI value (typically 0.15-0.70)
        reference_distribution: Array of PCI values from healthy reference population

    Returns:
        phi_normalized: Value in [0,1]
    """
    # Compute percentile rank
    percentile = np.mean(reference_distribution <= pci_raw)

    # Clip extreme values
    percentile = np.clip(percentile, 0.001, 0.999)

    return percentile
```

**Clinical thresholds** (after normalization):
- Φ < 0.15: Vegetative state / deep anesthesia
- Φ = 0.15-0.35: Minimally conscious state
- Φ = 0.35-0.50: Emerging consciousness
- Φ > 0.50: Full wakefulness

**Approximation when PCI unavailable**:
- Use Lempel-Ziv complexity (LZc) of spontaneous EEG
- Correlation with PCI: r ≈ 0.75
- Normalization: same percentile-based approach

---

### B (Binding): Gamma Phase-Locking Value

**Raw metric**: PLV in gamma band (30-50 Hz) across electrode pairs

**Natural range**: 0-1 (already bounded)

**Reference distribution** (healthy adults, eyes-open):
- 5th percentile: 0.15
- Median: 0.35
- 95th percentile: 0.55

**Normalization procedure**:

```python
def normalize_binding(plv_raw, reference_distribution):
    """
    Normalize gamma PLV to [0,1].

    Note: PLV is already bounded [0,1] but typically ranges 0.1-0.6.
    We rescale to use the full range based on population distribution.
    """
    percentile = np.mean(reference_distribution <= plv_raw)
    percentile = np.clip(percentile, 0.001, 0.999)
    return percentile
```

**State effects**:
- Sleep N3: B drops to ~0.10-0.20
- Anesthesia: B drops to ~0.05-0.15
- Meditation: B may increase to ~0.50-0.60

**Approximation when gamma PLV unavailable**:
- Use cross-frequency coupling (theta-gamma)
- Use inter-trial coherence in visual tasks
- Correlation with PLV: r ≈ 0.65

---

### W (Workspace): P300 Amplitude + Global Signal

**Raw metrics**:
1. P300 amplitude (μV) from oddball task
2. Global signal variance from fMRI

**Natural range**:
- P300: 0-25 μV (no natural upper bound)
- Global signal variance: arbitrary units

**Reference distribution** (healthy adults, oddball task):
- 5th percentile: 3.5 μV
- Median: 8.5 μV
- 95th percentile: 18.0 μV

**Normalization procedure**:

```python
def normalize_workspace(p300_amplitude, reference_distribution):
    """
    Normalize P300 amplitude to [0,1].

    P300 reflects workspace ignition - larger amplitudes indicate
    stronger global broadcast.
    """
    percentile = np.mean(reference_distribution <= p300_amplitude)
    percentile = np.clip(percentile, 0.001, 0.999)
    return percentile

def normalize_workspace_fmri(global_variance, reference_distribution):
    """
    Alternative: normalize fMRI global signal variance.
    """
    percentile = np.mean(reference_distribution <= global_variance)
    percentile = np.clip(percentile, 0.001, 0.999)
    return percentile
```

**Combined measure** (when both available):
$$W = 0.6 \times W_{P300} + 0.4 \times W_{fMRI}$$

**State effects**:
- Sleep: P300 absent or severely reduced
- Anesthesia: P300 abolished
- Attention lapses: P300 reduced

---

### A (Attention): Alpha Suppression

**Raw metric**: Alpha power change (8-12 Hz) during attention vs. baseline

**Natural range**: -80% to +50% (suppression is negative)

**Reference distribution** (healthy adults, visual attention task):
- 5th percentile: -65% (strong suppression)
- Median: -35%
- 95th percentile: +5% (no suppression)

**Normalization procedure**:

```python
def normalize_attention(alpha_change_percent, reference_distribution):
    """
    Normalize alpha suppression to [0,1].

    More negative values (stronger suppression) = higher attention.
    We invert so that high A = strong attention.
    """
    # Invert: stronger suppression → higher percentile
    inverted_distribution = -reference_distribution
    inverted_value = -alpha_change_percent

    percentile = np.mean(inverted_distribution <= inverted_value)
    percentile = np.clip(percentile, 0.001, 0.999)
    return percentile
```

**Alternative metrics** (in order of preference):
1. Alpha lateralization index (attention to one hemifield)
2. Pupil dilation (arousal/attention proxy)
3. Reaction time variability (inverse: high variability = low attention)

**State effects**:
- Sleep onset: Alpha suppression fails (A → 0)
- Anesthesia: Alpha power increases paradoxically
- Flow states: Strong sustained suppression

---

### R (Recursion/HOT): Meta-cognitive Accuracy

**Raw metric**: Meta-d' (meta-cognitive sensitivity)

**Natural range**: -1 to +3 (theoretical unbounded positive)

**Reference distribution** (healthy adults, confidence judgment task):
- 5th percentile: 0.2
- Median: 1.1
- 95th percentile: 2.3

**Normalization procedure**:

```python
def normalize_recursion(meta_d_prime, reference_distribution):
    """
    Normalize meta-d' to [0,1].

    Meta-d' measures how well confidence tracks accuracy.
    Higher values indicate better meta-cognitive monitoring.
    """
    percentile = np.mean(reference_distribution <= meta_d_prime)
    percentile = np.clip(percentile, 0.001, 0.999)
    return percentile
```

**Alternative metrics**:
1. Theory of mind task accuracy
2. Metacognitive efficiency (meta-d'/d')
3. Confidence calibration
4. Prefrontal connectivity strength

**State effects**:
- Dreaming: R variable (lucid dreams: high R; non-lucid: low R)
- Psychedelics: R often disrupted (ego dissolution)
- Meditation: R may increase (meta-awareness training)

---

## Clinical Application Guidelines

### 1. Minimum Requirements

For clinical assessment, we require:
- At least 3 of 5 components measured directly
- Remaining components may be approximated or marked as "unavailable"
- Confidence intervals should be reported

### 2. Component Weighting When Missing

If a component cannot be measured:

```python
def compute_c_with_missing(components, available_mask):
    """
    Compute C when some components are unavailable.

    Args:
        components: dict with keys 'phi', 'b', 'w', 'a', 'r'
        available_mask: dict of booleans indicating which are measured

    Returns:
        c_value: Consciousness estimate
        confidence: Confidence interval
    """
    available_values = [v for k, v in components.items() if available_mask[k]]

    if len(available_values) < 3:
        raise ValueError("At least 3 components required for valid estimate")

    # Use min of available components
    c_value = min(available_values)

    # Wider confidence interval when components missing
    n_missing = 5 - len(available_values)
    confidence_width = 0.1 + 0.05 * n_missing

    return c_value, (c_value - confidence_width, c_value + confidence_width)
```

### 3. Cross-Population Considerations

Reference distributions may need adjustment for:
- **Age**: Children and elderly have different baselines
- **Medication**: Many drugs affect specific components
- **Neurological conditions**: Baseline shifts expected

---

## Validation Requirements

### Test-Retest Reliability

All normalized measures should achieve:
- ICC > 0.75 for same-session repeated measures
- ICC > 0.60 for between-session (1 week apart)

### Cross-Site Reproducibility

Multi-site studies should demonstrate:
- Between-site correlation r > 0.85
- Systematic bias < 0.10 (normalized units)

### Construct Validity

Normalized components should:
- Discriminate conscious vs. unconscious states (AUC > 0.80)
- Track within-subject state changes (anesthesia, sleep)
- Correlate with behavioral measures (r > 0.50)

---

## Implementation Notes

### Reference Dataset

A canonical reference dataset is available at:
- Location: `data/reference/healthy_adults_n50.npz`
- Contains: All five component distributions from 50 healthy adults
- Update frequency: Annually with expanded sample

### Software Implementation

Normalization functions are implemented in:
- Rust: `src/hdc/normalization.rs`
- Python: `symthaea/normalization.py`

Both implementations produce identical results (validated to 6 decimal places).

---

## Version History

- **v1.0** (2026-01-16): Initial normalization standards
