# Rigorous Algorithmic Specifications for the Five-Component Framework

## 1. First-Principles Derivation of the Minimum Function

### 1.1 The Problem

Why C = min(Φ, B, W, A, R) and not:
- C = Φ × B × W × A × R (product)
- C = (Φ + B + W + A + R) / 5 (mean)
- C = (Φ × B × W × A × R)^0.2 (geometric mean)

### 1.2 The Information Processing Loop Argument

**Theorem 1**: If consciousness requires a complete information processing loop, then the minimum function is the correct aggregation.

**Proof sketch**:

Consider consciousness as a cyclic process:

```
    ┌─────────────────────────────────────────────┐
    │                                             │
    ▼                                             │
Sensory Input → Integration(Φ) → Binding(B) → Workspace(W) → Recursion(R)
                                      ↑              │
                                      └── Attention(A) ──┘
```

**Axiom 1 (Loop Completeness)**: Conscious experience requires completing the full loop. A break at any point halts the process.

**Axiom 2 (Rate Limitation)**: Information flows through the loop at the rate of the slowest component (bottleneck principle, cf. Theory of Constraints).

**Axiom 3 (Normalization)**: Each component processes information at rate c_i ∈ [0, 1], where 1 = maximum biological capacity.

**Derivation**:
Let I(t) = information integrated into conscious experience by time t.

By Axiom 2, the rate of conscious information processing is:
```
dI/dt = min(rate_Φ, rate_B, rate_W, rate_A, rate_R)
```

At steady state, integrated conscious information:
```
I_∞ = min(Φ, B, W, A, R)
```

If we define C = I_∞ (consciousness level = steady-state integrated information), then:
```
C = min(Φ, B, W, A, R)  ∎
```

**Why not product?**
Product would imply: if Φ = 0.9, B = 0.9, W = 0.9, A = 0.9, R = 0.9, then C = 0.9^5 = 0.59
But empirically, healthy wake consciousness should be C ≈ 0.9, not 0.59.

**Why not mean?**
Mean would imply: if A = 0 (coma patient with no attention), C = (0.6 + 0.5 + 0.4 + 0 + 0.3)/5 = 0.36
But empirically, such a patient is C ≈ 0, not 0.36.

### 1.3 Causal Necessity Formalization

**Definition (Causal Necessity)**: Component X is causally necessary for consciousness iff:
```
∀ states s: X(s) = 0 → C(s) = 0
```

**Theorem 2**: If all five components are causally necessary, then C = min(Φ, B, W, A, R).

**Proof**:
Let f: [0,1]^5 → [0,1] be the aggregation function.

By causal necessity, for each i:
```
c_i = 0 → f(c_1, ..., c_5) = 0
```

The only continuous, monotonically increasing function satisfying this for all i is:
```
f(c_1, ..., c_5) = min(c_1, ..., c_5) × g(c_1, ..., c_5)
```

where g ≥ 0 is a modifier. If we additionally require:
- f(1, 1, 1, 1, 1) = 1 (maximum consciousness when all components maximal)
- f is smooth

Then g = 1 and f = min.  ∎

---

## 2. Precise Algorithmic Definitions

### 2.1 Integration (Φ): Perturbational Complexity Index Approximation

**Theoretical construct**: Integrated information per IIT 3.0
**Computational problem**: True Φ is intractable (O(2^n) for n nodes)
**Approximation**: Lempel-Ziv complexity of EEG response to perturbation

**Algorithm Φ-LZc**:

```python
def compute_phi_lzc(eeg_data: np.ndarray, sfreq: float) -> float:
    """
    Compute Φ approximation via Lempel-Ziv complexity.

    Parameters
    ----------
    eeg_data : np.ndarray
        Shape (n_channels, n_samples). Preprocessed EEG data.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    phi : float
        Normalized Φ estimate in [0, 1].

    Algorithm
    ---------
    1. Bandpass filter 0.5-45 Hz (remove line noise and drift)
    2. Binarize each channel at its median (above=1, below=0)
    3. Concatenate all channels into single binary string
    4. Compute Lempel-Ziv complexity (number of unique substrings)
    5. Normalize by maximum possible complexity

    References
    ----------
    Casali et al. (2013) Sci Transl Med
    Schartner et al. (2017) PLoS Comp Bio
    """
    from scipy.signal import butter, filtfilt

    # Step 1: Bandpass filter
    nyq = sfreq / 2
    b, a = butter(4, [0.5/nyq, 45/nyq], btype='band')
    filtered = filtfilt(b, a, eeg_data, axis=1)

    # Step 2: Binarize at median
    medians = np.median(filtered, axis=1, keepdims=True)
    binary = (filtered > medians).astype(int)

    # Step 3: Concatenate channels
    binary_string = ''.join(str(x) for row in binary for x in row)

    # Step 4: Lempel-Ziv complexity
    lzc = _lempel_ziv_complexity(binary_string)

    # Step 5: Normalize
    n = len(binary_string)
    max_lzc = n / np.log2(n)  # Theoretical maximum for random string
    phi = min(1.0, lzc / max_lzc)

    return phi


def _lempel_ziv_complexity(s: str) -> int:
    """Compute Lempel-Ziv complexity of binary string."""
    n = len(s)
    if n == 0:
        return 0

    complexity = 1
    prefix_length = 1
    suffix_start = 1

    while suffix_start + prefix_length <= n:
        # Check if current component exists in prefix
        if s[suffix_start:suffix_start + prefix_length] in s[:suffix_start + prefix_length - 1]:
            prefix_length += 1
        else:
            complexity += 1
            suffix_start += prefix_length
            prefix_length = 1

    return complexity
```

**Validation requirements**:
- Test on synthetic data with known complexity
- Compare with published PCI values on same recordings
- Report correlation with PCI where TMS-EEG available

---

### 2.2 Binding (B): Gamma-Band Phase Synchrony

**Theoretical construct**: Temporal binding via gamma oscillations (30-80 Hz)
**Measure**: Global phase-locking value across electrode pairs

**Algorithm B-PLV**:

```python
def compute_binding_plv(eeg_data: np.ndarray, sfreq: float) -> float:
    """
    Compute Binding via gamma-band phase-locking value.

    Parameters
    ----------
    eeg_data : np.ndarray
        Shape (n_channels, n_samples). Preprocessed EEG data.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    binding : float
        Global PLV in [0, 1]. 0 = no synchrony, 1 = perfect synchrony.

    Algorithm
    ---------
    1. Bandpass filter 30-80 Hz (gamma band)
    2. Extract instantaneous phase via Hilbert transform
    3. For each channel pair, compute PLV = |mean(exp(i*Δφ))|
    4. Average across all pairs
    5. Apply sigmoid normalization to [0, 1]

    References
    ----------
    Lachaux et al. (1999) Human Brain Mapping
    Singer (1999) Neuron
    """
    from scipy.signal import butter, filtfilt, hilbert

    n_channels = eeg_data.shape[0]

    # Step 1: Bandpass filter (gamma: 30-80 Hz)
    nyq = sfreq / 2
    if 80 / nyq >= 1:
        high_cutoff = nyq * 0.9  # Adjust for low sampling rates
    else:
        high_cutoff = 80
    b, a = butter(4, [30/nyq, high_cutoff/nyq], btype='band')
    gamma = filtfilt(b, a, eeg_data, axis=1)

    # Step 2: Extract phase via Hilbert transform
    analytic = hilbert(gamma, axis=1)
    phase = np.angle(analytic)

    # Step 3-4: Compute pairwise PLV and average
    plv_sum = 0.0
    n_pairs = 0
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phase[i, :] - phase[j, :]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_sum += plv
            n_pairs += 1

    mean_plv = plv_sum / n_pairs if n_pairs > 0 else 0

    # Step 5: Sigmoid normalization
    # Empirically, mean PLV ranges ~0.1-0.4 in typical EEG
    # Normalize so 0.1 → 0.2, 0.4 → 0.8
    binding = 1 / (1 + np.exp(-10 * (mean_plv - 0.25)))

    return binding
```

---

### 2.3 Workspace (W): Global Signal Variance

**Theoretical construct**: Information globally broadcast to specialized processors
**Measure**: Variance of global field power + long-range connectivity

**Algorithm W-GFP**:

```python
def compute_workspace_gfp(eeg_data: np.ndarray, sfreq: float) -> float:
    """
    Compute Workspace via global field power variance and connectivity.

    Parameters
    ----------
    eeg_data : np.ndarray
        Shape (n_channels, n_samples). Preprocessed EEG data.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    workspace : float
        Workspace activation in [0, 1].

    Algorithm
    ---------
    1. Compute global field power: GFP = std across channels at each time
    2. Compute variance of GFP (high variance = dynamic broadcasting)
    3. Compute long-range connectivity (frontal-posterior coherence)
    4. Combine: W = 0.5 * GFP_variance + 0.5 * connectivity
    5. Normalize to [0, 1]

    References
    ----------
    Dehaene & Changeux (2011) Neuron
    Mashour et al. (2020) NEJM
    """
    from scipy.signal import coherence

    n_channels = eeg_data.shape[0]

    # Step 1: Global Field Power
    gfp = np.std(eeg_data, axis=0)

    # Step 2: GFP variance (normalized)
    gfp_var = np.var(gfp) / (np.mean(gfp) ** 2 + 1e-10)
    gfp_var_norm = min(1.0, gfp_var / 2.0)  # Empirical scaling

    # Step 3: Long-range connectivity (simplified: first half vs second half of channels)
    # In practice, would use actual electrode locations
    n_half = n_channels // 2
    frontal = eeg_data[:n_half, :].mean(axis=0)
    posterior = eeg_data[n_half:, :].mean(axis=0)

    # Coherence in alpha-beta band (8-30 Hz) - workspace communication
    f, coh = coherence(frontal, posterior, fs=sfreq, nperseg=int(sfreq * 2))
    alpha_beta_mask = (f >= 8) & (f <= 30)
    mean_coherence = np.mean(coh[alpha_beta_mask]) if np.any(alpha_beta_mask) else 0

    # Step 4: Combine
    workspace = 0.5 * gfp_var_norm + 0.5 * mean_coherence

    # Step 5: Ensure [0, 1]
    workspace = np.clip(workspace, 0, 1)

    return workspace
```

---

### 2.4 Attention (A): Alpha Suppression + Arousal Index

**Theoretical construct**: Precision-weighted selection of content for workspace access
**Measure**: Beta/Alpha ratio (arousal) + alpha desynchronization (engagement)

**Algorithm A-Arousal**:

```python
def compute_attention_arousal(eeg_data: np.ndarray, sfreq: float) -> float:
    """
    Compute Attention via arousal index and alpha suppression.

    Parameters
    ----------
    eeg_data : np.ndarray
        Shape (n_channels, n_samples). Preprocessed EEG data.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    attention : float
        Attention/arousal level in [0, 1].

    Algorithm
    ---------
    1. Compute power spectral density
    2. Extract band powers: delta (0.5-4), theta (4-8), alpha (8-13), beta (13-30)
    3. Arousal index = beta / (delta + theta)
    4. Alpha ratio = alpha / total_power
    5. Attention = 0.6 * arousal_normalized + 0.4 * (1 - alpha_ratio)
       (High attention = high arousal + low alpha)

    References
    ----------
    Klimesch (1999) Brain Research Reviews
    Pfurtscheller & Lopes da Silva (1999) Clinical Neurophysiology
    """
    from scipy.signal import welch

    # Step 1: Compute PSD
    f, psd = welch(eeg_data, fs=sfreq, nperseg=int(sfreq * 2), axis=1)

    # Average across channels
    psd_mean = np.mean(psd, axis=0)

    # Step 2: Extract band powers
    delta_mask = (f >= 0.5) & (f < 4)
    theta_mask = (f >= 4) & (f < 8)
    alpha_mask = (f >= 8) & (f < 13)
    beta_mask = (f >= 13) & (f < 30)

    delta_power = np.mean(psd_mean[delta_mask]) if np.any(delta_mask) else 1e-10
    theta_power = np.mean(psd_mean[theta_mask]) if np.any(theta_mask) else 1e-10
    alpha_power = np.mean(psd_mean[alpha_mask]) if np.any(alpha_mask) else 0
    beta_power = np.mean(psd_mean[beta_mask]) if np.any(beta_mask) else 0
    total_power = np.mean(psd_mean)

    # Step 3: Arousal index
    arousal = beta_power / (delta_power + theta_power + 1e-10)

    # Normalize arousal (empirically ranges 0.1-2.0)
    arousal_norm = min(1.0, arousal / 1.5)

    # Step 4: Alpha ratio (relative power)
    alpha_ratio = alpha_power / (total_power + 1e-10)

    # Step 5: Combine
    # High attention = high arousal + low alpha (alpha suppression indicates engagement)
    attention = 0.6 * arousal_norm + 0.4 * (1 - min(1.0, alpha_ratio * 3))

    attention = np.clip(attention, 0, 1)

    return attention
```

---

### 2.5 Recursion (R): Frontal-Posterior Theta Coupling

**Theoretical construct**: Meta-representation, higher-order thought
**Measure**: Theta phase synchrony between frontal and posterior regions

**Algorithm R-Theta**:

```python
def compute_recursion_theta(eeg_data: np.ndarray, sfreq: float,
                            frontal_idx: list = None,
                            posterior_idx: list = None) -> float:
    """
    Compute Recursion via frontal-posterior theta coupling.

    Parameters
    ----------
    eeg_data : np.ndarray
        Shape (n_channels, n_samples). Preprocessed EEG data.
    sfreq : float
        Sampling frequency in Hz.
    frontal_idx : list, optional
        Indices of frontal channels. Default: first third.
    posterior_idx : list, optional
        Indices of posterior channels. Default: last third.

    Returns
    -------
    recursion : float
        Meta-cognitive capacity estimate in [0, 1].

    Algorithm
    ---------
    1. Extract frontal and posterior channel groups
    2. Bandpass filter to theta (4-8 Hz)
    3. Compute instantaneous phase via Hilbert transform
    4. Compute phase-locking value between frontal and posterior
    5. Normalize to [0, 1]

    Rationale
    ---------
    Theta oscillations coordinate frontal-posterior communication for
    metacognitive processes. High theta PLV indicates active higher-order
    processing.

    References
    ----------
    Rutishauser et al. (2010) Science
    Fleming & Dolan (2012) TICS
    """
    from scipy.signal import butter, filtfilt, hilbert

    n_channels = eeg_data.shape[0]

    # Default channel grouping
    if frontal_idx is None:
        frontal_idx = list(range(n_channels // 3))
    if posterior_idx is None:
        posterior_idx = list(range(2 * n_channels // 3, n_channels))

    # Step 1: Extract channel groups
    frontal = eeg_data[frontal_idx, :].mean(axis=0)
    posterior = eeg_data[posterior_idx, :].mean(axis=0)

    # Step 2: Bandpass filter (theta: 4-8 Hz)
    nyq = sfreq / 2
    b, a = butter(4, [4/nyq, 8/nyq], btype='band')
    frontal_theta = filtfilt(b, a, frontal)
    posterior_theta = filtfilt(b, a, posterior)

    # Step 3: Extract phase
    frontal_phase = np.angle(hilbert(frontal_theta))
    posterior_phase = np.angle(hilbert(posterior_theta))

    # Step 4: Phase-locking value
    phase_diff = frontal_phase - posterior_phase
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    # Step 5: Normalize (theta PLV typically 0.1-0.5)
    recursion = min(1.0, plv * 2.5)

    return recursion
```

---

## 3. Attention Primacy Formalization

### 3.1 The Hypothesis

**Strong form**: A is multiplicatively gating:
```
C = A × min(Φ, B, W, R)
```

**Weak form**: A has a lower effective threshold:
```
C = min(Φ, B, W, A, R)  where θ_A < θ_others
```

### 3.2 Testable Predictions

| # | Prediction | Test | Expected if True |
|---|------------|------|------------------|
| P1 | A collapses before Φ at sleep onset | PSG study | A decreases >50% while Φ decreases <20% |
| P2 | Preserving A maintains C despite low Φ | Modafinil + sleep deprivation | C correlation with A > correlation with Φ |
| P3 | A-targeted interventions have larger effect | Compare modafinil (A) vs. memantine (Φ) | Effect size d > 0.5 difference |
| P4 | DOC patients with high Φ, low A are unconscious | EEG + behavioral assessment | High Φ / low A → VS diagnosis |

### 3.3 Falsification Criteria

The Attention Primacy hypothesis is **falsified** if:
1. Any study shows Φ collapsing before A at sleep onset (with matched measurement sensitivity)
2. Preserving A via pharmacology does NOT maintain consciousness when Φ is low
3. DOC patients show high A, low Φ profile AND are diagnosed VS (not MCS)

---

## 4. Phase Transition Formalization

### 4.1 The Model

```python
def consciousness_with_threshold(C_raw: float, C_star: float = 0.15,
                                  steepness: float = 20) -> float:
    """
    Consciousness with phase transition at threshold C*.

    Parameters
    ----------
    C_raw : float
        Raw consciousness score = min(Φ, B, W, A, R)
    C_star : float
        Critical threshold for phase transition
    steepness : float
        Sigmoid steepness (higher = sharper transition)

    Returns
    -------
    C_phenomenal : float
        Phenomenal consciousness level in [0, 1]
    """
    return 1 / (1 + np.exp(-steepness * (C_raw - C_star)))
```

### 4.2 Testable Predictions

| # | Prediction | Test | Expected if True |
|---|------------|------|------------------|
| P5 | Response probability follows sigmoid, not linear | Propofol titration | Sigmoid fit R² > Linear fit R² |
| P6 | Critical slowing near C* | Time-series analysis at transition | Autocorrelation increases near C* |
| P7 | Hysteresis | Compare induction vs emergence | C* for loss > C* for recovery |

---

## 5. Validation Protocol

### 5.1 Preregistration Template

```yaml
Study: Five-Component Consciousness Framework Validation
Registration: OSF (url to be added)
Date: YYYY-MM-DD

Hypotheses:
  H1: C = min(Φ,B,W,A,R) predicts consciousness level (r > 0.5)
  H2: A shows primacy - collapses first at sleep onset
  H3: Phase transition exists at C* ≈ 0.15

Sample Size:
  N = 40 (sleep study)
  Power: 0.80 for r = 0.5 at α = 0.05
  Justification: G*Power calculation

Exclusion Criteria:
  - Neurological disorders
  - Psychoactive medication
  - Poor EEG signal quality (>20% epochs rejected)

Analysis Plan:
  Primary: Spearman correlation, C vs expert consciousness rating
  Secondary: Compare A trajectory vs Φ trajectory
  Exploratory: Sigmoid vs linear fit for phase transition

Statistical Thresholds:
  α = 0.05 (two-tailed)
  Multiple comparison correction: Bonferroni for 3 primary tests

Data & Code Sharing:
  All data: OSF repository
  All code: GitHub with DOI via Zenodo
```

### 5.2 Success Criteria

The framework is **supported** if:
- H1: r > 0.5, p < 0.05
- H2: A drops >3x faster than Φ at sleep onset
- H3: Sigmoid fit is significantly better than linear (likelihood ratio p < 0.05)

The framework is **refuted** if:
- H1: r < 0.3
- H2: Φ drops before A
- H3: Linear fit is better than sigmoid

---

## 6. Updated Code Implementation

See `analysis/compute_components_v2.py` for the complete rigorous implementation incorporating all specifications above.
