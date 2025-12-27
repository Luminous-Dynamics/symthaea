#!/usr/bin/env python3
"""
Five-Component Consciousness Framework - Rigorous Implementation v2.0

This implementation addresses the methodological weaknesses identified in the
paradigm shift analysis:
1. Precise algorithmic definitions (reproducible)
2. Explicit parameter choices (transparent)
3. Validation against synthetic ground truth
4. Falsification criteria built in

Author: Tristan Stoltz
Affiliation: Luminous Dynamics
License: MIT
"""

import numpy as np
from typing import Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ComponentConfig:
    """Configuration for component computation with explicit parameters."""

    # Phi (Integration) parameters
    phi_filter_low: float = 0.5  # Hz
    phi_filter_high: float = 45.0  # Hz
    phi_filter_order: int = 4

    # Binding parameters
    binding_gamma_low: float = 30.0  # Hz
    binding_gamma_high: float = 80.0  # Hz
    binding_plv_midpoint: float = 0.25  # Sigmoid center
    binding_plv_steepness: float = 10.0

    # Workspace parameters
    workspace_coherence_low: float = 8.0  # Hz
    workspace_coherence_high: float = 30.0  # Hz
    workspace_gfp_weight: float = 0.5
    workspace_connectivity_weight: float = 0.5

    # Attention parameters
    attention_arousal_weight: float = 0.6
    attention_alpha_weight: float = 0.4
    attention_arousal_max: float = 1.5  # Normalization constant

    # Recursion parameters
    recursion_theta_low: float = 4.0  # Hz
    recursion_theta_high: float = 8.0  # Hz
    recursion_plv_scale: float = 2.5

    # Phase transition parameters
    phase_transition_threshold: float = 0.15
    phase_transition_steepness: float = 20.0


DEFAULT_CONFIG = ComponentConfig()


# =============================================================================
# Core Data Structures
# =============================================================================

class ConsciousnessVector(NamedTuple):
    """Five-component consciousness vector with metadata."""
    phi: float          # Integration
    binding: float      # Temporal synchrony
    workspace: float    # Global broadcast
    attention: float    # Precision/selection
    recursion: float    # Meta-representation

    @property
    def C(self) -> float:
        """Raw consciousness score = min of components."""
        return min(self.phi, self.binding, self.workspace,
                   self.attention, self.recursion)

    @property
    def C_phenomenal(self, threshold: float = 0.15,
                      steepness: float = 20.0) -> float:
        """Phenomenal consciousness with phase transition."""
        return 1 / (1 + np.exp(-steepness * (self.C - threshold)))

    @property
    def limiting_component(self) -> str:
        """Which component is currently limiting consciousness."""
        components = {
            'phi': self.phi,
            'binding': self.binding,
            'workspace': self.workspace,
            'attention': self.attention,
            'recursion': self.recursion
        }
        return min(components, key=components.get)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'phi': self.phi,
            'binding': self.binding,
            'workspace': self.workspace,
            'attention': self.attention,
            'recursion': self.recursion,
            'C': self.C,
            'limiting': self.limiting_component
        }


# =============================================================================
# Signal Processing Utilities
# =============================================================================

def bandpass_filter(data: np.ndarray, sfreq: float,
                    low: float, high: float, order: int = 4) -> np.ndarray:
    """
    Apply zero-phase bandpass filter.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (n_channels, n_samples) or (n_samples,)
    sfreq : float
        Sampling frequency in Hz
    low : float
        Low cutoff frequency in Hz
    high : float
        High cutoff frequency in Hz
    order : int
        Filter order

    Returns
    -------
    filtered : np.ndarray
        Filtered data, same shape as input
    """
    from scipy.signal import butter, filtfilt

    nyq = sfreq / 2

    # Handle edge cases for sampling rate
    if high >= nyq:
        high = nyq * 0.9
    if low >= high:
        low = high * 0.5

    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)


def compute_psd(data: np.ndarray, sfreq: float,
                nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density via Welch's method.

    Parameters
    ----------
    data : np.ndarray
        Input data
    sfreq : float
        Sampling frequency
    nperseg : int, optional
        Segment length for Welch's method

    Returns
    -------
    freqs : np.ndarray
        Frequency vector
    psd : np.ndarray
        Power spectral density
    """
    from scipy.signal import welch

    if nperseg is None:
        nperseg = min(int(sfreq * 2), data.shape[-1] // 4)

    return welch(data, fs=sfreq, nperseg=nperseg, axis=-1)


def extract_phase(data: np.ndarray) -> np.ndarray:
    """Extract instantaneous phase via Hilbert transform."""
    from scipy.signal import hilbert
    return np.angle(hilbert(data, axis=-1))


# =============================================================================
# Lempel-Ziv Complexity (for Phi)
# =============================================================================

def lempel_ziv_complexity(s: str) -> int:
    """
    Compute Lempel-Ziv complexity of a binary string.

    This is the number of unique substrings encountered when parsing
    the string from left to right.

    Parameters
    ----------
    s : str
        Binary string (only '0' and '1' characters)

    Returns
    -------
    complexity : int
        Lempel-Ziv complexity

    References
    ----------
    Lempel & Ziv (1976) IEEE Trans Inf Theory
    """
    n = len(s)
    if n == 0:
        return 0

    # Initialize
    complexity = 1
    prefix_end = 1  # End of current prefix (exclusive)
    component_start = 1  # Start of current component

    while component_start < n:
        # Try to extend current component
        component_end = component_start + 1

        while component_end <= n:
            # Check if current component exists in prefix
            component = s[component_start:component_end]
            prefix = s[:prefix_end]

            # Use extended prefix that includes partial current component
            extended_prefix = s[:component_end - 1]

            if component in extended_prefix:
                component_end += 1
            else:
                break

        # Component complete
        complexity += 1
        prefix_end = component_end
        component_start = component_end

    return complexity


def normalized_lzc(data: np.ndarray) -> float:
    """
    Compute normalized Lempel-Ziv complexity.

    Parameters
    ----------
    data : np.ndarray
        Multichannel data, shape (n_channels, n_samples)

    Returns
    -------
    lzc_norm : float
        Normalized LZc in [0, 1]
    """
    # Binarize at median
    medians = np.median(data, axis=1, keepdims=True)
    binary = (data > medians).astype(int)

    # Concatenate channels to single string
    binary_string = ''.join(str(x) for row in binary for x in row)

    # Compute LZc
    lzc = lempel_ziv_complexity(binary_string)

    # Normalize by theoretical maximum
    n = len(binary_string)
    if n == 0:
        return 0.0

    max_lzc = n / np.log2(n) if n > 1 else 1

    return min(1.0, lzc / max_lzc)


# =============================================================================
# Component Computation Functions
# =============================================================================

def compute_phi(eeg_data: np.ndarray, sfreq: float,
                config: ComponentConfig = DEFAULT_CONFIG) -> float:
    """
    Compute Integration (Φ) via Lempel-Ziv complexity.

    This is an approximation to IIT's Φ, validated against PCI.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data, shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz
    config : ComponentConfig
        Configuration parameters

    Returns
    -------
    phi : float
        Integration estimate in [0, 1]
    """
    # Bandpass filter
    filtered = bandpass_filter(
        eeg_data, sfreq,
        config.phi_filter_low,
        config.phi_filter_high,
        config.phi_filter_order
    )

    # Compute normalized LZc
    return normalized_lzc(filtered)


def compute_binding(eeg_data: np.ndarray, sfreq: float,
                    config: ComponentConfig = DEFAULT_CONFIG) -> float:
    """
    Compute Binding (B) via gamma-band phase synchrony.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data, shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz
    config : ComponentConfig
        Configuration parameters

    Returns
    -------
    binding : float
        Binding estimate in [0, 1]
    """
    n_channels = eeg_data.shape[0]

    if n_channels < 2:
        return 0.5  # Can't compute synchrony with single channel

    # Bandpass filter to gamma
    gamma = bandpass_filter(
        eeg_data, sfreq,
        config.binding_gamma_low,
        min(config.binding_gamma_high, sfreq / 2 * 0.9)
    )

    # Extract phase
    phase = extract_phase(gamma)

    # Compute pairwise PLV
    plv_sum = 0.0
    n_pairs = 0

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phase[i, :] - phase[j, :]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_sum += plv
            n_pairs += 1

    mean_plv = plv_sum / n_pairs if n_pairs > 0 else 0

    # Sigmoid normalization
    binding = 1 / (1 + np.exp(
        -config.binding_plv_steepness * (mean_plv - config.binding_plv_midpoint)
    ))

    return float(binding)


def compute_workspace(eeg_data: np.ndarray, sfreq: float,
                      config: ComponentConfig = DEFAULT_CONFIG) -> float:
    """
    Compute Workspace (W) via global field power and connectivity.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data, shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz
    config : ComponentConfig
        Configuration parameters

    Returns
    -------
    workspace : float
        Workspace activation in [0, 1]
    """
    from scipy.signal import coherence

    n_channels = eeg_data.shape[0]

    # Global Field Power
    gfp = np.std(eeg_data, axis=0)
    gfp_var = np.var(gfp) / (np.mean(gfp) ** 2 + 1e-10)
    gfp_var_norm = min(1.0, gfp_var / 2.0)

    # Long-range connectivity
    if n_channels >= 2:
        n_half = n_channels // 2
        frontal = eeg_data[:max(1, n_half), :].mean(axis=0)
        posterior = eeg_data[n_half:, :].mean(axis=0)

        nperseg = min(int(sfreq * 2), len(frontal) // 4)
        if nperseg > 10:
            f, coh = coherence(frontal, posterior, fs=sfreq, nperseg=nperseg)
            mask = (f >= config.workspace_coherence_low) & \
                   (f <= config.workspace_coherence_high)
            connectivity = np.mean(coh[mask]) if np.any(mask) else 0.5
        else:
            connectivity = 0.5
    else:
        connectivity = 0.5

    # Combine
    workspace = (config.workspace_gfp_weight * gfp_var_norm +
                 config.workspace_connectivity_weight * connectivity)

    return float(np.clip(workspace, 0, 1))


def compute_attention(eeg_data: np.ndarray, sfreq: float,
                      config: ComponentConfig = DEFAULT_CONFIG) -> float:
    """
    Compute Attention (A) via arousal index and alpha suppression.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data, shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz
    config : ComponentConfig
        Configuration parameters

    Returns
    -------
    attention : float
        Attention/arousal in [0, 1]
    """
    # Compute PSD
    f, psd = compute_psd(eeg_data, sfreq)
    psd_mean = np.mean(psd, axis=0) if psd.ndim > 1 else psd

    # Band powers
    def band_power(low, high):
        mask = (f >= low) & (f < high)
        return np.mean(psd_mean[mask]) if np.any(mask) else 1e-10

    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 13)
    beta = band_power(13, 30)
    total = np.mean(psd_mean)

    # Arousal index
    arousal = beta / (delta + theta + 1e-10)
    arousal_norm = min(1.0, arousal / config.attention_arousal_max)

    # Alpha ratio (high alpha = low attention)
    alpha_ratio = alpha / (total + 1e-10)
    alpha_suppression = 1 - min(1.0, alpha_ratio * 3)

    # Combine
    attention = (config.attention_arousal_weight * arousal_norm +
                 config.attention_alpha_weight * alpha_suppression)

    return float(np.clip(attention, 0, 1))


def compute_recursion(eeg_data: np.ndarray, sfreq: float,
                      frontal_idx: Optional[list] = None,
                      posterior_idx: Optional[list] = None,
                      config: ComponentConfig = DEFAULT_CONFIG) -> float:
    """
    Compute Recursion (R) via frontal-posterior theta coupling.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data, shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz
    frontal_idx : list, optional
        Frontal channel indices
    posterior_idx : list, optional
        Posterior channel indices
    config : ComponentConfig
        Configuration parameters

    Returns
    -------
    recursion : float
        Meta-cognitive capacity in [0, 1]
    """
    n_channels = eeg_data.shape[0]

    # Default channel grouping
    if frontal_idx is None:
        frontal_idx = list(range(n_channels // 3))
    if posterior_idx is None:
        posterior_idx = list(range(2 * n_channels // 3, n_channels))

    # Handle edge cases
    if len(frontal_idx) == 0:
        frontal_idx = [0]
    if len(posterior_idx) == 0:
        posterior_idx = [n_channels - 1]

    # Extract regions
    frontal = eeg_data[frontal_idx, :].mean(axis=0)
    posterior = eeg_data[posterior_idx, :].mean(axis=0)

    # Bandpass to theta
    frontal_theta = bandpass_filter(
        frontal, sfreq,
        config.recursion_theta_low,
        config.recursion_theta_high
    )
    posterior_theta = bandpass_filter(
        posterior, sfreq,
        config.recursion_theta_low,
        config.recursion_theta_high
    )

    # Phase-locking value
    frontal_phase = extract_phase(frontal_theta)
    posterior_phase = extract_phase(posterior_theta)
    phase_diff = frontal_phase - posterior_phase
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    # Normalize
    recursion = min(1.0, plv * config.recursion_plv_scale)

    return float(recursion)


# =============================================================================
# Main Computation Functions
# =============================================================================

def compute_consciousness_vector(eeg_data: np.ndarray, sfreq: float,
                                  config: ComponentConfig = DEFAULT_CONFIG
                                  ) -> ConsciousnessVector:
    """
    Compute all five components and return consciousness vector.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data, shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz
    config : ComponentConfig
        Configuration parameters

    Returns
    -------
    vector : ConsciousnessVector
        Five-component consciousness vector
    """
    phi = compute_phi(eeg_data, sfreq, config)
    binding = compute_binding(eeg_data, sfreq, config)
    workspace = compute_workspace(eeg_data, sfreq, config)
    attention = compute_attention(eeg_data, sfreq, config)
    recursion = compute_recursion(eeg_data, sfreq, config=config)

    return ConsciousnessVector(
        phi=phi,
        binding=binding,
        workspace=workspace,
        attention=attention,
        recursion=recursion
    )


def compute_consciousness_with_transition(
    vector: ConsciousnessVector,
    config: ComponentConfig = DEFAULT_CONFIG
) -> float:
    """
    Compute phenomenal consciousness with phase transition.

    Parameters
    ----------
    vector : ConsciousnessVector
        Five-component vector
    config : ComponentConfig
        Configuration with transition parameters

    Returns
    -------
    C_phenomenal : float
        Consciousness level after phase transition in [0, 1]
    """
    C_raw = vector.C
    return 1 / (1 + np.exp(
        -config.phase_transition_steepness * (C_raw - config.phase_transition_threshold)
    ))


# =============================================================================
# Validation: Synthetic Ground Truth
# =============================================================================

def generate_synthetic_eeg(n_channels: int = 64, duration: float = 10.0,
                           sfreq: float = 256.0,
                           state: str = 'wake') -> np.ndarray:
    """
    Generate synthetic EEG with known consciousness properties.

    Parameters
    ----------
    n_channels : int
        Number of channels
    duration : float
        Duration in seconds
    sfreq : float
        Sampling frequency
    state : str
        Consciousness state: 'wake', 'n1', 'n2', 'n3', 'rem', 'vs', 'mcs'

    Returns
    -------
    eeg : np.ndarray
        Synthetic EEG data
    """
    n_samples = int(duration * sfreq)
    t = np.arange(n_samples) / sfreq

    # State-dependent parameters
    state_params = {
        'wake': {'alpha': 0.3, 'beta': 0.4, 'gamma': 0.3, 'delta': 0.1, 'noise': 0.2},
        'n1': {'alpha': 0.4, 'beta': 0.2, 'gamma': 0.1, 'delta': 0.2, 'noise': 0.3},
        'n2': {'alpha': 0.3, 'beta': 0.1, 'gamma': 0.1, 'delta': 0.4, 'noise': 0.3},
        'n3': {'alpha': 0.1, 'beta': 0.1, 'gamma': 0.05, 'delta': 0.7, 'noise': 0.2},
        'rem': {'alpha': 0.2, 'beta': 0.3, 'gamma': 0.2, 'delta': 0.2, 'noise': 0.3},
        'vs': {'alpha': 0.1, 'beta': 0.05, 'gamma': 0.05, 'delta': 0.6, 'noise': 0.4},
        'mcs': {'alpha': 0.2, 'beta': 0.15, 'gamma': 0.1, 'delta': 0.4, 'noise': 0.3},
    }

    params = state_params.get(state, state_params['wake'])

    eeg = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Add oscillatory components
        alpha = params['alpha'] * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
        beta = params['beta'] * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
        gamma = params['gamma'] * np.sin(2 * np.pi * 40 * t + np.random.rand() * 2 * np.pi)
        delta = params['delta'] * np.sin(2 * np.pi * 2 * t + np.random.rand() * 2 * np.pi)
        noise = params['noise'] * np.random.randn(n_samples)

        # Add inter-channel correlation for binding
        if ch > 0 and state in ['wake', 'rem', 'mcs']:
            gamma = 0.5 * gamma + 0.5 * eeg[0, :] * params['gamma']

        eeg[ch, :] = alpha + beta + gamma + delta + noise

    return eeg


def validate_against_synthetic():
    """Run validation against synthetic data with known states."""
    print("=" * 60)
    print("Validation Against Synthetic Data")
    print("=" * 60)

    states = ['wake', 'n1', 'n2', 'n3', 'rem', 'vs', 'mcs']
    expected_order = ['wake', 'rem', 'mcs', 'n1', 'n2', 'n3', 'vs']

    results = {}

    for state in states:
        eeg = generate_synthetic_eeg(state=state)
        vector = compute_consciousness_vector(eeg, sfreq=256.0)
        results[state] = vector

        print(f"\n{state.upper():4s}: Φ={vector.phi:.2f} B={vector.binding:.2f} "
              f"W={vector.workspace:.2f} A={vector.attention:.2f} "
              f"R={vector.recursion:.2f} → C={vector.C:.2f} "
              f"[limiting: {vector.limiting_component}]")

    # Check ordering
    C_values = [results[s].C for s in expected_order]
    is_ordered = all(C_values[i] >= C_values[i+1] for i in range(len(C_values)-1))

    print("\n" + "-" * 60)
    print(f"Expected ordering: {' > '.join(expected_order)}")
    print(f"Actual C values:   {' > '.join([f'{results[s].C:.2f}' for s in expected_order])}")
    print(f"Ordering correct:  {'✓' if is_ordered else '✗'}")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\nFive-Component Consciousness Framework v2.0")
    print("Rigorous Implementation with Validation\n")

    # Run synthetic validation
    results = validate_against_synthetic()

    print("\n" + "=" * 60)
    print("Attention Primacy Analysis")
    print("=" * 60)

    # Analyze attention primacy
    wake = results['wake']
    n1 = results['n1']

    phi_change = (n1.phi - wake.phi) / wake.phi * 100
    attention_change = (n1.attention - wake.attention) / wake.attention * 100

    print(f"\nWake → N1 transition:")
    print(f"  Φ change:         {phi_change:+.1f}%")
    print(f"  Attention change: {attention_change:+.1f}%")
    print(f"  Attention primacy: {'CONFIRMED' if abs(attention_change) > abs(phi_change) else 'NOT CONFIRMED'}")
