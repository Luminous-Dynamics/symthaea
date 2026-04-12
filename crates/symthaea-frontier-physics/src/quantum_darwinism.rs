// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quantum Darwinism — the emergence of classical reality.
//!
//! Quantum Darwinism (Zurek 2003) explains how classical objectivity emerges
//! from quantum mechanics: the environment selects "pointer states" that
//! survive decoherence, and multiple independent observers can independently
//! discover the same information about the system.
//!
//! This is directly relevant to consciousness: we perceive a classical world
//! precisely because quantum Darwinism filters quantum superpositions into
//! the pointer basis that our neural hardware evolved to detect.
//!
//! References:
//! - Zurek, W. H. (2003). Rev. Mod. Phys. 75, 715.
//! - Zurek, W. H. (2009). Nature Physics 5, 181.
//! - Brandão et al. (2015). Nature Communications 6, 7908.

/// Redundancy of information: how many copies of the system's state
/// are stored in the environment.
///
/// R_δ = N_env / f_δ where f_δ is the fraction of the environment
/// needed to determine the system state to accuracy δ.
///
/// Classical objectivity requires R >> 1: many independent observers
/// can learn the same information.
pub fn redundancy(n_env_fragments: usize, fraction_needed: f64) -> f64 {
    if fraction_needed <= 0.0 {
        return f64::INFINITY;
    }
    n_env_fragments as f64 / fraction_needed
}

/// Mutual information plateau: the hallmark of quantum Darwinism.
///
/// I(S:F) as a function of fragment size f should show:
/// - Rise from 0 to H(S) at small f (encoding phase)
/// - Plateau at H(S) for a wide range of f (redundancy)
/// - Rise to 2H(S) at f → 1 (quantum correlations)
///
/// Returns true if the mutual information profile shows a plateau.
pub fn has_classical_plateau(
    mutual_info_vs_fraction: &[(f64, f64)], // (fraction, MI)
    system_entropy: f64,
    tolerance: f64,
) -> bool {
    let plateau_value = system_entropy;

    // Count how many points are within tolerance of the plateau
    let plateau_count = mutual_info_vs_fraction
        .iter()
        .filter(|(f, mi)| *f > 0.1 && *f < 0.9 && (mi - plateau_value).abs() < tolerance)
        .count();

    // Classical behavior: most of the curve is at the plateau
    let total = mutual_info_vs_fraction
        .iter()
        .filter(|(f, _)| *f > 0.1 && *f < 0.9)
        .count();

    if total == 0 {
        return false;
    }
    (plateau_count as f64 / total as f64) > 0.5
}

/// Pointer states: eigenstates of the system-environment interaction.
/// These are the states that survive decoherence — the "fittest" states
/// in the quantum Darwinian selection process.
///
/// For a system coupled to an environment via H_SE = Σ |s_i⟩⟨s_i| ⊗ B_i,
/// the pointer states are {|s_i⟩}.
///
/// Returns pointer state overlaps with given state: |⟨s_i|ψ⟩|²
pub fn pointer_state_decomposition(
    state_re: &[f64],
    state_im: &[f64],
    pointer_states: &[Vec<f64>], // pointer_states[i][j] = ⟨j|s_i⟩ (real)
) -> Vec<f64> {
    let mut overlaps = Vec::with_capacity(pointer_states.len());

    for ps in pointer_states {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, &ps_j) in ps.iter().enumerate() {
            re += ps_j * state_re[j];
            im += ps_j * state_im[j];
        }
        overlaps.push(re * re + im * im);
    }

    overlaps
}

/// Decoherence rate for a superposition of pointer states.
/// Γ_ij = γ × |⟨E_i|E_j⟩|² where |E_i⟩ are the environment states
/// correlated with pointer states |s_i⟩.
///
/// For orthogonal environment states: Γ_ij = γ (1 - δ_ij)
/// Off-diagonal elements of ρ_S decay at rate Γ.
pub fn decoherence_rate(coupling_strength: f64, env_overlap: f64) -> f64 {
    coupling_strength * (1.0 - env_overlap * env_overlap)
}

/// Quantum discord: the quantum part of correlations.
/// Discord = I(S:E) - J(S:E) where J is the classical mutual information.
///
/// For a classically correlated state (after decoherence): Discord → 0
/// For a quantum state (before decoherence): Discord > 0
///
/// This quantifies "quantumness" of correlations.
pub fn quantum_discord_estimate(total_mutual_info: f64, classical_mutual_info: f64) -> f64 {
    (total_mutual_info - classical_mutual_info).max(0.0)
}

/// Einselection (environment-induced superselection): the process by which
/// the environment selects the preferred basis.
///
/// The einselection rate tells us how fast superpositions collapse into
/// the pointer basis. For a system of size N coupled to an environment
/// of size M: rate ∝ M × g² where g is the coupling.
pub fn einselection_rate(n_env_modes: usize, coupling: f64) -> f64 {
    n_env_modes as f64 * coupling * coupling
}

/// Quantum-to-classical transition parameter.
/// χ = (system size) × (coupling)² × (environment size) / (system energy gap)²
///
/// χ >> 1: classical behavior (rapid decoherence)
/// χ << 1: quantum behavior (coherence preserved)
pub fn quantum_classical_parameter(
    n_system: usize,
    n_environment: usize,
    coupling: f64,
    energy_gap: f64,
) -> f64 {
    if energy_gap.abs() < 1e-20 {
        return f64::INFINITY;
    }
    n_system as f64 * coupling * coupling * n_environment as f64 / (energy_gap * energy_gap)
}

/// Objectivity criterion: is the system's state objectively observable?
/// Requires: (1) High redundancy R >> 1, and (2) Low discord.
pub fn is_objective(redundancy: f64, discord: f64, threshold: f64) -> bool {
    redundancy > 10.0 && discord < threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_redundancy_classical() {
        // 1000 environment fragments, need 1% to determine state
        let r = redundancy(1000, 0.01);
        assert!((r - 100000.0).abs() < 1.0, "R = {:.0}", r);
    }

    #[test]
    fn test_plateau_detection() {
        let h_s = 1.0; // System entropy
                       // Classical: plateau at H(S) for most fractions
        let profile: Vec<(f64, f64)> = (1..=10)
            .map(|i| {
                let f = i as f64 / 10.0;
                let mi = if f < 0.1 {
                    f * 10.0 * h_s
                } else if f > 0.9 {
                    h_s + f * h_s
                } else {
                    h_s
                };
                (f, mi)
            })
            .collect();

        assert!(has_classical_plateau(&profile, h_s, 0.1));
    }

    #[test]
    fn test_pointer_state_decomposition() {
        // State |ψ⟩ = |0⟩ in pointer basis {|0⟩, |1⟩}
        let state_re = vec![1.0, 0.0];
        let state_im = vec![0.0, 0.0];
        let pointer_states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let overlaps = pointer_state_decomposition(&state_re, &state_im, &pointer_states);
        assert!((overlaps[0] - 1.0).abs() < 1e-14);
        assert!(overlaps[1].abs() < 1e-14);
    }

    #[test]
    fn test_decoherence_orthogonal_env() {
        // Orthogonal environment: overlap = 0 → max decoherence
        let gamma = decoherence_rate(1.0, 0.0);
        assert!((gamma - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_decoherence_identical_env() {
        // Identical environment: overlap = 1 → no decoherence
        let gamma = decoherence_rate(1.0, 1.0);
        assert!(gamma.abs() < 1e-14);
    }

    #[test]
    fn test_quantum_classical_parameter() {
        // Large system + environment + strong coupling → classical
        let chi = quantum_classical_parameter(100, 10000, 0.1, 0.01);
        assert!(chi > 1.0, "Should be classical regime: χ = {:.2}", chi);

        // Small system + weak coupling → quantum
        let chi_q = quantum_classical_parameter(2, 10, 0.001, 1.0);
        assert!(chi_q < 1.0, "Should be quantum regime: χ = {:.6}", chi_q);
    }

    #[test]
    fn test_objectivity_requires_both() {
        assert!(is_objective(100.0, 0.01, 0.1)); // High R, low discord → objective
        assert!(!is_objective(1.0, 0.01, 0.1)); // Low R → not objective
        assert!(!is_objective(100.0, 0.5, 0.1)); // High discord → not objective
    }
}
