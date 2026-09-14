// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multiscale causal analysis with explicit, auditable coarse-graining.
//!
//! GEOM-002A measures how effective information changes across declared scales.
//! It does not interpret a positive macro advantage as consciousness, downward
//! causation, or evidence for any particular ontology.
//!
//! Each coarse TPM is derived from its parent TPM. For a macrostate A, the
//! intervention distribution is uniform over the parent states assigned to A.
//! Parent transition probabilities are then aggregated by destination
//! macrostate. This makes the coarse-graining semantics explicit and prevents
//! comparisons between unrelated transition systems.

use super::causal_emergence::{degeneracy, determinism, effective_information};
use std::fmt;

const ROW_SUM_TOLERANCE: f64 = 1.0e-9;

/// Errors that invalidate a multiscale causal measurement.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiscaleCausalError {
    EmptyLabel,
    TooFewFineStates { observed: usize },
    EmptyTpm,
    NonSquareTpm { row: usize, expected: usize, observed: usize },
    NonFiniteProbability { row: usize, column: usize },
    NegativeProbability { row: usize, column: usize, value: f64 },
    RowNotStochastic { row: usize, sum: f64 },
    EmptyCoarseGraining,
    AssignmentLengthMismatch { expected: usize, observed: usize },
    InvalidMacroState { fine_state: usize, macro_state: usize, macro_states: usize },
    EmptyMacroState { macro_state: usize },
    NonReducingCoarseGraining { parent_states: usize, macro_states: usize },
    InvalidDecisionThreshold { value: f64 },
}

impl fmt::Display for MultiscaleCausalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLabel => write!(f, "scale labels must not be empty"),
            Self::TooFewFineStates { observed } => {
                write!(f, "finest TPM requires at least two states; observed {observed}")
            }
            Self::EmptyTpm => write!(f, "transition probability matrix must not be empty"),
            Self::NonSquareTpm { row, expected, observed } => write!(
                f,
                "TPM row {row} has length {observed}; expected square width {expected}"
            ),
            Self::NonFiniteProbability { row, column } => {
                write!(f, "TPM[{row}][{column}] must be finite")
            }
            Self::NegativeProbability { row, column, value } => write!(
                f,
                "TPM[{row}][{column}] must be non-negative; observed {value}"
            ),
            Self::RowNotStochastic { row, sum } => write!(
                f,
                "TPM row {row} must sum to 1 within tolerance; observed {sum}"
            ),
            Self::EmptyCoarseGraining => write!(f, "coarse-graining assignment must not be empty"),
            Self::AssignmentLengthMismatch { expected, observed } => write!(
                f,
                "coarse-graining assignment length {observed}; expected {expected} parent states"
            ),
            Self::InvalidMacroState { fine_state, macro_state, macro_states } => write!(
                f,
                "fine state {fine_state} maps to macro state {macro_state}, outside 0..{macro_states}"
            ),
            Self::EmptyMacroState { macro_state } => {
                write!(f, "macro state {macro_state} has no constituent parent states")
            }
            Self::NonReducingCoarseGraining { parent_states, macro_states } => write!(
                f,
                "coarse graining must reduce state count: parent={parent_states}, macro={macro_states}"
            ),
            Self::InvalidDecisionThreshold { value } => write!(
                f,
                "minimum causal-advantage threshold must be finite and non-negative; observed {value}"
            ),
        }
    }
}

impl std::error::Error for MultiscaleCausalError {}

/// A declared map from parent states to a smaller set of macrostates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoarseGrainingSpec {
    pub label: String,
    /// `assignment[parent_state] = macro_state`.
    pub assignment: Vec<usize>,
}

impl CoarseGrainingSpec {
    pub fn new(label: impl Into<String>, assignment: Vec<usize>) -> Self {
        Self {
            label: label.into(),
            assignment,
        }
    }
}

/// Preregistered criterion for reporting a causal advantage.
///
/// Passing this rule means only that some declared coarser scale has effective
/// information exceeding the finest scale by at least the declared number of
/// bits. It is not a consciousness classifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalAdvantageRule {
    pub min_advantage_bits: f64,
}

impl CausalAdvantageRule {
    pub fn new(min_advantage_bits: f64) -> Result<Self, MultiscaleCausalError> {
        if !min_advantage_bits.is_finite() || min_advantage_bits < 0.0 {
            return Err(MultiscaleCausalError::InvalidDecisionThreshold {
                value: min_advantage_bits,
            });
        }
        Ok(Self { min_advantage_bits })
    }
}

/// Measurements for one scale in a multiscale causal sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalScaleMetrics {
    pub label: String,
    pub state_count: usize,
    pub effective_information_bits: f64,
    pub determinism: f64,
    pub degeneracy: f64,
    /// EI(scale) - EI(previous scale). Zero at the finest scale.
    pub gain_vs_previous_bits: f64,
    /// EI(scale) - EI(finest scale). Zero at the finest scale.
    pub advantage_vs_finest_bits: f64,
}

/// Preregistered-rule result for a completed sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalAdvantageDecision {
    pub criterion_met: bool,
    pub peak_scale_index: usize,
    pub peak_advantage_bits: f64,
    pub min_advantage_bits: f64,
}

/// Complete multiscale measurement result.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiscaleCausalSweep {
    pub scales: Vec<CausalScaleMetrics>,
    pub decision: CausalAdvantageDecision,
}

/// Validate that a TPM is finite, square, non-negative, and row stochastic.
pub fn validate_tpm(tpm: &[Vec<f64>]) -> Result<(), MultiscaleCausalError> {
    if tpm.is_empty() {
        return Err(MultiscaleCausalError::EmptyTpm);
    }

    let n = tpm.len();
    for (row_index, row) in tpm.iter().enumerate() {
        if row.len() != n {
            return Err(MultiscaleCausalError::NonSquareTpm {
                row: row_index,
                expected: n,
                observed: row.len(),
            });
        }

        let mut sum = 0.0;
        for (column_index, &probability) in row.iter().enumerate() {
            if !probability.is_finite() {
                return Err(MultiscaleCausalError::NonFiniteProbability {
                    row: row_index,
                    column: column_index,
                });
            }
            if probability < 0.0 {
                return Err(MultiscaleCausalError::NegativeProbability {
                    row: row_index,
                    column: column_index,
                    value: probability,
                });
            }
            sum += probability;
        }

        if (sum - 1.0).abs() > ROW_SUM_TOLERANCE {
            return Err(MultiscaleCausalError::RowNotStochastic {
                row: row_index,
                sum,
            });
        }
    }

    Ok(())
}

/// Derive a coarse TPM using uniform interventions within each macrostate.
pub fn coarse_grain_tpm(
    parent_tpm: &[Vec<f64>],
    assignment: &[usize],
) -> Result<Vec<Vec<f64>>, MultiscaleCausalError> {
    validate_tpm(parent_tpm)?;

    let parent_states = parent_tpm.len();
    if assignment.is_empty() {
        return Err(MultiscaleCausalError::EmptyCoarseGraining);
    }
    if assignment.len() != parent_states {
        return Err(MultiscaleCausalError::AssignmentLengthMismatch {
            expected: parent_states,
            observed: assignment.len(),
        });
    }

    let macro_states = assignment.iter().copied().max().map_or(0, |max| max + 1);
    if macro_states == 0 {
        return Err(MultiscaleCausalError::EmptyCoarseGraining);
    }
    if macro_states >= parent_states {
        return Err(MultiscaleCausalError::NonReducingCoarseGraining {
            parent_states,
            macro_states,
        });
    }

    let mut members = vec![Vec::new(); macro_states];
    for (fine_state, &macro_state) in assignment.iter().enumerate() {
        if macro_state >= macro_states {
            return Err(MultiscaleCausalError::InvalidMacroState {
                fine_state,
                macro_state,
                macro_states,
            });
        }
        members[macro_state].push(fine_state);
    }

    for (macro_state, group) in members.iter().enumerate() {
        if group.is_empty() {
            return Err(MultiscaleCausalError::EmptyMacroState { macro_state });
        }
    }

    let mut coarse = vec![vec![0.0; macro_states]; macro_states];
    for macro_from in 0..macro_states {
        let intervention_weight = 1.0 / members[macro_from].len() as f64;
        for &fine_from in &members[macro_from] {
            for fine_to in 0..parent_states {
                let macro_to = assignment[fine_to];
                coarse[macro_from][macro_to] +=
                    intervention_weight * parent_tpm[fine_from][fine_to];
            }
        }
    }

    validate_tpm(&coarse)?;
    Ok(coarse)
}

fn metrics_for_scale(
    label: &str,
    tpm: &[Vec<f64>],
    previous_ei: Option<f64>,
    finest_ei: f64,
) -> CausalScaleMetrics {
    let ei = effective_information(tpm);
    CausalScaleMetrics {
        label: label.to_owned(),
        state_count: tpm.len(),
        effective_information_bits: ei,
        determinism: determinism(tpm),
        degeneracy: degeneracy(tpm),
        gain_vs_previous_bits: previous_ei.map_or(0.0, |previous| ei - previous),
        advantage_vs_finest_bits: ei - finest_ei,
    }
}

/// Run a fine-to-coarse causal sweep using explicit recursive coarse-grainings.
pub fn analyze_multiscale_causality(
    finest_label: &str,
    finest_tpm: &[Vec<f64>],
    coarse_grainings: &[CoarseGrainingSpec],
    rule: CausalAdvantageRule,
) -> Result<MultiscaleCausalSweep, MultiscaleCausalError> {
    if finest_label.trim().is_empty() {
        return Err(MultiscaleCausalError::EmptyLabel);
    }
    validate_tpm(finest_tpm)?;
    if finest_tpm.len() < 2 {
        return Err(MultiscaleCausalError::TooFewFineStates {
            observed: finest_tpm.len(),
        });
    }

    let finest_ei = effective_information(finest_tpm);
    let mut scales = vec![metrics_for_scale(finest_label, finest_tpm, None, finest_ei)];
    let mut current_tpm = finest_tpm.to_vec();
    let mut previous_ei = finest_ei;

    for spec in coarse_grainings {
        if spec.label.trim().is_empty() {
            return Err(MultiscaleCausalError::EmptyLabel);
        }
        let coarse_tpm = coarse_grain_tpm(&current_tpm, &spec.assignment)?;
        let metrics = metrics_for_scale(&spec.label, &coarse_tpm, Some(previous_ei), finest_ei);
        previous_ei = metrics.effective_information_bits;
        scales.push(metrics);
        current_tpm = coarse_tpm;
    }

    let mut peak_scale_index = 0usize;
    for index in 1..scales.len() {
        if scales[index].effective_information_bits
            > scales[peak_scale_index].effective_information_bits
        {
            peak_scale_index = index;
        }
    }

    let peak_advantage_bits = scales[peak_scale_index].advantage_vs_finest_bits;
    let criterion_met = peak_scale_index > 0 && peak_advantage_bits >= rule.min_advantage_bits;

    Ok(MultiscaleCausalSweep {
        scales,
        decision: CausalAdvantageDecision {
            criterion_met,
            peak_scale_index,
            peak_advantage_bits,
            min_advantage_bits: rule.min_advantage_bits,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1.0e-10;

    fn uniform_tpm(states: usize) -> Vec<Vec<f64>> {
        vec![vec![1.0 / states as f64; states]; states]
    }

    #[test]
    fn rejects_malformed_tpms() {
        assert!(matches!(
            validate_tpm(&[vec![0.5, 0.5], vec![1.0]]),
            Err(MultiscaleCausalError::NonSquareTpm { .. })
        ));
        assert!(matches!(
            validate_tpm(&[vec![0.8, 0.3], vec![0.5, 0.5]]),
            Err(MultiscaleCausalError::RowNotStochastic { .. })
        ));
        assert!(matches!(
            validate_tpm(&[vec![1.1, -0.1], vec![0.5, 0.5]]),
            Err(MultiscaleCausalError::NegativeProbability { .. })
        ));
    }

    #[test]
    fn uniform_null_has_zero_effective_information_at_every_scale() {
        let fine = uniform_tpm(4);
        let specs = [CoarseGrainingSpec::new("macro", vec![0, 0, 1, 1])];
        let rule = CausalAdvantageRule::new(0.05).expect("valid rule");
        let sweep = analyze_multiscale_causality("fine", &fine, &specs, rule)
            .expect("valid sweep");

        assert_eq!(sweep.scales.len(), 2);
        assert!(sweep.scales.iter().all(|scale| {
            scale.effective_information_bits.abs() < TOLERANCE
                && scale.advantage_vs_finest_bits.abs() < TOLERANCE
        }));
        assert!(!sweep.decision.criterion_met);
    }

    #[test]
    fn deterministic_identity_has_expected_fine_scale_information() {
        let fine = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let specs = [CoarseGrainingSpec::new("macro", vec![0, 0, 1, 1])];
        let rule = CausalAdvantageRule::new(0.0).expect("valid rule");
        let sweep = analyze_multiscale_causality("fine", &fine, &specs, rule)
            .expect("valid sweep");

        assert!((sweep.scales[0].effective_information_bits - 2.0).abs() < TOLERANCE);
        assert!((sweep.scales[1].effective_information_bits - 1.0).abs() < TOLERANCE);
        assert!(!sweep.decision.criterion_met);
    }

    #[test]
    fn asymmetric_degeneracy_can_produce_a_coarse_causal_advantage() {
        // States 0,1,2 form macro A and all transition to state 3.
        // State 3 forms macro B and transitions uniformly into macro A.
        // Under a uniform intervention over four fine states, degeneracy lowers
        // fine-scale EI. The declared two-state macro dynamics are deterministic.
        let fine = vec![
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
        ];
        let specs = [CoarseGrainingSpec::new("macro", vec![0, 0, 0, 1])];
        let rule = CausalAdvantageRule::new(0.1).expect("valid rule");
        let sweep = analyze_multiscale_causality("fine", &fine, &specs, rule)
            .expect("valid sweep");

        assert!(sweep.scales[0].effective_information_bits < 1.0);
        assert!((sweep.scales[1].effective_information_bits - 1.0).abs() < TOLERANCE);
        assert!(sweep.decision.peak_advantage_bits > 0.1);
        assert_eq!(sweep.decision.peak_scale_index, 1);
        assert!(sweep.decision.criterion_met);
    }

    #[test]
    fn coarse_graining_must_be_surjective_and_state_reducing() {
        let fine = uniform_tpm(4);

        assert!(matches!(
            coarse_grain_tpm(&fine, &[0, 0, 2, 2]),
            Err(MultiscaleCausalError::EmptyMacroState { macro_state: 1 })
        ));
        assert!(matches!(
            coarse_grain_tpm(&fine, &[0, 1, 2, 3]),
            Err(MultiscaleCausalError::NonReducingCoarseGraining { .. })
        ));
        assert!(matches!(
            coarse_grain_tpm(&fine, &[0, 0]),
            Err(MultiscaleCausalError::AssignmentLengthMismatch { .. })
        ));
    }

    #[test]
    fn decision_threshold_is_explicit_and_fail_closed() {
        assert!(CausalAdvantageRule::new(-0.1).is_err());
        assert!(CausalAdvantageRule::new(f64::NAN).is_err());
        assert_eq!(
            CausalAdvantageRule::new(0.25).expect("valid rule").min_advantage_bits,
            0.25
        );
    }
}
