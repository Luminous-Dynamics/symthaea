//! Dynamic structural-bias reservoir transition kernel.
//!
//! Implements the recurrent state update once the train-reference library has
//! selected its maximum raw-dot-product match. It is not a full ESN trainer.

use std::error::Error;
use std::fmt;

use crate::structural_bias::{ReferenceLibraryError, TrainingReferenceLibrary};

const MAX_RESERVOIR_DIMENSION: usize = 2_048;
const MAX_INPUT_DIMENSION: usize = 4_096;
const MAX_WEIGHT_CELLS: usize = 4_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReservoirCellError {
    InvalidDimension,
    WeightMatrixSizeMismatch,
    WeightBudgetExceeded,
    NonFiniteWeight,
    InvalidLeakRate,
    StateDimensionMismatch,
    InputDimensionMismatch,
    NonFiniteState,
    NonFiniteInput,
    ReferenceLibraryDimensionMismatch,
    ReferenceLookup(ReferenceLibraryError),
    NonFiniteTransition,
}

impl fmt::Display for ReservoirCellError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension => write!(formatter, "reservoir or input dimension is invalid"),
            Self::WeightMatrixSizeMismatch => write!(formatter, "reservoir weight matrix has wrong size"),
            Self::WeightBudgetExceeded => write!(formatter, "reservoir weights exceed configured cell budget"),
            Self::NonFiniteWeight => write!(formatter, "reservoir weight is non-finite"),
            Self::InvalidLeakRate => write!(formatter, "leak rate must be finite and in (0, 1]"),
            Self::StateDimensionMismatch => write!(formatter, "current reservoir state has wrong dimension"),
            Self::InputDimensionMismatch => write!(formatter, "reservoir input has wrong dimension"),
            Self::NonFiniteState => write!(formatter, "current reservoir state is non-finite"),
            Self::NonFiniteInput => write!(formatter, "reservoir input is non-finite"),
            Self::ReferenceLibraryDimensionMismatch => write!(formatter, "reference library has wrong state dimension"),
            Self::ReferenceLookup(error) => write!(formatter, "{error}"),
            Self::NonFiniteTransition => write!(formatter, "reservoir transition is non-finite"),
        }
    }
}

impl Error for ReservoirCellError {}

impl From<ReferenceLibraryError> for ReservoirCellError {
    fn from(error: ReferenceLibraryError) -> Self {
        Self::ReferenceLookup(error)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReservoirTransition {
    pub next_state: Vec<f64>,
    pub selected_reference_index: usize,
    pub selected_raw_dot_product: f64,
    pub correction_norm: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralBiasReservoirCell {
    reservoir_dimension: usize,
    input_dimension: usize,
    leak_rate: f64,
    recurrent_weights: Vec<f64>,
    input_weights: Vec<f64>,
}

impl StructuralBiasReservoirCell {
    /// Matrices are row-major: recurrent N-by-N and input N-by-M.
    pub fn new(
        reservoir_dimension: usize,
        input_dimension: usize,
        leak_rate: f64,
        recurrent_weights: Vec<f64>,
        input_weights: Vec<f64>,
    ) -> Result<Self, ReservoirCellError> {
        if reservoir_dimension == 0
            || reservoir_dimension > MAX_RESERVOIR_DIMENSION
            || input_dimension == 0
            || input_dimension > MAX_INPUT_DIMENSION
        {
            return Err(ReservoirCellError::InvalidDimension);
        }
        if !leak_rate.is_finite() || leak_rate <= 0.0 || leak_rate > 1.0 {
            return Err(ReservoirCellError::InvalidLeakRate);
        }

        let recurrent_cells = reservoir_dimension
            .checked_mul(reservoir_dimension)
            .ok_or(ReservoirCellError::WeightBudgetExceeded)?;
        let input_cells = reservoir_dimension
            .checked_mul(input_dimension)
            .ok_or(ReservoirCellError::WeightBudgetExceeded)?;
        let total_cells = recurrent_cells
            .checked_add(input_cells)
            .ok_or(ReservoirCellError::WeightBudgetExceeded)?;
        if total_cells > MAX_WEIGHT_CELLS {
            return Err(ReservoirCellError::WeightBudgetExceeded);
        }
        if recurrent_weights.len() != recurrent_cells || input_weights.len() != input_cells {
            return Err(ReservoirCellError::WeightMatrixSizeMismatch);
        }
        if recurrent_weights
            .iter()
            .chain(input_weights.iter())
            .any(|weight| !weight.is_finite())
        {
            return Err(ReservoirCellError::NonFiniteWeight);
        }

        Ok(Self {
            reservoir_dimension,
            input_dimension,
            leak_rate,
            recurrent_weights,
            input_weights,
        })
    }

    pub fn reservoir_dimension(&self) -> usize {
        self.reservoir_dimension
    }

    pub fn input_dimension(&self) -> usize {
        self.input_dimension
    }

    pub fn leak_rate(&self) -> f64 {
        self.leak_rate
    }

    /// Apply one update using Wr*r + Win*u + Wr*(reference - r) inside tanh.
    pub fn step(
        &self,
        current_state: &[f64],
        input: &[f64],
        reference_library: &TrainingReferenceLibrary,
    ) -> Result<ReservoirTransition, ReservoirCellError> {
        if current_state.len() != self.reservoir_dimension {
            return Err(ReservoirCellError::StateDimensionMismatch);
        }
        if input.len() != self.input_dimension {
            return Err(ReservoirCellError::InputDimensionMismatch);
        }
        if reference_library.state_dimension() != self.reservoir_dimension {
            return Err(ReservoirCellError::ReferenceLibraryDimensionMismatch);
        }
        if current_state.iter().any(|value| !value.is_finite()) {
            return Err(ReservoirCellError::NonFiniteState);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(ReservoirCellError::NonFiniteInput);
        }

        let mut correction = vec![0.0; self.reservoir_dimension];
        let (selected_reference_index, selected_raw_dot_product) =
            reference_library.select_max_dot_product_into(current_state, &mut correction)?;
        let correction_norm = stable_l2_norm(&correction);
        if !correction_norm.is_finite() {
            return Err(ReservoirCellError::NonFiniteTransition);
        }

        let mut next_state = Vec::with_capacity(self.reservoir_dimension);
        for row in 0..self.reservoir_dimension {
            let recurrent_start = row * self.reservoir_dimension;
            let recurrent_row =
                &self.recurrent_weights[recurrent_start..recurrent_start + self.reservoir_dimension];
            let recurrent_drive = recurrent_row
                .iter()
                .zip(current_state.iter())
                .fold(0.0, |sum, (weight, value)| weight.mul_add(*value, sum));
            let structural_bias_drive = recurrent_row
                .iter()
                .zip(correction.iter())
                .fold(0.0, |sum, (weight, value)| weight.mul_add(*value, sum));

            let input_start = row * self.input_dimension;
            let input_row = &self.input_weights[input_start..input_start + self.input_dimension];
            let input_drive = input_row
                .iter()
                .zip(input.iter())
                .fold(0.0, |sum, (weight, value)| weight.mul_add(*value, sum));

            let preactivation = recurrent_drive + input_drive + structural_bias_drive;
            let next = (1.0 - self.leak_rate) * current_state[row]
                + self.leak_rate * preactivation.tanh();
            if !next.is_finite() {
                return Err(ReservoirCellError::NonFiniteTransition);
            }
            next_state.push(next);
        }

        Ok(ReservoirTransition {
            next_state,
            selected_reference_index,
            selected_raw_dot_product,
            correction_norm,
        })
    }
}

fn stable_l2_norm(values: &[f64]) -> f64 {
    let scale = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    if !scale.is_finite() {
        return f64::INFINITY;
    }
    let sum = values
        .iter()
        .map(|value| {
            let normalized = value / scale;
            normalized * normalized
        })
        .sum::<f64>();
    scale * sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn library(states: &[Vec<f64>]) -> TrainingReferenceLibrary {
        TrainingReferenceLibrary::from_training_states(states).unwrap()
    }

    #[test]
    fn transition_uses_recurrent_matrix_for_reference_difference() {
        let cell = StructuralBiasReservoirCell::new(1, 1, 0.25, vec![2.0], vec![3.0]).unwrap();
        let refs = library(&[vec![0.8], vec![-0.5]]);
        let result = cell.step(&[0.2], &[0.5], &refs).unwrap();

        let expected = 0.75 * 0.2 + 0.25 * (2.0 * 0.2 + 3.0 * 0.5 + 2.0 * (0.8 - 0.2)).tanh();
        assert_eq!(result.selected_reference_index, 0);
        assert!((result.selected_raw_dot_product - 0.16).abs() < 1e-12);
        assert!((result.correction_norm - 0.6).abs() < 1e-12);
        assert!((result.next_state[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn recurrent_state_plus_bias_equals_recurrent_matrix_times_reference() {
        let cell = StructuralBiasReservoirCell::new(
            2,
            1,
            0.5,
            vec![1.0, 2.0, -1.0, 0.5],
            vec![0.25, -0.75],
        )
        .unwrap();
        let current = [0.2, 0.4];
        let refs = library(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let selected = refs.select_max_dot_product(&current).unwrap();
        let result = cell.step(&current, &[0.3], &refs).unwrap();

        let reference = refs.reference_state(selected.reference_index).unwrap();
        let recurrent = [
            reference[0] + 2.0 * reference[1],
            -reference[0] + 0.5 * reference[1],
        ];
        for row in 0..2 {
            let input_drive = cell.input_weights[row] * 0.3;
            let expected = 0.5 * current[row] + 0.5 * (recurrent[row] + input_drive).tanh();
            assert!((result.next_state[row] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_invalid_dimensions_weights_and_leak_rate() {
        assert_eq!(
            StructuralBiasReservoirCell::new(0, 1, 0.5, vec![], vec![]),
            Err(ReservoirCellError::InvalidDimension)
        );
        assert_eq!(
            StructuralBiasReservoirCell::new(1, 1, 0.0, vec![1.0], vec![1.0]),
            Err(ReservoirCellError::InvalidLeakRate)
        );
        assert_eq!(
            StructuralBiasReservoirCell::new(1, 1, 0.5, vec![], vec![1.0]),
            Err(ReservoirCellError::WeightMatrixSizeMismatch)
        );
        assert_eq!(
            StructuralBiasReservoirCell::new(1, 1, 0.5, vec![f64::NAN], vec![1.0]),
            Err(ReservoirCellError::NonFiniteWeight)
        );
    }

    #[test]
    fn rejects_state_input_and_reference_dimension_mismatches() {
        let cell = StructuralBiasReservoirCell::new(2, 1, 0.5, vec![1.0; 4], vec![1.0; 2])
            .unwrap();
        let refs = library(&[vec![0.0, 1.0]]);
        assert_eq!(
            cell.step(&[0.0], &[1.0], &refs),
            Err(ReservoirCellError::StateDimensionMismatch)
        );
        assert_eq!(
            cell.step(&[0.0, 1.0], &[1.0, 2.0], &refs),
            Err(ReservoirCellError::InputDimensionMismatch)
        );
        let wrong_refs = library(&[vec![1.0]]);
        assert_eq!(
            cell.step(&[0.0, 1.0], &[1.0], &wrong_refs),
            Err(ReservoirCellError::ReferenceLibraryDimensionMismatch)
        );
    }

    #[test]
    fn rejects_nonfinite_state_and_input() {
        let cell = StructuralBiasReservoirCell::new(1, 1, 0.5, vec![1.0], vec![1.0]).unwrap();
        let refs = library(&[vec![0.0]]);
        assert_eq!(
            cell.step(&[f64::NAN], &[0.0], &refs),
            Err(ReservoirCellError::NonFiniteState)
        );
        assert_eq!(
            cell.step(&[0.0], &[f64::INFINITY], &refs),
            Err(ReservoirCellError::NonFiniteInput)
        );
    }
}
