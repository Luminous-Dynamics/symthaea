//! Train-derived reference-state lookup for dynamic structural-bias RC.
//!
//! This module implements only the reference-library and raw-dot-product
//! selection primitive described by Liu et al. (Physical Review Research 8,
//! 023114, 2026). It does not implement an ESN, train reservoir weights, or
//! apply the learned recurrent bias matrix. Callers must build this library
//! from the training reference window only; validation/confirmatory samples
//! must never be inserted.

use std::error::Error;
use std::fmt;

const MAX_REFERENCE_STATES: usize = 65_536;
const MAX_STATE_DIMENSION: usize = 4_096;
const MAX_REFERENCE_STATE_CELLS: usize = 4_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceLibraryError {
    EmptyReferenceWindow,
    EmptyState,
    InconsistentStateDimension,
    StateDimensionTooLarge,
    TooManyReferenceStates,
    ReferenceLibraryTooLarge,
    NonFiniteReferenceValue,
    QueryDimensionMismatch,
    NonFiniteQueryValue,
    NonFiniteSimilarity,
    OutputDimensionMismatch,
}

impl fmt::Display for ReferenceLibraryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyReferenceWindow => write!(formatter, "training reference window is empty"),
            Self::EmptyState => write!(formatter, "reservoir reference state is empty"),
            Self::InconsistentStateDimension => {
                write!(formatter, "reference states have inconsistent dimensions")
            }
            Self::StateDimensionTooLarge => {
                write!(
                    formatter,
                    "reference state dimension exceeds the configured maximum"
                )
            }
            Self::TooManyReferenceStates => {
                write!(
                    formatter,
                    "reference state count exceeds the configured maximum"
                )
            }
            Self::ReferenceLibraryTooLarge => {
                write!(
                    formatter,
                    "reference state library exceeds the configured cell budget"
                )
            }
            Self::NonFiniteReferenceValue => {
                write!(formatter, "reference library contains a non-finite value")
            }
            Self::QueryDimensionMismatch => {
                write!(
                    formatter,
                    "query state dimension differs from the reference library"
                )
            }
            Self::NonFiniteQueryValue => {
                write!(formatter, "query state contains a non-finite value")
            }
            Self::NonFiniteSimilarity => {
                write!(formatter, "raw dot-product similarity is non-finite")
            }
            Self::OutputDimensionMismatch => {
                write!(formatter, "correction output has the wrong dimension")
            }
        }
    }
}

impl Error for ReferenceLibraryError {}

/// Match selected by maximum raw dot-product similarity.
///
/// correction is selected_reference_state minus query_state, the direction
/// used by the paper's dynamic-bias rule before multiplication by its learned
/// recurrent bias matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceMatch {
    pub reference_index: usize,
    pub raw_dot_product: f64,
    pub correction: Vec<f64>,
}

/// Compact row-major library of reservoir states captured during training.
///
/// This type intentionally does not implement Serde serialization. It is a
/// model-side structure built from a declared training reference window, not
/// an evaluator manifest or hidden-truth container.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingReferenceLibrary {
    state_dimension: usize,
    reference_count: usize,
    states: Vec<f64>,
}

impl TrainingReferenceLibrary {
    /// Capture a train-only reference window.
    ///
    /// The caller supplies reservoir states captured after the warm-up period
    /// and before readout fitting/prediction. The API cannot inspect provenance
    /// of its input slice; the experiment harness must bind its caller to the
    /// frozen training window and must not pass validation/test states.
    pub fn from_training_states(
        training_reference_states: &[Vec<f64>],
    ) -> Result<Self, ReferenceLibraryError> {
        if training_reference_states.is_empty() {
            return Err(ReferenceLibraryError::EmptyReferenceWindow);
        }
        if training_reference_states.len() > MAX_REFERENCE_STATES {
            return Err(ReferenceLibraryError::TooManyReferenceStates);
        }

        let first = training_reference_states
            .first()
            .ok_or(ReferenceLibraryError::EmptyReferenceWindow)?;
        if first.is_empty() {
            return Err(ReferenceLibraryError::EmptyState);
        }
        let state_dimension = first.len();
        if state_dimension > MAX_STATE_DIMENSION {
            return Err(ReferenceLibraryError::StateDimensionTooLarge);
        }
        if training_reference_states
            .len()
            .checked_mul(state_dimension)
            .is_none_or(|cells| cells > MAX_REFERENCE_STATE_CELLS)
        {
            return Err(ReferenceLibraryError::ReferenceLibraryTooLarge);
        }

        let mut states = Vec::with_capacity(training_reference_states.len() * state_dimension);
        for state in training_reference_states {
            if state.len() != state_dimension {
                return Err(ReferenceLibraryError::InconsistentStateDimension);
            }
            if state.iter().any(|value| !value.is_finite()) {
                return Err(ReferenceLibraryError::NonFiniteReferenceValue);
            }
            states.extend_from_slice(state);
        }

        Ok(Self {
            state_dimension,
            reference_count: training_reference_states.len(),
            states,
        })
    }

    pub fn state_dimension(&self) -> usize {
        self.state_dimension
    }

    pub fn reference_count(&self) -> usize {
        self.reference_count
    }

    pub fn reference_state(&self, index: usize) -> Option<&[f64]> {
        if index >= self.reference_count {
            return None;
        }
        let start = index * self.state_dimension;
        self.states.get(start..start + self.state_dimension)
    }

    /// Select the first state attaining the maximum raw dot product.
    ///
    /// No cosine normalization, Euclidean-distance transform, or softmax is
    /// applied. Ties preserve the earliest reference index. This matches the
    /// paper's stated argmax rule and makes selection deterministic.
    pub fn select_max_dot_product(
        &self,
        query_state: &[f64],
    ) -> Result<ReferenceMatch, ReferenceLibraryError> {
        let mut correction = vec![0.0; self.state_dimension];
        let (reference_index, raw_dot_product) =
            self.select_max_dot_product_into(query_state, &mut correction)?;
        Ok(ReferenceMatch {
            reference_index,
            raw_dot_product,
            correction,
        })
    }

    /// Write selected_reference minus query_state to a caller-provided buffer.
    /// Returns (reference_index, raw_dot_product) without allocating.
    pub fn select_max_dot_product_into(
        &self,
        query_state: &[f64],
        correction_output: &mut [f64],
    ) -> Result<(usize, f64), ReferenceLibraryError> {
        if query_state.len() != self.state_dimension {
            return Err(ReferenceLibraryError::QueryDimensionMismatch);
        }
        if correction_output.len() != self.state_dimension {
            return Err(ReferenceLibraryError::OutputDimensionMismatch);
        }
        if query_state.iter().any(|value| !value.is_finite()) {
            return Err(ReferenceLibraryError::NonFiniteQueryValue);
        }

        let mut best_index = 0usize;
        let mut best_similarity = f64::NEG_INFINITY;
        for index in 0..self.reference_count {
            let start = index * self.state_dimension;
            let reference = &self.states[start..start + self.state_dimension];
            let similarity = query_state
                .iter()
                .zip(reference.iter())
                .fold(0.0, |sum, (query, stored)| query.mul_add(*stored, sum));
            if !similarity.is_finite() {
                return Err(ReferenceLibraryError::NonFiniteSimilarity);
            }
            if similarity > best_similarity {
                best_similarity = similarity;
                best_index = index;
            }
        }

        let start = best_index * self.state_dimension;
        let reference = &self.states[start..start + self.state_dimension];
        for ((output, stored), query) in correction_output
            .iter_mut()
            .zip(reference.iter())
            .zip(query_state.iter())
        {
            *output = stored - query;
            if !output.is_finite() {
                return Err(ReferenceLibraryError::NonFiniteSimilarity);
            }
        }
        Ok((best_index, best_similarity))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_by_raw_dot_product_not_cosine_similarity() {
        let library =
            TrainingReferenceLibrary::from_training_states(&[vec![2.0, 0.0], vec![0.2, 0.2]])
                .unwrap();

        // The first state has the larger raw dot product (2.0 vs 0.4),
        // while the second is more cosine-aligned with this query.
        let selected = library.select_max_dot_product(&[1.0, 1.0]).unwrap();
        assert_eq!(selected.reference_index, 0);
        assert_eq!(selected.raw_dot_product, 2.0);
        assert_eq!(selected.correction, vec![1.0, -1.0]);
    }

    #[test]
    fn maximum_dot_product_ties_choose_the_first_reference_state() {
        let library =
            TrainingReferenceLibrary::from_training_states(&[vec![1.0, 0.0], vec![0.0, 1.0]])
                .unwrap();
        let selected = library.select_max_dot_product(&[1.0, 1.0]).unwrap();
        assert_eq!(selected.reference_index, 0);
        assert_eq!(selected.raw_dot_product, 1.0);
        assert_eq!(selected.correction, vec![0.0, -1.0]);
    }

    #[test]
    fn reusable_output_buffer_matches_allocating_selection() {
        let library = TrainingReferenceLibrary::from_training_states(&[
            vec![0.0, 1.0, 2.0],
            vec![2.0, 0.0, 1.0],
            vec![1.0, 3.0, 0.0],
        ])
        .unwrap();
        let allocated = library.select_max_dot_product(&[1.0, 0.5, 0.25]).unwrap();
        let mut output = vec![0.0; 3];
        let (index, similarity) = library
            .select_max_dot_product_into(&[1.0, 0.5, 0.25], &mut output)
            .unwrap();

        assert_eq!(index, allocated.reference_index);
        assert_eq!(similarity, allocated.raw_dot_product);
        assert_eq!(output, allocated.correction);
    }

    #[test]
    fn rejects_empty_mixed_nonfinite_and_oversized_reference_windows() {
        assert_eq!(
            TrainingReferenceLibrary::from_training_states(&[]),
            Err(ReferenceLibraryError::EmptyReferenceWindow)
        );
        assert_eq!(
            TrainingReferenceLibrary::from_training_states(&[vec![]]),
            Err(ReferenceLibraryError::EmptyState)
        );
        assert_eq!(
            TrainingReferenceLibrary::from_training_states(&[vec![1.0], vec![1.0, 2.0]]),
            Err(ReferenceLibraryError::InconsistentStateDimension)
        );
        assert_eq!(
            TrainingReferenceLibrary::from_training_states(&[vec![f64::NAN]]),
            Err(ReferenceLibraryError::NonFiniteReferenceValue)
        );
        assert_eq!(
            TrainingReferenceLibrary::from_training_states(&vec![
                vec![0.0];
                MAX_REFERENCE_STATES + 1
            ]),
            Err(ReferenceLibraryError::TooManyReferenceStates)
        );
    }

    #[test]
    fn rejects_query_shape_and_nonfinite_values() {
        let library = TrainingReferenceLibrary::from_training_states(&[vec![1.0, 0.0]]).unwrap();
        assert_eq!(
            library.select_max_dot_product(&[1.0]),
            Err(ReferenceLibraryError::QueryDimensionMismatch)
        );
        assert_eq!(
            library.select_max_dot_product(&[1.0, f64::INFINITY]),
            Err(ReferenceLibraryError::NonFiniteQueryValue)
        );

        let mut wrong_output = vec![0.0];
        assert_eq!(
            library.select_max_dot_product_into(&[1.0, 0.0], &mut wrong_output),
            Err(ReferenceLibraryError::OutputDimensionMismatch)
        );
    }

    #[test]
    fn rejects_nonfinite_dot_product_overflow() {
        let library =
            TrainingReferenceLibrary::from_training_states(&[vec![1e308, 1e308]]).unwrap();
        assert_eq!(
            library.select_max_dot_product(&[1e308, 1e308]),
            Err(ReferenceLibraryError::NonFiniteSimilarity)
        );
    }
}
