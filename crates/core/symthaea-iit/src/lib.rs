// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A small, explicit causal-model substrate for a future Rust-native IIT engine.
//!
//! # Scope
//!
//! Version 0.1 defines and validates a dense, binary, factored transition-probability
//! model. It deliberately does not calculate IIT phi, system phi_s, structure Phi,
//! a cause-effect structure, or consciousness. These are separate algorithms and
//! must not be inferred from a valid TPM.
//!
//! # Model contract
//!
//! * A state is a bool slice of length n_units.
//! * Unit 0 is the least-significant bit in the state index.
//! * The TPM is row-major [past_state_index][next_unit].
//! * Each TPM cell is P(next_unit = 1 | entire past state).
//! * The factored transition probability assumes next units are conditionally
//!   independent given the current state, as in a state-by-node TPM.
//! * Connectivity is an explicit, directed matrix [source_unit][target_unit].
//!   It is input metadata, not inferred from the TPM. The caller is responsible
//!   for ensuring declared connectivity and the transition model describe the
//!   same system.
//! * The model digest covers the exact dense TPM and connectivity matrix, not an
//!   evaluation state or an IIT formalism. The evaluation digest binds model bytes
//!   to one state. Neither digest proves that the model is scientifically correct.
//!
//! This crate is intentionally small before implementing repertoire construction,
//! partitions, minimum-information searches, or formal IIT quantities. Those stages
//! need formalism-specific definitions and independent golden fixtures.

use sha2::{Digest, Sha256};
use thiserror::Error;

/// Maximum dense binary units accepted by this in-memory v0.1 representation.
///
/// This is a representation guard, not an IIT tractability claim. A dense binary
/// TPM stores 2^n * n probabilities, so storage grows exponentially with n.
pub const MAX_DENSE_UNITS: usize = 16;

const MODEL_MAGIC: &[u8] = b"SYMTHAEA-IIT-BINARY-TPM\0v0.1\0";
const EVALUATION_MAGIC: &[u8] = b"SYMTHAEA-IIT-EVALUATION\0v0.1\0";
const ANALYSIS_REQUEST_MAGIC: &[u8] = b"SYMTHAEA-IIT-ANALYSIS-REQUEST\0v0.1\0";

/// Errors caused by invalid model inputs or invalid state requests.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ModelError {
    #[error("unit count must be in 1..=16; got {units}")]
    InvalidUnitCount { units: usize },

    #[error("TPM length mismatch: expected {expected} state-by-unit values, got {actual}")]
    TransitionTableLength { expected: usize, actual: usize },

    #[error("connectivity length mismatch: expected {expected} directed entries, got {actual}")]
    ConnectivityLength { expected: usize, actual: usize },

    #[error("TPM value at flat index {index} must be finite and in [0, 1]; got {value}")]
    InvalidProbability { index: usize, value: f64 },

    #[error("state length mismatch: expected {expected} bits, got {actual}")]
    StateLength { expected: usize, actual: usize },

    #[error("unit index {index} is outside 0..{unit_count}")]
    UnitIndexOutOfRange { index: usize, unit_count: usize },

    #[error("quantity {quantity} is not supported by formalism {formalism}")]
    IncompatibleQuantity { quantity: &'static str, formalism: &'static str },

    #[error("connectivity omits causal influence {source_unit}->{target_unit} between states {state_without_source} and {state_with_source}")]
    MissingCausalEdge {
        source_unit: usize,
        target_unit: usize,
        state_without_source: usize,
        state_with_source: usize,
    },
}


/// Errors in probability-repertoire construction.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RepertoireError {
    #[error(transparent)]
    Model(#[from] ModelError),

    #[error("subset {subset} contains duplicate unit {unit}")]
    DuplicateSubsetUnit { subset: &'static str, unit: usize },

    #[error("cause repertoire has zero or non-finite likelihood mass")]
    UndefinedCauseRepertoire,
}

/// Dense, binary, factored transition-probability model.
///
/// transition_probabilities[state_index * n_units + unit] stores the probability
/// that unit is ON at the next step, given the complete current state indexed by
/// state_index. This format does not encode an IIT formalism or a value of Phi.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryCausalModel {
    n_units: usize,
    transition_probabilities: Vec<f64>,
    /// Row-major directed adjacency: connectivity[source * n_units + target].
    connectivity: Vec<bool>,
}

impl BinaryCausalModel {
    /// Construct a model after validating all shape and probability invariants.
    pub fn new(
        n_units: usize,
        transition_probabilities: Vec<f64>,
        connectivity: Vec<bool>,
    ) -> Result<Self, ModelError> {
        if !(1..=MAX_DENSE_UNITS).contains(&n_units) {
            return Err(ModelError::InvalidUnitCount { units: n_units });
        }

        // n_units is bounded above, so this shift and multiplication are safe.
        let state_count = 1usize << n_units;
        let expected_tpm_len = state_count * n_units;
        if transition_probabilities.len() != expected_tpm_len {
            return Err(ModelError::TransitionTableLength {
                expected: expected_tpm_len,
                actual: transition_probabilities.len(),
            });
        }

        let expected_connectivity_len = n_units * n_units;
        if connectivity.len() != expected_connectivity_len {
            return Err(ModelError::ConnectivityLength {
                expected: expected_connectivity_len,
                actual: connectivity.len(),
            });
        }

        for (index, value) in transition_probabilities.iter().copied().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(ModelError::InvalidProbability { index, value });
            }
        }

        // A declared connectivity matrix may include extra edges, but it must not
        // omit an influence present in the TPM. Use exact inequality as a
        // conservative rule: even a small nonzero numeric effect requires an edge.
        // State pairs differ only in the candidate source unit.
        for source_unit in 0..n_units {
            for target_unit in 0..n_units {
                if connectivity[source_unit * n_units + target_unit] {
                    continue;
                }
                for state_without_source in 0..state_count {
                    if ((state_without_source >> source_unit) & 1) != 0 {
                        continue;
                    }
                    let state_with_source = state_without_source | (1usize << source_unit);
                    let p_without = transition_probabilities
                        [state_without_source * n_units + target_unit];
                    let p_with = transition_probabilities
                        [state_with_source * n_units + target_unit];
                    if p_without != p_with {
                        return Err(ModelError::MissingCausalEdge {
                            source_unit,
                            target_unit,
                            state_without_source,
                            state_with_source,
                        });
                    }
                }
            }
        }

        Ok(Self {
            n_units,
            transition_probabilities,
            connectivity,
        })
    }

    /// Number of binary causal units in the model.
    pub fn n_units(&self) -> usize {
        self.n_units
    }

    /// Number of possible current states (2^n_units).
    pub fn n_states(&self) -> usize {
        1usize << self.n_units
    }

    /// Read the state-by-unit TPM in its documented row-major order.
    pub fn transition_probabilities(&self) -> &[f64] {
        &self.transition_probabilities
    }

    /// Read the declared directed connectivity matrix in source-major order.
    pub fn connectivity(&self) -> &[bool] {
        &self.connectivity
    }

    /// Whether the declared directed edge source -> target is present.
    pub fn has_edge(&self, source: usize, target: usize) -> Result<bool, ModelError> {
        self.check_unit(source)?;
        self.check_unit(target)?;
        Ok(self.connectivity[source * self.n_units + target])
    }

    /// State index with unit 0 as the least-significant bit.
    pub fn state_index(&self, state: &[bool]) -> Result<usize, ModelError> {
        self.check_state(state)?;
        Ok(state.iter().enumerate().fold(0usize, |index, (unit, on)| {
            if *on { index | (1usize << unit) } else { index }
        }))
    }

    /// Probability that target_unit is ON next, conditional on current_state.
    pub fn probability_next_one(
        &self,
        current_state: &[bool],
        target_unit: usize,
    ) -> Result<f64, ModelError> {
        self.check_unit(target_unit)?;
        let state_index = self.state_index(current_state)?;
        Ok(self.transition_probabilities[state_index * self.n_units + target_unit])
    }

    /// Probability of a complete next state under the factored TPM.
    ///
    /// This is the product of per-unit conditional probabilities. It is valid for
    /// a factored TPM with conditional independence of next units given the current
    /// state; it is not a general joint-TPM representation.
    pub fn transition_probability(
        &self,
        current_state: &[bool],
        next_state: &[bool],
    ) -> Result<f64, ModelError> {
        let current_index = self.state_index(current_state)?;
        self.check_state(next_state)?;
        let row_start = current_index * self.n_units;
        let probability = next_state.iter().enumerate().map(|(unit, next_on)| {
            let p_on = self.transition_probabilities[row_start + unit];
            if *next_on { p_on } else { 1.0 - p_on }
        }).product();
        Ok(probability)
    }


    /// Compute an effect repertoire using a product of per-purview-unit marginals.
    ///
    /// For each target unit, fix mechanism values and marginalize the non-mechanism
    /// units over a uniform maximum-entropy background. The full purview distribution
    /// is the product of these per-unit distributions, following PyPhi's factored
    /// repertoire convention. This deliberately does not preserve correlations induced
    /// solely by mixing over the same unconstrained hidden state.
    ///
    /// The result indexes future purview states with the first listed purview unit as
    /// the least-significant bit. This primitive does not compute mechanism phi.
    pub fn effect_repertoire(
        &self,
        mechanism: &[usize],
        current_state: &[bool],
        purview: &[usize],
    ) -> Result<Vec<f64>, RepertoireError> {
        self.check_state(current_state)?;
        self.validate_subset(mechanism, "mechanism")?;
        self.validate_subset(purview, "purview")?;

        let unconstrained_count = 1usize << (self.n_units - mechanism.len());
        let background_weight = 1.0 / unconstrained_count as f64;
        let mut per_unit_p_on = Vec::with_capacity(purview.len());

        for target_unit in purview.iter().copied() {
            let mut p_on = 0.0;
            for past_index in 0..self.n_states() {
                let mechanism_matches = mechanism.iter().all(|unit| {
                    (((past_index >> *unit) & 1) != 0) == current_state[*unit]
                });
                if mechanism_matches {
                    p_on += background_weight
                        * self.transition_probabilities[past_index * self.n_units + target_unit];
                }
            }
            per_unit_p_on.push(p_on);
        }

        let purview_state_count = 1usize << purview.len();
        let mut repertoire = vec![1.0; purview_state_count];
        for future_state in 0..purview_state_count {
            for (slot, p_on) in per_unit_p_on.iter().copied().enumerate() {
                let target_on = ((future_state >> slot) & 1) != 0;
                repertoire[future_state] *= if target_on { p_on } else { 1.0 - p_on };
            }
        }
        Ok(repertoire)
    }

    /// Compute a cause repertoire as a normalized product of single-unit cause repertoires.
    ///
    /// Each mechanism unit contributes a distribution over past purview states:
    /// likelihoods are obtained from the unit's TPM column under a uniform prior
    /// over full past states, marginalized to the requested purview, then normalized.
    /// The per-mechanism-unit distributions are combined by pointwise multiplication
    /// and normalized again. If a mechanism unit's observed state has zero likelihood
    /// under every past state, this returns an error rather than NaN.
    ///
    /// This uses the factored multi-unit cause-repertoire convention described in
    /// PyPhi's repertoire documentation; it is a primitive, not a full IIT calculation.
    pub fn cause_repertoire(
        &self,
        mechanism: &[usize],
        current_state: &[bool],
        purview: &[usize],
    ) -> Result<Vec<f64>, RepertoireError> {
        self.check_state(current_state)?;
        self.validate_subset(mechanism, "mechanism")?;
        self.validate_subset(purview, "purview")?;

        let purview_state_count = 1usize << purview.len();
        if mechanism.is_empty() {
            return Ok(vec![1.0 / purview_state_count as f64; purview_state_count]);
        }

        let mut repertoire = vec![1.0; purview_state_count];
        for mechanism_unit in mechanism.iter().copied() {
            let mut single_unit = vec![0.0; purview_state_count];
            for past_index in 0..self.n_states() {
                let purview_state = purview.iter().copied().enumerate().fold(
                    0usize,
                    |state_index, (slot, unit)| {
                        if ((past_index >> unit) & 1) != 0 {
                            state_index | (1usize << slot)
                        } else {
                            state_index
                        }
                    },
                );
                let p_on = self.transition_probabilities
                    [past_index * self.n_units + mechanism_unit];
                let likelihood = if current_state[mechanism_unit] {
                    p_on
                } else {
                    1.0 - p_on
                };
                single_unit[purview_state] += likelihood;
            }

            let single_mass: f64 = single_unit.iter().sum();
            if !single_mass.is_finite() || single_mass <= 0.0 {
                return Err(RepertoireError::UndefinedCauseRepertoire);
            }
            for probability in &mut single_unit {
                *probability /= single_mass;
            }
            for (joint, single) in repertoire.iter_mut().zip(single_unit) {
                *joint *= single;
            }
        }

        let mass: f64 = repertoire.iter().sum();
        if !mass.is_finite() || mass <= 0.0 {
            return Err(RepertoireError::UndefinedCauseRepertoire);
        }
        for probability in &mut repertoire {
            *probability /= mass;
        }
        Ok(repertoire)
    }

    fn validate_subset(&self, subset: &[usize], name: &'static str) -> Result<(), RepertoireError> {
        let mut seen = vec![false; self.n_units];
        for unit in subset.iter().copied() {
            self.check_unit(unit)?;
            if seen[unit] {
                return Err(RepertoireError::DuplicateSubsetUnit { subset: name, unit });
            }
            seen[unit] = true;
        }
        Ok(())
    }

    /// SHA-256 of the v0.1 canonical binary model encoding.
    ///
    /// Encoding: magic/version bytes; n_units as little-endian u32; each TPM
    /// probability as little-endian IEEE-754 f64 bits in row-major order (negative
    /// zero is normalized to positive zero); then each connectivity bool as 0/1.
    /// This encoding is independent of Rust struct layout and debug formatting.
    pub fn model_sha256(&self) -> String {
        hex_digest(&self.canonical_bytes())
    }

    /// SHA-256 binding the canonical model and one evaluation state.
    pub fn evaluation_sha256(&self, state: &[bool]) -> Result<String, ModelError> {
        self.check_state(state)?;
        let canonical_model = self.canonical_bytes();
        let mut bytes = Vec::with_capacity(
            EVALUATION_MAGIC.len() + canonical_model.len() + state.len(),
        );
        bytes.extend_from_slice(EVALUATION_MAGIC);
        bytes.extend_from_slice(&canonical_model);
        bytes.extend(state.iter().map(|on| u8::from(*on)));
        Ok(hex_digest(&bytes))
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            MODEL_MAGIC.len() + 4 + self.transition_probabilities.len() * 8
                + self.connectivity.len(),
        );
        bytes.extend_from_slice(MODEL_MAGIC);
        bytes.extend_from_slice(&(self.n_units as u32).to_le_bytes());
        for probability in &self.transition_probabilities {
            let canonical = if *probability == 0.0 { 0.0 } else { *probability };
            bytes.extend_from_slice(&canonical.to_bits().to_le_bytes());
        }
        bytes.extend(self.connectivity.iter().map(|edge| u8::from(*edge)));
        bytes
    }

    fn check_state(&self, state: &[bool]) -> Result<(), ModelError> {
        if state.len() != self.n_units {
            return Err(ModelError::StateLength {
                expected: self.n_units,
                actual: state.len(),
            });
        }
        Ok(())
    }

    fn check_unit(&self, index: usize) -> Result<(), ModelError> {
        if index >= self.n_units {
            return Err(ModelError::UnitIndexOutOfRange {
                index,
                unit_count: self.n_units,
            });
        }
        Ok(())
    }
}


/// Explicit IIT formalism profile. Profiles are not interchangeable defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormalismProfile {
    /// Original IIT 3.0 formalism (PyPhi preset: IIT_3_0).
    Iit3_0,
    /// IIT 4.0 as published in 2023, without the later intrinsic-information
    /// requirement in the system-level phi_s measure.
    Iit4_0_2023,
    /// IIT 4.0 with the intrinsic-information requirement described in 2026.
    Iit4_0_2026,
}

impl FormalismProfile {
    /// Stable external identifier intended for manifests and request digests.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Iit3_0 => "IIT_3_0",
            Self::Iit4_0_2023 => "IIT_4_0_2023",
            Self::Iit4_0_2026 => "IIT_4_0_2026",
        }
    }
}

/// Requested result identity. Distinct variants prevent conflating IIT 3.0 Phi,
/// IIT 4.0 system phi_s, mechanism small phi, and structure-level big Phi.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalysisQuantity {
    Iit3SystemPhi,
    Iit4SystemPhiS,
    Iit4MechanismSmallPhi,
    Iit4CauseEffectStructure,
    Iit4StructureBigPhi,
}

impl AnalysisQuantity {
    /// Stable external identifier intended for manifests and request digests.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Iit3SystemPhi => "iit3_system_phi",
            Self::Iit4SystemPhiS => "iit4_system_phi_s",
            Self::Iit4MechanismSmallPhi => "iit4_mechanism_small_phi",
            Self::Iit4CauseEffectStructure => "iit4_cause_effect_structure",
            Self::Iit4StructureBigPhi => "iit4_structure_big_phi",
        }
    }

    fn compatible_with(self, formalism: FormalismProfile) -> bool {
        match (self, formalism) {
            (Self::Iit3SystemPhi, FormalismProfile::Iit3_0) => true,
            (
                Self::Iit4SystemPhiS
                | Self::Iit4MechanismSmallPhi
                | Self::Iit4CauseEffectStructure
                | Self::Iit4StructureBigPhi,
                FormalismProfile::Iit4_0_2023 | FormalismProfile::Iit4_0_2026,
            ) => true,
            _ => false,
        }
    }
}

/// Complete identity of one requested calculation; no calculation algorithm is
/// implied by creating or hashing this record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisRequest {
    state: Vec<bool>,
    formalism: FormalismProfile,
    quantity: AnalysisQuantity,
}

impl AnalysisRequest {
    /// Create a request only when state shape and quantity/formalism are compatible.
    pub fn new(
        model: &BinaryCausalModel,
        state: Vec<bool>,
        formalism: FormalismProfile,
        quantity: AnalysisQuantity,
    ) -> Result<Self, ModelError> {
        model.check_state(&state)?;
        if !quantity.compatible_with(formalism) {
            return Err(ModelError::IncompatibleQuantity {
                quantity: quantity.id(),
                formalism: formalism.id(),
            });
        }
        Ok(Self { state, formalism, quantity })
    }

    pub fn state(&self) -> &[bool] {
        &self.state
    }

    pub fn formalism(&self) -> FormalismProfile {
        self.formalism
    }

    pub fn quantity(&self) -> AnalysisQuantity {
        self.quantity
    }

    /// SHA-256 binding model bytes, evaluation state, formalism, and quantity.
    ///
    /// A qualified result receipt should also bind build identity, algorithm
    /// version, numerical policy, and partition-search configuration.
    pub fn request_sha256(&self, model: &BinaryCausalModel) -> String {
        let mut bytes = Vec::with_capacity(
            ANALYSIS_REQUEST_MAGIC.len()
                + model.canonical_bytes().len()
                + self.state.len()
                + self.formalism.id().len()
                + self.quantity.id().len()
                + 2,
        );
        bytes.extend_from_slice(ANALYSIS_REQUEST_MAGIC);
        bytes.extend_from_slice(&model.canonical_bytes());
        bytes.extend(self.state.iter().map(|on| u8::from(*on)));
        bytes.push(0);
        bytes.extend_from_slice(self.formalism.id().as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.quantity.id().as_bytes());
        hex_digest(&bytes)
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        // Writing to a String is infallible.
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_unit_model() -> BinaryCausalModel {
        // State indices 0..3 map to [false,false], [true,false],
        // [false,true], [true,true]; unit 0 is the least-significant bit.
        // Each row contains P(A_next=1), P(B_next=1).
        BinaryCausalModel::new(
            2,
            vec![
                0.1, 0.2,
                0.9, 0.2,
                0.1, 0.8,
                0.9, 0.8,
            ],
            vec![
                true, true,
                false, true,
            ],
        ).expect("valid fixture")
    }

    #[test]
    fn validates_dimensions_and_probabilities() {
        assert!(matches!(
            BinaryCausalModel::new(0, vec![], vec![]),
            Err(ModelError::InvalidUnitCount { .. })
        ));
        assert!(matches!(
            BinaryCausalModel::new(2, vec![0.5; 7], vec![false; 4]),
            Err(ModelError::TransitionTableLength { expected: 8, actual: 7 })
        ));
        assert!(matches!(
            BinaryCausalModel::new(2, vec![0.5; 8], vec![false; 3]),
            Err(ModelError::ConnectivityLength { expected: 4, actual: 3 })
        ));
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
            let mut tpm = vec![0.5; 8];
            tpm[3] = invalid;
            assert!(matches!(
                BinaryCausalModel::new(2, tpm, vec![false; 4]),
                Err(ModelError::InvalidProbability { index: 3, .. })
            ));
        }
    }

    #[test]
    fn uses_documented_little_endian_state_index() {
        let model = two_unit_model();
        assert_eq!(model.state_index(&[false, false]).unwrap(), 0);
        assert_eq!(model.state_index(&[true, false]).unwrap(), 1);
        assert_eq!(model.state_index(&[false, true]).unwrap(), 2);
        assert_eq!(model.state_index(&[true, true]).unwrap(), 3);
        assert_eq!(model.probability_next_one(&[false, true], 0).unwrap(), 0.1);
    }

    #[test]
    fn computes_factored_transition_probabilities() {
        let model = two_unit_model();
        assert!((model.transition_probability(&[true, false], &[true, true]).unwrap() - 0.18).abs() < 1e-12);
        assert!((model.transition_probability(&[true, false], &[false, true]).unwrap() - 0.02).abs() < 1e-12);
        assert!((model.transition_probability(&[false, false], &[false, false]).unwrap() - 0.72).abs() < 1e-12);
    }

    #[test]
    fn preserves_directed_connectivity_semantics() {
        let model = two_unit_model();
        assert!(model.has_edge(0, 1).unwrap());
        assert!(!model.has_edge(1, 0).unwrap());
        assert!(matches!(
            model.has_edge(2, 0),
            Err(ModelError::UnitIndexOutOfRange { index: 2, unit_count: 2 })
        ));
    }

    #[test]
    fn digests_are_deterministic_and_cover_model_state_and_connectivity() {
        let model = two_unit_model();
        let same = two_unit_model();
        assert_eq!(model.model_sha256(), same.model_sha256());
        assert_eq!(
            model.evaluation_sha256(&[true, false]).unwrap(),
            same.evaluation_sha256(&[true, false]).unwrap()
        );
        assert_ne!(
            model.evaluation_sha256(&[true, false]).unwrap(),
            model.evaluation_sha256(&[false, true]).unwrap()
        );

        let changed_tpm = BinaryCausalModel::new(
            2,
            vec![0.11, 0.2, 0.9, 0.2, 0.1, 0.8, 0.9, 0.8],
            vec![true, true, false, true],
        ).unwrap();
        assert_ne!(model.model_sha256(), changed_tpm.model_sha256());

        let changed_connectivity = BinaryCausalModel::new(
            2,
            model.transition_probabilities().to_vec(),
            vec![true, false, false, true],
        ).unwrap();
        assert_ne!(model.model_sha256(), changed_connectivity.model_sha256());
    }

    #[test]
    fn negative_zero_is_canonicalized_for_model_identity() {
        let positive_zero = BinaryCausalModel::new(1, vec![0.0, 1.0], vec![true]).unwrap();
        let negative_zero = BinaryCausalModel::new(1, vec![-0.0, 1.0], vec![true]).unwrap();
        assert_eq!(positive_zero.model_sha256(), negative_zero.model_sha256());
    }

    #[test]
    fn rejects_state_length_mismatch() {
        let model = two_unit_model();
        assert!(matches!(
            model.transition_probability(&[true], &[false, true]),
            Err(ModelError::StateLength { expected: 2, actual: 1 })
        ));
        assert!(matches!(
            model.evaluation_sha256(&[false]),
            Err(ModelError::StateLength { expected: 2, actual: 1 })
        ));
    }
    #[test]
    fn formalism_and_quantity_are_typed_and_compatible() {
        let model = two_unit_model();
        assert!(AnalysisRequest::new(
            &model,
            vec![true, false],
            FormalismProfile::Iit4_0_2023,
            AnalysisQuantity::Iit4SystemPhiS
        ).is_ok());
        assert!(matches!(
            AnalysisRequest::new(
                &model,
                vec![true, false],
                FormalismProfile::Iit3_0,
                AnalysisQuantity::Iit4StructureBigPhi
            ),
            Err(ModelError::IncompatibleQuantity { .. })
        ));
        assert!(matches!(
            AnalysisRequest::new(
                &model,
                vec![true],
                FormalismProfile::Iit4_0_2026,
                AnalysisQuantity::Iit4SystemPhiS
            ),
            Err(ModelError::StateLength { .. })
        ));
        assert_eq!(FormalismProfile::Iit4_0_2023.id(), "IIT_4_0_2023");
        assert_eq!(FormalismProfile::Iit4_0_2026.id(), "IIT_4_0_2026");
        assert_ne!(
            AnalysisQuantity::Iit4SystemPhiS.id(),
            AnalysisQuantity::Iit4StructureBigPhi.id()
        );
    }

    #[test]
    fn analysis_request_digest_binds_formalism_and_quantity() {
        let model = two_unit_model();
        let req_2023 = AnalysisRequest::new(
            &model,
            vec![true, false],
            FormalismProfile::Iit4_0_2023,
            AnalysisQuantity::Iit4SystemPhiS,
        ).unwrap();
        let req_2026 = AnalysisRequest::new(
            &model,
            vec![true, false],
            FormalismProfile::Iit4_0_2026,
            AnalysisQuantity::Iit4SystemPhiS,
        ).unwrap();
        let req_structure = AnalysisRequest::new(
            &model,
            vec![true, false],
            FormalismProfile::Iit4_0_2023,
            AnalysisQuantity::Iit4StructureBigPhi,
        ).unwrap();
        assert_ne!(req_2023.request_sha256(&model), req_2026.request_sha256(&model));
        assert_ne!(req_2023.request_sha256(&model), req_structure.request_sha256(&model));
    }

    #[test]
    fn identity_model_has_point_mass_cause_and_effect_repertoires() {
        let model = BinaryCausalModel::new(1, vec![0.0, 1.0], vec![true]).unwrap();
        let effect = model.effect_repertoire(&[0], &[true], &[0]).unwrap();
        let cause = model.cause_repertoire(&[0], &[true], &[0]).unwrap();
        assert_eq!(effect, vec![0.0, 1.0]);
        assert_eq!(cause, vec![0.0, 1.0]);

        let effect_off = model.effect_repertoire(&[0], &[false], &[0]).unwrap();
        let cause_off = model.cause_repertoire(&[0], &[false], &[0]).unwrap();
        assert_eq!(effect_off, vec![1.0, 0.0]);
        assert_eq!(cause_off, vec![1.0, 0.0]);
    }

    #[test]
    fn effect_repertoire_averages_each_target_over_unconstrained_units() {
        // Unit B's next-state probability is 0.3 for every past state, independently of A.
        let model = BinaryCausalModel::new(
            2,
            vec![0.1, 0.3, 0.9, 0.3, 0.1, 0.3, 0.9, 0.3],
            vec![true, false, false, false],
        ).unwrap();
        let repertoire = model.effect_repertoire(&[0], &[true, false], &[1]).unwrap();
        assert!((repertoire[0] - 0.7).abs() < 1e-12);
        assert!((repertoire[1] - 0.3).abs() < 1e-12);
        assert!((repertoire.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn effect_repertoire_factorizes_per_unit_instead_of_mixing_hidden_state_correlations() {
        // Given A=ON, B is unconstrained. Both next-unit marginals average to 0.5,
        // so the declared factored repertoire is uniform, not a 50/50 mixture of 00 and 11.
        let model = BinaryCausalModel::new(
            2,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![false, false, true, true],
        ).unwrap();
        let repertoire = model.effect_repertoire(&[0], &[true, false], &[0, 1]).unwrap();
        for probability in repertoire {
            assert!((probability - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn cause_repertoire_is_uniform_when_past_purview_is_irrelevant() {
        // A's next probability depends on past A but not past B; conditioning on current A
        // therefore leaves the B purview at its uniform prior.
        let model = BinaryCausalModel::new(
            2,
            vec![0.1, 0.3, 0.9, 0.3, 0.1, 0.3, 0.9, 0.3],
            vec![true, false, false, false],
        ).unwrap();
        let repertoire = model.cause_repertoire(&[0], &[true, false], &[1]).unwrap();
        assert!((repertoire[0] - 0.5).abs() < 1e-12);
        assert!((repertoire[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn repertoire_rejects_duplicate_subset_units() {
        let model = two_unit_model();
        assert!(matches!(
            model.effect_repertoire(&[0, 0], &[true, false], &[1]),
            Err(RepertoireError::DuplicateSubsetUnit { subset: "mechanism", unit: 0 })
        ));
        assert!(matches!(
            model.cause_repertoire(&[0], &[true, false], &[1, 1]),
            Err(RepertoireError::DuplicateSubsetUnit { subset: "purview", unit: 1 })
        ));
    }

    #[test]
    fn cause_repertoire_fails_closed_for_zero_likelihood() {
        let model = BinaryCausalModel::new(1, vec![0.0, 0.0], vec![true]).unwrap();
        assert!(matches!(
            model.cause_repertoire(&[0], &[true], &[0]),
            Err(RepertoireError::UndefinedCauseRepertoire)
        ));
    }

    #[test]
    fn rejects_connectivity_that_omits_a_tpm_influence() {
        // Unit 1 changes target 1's transition probability, but the declared
        // graph omits edge 1 -> 1. PyPhi likewise rejects such under-specified connectivity.
        assert!(matches!(
            BinaryCausalModel::new(
                2,
                vec![0.1, 0.2, 0.9, 0.2, 0.1, 0.8, 0.9, 0.8],
                vec![true, false, true, false]
            ),
            Err(ModelError::MissingCausalEdge {
                source_unit: 1,
                target_unit: 1,
                ..
            })
        ));
    }

}
