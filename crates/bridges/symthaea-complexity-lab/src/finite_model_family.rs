//! Exact posterior envelopes over an explicitly enumerated finite model family.
//!
//! This is a bounded first step toward credal inference. Each family member is
//! one complete finite factor model with the same variable/cardinality manifest.
//! The implementation computes exact *unary* state-probability bounds across
//! supported models. It does not yet optimize over continuous credal sets,
//! arbitrary joint query events, or model weights.
//!
//! Model digests are supplied by the caller and must be SHA-256 digests of the
//! repository's canonical model transcript. This module validates their shape
//! and uniqueness but does not recompute the transcript hash. A later
//! proof-carrying family certificate must bind/recompute those canonical inputs.
//!
//! A model with zero evidence mass contributes no conditional posterior. It is
//! counted explicitly; its undefined posterior is never replaced by a uniform
//! distribution. If only some models support the evidence, the returned bounds
//! are over that supported subfamily and carry PartialConditioningDomain status.
//! No claim is made about models outside the caller-declared family.

use crate::rational_probability::{
    posterior_marginals_expectation_semiring, RationalInferenceError,
};
use crate::semiring::{CheckedRational, EliminationStats, ExactFactorGraph};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;


/// Canonical transcript encoding failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonicalModelTranscriptError {
    LengthOutOfRange { field: &'static str, value: usize },
    NegativeProbabilityPotential { factor_index: usize },
}

impl fmt::Display for CanonicalModelTranscriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthOutOfRange { field, value } => {
                write!(f, "{field} length/value {value} does not fit canonical u64 encoding")
            }
            Self::NegativeProbabilityPotential { factor_index } => {
                write!(f, "probability model factor {factor_index} contains a negative potential")
            }
        }
    }
}

impl Error for CanonicalModelTranscriptError {}

/// Encode a rational probability factor graph into a deterministic byte transcript.
///
/// The transcript is suitable as the input to SHA-256 or another explicitly
/// versioned digest function. This function deliberately does not claim to
/// produce a cryptographic digest: the caller must hash the returned bytes and
/// bind the resulting digest to these exact bytes in any certificate.
///
/// Canonicalization rules:
/// - Variable identities are their stable ordinal indices; cardinalities are
///   encoded in ordinal order.
/// - Factors are sorted lexicographically by scope, then by their canonical
///   rational table values. Duplicate factors remain duplicated.
/// - The order of variables inside a factor scope is preserved because it
///   defines table-coordinate meaning.
/// - Table values are reduced CheckedRational numerator/positive denominator
///   pairs encoded as signed i128 big-endian bytes.
/// - The transcript is explicitly versioned and declares probability-potential
///   semantics and the table-order convention.
pub fn canonical_model_transcript(
    graph: &ExactFactorGraph<CheckedRational>,
) -> Result<Vec<u8>, CanonicalModelTranscriptError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"SYMPTHAEA-PGM-RATIONAL-MODEL-V1\0");
    bytes.extend_from_slice(b"semantics=nonnegative-probability-potentials\0");
    bytes.extend_from_slice(b"scope-order=first-variable-most-significant\0");

    append_len(&mut bytes, "variable count", graph.variable_count())?;
    for &cardinality in graph.cardinalities() {
        append_len(&mut bytes, "variable cardinality", cardinality)?;
    }

    let mut factors = graph.factors().iter().collect::<Vec<_>>();
    for (factor_index, factor) in factors.iter().enumerate() {
        if factor.values.iter().any(CheckedRational::is_negative) {
            return Err(CanonicalModelTranscriptError::NegativeProbabilityPotential {
                factor_index,
            });
        }
    }
    factors.sort_by(|left, right| {
        let scope_order = left.scope.cmp(&right.scope);
        if scope_order != std::cmp::Ordering::Equal {
            return scope_order;
        }
        for (left_value, right_value) in left.values.iter().zip(&right.values) {
            let value_order = (
                left_value.numerator(),
                left_value.denominator(),
            ).cmp(&(
                right_value.numerator(),
                right_value.denominator(),
            ));
            if value_order != std::cmp::Ordering::Equal {
                return value_order;
            }
        }
        left.values.len().cmp(&right.values.len())
    });

    append_len(&mut bytes, "factor count", factors.len())?;
    for factor in factors {
        append_len(&mut bytes, "factor scope length", factor.scope.len())?;
        for &variable in &factor.scope {
            append_len(&mut bytes, "factor variable index", variable)?;
        }
        append_len(&mut bytes, "factor table length", factor.values.len())?;
        for value in &factor.values {
            bytes.extend_from_slice(&value.numerator().to_be_bytes());
            bytes.extend_from_slice(&value.denominator().to_be_bytes());
        }
    }

    Ok(bytes)
}

fn append_len(
    output: &mut Vec<u8>,
    field: &'static str,
    value: usize,
) -> Result<(), CanonicalModelTranscriptError> {
    let encoded = u64::try_from(value).map_err(|_| {
        CanonicalModelTranscriptError::LengthOutOfRange { field, value }
    })?;
    output.extend_from_slice(&encoded.to_be_bytes());
    Ok(())
}

/// One model in the caller-declared finite family.
///
/// The digest must be computed from the canonical model transcript. Factor
/// insertion order should not affect that transcript where the model semantics
/// are equivalent; factor scope/table ordering must remain part of the identity.
#[derive(Clone, Copy, Debug)]
pub struct FiniteModelRef<'a> {
    pub model_digest: &'a str,
    pub graph: &'a ExactFactorGraph<CheckedRational>,
}

/// Failure to define or evaluate a finite model family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FiniteFamilyError {
    EmptyFamily,
    InvalidModelDigest { index: usize, digest: String },
    DuplicateModelDigest { digest: String },
    MissingStateBounds { variable_index: usize, state_index: usize },
    InvalidQueryVariable { variable_index: usize },
    DuplicateQueryVariable { variable_index: usize },
    QueryStateOutOfRange { variable_index: usize, state: usize, cardinality: usize },
    CardinalityManifestMismatch {
        model_digest: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    ModelInference {
        model_digest: String,
        source: RationalInferenceError,
    },
}

impl fmt::Display for FiniteFamilyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFamily => write!(f, "finite model family must contain at least one model"),
            Self::InvalidModelDigest { index, digest } => write!(
                f,
                "model at family index {index} has invalid SHA-256 digest {digest:?}"
            ),
            Self::DuplicateModelDigest { digest } => {
                write!(f, "finite model family repeats model digest {digest}")
            }
            Self::MissingStateBounds { variable_index, state_index } => write!(
                f,
                "no posterior bound was accumulated for variable {variable_index}, state {state_index}"
            ),
            Self::InvalidQueryVariable { variable_index } => {
                write!(f, "joint event references unknown variable {variable_index}")
            }
            Self::DuplicateQueryVariable { variable_index } => {
                write!(f, "joint event assigns variable {variable_index} more than once")
            }
            Self::QueryStateOutOfRange { variable_index, state, cardinality } => write!(
                f,
                "joint event state {state} is outside variable {variable_index} cardinality {cardinality}"
            ),
            Self::CardinalityManifestMismatch {
                model_digest,
                expected,
                actual,
            } => write!(
                f,
                "model {model_digest} has cardinalities {actual:?}; family manifest requires {expected:?}"
            ),
            Self::ModelInference {
                model_digest,
                source,
            } => write!(f, "exact inference failed for model {model_digest}: {source}"),
        }
    }
}

impl Error for FiniteFamilyError {}

/// Conditional-inference domain covered by the returned posterior envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConditioningDomainStatus {
    /// Every candidate model assigns positive mass to the supplied evidence.
    Complete,
    /// At least one model has zero evidence mass; bounds cover only supported
    /// members. Callers that require universal evidence support must abstain.
    PartialConditioningDomain,
    /// No model in the declared family supports the evidence. No posterior
    /// bounds are returned.
    NoSupportedModels,
}

/// Exact lower/upper probability for one unary state event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateProbabilityBounds {
    pub lower: CheckedRational,
    pub upper: CheckedRational,
    /// Lexicographically smallest model digest attaining the lower endpoint.
    pub lower_witness_model: String,
    /// Lexicographically smallest model digest attaining the upper endpoint.
    pub upper_witness_model: String,
}

/// Per-model exact evaluation record.
///
/// For zero-evidence models, evidence_mass is exactly zero and posterior
/// marginals/statistics are None because conditional probabilities are undefined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPosteriorRecord {
    pub model_digest: String,
    pub evidence_mass: CheckedRational,
    pub posterior_marginals: Option<Vec<Vec<CheckedRational>>>,
    pub elimination_stats: Option<EliminationStats>,
}

/// Exact posterior lower/upper bounds for every variable state in the manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteFamilyPosteriorEnvelope {
    pub variable_cardinalities: Vec<usize>,
    pub model_count: usize,
    pub supported_model_count: usize,
    pub zero_evidence_model_count: usize,
    pub conditioning_status: ConditioningDomainStatus,
    /// None precisely when no family member supports the evidence.
    /// Otherwise indexed by variable, then by state.
    pub bounds: Option<Vec<Vec<StateProbabilityBounds>>>,
    /// Sorted by model digest, making output independent of caller insertion order.
    pub model_records: Vec<ModelPosteriorRecord>,
}

/// Evaluate exact unary posterior envelopes over a finite family.
///
/// All models must share one variable/cardinality manifest and every digest
/// must be a unique lowercase 64-character hexadecimal SHA-256 identifier.
/// The digest bytes are not recomputed here; callers must generate them from
/// the canonical model encoding and an independent certificate checker must
/// bind those encodings to the evaluated models.
///
/// The supplied elimination order is applied unchanged to every model. The
/// expectation-semiring posterior evaluator computes the partition/evidence
/// mass and all unary state marginals in one elimination pass per model.
/// Arithmetic remains bounded by CheckedRational's checked i128 domain.
///
/// The envelope is an unweighted set-valued result, not a Bayesian model
/// average. No family-member probabilities or weights are inferred.
pub fn evaluate_finite_model_family(
    models: &[FiniteModelRef<'_>],
    elimination_order: &[usize],
) -> Result<FiniteFamilyPosteriorEnvelope, FiniteFamilyError> {
    if models.is_empty() {
        return Err(FiniteFamilyError::EmptyFamily);
    }

    let mut ordered = models.iter().copied().collect::<Vec<_>>();
    for (index, model) in ordered.iter().enumerate() {
        if !is_sha256_digest(model.model_digest) {
            return Err(FiniteFamilyError::InvalidModelDigest {
                index,
                digest: model.model_digest.to_owned(),
            });
        }
    }
    ordered.sort_by(|left, right| left.model_digest.cmp(right.model_digest));

    let mut seen = BTreeSet::new();
    for model in &ordered {
        if !seen.insert(model.model_digest) {
            return Err(FiniteFamilyError::DuplicateModelDigest {
                digest: model.model_digest.to_owned(),
            });
        }
    }

    let cardinalities = ordered[0].graph.cardinalities().to_vec();
    let mut lower_upper: Vec<Vec<Option<StateProbabilityBounds>>> = cardinalities
        .iter()
        .map(|&cardinality| vec![None; cardinality])
        .collect();

    let mut model_records = Vec::with_capacity(ordered.len());
    let mut supported_model_count = 0usize;
    let mut zero_evidence_model_count = 0usize;

    for model in ordered {
        let model_cardinalities = model.graph.cardinalities();
        if model_cardinalities != cardinalities.as_slice() {
            return Err(FiniteFamilyError::CardinalityManifestMismatch {
                model_digest: model.model_digest.to_owned(),
                expected: cardinalities.clone(),
                actual: model_cardinalities.to_vec(),
            });
        }

        let posterior = match posterior_marginals_expectation_semiring(
            model.graph,
            elimination_order,
        ) {
            Ok(posterior) => posterior,
            Err(RationalInferenceError::ZeroPartitionFunction) => {
                zero_evidence_model_count += 1;
                model_records.push(ModelPosteriorRecord {
                    model_digest: model.model_digest.to_owned(),
                    evidence_mass: CheckedRational::zero(),
                    posterior_marginals: None,
                    elimination_stats: None,
                });
                continue;
            }
            Err(source) => {
                return Err(FiniteFamilyError::ModelInference {
                    model_digest: model.model_digest.to_owned(),
                    source,
                });
            }
        };

        supported_model_count += 1;
        for (variable_index, marginal) in posterior.marginals.iter().enumerate() {
            for (state, &probability) in marginal.iter().enumerate() {
                match &mut lower_upper[variable_index][state] {
                    None => {
                        lower_upper[variable_index][state] = Some(StateProbabilityBounds {
                            lower: probability,
                            upper: probability,
                            lower_witness_model: model.model_digest.to_owned(),
                            upper_witness_model: model.model_digest.to_owned(),
                        });
                    }
                    Some(bounds) => {
                        // Models are processed in digest order. Strict comparison
                        // preserves the lexicographically first witness on ties.
                        if probability < bounds.lower {
                            bounds.lower = probability;
                            bounds.lower_witness_model = model.model_digest.to_owned();
                        }
                        if probability > bounds.upper {
                            bounds.upper = probability;
                            bounds.upper_witness_model = model.model_digest.to_owned();
                        }
                    }
                }
            }
        }

        model_records.push(ModelPosteriorRecord {
            model_digest: model.model_digest.to_owned(),
            evidence_mass: posterior.partition_function,
            posterior_marginals: Some(posterior.marginals),
            elimination_stats: Some(posterior.partition_stats),
        });
    }

    let conditioning_status = if supported_model_count == 0 {
        ConditioningDomainStatus::NoSupportedModels
    } else if zero_evidence_model_count > 0 {
        ConditioningDomainStatus::PartialConditioningDomain
    } else {
        ConditioningDomainStatus::Complete
    };

    let bounds = if supported_model_count == 0 {
        None
    } else {
        let mut completed = Vec::with_capacity(lower_upper.len());
        for (variable_index, variable_bounds) in lower_upper.into_iter().enumerate() {
            let mut states = Vec::with_capacity(variable_bounds.len());
            for (state_index, state_bounds) in variable_bounds.into_iter().enumerate() {
                // Keep even impossible internal states as a typed failure rather
                // than allowing a malformed result to panic during assembly.
                states.push(state_bounds.ok_or(FiniteFamilyError::MissingStateBounds {
                    variable_index,
                    state_index,
                })?);
            }
            completed.push(states);
        }
        Some(completed)
    };

    Ok(FiniteFamilyPosteriorEnvelope {
        variable_cardinalities: cardinalities,
        model_count: models.len(),
        supported_model_count,
        zero_evidence_model_count,
        conditioning_status,
        bounds,
        model_records,
    })
}


/// Exact posterior lower/upper probability for an arbitrary joint assignment event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointEventProbabilityBounds {
    pub lower: CheckedRational,
    pub upper: CheckedRational,
    pub lower_witness_model: String,
    pub upper_witness_model: String,
}

/// Per-model exact evaluation record for a joint assignment event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointEventModelRecord {
    pub model_digest: String,
    pub evidence_mass: CheckedRational,
    pub event_mass: Option<CheckedRational>,
    pub conditional_probability: Option<CheckedRational>,
    pub evidence_elimination_stats: Option<EliminationStats>,
    pub event_elimination_stats: Option<EliminationStats>,
}

/// Exact envelope for one joint assignment event over a finite model family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteFamilyJointEventEnvelope {
    /// Canonically sorted (variable, state) pairs.
    pub query_assignments: Vec<(usize, usize)>,
    pub model_count: usize,
    pub supported_model_count: usize,
    pub zero_evidence_model_count: usize,
    pub conditioning_status: ConditioningDomainStatus,
    pub bounds: Option<JointEventProbabilityBounds>,
    pub model_records: Vec<JointEventModelRecord>,
}

/// Evaluate exact conditional probability bounds for a conjunction of assignments.
///
/// The event is a conjunction X_i = state_i. It is evaluated with exact
/// variable elimination after adding indicator factors to each model. This
/// computes joint factor mass directly; it never multiplies singleton
/// marginals and therefore preserves dependencies.
///
/// The empty assignment list denotes the certain event, conditional on the
/// evidence being supported by a model. Evidence is already encoded in each
/// candidate graph as factors.
pub fn evaluate_finite_model_joint_event(
    models: &[FiniteModelRef<'_>],
    query_assignments: &[(usize, usize)],
    elimination_order: &[usize],
) -> Result<FiniteFamilyJointEventEnvelope, FiniteFamilyError> {
    if models.is_empty() {
        return Err(FiniteFamilyError::EmptyFamily);
    }

    let mut ordered = models.iter().copied().collect::<Vec<_>>();
    for (index, model) in ordered.iter().enumerate() {
        if !is_sha256_digest(model.model_digest) {
            return Err(FiniteFamilyError::InvalidModelDigest {
                index,
                digest: model.model_digest.to_owned(),
            });
        }
    }
    ordered.sort_by(|left, right| left.model_digest.cmp(right.model_digest));

    let mut seen = BTreeSet::new();
    for model in &ordered {
        if !seen.insert(model.model_digest) {
            return Err(FiniteFamilyError::DuplicateModelDigest {
                digest: model.model_digest.to_owned(),
            });
        }
    }

    let cardinalities = ordered[0].graph.cardinalities().to_vec();
    let mut canonical_query = query_assignments.to_vec();
    canonical_query.sort_unstable_by_key(|(variable, _)| *variable);

    for (position, &(variable, state)) in canonical_query.iter().enumerate() {
        let cardinality = cardinalities.get(variable).copied().ok_or(
            FiniteFamilyError::InvalidQueryVariable { variable_index: variable },
        )?;
        if position > 0 && canonical_query[position - 1].0 == variable {
            return Err(FiniteFamilyError::DuplicateQueryVariable {
                variable_index: variable,
            });
        }
        if state >= cardinality {
            return Err(FiniteFamilyError::QueryStateOutOfRange {
                variable_index: variable,
                state,
                cardinality,
            });
        }
    }

    let mut model_records = Vec::with_capacity(ordered.len());
    let mut supported_model_count = 0usize;
    let mut zero_evidence_model_count = 0usize;
    let mut endpoint_bounds: Option<JointEventProbabilityBounds> = None;

    for model in ordered {
        let model_cardinalities = model.graph.cardinalities();
        if model_cardinalities != cardinalities.as_slice() {
            return Err(FiniteFamilyError::CardinalityManifestMismatch {
                model_digest: model.model_digest.to_owned(),
                expected: cardinalities.clone(),
                actual: model_cardinalities.to_vec(),
            });
        }

        for (factor_index, factor) in model.graph.factors().iter().enumerate() {
            for value in &factor.values {
                if value.is_negative() {
                    return Err(FiniteFamilyError::ModelInference {
                        model_digest: model.model_digest.to_owned(),
                        source: RationalInferenceError::NegativePotential {
                            factor_index,
                            numerator: value.numerator(),
                            denominator: value.denominator(),
                        },
                    });
                }
            }
        }

        let (evidence_mass, evidence_stats) = model.graph.evaluate(elimination_order)
            .map_err(|source| FiniteFamilyError::ModelInference {
                model_digest: model.model_digest.to_owned(),
                source: RationalInferenceError::Elimination(source),
            })?;

        if evidence_mass.is_zero() {
            zero_evidence_model_count += 1;
            model_records.push(JointEventModelRecord {
                model_digest: model.model_digest.to_owned(),
                evidence_mass: CheckedRational::zero(),
                event_mass: None,
                conditional_probability: None,
                evidence_elimination_stats: None,
                event_elimination_stats: None,
            });
            continue;
        }

        let mut conditioned = model.graph.clone();
        for &(variable, state) in &canonical_query {
            let mut indicator = vec![CheckedRational::zero(); cardinalities[variable]];
            indicator[state] = CheckedRational::one();
            conditioned.add_factor(vec![variable], indicator)
                .map_err(|source| FiniteFamilyError::ModelInference {
                    model_digest: model.model_digest.to_owned(),
                    source: RationalInferenceError::Elimination(source),
                })?;
        }

        let (event_mass, event_stats) = conditioned.evaluate(elimination_order)
            .map_err(|source| FiniteFamilyError::ModelInference {
                model_digest: model.model_digest.to_owned(),
                source: RationalInferenceError::Elimination(source),
            })?;
        let probability = event_mass.checked_div(evidence_mass)
            .map_err(|source| FiniteFamilyError::ModelInference {
                model_digest: model.model_digest.to_owned(),
                source: RationalInferenceError::Arithmetic(source),
            })?;

        supported_model_count += 1;
        match &mut endpoint_bounds {
            None => {
                endpoint_bounds = Some(JointEventProbabilityBounds {
                    lower: probability,
                    upper: probability,
                    lower_witness_model: model.model_digest.to_owned(),
                    upper_witness_model: model.model_digest.to_owned(),
                });
            }
            Some(bounds) => {
                // Digest order plus strict comparisons gives stable first witnesses.
                if probability < bounds.lower {
                    bounds.lower = probability;
                    bounds.lower_witness_model = model.model_digest.to_owned();
                }
                if probability > bounds.upper {
                    bounds.upper = probability;
                    bounds.upper_witness_model = model.model_digest.to_owned();
                }
            }
        }

        model_records.push(JointEventModelRecord {
            model_digest: model.model_digest.to_owned(),
            evidence_mass,
            event_mass: Some(event_mass),
            conditional_probability: Some(probability),
            evidence_elimination_stats: Some(evidence_stats),
            event_elimination_stats: Some(event_stats),
        });
    }

    let conditioning_status = if supported_model_count == 0 {
        ConditioningDomainStatus::NoSupportedModels
    } else if zero_evidence_model_count > 0 {
        ConditioningDomainStatus::PartialConditioningDomain
    } else {
        ConditioningDomainStatus::Complete
    };

    Ok(FiniteFamilyJointEventEnvelope {
        query_assignments: canonical_query,
        model_count: models.len(),
        supported_model_count,
        zero_evidence_model_count,
        conditioning_status,
        bounds: endpoint_bounds,
        model_records,
    })
}

fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(numerator: i128, denominator: i128) -> CheckedRational {
        CheckedRational::new(numerator, denominator).unwrap()
    }

    fn graph(weights: [CheckedRational; 2]) -> ExactFactorGraph<CheckedRational> {
        let mut graph = ExactFactorGraph::new(
            vec![2],
            CheckedRational::zero(),
            CheckedRational::one(),
        )
        .unwrap();
        graph.add_factor(vec![0], weights.to_vec()).unwrap();
        graph
    }

    fn digest(byte: char) -> String {
        std::iter::repeat(byte).take(64).collect()
    }

    #[test]
    fn unary_posterior_bounds_are_exact_and_have_endpoint_witnesses() {
        let model_a = graph([q(3, 1), q(1, 1)]);
        let model_b = graph([q(1, 1), q(3, 1)]);
        let id_a = digest('a');
        let id_b = digest('b');
        let models = [
            FiniteModelRef { model_digest: &id_a, graph: &model_a },
            FiniteModelRef { model_digest: &id_b, graph: &model_b },
        ];

        let result = evaluate_finite_model_family(&models, &[0]).unwrap();
        let bounds = &result.bounds.as_ref().unwrap()[0];

        assert_eq!(result.model_count, 2);
        assert_eq!(result.supported_model_count, 2);
        assert_eq!(result.zero_evidence_model_count, 0);
        assert_eq!(result.conditioning_status, ConditioningDomainStatus::Complete);
        assert_eq!(bounds[0].lower, q(1, 4));
        assert_eq!(bounds[0].upper, q(3, 4));
        assert_eq!(bounds[0].lower_witness_model, id_b);
        assert_eq!(bounds[0].upper_witness_model, id_a);
        assert_eq!(bounds[1].lower, q(1, 4));
        assert_eq!(bounds[1].upper, q(3, 4));
    }

    #[test]
    fn input_order_does_not_change_envelope_or_witness_selection() {
        let model_a = graph([q(3, 1), q(1, 1)]);
        let model_b = graph([q(1, 1), q(3, 1)]);
        let id_a = digest('a');
        let id_b = digest('b');

        let forward = [
            FiniteModelRef { model_digest: &id_a, graph: &model_a },
            FiniteModelRef { model_digest: &id_b, graph: &model_b },
        ];
        let reverse = [forward[1], forward[0]];

        assert_eq!(
            evaluate_finite_model_family(&forward, &[0]).unwrap(),
            evaluate_finite_model_family(&reverse, &[0]).unwrap()
        );
    }

    #[test]
    fn zero_evidence_models_are_counted_but_never_given_a_posterior() {
        let supported = graph([q(1, 1), q(1, 1)]);
        let mut impossible = graph([q(1, 1), q(0, 1)]);
        impossible.add_factor(vec![0], vec![q(0, 1), q(1, 1)]).unwrap();
        let id_a = digest('a');
        let id_b = digest('b');
        let models = [
            FiniteModelRef { model_digest: &id_a, graph: &supported },
            FiniteModelRef { model_digest: &id_b, graph: &impossible },
        ];

        let result = evaluate_finite_model_family(&models, &[0]).unwrap();
        assert_eq!(result.conditioning_status, ConditioningDomainStatus::PartialConditioningDomain);
        assert_eq!(result.supported_model_count, 1);
        assert_eq!(result.zero_evidence_model_count, 1);
        let unsupported = result.model_records.iter().find(|row| row.model_digest == id_b).unwrap();
        assert_eq!(unsupported.evidence_mass, CheckedRational::zero());
        assert_eq!(unsupported.posterior_marginals, None);
        assert!(result.bounds.is_some());
    }

    #[test]
    fn no_supported_models_returns_no_bounds_not_a_fabricated_distribution() {
        let mut impossible_a = graph([q(1, 1), q(0, 1)]);
        impossible_a.add_factor(vec![0], vec![q(0, 1), q(1, 1)]).unwrap();
        let mut impossible_b = graph([q(0, 1), q(1, 1)]);
        impossible_b.add_factor(vec![0], vec![q(1, 1), q(0, 1)]).unwrap();
        let id_a = digest('a');
        let id_b = digest('b');
        let models = [
            FiniteModelRef { model_digest: &id_a, graph: &impossible_a },
            FiniteModelRef { model_digest: &id_b, graph: &impossible_b },
        ];

        let result = evaluate_finite_model_family(&models, &[0]).unwrap();
        assert_eq!(result.conditioning_status, ConditioningDomainStatus::NoSupportedModels);
        assert_eq!(result.supported_model_count, 0);
        assert_eq!(result.zero_evidence_model_count, 2);
        assert_eq!(result.bounds, None);
    }

    #[test]
    fn duplicate_or_malformed_model_digests_are_rejected() {
        let model = graph([q(1, 1), q(1, 1)]);
        let id = digest('a');
        let duplicate = [
            FiniteModelRef { model_digest: &id, graph: &model },
            FiniteModelRef { model_digest: &id, graph: &model },
        ];
        assert!(matches!(
            evaluate_finite_model_family(&duplicate, &[0]),
            Err(FiniteFamilyError::DuplicateModelDigest { .. })
        ));

        let malformed = [FiniteModelRef {
            model_digest: "not-a-digest",
            graph: &model,
        }];
        assert!(matches!(
            evaluate_finite_model_family(&malformed, &[0]),
            Err(FiniteFamilyError::InvalidModelDigest { .. })
        ));
    }

    #[test]
    fn mismatched_variable_cardinality_manifests_are_rejected() {
        let binary = graph([q(1, 1), q(1, 1)]);
        let mut ternary = ExactFactorGraph::new(
            vec![3],
            CheckedRational::zero(),
            CheckedRational::one(),
        ).unwrap();
        ternary.add_factor(vec![0], vec![q(1, 1), q(1, 1), q(1, 1)]).unwrap();
        let id_a = digest('a');
        let id_b = digest('b');
        let models = [
            FiniteModelRef { model_digest: &id_a, graph: &binary },
            FiniteModelRef { model_digest: &id_b, graph: &ternary },
        ];

        assert!(matches!(
            evaluate_finite_model_family(&models, &[0]),
            Err(FiniteFamilyError::CardinalityManifestMismatch { .. })
        ));
    }

    #[test]
    fn invalid_elimination_order_is_reported_with_model_identity() {
        let model = graph([q(1, 1), q(1, 1)]);
        let id = digest('a');
        let models = [FiniteModelRef { model_digest: &id, graph: &model }];

        assert!(matches!(
            evaluate_finite_model_family(&models, &[1]),
            Err(FiniteFamilyError::ModelInference { .. })
        ));
    }

    #[test]
    fn negative_probability_potential_fails_closed_for_the_whole_family() {
        let negative = graph([q(1, 1), q(-1, 1)]);
        let id = digest('a');
        let models = [FiniteModelRef { model_digest: &id, graph: &negative }];

        assert!(matches!(
            evaluate_finite_model_family(&models, &[0]),
            Err(FiniteFamilyError::ModelInference {
                source: RationalInferenceError::NegativePotential { .. },
                ..
            })
        ));
    }

    #[test]
    fn arbitrary_joint_event_bounds_use_joint_factor_mass_not_singleton_products() {
        let mut model_a = ExactFactorGraph::new(
            vec![2, 2],
            CheckedRational::zero(),
            CheckedRational::one(),
        ).unwrap();
        model_a.add_factor(vec![0, 1], vec![q(1, 1), q(2, 1), q(3, 1), q(4, 1)]).unwrap();

        let mut model_b = ExactFactorGraph::new(
            vec![2, 2],
            CheckedRational::zero(),
            CheckedRational::one(),
        ).unwrap();
        model_b.add_factor(vec![0, 1], vec![q(4, 1), q(3, 1), q(2, 1), q(1, 1)]).unwrap();

        let id_a = digest('a');
        let id_b = digest('b');
        let models = [
            FiniteModelRef { model_digest: &id_a, graph: &model_a },
            FiniteModelRef { model_digest: &id_b, graph: &model_b },
        ];
        let result = evaluate_finite_model_joint_event(&models, &[(0, 1), (1, 0)], &[0, 1]).unwrap();

        let bounds = result.bounds.unwrap();
        assert_eq!(bounds.lower, q(1, 5));
        assert_eq!(bounds.upper, q(3, 10));
        assert_eq!(bounds.lower_witness_model, id_b);
        assert_eq!(bounds.upper_witness_model, id_a);
        assert_eq!(result.conditioning_status, ConditioningDomainStatus::Complete);
    }

    #[test]
    fn joint_event_query_order_is_canonical_and_duplicate_variables_reject() {
        let mut model = ExactFactorGraph::new(
            vec![2, 2],
            CheckedRational::zero(),
            CheckedRational::one(),
        ).unwrap();
        model.add_factor(vec![0, 1], vec![q(1, 1), q(2, 1), q(3, 1), q(4, 1)]).unwrap();
        let id = digest('a');
        let models = [FiniteModelRef { model_digest: &id, graph: &model }];

        let forward = evaluate_finite_model_joint_event(&models, &[(0, 1), (1, 0)], &[0, 1]).unwrap();
        let reverse = evaluate_finite_model_joint_event(&models, &[(1, 0), (0, 1)], &[0, 1]).unwrap();
        assert_eq!(forward, reverse);

        assert!(matches!(
            evaluate_finite_model_joint_event(&models, &[(0, 0), (0, 1)], &[0, 1]),
            Err(FiniteFamilyError::DuplicateQueryVariable { variable_index: 0 })
        ));
    }

    #[test]
    fn empty_joint_event_is_certain_for_every_supported_model() {
        let model = graph([q(3, 1), q(1, 1)]);
        let id = digest('a');
        let models = [FiniteModelRef { model_digest: &id, graph: &model }];
        let result = evaluate_finite_model_joint_event(&models, &[], &[0]).unwrap();
        let bounds = result.bounds.unwrap();
        assert_eq!(bounds.lower, CheckedRational::one());
        assert_eq!(bounds.upper, CheckedRational::one());
    }


    #[test]
    fn canonical_model_transcript_is_factor_insertion_order_invariant() {
        let mut forward = ExactFactorGraph::new(
            vec![2, 2],
            CheckedRational::zero(),
            CheckedRational::one(),
        ).unwrap();
        forward.add_factor(vec![0], vec![q(1, 3), q(2, 3)]).unwrap();
        forward.add_factor(vec![1], vec![q(1, 4), q(3, 4)]).unwrap();
        forward.add_factor(vec![0, 1], vec![q(1, 1), q(2, 1), q(3, 1), q(4, 1)]).unwrap();

        let mut reverse = ExactFactorGraph::new(
            vec![2, 2],
            CheckedRational::zero(),
            CheckedRational::one(),
        ).unwrap();
        reverse.add_factor(vec![0, 1], vec![q(1, 1), q(2, 1), q(3, 1), q(4, 1)]).unwrap();
        reverse.add_factor(vec![1], vec![q(1, 4), q(3, 4)]).unwrap();
        reverse.add_factor(vec![0], vec![q(1, 3), q(2, 3)]).unwrap();

        assert_eq!(
            canonical_model_transcript(&forward).unwrap(),
            canonical_model_transcript(&reverse).unwrap()
        );
    }

    #[test]
    fn canonical_model_transcript_changes_when_exact_factor_changes() {
        let mut left = graph([q(3, 1), q(1, 1)]);
        let right = graph([q(2, 1), q(1, 1)]);
        assert_ne!(
            canonical_model_transcript(&left).unwrap(),
            canonical_model_transcript(&right).unwrap()
        );
        left.add_factor(vec![0], vec![q(1, 1), q(0, 1)]).unwrap();
        assert_ne!(
            canonical_model_transcript(&left).unwrap(),
            canonical_model_transcript(&right).unwrap()
        );
    }

    #[test]
    fn canonical_model_transcript_rejects_negative_probability_potentials() {
        let model = graph([q(1, 1), q(-1, 1)]);
        assert!(matches!(
            canonical_model_transcript(&model),
            Err(CanonicalModelTranscriptError::NegativeProbabilityPotential { factor_index: 0 })
        ));
    }

}
