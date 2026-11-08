//! Exact posterior envelopes over an explicitly enumerated finite model family.
//!
//! This is a bounded first step toward credal inference. Each family member is
//! one complete finite factor model with the same variable/cardinality manifest.
//! The implementation computes exact *unary* state-probability bounds across
//! supported models. It does not yet optimize over continuous credal sets,
//! arbitrary joint query events, or model weights.
//!
//! Model identity is derived from a versioned canonical transcript and SHA-256.
//! Every finite-family query recomputes the content digest and checks it against
//! the supplied model reference before inference, preventing a digest from being
//! reused for altered factor content.
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
use sha2::{Digest, Sha256};


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
        if factor.values.iter().any(|value| value.is_negative()) {
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


/// Compute the lowercase SHA-256 content identity of a canonical rational model.
///
/// The digest is computed here rather than accepted as caller-supplied metadata.
/// The versioned transcript is the hash preimage; changing its encoding requires
/// a new transcript version and therefore a new model identity.
pub fn canonical_model_sha256_hex(
    graph: &ExactFactorGraph<CheckedRational>,
) -> Result<String, CanonicalModelTranscriptError> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let transcript = canonical_model_transcript(graph)?;
    let digest = Sha256::digest(&transcript);
    let mut encoded = String::with_capacity(64);
    for &byte in digest.iter() {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(encoded)
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
/// The digest is recomputed from the canonical model transcript at evaluation
/// time. Factor insertion order does not affect identity; factor scope/table
/// order and duplicate factor multiplicity remain part of the model semantics.
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
    ModelDigestMismatch { index: usize, supplied: String, computed: String },
    CanonicalModelEncoding { index: usize, source: CanonicalModelTranscriptError },
    DuplicateModelDigest { digest: String },
    MissingStateBounds { variable_index: usize, state_index: usize },
    MissingQueryAssignmentWitness { query_assignment_index: usize },
    InvalidQueryVariable { variable_index: usize },
    DuplicateQueryVariable { variable_index: usize },
    QueryStateOutOfRange { variable_index: usize, state: usize, cardinality: usize },
    AssignmentSpaceOverflow,
    AssignmentLimitExceeded { required: usize, limit: usize },
    EnumerationPartitionMismatch {
        model_digest: String,
        elimination_mass: CheckedRational,
        enumerated_mass: CheckedRational,
    },
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
            Self::ModelDigestMismatch { index, supplied, computed } => write!(
                f,
                "model at family index {index} supplied digest {supplied}, but canonical model content hashes to {computed}"
            ),
            Self::CanonicalModelEncoding { index, source } => write!(
                f,
                "model at family index {index} cannot be canonically encoded: {source}"
            ),
            Self::DuplicateModelDigest { digest } => {
                write!(f, "finite model family repeats model digest {digest}")
            }
            Self::MissingStateBounds { variable_index, state_index } => write!(
                f,
                "no posterior bound was accumulated for variable {variable_index}, state {state_index}"
            ),
            Self::MissingQueryAssignmentWitness { query_assignment_index } => write!(
                f,
                "no model witness was accumulated for query assignment {query_assignment_index}"
            ),
            Self::InvalidQueryVariable { variable_index } => {
                write!(f, "query references unknown variable {variable_index}")
            }
            Self::DuplicateQueryVariable { variable_index } => {
                write!(f, "query repeats variable {variable_index}")
            }
            Self::QueryStateOutOfRange { variable_index, state, cardinality } => write!(
                f,
                "query state {state} is outside variable {variable_index} cardinality {cardinality}"
            ),
            Self::AssignmentSpaceOverflow => write!(f, "full assignment-space size overflowed usize"),
            Self::AssignmentLimitExceeded { required, limit } => write!(
                f,
                "exact enumeration requires {required} full assignments; configured limit is {limit}"
            ),
            Self::EnumerationPartitionMismatch {
                model_digest,
                elimination_mass,
                enumerated_mass,
            } => write!(
                f,
                "direct assignment enumeration for model {model_digest} produced {enumerated_mass}, but variable elimination produced {elimination_mass}"
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
/// All models must share one variable/cardinality manifest. Every supplied
/// digest must be a lowercase SHA-256 hex string and must match the digest
/// recomputed from that model's canonical transcript.
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

    let ordered = ordered_model_refs(models)?;

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

    let ordered = ordered_model_refs(models)?;

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


/// A maximizing assignment and the model witnessing its probability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CredalMapAssignment {
    pub assignment: Vec<usize>,
    pub probability: CheckedRational,
    pub witness_model: String,
}

/// Per-model posterior over the requested MAP query variables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CredalMapModelRecord {
    pub model_digest: String,
    pub evidence_mass: CheckedRational,
    /// Indexed by the canonical mixed-radix query assignment order.
    pub query_probabilities: Option<Vec<CheckedRational>>,
    pub elimination_stats: Option<EliminationStats>,
}

/// Exact finite-family maximax and maximin MAP over an explicit model set.
///
/// This is an exact small-instance reference for a *finite enumerated* family,
/// not a solver over arbitrary convex credal sets. For each query assignment q:
///
///   maximax chooses argmax_q max_m P_m(q | evidence)
///   maximin chooses argmax_q min_m P_m(q | evidence)
///
/// Maximin is withheld when any family member assigns zero evidence mass;
/// silently dropping that member would change the lower envelope. Maximax is
/// returned over supported models only, with the conditioning-domain status
/// making partial support explicit.
///
/// The implementation directly enumerates all complete assignments, sums
/// hidden-variable assignments into each query-state mass, and checks that the
/// enumerated mass equals the exact VE partition function for every supported
/// model. The caller must set a full-assignment limit to bound work and memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteFamilyCredalMapResult {
    pub query_variables: Vec<usize>,
    pub query_cardinalities: Vec<usize>,
    pub model_count: usize,
    pub supported_model_count: usize,
    pub zero_evidence_model_count: usize,
    pub conditioning_status: ConditioningDomainStatus,
    /// Canonical representative of the maximax optimum set (first assignment
    /// in ascending mixed-radix query order). Retained for compatibility.
    pub maximax: Option<CredalMapAssignment>,
    /// Every assignment attaining the global maximax objective, in canonical
    /// query-assignment order. Each entry carries its endpoint model witness.
    pub maximax_optima: Vec<CredalMapAssignment>,
    /// Canonical representative of the maximin optimum set, when the
    /// conditioning domain is complete. Retained for compatibility.
    pub maximin: Option<CredalMapAssignment>,
    /// Every assignment attaining the global maximin objective. Empty when
    /// maximin is withheld for partial or empty evidence support.
    pub maximin_optima: Vec<CredalMapAssignment>,
    pub model_records: Vec<CredalMapModelRecord>,
}

#[derive(Clone, Copy)]
struct ProbabilityWitness {
    probability: CheckedRational,
    model_index: usize,
}

/// Exact finite-family maximax/maximin MAP on a caller-selected query subset.
///
/// Query variables are canonicalized into ascending variable-index order.
/// Assignments are returned in that same order. All unqueried variables are
/// summed out exactly. The family is unweighted; no model averaging is inferred.
pub fn evaluate_finite_family_credal_map(
    models: &[FiniteModelRef<'_>],
    query_variables: &[usize],
    elimination_order: &[usize],
    max_full_assignments: usize,
) -> Result<FiniteFamilyCredalMapResult, FiniteFamilyError> {
    if models.is_empty() {
        return Err(FiniteFamilyError::EmptyFamily);
    }
    if max_full_assignments == 0 {
        return Err(FiniteFamilyError::AssignmentLimitExceeded {
            required: 1,
            limit: 0,
        });
    }

    let ordered = ordered_model_refs(models)?;

    let cardinalities = ordered[0].graph.cardinalities().to_vec();
    let mut canonical_query = query_variables.to_vec();
    canonical_query.sort_unstable();
    for (position, &variable) in canonical_query.iter().enumerate() {
        if variable >= cardinalities.len() {
            return Err(FiniteFamilyError::InvalidQueryVariable {
                variable_index: variable,
            });
        }
        if position > 0 && canonical_query[position - 1] == variable {
            return Err(FiniteFamilyError::DuplicateQueryVariable {
                variable_index: variable,
            });
        }
    }
    let query_cardinalities = canonical_query
        .iter()
        .map(|&variable| cardinalities[variable])
        .collect::<Vec<_>>();

    let full_assignment_count = cardinalities.iter().try_fold(1usize, |count, &cardinality| {
        count.checked_mul(cardinality)
    }).ok_or(FiniteFamilyError::AssignmentSpaceOverflow)?;
    if full_assignment_count > max_full_assignments {
        return Err(FiniteFamilyError::AssignmentLimitExceeded {
            required: full_assignment_count,
            limit: max_full_assignments,
        });
    }
    let query_assignment_count = query_cardinalities.iter().try_fold(1usize, |count, &cardinality| {
        count.checked_mul(cardinality)
    }).ok_or(FiniteFamilyError::AssignmentSpaceOverflow)?;

    let mut max_by_query: Vec<Option<ProbabilityWitness>> =
        vec![None; query_assignment_count];
    let mut min_by_query: Vec<Option<ProbabilityWitness>> =
        vec![None; query_assignment_count];
    let mut model_records = Vec::with_capacity(ordered.len());
    let mut supported_model_count = 0usize;
    let mut zero_evidence_model_count = 0usize;

    for (model_index, model) in ordered.iter().enumerate() {
        if model.graph.cardinalities() != cardinalities.as_slice() {
            return Err(FiniteFamilyError::CardinalityManifestMismatch {
                model_digest: model.model_digest.to_owned(),
                expected: cardinalities.clone(),
                actual: model.graph.cardinalities().to_vec(),
            });
        }

        for (factor_index, factor) in model.graph.factors().iter().enumerate() {
            if let Some(value) = factor.values.iter().find(|value| value.is_negative()) {
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

        let (evidence_mass, elimination_stats) = model.graph.evaluate(elimination_order)
            .map_err(|source| FiniteFamilyError::ModelInference {
                model_digest: model.model_digest.to_owned(),
                source: RationalInferenceError::Elimination(source),
            })?;

        if evidence_mass.is_zero() {
            zero_evidence_model_count += 1;
            model_records.push(CredalMapModelRecord {
                model_digest: model.model_digest.to_owned(),
                evidence_mass: CheckedRational::zero(),
                query_probabilities: None,
                elimination_stats: Some(elimination_stats),
            });
            continue;
        }

        let mut query_masses = vec![CheckedRational::zero(); query_assignment_count];
        let mut enumerated_partition = CheckedRational::zero();
        let mut assignment = vec![0usize; cardinalities.len()];

        for full_index in 0..full_assignment_count {
            decode_mixed_radix(full_index, &cardinalities, &mut assignment);

            let mut joint_mass = CheckedRational::one();
            for factor in model.graph.factors() {
                let table_index = factor.scope.iter().fold(0usize, |index, &variable| {
                    index * cardinalities[variable] + assignment[variable]
                });
                joint_mass = joint_mass.checked_mul(factor.values[table_index])
                    .map_err(|source| FiniteFamilyError::ModelInference {
                        model_digest: model.model_digest.to_owned(),
                        source: RationalInferenceError::Arithmetic(source),
                    })?;
                if joint_mass.is_zero() {
                    break;
                }
            }

            enumerated_partition = enumerated_partition.checked_add(joint_mass)
                .map_err(|source| FiniteFamilyError::ModelInference {
                    model_digest: model.model_digest.to_owned(),
                    source: RationalInferenceError::Arithmetic(source),
                })?;

            let query_index = canonical_query.iter().fold(0usize, |index, &variable| {
                index * cardinalities[variable] + assignment[variable]
            });
            query_masses[query_index] = query_masses[query_index].checked_add(joint_mass)
                .map_err(|source| FiniteFamilyError::ModelInference {
                    model_digest: model.model_digest.to_owned(),
                    source: RationalInferenceError::Arithmetic(source),
                })?;
        }

        if enumerated_partition != evidence_mass {
            return Err(FiniteFamilyError::EnumerationPartitionMismatch {
                model_digest: model.model_digest.to_owned(),
                elimination_mass: evidence_mass,
                enumerated_mass: enumerated_partition,
            });
        }

        let mut query_probabilities = Vec::with_capacity(query_assignment_count);
        for (query_index, query_mass) in query_masses.into_iter().enumerate() {
            let probability = query_mass.checked_div(evidence_mass)
                .map_err(|source| FiniteFamilyError::ModelInference {
                    model_digest: model.model_digest.to_owned(),
                    source: RationalInferenceError::Arithmetic(source),
                })?;
            query_probabilities.push(probability);

            match max_by_query[query_index] {
                None => {
                    max_by_query[query_index] = Some(ProbabilityWitness {
                        probability,
                        model_index,
                    });
                    min_by_query[query_index] = Some(ProbabilityWitness {
                        probability,
                        model_index,
                    });
                }
                Some(current_max) => {
                    if probability > current_max.probability {
                        max_by_query[query_index] = Some(ProbabilityWitness {
                            probability,
                            model_index,
                        });
                    }
                    let current_min = min_by_query[query_index].ok_or(
                        FiniteFamilyError::MissingQueryAssignmentWitness {
                            query_assignment_index: query_index,
                        },
                    )?;
                    if probability < current_min.probability {
                        min_by_query[query_index] = Some(ProbabilityWitness {
                            probability,
                            model_index,
                        });
                    }
                }
            }
        }

        supported_model_count += 1;
        model_records.push(CredalMapModelRecord {
            model_digest: model.model_digest.to_owned(),
            evidence_mass,
            query_probabilities: Some(query_probabilities),
            elimination_stats: Some(elimination_stats),
        });
    }

    let conditioning_status = if supported_model_count == 0 {
        ConditioningDomainStatus::NoSupportedModels
    } else if zero_evidence_model_count > 0 {
        ConditioningDomainStatus::PartialConditioningDomain
    } else {
        ConditioningDomainStatus::Complete
    };

    let maximax_optima = if supported_model_count == 0 {
        Vec::new()
    } else {
        collect_optimal_assignments(&max_by_query, &ordered, &query_cardinalities)?
    };
    let maximax = maximax_optima.first().cloned();

    let maximin_optima = if conditioning_status != ConditioningDomainStatus::Complete {
        Vec::new()
    } else {
        collect_optimal_assignments(&min_by_query, &ordered, &query_cardinalities)?
    };
    let maximin = maximin_optima.first().cloned();

    Ok(FiniteFamilyCredalMapResult {
        query_variables: canonical_query,
        query_cardinalities,
        model_count: models.len(),
        supported_model_count,
        zero_evidence_model_count,
        conditioning_status,
        maximax,
        maximax_optima,
        maximin,
        maximin_optima,
        model_records,
    })
}

fn collect_optimal_assignments(
    witnesses: &[Option<ProbabilityWitness>],
    ordered_models: &[FiniteModelRef<'_>],
    query_cardinalities: &[usize],
) -> Result<Vec<CredalMapAssignment>, FiniteFamilyError> {
    let mut best_probability: Option<CheckedRational> = None;
    for (query_assignment_index, witness) in witnesses.iter().enumerate() {
        let witness = witness.ok_or(FiniteFamilyError::MissingQueryAssignmentWitness {
            query_assignment_index,
        })?;
        if best_probability
            .as_ref()
            .map_or(true, |best| witness.probability > *best)
        {
            best_probability = Some(witness.probability);
        }
    }

    let Some(best_probability) = best_probability else {
        return Ok(Vec::new());
    };

    let mut optima = Vec::new();
    for (query_assignment_index, witness) in witnesses.iter().enumerate() {
        let witness = witness.ok_or(FiniteFamilyError::MissingQueryAssignmentWitness {
            query_assignment_index,
        })?;
        if witness.probability == best_probability {
            optima.push(CredalMapAssignment {
                assignment: decode_query_assignment(query_assignment_index, query_cardinalities),
                probability: witness.probability,
                witness_model: ordered_models[witness.model_index].model_digest.to_owned(),
            });
        }
    }
    Ok(optima)
}

fn decode_mixed_radix(mut index: usize, cardinalities: &[usize], output: &mut [usize]) {
    for position in (0..cardinalities.len()).rev() {
        output[position] = index % cardinalities[position];
        index /= cardinalities[position];
    }
}

fn decode_query_assignment(mut index: usize, cardinalities: &[usize]) -> Vec<usize> {
    let mut assignment = vec![0usize; cardinalities.len()];
    for position in (0..cardinalities.len()).rev() {
        assignment[position] = index % cardinalities[position];
        index /= cardinalities[position];
    }
    assignment
}


fn ordered_model_refs<'a>(
    models: &[FiniteModelRef<'a>],
) -> Result<Vec<FiniteModelRef<'a>>, FiniteFamilyError> {
    if models.is_empty() {
        return Err(FiniteFamilyError::EmptyFamily);
    }

    let mut ordered = models.to_vec();
    for (index, model) in ordered.iter().enumerate() {
        if !is_sha256_digest(model.model_digest) {
            return Err(FiniteFamilyError::InvalidModelDigest {
                index,
                digest: model.model_digest.to_owned(),
            });
        }

        let computed = canonical_model_sha256_hex(model.graph).map_err(|source| {
            match source {
                CanonicalModelTranscriptError::NegativeProbabilityPotential { factor_index } => {
                    match model.graph.factors()[factor_index]
                        .values.iter().find(|value| value.is_negative()).copied()
                    {
                        Some(value) => FiniteFamilyError::ModelInference {
                            model_digest: model.model_digest.to_owned(),
                            source: RationalInferenceError::NegativePotential {
                                factor_index,
                                numerator: value.numerator(),
                                denominator: value.denominator(),
                            },
                        },
                        None => FiniteFamilyError::CanonicalModelEncoding {
                            index,
                            source: CanonicalModelTranscriptError::NegativeProbabilityPotential {
                                factor_index,
                            },
                        },
                    }
                }
                other => FiniteFamilyError::CanonicalModelEncoding {
                    index,
                    source: other,
                },
            }
        })?;
        if computed != model.model_digest {
            return Err(FiniteFamilyError::ModelDigestMismatch {
                index,
                supplied: model.model_digest.to_owned(),
                computed,
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
    Ok(ordered)
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

    fn digest(graph: &ExactFactorGraph<CheckedRational>) -> String {
        canonical_model_sha256_hex(graph).unwrap()
    }

    #[test]
    fn unary_posterior_bounds_are_exact_and_have_endpoint_witnesses() {
        let model_a = graph([q(3, 1), q(1, 1)]);
        let model_b = graph([q(1, 1), q(3, 1)]);
        let id_a = digest(&model_a);
        let id_b = digest(&model_b);
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
        let id_a = digest(&model_a);
        let id_b = digest(&model_b);

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
        let id_a = digest(&supported);
        let id_b = digest(&impossible);
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
        let id_a = digest(&impossible_a);
        let id_b = digest(&impossible_b);
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
        let id = digest(&model);
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
        let id_a = digest(&binary);
        let id_b = digest(&ternary);
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
        let id = digest(&model);
        let models = [FiniteModelRef { model_digest: &id, graph: &model }];

        assert!(matches!(
            evaluate_finite_model_family(&models, &[1]),
            Err(FiniteFamilyError::ModelInference { .. })
        ));
    }

    #[test]
    fn negative_probability_potential_fails_closed_for_the_whole_family() {
        let negative = graph([q(1, 1), q(-1, 1)]);
        let id = "a".repeat(64); // Valid syntax; invalid models cannot be canonically encoded.
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

        let id_a = digest(&model_a);
        let id_b = digest(&model_b);
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
        let id = digest(&model);
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
        let id = digest(&model);
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


    #[test]
    fn finite_family_maximax_and_maximin_map_are_distinct_exact_objectives() {
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

        let id_a = digest(&model_a);
        let id_b = digest(&model_b);
        let models = [
            FiniteModelRef { model_digest: &id_a, graph: &model_a },
            FiniteModelRef { model_digest: &id_b, graph: &model_b },
        ];
        let result = evaluate_finite_family_credal_map(&models, &[0], &[0, 1], 4).unwrap();

        assert_eq!(result.conditioning_status, ConditioningDomainStatus::Complete);
        // Both query assignments tie for both objectives. The legacy fields
        // expose the first canonical representative, while the optima fields
        // preserve the complete argmax sets.
        assert_eq!(
            result.maximax_optima.iter().map(|item| item.assignment.clone()).collect::<Vec<_>>(),
            vec![vec![0], vec![1]]
        );
        assert_eq!(
            result.maximin_optima.iter().map(|item| item.assignment.clone()).collect::<Vec<_>>(),
            vec![vec![0], vec![1]]
        );
        assert!(result.maximax_optima.iter().all(|item| item.probability == q(7, 10)));
        assert!(result.maximin_optima.iter().all(|item| item.probability == q(3, 10)));
        let maximax = result.maximax.unwrap();
        assert_eq!(maximax.assignment, vec![0]);
        assert_eq!(maximax.probability, q(7, 10));
        assert_eq!(maximax.witness_model, id_b);

        let maximin = result.maximin.unwrap();
        assert_eq!(maximin.assignment, vec![0]);
        assert_eq!(maximin.probability, q(3, 10));
        assert_eq!(maximin.witness_model, id_a);
    }

    #[test]
    fn credal_map_sums_hidden_variables_and_canonicalizes_query_scope() {
        let mut model = ExactFactorGraph::new(
            vec![2, 2, 2],
            CheckedRational::zero(),
            CheckedRational::one(),
        ).unwrap();
        model.add_factor(vec![0, 1], vec![q(1, 1), q(2, 1), q(3, 1), q(4, 1)]).unwrap();
        model.add_factor(vec![1, 2], vec![q(4, 1), q(3, 1), q(2, 1), q(1, 1)]).unwrap();
        let id = digest(&model);
        let models = [FiniteModelRef { model_digest: &id, graph: &model }];

        let left = evaluate_finite_family_credal_map(&models, &[0, 2], &[0, 1, 2], 8).unwrap();
        let right = evaluate_finite_family_credal_map(&models, &[2, 0], &[0, 1, 2], 8).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.query_variables, vec![0, 2]);
        assert_eq!(
            left.model_records[0].query_probabilities.as_ref().unwrap(),
            &vec![q(4, 23), q(5, 46), q(10, 23), q(13, 46)]
        );
        assert_eq!(left.model_records[0].evidence_mass, q(46, 1));
    }

    #[test]
    fn empty_credal_map_query_is_the_certain_event() {
        let model = graph([q(3, 1), q(1, 1)]);
        let id = digest(&model);
        let models = [FiniteModelRef { model_digest: &id, graph: &model }];
        let result = evaluate_finite_family_credal_map(&models, &[], &[0], 2).unwrap();
        assert_eq!(result.maximax.as_ref().unwrap().assignment, Vec::<usize>::new());
        assert_eq!(result.maximax.as_ref().unwrap().probability, CheckedRational::one());
        assert_eq!(result.maximin.as_ref().unwrap().probability, CheckedRational::one());
        assert_eq!(result.maximax_optima.len(), 1);
        assert_eq!(result.maximin_optima.len(), 1);
    }

    #[test]
    fn credal_map_rejects_unknown_or_duplicate_query_variables() {
        let model = graph([q(3, 1), q(1, 1)]);
        let id = digest(&model);
        let models = [FiniteModelRef { model_digest: &id, graph: &model }];
        assert!(matches!(
            evaluate_finite_family_credal_map(&models, &[1], &[0], 2),
            Err(FiniteFamilyError::InvalidQueryVariable { variable_index: 1 })
        ));
        assert!(matches!(
            evaluate_finite_family_credal_map(&models, &[0, 0], &[0], 2),
            Err(FiniteFamilyError::DuplicateQueryVariable { variable_index: 0 })
        ));
    }

    #[test]
    fn partial_evidence_support_withholds_maximin_but_marks_maximax_partial() {
        let supported = graph([q(1, 1), q(1, 1)]);
        let mut impossible = graph([q(1, 1), q(0, 1)]);
        impossible.add_factor(vec![0], vec![q(0, 1), q(1, 1)]).unwrap();
        let id_a = digest(&supported);
        let id_b = digest(&impossible);
        let models = [
            FiniteModelRef { model_digest: &id_a, graph: &supported },
            FiniteModelRef { model_digest: &id_b, graph: &impossible },
        ];

        let result = evaluate_finite_family_credal_map(&models, &[0], &[0], 2).unwrap();
        assert_eq!(result.conditioning_status, ConditioningDomainStatus::PartialConditioningDomain);
        assert!(result.maximax.is_some());
        assert_eq!(result.maximax_optima.len(), 1);
        assert_eq!(result.maximin, None);
        assert!(result.maximin_optima.is_empty());
    }

    #[test]
    fn credal_map_enforces_full_assignment_resource_limit() {
        let model = ExactFactorGraph::new(
            vec![2, 2, 2],
            CheckedRational::zero(),
            CheckedRational::one(),
        ).unwrap();
        let id = digest(&model);
        let models = [FiniteModelRef { model_digest: &id, graph: &model }];
        assert!(matches!(
            evaluate_finite_family_credal_map(&models, &[0], &[0, 1, 2], 7),
            Err(FiniteFamilyError::AssignmentLimitExceeded { required: 8, limit: 7 })
        ));
    }


    #[test]
    fn canonical_sha256_matches_independent_known_answer() {
        let model = graph([q(3, 1), q(1, 1)]);
        assert_eq!(
            canonical_model_sha256_hex(&model).unwrap(),
            "fd389042cfcfcdacbcf87c1c14ff5cfa16b5a71d3974aece32d7d6092c7e0654"
        );
    }

    #[test]
    fn model_reference_rejects_digest_that_does_not_match_graph_content() {
        let supplied_model = graph([q(3, 1), q(1, 1)]);
        let changed_model = graph([q(2, 1), q(1, 1)]);
        let supplied_digest = digest(&supplied_model);
        let models = [FiniteModelRef {
            model_digest: &supplied_digest,
            graph: &changed_model,
        }];
        assert!(matches!(
            evaluate_finite_model_family(&models, &[0]),
            Err(FiniteFamilyError::ModelDigestMismatch { .. })
        ));
    }


    #[test]
    fn conditional_posterior_family_map_differs_from_joint_mass_credal_map() {
        let prior_rows = [
            [q(1, 5), q(4, 5)],
            [q(4, 5), q(1, 5)],
        ];
        let b1_given_a0 = [q(1, 10), q(9, 10)];
        let b0_given_a0 = [q(9, 10), q(1, 10)];
        let b1_given_a1 = [q(1, 5), q(7, 10)];
        let b0_given_a1 = [q(4, 5), q(3, 10)];

        let mut owned_models = Vec::new();
        for prior_index in 0..2 {
            for row_a0_index in 0..2 {
                for row_a1_index in 0..2 {
                    let mut model = ExactFactorGraph::new(
                        vec![2, 2],
                        CheckedRational::zero(),
                        CheckedRational::one(),
                    ).unwrap();
                    model.add_factor(
                        vec![0],
                        prior_rows[prior_index].to_vec(),
                    ).unwrap();
                    // Scope [A, B]: B is the fastest-changing table coordinate.
                    model.add_factor(
                        vec![0, 1],
                        vec![
                            b0_given_a0[row_a0_index].clone(),
                            b1_given_a0[row_a0_index].clone(),
                            b0_given_a1[row_a1_index].clone(),
                            b1_given_a1[row_a1_index].clone(),
                        ],
                    ).unwrap();
                    // Evidence B = 1.
                    model.add_factor(
                        vec![1],
                        vec![CheckedRational::zero(), CheckedRational::one()],
                    ).unwrap();
                    owned_models.push(model);
                }
            }
        }

        assert_eq!(owned_models.len(), 8);
        let digests = owned_models.iter().map(digest).collect::<Vec<_>>();
        let model_refs = owned_models
            .iter()
            .zip(digests.iter())
            .map(|(model, model_digest)| FiniteModelRef {
                model_digest: model_digest,
                graph: model,
            })
            .collect::<Vec<_>>();

        // The finite-family API computes extrema of P(A | B=1) across models.
        let conditional = evaluate_finite_family_credal_map(
            &model_refs,
            &[0],
            &[0, 1],
            4,
        ).unwrap();
        let maximax = conditional.maximax.as_ref().unwrap();
        let maximin = conditional.maximin.as_ref().unwrap();
        assert_eq!(maximax.assignment, vec![1]);
        assert_eq!(maximax.probability, q(28, 29));
        assert_eq!(maximin.assignment, vec![1]);
        assert_eq!(maximin.probability, q(1, 19));

        // Independently recompute the joint assignment masses P(A=a, B=1).
        let mut lower = [None::<CheckedRational>, None];
        let mut upper = [None::<CheckedRational>, None];
        for model in &owned_models {
            for state in 0..2 {
                let mut event_model = model.clone();
                let mut indicator = vec![
                    CheckedRational::zero(),
                    CheckedRational::zero(),
                ];
                indicator[state] = CheckedRational::one();
                event_model.add_factor(vec![0], indicator).unwrap();
                let (joint_mass, _) = event_model.evaluate(&[0, 1]).unwrap();

                if lower[state]
                    .as_ref()
                    .map_or(true, |current| &joint_mass < current)
                {
                    lower[state] = Some(joint_mass.clone());
                }
                if upper[state]
                    .as_ref()
                    .map_or(true, |current| &joint_mass > current)
                {
                    upper[state] = Some(joint_mass);
                }
            }
        }

        assert_eq!(lower[0].as_ref().unwrap(), &q(1, 50));
        assert_eq!(upper[0].as_ref().unwrap(), &q(18, 25));
        assert_eq!(lower[1].as_ref().unwrap(), &q(1, 25));
        assert_eq!(upper[1].as_ref().unwrap(), &q(14, 25));

        // Joint-mass maximax selects A=0, unlike conditional-posterior maximax.
        assert!(upper[0].as_ref().unwrap() > upper[1].as_ref().unwrap());
        // Joint-mass maximin selects A=1.
        assert!(lower[1].as_ref().unwrap() > lower[0].as_ref().unwrap());
    }

}
