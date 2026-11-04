//! Minimal reference-set decision envelope evaluator.
//!
//! This crate implements only the pairwise, single-measurand comparison core
//! specified by SAM-REF-DECISION-001 (#10318). Callers remain responsible for
//! authenticating reference qualifications, freezing the action set, defining
//! hard constraints and retaining source receipts. This crate does not prove
//! reference independence or physical truth.

#![forbid(unsafe_code)]

use std::collections::{BTreeSet, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    HigherIsBetter,
    LowerIsBetter,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntervalSemantics {
    HardBound,
    MeasurementInterval,
    ModelUncertaintyInterval,
    StatisticalInterval {
        method: String,
        confidence_ppm: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Applicability {
    Applicable,
    Inapplicable,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Qualification {
    Qualified {
        record_id: String,
        source_artifact_digest: String,
    },
    NotQualified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feasibility {
    Feasible,
    Infeasible,
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DifferenceInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetricEvidence {
    BoundedInterval {
        lower: f64,
        upper: f64,
        semantics: IntervalSemantics,
    },
    PointEstimateOnly {
        value: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetricOutcome {
    Observed(MetricEvidence),
    Missing,
    Censored,
    TimedOut,
    Unsupported,
    ExecutionFailed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActionResult {
    pub receipt_id: String,
    pub scenario_id: String,
    pub action_set_id: String,
    pub reference_id: String,
    pub action_id: String,
    pub metric_id: String,
    pub unit: String,
    pub execution_context_digest: String,
    pub action_payload_digest: String,
    pub output_digest: String,
    pub feasibility: Feasibility,
    pub outcome: MetricOutcome,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceRow {
    pub reference_id: String,
    pub dependency_family_id: String,
    pub qualification: Qualification,
    pub applicability: Applicability,
    pub action_a: Option<ActionResult>,
    pub action_b: Option<ActionResult>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricDefinition {
    pub metric_id: String,
    pub unit: String,
    pub direction: Direction,
    pub equivalence_margin: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonContext {
    pub scenario_id: String,
    pub action_set_id: String,
    pub action_set_complete: bool,
    /// Frozen expected portfolio membership; omission of a reference row is incomplete.
    pub expected_reference_ids: Vec<String>,
    pub action_a_id: String,
    pub action_b_id: String,
    pub metric: MetricDefinition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preference {
    ActionA,
    ActionB,
    EquivalentWithinMargin,
    Unresolved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionClass {
    PortfolioDominatesA,
    PortfolioDominatesB,
    PortfolioEquivalent,
    ReferenceSensitive,
    Unresolved,
    PointwiseOnly,
    Incomplete,
    OutOfScope,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IncompleteReason {
    ActionSetDeclaredIncomplete,
    MissingExpectedReference(String),
    UnexpectedReference(String),
    ApplicabilityUnknown { reference_id: String },
    MissingActionResult {
        reference_id: String,
        action_id: String,
    },
    MissingOutcome {
        reference_id: String,
        action_id: String,
    },
    CensoredOutcome {
        reference_id: String,
        action_id: String,
    },
    TimedOutOutcome {
        reference_id: String,
        action_id: String,
    },
    UnsupportedOutcome {
        reference_id: String,
        action_id: String,
    },
    ExecutionFailed {
        reference_id: String,
        action_id: String,
    },
    UnknownFeasibility {
        reference_id: String,
        action_id: String,
    },
    InfeasibleAction {
        reference_id: String,
        action_id: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnresolvedReason {
    MixedPointAndIntervalEvidence,
    IncompatibleIntervalSemantics,
    OverlappingIntervals,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceComparison {
    pub reference_id: String,
    pub dependency_family_id: String,
    pub applicability: Applicability,
    pub difference: Option<DifferenceInterval>,
    pub point_preference: Option<Preference>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionEnvelope {
    pub classification: DecisionClass,
    pub incomplete_reason: Option<IncompleteReason>,
    pub unresolved_reason: Option<UnresolvedReason>,
    pub applicable_reference_count: usize,
    pub inapplicable_reference_count: usize,
    // Descriptive common-mode family count. Never used as vote weight.
    pub dependency_family_count: usize,
    pub comparisons: Vec<ReferenceComparison>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvaluationError {
    EmptyField(&'static str),
    DuplicateActionId,
    InvalidEquivalenceMargin,
    DuplicateActionId,
    DuplicateReferenceId(String),
    InvalidDigest { field: &'static str },
    ContextMismatch { reference_id: String, action_id: String },
    NonFiniteDifference { reference_id: String },
    DuplicateReceiptId(String),
    NonFiniteMetricValue {
        reference_id: String,
        action_id: String,
    },
    InvalidMetricInterval {
        reference_id: String,
        action_id: String,
    },
    InvalidStatisticalInterval {
        reference_id: String,
    },
}

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for EvaluationError {}

/// Evaluate one metric for two frozen, caller-declared candidate actions.
///
/// A complete action set and known-feasible candidate actions are prerequisites.
/// Missingness, censoring, timeouts and unsupported reference cells produce an
/// Incomplete envelope; they are never imputed or counted as agreement.
pub fn evaluate_pair(
    context: &ComparisonContext,
    rows: &[ReferenceRow],
) -> Result<DecisionEnvelope, EvaluationError> {
    validate_context(context)?;

    if !context.action_set_complete {
        return Ok(empty_envelope(
            DecisionClass::Incomplete,
            Some(IncompleteReason::ActionSetDeclaredIncomplete),
        ));
    }

    let mut ordered_rows: Vec<&ReferenceRow> = rows.iter().collect();
    ordered_rows.sort_by(|left, right| left.reference_id.cmp(&right.reference_id));

    let mut seen_references = HashSet::with_capacity(ordered_rows.len());
    let mut seen_receipts = HashSet::with_capacity(ordered_rows.len() * 2);
    let mut expected = BTreeSet::new();
    for reference_id in &context.expected_reference_ids {
        validate_nonempty(reference_id, "expected_reference_id")?;
        if !expected.insert(reference_id.clone()) {
            return Err(EvaluationError::DuplicateReferenceId(reference_id.clone()));
        }
    }
    let provided: BTreeSet<String> =
        ordered_rows.iter().map(|row| row.reference_id.clone()).collect();
    if let Some(missing) = expected.difference(&provided).next() {
        return Ok(empty_envelope(
            DecisionClass::Incomplete,
            Some(IncompleteReason::MissingExpectedReference(missing.clone())),
        ));
    }
    if let Some(unexpected) = provided.difference(&expected).next() {
        return Ok(empty_envelope(
            DecisionClass::Incomplete,
            Some(IncompleteReason::UnexpectedReference(unexpected.clone())),
        ));
    }

    for row in &ordered_rows {
        validate_nonempty(&row.reference_id, "reference_id")?;
        validate_nonempty(&row.dependency_family_id, "dependency_family_id")?;
        if !seen_references.insert(row.reference_id.as_str()) {
            return Err(EvaluationError::DuplicateReferenceId(row.reference_id.clone()));
        }
        for result in [row.action_a.as_ref(), row.action_b.as_ref()]
            .into_iter()
            .flatten()
        {
            validate_nonempty(&result.receipt_id, "receipt_id")?;
            validate_digest(&result.execution_context_digest, "execution_context_digest")?;
            validate_digest(&result.action_payload_digest, "action_payload_digest")?;
            validate_digest(&result.output_digest, "output_digest")?;
            if !seen_receipts.insert(result.receipt_id.as_str()) {
                return Err(EvaluationError::DuplicateReceiptId(result.receipt_id.clone()));
            }
        }
    }

    let mut comparisons = Vec::with_capacity(ordered_rows.len());
    let mut families = BTreeSet::new();
    let mut applicable_count = 0usize;
    let mut inapplicable_count = 0usize;
    let mut common_semantics: Option<IntervalSemantics> = None;
    let mut common_action_payloads = std::collections::BTreeMap::new();
    let mut evidence_kind: Option<EvidenceKind> = None;
    let mut mixed_evidence = false;
    let mut incompatible_semantics = false;

    for row in ordered_rows {
        match row.applicability {
            Applicability::Unknown => {
                return Ok(DecisionEnvelope {
                    classification: DecisionClass::Incomplete,
                    incomplete_reason: Some(IncompleteReason::ApplicabilityUnknown {
                        reference_id: row.reference_id.clone(),
                    }),
                    unresolved_reason: None,
                    applicable_reference_count: applicable_count,
                    inapplicable_reference_count: inapplicable_count,
                    dependency_family_count: families.len(),
                    comparisons,
                });
            }
            Applicability::Inapplicable => {
                inapplicable_count += 1;
                comparisons.push(ReferenceComparison {
                    reference_id: row.reference_id.clone(),
                    dependency_family_id: row.dependency_family_id.clone(),
                    applicability: row.applicability,
                    difference: None,
                    point_preference: None,
                });
                continue;
            }
            Applicability::Applicable => {}
        }

        let Qualification::Qualified {
            record_id,
            source_artifact_digest,
        } = &row.qualification else {
            return Ok(DecisionEnvelope {
                classification: DecisionClass::OutOfScope,
                incomplete_reason: None,
                unresolved_reason: None,
                applicable_reference_count: applicable_count,
                inapplicable_reference_count: inapplicable_count,
                dependency_family_count: families.len(),
                comparisons,
            });
        };
        validate_nonempty(record_id, "qualification_record_id")?;
        validate_digest(source_artifact_digest, "reference_source_artifact_digest")?;

        applicable_count += 1;
        families.insert(row.dependency_family_id.clone());

        let Some(action_a) = row.action_a.as_ref() else {
            return Ok(incomplete_envelope(
                IncompleteReason::MissingActionResult {
                    reference_id: row.reference_id.clone(),
                    action_id: context.action_a_id.clone(),
                },
                applicable_count,
                inapplicable_count,
                families.len(),
                comparisons,
            ));
        };
        let Some(action_b) = row.action_b.as_ref() else {
            return Ok(incomplete_envelope(
                IncompleteReason::MissingActionResult {
                    reference_id: row.reference_id.clone(),
                    action_id: context.action_b_id.clone(),
                },
                applicable_count,
                inapplicable_count,
                families.len(),
                comparisons,
            ));
        };

        validate_action_identity(context, row, &context.action_a_id, action_a)?;
        validate_action_identity(context, row, &context.action_b_id, action_b)?;
        for (action_id, action) in [
            (context.action_a_id.as_str(), action_a),
            (context.action_b_id.as_str(), action_b),
        ] {
            if let Some(existing) = common_action_payloads.get(action_id) {
                if existing != &action.action_payload_digest {
                    return Err(EvaluationError::ContextMismatch {
                        reference_id: row.reference_id.clone(),
                        action_id: action_id.to_owned(),
                    });
                }
            } else {
                common_action_payloads.insert(action_id.to_owned(), action.action_payload_digest.clone());
            }
        }
        if action_a.execution_context_digest != action_b.execution_context_digest {
            return Err(EvaluationError::ContextMismatch {
                reference_id: row.reference_id.clone(),
                action_id: "shared_execution_context".into(),
            });
        }

        for (action_id, action) in [
            (context.action_a_id.as_str(), action_a),
            (context.action_b_id.as_str(), action_b),
        ] {
            match action.feasibility {
                Feasibility::Feasible => {}
                Feasibility::Unknown => {
                    return Ok(incomplete_envelope(
                        IncompleteReason::UnknownFeasibility {
                            reference_id: row.reference_id.clone(),
                            action_id: action_id.to_owned(),
                        },
                        applicable_count,
                        inapplicable_count,
                        families.len(),
                        comparisons,
                    ));
                }
                Feasibility::Infeasible => {
                    return Ok(incomplete_envelope(
                        IncompleteReason::InfeasibleAction {
                            reference_id: row.reference_id.clone(),
                            action_id: action_id.to_owned(),
                        },
                        applicable_count,
                        inapplicable_count,
                        families.len(),
                        comparisons,
                    ));
                }
            }
        }

        let evidence_a = match observed_evidence(
            &row.reference_id,
            &context.action_a_id,
            &action_a.outcome,
        ) {
            Ok(evidence) => evidence,
            Err(reason) => {
                return Ok(incomplete_envelope(
                    reason,
                    applicable_count,
                    inapplicable_count,
                    families.len(),
                    comparisons,
                ));
            }
        };
        let evidence_b = match observed_evidence(
            &row.reference_id,
            &context.action_b_id,
            &action_b.outcome,
        ) {
            Ok(evidence) => evidence,
            Err(reason) => {
                return Ok(incomplete_envelope(
                    reason,
                    applicable_count,
                    inapplicable_count,
                    families.len(),
                    comparisons,
                ));
            }
        };

        validate_metric_evidence(&row.reference_id, &context.action_a_id, evidence_a)?;
        validate_metric_evidence(&row.reference_id, &context.action_b_id, evidence_b)?;

        match (evidence_a, evidence_b) {
            (
                MetricEvidence::BoundedInterval {
                    lower: a_lower,
                    upper: a_upper,
                    semantics: a_semantics,
                },
                MetricEvidence::BoundedInterval {
                    lower: b_lower,
                    upper: b_upper,
                    semantics: b_semantics,
                },
            ) => {
                if a_semantics != b_semantics {
                    incompatible_semantics = true;
                }
                if let Some(existing) = common_semantics.as_ref() {
                    if existing != a_semantics || existing != b_semantics {
                        incompatible_semantics = true;
                    }
                } else {
                    common_semantics = Some(a_semantics.clone());
                }

                let (a_lower, a_upper) =
                    favorable_interval(*a_lower, *a_upper, context.metric.direction);
                let (b_lower, b_upper) =
                    favorable_interval(*b_lower, *b_upper, context.metric.direction);
                if a_semantics == b_semantics {
                    let difference = DifferenceInterval {
                        lower: a_lower - b_upper,
                        upper: a_upper - b_lower,
                    };
                    if !difference.lower.is_finite() || !difference.upper.is_finite() {
                        return Err(EvaluationError::NonFiniteDifference {
                            reference_id: row.reference_id.clone(),
                        });
                    }
                    comparisons.push(ReferenceComparison {
                        reference_id: row.reference_id.clone(),
                        dependency_family_id: row.dependency_family_id.clone(),
                        applicability: row.applicability,
                        difference: Some(difference),
                        point_preference: None,
                    });
                } else {
                    comparisons.push(ReferenceComparison {
                        reference_id: row.reference_id.clone(),
                        dependency_family_id: row.dependency_family_id.clone(),
                        applicability: row.applicability,
                        difference: None,
                        point_preference: None,
                    });
                }
                mark_evidence_kind(&mut evidence_kind, EvidenceKind::Interval, &mut mixed_evidence);
            }
            (
                MetricEvidence::PointEstimateOnly { value: a_value },
                MetricEvidence::PointEstimateOnly { value: b_value },
            ) => {
                let (a_value, b_value) = match context.metric.direction {
                    Direction::HigherIsBetter => (*a_value, *b_value),
                    Direction::LowerIsBetter => (-*a_value, -*b_value),
                };
                let difference = a_value - b_value;
                if !difference.is_finite() {
                    return Err(EvaluationError::NonFiniteDifference {
                        reference_id: row.reference_id.clone(),
                    });
                }
                let preference = if difference > context.metric.equivalence_margin {
                    Preference::ActionA
                } else if difference < -context.metric.equivalence_margin {
                    Preference::ActionB
                } else {
                    Preference::EquivalentWithinMargin
                };
                comparisons.push(ReferenceComparison {
                    reference_id: row.reference_id.clone(),
                    dependency_family_id: row.dependency_family_id.clone(),
                    applicability: row.applicability,
                    difference: None,
                    point_preference: Some(preference),
                });
                mark_evidence_kind(&mut evidence_kind, EvidenceKind::Point, &mut mixed_evidence);
            }
            _ => {
                mixed_evidence = true;
                mark_evidence_kind(&mut evidence_kind, EvidenceKind::Mixed, &mut mixed_evidence);
                comparisons.push(ReferenceComparison {
                    reference_id: row.reference_id.clone(),
                    dependency_family_id: row.dependency_family_id.clone(),
                    applicability: row.applicability,
                    difference: None,
                    point_preference: None,
                });
            }
        }
    }

    if applicable_count == 0 {
        return Ok(DecisionEnvelope {
            classification: DecisionClass::OutOfScope,
            incomplete_reason: None,
            unresolved_reason: None,
            applicable_reference_count: 0,
            inapplicable_reference_count: inapplicable_count,
            dependency_family_count: 0,
            comparisons,
        });
    }

    if mixed_evidence {
        return Ok(DecisionEnvelope {
            classification: DecisionClass::Unresolved,
            incomplete_reason: None,
            unresolved_reason: Some(UnresolvedReason::MixedPointAndIntervalEvidence),
            applicable_reference_count: applicable_count,
            inapplicable_reference_count: inapplicable_count,
            dependency_family_count: families.len(),
            comparisons,
        });
    }

    if incompatible_semantics {
        return Ok(DecisionEnvelope {
            classification: DecisionClass::Unresolved,
            incomplete_reason: None,
            unresolved_reason: Some(UnresolvedReason::IncompatibleIntervalSemantics),
            applicable_reference_count: applicable_count,
            inapplicable_reference_count: inapplicable_count,
            dependency_family_count: families.len(),
            comparisons,
        });
    }

    let classification = match evidence_kind {
        Some(EvidenceKind::Point) => DecisionClass::PointwiseOnly,
        Some(EvidenceKind::Interval) => {
            classify_intervals(&comparisons, context.metric.equivalence_margin)
        }
        Some(EvidenceKind::Mixed) => DecisionClass::Unresolved,
        None => DecisionClass::OutOfScope,
    };

    let unresolved_reason = if classification == DecisionClass::Unresolved {
        Some(UnresolvedReason::OverlappingIntervals)
    } else {
        None
    };

    Ok(DecisionEnvelope {
        classification,
        incomplete_reason: None,
        unresolved_reason,
        applicable_reference_count: applicable_count,
        inapplicable_reference_count: inapplicable_count,
        dependency_family_count: families.len(),
        comparisons,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EvidenceKind {
    Point,
    Interval,
    Mixed,
}

fn mark_evidence_kind(
    current: &mut Option<EvidenceKind>,
    next: EvidenceKind,
    mixed: &mut bool,
) {
    match *current {
        None => *current = Some(next),
        Some(existing) if existing == next => {}
        Some(_) => {
            *mixed = true;
            *current = Some(EvidenceKind::Mixed);
        }
    }
}

fn validate_context(context: &ComparisonContext) -> Result<(), EvaluationError> {
    validate_nonempty(&context.scenario_id, "scenario_id")?;
    validate_nonempty(&context.action_set_id, "action_set_id")?;
    validate_nonempty(&context.action_a_id, "action_a_id")?;
    validate_nonempty(&context.action_b_id, "action_b_id")?;
    validate_nonempty(&context.metric.metric_id, "metric_id")?;
    validate_nonempty(&context.metric.unit, "metric_unit")?;
    if context.action_a_id == context.action_b_id {
        return Err(EvaluationError::DuplicateActionId);
    }
    if context.expected_reference_ids.is_empty() {
        return Err(EvaluationError::EmptyField("expected_reference_ids"));
    }
    if !context.metric.equivalence_margin.is_finite()
        || context.metric.equivalence_margin < 0.0
    {
        return Err(EvaluationError::InvalidEquivalenceMargin);
    }
    Ok(())
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), EvaluationError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(EvaluationError::InvalidDigest { field });
    }
    Ok(())
}

fn validate_action_identity(
    context: &ComparisonContext,
    row: &ReferenceRow,
    expected_action_id: &str,
    action: &ActionResult,
) -> Result<(), EvaluationError> {
    let matches = action.scenario_id == context.scenario_id
        && action.action_set_id == context.action_set_id
        && action.reference_id == row.reference_id
        && action.action_id == expected_action_id
        && action.metric_id == context.metric.metric_id
        && action.unit == context.metric.unit;
    if !matches {
        return Err(EvaluationError::ContextMismatch {
            reference_id: row.reference_id.clone(),
            action_id: expected_action_id.to_owned(),
        });
    }
    Ok(())
}

fn validate_nonempty(value: &str, field: &'static str) -> Result<(), EvaluationError> {
    if value.trim().is_empty() {
        Err(EvaluationError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn observed_evidence<'a>(
    reference_id: &str,
    action_id: &str,
    outcome: &'a MetricOutcome,
) -> Result<&'a MetricEvidence, IncompleteReason> {
    match outcome {
        MetricOutcome::Observed(evidence) => Ok(evidence),
        MetricOutcome::Missing => Err(IncompleteReason::MissingOutcome {
            reference_id: reference_id.to_owned(),
            action_id: action_id.to_owned(),
        }),
        MetricOutcome::Censored => Err(IncompleteReason::CensoredOutcome {
            reference_id: reference_id.to_owned(),
            action_id: action_id.to_owned(),
        }),
        MetricOutcome::TimedOut => Err(IncompleteReason::TimedOutOutcome {
            reference_id: reference_id.to_owned(),
            action_id: action_id.to_owned(),
        }),
        MetricOutcome::Unsupported => Err(IncompleteReason::UnsupportedOutcome {
            reference_id: reference_id.to_owned(),
            action_id: action_id.to_owned(),
        }),
        MetricOutcome::ExecutionFailed => Err(IncompleteReason::ExecutionFailed {
            reference_id: reference_id.to_owned(),
            action_id: action_id.to_owned(),
        }),
    }
}

fn validate_metric_evidence(
    reference_id: &str,
    action_id: &str,
    evidence: &MetricEvidence,
) -> Result<(), EvaluationError> {
    match evidence {
        MetricEvidence::BoundedInterval { lower, upper, semantics } => {
            if !lower.is_finite() || !upper.is_finite() {
                return Err(EvaluationError::NonFiniteMetricValue {
                    reference_id: reference_id.to_owned(),
                    action_id: action_id.to_owned(),
                });
            }
            if lower > upper {
                return Err(EvaluationError::InvalidMetricInterval {
                    reference_id: reference_id.to_owned(),
                    action_id: action_id.to_owned(),
                });
            }
            if let IntervalSemantics::StatisticalInterval { method, confidence_ppm } = semantics {
                validate_nonempty(method, "interval_method")?;
                if *confidence_ppm == 0 || *confidence_ppm >= 1_000_000 {
                    return Err(EvaluationError::InvalidStatisticalInterval {
                        reference_id: reference_id.to_owned(),
                    });
                }
            }
        }
        MetricEvidence::PointEstimateOnly { value } => {
            if !value.is_finite() {
                return Err(EvaluationError::NonFiniteMetricValue {
                    reference_id: reference_id.to_owned(),
                    action_id: action_id.to_owned(),
                });
            }
        }
    }
    Ok(())
}

fn favorable_interval(
    lower: f64,
    upper: f64,
    direction: Direction,
) -> (f64, f64) {
    match direction {
        Direction::HigherIsBetter => (lower, upper),
        Direction::LowerIsBetter => (-upper, -lower),
    }
}

fn classify_intervals(
    comparisons: &[ReferenceComparison],
    margin: f64,
) -> DecisionClass {
    let mut all_a = true;
    let mut all_b = true;
    let mut all_equivalent = true;
    let mut any_a = false;
    let mut any_b = false;

    for comparison in comparisons {
        let Some(difference) = comparison.difference.as_ref() else {
            continue;
        };
        let a = difference.lower > margin;
        let b = difference.upper < -margin;
        let equivalent = difference.lower >= -margin && difference.upper <= margin;

        all_a &= a;
        all_b &= b;
        all_equivalent &= equivalent;
        any_a |= a;
        any_b |= b;
    }

    if any_a && any_b {
        DecisionClass::ReferenceSensitive
    } else if all_a {
        DecisionClass::PortfolioDominatesA
    } else if all_b {
        DecisionClass::PortfolioDominatesB
    } else if all_equivalent {
        DecisionClass::PortfolioEquivalent
    } else {
        DecisionClass::Unresolved
    }
}

fn empty_envelope(
    classification: DecisionClass,
    incomplete_reason: Option<IncompleteReason>,
) -> DecisionEnvelope {
    DecisionEnvelope {
        classification,
        incomplete_reason,
        unresolved_reason: None,
        applicable_reference_count: 0,
        inapplicable_reference_count: 0,
        dependency_family_count: 0,
        comparisons: Vec::new(),
    }
}

fn incomplete_envelope(
    reason: IncompleteReason,
    applicable_reference_count: usize,
    inapplicable_reference_count: usize,
    dependency_family_count: usize,
    comparisons: Vec<ReferenceComparison>,
) -> DecisionEnvelope {
    DecisionEnvelope {
        classification: DecisionClass::Incomplete,
        incomplete_reason: Some(reason),
        unresolved_reason: None,
        applicable_reference_count,
        inapplicable_reference_count,
        dependency_family_count,
        comparisons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(direction: Direction, margin: f64) -> ComparisonContext {
        ComparisonContext {
            scenario_id: "crossing-001".into(),
            action_set_id: "actions-v1".into(),
            action_set_complete: true,
            expected_reference_ids: Vec::new(),
            action_a_id: "continue".into(),
            action_b_id: "yield".into(),
            metric: MetricDefinition {
                metric_id: "clearance-margin".into(),
                unit: "m".into(),
                direction,
                equivalence_margin: margin,
            },
        }
    }

    fn semantics() -> IntervalSemantics {
        IntervalSemantics::ModelUncertaintyInterval
    }

    fn interval(lower: f64, upper: f64) -> MetricOutcome {
        MetricOutcome::Observed(MetricEvidence::BoundedInterval {
            lower,
            upper,
            semantics: semantics(),
        })
    }

    fn point(value: f64) -> MetricOutcome {
        MetricOutcome::Observed(MetricEvidence::PointEstimateOnly { value })
    }

    fn action(reference_id: &str, action_id: &str, receipt_id: &str, outcome: MetricOutcome) -> ActionResult {
        let action_payload_digest = if action_id == "continue" { "b" } else { "c" };
        ActionResult {
            receipt_id: receipt_id.into(),
            scenario_id: "crossing-001".into(),
            action_set_id: "actions-v1".into(),
            reference_id: reference_id.into(),
            action_id: action_id.into(),
            metric_id: "clearance-margin".into(),
            unit: "m".into(),
            execution_context_digest: "d".repeat(64),
            action_payload_digest: action_payload_digest.repeat(64),
            output_digest: format!(
                "{}{}",
                reference_id
                    .bytes()
                    .find(u8::is_ascii_hexdigit)
                    .map(char::from)
                    .unwrap_or('a')
                    .to_ascii_lowercase(),
                if action_id == "continue" { "e" } else { "f" }.repeat(63)
            ),
            feasibility: Feasibility::Feasible,
            outcome,
        }
    }

    fn evaluate(
        context: &ComparisonContext,
        rows: &[ReferenceRow],
    ) -> Result<DecisionEnvelope, EvaluationError> {
        let mut bound_context = context.clone();
        if bound_context.expected_reference_ids.is_empty() {
            bound_context.expected_reference_ids =
                rows.iter().map(|row| row.reference_id.clone()).collect();
        }
        evaluate_pair(&bound_context, rows)
    }

    fn row(
        reference_id: &str,
        family: &str,
        a: MetricOutcome,
        b: MetricOutcome,
    ) -> ReferenceRow {
        ReferenceRow {
            reference_id: reference_id.into(),
            dependency_family_id: family.into(),
            qualification: Qualification::Qualified {
                record_id: format!("qualification-{reference_id}"),
                source_artifact_digest: "a".repeat(64),
            },
            applicability: Applicability::Applicable,
            action_a: Some(action(reference_id, "continue", &format!("{reference_id}-a"), a)),
            action_b: Some(action(reference_id, "yield", &format!("{reference_id}-b"), b)),
        }
    }

    #[test]
    fn unanimous_interval_preference_is_portfolio_dominance() {
        let rows = vec![
            row("r0", "analytic", interval(0.20, 0.25), interval(-0.10, -0.05)),
            row("r1", "independent-code", interval(0.15, 0.22), interval(-0.08, -0.02)),
        ];
        let envelope = evaluate(&context(Direction::HigherIsBetter, 0.02), &rows).unwrap();
        assert_eq!(envelope.classification, DecisionClass::PortfolioDominatesA);
        assert_eq!(envelope.applicable_reference_count, 2);
        assert_eq!(envelope.dependency_family_count, 2);
    }

    #[test]
    fn opposite_reference_preferences_are_not_averaged() {
        let rows = vec![
            row("r0", "family-a", interval(0.20, 0.30), interval(-0.10, -0.05)),
            row("r1", "family-b", interval(-0.35, -0.25), interval(0.20, 0.30)),
        ];
        let envelope = evaluate_pair(&context(Direction::HigherIsBetter, 0.02), &rows).unwrap();
        assert_eq!(envelope.classification, DecisionClass::ReferenceSensitive);
    }

    #[test]
    fn overlapping_intervals_remain_unresolved() {
        let rows = vec![
            row("r0", "family-a", interval(0.72, 0.82), interval(0.78, 0.88)),
            row("r1", "family-b", interval(0.70, 0.80), interval(0.76, 0.86)),
        ];
        let envelope = evaluate_pair(&context(Direction::HigherIsBetter, 0.02), &rows).unwrap();
        assert_eq!(envelope.classification, DecisionClass::Unresolved);
        assert_eq!(
            envelope.unresolved_reason,
            Some(UnresolvedReason::OverlappingIntervals)
        );
    }

    #[test]
    fn equivalence_requires_each_reference_interval_inside_margin() {
        let rows = vec![
            row("r0", "family-a", interval(1.00, 1.01), interval(1.00, 1.01)),
            row("r1", "family-b", interval(2.00, 2.01), interval(2.005, 2.01)),
        ];
        let envelope = evaluate_pair(&context(Direction::HigherIsBetter, 0.02), &rows).unwrap();
        assert_eq!(envelope.classification, DecisionClass::PortfolioEquivalent);
    }

    #[test]
    fn lower_is_better_is_normalized_before_comparison() {
        let rows = vec![
            row("r0", "analytic", interval(1.0, 1.2), interval(2.0, 2.2)),
            row("r1", "independent", interval(0.8, 1.1), interval(1.8, 2.0)),
        ];
        let envelope = evaluate_pair(&context(Direction::LowerIsBetter, 0.01), &rows).unwrap();
        assert_eq!(envelope.classification, DecisionClass::PortfolioDominatesA);
    }

    #[test]
    fn point_estimates_are_pointwise_only_not_zero_width_intervals() {
        let rows = vec![
            row("r0", "family-a", point(4.0), point(2.0)),
            row("r1", "family-b", point(3.0), point(1.0)),
        ];
        let envelope = evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &rows).unwrap();
        assert_eq!(envelope.classification, DecisionClass::PointwiseOnly);
        assert!(envelope.comparisons.iter().all(|item| item.difference.is_none()));
        assert!(envelope.comparisons.iter().all(|item| item.point_preference == Some(Preference::ActionA)));
    }

    #[test]
    fn mixed_point_and_interval_evidence_is_unresolved() {
        let rows = vec![
            row("r0", "family-a", interval(2.0, 2.2), interval(1.0, 1.2)),
            row("r1", "family-b", point(2.0), point(1.0)),
        ];
        let envelope = evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &rows).unwrap();
        assert_eq!(envelope.classification, DecisionClass::Unresolved);
        assert_eq!(
            envelope.unresolved_reason,
            Some(UnresolvedReason::MixedPointAndIntervalEvidence)
        );
    }

    #[test]
    fn different_interval_semantics_cannot_be_pooled() {
        let mut second = row("r1", "family-b", interval(2.0, 2.2), interval(1.0, 1.2));
        for result in [&mut second.action_a, &mut second.action_b].into_iter().flatten() {
            if let MetricOutcome::Observed(MetricEvidence::BoundedInterval { semantics, .. }) = &mut result.outcome {
                *semantics = IntervalSemantics::MeasurementInterval;
            }
        }
        let rows = vec![
            row("r0", "family-a", interval(2.0, 2.2), interval(1.0, 1.2)),
            second,
        ];
        let envelope = evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &rows).unwrap();
        assert_eq!(envelope.classification, DecisionClass::Unresolved);
        assert_eq!(
            envelope.unresolved_reason,
            Some(UnresolvedReason::IncompatibleIntervalSemantics)
        );
    }

    #[test]
    fn inapplicable_reference_is_excluded_only_for_this_measurand() {
        let mut actor_only = row("actor-only", "actor-policy", interval(5.0, 6.0), interval(1.0, 2.0));
        actor_only.applicability = Applicability::Inapplicable;
        actor_only.action_a = None;
        actor_only.action_b = None;

        let rows = vec![
            row("r0", "analytic", interval(2.0, 2.2), interval(1.0, 1.2)),
            actor_only,
        ];
        let envelope = evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &rows).unwrap();
        assert_eq!(envelope.classification, DecisionClass::PortfolioDominatesA);
        assert_eq!(envelope.applicable_reference_count, 1);
        assert_eq!(envelope.inapplicable_reference_count, 1);
    }

    #[test]
    fn unknown_applicability_blocks_complete_portfolio_claim() {
        let mut row = row("r0", "family-a", interval(2.0, 2.2), interval(1.0, 1.2));
        row.applicability = Applicability::Unknown;
        let envelope = evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &[row]).unwrap();
        assert_eq!(envelope.classification, DecisionClass::Incomplete);
        assert_eq!(
            envelope.incomplete_reason,
            Some(IncompleteReason::ApplicabilityUnknown {
                reference_id: "r0".into()
            })
        );
    }

    #[test]
    fn timed_out_required_cell_is_incomplete_not_agreement() {
        let rows = vec![row(
            "r0",
            "family-a",
            MetricOutcome::TimedOut,
            interval(1.0, 1.2),
        )];
        let envelope = evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &rows).unwrap();
        assert_eq!(envelope.classification, DecisionClass::Incomplete);
        assert_eq!(
            envelope.incomplete_reason,
            Some(IncompleteReason::TimedOutOutcome {
                reference_id: "r0".into(),
                action_id: "continue".into()
            })
        );
    }

    #[test]
    fn infeasible_or_unknown_action_is_never_ranked_by_utility() {
        let mut bad = row("r0", "family-a", interval(9.0, 10.0), interval(1.0, 2.0));
        bad.action_a.as_mut().unwrap().feasibility = Feasibility::Infeasible;
        let envelope = evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &[bad]).unwrap();
        assert_eq!(envelope.classification, DecisionClass::Incomplete);
        assert!(matches!(envelope.incomplete_reason, Some(IncompleteReason::InfeasibleAction { .. })));
    }

    #[test]
    fn duplicate_reference_and_receipt_identities_are_rejected() {
        let duplicate_reference = vec![
            row("r0", "family-a", interval(2.0, 2.2), interval(1.0, 1.2)),
            row("r0", "family-b", interval(2.0, 2.2), interval(1.0, 1.2)),
        ];
        assert_eq!(
            evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &duplicate_reference),
            Err(EvaluationError::DuplicateReferenceId("r0".into()))
        );

        let mut duplicate_receipt = row("r0", "family-a", interval(2.0, 2.2), interval(1.0, 1.2));
        duplicate_receipt.action_b.as_mut().unwrap().receipt_id = "r0-a".into();
        assert_eq!(
            evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &[duplicate_receipt]),
            Err(EvaluationError::DuplicateReceiptId("r0-a".into()))
        );
    }

    #[test]
    fn incomplete_candidate_set_suppresses_comparison() {
        let mut ctx = context(Direction::HigherIsBetter, 0.1);
        ctx.action_set_complete = false;
        let rows = vec![row("r0", "family-a", interval(2.0, 2.2), interval(1.0, 1.2))];
        let envelope = evaluate(&ctx, &rows).unwrap();
        assert_eq!(envelope.classification, DecisionClass::Incomplete);
        assert_eq!(
            envelope.incomplete_reason,
            Some(IncompleteReason::ActionSetDeclaredIncomplete)
        );
    }

    #[test]
    fn common_mode_family_is_metadata_not_a_vote_weight() {
        let rows = vec![
            row("r0", "shared-root", interval(2.0, 2.2), interval(1.0, 1.2)),
            row("r1", "shared-root", interval(2.1, 2.3), interval(1.0, 1.2)),
        ];
        let envelope = evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &rows).unwrap();
        assert_eq!(envelope.classification, DecisionClass::PortfolioDominatesA);
        assert_eq!(envelope.applicable_reference_count, 2);
        assert_eq!(envelope.dependency_family_count, 1);
    }

    #[test]
    fn input_reference_order_does_not_change_canonical_result() {
        let first = row("r1", "family-b", interval(2.0, 2.2), interval(1.0, 1.2));
        let second = row("r0", "family-a", interval(2.1, 2.3), interval(1.0, 1.2));
        let left = evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &[first.clone(), second.clone()]).unwrap();
        let right = evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &[second, first]).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.comparisons[0].reference_id, "r0");
    }

    #[test]
    fn missing_expected_reference_suppresses_portfolio_dominance() {
        let mut ctx = context(Direction::HigherIsBetter, 0.1);
        ctx.expected_reference_ids = vec!["r0".into(), "r1".into()];
        let rows = vec![row("r0", "family-a", interval(2.0, 2.2), interval(1.0, 1.2))];
        let envelope = evaluate_pair(&ctx, &rows).unwrap();
        assert_eq!(envelope.classification, DecisionClass::Incomplete);
        assert_eq!(
            envelope.incomplete_reason,
            Some(IncompleteReason::MissingExpectedReference("r1".into()))
        );
    }

    #[test]
    fn mismatched_action_context_is_rejected() {
        let mut bad = row("r0", "family-a", interval(2.0, 2.2), interval(1.0, 1.2));
        bad.action_a.as_mut().unwrap().action_set_id = "other-actions".into();
        assert!(matches!(
            evaluate(&context(Direction::HigherIsBetter, 0.1), &[bad]),
            Err(EvaluationError::ContextMismatch { .. })
        ));
    }

    #[test]
    fn different_action_payloads_across_references_reject_pairing() {
        let mut second = row("r1", "family-b", interval(2.0, 2.2), interval(1.0, 1.2));
        second.action_a.as_mut().unwrap().action_payload_digest = "9".repeat(64);
        let rows = vec![
            row("r0", "family-a", interval(2.0, 2.2), interval(1.0, 1.2)),
            second,
        ];
        assert!(matches!(
            evaluate(&context(Direction::HigherIsBetter, 0.1), &rows),
            Err(EvaluationError::ContextMismatch { .. })
        ));
    }

    #[test]
    fn overflowing_pairwise_difference_is_rejected() {
        let rows = vec![row("r0", "family-a", point(f64::MAX), point(-f64::MAX))];
        assert!(matches!(
            evaluate(&context(Direction::HigherIsBetter, 0.1), &rows),
            Err(EvaluationError::NonFiniteDifference { .. })
        ));
    }

    #[test]
    fn no_applicable_qualified_reference_is_out_of_scope() {
        let mut unqualified = row("r0", "family-a", interval(2.0, 2.2), interval(1.0, 1.2));
        unqualified.qualification = Qualification::NotQualified;
        let envelope = evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &[unqualified]).unwrap();
        assert_eq!(envelope.classification, DecisionClass::OutOfScope);
    }

    #[test]
    fn invalid_intervals_and_nonfinite_values_are_rejected() {
        let bad = MetricOutcome::Observed(MetricEvidence::BoundedInterval {
            lower: 3.0,
            upper: 2.0,
            semantics: semantics(),
        });
        let rows = vec![row("r0", "family-a", bad, interval(1.0, 1.2))];
        assert!(matches!(
            evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &rows),
            Err(EvaluationError::InvalidMetricInterval { .. })
        ));

        let nan = MetricOutcome::Observed(MetricEvidence::PointEstimateOnly { value: f64::NAN });
        let rows = vec![row("r0", "family-a", nan, point(1.0))];
        assert!(matches!(
            evaluate_pair(&context(Direction::HigherIsBetter, 0.1), &rows),
            Err(EvaluationError::NonFiniteMetricValue { .. })
        ));
    }
}
