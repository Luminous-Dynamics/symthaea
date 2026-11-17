//! Pareto and Q_ENG trade-off audit for lifetime fusion plant service.
//!
//! Q_ENG is a scalar comparator over availability and normalized engineering
//! burden. This module deliberately keeps the wider service/cost objective
//! vector visible: a scalar preference between non-dominated candidates is a
//! trade-off, not a universal ranking.

use std::collections::BTreeSet;

pub const PARETO_TRADEOFF_VERSION: &str = "fusion-pareto-tradeoff/0.1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ObjectiveDimension {
    LifetimeNetElectricEnergyMwh,
    WholeLifeCost,
    OperationalAvailability,
    PlannedDowntimeHours,
    UnplannedDowntimeHours,
    EngineeringBurdenIndex,
    NetExportBoundWidthMwe,
}

impl ObjectiveDimension {
    pub const ALL: [Self; 7] = [
        Self::LifetimeNetElectricEnergyMwh,
        Self::WholeLifeCost,
        Self::OperationalAvailability,
        Self::PlannedDowntimeHours,
        Self::UnplannedDowntimeHours,
        Self::EngineeringBurdenIndex,
        Self::NetExportBoundWidthMwe,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::LifetimeNetElectricEnergyMwh => "lifetime-net-electric-energy-mwh",
            Self::WholeLifeCost => "whole-life-cost",
            Self::OperationalAvailability => "operational-availability-fraction",
            Self::PlannedDowntimeHours => "planned-downtime-hours",
            Self::UnplannedDowntimeHours => "unplanned-downtime-hours",
            Self::EngineeringBurdenIndex => "engineering-burden-index",
            Self::NetExportBoundWidthMwe => "net-export-lower-bound-width-mwe",
        }
    }

    const fn direction(self) -> ObjectiveDirection {
        match self {
            Self::LifetimeNetElectricEnergyMwh | Self::OperationalAvailability => {
                ObjectiveDirection::Maximize
            }
            Self::WholeLifeCost
            | Self::PlannedDowntimeHours
            | Self::UnplannedDowntimeHours
            | Self::EngineeringBurdenIndex
            | Self::NetExportBoundWidthMwe => ObjectiveDirection::Minimize,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObjectiveDirection {
    Maximize,
    Minimize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateEligibility {
    Eligible,
    BlockedByHardGate,
    UnknownHardGate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectiveComparisonProfile {
    pub profile_id: String,
    pub profile_sha256: String,
    pub service_boundary_ref: String,
    pub lifetime_horizon_ref: String,
    /// Pins currency, price year, financing/discount assumptions and cost scope.
    pub cost_basis_ref: String,
    /// Pins the externally defined normalization used to produce EBI.
    pub ebi_normalization_ref: String,
    /// Pins the UQ method and simultaneous/one-sided coverage definition.
    pub uncertainty_basis_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SourcedScalar {
    pub value: f64,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateObjectiveVector {
    pub candidate_id: String,
    pub candidate_revision: String,
    pub profile_id: String,
    pub profile_sha256: String,
    pub objective_vector_receipt_ref: String,
    pub objective_vector_receipt_sha256: String,
    pub eligibility: CandidateEligibility,
    /// MWh delivered at the declared service boundary over the fixed horizon.
    pub lifetime_net_electric_energy_mwh: Option<SourcedScalar>,
    /// Cost on the profile's fixed currency/price-year/discount/accounting basis.
    pub whole_life_cost: Option<SourcedScalar>,
    pub operational_availability: Option<SourcedScalar>,
    pub planned_downtime_hours: Option<SourcedScalar>,
    pub unplanned_downtime_hours: Option<SourcedScalar>,
    /// Dimensionless EBI calculated upstream under the frozen normalization.
    pub engineering_burden_index: Option<SourcedScalar>,
    /// Width of a method/coverage-matched net-export uncertainty interval.
    pub net_export_bound_width_mwe: Option<SourcedScalar>,
    pub ebi_normalization_ref: String,
    pub uncertainty_basis_ref: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QEngValue {
    pub availability_fraction: f64,
    pub engineering_burden_index: f64,
    pub q_eng: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DominanceRelation {
    FirstDominatesSecond,
    SecondDominatesFirst,
    Equivalent,
    NonDominatedTradeoff,
    NotComparable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarPreference {
    First,
    Second,
    Tie,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarTradeoffDisposition {
    NoScalarPreference,
    ScalarPreferenceAgreesWithDominance,
    ScalarPreferenceHidesOmittedObjectiveTradeoff,
    ScalarPreferenceBetweenEquivalentVectors,
    ScalarAndVectorDominanceInconsistent,
    NotComparable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComparisonDiagnostic {
    EmptyProfileIdentity,
    InvalidProfileDigest,
    EmptyProfileBoundary,
    EmptyLifetimeHorizon,
    EmptyCostBasis,
    EmptyEbiNormalization,
    EmptyUncertaintyBasis,
    EmptyCandidateIdentity,
    DuplicateCandidateIdentity,
    EmptyObjectiveReceipt,
    InvalidObjectiveReceiptDigest,
    ProfileIdentityMismatch,
    ProfileDigestMismatch,
    EbiNormalizationMismatch,
    UncertaintyBasisMismatch,
    CandidateBlockedByHardGate,
    CandidateHardGateUnknown,
    MissingObjective(ObjectiveDimension),
    NonFiniteObjective(ObjectiveDimension),
    InvalidObjectiveRange(ObjectiveDimension),
    MissingObjectiveEvidence(ObjectiveDimension),
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseTradeoffReport {
    pub contract_version: &'static str,
    pub first_candidate_id: String,
    pub second_candidate_id: String,
    pub first_q_eng: Option<QEngValue>,
    pub second_q_eng: Option<QEngValue>,
    pub scalar_preference: ScalarPreference,
    pub dominance: DominanceRelation,
    pub scalar_tradeoff: ScalarTradeoffDisposition,
    /// Dimensions on which the scalar-preferred candidate is worse.
    pub scalar_winner_worse_dimensions: Vec<ObjectiveDimension>,
    pub first_diagnostics: Vec<ComparisonDiagnostic>,
    pub second_diagnostics: Vec<ComparisonDiagnostic>,
    pub claim_ceiling: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParetoFrontStatus {
    NonDominated,
    Dominated,
    NotComparable,
    BlockedByHardGate,
    UnknownHardGate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParetoCandidateStatus {
    pub candidate_id: String,
    pub status: ParetoFrontStatus,
    pub dominated_by: Vec<String>,
    pub diagnostics: Vec<ComparisonDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParetoFrontReport {
    pub contract_version: &'static str,
    pub profile_id: String,
    pub frontier_candidate_ids: Vec<String>,
    pub candidates: Vec<ParetoCandidateStatus>,
    pub diagnostics: Vec<ComparisonDiagnostic>,
    pub claim_ceiling: &'static str,
}

pub fn calculate_q_eng(
    availability_fraction: f64,
    engineering_burden_index: f64,
) -> Option<QEngValue> {
    if !availability_fraction.is_finite()
        || !(0.0..=1.0).contains(&availability_fraction)
        || !engineering_burden_index.is_finite()
        || engineering_burden_index <= 0.0
    {
        return None;
    }

    let q_eng = availability_fraction / engineering_burden_index;
    q_eng.is_finite().then_some(QEngValue {
        availability_fraction,
        engineering_burden_index,
        q_eng,
    })
}

fn validate_profile(profile: &ObjectiveComparisonProfile) -> Vec<ComparisonDiagnostic> {
    let mut diagnostics = Vec::new();
    if profile.profile_id.trim().is_empty() {
        diagnostics.push(ComparisonDiagnostic::EmptyProfileIdentity);
    }
    if !is_sha256(&profile.profile_sha256) {
        diagnostics.push(ComparisonDiagnostic::InvalidProfileDigest);
    }
    if profile.service_boundary_ref.trim().is_empty() {
        diagnostics.push(ComparisonDiagnostic::EmptyProfileBoundary);
    }
    if profile.lifetime_horizon_ref.trim().is_empty() {
        diagnostics.push(ComparisonDiagnostic::EmptyLifetimeHorizon);
    }
    if profile.cost_basis_ref.trim().is_empty() {
        diagnostics.push(ComparisonDiagnostic::EmptyCostBasis);
    }
    if profile.ebi_normalization_ref.trim().is_empty() {
        diagnostics.push(ComparisonDiagnostic::EmptyEbiNormalization);
    }
    if profile.uncertainty_basis_ref.trim().is_empty() {
        diagnostics.push(ComparisonDiagnostic::EmptyUncertaintyBasis);
    }
    diagnostics
}

fn validate_candidate(
    profile: &ObjectiveComparisonProfile,
    candidate: &CandidateObjectiveVector,
 -> Vec<ComparisonDiagnostic> {
    let mut diagnostics = Vec::new();
    if candidate.candidate_id.trim().is_empty() || candidate.candidate_revision.trim().is_empty() {
        diagnostics.push(ComparisonDiagnostic::EmptyCandidateIdentity);
    }
    if candidate.objective_vector_receipt_ref.trim().is_empty() {
        diagnostics.push(ComparisonDiagnostic::EmptyObjectiveReceipt);
    }
    if !is_sha256(&candidate.objective_vector_receipt_sha256) {
        diagnostics.push(ComparisonDiagnostic::InvalidObjectiveReceiptDigest);
    }
    if candidate.profile_id != profile.profile_id {
        diagnostics.push(ComparisonDiagnostic::ProfileIdentityMismatch);
    }
    if candidate.profile_sha256 != profile.profile_sha256 {
        diagnostics.push(ComparisonDiagnostic::ProfileDigestMismatch);
    }
    if candidate.ebi_normalization_ref != profile.ebi_normalization_ref {
        diagnostics.push(ComparisonDiagnostic::EbiNormalizationMismatch);
    }
    if candidate.uncertainty_basis_ref != profile.uncertainty_basis_ref {
        diagnostics.push(ComparisonDiagnostic::UncertaintyBasisMismatch);
    }
    match candidate.eligibility {
        CandidateEligibility::BlockedByHardGate => {
            diagnostics.push(ComparisonDiagnostic::CandidateBlockedByHardGate);
        }
        CandidateEligibility::UnknownHardGate => {
            diagnostics.push(ComparisonDiagnostic::CandidateHardGateUnknown);
        }
        CandidateEligibility::Eligible => {}
    }

    for dimension in ObjectiveDimension::ALL {
        let Some(value) = objective_value(candidate, dimension) else {
            diagnostics.push(ComparisonDiagnostic::MissingObjective(dimension));
            continue;
        };
        if !value.value.is_finite() {
            diagnostics.push(ComparisonDiagnostic::NonFiniteObjective(dimension));
        }
        if value.evidence_ref.trim().is_empty() {
            diagnostics.push(ComparisonDiagnostic::MissingObjectiveEvidence(dimension));
        }
        let invalid_range = match dimension {
            ObjectiveDimension::OperationalAvailability => !(0.0..=1.0).contains(&value.value),
            ObjectiveDimension::WholeLifeCost
            | ObjectiveDimension::PlannedDowntimeHours
            | ObjectiveDimension::UnplannedDowntimeHours
            | ObjectiveDimension::NetExportBoundWidthMwe => value.value < 0.0,
            ObjectiveDimension::EngineeringBurdenIndex => value.value <= 0.0,
            ObjectiveDimension::LifetimeNetElectricEnergyMwh => false,
        };
        if invalid_range {
            diagnostics.push(ComparisonDiagnostic::InvalidObjectiveRange(dimension));
        }
    }

    diagnostics
}

fn objective_value(
    candidate: &CandidateObjectiveVector,
    dimension: ObjectiveDimension,
) -> Option<&SourcedScalar> {
    match dimension {
        ObjectiveDimension::LifetimeNetElectricEnergyMwh => {
            candidate.lifetime_net_electric_energy_mwh.as_ref()
        }
        ObjectiveDimension::WholeLifeCost => candidate.whole_life_cost.as_ref(),
        ObjectiveDimension::OperationalAvailability => candidate.operational_availability.as_ref(),
        ObjectiveDimension::PlannedDowntimeHours => candidate.planned_downtime_hours.as_ref(),
        ObjectiveDimension::UnplannedDowntimeHours => candidate.unplanned_downtime_hours.as_ref(),
        ObjectiveDimension::EngineeringBurdenIndex => candidate.engineering_burden_index.as_ref(),
        ObjectiveDimension::NetExportBoundWidthMwe => candidate.net_export_bound_width_mwe.as_ref(),
    }
}

fn q_eng(candidate: &CandidateObjectiveVector) -> Option<QEngValue> {
    let availability = candidate.operational_availability.as_ref()?.value;
    let burden = candidate.engineering_burden_index.as_ref()?.value;
    calculate_q_eng(availability, burden)
}

fn scalar_preference(first: Option<QEngValue>, second: Option<QEngValue>) -> ScalarPreference {
    match (first, second) {
        (Some(first), Some(second)) if first.q_eng > second.q_eng => ScalarPreference::First,
        (Some(first), Some(second)) if second.q_eng > first.q_eng => ScalarPreference::Second,
        (Some(_), Some(_)) => ScalarPreference::Tie,
        _ => ScalarPreference::Unavailable,
    }
}

fn dominance(
    first: &CandidateObjectiveVector,
    second: &CandidateObjectiveVector,
) -> DominanceRelation {
    let mut first_better = false;
    let mut second_better = false;

    for dimension in ObjectiveDimension::ALL {
        let Some(first_value) = objective_value(first, dimension).map(|item| item.value) else {
            return DominanceRelation::NotComparable;
        };
        let Some(second_value) = objective_value(second, dimension).map(|item| item.value) else {
            return DominanceRelation::NotComparable;
        };
        match dimension.direction() {
            ObjectiveDirection::Maximize => {
                first_better |= first_value > second_value;
                second_better |= second_value > first_value;
            }
            ObjectiveDirection::Minimize => {
                first_better |= first_value < second_value;
                second_better |= second_value < first_value;
            }
        }
    }

    match (first_better, second_better) {
        (true, false) => DominanceRelation::FirstDominatesSecond,
        (false, true) => DominanceRelation::SecondDominatesFirst,
        (true, true) => DominanceRelation::NonDominatedTradeoff,
        (false, false) => DominanceRelation::Equivalent,
    }
}

fn worse_dimensions(
    winner: &CandidateObjectiveVector,
    other: &CandidateObjectiveVector,
) -> Vec<ObjectiveDimension> {
    ObjectiveDimension::ALL
        .into_iter()
        .filter(|dimension| {
            let Some(winner_value) = objective_value(winner, *dimension).map(|item| item.value)
            else {
                return false;
            };
            let Some(other_value) = objective_value(other, *dimension).map(|item| item.value)
            else {
                return false;
            };
            match dimension.direction() {
                ObjectiveDirection::Maximize => winner_value < other_value,
                ObjectiveDirection::Minimize => winner_value > other_value,
            }
        })
        .collect()
}

pub fn compare_pair(
    profile: &ObjectiveComparisonProfile,
    first: &CandidateObjectiveVector,
    second: &CandidateObjectiveVector,
) -> PairwiseTradeoffReport {
    let profile_diagnostics = validate_profile(profile);
    let mut first_diagnostics = validate_candidate(profile, first;
    let mut second_diagnostics = validate_candidate(profile, second;
    first_diagnostics.extend(profile_diagnostics.iter().copied());
    second_diagnostics.extend(profile_diagnostics.iter().copied());
    if first.candidate_id == second.candidate_id {
        first_diagnostics.push(ComparisonDiagnostic::DuplicateCandidateIdentity);
        second_diagnostics.push(ComparisonDiagnostic::DuplicateCandidateIdentity);
    }

    let first_q_eng = q_eng(first);
    let second_q_eng = q_eng(second);
    let scalar_preference = scalar_preference(first_q_eng, second_q_eng);
    let comparable = first_diagnostics.is_empty() && second_diagnostics.is_empty();
    let relation = if comparable {
        dominance(first, second)
    } else {
        DominanceRelation::NotComparable
    };

    let scalar_winner_worse_dimensions = match scalar_preference {
        ScalarPreference::First if comparable => worse_dimensions(first, second),
        ScalarPreference::Second if comparable => worse_dimensions(second, first),
        _ => Vec::new(),
    };

    let scalar_tradeoff = match (relation, scalar_preference) {
        (_, ScalarPreference::Unavailable) | (DominanceRelation::NotComparable, _) => {
            ScalarTradeoffDisposition::NotComparable
        }
        (DominanceRelation::FirstDominatesSecond, ScalarPreference::First)
        | (DominanceRelation::SecondDominatesFirst, ScalarPreference::Second) => {
            ScalarTradeoffDisposition::ScalarPreferenceAgreesWithDominance
        }
        (DominanceRelation::FirstDominatesSecond, ScalarPreference::Second)
        | (DominanceRelation::SecondDominatesFirst, ScalarPreference::First) => {
            // This should be impossible when Q_ENG and the vector use the same
            // availability and EBI values. Keep it visible as a contract defect.
            ScalarTradeoffDisposition::ScalarAndVectorDominanceInconsistent
        }
        (DominanceRelation::Equivalent, ScalarPreference::Tie) => {
            ScalarTradeoffDisposition::NoScalarPreference
        }
        (DominanceRelation::Equivalent, _) => {
            ScalarTradeoffDisposition::ScalarPreferenceBetweenEquivalentVectors
        }
        (
            DominanceRelation::NonDominatedTradeoff,
            ScalarPreference::First | ScalarPreference::Second,
        ) => ScalarTradeoffDisposition::ScalarPreferenceHidesOmittedObjectiveTradeoff,
        (_, ScalarPreference::Tie) => ScalarTradeoffDisposition::NoScalarPreference,
    };

    PairwiseTradeoffReport {
        contract_version: PARETO_TRADEOFF_VERSION,
        first_candidate_id: first.candidate_id.clone(),
        second_candidate_id: second.candidate_id.clone(),
        first_q_eng,
        second_q_eng,
        scalar_preference,
        dominance: relation,
        scalar_tradeoff,
        scalar_winner_worse_dimensions,
        first_diagnostics,
        second_diagnostics,
        claim_ceiling: "scalar/Pareto comparison over source-bound objective vectors only",
    }
}

pub fn extract_pareto_front(
    profile: &ObjectiveComparisonProfile,
    candidates: &[CandidateObjectiveVector],
) -> ParetoFrontReport {
    let profile_diagnostics = validate_profile(profile);
    let mut diagnostics = profile_diagnostics.clone();
    let mut seen = BTreeSet::new();
    let mut statuses = Vec::with_capacity(candidates.len());

    let mut duplicate_ids = BTreeSet::new();
    for candidate in candidates {
        if !seen.insert(candidate.candidate_id.as_str()) {
            diagnostics.push(ComparisonDiagnostic::DuplicateCandidateIdentity);
            duplicate_ids.insert(candidate.candidate_id.as_str());
        }
    }

    for candidate in candidates {
        let mut candidate_diagnostics = validate_candidate(profile, candidate;
        candidate_diagnostics.extend(profile_diagnostics.iter().copied());
        if duplicate_ids.contains(candidate.candidate_id.as_str()) {
            candidate_diagnostics.push(ComparisonDiagnostic::DuplicateCandidateIdentity);
        }
        let base_status = match candidate.eligibility {
            CandidateEligibility::BlockedByHardGate => ParetoFrontStatus::BlockedByHardGate,
            CandidateEligibility::UnknownHardGate => ParetoFrontStatus::UnknownHardGate,
            CandidateEligibility::Eligible if !candidate_diagnostics.is_empty() => {
                ParetoFrontStatus::NotComparable
            }
            CandidateEligibility::Eligible => ParetoFrontStatus::NonDominated,
        };
        statuses.push(ParetoCandidateStatus {
            candidate_id: candidate.candidate_id.clone(),
            status: base_status,
            dominated_by: Vec::new(),
            diagnostics: candidate_diagnostics,
        });
    }

    for first_index in 0..candidates.len() {
        if statuses[first_index].status != ParetoFrontStatus::NonDominated {
            continue;
        }
        for second_index in 0..candidates.len() {
            if first_index == second_index
                || statuses[second_index].status != ParetoFrontStatus::NonDominated
            {
                continue;
            }
            if dominance(&candidates[second_index], &candidates[first_index])
                == DominanceRelation::FirstDominatesSecond
            {
                statuses[first_index]
                    .dominated_by
                    .push(candidates[second_index].candidate_id.clone());
            }
        }
        if !statuses[first_index].dominated_by.is_empty() {
            statuses[first_index].status = ParetoFrontStatus::Dominated;
        }
    }

    statuses.sort_by(|first, second| first.candidate_id.cmp(&second.candidate_id));
    for status in &mut statuses {
        status.dominated_by.sort();
    }
    let frontier_candidate_ids = statuses
        .iter()
        .filter(|status| status.status == ParetoFrontStatus::NonDominated)
        .map(|status| status.candidate_id.clone())
        .collect();

    ParetoFrontReport {
        contract_version: PARETO_TRADEOFF_VERSION,
        profile_id: profile.profile_id.clone(),
        frontier_candidate_ids,
        candidates: statuses,
        diagnostics,
        claim_ceiling: "Pareto frontier over complete eligible objective vectors only",
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn profile() -> ObjectiveComparisonProfile {
        ObjectiveComparisonProfile {
            profile_id: "fusion-service-vector-v1".to_string(),
            profile_sha256: sha('a'),
            service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
            lifetime_horizon_ref: "plant-life-30-years-v1".to_string(),
            cost_basis_ref: "USD-2026-real-discount-6pct-whole-life-v1".to_string(),
            ebi_normalization_ref: "qeng-ebi-normalization-paper-v1".to_string(),
            uncertainty_basis_ref: "net-export-one-sided-95pct-method-v1".to_string(),
        }
    }

    fn scalar(value: f64, name: &str) -> Option<SourcedScalar> {
        Some(SourcedScalar {
            value,
            evidence_ref: format!("evidence-{name}"),
        })
    }

    fn candidate(id: &str, values: [f64; 7] -> CandidateObjectiveVector {
        let [energy, cost, availability, planned, unplanned, ebi, interval_width] = values;
        CandidateObjectiveVector {
            candidate_id: id.to_string(),
            candidate_revision: format!("{id}-r1"),
            profile_id: profile().profile_id,
            profile_sha256: profile().profile_sha256,
            objective_vector_receipt_ref: format!("vector-receipt-{id}"),
            objective_vector_receipt_sha256: sha('b'),
            eligibility: CandidateEligibility::Eligible,
            lifetime_net_electric_energy_mwh: scalar(energy, &format!("{id}-energy")),
            whole_life_cost: scalar(cost, &format!("{id}-cost")),
            operational_availability: scalar(availability, &format!("{id}-availability")),
            planned_downtime_hours: scalar(planned, &format!("{id}-planned")),
            unplanned_downtime_hours: scalar(unplanned, &format!("{id}-unplanned")),
            engineering_burden_index: scalar(ebi, &format!("{id}-ebi")),
            net_export_bound_width_mwe: scalar(interval_width, &format!("{id}-uncertainty")),
            ebi_normalization_ref: profile().ebi_normalization_ref,
            uncertainty_basis_ref: profile().uncertainty_basis_ref,
        }
    }

    #[test]
    fn q_eng_is_availability_over_positive_normalized_ebi() {
        let value = calculate_q_eng(0.8, 0.4).unwrap();
        assert_eq!(value.q_eng, 2.0);
    }

    #[test]
    fn better_availability_and_lower_ebi_cannot_reduce_q_eng() {
        let lower = calculate_q_eng(0.7, 0.8).unwrap();
        let higher = calculate_q_eng(0.9, 0.6).unwrap();
        assert!(higher.q_eng > lower.q_eng);
    }

    #[test]
    fn q_eng_is_monotone_in_each_input() {
        assert!(
            calculate_q_eng(0.8, 0.5).unwrap().q_eng > calculate_q_eng(0.7, 0.5).unwrap().q_eng
        );
        assert!(
            calculate_q_eng(0.8, 0.4).unwrap().q_eng > calculate_q_eng(0.8, 0.5).unwrap().q_eng
        );
    }

    #[test]
    fn q_eng_invalid_ebi_or_availability_is_refused() {
        assert!(calculate_q_eng(0.8, 0.0).is_none());
        assert!(calculate_q_eng(0.8, -0.1).is_none());
        assert!(calculate_q_eng(0.8, f64::NAN).is_none());
        assert!(calculate_q_eng(1.1, 0.5).is_none());
        assert!(calculate_q_eng(f64::INFINITY, 0.5).is_none());
    }

    #[test]
    fn q_eng_scales_inversely_with_ebi() {
        let baseline = calculate_q_eng(0.8, 0.4).unwrap().q_eng;
        let scaled_ebi = calculate_q_eng(0.8, 0.8).unwrap().q_eng;
        assert_eq!(scaled_ebi, baseline / 2.0);
    }

    #[test]
    fn q_eng_preference_can_hide_lifetime_energy_and_cost_tradeoff() {
        let profile = profile();
        let high_qeng = candidate(
            "high-qeng",
            8_000.0,
            12_000.0,
            0.95,
            500.0,
            300.0,
            0.8,
            25.0,
        ;
        let high_service = candidate(
            "high-service",
            12_000.0,
            8_000.0,
            0.90,
            650.0,
            350.0,
            0.9,
            15.0,
        ;
        let report = compare_pair(&profile, &high_qeng, &high_service);
        assert_eq!(report.scalar_preference, ScalarPreference::First);
        assert_eq!(report.dominance, DominanceRelation::NonDominatedTradeoff);
        assert!(
            report
                .scalar_winner_worse_dimensions
                .contains(&ObjectiveDimension::LifetimeNetElectricEnergyMwh)
        );
        assert!(
            report
                .scalar_winner_worse_dimensions
                .contains(&ObjectiveDimension::WholeLifeCost)
        );
        assert_eq!(
            report.scalar_tradeoff,
            ScalarTradeoffDisposition::ScalarPreferenceHidesOmittedObjectiveTradeoff
        );
    }

    #[test]
    fn q_eng_ratio_can_trade_availability_for_ebi() {
        let profile = profile();
        let lower_availability = candidate(
            "lower-availability",
            8_000.0,
            12_000.0,
            0.80,
            500.0,
            300.0,
            0.20,
            25.0,
        ;
        let higher_availability = candidate(
            "higher-availability",
            12_000.0,
            8_000.0,
            0.95,
            650.0,
            350.0,
            0.40,
            15.0,
        ;
        let report = compare_pair(&profile, &lower_availability, &higher_availability);
        assert_eq!(report.scalar_preference, ScalarPreference::First);
        assert_eq!(report.dominance, DominanceRelation::NonDominatedTradeoff);
        assert!(
            report
                .scalar_winner_worse_dimensions
                .contains(&ObjectiveDimension::OperationalAvailability)
        );
        assert!(
            report
                .scalar_winner_worse_dimensions
                .contains(&ObjectiveDimension::LifetimeNetElectricEnergyMwh)
        );
    }

    #[test]
    fn q_eng_cannot_reverse_true_vector_dominance_when_its_inputs_are_in_vector() {
        let profile = profile();
        let first = candidate("dominant", [12_000.0, 7_000.0, 0.95, 300.0, 200.0, 0.5, 10.0]);
        let second = candidate(
            "dominated",
            10_000.0,
            9_000.0,
            0.85,
            500.0,
            300.0,
            0.8,
            20.0,
        ;
        let report = compare_pair(&profile, &first, &second);
        assert_eq!(report.dominance, DominanceRelation::FirstDominatesSecond);
        assert_eq!(report.scalar_preference, ScalarPreference::First);
        assert_eq!(
            report.scalar_tradeoff,
            ScalarTradeoffDisposition::ScalarPreferenceAgreesWithDominance
        );
    }

    #[test]
    fn same_q_eng_ratio_does_not_mean_same_objective_vector() {
        let profile = profile();
        let first = candidate("ratio-a", [8_000.0, 12_000.0, 0.8, 500.0, 400.0, 0.4, 20.0]);
        let second = candidate("ratio-b", [11_000.0, 7_000.0, 0.9, 300.0, 200.0, 0.45, 10.0]);
        assert_eq!(q_eng(&first).unwrap().q_eng, q_eng(&second).unwrap().q_eng);
        let report = compare_pair(&profile, &first, &second);
        assert_eq!(report.scalar_preference, ScalarPreference::Tie);
        assert_eq!(report.dominance, DominanceRelation::NonDominatedTradeoff);
    }

    #[test]
    fn scalar_winner_is_not_universally_better_when_vectors_are_incomparable() {
        let profile = profile();
        let first = candidate("first", [9_000.0, 10_000.0, 0.9, 400.0, 300.0, 0.5, 20.0]);
        let second = candidate("second", [11_000.0, 8_000.0, 0.85, 450.0, 350.0, 0.5, 25.0]);
        let report = compare_pair(&profile, &first, &second);
        assert_eq!(report.dominance, DominanceRelation::NonDominatedTradeoff);
        assert_eq!(report.scalar_preference, ScalarPreference::First);
        assert_eq!(
            report.scalar_tradeoff,
            ScalarTradeoffDisposition::ScalarPreferenceHidesOmittedObjectiveTradeoff
        );
    }

    #[test]
    fn missing_lifetime_energy_is_not_imputed() {
        let profile = profile();
        let mut first = candidate("first", [9_000.0, 10_000.0, 0.9, 400.0, 300.0, 0.5, 20.0]);
        first.lifetime_net_electric_energy_mwh = None;
        let second = candidate("second", [11_000.0, 8_000.0, 0.85, 450.0, 350.0, 0.5, 25.0]);
        let report = compare_pair(&profile, &first, &second);
        assert_eq!(report.dominance, DominanceRelation::NotComparable);
        assert!(
            report
                .first_diagnostics
                .contains(&ComparisonDiagnostic::MissingObjective(
                    ObjectiveDimension::LifetimeNetElectricEnergyMwh
                ))
        );
        assert_eq!(report.scalar_preference, ScalarPreference::First);
    }

    #[test]
    fn mixed_cost_or_uncertainty_profile_is_not_comparable() {
        let profile = profile();
        let first = candidate("first", [9_000.0, 10_000.0, 0.9, 400.0, 300.0, 0.5, 20.0]);
        let mut second = candidate("second", [11_000.0, 8_000.0, 0.85, 450.0, 350.0, 0.5, 25.0]);
        second.profile_sha256 = sha('c');
        let report = compare_pair(&profile, &first, &second);
        assert_eq!(report.dominance, DominanceRelation::NotComparable);
        assert!(
            report
                .second_diagnostics
                .contains(&ComparisonDiagnostic::ProfileDigestMismatch)
        );
    }

    #[test]
    fn missing_ebi_provenance_is_not_comparable() {
        let profile = profile();
        let first = candidate("first", [9_000.0, 10_000.0, 0.9, 400.0, 300.0, 0.5, 20.0]);
        let mut second = candidate("second", [11_000.0, 8_000.0, 0.85, 450.0, 350.0, 0.5, 25.0]);
        second.ebi_normalization_ref.clear();
        let report = compare_pair(&profile, &first, &second);
        assert_eq!(report.dominance, DominanceRelation::NotComparable);
        assert!(
            report
                .second_diagnostics
                .contains(&ComparisonDiagnostic::EbiNormalizationMismatch)
        );
    }

    #[test]
    fn a_blocked_hard_gate_cannot_become_pareto_attractive() {
        let profile = profile();
        let mut first = candidate("blocked", [20_000.0, 1_000.0, 1.0, 0.0, 0.0, 0.1, 1.0]);
        first.eligibility = CandidateEligibility::BlockedByHardGate;
        let second = candidate("eligible", [10_000.0, 10_000.0, 0.8, 500.0, 300.0, 0.5, 20.0]);
        let report = extract_pareto_front(&profile, &[first, second]);
        assert_eq!(
            report
                .candidates
                .iter()
                .find(|candidate| candidate.candidate_id == "blocked")
                .unwrap()
                .status,
            ParetoFrontStatus::BlockedByHardGate
        );
        assert_eq!(report.frontier_candidate_ids, vec!["eligible".to_string()]);
    }

    #[test]
    fn frontier_is_order_invariant_and_sorted() {
        let profile = profile();
        let first = candidate("z", [9_000.0, 10_000.0, 0.9, 400.0, 300.0, 0.5, 20.0]);
        let second = candidate("a", [11_000.0, 8_000.0, 0.85, 450.0, 350.0, 0.5, 25.0]);
        let third = candidate("dominated", [7_000.0, 12_000.0, 0.7, 600.0, 450.0, 0.8, 35.0]);
        let first_report =
            extract_pareto_front(&profile, &[first.clone(), second.clone(), third.clone()]);
        let second_report = extract_pareto_front(&profile, &[third, second, first]);
        assert_eq!(first_report, second_report);
        assert_eq!(
            first_report.frontier_candidate_ids,
            vec!["a".to_string(), "z".to_string()]
        );
    }

    #[test]
    fn adding_a_dominated_candidate_does_not_change_existing_frontier() {
        let profile = profile();
        let first = candidate("a", [9_000.0, 10_000.0, 0.9, 400.0, 300.0, 0.5, 20.0]);
        let second = candidate("b", [11_000.0, 8_000.0, 0.85, 450.0, 350.0, 0.5, 25.0]);
        let base = extract_pareto_front(&profile, &[first.clone(), second.clone()]);
        let dominated = candidate("c", [5_000.0, 20_000.0, 0.5, 800.0, 600.0, 1.0, 50.0]);
        let extended = extract_pareto_front(&profile, &[first, second, dominated]);
        assert_eq!(base.frontier_candidate_ids, extended.frontier_candidate_ids);
    }

    #[test]
    fn equivalent_vectors_are_explicitly_tied() {
        let profile = profile();
        let first = candidate("first", [9_000.0, 10_000.0, 0.9, 400.0, 300.0, 0.5, 20.0]);
        let mut second = candidate("second", [9_000.0, 10_000.0, 0.9, 400.0, 300.0, 0.5, 20.0]);
        second.objective_vector_receipt_ref = "other-receipt".to_string();
        let report = compare_pair(&profile, &first, &second);
        assert_eq!(report.dominance, DominanceRelation::Equivalent);
        assert_eq!(report.scalar_preference, ScalarPreference::Tie);
    }

    #[test]
    fn non_finite_and_invalid_objective_values_are_rejected() {
        let profile = profile();
        let mut first = candidate("first", [9_000.0, 10_000.0, 0.9, 400.0, 300.0, 0.5, 20.0]);
        first.whole_life_cost.as_mut().unwrap().value = f64::INFINITY;
        let second = candidate("second", [11_000.0, 8_000.0, 0.85, 450.0, 350.0, 0.5, 25.0]);
        let report = compare_pair(&profile, &first, &second);
        assert_eq!(report.dominance, DominanceRelation::NotComparable);
        assert!(
            report
                .first_diagnostics
                .contains(&ComparisonDiagnostic::NonFiniteObjective(
                    ObjectiveDimension::WholeLifeCost
                ))
        );
    }

    #[test]
    fn pairwise_comparison_rejects_reused_candidate_identity() {
        let profile = profile();
        let first = candidate("same", [9_000.0, 10_000.0, 0.9, 400.0, 300.0, 0.5, 20.0]);
        let mut second = candidate("same", [11_000.0, 8_000.0, 0.85, 450.0, 350.0, 0.5, 25.0]);
        second.candidate_revision = "same-r2".to_string();
        let report = compare_pair(&profile, &first, &second);
        assert_eq!(report.dominance, DominanceRelation::NotComparable);
        assert!(
            report
                .first_diagnostics
                .contains(&ComparisonDiagnostic::DuplicateCandidateIdentity)
        );
        assert!(
            report
                .second_diagnostics
                .contains(&ComparisonDiagnostic::DuplicateCandidateIdentity)
        );
    }

    #[test]
    fn duplicate_candidate_ids_are_reported_in_frontier() {
        let profile = profile();
        let first = candidate("same", [9_000.0, 10_000.0, 0.9, 400.0, 300.0, 0.5, 20.0]);
        let second = candidate("same", [11_000.0, 8_000.0, 0.85, 450.0, 350.0, 0.5, 25.0]);
        let report = extract_pareto_front(&profile, &[first, second]);
        assert!(!report.diagnostics.is_empty());
    }
}
