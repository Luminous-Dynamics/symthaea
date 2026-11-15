//! Service-boundary rank-reversal benchmark.
//!
//! Compares candidate rankings across plasma gain, gross generation, net
//! export and lifecycle net energy. It composes the power-balance and
//! lifecycle ledgers; it does not create a weighted utility or new optimizer.

use std::collections::BTreeSet;

use crate::EvidenceClass;
use super::lifecycle_service_ledger::{
    LifecycleServiceDisposition, LifecycleServiceReport,
};
use super::plant_service_gate::{
    PowerBalanceDisposition, PowerBalanceProfile, PowerBalanceReport, PowerBalanceRun,
    evaluate_plant_power_balance,
};

pub const SERVICE_BOUNDARY_RANK_VERSION: &str = "fusion-service-boundary-rank/0.1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ServiceRankingMetric {
    PlasmaGainQ,
    GrossElectricMwe,
    NetElectricMwe,
    LifetimeNetEnergyMwh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NetServiceDisposition {
    NetImport,
    NoNetExport,
    NetExportBelowReserve,
    NetExportQualified,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateServiceInput {
    pub candidate_id: String,
    pub candidate_revision: String,
    pub geometry_sha256: String,
    pub power_profile: PowerBalanceProfile,
    pub power_run: PowerBalanceRun,
    pub power_report: PowerBalanceReport,
    pub power_report_ref: String,
    pub plasma_gain_q: f64,
    pub plasma_gain_run_id: String,
    pub plasma_gain_output_sha256: String,
    pub plasma_gain_evidence_ref: String,
    pub lifecycle_report_ref: String,
    pub lifecycle_report_sha256: String,
    pub lifecycle_report: LifecycleServiceReport,
    pub whole_life_cost: f64,
    pub cost_currency_ref: String,
    pub cost_scope_ref: String,
    pub cost_evidence_ref: String,
    pub evidence_class: EvidenceClass,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServiceBoundaryRankingProfile {
    pub service_boundary_ref: String,
    pub unit_ref: String,
    pub power_sign_convention_ref: String,
    pub lifecycle_sign_convention_ref: String,
    pub lifecycle_horizon_ref: String,
    pub lifecycle_horizon_seconds: u64,
    pub cost_currency_ref: String,
    pub cost_scope_ref: String,
    pub required_net_export_reserve_mwe: f64,
    pub numerical_tolerance_mwh: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceBoundaryRankError {
    EmptyCandidateIdentity,
    DuplicateCandidateId,
    InvalidGeometryDigest,
    InvalidPowerRunDigest,
    InvalidPowerReportReference,
    CandidatePowerRevisionMismatch,
    CandidatePowerReportMismatch,
    PowerBalanceNotAdmissible,
    ServiceBoundaryMismatch,
    UnitMismatch,
    SignConventionMismatch,
    InvalidPlasmaGain,
    EmptyPlasmaGainEvidence,
    PlasmaGainRunMismatch,
    PlasmaGainOutputMismatch,
    InvalidPlasmaGainDigest,
    LifecycleNotValid,
    InvalidLifecycleDigest,
    EmptyLifecycleReference,
    LifecycleBoundaryMismatch,
    LifecycleUnitMismatch,
    LifecycleSignConventionMismatch,
    LifecycleHorizonMismatch,
    LifecycleHorizonLengthMismatch,
    LifecycleMetricsMissing,
    LifecycleMetricsInvalid,
    LifecycleEnergyIdentityMismatch,
    LifecyclePowerOutputUnbound,
    InvalidWholeLifeCost,
    CostCurrencyMismatch,
    CostScopeMismatch,
    EmptyCostEvidence,
    EmptyProfileIdentity,
    InvalidReserve,
    ReserveMismatch,
    InvalidTolerance,
    EmptyCandidateSet,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateServiceFinding {
    pub candidate_id: String,
    pub candidate_revision: String,
    pub solver_run_id: String,
    pub solver_id: String,
    pub output_sha256: String,
    pub plasma_gain_q: f64,
    pub gross_electric_mwe: f64,
    pub variable_recirculated_mwe: f64,
    pub fixed_recirculated_mwe: f64,
    pub total_recirculated_mwe: f64,
    pub net_electric_mwe: f64,
    pub availability_fraction: f64,
    pub lifetime_net_energy_mwh: f64,
    pub export_energy_mwh: f64,
    pub import_energy_mwh: f64,
    pub whole_life_cost: f64,
    pub net_service_disposition: NetServiceDisposition,
    pub power_report_ref: String,
    pub lifecycle_report_ref: String,
    pub lifecycle_report_sha256: String,
    pub plasma_gain_evidence_ref: String,
    pub cost_evidence_ref: String,
    pub evidence_class: EvidenceClass,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryRanking {
    pub metric: ServiceRankingMetric,
    /// Highest value first. Candidate ID is the stable tie-breaker.
    pub ordered_candidate_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairwiseMetricWinner {
    pub metric: ServiceRankingMetric,
    pub winner_candidate_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairwiseRankReversal {
    pub first_candidate_id: String,
    pub second_candidate_id: String,
    pub metric_winners: Vec<PairwiseMetricWinner>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServiceParetoPoint {
    pub candidate_id: String,
    pub whole_life_cost: f64,
    pub lifetime_net_energy_mwh: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServiceBoundaryRankingReport {
    pub contract_version: &'static str,
    pub profile: ServiceBoundaryRankingProfile,
    pub findings: Vec<CandidateServiceFinding>,
    pub rankings: Vec<BoundaryRanking>,
    pub pairwise_rank_reversals: Vec<PairwiseRankReversal>,
    /// Only candidates that clear the declared net-export reserve enter this
    /// cost/lifetime-energy front.
    pub qualified_cost_energy_pareto_front: Vec<ServiceParetoPoint>,
    pub claim_ceiling: &'static str,
}

/// Validate and rank service-boundary metrics without collapsing them into a
/// scalar. The power and lifecycle reports are recomputed/checked by their
/// respective owners before they are admitted to this benchmark.
pub fn compare_service_boundary_rankings(
    profile: &ServiceBoundaryRankingProfile,
    candidates: &[CandidateServiceInput],
) -> Result<ServiceBoundaryRankingReport, Vec<ServiceBoundaryRankError>> {
    let mut errors = validate_profile(profile);
    if candidates.is_empty() {
        errors.push(ServiceBoundaryRankError::EmptyCandidateSet);
    }

    let mut seen = BTreeSet::new();
    for candidate in candidates {
        if !seen.insert(candidate.candidate_id.as_str()) {
            errors.push(ServiceBoundaryRankError::DuplicateCandidateId);
        }
        errors.extend(validate_candidate(profile, candidate));
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    let mut findings: Vec<CandidateServiceFinding> = candidates
        .iter()
        .map(|candidate| build_finding(profile, candidate))
        .collect();
    findings.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let rankings = vec![
        ranking(&findings, ServiceRankingMetric::PlasmaGainQ, |row| row.plasma_gain_q),
        ranking(&findings, ServiceRankingMetric::GrossElectricMwe, |row| row.gross_electric_mwe),
        ranking(&findings, ServiceRankingMetric::NetElectricMwe, |row| row.net_electric_mwe),
        ranking(&findings, ServiceRankingMetric::LifetimeNetEnergyMwh, |row| {
            row.lifetime_net_energy_mwh
        }),
    ];

    let pairwise_rank_reversals = rank_reversals(&findings);
    let qualified_cost_energy_pareto_front = cost_energy_front(&findings);

    Ok(ServiceBoundaryRankingReport {
        contract_version: SERVICE_BOUNDARY_RANK_VERSION,
        profile: profile.clone(),
        findings,
        rankings,
        pairwise_rank_reversals,
        qualified_cost_energy_pareto_front,
        claim_ceiling: "service-boundary ranking and rank-reversal benchmark only",
    })
}

fn validate_profile(
    profile: &ServiceBoundaryRankingProfile,
) -> Vec<ServiceBoundaryRankError> {
    let mut errors = Vec::new();
    if profile.service_boundary_ref.trim().is_empty()
        || profile.lifecycle_horizon_ref.trim().is_empty()
        || profile.cost_currency_ref.trim().is_empty()
        || profile.cost_scope_ref.trim().is_empty()
    {
        errors.push(ServiceBoundaryRankError::EmptyProfileIdentity);
    }
    if profile.unit_ref != "MWe" {
        errors.push(ServiceBoundaryRankError::UnitMismatch);
    }
    if profile.power_sign_convention_ref != "gross-and-load-positive-net-export-positive-v1"
        || profile.lifecycle_sign_convention_ref
            != "net-export-positive-import-negative-v1"
    {
        errors.push(ServiceBoundaryRankError::SignConventionMismatch);
    }
    if profile.lifecycle_horizon_seconds == 0 {
        errors.push(ServiceBoundaryRankError::LifecycleHorizonLengthMismatch);
    }
    if !profile.required_net_export_reserve_mwe.is_finite()
        || profile.required_net_export_reserve_mwe < 0.0
    {
        errors.push(ServiceBoundaryRankError::InvalidReserve);
    }
    if !profile.numerical_tolerance_mwh.is_finite() || profile.numerical_tolerance_mwh < 0.0 {
        errors.push(ServiceBoundaryRankError::InvalidTolerance);
    }
    errors
}

fn validate_candidate(
    profile: &ServiceBoundaryRankingProfile,
    candidate: &CandidateServiceInput,
) -> Vec<ServiceBoundaryRankError> {
    let mut errors = Vec::new();

    if candidate.candidate_id.trim().is_empty()
        || candidate.candidate_revision.trim().is_empty()
        || candidate.power_run.run_id.trim().is_empty()
    {
        errors.push(ServiceBoundaryRankError::EmptyCandidateIdentity);
    }
    if !is_sha256(&candidate.geometry_sha256) {
        errors.push(ServiceBoundaryRankError::InvalidGeometryDigest);
    }
    if !is_sha256(&candidate.power_run.output_sha256) {
        errors.push(ServiceBoundaryRankError::InvalidPowerRunDigest);
    }
    if candidate.power_report_ref.trim().is_empty() {
        errors.push(ServiceBoundaryRankError::InvalidPowerReportReference);
    }
    if candidate.power_run.candidate_revision != candidate.candidate_revision {
        errors.push(ServiceBoundaryRankError::CandidatePowerRevisionMismatch);
    }

    if candidate.power_run.service_boundary_ref != profile.service_boundary_ref
        || candidate.power_profile.service_boundary_ref != profile.service_boundary_ref
    {
        errors.push(ServiceBoundaryRankError::ServiceBoundaryMismatch);
    }
    if candidate.power_run.unit_ref != profile.unit_ref
        || candidate.power_profile.unit_ref != profile.unit_ref
    {
        errors.push(ServiceBoundaryRankError::UnitMismatch);
    }
    if candidate.power_run.sign_convention_ref != profile.power_sign_convention_ref
        || candidate.power_profile.sign_convention_ref != profile.power_sign_convention_ref
    {
        errors.push(ServiceBoundaryRankError::SignConventionMismatch);
    }
    if candidate.power_profile.required_net_export_reserve_mw
        != profile.required_net_export_reserve_mwe
    {
        errors.push(ServiceBoundaryRankError::ReserveMismatch);
    }

    let recomputed = super::plant_service_gate::evaluate_plant_power_balance(
        &candidate.power_profile,
        &candidate.power_run,
    );
    if recomputed != candidate.power_report {
        errors.push(ServiceBoundaryRankError::CandidatePowerReportMismatch);
    }
    if matches!(
        candidate.power_report.disposition,
        PowerBalanceDisposition::NotEvaluable | PowerBalanceDisposition::BalanceInconsistent
    ) {
        errors.push(ServiceBoundaryRankError::PowerBalanceNotAdmissible);
    }

    if !candidate.plasma_gain_q.is_finite() || candidate.plasma_gain_q < 0.0 {
        errors.push(ServiceBoundaryRankError::InvalidPlasmaGain);
    }
    if candidate.plasma_gain_evidence_ref.trim().is_empty() {
        errors.push(ServiceBoundaryRankError::EmptyPlasmaGainEvidence);
    }
    if candidate.plasma_gain_run_id != candidate.power_run.run_id {
        errors.push(ServiceBoundaryRankError::PlasmaGainRunMismatch);
    }
    if candidate.plasma_gain_output_sha256 != candidate.power_run.output_sha256 {
        errors.push(ServiceBoundaryRankError::PlasmaGainOutputMismatch);
    }
    if !is_sha256(&candidate.plasma_gain_output_sha256) {
        errors.push(ServiceBoundaryRankError::InvalidPlasmaGainDigest);
    }

    let lifecycle = &candidate.lifecycle_report;
    if lifecycle.disposition != LifecycleServiceDisposition::ValidDeterministicLedger {
        errors.push(ServiceBoundaryRankError::LifecycleNotValid);
    }
    if candidate.lifecycle_report_ref.trim().is_empty()
        || !is_sha256(&candidate.lifecycle_report_sha256)
    {
        errors.push(ServiceBoundaryRankError::InvalidLifecycleDigest);
    }
    if lifecycle.service_boundary_ref != profile.service_boundary_ref {
        errors.push(ServiceBoundaryRankError::LifecycleBoundaryMismatch);
    }
    if lifecycle.unit_ref != profile.unit_ref {
        errors.push(ServiceBoundaryRankError::LifecycleUnitMismatch);
    }
    if lifecycle.sign_convention_ref != profile.lifecycle_sign_convention_ref {
        errors.push(ServiceBoundaryRankError::LifecycleSignConventionMismatch);
    }
    if lifecycle.horizon_ref != profile.lifecycle_horizon_ref {
        errors.push(ServiceBoundaryRankError::LifecycleHorizonMismatch);
    }
    if lifecycle.horizon_seconds != Some(profile.lifecycle_horizon_seconds)
        || lifecycle
            .horizon_end_second
            .checked_sub(lifecycle.horizon_start_second)
            != Some(profile.lifecycle_horizon_seconds)
    {
        errors.push(ServiceBoundaryRankError::LifecycleHorizonLengthMismatch);
    }
    let metrics_present = lifecycle.availability_fraction.is_some()
        && lifecycle.signed_net_energy_mwh.is_some()
        && lifecycle.export_energy_mwh.is_some()
        && lifecycle.import_energy_mwh.is_some();
    if !metrics_present {
        errors.push(ServiceBoundaryRankError::LifecycleMetricsMissing);
    } else {
        let availability = lifecycle.availability_fraction.unwrap();
        let signed = lifecycle.signed_net_energy_mwh.unwrap();
        let export = lifecycle.export_energy_mwh.unwrap();
        let import = lifecycle.import_energy_mwh.unwrap();
        if !availability.is_finite()
            || !(0.0..=1.0).contains(&availability)
            || !signed.is_finite()
            || !export.is_finite()
            || export < 0.0
            || !import.is_finite()
            || import < 0.0
        {
            errors.push(ServiceBoundaryRankError::LifecycleMetricsInvalid);
        } else if (signed - (export - import)).abs() > profile.numerical_tolerance_mwh
        {
            errors.push(ServiceBoundaryRankError::LifecycleEnergyIdentityMismatch);
        }
    }

    let power_receipt_bound = lifecycle.segment_receipts.iter().any(|segment| {
        segment.power_balance_report_ref == candidate.power_report_ref
            && segment.output_sha256 == candidate.power_run.output_sha256
    });
    if !power_receipt_bound {
        errors.push(ServiceBoundaryRankError::LifecyclePowerOutputUnbound);
    }

    if !candidate.whole_life_cost.is_finite() || candidate.whole_life_cost <= 0.0 {
        errors.push(ServiceBoundaryRankError::InvalidWholeLifeCost);
    }
    if candidate.cost_currency_ref != profile.cost_currency_ref {
        errors.push(ServiceBoundaryRankError::CostCurrencyMismatch);
    }
    if candidate.cost_scope_ref != profile.cost_scope_ref {
        errors.push(ServiceBoundaryRankError::CostScopeMismatch);
    }
    if candidate.cost_evidence_ref.trim().is_empty() {
        errors.push(ServiceBoundaryRankError::EmptyCostEvidence);
    }

    errors
}

fn build_finding(
    profile: &ServiceBoundaryRankingProfile,
    candidate: &CandidateServiceInput,
) -> CandidateServiceFinding {
    let power = &candidate.power_report;
    let net = power.reported_net_electric_mw.expect("validated power report has net output");
    let gross = power.gross_electric_mw.expect("validated power report has gross output");
    let variable = power.variable_recirculated_mw.expect("validated power report has variable recirculation");
    let fixed = power.fixed_recirculated_mw.expect("validated power report has fixed recirculation");
    let total = power.total_recirculated_mw.expect("validated power report has total recirculation");
    let lifecycle = &candidate.lifecycle_report;
    let availability = lifecycle.availability_fraction.expect("validated lifecycle has availability");
    let lifetime_net = lifecycle.signed_net_energy_mwh.expect("validated lifecycle has energy");
    let export = lifecycle.export_energy_mwh.expect("validated lifecycle has export");
    let import = lifecycle.import_energy_mwh.expect("validated lifecycle has import");

    let net_service_disposition = match power.disposition {
        PowerBalanceDisposition::NetImport => NetServiceDisposition::NetImport,
        PowerBalanceDisposition::NoNetExport => NetServiceDisposition::NoNetExport,
        PowerBalanceDisposition::NetExportBelowReserve => {
            NetServiceDisposition::NetExportBelowReserve
        }
        PowerBalanceDisposition::NetExportQualifiedForDeclaredBoundary => {
            NetServiceDisposition::NetExportQualified
        }
        PowerBalanceDisposition::NotEvaluable | PowerBalanceDisposition::BalanceInconsistent => {
            unreachable!("validated power report cannot have a non-admissible disposition")
        }
    };

    CandidateServiceFinding {
        candidate_id: candidate.candidate_id.clone(),
        candidate_revision: candidate.candidate_revision.clone(),
        solver_run_id: candidate.power_run.run_id.clone(),
        solver_id: candidate.power_run.solver_id.clone(),
        output_sha256: candidate.power_run.output_sha256.clone(),
        plasma_gain_q: candidate.plasma_gain_q,
        gross_electric_mwe: gross,
        variable_recirculated_mwe: variable,
        fixed_recirculated_mwe: fixed,
        total_recirculated_mwe: total,
        net_electric_mwe: net,
        availability_fraction: availability,
        lifetime_net_energy_mwh: lifetime_net,
        export_energy_mwh: export,
        import_energy_mwh: import,
        whole_life_cost: candidate.whole_life_cost,
        net_service_disposition,
        power_report_ref: candidate.power_report_ref.clone(),
        lifecycle_report_ref: candidate.lifecycle_report_ref.clone(),
        lifecycle_report_sha256: candidate.lifecycle_report_sha256.clone(),
        plasma_gain_evidence_ref: candidate.plasma_gain_evidence_ref.clone(),
        cost_evidence_ref: candidate.cost_evidence_ref.clone(),
        evidence_class: candidate.evidence_class,
    }
}

fn ranking(
    findings: &[CandidateServiceFinding],
    metric: ServiceRankingMetric,
    value: impl Fn(&CandidateServiceFinding) -> f64,
) -> BoundaryRanking {
    let mut ordered = findings.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        value(right)
            .total_cmp(&value(left))
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    BoundaryRanking {
        metric,
        ordered_candidate_ids: ordered.into_iter().map(|row| row.candidate_id.clone()).collect(),
    }
}

fn rank_reversals(findings: &[CandidateServiceFinding]) -> Vec<PairwiseRankReversal> {
    let mut reversals = Vec::new();
    for first in 0..findings.len() {
        for second in (first + 1)..findings.len() {
            let a = &findings[first];
            let b = &findings[second];
            let metrics = [
                (
                    ServiceRankingMetric::PlasmaGainQ,
                    a.plasma_gain_q,
                    b.plasma_gain_q,
                ),
                (
                    ServiceRankingMetric::GrossElectricMwe,
                    a.gross_electric_mwe,
                    b.gross_electric_mwe,
                ),
                (
                    ServiceRankingMetric::NetElectricMwe,
                    a.net_electric_mwe,
                    b.net_electric_mwe,
                ),
                (
                    ServiceRankingMetric::LifetimeNetEnergyMwh,
                    a.lifetime_net_energy_mwh,
                    b.lifetime_net_energy_mwh,
                ),
            ];
            let winners: Vec<PairwiseMetricWinner> = metrics
                .into_iter()
                .filter_map(|(metric, first_value, second_value)| {
                    match first_value.total_cmp(&second_value) {
                        std::cmp::Ordering::Greater => Some(PairwiseMetricWinner {
                            metric,
                            winner_candidate_id: a.candidate_id.clone(),
                        }),
                        std::cmp::Ordering::Less => Some(PairwiseMetricWinner {
                            metric,
                            winner_candidate_id: b.candidate_id.clone(),
                        }),
                        std::cmp::Ordering::Equal => None,
                    }
                })
                .collect();
            let distinct_winners: BTreeSet<&str> = winners
                .iter()
                .map(|winner| winner.winner_candidate_id.as_str())
                .collect();
            if distinct_winners.len() > 1 {
                reversals.push(PairwiseRankReversal {
                    first_candidate_id: a.candidate_id.clone(),
                    second_candidate_id: b.candidate_id.clone(),
                    metric_winners: winners,
                });
            }
        }
    }
    reversals
}

fn cost_energy_front(findings: &[CandidateServiceFinding]) -> Vec<ServiceParetoPoint> {
    let qualified: Vec<&CandidateServiceFinding> = findings
        .iter()
        .filter(|row| row.net_service_disposition == NetServiceDisposition::NetExportQualified)
        .collect();
    let mut front = qualified
        .iter()
        .filter(|candidate| {
            !qualified.iter().any(|other| {
                other.candidate_id != candidate.candidate_id
                    && other.whole_life_cost <= candidate.whole_life_cost
                    && other.lifetime_net_energy_mwh >= candidate.lifetime_net_energy_mwh
                    && (other.whole_life_cost < candidate.whole_life_cost
                        || other.lifetime_net_energy_mwh > candidate.lifetime_net_energy_mwh)
            })
        })
        .map(|candidate| ServiceParetoPoint {
            candidate_id: candidate.candidate_id.clone(),
            whole_life_cost: candidate.whole_life_cost,
            lifetime_net_energy_mwh: candidate.lifetime_net_energy_mwh,
        })
        .collect::<Vec<_>>();
    front.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    front
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle_service_ledger::{
        AvailabilityEvent, LifecycleEvidenceClass, LifecycleServicePlan,
        OutageCategory, PowerOperatingState, PowerStateSegment, ServiceImpact,
        evaluate_lifecycle_service_budget,
    };
    use crate::plant_service_gate::{
        PowerObservable, PowerTermEvidence, evaluate_plant_power_balance,
    };

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn profile() -> ServiceBoundaryRankingProfile {
        ServiceBoundaryRankingProfile {
            service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
            unit_ref: "MWe".to_string(),
            power_sign_convention_ref: "gross-and-load-positive-net-export-positive-v1".to_string(),
            lifecycle_sign_convention_ref: "net-export-positive-import-negative-v1".to_string(),
            lifecycle_horizon_ref: "synthetic-10000-second-horizon-v1".to_string(),
            lifecycle_horizon_seconds: 10_000,
            cost_currency_ref: "MUSD_2026".to_string(),
            cost_scope_ref: "discounted-whole-life-cost-v1".to_string(),
            required_net_export_reserve_mwe: 0.0,
            numerical_tolerance_mwh: 0.001,
        }
    }

    fn power_profile() -> PowerBalanceProfile {
        PowerBalanceProfile {
            schema_ref: "synthetic-systems-code-net-power-v1".to_string(),
            schema_sha256: sha('a'),
            service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
            unit_ref: "MWe".to_string(),
            averaging_window_ref: "steady-state-60s-v1".to_string(),
            sign_convention_ref: "gross-and-load-positive-net-export-positive-v1".to_string(),
            absolute_balance_tolerance_mw: 0.001,
            required_net_export_reserve_mw: 0.0,
            reserve_threshold_inclusive: true,
        }
    }

    fn candidate(
        candidate_id: &str,
        q: f64,
        gross: f64,
        variable: f64,
        fixed: f64,
        availability: f64,
        outage_power: f64,
        whole_life_cost: f64,
        evidence_character: char,
    ) -> CandidateServiceInput {
        let profile = power_profile();
        let output_sha = sha(evidence_character);
        let total = variable + fixed;
        let net = gross - total;
        let run = PowerBalanceRun {
            run_id: format!("solver-run-{candidate_id}"),
            solver_id: "synthetic-systems-code".to_string(),
            solver_revision: "synthetic-revision-r1".to_string(),
            candidate_revision: format!("design-{candidate_id}-r1"),
            output_sha256: output_sha.clone(),
            schema_ref: profile.schema_ref.clone(),
            schema_sha256: profile.schema_sha256.clone(),
            service_boundary_ref: profile.service_boundary_ref.clone(),
            unit_ref: profile.unit_ref.clone(),
            averaging_window_ref: profile.averaging_window_ref.clone(),
            sign_convention_ref: profile.sign_convention_ref.clone(),
            receipt_ref: format!("power-run-receipt-{candidate_id}"),
            terms: vec![
                PowerTermEvidence {
                    observable: PowerObservable::GrossElectric,
                    value_mw: gross,
                    source_locator: "synthetic.gross".to_string(),
                    evidence_ref: format!("gross-{candidate_id}"),
                },
                PowerTermEvidence {
                    observable: PowerObservable::VariableRecirculated,
                    value_mw: variable,
                    source_locator: "synthetic.variable".to_string(),
                    evidence_ref: format!("variable-{candidate_id}"),
                },
                PowerTermEvidence {
                    observable: PowerObservable::FixedRecirculated,
                    value_mw: fixed,
                    source_locator: "synthetic.fixed".to_string(),
                    evidence_ref: format!("fixed-{candidate_id}"),
                },
                PowerTermEvidence {
                    observable: PowerObservable::TotalRecirculated,
                    value_mw: total,
                    source_locator: "synthetic.total".to_string(),
                    evidence_ref: format!("total-{candidate_id}"),
                },
                PowerTermEvidence {
                    observable: PowerObservable::NetElectric,
                    value_mw: net,
                    source_locator: "synthetic.net".to_string(),
                    evidence_ref: format!("net-{candidate_id}"),
                },
            ],
        };
        let power_report = evaluate_plant_power_balance(&profile, &run);
        let reserve_seconds = ((1.0 - availability) * 10_000.0).round() as u64;
        let outage_start = 10_000 - reserve_seconds;
        let lifecycle_plan = LifecycleServicePlan {
            campaign_id: format!("lifecycle-{candidate_id}"),
            horizon_ref: "synthetic-10000-second-horizon-v1".to_string(),
            horizon_start_second: 0,
            horizon_end_second: 10_000,
            service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
            unit_ref: "MWe".to_string(),
            sign_convention_ref: "net-export-positive-import-negative-v1".to_string(),
            events: vec![AvailabilityEvent {
                event_id: format!("planned-outage-{candidate_id}"),
                category: OutageCategory::Planned,
                impact: ServiceImpact::FullOutage,
                subsystem_id: "synthetic-plant".to_string(),
                failure_mode_id: "synthetic-maintenance".to_string(),
                start_second: outage_start,
                end_second: 10_000,
                evidence_ref: format!("outage-evidence-{candidate_id}"),
                evidence_class: LifecycleEvidenceClass::Synthetic,
            }],
            power_segments: vec![
                PowerStateSegment {
                    segment_id: format!("service-{candidate_id}"),
                    start_second: 0,
                    end_second: outage_start,
                    net_power_mwe: net,
                    state: PowerOperatingState::PowerGeneration,
                    active_event_ids: vec![],
                    power_balance_report_ref: format!("power-report-{candidate_id}"),
                    output_sha256: output_sha.clone(),
                    evidence_ref: format!("service-evidence-{candidate_id}"),
                    evidence_class: LifecycleEvidenceClass::Synthetic,
                },
                PowerStateSegment {
                    segment_id: format!("outage-{candidate_id}"),
                    start_second: outage_start,
                    end_second: 10_000,
                    net_power_mwe: outage_power,
                    state: PowerOperatingState::PlannedMaintenance,
                    active_event_ids: vec![format!("planned-outage-{candidate_id}")],
                    power_balance_report_ref: format!("power-report-{candidate_id}"),
                    output_sha256: output_sha,
                    evidence_ref: format!("outage-power-evidence-{candidate_id}"),
                    evidence_class: LifecycleEvidenceClass::Synthetic,
                },
            ],
        };
        let lifecycle_report = evaluate_lifecycle_service_budget(&lifecycle_plan);
        CandidateServiceInput {
            candidate_id: candidate_id.to_string(),
            candidate_revision: format!("design-{candidate_id}-r1"),
            geometry_sha256: sha(evidence_character),
            power_profile: profile,
            power_run: run,
            power_report,
            power_report_ref: format!("power-report-{candidate_id}"),
            plasma_gain_q: q,
            plasma_gain_run_id: run.run_id.clone(),
            plasma_gain_output_sha256: run.output_sha256.clone(),
            plasma_gain_evidence_ref: format!("q-evidence-{candidate_id}"),
            lifecycle_report_ref: format!("lifecycle-report-{candidate_id}"),
            lifecycle_report_sha256: sha(evidence_character.to_ascii_uppercase()),
            lifecycle_report,
            whole_life_cost,
            cost_currency_ref: "MUSD_2026".to_string(),
            cost_scope_ref: "discounted-whole-life-cost-v1".to_string(),
            cost_evidence_ref: format!("cost-evidence-{candidate_id}"),
            evidence_class: EvidenceClass::SyntheticFixture,
        }
    }

    fn three_candidate_fixture() -> Vec<CandidateServiceInput> {
        vec![
            candidate("A", 9.9, 210.0, 175.0, 55.0, 0.90, -5.0, 4_000.0, 'a'),
            candidate("B", 6.5, 180.0, 100.0, 30.0, 0.85, -7.0, 4_200.0, 'b'),
            candidate("C", 7.2, 195.0, 120.0, 25.0, 0.95, -8.0, 4_900.0, 'c'),
        ]
    }

    #[test]
    fn plasma_gain_and_gross_power_winner_can_lose_on_net_service() {
        let report = compare_service_boundary_rankings(&profile(), &three_candidate_fixture()).unwrap();
        assert_eq!(
            report.rankings[0].ordered_candidate_ids,
            vec!["A", "C", "B"]
        );
        assert_eq!(
            report.rankings[1].ordered_candidate_ids,
            vec!["A", "C", "B"]
        );
        assert_eq!(
            report.findings.iter().find(|row| row.candidate_id == "A").unwrap().net_service_disposition,
            NetServiceDisposition::NetImport
        );
        assert_eq!(
            report.findings.iter().find(|row| row.candidate_id == "B").unwrap().net_service_disposition,
            NetServiceDisposition::NetExportQualified
        );
        assert_eq!(
            report.findings.iter().find(|row| row.candidate_id == "C").unwrap().net_service_disposition,
            NetServiceDisposition::NetExportQualified
        );
    }

    #[test]
    fn rank_reversal_and_cost_energy_pareto_tradeoff_are_visible() {
        let report = compare_service_boundary_rankings(&profile(), &three_candidate_fixture()).unwrap();
        assert!(report.pairwise_rank_reversals.iter().any(|reversal| {
            reversal.first_candidate_id == "A"
                && reversal.second_candidate_id == "B"
                && reversal.metric_winners.iter().any(|winner| {
                    winner.metric == ServiceRankingMetric::PlasmaGainQ
                        && winner.winner_candidate_id == "A"
                })
                && reversal.metric_winners.iter().any(|winner| {
                    winner.metric == ServiceRankingMetric::LifetimeNetEnergyMwh
                        && winner.winner_candidate_id == "B"
                })
        }));
        assert_eq!(
            report.qualified_cost_energy_pareto_front
                .iter()
                .map(|point| point.candidate_id.as_str())
                .collect::<Vec<_>>(),
            vec!["B", "C"]
        );
    }

    #[test]
    fn plasma_gain_must_bind_exact_solver_run_and_output_digest() {
        let mut candidates = three_candidate_fixture();
        candidates[0].plasma_gain_run_id = "another-run".to_string();
        assert!(compare_service_boundary_rankings(&profile(), &candidates).is_err());
        let mut candidates = three_candidate_fixture();
        candidates[0].plasma_gain_output_sha256 = sha('f');
        assert!(compare_service_boundary_rankings(&profile(), &candidates).is_err());
    }

    #[test]
    fn zero_net_electricity_is_not_export_qualified() {
        let mut candidates = three_candidate_fixture();
        let row = &mut candidates[1];
        row.power_run
            .terms
            .iter_mut()
            .find(|term| term.observable == PowerObservable::NetElectric)
            .unwrap()
            .value_mw = 0.0;
        row.power_run
            .terms
            .iter_mut()
            .find(|term| term.observable == PowerObservable::GrossElectric)
            .unwrap()
            .value_mw = 130.0;
        row.power_run.output_sha256 = sha('g');
        row.plasma_gain_output_sha256 = sha('g');
        row.lifecycle_report.segment_receipts
            .iter_mut()
            .for_each(|segment| segment.output_sha256 = sha('g'));
        row.lifecycle_report_sha256 = sha('D');
        row.lifecycle_report.signed_net_energy_mwh = Some(-7.0 * 1500.0 / 3600.0);
        row.lifecycle_report.export_energy_mwh = Some(0.0);
        row.lifecycle_report.import_energy_mwh = Some(7.0 * 1500.0 / 3600.0);
        for state in &mut row.lifecycle_report.energy_by_state {
            if state.state == PowerOperatingState::PowerGeneration {
                state.signed_net_energy_mwh = 0.0;
                state.export_energy_mwh = 0.0;
                state.import_energy_mwh = 0.0;
            } else if state.state == PowerOperatingState::PlannedMaintenance {
                state.signed_net_energy_mwh = -7.0 * 1500.0 / 3600.0;
                state.export_energy_mwh = 0.0;
                state.import_energy_mwh = 7.0 * 1500.0 / 3600.0;
            }
        }
        row.power_report = evaluate_plant_power_balance(&row.power_profile, &row.power_run);
        assert_eq!(
            compare_service_boundary_rankings(&profile(), &candidates)
                .unwrap()
                .findings
                .iter()
                .find(|finding| finding.candidate_id == "B")
                .unwrap()
                .net_service_disposition,
            NetServiceDisposition::NoNetExport
        );
    }

    #[test]
    fn lifecycle_report_from_another_candidate_horizon_is_rejected() {
        let mut candidates = three_candidate_fixture();
        candidates[0].lifecycle_report.horizon_ref = "different-horizon".to_string();
        assert!(compare_service_boundary_rankings(&profile(), &candidates).is_err());
    }

    #[test]
    fn stale_lifecycle_digest_is_rejected() {
        let mut candidates = three_candidate_fixture();
        candidates[0].lifecycle_report_sha256 = "bad".to_string();
        assert!(compare_service_boundary_rankings(&profile(), &candidates).is_err());
    }

    #[test]
    fn net_power_boundary_mismatch_is_not_comparable() {
        let mut candidates = three_candidate_fixture();
        candidates[0].power_run.service_boundary_ref = "gross-output".to_string();
        assert!(compare_service_boundary_rankings(&profile(), &candidates).is_err());
    }

    #[test]
    fn lifecycle_report_must_bind_exact_power_output_digest_and_report() {
        let mut candidates = three_candidate_fixture();
        candidates[0].lifecycle_report.segment_receipts[0].output_sha256 = sha('f');
        assert!(compare_service_boundary_rankings(&profile(), &candidates).is_err());
    }

    #[test]
    fn net_power_and_recirculation_identity_must_reconcile() {
        let mut candidates = three_candidate_fixture();
        candidates[0].power_run.terms[4].value_mw = 100.0;
        candidates[0].power_report = evaluate_plant_power_balance(
            &candidates[0].power_profile,
            &candidates[0].power_run,
        );
        assert_eq!(
            compare_service_boundary_rankings(&profile(), &candidates).unwrap_err(),
            vec![ServiceBoundaryRankError::CandidatePowerReportMismatch]
        );
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected() {
        let mut candidates = three_candidate_fixture();
        candidates[1].candidate_id = candidates[0].candidate_id.clone();
        assert!(compare_service_boundary_rankings(&profile(), &candidates).is_err());
    }

    #[test]
    fn horizon_currency_and_cost_scope_must_match() {
        let mut candidates = three_candidate_fixture();
        candidates[1].cost_currency_ref = "MUSD_2025".to_string();
        assert!(compare_service_boundary_rankings(&profile(), &candidates).is_err());
        let mut candidates = three_candidate_fixture();
        candidates[1].cost_scope_ref = "undiscounted-capex".to_string();
        assert!(compare_service_boundary_rankings(&profile(), &candidates).is_err());
    }

    #[test]
    fn input_candidate_order_does_not_change_deterministic_report() {
        let first = three_candidate_fixture();
        let mut second = three_candidate_fixture();
        second.reverse();
        let report_a = compare_service_boundary_rankings(&profile(), &first).unwrap();
        let report_b = compare_service_boundary_rankings(&profile(), &second).unwrap();
        assert_eq!(report_a.rankings, report_b.rankings);
        assert_eq!(report_a.pairwise_rank_reversals, report_b.pairwise_rank_reversals);
        assert_eq!(report_a.qualified_cost_energy_pareto_front, report_b.qualified_cost_energy_pareto_front);
    }
}
