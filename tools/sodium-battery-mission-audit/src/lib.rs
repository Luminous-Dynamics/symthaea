use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TechnologyLane {
    CommercialSodiumIon,
    ZeroExcessSodiumMetal,
    LithiumIronPhosphate,
    NickelManganeseCobalt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FrontierGroup {
    CommercialCellComparison,
    PostLithiumResearch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaturityClass {
    CommercialProduct,
    PilotLine,
    LaboratoryPouch,
    CoinCell,
    ComputationalOnly,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SystemBoundary {
    CompleteCell,
    Module,
    Pack,
    InstalledSystem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MassBasis {
    ActiveMaterial,
    Electrode,
    CompleteCellIncludingInactiveComponents,
    Module,
    Pack,
    InstalledSystem,
}

impl MassBasis {
    fn matches_boundary(self, boundary: SystemBoundary) -> bool {
        matches!(
            (self, boundary),
            (
                Self::CompleteCellIncludingInactiveComponents,
                SystemBoundary::CompleteCell
            ) | (Self::Module, SystemBoundary::Module)
                | (Self::Pack, SystemBoundary::Pack)
                | (Self::InstalledSystem, SystemBoundary::InstalledSystem)
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceClass {
    SyntheticFixture,
    Assumption,
    VendorClaim,
    LiteratureReported,
    Surrogate,
    Calculation,
    PhysicalMeasurement,
    IndependentLab,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RightsState {
    CandidateUnreviewed,
    ResearchUseAdmitted,
    EvaluationOnly,
    CommercialUseBlocked,
    RedistributionBlocked,
    Rejected,
}

impl RightsState {
    fn evaluation_admitted(self) -> bool {
        matches!(self, Self::ResearchUseAdmitted | Self::EvaluationOnly)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasurementMetric {
    SpecificEnergy,
    VolumetricEnergy,
    UsableCapacity,
    ChargeAcceptance,
    PeakPower,
    RoundTripEfficiency,
    CoulombicEfficiency,
    ImpedanceGrowth,
    PreheatEnergy,
    ProcessEnergy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifetimeStatus {
    EventObserved,
    RightCensored,
    CompetingEvent,
    InvalidTest,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AmbientScenario {
    pub scenario_id: String,
    pub condition_profile_id: String,
    pub ambient_c: i32,
    pub preheat_included_and_metered: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mission {
    pub mission_id: String,
    pub duty_cycle_ref: String,
    pub system_boundary: SystemBoundary,
    pub ambient_scenarios: Vec<AmbientScenario>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lane {
    pub lane_id: String,
    pub technology: TechnologyLane,
    pub frontier_group: FrontierGroup,
    pub maturity: MaturityClass,
    pub allowed_comparators: Vec<TechnologyLane>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Specimen {
    pub specimen_id: String,
    pub design_id: String,
    pub lane_id: String,
    pub chemistry_description: String,
    pub mass_basis: MassBasis,
    pub cell_mass_g: f64,
    pub cell_volume_cm3: f64,
    pub areal_loading_mg_cm2: f64,
    pub electrolyte_to_capacity_g_ah: f64,
    pub source_lineage_ref: String,
    pub batch_group_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestCondition {
    pub condition_id: String,
    pub specimen_id: String,
    pub condition_profile_id: String,
    pub temperature_c: i32,
    pub protocol_ref: String,
    pub calibration_ref: String,
    pub raw_data_digest: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    pub measurement_id: String,
    pub specimen_id: String,
    pub condition_id: String,
    pub metric: MeasurementMetric,
    pub value: f64,
    pub unit: String,
    pub system_boundary: SystemBoundary,
    pub evidence_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LifetimeOutcome {
    pub outcome_id: String,
    pub specimen_id: String,
    pub status: LifetimeStatus,
    pub observed_cycles: u32,
    pub event_cycle: Option<u32>,
    pub event_code: Option<String>,
    pub end_of_life_definition_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub evidence_id: String,
    pub evidence_class: EvidenceClass,
    pub source_digest: String,
    pub claim_scope: String,
    pub validity_domain: String,
    pub rights_state: RightsState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComparisonSet {
    pub comparison_set_id: String,
    pub frontier_group: FrontierGroup,
    pub mission_id: String,
    pub lane_ids: Vec<String>,
    pub specimen_ids: Vec<String>,
    pub condition_profile_ids: Vec<String>,
    pub system_boundary: SystemBoundary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComparisonPolicy {
    pub require_matched_conditions: bool,
    pub separate_frontier_groups: bool,
    pub allow_active_material_to_full_cell_extrapolation: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SodiumManifest {
    pub manifest_id: String,
    pub mission: Mission,
    pub lanes: Vec<Lane>,
    pub specimens: Vec<Specimen>,
    pub conditions: Vec<TestCondition>,
    pub measurements: Vec<Measurement>,
    pub lifetime_outcomes: Vec<LifetimeOutcome>,
    pub evidence: Vec<EvidenceRecord>,
    pub comparison_sets: Vec<ComparisonSet>,
    pub policy: ComparisonPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SodiumAuditReport {
    pub manifest_id: String,
    pub specimen_count: usize,
    pub condition_count: usize,
    pub measurement_count: usize,
    pub observed_lifetime_count: usize,
    pub right_censored_count: usize,
    pub competing_event_count: usize,
    pub invalid_or_unknown_lifetime_count: usize,
    pub comparison_set_ids: Vec<String>,
    pub frontier_groups: Vec<FrontierGroup>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SodiumAuditError(pub String);

impl std::fmt::Display for SodiumAuditError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for SodiumAuditError {}

fn require_nonempty(value: &str, label: &str) -> Result<(), SodiumAuditError> {
    if value.trim().is_empty() {
        return Err(SodiumAuditError(format!("{label} must not be empty")));
    }
    Ok(())
}

fn unique_ids<'a>(
    values: impl IntoIterator<Item = &'a str>,
    label: &str,
) -> Result<BTreeSet<String>, SodiumAuditError> {
    let mut set = BTreeSet::new();

    for value in values {
        require_nonempty(value, label)?;
        if !set.insert(value.to_string()) {
            return Err(SodiumAuditError(format!("duplicate {label}: {value}")));
        }
    }

    Ok(set)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_lifetime(outcome: &LifetimeOutcome) -> Result<(), SodiumAuditError> {
    require_nonempty(&outcome.end_of_life_definition_ref, "end-of-life definition")?;

    match outcome.status {
        LifetimeStatus::EventObserved => {
            let event_cycle = outcome.event_cycle.ok_or_else(|| {
                SodiumAuditError(format!(
                    "observed event {} has no event cycle",
                    outcome.outcome_id
                ))
            })?;

            if event_cycle == 0
                || event_cycle > outcome.observed_cycles
                || outcome.event_code.is_some()
            {
                return Err(SodiumAuditError(format!(
                    "observed event {} has invalid event semantics",
                    outcome.outcome_id
                )));
            }
        }
        LifetimeStatus::RightCensored => {
            if outcome.event_cycle.is_some() || outcome.event_code.is_some() {
                return Err(SodiumAuditError(format!(
                    "right-censored outcome {} must not contain an event",
                    outcome.outcome_id
                )));
            }
        }
        LifetimeStatus::CompetingEvent => {
            let event_cycle = outcome.event_cycle.ok_or_else(|| {
                SodiumAuditError(format!(
                    "competing event {} has no event cycle",
                    outcome.outcome_id
                ))
            })?;
            require_nonempty(
                outcome.event_code.as_deref().unwrap_or(""),
                "competing event code",
            )?;

            if event_cycle == 0 || event_cycle > outcome.observed_cycles {
                return Err(SodiumAuditError(format!(
                    "competing event {} lies outside observation horizon",
                    outcome.outcome_id
                )));
            }
        }
        LifetimeStatus::InvalidTest | LifetimeStatus::Unknown => {
            if outcome.event_cycle.is_some() || outcome.event_code.is_some() {
                return Err(SodiumAuditError(format!(
                    "invalid/unknown outcome {} must not assert an event",
                    outcome.outcome_id
                )));
            }
        }
    }

    Ok(())
}

fn expected_unit(metric: MeasurementMetric) -> &'static str {
    match metric {
        MeasurementMetric::SpecificEnergy => "Wh/kg",
        MeasurementMetric::VolumetricEnergy => "Wh/L",
        MeasurementMetric::UsableCapacity => "Ah",
        MeasurementMetric::ChargeAcceptance => "C",
        MeasurementMetric::PeakPower => "W/kg",
        MeasurementMetric::RoundTripEfficiency
        | MeasurementMetric::CoulombicEfficiency
        | MeasurementMetric::ImpedanceGrowth => "fraction",
        MeasurementMetric::PreheatEnergy | MeasurementMetric::ProcessEnergy => "Wh",
    }
}

/// Audit a sodium-battery mission comparison manifest.
///
/// This is an integrity audit of declared references and comparison boundaries,
/// not an electrochemical model, predictor, laboratory or battery qualification.
pub fn audit_sodium_manifest(
    manifest: &SodiumManifest,
) -> Result<SodiumAuditReport, SodiumAuditError> {
    require_nonempty(&manifest.manifest_id, "manifest ID")?;
    require_nonempty(&manifest.mission.mission_id, "mission ID")?;
    require_nonempty(&manifest.mission.duty_cycle_ref, "duty-cycle reference")?;

    if manifest.lanes.is_empty()
        || manifest.specimens.is_empty()
        || manifest.conditions.is_empty()
        || manifest.measurements.is_empty()
        || manifest.evidence.is_empty()
        || manifest.comparison_sets.is_empty()
    {
        return Err(SodiumAuditError(
            "lanes, specimens, conditions, measurements, evidence and comparison sets are required".into(),
        ));
    }

    if !manifest.policy.require_matched_conditions
        || !manifest.policy.separate_frontier_groups
        || manifest.policy.allow_active_material_to_full_cell_extrapolation
    {
        return Err(SodiumAuditError(
            "comparison policy must match conditions, separate frontier groups and forbid active-material-to-cell extrapolation".into(),
        ));
    }

    unique_ids(
        manifest.lanes.iter().map(|item| item.lane_id.as_str()),
        "technology lane ID",
    )?;
    unique_ids(
        manifest.specimens.iter().map(|item| item.specimen_id.as_str()),
        "specimen ID",
    )?;
    unique_ids(
        manifest.conditions.iter().map(|item| item.condition_id.as_str()),
        "condition ID",
    )?;
    unique_ids(
        manifest.measurements.iter().map(|item| item.measurement_id.as_str()),
        "measurement ID",
    )?;
    unique_ids(
        manifest.lifetime_outcomes.iter().map(|item| item.outcome_id.as_str()),
        "lifetime outcome ID",
    )?;
    unique_ids(
        manifest.evidence.iter().map(|item| item.evidence_id.as_str()),
        "evidence ID",
    )?;
    unique_ids(
        manifest.comparison_sets.iter().map(|item| item.comparison_set_id.as_str()),
        "comparison set ID",
    )?;

    let mut lane_by_id = BTreeMap::new();
    for lane in &manifest.lanes {
        lane_by_id.insert(lane.lane_id.as_str(), lane);
    }

    let mut specimen_by_id = BTreeMap::new();
    for specimen in &manifest.specimens {
        require_nonempty(&specimen.design_id, "design ID")?;
        require_nonempty(&specimen.chemistry_description, "chemistry description")?;
        require_nonempty(&specimen.source_lineage_ref, "sample lineage reference")?;
        require_nonempty(&specimen.batch_group_id, "batch group ID")?;

        for (name, value) in [
            ("cell mass", specimen.cell_mass_g),
            ("cell volume", specimen.cell_volume_cm3),
            ("areal loading", specimen.areal_loading_mg_cm2),
            ("electrolyte-to-capacity ratio", specimen.electrolyte_to_capacity_g_ah),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(SodiumAuditError(format!(
                    "specimen {} has invalid {name}",
                    specimen.specimen_id
                )));
            }
        }

        if specimen.cell_mass_g <= 0.0 || specimen.cell_volume_cm3 <= 0.0 {
            return Err(SodiumAuditError(format!(
                "specimen {} must have positive mass and volume",
                specimen.specimen_id
            )));
        }

        if !lane_by_id.contains_key(specimen.lane_id.as_str()) {
            return Err(SodiumAuditError(format!(
                "specimen {} references unknown technology lane {}",
                specimen.specimen_id, specimen.lane_id
            )));
        }

        specimen_by_id.insert(specimen.specimen_id.as_str(), specimen);
    }

    let profile_by_id: BTreeMap<&str, &AmbientScenario> = manifest
        .mission
        .ambient_scenarios
        .iter()
        .map(|scenario| (scenario.condition_profile_id.as_str(), scenario))
        .collect();

    if profile_by_id.len() != manifest.mission.ambient_scenarios.len()
        || profile_by_id.is_empty()
    {
        return Err(SodiumAuditError(
            "mission condition profile IDs must be nonempty and unique".into(),
        ));
    }

    for scenario in &manifest.mission.ambient_scenarios {
        require_nonempty(&scenario.scenario_id, "ambient scenario ID")?;
        require_nonempty(&scenario.condition_profile_id, "condition profile ID")?;
    }

    let mut condition_by_id = BTreeMap::new();
    let mut profiles_by_specimen: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();

    for condition in &manifest.conditions {
        require_nonempty(&condition.protocol_ref, "test protocol reference")?;
        require_nonempty(&condition.calibration_ref, "calibration reference")?;

        let specimen = specimen_by_id
            .get(condition.specimen_id.as_str())
            .ok_or_else(|| {
                SodiumAuditError(format!(
                    "condition {} references unknown specimen {}",
                    condition.condition_id, condition.specimen_id
                ))
            })?;

        let scenario = profile_by_id
            .get(condition.condition_profile_id.as_str())
            .ok_or_else(|| {
                SodiumAuditError(format!(
                    "condition {} references unknown condition profile {}",
                    condition.condition_id, condition.condition_profile_id
                ))
            })?;

        if condition.temperature_c != scenario.ambient_c {
            return Err(SodiumAuditError(format!(
                "condition {} temperature does not match mission profile",
                condition.condition_id
            )));
        }

        if !valid_digest(&condition.raw_data_digest) {
            return Err(SodiumAuditError(format!(
                "condition {} raw-data digest is malformed",
                condition.condition_id
            )));
        }

        profiles_by_specimen
            .entry(specimen.specimen_id.as_str())
            .or_default()
            .insert(condition.condition_profile_id.as_str());

        condition_by_id.insert(condition.condition_id.as_str(), condition);
    }

    let required_profiles: BTreeSet<&str> = profile_by_id.keys().copied().collect();
    for specimen in &manifest.specimens {
        let actual = profiles_by_specimen
            .get(specimen.specimen_id.as_str())
            .cloned()
            .unwrap_or_default();

        if actual != required_profiles {
            return Err(SodiumAuditError(format!(
                "specimen {} lacks one or more required mission condition profiles",
                specimen.specimen_id
            )));
        }
    }

    let mut evidence_by_id = BTreeMap::new();
    for evidence in &manifest.evidence {
        require_nonempty(&evidence.claim_scope, "evidence claim scope")?;
        require_nonempty(&evidence.validity_domain, "evidence validity domain")?;

        if !valid_digest(&evidence.source_digest) {
            return Err(SodiumAuditError(format!(
                "evidence {} has malformed source digest",
                evidence.evidence_id
            )));
        }

        if !evidence.rights_state.evaluation_admitted() {
            return Err(SodiumAuditError(format!(
                "evidence {} is not admitted for evaluation",
                evidence.evidence_id
            )));
        }

        evidence_by_id.insert(evidence.evidence_id.as_str(), evidence);
    }

    for measurement in &manifest.measurements {
        let specimen = specimen_by_id
            .get(measurement.specimen_id.as_str())
            .ok_or_else(|| {
                SodiumAuditError(format!(
                    "measurement {} references unknown specimen {}",
                    measurement.measurement_id, measurement.specimen_id
                ))
            })?;

        let condition = condition_by_id
            .get(measurement.condition_id.as_str())
            .ok_or_else(|| {
                SodiumAuditError(format!(
                    "measurement {} references unknown condition {}",
                    measurement.measurement_id, measurement.condition_id
                ))
            })?;

        if condition.specimen_id != specimen.specimen_id {
            return Err(SodiumAuditError(format!(
                "measurement {} condition belongs to a different specimen",
                measurement.measurement_id
            )));
        }

        if !measurement.value.is_finite() || measurement.value < 0.0 {
            return Err(SodiumAuditError(format!(
                "measurement {} has invalid numeric value",
                measurement.measurement_id
            )));
        }

        if measurement.unit != expected_unit(measurement.metric) {
            return Err(SodiumAuditError(format!(
                "measurement {} has an incompatible unit",
                measurement.measurement_id
            )));
        }

        if !specimen.mass_basis.matches_boundary(measurement.system_boundary) {
            return Err(SodiumAuditError(format!(
                "measurement {} system boundary does not match specimen mass basis",
                measurement.measurement_id
            )));
        }

        if !evidence_by_id.contains_key(measurement.evidence_id.as_str()) {
            return Err(SodiumAuditError(format!(
                "measurement {} references unknown or unadmitted evidence",
                measurement.measurement_id
            )));
        }
    }

    unique_ids(
        manifest.lifetime_outcomes.iter().map(|item| item.specimen_id.as_str()),
        "lifetime outcome specimen ID",
    )?;

    let outcome_specimens: BTreeSet<String> = manifest
        .lifetime_outcomes
        .iter()
        .map(|outcome| outcome.specimen_id.clone())
        .collect();
    let all_specimens: BTreeSet<String> = manifest
        .specimens
        .iter()
        .map(|specimen| specimen.specimen_id.clone())
        .collect();

    if outcome_specimens != all_specimens {
        return Err(SodiumAuditError(
            "every specimen must have exactly one explicit lifetime outcome".into(),
        ));
    }

    for outcome in &manifest.lifetime_outcomes {
        validate_lifetime(outcome)?;
    }

    let mut comparison_set_ids = Vec::new();
    let mut frontier_groups = BTreeSet::new();

    for set in &manifest.comparison_sets {
        require_nonempty(&set.mission_id, "comparison mission ID")?;

        if set.mission_id != manifest.mission.mission_id {
            return Err(SodiumAuditError(format!(
                "comparison set {} references a different mission",
                set.comparison_set_id
            )));
        }

        let lane_ids = unique_ids(
            set.lane_ids.iter().map(String::as_str),
            "comparison lane ID",
        )?;
        let specimen_ids = unique_ids(
            set.specimen_ids.iter().map(String::as_str),
            "comparison specimen ID",
        )?;
        let profile_ids = unique_ids(
            set.condition_profile_ids.iter().map(String::as_str),
            "comparison condition profile ID",
        )?;

        if !profile_ids.is_subset(&profile_by_id.keys().map(|key| key.to_string()).collect()) {
            return Err(SodiumAuditError(format!(
                "comparison set {} references an unknown condition profile",
                set.comparison_set_id
            )));
        }

        let mut observed_lanes = BTreeSet::new();
        for specimen_id in &specimen_ids {
            let specimen = specimen_by_id.get(specimen_id.as_str()).ok_or_else(|| {
                SodiumAuditError(format!(
                    "comparison set {} references unknown specimen {}",
                    set.comparison_set_id, specimen_id
                ))
            })?;

            if !lane_ids.contains(&specimen.lane_id) {
                return Err(SodiumAuditError(format!(
                    "comparison set {} omits specimen {} technology lane",
                    set.comparison_set_id, specimen_id
                )));
            }

            let lane = lane_by_id
                .get(specimen.lane_id.as_str())
                .expect("specimen lane validated");
            if lane.frontier_group != set.frontier_group {
                return Err(SodiumAuditError(format!(
                    "comparison set {} mixes technology frontier groups",
                    set.comparison_set_id
                )));
            }

            if !specimen.mass_basis.matches_boundary(set.system_boundary) {
                return Err(SodiumAuditError(format!(
                    "comparison set {} mixes incompatible mass boundaries",
                    set.comparison_set_id
                )));
            }

            observed_lanes.insert(specimen.lane_id.clone());
        }

        if observed_lanes != lane_ids {
            return Err(SodiumAuditError(format!(
                "comparison set {} declares an empty lane or omits all lane specimens",
                set.comparison_set_id
            )));
        }

        let declared_lanes: Vec<&Lane> = lane_ids
            .iter()
            .map(|id| lane_by_id.get(id.as_str()).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                SodiumAuditError(format!(
                    "comparison set {} references unknown technology lane",
                    set.comparison_set_id
                ))
            })?;

        for left in &declared_lanes {
            for right in &declared_lanes {
                if left.lane_id == right.lane_id {
                    continue;
                }

                if !left.allowed_comparators.contains(&right.technology)
                    || !right.allowed_comparators.contains(&left.technology)
                {
                    return Err(SodiumAuditError(format!(
                        "comparison set {} contains a disallowed technology-lane pair",
                        set.comparison_set_id
                    )));
                }
            }
        }

        comparison_set_ids.push(set.comparison_set_id.clone());
        frontier_groups.insert(set.frontier_group);
    }

    comparison_set_ids.sort();

    let mut observed_lifetime_count = 0;
    let mut right_censored_count = 0;
    let mut competing_event_count = 0;
    let mut invalid_or_unknown_lifetime_count = 0;

    for outcome in &manifest.lifetime_outcomes {
        match outcome.status {
            LifetimeStatus::EventObserved => observed_lifetime_count += 1,
            LifetimeStatus::RightCensored => right_censored_count += 1,
            LifetimeStatus::CompetingEvent => competing_event_count += 1,
            LifetimeStatus::InvalidTest | LifetimeStatus::Unknown => {
                invalid_or_unknown_lifetime_count += 1
            }
        }
    }

    Ok(SodiumAuditReport {
        manifest_id: manifest.manifest_id.clone(),
        specimen_count: manifest.specimens.len(),
        condition_count: manifest.conditions.len(),
        measurement_count: manifest.measurements.len(),
        observed_lifetime_count,
        right_censored_count,
        competing_event_count,
        invalid_or_unknown_lifetime_count,
        comparison_set_ids,
        frontier_groups: frontier_groups.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest() -> SodiumManifest {
        let mission = Mission {
            mission_id: "MISSION".into(),
            duty_cycle_ref: "DUTY".into(),
            system_boundary: SystemBoundary::CompleteCell,
            ambient_scenarios: vec![
                AmbientScenario {
                    scenario_id: "REF".into(),
                    condition_profile_id: "REF25".into(),
                    ambient_c: 25,
                    preheat_included_and_metered: true,
                },
                AmbientScenario {
                    scenario_id: "COLD".into(),
                    condition_profile_id: "COLD20".into(),
                    ambient_c: -20,
                    preheat_included_and_metered: true,
                },
            ],
        };

        let lanes = vec![
            Lane {
                lane_id: "SIB".into(),
                technology: TechnologyLane::CommercialSodiumIon,
                frontier_group: FrontierGroup::CommercialCellComparison,
                maturity: MaturityClass::PilotLine,
                allowed_comparators: vec![
                    TechnologyLane::CommercialSodiumIon,
                    TechnologyLane::LithiumIronPhosphate,
                ],
            },
            Lane {
                lane_id: "LFP".into(),
                technology: TechnologyLane::LithiumIronPhosphate,
                frontier_group: FrontierGroup::CommercialCellComparison,
                maturity: MaturityClass::CommercialProduct,
                allowed_comparators: vec![
                    TechnologyLane::CommercialSodiumIon,
                    TechnologyLane::LithiumIronPhosphate,
                ],
            },
            Lane {
                lane_id: "ZEM".into(),
                technology: TechnologyLane::ZeroExcessSodiumMetal,
                frontier_group: FrontierGroup::PostLithiumResearch,
                maturity: MaturityClass::LaboratoryPouch,
                allowed_comparators: vec![TechnologyLane::ZeroExcessSodiumMetal],
            },
        ];

        let specimens = vec![
            specimen("SIB-1", "DESIGN-SIB", "SIB"),
            specimen("SIB-2", "DESIGN-SIB", "SIB"),
            specimen("LFP-1", "DESIGN-LFP", "LFP"),
            specimen("LFP-2", "DESIGN-LFP", "LFP"),
            specimen("ZEM-1", "DESIGN-ZEM", "ZEM"),
        ];

        let mut conditions = Vec::new();
        for item in &specimens {
            for (profile, temperature) in [("REF25", 25), ("COLD20", -20)] {
                conditions.push(TestCondition {
                    condition_id: format!("{}-{profile}", item.specimen_id),
                    specimen_id: item.specimen_id.clone(),
                    condition_profile_id: profile.into(),
                    temperature_c: temperature,
                    protocol_ref: format!("PROTO-{profile}"),
                    calibration_ref: "CAL-1".into(),
                    raw_data_digest: "a".repeat(64),
                });
            }
        }

        let measurements = conditions
            .iter()
            .map(|condition| Measurement {
                measurement_id: format!("MEAS-{}", condition.condition_id),
                specimen_id: condition.specimen_id.clone(),
                condition_id: condition.condition_id.clone(),
                metric: MeasurementMetric::UsableCapacity,
                value: 2.5,
                unit: "Ah".into(),
                system_boundary: SystemBoundary::CompleteCell,
                evidence_id: "SYNTH-EVIDENCE".into(),
            })
            .collect();

        let lifetime_outcomes = vec![
            lifetime_event("SIB-1", 900),
            lifetime_censored("SIB-2", 650),
            lifetime_event("LFP-1", 1500),
            lifetime_censored("LFP-2", 1300),
            LifetimeOutcome {
                outcome_id: "LIFE-ZEM".into(),
                specimen_id: "ZEM-1".into(),
                status: LifetimeStatus::CompetingEvent,
                observed_cycles: 120,
                event_cycle: Some(120),
                event_code: Some("INVENTORY-LOSS".into()),
                end_of_life_definition_ref: "ZEM-ENDPOINT".into(),
            },
        ];

        SodiumManifest {
            manifest_id: "SYNTH-SODIUM".into(),
            mission,
            lanes,
            specimens,
            conditions,
            measurements,
            lifetime_outcomes,
            evidence: vec![EvidenceRecord {
                evidence_id: "SYNTH-EVIDENCE".into(),
                evidence_class: EvidenceClass::SyntheticFixture,
                source_digest: "b".repeat(64),
                claim_scope: "synthetic fixture only".into(),
                validity_domain: "unit tests only".into(),
                rights_state: RightsState::EvaluationOnly,
            }],
            comparison_sets: vec![
                ComparisonSet {
                    comparison_set_id: "COMMERCIAL".into(),
                    frontier_group: FrontierGroup::CommercialCellComparison,
                    mission_id: "MISSION".into(),
                    lane_ids: vec!["SIB".into(), "LFP".into()],
                    specimen_ids: vec![
                        "SIB-1".into(),
                        "SIB-2".into(),
                        "LFP-1".into(),
                        "LFP-2".into(),
                    ],
                    condition_profile_ids: vec!["REF25".into(), "COLD20".into()],
                    system_boundary: SystemBoundary::CompleteCell,
                },
                ComparisonSet {
                    comparison_set_id: "RESEARCH".into(),
                    frontier_group: FrontierGroup::PostLithiumResearch,
                    mission_id: "MISSION".into(),
                    lane_ids: vec!["ZEM".into()],
                    specimen_ids: vec!["ZEM-1".into()],
                    condition_profile_ids: vec!["REF25".into(), "COLD20".into()],
                    system_boundary: SystemBoundary::CompleteCell,
                },
            ],
            policy: ComparisonPolicy {
                require_matched_conditions: true,
                separate_frontier_groups: true,
                allow_active_material_to_full_cell_extrapolation: false,
            },
        }
    }

    fn specimen(id: &str, design: &str, lane: &str) -> Specimen {
        Specimen {
            specimen_id: id.into(),
            design_id: design.into(),
            lane_id: lane.into(),
            chemistry_description: "synthetic test chemistry".into(),
            mass_basis: MassBasis::CompleteCellIncludingInactiveComponents,
            cell_mass_g: 150.0,
            cell_volume_cm3: 85.0,
            areal_loading_mg_cm2: 18.0,
            electrolyte_to_capacity_g_ah: 3.0,
            source_lineage_ref: format!("LINEAGE-{id}"),
            batch_group_id: format!("BATCH-{id}"),
        }
    }

    fn lifetime_event(specimen_id: &str, event_cycle: u32) -> LifetimeOutcome {
        LifetimeOutcome {
            outcome_id: format!("LIFE-{specimen_id}"),
            specimen_id: specimen_id.into(),
            status: LifetimeStatus::EventObserved,
            observed_cycles: event_cycle,
            event_cycle: Some(event_cycle),
            event_code: None,
            end_of_life_definition_ref: "80PCT".into(),
        }
    }

    fn lifetime_censored(specimen_id: &str, observed_cycles: u32) -> LifetimeOutcome {
        LifetimeOutcome {
            outcome_id: format!("LIFE-{specimen_id}"),
            specimen_id: specimen_id.into(),
            status: LifetimeStatus::RightCensored,
            observed_cycles,
            event_cycle: None,
            event_code: None,
            end_of_life_definition_ref: "80PCT".into(),
        }
    }

    #[test]
    fn accepts_separated_commercial_and_zero_excess_frontiers() {
        let report = audit_sodium_manifest(&valid_manifest()).expect("valid fixture");
        assert_eq!(report.specimen_count, 5);
        assert_eq!(report.condition_count, 10);
        assert_eq!(report.measurement_count, 10);
        assert_eq!(report.observed_lifetime_count, 2);
        assert_eq!(report.right_censored_count, 2);
        assert_eq!(report.competing_event_count, 1);
        assert_eq!(report.frontier_groups.len(), 2);
    }

    #[test]
    fn rejects_zero_excess_sodium_metal_in_commercial_frontier() {
        let mut manifest = valid_manifest();
        manifest.comparison_sets[0].lane_ids.push("ZEM".into());
        manifest.comparison_sets[0].specimen_ids.push("ZEM-1".into());

        let error = audit_sodium_manifest(&manifest).expect_err("frontier mixing rejected");
        assert!(error.0.contains("mixes technology frontier groups"));
    }

    #[test]
    fn rejects_active_material_mass_promoted_to_complete_cell() {
        let mut manifest = valid_manifest();
        manifest.specimens[0].mass_basis = MassBasis::ActiveMaterial;

        let error = audit_sodium_manifest(&manifest).expect_err("mass boundary mismatch");
        assert!(error.0.contains("system boundary does not match specimen mass basis"));
    }

    #[test]
    fn rejects_condition_temperature_not_matching_mission_profile() {
        let mut manifest = valid_manifest();
        manifest.conditions[0].temperature_c = 10;

        let error = audit_sodium_manifest(&manifest).expect_err("condition mismatch");
        assert!(error.0.contains("temperature does not match mission profile"));
    }

    #[test]
    fn rejects_unadmitted_evidence_rights() {
        let mut manifest = valid_manifest();
        manifest.evidence[0].rights_state = RightsState::CandidateUnreviewed;

        let error = audit_sodium_manifest(&manifest).expect_err("unreviewed rights rejected");
        assert!(error.0.contains("not admitted for evaluation"));
    }

    #[test]
    fn rejects_unknown_measurement_evidence() {
        let mut manifest = valid_manifest();
        manifest.measurements[0].evidence_id = "MISSING".into();

        let error = audit_sodium_manifest(&manifest).expect_err("missing evidence rejected");
        assert!(error.0.contains("unknown or unadmitted evidence"));
    }

    #[test]
    fn rejects_right_censored_outcome_with_pseudo_event() {
        let mut manifest = valid_manifest();
        manifest.lifetime_outcomes[1].event_cycle = Some(650);

        let error = audit_sodium_manifest(&manifest).expect_err("censoring semantics rejected");
        assert!(error.0.contains("must not contain an event"));
    }

    #[test]
    fn rejects_competing_event_without_code() {
        let mut manifest = valid_manifest();
        manifest.lifetime_outcomes[4].event_code = None;

        let error = audit_sodium_manifest(&manifest).expect_err("competing code required");
        assert!(error.0.contains("competing event code"));
    }

    #[test]
    fn rejects_measurement_joined_to_other_specimen_condition() {
        let mut manifest = valid_manifest();
        manifest.measurements[0].specimen_id = "LFP-1".into();

        let error = audit_sodium_manifest(&manifest).expect_err("cross-specimen join rejected");
        assert!(error.0.contains("condition belongs to a different specimen"));
    }

    #[test]
    fn rejects_invalid_raw_data_digest() {
        let mut manifest = valid_manifest();
        manifest.conditions[0].raw_data_digest = "placeholder".into();

        let error = audit_sodium_manifest(&manifest).expect_err("digest rejected");
        assert!(error.0.contains("raw-data digest is malformed"));
    }

    #[test]
    fn rejects_comparison_lane_that_is_not_authorized() {
        let mut manifest = valid_manifest();
        manifest.lanes[0].allowed_comparators = vec![TechnologyLane::CommercialSodiumIon];

        let error = audit_sodium_manifest(&manifest).expect_err("comparator policy rejected");
        assert!(error.0.contains("disallowed technology-lane pair"));
    }

    #[test]
    fn report_is_invariant_to_input_order() {
        let mut first = valid_manifest();
        let first_report = audit_sodium_manifest(&first).expect("first valid manifest");

        first.lanes.reverse();
        first.specimens.reverse();
        first.conditions.reverse();
        first.measurements.reverse();
        first.lifetime_outcomes.reverse();
        first.comparison_sets.reverse();

        let second_report = audit_sodium_manifest(&first).expect("permuted valid manifest");
        assert_eq!(first_report, second_report);
    }
}
