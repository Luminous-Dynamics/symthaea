#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TechnologyLane {
    SulfideLithiumMetal,
    Halide,
    Oxide,
    PolymerComposite,
    SiliconAssb,
    HybridSolidLiquid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TestLevel {
    FullCell,
    SymmetricCell,
    HalfCell,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceClass {
    PhysicalMeasurement,
    IndependentLaboratory,
    Literature,
    Simulation,
    Synthetic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    Candidate,
    AdmittedForResearch,
    AdmittedForEvaluation,
    Blocked,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutcomeStatus {
    ObservedEol,
    RightCensored,
    Failed,
    Invalid,
    Incomplete,
    SyntheticOutcome,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatchedAxis {
    TechnologyLane,
    ChemistryFamily,
    ElectrolyteFamily,
    CathodeFamily,
    AnodeConfiguration,
    CellFormat,
    TestLevel,
    ArealCapacity,
    CathodeLoading,
    ElectrolyteThickness,
    Temperature,
    CurrentDensity,
    VoltageWindow,
    CycleProtocol,
}

impl MatchedAxis {
    pub const REQUIRED: [Self; 14] = [
        Self::TechnologyLane,
        Self::ChemistryFamily,
        Self::ElectrolyteFamily,
        Self::CathodeFamily,
        Self::AnodeConfiguration,
        Self::CellFormat,
        Self::TestLevel,
        Self::ArealCapacity,
        Self::CathodeLoading,
        Self::ElectrolyteThickness,
        Self::Temperature,
        Self::CurrentDensity,
        Self::VoltageWindow,
        Self::CycleProtocol,
    ];
}

#[derive(Clone, Debug, PartialEq)]
pub struct MassInventory {
    pub cathode_active_mg: f64,
    pub cathode_inactive_mg: f64,
    pub anode_active_mg: f64,
    pub anode_inactive_mg: f64,
    pub electrolyte_mg: f64,
    pub separator_or_support_mg: f64,
    pub current_collectors_and_tabs_mg: f64,
    pub casing_and_seals_mg: f64,
    pub declared_full_cell_mass_mg: f64,
    pub closure_tolerance_mg: f64,
    pub boundary_digest: String,
    pub declared_full_cell_volume_cm3: f64,
    pub volume_boundary_digest: String,
}

impl MassInventory {
    pub fn component_sum_mg(&self) -> f64 {
        self.cathode_active_mg
            + self.cathode_inactive_mg
            + self.anode_active_mg
            + self.anode_inactive_mg
            + self.electrolyte_mg
            + self.separator_or_support_mg
            + self.current_collectors_and_tabs_mg
            + self.casing_and_seals_mg
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DesignDescriptor {
    pub design_id: String,
    pub revision: String,
    pub technology_lane: TechnologyLane,
    pub chemistry_family: String,
    pub electrolyte_family: String,
    pub cathode_family: String,
    pub anode_configuration: String,
    pub cell_format: String,
    pub test_level: TestLevel,
    pub areal_capacity_mah_cm2: f64,
    pub cathode_loading_mg_cm2: f64,
    pub electrolyte_thickness_um: f64,
    pub temperature_c: f64,
    pub current_density_ma_cm2: f64,
    pub voltage_window: String,
    pub cycle_protocol: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PressureEvidence {
    pub fabrication_pressure_mpa: f64,
    pub operating_pressure_mpa: f64,
    pub measurement_method: String,
    pub instrument_ref: String,
    pub uncertainty_mpa: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceRecord {
    pub evidence_id: String,
    pub source_digest: String,
    pub evidence_class: EvidenceClass,
    pub admission: Admission,
    pub claim_scope: String,
    pub rights_ref: String,
    pub metrology_ref: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CellObservation {
    pub observation_id: String,
    pub cell_id: String,
    pub batch_id: String,
    pub manufacturing_line: String,
    pub process_revision: String,
    pub design: DesignDescriptor,
    pub pressure: PressureEvidence,
    pub mass_inventory: Option<MassInventory>,
    pub cycles_observed: u32,
    pub outcome_status: OutcomeStatus,
    pub event_cycle: Option<u32>,
    pub capacity_retention_permille: Option<u16>,
    pub impedance_growth_permille: Option<u16>,
    pub evidence_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PressureComparison {
    pub comparison_id: String,
    pub baseline_observation_id: String,
    pub candidate_observation_id: String,
    pub matched_axes: Vec<MatchedAxis>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Manifest {
    pub campaign_id: String,
    pub evidence: Vec<EvidenceRecord>,
    pub observations: Vec<CellObservation>,
    pub comparisons: Vec<PressureComparison>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComparisonReport {
    pub comparison_id: String,
    pub baseline_observation_id: String,
    pub candidate_observation_id: String,
    pub matched_axis_count: usize,
    pub pressure_measurement_method: String,
    pub baseline_fabrication_pressure_milli_mpa: u64,
    pub candidate_fabrication_pressure_milli_mpa: u64,
    pub fabrication_pressure_delta_milli_mpa: i64,
    pub baseline_operating_pressure_milli_mpa: u64,
    pub candidate_operating_pressure_milli_mpa: u64,
    pub pressure_delta_milli_mpa: i64,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditReport {
    pub campaign_id: String,
    pub observation_count: usize,
    pub full_cell_count: usize,
    pub symmetric_cell_count: usize,
    pub half_cell_count: usize,
    pub synthetic_outcome_count: usize,
    pub physical_observation_count: usize,
    pub right_censored_count: usize,
    pub failed_or_invalid_count: usize,
    pub incomplete_count: usize,
    pub comparison_count: usize,
    pub comparisons: Vec<ComparisonReport>,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditError(pub String);

impl std::fmt::Display for AuditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AuditError {}

fn require_nonempty(value: &str, name: &str) -> Result<(), AuditError> {
    if value.trim().is_empty() {
        return Err(AuditError(format!("{name} must not be empty")));
    }
    Ok(())
}

fn finite_nonnegative(value: f64, name: &str) -> Result<(), AuditError> {
    if !value.is_finite() || value < 0.0 {
        return Err(AuditError(format!("{name} must be finite and nonnegative")));
    }
    Ok(())
}

fn finite_positive(value: f64, name: &str) -> Result<(), AuditError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(AuditError(format!("{name} must be finite and positive")));
    }
    Ok(())
}

fn validate_digest(value: &str, name: &str) -> Result<(), AuditError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(AuditError(format!(
            "{name} must be lowercase 64-character SHA-256"
        )));
    }
    Ok(())
}

fn validate_unique<'a>(
    values: impl IntoIterator<Item = &'a str>,
    name: &str,
) -> Result<(), AuditError> {
    let mut seen = BTreeSet::new();
    for value in values {
        require_nonempty(value, name)?;
        if !seen.insert(value) {
            return Err(AuditError(format!("duplicate {name}: {value}")));
        }
    }
    Ok(())
}

fn evidence_registry(
    records: &[EvidenceRecord],
) -> Result<BTreeMap<&str, &EvidenceRecord>, AuditError> {
    if records.is_empty() {
        return Err(AuditError("evidence registry must not be empty".into()));
    }
    validate_unique(
        records.iter().map(|r| r.evidence_id.as_str()),
        "evidence ID",
    )?;
    let mut map = BTreeMap::new();
    for record in records {
        validate_digest(&record.source_digest, "evidence source digest")?;
        require_nonempty(&record.claim_scope, "evidence claim scope")?;
        require_nonempty(&record.rights_ref, "evidence rights reference")?;
        if (record.evidence_class == EvidenceClass::PhysicalMeasurement
            || record.evidence_class == EvidenceClass::IndependentLaboratory)
            && record.metrology_ref.as_deref().is_none_or(str::is_empty)
        {
            return Err(AuditError(format!(
                "physical evidence {} requires a metrology reference",
                record.evidence_id
            )));
        }
        map.insert(record.evidence_id.as_str(), record);
    }
    Ok(map)
}

fn admitted_evidence<'a>(
    registry: &'a BTreeMap<&'a str, &'a EvidenceRecord>,
    id: &str,
    owner: &str,
) -> Result<&'a EvidenceRecord, AuditError> {
    let record = registry
        .get(id)
        .copied()
        .ok_or_else(|| AuditError(format!("{owner} references unknown evidence {id}")))?;
    if record.admission != Admission::AdmittedForEvaluation {
        return Err(AuditError(format!(
            "{owner} evidence {id} is not admitted for evaluation"
        )));
    }
    Ok(record)
}

fn validate_design(d: &DesignDescriptor) -> Result<(), AuditError> {
    for (v, n) in [
        (d.design_id.as_str(), "design ID"),
        (d.revision.as_str(), "design revision"),
        (d.chemistry_family.as_str(), "chemistry family"),
        (d.electrolyte_family.as_str(), "electrolyte family"),
        (d.cathode_family.as_str(), "cathode family"),
        (d.anode_configuration.as_str(), "anode configuration"),
        (d.cell_format.as_str(), "cell format"),
        (d.voltage_window.as_str(), "voltage window"),
        (d.cycle_protocol.as_str(), "cycle protocol"),
    ] {
        require_nonempty(v, n)?;
    }
    finite_positive(d.areal_capacity_mah_cm2, "areal capacity")?;
    finite_positive(d.cathode_loading_mg_cm2, "cathode loading")?;
    finite_positive(d.electrolyte_thickness_um, "electrolyte thickness")?;
    finite_positive(d.temperature_c + 273.15, "absolute temperature")?;
    finite_positive(d.current_density_ma_cm2, "current density")?;
    Ok(())
}

fn validate_pressure(p: &PressureEvidence) -> Result<(), AuditError> {
    finite_nonnegative(p.fabrication_pressure_mpa, "fabrication pressure")?;
    finite_nonnegative(p.operating_pressure_mpa, "operating pressure")?;
    finite_nonnegative(p.uncertainty_mpa, "pressure uncertainty")?;
    require_nonempty(&p.measurement_method, "pressure measurement method")?;
    require_nonempty(&p.instrument_ref, "pressure instrument reference")?;
    if p.uncertainty_mpa > p.fabrication_pressure_mpa.max(p.operating_pressure_mpa)
        && p.uncertainty_mpa > 0.0
    {
        return Err(AuditError(
            "pressure uncertainty exceeds both declared pressure magnitudes".into(),
        ));
    }
    Ok(())
}

fn validate_mass(m: &MassInventory) -> Result<(), AuditError> {
    for (v, n) in [
        (m.cathode_active_mg, "cathode active mass"),
        (m.cathode_inactive_mg, "cathode inactive composite mass"),
        (m.anode_active_mg, "anode active mass"),
        (m.anode_inactive_mg, "anode inactive mass"),
        (m.electrolyte_mg, "electrolyte mass"),
        (m.separator_or_support_mg, "separator/support mass"),
        (
            m.current_collectors_and_tabs_mg,
            "current collector/tab mass",
        ),
        (m.casing_and_seals_mg, "casing/seal mass"),
    ] {
        finite_nonnegative(v, n)?;
    }
    finite_positive(m.declared_full_cell_mass_mg, "declared full-cell mass")?;
    finite_nonnegative(m.closure_tolerance_mg, "mass closure tolerance")?;
    validate_digest(&m.boundary_digest, "complete-cell boundary digest")?;
    finite_positive(
        m.declared_full_cell_volume_cm3,
        "declared full-cell external volume",
    )?;
    validate_digest(
        &m.volume_boundary_digest,
        "complete-cell volume boundary digest",
    )?;
    let delta = (m.component_sum_mg() - m.declared_full_cell_mass_mg).abs();
    if delta > m.closure_tolerance_mg {
        return Err(AuditError(format!(
            "full-cell mass does not close: delta={delta} mg exceeds tolerance={} mg",
            m.closure_tolerance_mg
        )));
    }
    Ok(())
}

fn validate_status(o: &CellObservation, e: &EvidenceRecord) -> Result<(), AuditError> {
    match o.outcome_status {
        OutcomeStatus::ObservedEol => {
            if e.evidence_class == EvidenceClass::Synthetic
                || e.evidence_class == EvidenceClass::Simulation
            {
                return Err(AuditError(format!(
                    "observed EOL for {} requires measured evidence",
                    o.observation_id
                )));
            }
            let cycle = o.event_cycle.ok_or_else(|| {
                AuditError(format!(
                    "observed EOL for {} requires event cycle",
                    o.observation_id
                ))
            })?;
            if cycle == 0 || cycle > o.cycles_observed {
                return Err(AuditError(
                    "EOL event cycle outside observed exposure".into(),
                ));
            }
        }
        OutcomeStatus::RightCensored => {
            if e.evidence_class == EvidenceClass::Synthetic
                || e.evidence_class == EvidenceClass::Simulation
            {
                return Err(AuditError(format!(
                    "right-censored exposure for {} requires measured evidence",
                    o.observation_id
                )));
            }
            if o.cycles_observed == 0 || o.event_cycle.is_some() {
                return Err(AuditError(format!(
                    "right-censored {} requires observed exposure and no EOL event",
                    o.observation_id
                )));
            }
        }
        OutcomeStatus::Failed | OutcomeStatus::Invalid | OutcomeStatus::Incomplete => {
            if o.event_cycle.is_some() {
                return Err(AuditError(format!(
                    "failed/invalid/incomplete observation {} cannot claim observed EOL",
                    o.observation_id
                )));
            }
        }
        OutcomeStatus::SyntheticOutcome => {
            if e.evidence_class != EvidenceClass::Synthetic {
                return Err(AuditError(format!(
                    "synthetic outcome {} requires synthetic evidence",
                    o.observation_id
                )));
            }
            if o.cycles_observed == 0 {
                return Err(AuditError(
                    "synthetic observation needs nonzero exposure".into(),
                ));
            }
            if o.event_cycle
                .is_some_and(|c| c == 0 || c > o.cycles_observed)
            {
                return Err(AuditError("synthetic event cycle outside exposure".into()));
            }
        }
    }
    if let Some(v) = o.capacity_retention_permille
        && v > 1000
    {
        return Err(AuditError(
            "capacity retention must be <=1000 permille".into(),
        ));
    }
    Ok(())
}

fn axis_matches(a: &DesignDescriptor, b: &DesignDescriptor, axis: MatchedAxis) -> bool {
    match axis {
        MatchedAxis::TechnologyLane => a.technology_lane == b.technology_lane,
        MatchedAxis::ChemistryFamily => a.chemistry_family == b.chemistry_family,
        MatchedAxis::ElectrolyteFamily => a.electrolyte_family == b.electrolyte_family,
        MatchedAxis::CathodeFamily => a.cathode_family == b.cathode_family,
        MatchedAxis::AnodeConfiguration => a.anode_configuration == b.anode_configuration,
        MatchedAxis::CellFormat => a.cell_format == b.cell_format,
        MatchedAxis::TestLevel => a.test_level == b.test_level,
        MatchedAxis::ArealCapacity => {
            (a.areal_capacity_mah_cm2 - b.areal_capacity_mah_cm2).abs() <= 1e-9
        }
        MatchedAxis::CathodeLoading => {
            (a.cathode_loading_mg_cm2 - b.cathode_loading_mg_cm2).abs() <= 1e-9
        }
        MatchedAxis::ElectrolyteThickness => {
            (a.electrolyte_thickness_um - b.electrolyte_thickness_um).abs() <= 1e-9
        }
        MatchedAxis::Temperature => (a.temperature_c - b.temperature_c).abs() <= 1e-9,
        MatchedAxis::CurrentDensity => {
            (a.current_density_ma_cm2 - b.current_density_ma_cm2).abs() <= 1e-9
        }
        MatchedAxis::VoltageWindow => a.voltage_window == b.voltage_window,
        MatchedAxis::CycleProtocol => a.cycle_protocol == b.cycle_protocol,
    }
}

fn pressure_milli_mpa(v: f64) -> u64 {
    (v * 1000.0).round() as u64
}

pub fn audit_manifest(manifest: &Manifest) -> Result<AuditReport, AuditError> {
    require_nonempty(&manifest.campaign_id, "campaign ID")?;
    if manifest.observations.is_empty() {
        return Err(AuditError("campaign requires observations".into()));
    }
    let registry = evidence_registry(&manifest.evidence)?;
    validate_unique(
        manifest
            .observations
            .iter()
            .map(|o| o.observation_id.as_str()),
        "observation ID",
    )?;
    validate_unique(
        manifest
            .comparisons
            .iter()
            .map(|c| c.comparison_id.as_str()),
        "comparison ID",
    )?;

    let mut observation_map = BTreeMap::new();
    let mut full_cell_count = 0;
    let mut symmetric_cell_count = 0;
    let mut half_cell_count = 0;
    let mut synthetic_outcome_count = 0;
    let mut physical_observation_count = 0;
    let mut right_censored_count = 0;
    let mut failed_or_invalid_count = 0;
    let mut incomplete_count = 0;

    for o in &manifest.observations {
        require_nonempty(&o.observation_id, "observation ID")?;
        require_nonempty(&o.cell_id, "cell ID")?;
        require_nonempty(&o.batch_id, "manufacturing batch ID")?;
        require_nonempty(&o.manufacturing_line, "manufacturing line ID")?;
        require_nonempty(&o.process_revision, "manufacturing process revision")?;
        validate_design(&o.design)?;
        validate_pressure(&o.pressure)?;
        let evidence = admitted_evidence(
            &registry,
            &o.evidence_id,
            &format!("observation {}", o.observation_id),
        )?;
        match o.design.test_level {
            TestLevel::FullCell => {
                let inventory = o.mass_inventory.as_ref().ok_or_else(|| {
                    AuditError(format!(
                        "full-cell observation {} requires mass inventory",
                        o.observation_id
                    ))
                })?;
                validate_mass(inventory)?;
                full_cell_count += 1;
            }
            TestLevel::SymmetricCell => {
                if o.mass_inventory.is_some() {
                    return Err(AuditError(format!(
                        "symmetric-cell observation {} must not be promoted as full-cell mass inventory",
                        o.observation_id
                    )));
                }
                symmetric_cell_count += 1;
            }
            TestLevel::HalfCell => {
                if o.mass_inventory.is_some() {
                    return Err(AuditError(format!(
                        "half-cell observation {} must not carry complete-cell mass inventory",
                        o.observation_id
                    )));
                }
                half_cell_count += 1;
            }
        }
        validate_status(o, evidence)?;
        match o.outcome_status {
            OutcomeStatus::SyntheticOutcome => synthetic_outcome_count += 1,
            OutcomeStatus::ObservedEol => physical_observation_count += 1,
            OutcomeStatus::RightCensored => {
                right_censored_count += 1;
                physical_observation_count += 1;
            }
            OutcomeStatus::Failed | OutcomeStatus::Invalid => failed_or_invalid_count += 1,
            OutcomeStatus::Incomplete => incomplete_count += 1,
        }
        observation_map.insert(o.observation_id.as_str(), o);
    }

    let mut reports = Vec::new();
    for comparison in &manifest.comparisons {
        require_nonempty(&comparison.comparison_id, "comparison ID")?;
        let baseline = observation_map
            .get(comparison.baseline_observation_id.as_str())
            .copied()
            .ok_or_else(|| {
                AuditError(format!(
                    "comparison {} references missing baseline",
                    comparison.comparison_id
                ))
            })?;
        let candidate = observation_map
            .get(comparison.candidate_observation_id.as_str())
            .copied()
            .ok_or_else(|| {
                AuditError(format!(
                    "comparison {} references missing candidate",
                    comparison.comparison_id
                ))
            })?;

        if comparison.baseline_observation_id == comparison.candidate_observation_id {
            return Err(AuditError(
                "comparison must use distinct observations".into(),
            ));
        }
        if baseline.design.technology_lane != candidate.design.technology_lane {
            return Err(AuditError(format!(
                "comparison {} crosses technology lanes; stratify rather than pool",
                comparison.comparison_id
            )));
        }
        if baseline.design.test_level != candidate.design.test_level {
            return Err(AuditError(format!(
                "comparison {} mixes full, symmetric or half-cell test levels",
                comparison.comparison_id
            )));
        }
        if baseline.design.design_id != candidate.design.design_id
            || baseline.design.revision != candidate.design.revision
        {
            return Err(AuditError(format!(
                "comparison {} changes design identity/revision; only operating pressure may vary",
                comparison.comparison_id
            )));
        }
        if baseline.batch_id != candidate.batch_id
            || baseline.manufacturing_line != candidate.manufacturing_line
            || baseline.process_revision != candidate.process_revision
        {
            return Err(AuditError(format!(
                "comparison {} changes batch, manufacturing line or process revision; stratify the process effect separately",
                comparison.comparison_id
            )));
        }
        if baseline.pressure.measurement_method != candidate.pressure.measurement_method {
            return Err(AuditError(format!(
                "comparison {} uses incompatible pressure measurement methods",
                comparison.comparison_id
            )));
        }
        let fabrication_pressure_delta = (candidate.pressure.fabrication_pressure_mpa
            - baseline.pressure.fabrication_pressure_mpa)
            .abs();
        let fabrication_pressure_tolerance =
            baseline.pressure.uncertainty_mpa + candidate.pressure.uncertainty_mpa;
        if fabrication_pressure_delta > fabrication_pressure_tolerance {
            return Err(AuditError(format!(
                "comparison {} confounds operating pressure with fabrication pressure (delta={} MPa exceeds combined uncertainty={} MPa)",
                comparison.comparison_id,
                fabrication_pressure_delta,
                fabrication_pressure_tolerance
            )));
        }
        if baseline.pressure.operating_pressure_mpa.to_bits()
            == candidate.pressure.operating_pressure_mpa.to_bits()
        {
            return Err(AuditError(format!(
                "comparison {} must vary operating pressure to qualify a pressure comparison",
                comparison.comparison_id
            )));
        }
        validate_unique(
            comparison.matched_axes.iter().map(|axis| match axis {
                MatchedAxis::TechnologyLane => "technology_lane",
                MatchedAxis::ChemistryFamily => "chemistry_family",
                MatchedAxis::ElectrolyteFamily => "electrolyte_family",
                MatchedAxis::CathodeFamily => "cathode_family",
                MatchedAxis::AnodeConfiguration => "anode_configuration",
                MatchedAxis::CellFormat => "cell_format",
                MatchedAxis::TestLevel => "test_level",
                MatchedAxis::ArealCapacity => "areal_capacity",
                MatchedAxis::CathodeLoading => "cathode_loading",
                MatchedAxis::ElectrolyteThickness => "electrolyte_thickness",
                MatchedAxis::Temperature => "temperature",
                MatchedAxis::CurrentDensity => "current_density",
                MatchedAxis::VoltageWindow => "voltage_window",
                MatchedAxis::CycleProtocol => "cycle_protocol",
            }),
            "matched axis",
        )?;
        let matched: BTreeSet<_> = comparison.matched_axes.iter().copied().collect();
        for required in MatchedAxis::REQUIRED {
            if !matched.contains(&required) {
                return Err(AuditError(format!(
                    "comparison {} omits required matched axis {required:?}",
                    comparison.comparison_id
                )));
            }
        }
        for axis in &comparison.matched_axes {
            if !axis_matches(&baseline.design, &candidate.design, *axis) {
                return Err(AuditError(format!(
                    "comparison {} declares {:?} matched but values differ",
                    comparison.comparison_id, axis
                )));
            }
        }
        if baseline.design.test_level == TestLevel::FullCell
            && (baseline.mass_inventory.is_none() || candidate.mass_inventory.is_none())
        {
            return Err(AuditError(
                "full-cell comparison requires complete-cell mass on both sides".into(),
            ));
        }
        let delta = (candidate.pressure.operating_pressure_mpa
            - baseline.pressure.operating_pressure_mpa)
            * 1000.0;
        reports.push(ComparisonReport {
            comparison_id: comparison.comparison_id.clone(),
            baseline_observation_id: baseline.observation_id.clone(),
            candidate_observation_id: candidate.observation_id.clone(),
            matched_axis_count: comparison.matched_axes.len(),
            pressure_measurement_method: baseline.pressure.measurement_method.clone(),
            baseline_fabrication_pressure_milli_mpa: pressure_milli_mpa(
                baseline.pressure.fabrication_pressure_mpa,
            ),
            candidate_fabrication_pressure_milli_mpa: pressure_milli_mpa(
                candidate.pressure.fabrication_pressure_mpa,
            ),
            fabrication_pressure_delta_milli_mpa: ((candidate.pressure.fabrication_pressure_mpa
                - baseline.pressure.fabrication_pressure_mpa)
                * 1000.0)
                .round() as i64,
            baseline_operating_pressure_milli_mpa: pressure_milli_mpa(baseline.pressure.operating_pressure_mpa),
            candidate_operating_pressure_milli_mpa: pressure_milli_mpa(candidate.pressure.operating_pressure_mpa),
            pressure_delta_milli_mpa: delta.round() as i64,
            claim_ceiling: "Matched-condition pressure comparison and evidence semantics only; no cell-performance, safety or manufacturing claim.".into(),
        });
    }
    reports.sort_by(|a, b| a.comparison_id.cmp(&b.comparison_id));
    Ok(AuditReport {
        campaign_id: manifest.campaign_id.clone(),
        observation_count: manifest.observations.len(),
        full_cell_count,
        symmetric_cell_count,
        half_cell_count,
        synthetic_outcome_count,
        physical_observation_count,
        right_censored_count,
        failed_or_invalid_count,
        incomplete_count,
        comparison_count: manifest.comparisons.len(),
        comparisons: reports,
        claim_ceiling: "Evidence, matched-condition, pressure-separation and mass-boundary semantics only. No electrochemical performance, life, safety, scale-up or commercial claim.".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn evidence(id: &str, ch: char) -> EvidenceRecord {
        EvidenceRecord {
            evidence_id: id.into(),
            source_digest: digest(ch),
            evidence_class: EvidenceClass::Synthetic,
            admission: Admission::AdmittedForEvaluation,
            claim_scope: "synthetic fixture".into(),
            rights_ref: "SYNTHETIC-ONLY-NO-PRODUCTION-DATA".into(),
            metrology_ref: None,
        }
    }

    fn design(id: &str, level: TestLevel) -> DesignDescriptor {
        DesignDescriptor {
            design_id: id.into(),
            revision: "r1".into(),
            technology_lane: TechnologyLane::SulfideLithiumMetal,
            chemistry_family: "Li-metal|NMC811".into(),
            electrolyte_family: "sulfide-composite".into(),
            cathode_family: "NMC811".into(),
            anode_configuration: "lithium-metal|porous-carbon-current-collector".into(),
            cell_format: "pouch".into(),
            test_level: level,
            areal_capacity_mah_cm2: 2.4,
            cathode_loading_mg_cm2: 20.0,
            electrolyte_thickness_um: 35.0,
            temperature_c: 25.0,
            current_density_ma_cm2: 0.24,
            voltage_window: "2.8-4.3V".into(),
            cycle_protocol: "0.1C-charge-discharge-r1".into(),
        }
    }

    fn mass() -> MassInventory {
        MassInventory {
            cathode_active_mg: 100.0,
            cathode_inactive_mg: 20.0,
            anode_active_mg: 10.0,
            anode_inactive_mg: 2.0,
            electrolyte_mg: 35.0,
            separator_or_support_mg: 8.0,
            current_collectors_and_tabs_mg: 18.0,
            casing_and_seals_mg: 27.0,
            declared_full_cell_mass_mg: 220.0,
            closure_tolerance_mg: 0.1,
            boundary_digest: digest('9'),
            declared_full_cell_volume_cm3: 1.2,
            volume_boundary_digest: digest('8'),
        }
    }

    fn observation(id: &str, d: DesignDescriptor, op: f64, eid: &str) -> CellObservation {
        let full = d.test_level == TestLevel::FullCell;
        CellObservation {
            observation_id: id.into(),
            cell_id: format!("CELL-{id}"),
            batch_id: "BATCH-001".into(),
            manufacturing_line: "PILOT-LINE-01".into(),
            process_revision: "PROCESS-R1".into(),
            design: d,
            pressure: PressureEvidence {
                fabrication_pressure_mpa: 100.0,
                operating_pressure_mpa: op,
                measurement_method: "calibrated load-cell fixture".into(),
                instrument_ref: "PRESSURE-MET-01".into(),
                uncertainty_mpa: 0.05,
            },
            mass_inventory: if full { Some(mass()) } else { None },
            cycles_observed: 100,
            outcome_status: OutcomeStatus::SyntheticOutcome,
            event_cycle: Some(90),
            capacity_retention_permille: Some(950),
            impedance_growth_permille: Some(120),
            evidence_id: eid.into(),
        }
    }

    fn axes() -> Vec<MatchedAxis> {
        MatchedAxis::REQUIRED.to_vec()
    }

    fn fixture() -> Manifest {
        Manifest {
            campaign_id: "ASSB-LOW-PRESSURE-001".into(),
            evidence: vec![
                evidence("E-HIGH", 'a'),
                evidence("E-LOW", 'b'),
                evidence("E-SYM", 'c'),
            ],
            observations: vec![
                observation(
                    "HIGH-P",
                    design("ASSB-SULFIDE-DESIGN-A", TestLevel::FullCell),
                    5.0,
                    "E-HIGH",
                ),
                observation(
                    "LOW-P",
                    design("ASSB-SULFIDE-DESIGN-A", TestLevel::FullCell),
                    0.3,
                    "E-LOW",
                ),
                observation(
                    "SYM",
                    design("ASSB-SULFIDE-SYM", TestLevel::SymmetricCell),
                    5.0,
                    "E-SYM",
                ),
            ],
            comparisons: vec![PressureComparison {
                comparison_id: "HIGH-VS-LOW-P".into(),
                baseline_observation_id: "HIGH-P".into(),
                candidate_observation_id: "LOW-P".into(),
                matched_axes: axes(),
            }],
        }
    }

    #[test]
    fn matched_pressure_fixture_audits_with_pressure_separate_from_fabrication() {
        let r = audit_manifest(&fixture()).expect("fixture valid");
        assert_eq!(r.full_cell_count, 2);
        assert_eq!(r.symmetric_cell_count, 1);
        assert_eq!(r.comparison_count, 1);
        assert_eq!(r.comparisons[0].baseline_operating_pressure_milli_mpa, 5000);
        assert_eq!(r.comparisons[0].candidate_operating_pressure_milli_mpa, 300);
        assert_eq!(
            r.comparisons[0].baseline_fabrication_pressure_milli_mpa,
            100000
        );
        assert_eq!(
            r.comparisons[0].candidate_fabrication_pressure_milli_mpa,
            100000
        );
        assert_eq!(r.comparisons[0].fabrication_pressure_delta_milli_mpa, 0);
        assert_eq!(
            r.comparisons[0].pressure_measurement_method,
            "calibrated load-cell fixture"
        );
        assert_eq!(r.comparisons[0].pressure_delta_milli_mpa, -4700);
    }

    #[test]
    fn full_cell_mass_must_close() {
        let mut m = fixture();
        m.observations[0]
            .mass_inventory
            .as_mut()
            .unwrap()
            .declared_full_cell_mass_mg = 250.0;
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn full_cell_requires_mass_inventory() {
        let mut m = fixture();
        m.observations[0].mass_inventory = None;
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn symmetric_cell_cannot_claim_full_cell_mass() {
        let mut m = fixture();
        m.observations[2].mass_inventory = Some(mass());
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn missing_pressure_measurement_method_is_rejected() {
        let mut m = fixture();
        m.observations[0].pressure.measurement_method.clear();
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn nonfinite_pressure_is_rejected() {
        let mut m = fixture();
        m.observations[0].pressure.operating_pressure_mpa = f64::NAN;
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn zero_fabrication_pressure_remains_a_declared_measurement() {
        let mut m = fixture();
        m.observations[0].pressure.fabrication_pressure_mpa = 0.0;
        m.observations[1].pressure.fabrication_pressure_mpa = 0.0;
        assert!(audit_manifest(&m).is_ok());
        // Zero fabrication pressure is allowed; it is a real declared measurement,
        // not a missing value. Operating pressure remains independently required.
    }

    #[test]
    fn fabrication_pressure_must_be_controlled_within_measurement_uncertainty() {
        let mut m = fixture();
        m.observations[1].pressure.fabrication_pressure_mpa = 101.0;
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn fabrication_pressure_within_combined_uncertainty_is_comparable() {
        let mut m = fixture();
        m.observations[1].pressure.fabrication_pressure_mpa = 100.08;
        assert!(audit_manifest(&m).is_ok());
    }

    #[test]
    fn different_pressure_measurement_methods_are_rejected() {
        let mut m = fixture();
        m.observations[1].pressure.measurement_method = "different fixture method".into();
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn process_batch_line_and_design_revision_must_match() {
        let mut m = fixture();
        m.observations[1].batch_id = "BATCH-002".into();
        assert!(audit_manifest(&m).is_err());
        let mut m = fixture();
        m.observations[1].process_revision = "PROCESS-R2".into();
        assert!(audit_manifest(&m).is_err());
        let mut m = fixture();
        m.observations[1].design.revision = "r2".into();
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn complete_cell_volume_must_be_positive_and_bound() {
        let mut m = fixture();
        m.observations[0]
            .mass_inventory
            .as_mut()
            .unwrap()
            .declared_full_cell_volume_cm3 = 0.0;
        assert!(audit_manifest(&m).is_err());
        let mut m = fixture();
        m.observations[0]
            .mass_inventory
            .as_mut()
            .unwrap()
            .volume_boundary_digest = "bad".into();
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn missing_cell_or_process_identity_is_rejected() {
        let mut m = fixture();
        m.observations[0].cell_id.clear();
        assert!(audit_manifest(&m).is_err());
        let mut m = fixture();
        m.observations[0].batch_id.clear();
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn comparison_must_change_operating_pressure() {
        let mut m = fixture();
        m.observations[1].pressure.operating_pressure_mpa = 5.0;
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn technology_lanes_cannot_be_pooled() {
        let mut m = fixture();
        m.observations[1].design.technology_lane = TechnologyLane::Halide;
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn symmetric_and_full_cell_levels_cannot_be_pooled() {
        let mut m = fixture();
        m.observations[1].design.test_level = TestLevel::SymmetricCell;
        m.observations[1].mass_inventory = None;
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn false_matched_axis_is_rejected() {
        let mut m = fixture();
        m.observations[1].design.temperature_c = 45.0;
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn missing_matched_axis_is_rejected() {
        let mut m = fixture();
        m.comparisons[0].matched_axes.pop();
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn right_censoring_remains_distinct() {
        let mut m = fixture();
        m.evidence[0].evidence_class = EvidenceClass::PhysicalMeasurement;
        m.evidence[0].metrology_ref = Some("CALIBRATED-METROLOGY-REF".into());
        m.observations[0].outcome_status = OutcomeStatus::RightCensored;
        m.observations[0].event_cycle = None;
        let r = audit_manifest(&m).expect("right censor valid");
        assert_eq!(r.right_censored_count, 1);
    }

    #[test]
    fn synthetic_result_cannot_be_promoted_to_physical_eol() {
        let mut m = fixture();
        m.observations[0].outcome_status = OutcomeStatus::ObservedEol;
        m.observations[0].event_cycle = Some(90);
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn missing_evidence_admission_is_rejected() {
        let mut m = fixture();
        m.evidence[0].admission = Admission::Unknown;
        assert!(audit_manifest(&m).is_err());
    }

    #[test]
    fn input_permutation_preserves_report() {
        let m = fixture();
        let expected = audit_manifest(&m).expect("fixture valid");
        let mut p = m;
        p.observations.reverse();
        p.evidence.reverse();
        p.comparisons.reverse();
        assert_eq!(
            expected,
            audit_manifest(&p).expect("permuted fixture valid")
        );
    }
}
