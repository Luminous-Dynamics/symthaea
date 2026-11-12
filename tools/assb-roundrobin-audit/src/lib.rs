#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceClass {
    PhysicalMeasurement,
    IndependentLaboratory,
    Synthetic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    Candidate,
    AdmittedForEvaluation,
    Blocked,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtocolMode {
    CommonProtocol,
    LabNative,
    ProtocolDeviation,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupComparability {
    KnownAnswerFixtureOnly,
    DescriptiveOnly,
    WithinLabOnly,
    CandidateCrossLabVarianceAnalysis,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceRecord {
    pub evidence_id: String,
    pub source_digest: String,
    pub evidence_class: EvidenceClass,
    pub admission: Admission,
    pub rights_ref: String,
    pub metrology_ref: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Laboratory {
    pub lab_id: String,
    pub blinded_lab_code: String,
    pub authority_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fixture {
    pub fixture_id: String,
    pub lab_id: String,
    pub revision: String,
    pub measurement_method: String,
    pub instrument_ref: String,
    pub calibration_evidence_id: String,
    pub pressure_range_min_mpa: f64,
    pub pressure_range_max_mpa: f64,
    pub pressure_uncertainty_mpa: f64,
    pub pressure_homogeneity_percent: f64,
    pub setpoint_accuracy_mape_percent: f64,
    pub repeatability_cv_percent: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CellIdentity {
    pub cell_id: String,
    pub blinded_sample_code: String,
    pub design_id: String,
    pub design_revision: String,
    pub test_level: String,
    pub material_lot_id: String,
    pub process_revision: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub observation_id: String,
    pub cell_id: String,
    pub lab_id: String,
    pub fixture_id: String,
    pub replicate_group_id: String,
    pub protocol_mode: ProtocolMode,
    pub protocol_revision: String,
    pub condition_digest: String,
    pub pressure_setpoint_mpa: f64,
    pub measured_operating_pressure_mpa: f64,
    pub exposure_cycles: u32,
    pub outcome_status: OutcomeStatus,
    pub event_cycle: Option<u32>,
    pub raw_data_digest: String,
    pub evidence_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Manifest {
    pub campaign_id: String,
    pub evidence: Vec<EvidenceRecord>,
    pub laboratories: Vec<Laboratory>,
    pub fixtures: Vec<Fixture>,
    pub cells: Vec<CellIdentity>,
    pub observations: Vec<Observation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupReport {
    pub replicate_group_id: String,
    pub protocol_mode: ProtocolMode,
    pub observation_count: usize,
    pub laboratory_count: usize,
    pub minimum_independent_cells_per_lab: usize,
    pub comparability: GroupComparability,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditReport {
    pub campaign_id: String,
    pub laboratory_count: usize,
    pub fixture_count: usize,
    pub cell_count: usize,
    pub observation_count: usize,
    pub physical_observation_count: usize,
    pub synthetic_observation_count: usize,
    pub right_censored_count: usize,
    pub failed_invalid_incomplete_count: usize,
    pub cross_lab_candidate_group_count: usize,
    pub groups: Vec<GroupReport>,
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

fn nonempty(value: &str, name: &str) -> Result<(), AuditError> {
    if value.trim().is_empty() {
        return Err(AuditError(format!("{name} must not be empty")));
    }
    Ok(())
}

fn digest(value: &str, name: &str) -> Result<(), AuditError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(AuditError(format!("{name} must be a lowercase SHA-256 digest")));
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

fn unique<'a>(
    values: impl IntoIterator<Item = &'a str>,
    name: &str,
) -> Result<(), AuditError> {
    let mut found = BTreeSet::new();
    for value in values {
        nonempty(value, name)?;
        if !found.insert(value) {
            return Err(AuditError(format!("duplicate {name}: {value}")));
        }
    }
    Ok(())
}

fn evidence_map(
    records: &[EvidenceRecord],
) -> Result<BTreeMap<&str, &EvidenceRecord>, AuditError> {
    unique(records.iter().map(|r| r.evidence_id.as_str()), "evidence ID")?;
    if records.is_empty() {
        return Err(AuditError("evidence registry must not be empty".into()));
    }
    let mut map = BTreeMap::new();
    for record in records {
        digest(&record.source_digest, "evidence source digest")?;
        nonempty(&record.rights_ref, "data-rights reference")?;
        if matches!(
            record.evidence_class,
            EvidenceClass::PhysicalMeasurement | EvidenceClass::IndependentLaboratory
        ) && record.metrology_ref.as_deref().is_none_or(str::is_empty)
        {
            return Err(AuditError(format!(
                "measured evidence {} requires a metrology reference",
                record.evidence_id
            )));
        }
        map.insert(record.evidence_id.as_str(), record);
    }
    Ok(map)
}

fn admitted<'a>(
    evidence: &'a BTreeMap<&'a str, &'a EvidenceRecord>,
    id: &str,
    owner: &str,
) -> Result<&'a EvidenceRecord, AuditError> {
    let record = evidence
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

fn validate_outcome(
    observation: &Observation,
    evidence: &EvidenceRecord,
) -> Result<(), AuditError> {
    match observation.outcome_status {
        OutcomeStatus::ObservedEol => {
            if evidence.evidence_class == EvidenceClass::Synthetic {
                return Err(AuditError("synthetic evidence cannot report physical EOL".into()));
            }
            let cycle = observation
                .event_cycle
                .ok_or_else(|| AuditError("observed EOL requires event cycle".into()))?;
            if cycle == 0 || cycle > observation.exposure_cycles {
                return Err(AuditError("EOL event cycle outside observed exposure".into()));
            }
        }
        OutcomeStatus::RightCensored => {
            if evidence.evidence_class == EvidenceClass::Synthetic {
                return Err(AuditError("synthetic evidence cannot report physical censoring".into()));
            }
            if observation.exposure_cycles == 0 || observation.event_cycle.is_some() {
                return Err(AuditError(
                    "right-censored exposure requires observed cycles and no EOL event".into(),
                ));
            }
        }
        OutcomeStatus::Failed | OutcomeStatus::Invalid | OutcomeStatus::Incomplete => {
            if evidence.evidence_class == EvidenceClass::Synthetic {
                return Err(AuditError(
                    "synthetic evidence cannot be promoted to a physical failure/invalid record"
                        .into(),
                ));
            }
            if observation.event_cycle.is_some() {
                return Err(AuditError(
                    "failed/invalid/incomplete record cannot claim observed EOL".into(),
                ));
            }
        }
        OutcomeStatus::SyntheticOutcome => {
            if evidence.evidence_class != EvidenceClass::Synthetic {
                return Err(AuditError(
                    "synthetic outcome requires synthetic evidence class".into(),
                ));
            }
        }
    }
    Ok(())
}

fn group_comparability(
    mode: ProtocolMode,
    rows: &[&Observation],
    cells: &BTreeMap<&str, &CellIdentity>,
    evidence: &BTreeMap<&str, &EvidenceRecord>,
) -> Result<GroupReport, AuditError> {
    let group_id = rows[0].replicate_group_id.as_str();
    let labs: BTreeSet<_> = rows.iter().map(|o| o.lab_id.as_str()).collect();
    let mut per_lab = BTreeMap::<&str, BTreeSet<&str>>::new();
    let mut per_group_cells = BTreeSet::new();

    for row in rows {
        let cell = cells.get(row.cell_id.as_str()).copied().ok_or_else(|| {
            AuditError(format!("group {group_id} references missing cell {}", row.cell_id))
        })?;
        if !per_group_cells.insert(row.cell_id.as_str()) {
            return Err(AuditError(format!(
                "group {group_id} repeats physical cell {}; repeated measurement is not an independent replicate",
                row.cell_id
            )));
        }
        per_lab
            .entry(row.lab_id.as_str())
            .or_default()
            .insert(cell.cell_id.as_str());
        if row.protocol_mode != mode {
            return Err(AuditError(format!(
                "group {group_id} mixes protocol modes"
            )));
        }
    }

    if mode == ProtocolMode::CommonProtocol {
        let first = rows[0];
        let first_cell = cells[first.cell_id.as_str()];
        for row in rows.iter().skip(1) {
            let cell = cells[row.cell_id.as_str()];
            if row.protocol_revision != first.protocol_revision
                || row.condition_digest != first.condition_digest
                || row.pressure_setpoint_mpa.to_bits() != first.pressure_setpoint_mpa.to_bits()
                || cell.design_id != first_cell.design_id
                || cell.design_revision != first_cell.design_revision
                || cell.test_level != first_cell.test_level
                || cell.material_lot_id != first_cell.material_lot_id
                || cell.process_revision != first_cell.process_revision
            {
                return Err(AuditError(format!(
                    "COMMON_PROTOCOL group {group_id} mixes design, lot, process, protocol, condition or setpoint"
                )));
            }
        }
    } else {
        let first_lab = rows[0].lab_id.as_str();
        if rows.iter().any(|row| row.lab_id != first_lab) {
            return Err(AuditError(format!(
                "LAB_NATIVE and PROTOCOL_DEVIATION groups must be scoped to one lab; do not pool across labs"
            )));
        }
    }

    let minimum = per_lab.values().map(BTreeSet::len).min().unwrap_or(0);
    let synthetic_count = rows
        .iter()
        .filter(|row| {
            evidence
                .get(row.evidence_id.as_str())
                .is_some_and(|e| e.evidence_class == EvidenceClass::Synthetic)
        })
        .count();
    if synthetic_count != 0 && synthetic_count != rows.len() {
        return Err(AuditError(format!(
            "replicate group {group_id} mixes synthetic fixtures with physical observations"
        )));
    }
    let synthetic = synthetic_count == rows.len();
    let comparability = if synthetic {
        GroupComparability::KnownAnswerFixtureOnly
    } else {
        match mode {
            ProtocolMode::LabNative => GroupComparability::WithinLabOnly,
            ProtocolMode::ProtocolDeviation => GroupComparability::DescriptiveOnly,
            ProtocolMode::CommonProtocol if labs.len() >= 3 && minimum >= 3 => {
                GroupComparability::CandidateCrossLabVarianceAnalysis
            }
            ProtocolMode::CommonProtocol => GroupComparability::DescriptiveOnly,
        }
    };

    Ok(GroupReport {
        replicate_group_id: group_id.to_string(),
        protocol_mode: mode,
        observation_count: rows.len(),
        laboratory_count: labs.len(),
        minimum_independent_cells_per_lab: minimum,
        comparability,
    })
}

pub fn audit_manifest(manifest: &Manifest) -> Result<AuditReport, AuditError> {
    nonempty(&manifest.campaign_id, "campaign ID")?;
    if manifest.observations.is_empty() {
        return Err(AuditError("manifest requires observations".into()));
    }
    let evidence = evidence_map(&manifest.evidence)?;
    unique(manifest.laboratories.iter().map(|l| l.lab_id.as_str()), "lab ID")?;
    unique(
        manifest.laboratories.iter().map(|l| l.blinded_lab_code.as_str()),
        "blinded lab code",
    )?;
    unique(
        manifest.fixtures.iter().map(|f| f.fixture_id.as_str()),
        "fixture ID",
    )?;
    unique(manifest.cells.iter().map(|c| c.cell_id.as_str()), "cell ID")?;
    unique(
        manifest.cells.iter().map(|c| c.blinded_sample_code.as_str()),
        "blinded sample code",
    )?;
    unique(
        manifest.observations.iter().map(|o| o.observation_id.as_str()),
        "observation ID",
    )?;

    let lab_map: BTreeMap<_, _> = manifest
        .laboratories
        .iter()
        .map(|lab| (lab.lab_id.as_str(), lab))
        .collect();
    if lab_map.is_empty() {
        return Err(AuditError("at least one laboratory is required".into()));
    }
    for lab in &manifest.laboratories {
        nonempty(&lab.authority_ref, "laboratory authority reference")?;
    }

    let mut fixture_map = BTreeMap::new();
    for fixture in &manifest.fixtures {
        nonempty(&fixture.revision, "fixture revision")?;
        nonempty(&fixture.measurement_method, "fixture measurement method")?;
        nonempty(&fixture.instrument_ref, "fixture instrument reference")?;
        finite_nonnegative(fixture.pressure_range_min_mpa, "fixture pressure minimum")?;
        finite_positive(fixture.pressure_range_max_mpa, "fixture pressure maximum")?;
        if fixture.pressure_range_min_mpa >= fixture.pressure_range_max_mpa {
            return Err(AuditError("fixture pressure range must have max > min".into()));
        }
        for (value, name) in [
            (fixture.pressure_uncertainty_mpa, "fixture pressure uncertainty"),
            (fixture.pressure_homogeneity_percent, "pressure homogeneity"),
            (fixture.setpoint_accuracy_mape_percent, "setpoint accuracy MAPE"),
            (fixture.repeatability_cv_percent, "pressure repeatability CV"),
        ] {
            finite_nonnegative(value, name)?;
        }
        if !lab_map.contains_key(fixture.lab_id.as_str()) {
            return Err(AuditError(format!(
                "fixture {} references unknown laboratory {}",
                fixture.fixture_id, fixture.lab_id
            )));
        }
        let calibration = admitted(
            &evidence,
            &fixture.calibration_evidence_id,
            &format!("fixture {}", fixture.fixture_id),
        )?;
        if calibration.evidence_class == EvidenceClass::Synthetic
            && !fixture.calibration_evidence_id.starts_with("SYNTHETIC")
        {
            return Err(AuditError(
                "synthetic fixture calibration evidence must be visibly synthetic".into(),
            ));
        }
        fixture_map.insert(fixture.fixture_id.as_str(), fixture);
    }

    let cell_map: BTreeMap<_, _> = manifest
        .cells
        .iter()
        .map(|cell| (cell.cell_id.as_str(), cell))
        .collect();
    for cell in &manifest.cells {
        for (value, label) in [
            (cell.blinded_sample_code.as_str(), "blinded sample code"),
            (cell.design_id.as_str(), "design ID"),
            (cell.design_revision.as_str(), "design revision"),
            (cell.test_level.as_str(), "test level"),
            (cell.material_lot_id.as_str(), "material lot ID"),
            (cell.process_revision.as_str(), "process revision"),
        ] {
            nonempty(value, label)?;
        }
    }

    let mut group_rows = BTreeMap::<&str, Vec<&Observation>>::new();
    let mut physical_observation_count = 0;
    let mut synthetic_observation_count = 0;
    let mut right_censored_count = 0;
    let mut failed_invalid_incomplete_count = 0;

    for observation in &manifest.observations {
        nonempty(&observation.replicate_group_id, "replicate group ID")?;
        nonempty(&observation.protocol_revision, "protocol revision")?;
        digest(&observation.condition_digest, "test condition digest")?;
        digest(&observation.raw_data_digest, "raw data digest")?;
        finite_nonnegative(observation.pressure_setpoint_mpa, "pressure setpoint")?;
        finite_nonnegative(
            observation.measured_operating_pressure_mpa,
            "measured operating pressure",
        )?;

        if !lab_map.contains_key(observation.lab_id.as_str()) {
            return Err(AuditError(format!(
                "observation {} references unknown laboratory",
                observation.observation_id
            )));
        }
        let fixture = fixture_map
            .get(observation.fixture_id.as_str())
            .copied()
            .ok_or_else(|| {
                AuditError(format!(
                    "observation {} references unknown fixture",
                    observation.observation_id
                ))
            })?;
        if fixture.lab_id != observation.lab_id {
            return Err(AuditError(format!(
                "observation {} uses a fixture owned by another laboratory",
                observation.observation_id
            )));
        }
        if observation.pressure_setpoint_mpa < fixture.pressure_range_min_mpa
            || observation.pressure_setpoint_mpa > fixture.pressure_range_max_mpa
            || observation.measured_operating_pressure_mpa < fixture.pressure_range_min_mpa
            || observation.measured_operating_pressure_mpa > fixture.pressure_range_max_mpa
        {
            return Err(AuditError(format!(
                "observation {} pressure is outside fixture {} qualified range",
                observation.observation_id, fixture.fixture_id
            )));
        }
        if !cell_map.contains_key(observation.cell_id.as_str()) {
            return Err(AuditError(format!(
                "observation {} references unknown cell",
                observation.observation_id
            )));
        }
        let source = admitted(
            &evidence,
            &observation.evidence_id,
            &format!("observation {}", observation.observation_id),
        )?;
        validate_outcome(observation, source)?;
        match observation.outcome_status {
            OutcomeStatus::SyntheticOutcome => synthetic_observation_count += 1,
            OutcomeStatus::ObservedEol => physical_observation_count += 1,
            OutcomeStatus::RightCensored => {
                physical_observation_count += 1;
                right_censored_count += 1;
            }
            OutcomeStatus::Failed | OutcomeStatus::Invalid | OutcomeStatus::Incomplete => {
                physical_observation_count += 1;
                failed_invalid_incomplete_count += 1;
            }
        }
        group_rows
            .entry(observation.replicate_group_id.as_str())
            .or_default()
            .push(observation);
    }

    let mut groups = Vec::new();
    let mut cross_lab_candidate_group_count = 0;
    for rows in group_rows.values() {
        let report = group_comparability(rows[0].protocol_mode, rows, &cell_map, &evidence)?;
        if report.comparability == GroupComparability::CandidateCrossLabVarianceAnalysis {
            cross_lab_candidate_group_count += 1;
        }
        groups.push(report);
    }
    groups.sort_by(|a, b| a.replicate_group_id.cmp(&b.replicate_group_id));

    Ok(AuditReport {
        campaign_id: manifest.campaign_id.clone(),
        laboratory_count: manifest.laboratories.len(),
        fixture_count: manifest.fixtures.len(),
        cell_count: manifest.cells.len(),
        observation_count: manifest.observations.len(),
        physical_observation_count,
        synthetic_observation_count,
        right_censored_count,
        failed_invalid_incomplete_count,
        cross_lab_candidate_group_count,
        groups,
        claim_ceiling: "Identity, provenance, fixture-range, replicate and grouping semantics only. No statistical pooling, physical reproducibility, electrochemical performance, safety or commercial claim.".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn ev(id: &str, ch: char, class: EvidenceClass) -> EvidenceRecord {
        EvidenceRecord {
            evidence_id: id.into(),
            source_digest: d(ch),
            evidence_class: class,
            admission: Admission::AdmittedForEvaluation,
            rights_ref: format!("RIGHTS-{id}"),
            metrology_ref: (class != EvidenceClass::Synthetic).then(|| format!("MET-{id}")),
        }
    }

    fn lab(id: &str, code: &str) -> Laboratory {
        Laboratory {
            lab_id: id.into(),
            blinded_lab_code: code.into(),
            authority_ref: format!("AUTH-{id}"),
        }
    }

    fn fixture(id: &str, lab_id: &str, cal: &str) -> Fixture {
        Fixture {
            fixture_id: id.into(),
            lab_id: lab_id.into(),
            revision: "fixture-r1".into(),
            measurement_method: "calibrated-load-cell".into(),
            instrument_ref: format!("LOADCELL-{id}"),
            calibration_evidence_id: cal.into(),
            pressure_range_min_mpa: 0.0,
            pressure_range_max_mpa: 20.0,
            pressure_uncertainty_mpa: 0.05,
            pressure_homogeneity_percent: 3.0,
            setpoint_accuracy_mape_percent: 2.5,
            repeatability_cv_percent: 2.0,
        }
    }

    fn cell(id: &str, code: &str) -> CellIdentity {
        CellIdentity {
            cell_id: id.into(),
            blinded_sample_code: code.into(),
            design_id: "ASSB-REF-A".into(),
            design_revision: "r1".into(),
            test_level: "full_cell".into(),
            material_lot_id: "REF-LOT-01".into(),
            process_revision: "PROCESS-r1".into(),
        }
    }

    fn observation(
        id: &str,
        cell_id: &str,
        lab_id: &str,
        fixture_id: &str,
        evidence_id: &str,
    ) -> Observation {
        Observation {
            observation_id: id.into(),
            cell_id: cell_id.into(),
            lab_id: lab_id.into(),
            fixture_id: fixture_id.into(),
            replicate_group_id: "COMMON-ROUND-001".into(),
            protocol_mode: ProtocolMode::CommonProtocol,
            protocol_revision: "PROTOCOL-r1".into(),
            condition_digest: d('f'),
            pressure_setpoint_mpa: 0.3,
            measured_operating_pressure_mpa: 0.3,
            exposure_cycles: 100,
            outcome_status: OutcomeStatus::SyntheticOutcome,
            event_cycle: Some(90),
            raw_data_digest: d('a'),
            evidence_id: evidence_id.into(),
        }
    }

    fn fixture_manifest() -> Manifest {
        let mut evidence = vec![
            ev("SYNTHETIC-CAL-A", '1', EvidenceClass::Synthetic),
            ev("SYNTHETIC-CAL-B", '2', EvidenceClass::Synthetic),
            ev("SYNTHETIC-CAL-C", '3', EvidenceClass::Synthetic),
        ];
        let mut labs = Vec::new();
        let mut fixtures = Vec::new();
        let mut cells = Vec::new();
        let mut observations = Vec::new();
        for (lab_id, code, fixture_id, cal) in [
            ("LAB-A", "BLIND-LAB-01", "FIX-A", "SYNTHETIC-CAL-A"),
            ("LAB-B", "BLIND-LAB-02", "FIX-B", "SYNTHETIC-CAL-B"),
            ("LAB-C", "BLIND-LAB-03", "FIX-C", "SYNTHETIC-CAL-C"),
        ] {
            labs.push(lab(lab_id, code));
            fixtures.push(fixture(fixture_id, lab_id, cal));
            for n in 1..=3 {
                let cell_id = format!("{lab_id}-CELL-{n}");
                let obs_id = format!("{lab_id}-OBS-{n}");
                let evidence_id = format!("SYNTHETIC-EVIDENCE-{}-{n}", &lab_id[4..]);
                evidence.push(ev(&evidence_id, char::from_digit((n + 3) as u32, 10).unwrap(), EvidenceClass::Synthetic));
                cells.push(cell(&cell_id, &format!("{lab_id}-BLIND-{n}")));
                observations.push(observation(
                    &obs_id,
                    &cell_id,
                    lab_id,
                    fixture_id,
                    &evidence_id,
                ));
            }
        }
        Manifest {
            campaign_id: "ASSB-ROUNDROBIN-001".into(),
            evidence,
            laboratories: labs,
            fixtures,
            cells,
            observations,
        }
    }

    #[test]
    fn synthetic_three_lab_triplicate_is_fixture_only_not_physical_reproducibility() {
        let report = audit_manifest(&fixture_manifest()).expect("synthetic fixture valid");
        assert_eq!(report.laboratory_count, 3);
        assert_eq!(report.cell_count, 9);
        assert_eq!(report.observation_count, 9);
        assert_eq!(
            report.groups[0].comparability,
            GroupComparability::KnownAnswerFixtureOnly
        );
        assert_eq!(report.cross_lab_candidate_group_count, 0);
    }

    #[test]
    fn common_protocol_physical_three_lab_triplicate_is_candidate_for_variance_analysis() {
        let mut manifest = fixture_manifest();
        for evidence in &mut manifest.evidence {
            if evidence.evidence_class == EvidenceClass::Synthetic
                && !evidence.evidence_id.starts_with("SYNTHETIC-CAL")
            {
                evidence.evidence_class = EvidenceClass::PhysicalMeasurement;
                evidence.metrology_ref = Some(format!("MET-{}", evidence.evidence_id));
            }
            if evidence.evidence_id.starts_with("SYNTHETIC-CAL") {
                evidence.evidence_class = EvidenceClass::IndependentLaboratory;
                evidence.metrology_ref = Some(format!("CAL-{}", evidence.evidence_id));
            }
        }
        for observation in &mut manifest.observations {
            observation.outcome_status = OutcomeStatus::RightCensored;
            observation.event_cycle = None;
        }
        let report = audit_manifest(&manifest).expect("measured round robin valid");
        assert_eq!(report.cross_lab_candidate_group_count, 1);
        assert_eq!(
            report.groups[0].comparability,
            GroupComparability::CandidateCrossLabVarianceAnalysis
        );
    }

    #[test]
    fn repeated_physical_cell_cannot_count_as_independent_replicate() {
        let mut manifest = fixture_manifest();
        manifest.observations[1].cell_id = manifest.observations[0].cell_id.clone();
        assert!(audit_manifest(&manifest).is_err());
    }

    #[test]
    fn fixture_must_belong_to_observation_laboratory() {
        let mut manifest = fixture_manifest();
        manifest.observations[0].lab_id = "LAB-B".into();
        assert!(audit_manifest(&manifest).is_err());
    }

    #[test]
    fn common_protocol_cannot_hide_condition_or_protocol_mismatch() {
        let mut manifest = fixture_manifest();
        manifest.observations[0].protocol_revision = "PROTOCOL-r2".into();
        assert!(audit_manifest(&manifest).is_err());
        let mut manifest = fixture_manifest();
        manifest.observations[0].condition_digest = d('e');
        assert!(audit_manifest(&manifest).is_err());
    }

    #[test]
    fn cross_lab_native_protocol_group_is_rejected_from_pooling() {
        let mut manifest = fixture_manifest();
        manifest.observations = vec![
            manifest.observations[0].clone(),
            manifest.observations[3].clone(),
            manifest.observations[6].clone(),
        ];
        for observation in &mut manifest.observations {
            observation.protocol_mode = ProtocolMode::LabNative;
        }
        assert!(audit_manifest(&manifest).is_err());
    }

    #[test]
    fn lab_native_protocol_is_scoped_to_single_lab() {
        let mut manifest = fixture_manifest();
        manifest.observations.retain(|o| o.lab_id == "LAB-A");
        manifest.observations[0].protocol_mode = ProtocolMode::LabNative;
        manifest.observations[1].protocol_mode = ProtocolMode::LabNative;
        manifest.observations[2].protocol_mode = ProtocolMode::LabNative;
        manifest.observations[1].protocol_revision = "LOCAL-r2".into();
        let report = audit_manifest(&manifest).expect("lab-native arm is descriptive");
        assert_eq!(
            report.groups[0].comparability,
            GroupComparability::KnownAnswerFixtureOnly
        );
    }

    #[test]
    fn right_censoring_remains_distinct() {
        let mut manifest = fixture_manifest();
        manifest.laboratories.retain(|lab| lab.lab_id == "LAB-A");
        manifest.fixtures.retain(|fixture| fixture.fixture_id == "FIX-A");
        manifest.cells.retain(|cell| cell.cell_id == "LAB-A-CELL-1");
        manifest
            .observations
            .retain(|observation| observation.observation_id == "LAB-A-OBS-1");
        manifest.evidence.retain(|record| {
            record.evidence_id == "SYNTHETIC-CAL-A"
                || record.evidence_id == "SYNTHETIC-EVIDENCE-A-1"
        });
        for record in &mut manifest.evidence {
            record.evidence_class = if record.evidence_id == "SYNTHETIC-CAL-A" {
                EvidenceClass::IndependentLaboratory
            } else {
                EvidenceClass::PhysicalMeasurement
            };
            record.metrology_ref = Some(format!("MET-{}", record.evidence_id));
        }
        manifest.observations[0].outcome_status = OutcomeStatus::RightCensored;
        manifest.observations[0].event_cycle = None;
        let report = audit_manifest(&manifest).expect("physical right-censored record valid");
        assert_eq!(report.right_censored_count, 1);
    }

    #[test]
    fn physical_eol_requires_physical_evidence() {
        let mut manifest = fixture_manifest();
        manifest.observations[0].outcome_status = OutcomeStatus::ObservedEol;
        manifest.observations[0].event_cycle = Some(90);
        assert!(audit_manifest(&manifest).is_err());
    }

    #[test]
    fn fixture_pressure_cannot_exceed_qualified_range() {
        let mut manifest = fixture_manifest();
        manifest.observations[0].measured_operating_pressure_mpa = 30.0;
        assert!(audit_manifest(&manifest).is_err());
    }

    #[test]
    fn missing_calibration_record_is_rejected() {
        let mut manifest = fixture_manifest();
        manifest.fixtures[0].calibration_evidence_id = "MISSING".into();
        assert!(audit_manifest(&manifest).is_err());
    }

    #[test]
    fn blinded_codes_and_raw_data_digests_are_required() {
        let mut manifest = fixture_manifest();
        manifest.cells[0].blinded_sample_code.clear();
        assert!(audit_manifest(&manifest).is_err());
        let mut manifest = fixture_manifest();
        manifest.observations[0].raw_data_digest = "missing".into();
        assert!(audit_manifest(&manifest).is_err());
    }

    #[test]
    fn input_permutation_preserves_report() {
        let manifest = fixture_manifest();
        let expected = audit_manifest(&manifest).expect("fixture valid");
        let mut permuted = manifest;
        permuted.evidence.reverse();
        permuted.cells.reverse();
        permuted.observations.reverse();
        permuted.fixtures.reverse();
        permuted.laboratories.reverse();
        assert_eq!(
            expected,
            audit_manifest(&permuted).expect("permuted fixture valid")
        );
    }
}
