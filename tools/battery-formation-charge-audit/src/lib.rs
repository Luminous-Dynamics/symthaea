#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceClass {
    SyntheticFixture,
    Assumption,
    VendorClaim,
    LiteratureReported,
    Simulation,
    Surrogate,
    Calculation,
    PhysicalMeasurement,
    IndependentLaboratory,
}

impl EvidenceClass {
    fn is_physical(self) -> bool {
        matches!(
            self,
            Self::PhysicalMeasurement | Self::IndependentLaboratory
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceAdmission {
    Candidate,
    AdmittedForResearch,
    AdmittedForEvaluation,
    Blocked,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub evidence_id: String,
    pub source_digest: String,
    pub evidence_class: EvidenceClass,
    pub admission: EvidenceAdmission,
    pub claim_scope: String,
    pub validity_domain: String,
    pub metrology_ref: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BlockAxis {
    Lot,
    Line,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HoldoutAxis {
    Lot,
    Line,
    ProtocolPair,
    Design,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataSplit {
    Train,
    Validation,
    Test,
    Prospective,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolArm {
    pub arm_id: String,
    pub design_id: String,
    pub design_revision: String,
    pub formation_profile_id: String,
    pub formation_revision: String,
    pub charge_policy_id: String,
    pub charge_policy_revision: String,
    pub protocol_evidence_id: String,
    pub is_reference: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifetimeStatus {
    ObservedEol,
    RightCensored,
    CompetingFailure,
    Invalid,
    Unknown,
    SyntheticOutcome,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellObservation {
    pub cell_id: String,
    pub arm_id: String,
    pub design_id: String,
    pub design_revision: String,
    pub lot_id: String,
    pub lot_lineage_digest: String,
    pub line_id: String,
    pub line_identity_digest: String,
    pub split: DataSplit,
    pub feature_cutoff_cycle: u32,
    pub cycles_observed: u32,
    pub status: LifetimeStatus,
    pub event_cycle: Option<u32>,
    pub failure_mode: Option<String>,
    pub outcome_evidence_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Campaign {
    pub campaign_id: String,
    pub mission_id: String,
    pub blocking_axis: BlockAxis,
    pub holdout_axis: HoldoutAxis,
    pub maximum_feature_cutoff_cycle: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub campaign: Campaign,
    pub arms: Vec<ProtocolArm>,
    pub cells: Vec<CellObservation>,
    pub evidence: Vec<EvidenceRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditReport {
    pub campaign_id: String,
    pub arm_count: usize,
    pub formation_profile_count: usize,
    pub charge_policy_count: usize,
    pub cell_count: usize,
    pub lot_count: usize,
    pub line_count: usize,
    pub block_count: usize,
    pub holdout_group_count: usize,
    pub observed_eol_count: usize,
    pub right_censored_count: usize,
    pub competing_failure_count: usize,
    pub invalid_or_unknown_count: usize,
    pub synthetic_outcome_count: usize,
    pub physical_lifetime_outcome_count: usize,
    pub maximum_observed_feature_cutoff_cycle: u32,
    pub reference_arm_id: String,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditError(pub String);

impl std::fmt::Display for AuditError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for AuditError {}

fn require_nonempty(value: &str, label: &str) -> Result<(), AuditError> {
    if value.trim().is_empty() {
        return Err(AuditError(format!("{label} must not be empty")));
    }
    Ok(())
}

fn validate_unique<'a>(
    values: impl IntoIterator<Item = &'a str>,
    label: &str,
) -> Result<(), AuditError> {
    let mut seen = BTreeSet::new();
    for value in values {
        require_nonempty(value, label)?;
        if !seen.insert(value) {
            return Err(AuditError(format!("duplicate {label}: {value}")));
        }
    }
    Ok(())
}

fn validate_digest(value: &str, label: &str) -> Result<(), AuditError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(AuditError(format!(
            "{label} must be a lowercase 64-character SHA-256 digest"
        )));
    }
    Ok(())
}

fn evidence_registry(
    evidence: &[EvidenceRecord],
) -> Result<BTreeMap<&str, &EvidenceRecord>, AuditError> {
    validate_unique(
        evidence.iter().map(|record| record.evidence_id.as_str()),
        "evidence ID",
    )?;
    if evidence.is_empty() {
        return Err(AuditError("evidence registry must not be empty".into()));
    }

    let mut registry = BTreeMap::new();
    for record in evidence {
        validate_digest(&record.source_digest, "evidence source digest")?;
        require_nonempty(&record.claim_scope, "evidence claim scope")?;
        require_nonempty(&record.validity_domain, "evidence validity domain")?;
        if record.evidence_class.is_physical()
            && record
                .metrology_ref
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
        {
            return Err(AuditError(format!(
                "physical evidence {} requires a metrology reference",
                record.evidence_id
            )));
        }
        registry.insert(record.evidence_id.as_str(), record);
    }
    Ok(registry)
}

fn require_admitted<'a>(
    registry: &'a BTreeMap<&str, &EvidenceRecord>,
    evidence_id: &str,
    owner: &str,
) -> Result<&'a EvidenceRecord, AuditError> {
    let record = registry.get(evidence_id).copied().ok_or_else(|| {
        AuditError(format!(
            "{owner} references unknown evidence ID {evidence_id}"
        ))
    })?;
    if record.admission != EvidenceAdmission::AdmittedForEvaluation {
        return Err(AuditError(format!(
            "{owner} references evidence {evidence_id} that is not admitted for evaluation"
        )));
    }
    Ok(record)
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum HoldoutGroup {
    Lot(String),
    Line(String),
    ProtocolPair {
        formation_profile_id: String,
        formation_revision: String,
        charge_policy_id: String,
        charge_policy_revision: String,
    },
    Design {
        design_id: String,
        design_revision: String,
    },
}

fn holdout_group(cell: &CellObservation, arm: &ProtocolArm, axis: HoldoutAxis) -> HoldoutGroup {
    match axis {
        HoldoutAxis::Lot => HoldoutGroup::Lot(cell.lot_id.clone()),
        HoldoutAxis::Line => HoldoutGroup::Line(cell.line_id.clone()),
        HoldoutAxis::ProtocolPair => HoldoutGroup::ProtocolPair {
            formation_profile_id: arm.formation_profile_id.clone(),
            formation_revision: arm.formation_revision.clone(),
            charge_policy_id: arm.charge_policy_id.clone(),
            charge_policy_revision: arm.charge_policy_revision.clone(),
        },
        HoldoutAxis::Design => HoldoutGroup::Design {
            design_id: cell.design_id.clone(),
            design_revision: cell.design_revision.clone(),
        },
    }
}

fn validate_lifetime_status(
    cell: &CellObservation,
    evidence: &EvidenceRecord,
) -> Result<(), AuditError> {
    match cell.status {
        LifetimeStatus::ObservedEol => {
            if !evidence.evidence_class.is_physical() {
                return Err(AuditError(format!(
                    "observed EOL for cell {} requires physical evidence",
                    cell.cell_id
                )));
            }
            let event_cycle = cell.event_cycle.ok_or_else(|| {
                AuditError(format!(
                    "observed EOL for cell {} requires an event cycle",
                    cell.cell_id
                ))
            })?;
            if event_cycle == 0 || event_cycle > cell.cycles_observed {
                return Err(AuditError(format!(
                    "observed EOL event for cell {} is outside observed cycles",
                    cell.cell_id
                )));
            }
            if cell.failure_mode.is_some() {
                return Err(AuditError(format!(
                    "observed EOL for cell {} cannot carry a competing failure mode",
                    cell.cell_id
                )));
            }
        }
        LifetimeStatus::RightCensored => {
            if !evidence.evidence_class.is_physical() {
                return Err(AuditError(format!(
                    "right-censored life for cell {} requires physical evidence",
                    cell.cell_id
                )));
            }
            if cell.cycles_observed == 0 || cell.event_cycle.is_some() {
                return Err(AuditError(format!(
                    "right-censored cell {} must have observed exposure and no EOL event",
                    cell.cell_id
                )));
            }
            if cell.failure_mode.is_some() {
                return Err(AuditError(format!(
                    "right-censored cell {} cannot carry a competing failure mode",
                    cell.cell_id
                )));
            }
        }
        LifetimeStatus::CompetingFailure => {
            if !evidence.evidence_class.is_physical() {
                return Err(AuditError(format!(
                    "competing failure for cell {} requires physical evidence",
                    cell.cell_id
                )));
            }
            let event_cycle = cell.event_cycle.ok_or_else(|| {
                AuditError(format!(
                    "competing failure for cell {} requires its event cycle",
                    cell.cell_id
                ))
            })?;
            if event_cycle == 0 || event_cycle > cell.cycles_observed {
                return Err(AuditError(format!(
                    "competing failure event for cell {} is outside observed cycles",
                    cell.cell_id
                )));
            }
            if cell
                .failure_mode
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
            {
                return Err(AuditError(format!(
                    "competing failure for cell {} requires a failure mode",
                    cell.cell_id
                )));
            }
        }
        LifetimeStatus::SyntheticOutcome => {
            if evidence.evidence_class != EvidenceClass::SyntheticFixture {
                return Err(AuditError(format!(
                    "synthetic outcome for cell {} requires synthetic-fixture evidence",
                    cell.cell_id
                )));
            }
            if cell.cycles_observed == 0 {
                return Err(AuditError(format!(
                    "synthetic outcome for cell {} requires a nonzero cycle count",
                    cell.cell_id
                )));
            }
            if cell
                .event_cycle
                .is_some_and(|cycle| cycle == 0 || cycle > cell.cycles_observed)
            {
                return Err(AuditError(format!(
                    "synthetic event for cell {} is outside observed cycles",
                    cell.cell_id
                )));
            }
        }
        LifetimeStatus::Invalid | LifetimeStatus::Unknown => {
            if cell.event_cycle.is_some() {
                return Err(AuditError(format!(
                    "invalid/unknown cell {} cannot claim an EOL event",
                    cell.cell_id
                )));
            }
        }
    }
    Ok(())
}

/// Audit a finite formation × charge-policy campaign.
/// This validates design-of-experiments and evidence semantics, not electrochemical performance.
pub fn audit_campaign(manifest: &Manifest) -> Result<AuditReport, AuditError> {
    let campaign = &manifest.campaign;
    require_nonempty(&campaign.campaign_id, "campaign ID")?;
    require_nonempty(&campaign.mission_id, "mission ID")?;
    if campaign.maximum_feature_cutoff_cycle == 0 {
        return Err(AuditError(
            "maximum feature cutoff cycle must be nonzero".into(),
        ));
    }
    if manifest.arms.is_empty() || manifest.cells.is_empty() {
        return Err(AuditError(
            "campaign requires protocol arms and cell observations".into(),
        ));
    }

    let registry = evidence_registry(&manifest.evidence)?;
    validate_unique(
        manifest.arms.iter().map(|arm| arm.arm_id.as_str()),
        "protocol arm ID",
    )?;
    validate_unique(
        manifest.cells.iter().map(|cell| cell.cell_id.as_str()),
        "physical cell ID",
    )?;

    let mut arms_by_id = BTreeMap::new();
    let mut reference_arm_ids = Vec::new();
    let mut design_formations: BTreeMap<(String, String), BTreeSet<(String, String)>> =
        BTreeMap::new();
    let mut design_charge_policies: BTreeMap<(String, String), BTreeSet<(String, String)>> =
        BTreeMap::new();
    let mut design_combinations: BTreeSet<(String, String, String, String, String, String)> =
        BTreeSet::new();
    let mut formation_profiles = BTreeSet::new();
    let mut charge_policies = BTreeSet::new();

    for arm in &manifest.arms {
        for (value, label) in [
            (arm.design_id.as_str(), "arm design ID"),
            (arm.design_revision.as_str(), "arm design revision"),
            (arm.formation_profile_id.as_str(), "formation profile ID"),
            (arm.formation_revision.as_str(), "formation revision"),
            (arm.charge_policy_id.as_str(), "charge policy ID"),
            (
                arm.charge_policy_revision.as_str(),
                "charge policy revision",
            ),
        ] {
            require_nonempty(value, label)?;
        }
        require_admitted(
            &registry,
            &arm.protocol_evidence_id,
            &format!("arm {}", arm.arm_id),
        )?;
        let design_key = (arm.design_id.clone(), arm.design_revision.clone());
        design_formations
            .entry(design_key.clone())
            .or_default()
            .insert((
                arm.formation_profile_id.clone(),
                arm.formation_revision.clone(),
            ));
        design_charge_policies
            .entry(design_key.clone())
            .or_default()
            .insert((
                arm.charge_policy_id.clone(),
                arm.charge_policy_revision.clone(),
            ));
        design_combinations.insert((
            arm.design_id.clone(),
            arm.design_revision.clone(),
            arm.formation_profile_id.clone(),
            arm.formation_revision.clone(),
            arm.charge_policy_id.clone(),
            arm.charge_policy_revision.clone(),
        ));
        formation_profiles.insert((
            arm.formation_profile_id.clone(),
            arm.formation_revision.clone(),
        ));
        charge_policies.insert((
            arm.charge_policy_id.clone(),
            arm.charge_policy_revision.clone(),
        ));
        if arm.is_reference {
            reference_arm_ids.push(arm.arm_id.clone());
        }
        arms_by_id.insert(arm.arm_id.as_str(), arm);
    }

    if reference_arm_ids.len() != 1 {
        return Err(AuditError(format!(
            "campaign requires exactly one reference arm; found {}",
            reference_arm_ids.len()
        )));
    }
    if formation_profiles.len() < 2 || charge_policies.len() < 2 {
        return Err(AuditError(
            "campaign requires at least two formation profiles and two charge policies".into(),
        ));
    }

    for (design_key, formations) in &design_formations {
        let charges = design_charge_policies
            .get(design_key)
            .ok_or_else(|| AuditError("design charge-policy set is missing".into()))?;
        if formations.len() < 2 || charges.len() < 2 {
            return Err(AuditError(format!(
                "design {}@{} requires at least two formation and two charge levels",
                design_key.0, design_key.1
            )));
        }
        for (formation_id, formation_revision) in formations {
            for (charge_id, charge_revision) in charges {
                if !design_combinations.contains(&(
                    design_key.0.clone(),
                    design_key.1.clone(),
                    formation_id.clone(),
                    formation_revision.clone(),
                    charge_id.clone(),
                    charge_revision.clone(),
                )) {
                    return Err(AuditError(format!(
                        "incomplete factorial: {}@{} lacks {}@{} × {}@{}",
                        design_key.0,
                        design_key.1,
                        formation_id,
                        formation_revision,
                        charge_id,
                        charge_revision
                    )));
                }
            }
        }
    }

    let mut lot_digests = BTreeMap::new();
    let mut line_digests = BTreeMap::new();
    let mut split_by_group = BTreeMap::<HoldoutGroup, DataSplit>::new();
    let mut used_holdout_groups = BTreeSet::<HoldoutGroup>::new();
    let mut arms_by_block = BTreeMap::<String, BTreeSet<String>>::new();
    let mut observed_eol_count = 0;
    let mut right_censored_count = 0;
    let mut competing_failure_count = 0;
    let mut invalid_or_unknown_count = 0;
    let mut synthetic_outcome_count = 0;
    let mut physical_lifetime_outcome_count = 0;
    let mut maximum_observed_feature_cutoff_cycle = 0;

    for cell in &manifest.cells {
        for (value, label) in [
            (cell.arm_id.as_str(), "cell arm ID"),
            (cell.design_id.as_str(), "cell design ID"),
            (cell.design_revision.as_str(), "cell design revision"),
            (cell.lot_id.as_str(), "lot ID"),
            (cell.line_id.as_str(), "line ID"),
        ] {
            require_nonempty(value, label)?;
        }
        validate_digest(&cell.lot_lineage_digest, "lot lineage digest")?;
        validate_digest(&cell.line_identity_digest, "line identity digest")?;
        if cell.feature_cutoff_cycle == 0
            || cell.feature_cutoff_cycle > campaign.maximum_feature_cutoff_cycle
        {
            return Err(AuditError(format!(
                "cell {} feature cutoff exceeds the declared prediction-time budget",
                cell.cell_id
            )));
        }
        maximum_observed_feature_cutoff_cycle =
            maximum_observed_feature_cutoff_cycle.max(cell.feature_cutoff_cycle);

        let arm = arms_by_id
            .get(cell.arm_id.as_str())
            .copied()
            .ok_or_else(|| {
                AuditError(format!(
                    "cell {} references unknown protocol arm {}",
                    cell.cell_id, cell.arm_id
                ))
            })?;
        if cell.design_id != arm.design_id || cell.design_revision != arm.design_revision {
            return Err(AuditError(format!(
                "cell {} design identity differs from its protocol arm",
                cell.cell_id
            )));
        }

        let outcome_evidence = require_admitted(
            &registry,
            &cell.outcome_evidence_id,
            &format!("cell {}", cell.cell_id),
        )?;
        validate_lifetime_status(cell, outcome_evidence)?;

        let group = holdout_group(cell, arm, campaign.holdout_axis);
        match split_by_group.insert(group.clone(), cell.split) {
            Some(previous) if previous != cell.split => {
                return Err(AuditError(format!(
                    "holdout group {group:?} crosses {:?} and {:?} splits",
                    previous, cell.split
                )));
            }
            _ => {}
        }
        used_holdout_groups.insert(group);

        let block_id = match campaign.blocking_axis {
            BlockAxis::Lot => cell.lot_id.clone(),
            BlockAxis::Line => cell.line_id.clone(),
        };
        arms_by_block
            .entry(block_id)
            .or_default()
            .insert(cell.arm_id.clone());

        match lot_digests.insert(cell.lot_id.as_str(), cell.lot_lineage_digest.as_str()) {
            Some(previous) if previous != cell.lot_lineage_digest.as_str() => {
                return Err(AuditError(format!(
                    "lot {} has conflicting lineage digests",
                    cell.lot_id
                )));
            }
            _ => {}
        }
        match line_digests.insert(cell.line_id.as_str(), cell.line_identity_digest.as_str()) {
            Some(previous) if previous != cell.line_identity_digest.as_str() => {
                return Err(AuditError(format!(
                    "line {} has conflicting identity digests",
                    cell.line_id
                )));
            }
            _ => {}
        }

        match cell.status {
            LifetimeStatus::ObservedEol => {
                observed_eol_count += 1;
                physical_lifetime_outcome_count += 1;
            }
            LifetimeStatus::RightCensored => {
                right_censored_count += 1;
                physical_lifetime_outcome_count += 1;
            }
            LifetimeStatus::CompetingFailure => {
                competing_failure_count += 1;
                physical_lifetime_outcome_count += 1;
            }
            LifetimeStatus::Invalid | LifetimeStatus::Unknown => {
                invalid_or_unknown_count += 1;
            }
            LifetimeStatus::SyntheticOutcome => synthetic_outcome_count += 1,
        }
    }

    let expected_arm_ids: BTreeSet<String> = manifest
        .arms
        .iter()
        .map(|arm| arm.arm_id.clone())
        .collect();
    for (block_id, block_arm_ids) in &arms_by_block {
        if block_arm_ids != &expected_arm_ids {
            return Err(AuditError(format!(
                "block {block_id} does not contain the full protocol-arm set"
            )));
        }
    }

    // Reverse-map digests to stable group identities to reject aliases.
    let mut lot_ids_by_digest = BTreeMap::<&str, &str>::new();
    for (lot_id, digest) in &lot_digests {
        match lot_ids_by_digest.insert(*digest, *lot_id) {
            Some(previous) if previous != *lot_id => {
                return Err(AuditError(format!(
                    "lot lineage digest is aliased by {previous} and {lot_id}"
                )));
            }
            _ => {}
        }
    }
    let mut line_ids_by_digest = BTreeMap::<&str, &str>::new();
    for (line_id, digest) in &line_digests {
        match line_ids_by_digest.insert(*digest, *line_id) {
            Some(previous) if previous != *line_id => {
                return Err(AuditError(format!(
                    "line identity digest is aliased by {previous} and {line_id}"
                )));
            }
            _ => {}
        }
    }

    let reference_arm_id = reference_arm_ids
        .into_iter()
        .next()
        .ok_or_else(|| AuditError("reference arm disappeared".into()))?;
    Ok(AuditReport {
        campaign_id: campaign.campaign_id.clone(),
        arm_count: manifest.arms.len(),
        formation_profile_count: formation_profiles.len(),
        charge_policy_count: charge_policies.len(),
        cell_count: manifest.cells.len(),
        lot_count: lot_digests.len(),
        line_count: line_digests.len(),
        block_count: arms_by_block.len(),
        holdout_group_count: used_holdout_groups.len(),
        observed_eol_count,
        right_censored_count,
        competing_failure_count,
        invalid_or_unknown_count,
        synthetic_outcome_count,
        physical_lifetime_outcome_count,
        maximum_observed_feature_cutoff_cycle,
        reference_arm_id,
        claim_ceiling: "Finite campaign design, split and evidence semantics only; no electrochemical performance, cycle-life gain, safety or commercial claim.".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn evidence(
        evidence_id: &str,
        character: char,
        evidence_class: EvidenceClass,
        metrology_ref: Option<&str>,
    ) -> EvidenceRecord {
        EvidenceRecord {
            evidence_id: evidence_id.into(),
            source_digest: digest(character),
            evidence_class,
            admission: EvidenceAdmission::AdmittedForEvaluation,
            claim_scope: format!("test scope for {evidence_id}"),
            validity_domain: "synthetic test fixture".into(),
            metrology_ref: metrology_ref.map(str::to_string),
        }
    }

    fn fixture() -> Manifest {
        let arms = vec![
            ProtocolArm {
                arm_id: "reference".into(),
                design_id: "CELL-DESIGN-A".into(),
                design_revision: "r1".into(),
                formation_profile_id: "FORM-BASE".into(),
                formation_revision: "r1".into(),
                charge_policy_id: "CHARGE-BASE".into(),
                charge_policy_revision: "r1".into(),
                protocol_evidence_id: "E-ARM".into(),
                is_reference: true,
            },
            ProtocolArm {
                arm_id: "formation-only".into(),
                design_id: "CELL-DESIGN-A".into(),
                design_revision: "r1".into(),
                formation_profile_id: "FORM-CANDIDATE".into(),
                formation_revision: "r1".into(),
                charge_policy_id: "CHARGE-BASE".into(),
                charge_policy_revision: "r1".into(),
                protocol_evidence_id: "E-ARM".into(),
                is_reference: false,
            },
            ProtocolArm {
                arm_id: "charge-only".into(),
                design_id: "CELL-DESIGN-A".into(),
                design_revision: "r1".into(),
                formation_profile_id: "FORM-BASE".into(),
                formation_revision: "r1".into(),
                charge_policy_id: "CHARGE-FAST".into(),
                charge_policy_revision: "r1".into(),
                protocol_evidence_id: "E-ARM".into(),
                is_reference: false,
            },
            ProtocolArm {
                arm_id: "joint".into(),
                design_id: "CELL-DESIGN-A".into(),
                design_revision: "r1".into(),
                formation_profile_id: "FORM-CANDIDATE".into(),
                formation_revision: "r1".into(),
                charge_policy_id: "CHARGE-FAST".into(),
                charge_policy_revision: "r1".into(),
                protocol_evidence_id: "E-ARM".into(),
                is_reference: false,
            },
        ];
        let lot_data = [
            ("LOT-A", '1', DataSplit::Train),
            ("LOT-B", '2', DataSplit::Validation),
            ("LOT-C", '3', DataSplit::Test),
        ];
        let arm_data = [
            ("reference", 'a'),
            ("formation-only", 'b'),
            ("charge-only", 'c'),
            ("joint", 'd'),
        ];
        let mut cells = Vec::new();
        for (lot_id, lot_char, split) in lot_data {
            for (arm_id, _) in arm_data {
                let suffix = cells.len() + 1;
                cells.push(CellObservation {
                    cell_id: format!("CELL-{suffix:03}"),
                    arm_id: arm_id.into(),
                    design_id: "CELL-DESIGN-A".into(),
                    design_revision: "r1".into(),
                    lot_id: lot_id.into(),
                    lot_lineage_digest: digest(lot_char),
                    line_id: "LINE-1".into(),
                    line_identity_digest: digest('4'),
                    split,
                    feature_cutoff_cycle: 20,
                    cycles_observed: 500,
                    status: LifetimeStatus::SyntheticOutcome,
                    event_cycle: Some(420 + suffix as u32),
                    failure_mode: None,
                    outcome_evidence_id: "E-OUTCOME".into(),
                });
            }
        }
        Manifest {
            campaign: Campaign {
                campaign_id: "FORMATION-CHARGE-CO-DESIGN-001".into(),
                mission_id: "EV-CITY-DELIVERY-DUTY-01".into(),
                blocking_axis: BlockAxis::Lot,
                holdout_axis: HoldoutAxis::Lot,
                maximum_feature_cutoff_cycle: 50,
            },
            arms,
            cells,
            evidence: vec![
                evidence("E-ARM", '5', EvidenceClass::SyntheticFixture, None),
                evidence("E-OUTCOME", '6', EvidenceClass::SyntheticFixture, None),
            ],
        }
    }

    #[test]
    fn factorial_fixture_closes_without_claiming_physical_lifetime() {
        let report = audit_campaign(&fixture()).expect("synthetic factorial should close");
        assert_eq!(report.arm_count, 4);
        assert_eq!(report.formation_profile_count, 2);
        assert_eq!(report.charge_policy_count, 2);
        assert_eq!(report.cell_count, 12);
        assert_eq!(report.lot_count, 3);
        assert_eq!(report.line_count, 1);
        assert_eq!(report.block_count, 3);
        assert_eq!(report.observed_eol_count, 0);
        assert_eq!(report.synthetic_outcome_count, 12);
        assert_eq!(report.physical_lifetime_outcome_count, 0);
        assert_eq!(report.reference_arm_id, "reference");
    }

    #[test]
    fn matched_block_must_contain_every_protocol_arm() {
        let mut value = fixture();
        value
            .cells
            .retain(|cell| !(cell.lot_id == "LOT-A" && cell.arm_id == "joint"));
        assert!(audit_campaign(&value).is_err());
    }

    #[test]
    fn line_blocking_requires_every_arm_on_each_line() {
        let mut value = fixture();
        value.campaign.blocking_axis = BlockAxis::Line;
        value.cells.push({
            let mut cell = value.cells[0].clone();
            cell.cell_id = "CELL-EXTRA".into();
            cell.lot_id = "LOT-D".into();
            cell.lot_lineage_digest = digest('8');
            cell.line_id = "LINE-2".into();
            cell.line_identity_digest = digest('9');
            cell.arm_id = "reference".into();
            cell
        });
        assert!(audit_campaign(&value).is_err());
    }

    #[test]
    fn incomplete_factorial_is_rejected() {
        let mut value = fixture();
        value.arms.pop();
        value.cells.retain(|cell| cell.arm_id != "joint");
        assert!(audit_campaign(&value).is_err());
    }

    #[test]
    fn exactly_one_reference_arm_is_required() {
        let mut value = fixture();
        value.arms[1].is_reference = true;
        assert!(audit_campaign(&value).is_err());
    }

    #[test]
    fn protocol_arm_evidence_must_be_admitted() {
        let mut value = fixture();
        value.evidence[0].admission = EvidenceAdmission::AdmittedForResearch;
        assert!(audit_campaign(&value).is_err());
    }

    #[test]
    fn cell_design_must_match_its_arm() {
        let mut value = fixture();
        value.cells[0].design_revision = "r2".into();
        assert!(audit_campaign(&value).is_err());
    }

    #[test]
    fn lot_holdout_group_cannot_cross_splits() {
        let mut value = fixture();
        value.cells[1].split = DataSplit::Test;
        assert!(audit_campaign(&value).is_err());
    }

    #[test]
    fn protocol_pair_holdout_group_cannot_cross_splits() {
        let mut value = fixture();
        value.campaign.holdout_axis = HoldoutAxis::ProtocolPair;
        for cell in &mut value.cells {
            cell.split = DataSplit::Train;
        }
        value.cells[4].split = DataSplit::Test;
        assert!(audit_campaign(&value).is_err());
    }

    #[test]
    fn protocol_pair_ids_with_delimiters_do_not_collide() {
        let mut value = fixture();
        value.campaign.holdout_axis = HoldoutAxis::ProtocolPair;
        for cell in &mut value.cells {
            cell.split = DataSplit::Train;
        }
        value.arms[0].formation_profile_id = "A".into();
        value.arms[0].formation_revision = "B|C".into();
        value.arms[0].charge_policy_id = "D".into();
        value.arms[0].charge_policy_revision = "E".into();
        value.arms[1].formation_profile_id = "A".into();
        value.arms[1].formation_revision = "B".into();
        value.arms[1].charge_policy_id = "D".into();
        value.arms[1].charge_policy_revision = "E".into();
        value.arms[2].formation_profile_id = "A".into();
        value.arms[2].formation_revision = "B|C".into();
        value.arms[2].charge_policy_id = "C|D".into();
        value.arms[2].charge_policy_revision = "E".into();
        value.arms[3].formation_profile_id = "A".into();
        value.arms[3].formation_revision = "B".into();
        value.arms[3].charge_policy_id = "C|D".into();
        value.arms[3].charge_policy_revision = "E".into();

        for cell in &mut value.cells {
            if cell.arm_id == "joint" {
                cell.split = DataSplit::Test;
            }
        }
        assert!(audit_campaign(&value).is_ok());
    }

    #[test]
    fn feature_cutoff_cannot_exceed_prediction_time_budget() {
        let mut value = fixture();
        value.cells[0].feature_cutoff_cycle = 51;
        assert!(audit_campaign(&value).is_err());
    }

    #[test]
    fn synthetic_outcome_cannot_be_promoted_to_physical_eol() {
        let mut value = fixture();
        value.cells[0].status = LifetimeStatus::ObservedEol;
        value.cells[0].event_cycle = Some(400);
        assert!(audit_campaign(&value).is_err());
    }

    #[test]
    fn physical_eol_requires_metrology_admitted_evidence() {
        let mut value = fixture();
        value.evidence.push(evidence(
            "E-PHYSICAL",
            '7',
            EvidenceClass::PhysicalMeasurement,
            Some("METROLOGY-01"),
        ));
        value.cells[0].status = LifetimeStatus::ObservedEol;
        value.cells[0].event_cycle = Some(400);
        value.cells[0].outcome_evidence_id = "E-PHYSICAL".into();
        let report = audit_campaign(&value).expect("physical observation is admissible");
        assert_eq!(report.physical_lifetime_outcome_count, 1);
        assert_eq!(report.observed_eol_count, 1);
    }

    #[test]
    fn physical_evidence_without_metrology_is_rejected() {
        let mut value = fixture();
        value.evidence.push(evidence(
            "E-PHYSICAL",
            '7',
            EvidenceClass::PhysicalMeasurement,
            None,
        ));
        assert!(audit_campaign(&value).is_err());
    }

    #[test]
    fn right_censored_cell_is_not_an_observed_eol() {
        let mut value = fixture();
        value.evidence.push(evidence(
            "E-PHYSICAL",
            '7',
            EvidenceClass::PhysicalMeasurement,
            Some("METROLOGY-01"),
        ));
        value.cells[0].status = LifetimeStatus::RightCensored;
        value.cells[0].event_cycle = None;
        value.cells[0].outcome_evidence_id = "E-PHYSICAL".into();
        let report = audit_campaign(&value).expect("right censor is valid");
        assert_eq!(report.right_censored_count, 1);
        assert_eq!(report.observed_eol_count, 0);
    }

    #[test]
    fn competing_failure_requires_event_cycle_and_failure_mode() {
        let mut value = fixture();
        value.evidence.push(evidence(
            "E-PHYSICAL",
            '7',
            EvidenceClass::PhysicalMeasurement,
            Some("METROLOGY-01"),
        ));
        value.cells[0].status = LifetimeStatus::CompetingFailure;
        value.cells[0].event_cycle = Some(100);
        value.cells[0].failure_mode = Some("separator_short".into());
        value.cells[0].outcome_evidence_id = "E-PHYSICAL".into();
        let report = audit_campaign(&value).expect("competing failure remains distinct");
        assert_eq!(report.competing_failure_count, 1);
    }

    #[test]
    fn duplicate_lot_lineage_identity_is_rejected() {
        let mut value = fixture();
        value.cells[4].lot_lineage_digest = value.cells[0].lot_lineage_digest.clone();
        assert!(audit_campaign(&value).is_err());
    }

    #[test]
    fn input_permutation_does_not_change_report() {
        let value = fixture();
        let expected = audit_campaign(&value).expect("fixture should audit");
        let mut permuted = value;
        permuted.cells.reverse();
        permuted.arms.reverse();
        permuted.evidence.reverse();
        let actual = audit_campaign(&permuted).expect("permuted fixture should audit");
        assert_eq!(expected, actual);
    }
}
