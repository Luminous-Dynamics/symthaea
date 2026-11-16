//! Conservative hierarchy-aware split-leakage auditing for Subterra datasets.
//!
//! A row is not an independent scientific unit merely because it has a
//! different filename or case ID. This auditor connects rows that share a
//! site, physical acquisition, or declared common-cause group, then requires
//! every connected component to remain inside one data split.
//!
//! The resulting components are conservative split-grouping units, NOT proof
//! of statistical independence. Undeclared common causes remain undetected;
//! group IDs are caller assertions and are not authenticated here.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataSplit {
    Training,
    Calibration,
    Evaluation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharedCauseKind {
    InstrumentCalibration,
    EnvironmentalEpisode,
    ProcessingLineage,
    ReferenceLabelCampaign,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DependencyKind {
    Site,
    PhysicalAcquisition,
    InstrumentCalibration,
    EnvironmentalEpisode,
    ProcessingLineage,
    ReferenceLabelCampaign,
    Other,
}

impl From<SharedCauseKind> for DependencyKind {
    fn from(value: SharedCauseKind) -> Self {
        match value {
            SharedCauseKind::InstrumentCalibration => Self::InstrumentCalibration,
            SharedCauseKind::EnvironmentalEpisode => Self::EnvironmentalEpisode,
            SharedCauseKind::ProcessingLineage => Self::ProcessingLineage,
            SharedCauseKind::ReferenceLabelCampaign => Self::ReferenceLabelCampaign,
            SharedCauseKind::Other => Self::Other,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedCauseGroup {
    pub kind: SharedCauseKind,
    /// Stable ID for one shared episode/campaign/lineage, not a global
    /// instrument or software name that would join the entire dataset.
    pub group_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageCase {
    pub case_id: String,
    pub site_id: String,
    /// All physical acquisitions contributing to this raw or derived record.
    /// A fused map/product may name multiple acquisition lineages; each joins
    /// the record to every contributing dependency component.
    pub acquisition_lineage_ids: Vec<String>,
    pub split: DataSplit,
    /// Caller-declared shared causes that must not cross split boundaries.
    pub shared_causes: Vec<SharedCauseGroup>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageDataset {
    pub dataset_id: String,
    pub cases: Vec<LineageCase>,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationArtifactKind {
    RawAcquisition,
    DerivedProduct,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestSharedCauseGroup {
    pub kind: SharedCauseKind,
    pub group_id: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationLineageRecord {
    pub record_id: String,
    pub site_id: String,
    pub observation_id: String,
    /// Exact union of all contributing physical acquisition IDs.
    pub acquisition_lineage_ids: Vec<String>,
    pub artifact_kind: ObservationArtifactKind,
    pub source_record_ids: Vec<String>,
    pub shared_cause_groups: Vec<ManifestSharedCauseGroup>,
    pub provenance_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationLineageManifest {
    pub schema_version: String,
    pub manifest_id: String,
    pub dataset_id: String,
    pub records: Vec<ObservationLineageRecord>,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    InvalidJson(String),
    UnsupportedSchemaVersion(String),
    EmptyManifestId,
    EmptyDatasetId,
    EmptyManifestRecords,
    EmptyRecordField { record_id: String, field: &'static str },
    DuplicateRecordId(String),
    DuplicateSourceRecord { record_id: String, source_record_id: String },
    DuplicateAcquisitionLineage { record_id: String, acquisition_lineage_id: String },
    DuplicateSharedCauseGroup {
        record_id: String,
        kind: SharedCauseKind,
        group_id: String,
    },
    RawAcquisitionHasSources(String),
    RawAcquisitionMustNameExactlyOne(String),
    DerivedProductHasNoSources(String),
    UnknownSourceRecord { record_id: String, source_record_id: String },
    SourceGraphCycle,
    DerivedAcquisitionLineageMismatch(String),
    MissingSplitAssignment(String),
    UnknownSplitAssignment(String),
    SplitAudit(AuditError),
    MissingSiteEvidenceDocuments,
    InvalidSiteEvidenceJson(String),
    UnsupportedSiteEvidenceSchemaVersion(String),
    EmptySiteEvidenceSiteId,
    DuplicateSiteEvidenceSite(String),
    EmptySiteEvidenceObservationId { site_id: String },
    DuplicateSiteEvidenceObservationId {
        observation_id: String,
        first_site_id: String,
        duplicate_site_id: String,
    },
    ManifestObservationNotFound {
        record_id: String,
        site_id: String,
        observation_id: String,
    },
    ManifestObservationSiteMismatch {
        record_id: String,
        expected_site_id: String,
        actual_site_id: String,
    },
    DuplicateManifestObservationId {
        observation_id: String,
        first_record_id: String,
        duplicate_record_id: String,
    },
    SiteEvidenceObservationUnmapped {
        site_id: String,
        observation_id: String,
    },
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ManifestError {}


#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
struct SiteEvidenceObservationProjection {
    schema_version: String,
    site_id: String,
    observations: Vec<SiteEvidenceObservationProjectionItem>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
struct SiteEvidenceObservationProjectionItem {
    observation_id: String,
}

pub fn parse_lineage_manifest(input: &str) -> Result<ObservationLineageManifest, ManifestError> {
    let manifest: ObservationLineageManifest =
        serde_json::from_str(input).map_err(|error| ManifestError::InvalidJson(error.to_string()))?;
    manifest.validate()?;
    Ok(manifest)
}

impl ObservationLineageManifest {
    /// Validate the typed v0.1 sidecar, including graph constraints that JSON
    /// Schema cannot express: source references, acyclicity and exact inherited
    /// acquisition lineage for every derived product.
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.schema_version != "0.1" {
            return Err(ManifestError::UnsupportedSchemaVersion(
                self.schema_version.clone(),
            ));
        }
        if !nonempty(&self.manifest_id) {
            return Err(ManifestError::EmptyManifestId);
        }
        if !nonempty(&self.dataset_id) {
            return Err(ManifestError::EmptyDatasetId);
        }
        if self.records.is_empty() {
            return Err(ManifestError::EmptyManifestRecords);
        }

        let mut records_by_id = BTreeMap::<&str, &ObservationLineageRecord>::new();
        for record in &self.records {
            for (value, field) in [
                (record.record_id.as_str(), "record_id"),
                (record.site_id.as_str(), "site_id"),
                (record.observation_id.as_str(), "observation_id"),
                (record.provenance_ref.as_str(), "provenance_ref"),
            ] {
                if !nonempty(value) {
                    return Err(ManifestError::EmptyRecordField {
                        record_id: record.record_id.clone(),
                        field,
                    });
                }
            }

            if records_by_id.insert(record.record_id.as_str(), record).is_some() {
                return Err(ManifestError::DuplicateRecordId(record.record_id.clone()));
            }
            if record.acquisition_lineage_ids.is_empty() {
                return Err(ManifestError::EmptyRecordField {
                    record_id: record.record_id.clone(),
                    field: "acquisition_lineage_ids",
                });
            }

            let mut acquisition_ids = BTreeSet::new();
            for acquisition_id in &record.acquisition_lineage_ids {
                if !nonempty(acquisition_id) {
                    return Err(ManifestError::EmptyRecordField {
                        record_id: record.record_id.clone(),
                        field: "acquisition_lineage_id",
                    });
                }
                if !acquisition_ids.insert(acquisition_id.as_str()) {
                    return Err(ManifestError::DuplicateAcquisitionLineage {
                        record_id: record.record_id.clone(),
                        acquisition_lineage_id: acquisition_id.clone(),
                    });
                }
            }

            let mut source_ids = BTreeSet::new();
            for source_id in &record.source_record_ids {
                if !nonempty(source_id) {
                    return Err(ManifestError::EmptyRecordField {
                        record_id: record.record_id.clone(),
                        field: "source_record_id",
                    });
                }
                if !source_ids.insert(source_id.as_str()) {
                    return Err(ManifestError::DuplicateSourceRecord {
                        record_id: record.record_id.clone(),
                        source_record_id: source_id.clone(),
                    });
                }
            }

            let mut cause_ids = BTreeSet::new();
            for cause in &record.shared_cause_groups {
                if !nonempty(&cause.group_id) {
                    return Err(ManifestError::EmptyRecordField {
                        record_id: record.record_id.clone(),
                        field: "shared_cause_group_id",
                    });
                }
                if !cause_ids.insert((cause.kind, cause.group_id.as_str())) {
                    return Err(ManifestError::DuplicateSharedCauseGroup {
                        record_id: record.record_id.clone(),
                        kind: cause.kind,
                        group_id: cause.group_id.clone(),
                    });
                }
            }

            match record.artifact_kind {
                ObservationArtifactKind::RawAcquisition => {
                    if !record.source_record_ids.is_empty() {
                        return Err(ManifestError::RawAcquisitionHasSources(
                            record.record_id.clone(),
                        ));
                    }
                    if record.acquisition_lineage_ids.len() != 1 {
                        return Err(ManifestError::RawAcquisitionMustNameExactlyOne(
                            record.record_id.clone(),
                        ));
                    }
                }
                ObservationArtifactKind::DerivedProduct => {
                    if record.source_record_ids.is_empty() {
                        return Err(ManifestError::DerivedProductHasNoSources(
                            record.record_id.clone(),
                        ));
                    }
                }
            }
        }

        let mut indegree = self
            .records
            .iter()
            .map(|record| (record.record_id.as_str(), 0usize))
            .collect::<BTreeMap<_, _>>();
        let mut children = BTreeMap::<&str, Vec<&str>>::new();

        for record in &self.records {
            for source_id in &record.source_record_ids {
                if !records_by_id.contains_key(source_id.as_str()) {
                    return Err(ManifestError::UnknownSourceRecord {
                        record_id: record.record_id.clone(),
                        source_record_id: source_id.clone(),
                    });
                }
                *indegree
                    .get_mut(record.record_id.as_str())
                    .ok_or(ManifestError::SourceGraphCycle)? += 1;
                children
                    .entry(source_id.as_str())
                    .or_default()
                    .push(record.record_id.as_str());
            }
        }

        let mut ready = indegree
            .iter()
            .filter_map(|(record_id, degree)| (*degree == 0).then_some(*record_id))
            .collect::<VecDeque<_>>();
        let mut visited_count = 0usize;
        while let Some(record_id) = ready.pop_front() {
            visited_count += 1;
            if let Some(descendants) = children.get(record_id) {
                for child_id in descendants {
                    let child_degree = indegree
                        .get_mut(child_id)
                        .ok_or(ManifestError::SourceGraphCycle)?;
                    *child_degree = child_degree.saturating_sub(1);
                    if *child_degree == 0 {
                        ready.push_back(child_id);
                    }
                }
            }
        }
        if visited_count != self.records.len() {
            return Err(ManifestError::SourceGraphCycle);
        }

        for record in &self.records {
            if record.artifact_kind == ObservationArtifactKind::DerivedProduct {
                let expected_acquisitions = record
                    .source_record_ids
                    .iter()
                    .filter_map(|source_id| records_by_id.get(source_id.as_str()))
                    .flat_map(|source| source.acquisition_lineage_ids.iter().map(String::as_str))
                    .collect::<BTreeSet<_>>();
                let declared_acquisitions = record
                    .acquisition_lineage_ids
                    .iter()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>();
                if expected_acquisitions != declared_acquisitions {
                    return Err(ManifestError::DerivedAcquisitionLineageMismatch(
                        record.record_id.clone(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Convert the producer-side manifest to split-audit rows. Split
    /// assignments are deliberately supplied by the evaluation protocol, not
    /// embedded in Symtropy's physical evidence record.
    pub fn to_lineage_dataset(
        &self,
        split_by_record_id: &BTreeMap<String, DataSplit>,
    ) -> Result<LineageDataset, ManifestError> {
        self.validate()?;
        let known_record_ids = self
            .records
            .iter()
            .map(|record| record.record_id.as_str())
            .collect::<BTreeSet<_>>();

        for record in &self.records {
            if !split_by_record_id.contains_key(&record.record_id) {
                return Err(ManifestError::MissingSplitAssignment(
                    record.record_id.clone(),
                ));
            }
        }
        for record_id in split_by_record_id.keys() {
            if !known_record_ids.contains(record_id.as_str()) {
                return Err(ManifestError::UnknownSplitAssignment(
                    record_id.clone(),
                ));
            }
        }

        let cases = self
            .records
            .iter()
            .map(|record| LineageCase {
                case_id: record.record_id.clone(),
                site_id: record.site_id.clone(),
                acquisition_lineage_ids: record.acquisition_lineage_ids.clone(),
                split: split_by_record_id[&record.record_id],
                shared_causes: record
                    .shared_cause_groups
                    .iter()
                    .map(|cause| SharedCauseGroup {
                        kind: cause.kind,
                        group_id: cause.group_id.clone(),
                    })
                    .collect(),
            })
            .collect();

        Ok(LineageDataset {
            dataset_id: self.dataset_id.clone(),
            cases,
        })
    }


    /// Cross-check every lineage record against the observation arrays in
    /// schema-conforming Symtropy site-evidence JSON documents. This is a
    /// strict projection of the required fields; the producer's JSON Schema
    /// remains responsible for validating the full site-evidence document.
    pub fn validate_against_site_evidence_json(
        &self,
        site_evidence_json: &[&str],
    ) -> Result<(), ManifestError> {
        self.validate()?;
        if site_evidence_json.is_empty() {
            return Err(ManifestError::MissingSiteEvidenceDocuments);
        }

        let mut site_ids = BTreeSet::<String>::new();
        let mut observations_by_id = BTreeMap::<String, String>::new();

        for input in site_evidence_json {
            let site: SiteEvidenceObservationProjection =
                serde_json::from_str(input)
                    .map_err(|error| ManifestError::InvalidSiteEvidenceJson(error.to_string()))?;
            if site.schema_version != "0.1" {
                return Err(ManifestError::UnsupportedSiteEvidenceSchemaVersion(
                    site.schema_version,
                ));
            }
            if !nonempty(&site.site_id) {
                return Err(ManifestError::EmptySiteEvidenceSiteId);
            }
            if !site_ids.insert(site.site_id.clone()) {
                return Err(ManifestError::DuplicateSiteEvidenceSite(site.site_id));
            }

            let mut local_observations = BTreeSet::new();
            for observation in site.observations {
                if !nonempty(&observation.observation_id) {
                    return Err(ManifestError::EmptySiteEvidenceObservationId {
                        site_id: site.site_id.clone(),
                    });
                }
                if !local_observations.insert(observation.observation_id.clone()) {
                    return Err(ManifestError::DuplicateSiteEvidenceObservationId {
                        observation_id: observation.observation_id,
                        first_site_id: site.site_id.clone(),
                        duplicate_site_id: site.site_id.clone(),
                    });
                }
                if let Some(first_site_id) = observations_by_id.insert(
                    observation.observation_id.clone(),
                    site.site_id.clone(),
                ) {
                    return Err(ManifestError::DuplicateSiteEvidenceObservationId {
                        observation_id: observation.observation_id,
                        first_site_id,
                        duplicate_site_id: site.site_id.clone(),
                    });
                }
            }
        }

        let mut manifest_observations = BTreeMap::<&str, &str>::new();
        for record in &self.records {
            if let Some(first_record_id) = manifest_observations.insert(
                record.observation_id.as_str(),
                record.record_id.as_str(),
            ) {
                return Err(ManifestError::DuplicateManifestObservationId {
                    observation_id: record.observation_id.clone(),
                    first_record_id: first_record_id.to_string(),
                    duplicate_record_id: record.record_id.clone(),
                });
            }

            match observations_by_id.get(&record.observation_id) {
                None => {
                    return Err(ManifestError::ManifestObservationNotFound {
                        record_id: record.record_id.clone(),
                        site_id: record.site_id.clone(),
                        observation_id: record.observation_id.clone(),
                    });
                }
                Some(actual_site_id) if actual_site_id != &record.site_id => {
                    return Err(ManifestError::ManifestObservationSiteMismatch {
                        record_id: record.record_id.clone(),
                        expected_site_id: record.site_id.clone(),
                        actual_site_id: actual_site_id.clone(),
                    });
                }
                Some(_) => {}
            }
        }

        for (observation_id, site_id) in &observations_by_id {
            if !manifest_observations.contains_key(observation_id.as_str()) {
                return Err(ManifestError::SiteEvidenceObservationUnmapped {
                    site_id: site_id.clone(),
                    observation_id: observation_id.clone(),
                });
            }
        }

        Ok(())
    }

    pub fn audit_with_site_evidence_json(
        &self,
        site_evidence_json: &[&str],
        split_by_record_id: &BTreeMap<String, DataSplit>,
    ) -> Result<DependencyAuditReport, ManifestError> {
        self.validate_against_site_evidence_json(site_evidence_json)?;
        self.audit_with_splits(split_by_record_id)
    }

    pub fn audit_with_splits(
        &self,
        split_by_record_id: &BTreeMap<String, DataSplit>,
    ) -> Result<DependencyAuditReport, ManifestError> {
        self.to_lineage_dataset(split_by_record_id)?
            .audit()
            .map_err(ManifestError::SplitAudit)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditError {
    EmptyDatasetId,
    EmptyCaseSet,
    EmptyField {
        case_id: String,
        field: &'static str,
    },
    DuplicateCaseId(String),
    SiteAppearsInMultipleSplits {
        site_id: String,
        first_split: DataSplit,
        conflicting_split: DataSplit,
    },
    DependencyGroupCrossesSplits {
        kind: DependencyKind,
        group_id: String,
        first_split: DataSplit,
        conflicting_split: DataSplit,
    },
    DuplicateSharedCauseGroup {
        case_id: String,
        kind: SharedCauseKind,
        group_id: String,
    },
}

impl std::fmt::Display for AuditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AuditError {}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey {
    kind: DependencyKind,
    group_id: String,
}

impl GroupKey {
    fn new(kind: DependencyKind, group_id: &str) -> Self {
        Self {
            kind,
            group_id: group_id.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyGroupAudit {
    pub kind: DependencyKind,
    pub group_id: String,
    pub case_count: usize,
    pub distinct_site_count: usize,
    pub split: DataSplit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyComponent {
    /// Lexicographically smallest case ID in this component; deterministic
    /// for the same input identities regardless of row order.
    pub representative_case_id: String,
    pub split: DataSplit,
    pub case_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyAuditReport {
    pub dataset_id: String,
    pub total_cases: usize,
    pub unique_sites: usize,
    pub unique_acquisitions: usize,
    pub declared_shared_cause_memberships: usize,
    /// Includes automatic Site and PhysicalAcquisition groups.
    pub dependency_group_count: usize,
    /// Transitive connected components under the declared grouping contract.
    pub dependency_component_count: usize,
    pub singleton_component_count: usize,
    pub largest_component_case_count: usize,
    pub groups: Vec<DependencyGroupAudit>,
    pub components: Vec<DependencyComponent>,
}

#[derive(Debug, Clone)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            let root = self.find(self.parent[value]);
            self.parent[value] = root;
        }
        self.parent[value]
    }

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }

        if self.rank[left_root] < self.rank[right_root] {
            self.parent[left_root] = right_root;
        } else if self.rank[left_root] > self.rank[right_root] {
            self.parent[right_root] = left_root;
        } else {
            self.parent[right_root] = left_root;
            self.rank[left_root] = self.rank[left_root].saturating_add(1);
        }
    }
}

impl LineageDataset {
    /// Validate split isolation and calculate conservative dependency
    /// components. Repeated rows from one acquisition are allowed, but they
    /// can never increase the number of dependency components.
    pub fn audit(&self) -> Result<DependencyAuditReport, AuditError> {
        if !nonempty(&self.dataset_id) {
            return Err(AuditError::EmptyDatasetId);
        }
        if self.cases.is_empty() {
            return Err(AuditError::EmptyCaseSet);
        }

        let mut case_ids = BTreeSet::new();
        let mut site_splits = BTreeMap::<String, DataSplit>::new();
        let mut group_splits = BTreeMap::<GroupKey, DataSplit>::new();
        let mut group_members = BTreeMap::<GroupKey, Vec<usize>>::new();
        let mut unique_sites = BTreeSet::new();
        let mut unique_acquisitions = BTreeSet::new();
        let mut declared_shared_cause_memberships = 0usize;

        for (case_index, case) in self.cases.iter().enumerate() {
            for (value, field) in [
                (case.case_id.as_str(), "case_id"),
                (case.site_id.as_str(), "site_id"),
            ] {
                if !nonempty(value) {
                    return Err(AuditError::EmptyField {
                        case_id: case.case_id.clone(),
                        field,
                    });
                }
            }

            if !case_ids.insert(case.case_id.as_str()) {
                return Err(AuditError::DuplicateCaseId(case.case_id.clone()));
            }
            unique_sites.insert(case.site_id.as_str());

            match site_splits.get(&case.site_id) {
                Some(first_split) if *first_split != case.split => {
                    return Err(AuditError::SiteAppearsInMultipleSplits {
                        site_id: case.site_id.clone(),
                        first_split: *first_split,
                        conflicting_split: case.split,
                    });
                }
                Some(_) => {}
                None => {
                    site_splits.insert(case.site_id.clone(), case.split);
                }
            }

            if case.acquisition_lineage_ids.is_empty() {
                return Err(AuditError::EmptyAcquisitionLineageSet {
                    case_id: case.case_id.clone(),
                });
            }
            let mut case_acquisitions = BTreeSet::new();
            let mut keys = vec![GroupKey::new(DependencyKind::Site, &case.site_id)];
            for acquisition_lineage_id in &case.acquisition_lineage_ids {
                if !nonempty(acquisition_lineage_id) {
                    return Err(AuditError::EmptyField {
                        case_id: case.case_id.clone(),
                        field: "acquisition_lineage_id",
                    });
                }
                if !case_acquisitions.insert(acquisition_lineage_id.as_str()) {
                    return Err(AuditError::DuplicateAcquisitionLineage(
                        acquisition_lineage_id.clone(),
                    ));
                }
                unique_acquisitions.insert(acquisition_lineage_id.as_str());
                keys.push(GroupKey::new(
                    DependencyKind::PhysicalAcquisition,
                    acquisition_lineage_id,
                ));
            }

            let mut case_groups = BTreeSet::new();
            for shared_cause in &case.shared_causes {
                if !nonempty(&shared_cause.group_id) {
                    return Err(AuditError::EmptyField {
                        case_id: case.case_id.clone(),
                        field: "shared_cause_group_id",
                    });
                }
                if !case_groups.insert((shared_cause.kind, shared_cause.group_id.as_str())) {
                    return Err(AuditError::DuplicateSharedCauseGroup {
                        case_id: case.case_id.clone(),
                        kind: shared_cause.kind,
                        group_id: shared_cause.group_id.clone(),
                    });
                }
                declared_shared_cause_memberships += 1;
                keys.push(GroupKey::new(
                    shared_cause.kind.into(),
                    &shared_cause.group_id,
                ));
            }

            for key in keys {
                match group_splits.get(&key) {
                    Some(first_split) if *first_split != case.split => {
                        if key.kind == DependencyKind::Site {
                            return Err(AuditError::SiteAppearsInMultipleSplits {
                                site_id: key.group_id,
                                first_split: *first_split,
                                conflicting_split: case.split,
                            });
                        }
                        return Err(AuditError::DependencyGroupCrossesSplits {
                            kind: key.kind,
                            group_id: key.group_id,
                            first_split: *first_split,
                            conflicting_split: case.split,
                        });
                    }
                    Some(_) => {}
                    None => {
                        group_splits.insert(key.clone(), case.split);
                    }
                }
                group_members.entry(key).or_default().push(case_index);
            }
        }

        let mut union_find = UnionFind::new(self.cases.len());
        for members in group_members.values() {
            if let Some(first) = members.first().copied() {
                for member in members.iter().skip(1).copied() {
                    union_find.union(first, member);
                }
            }
        }

        let mut component_members = BTreeMap::<usize, Vec<usize>>::new();
        for case_index in 0..self.cases.len() {
            let root = union_find.find(case_index);
            component_members.entry(root).or_default().push(case_index);
        }

        let mut components = component_members
            .into_values()
            .map(|members| {
                let mut case_ids = members
                    .iter()
                    .map(|index| self.cases[*index].case_id.clone())
                    .collect::<Vec<_>>();
                case_ids.sort();
                DependencyComponent {
                    representative_case_id: case_ids[0].clone(),
                    split: self.cases[members[0]].split,
                    case_ids,
                }
            })
            .collect::<Vec<_>>();
        components.sort_by(|left, right| {
            left.representative_case_id
                .cmp(&right.representative_case_id)
        });

        let mut groups = group_members
            .into_iter()
            .map(|(key, members)| {
                let distinct_sites = members
                    .iter()
                    .map(|index| self.cases[*index].site_id.as_str())
                    .collect::<BTreeSet<_>>();
                DependencyGroupAudit {
                    kind: key.kind,
                    group_id: key.group_id,
                    case_count: members.len(),
                    distinct_site_count: distinct_sites.len(),
                    split: self.cases[members[0]].split,
                }
            })
            .collect::<Vec<_>>();
        groups.sort_by(|left, right| {
            left.kind
                .cmp(&right.kind)
                .then_with(|| left.group_id.cmp(&right.group_id))
        });

        let singleton_component_count = components
            .iter()
            .filter(|component| component.case_ids.len() == 1)
            .count();
        let largest_component_case_count = components
            .iter()
            .map(|component| component.case_ids.len())
            .max()
            .unwrap_or(0);

        Ok(DependencyAuditReport {
            dataset_id: self.dataset_id.clone(),
            total_cases: self.cases.len(),
            unique_sites: unique_sites.len(),
            unique_acquisitions: unique_acquisitions.len(),
            declared_shared_cause_memberships,
            dependency_group_count: groups.len(),
            dependency_component_count: components.len(),
            singleton_component_count,
            largest_component_case_count,
            groups,
            components,
        })
    }
}


pub const MINIMUM_CLUSTER_BOOTSTRAP_CLUSTERS: usize = 20;
pub const MINIMUM_CLUSTER_BOOTSTRAP_REPLICATES: usize = 200;
pub const MAXIMUM_CLUSTER_BOOTSTRAP_REPLICATES: usize = 1_000_000;

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterScorePair {
    /// Must identify a dependency component from the lineage audit, not a raw row.
    pub dependency_component_id: String,
    /// One pre-aggregated score for this component and the baseline policy.
    pub baseline_score: f64,
    /// One pre-aggregated score for this component and the candidate policy.
    pub candidate_score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterBootstrapConfig {
    pub replicates: usize,
    pub seed: u64,
    /// Must be strictly between 0 and 1 (for example, 0.95).
    pub confidence_level: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapInterval {
    pub lower_bound: f64,
    pub upper_bound: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedClusterBootstrapReport {
    pub method_id: &'static str,
    pub cluster_count: usize,
    pub replicates: usize,
    pub seed: u64,
    pub confidence_level: f64,
    pub mean_baseline_score: f64,
    pub mean_candidate_score: f64,
    /// Candidate minus baseline. For loss metrics, negative favors candidate.
    pub mean_candidate_minus_baseline: f64,
    pub candidate_minus_baseline_interval: BootstrapInterval,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootstrapError {
    EmptyClusterScores,
    TooFewClusters { observed: usize, minimum: usize },
    EmptyDependencyComponentId,
    DuplicateDependencyComponentId(String),
    NonFiniteScore { dependency_component_id: String },
    NumericalOverflow,
    InvalidConfidenceLevel,
    TooFewReplicates { observed: usize, minimum: usize },
    TooManyReplicates { observed: usize, maximum: usize },
}

impl std::fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for BootstrapError {}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper_exclusive: usize) -> usize {
        (((self.next_u64() as u128) * (upper_exclusive as u128)) >> 64) as usize
    }
}

/// Paired percentile bootstrap over already aggregated dependency components.
///
/// Each component receives equal weight. This intentionally prevents many
/// correlated rows inside one component from dominating the estimate. The
/// caller must orient the score consistently; for loss metrics, lower is
/// better and a negative candidate-minus-baseline interval favors candidate.
///
/// This routine does not prove components are independent. Its interval is
/// only as defensible as the caller's dependency grouping and sampling frame.
pub fn paired_cluster_bootstrap(
    scores: &[ClusterScorePair],
    config: ClusterBootstrapConfig,
) -> Result<PairedClusterBootstrapReport, BootstrapError> {
    if scores.is_empty() {
        return Err(BootstrapError::EmptyClusterScores);
    }
    if scores.len() < MINIMUM_CLUSTER_BOOTSTRAP_CLUSTERS {
        return Err(BootstrapError::TooFewClusters {
            observed: scores.len(),
            minimum: MINIMUM_CLUSTER_BOOTSTRAP_CLUSTERS,
        });
    }
    if config.replicates < MINIMUM_CLUSTER_BOOTSTRAP_REPLICATES {
        return Err(BootstrapError::TooFewReplicates {
            observed: config.replicates,
            minimum: MINIMUM_CLUSTER_BOOTSTRAP_REPLICATES,
        });
    }
    if config.replicates > MAXIMUM_CLUSTER_BOOTSTRAP_REPLICATES {
        return Err(BootstrapError::TooManyReplicates {
            observed: config.replicates,
            maximum: MAXIMUM_CLUSTER_BOOTSTRAP_REPLICATES,
        });
    }
    if !config.confidence_level.is_finite()
        || config.confidence_level <= 0.0
        || config.confidence_level >= 1.0
    {
        return Err(BootstrapError::InvalidConfidenceLevel);
    }

    let mut ordered_scores = scores.to_vec();
    ordered_scores.sort_by(|left, right| {
        left.dependency_component_id
            .cmp(&right.dependency_component_id)
    });
    let mut seen = BTreeSet::new();
    for score in &ordered_scores {
        if !nonempty(&score.dependency_component_id) {
            return Err(BootstrapError::EmptyDependencyComponentId);
        }
        if !seen.insert(score.dependency_component_id.as_str()) {
            return Err(BootstrapError::DuplicateDependencyComponentId(
                score.dependency_component_id.clone(),
            ));
        }
        if !score.baseline_score.is_finite() || !score.candidate_score.is_finite() {
            return Err(BootstrapError::NonFiniteScore {
                dependency_component_id: score.dependency_component_id.clone(),
            });
        }
        if !(score.candidate_score - score.baseline_score).is_finite() {
            return Err(BootstrapError::NumericalOverflow);
        }
    }

    let cluster_count = ordered_scores.len();
    let baseline_sum = ordered_scores.iter().map(|score| score.baseline_score).sum::<f64>();
    let candidate_sum = ordered_scores.iter().map(|score| score.candidate_score).sum::<f64>();
    if !baseline_sum.is_finite() || !candidate_sum.is_finite() {
        return Err(BootstrapError::NumericalOverflow);
    }
    let mean_baseline_score = baseline_sum / cluster_count as f64;
    let mean_candidate_score = candidate_sum / cluster_count as f64;
    if !mean_baseline_score.is_finite() || !mean_candidate_score.is_finite() {
        return Err(BootstrapError::NumericalOverflow);
    }

    let deltas = ordered_scores.iter()
        .map(|score| score.candidate_score - score.baseline_score)
        .collect::<Vec<_>>();
    let delta_sum = deltas.iter().sum::<f64>();
    let mean_candidate_minus_baseline = delta_sum / cluster_count as f64;
    if !delta_sum.is_finite() || !mean_candidate_minus_baseline.is_finite() {
        return Err(BootstrapError::NumericalOverflow);
    }
    let mut rng = SplitMix64::new(config.seed);
    let mut bootstrap_means = Vec::with_capacity(config.replicates);
    for _ in 0..config.replicates {
        let mut sum = 0.0;
        for _ in 0..cluster_count {
            sum += deltas[rng.index(cluster_count)];
        }
        let mean = sum / cluster_count as f64;
        if !mean.is_finite() {
            return Err(BootstrapError::NumericalOverflow);
        }
        bootstrap_means.push(mean);
    }
    bootstrap_means.sort_by(f64::total_cmp);

    let tail_probability = (1.0 - config.confidence_level) / 2.0;
    let last_index = (bootstrap_means.len() - 1) as f64;
    let lower_index = (tail_probability * last_index).floor() as usize;
    let upper_index = ((1.0 - tail_probability) * last_index).ceil() as usize;
    let candidate_minus_baseline_interval = BootstrapInterval {
        lower_bound: bootstrap_means[lower_index],
        upper_bound: bootstrap_means[upper_index],
    };

    Ok(PairedClusterBootstrapReport {
        method_id: "paired-percentile-cluster-bootstrap-v0.1",
        cluster_count,
        replicates: config.replicates,
        seed: config.seed,
        confidence_level: config.confidence_level,
        mean_baseline_score,
        mean_candidate_score,
        mean_candidate_minus_baseline,
        candidate_minus_baseline_interval,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(
        case_id: &str,
        site_id: &str,
        acquisition_id: &str,
        split: DataSplit,
    ) -> LineageCase {
        LineageCase {
            case_id: case_id.into(),
            site_id: site_id.into(),
            acquisition_lineage_ids: vec![acquisition_id.into()],
            split,
            shared_causes: Vec::new(),
        }
    }

    fn shared(kind: SharedCauseKind, id: &str) -> SharedCauseGroup {
        SharedCauseGroup {
            kind,
            group_id: id.into(),
        }
    }

    fn dataset(cases: Vec<LineageCase>) -> LineageDataset {
        LineageDataset {
            dataset_id: "subterra-lineage-fixture-v1".into(),
            cases,
        }
    }

    #[test]
    fn repeated_derived_records_from_one_acquisition_form_one_component() {
        let data = dataset(vec![
            case("raw", "site-a", "acq-1", DataSplit::Training),
            case("mesh", "site-a", "acq-1", DataSplit::Training),
            case("features", "site-a", "acq-1", DataSplit::Training),
        ]);
        let report = data.audit().unwrap();
        assert_eq!(report.total_cases, 3);
        assert_eq!(report.unique_acquisitions, 1);
        assert_eq!(report.dependency_component_count, 1);
        assert_eq!(report.largest_component_case_count, 3);
    }

    #[test]
    fn different_acquisitions_from_one_site_remain_one_site_component() {
        let data = dataset(vec![
            case("scan-1", "site-a", "acq-1", DataSplit::Training),
            case("scan-2", "site-a", "acq-2", DataSplit::Training),
        ]);
        let report = data.audit().unwrap();
        assert_eq!(report.unique_acquisitions, 2);
        assert_eq!(report.dependency_component_count, 1);
        assert_eq!(report.components[0].case_ids, vec!["scan-1".to_string(), "scan-2".to_string()]);
    }

    #[test]
    fn shared_calibration_session_joins_distinct_sites() {
        let mut left = case("left", "site-a", "acq-a", DataSplit::Training);
        left.shared_causes.push(shared(
            SharedCauseKind::InstrumentCalibration,
            "cal-session-7",
        ));
        let mut right = case("right", "site-b", "acq-b", DataSplit::Training);
        right.shared_causes.push(shared(
            SharedCauseKind::InstrumentCalibration,
            "cal-session-7",
        ));
        let report = dataset(vec![left, right]).audit().unwrap();
        assert_eq!(report.unique_sites, 2);
        assert_eq!(report.dependency_component_count, 1);
        assert_eq!(report.components[0].case_ids, vec!["left", "right"]);
    }

    #[test]
    fn transitive_shared_causes_join_components() {
        let mut a = case("a", "site-a", "acq-a", DataSplit::Training);
        a.shared_causes
            .push(shared(SharedCauseKind::EnvironmentalEpisode, "storm-1"));
        let mut b = case("b", "site-b", "acq-b", DataSplit::Training);
        b.shared_causes
            .push(shared(SharedCauseKind::EnvironmentalEpisode, "storm-1"));
        b.shared_causes
            .push(shared(SharedCauseKind::ProcessingLineage, "pipeline-run-9"));
        let mut c = case("c", "site-c", "acq-c", DataSplit::Training);
        c.shared_causes
            .push(shared(SharedCauseKind::ProcessingLineage, "pipeline-run-9"));
        let report = dataset(vec![a, b, c]).audit().unwrap();
        assert_eq!(report.dependency_component_count, 1);
        assert_eq!(report.largest_component_case_count, 3);
    }

    #[test]
    fn shared_cause_group_crossing_splits_is_rejected() {
        let mut a = case("a", "site-a", "acq-a", DataSplit::Training);
        a.shared_causes
            .push(shared(SharedCauseKind::EnvironmentalEpisode, "cave-episode-1"));
        let mut b = case("b", "site-b", "acq-b", DataSplit::Evaluation);
        b.shared_causes
            .push(shared(SharedCauseKind::EnvironmentalEpisode, "cave-episode-1"));
        assert_eq!(
            dataset(vec![a, b]).audit(),
            Err(AuditError::DependencyGroupCrossesSplits {
                kind: DependencyKind::EnvironmentalEpisode,
                group_id: "cave-episode-1".into(),
                first_split: DataSplit::Training,
                conflicting_split: DataSplit::Evaluation,
            })
        );
    }

    #[test]
    fn physical_acquisition_cannot_cross_splits() {
        let data = dataset(vec![
            case("raw", "site-a", "acq-shared", DataSplit::Training),
            case("derived", "site-b", "acq-shared", DataSplit::Evaluation),
        ]);
        assert_eq!(
            data.audit(),
            Err(AuditError::DependencyGroupCrossesSplits {
                kind: DependencyKind::PhysicalAcquisition,
                group_id: "acq-shared".into(),
                first_split: DataSplit::Training,
                conflicting_split: DataSplit::Evaluation,
            })
        );
    }

    #[test]
    fn site_cannot_cross_splits() {
        let data = dataset(vec![
            case("a", "site-shared", "acq-a", DataSplit::Training),
            case("b", "site-shared", "acq-b", DataSplit::Calibration),
        ]);
        assert!(matches!(
            data.audit(),
            Err(AuditError::SiteAppearsInMultipleSplits { site_id, .. })
                if site_id == "site-shared"
        ));
    }

    #[test]
    fn duplicate_case_ids_are_rejected() {
        let data = dataset(vec![
            case("same", "site-a", "acq-a", DataSplit::Training),
            case("same", "site-b", "acq-b", DataSplit::Training),
        ]);
        assert_eq!(data.audit(), Err(AuditError::DuplicateCaseId("same".into())));
    }

    #[test]
    fn blank_dataset_and_empty_case_set_are_rejected() {
        assert_eq!(
            dataset(vec![]).audit(),
            Err(AuditError::EmptyCaseSet)
        );
        let mut data = dataset(vec![case("a", "site-a", "acq-a", DataSplit::Training)]);
        data.dataset_id = "  ".into();
        assert_eq!(data.audit(), Err(AuditError::EmptyDatasetId));
    }

    #[test]
    fn blank_case_identity_fields_are_rejected() {
        let mut data = dataset(vec![case("a", "site-a", "acq-a", DataSplit::Training)]);
        data.cases[0].site_id.clear();
        assert_eq!(
            data.audit(),
            Err(AuditError::EmptyField {
                case_id: "a".into(),
                field: "site_id",
            })
        );
    }

    #[test]
    fn duplicate_shared_cause_in_one_case_is_rejected() {
        let mut record = case("a", "site-a", "acq-a", DataSplit::Training);
        record.shared_causes = vec![
            shared(SharedCauseKind::InstrumentCalibration, "cal-1"),
            shared(SharedCauseKind::InstrumentCalibration, "cal-1"),
        ];
        assert_eq!(
            dataset(vec![record]).audit(),
            Err(AuditError::DuplicateSharedCauseGroup {
                case_id: "a".into(),
                kind: SharedCauseKind::InstrumentCalibration,
                group_id: "cal-1".into(),
            })
        );
    }

    #[test]
    fn blank_shared_cause_group_id_is_rejected() {
        let mut record = case("a", "site-a", "acq-a", DataSplit::Training);
        record
            .shared_causes
            .push(shared(SharedCauseKind::ProcessingLineage, " "));
        assert_eq!(
            dataset(vec![record]).audit(),
            Err(AuditError::EmptyField {
                case_id: "a".into(),
                field: "shared_cause_group_id",
            })
        );
    }

    #[test]
    fn unrelated_sites_and_acquisitions_form_separate_components() {
        let data = dataset(vec![
            case("a", "site-a", "acq-a", DataSplit::Training),
            case("b", "site-b", "acq-b", DataSplit::Training),
            case("c", "site-c", "acq-c", DataSplit::Evaluation),
        ]);
        let report = data.audit().unwrap();
        assert_eq!(report.dependency_component_count, 3);
        assert_eq!(report.singleton_component_count, 3);
    }

    #[test]
    fn group_and_component_summaries_are_exact() {
        let mut a = case("a", "site-a", "acq-a", DataSplit::Training);
        a.shared_causes
            .push(shared(SharedCauseKind::ReferenceLabelCampaign, "panel-2"));
        let mut b = case("b", "site-b", "acq-b", DataSplit::Training);
        b.shared_causes
            .push(shared(SharedCauseKind::ReferenceLabelCampaign, "panel-2"));
        let c = case("c", "site-c", "acq-c", DataSplit::Evaluation);
        let report = dataset(vec![a, b, c]).audit().unwrap();
        assert_eq!(report.total_cases, 3);
        assert_eq!(report.unique_sites, 3);
        assert_eq!(report.unique_acquisitions, 3);
        assert_eq!(report.declared_shared_cause_memberships, 2);
        assert_eq!(report.dependency_group_count, 7);
        assert_eq!(report.dependency_component_count, 2);
        assert_eq!(report.singleton_component_count, 1);
        assert_eq!(report.largest_component_case_count, 2);
        let label_group = report.groups.iter().find(|group| {
            group.kind == DependencyKind::ReferenceLabelCampaign
                && group.group_id == "panel-2"
        }).unwrap();
        assert_eq!(label_group.case_count, 2);
        assert_eq!(label_group.distinct_site_count, 2);
    }

    #[test]
    fn audit_is_deterministic_under_row_permutation() {
        let a = case("a", "site-a", "acq-a", DataSplit::Training);
        let b = case("b", "site-b", "acq-b", DataSplit::Training);
        let c = case("c", "site-c", "acq-c", DataSplit::Evaluation);
        let left = dataset(vec![a.clone(), b.clone(), c.clone()]).audit().unwrap();
        let right = dataset(vec![c, b, a]).audit().unwrap();
        assert_eq!(left, right);
    }

    fn bootstrap_scores() -> Vec<ClusterScorePair> {
        (0..20)
            .map(|index| ClusterScorePair {
                dependency_component_id: format!("component-{index:02}"),
                baseline_score: 2.0 + index as f64 / 10.0,
                candidate_score: 1.0 + index as f64 / 10.0,
            })
            .collect()
    }

    fn bootstrap_config() -> ClusterBootstrapConfig {
        ClusterBootstrapConfig {
            replicates: 1_000,
            seed: 0x5EED_2026,
            confidence_level: 0.95,
        }
    }

    #[test]
    fn paired_bootstrap_uses_equal_component_weight_and_reports_effect() {
        let report = paired_cluster_bootstrap(&bootstrap_scores(), bootstrap_config()).unwrap();
        assert_eq!(report.cluster_count, 20);
        assert_eq!(report.replicates, 1_000);
        assert_eq!(report.seed, 0x5EED_2026);
        assert!((report.mean_baseline_score - 2.95).abs() < 1.0e-12);
        assert!((report.mean_candidate_score - 1.95).abs() < 1.0e-12);
        assert!((report.mean_candidate_minus_baseline + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn bootstrap_is_reproducible_under_input_permutation() {
        let scores = bootstrap_scores();
        let first = paired_cluster_bootstrap(&scores, bootstrap_config()).unwrap();
        let mut reversed = scores;
        reversed.reverse();
        let second = paired_cluster_bootstrap(&reversed, bootstrap_config()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn constant_paired_delta_produces_degenerate_interval() {
        let report = paired_cluster_bootstrap(&bootstrap_scores(), bootstrap_config()).unwrap();
        assert!((report.candidate_minus_baseline_interval.lower_bound + 1.0).abs() < 1.0e-12);
        assert!((report.candidate_minus_baseline_interval.upper_bound + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn bootstrap_rejects_duplicate_component_ids() {
        let mut scores = bootstrap_scores();
        scores[1].dependency_component_id = scores[0].dependency_component_id.clone();
        assert_eq!(
            paired_cluster_bootstrap(&scores, bootstrap_config()),
            Err(BootstrapError::DuplicateDependencyComponentId("component-00".into()))
        );
    }

    #[test]
    fn bootstrap_rejects_non_finite_scores() {
        let mut scores = bootstrap_scores();
        scores[0].candidate_score = f64::NAN;
        assert_eq!(
            paired_cluster_bootstrap(&scores, bootstrap_config()),
            Err(BootstrapError::NonFiniteScore {
                dependency_component_id: "component-00".into()
            })
        );
    }

    #[test]
    fn bootstrap_rejects_too_few_dependency_components() {
        let scores = bootstrap_scores().into_iter().take(19).collect::<Vec<_>>();
        assert_eq!(
            paired_cluster_bootstrap(&scores, bootstrap_config()),
            Err(BootstrapError::TooFewClusters {
                observed: 19,
                minimum: MINIMUM_CLUSTER_BOOTSTRAP_CLUSTERS,
            })
        );
    }

    #[test]
    fn bootstrap_rejects_invalid_replicate_counts() {
        let scores = bootstrap_scores();
        let mut config = bootstrap_config();
        config.replicates = MINIMUM_CLUSTER_BOOTSTRAP_REPLICATES - 1;
        assert!(matches!(
            paired_cluster_bootstrap(&scores, config),
            Err(BootstrapError::TooFewReplicates { .. })
        ));
        config.replicates = MAXIMUM_CLUSTER_BOOTSTRAP_REPLICATES + 1;
        assert!(matches!(
            paired_cluster_bootstrap(&scores, config),
            Err(BootstrapError::TooManyReplicates { .. })
        ));
    }

    #[test]
    fn bootstrap_rejects_invalid_confidence_level() {
        let scores = bootstrap_scores();
        let mut config = bootstrap_config();
        config.confidence_level = 1.0;
        assert_eq!(
            paired_cluster_bootstrap(&scores, config),
            Err(BootstrapError::InvalidConfidenceLevel)
        );
        config.confidence_level = f64::NAN;
        assert_eq!(
            paired_cluster_bootstrap(&scores, config),
            Err(BootstrapError::InvalidConfidenceLevel)
        );
    }

    #[test]
    fn bootstrap_rejects_empty_component_identity() {
        let mut scores = bootstrap_scores();
        scores[0].dependency_component_id = " ".into();
        assert_eq!(
            paired_cluster_bootstrap(&scores, bootstrap_config()),
            Err(BootstrapError::EmptyDependencyComponentId)
        );
    }

    #[test]
    fn bootstrap_rejects_empty_input() {
        assert_eq!(
            paired_cluster_bootstrap(&[], bootstrap_config()),
            Err(BootstrapError::EmptyClusterScores)
        );
    }

    #[test]
    fn fused_product_joins_every_contributing_physical_acquisition() {
        let a = case("raw-a", "site-a", "acq-a", DataSplit::Training);
        let b = case("raw-b", "site-b", "acq-b", DataSplit::Training);
        let mut fused = case("fused", "regional-product", "placeholder", DataSplit::Training);
        fused.acquisition_lineage_ids = vec!["acq-a".into(), "acq-b".into()];
        let report = dataset(vec![a, b, fused]).audit().unwrap();
        assert_eq!(report.unique_acquisitions, 2);
        assert_eq!(report.dependency_component_count, 1);
        assert_eq!(report.largest_component_case_count, 3);
    }

    #[test]
    fn empty_or_duplicate_acquisition_lineages_are_rejected() {
        let mut data = dataset(vec![case("a", "site-a", "acq-a", DataSplit::Training)]);
        data.cases[0].acquisition_lineage_ids.clear();
        assert_eq!(
            data.audit(),
            Err(AuditError::EmptyAcquisitionLineageSet {
                case_id: "a".into()
            })
        );

        data.cases[0].acquisition_lineage_ids = vec!["acq-a".into(), "acq-a".into()];
        assert_eq!(
            data.audit(),
            Err(AuditError::DuplicateAcquisitionLineage("acq-a".into()))
        );
    }

    #[test]
    fn fused_product_cannot_smuggle_a_shared_acquisition_across_splits() {
        let mut fused = case("fused", "site-a", "acq-a", DataSplit::Training);
        fused.acquisition_lineage_ids.push("acq-b".into());
        let b = case("raw-b", "site-b", "acq-b", DataSplit::Evaluation);
        assert_eq!(
            dataset(vec![fused, b]).audit(),
            Err(AuditError::DependencyGroupCrossesSplits {
                kind: DependencyKind::PhysicalAcquisition,
                group_id: "acq-b".into(),
                first_split: DataSplit::Training,
                conflicting_split: DataSplit::Evaluation,
            })
        );
    }

    const SYMTROPY_LINEAGE_FIXTURE: &str =
        include_str!("../fixtures/symtropy-observation-lineage-synthetic.json");

    fn fixture_manifest() -> ObservationLineageManifest {
        parse_lineage_manifest(SYMTROPY_LINEAGE_FIXTURE).unwrap()
    }

    fn fixture_splits(manifest: &ObservationLineageManifest) -> BTreeMap<String, DataSplit> {
        manifest.records.iter().map(|record| {
            let split = if record.site_id == "synthetic-site-c" {
                DataSplit::Evaluation
            } else {
                DataSplit::Training
            };
            (record.record_id.clone(), split)
        }).collect()
    }

    #[test]
    fn symtropy_lineage_fixture_parses_and_audits_transitively() {
        let manifest = fixture_manifest();
        assert_eq!(manifest.records.len(), 6);
        let report = manifest.audit_with_splits(&fixture_splits(&manifest)).unwrap();
        assert_eq!(report.total_cases, 6);
        assert_eq!(report.unique_sites, 3);
        assert_eq!(report.unique_acquisitions, 4);
        assert_eq!(report.dependency_component_count, 2);
        assert_eq!(report.singleton_component_count, 1);
        assert_eq!(report.largest_component_case_count, 5);
    }

    #[test]
    fn manifest_split_map_must_cover_exactly_every_record() {
        let manifest = fixture_manifest();
        let mut splits = fixture_splits(&manifest);
        splits.remove("record-site-a-raw-01");
        assert_eq!(
            manifest.to_lineage_dataset(&splits),
            Err(ManifestError::MissingSplitAssignment("record-site-a-raw-01".into()))
        );

        let mut splits = fixture_splits(&manifest);
        splits.insert("not-a-record".into(), DataSplit::Training);
        assert_eq!(
            manifest.to_lineage_dataset(&splits),
            Err(ManifestError::UnknownSplitAssignment("not-a-record".into()))
        );
    }

    #[test]
    fn declared_common_cause_cannot_cross_training_and_evaluation() {
        let manifest = fixture_manifest();
        let mut splits = fixture_splits(&manifest);
        splits.insert("record-site-b-raw-01".into(), DataSplit::Evaluation);
        assert!(matches!(
            manifest.audit_with_splits(&splits),
            Err(ManifestError::SplitAudit(
                AuditError::DependencyGroupCrossesSplits {
                    kind: DependencyKind::InstrumentCalibration,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn source_records_must_exist() {
        let mut manifest = fixture_manifest();
        manifest.records
            .iter_mut()
            .find(|record| record.record_id == "record-site-a-mesh-01")
            .unwrap()
            .source_record_ids[0] = "missing-parent".into();
        assert!(matches!(
            manifest.validate(),
            Err(ManifestError::UnknownSourceRecord {
                record_id,
                source_record_id
            }) if record_id == "record-site-a-mesh-01" && source_record_id == "missing-parent"
        ));
    }

    #[test]
    fn source_graph_must_be_acyclic() {
        let mut manifest = fixture_manifest();
        manifest.records[0].artifact_kind = ObservationArtifactKind::DerivedProduct;
        manifest.records[0].source_record_ids = vec![manifest.records[2].record_id.clone()];
        manifest.records[0].acquisition_lineage_ids = vec!["acquisition-a-01".into()];
        // The mesh record already points back to record-site-a-raw-01.
        assert_eq!(manifest.validate(), Err(ManifestError::SourceGraphCycle));
    }

    #[test]
    fn derived_product_must_preserve_exact_union_of_parent_acquisitions() {
        let mut manifest = fixture_manifest();
        let fused = manifest.records
            .iter_mut()
            .find(|record| record.record_id == "record-site-a-features-01")
            .unwrap();
        fused.acquisition_lineage_ids = vec!["acquisition-a-01".into()];
        assert_eq!(
            manifest.validate(),
            Err(ManifestError::DerivedAcquisitionLineageMismatch(
                "record-site-a-features-01".into()
            ))
        );
    }

    #[test]
    fn raw_and_derived_record_shapes_are_checked() {
        let mut manifest = fixture_manifest();
        manifest.records[0].source_record_ids.push("anything".into());
        assert_eq!(
            manifest.validate(),
            Err(ManifestError::RawAcquisitionHasSources(
                "record-site-a-raw-01".into()
            ))
        );

        let mut manifest = fixture_manifest();
        manifest.records
            .iter_mut()
            .find(|record| record.record_id == "record-site-a-mesh-01")
            .unwrap()
            .source_record_ids
            .clear();
        assert_eq!(
            manifest.validate(),
            Err(ManifestError::DerivedProductHasNoSources(
                "record-site-a-mesh-01".into()
            ))
        );
    }

    #[test]
    fn malformed_json_and_unknown_fields_are_rejected() {
        assert!(matches!(
            parse_lineage_manifest("{"),
            Err(ManifestError::InvalidJson(_))
        ));
        let json = SYMTROPY_LINEAGE_FIXTURE.replace(
            "\"schema_version\": \"0.1\"",
            "\"schema_version\": \"0.1\",\n  \"unreviewed_extra\": true"
        );
        assert!(matches!(
            parse_lineage_manifest(&json),
            Err(ManifestError::InvalidJson(_))
        ));
    }

    #[test]
    fn duplicate_record_ids_are_rejected() {
        let mut manifest = fixture_manifest();
        manifest.records[1].record_id = manifest.records[0].record_id.clone();
        assert_eq!(
            manifest.validate(),
            Err(ManifestError::DuplicateRecordId(
                "record-site-a-raw-01".into()
            ))
        );
    }

    #[test]
    fn duplicate_acquisition_and_shared_cause_memberships_are_rejected() {
        let mut manifest = fixture_manifest();
        manifest.records[0].acquisition_lineage_ids
            .push("acquisition-a-01".into());
        assert!(matches!(
            manifest.validate(),
            Err(ManifestError::DuplicateAcquisitionLineage { .. })
        ));

        let mut manifest = fixture_manifest();
        let cause = manifest.records[0].shared_cause_groups[0].clone();
        manifest.records[0].shared_cause_groups.push(cause);
        assert!(matches!(
            manifest.validate(),
            Err(ManifestError::DuplicateSharedCauseGroup { .. })
        ));
    }

    #[test]
    fn raw_acquisition_must_name_one_physical_acquisition() {
        let mut manifest = fixture_manifest();
        manifest.records[0].acquisition_lineage_ids.push("extra-acq".into());
        assert_eq!(
            manifest.validate(),
            Err(ManifestError::RawAcquisitionMustNameExactlyOne(
                "record-site-a-raw-01".into()
            ))
        );
    }

    const SITE_EVIDENCE_A: &str =
        include_str!("../fixtures/site-evidence-lineage-a-synthetic.json");
    const SITE_EVIDENCE_B: &str =
        include_str!("../fixtures/site-evidence-lineage-b-synthetic.json");
    const SITE_EVIDENCE_C: &str =
        include_str!("../fixtures/site-evidence-lineage-c-synthetic.json");

    fn site_evidence_fixture_set() -> [&'static str; 3] {
        [SITE_EVIDENCE_A, SITE_EVIDENCE_B, SITE_EVIDENCE_C]
    }

    #[test]
    fn lineage_observations_resolve_exactly_once_to_their_site_evidence() {
        let manifest = fixture_manifest();
        assert!(manifest
            .validate_against_site_evidence_json(&site_evidence_fixture_set())
            .is_ok());
        let report = manifest
            .audit_with_site_evidence_json(
                &site_evidence_fixture_set(),
                &fixture_splits(&manifest),
            )
            .unwrap();
        assert_eq!(report.total_cases, 6);
        assert_eq!(report.unique_sites, 3);
    }

    #[test]
    fn site_evidence_fixture_set_must_cover_all_records() {
        let manifest = fixture_manifest();
        assert_eq!(
            manifest.validate_against_site_evidence_json(&[]),
            Err(ManifestError::MissingSiteEvidenceDocuments)
        );
        assert!(matches!(
            manifest.validate_against_site_evidence_json(&[SITE_EVIDENCE_A, SITE_EVIDENCE_B]),
            Err(ManifestError::ManifestObservationNotFound { .. })
        ));
    }

    #[test]
    fn observation_ids_must_be_unique_across_site_documents() {
        let manifest = fixture_manifest();
        let duplicate = SITE_EVIDENCE_B.replace(
            "obs-site-b-raw-01",
            "obs-site-a-raw-01"
        );
        assert!(matches!(
            manifest.validate_against_site_evidence_json(&[
                SITE_EVIDENCE_A,
                &duplicate,
                SITE_EVIDENCE_C
            ]),
            Err(ManifestError::DuplicateSiteEvidenceObservationId { .. })
        ));
    }

    #[test]
    fn manifest_observation_must_belong_to_the_declared_site() {
        let mut manifest = fixture_manifest();
        manifest.records[0].site_id = "synthetic-site-b".into();
        assert_eq!(
            manifest.validate_against_site_evidence_json(&site_evidence_fixture_set()),
            Err(ManifestError::ManifestObservationSiteMismatch {
                record_id: "record-site-a-raw-01".into(),
                expected_site_id: "synthetic-site-b".into(),
                actual_site_id: "synthetic-site-a".into()
            })
        );
    }

    #[test]
    fn every_site_evidence_observation_must_have_a_lineage_record() {
        let manifest = fixture_manifest();
        let extra = SITE_EVIDENCE_C.replace(
            "obs-site-c-raw-01",
            "obs-site-c-raw-01\", \"observation_id\": \"unmapped-extra"
        );
        assert!(matches!(
            manifest.validate_against_site_evidence_json(&[
                SITE_EVIDENCE_A,
                SITE_EVIDENCE_B,
                &extra
            ]),
            Err(ManifestError::InvalidSiteEvidenceJson(_))
                | Err(ManifestError::SiteEvidenceObservationUnmapped { .. })
        ));
    }

    #[test]
    fn duplicate_manifest_observation_bindings_are_rejected() {
        let mut manifest = fixture_manifest();
        manifest.records[1].observation_id = manifest.records[0].observation_id.clone();
        assert!(matches!(
            manifest.validate_against_site_evidence_json(&site_evidence_fixture_set()),
            Err(ManifestError::DuplicateManifestObservationId { .. })
        ));
    }

    #[test]
    fn unsupported_or_malformed_site_evidence_json_is_rejected() {
        let manifest = fixture_manifest();
        assert!(matches!(
            manifest.validate_against_site_evidence_json(&["{"]),
            Err(ManifestError::InvalidSiteEvidenceJson(_))
        ));
        let unsupported = SITE_EVIDENCE_A.replace(
            "\"schema_version\": \"0.1\"",
            "\"schema_version\": \"0.2\""
        );
        assert_eq!(
            manifest.validate_against_site_evidence_json(&[
                &unsupported,
                SITE_EVIDENCE_B,
                SITE_EVIDENCE_C
            ]),
            Err(ManifestError::UnsupportedSiteEvidenceSchemaVersion("0.2".into()))
        );
    }

    #[test]
    fn duplicate_site_documents_and_local_observations_are_rejected() {
        let manifest = fixture_manifest();
        assert_eq!(
            manifest.validate_against_site_evidence_json(&[
                SITE_EVIDENCE_A,
                SITE_EVIDENCE_A,
                SITE_EVIDENCE_C
            ]),
            Err(ManifestError::DuplicateSiteEvidenceSite("synthetic-site-a".into()))
        );
        let duplicate_local = SITE_EVIDENCE_A.replace(
            "\"observation_id\": \"obs-site-a-raw-02\"",
            "\"observation_id\": \"obs-site-a-raw-01\""
        );
        assert!(matches!(
            manifest.validate_against_site_evidence_json(&[
                &duplicate_local,
                SITE_EVIDENCE_B,
                SITE_EVIDENCE_C
            ]),
            Err(ManifestError::DuplicateSiteEvidenceObservationId { .. })
        ));
    }

}
