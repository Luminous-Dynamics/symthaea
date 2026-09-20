// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Executable OQMD v1.7 -> Fe/Co/Zr extraction above the frozen MAG-DATA contracts.
//!
//! qmpy 1.4 is used only as a source-schema compatibility adapter. Rust owns
//! content addressing, multiplicity refusal, composition/structure
//! canonicalization, compact-corpus validation, and extraction receipts.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use symthaea_materials_historical_extraction::{
    CompactHistoricalCorpus, HistoricalExtractionReceipt, NormalizedOqmdRecord,
    OqmdExtractionProtocol, bind_extraction_receipt, canonical_decimal,
};
use symthaea_materials_historical_import::{
    HistoricalImportProfile, ImportExtractionBinding, bind_import_to_extraction,
};
use symthaea_materials_oqmd_db_state::VerifiedImportedDatabaseState;
use symthaea_materials_snapshot_acquisition::HistoricalSnapshotAcquisitionReceipt;
use symthaea_process_capture::{EnvironmentPolicy, ProcessSpec};
use symthaea_process_file_io::{
    BoundFileIoProcessCapture, BoundFileIoProcessRequest, FileIoLauncherArtifact,
    NewStdoutFile, capture_process_with_file_io,
};
use symthaea_process_stdin::ContentAddressedStdinFile;
use thiserror::Error;

/// qmpy release whose public FormationEnergy-list semantics are reproduced.
pub const QMPY_VERSION: &str = "1.4.0";
/// Exact reviewed qmpy source commit for the 1.4.0 adapter semantics.
pub const QMPY_SOURCE_COMMIT: &str = "dede5bdf4aa3ea1187a7bc273e86336c24aadb25";
/// Composition canonicalization contract frozen by MAG-DATA-003.
pub const COMPOSITION_CANONICALIZER_ID: &str = "reduced-integer-stoichiometry-v1";
/// Structure canonicalization contract frozen by MAG-DATA-003.
pub const STRUCTURE_CANONICALIZER_ID: &str = "species-lattice-fractional-sites-v1";
/// Property/method condition contract frozen by MAG-DATA-003.
pub const PROPERTY_CONDITION_POLICY_ID: &str = "oqmd-dft-method-condition-v1";
const STRUCTURE_SCALE: f64 = 100_000_000.0;

/// Exact local executable or manifest artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalArtifact {
    /// Absolute UTF-8 path.
    pub path: String,
    /// SHA-256 of exact file bytes.
    pub sha256: String,
}

impl LocalArtifact {
    /// Observe one exact local artifact.
    pub fn observe(path: &Path) -> Result<Self, ExtractionExecutorError> {
        let path_text = absolute_utf8(path)?;
        let (sha256, _) = hash_file(path)?;
        Ok(Self {
            path: path_text,
            sha256,
        })
    }

    fn validate_and_rehash(&self) -> Result<(), ExtractionExecutorError> {
        validate_sha256(&self.sha256)?;
        if !Path::new(&self.path).is_absolute() {
            return Err(ExtractionExecutorError::PathNotAbsolute(self.path.clone()));
        }
        let (actual, _) = hash_file(Path::new(&self.path))?;
        if !actual.eq_ignore_ascii_case(&self.sha256) {
            return Err(ExtractionExecutorError::ArtifactDigestMismatch(
                self.path.clone(),
            ));
        }
        Ok(())
    }
}

/// Preregistered local qmpy extraction execution plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QmpyExtractionPlan {
    /// Plan schema version.
    pub schema_version: u32,
    /// Exact Python executable.
    pub python: LocalArtifact,
    /// Exact qmpy source adapter script.
    pub adapter_script: LocalArtifact,
    /// Exact qmpy/Nix package manifest.
    pub qmpy_artifact_manifest: LocalArtifact,
    /// Exact Rust canonicalizer/extractor executable.
    pub rust_extractor_artifact: LocalArtifact,
    /// Exact extraction/import environment manifest.
    pub execution_environment_manifest: LocalArtifact,
    /// Exact SIM-PROC-004 launcher artifact.
    pub file_io_launcher: FileIoLauncherArtifact,
    /// Exact empty stdin file used by the output-producing adapter.
    pub empty_stdin: ContentAddressedStdinFile,
    /// qmpy version asserted by the environment.
    pub qmpy_version: String,
    /// Exact qmpy source commit asserted by the environment.
    pub qmpy_source_commit: String,
    /// Imported database name.
    pub database_name: String,
    /// MySQL user; the historical authority path does not accept a password.
    pub database_user: String,
    /// Isolated database Unix socket.
    pub unix_socket_path: String,
    /// Create-new raw NDJSON output path.
    pub raw_ndjson_path: String,
    /// Adapter wall-clock bound.
    pub timeout_ms: u64,
    /// Bounded adapter diagnostic limit.
    pub max_output_bytes: usize,
}

impl QmpyExtractionPlan {
    /// Validate source identity, imported state, and every local artifact.
    pub fn validate_against(
        &self,
        protocol: &OqmdExtractionProtocol,
        profile: &HistoricalImportProfile,
        acquisition: &HistoricalSnapshotAcquisitionReceipt,
        database_state: &VerifiedImportedDatabaseState,
    ) -> Result<(), ExtractionExecutorError> {
        if self.schema_version != 1 {
            return Err(ExtractionExecutorError::UnsupportedPlanSchema(
                self.schema_version,
            ));
        }
        protocol
            .validate()
            .map_err(|error| ExtractionExecutorError::Protocol(error.to_string()))?;
        if protocol.composition_canonicalizer_id != COMPOSITION_CANONICALIZER_ID
            || protocol.structure_canonicalizer_id != STRUCTURE_CANONICALIZER_ID
            || protocol.property_condition_policy_id != PROPERTY_CONDITION_POLICY_ID
        {
            return Err(ExtractionExecutorError::CanonicalizerContractMismatch);
        }
        profile
            .validate_against(protocol)
            .map_err(|error| ExtractionExecutorError::Import(error.to_string()))?;
        acquisition
            .validate_against(protocol)
            .map_err(|error| ExtractionExecutorError::Acquisition(error.to_string()))?;
        database_state
            .evidence_sha256()
            .map_err(|error| ExtractionExecutorError::DatabaseState(error.to_string()))?;
        database_state
            .import_receipt
            .validate_against(protocol, profile, acquisition)
            .map_err(|error| ExtractionExecutorError::Import(error.to_string()))?;
        if self.qmpy_version != QMPY_VERSION || self.qmpy_source_commit != QMPY_SOURCE_COMMIT {
            return Err(ExtractionExecutorError::UnexpectedQmpyIdentity);
        }
        if self.database_name != database_state.inventory.database_name {
            return Err(ExtractionExecutorError::DatabaseNameMismatch);
        }
        if self.database_user.trim().is_empty() {
            return Err(ExtractionExecutorError::EmptyDatabaseUser);
        }
        if !Path::new(&self.unix_socket_path).is_absolute() {
            return Err(ExtractionExecutorError::PathNotAbsolute(
                self.unix_socket_path.clone(),
            ));
        }
        if !Path::new(&self.raw_ndjson_path).is_absolute() {
            return Err(ExtractionExecutorError::PathNotAbsolute(
                self.raw_ndjson_path.clone(),
            ));
        }
        if self.timeout_ms == 0 || self.max_output_bytes == 0 {
            return Err(ExtractionExecutorError::InvalidExecutionBounds);
        }
        for artifact in [
            &self.python,
            &self.adapter_script,
            &self.qmpy_artifact_manifest,
            &self.rust_extractor_artifact,
            &self.execution_environment_manifest,
        ] {
            artifact.validate_and_rehash()?;
        }
        if !self
            .execution_environment_manifest
            .sha256
            .eq_ignore_ascii_case(&profile.import_environment_manifest_sha256)
        {
            return Err(ExtractionExecutorError::EnvironmentManifestMismatch);
        }
        let observed_launcher = FileIoLauncherArtifact::observe(Path::new(
            &self.file_io_launcher.path,
        ))
        .map_err(|error| ExtractionExecutorError::FileIo(error.to_string()))?;
        if observed_launcher != self.file_io_launcher {
            return Err(ExtractionExecutorError::FileIoLauncherMismatch);
        }
        self.empty_stdin
            .validate()
            .map_err(|error| ExtractionExecutorError::Stdin(error.to_string()))?;
        if self.empty_stdin.bytes != 0
            || !self.empty_stdin.sha256.eq_ignore_ascii_case(&sha256_hex(&[]))
        {
            return Err(ExtractionExecutorError::AdapterStdinNotEmpty);
        }
        Ok(())
    }

    /// Deterministic plan identity after full validation.
    pub fn plan_sha256(
        &self,
        protocol: &OqmdExtractionProtocol,
        profile: &HistoricalImportProfile,
        acquisition: &HistoricalSnapshotAcquisitionReceipt,
        database_state: &VerifiedImportedDatabaseState,
    ) -> Result<String, ExtractionExecutorError> {
        self.validate_against(protocol, profile, acquisition, database_state)?;
        Ok(serialized_plan_sha(self)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RawSiteOccupant {
    element: String,
    occupancy: String,
    oxidation_state: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RawQmpySite {
    fractional_coordinate: [String; 3],
    occupants: Vec<RawSiteOccupant>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RawQmpyStructure {
    lattice: [[String; 3]; 3],
    sites: Vec<RawQmpySite>,
    spacegroup: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RawQmpyFormationEnergyRecord {
    schema_version: u32,
    formation_energy_id: u64,
    entry_id: u64,
    duplicate_entry_id: Option<u64>,
    name: String,
    composition_formula: String,
    element_set: Vec<String>,
    prototype: Option<String>,
    natoms: u32,
    ntypes: u16,
    delta_e_ev_atom: Option<String>,
    stability_ev_atom: Option<String>,
    band_gap_ev: Option<String>,
    calculation_id: u64,
    calculation_label: Option<String>,
    fit: String,
    icsd_id: Option<String>,
    structure: Option<RawQmpyStructure>,
}

impl RawQmpyFormationEnergyRecord {
    fn validate(&self, protocol: &OqmdExtractionProtocol) -> Result<(), ExtractionExecutorError> {
        if self.schema_version != 1 {
            return Err(ExtractionExecutorError::UnsupportedRawSchema(
                self.schema_version,
            ));
        }
        if self.formation_energy_id == 0 || self.entry_id == 0 || self.calculation_id == 0 {
            return Err(ExtractionExecutorError::InvalidSourceId(self.entry_id));
        }
        if self.name.trim().is_empty() || self.composition_formula.trim().is_empty() {
            return Err(ExtractionExecutorError::EmptySourceField(self.entry_id));
        }
        if self.fit != "standard" {
            return Err(ExtractionExecutorError::UnexpectedFit(self.fit.clone()));
        }
        if self.duplicate_entry_id == Some(self.entry_id) {
            return Err(ExtractionExecutorError::SelfDuplicate(self.entry_id));
        }
        if self.element_set.is_empty()
            || self
                .element_set
                .windows(2)
                .any(|window| window[0].as_str() >= window[1].as_str())
            || self.ntypes as usize != self.element_set.len()
        {
            return Err(ExtractionExecutorError::InvalidElementSet(self.entry_id));
        }
        let allowed: BTreeSet<&str> = protocol.allowed_elements.iter().map(String::as_str).collect();
        if self
            .element_set
            .iter()
            .any(|element| !allowed.contains(element.as_str()))
        {
            return Err(ExtractionExecutorError::ElementOutsideProtocol(self.entry_id));
        }
        if self.natoms == 0 {
            return Err(ExtractionExecutorError::InvalidAtomCount(self.entry_id));
        }
        for value in [
            self.delta_e_ev_atom.as_deref(),
            self.stability_ev_atom.as_deref(),
            self.band_gap_ev.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            parse_finite(value)?;
        }
        if let Some(structure) = &self.structure {
            for vector in &structure.lattice {
                for value in vector {
                    parse_finite(value)?;
                }
            }
            if structure.sites.is_empty() {
                return Err(ExtractionExecutorError::EmptyStructure(self.entry_id));
            }
            for site in &structure.sites {
                for value in &site.fractional_coordinate {
                    parse_finite(value)?;
                }
                for occupant in &site.occupants {
                    if occupant.element.trim().is_empty() {
                        return Err(ExtractionExecutorError::InvalidStructureOccupant(
                            self.entry_id,
                        ));
                    }
                    let occupancy = parse_finite(&occupant.occupancy)?;
                    if !(occupancy > 0.0 && occupancy <= 1.0) {
                        return Err(ExtractionExecutorError::InvalidOccupancy(self.entry_id));
                    }
                }
            }
        }
        Ok(())
    }
}

/// One OQMD entry carrying more than one qmpy standard-fit row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceMultiplicity {
    /// OQMD entry ID.
    pub entry_id: u64,
    /// FormationEnergy IDs in strict ascending order.
    pub formation_energy_ids: Vec<u64>,
}

/// Explicit source-row multiplicity report produced before corpus construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceMultiplicityReport {
    /// Report schema version.
    pub schema_version: u32,
    /// Exact raw qmpy row count.
    pub raw_record_count: u64,
    /// Entries that cannot be collapsed under the frozen v1 one-record-per-entry rule.
    pub ambiguous_entries: Vec<SourceMultiplicity>,
}

impl SourceMultiplicityReport {
    /// Whether the frozen v1 corpus can be constructed without an invented selector.
    pub fn is_unambiguous(&self) -> bool {
        self.ambiguous_entries.is_empty()
    }

    /// Revalidate canonical ordering and content.
    pub fn validate(&self) -> Result<(), ExtractionExecutorError> {
        if self.schema_version != 1 {
            return Err(ExtractionExecutorError::UnsupportedMultiplicitySchema(
                self.schema_version,
            ));
        }
        if self.raw_record_count == 0 {
            return Err(ExtractionExecutorError::EmptyRawExtract);
        }
        let mut previous_entry = None;
        for item in &self.ambiguous_entries {
            if item.entry_id == 0
                || item.formation_energy_ids.len() < 2
                || item
                    .formation_energy_ids
                    .windows(2)
                    .any(|window| window[0] >= window[1])
                || previous_entry.is_some_and(|previous| item.entry_id <= previous)
            {
                return Err(ExtractionExecutorError::NonCanonicalMultiplicityReport);
            }
            previous_entry = Some(item.entry_id);
        }
        Ok(())
    }

    /// Deterministic report identity.
    pub fn report_sha256(&self) -> Result<String, ExtractionExecutorError> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Raw qmpy adapter process/file evidence before scientific normalization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QmpyAdapterExecutionEvidence {
    /// Exact extraction-plan identity.
    pub plan_sha256: String,
    /// Exact imported-database-state identity.
    pub database_state_evidence_sha256: String,
    /// Exact qmpy source-package manifest identity.
    pub qmpy_artifact_sha256: String,
    /// Exact SIM-PROC-004 file/process evidence.
    pub process: BoundFileIoProcessCapture,
}

impl QmpyAdapterExecutionEvidence {
    /// Deterministic raw adapter evidence identity.
    pub fn evidence_sha256(&self) -> Result<String, ExtractionExecutorError> {
        validate_sha256(&self.plan_sha256)?;
        validate_sha256(&self.database_state_evidence_sha256)?;
        validate_sha256(&self.qmpy_artifact_sha256)?;
        self.process
            .capture_sha256()
            .map_err(|error| ExtractionExecutorError::FileIo(error.to_string()))?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Result of parsing/canonicalizing one completed qmpy adapter artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizationAttempt {
    /// Source multiplicity is always materialized, including on refusal.
    pub multiplicity: SourceMultiplicityReport,
    /// Canonical corpus, absent when source multiplicity is ambiguous.
    pub corpus: Option<CompactHistoricalCorpus>,
}

/// Complete executable historical extraction evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletedQmpyExtraction {
    /// Raw adapter execution evidence.
    pub raw_execution: QmpyAdapterExecutionEvidence,
    /// Source multiplicity report.
    pub multiplicity: SourceMultiplicityReport,
    /// Frozen compact historical corpus.
    pub corpus: CompactHistoricalCorpus,
    /// MAG-DATA-003 extraction receipt.
    pub extraction_receipt: HistoricalExtractionReceipt,
    /// MAG-DATA-008 import -> extraction binding.
    pub import_extraction_binding: ImportExtractionBinding,
}

impl CompletedQmpyExtraction {
    /// Revalidate nested corpus/receipt objects and compute complete evidence identity.
    pub fn evidence_sha256(
        &self,
        protocol: &OqmdExtractionProtocol,
    ) -> Result<String, ExtractionExecutorError> {
        self.raw_execution.evidence_sha256()?;
        self.multiplicity.report_sha256()?;
        if !self.multiplicity.is_unambiguous() {
            return Err(ExtractionExecutorError::NonCanonicalMultiplicityReport);
        }
        self.corpus
            .validate_for(protocol)
            .map_err(|error| ExtractionExecutorError::Historical(error.to_string()))?;
        self.extraction_receipt
            .receipt_sha256()
            .map_err(|error| ExtractionExecutorError::Historical(error.to_string()))?;
        self.import_extraction_binding
            .binding_sha256()
            .map_err(|error| ExtractionExecutorError::Import(error.to_string()))?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Execute the exact qmpy adapter into a create-new NDJSON artifact.
pub fn execute_qmpy_adapter(
    protocol: &OqmdExtractionProtocol,
    profile: &HistoricalImportProfile,
    acquisition: &HistoricalSnapshotAcquisitionReceipt,
    database_state: &VerifiedImportedDatabaseState,
    plan: &QmpyExtractionPlan,
) -> Result<QmpyAdapterExecutionEvidence, ExtractionExecutorError> {
    plan.validate_against(protocol, profile, acquisition, database_state)?;
    let plan_sha = plan.plan_sha256(protocol, profile, acquisition, database_state)?;
    let database_state_sha = database_state
        .evidence_sha256()
        .map_err(|error| ExtractionExecutorError::DatabaseState(error.to_string()))?;

    let mut environment = BTreeMap::new();
    environment.insert(
        "DJANGO_SETTINGS_MODULE".to_string(),
        "qmpy.db.settings".to_string(),
    );
    environment.insert("LANG".to_string(), "C.UTF-8".to_string());
    environment.insert("LC_ALL".to_string(), "C.UTF-8".to_string());
    environment.insert("PYTHONHASHSEED".to_string(), "0".to_string());
    environment.insert("TZ".to_string(), "UTC".to_string());
    environment.insert("qmdb_v1_1_name".to_string(), plan.database_name.clone());
    environment.insert("qmdb_v1_1_user".to_string(), plan.database_user.clone());
    environment.insert("qmdb_v1_1_host".to_string(), plan.unix_socket_path.clone());
    environment.insert("qmdb_v1_1_port".to_string(), String::new());

    let request = BoundFileIoProcessRequest {
        launcher: plan.file_io_launcher.clone(),
        stdin: plan.empty_stdin.clone(),
        stdout: NewStdoutFile::new(Path::new(&plan.raw_ndjson_path))
            .map_err(|error| ExtractionExecutorError::FileIo(error.to_string()))?,
        target: ProcessSpec {
            command: plan.python.path.clone(),
            args: vec![plan.adapter_script.path.clone()],
            environment,
            environment_policy: EnvironmentPolicy::ClearAndSet,
            timeout_ms: plan.timeout_ms,
            max_output_bytes: plan.max_output_bytes,
        },
    };
    let process = capture_process_with_file_io(&request)
        .map_err(|error| ExtractionExecutorError::FileIo(error.to_string()))?;
    let evidence = QmpyAdapterExecutionEvidence {
        plan_sha256: plan_sha,
        database_state_evidence_sha256: database_state_sha,
        qmpy_artifact_sha256: plan.qmpy_artifact_manifest.sha256.to_ascii_lowercase(),
        process,
    };
    evidence.evidence_sha256()?;
    Ok(evidence)
}

/// Parse exact raw adapter bytes and attempt frozen-v1 normalization.
pub fn normalize_adapter_output(
    protocol: &OqmdExtractionProtocol,
    plan: &QmpyExtractionPlan,
    execution: &QmpyAdapterExecutionEvidence,
) -> Result<NormalizationAttempt, ExtractionExecutorError> {
    if !execution
        .plan_sha256
        .eq_ignore_ascii_case(&serialized_plan_sha(plan)?)
        || !execution
            .qmpy_artifact_sha256
            .eq_ignore_ascii_case(&plan.qmpy_artifact_manifest.sha256)
    {
        return Err(ExtractionExecutorError::AdapterPlanMismatch);
    }
    if !execution.process.complete_output() {
        return Err(ExtractionExecutorError::AdapterProcessDidNotSucceed);
    }
    if execution.process.process_capture.stderr_truncated {
        return Err(ExtractionExecutorError::AdapterDiagnosticsTruncated);
    }
    let observed = execution
        .process
        .observed_stdout
        .as_ref()
        .ok_or(ExtractionExecutorError::MissingAdapterOutput)?;
    if observed.path != plan.raw_ndjson_path || observed.bytes == 0 {
        return Err(ExtractionExecutorError::AdapterOutputIdentityMismatch);
    }
    let (actual_sha, actual_bytes) = hash_file(Path::new(&observed.path))?;
    if actual_bytes != observed.bytes || !actual_sha.eq_ignore_ascii_case(&observed.sha256) {
        return Err(ExtractionExecutorError::AdapterOutputIdentityMismatch);
    }

    let records = read_raw_ndjson(Path::new(&observed.path), protocol)?;
    let multiplicity = source_multiplicity_report(&records)?;
    if !multiplicity.is_unambiguous() {
        return Ok(NormalizationAttempt {
            multiplicity,
            corpus: None,
        });
    }
    let normalized = records
        .iter()
        .map(|record| normalize_record(protocol, record))
        .collect::<Result<Vec<_>, _>>()?;
    let corpus = CompactHistoricalCorpus::from_records(protocol, normalized)
        .map_err(|error| ExtractionExecutorError::Historical(error.to_string()))?;
    Ok(NormalizationAttempt {
        multiplicity,
        corpus: Some(corpus),
    })
}

/// Bind a successful unambiguous normalization into the frozen historical receipts.
pub fn finalize_extraction(
    protocol: &OqmdExtractionProtocol,
    profile: &HistoricalImportProfile,
    acquisition: &HistoricalSnapshotAcquisitionReceipt,
    database_state: &VerifiedImportedDatabaseState,
    plan: &QmpyExtractionPlan,
    raw_execution: QmpyAdapterExecutionEvidence,
    normalization: NormalizationAttempt,
) -> Result<CompletedQmpyExtraction, ExtractionExecutorError> {
    let expected_plan = plan.plan_sha256(protocol, profile, acquisition, database_state)?;
    let database_state_sha = database_state
        .evidence_sha256()
        .map_err(|error| ExtractionExecutorError::DatabaseState(error.to_string()))?;
    raw_execution.evidence_sha256()?;
    if !raw_execution.plan_sha256.eq_ignore_ascii_case(&expected_plan)
        || !raw_execution
            .database_state_evidence_sha256
            .eq_ignore_ascii_case(&database_state_sha)
        || !raw_execution
            .qmpy_artifact_sha256
            .eq_ignore_ascii_case(&plan.qmpy_artifact_manifest.sha256)
    {
        return Err(ExtractionExecutorError::AdapterPlanMismatch);
    }
    normalization.multiplicity.validate()?;
    let corpus = normalization
        .corpus
        .ok_or_else(|| ExtractionExecutorError::AmbiguousSourceRows {
            report_sha256: normalization
                .multiplicity
                .report_sha256()
                .unwrap_or_else(|_| "invalid-report".to_string()),
        })?;
    corpus
        .validate_for(protocol)
        .map_err(|error| ExtractionExecutorError::Historical(error.to_string()))?;

    let import_receipt_sha = database_state
        .import_receipt
        .receipt_sha256(protocol, profile, acquisition)
        .map_err(|error| ExtractionExecutorError::Import(error.to_string()))?;
    let schema_sha = database_state
        .inventory
        .inventory_sha256()
        .map_err(|error| ExtractionExecutorError::DatabaseState(error.to_string()))?;
    let composition_impl_sha = component_implementation_sha(
        COMPOSITION_CANONICALIZER_ID,
        &plan.rust_extractor_artifact.sha256,
    );
    let structure_impl_sha = component_implementation_sha(
        STRUCTURE_CANONICALIZER_ID,
        &plan.rust_extractor_artifact.sha256,
    );
    let extraction_receipt = bind_extraction_receipt(
        protocol,
        &corpus,
        &acquisition.compressed_snapshot_sha256,
        &profile.import_environment_manifest_sha256,
        &schema_sha,
        &import_receipt_sha,
        &plan.rust_extractor_artifact.sha256,
        &plan.adapter_script.sha256,
        &plan.qmpy_artifact_manifest.sha256,
        &composition_impl_sha,
        &structure_impl_sha,
    )
    .map_err(|error| ExtractionExecutorError::Historical(error.to_string()))?;
    let import_extraction_binding = bind_import_to_extraction(
        protocol,
        profile,
        acquisition,
        &database_state.import_receipt,
        &extraction_receipt,
    )
    .map_err(|error| ExtractionExecutorError::Import(error.to_string()))?;

    let completed = CompletedQmpyExtraction {
        raw_execution,
        multiplicity: normalization.multiplicity,
        corpus,
        extraction_receipt,
        import_extraction_binding,
    };
    completed.evidence_sha256(protocol)?;
    Ok(completed)
}

fn source_multiplicity_report(
    records: &[RawQmpyFormationEnergyRecord],
) -> Result<SourceMultiplicityReport, ExtractionExecutorError> {
    if records.is_empty() {
        return Err(ExtractionExecutorError::EmptyRawExtract);
    }
    let mut by_entry: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut formation_ids = BTreeSet::new();
    for record in records {
        if !formation_ids.insert(record.formation_energy_id) {
            return Err(ExtractionExecutorError::DuplicateFormationEnergyId(
                record.formation_energy_id,
            ));
        }
        by_entry
            .entry(record.entry_id)
            .or_default()
            .push(record.formation_energy_id);
    }
    let ambiguous_entries = by_entry
        .into_iter()
        .filter_map(|(entry_id, mut ids)| {
            ids.sort_unstable();
            ids.dedup();
            (ids.len() > 1).then_some(SourceMultiplicity {
                entry_id,
                formation_energy_ids: ids,
            })
        })
        .collect();
    let report = SourceMultiplicityReport {
        schema_version: 1,
        raw_record_count: records.len() as u64,
        ambiguous_entries,
    };
    report.validate()?;
    Ok(report)
}

fn read_raw_ndjson(
    path: &Path,
    protocol: &OqmdExtractionProtocol,
) -> Result<Vec<RawQmpyFormationEnergyRecord>, ExtractionExecutorError> {
    let file = File::open(path).map_err(ExtractionExecutorError::Io)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut previous_key: Option<(u64, u64)> = None;
    for (line_index, line) in reader.lines().enumerate() {
        let line = line.map_err(ExtractionExecutorError::Io)?;
        if line.trim().is_empty() {
            return Err(ExtractionExecutorError::EmptyNdjsonLine(line_index + 1));
        }
        let record: RawQmpyFormationEnergyRecord = serde_json::from_str(&line)?;
        record.validate(protocol)?;
        let key = (record.entry_id, record.formation_energy_id);
        if previous_key.is_some_and(|previous| key <= previous) {
            return Err(ExtractionExecutorError::NonCanonicalRawOrder);
        }
        previous_key = Some(key);
        records.push(record);
    }
    if records.is_empty() {
        return Err(ExtractionExecutorError::EmptyRawExtract);
    }
    Ok(records)
}

fn normalize_record(
    protocol: &OqmdExtractionProtocol,
    raw: &RawQmpyFormationEnergyRecord,
) -> Result<NormalizedOqmdRecord, ExtractionExecutorError> {
    raw.validate(protocol)?;
    let (composition_sha256, composition_elements) = canonicalize_composition(&raw.composition_formula)?;
    if composition_elements != raw.element_set {
        return Err(ExtractionExecutorError::CompositionElementMismatch(raw.entry_id));
    }
    let structure_sha256 = raw
        .structure
        .as_ref()
        .map(|structure| canonicalize_structure(raw.entry_id, structure))
        .transpose()?;
    let spacegroup = raw
        .structure
        .as_ref()
        .and_then(|structure| structure.spacegroup.clone());
    Ok(NormalizedOqmdRecord {
        entry_id: raw.entry_id,
        name: raw.name.clone(),
        element_set: raw.element_set.clone(),
        composition_sha256,
        structure_sha256,
        duplicate_entry_id: raw.duplicate_entry_id,
        spacegroup,
        prototype: raw.prototype.clone(),
        natoms: raw.natoms,
        ntypes: raw.ntypes,
        delta_e_ev_atom: canonical_optional_decimal(raw.delta_e_ev_atom.as_deref())?,
        stability_ev_atom: canonical_optional_decimal(raw.stability_ev_atom.as_deref())?,
        band_gap_ev: canonical_optional_decimal(raw.band_gap_ev.as_deref())?,
        calculation_label: raw.calculation_label.clone(),
        fit: Some(raw.fit.clone()),
        icsd_id: raw.icsd_id.clone(),
        property_condition_signature: property_condition_signature(raw)?,
    })
}

fn canonical_optional_decimal(
    value: Option<&str>,
) -> Result<Option<String>, ExtractionExecutorError> {
    value
        .map(|text| {
            canonical_decimal(parse_finite(text)?)
                .map_err(|error| ExtractionExecutorError::Historical(error.to_string()))
        })
        .transpose()
}

#[derive(Serialize)]
struct PropertyCondition<'a> {
    contract_id: &'a str,
    qmpy_version: &'a str,
    fit: &'a str,
    calculation_label: Option<&'a str>,
}

fn property_condition_signature(
    raw: &RawQmpyFormationEnergyRecord,
) -> Result<String, ExtractionExecutorError> {
    Ok(sha256_hex(&serde_json::to_vec(&PropertyCondition {
        contract_id: PROPERTY_CONDITION_POLICY_ID,
        qmpy_version: QMPY_VERSION,
        fit: raw.fit.as_str(),
        calculation_label: raw.calculation_label.as_deref(),
    })?))
}

fn canonicalize_composition(
    formula: &str,
) -> Result<(String, Vec<String>), ExtractionExecutorError> {
    let mut parts: BTreeMap<String, (u128, u128)> = BTreeMap::new();
    for token in formula.split_whitespace() {
        let split = token
            .char_indices()
            .find_map(|(index, ch)| (!ch.is_ascii_alphabetic()).then_some(index))
            .unwrap_or(token.len());
        let (element, amount_text) = token.split_at(split);
        if element.is_empty()
            || !element.chars().enumerate().all(|(index, ch)| {
                if index == 0 {
                    ch.is_ascii_uppercase()
                } else {
                    ch.is_ascii_lowercase()
                }
            })
            || parts.contains_key(element)
        {
            return Err(ExtractionExecutorError::InvalidCompositionFormula(
                formula.to_string(),
            ));
        }
        let ratio = if amount_text.is_empty() {
            (1, 1)
        } else {
            decimal_ratio(amount_text)?
        };
        if ratio.0 == 0 {
            return Err(ExtractionExecutorError::InvalidCompositionFormula(
                formula.to_string(),
            ));
        }
        parts.insert(element.to_string(), ratio);
    }
    if parts.is_empty() {
        return Err(ExtractionExecutorError::InvalidCompositionFormula(
            formula.to_string(),
        ));
    }
    let mut common_denominator = 1_u128;
    for (_, denominator) in parts.values() {
        common_denominator = checked_lcm(common_denominator, *denominator)?;
    }
    let mut integer_counts = Vec::with_capacity(parts.len());
    for (element, (numerator, denominator)) in &parts {
        let factor = common_denominator
            .checked_div(*denominator)
            .ok_or(ExtractionExecutorError::CompositionOverflow)?;
        integer_counts.push((
            element.clone(),
            numerator
                .checked_mul(factor)
                .ok_or(ExtractionExecutorError::CompositionOverflow)?,
        ));
    }
    let divisor = integer_counts
        .iter()
        .map(|(_, count)| *count)
        .reduce(gcd_u128)
        .ok_or_else(|| ExtractionExecutorError::InvalidCompositionFormula(formula.to_string()))?;
    let canonical = integer_counts
        .iter()
        .map(|(element, count)| format!("{element}:{}", count / divisor))
        .collect::<Vec<_>>()
        .join("|");
    Ok((
        sha256_hex(canonical.as_bytes()),
        integer_counts
            .into_iter()
            .map(|(element, _)| element)
            .collect(),
    ))
}

fn decimal_ratio(text: &str) -> Result<(u128, u128), ExtractionExecutorError> {
    let original = text.trim();
    if original.is_empty() || original.starts_with('-') {
        return Err(ExtractionExecutorError::InvalidCompositionNumber(
            original.to_string(),
        ));
    }
    let unsigned = original.strip_prefix('+').unwrap_or(original);
    let (mantissa, exponent) = match unsigned.find(|ch| ch == 'e' || ch == 'E') {
        Some(index) => {
            let exponent = unsigned[index + 1..]
                .parse::<i32>()
                .map_err(|_| {
                    ExtractionExecutorError::InvalidCompositionNumber(original.to_string())
                })?;
            (&unsigned[..index], exponent)
        }
        None => (unsigned, 0),
    };
    let mut digits = String::new();
    let mut fractional_digits = 0_u32;
    let mut seen_decimal = false;
    for ch in mantissa.chars() {
        match ch {
            '0'..='9' => {
                digits.push(ch);
                if seen_decimal {
                    fractional_digits += 1;
                }
            }
            '.' if !seen_decimal => seen_decimal = true,
            _ => {
                return Err(ExtractionExecutorError::InvalidCompositionNumber(
                    original.to_string(),
                ));
            }
        }
    }
    if digits.is_empty() || fractional_digits > 30 || exponent.unsigned_abs() > 30 {
        return Err(ExtractionExecutorError::InvalidCompositionNumber(
            original.to_string(),
        ));
    }
    let mut numerator = digits
        .parse::<u128>()
        .map_err(|_| ExtractionExecutorError::CompositionOverflow)?;
    let mut denominator = checked_pow10(fractional_digits)?;
    if exponent > 0 {
        numerator = numerator
            .checked_mul(checked_pow10(exponent as u32)?)
            .ok_or(ExtractionExecutorError::CompositionOverflow)?;
    } else if exponent < 0 {
        denominator = denominator
            .checked_mul(checked_pow10((-exponent) as u32)?)
            .ok_or(ExtractionExecutorError::CompositionOverflow)?;
    }
    let divisor = gcd_u128(numerator, denominator);
    Ok((numerator / divisor, denominator / divisor))
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct CanonicalOccupant {
    element: String,
    oxidation_state: Option<i32>,
    occupancy_q1e8: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct CanonicalSite {
    fractional_q1e8: [i64; 3],
    occupants: Vec<CanonicalOccupant>,
}

#[derive(Serialize)]
struct CanonicalStructure {
    contract_id: &'static str,
    lattice_q1e8: [[i64; 3]; 3],
    sites: Vec<CanonicalSite>,
}

fn canonicalize_structure(
    entry_id: u64,
    structure: &RawQmpyStructure,
) -> Result<String, ExtractionExecutorError> {
    let mut lattice = [[0_i64; 3]; 3];
    for (row_index, row) in structure.lattice.iter().enumerate() {
        for (column_index, value) in row.iter().enumerate() {
            lattice[row_index][column_index] = quantize(parse_finite(value)?)?;
        }
    }
    let mut sites = Vec::with_capacity(structure.sites.len());
    for site in &structure.sites {
        let mut coordinate = [0_i64; 3];
        for (index, value) in site.fractional_coordinate.iter().enumerate() {
            let wrapped = parse_finite(value)?.rem_euclid(1.0);
            let mut quantized = quantize(wrapped)?;
            if quantized == STRUCTURE_SCALE as i64 {
                quantized = 0;
            }
            coordinate[index] = quantized;
        }
        let mut occupants = site
            .occupants
            .iter()
            .map(|occupant| {
                let occupancy = parse_finite(&occupant.occupancy)?;
                if !(occupancy > 0.0 && occupancy <= 1.0) {
                    return Err(ExtractionExecutorError::InvalidOccupancy(entry_id));
                }
                Ok(CanonicalOccupant {
                    element: occupant.element.clone(),
                    oxidation_state: occupant.oxidation_state,
                    occupancy_q1e8: quantize(occupancy)?,
                })
            })
            .collect::<Result<Vec<_>, ExtractionExecutorError>>()?;
        occupants.sort();
        sites.push(CanonicalSite {
            fractional_q1e8: coordinate,
            occupants,
        });
    }
    sites.sort();
    Ok(sha256_hex(&serde_json::to_vec(&CanonicalStructure {
        contract_id: STRUCTURE_CANONICALIZER_ID,
        lattice_q1e8: lattice,
        sites,
    })?))
}

fn quantize(value: f64) -> Result<i64, ExtractionExecutorError> {
    let scaled = value * STRUCTURE_SCALE;
    if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
        return Err(ExtractionExecutorError::StructureQuantizationOverflow);
    }
    Ok(scaled.round() as i64)
}

fn parse_finite(text: &str) -> Result<f64, ExtractionExecutorError> {
    let value = text
        .parse::<f64>()
        .map_err(|_| ExtractionExecutorError::InvalidNumericText(text.to_string()))?;
    if !value.is_finite() {
        return Err(ExtractionExecutorError::InvalidNumericText(text.to_string()));
    }
    Ok(value)
}

fn component_implementation_sha(contract_id: &str, executable_sha256: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(contract_id.as_bytes());
    digest.update([0]);
    digest.update(executable_sha256.as_bytes());
    format!("{:x}", digest.finalize())
}

fn serialized_plan_sha(plan: &QmpyExtractionPlan) -> Result<String, ExtractionExecutorError> {
    Ok(sha256_hex(&serde_json::to_vec(plan)?))
}

fn checked_pow10(exponent: u32) -> Result<u128, ExtractionExecutorError> {
    let mut value = 1_u128;
    for _ in 0..exponent {
        value = value
            .checked_mul(10)
            .ok_or(ExtractionExecutorError::CompositionOverflow)?;
    }
    Ok(value)
}

fn checked_lcm(a: u128, b: u128) -> Result<u128, ExtractionExecutorError> {
    let divisor = gcd_u128(a, b);
    a.checked_div(divisor)
        .and_then(|reduced| reduced.checked_mul(b))
        .ok_or(ExtractionExecutorError::CompositionOverflow)
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

fn hash_file(path: &Path) -> Result<(String, u64), ExtractionExecutorError> {
    let file = File::open(path).map_err(ExtractionExecutorError::Io)?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut chunk = [0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut chunk).map_err(ExtractionExecutorError::Io)?;
        if read == 0 {
            break;
        }
        digest.update(&chunk[..read]);
        bytes = bytes
            .checked_add(read as u64)
            .ok_or(ExtractionExecutorError::ByteCountOverflow)?;
    }
    Ok((format!("{:x}", digest.finalize()), bytes))
}

fn absolute_utf8(path: &Path) -> Result<String, ExtractionExecutorError> {
    if !path.is_absolute() {
        return Err(ExtractionExecutorError::PathNotAbsolute(
            path.display().to_string(),
        ));
    }
    path.to_str()
        .map(str::to_string)
        .ok_or(ExtractionExecutorError::NonUtf8Path)
}

fn validate_sha256(value: &str) -> Result<(), ExtractionExecutorError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ExtractionExecutorError::InvalidSha256(value.to_string()));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Executable extraction failure or scientific refusal.
#[derive(Debug, Error)]
pub enum ExtractionExecutorError {
    /// Plan schema unsupported.
    #[error("unsupported qmpy extraction plan schema {0}")]
    UnsupportedPlanSchema(u32),
    /// Raw transport schema unsupported.
    #[error("unsupported qmpy raw-row schema {0}")]
    UnsupportedRawSchema(u32),
    /// Multiplicity schema unsupported.
    #[error("unsupported source-multiplicity schema {0}")]
    UnsupportedMultiplicitySchema(u32),
    /// Frozen canonicalizer IDs differ from this implementation.
    #[error("historical extraction canonicalizer contract mismatch")]
    CanonicalizerContractMismatch,
    /// qmpy version/commit differs from reviewed source semantics.
    #[error("unexpected qmpy version/source commit")]
    UnexpectedQmpyIdentity,
    /// Imported database name differs.
    #[error("qmpy extraction database name differs from verified import")]
    DatabaseNameMismatch,
    /// Database user missing.
    #[error("qmpy extraction database user cannot be empty")]
    EmptyDatabaseUser,
    /// Execution bounds invalid.
    #[error("qmpy extraction execution bounds must be nonzero")]
    InvalidExecutionBounds,
    /// Path must be absolute.
    #[error("required path is not absolute: {0}")]
    PathNotAbsolute(String),
    /// Path not UTF-8.
    #[error("required path is not UTF-8")]
    NonUtf8Path,
    /// Local artifact changed after preregistration.
    #[error("local extraction artifact digest mismatch: {0}")]
    ArtifactDigestMismatch(String),
    /// Extraction environment differs from import environment.
    #[error("qmpy extraction environment manifest differs from import profile")]
    EnvironmentManifestMismatch,
    /// File-I/O launcher bytes differ from the preregistered artifact.
    #[error("SIM-PROC file-I/O launcher differs from preregistered identity")]
    FileIoLauncherMismatch,
    /// Adapter stdin is not exact empty bytes.
    #[error("qmpy adapter stdin must be the exact empty artifact")]
    AdapterStdinNotEmpty,
    /// Source primary key invalid.
    #[error("invalid qmpy source identifier for entry {0}")]
    InvalidSourceId(u64),
    /// Required source field empty.
    #[error("required qmpy source field empty for entry {0}")]
    EmptySourceField(u64),
    /// Non-standard fit observed.
    #[error("qmpy adapter emitted non-standard fit {0}")]
    UnexpectedFit(String),
    /// qmpy self-duplicate convention was not normalized away.
    #[error("qmpy entry {0} contains a self duplicate link")]
    SelfDuplicate(u64),
    /// Element set malformed.
    #[error("invalid qmpy element set for entry {0}")]
    InvalidElementSet(u64),
    /// Element outside frozen space.
    #[error("qmpy entry {0} contains an element outside Fe/Co/Zr")]
    ElementOutsideProtocol(u64),
    /// Invalid atom count.
    #[error("qmpy entry {0} has invalid atom count")]
    InvalidAtomCount(u64),
    /// Structure present but lacks sites.
    #[error("qmpy entry {0} structure has no sites")]
    EmptyStructure(u64),
    /// Structure occupant malformed.
    #[error("qmpy entry {0} has malformed site occupant")]
    InvalidStructureOccupant(u64),
    /// Occupancy outside (0,1].
    #[error("qmpy entry {0} has invalid occupancy")]
    InvalidOccupancy(u64),
    /// Invalid/non-finite numeric source text.
    #[error("invalid/non-finite source numeric text: {0}")]
    InvalidNumericText(String),
    /// Composition formula malformed.
    #[error("invalid qmpy composition formula: {0}")]
    InvalidCompositionFormula(String),
    /// Composition amount malformed.
    #[error("invalid qmpy composition amount: {0}")]
    InvalidCompositionNumber(String),
    /// Composition arithmetic overflow.
    #[error("composition canonicalization overflow")]
    CompositionOverflow,
    /// Formula element set differs from qmpy element-set evidence.
    #[error("composition formula/element-set mismatch for entry {0}")]
    CompositionElementMismatch(u64),
    /// Structure quantization overflow.
    #[error("structure coordinate/lattice quantization overflow")]
    StructureQuantizationOverflow,
    /// Duplicate source FormationEnergy ID.
    #[error("duplicate qmpy FormationEnergy id {0}")]
    DuplicateFormationEnergyId(u64),
    /// Empty raw extraction.
    #[error("qmpy raw extraction is empty")]
    EmptyRawExtract,
    /// Empty NDJSON line.
    #[error("empty qmpy NDJSON line {0}")]
    EmptyNdjsonLine(usize),
    /// Raw rows not ordered by `(entry_id, formation_energy_id)`.
    #[error("qmpy raw extraction order is noncanonical")]
    NonCanonicalRawOrder,
    /// Stored multiplicity report malformed/noncanonical.
    #[error("source-multiplicity report is noncanonical")]
    NonCanonicalMultiplicityReport,
    /// Stored raw execution does not bind the supplied plan/state/qmpy artifact.
    #[error("qmpy raw execution does not match extraction plan/state")]
    AdapterPlanMismatch,
    /// Adapter target did not complete successfully.
    #[error("qmpy adapter process did not complete successfully")]
    AdapterProcessDidNotSucceed,
    /// Adapter stderr exceeded bound.
    #[error("qmpy adapter diagnostics were truncated")]
    AdapterDiagnosticsTruncated,
    /// Adapter produced no output.
    #[error("qmpy adapter output missing")]
    MissingAdapterOutput,
    /// Adapter output mutated or differs from captured identity.
    #[error("qmpy adapter output identity mismatch")]
    AdapterOutputIdentityMismatch,
    /// Frozen one-record-per-entry contract cannot select among source rows.
    #[error("qmpy source multiplicity is ambiguous; report {report_sha256}")]
    AmbiguousSourceRows {
        /// Multiplicity-report identity.
        report_sha256: String,
    },
    /// SHA text malformed.
    #[error("invalid SHA-256: {0}")]
    InvalidSha256(String),
    /// Byte count overflow.
    #[error("file byte count overflowed u64")]
    ByteCountOverflow,
    /// Protocol layer failed.
    #[error("historical extraction protocol failure: {0}")]
    Protocol(String),
    /// Acquisition layer failed.
    #[error("historical acquisition failure: {0}")]
    Acquisition(String),
    /// Import layer failed.
    #[error("historical import failure: {0}")]
    Import(String),
    /// Imported database state failed validation.
    #[error("imported database state failure: {0}")]
    DatabaseState(String),
    /// Frozen historical extraction layer failed.
    #[error("historical extraction failure: {0}")]
    Historical(String),
    /// SIM-PROC stdin layer failed.
    #[error("stdin binding failure: {0}")]
    Stdin(String),
    /// SIM-PROC file-I/O layer failed.
    #[error("file-I/O binding failure: {0}")]
    FileIo(String),
    /// File I/O failed.
    #[error("file I/O failure: {0}")]
    Io(#[source] std::io::Error),
    /// JSON failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(entry_id: u64, formation_energy_id: u64) -> RawQmpyFormationEnergyRecord {
        RawQmpyFormationEnergyRecord {
            schema_version: 1,
            formation_energy_id,
            entry_id,
            duplicate_entry_id: None,
            name: "Co2 Fe1".to_string(),
            composition_formula: "Co2 Fe1".to_string(),
            element_set: vec!["Co".to_string(), "Fe".to_string()],
            prototype: None,
            natoms: 3,
            ntypes: 2,
            delta_e_ev_atom: Some("-0.5".to_string()),
            stability_ev_atom: Some("0.01".to_string()),
            band_gap_ev: Some("0".to_string()),
            calculation_id: formation_energy_id + 100,
            calculation_label: Some("static".to_string()),
            fit: "standard".to_string(),
            icsd_id: None,
            structure: None,
        }
    }

    #[test]
    fn reduced_integer_stoichiometry_is_scale_invariant() {
        let (left, left_elements) = canonicalize_composition("Co2 Fe1 Zr1").unwrap();
        let (right, right_elements) = canonicalize_composition("Co4 Fe2 Zr2").unwrap();
        assert_eq!(left, right);
        assert_eq!(left_elements, right_elements);
    }

    #[test]
    fn decimal_stoichiometry_reduces_exactly() {
        let (left, _) = canonicalize_composition("Co0.5 Fe1.0 Zr1.5").unwrap();
        let (right, _) = canonicalize_composition("Co1 Fe2 Zr3").unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn structure_identity_ignores_site_and_occupant_order() {
        let site_a = RawQmpySite {
            fractional_coordinate: ["0".into(), "0".into(), "0".into()],
            occupants: vec![RawSiteOccupant {
                element: "Fe".into(),
                occupancy: "1.0".into(),
                oxidation_state: None,
            }],
        };
        let site_b = RawQmpySite {
            fractional_coordinate: ["0.5".into(), "0.5".into(), "0.5".into()],
            occupants: vec![
                RawSiteOccupant {
                    element: "Co".into(),
                    occupancy: "0.5".into(),
                    oxidation_state: None,
                },
                RawSiteOccupant {
                    element: "Zr".into(),
                    occupancy: "0.5".into(),
                    oxidation_state: None,
                },
            ],
        };
        let lattice = [
            ["3".into(), "0".into(), "0".into()],
            ["0".into(), "3".into(), "0".into()],
            ["0".into(), "0".into(), "3".into()],
        ];
        let first = RawQmpyStructure {
            lattice: lattice.clone(),
            sites: vec![site_a.clone(), site_b.clone()],
            spacegroup: None,
        };
        let mut reversed_b = site_b;
        reversed_b.occupants.reverse();
        let second = RawQmpyStructure {
            lattice,
            sites: vec![reversed_b, site_a],
            spacegroup: Some("ignored-for-coordinate-identity".into()),
        };
        assert_eq!(
            canonicalize_structure(1, &first).unwrap(),
            canonicalize_structure(1, &second).unwrap()
        );
    }

    #[test]
    fn source_multiplicity_is_never_silently_collapsed() {
        let report = source_multiplicity_report(&[raw(7, 70), raw(7, 71), raw(8, 80)]).unwrap();
        assert!(!report.is_unambiguous());
        assert_eq!(report.ambiguous_entries.len(), 1);
        assert_eq!(report.ambiguous_entries[0].entry_id, 7);
        assert_eq!(report.ambiguous_entries[0].formation_energy_ids, vec![70, 71]);
        assert_eq!(report.report_sha256().unwrap().len(), 64);
    }
}
