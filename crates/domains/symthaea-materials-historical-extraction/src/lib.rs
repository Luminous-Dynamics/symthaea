// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preregistered historical-corpus extraction for materials rediscovery benchmarks.
//!
//! This crate freezes *how* a historical database snapshot will be reduced into a
//! compact benchmark corpus before model scoring. Extraction does not decide whether
//! a benchmark target is uncontaminated; that authority remains in
//! `symthaea-materials-corpus-audit`.
//!
//! The first protocol is OQMD v1.7 -> Fe/Co/Zr. It deliberately preserves every
//! unary, binary, and ternary source record whose non-empty element set is a subset
//! of `{Co, Fe, Zr}`. No energy, hull-distance, structure, or property threshold may
//! be applied before the compact corpus is frozen and audited.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};
use symthaea_materials_corpus_audit::{HistoricalCorpusRecord, PropertyLabelRef};
use thiserror::Error;

/// Precision actually established for a historical database release date.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalReleaseDate {
    /// Only a year and month are established by the source.
    YearMonth {
        /// Gregorian year.
        year: u16,
        /// Gregorian month, 1-12.
        month: u8,
    },
    /// An exact Gregorian date is established by the source.
    FullDate {
        /// Gregorian year.
        year: u16,
        /// Gregorian month, 1-12.
        month: u8,
        /// Gregorian day, 1-31. Calendar-day validity is checked conservatively.
        day: u8,
    },
}

impl HistoricalReleaseDate {
    fn validate(&self) -> Result<(), HistoricalExtractionError> {
        let (year, month, day) = match self {
            Self::YearMonth { year, month } => (*year, *month, None),
            Self::FullDate { year, month, day } => (*year, *month, Some(*day)),
        };
        if year < 1900 || !(1..=12).contains(&month) {
            return Err(HistoricalExtractionError::InvalidReleaseDate);
        }
        if day.is_some_and(|day| !(1..=31).contains(&day)) {
            return Err(HistoricalExtractionError::InvalidReleaseDate);
        }
        Ok(())
    }
}

/// OQMD fields/semantic inputs required in the normalized historical extract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OqmdSourceField {
    /// OQMD entry identifier.
    EntryId,
    /// Source compound/formula name.
    Name,
    /// Exact source composition from which the canonical element set is derived.
    Composition,
    /// Preferred duplicate entry link when OQMD supplies one.
    DuplicateEntryId,
    /// Space-group label.
    Spacegroup,
    /// Structure prototype label.
    Prototype,
    /// Number of atoms in the source cell.
    Natoms,
    /// Number of element types.
    Ntypes,
    /// Atomic sites/species/coordinates needed for canonical structure identity.
    Sites,
    /// Unit-cell vectors needed for canonical structure identity.
    UnitCell,
    /// OQMD formation energy per atom (`delta_e`).
    FormationEnergyPerAtom,
    /// OQMD hull distance (`stability`).
    StabilityPerAtom,
    /// Band gap when present.
    BandGap,
    /// OQMD calculation label.
    CalculationLabel,
    /// OQMD thermodynamic fit identifier.
    Fit,
    /// ICSD identifier when present.
    IcsdId,
}

/// Which phase-space rows enter the raw compact corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementScopePolicy {
    /// Include every record with a non-empty element set that is a subset of the allowed set.
    ///
    /// For `{Co, Fe, Zr}` this preserves unary, binary, and ternary phases needed for
    /// thermodynamic context rather than keeping ternaries only.
    AllNonemptySubsets,
}

/// Duplicate handling before contamination analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DuplicatePolicy {
    /// Preserve every source entry and retain the source's preferred-duplicate link.
    PreserveAllWithPreferredLink,
}

/// Scientific-value filtering permitted before corpus freeze.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefilterPolicy {
    /// Apply no energy, stability, band-gap, structure-quality, or property-value threshold.
    NoneBeforeCorpusFreeze,
}

/// Stable ordering of normalized source records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusOrderingPolicy {
    /// Sort strictly by immutable OQMD entry ID ascending.
    EntryIdAscending,
}

/// Preregistered extraction protocol independent of the eventual downloaded bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OqmdExtractionProtocol {
    /// Protocol schema version.
    pub schema_version: u32,
    /// Source provider identifier.
    pub provider: String,
    /// Historical OQMD database version.
    pub database_version: String,
    /// Documented compressed dump filename.
    pub dump_filename: String,
    /// Release date at exactly the precision supported by the source.
    pub release_date: HistoricalReleaseDate,
    /// Database engine documented for the dump.
    pub database_engine: String,
    /// qmpy API version documented as compatible with this database release.
    pub qmpy_api_version: String,
    /// Allowed element symbols in canonical lexical order.
    pub allowed_elements: Vec<String>,
    /// Phase-space inclusion policy.
    pub element_scope: ElementScopePolicy,
    /// Required source fields for deterministic normalized records.
    pub required_fields: Vec<OqmdSourceField>,
    /// Source-duplicate handling.
    pub duplicate_policy: DuplicatePolicy,
    /// Pre-freeze property filtering policy.
    pub prefilter_policy: PrefilterPolicy,
    /// Stable corpus ordering rule.
    pub ordering_policy: CorpusOrderingPolicy,
    /// Composition canonicalizer contract identifier.
    pub composition_canonicalizer_id: String,
    /// Structure canonicalizer contract identifier.
    pub structure_canonicalizer_id: String,
    /// Property-condition canonicalization contract identifier.
    pub property_condition_policy_id: String,
    /// Historical data-license identifier.
    pub source_license: String,
    /// Source page supporting the data-license assertion.
    pub source_license_url: String,
    /// Documented source download page or immutable source locator.
    pub source_url: String,
}

impl OqmdExtractionProtocol {
    /// Validate protocol invariants before acquisition/extraction begins.
    pub fn validate(&self) -> Result<(), HistoricalExtractionError> {
        if self.schema_version != 1 {
            return Err(HistoricalExtractionError::UnsupportedProtocolSchema(
                self.schema_version,
            ));
        }
        for (name, value) in [
            ("provider", &self.provider),
            ("database_version", &self.database_version),
            ("dump_filename", &self.dump_filename),
            ("database_engine", &self.database_engine),
            ("qmpy_api_version", &self.qmpy_api_version),
            ("composition_canonicalizer_id", &self.composition_canonicalizer_id),
            ("structure_canonicalizer_id", &self.structure_canonicalizer_id),
            ("property_condition_policy_id", &self.property_condition_policy_id),
            ("source_license", &self.source_license),
            ("source_license_url", &self.source_license_url),
            ("source_url", &self.source_url),
        ] {
            nonempty(name, value)?;
        }
        self.release_date.validate()?;
        if !self.database_engine.eq_ignore_ascii_case("mysql") {
            return Err(HistoricalExtractionError::UnexpectedDatabaseEngine(
                self.database_engine.clone(),
            ));
        }
        validate_sorted_unique_elements(&self.allowed_elements)?;
        validate_required_fields(&self.required_fields)?;
        Ok(())
    }

    /// Deterministic SHA-256 commitment to the complete preregistered protocol.
    pub fn protocol_sha256(&self) -> Result<String, HistoricalExtractionError> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Canonical first historical protocol: OQMD v1.7, all Fe/Co/Zr subset phases.
pub fn oqmd_v17_fe_co_zr_protocol() -> OqmdExtractionProtocol {
    OqmdExtractionProtocol {
        schema_version: 1,
        provider: "OQMD".to_string(),
        database_version: "1.7".to_string(),
        dump_filename: "qmdb__v1_7__052025.sql.gz".to_string(),
        release_date: HistoricalReleaseDate::YearMonth {
            year: 2025,
            month: 5,
        },
        database_engine: "MySQL".to_string(),
        qmpy_api_version: "1.4".to_string(),
        allowed_elements: vec!["Co".to_string(), "Fe".to_string(), "Zr".to_string()],
        element_scope: ElementScopePolicy::AllNonemptySubsets,
        required_fields: vec![
            OqmdSourceField::EntryId,
            OqmdSourceField::Name,
            OqmdSourceField::Composition,
            OqmdSourceField::DuplicateEntryId,
            OqmdSourceField::Spacegroup,
            OqmdSourceField::Prototype,
            OqmdSourceField::Natoms,
            OqmdSourceField::Ntypes,
            OqmdSourceField::Sites,
            OqmdSourceField::UnitCell,
            OqmdSourceField::FormationEnergyPerAtom,
            OqmdSourceField::StabilityPerAtom,
            OqmdSourceField::BandGap,
            OqmdSourceField::CalculationLabel,
            OqmdSourceField::Fit,
            OqmdSourceField::IcsdId,
        ],
        duplicate_policy: DuplicatePolicy::PreserveAllWithPreferredLink,
        prefilter_policy: PrefilterPolicy::NoneBeforeCorpusFreeze,
        ordering_policy: CorpusOrderingPolicy::EntryIdAscending,
        composition_canonicalizer_id: "reduced-integer-stoichiometry-v1".to_string(),
        structure_canonicalizer_id: "species-lattice-fractional-sites-v1".to_string(),
        property_condition_policy_id: "oqmd-dft-method-condition-v1".to_string(),
        source_license: "CC-BY-4.0".to_string(),
        source_license_url: "https://oqmd.org/documentation/overview".to_string(),
        source_url: "https://oqmd.org/download/".to_string(),
    }
}

/// One deterministic normalized OQMD source row before benchmark contamination analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedOqmdRecord {
    /// Immutable OQMD entry identifier.
    pub entry_id: u64,
    /// Source name/formula.
    pub name: String,
    /// Canonical element symbols derived from exact composition, in lexical order.
    pub element_set: Vec<String>,
    /// Exact canonical composition digest.
    pub composition_sha256: String,
    /// Exact canonical structure digest when structure identity is resolvable.
    pub structure_sha256: Option<String>,
    /// OQMD preferred duplicate entry when supplied.
    pub duplicate_entry_id: Option<u64>,
    /// Space-group label when supplied.
    pub spacegroup: Option<String>,
    /// Structure prototype label when supplied.
    pub prototype: Option<String>,
    /// Number of atoms in the source cell.
    pub natoms: u32,
    /// Number of element types.
    pub ntypes: u16,
    /// Canonical finite decimal text for formation energy in eV/atom.
    pub delta_e_ev_atom: Option<String>,
    /// Canonical finite decimal text for hull distance in eV/atom.
    pub stability_ev_atom: Option<String>,
    /// Canonical finite decimal text for band gap in eV.
    pub band_gap_ev: Option<String>,
    /// Source calculation label.
    pub calculation_label: Option<String>,
    /// Source thermodynamic fit identifier.
    pub fit: Option<String>,
    /// ICSD identifier when supplied.
    pub icsd_id: Option<String>,
    /// Exact condition/method signature attached to source-derived property labels.
    pub property_condition_signature: String,
}

impl NormalizedOqmdRecord {
    /// Validate this row against the preregistered extraction protocol.
    pub fn validate_for(
        &self,
        protocol: &OqmdExtractionProtocol,
    ) -> Result<(), HistoricalExtractionError> {
        protocol.validate()?;
        if self.entry_id == 0 {
            return Err(HistoricalExtractionError::InvalidEntryId(self.entry_id));
        }
        nonempty("name", &self.name)?;
        validate_sorted_unique_elements(&self.element_set)?;
        let allowed: BTreeSet<&str> = protocol.allowed_elements.iter().map(String::as_str).collect();
        if self
            .element_set
            .iter()
            .any(|element| !allowed.contains(element.as_str()))
        {
            return Err(HistoricalExtractionError::ElementOutsideProtocol {
                entry_id: self.entry_id,
            });
        }
        if self.ntypes as usize != self.element_set.len() || self.natoms == 0 {
            return Err(HistoricalExtractionError::InvalidStructureCounts {
                entry_id: self.entry_id,
            });
        }
        sha256(&self.composition_sha256)?;
        if let Some(structure) = &self.structure_sha256 {
            sha256(structure)?;
        }
        if self.duplicate_entry_id == Some(self.entry_id) {
            return Err(HistoricalExtractionError::SelfDuplicateLink(self.entry_id));
        }
        for (field, value) in [
            ("delta_e_ev_atom", &self.delta_e_ev_atom),
            ("stability_ev_atom", &self.stability_ev_atom),
            ("band_gap_ev", &self.band_gap_ev),
        ] {
            if let Some(value) = value {
                validate_canonical_decimal(field, value)?;
            }
        }
        if self.delta_e_ev_atom.is_some()
            || self.stability_ev_atom.is_some()
            || self.band_gap_ev.is_some()
        {
            nonempty(
                "property_condition_signature",
                &self.property_condition_signature,
            )?;
        }
        Ok(())
    }

    /// Convert this source row into the narrow record consumed by contamination audit.
    pub fn to_audit_record(&self) -> HistoricalCorpusRecord {
        HistoricalCorpusRecord {
            record_id: format!("oqmd-entry:{}", self.entry_id),
            composition_sha256: self.composition_sha256.clone(),
            structure_sha256: self.structure_sha256.clone(),
            property_labels: self.property_labels(),
        }
    }

    /// Source property labels that are actually present for this exact row.
    ///
    /// No magnetic property is synthesized here. OQMD formation energy, stability,
    /// and band gap are the only labels represented by this v1 normalized record.
    pub fn property_labels(&self) -> Vec<PropertyLabelRef> {
        let mut labels = Vec::new();
        if self.delta_e_ev_atom.is_some() {
            labels.push(PropertyLabelRef {
                property_id: "formation_energy_ev_atom".to_string(),
                condition_signature: self.property_condition_signature.clone(),
            });
        }
        if self.stability_ev_atom.is_some() {
            labels.push(PropertyLabelRef {
                property_id: "hull_distance_ev_atom".to_string(),
                condition_signature: self.property_condition_signature.clone(),
            });
        }
        if self.band_gap_ev.is_some() {
            labels.push(PropertyLabelRef {
                property_id: "band_gap_ev".to_string(),
                condition_signature: self.property_condition_signature.clone(),
            });
        }
        labels
    }
}

/// Frozen compact historical corpus produced before target contamination analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactHistoricalCorpus {
    /// Corpus schema version.
    pub schema_version: u32,
    /// Exact preregistered extraction protocol digest.
    pub protocol_sha256: String,
    /// All qualifying records ordered by OQMD entry ID ascending.
    pub records: Vec<NormalizedOqmdRecord>,
}

impl CompactHistoricalCorpus {
    /// Build and validate a compact corpus without target/property-value filtering.
    pub fn from_records(
        protocol: &OqmdExtractionProtocol,
        mut records: Vec<NormalizedOqmdRecord>,
    ) -> Result<Self, HistoricalExtractionError> {
        protocol.validate()?;
        if records.is_empty() {
            return Err(HistoricalExtractionError::EmptyCorpus);
        }
        for record in &records {
            record.validate_for(protocol)?;
        }
        records.sort_by_key(|record| record.entry_id);
        let mut ids = HashSet::new();
        for record in &records {
            if !ids.insert(record.entry_id) {
                return Err(HistoricalExtractionError::DuplicateEntryId(record.entry_id));
            }
        }
        let corpus = Self {
            schema_version: 1,
            protocol_sha256: protocol.protocol_sha256()?,
            records,
        };
        corpus.validate_for(protocol)?;
        Ok(corpus)
    }

    /// Revalidate serialized/deserialized corpus bytes against their exact protocol.
    pub fn validate_for(
        &self,
        protocol: &OqmdExtractionProtocol,
    ) -> Result<(), HistoricalExtractionError> {
        if self.schema_version != 1 {
            return Err(HistoricalExtractionError::UnsupportedCorpusSchema(
                self.schema_version,
            ));
        }
        let expected_protocol = protocol.protocol_sha256()?;
        if self.protocol_sha256 != expected_protocol {
            return Err(HistoricalExtractionError::ProtocolCorpusMismatch);
        }
        if self.records.is_empty() {
            return Err(HistoricalExtractionError::EmptyCorpus);
        }
        let mut previous = None;
        for record in &self.records {
            record.validate_for(protocol)?;
            if previous.is_some_and(|prior| record.entry_id <= prior) {
                return Err(HistoricalExtractionError::NonCanonicalRecordOrder);
            }
            previous = Some(record.entry_id);
        }
        Ok(())
    }

    /// Deterministic digest of the complete compact corpus after full revalidation.
    pub fn corpus_sha256(
        &self,
        protocol: &OqmdExtractionProtocol,
    ) -> Result<String, HistoricalExtractionError> {
        self.validate_for(protocol)?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }

    /// Convert all records for the separate MAG-DATA-002 contamination audit.
    pub fn audit_records(
        &self,
        protocol: &OqmdExtractionProtocol,
    ) -> Result<Vec<HistoricalCorpusRecord>, HistoricalExtractionError> {
        self.validate_for(protocol)?;
        Ok(self
            .records
            .iter()
            .map(NormalizedOqmdRecord::to_audit_record)
            .collect())
    }
}

/// Evidence binding one actual historical dump import/extraction to its preregistered protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalExtractionReceipt {
    /// Receipt schema version.
    pub schema_version: u32,
    /// Extraction protocol commitment frozen before scoring.
    pub protocol_sha256: String,
    /// Exact compressed historical SQL dump bytes.
    pub compressed_dump_sha256: String,
    /// Exact MySQL/Nix/container import environment artifact.
    pub import_environment_sha256: String,
    /// Exact imported-schema inventory artifact.
    pub schema_inventory_sha256: String,
    /// Exact import log/receipt artifact.
    pub import_receipt_sha256: String,
    /// Exact extractor implementation artifact.
    pub extractor_artifact_sha256: String,
    /// Exact SQL/query artifact emitted/used by the extractor.
    pub extraction_query_sha256: String,
    /// Exact qmpy v1.4 package/source artifact used for schema semantics, if any.
    pub qmpy_artifact_sha256: String,
    /// Exact composition canonicalizer implementation artifact.
    pub composition_canonicalizer_sha256: String,
    /// Exact structure canonicalizer implementation artifact.
    pub structure_canonicalizer_sha256: String,
    /// Exact compact corpus digest.
    pub compact_corpus_sha256: String,
    /// Number of preserved normalized source records.
    pub record_count: u64,
}

impl HistoricalExtractionReceipt {
    /// Deterministic receipt identity.
    pub fn receipt_sha256(&self) -> Result<String, HistoricalExtractionError> {
        if self.schema_version != 1 {
            return Err(HistoricalExtractionError::UnsupportedReceiptSchema(
                self.schema_version,
            ));
        }
        for digest in [
            &self.protocol_sha256,
            &self.compressed_dump_sha256,
            &self.import_environment_sha256,
            &self.schema_inventory_sha256,
            &self.import_receipt_sha256,
            &self.extractor_artifact_sha256,
            &self.extraction_query_sha256,
            &self.qmpy_artifact_sha256,
            &self.composition_canonicalizer_sha256,
            &self.structure_canonicalizer_sha256,
            &self.compact_corpus_sha256,
        ] {
            sha256(digest)?;
        }
        if self.record_count == 0 {
            return Err(HistoricalExtractionError::EmptyCorpus);
        }
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Bind an executed historical extraction to exact source/environment/tool artifacts.
#[allow(clippy::too_many_arguments)]
pub fn bind_extraction_receipt(
    protocol: &OqmdExtractionProtocol,
    corpus: &CompactHistoricalCorpus,
    compressed_dump_sha256: &str,
    import_environment_sha256: &str,
    schema_inventory_sha256: &str,
    import_receipt_sha256: &str,
    extractor_artifact_sha256: &str,
    extraction_query_sha256: &str,
    qmpy_artifact_sha256: &str,
    composition_canonicalizer_sha256: &str,
    structure_canonicalizer_sha256: &str,
) -> Result<HistoricalExtractionReceipt, HistoricalExtractionError> {
    corpus.validate_for(protocol)?;
    let protocol_sha = protocol.protocol_sha256()?;
    for digest in [
        compressed_dump_sha256,
        import_environment_sha256,
        schema_inventory_sha256,
        import_receipt_sha256,
        extractor_artifact_sha256,
        extraction_query_sha256,
        qmpy_artifact_sha256,
        composition_canonicalizer_sha256,
        structure_canonicalizer_sha256,
    ] {
        sha256(digest)?;
    }
    let receipt = HistoricalExtractionReceipt {
        schema_version: 1,
        protocol_sha256: protocol_sha,
        compressed_dump_sha256: compressed_dump_sha256.to_ascii_lowercase(),
        import_environment_sha256: import_environment_sha256.to_ascii_lowercase(),
        schema_inventory_sha256: schema_inventory_sha256.to_ascii_lowercase(),
        import_receipt_sha256: import_receipt_sha256.to_ascii_lowercase(),
        extractor_artifact_sha256: extractor_artifact_sha256.to_ascii_lowercase(),
        extraction_query_sha256: extraction_query_sha256.to_ascii_lowercase(),
        qmpy_artifact_sha256: qmpy_artifact_sha256.to_ascii_lowercase(),
        composition_canonicalizer_sha256: composition_canonicalizer_sha256.to_ascii_lowercase(),
        structure_canonicalizer_sha256: structure_canonicalizer_sha256.to_ascii_lowercase(),
        compact_corpus_sha256: corpus.corpus_sha256(protocol)?,
        record_count: corpus.records.len() as u64,
    };
    receipt.receipt_sha256()?;
    Ok(receipt)
}

/// Convert one finite number into the deterministic decimal form required by v1 records.
pub fn canonical_decimal(value: f64) -> Result<String, HistoricalExtractionError> {
    if !value.is_finite() {
        return Err(HistoricalExtractionError::NonFiniteDecimal);
    }
    Ok(value.to_string())
}

fn validate_canonical_decimal(
    field: &'static str,
    value: &str,
) -> Result<(), HistoricalExtractionError> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| HistoricalExtractionError::InvalidDecimal {
            field,
            value: value.to_string(),
        })?;
    if !parsed.is_finite() {
        return Err(HistoricalExtractionError::NonFiniteDecimal);
    }
    let canonical = parsed.to_string();
    if canonical != value {
        return Err(HistoricalExtractionError::NonCanonicalDecimal {
            field,
            value: value.to_string(),
            canonical,
        });
    }
    Ok(())
}

fn validate_sorted_unique_elements(elements: &[String]) -> Result<(), HistoricalExtractionError> {
    if elements.is_empty() {
        return Err(HistoricalExtractionError::EmptyElementSet);
    }
    for element in elements {
        nonempty("element", element)?;
    }
    if elements
        .windows(2)
        .any(|window| window[0].as_str() >= window[1].as_str())
    {
        return Err(HistoricalExtractionError::NonCanonicalElementOrder);
    }
    Ok(())
}

fn validate_required_fields(fields: &[OqmdSourceField]) -> Result<(), HistoricalExtractionError> {
    let present: BTreeSet<OqmdSourceField> = fields.iter().copied().collect();
    if present.len() != fields.len() {
        return Err(HistoricalExtractionError::DuplicateRequiredField);
    }
    let required = [
        OqmdSourceField::EntryId,
        OqmdSourceField::Name,
        OqmdSourceField::Composition,
        OqmdSourceField::DuplicateEntryId,
        OqmdSourceField::Spacegroup,
        OqmdSourceField::Prototype,
        OqmdSourceField::Natoms,
        OqmdSourceField::Ntypes,
        OqmdSourceField::Sites,
        OqmdSourceField::UnitCell,
        OqmdSourceField::FormationEnergyPerAtom,
        OqmdSourceField::StabilityPerAtom,
        OqmdSourceField::BandGap,
        OqmdSourceField::CalculationLabel,
        OqmdSourceField::Fit,
        OqmdSourceField::IcsdId,
    ];
    if required.iter().any(|field| !present.contains(field)) {
        return Err(HistoricalExtractionError::MissingRequiredField);
    }
    Ok(())
}

fn nonempty(name: &'static str, value: &str) -> Result<(), HistoricalExtractionError> {
    if value.trim().is_empty() {
        return Err(HistoricalExtractionError::EmptyField(name));
    }
    Ok(())
}

fn sha256(value: &str) -> Result<(), HistoricalExtractionError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(HistoricalExtractionError::InvalidSha256(value.to_string()));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Historical extraction validation failures.
#[derive(Debug, Error)]
pub enum HistoricalExtractionError {
    /// Protocol schema is unsupported.
    #[error("unsupported extraction protocol schema {0}")]
    UnsupportedProtocolSchema(u32),
    /// Compact corpus schema is unsupported.
    #[error("unsupported compact corpus schema {0}")]
    UnsupportedCorpusSchema(u32),
    /// Receipt schema is unsupported.
    #[error("unsupported extraction receipt schema {0}")]
    UnsupportedReceiptSchema(u32),
    /// Required text field is empty.
    #[error("required field {0} is empty")]
    EmptyField(&'static str),
    /// Release-date metadata is malformed.
    #[error("historical release date is invalid")]
    InvalidReleaseDate,
    /// Dump engine differs from the v1 OQMD contract.
    #[error("unexpected database engine {0}; OQMD v1 historical dumps require MySQL")]
    UnexpectedDatabaseEngine(String),
    /// Element set is empty.
    #[error("element set must be non-empty")]
    EmptyElementSet,
    /// Elements are not strictly lexically ordered/unique.
    #[error("element symbols must be strictly ordered and unique")]
    NonCanonicalElementOrder,
    /// Required source-field list contains a duplicate.
    #[error("required OQMD field list contains duplicates")]
    DuplicateRequiredField,
    /// Required source-field list omits a mandatory field.
    #[error("required OQMD field list omits a mandatory v1 field")]
    MissingRequiredField,
    /// Entry ID is not valid.
    #[error("invalid OQMD entry id {0}")]
    InvalidEntryId(u64),
    /// Record contains an element outside the preregistered phase space.
    #[error("OQMD entry {entry_id} contains an element outside the extraction protocol")]
    ElementOutsideProtocol {
        /// Offending OQMD entry.
        entry_id: u64,
    },
    /// natoms/ntypes do not agree with the normalized row.
    #[error("OQMD entry {entry_id} has invalid atom/type counts")]
    InvalidStructureCounts {
        /// Offending OQMD entry.
        entry_id: u64,
    },
    /// Source duplicate link points to itself.
    #[error("OQMD entry {0} points to itself as preferred duplicate")]
    SelfDuplicateLink(u64),
    /// SHA-256 field is malformed.
    #[error("invalid SHA-256: {0}")]
    InvalidSha256(String),
    /// Decimal cannot be parsed.
    #[error("invalid decimal in {field}: {value}")]
    InvalidDecimal {
        /// Field name.
        field: &'static str,
        /// Source value.
        value: String,
    },
    /// Decimal is finite but not in canonical Rust v1 representation.
    #[error("non-canonical decimal in {field}: {value}; canonical form is {canonical}")]
    NonCanonicalDecimal {
        /// Field name.
        field: &'static str,
        /// Supplied form.
        value: String,
        /// Required deterministic form.
        canonical: String,
    },
    /// Numeric field is NaN or infinite.
    #[error("numeric source property must be finite")]
    NonFiniteDecimal,
    /// No records survived the element-set extraction rule.
    #[error("historical compact corpus is empty")]
    EmptyCorpus,
    /// OQMD entry IDs are not unique.
    #[error("duplicate OQMD entry id {0}")]
    DuplicateEntryId(u64),
    /// Serialized compact records are not strictly ordered by entry ID.
    #[error("compact corpus records are not strictly ordered by OQMD entry id")]
    NonCanonicalRecordOrder,
    /// Corpus was built under a different protocol commitment.
    #[error("compact corpus protocol digest does not match extraction protocol")]
    ProtocolCorpusMismatch,
    /// JSON serialization failed.
    #[error("serialization failure: {0}")]
    Serialization(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn record(entry_id: u64, elements: &[&str]) -> NormalizedOqmdRecord {
        NormalizedOqmdRecord {
            entry_id,
            name: format!("record-{entry_id}"),
            element_set: elements.iter().map(|element| (*element).to_string()).collect(),
            composition_sha256: sha('a'),
            structure_sha256: Some(sha('b')),
            duplicate_entry_id: None,
            spacegroup: Some("Pm-3m".to_string()),
            prototype: None,
            natoms: 2,
            ntypes: elements.len() as u16,
            delta_e_ev_atom: Some("-0.25".to_string()),
            stability_ev_atom: Some("0".to_string()),
            band_gap_ev: None,
            calculation_label: Some("static".to_string()),
            fit: Some("standard".to_string()),
            icsd_id: None,
            property_condition_signature: "oqmd-v1.7|fit=standard|calculation=static".to_string(),
        }
    }

    #[test]
    fn canonical_protocol_freezes_real_v17_source_semantics() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        protocol.validate().unwrap();
        assert_eq!(protocol.database_engine, "MySQL");
        assert_eq!(protocol.qmpy_api_version, "1.4");
        assert_eq!(protocol.source_license, "CC-BY-4.0");
        assert_eq!(
            protocol.release_date,
            HistoricalReleaseDate::YearMonth {
                year: 2025,
                month: 5
            }
        );
        assert_eq!(
            protocol.allowed_elements,
            vec!["Co".to_string(), "Fe".to_string(), "Zr".to_string()]
        );
        assert_eq!(protocol.prefilter_policy, PrefilterPolicy::NoneBeforeCorpusFreeze);
    }

    #[test]
    fn all_subset_phases_are_allowed_but_foreign_elements_fail() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        record(1, &["Fe"]).validate_for(&protocol).unwrap();
        record(2, &["Co", "Fe"]).validate_for(&protocol).unwrap();
        record(3, &["Co", "Fe", "Zr"])
            .validate_for(&protocol)
            .unwrap();
        assert!(matches!(
            record(4, &["Fe", "Ni"]).validate_for(&protocol),
            Err(HistoricalExtractionError::ElementOutsideProtocol { entry_id: 4 })
        ));
    }

    #[test]
    fn compact_corpus_orders_by_entry_and_preserves_duplicate_rows() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let mut later = record(20, &["Fe", "Zr"]);
        later.duplicate_entry_id = Some(10);
        let earlier = record(10, &["Fe", "Zr"]);
        let corpus = CompactHistoricalCorpus::from_records(&protocol, vec![later, earlier]).unwrap();
        assert_eq!(corpus.records.len(), 2);
        assert_eq!(corpus.records[0].entry_id, 10);
        assert_eq!(corpus.records[1].entry_id, 20);
        assert_eq!(corpus.records[1].duplicate_entry_id, Some(10));
        corpus.validate_for(&protocol).unwrap();
    }

    #[test]
    fn deserialized_or_mutated_bad_order_fails_at_consumption_boundary() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let mut corpus = CompactHistoricalCorpus::from_records(
            &protocol,
            vec![record(1, &["Fe"]), record(2, &["Co", "Fe"])],
        )
        .unwrap();
        corpus.records.swap(0, 1);
        assert!(matches!(
            corpus.corpus_sha256(&protocol),
            Err(HistoricalExtractionError::NonCanonicalRecordOrder)
        ));
        assert!(corpus.audit_records(&protocol).is_err());
    }

    #[test]
    fn source_properties_map_to_audit_labels_without_inventing_magnetics() {
        let row = record(1, &["Co", "Fe", "Zr"]);
        let labels = row.property_labels();
        assert_eq!(labels.len(), 2);
        assert!(labels.iter().any(|label| label.property_id == "formation_energy_ev_atom"));
        assert!(labels.iter().any(|label| label.property_id == "hull_distance_ev_atom"));
        assert!(!labels.iter().any(|label| label.property_id.contains("magnet")));
    }

    #[test]
    fn unresolved_structure_remains_unresolved_for_contamination_audit() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let mut row = record(1, &["Co", "Fe"]);
        row.structure_sha256 = None;
        let corpus = CompactHistoricalCorpus::from_records(&protocol, vec![row]).unwrap();
        let audit = corpus.audit_records(&protocol).unwrap();
        assert!(audit[0].structure_sha256.is_none());
    }

    #[test]
    fn numeric_text_must_be_canonical_and_finite() {
        assert_eq!(canonical_decimal(1.0).unwrap(), "1");
        assert!(canonical_decimal(f64::NAN).is_err());
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let mut row = record(1, &["Fe"]);
        row.delta_e_ev_atom = Some("-0.2500".to_string());
        assert!(matches!(
            row.validate_for(&protocol),
            Err(HistoricalExtractionError::NonCanonicalDecimal { .. })
        ));
    }

    #[test]
    fn executed_receipt_binds_dump_environment_query_and_compact_corpus() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let corpus = CompactHistoricalCorpus::from_records(
            &protocol,
            vec![record(1, &["Fe"]), record(2, &["Co", "Fe", "Zr"])],
        )
        .unwrap();
        let receipt = bind_extraction_receipt(
            &protocol,
            &corpus,
            &sha('1'),
            &sha('2'),
            &sha('3'),
            &sha('4'),
            &sha('5'),
            &sha('6'),
            &sha('7'),
            &sha('8'),
            &sha('9'),
        )
        .unwrap();
        assert_eq!(receipt.record_count, 2);
        assert_eq!(
            receipt.compact_corpus_sha256,
            corpus.corpus_sha256(&protocol).unwrap()
        );
        assert_eq!(receipt.protocol_sha256, protocol.protocol_sha256().unwrap());
        assert_ne!(receipt.receipt_sha256().unwrap(), protocol.protocol_sha256().unwrap());
    }
}
