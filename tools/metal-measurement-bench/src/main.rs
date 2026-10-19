use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{Cursor, Write},
    path::{Component, Path, PathBuf},
    process::ExitCode,
};
use thiserror::Error;

type Result<T> = std::result::Result<T, AdapterError>;

#[derive(Debug, Error)]
enum AdapterError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("validation failed: {0}")]
    Validation(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceManifest {
    schema_version: String,
    dataset: DatasetIdentity,
    related_sources: Vec<RelatedSource>,
    source_objects: Vec<SourceObject>,
    challenge_context: ChallengeContext,
    reuse_policy: ReusePolicy,
    artifacts: Vec<SourceArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetIdentity {
    id: String,
    doi: String,
    title: String,
    publisher: String,
    source_url: String,
    repository_url: String,
    descriptive_readme_url: String,
    citation_policy: String,
    source_revision: String,
    metadata_accessed_on: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RelatedSource {
    id: String,
    doi: String,
    role: String,
    url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceObject {
    id: String,
    source_doi: String,
    source_url: String,
    source_revision: String,
    accessed_on: String,
    terms_url: String,
    terms_reviewed: bool,
    local_use_authorized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChallengeContext {
    material: String,
    process: String,
    condition: String,
    test_method: String,
    strain_rate_per_second: f64,
    gauge_length_mm: f64,
    target_specimen_count: usize,
    target_parent_specimen_id: String,
    target_definition: String,
    split_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReusePolicy {
    artifact_terms_reviewed: bool,
    local_use_authorized: bool,
    repository_bundling_authorized: bool,
    status_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceArtifact {
    relative_path: String,
    source_object_id: String,
    source_doi: String,
    artifact_revision: String,
    accessed_on: String,
    source_url: String,
    terms_url: String,
    terms_reviewed: bool,
    local_use_authorized: bool,
    repository_bundling_authorized: bool,
    sha256: String,
    size_bytes: u64,
    media_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationSet {
    schema_version: String,
    records: Vec<Observation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    record_id: String,
    source_doi: String,
    specimen_id: String,
    lineage_group_id: String,
    source_object_id: String,
    source_row_id: String,
    source_field: String,
    variable_id: String,
    role: RecordRole,
    split: Split,
    raw_value: Option<f64>,
    raw_unit: Option<String>,
    censor_lower_bound: Option<f64>,
    censor_upper_bound: Option<f64>,
    normalized_value: Option<f64>,
    normalized_unit: Option<String>,
    conversion_note: Option<String>,
    reported_uncertainty: Option<f64>,
    uncertainty_unit: Option<String>,
    censoring: Censoring,
    metadata_note: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RecordRole {
    Predictor,
    TargetOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Split {
    Calibration,
    HeldOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Censoring {
    Uncensored,
    LeftCensored,
    RightCensored,
    IntervalCensored,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
struct CsvColumnInventory {
    source_header: String,
    column_index_zero_based: usize,
    nonempty_cells: usize,
    empty_cells: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CsvInventory {
    artifact_relative_path: String,
    artifact_sha256: String,
    data_row_count: usize,
    columns: Vec<CsvColumnInventory>,
    row_identity_note: &'static str,
    claim_ceiling: &'static str,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CsvMappingSpec {
    schema_version: String,
    artifact_relative_path: String,
    source_row_id_header: String,
    columns: Vec<CsvColumnMapping>,
    excluded_columns: Vec<CsvExcludedColumn>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CsvExcludedColumn {
    source_header: String,
    reason: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CsvColumnMapping {
    source_header: String,
    canonical_field_id: String,
    raw_unit: String,
    role: CsvFieldRole,
    value_kind: CsvValueKind,
    missing_tokens: Vec<String>,
    missing_policy: CsvMissingPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum CsvFieldRole {
    Metadata,
    Predictor,
    TargetOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum CsvValueKind {
    Text,
    FiniteNumber,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum CsvMissingPolicy {
    Reject,
    PreserveMissing,
}

#[derive(Debug, Clone, Serialize)]
struct CsvMappedColumnReceipt {
    source_header: String,
    canonical_field_id: String,
    raw_unit: String,
    role: CsvFieldRole,
    value_kind: CsvValueKind,
    missing_tokens: Vec<String>,
    missing_token_counts: BTreeMap<String, usize>,
    missing_policy: CsvMissingPolicy,
    nonempty_cells: usize,
    numeric_values_validated: usize,
    preserved_missing_cells: usize,
    preserved_missing_token_cells: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CsvExcludedColumnReceipt {
    source_header: String,
    reason: String,
    nonempty_cells: usize,
    empty_cells: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CsvReconciliationReceipt {
    artifact_relative_path: String,
    artifact_sha256: String,
    mapping_sha256: String,
    headers_sha256: String,
    source_row_id_header: String,
    ordered_source_row_ids_sha256: String,
    data_row_count: usize,
    unique_source_row_ids: usize,
    mapped_column_count: usize,
    columns: Vec<CsvMappedColumnReceipt>,
    excluded_column_count: usize,
    excluded_columns: Vec<CsvExcludedColumnReceipt>,
    reconciliation_note: &'static str,
    claim_ceiling: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct ZipMemberInventory {
    member_path: String,
    compressed_size_bytes: u64,
    uncompressed_size_bytes: u64,
    crc32_hex: String,
    is_directory: bool,
    unix_mode: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
struct ZipInventory {
    artifact_relative_path: String,
    archive_sha256: String,
    member_count: usize,
    total_uncompressed_bytes: u64,
    members: Vec<ZipMemberInventory>,
    inventory_note: &'static str,
    claim_ceiling: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct AuditReceipt {
    receipt_schema_version: &'static str,
    adapter_version: &'static str,
    source_commit: Option<String>,
    operation: &'static str,
    status: &'static str,
    manifest_sha256: String,
    evidence_input_sha256: Option<String>,
    artifact_count: usize,
    record_count: usize,
    held_out_outcome_groups: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    csv_inventory: Option<CsvInventory>,
    #[serde(skip_serializing_if = "Option::is_none")]
    csv_reconciliation: Option<CsvReconciliationReceipt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    zip_inventory: Option<ZipInventory>,
    claim_ceiling: &'static str,
}

fn io_error(path: &Path, source: std::io::Error) -> AdapterError {
    AdapterError::Io {
        path: path.display().to_string(),
        source,
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<(T, Vec<u8>)> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    let value = serde_json::from_slice(&bytes)?;
    Ok((value, bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn validate_csv_quote_syntax(bytes: &[u8]) -> Result<()> {
    let mut index = 0usize;
    let mut in_quotes = false;
    let mut at_field_start = true;
    let mut after_closing_quote = false;

    while index < bytes.len() {
        let byte = bytes[index];
        if in_quotes {
            if byte == b'"' {
                if bytes.get(index + 1) == Some(&b'"') {
                    index += 2;
                    continue;
                }
                in_quotes = false;
                after_closing_quote = true;
            }
            index += 1;
            continue;
        }

        if after_closing_quote {
            match byte {
                b',' => {
                    at_field_start = true;
                    after_closing_quote = false;
                    index += 1;
                }
                b'\n' => {
                    at_field_start = true;
                    after_closing_quote = false;
                    index += 1;
                }
                b'\r' if bytes.get(index + 1) == Some(&b'\n') => {
                    at_field_start = true;
                    after_closing_quote = false;
                    index += 2;
                }
                _ => {
                    return Err(AdapterError::Validation(
                        "CSV has characters after a closing quote before the field delimiter".into(),
                    ));
                }
            }
            continue;
        }

        match byte {
            b'"' if at_field_start => {
                in_quotes = true;
                at_field_start = false;
                index += 1;
            }
            b'"' => {
                return Err(AdapterError::Validation(
                    "CSV quote appears inside an unquoted field".into(),
                ));
            }
            b',' => {
                at_field_start = true;
                index += 1;
            }
            b'\n' => {
                at_field_start = true;
                index += 1;
            }
            b'\r' if bytes.get(index + 1) == Some(&b'\n') => {
                at_field_start = true;
                index += 2;
            }
            _ => {
                at_field_start = false;
                index += 1;
            }
        }
    }

    if in_quotes {
        return Err(AdapterError::Validation(
            "CSV ends inside an unterminated quoted field".into(),
        ));
    }
    Ok(())
}

fn is_portable_zip_component(component: &str) -> bool {
    if component.is_empty()
        || component == "."
        || component == ".."
        || component.ends_with('.')
        || component.ends_with(' ')
    {
        return false;
    }
    let stem = component.split('.').next().unwrap_or_default();
    let upper = stem.to_ascii_uppercase();
    let reserved = matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (upper.len() == 4
            && (upper.starts_with("COM") || upper.starts_with("LPT"))
            && upper.as_bytes()[3].is_ascii_digit()
            && (b'1'..=b'9').contains(&upper.as_bytes()[3]));
    !reserved
}

fn inspect_zip_bytes(bytes: &[u8], relative_path: &str) -> Result<ZipInventory> {
    const MAX_MEMBERS: usize = 100_000;

    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor)?;
    if archive.len() > MAX_MEMBERS {
        return Err(AdapterError::Validation(format!(
            "ZIP archive has {} members, exceeding the inventory limit of {MAX_MEMBERS}",
            archive.len()
        )));
    }

    let mut members = Vec::with_capacity(archive.len());
    let mut exact_names = BTreeSet::new();
    let mut portable_names = BTreeSet::new();
    let mut total_uncompressed_bytes = 0u64;

    for index in 0..archive.len() {
        let member = archive.by_index(index)?;
        let raw_name = std::str::from_utf8(member.name_raw()).map_err(|_| {
            AdapterError::Validation(format!(
                "ZIP member {index} has a non-UTF-8 path; refusing ambiguous portable identity"
            ))
        })?;
        if raw_name.is_empty()
            || raw_name.contains('\\')
            || raw_name.contains(':')
            || raw_name.starts_with('/')
            || raw_name.starts_with("//")
        {
            return Err(AdapterError::Validation(format!(
                "ZIP member path {raw_name:?} is empty, non-portable, or absolute"
            )));
        }

        let path_without_directory_suffix = raw_name.strip_suffix('/').unwrap_or(raw_name);
        if path_without_directory_suffix.is_empty()
            || path_without_directory_suffix
                .split('/')
                .any(|component| !is_portable_zip_component(component))
        {
            return Err(AdapterError::Validation(format!(
                "ZIP member path {raw_name:?} contains empty, traversal, normalized, or non-portable components"
            )));
        }
        let member_path = Path::new(raw_name);
        if member_path.components().any(|component| {
            !matches!(component, Component::Normal(_))
        }) {
            return Err(AdapterError::Validation(format!(
                "ZIP member path {raw_name:?} contains traversal or non-normal path components"
            )));
        }

        let unix_mode = member.unix_mode();
        if unix_mode.is_some_and(|mode| mode & 0o170000 == 0o120000) {
            return Err(AdapterError::Validation(format!(
                "ZIP member {raw_name:?} is a symlink; archives with symlinks are not accepted"
            )));
        }

        if !exact_names.insert(raw_name.to_string()) {
            return Err(AdapterError::Validation(format!(
                "ZIP archive contains duplicate member path {raw_name:?}"
            )));
        }
        let portable_key = raw_name.to_lowercase();
        if !portable_names.insert(portable_key) {
            return Err(AdapterError::Validation(format!(
                "ZIP archive contains case-colliding member path {raw_name:?}"
            )));
        }

        total_uncompressed_bytes = total_uncompressed_bytes
            .checked_add(member.size())
            .ok_or_else(|| AdapterError::Validation(
                "ZIP total uncompressed size overflow".into()
            ))?;

        members.push(ZipMemberInventory {
            member_path: raw_name.to_string(),
            compressed_size_bytes: member.compressed_size(),
            uncompressed_size_bytes: member.size(),
            crc32_hex: format!("{:08x}", member.crc32()),
            is_directory: member.is_dir(),
            unix_mode,
        });
    }

    Ok(ZipInventory {
        artifact_relative_path: relative_path.to_string(),
        archive_sha256: sha256_hex(bytes),
        member_count: members.len(),
        total_uncompressed_bytes,
        members,
        inventory_note: "central-directory inventory only; member content was not extracted or individually cryptographically hashed",
        claim_ceiling: "archive structure and path-safety inventory only; not semantic data validation, source permission, or model-accuracy evidence",
    })
}

fn inspect_csv_bytes(bytes: &[u8], relative_path: &str) -> Result<CsvInventory> {
    validate_csv_quote_syntax(bytes)?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(bytes);

    let headers = reader.headers()?.clone();
    if headers.is_empty() {
        return Err(AdapterError::Validation(
            "CSV must contain a non-empty header row".into(),
        ));
    }

    let mut seen_headers = BTreeSet::new();
    for header in headers.iter() {
        let normalized = header.trim().to_lowercase();
        if normalized.is_empty() {
            return Err(AdapterError::Validation(
                "CSV header names must not be blank".into(),
            ));
        }
        if !seen_headers.insert(normalized.clone()) {
            return Err(AdapterError::Validation(format!(
                "CSV contains duplicate or whitespace/case-colliding header {header:?}"
            )));
        }
    }

    let mut columns: Vec<CsvColumnInventory> = headers
        .iter()
        .enumerate()
        .map(|(index, header)| CsvColumnInventory {
            source_header: header.to_string(),
            column_index_zero_based: index,
            nonempty_cells: 0,
            empty_cells: 0,
        })
        .collect();

    let mut data_row_count = 0usize;
    for row in reader.records() {
        let row = row?;
        data_row_count = data_row_count.checked_add(1).ok_or_else(|| {
            AdapterError::Validation("CSV row count overflow".into())
        })?;
        for (index, cell) in row.iter().enumerate() {
            let column = columns.get_mut(index).ok_or_else(|| {
                AdapterError::Validation("CSV row has more cells than the header".into())
            })?;
            if cell.trim().is_empty() {
                column.empty_cells = column.empty_cells.checked_add(1).ok_or_else(|| {
                    AdapterError::Validation("CSV empty-cell count overflow".into())
                })?;
            } else {
                column.nonempty_cells = column.nonempty_cells.checked_add(1).ok_or_else(|| {
                    AdapterError::Validation("CSV nonempty-cell count overflow".into())
                })?;
            }
        }
    }

    Ok(CsvInventory {
        artifact_relative_path: relative_path.to_string(),
        artifact_sha256: sha256_hex(bytes),
        data_row_count,
        columns,
        row_identity_note: "rows are reported as parsed-record counts only; data-row ordinal is not an upstream row identifier",
        claim_ceiling: "CSV byte/shape inventory only; no semantic field mapping or model-accuracy claim",
    })
}

fn reconcile_csv_bytes(
    bytes: &[u8],
    mapping: &CsvMappingSpec,
    mapping_bytes: &[u8],
    expected_relative_path: &str,
) -> Result<CsvReconciliationReceipt> {
    if mapping.schema_version != "0.1.0" {
        return Err(AdapterError::Validation(format!(
            "unsupported CSV mapping schema_version {:?}",
            mapping.schema_version
        )));
    }

    let declared_path = safe_relative_path(&mapping.artifact_relative_path)?;
    if declared_path.to_string_lossy() != expected_relative_path {
        return Err(AdapterError::Validation(format!(
            "mapping artifact path {:?} does not match requested manifest path {:?}",
            mapping.artifact_relative_path, expected_relative_path
        )));
    }
    if !nonblank(&mapping.source_row_id_header) {
        return Err(AdapterError::Validation(
            "source_row_id_header must identify an exact nonblank source header".into(),
        ));
    }
    if mapping.columns.is_empty() {
        return Err(AdapterError::Validation(
            "CSV mapping must declare at least one semantic column".into(),
        ));
    }

    let inventory = inspect_csv_bytes(bytes, expected_relative_path)?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(bytes);
    let headers = reader.headers()?.clone();

    let row_id_index = headers.iter()
        .position(|header| header == mapping.source_row_id_header)
        .ok_or_else(|| AdapterError::Validation(format!(
            "source row identifier header {:?} is absent from the exact CSV header",
            mapping.source_row_id_header
        )))?;

    let mut source_headers = BTreeSet::new();
    let mut canonical_fields = BTreeSet::new();
    let mut mapped_indices = Vec::with_capacity(mapping.columns.len());
    for column in &mapping.columns {
        if !nonblank(&column.source_header)
            || !nonblank(&column.canonical_field_id)
            || !nonblank(&column.raw_unit)
        {
            return Err(AdapterError::Validation(
                "mapped source header, canonical field ID, and raw unit must be explicit".into(),
            ));
        }

        let mut declared_missing_tokens = BTreeSet::new();
        for token in &column.missing_tokens {
            if token.trim().is_empty() || token != token.trim() {
                return Err(AdapterError::Validation(format!(
                    "missing-value tokens for source field {:?} must be nonblank exact tokens without surrounding whitespace",
                    column.source_header
                )));
            }
            if !declared_missing_tokens.insert(token.clone()) {
                return Err(AdapterError::Validation(format!(
                    "missing-value token {:?} is duplicated for source field {:?}",
                    token, column.source_header
                )));
            }
        }
        if column.source_header == mapping.source_row_id_header {
            return Err(AdapterError::Validation(format!(
                "source row identity column {:?} cannot also be mapped as a measurement field",
                column.source_header
            )));
        }
        if !source_headers.insert(column.source_header.clone()) {
            return Err(AdapterError::Validation(format!(
                "source header {:?} is mapped more than once",
                column.source_header
            )));
        }
        if !canonical_fields.insert(column.canonical_field_id.clone()) {
            return Err(AdapterError::Validation(format!(
                "canonical field ID {:?} is mapped more than once",
                column.canonical_field_id
            )));
        }
        let index = headers.iter()
            .position(|header| header == column.source_header)
            .ok_or_else(|| AdapterError::Validation(format!(
                "mapped source header {:?} is absent from the exact CSV header",
                column.source_header
            )))?;
        mapped_indices.push(index);
    }

    let mut excluded_headers = BTreeSet::new();
    let mut excluded_indices = Vec::with_capacity(mapping.excluded_columns.len());
    for excluded in &mapping.excluded_columns {
        if !nonblank(&excluded.source_header) || !nonblank(&excluded.reason) {
            return Err(AdapterError::Validation(
                "excluded source header and exclusion reason must be explicit".into(),
            ));
        }
        if excluded.source_header == mapping.source_row_id_header {
            return Err(AdapterError::Validation(
                "the source row identifier cannot be excluded as a data column".into(),
            ));
        }
        if source_headers.contains(&excluded.source_header) {
            return Err(AdapterError::Validation(format!(
                "source header {:?} cannot be both mapped and excluded",
                excluded.source_header
            )));
        }
        if !excluded_headers.insert(excluded.source_header.clone()) {
            return Err(AdapterError::Validation(format!(
                "source header {:?} is excluded more than once",
                excluded.source_header
            )));
        }
        let index = headers.iter()
            .position(|header| header == excluded.source_header)
            .ok_or_else(|| AdapterError::Validation(format!(
                "excluded source header {:?} is absent from the exact CSV header",
                excluded.source_header
            )))?;
        excluded_indices.push(index);
    }

    for header in headers.iter() {
        if header != mapping.source_row_id_header
            && !source_headers.contains(header)
            && !excluded_headers.contains(header)
        {
            return Err(AdapterError::Validation(format!(
                "source header {:?} has neither a field mapping nor an explicit exclusion reason",
                header
            )));
        }
    }

    let mut excluded_receipts: Vec<CsvExcludedColumnReceipt> = mapping.excluded_columns.iter()
        .map(|excluded| CsvExcludedColumnReceipt {
            source_header: excluded.source_header.clone(),
            reason: excluded.reason.clone(),
            nonempty_cells: 0,
            empty_cells: 0,
        })
        .collect();

    let mut rows = 0usize;
    let mut seen_row_ids = BTreeSet::new();
    let mut row_ids_hasher = Sha256::new();
    let mut mapped_receipts: Vec<CsvMappedColumnReceipt> = mapping.columns.iter()
        .map(|column| CsvMappedColumnReceipt {
            source_header: column.source_header.clone(),
            canonical_field_id: column.canonical_field_id.clone(),
            raw_unit: column.raw_unit.clone(),
            role: column.role,
            value_kind: column.value_kind,
            missing_tokens: column.missing_tokens.clone(),
            missing_token_counts: column.missing_tokens.iter().cloned().map(|token| (token, 0)).collect(),
            missing_policy: column.missing_policy,
            nonempty_cells: 0,
            numeric_values_validated: 0,
            preserved_missing_cells: 0,
            preserved_missing_token_cells: 0,
        })
        .collect();

    for (record_index, result) in reader.records().enumerate() {
        let record = result?;
        rows = rows.checked_add(1).ok_or_else(|| {
            AdapterError::Validation("CSV reconciliation row count overflow".into())
        })?;
        let row_id = record.get(row_id_index).unwrap_or_default();
        if row_id.trim().is_empty() {
            return Err(AdapterError::Validation(format!(
                "CSV data record {} has a blank source-native row ID in header {:?}",
                record_index + 1, mapping.source_row_id_header
            )));
        }
        if row_id != row_id.trim() {
            return Err(AdapterError::Validation(format!(
                "CSV source-native row ID in record {} has surrounding whitespace; identity normalization is not implicit",
                record_index + 1
            )));
        }
        if !seen_row_ids.insert(row_id.to_string()) {
            return Err(AdapterError::Validation(format!(
                "CSV source-native row ID {:?} is duplicated", row_id
            )));
        }
        let row_id_bytes = row_id.as_bytes();
        row_ids_hasher.update((row_id_bytes.len() as u64).to_be_bytes());
        row_ids_hasher.update(row_id_bytes);

        for (mapping_index, source_index) in mapped_indices.iter().enumerate() {
            let column_spec = &mapping.columns[mapping_index];
            let cell = record.get(*source_index).unwrap_or_default();
            let missing_token = column_spec.missing_tokens.iter().find(|token| token.as_str() == cell);
            if cell.trim().is_empty() || missing_token.is_some() {
                if column_spec.missing_policy == CsvMissingPolicy::Reject {
                    return Err(AdapterError::Validation(format!(
                        "CSV record {} has a missing value in required source field {:?}",
                        record_index + 1, column_spec.source_header
                    )));
                }
                if let Some(token) = missing_token {
                    let count = mapped_receipts[mapping_index]
                        .missing_token_counts
                        .get_mut(token)
                        .ok_or_else(|| AdapterError::Validation(
                            "declared missing-value token has no receipt counter".into()
                        ))?;
                    *count = count.checked_add(1).ok_or_else(|| AdapterError::Validation(
                        "missing-value token count overflow".into()
                    ))?;
                    mapped_receipts[mapping_index].preserved_missing_token_cells =
                        mapped_receipts[mapping_index].preserved_missing_token_cells.checked_add(1)
                        .ok_or_else(|| AdapterError::Validation(
                            "preserved missing-token count overflow".into()
                        ))?;
                } else {
                    mapped_receipts[mapping_index].preserved_missing_cells =
                        mapped_receipts[mapping_index].preserved_missing_cells.checked_add(1)
                        .ok_or_else(|| AdapterError::Validation(
                            "CSV blank-cell count overflow".into()
                        ))?;
                }
                continue;
            }

            if column_spec.value_kind == CsvValueKind::FiniteNumber {
                let parsed = cell.trim().parse::<f64>().map_err(|_| {
                    AdapterError::Validation(format!(
                        "CSV record {} field {:?} is not a valid finite number; source text {:?}",
                        record_index + 1, column_spec.source_header, cell
                    ))
                })?;
                if !parsed.is_finite() {
                    return Err(AdapterError::Validation(format!(
                        "CSV record {} field {:?} contains a non-finite numeric token {:?}",
                        record_index + 1, column_spec.source_header, cell
                    )));
                }
                mapped_receipts[mapping_index].numeric_values_validated =
                    mapped_receipts[mapping_index].numeric_values_validated.checked_add(1)
                    .ok_or_else(|| AdapterError::Validation(
                        "numeric cell count overflow".into()
                    ))?;
            }

            mapped_receipts[mapping_index].nonempty_cells =
                mapped_receipts[mapping_index].nonempty_cells.checked_add(1)
                .ok_or_else(|| AdapterError::Validation(
                    "CSV nonempty-cell count overflow".into()
                ))?;
        }

        for (excluded_index, source_index) in excluded_indices.iter().enumerate() {
            let cell = record.get(*source_index).unwrap_or_default();
            if cell.trim().is_empty() {
                excluded_receipts[excluded_index].empty_cells =
                    excluded_receipts[excluded_index].empty_cells.checked_add(1)
                    .ok_or_else(|| AdapterError::Validation(
                        "excluded CSV empty-cell count overflow".into()
                    ))?;
            } else {
                excluded_receipts[excluded_index].nonempty_cells =
                    excluded_receipts[excluded_index].nonempty_cells.checked_add(1)
                    .ok_or_else(|| AdapterError::Validation(
                        "excluded CSV nonempty-cell count overflow".into()
                    ))?;
            }
        }
    }

    if rows != inventory.data_row_count || seen_row_ids.len() != rows {
        return Err(AdapterError::Validation(
            "CSV reconciliation row count or source row identity accounting mismatch".into(),
        ));
    }

    let mut header_hasher = Sha256::new();
    for header in headers.iter() {
        let header_bytes = header.as_bytes();
        header_hasher.update((header_bytes.len() as u64).to_be_bytes());
        header_hasher.update(header_bytes);
    }
    let headers_sha256 = format!("{:x}", header_hasher.finalize());
    let ordered_source_row_ids_sha256 = format!("{:x}", row_ids_hasher.finalize());

    Ok(CsvReconciliationReceipt {
        artifact_relative_path: expected_relative_path.to_string(),
        artifact_sha256: sha256_hex(bytes),
        mapping_sha256: sha256_hex(mapping_bytes),
        headers_sha256,
        source_row_id_header: mapping.source_row_id_header.clone(),
        ordered_source_row_ids_sha256,
        data_row_count: rows,
        unique_source_row_ids: seen_row_ids.len(),
        mapped_column_count: mapped_receipts.len(),
        columns: mapped_receipts,
        excluded_column_count: excluded_receipts.len(),
        excluded_columns: excluded_receipts,
        reconciliation_note: "all data headers are either explicitly mapped or excluded with a reason; all parsed rows and source-native row IDs were accounted for; missing values follow the declared field policy",
        claim_ceiling: "source row/header coverage integrity only; no numeric parsing, unit conversion, prediction accuracy, or physical qualification claim",
    })
}

fn nonblank(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_http_url(value: &str) -> bool {
    let Some((scheme, remainder)) = value.split_once("://") else {
        return false;
    };
    if !matches!(scheme, "https" | "http") {
        return false;
    }

    // Reject empty authorities and whitespace instead of accepting any string
    // that merely starts with an HTTP(S) scheme prefix.
    let authority = remainder
        .split(|character| matches!(character, '/' | '?' | '#'))
        .next()
        .unwrap_or_default();
    if authority.is_empty() || authority.chars().any(char::is_whitespace) {
        return false;
    }
    let host_port = authority.rsplit('@').next().unwrap_or_default();
    if host_port.is_empty() || host_port.starts_with(':') || host_port.ends_with(':') {
        return false;
    }
    if host_port.starts_with('[') {
        let Some(close_bracket) = host_port.find(']') else {
            return false;
        };
        if close_bracket <= 1 {
            return false;
        }
        let suffix = &host_port[close_bracket + 1..];
        suffix.is_empty()
            || (suffix.starts_with(':')
                && suffix[1..].parse::<u16>().is_ok())
    } else {
        let mut pieces = host_port.split(':');
        let host = pieces.next().unwrap_or_default();
        let port = pieces.next();
        !host.is_empty()
            && pieces.next().is_none()
            && port.is_none_or(|number| number.parse::<u16>().is_ok())
            && !host.contains('[')
            && !host.contains(']')
    }
}

fn is_valid_iso_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || !bytes[..4].iter().all(|byte| byte.is_ascii_digit())
        || !bytes[5..7].iter().all(|byte| byte.is_ascii_digit())
        || !bytes[8..10].iter().all(|byte| byte.is_ascii_digit())
    {
        return false;
    }

    let Ok(year) = value[..4].parse::<u32>() else {
        return false;
    };
    let Ok(month) = value[5..7].parse::<u32>() else {
        return false;
    };
    let Ok(day) = value[8..10].parse::<u32>() else {
        return false;
    };
    if year == 0 || !(1..=12).contains(&month) {
        return false;
    }

    let leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days_in_month).contains(&day)
}

fn is_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn safe_relative_path(raw: &str) -> Result<PathBuf> {
    let path = Path::new(raw);
    if raw.is_empty()
        || raw.contains('\\')
        || path.is_absolute()
        || path.components().any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(AdapterError::Validation(format!(
            "artifact path must be a safe relative path: {raw:?}"
        )));
    }
    Ok(path.to_path_buf())
}

fn validate_manifest(manifest: &SourceManifest) -> Result<()> {
    if manifest.schema_version != "0.1.0" {
        return Err(AdapterError::Validation(format!(
            "unsupported manifest schema_version {:?}",
            manifest.schema_version
        )));
    }

    for (name, value) in [
        ("dataset.id", manifest.dataset.id.as_str()),
        ("dataset.doi", manifest.dataset.doi.as_str()),
        ("dataset.title", manifest.dataset.title.as_str()),
        ("dataset.publisher", manifest.dataset.publisher.as_str()),
        ("dataset.source_url", manifest.dataset.source_url.as_str()),
        ("dataset.repository_url", manifest.dataset.repository_url.as_str()),
        ("dataset.descriptive_readme_url", manifest.dataset.descriptive_readme_url.as_str()),
        ("dataset.citation_policy", manifest.dataset.citation_policy.as_str()),
        ("dataset.source_revision", manifest.dataset.source_revision.as_str()),
        ("dataset.metadata_accessed_on", manifest.dataset.metadata_accessed_on.as_str()),
        ("challenge_context.material", manifest.challenge_context.material.as_str()),
        ("challenge_context.process", manifest.challenge_context.process.as_str()),
        ("challenge_context.condition", manifest.challenge_context.condition.as_str()),
        ("challenge_context.test_method", manifest.challenge_context.test_method.as_str()),
        ("challenge_context.target_parent_specimen_id", manifest.challenge_context.target_parent_specimen_id.as_str()),
        ("challenge_context.target_definition", manifest.challenge_context.target_definition.as_str()),
        ("challenge_context.split_policy", manifest.challenge_context.split_policy.as_str()),
        ("reuse_policy.status_note", manifest.reuse_policy.status_note.as_str()),
    ] {
        if !nonblank(value) {
            return Err(AdapterError::Validation(format!("{name} must not be blank")));
        }
    }

    if !manifest.dataset.doi.starts_with("10.") {
        return Err(AdapterError::Validation("dataset DOI must be explicit and DOI-shaped".into()));
    }
    if !is_valid_iso_date(&manifest.dataset.metadata_accessed_on) {
        return Err(AdapterError::Validation("metadata_accessed_on must use YYYY-MM-DD syntax".into()));
    }
    for (field, url) in [
        ("dataset.source_url", manifest.dataset.source_url.as_str()),
        ("dataset.repository_url", manifest.dataset.repository_url.as_str()),
        ("dataset.descriptive_readme_url", manifest.dataset.descriptive_readme_url.as_str()),
    ] {
        if !is_http_url(url) {
            return Err(AdapterError::Validation(format!("{field} must be an HTTP(S) URL")));
        }
    }

    let context = &manifest.challenge_context;
    if !context.strain_rate_per_second.is_finite() || context.strain_rate_per_second <= 0.0 {
        return Err(AdapterError::Validation(
            "strain_rate_per_second must be finite and positive".into(),
        ));
    }
    if !context.gauge_length_mm.is_finite() || context.gauge_length_mm <= 0.0 {
        return Err(AdapterError::Validation(
            "gauge_length_mm must be finite and positive".into(),
        ));
    }
    if context.target_specimen_count == 0 {
        return Err(AdapterError::Validation("target_specimen_count must be positive".into()));
    }
    if manifest.reuse_policy.repository_bundling_authorized
        && (!manifest.reuse_policy.artifact_terms_reviewed
            || !manifest.reuse_policy.local_use_authorized)
    {
        return Err(AdapterError::Validation(
            "repository bundling requires artifact-specific terms review and local-use authorization".into(),
        ));
    }
    if manifest.reuse_policy.local_use_authorized
        && !manifest.reuse_policy.artifact_terms_reviewed
    {
        return Err(AdapterError::Validation(
            "local use cannot be authorized before artifact-specific terms review".into(),
        ));
    }

    let mut source_ids = BTreeSet::new();
    source_ids.insert(manifest.dataset.id.clone());
    for source in &manifest.related_sources {
        if !nonblank(&source.id)
            || !nonblank(&source.doi)
            || !nonblank(&source.role)
            || !nonblank(&source.url)
        {
            return Err(AdapterError::Validation(
                "related source id, DOI/status, role, and URL must be explicit".into(),
            ));
        }
        if !source_ids.insert(source.id.clone()) {
            return Err(AdapterError::Validation(format!(
                "duplicate source id {:?}", source.id
            )));
        }
        if !source.doi.starts_with("10.") && !source.doi.starts_with("not_applicable_") {
            return Err(AdapterError::Validation(format!(
                "related source {:?} must provide a DOI or an explicit non-DOI status",
                source.id
            )));
        }
        if !is_http_url(&source.url) {
            return Err(AdapterError::Validation(format!(
                "related source {:?} URL must be HTTP(S)", source.id
            )));
        }
    }

    let mut source_object_ids: BTreeSet<String> = BTreeSet::new();
    let mut declared_source_dois = BTreeSet::new();
    declared_source_dois.insert(manifest.dataset.doi.as_str());
    for source in &manifest.related_sources {
        if !source.doi.starts_with("not_applicable_") {
            declared_source_dois.insert(source.doi.as_str());
        }
    }
    for object in &manifest.source_objects {
        for (field, value) in [
            ("id", object.id.as_str()),
            ("source_doi", object.source_doi.as_str()),
            ("source_revision", object.source_revision.as_str()),
        ] {
            if !nonblank(value) {
                return Err(AdapterError::Validation(format!(
                    "source object {field} must not be blank"
                )));
            }
        }
        if !source_object_ids.insert(object.id.clone()) {
            return Err(AdapterError::Validation(format!(
                "duplicate source object id {:?}", object.id
            )));
        }
        if !declared_source_dois.contains(object.source_doi.as_str()) {
            return Err(AdapterError::Validation(format!(
                "source object {:?} references undeclared DOI {:?}",
                object.id, object.source_doi
            )));
        }
        if !is_http_url(&object.source_url) || !is_http_url(&object.terms_url) {
            return Err(AdapterError::Validation(format!(
                "source object {:?} source and terms URLs must be HTTP(S)", object.id
            )));
        }
        if !is_valid_iso_date(&object.accessed_on) {
            return Err(AdapterError::Validation(format!(
                "source object {:?} accessed_on must use YYYY-MM-DD syntax", object.id
            )));
        }
        if object.local_use_authorized && !object.terms_reviewed {
            return Err(AdapterError::Validation(format!(
                "source object {:?} local use is authorized without terms review", object.id
            )));
        }
    }

    let mut artifact_paths = BTreeSet::new();
    for artifact in &manifest.artifacts {
        safe_relative_path(&artifact.relative_path)?;
        if !artifact_paths.insert(artifact.relative_path.clone()) {
            return Err(AdapterError::Validation(format!(
                "duplicate artifact path {:?}", artifact.relative_path
            )));
        }
        for (field, value) in [
            ("source_object_id", artifact.source_object_id.as_str()),
            ("artifact_revision", artifact.artifact_revision.as_str()),
        ] {
            if !nonblank(value) {
                return Err(AdapterError::Validation(format!(
                    "artifact {:?} {field} must not be blank",
                    artifact.relative_path
                )));
            }
        }
        if !is_valid_iso_date(&artifact.accessed_on) {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} accessed_on must use YYYY-MM-DD syntax",
                artifact.relative_path
            )));
        }
        if !is_http_url(&artifact.terms_url) {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} terms_url must be HTTP(S)",
                artifact.relative_path
            )));
        }
        if artifact.local_use_authorized && !artifact.terms_reviewed {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} local use is authorized without terms review",
                artifact.relative_path
            )));
        }
        if artifact.repository_bundling_authorized
            && (!artifact.terms_reviewed || !artifact.local_use_authorized)
        {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} bundling requires terms review and local-use authorization",
                artifact.relative_path
            )));
        }
        if !source_object_ids.contains(&artifact.source_object_id) {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} references undeclared source object {:?}",
                artifact.relative_path, artifact.source_object_id
            )));
        }
        let source_object = manifest.source_objects.iter()
            .find(|object| object.id == artifact.source_object_id)
            .expect("source object existence checked above");
        if source_object.source_doi != artifact.source_doi {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} DOI {:?} does not match source object DOI {:?}",
                artifact.relative_path, artifact.source_doi, source_object.source_doi
            )));
        }
        if source_object.source_revision != artifact.artifact_revision {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} revision {:?} does not match declared source object revision {:?}",
                artifact.relative_path, artifact.artifact_revision, source_object.source_revision
            )));
        }
        if !source_object.terms_reviewed || !source_object.local_use_authorized {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} is blocked by source-object reuse permissions",
                artifact.relative_path
            )));
        }
        if !is_hex_sha256(&artifact.sha256) {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} must declare a lowercase 64-character SHA-256 digest",
                artifact.relative_path
            )));
        }
        if artifact.size_bytes == 0 {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} size_bytes must be positive",
                artifact.relative_path
            )));
        }
        if !nonblank(&artifact.media_type) {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} media_type must not be blank",
                artifact.relative_path
            )));
        }
        if !is_http_url(&artifact.source_url) {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} source_url must be HTTP(S)",
                artifact.relative_path
            )));
        }
    }

    if !manifest.artifacts.is_empty()
        && (!manifest.reuse_policy.artifact_terms_reviewed
            || !manifest.reuse_policy.local_use_authorized
            || manifest.artifacts.iter().any(|artifact| {
                !artifact.terms_reviewed || !artifact.local_use_authorized
            }))
    {
        return Err(AdapterError::Validation(
            "artifacts are declared but manifest and per-artifact terms/local-use permissions are not reviewed".into(),
        ));
    }
    if manifest.reuse_policy.repository_bundling_authorized
        && manifest.artifacts.iter().any(|artifact| !artifact.repository_bundling_authorized)
    {
        return Err(AdapterError::Validation(
            "manifest enables bundling but at least one artifact forbids or has not approved it".into(),
        ));
    }
    Ok(())
}

fn validate_records(
    manifest: &SourceManifest,
    observations: &ObservationSet,
) -> Result<Vec<String>> {
    if observations.schema_version != "0.1.0" {
        return Err(AdapterError::Validation(format!(
            "unsupported observation schema_version {:?}", observations.schema_version
        )));
    }
    if observations.records.is_empty() {
        return Err(AdapterError::Validation("observation set must not be empty".into()));
    }

    let mut declared_dois: BTreeSet<String> = BTreeSet::new();
    declared_dois.insert(manifest.dataset.doi.clone());
    for source in &manifest.related_sources {
        if !source.doi.starts_with("not_applicable_") {
            declared_dois.insert(source.doi.clone());
        }
    }
    let declared_objects: BTreeMap<String, (String, bool)> = manifest.source_objects.iter()
        .map(|object| (
            object.id.clone(),
            (object.source_doi.clone(), object.terms_reviewed && object.local_use_authorized),
        ))
        .collect();
    let declared_artifact_objects: BTreeSet<String> = manifest.artifacts.iter()
        .map(|artifact| artifact.source_object_id.clone())
        .collect();
    if !manifest.reuse_policy.artifact_terms_reviewed || !manifest.reuse_policy.local_use_authorized {
        return Err(AdapterError::Validation(
            "record validation requires manifest-level source terms and local-use authorization".into(),
        ));
    }

    let target_group = manifest.challenge_context.target_parent_specimen_id.as_str();
    let mut record_ids: BTreeSet<String> = BTreeSet::new();
    let mut source_cells: BTreeSet<(String, String, String)> = BTreeSet::new();
    let mut source_field_contracts: BTreeMap<(String, String), (String, String)> = BTreeMap::new();
    let mut normalized_units: BTreeMap<String, String> = BTreeMap::new();
    let mut calibration_outcome_groups = BTreeSet::new();
    let mut held_out_outcome_groups = BTreeSet::new();

    for record in &observations.records {
        for (name, value) in [
            ("record_id", record.record_id.as_str()),
            ("source_doi", record.source_doi.as_str()),
            ("specimen_id", record.specimen_id.as_str()),
            ("lineage_group_id", record.lineage_group_id.as_str()),
            ("source_object_id", record.source_object_id.as_str()),
            ("source_row_id", record.source_row_id.as_str()),
            ("source_field", record.source_field.as_str()),
            ("variable_id", record.variable_id.as_str()),
        ] {
            if !nonblank(value) {
                return Err(AdapterError::Validation(format!("record field {name} must not be blank")));
            }
            if matches!(value.trim().to_ascii_lowercase().as_str(), "unknown" | "n/a" | "null") {
                return Err(AdapterError::Validation(format!(
                    "record field {name} must use a stable identity; preserve unknown metadata separately"
                )));
            }
        }
        if !record_ids.insert(record.record_id.clone()) {
            return Err(AdapterError::Validation(format!(
                "duplicate record_id {:?}", record.record_id
            )));
        }
        if !declared_dois.contains(&record.source_doi) {
            return Err(AdapterError::Validation(format!(
                "record {:?} references undeclared source DOI {:?}",
                record.record_id, record.source_doi
            )));
        }
        let (object_doi, object_authorized) = declared_objects.get(&record.source_object_id)
            .ok_or_else(|| AdapterError::Validation(format!(
                "record {:?} references undeclared source object {:?}",
                record.record_id, record.source_object_id
            )))?;
        if object_doi != &record.source_doi {
            return Err(AdapterError::Validation(format!(
                "record {:?} DOI {:?} does not match object {:?} DOI {:?}",
                record.record_id, record.source_doi, record.source_object_id, object_doi
            )));
        }
        if !*object_authorized {
            return Err(AdapterError::Validation(format!(
                "record {:?} cannot be evaluated until source-object terms and local-use permission are reviewed",
                record.record_id
            )));
        }
        if !declared_artifact_objects.contains(&record.source_object_id) {
            return Err(AdapterError::Validation(format!(
                "record {:?} source object {:?} has no declared artifact digest in the manifest",
                record.record_id, record.source_object_id
            )));
        }

        let raw_unit = record.raw_unit.as_deref().filter(|unit| nonblank(unit))
            .ok_or_else(|| AdapterError::Validation(format!(
                "record {:?} must retain an explicit raw measurement/bound unit",
                record.record_id
            )))?;

        let cell_key = (
            record.source_object_id.clone(),
            record.source_row_id.clone(),
            record.source_field.clone(),
        );
        if !source_cells.insert(cell_key) {
            return Err(AdapterError::Validation(format!(
                "record {:?} duplicates a source object/row/field cell",
                record.record_id
            )));
        }

        let source_key = (record.source_object_id.clone(), record.source_field.clone());
        let field_contract = (record.variable_id.clone(), raw_unit.to_string());
        if let Some(previous) = source_field_contracts.get(&source_key) {
            if previous != &field_contract {
                return Err(AdapterError::Validation(format!(
                    "source field {:?} changes variable identity or raw unit within object {:?}: {:?} vs {:?}",
                    record.source_field, record.source_object_id, previous, field_contract
                )));
            }
        } else {
            source_field_contracts.insert(source_key, field_contract);
        }

        match record.censoring {
            Censoring::Uncensored => {
                let value = record.raw_value.ok_or_else(|| AdapterError::Validation(format!(
                    "record {:?} is uncensored but has no raw value", record.record_id
                )))?;
                if !value.is_finite() || record.censor_lower_bound.is_some() || record.censor_upper_bound.is_some() {
                    return Err(AdapterError::Validation(format!(
                        "record {:?} uncensored value must be finite and have no censor bounds", record.record_id
                    )));
                }
            }
            Censoring::LeftCensored => {
                if record.raw_value.is_some()
                    || record.censor_lower_bound.is_some()
                    || record.censor_upper_bound.is_none_or(|bound| !bound.is_finite())
                {
                    return Err(AdapterError::Validation(format!(
                        "record {:?} left-censored value needs an upper bound only and no fabricated point value",
                        record.record_id
                    )));
                }
            }
            Censoring::RightCensored => {
                if record.raw_value.is_some()
                    || record.censor_lower_bound.is_none_or(|bound| !bound.is_finite())
                    || record.censor_upper_bound.is_some()
                {
                    return Err(AdapterError::Validation(format!(
                        "record {:?} right-censored value needs a lower bound only and no fabricated point value",
                        record.record_id
                    )));
                }
            }
            Censoring::IntervalCensored => {
                match (record.raw_value, record.censor_lower_bound, record.censor_upper_bound) {
                    (None, Some(lower), Some(upper)) if lower.is_finite() && upper.is_finite() && lower < upper => {}
                    _ => return Err(AdapterError::Validation(format!(
                        "record {:?} interval-censored value needs finite lower/upper bounds with lower < upper and no fabricated point value",
                        record.record_id
                    ))),
                }
            }
            Censoring::Unknown => {
                if record.metadata_note.as_deref().is_none_or(|note| !nonblank(note)) {
                    return Err(AdapterError::Validation(format!(
                        "record {:?} has unknown censoring without an explanatory metadata note",
                        record.record_id
                    )));
                }
                if record.raw_value.is_some_and(|value| !value.is_finite())
                    || record.censor_lower_bound.is_some_and(|value| !value.is_finite())
                    || record.censor_upper_bound.is_some_and(|value| !value.is_finite())
                {
                    return Err(AdapterError::Validation(format!(
                        "record {:?} unknown-censoring fields must still be finite when present",
                        record.record_id
                    )));
                }
            }
        }

        if let Some(value) = record.normalized_value {
            if !value.is_finite() {
                return Err(AdapterError::Validation(format!(
                    "record {:?} normalized_value must be finite", record.record_id
                )));
            }
            let normalized_unit = record.normalized_unit.as_deref().unwrap_or_default();
            if let Some(previous) = normalized_units.get(&record.variable_id) {
                if previous != normalized_unit {
                    return Err(AdapterError::Validation(format!(
                        "variable {:?} changes normalized unit from {:?} to {:?}",
                        record.variable_id, previous, normalized_unit
                    )));
                }
            } else {
                normalized_units.insert(record.variable_id.clone(), normalized_unit.to_string());
            }
            if record.raw_value.is_none()
                || record.normalized_unit.as_deref().is_none_or(|unit| !nonblank(unit))
                || record.conversion_note.as_deref().is_none_or(|note| !nonblank(note))
            {
                return Err(AdapterError::Validation(format!(
                    "record {:?} normalized value requires preserved raw value, normalized unit, and explicit conversion note",
                    record.record_id
                )));
            }
        } else if record.normalized_unit.is_some() || record.conversion_note.is_some() {
            return Err(AdapterError::Validation(format!(
                "record {:?} declares conversion metadata without a normalized value", record.record_id
            )));
        }

        if let Some(uncertainty) = record.reported_uncertainty {
            if !uncertainty.is_finite() || uncertainty < 0.0 {
                return Err(AdapterError::Validation(format!(
                    "record {:?} uncertainty must be finite and non-negative", record.record_id
                )));
            }
            if record.uncertainty_unit.as_deref().is_none_or(|unit| !nonblank(unit)) {
                return Err(AdapterError::Validation(format!(
                    "record {:?} uncertainty requires an explicit unit", record.record_id
                )));
            }
        } else if record.uncertainty_unit.is_some() {
            return Err(AdapterError::Validation(format!(
                "record {:?} declares uncertainty unit without uncertainty value", record.record_id
            )));
        }

        if record.role == RecordRole::TargetOutcome {
            if record.split == Split::Calibration {
                if record.lineage_group_id == target_group {
                    return Err(AdapterError::Validation(format!(
                        "target lineage {:?} appears in calibration outcomes", target_group
                    )));
                }
                calibration_outcome_groups.insert(record.lineage_group_id.clone());
            } else {
                if record.lineage_group_id != target_group {
                    return Err(AdapterError::Validation(format!(
                        "held-out target outcome must belong to declared target lineage {:?}",
                        target_group
                    )));
                }
                held_out_outcome_groups.insert(record.lineage_group_id.clone());
            }
        }
    }

    if calibration_outcome_groups
        .iter()
        .any(|group| held_out_outcome_groups.contains(group))
    {
        return Err(AdapterError::Validation(
            "outcome lineage group leakage between calibration and held-out splits".into(),
        ));
    }
    if held_out_outcome_groups.is_empty() {
        return Err(AdapterError::Validation(
            "no held-out target outcomes found; do not emit predictive metrics".into(),
        ));
    }
    Ok(held_out_outcome_groups.into_iter().collect())
}

fn verify_artifacts(manifest: &SourceManifest, root: &Path) -> Result<()> {
    if manifest.artifacts.is_empty() {
        return Err(AdapterError::Validation(
            "manifest is metadata-only: no source artifacts are declared or verified".into(),
        ));
    }
    if !manifest.reuse_policy.artifact_terms_reviewed || !manifest.reuse_policy.local_use_authorized {
        return Err(AdapterError::Validation(
            "artifact verification blocked until artifact terms and local-use permissions are reviewed".into(),
        ));
    }
    let canonical_root = fs::canonicalize(root).map_err(|error| io_error(root, error))?;
    for artifact in &manifest.artifacts {
        let relative = safe_relative_path(&artifact.relative_path)?;
        let requested_path = canonical_root.join(relative);
        let path = fs::canonicalize(&requested_path).map_err(|error| io_error(&requested_path, error))?;
        if !path.starts_with(&canonical_root) {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} resolves outside the declared artifact root",
                artifact.relative_path
            )));
        }
        if !path.is_file() {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} is not a regular file",
                artifact.relative_path
            )));
        }
        let bytes = fs::read(&path).map_err(|error| io_error(&path, error))?;
        if bytes.len() as u64 != artifact.size_bytes {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} size mismatch: manifest {}, actual {}",
                artifact.relative_path, artifact.size_bytes, bytes.len()
            )));
        }
        let actual = sha256_hex(&bytes);
        if actual != artifact.sha256 {
            return Err(AdapterError::Validation(format!(
                "artifact {:?} SHA-256 mismatch: expected {}, actual {}",
                artifact.relative_path, artifact.sha256, actual
            )));
        }
    }
    Ok(())
}

fn manifest_receipt(
    operation: &'static str,
    status: &'static str,
    manifest_bytes: &[u8],
    evidence_input_sha256: Option<String>,
    artifact_count: usize,
    record_count: usize,
    groups: Vec<String>,
) -> AuditReceipt {
    AuditReceipt {
        receipt_schema_version: "0.1.0",
        adapter_version: env!("CARGO_PKG_VERSION"),
        source_commit: env::var("GITHUB_SHA").ok().filter(|value| nonblank(value)),
        operation,
        status,
        manifest_sha256: sha256_hex(manifest_bytes),
        evidence_input_sha256,
        artifact_count,
        record_count,
        held_out_outcome_groups: groups,
        csv_inventory: None,
        csv_reconciliation: None,
        zip_inventory: None,
        claim_ceiling: "source/contract integrity only; not evidence of model accuracy or physical qualification",
    }
}

fn run(args: &[String]) -> Result<AuditReceipt> {
    if args.len() < 3 {
        return Err(AdapterError::Validation(
            "usage: metal-measurement-bench validate-manifest MANIFEST.json | verify-artifacts MANIFEST.json ARTIFACT_DIR | inspect-csv MANIFEST.json ARTIFACT_DIR RELATIVE_CSV_PATH | inspect-zip MANIFEST.json ARTIFACT_DIR RELATIVE_ZIP_PATH | reconcile-csv MANIFEST.json ARTIFACT_DIR MAPPING.json RELATIVE_CSV_PATH | validate-records MANIFEST.json OBSERVATIONS.json ARTIFACT_DIR".into(),
        ));
    }
    let operation = args[1].as_str();
    let manifest_path = Path::new(&args[2]);
    let (manifest, manifest_bytes): (SourceManifest, Vec<u8>) = read_json(manifest_path)?;
    validate_manifest(&manifest)?;

    match operation {
        "validate-manifest" => {
            if args.len() != 3 {
                return Err(AdapterError::Validation("validate-manifest accepts exactly one path".into()));
            }
            let status = if manifest.artifacts.is_empty() {
                "VALID_METADATA_ONLY_NO_ARTIFACTS"
            } else {
                "VALID_MANIFEST_ARTIFACTS_DECLARED_NOT_HASH_CHECKED"
            };
            Ok(manifest_receipt(
                "validate-manifest", status, &manifest_bytes, None, manifest.artifacts.len(), 0, Vec::new(),
            ))
        }
        "verify-artifacts" => {
            if args.len() != 4 {
                return Err(AdapterError::Validation("verify-artifacts requires an artifact directory".into()));
            }
            verify_artifacts(&manifest, Path::new(&args[3]))?;
            Ok(manifest_receipt(
                "verify-artifacts", "ARTIFACT_BYTES_AND_DIGESTS_VERIFIED",
                &manifest_bytes, None, manifest.artifacts.len(), 0, Vec::new(),
            ))
        }
        "inspect-csv" => {
            if args.len() != 5 {
                return Err(AdapterError::Validation(
                    "inspect-csv requires an artifact directory and a manifest-declared relative CSV path".into(),
                ));
            }
            let relative_path = safe_relative_path(&args[4])?;
            let relative_string = relative_path.to_string_lossy().to_string();
            if !relative_string.to_lowercase().ends_with(".csv") {
                return Err(AdapterError::Validation(
                    "inspect-csv accepts only a .csv artifact path".into(),
                ));
            }
            let artifact = manifest.artifacts.iter()
                .find(|artifact| artifact.relative_path == relative_string)
                .ok_or_else(|| AdapterError::Validation(format!(
                    "CSV path {:?} is not declared by the manifest", relative_string
                )))?;

            let root = Path::new(&args[3]);
            verify_artifacts(&manifest, root)?;
            let canonical_root = fs::canonicalize(root).map_err(|error| io_error(root, error))?;
            let requested_path = canonical_root.join(&relative_path);
            let canonical_path = fs::canonicalize(&requested_path)
                .map_err(|error| io_error(&requested_path, error))?;
            if !canonical_path.starts_with(&canonical_root) || !canonical_path.is_file() {
                return Err(AdapterError::Validation(format!(
                    "CSV artifact {:?} is outside the verified root or is not a regular file",
                    relative_string
                )));
            }

            // Parse the exact byte buffer after rechecking it, rather than reopening
            // the file in the CSV reader after verification (avoids a verify/read TOCTOU).
            let bytes = fs::read(&canonical_path).map_err(|error| io_error(&canonical_path, error))?;
            if bytes.len() as u64 != artifact.size_bytes || sha256_hex(&bytes) != artifact.sha256 {
                return Err(AdapterError::Validation(format!(
                    "CSV artifact {:?} changed after initial verification", relative_string
                )));
            }
            let inventory = inspect_csv_bytes(&bytes, &relative_string)?;
            let mut receipt = manifest_receipt(
                "inspect-csv",
                "CSV_SHAPE_INVENTORIED_NOT_SEMANTICALLY_MAPPED",
                &manifest_bytes,
                Some(sha256_hex(&bytes)),
                manifest.artifacts.len(),
                inventory.data_row_count,
                Vec::new(),
            );
            receipt.csv_inventory = Some(inventory);
            Ok(receipt)
        }
        "inspect-zip" => {
            if args.len() != 5 {
                return Err(AdapterError::Validation(
                    "inspect-zip requires artifact root and manifest-declared relative ZIP path".into(),
                ));
            }
            let requested_relative = safe_relative_path(&args[4])?;
            let relative_string = requested_relative.to_string_lossy().to_string();
            if !relative_string.to_lowercase().ends_with(".zip") {
                return Err(AdapterError::Validation(
                    "inspect-zip accepts only a .zip artifact path".into(),
                ));
            }
            let artifact = manifest.artifacts.iter()
                .find(|artifact| artifact.relative_path == relative_string)
                .ok_or_else(|| AdapterError::Validation(format!(
                    "ZIP path {:?} is not declared by the manifest", relative_string
                )))?;

            let root = Path::new(&args[3]);
            verify_artifacts(&manifest, root)?;
            let canonical_root = fs::canonicalize(root).map_err(|error| io_error(root, error))?;
            let requested_path = canonical_root.join(&requested_relative);
            let canonical_path = fs::canonicalize(&requested_path)
                .map_err(|error| io_error(&requested_path, error))?;
            if !canonical_path.starts_with(&canonical_root) || !canonical_path.is_file() {
                return Err(AdapterError::Validation(format!(
                    "ZIP artifact {:?} is outside the verified root or is not a regular file",
                    relative_string
                )));
            }
            let bytes = fs::read(&canonical_path).map_err(|error| io_error(&canonical_path, error))?;
            if bytes.len() as u64 != artifact.size_bytes || sha256_hex(&bytes) != artifact.sha256 {
                return Err(AdapterError::Validation(format!(
                    "ZIP artifact {:?} changed after initial verification", relative_string
                )));
            }
            let inventory = inspect_zip_bytes(&bytes, &relative_string)?;
            let mut receipt = manifest_receipt(
                "inspect-zip",
                "ZIP_STRUCTURE_INVENTORIED_NO_EXTRACTION",
                &manifest_bytes,
                Some(sha256_hex(&bytes)),
                manifest.artifacts.len(),
                inventory.member_count,
                Vec::new(),
            );
            receipt.zip_inventory = Some(inventory);
            Ok(receipt)
        }
        "reconcile-csv" => {
            if args.len() != 6 {
                return Err(AdapterError::Validation(
                    "reconcile-csv requires artifact root, mapping JSON and manifest-declared relative CSV path".into(),
                ));
            }
            let (mapping, mapping_bytes): (CsvMappingSpec, Vec<u8>) =
                read_json(Path::new(&args[4]))?;
            let requested_relative = safe_relative_path(&args[5])?;
            let relative_string = requested_relative.to_string_lossy().to_string();
            if !relative_string.to_lowercase().ends_with(".csv") {
                return Err(AdapterError::Validation(
                    "reconcile-csv accepts only a .csv artifact path".into(),
                ));
            }
            let artifact = manifest.artifacts.iter()
                .find(|artifact| artifact.relative_path == relative_string)
                .ok_or_else(|| AdapterError::Validation(format!(
                    "CSV path {:?} is not declared by the manifest", relative_string
                )))?;

            let root = Path::new(&args[3]);
            verify_artifacts(&manifest, root)?;
            let canonical_root = fs::canonicalize(root).map_err(|error| io_error(root, error))?;
            let requested_path = canonical_root.join(&requested_relative);
            let canonical_path = fs::canonicalize(&requested_path)
                .map_err(|error| io_error(&requested_path, error))?;
            if !canonical_path.starts_with(&canonical_root) || !canonical_path.is_file() {
                return Err(AdapterError::Validation(format!(
                    "CSV artifact {:?} is outside the verified root or is not a regular file",
                    relative_string
                )));
            }
            let bytes = fs::read(&canonical_path).map_err(|error| io_error(&canonical_path, error))?;
            if bytes.len() as u64 != artifact.size_bytes || sha256_hex(&bytes) != artifact.sha256 {
                return Err(AdapterError::Validation(format!(
                    "CSV artifact {:?} changed after initial verification", relative_string
                )));
            }
            let reconciliation = reconcile_csv_bytes(
                &bytes, &mapping, &mapping_bytes, &relative_string
            )?;
            let mut receipt = manifest_receipt(
                "reconcile-csv",
                "CSV_SOURCE_ROWS_AND_EXPLICIT_FIELDS_RECONCILED",
                &manifest_bytes,
                Some(sha256_hex(&bytes)),
                manifest.artifacts.len(),
                reconciliation.data_row_count,
                Vec::new(),
            );
            receipt.csv_reconciliation = Some(reconciliation);
            Ok(receipt)
        }
        "validate-records" => {
            if args.len() != 5 {
                return Err(AdapterError::Validation(
                    "validate-records requires an observation JSON path and an artifact directory".into(),
                ));
            }
            let (observations, observation_bytes): (ObservationSet, Vec<u8>) =
                read_json(Path::new(&args[3]))?;
            verify_artifacts(&manifest, Path::new(&args[4]))?;
            let groups = validate_records(&manifest, &observations)?;
            Ok(manifest_receipt(
                "validate-records",
                "RECORD_CONTRACT_SPLIT_AND_SOURCE_BYTES_VALID",
                &manifest_bytes, Some(sha256_hex(&observation_bytes)), manifest.artifacts.len(),
                observations.records.len(), groups,
            ))
        }
        _ => Err(AdapterError::Validation(format!("unknown operation {operation:?}"))),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    match run(&args) {
        Ok(receipt) => match serde_json::to_string_pretty(&receipt) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("could not serialize receipt: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn manifest() -> SourceManifest {
        SourceManifest {
            schema_version: "0.1.0".into(),
            dataset: DatasetIdentity {
                id: "AMB2025-02".into(),
                doi: "10.18434/mds2-3735".into(),
                title: "AMB2025-02 tensile profile".into(),
                publisher: "NIST".into(),
                source_url: "https://doi.org/10.18434/mds2-3735".into(),
                repository_url: "https://data.nist.gov/od/id/mds2-3735".into(),
                descriptive_readme_url: "https://www.nist.gov/ambench".into(),
                citation_policy: "Use DOI landing citation verbatim".into(),
                source_revision: "DOI record 10.18434/mds2-3735; artifact inventory pending".into(),
                metadata_accessed_on: "2026-10-18".into(),
            },
            related_sources: vec![RelatedSource {
                id: "AMB2022-01-serial-sectioning-xct".into(),
                doi: "10.18434/mds2-2767".into(),
                role: "serial-sectioning and XCT; EBSD-related calibration source".into(),
                url: "https://doi.org/10.18434/mds2-2767".into(),
            }],
            source_objects: Vec::new(),
            challenge_context: ChallengeContext {
                material: "IN718".into(),
                process: "PBF-LB".into(),
                condition: "as-built; no heat treatment".into(),
                test_method: "ASTM E8".into(),
                strain_rate_per_second: 0.001,
                gauge_length_mm: 3.0,
                target_specimen_count: 8,
                target_parent_specimen_id: "AMB2022-718-AMMT-B7-P4".into(),
                target_definition: "average tensile properties".into(),
                split_policy: "hold target lineage outcome records out as one group".into(),
            },
            reuse_policy: ReusePolicy {
                artifact_terms_reviewed: false,
                local_use_authorized: false,
                repository_bundling_authorized: false,
                status_note: "link only until reviewed".into(),
            },
            artifacts: Vec::new(),
        }
    }

    fn record_manifest() -> SourceManifest {
        let mut value = manifest();
        value.reuse_policy.artifact_terms_reviewed = true;
        value.reuse_policy.local_use_authorized = true;
        value.source_objects.push(SourceObject {
            id: "amb2025-02-targets".into(),
            source_doi: "10.18434/mds2-3735".into(),
            source_url: "https://data.nist.gov/od/id/mds2-3735".into(),
            source_revision: "fixture-object-v1".into(),
            accessed_on: "2026-10-18".into(),
            terms_url: "https://data.nist.gov/od/id/mds2-3735".into(),
            terms_reviewed: true,
            local_use_authorized: true,
        });
        value.artifacts.push(SourceArtifact {
            relative_path: "fixture.csv".into(),
            source_object_id: "amb2025-02-targets".into(),
            source_doi: "10.18434/mds2-3735".into(),
            artifact_revision: "fixture-object-v1".into(),
            accessed_on: "2026-10-18".into(),
            source_url: "https://data.nist.gov/od/id/mds2-3735".into(),
            terms_url: "https://data.nist.gov/od/id/mds2-3735".into(),
            terms_reviewed: true,
            local_use_authorized: true,
            repository_bundling_authorized: false,
            sha256: "a".repeat(64),
            size_bytes: 1,
            media_type: "text/csv".into(),
        });
        value
    }

    fn record(
        record_id: &str, group: &str, specimen: &str, role: RecordRole,
        split: Split, variable: &str, value: f64,
    ) -> Observation {
        Observation {
            record_id: record_id.into(),
            source_doi: "10.18434/mds2-3735".into(),
            specimen_id: specimen.into(),
            lineage_group_id: group.into(),
            source_object_id: "amb2025-02-targets".into(),
            source_row_id: record_id.into(),
            source_field: variable.into(),
            variable_id: variable.into(),
            role, split,
            raw_value: Some(value),
            raw_unit: Some("MPa".into()),
            censor_lower_bound: None,
            censor_upper_bound: None,
            normalized_value: None,
            normalized_unit: None,
            conversion_note: None,
            reported_uncertainty: None,
            uncertainty_unit: None,
            censoring: Censoring::Uncensored,
            metadata_note: None,
        }
    }

    fn fixture_csv_mapping() -> CsvMappingSpec {
        CsvMappingSpec {
            schema_version: "0.1.0".into(),
            artifact_relative_path: "fixture.csv".into(),
            source_row_id_header: "source_row_id".into(),
            excluded_columns: Vec::new(),
            columns: vec![
                CsvColumnMapping {
                    source_header: "stress_mpa".into(),
                    canonical_field_id: "tensile.stress".into(),
                    raw_unit: "MPa".into(),
                    role: CsvFieldRole::TargetOutcome,
                    value_kind: CsvValueKind::FiniteNumber,
                    missing_tokens: Vec::new(),
                    missing_policy: CsvMissingPolicy::PreserveMissing,
                },
                CsvColumnMapping {
                    source_header: "specimen_note".into(),
                    canonical_field_id: "specimen.note".into(),
                    raw_unit: "text".into(),
                    role: CsvFieldRole::Metadata,
                    value_kind: CsvValueKind::Text,
                    missing_tokens: Vec::new(),
                    missing_policy: CsvMissingPolicy::PreserveMissing,
                },
            ],
        }
    }

    #[test]
    fn reconcile_csv_operation_binds_verified_inputs_and_mapping_to_receipt() {
        let temp = tempdir().unwrap();
        let artifact_root = temp.path().join("artifacts");
        fs::create_dir_all(&artifact_root).unwrap();
        let artifact_bytes = b"source_row_id,stress_mpa,specimen_note\nR-001,920,as-built\nR-002,,repeat\n";
        fs::write(artifact_root.join("fixture.csv"), artifact_bytes).unwrap();

        let mut manifest = record_manifest();
        manifest.artifacts[0].sha256 = sha256_hex(artifact_bytes);
        manifest.artifacts[0].size_bytes = artifact_bytes.len() as u64;
        let manifest_path = temp.path().join("manifest.json");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let mapping = fixture_csv_mapping();
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        let mapping_path = temp.path().join("mapping.json");
        fs::write(&mapping_path, &mapping_bytes).unwrap();

        let args = vec![
            "metal-measurement-bench".to_string(),
            "reconcile-csv".to_string(),
            manifest_path.display().to_string(),
            artifact_root.display().to_string(),
            mapping_path.display().to_string(),
            "fixture.csv".to_string(),
        ];
        let receipt = run(&args).unwrap();
        assert_eq!(receipt.operation, "reconcile-csv");
        assert_eq!(receipt.status, "CSV_SOURCE_ROWS_AND_EXPLICIT_FIELDS_RECONCILED");
        assert_eq!(receipt.record_count, 2);
        assert_eq!(receipt.evidence_input_sha256.as_deref(), Some(sha256_hex(artifact_bytes).as_str()));
        let reconciliation = receipt.csv_reconciliation.unwrap();
        assert_eq!(reconciliation.artifact_sha256, sha256_hex(artifact_bytes));
        assert_eq!(reconciliation.mapping_sha256, sha256_hex(&mapping_bytes));
        assert_eq!(reconciliation.data_row_count, 2);
        assert_eq!(reconciliation.unique_source_row_ids, 2);
        assert_eq!(reconciliation.columns[0].preserved_missing_cells, 1);

        let mut undeclared_args = args;
        undeclared_args[5] = "unlisted.csv".to_string();
        assert!(run(&undeclared_args).is_err());
    }

    #[test]
    fn committed_csv_mapping_schema_and_receipt_operation_are_bound() {
        let mapping_schema: serde_json::Value = serde_json::from_str(
            include_str!("../schemas/csv-mapping-v0.1.schema.json")
        ).unwrap();
        let required = mapping_schema["required"].as_array().unwrap();
        for field in ["schema_version", "artifact_relative_path", "source_row_id_header", "columns", "excluded_columns"] {
            assert!(required.iter().any(|value| value == field),
                "mapping schema does not require {field}");
        }
        assert!(mapping_schema["properties"]["columns"]["items"]["properties"]["missing_policy"]["enum"]
            .as_array().unwrap().iter().any(|value| value == "preserve_missing"));
        assert!(mapping_schema["properties"]["columns"]["items"]["required"].as_array().unwrap()
            .iter().any(|value| value == "value_kind"));
        assert!(mapping_schema["properties"]["columns"]["items"]["properties"]["missing_tokens"]["uniqueItems"] == true);

        let receipt_schema: serde_json::Value = serde_json::from_str(
            include_str!("../schemas/audit-receipt-v0.1.schema.json")
        ).unwrap();
        assert!(receipt_schema["properties"]["csv_reconciliation"]["properties"]["mapping_sha256"].is_object());
        assert!(receipt_schema["properties"]["csv_reconciliation"]["properties"]["columns"]["items"]["properties"]["numeric_values_validated"].is_object());
        assert!(receipt_schema["properties"]["csv_reconciliation"]["properties"]["excluded_columns"].is_object());
        assert!(receipt_schema["properties"]["zip_inventory"]["properties"]["archive_sha256"].is_object());
        assert!(receipt_schema["allOf"].as_array().unwrap().iter().any(|condition| {
            condition["if"]["properties"]["operation"]["const"] == "inspect-zip"
                && condition["then"]["required"].as_array().unwrap()
                    .iter().any(|value| value == "zip_inventory")
        }));
        assert!(receipt_schema["allOf"].as_array().unwrap().iter().any(|condition| {
            condition["if"]["properties"]["operation"]["const"] == "reconcile-csv"
                && condition["then"]["required"].as_array().unwrap()
                    .iter().any(|value| value == "csv_reconciliation")
        }));
    }

    #[test]
    fn csv_reconciliation_binds_exact_headers_units_roles_and_source_row_ids() {
        let bytes = b"source_row_id,stress_mpa,specimen_note\nR-001,920,as-built\nR-002,,repeat\n";
        let mapping = fixture_csv_mapping();
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        let result = reconcile_csv_bytes(bytes, &mapping, &mapping_bytes, "fixture.csv").unwrap();
        assert_eq!(result.artifact_sha256, sha256_hex(bytes));
        assert_eq!(result.mapping_sha256, sha256_hex(&mapping_bytes));
        assert_eq!(result.data_row_count, 2);
        assert_eq!(result.unique_source_row_ids, 2);
        assert_eq!(result.mapped_column_count, 2);
        assert_eq!(result.excluded_column_count, 0);
        assert_eq!(result.columns[0].source_header, "stress_mpa");
        assert_eq!(result.columns[0].raw_unit, "MPa");
        assert_eq!(result.columns[0].role, CsvFieldRole::TargetOutcome);
        assert_eq!(result.columns[0].preserved_missing_cells, 1);
        assert_eq!(result.columns[1].nonempty_cells, 2);
    }

    #[test]
    fn csv_reconciliation_rejects_surrounding_whitespace_in_source_row_ids() {
        let bytes = b"source_row_id,stress_mpa,specimen_note\n R-001,920,note\n";
        let mapping = fixture_csv_mapping();
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        assert!(reconcile_csv_bytes(bytes, &mapping, &mapping_bytes, "fixture.csv").is_err());
    }

    #[test]
    fn csv_reconciliation_rejects_unmapped_columns_without_disposition() {
        let bytes = b"source_row_id,stress_mpa,specimen_note,export_comment\nR-001,920,note,unused\n";
        let mapping = fixture_csv_mapping();
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        assert!(reconcile_csv_bytes(bytes, &mapping, &mapping_bytes, "fixture.csv").is_err());
    }

    #[test]
    fn csv_reconciliation_counts_every_explicitly_excluded_column() {
        let bytes = b"source_row_id,stress_mpa,specimen_note,export_comment\nR-001,920,as-built,legacy-a\nR-002,,repeat,\n";
        let mut mapping = fixture_csv_mapping();
        mapping.excluded_columns.push(CsvExcludedColumn {
            source_header: "export_comment".into(),
            reason: "Retain source traceability, exclude from this profile's predictors and targets.".into(),
        });
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        let result = reconcile_csv_bytes(bytes, &mapping, &mapping_bytes, "fixture.csv").unwrap();
        assert_eq!(result.mapped_column_count, 2);
        assert_eq!(result.excluded_column_count, 1);
        assert_eq!(result.excluded_columns[0].source_header, "export_comment");
        assert_eq!(result.excluded_columns[0].nonempty_cells, 1);
        assert_eq!(result.excluded_columns[0].empty_cells, 1);
        assert!(result.excluded_columns[0].reason.contains("exclude from this profile"));
    }

    #[test]
    fn csv_reconciliation_rejects_ambiguous_or_unjustified_exclusions() {
        let bytes = b"source_row_id,stress_mpa,specimen_note,export_comment\nR-001,920,note,unused\n";

        let mut overlaps_mapped = fixture_csv_mapping();
        overlaps_mapped.excluded_columns.push(CsvExcludedColumn {
            source_header: "stress_mpa".into(),
            reason: "incorrect overlap".into(),
        });
        let bytes_for_mapping = serde_json::to_vec(&overlaps_mapped).unwrap();
        assert!(reconcile_csv_bytes(bytes, &overlaps_mapped, &bytes_for_mapping, "fixture.csv").is_err());

        let mut unknown_header = fixture_csv_mapping();
        unknown_header.excluded_columns.push(CsvExcludedColumn {
            source_header: "not_in_source.csv".into(),
            reason: "header does not exist".into(),
        });
        let bytes_for_mapping = serde_json::to_vec(&unknown_header).unwrap();
        assert!(reconcile_csv_bytes(bytes, &unknown_header, &bytes_for_mapping, "fixture.csv").is_err());

        let mut missing_reason = fixture_csv_mapping();
        missing_reason.excluded_columns.push(CsvExcludedColumn {
            source_header: "export_comment".into(),
            reason: "  ".into(),
        });
        let bytes_for_mapping = serde_json::to_vec(&missing_reason).unwrap();
        assert!(reconcile_csv_bytes(bytes, &missing_reason, &bytes_for_mapping, "fixture.csv").is_err());
    }

    fn zip_fixture(entries: &[(&str, &[u8], u32)]) -> Vec<u8> {
        let cursor = Cursor::new(Vec::new());
        let mut writer = zip::ZipWriter::new(cursor);
        for (name, bytes, mode) in entries {
            let options = zip::write::SimpleFileOptions::default().unix_permissions(*mode);
            writer.start_file(*name, options).unwrap();
            writer.write_all(bytes).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    #[test]
    fn inspect_zip_operation_binds_verified_archive_to_receipt() {
        let temp = tempdir().unwrap();
        let artifact_root = temp.path().join("artifacts");
        fs::create_dir_all(&artifact_root).unwrap();
        let archive_bytes = zip_fixture(&[
            ("calibration.csv", b"source_row_id,value\nR-001,3.5\n", 0o100644),
            ("README.txt", b"metadata", 0o100644),
        ]);
        fs::write(artifact_root.join("source.zip"), &archive_bytes).unwrap();

        let mut manifest = record_manifest();
        manifest.artifacts[0].relative_path = "source.zip".into();
        manifest.artifacts[0].media_type = "application/zip".into();
        manifest.artifacts[0].sha256 = sha256_hex(&archive_bytes);
        manifest.artifacts[0].size_bytes = archive_bytes.len() as u64;
        let manifest_path = temp.path().join("manifest.json");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let args = vec![
            "metal-measurement-bench".to_string(),
            "inspect-zip".to_string(),
            manifest_path.display().to_string(),
            artifact_root.display().to_string(),
            "source.zip".to_string(),
        ];
        let receipt = run(&args).unwrap();
        assert_eq!(receipt.operation, "inspect-zip");
        assert_eq!(receipt.status, "ZIP_STRUCTURE_INVENTORIED_NO_EXTRACTION");
        assert_eq!(receipt.record_count, 2);
        assert_eq!(receipt.evidence_input_sha256.as_deref(), Some(sha256_hex(&archive_bytes).as_str()));
        let inventory = receipt.zip_inventory.unwrap();
        assert_eq!(inventory.archive_sha256, sha256_hex(&archive_bytes));
        assert_eq!(inventory.member_count, 2);

        let mut undeclared_args = args;
        undeclared_args[4] = "unlisted.zip".to_string();
        assert!(run(&undeclared_args).is_err());
    }

    fn zip_fixture_symlink() -> Vec<u8> {
        let mut bytes = zip_fixture(&[("link", b"target.csv", 0o100644)]);
        let central_signature = [0x50, 0x4b, 0x01, 0x02];
        let offset = bytes.windows(4)
            .position(|window| window == central_signature)
            .expect("ZIP fixture has a central-directory file header");
        // ZIP's 'version made by' high byte identifies Unix, and the upper
        // 16 external-attribute bits contain the Unix st_mode value.
        bytes[offset + 4..offset + 6]
            .copy_from_slice(&((3u16 << 8) | 20).to_le_bytes());
        bytes[offset + 38..offset + 42]
            .copy_from_slice(&((0o120777u32) << 16).to_le_bytes());
        bytes
    }

    #[test]
    fn zip_inventory_binds_member_paths_sizes_crc_and_archive_digest() {
        let bytes = zip_fixture(&[
            ("data.csv", b"row,value\nR-1,1.5\n", 0o100644),
            ("readme.txt", b"metadata only", 0o100644),
        ]);
        let inventory = inspect_zip_bytes(&bytes, "source.zip").unwrap();
        assert_eq!(inventory.archive_sha256, sha256_hex(&bytes));
        assert_eq!(inventory.member_count, 2);
        assert_eq!(inventory.members[0].member_path, "data.csv");
        assert_eq!(inventory.members[0].uncompressed_size_bytes, b"row,value\nR-1,1.5\n".len() as u64);
        assert_eq!(inventory.members[0].crc32_hex.len(), 8);
        assert!(inventory.total_uncompressed_bytes > 0);
        assert!(inventory.inventory_note.contains("not extracted"));
    }

    #[test]
    fn zip_inventory_rejects_traversal_absolute_and_backslash_paths() {
        for member in [
            "../escape.csv",
            "/absolute.csv",
            "nested/../../escape.csv",
            r"nested\escape.csv",
            "C:/drive.csv",
            "nested//file.csv",
            "nested/./file.csv",
            "CON.txt",
            "trailing./file.csv",
            "folder /bad.csv",
            "LPT1.csv",
        ] {
            let bytes = zip_fixture(&[(member, b"x", 0o100644)]);
            assert!(inspect_zip_bytes(&bytes, "source.zip").is_err(), "accepted unsafe member {member:?}");
        }
    }

    #[test]
    fn zip_inventory_rejects_symlinks_and_case_collisions() {
        let symlink = zip_fixture_symlink();
        assert!(inspect_zip_bytes(&symlink, "source.zip").is_err());

        // The ZIP writer rejects exact duplicate names before producing bytes;
        // this still exercises an on-disk portability collision at inventory time.
        let case_collision = zip_fixture(&[
            ("Data.csv", b"a", 0o100644),
            ("data.csv", b"b", 0o100644),
        ]);
        assert!(inspect_zip_bytes(&case_collision, "source.zip").is_err());
    }

    #[test]
    fn csv_reconciliation_rejects_malformed_and_non_finite_numeric_cells() {
        for value in ["NaN", "inf", "-inf", "not-a-number", "1.2.3"] {
            let input = format!(
                "source_row_id,stress_mpa,specimen_note\nR-001,{value},note\n"
            );
            let mapping = fixture_csv_mapping();
            let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
            assert!(
                reconcile_csv_bytes(input.as_bytes(), &mapping, &mapping_bytes, "fixture.csv").is_err(),
                "accepted invalid finite-number cell {value:?}"
            );
        }
    }

    #[test]
    fn csv_reconciliation_preserves_declared_missing_tokens_without_imputation() {
        let bytes = b"source_row_id,stress_mpa,specimen_note\nR-001,920,as-built\nR-002,NA,repeat\n";
        let mut mapping = fixture_csv_mapping();
        mapping.columns[0].missing_tokens.push("NA".into());
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        let result = reconcile_csv_bytes(bytes, &mapping, &mapping_bytes, "fixture.csv").unwrap();
        assert_eq!(result.columns[0].numeric_values_validated, 1);
        assert_eq!(result.columns[0].nonempty_cells, 1);
        assert_eq!(result.columns[0].preserved_missing_token_cells, 1);
        assert_eq!(result.columns[0].missing_token_counts.get("NA"), Some(&1));

        mapping.columns[0].missing_policy = CsvMissingPolicy::Reject;
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        assert!(reconcile_csv_bytes(bytes, &mapping, &mapping_bytes, "fixture.csv").is_err());
    }

    #[test]
    fn csv_reconciliation_rejects_undeclared_numeric_sentinels_and_bad_token_contracts() {
        let bytes = b"source_row_id,stress_mpa,specimen_note\nR-001,NA,note\n";
        let mapping = fixture_csv_mapping();
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        assert!(reconcile_csv_bytes(bytes, &mapping, &mapping_bytes, "fixture.csv").is_err());

        let mut duplicate_tokens = fixture_csv_mapping();
        duplicate_tokens.columns[0].missing_tokens = vec!["NA".into(), "NA".into()];
        let mapping_bytes = serde_json::to_vec(&duplicate_tokens).unwrap();
        assert!(reconcile_csv_bytes(bytes, &duplicate_tokens, &mapping_bytes, "fixture.csv").is_err());

        let mut whitespace_token = fixture_csv_mapping();
        whitespace_token.columns[0].missing_tokens = vec![" NA ".into()];
        let mapping_bytes = serde_json::to_vec(&whitespace_token).unwrap();
        assert!(reconcile_csv_bytes(bytes, &whitespace_token, &mapping_bytes, "fixture.csv").is_err());
    }

    #[test]
    fn csv_reconciliation_rejects_absent_header_and_repeated_row_ids() {
        let bytes = b"source_row_id,stress_mpa\nR-001,920\nR-002,930\n";
        let mut mapping = fixture_csv_mapping();
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        assert!(reconcile_csv_bytes(bytes, &mapping, &mapping_bytes, "fixture.csv").is_err());

        mapping.columns.truncate(1);
        let duplicate_rows = b"source_row_id,stress_mpa\nR-001,920\nR-001,930\n";
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        assert!(reconcile_csv_bytes(duplicate_rows, &mapping, &mapping_bytes, "fixture.csv").is_err());
    }

    #[test]
    fn csv_reconciliation_applies_missing_policy_without_silent_imputation() {
        let bytes = b"source_row_id,stress_mpa,specimen_note\nR-001,,note\n";
        let mut mapping = fixture_csv_mapping();
        mapping.columns[0].missing_policy = CsvMissingPolicy::Reject;
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        assert!(reconcile_csv_bytes(bytes, &mapping, &mapping_bytes, "fixture.csv").is_err());

        mapping.columns[0].missing_policy = CsvMissingPolicy::PreserveMissing;
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        let result = reconcile_csv_bytes(bytes, &mapping, &mapping_bytes, "fixture.csv").unwrap();
        assert_eq!(result.columns[0].preserved_missing_cells, 1);
        assert_eq!(result.columns[0].nonempty_cells, 0);
    }

    #[test]
    fn csv_reconciliation_rejects_mismatched_artifact_identity_and_duplicate_mapping_fields() {
        let bytes = b"source_row_id,stress_mpa,specimen_note\nR-001,920,note\n";
        let mut mapping = fixture_csv_mapping();
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        assert!(reconcile_csv_bytes(bytes, &mapping, &mapping_bytes, "other.csv").is_err());

        mapping.columns[1].canonical_field_id = mapping.columns[0].canonical_field_id.clone();
        let mapping_bytes = serde_json::to_vec(&mapping).unwrap();
        assert!(reconcile_csv_bytes(bytes, &mapping, &mapping_bytes, "fixture.csv").is_err());
    }

    #[test]
    fn inspect_csv_operation_binds_verified_artifact_to_receipt() {
        let temp = tempdir().unwrap();
        let artifact_root = temp.path().join("artifacts");
        fs::create_dir_all(&artifact_root).unwrap();
        let artifact_bytes = b"record_id,stress_mpa\nS-01,920\nS-02,940\n";
        fs::write(artifact_root.join("fixture.csv"), artifact_bytes).unwrap();

        let mut value = record_manifest();
        value.artifacts[0].sha256 = sha256_hex(artifact_bytes);
        value.artifacts[0].size_bytes = artifact_bytes.len() as u64;
        let manifest_path = temp.path().join("manifest.json");
        fs::write(&manifest_path, serde_json::to_vec(&value).unwrap()).unwrap();

        let args = vec![
            "metal-measurement-bench".to_string(),
            "inspect-csv".to_string(),
            manifest_path.display().to_string(),
            artifact_root.display().to_string(),
            "fixture.csv".to_string(),
        ];
        let receipt = run(&args).unwrap();
        assert_eq!(receipt.operation, "inspect-csv");
        assert_eq!(receipt.status, "CSV_SHAPE_INVENTORIED_NOT_SEMANTICALLY_MAPPED");
        assert_eq!(receipt.record_count, 2);
        assert_eq!(receipt.evidence_input_sha256.as_deref(), Some(sha256_hex(artifact_bytes).as_str()));
        let inventory = receipt.csv_inventory.unwrap();
        assert_eq!(inventory.artifact_sha256, sha256_hex(artifact_bytes));
        assert_eq!(inventory.data_row_count, 2);
        assert_eq!(inventory.columns[0].source_header, "record_id");

        let mut unmanifested_args = args;
        unmanifested_args[4] = "not-declared.csv".to_string();
        assert!(run(&unmanifested_args).is_err());
    }

    #[test]
    fn csv_inventory_counts_rows_and_empty_cells_without_inferencing_units() {
        let input = b"specimen,stress MPa,note\r\nA,950,\"necked, early\"\r\nB,, ok \r\nC,1020,\r\n";
        let inventory = inspect_csv_bytes(input, "synthetic-fixture.csv").unwrap();
        assert_eq!(inventory.data_row_count, 3);
        assert_eq!(inventory.artifact_sha256, sha256_hex(input));
        assert_eq!(inventory.columns.len(), 3);
        assert_eq!(inventory.columns[0].source_header, "specimen");
        assert_eq!(inventory.columns[0].nonempty_cells, 3);
        assert_eq!(inventory.columns[1].empty_cells, 1);
        assert_eq!(inventory.columns[1].nonempty_cells, 2);
        assert_eq!(inventory.columns[2].empty_cells, 1);
        assert!(inventory.row_identity_note.contains("not an upstream row identifier"));
        assert!(inventory.claim_ceiling.contains("no semantic field mapping"));
    }

    #[test]
    fn csv_inventory_rejects_blank_and_case_or_whitespace_colliding_headers() {
        for input in [
            &b",value\n1,2\n"[..],
            &b"value, value \n1,2\n"[..],
            &b"ID,id\na,b\n"[..],
        ] {
            assert!(inspect_csv_bytes(input, "fixture.csv").is_err());
        }
    }

    #[test]
    fn csv_inventory_rejects_inconsistent_record_width() {
        let input = b"id,value\na,1\nb,2,unexpected\n";
        assert!(inspect_csv_bytes(input, "fixture.csv").is_err());
    }

    #[test]
    fn csv_inventory_rejects_quotes_in_unquoted_fields_and_trailing_quote_junk() {
        for input in [
            &b"id,note\n1,un\"escaped\n"[..],
            &b"id,note\n1,\"quoted\" junk\n"[..],
        ] {
            assert!(inspect_csv_bytes(input, "fixture.csv").is_err());
        }
    }

    #[test]
    fn csv_inventory_rejects_unterminated_quoted_fields() {
        let input = b"id,note\n1,\"unterminated\n";
        assert!(inspect_csv_bytes(input, "fixture.csv").is_err());
    }

    #[test]
    fn csv_inventory_handles_quoted_newlines_as_one_record() {
        let input = b"id,note\n1,\"line one\nline two\"\n2,ok\n";
        let inventory = inspect_csv_bytes(input, "fixture.csv").unwrap();
        assert_eq!(inventory.data_row_count, 2);
        assert_eq!(inventory.columns[1].nonempty_cells, 2);
    }

    #[test]
    fn committed_receipt_schema_binds_csv_inventory_to_inspect_operation() {
        let schema: serde_json::Value = serde_json::from_str(
            include_str!("../schemas/audit-receipt-v0.1.schema.json")
        ).unwrap();
        let required = schema["required"].as_array().unwrap();
        for field in [
            "receipt_schema_version",
            "manifest_sha256",
            "evidence_input_sha256",
            "artifact_count",
            "record_count",
            "claim_ceiling",
        ] {
            assert!(required.iter().any(|value| value == field),
                "receipt schema does not require {field}");
        }
        assert!(schema["properties"]["csv_inventory"]["properties"]["artifact_sha256"].is_object());
        assert!(schema["properties"]["csv_inventory"]["properties"]["columns"]["items"]["properties"]["empty_cells"].is_object());
        assert!(schema["allOf"].as_array().unwrap().iter().any(|condition| {
            condition["if"]["properties"]["operation"]["const"] == "inspect-csv"
                && condition["then"]["required"].as_array().unwrap().iter()
                    .any(|value| value == "csv_inventory")
        }));
    }

    #[test]
    fn committed_observation_schema_binds_units_and_censoring_contract() {
        let schema: serde_json::Value = serde_json::from_str(include_str!("../schemas/observation-set-v0.1.schema.json")).unwrap();
        let record = &schema["properties"]["records"]["items"];
        assert_eq!(record["properties"]["raw_unit"]["type"], "string");
        for field in ["source_object_id", "source_row_id", "source_field", "raw_unit", "censor_lower_bound", "censor_upper_bound"] {
            assert!(record["properties"].get(field).is_some(), "missing schema field {field}");
        }
        for state in ["uncensored", "left_censored", "right_censored", "interval_censored"] {
            assert!(record["allOf"].as_array().unwrap().iter().any(|condition| condition["if"]["properties"]["censoring"]["const"] == state),
                "schema has no explicit bounds condition for {state}");
        }
    }

    #[test]
    fn committed_manifest_schema_requires_source_object_registry() {
        let schema: serde_json::Value = serde_json::from_str(include_str!("../schemas/measurement-manifest-v0.1.schema.json")).unwrap();
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|value| value == "source_objects"));
        let object_required = schema["properties"]["source_objects"]["items"]["required"].as_array().unwrap();
        for field in ["id", "source_doi", "source_url", "source_revision", "accessed_on", "terms_url", "terms_reviewed", "local_use_authorized"] {
            assert!(object_required.iter().any(|value| value == field), "source object schema does not require {field}");
        }
    }

    #[test]
    fn committed_manifest_schema_requires_per_artifact_rights_evidence() {
        let schema: serde_json::Value = serde_json::from_str(include_str!("../schemas/measurement-manifest-v0.1.schema.json")).unwrap();
        let required = schema["properties"]["artifacts"]["items"]["required"].as_array().unwrap();
        for field in ["source_object_id", "source_doi", "artifact_revision", "accessed_on", "terms_url", "terms_reviewed", "local_use_authorized", "repository_bundling_authorized", "sha256", "size_bytes"] {
            assert!(required.iter().any(|value| value == field), "artifact schema does not require {field}");
        }
    }
    #[test]
    fn committed_amb2025_profile_deserializes_and_validates() {
        let raw = include_str!("../schemas/amb2025-02-source-manifest-v0.1.json");
        let profile: SourceManifest = serde_json::from_str(raw).unwrap();
        validate_manifest(&profile).unwrap();
        assert_eq!(profile.dataset.doi, "10.18434/mds2-3735");
        assert!(profile.artifacts.is_empty());
    }

    #[test]
    fn metadata_only_manifest_is_valid_without_claiming_data_verification() {
        let value = manifest();
        validate_manifest(&value).unwrap();
        assert!(value.artifacts.is_empty());
        assert!(value.source_objects.is_empty());
        assert!(!value.reuse_policy.artifact_terms_reviewed);
    }

    #[test]
    fn repository_bundling_requires_review_and_local_use() {
        let mut value = manifest();
        value.reuse_policy.artifact_terms_reviewed = true;
        value.reuse_policy.repository_bundling_authorized = true;
        assert!(validate_manifest(&value).is_err());

        value.reuse_policy.local_use_authorized = true;
        assert!(validate_manifest(&value).is_ok());
    }

    #[test]
    fn rejects_local_artifacts_until_terms_reviewed() {
        let mut value = manifest();
        value.reuse_policy.artifact_terms_reviewed = true;
        value.reuse_policy.local_use_authorized = true;
        value.source_objects.push(SourceObject {
            id: "amb2025-02-targets".into(),
            source_doi: "10.18434/mds2-3735".into(),
            source_url: "https://data.nist.gov/od/id/mds2-3735".into(),
            source_revision: "source-revision-pending".into(),
            accessed_on: "2026-10-18".into(),
            terms_url: "https://data.nist.gov/terms".into(),
            terms_reviewed: true,
            local_use_authorized: true,
        });
        value.artifacts.push(SourceArtifact {
            relative_path: "results.csv".into(),
            source_object_id: "amb2025-02-targets".into(),
            source_doi: "10.18434/mds2-3735".into(),
            artifact_revision: "source-revision-pending".into(),
            accessed_on: "2026-10-18".into(),
            source_url: "https://data.nist.gov/example.csv".into(),
            terms_url: "https://data.nist.gov/terms".into(),
            terms_reviewed: false,
            local_use_authorized: false,
            repository_bundling_authorized: false,
            sha256: "a".repeat(64),
            size_bytes: 12,
            media_type: "text/csv".into(),
        });
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn rejects_path_traversal_in_artifact_paths() {
        assert!(safe_relative_path("../private.csv").is_err());
        assert!(safe_relative_path(r"..\private.csv").is_err());
        assert!(safe_relative_path("nested/data.csv").is_ok());
    }

    #[test]
    fn artifact_verifier_checks_both_bytes_and_digest() {
        let tmp = tempdir().unwrap();
        let content = b"source-bound bytes\n";
        fs::write(tmp.path().join("sample.csv"), content).unwrap();
        let mut value = record_manifest();
        value.artifacts[0].relative_path = "sample.csv".into();
        value.artifacts[0].sha256 = sha256_hex(content);
        value.artifacts[0].size_bytes = content.len() as u64;
        validate_manifest(&value).unwrap();
        verify_artifacts(&value, tmp.path()).unwrap();
        value.artifacts[0].sha256 = "0".repeat(64);
        assert!(verify_artifacts(&value, tmp.path()).is_err());
    }

    #[test]
    fn rejects_target_lineage_in_calibration_outcomes() {
        let value = record_manifest();
        let observations = ObservationSet {
            schema_version: "0.1.0".into(),
            records: vec![
                record("r1", "AMB2022-718-AMMT-B7-P4", "specimen-a", RecordRole::TargetOutcome, Split::Calibration, "yield_strength", 900.0),
                record("r2", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "ultimate_strength", 1100.0),
            ],
        };
        assert!(validate_records(&value, &observations).is_err());
    }

    #[test]
    fn accepts_disjoint_calibration_and_held_out_outcome_groups() {
        let value = record_manifest();
        let observations = ObservationSet {
            schema_version: "0.1.0".into(),
            records: vec![
                record("cal-1", "calibration-build-a", "specimen-cal-a", RecordRole::TargetOutcome, Split::Calibration, "yield_strength", 900.0),
                record("target-1", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0),
                record("target-2", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "ultimate_strength", 1100.0),
            ],
        };
        let groups = validate_records(&value, &observations).unwrap();
        assert_eq!(groups, vec![String::from("AMB2022-718-AMMT-B7-P4")]);
    }

    #[test]
    fn rejects_record_from_an_undeclared_source_object() {
        let value = record_manifest();
        let mut row = record("target", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        row.source_object_id = "invented-object-id".into();
        let observations = ObservationSet { schema_version: "0.1.0".into(), records: vec![row] };
        assert!(validate_records(&value, &observations).is_err());
    }

    #[test]
    fn rejects_record_whose_known_doi_does_not_match_its_source_object() {
        let value = record_manifest();
        let mut row = record("target", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        row.source_doi = "10.18434/mds2-2767".into();
        let observations = ObservationSet { schema_version: "0.1.0".into(), records: vec![row] };
        assert!(validate_records(&value, &observations).is_err());
    }
    #[test]
    fn rejects_record_when_its_source_object_has_no_artifact_digest() {
        let mut value = record_manifest();
        value.artifacts.clear();
        let observations = ObservationSet {
            schema_version: "0.1.0".into(),
            records: vec![record("target", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0)],
        };
        assert!(validate_records(&value, &observations).is_err());
    }

    #[test]
    fn rejects_normalization_without_preserved_source_and_conversion_note() {
        let value = record_manifest();
        let mut row = record("target", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        row.normalized_value = Some(0.92);
        row.normalized_unit = Some("GPa".into());
        let observations = ObservationSet { schema_version: "0.1.0".into(), records: vec![row] };
        assert!(validate_records(&value, &observations).is_err());
    }

    #[test]
    fn rejects_unknown_source_doi() {
        let value = record_manifest();
        let mut row = record("target", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        row.source_doi = "10.0000/not-declared".into();
        let observations = ObservationSet { schema_version: "0.1.0".into(), records: vec![row] };
        assert!(validate_records(&value, &observations).is_err());
    }

    #[test]
    fn rejects_duplicate_record_ids() {
        let value = record_manifest();
        let row1 = record("dup", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        let row2 = record("dup", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "ultimate_strength", 1100.0);
        let observations = ObservationSet { schema_version: "0.1.0".into(), records: vec![row1, row2] };
        assert!(validate_records(&value, &observations).is_err());
    }

    #[test]
    fn rejects_unknown_censoring_without_explanation() {
        let value = record_manifest();
        let mut row = record("target", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        row.censoring = Censoring::Unknown;
        let observations = ObservationSet { schema_version: "0.1.0".into(), records: vec![row] };
        assert!(validate_records(&value, &observations).is_err());
    }

    #[test]
    fn preserves_left_right_and_interval_censoring_without_fake_point_values() {
        let value = record_manifest();
        let mut left = record("left", "lineage-left", "spec-left", RecordRole::Predictor, Split::Calibration, "porosity", 0.0);
        left.raw_value = None;
        left.censoring = Censoring::LeftCensored;
        left.censor_upper_bound = Some(0.05);

        let mut right = record("right", "lineage-right", "spec-right", RecordRole::Predictor, Split::Calibration, "porosity", 0.0);
        right.raw_value = None;
        right.censoring = Censoring::RightCensored;
        right.censor_lower_bound = Some(0.8);

        let mut interval = record("interval", "lineage-interval", "spec-interval", RecordRole::Predictor, Split::Calibration, "porosity", 0.0);
        interval.raw_value = None;
        interval.censoring = Censoring::IntervalCensored;
        interval.censor_lower_bound = Some(0.1);
        interval.censor_upper_bound = Some(0.2);

        let mut target = record("target", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        target.source_field = "target_yield_strength".into();
        let observations = ObservationSet { schema_version: "0.1.0".into(), records: vec![left, right, interval, target] };
        assert!(validate_records(&value, &observations).is_ok());
    }

    #[test]
    fn rejects_censoring_bounds_that_conflict_with_state() {
        let value = record_manifest();
        let mut row = record("target", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        row.censoring = Censoring::IntervalCensored;
        row.censor_lower_bound = Some(950.0);
        row.censor_upper_bound = Some(900.0);
        let observations = ObservationSet { schema_version: "0.1.0".into(), records: vec![row] };
        assert!(validate_records(&value, &observations).is_err());
    }

    #[test]
    fn rejects_duplicate_source_object_row_field_cells() {
        let value = record_manifest();
        let first = record("obs-1", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        let mut second = record("obs-2", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        second.source_row_id = first.source_row_id.clone();
        let observations = ObservationSet {
            schema_version: "0.1.0".into(),
            records: vec![first, second],
        };
        assert!(validate_records(&value, &observations).is_err());
    }

    #[test]
    fn rejects_source_field_with_inconsistent_raw_units_or_identity() {
        let value = record_manifest();
        let one = record("r1", "cal-group", "spec-1", RecordRole::Predictor, Split::Calibration, "process_energy", 20.0);
        let mut two = record("r2", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        two.source_field = "process_energy".into();
        two.raw_unit = Some("GPa".into());
        let observations = ObservationSet { schema_version: "0.1.0".into(), records: vec![one, two] };
        assert!(validate_records(&value, &observations).is_err());
    }

    #[test]
    fn rejects_inconsistent_normalized_units_for_same_variable() {
        let value = record_manifest();
        let mut first = record("target-1", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        first.normalized_value = Some(0.92);
        first.normalized_unit = Some("GPa".into());
        first.conversion_note = Some("MPa to GPa".into());

        let mut second = record("target-2", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        second.source_field = "yield_strength_2".into();
        second.normalized_value = Some(920.0);
        second.normalized_unit = Some("MPa".into());
        second.conversion_note = Some("identity conversion".into());
        let observations = ObservationSet { schema_version: "0.1.0".into(), records: vec![first, second] };
        assert!(validate_records(&value, &observations).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn artifact_verifier_rejects_symlink_escape_from_declared_root() {
        use std::os::unix::fs::symlink;
        let root = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let bytes = b"outside data";
        let external = outside.path().join("external.csv");
        fs::write(&external, bytes).unwrap();
        symlink(&external, root.path().join("linked.csv")).unwrap();

        let mut value = record_manifest();
        value.artifacts[0].relative_path = "linked.csv".into();
        value.artifacts[0].sha256 = sha256_hex(bytes);
        value.artifacts[0].size_bytes = bytes.len() as u64;
        assert!(validate_manifest(&value).is_ok());
        assert!(verify_artifacts(&value, root.path()).is_err());
    }
    #[test]
    fn raw_value_and_unit_survive_normalization() {
        let value = record_manifest();
        let mut row = record("target", "AMB2022-718-AMMT-B7-P4", "aggregate-target", RecordRole::TargetOutcome, Split::HeldOut, "yield_strength", 920.0);
        row.normalized_value = Some(0.92);
        row.normalized_unit = Some("GPa".into());
        row.conversion_note = Some("1 MPa = 0.001 GPa; deterministic unit conversion".into());
        let observations = ObservationSet { schema_version: "0.1.0".into(), records: vec![row] };
        validate_records(&value, &observations).unwrap();
        assert_eq!(observations.records[0].raw_value, Some(920.0));
        assert_eq!(observations.records[0].raw_unit.as_deref(), Some("MPa"));
    }

    #[test]
    fn iso_date_validator_rejects_impossible_calendar_dates() {
        for invalid in [
            "2026-19-42",
            "2026-00-10",
            "2026-13-01",
            "2026-04-31",
            "2025-02-29",
            "0000-01-01",
            "2026/10/18",
        ] {
            assert!(!is_valid_iso_date(invalid), "accepted invalid date {invalid}");
        }
    }

    #[test]
    fn iso_date_validator_accepts_valid_month_ends_and_gregorian_leap_days() {
        for valid in [
            "2026-10-18",
            "2024-02-29",
            "2000-02-29",
            "2026-02-28",
            "2026-04-30",
            "2026-12-31",
        ] {
            assert!(is_valid_iso_date(valid), "rejected valid date {valid}");
        }
        assert!(!is_valid_iso_date("1900-02-29"), "1900 is not a Gregorian leap year");
    }

    #[test]
    fn http_url_validator_rejects_empty_or_malformed_authorities() {
        for valid in [
            "https://data.nist.gov/od/id/mds2-3735",
            "http://localhost:8080/manifest.json",
            "https://[2001:db8::1]:443/dataset",
        ] {
            assert!(is_http_url(valid), "rejected HTTP(S) URL {valid}");
        }
        for invalid in [
            "",
            "https://",
            "https:///resource",
            "https:// example.org/data",
            "https://@/data",
            "https://example.org:",
            "https://example.org:port/data",
            "ftp://example.org/data",
            "not-a-url",
        ] {
            assert!(!is_http_url(invalid), "accepted malformed URL {invalid}");
        }
    }
}
