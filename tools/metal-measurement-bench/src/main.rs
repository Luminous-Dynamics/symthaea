use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
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
struct AuditReceipt {
    receipt_schema_version: &'static str,
    operation: &'static str,
    status: &'static str,
    manifest_sha256: String,
    evidence_input_sha256: Option<String>,
    artifact_count: usize,
    record_count: usize,
    held_out_outcome_groups: Vec<String>,
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

fn nonblank(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_http_url(value: &str) -> bool {
    value.starts_with("https://") || value.starts_with("http://")
}

fn is_iso_date_shape(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes[..4].iter().all(|byte| byte.is_ascii_digit())
        && bytes[5..7].iter().all(|byte| byte.is_ascii_digit())
        && bytes[8..10].iter().all(|byte| byte.is_ascii_digit())
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
    if !is_iso_date_shape(&manifest.dataset.metadata_accessed_on) {
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
        if !is_iso_date_shape(&object.accessed_on) {
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
        if !is_iso_date_shape(&artifact.accessed_on) {
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

        let source_key = (record.source_doi.clone(), record.source_field.clone());
        let field_contract = (record.variable_id.clone(), raw_unit.to_string());
        if let Some(previous) = source_field_contracts.get(&source_key) {
            if previous != &field_contract {
                return Err(AdapterError::Validation(format!(
                    "source field {:?} changes variable identity or raw unit within source {:?}: {:?} vs {:?}",
                    record.source_field, record.source_doi, previous, field_contract
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
        operation,
        status,
        manifest_sha256: sha256_hex(manifest_bytes),
        evidence_input_sha256,
        artifact_count,
        record_count,
        held_out_outcome_groups: groups,
        claim_ceiling: "source/contract integrity only; not evidence of model accuracy or physical qualification",
    }
}

fn run(args: &[String]) -> Result<AuditReceipt> {
    if args.len() < 3 {
        return Err(AdapterError::Validation(
            "usage: metal-measurement-bench validate-manifest MANIFEST.json | verify-artifacts MANIFEST.json ARTIFACT_DIR | validate-records MANIFEST.json OBSERVATIONS.json ARTIFACT_DIR".into(),
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
                id: "AMB2022-01".into(),
                doi: "10.18434/mds2-2767".into(),
                role: "upstream microstructure".into(),
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
}
