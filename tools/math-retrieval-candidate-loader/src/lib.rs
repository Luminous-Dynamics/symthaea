// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MATH-RET-RUNTIME-002A — content-addressed candidate artifact loader.
//!
//! The research membership guard accepts an already-parsed candidate universe.
//! This production-hardening layer removes the remaining caller-trust seam: it
//! hashes, parses, validates, and retains the exact candidate artifact bytes
//! before constructing the immutable `FrozenCandidateUniverse` used by the
//! guard. Candidate identities are never supplied separately by the caller.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::{self, Write as _};
use std::fs;
use std::path::Path;
use symthaea_math_retrieval_membership_guard::FrozenCandidateUniverse;
use symthaea_math_retrieval_runtime_seam::Sha256Digest;

const VERSION: &str = "math-retrieval-candidate-set-v1";
const AUTHORITY: &str = "MeasurementOnly";
const SOURCE_IDENTITY_KIND: &str = "SourceObjectDigest";
const CANONICAL_ORDER: &str = "SourceObjectDigestAscending";
const ROOT_FIELDS: [&str; 10] = [
    "version",
    "candidate_set_id",
    "authority",
    "corpus_snapshot_sha256",
    "knowledge_boundary_sha256",
    "candidate_eligibility_policy_sha256",
    "source_identity_kind",
    "canonical_order",
    "candidate_count",
    "candidates",
];

/// Exact qualified expectations that the loaded artifact must satisfy.
///
/// These values come from the already-qualified graph/index binding. Candidate
/// identities themselves are intentionally absent: they are derived only from
/// the artifact bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateArtifactBinding {
    pub candidate_set_sha256: Sha256Digest,
    pub candidate_count: usize,
    pub corpus_snapshot_sha256: Sha256Digest,
    pub knowledge_boundary_sha256: Sha256Digest,
    pub candidate_eligibility_policy_sha256: Sha256Digest,
}

/// Immutable result of loading and validating one candidate artifact.
///
/// The exact bytes are retained so later evidence can refer to the bytes that
/// were actually admitted rather than reopening a mutable path.
#[derive(Debug, Clone)]
pub struct LoadedCandidateArtifact {
    artifact_bytes: Box<[u8]>,
    candidate_set_id: String,
    corpus_snapshot_sha256: Sha256Digest,
    knowledge_boundary_sha256: Sha256Digest,
    candidate_eligibility_policy_sha256: Sha256Digest,
    universe: FrozenCandidateUniverse,
}

impl LoadedCandidateArtifact {
    pub fn artifact_bytes(&self) -> &[u8] {
        &self.artifact_bytes
    }

    pub fn candidate_set_id(&self) -> &str {
        &self.candidate_set_id
    }

    pub fn candidate_set_sha256(&self) -> &Sha256Digest {
        self.universe.candidate_set_sha256()
    }

    pub fn candidate_count(&self) -> usize {
        self.universe.candidate_count()
    }

    pub fn corpus_snapshot_sha256(&self) -> &Sha256Digest {
        &self.corpus_snapshot_sha256
    }

    pub fn knowledge_boundary_sha256(&self) -> &Sha256Digest {
        &self.knowledge_boundary_sha256
    }

    pub fn candidate_eligibility_policy_sha256(&self) -> &Sha256Digest {
        &self.candidate_eligibility_policy_sha256
    }

    pub fn universe(&self) -> &FrozenCandidateUniverse {
        &self.universe
    }

    pub fn into_universe(self) -> FrozenCandidateUniverse {
        self.universe
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateLoaderError {
    Io(String),
    Json(String),
    Contract(String),
    Binding(String),
}

impl fmt::Display for CandidateLoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "candidate artifact I/O error: {message}"),
            Self::Json(message) => write!(f, "candidate artifact JSON error: {message}"),
            Self::Contract(message) => write!(f, "candidate artifact contract error: {message}"),
            Self::Binding(message) => write!(f, "candidate artifact binding error: {message}"),
        }
    }
}

impl std::error::Error for CandidateLoaderError {}

/// Trusted production boundary for the candidate-set artifact.
pub struct CandidateArtifactLoader;

impl CandidateArtifactLoader {
    /// Read one path exactly once and immediately load the resulting immutable
    /// bytes. The returned artifact never reopens `path`.
    pub fn load_file(
        path: impl AsRef<Path>,
        expected: &CandidateArtifactBinding,
    ) -> Result<LoadedCandidateArtifact, CandidateLoaderError> {
        let bytes = fs::read(path.as_ref())
            .map_err(|error| CandidateLoaderError::Io(error.to_string()))?;
        Self::load_bytes(&bytes, expected)
    }

    /// Hash, parse, validate, bind, and retain the exact artifact bytes.
    pub fn load_bytes(
        bytes: &[u8],
        expected: &CandidateArtifactBinding,
    ) -> Result<LoadedCandidateArtifact, CandidateLoaderError> {
        let actual_artifact_sha256 = sha256_digest(bytes);
        if actual_artifact_sha256 != expected.candidate_set_sha256 {
            return Err(CandidateLoaderError::Binding(format!(
                "artifact SHA-256 mismatch: expected {}, got {}",
                expected.candidate_set_sha256, actual_artifact_sha256
            )));
        }

        let value: Value = serde_json::from_slice(bytes)
            .map_err(|error| CandidateLoaderError::Json(error.to_string()))?;
        let root = value.as_object().ok_or_else(|| {
            CandidateLoaderError::Contract("root must be a JSON object".into())
        })?;
        validate_exact_root_fields(root)?;

        require_string(root, "version", VERSION)?;
        require_string(root, "authority", AUTHORITY)?;
        require_string(root, "source_identity_kind", SOURCE_IDENTITY_KIND)?;
        require_string(root, "canonical_order", CANONICAL_ORDER)?;

        let candidate_set_id = root
            .get("candidate_set_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                CandidateLoaderError::Contract(
                    "candidate_set_id must be a non-empty string".into(),
                )
            })?;
        if candidate_set_id.trim().is_empty() {
            return Err(CandidateLoaderError::Contract(
                "candidate_set_id must be a non-empty string".into(),
            ));
        }

        let corpus_snapshot_sha256 = parse_digest_field(root, "corpus_snapshot_sha256")?;
        let knowledge_boundary_sha256 = parse_digest_field(root, "knowledge_boundary_sha256")?;
        let candidate_eligibility_policy_sha256 =
            parse_digest_field(root, "candidate_eligibility_policy_sha256")?;

        compare_binding(
            "corpus_snapshot_sha256",
            &corpus_snapshot_sha256,
            &expected.corpus_snapshot_sha256,
        )?;
        compare_binding(
            "knowledge_boundary_sha256",
            &knowledge_boundary_sha256,
            &expected.knowledge_boundary_sha256,
        )?;
        compare_binding(
            "candidate_eligibility_policy_sha256",
            &candidate_eligibility_policy_sha256,
            &expected.candidate_eligibility_policy_sha256,
        )?;

        let candidate_count_u64 = root
            .get("candidate_count")
            .and_then(Value::as_u64)
            .filter(|count| *count >= 1)
            .ok_or_else(|| {
                CandidateLoaderError::Contract(
                    "candidate_count must be a positive integer".into(),
                )
            })?;
        let candidate_count = usize::try_from(candidate_count_u64).map_err(|_| {
            CandidateLoaderError::Contract("candidate_count exceeds platform usize".into())
        })?;
        if candidate_count != expected.candidate_count {
            return Err(CandidateLoaderError::Binding(format!(
                "candidate_count mismatch: expected {}, got {candidate_count}",
                expected.candidate_count
            )));
        }

        let array = root
            .get("candidates")
            .and_then(Value::as_array)
            .filter(|candidates| !candidates.is_empty())
            .ok_or_else(|| {
                CandidateLoaderError::Contract("candidates must be a non-empty array".into())
            })?;
        if array.len() != candidate_count {
            return Err(CandidateLoaderError::Contract(format!(
                "candidate_count={candidate_count} but candidates has {} items",
                array.len()
            )));
        }

        let mut candidate_text = Vec::with_capacity(array.len());
        let mut candidates = Vec::with_capacity(array.len());
        for (index, value) in array.iter().enumerate() {
            let text = value.as_str().ok_or_else(|| {
                CandidateLoaderError::Contract(format!(
                    "candidates[{index}] must be sha256:<64 lowercase hex>"
                ))
            })?;
            let digest = parse_digest_text(text, &format!("candidates[{index}]"))?;
            candidate_text.push(text);
            candidates.push(digest);
        }

        for pair in candidate_text.windows(2) {
            if pair[0] >= pair[1] {
                let reason = if pair[0] == pair[1] {
                    "duplicate source identities are forbidden"
                } else {
                    "SourceObjectDigestAscending order required"
                };
                return Err(CandidateLoaderError::Contract(reason.into()));
            }
        }

        let universe = FrozenCandidateUniverse::new(actual_artifact_sha256, candidates)
            .map_err(|error| CandidateLoaderError::Contract(error.to_string()))?;

        Ok(LoadedCandidateArtifact {
            artifact_bytes: bytes.to_vec().into_boxed_slice(),
            candidate_set_id: candidate_set_id.to_owned(),
            corpus_snapshot_sha256,
            knowledge_boundary_sha256,
            candidate_eligibility_policy_sha256,
            universe,
        })
    }
}

fn validate_exact_root_fields(root: &Map<String, Value>) -> Result<(), CandidateLoaderError> {
    let expected: BTreeSet<&str> = ROOT_FIELDS.into_iter().collect();
    let actual: BTreeSet<&str> = root.keys().map(String::as_str).collect();
    if actual != expected {
        let missing: Vec<_> = expected.difference(&actual).copied().collect();
        let extra: Vec<_> = actual.difference(&expected).copied().collect();
        return Err(CandidateLoaderError::Contract(format!(
            "root fields differ; missing={missing:?} extra={extra:?}"
        )));
    }
    Ok(())
}

fn require_string(
    root: &Map<String, Value>,
    field: &str,
    expected: &str,
) -> Result<(), CandidateLoaderError> {
    match root.get(field).and_then(Value::as_str) {
        Some(actual) if actual == expected => Ok(()),
        _ => Err(CandidateLoaderError::Contract(format!(
            "{field} must equal {expected:?}"
        ))),
    }
}

fn parse_digest_field(
    root: &Map<String, Value>,
    field: &str,
) -> Result<Sha256Digest, CandidateLoaderError> {
    let text = root.get(field).and_then(Value::as_str).ok_or_else(|| {
        CandidateLoaderError::Contract(format!(
            "{field} must be sha256:<64 lowercase hex>"
        ))
    })?;
    parse_digest_text(text, field)
}

fn parse_digest_text(text: &str, where_: &str) -> Result<Sha256Digest, CandidateLoaderError> {
    let bytes = text.as_bytes();
    let valid = bytes.len() == 71
        && bytes.starts_with(b"sha256:")
        && bytes[7..]
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte));
    if !valid {
        return Err(CandidateLoaderError::Contract(format!(
            "{where_} must be sha256:<64 lowercase hex>"
        )));
    }
    Sha256Digest::parse(text.to_owned())
        .map_err(|error| CandidateLoaderError::Contract(format!("{where_}: {error}")))
}

fn compare_binding(
    field: &str,
    actual: &Sha256Digest,
    expected: &Sha256Digest,
) -> Result<(), CandidateLoaderError> {
    if actual != expected {
        return Err(CandidateLoaderError::Binding(format!(
            "{field} mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn sha256_digest(bytes: &[u8]) -> Sha256Digest {
    let raw = Sha256::digest(bytes);
    let mut text = String::with_capacity(71);
    text.push_str("sha256:");
    for byte in raw {
        write!(&mut text, "{byte:02x}").expect("writing SHA-256 to String cannot fail");
    }
    Sha256Digest::parse(text).expect("internally generated SHA-256 must parse")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn d(n: u8) -> String {
        format!("sha256:{n:064x}")
    }

    fn parse(n: u8) -> Sha256Digest {
        Sha256Digest::parse(d(n)).unwrap()
    }

    fn fixture() -> Value {
        json!({
            "version": VERSION,
            "candidate_set_id": "fixture-v1",
            "authority": AUTHORITY,
            "corpus_snapshot_sha256": d(100),
            "knowledge_boundary_sha256": d(101),
            "candidate_eligibility_policy_sha256": d(102),
            "source_identity_kind": SOURCE_IDENTITY_KIND,
            "canonical_order": CANONICAL_ORDER,
            "candidate_count": 3,
            "candidates": [d(20), d(21), d(22)]
        })
    }

    fn bytes(value: &Value) -> Vec<u8> {
        serde_json::to_vec(value).unwrap()
    }

    fn binding(raw: &[u8], value: &Value) -> CandidateArtifactBinding {
        CandidateArtifactBinding {
            candidate_set_sha256: sha256_digest(raw),
            candidate_count: value["candidate_count"].as_u64().unwrap() as usize,
            corpus_snapshot_sha256: Sha256Digest::parse(
                value["corpus_snapshot_sha256"].as_str().unwrap().to_owned(),
            )
            .unwrap(),
            knowledge_boundary_sha256: Sha256Digest::parse(
                value["knowledge_boundary_sha256"].as_str().unwrap().to_owned(),
            )
            .unwrap(),
            candidate_eligibility_policy_sha256: Sha256Digest::parse(
                value["candidate_eligibility_policy_sha256"]
                    .as_str()
                    .unwrap()
                    .to_owned(),
            )
            .unwrap(),
        }
    }

    fn assert_contract_error(value: Value) {
        let raw = bytes(&value);
        let expected = binding(&raw, &value);
        let error = CandidateArtifactLoader::load_bytes(&raw, &expected).unwrap_err();
        assert!(matches!(error, CandidateLoaderError::Contract(_)));
    }

    #[test]
    fn sha256_matches_nist_abc_vector() {
        assert_eq!(
            sha256_digest(b"abc").to_string(),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn exact_artifact_bytes_reconstruct_frozen_universe() {
        let value = fixture();
        let raw = bytes(&value);
        let expected = binding(&raw, &value);
        let loaded = CandidateArtifactLoader::load_bytes(&raw, &expected).unwrap();

        assert_eq!(loaded.artifact_bytes(), raw);
        assert_eq!(loaded.candidate_set_id(), "fixture-v1");
        assert_eq!(loaded.candidate_count(), 3);
        assert_eq!(loaded.candidate_set_sha256(), &expected.candidate_set_sha256);
        assert!(loaded.universe().contains(&parse(20)));
        assert!(loaded.universe().contains(&parse(21)));
        assert!(loaded.universe().contains(&parse(22)));
    }

    #[test]
    fn one_byte_artifact_mutation_is_rejected_against_qualified_binding() {
        let value = fixture();
        let raw = bytes(&value);
        let expected = binding(&raw, &value);
        let mut mutated = raw.clone();
        mutated.push(b'\n');

        let error = CandidateArtifactLoader::load_bytes(&mutated, &expected).unwrap_err();
        assert!(matches!(error, CandidateLoaderError::Binding(_)));
    }

    #[test]
    fn structural_candidate_contract_attacks_are_rejected_even_with_rebound_hash() {
        let mut unsorted = fixture();
        unsorted["candidates"] = json!([d(21), d(20), d(22)]);
        assert_contract_error(unsorted);

        let mut duplicate = fixture();
        duplicate["candidates"] = json!([d(20), d(20), d(22)]);
        assert_contract_error(duplicate);

        let mut count = fixture();
        count["candidate_count"] = json!(2);
        assert_contract_error(count);

        let mut extra = fixture();
        extra["unexpected"] = json!(true);
        assert_contract_error(extra);

        let mut authority = fixture();
        authority["authority"] = json!("FormalAuthority");
        assert_contract_error(authority);

        let mut uppercase = fixture();
        uppercase["candidates"][0] = json!(format!("sha256:{}", "A".repeat(64)));
        assert_contract_error(uppercase);
    }

    #[test]
    fn qualified_metadata_bindings_are_checked_independently() {
        let value = fixture();
        let raw = bytes(&value);

        let mut expected = binding(&raw, &value);
        expected.corpus_snapshot_sha256 = parse(1);
        assert!(matches!(
            CandidateArtifactLoader::load_bytes(&raw, &expected).unwrap_err(),
            CandidateLoaderError::Binding(_)
        ));

        let mut expected = binding(&raw, &value);
        expected.knowledge_boundary_sha256 = parse(2);
        assert!(matches!(
            CandidateArtifactLoader::load_bytes(&raw, &expected).unwrap_err(),
            CandidateLoaderError::Binding(_)
        ));

        let mut expected = binding(&raw, &value);
        expected.candidate_eligibility_policy_sha256 = parse(3);
        assert!(matches!(
            CandidateArtifactLoader::load_bytes(&raw, &expected).unwrap_err(),
            CandidateLoaderError::Binding(_)
        ));

        let mut expected = binding(&raw, &value);
        expected.candidate_count = 4;
        assert!(matches!(
            CandidateArtifactLoader::load_bytes(&raw, &expected).unwrap_err(),
            CandidateLoaderError::Binding(_)
        ));
    }

    #[test]
    fn query_specific_exclusion_is_not_conflated_with_static_universe_loading() {
        let mut value = fixture();
        value["candidates"] = json!([d(20), d(21), d(250)]);
        let raw = bytes(&value);
        let expected = binding(&raw, &value);
        let loaded = CandidateArtifactLoader::load_bytes(&raw, &expected).unwrap();

        // 250 may be the query source in a later request. Static eligibility
        // still admits it; query-source exclusion remains the executor's job.
        assert!(loaded.universe().contains(&parse(250)));
    }

    #[test]
    fn already_loaded_universe_is_immune_to_later_path_substitution() {
        let value = fixture();
        let raw = bytes(&value);
        let expected = binding(&raw, &value);

        let unique = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "symthaea-math-ret-candidate-loader-{}-{unique}.json",
            std::process::id()
        ));
        fs::write(&path, &raw).unwrap();
        let loaded = CandidateArtifactLoader::load_file(&path, &expected).unwrap();

        let mut replacement = fixture();
        replacement["candidates"] = json!([d(30), d(31), d(32)]);
        fs::write(&path, bytes(&replacement)).unwrap();

        assert_eq!(loaded.artifact_bytes(), raw);
        assert!(loaded.universe().contains(&parse(20)));
        assert!(!loaded.universe().contains(&parse(30)));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn malformed_or_empty_candidate_documents_fail_closed() {
        let raw = b"not-json";
        let expected = CandidateArtifactBinding {
            candidate_set_sha256: sha256_digest(raw),
            candidate_count: 1,
            corpus_snapshot_sha256: parse(100),
            knowledge_boundary_sha256: parse(101),
            candidate_eligibility_policy_sha256: parse(102),
        };
        assert!(matches!(
            CandidateArtifactLoader::load_bytes(raw, &expected).unwrap_err(),
            CandidateLoaderError::Json(_)
        ));

        let mut value = fixture();
        value["candidate_set_id"] = json!("   ");
        assert_contract_error(value);

        let mut value = fixture();
        value["candidate_count"] = json!(0);
        value["candidates"] = json!([]);
        assert_contract_error(value);
    }
}
