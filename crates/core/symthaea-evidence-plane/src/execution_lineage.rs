// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical computational-input lineage for evidence-bearing executions.
//!
//! This module identifies the computational inputs and declared environment of
//! an execution. It deliberately does **not** identify one run occurrence:
//! timestamps, run IDs, outcomes, metrics, and output evidence belong in
//! receipts/reports that bind this lineage digest.
//!
//! Environment discovery is also deliberately outside this library. Callers
//! must explicitly supply relevant environment variables and immutable inputs;
//! library code must never scrape arbitrary process environment or secrets.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::Write as _;

const EXECUTION_LINEAGE_DOMAIN_V1: &[u8] = b"symthaea.evidence.execution-lineage.v1\0";

/// Digest algorithm used by an externally-produced immutable artifact digest.
///
/// The execution-lineage commitment itself is always BLAKE3; this enum permits
/// the lineage to bind existing SHA-256 evidence without re-hashing or erasing
/// the algorithm identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentDigestAlgorithmV1 {
    Blake3,
    Sha256,
}

impl ContentDigestAlgorithmV1 {
    const fn token(self) -> &'static str {
        match self {
            Self::Blake3 => "blake3",
            Self::Sha256 => "sha256",
        }
    }
}

/// Algorithm-labelled 256-bit content digest.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ContentDigestV1 {
    pub algorithm: ContentDigestAlgorithmV1,
    pub hex: String,
}

impl ContentDigestV1 {
    pub fn new(
        algorithm: ContentDigestAlgorithmV1,
        hex: impl Into<String>,
    ) -> Result<Self, ExecutionLineageErrorV1> {
        let digest = Self {
            algorithm,
            hex: hex.into().to_ascii_lowercase(),
        };
        digest.validate()?;
        Ok(digest)
    }

    pub fn validate(&self) -> Result<(), ExecutionLineageErrorV1> {
        if !is_lower_hex_digest(&self.hex) {
            return Err(ExecutionLineageErrorV1::InvalidDigest);
        }
        Ok(())
    }

    fn feed(&self, hasher: &mut blake3::Hasher) {
        feed_str(hasher, self.algorithm.token());
        feed_str(hasher, &self.hex);
    }
}

/// Named immutable artifact or dependency-lock digest.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NamedDigestV1 {
    pub name: String,
    pub digest: ContentDigestV1,
}

impl NamedDigestV1 {
    pub fn new(
        name: impl Into<String>,
        digest: ContentDigestV1,
    ) -> Result<Self, ExecutionLineageErrorV1> {
        let value = Self {
            name: name.into(),
            digest,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), ExecutionLineageErrorV1> {
        require_non_empty(&self.name)?;
        self.digest.validate()
    }
}

/// Named version/value included in the declared execution environment.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NamedValueV1 {
    pub name: String,
    pub value: String,
}

impl NamedValueV1 {
    pub fn new(
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, ExecutionLineageErrorV1> {
        let named = Self {
            name: name.into(),
            value: value.into(),
        };
        named.validate()?;
        Ok(named)
    }

    fn validate(&self) -> Result<(), ExecutionLineageErrorV1> {
        require_non_empty(&self.name)?;
        require_non_empty(&self.value)
    }
}

/// Caller-supplied inputs from which a canonical lineage is sealed.
///
/// Named collections may arrive in any order. Sealing sorts them and rejects
/// duplicate names so conflicting values cannot be silently selected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionLineageInputsV1 {
    pub source_repository: String,
    pub source_revision: String,
    pub source_tree_digest: ContentDigestV1,
    pub dependency_locks: Vec<NamedDigestV1>,
    pub toolchain: Vec<NamedValueV1>,
    pub host_triple: String,
    pub target_triple: String,
    /// Nix derivation/store/environment identity when execution is Nix-bound.
    /// `None` explicitly means no Nix environment identity was declared.
    pub nix_environment: Option<String>,
    pub feature_flags: Vec<String>,
    pub working_directory: String,
    /// Exact argv. Element 0 must identify the program; later empty arguments
    /// are allowed because an empty argv element can be semantically meaningful.
    pub command_argv: Vec<String>,
    /// Explicit allow-list only. Do not populate this by dumping process env.
    pub relevant_env: Vec<NamedValueV1>,
    pub input_artifacts: Vec<NamedDigestV1>,
}

/// Canonical, content-addressed computational lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionLineageV1 {
    pub schema_id: String,
    pub inputs: ExecutionLineageInputsV1,
    pub lineage_digest_hex: String,
}

impl ExecutionLineageV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.evidence.execution-lineage.v1";

    pub fn seal(
        mut inputs: ExecutionLineageInputsV1,
    ) -> Result<Self, ExecutionLineageErrorV1> {
        canonicalize_inputs(&mut inputs)?;
        validate_inputs(&inputs)?;
        let mut lineage = Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            inputs,
            lineage_digest_hex: String::new(),
        };
        lineage.lineage_digest_hex = lineage.compute_digest_hex()?;
        lineage.validate()?;
        Ok(lineage)
    }

    pub fn validate(&self) -> Result<(), ExecutionLineageErrorV1> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err(ExecutionLineageErrorV1::SchemaMismatch);
        }
        validate_inputs(&self.inputs)?;
        if !inputs_are_canonical(&self.inputs) {
            return Err(ExecutionLineageErrorV1::NonCanonicalOrdering);
        }
        if !is_lower_hex_digest(&self.lineage_digest_hex) {
            return Err(ExecutionLineageErrorV1::InvalidLineageDigest);
        }
        if self.lineage_digest_hex != self.compute_digest_hex()? {
            return Err(ExecutionLineageErrorV1::LineageDigestMismatch);
        }
        Ok(())
    }

    pub fn compute_digest_hex(&self) -> Result<String, ExecutionLineageErrorV1> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err(ExecutionLineageErrorV1::SchemaMismatch);
        }
        validate_inputs(&self.inputs)?;
        if !inputs_are_canonical(&self.inputs) {
            return Err(ExecutionLineageErrorV1::NonCanonicalOrdering);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(EXECUTION_LINEAGE_DOMAIN_V1);
        feed_str(&mut hasher, &self.schema_id);
        feed_str(&mut hasher, &self.inputs.source_repository);
        feed_str(&mut hasher, &self.inputs.source_revision);
        self.inputs.source_tree_digest.feed(&mut hasher);
        feed_named_digests(&mut hasher, &self.inputs.dependency_locks);
        feed_named_values(&mut hasher, &self.inputs.toolchain);
        feed_str(&mut hasher, &self.inputs.host_triple);
        feed_str(&mut hasher, &self.inputs.target_triple);
        feed_option_str(&mut hasher, self.inputs.nix_environment.as_deref());
        feed_strings(&mut hasher, &self.inputs.feature_flags);
        feed_str(&mut hasher, &self.inputs.working_directory);
        feed_strings(&mut hasher, &self.inputs.command_argv);
        feed_named_values(&mut hasher, &self.inputs.relevant_env);
        feed_named_digests(&mut hasher, &self.inputs.input_artifacts);
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

/// Field families reported when two otherwise-valid lineages differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionLineageFieldV1 {
    SourceRepository,
    SourceRevision,
    SourceTreeDigest,
    DependencyLocks,
    Toolchain,
    HostTriple,
    TargetTriple,
    NixEnvironment,
    FeatureFlags,
    WorkingDirectory,
    CommandArgv,
    RelevantEnvironment,
    InputArtifacts,
}

/// Deterministic field-level description of execution-lineage drift.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionLineageDriftV1 {
    pub prepared_lineage_digest_hex: String,
    pub observed_lineage_digest_hex: String,
    pub changed_fields: Vec<ExecutionLineageFieldV1>,
}

impl ExecutionLineageDriftV1 {
    pub fn between(
        prepared: &ExecutionLineageV1,
        observed: &ExecutionLineageV1,
    ) -> Result<Option<Self>, ExecutionLineageErrorV1> {
        prepared.validate()?;
        observed.validate()?;
        if prepared.lineage_digest_hex == observed.lineage_digest_hex {
            return Ok(None);
        }

        let left = &prepared.inputs;
        let right = &observed.inputs;
        let mut changed_fields = Vec::new();
        push_if_changed(
            &mut changed_fields,
            ExecutionLineageFieldV1::SourceRepository,
            left.source_repository != right.source_repository,
        );
        push_if_changed(
            &mut changed_fields,
            ExecutionLineageFieldV1::SourceRevision,
            left.source_revision != right.source_revision,
        );
        push_if_changed(
            &mut changed_fields,
            ExecutionLineageFieldV1::SourceTreeDigest,
            left.source_tree_digest != right.source_tree_digest,
        );
        push_if_changed(
            &mut changed_fields,
            ExecutionLineageFieldV1::DependencyLocks,
            left.dependency_locks != right.dependency_locks,
        );
        push_if_changed(
            &mut changed_fields,
            ExecutionLineageFieldV1::Toolchain,
            left.toolchain != right.toolchain,
        );
        push_if_changed(
            &mut changed_fields,
            ExecutionLineageFieldV1::HostTriple,
            left.host_triple != right.host_triple,
        );
        push_if_changed(
            &mut changed_fields,
            ExecutionLineageFieldV1::TargetTriple,
            left.target_triple != right.target_triple,
        );
        push_if_changed(
            &mut changed_fields,
            ExecutionLineageFieldV1::NixEnvironment,
            left.nix_environment != right.nix_environment,
        );
        push_if_changed(
            &mut changed_fields,
            ExecutionLineageFieldV1::FeatureFlags,
            left.feature_flags != right.feature_flags,
        );
        push_if_changed(
            &mut changed_fields,
            ExecutionLineageFieldV1::WorkingDirectory,
            left.working_directory != right.working_directory,
        );
        push_if_changed(
            &mut changed_fields,
            ExecutionLineageFieldV1::CommandArgv,
            left.command_argv != right.command_argv,
        );
        push_if_changed(
            &mut changed_fields,
            ExecutionLineageFieldV1::RelevantEnvironment,
            left.relevant_env != right.relevant_env,
        );
        push_if_changed(
            &mut changed_fields,
            ExecutionLineageFieldV1::InputArtifacts,
            left.input_artifacts != right.input_artifacts,
        );

        if changed_fields.is_empty() {
            return Err(ExecutionLineageErrorV1::UnclassifiedLineageDrift);
        }
        Ok(Some(Self {
            prepared_lineage_digest_hex: prepared.lineage_digest_hex.clone(),
            observed_lineage_digest_hex: observed.lineage_digest_hex.clone(),
            changed_fields,
        }))
    }
}

/// Decision produced when an evidence session observes an execution lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceLineageDecisionV1 {
    Stable,
    ReprepareBeforeEvidence(ExecutionLineageDriftV1),
    RefuseMixedLineageAfterEvidence(ExecutionLineageDriftV1),
}

/// Stateful enforcement of the no-mixed-evidence-lineage rule.
///
/// The guard never automatically adopts a changed lineage. Before evidence,
/// callers must explicitly `reprepare`; after evidence starts, re-prepare is
/// refused and a new evidence session/lineage root is required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLineageGuardV1 {
    pub schema_id: String,
    pub prepared_lineage: ExecutionLineageV1,
    pub evidence_records_committed: u64,
}

impl EvidenceLineageGuardV1 {
    pub const SCHEMA_ID: &'static str = "symthaea.evidence.lineage-guard.v1";

    pub fn prepare(
        lineage: ExecutionLineageV1,
    ) -> Result<Self, ExecutionLineageErrorV1> {
        lineage.validate()?;
        Ok(Self {
            schema_id: Self::SCHEMA_ID.to_string(),
            prepared_lineage: lineage,
            evidence_records_committed: 0,
        })
    }

    pub fn validate(&self) -> Result<(), ExecutionLineageErrorV1> {
        if self.schema_id != Self::SCHEMA_ID {
            return Err(ExecutionLineageErrorV1::GuardSchemaMismatch);
        }
        self.prepared_lineage.validate()
    }

    pub fn assess(
        &self,
        observed: &ExecutionLineageV1,
    ) -> Result<EvidenceLineageDecisionV1, ExecutionLineageErrorV1> {
        self.validate()?;
        observed.validate()?;
        let Some(drift) = ExecutionLineageDriftV1::between(&self.prepared_lineage, observed)? else {
            return Ok(EvidenceLineageDecisionV1::Stable);
        };
        if self.evidence_records_committed == 0 {
            Ok(EvidenceLineageDecisionV1::ReprepareBeforeEvidence(drift))
        } else {
            Ok(EvidenceLineageDecisionV1::RefuseMixedLineageAfterEvidence(
                drift,
            ))
        }
    }

    /// Explicitly adopt a new lineage only while no claim-bearing evidence has
    /// been committed under the current prepared lineage.
    pub fn reprepare(
        &mut self,
        lineage: ExecutionLineageV1,
    ) -> Result<(), ExecutionLineageErrorV1> {
        self.validate()?;
        lineage.validate()?;
        if self.evidence_records_committed != 0 {
            return Err(ExecutionLineageErrorV1::CannotReprepareAfterEvidence);
        }
        self.prepared_lineage = lineage;
        Ok(())
    }

    /// Commit one evidence record only when the observed lineage is exactly the
    /// prepared lineage. Returns the new committed-record count.
    pub fn commit_evidence(
        &mut self,
        observed: &ExecutionLineageV1,
    ) -> Result<u64, ExecutionLineageErrorV1> {
        match self.assess(observed)? {
            EvidenceLineageDecisionV1::Stable => {}
            EvidenceLineageDecisionV1::ReprepareBeforeEvidence(_) => {
                return Err(ExecutionLineageErrorV1::ReprepareRequiredBeforeEvidence);
            }
            EvidenceLineageDecisionV1::RefuseMixedLineageAfterEvidence(_) => {
                return Err(ExecutionLineageErrorV1::MixedLineageAfterEvidence);
            }
        }
        self.evidence_records_committed = self
            .evidence_records_committed
            .checked_add(1)
            .ok_or(ExecutionLineageErrorV1::EvidenceCountOverflow)?;
        Ok(self.evidence_records_committed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionLineageErrorV1 {
    EmptyField,
    InvalidDigest,
    DuplicateNamedEntry { category: &'static str, name: String },
    EmptyCommand,
    SchemaMismatch,
    GuardSchemaMismatch,
    NonCanonicalOrdering,
    InvalidLineageDigest,
    LineageDigestMismatch,
    UnclassifiedLineageDrift,
    ReprepareRequiredBeforeEvidence,
    MixedLineageAfterEvidence,
    CannotReprepareAfterEvidence,
    EvidenceCountOverflow,
}

impl fmt::Display for ExecutionLineageErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField => write!(f, "execution lineage contains an empty required field"),
            Self::InvalidDigest => write!(f, "execution lineage contains an invalid content digest"),
            Self::DuplicateNamedEntry { category, name } => {
                write!(f, "duplicate execution-lineage {category} entry: {name}")
            }
            Self::EmptyCommand => write!(f, "execution lineage command argv is empty or has no program"),
            Self::SchemaMismatch => write!(f, "execution lineage schema mismatch"),
            Self::GuardSchemaMismatch => write!(f, "evidence lineage guard schema mismatch"),
            Self::NonCanonicalOrdering => write!(f, "execution lineage is not canonically ordered"),
            Self::InvalidLineageDigest => write!(f, "execution lineage digest is invalid"),
            Self::LineageDigestMismatch => write!(f, "execution lineage digest mismatch"),
            Self::UnclassifiedLineageDrift => write!(f, "execution lineage changed without a classified field difference"),
            Self::ReprepareRequiredBeforeEvidence => write!(f, "execution lineage drifted before evidence; re-prepare is required"),
            Self::MixedLineageAfterEvidence => write!(f, "execution lineage drifted after evidence began; mixed lineage is refused"),
            Self::CannotReprepareAfterEvidence => write!(f, "cannot re-prepare an evidence lineage after evidence has begun"),
            Self::EvidenceCountOverflow => write!(f, "evidence-lineage committed record count overflow"),
        }
    }
}

impl std::error::Error for ExecutionLineageErrorV1 {}

fn canonicalize_inputs(
    inputs: &mut ExecutionLineageInputsV1,
) -> Result<(), ExecutionLineageErrorV1> {
    canonicalize_named_digests("dependency lock", &mut inputs.dependency_locks)?;
    canonicalize_named_values("toolchain", &mut inputs.toolchain)?;
    canonicalize_named_values("relevant environment", &mut inputs.relevant_env)?;
    canonicalize_named_digests("input artifact", &mut inputs.input_artifacts)?;
    for feature in &inputs.feature_flags {
        require_non_empty(feature)?;
    }
    inputs.feature_flags.sort();
    inputs.feature_flags.dedup();
    Ok(())
}

fn canonicalize_named_digests(
    category: &'static str,
    values: &mut [NamedDigestV1],
) -> Result<(), ExecutionLineageErrorV1> {
    for value in values.iter() {
        value.validate()?;
    }
    values.sort_by(|left, right| left.name.cmp(&right.name));
    reject_duplicate_names(category, values.iter().map(|value| value.name.as_str()))
}

fn canonicalize_named_values(
    category: &'static str,
    values: &mut [NamedValueV1],
) -> Result<(), ExecutionLineageErrorV1> {
    for value in values.iter() {
        value.validate()?;
    }
    values.sort_by(|left, right| left.name.cmp(&right.name));
    reject_duplicate_names(category, values.iter().map(|value| value.name.as_str()))
}

fn reject_duplicate_names<'a>(
    category: &'static str,
    names: impl Iterator<Item = &'a str>,
) -> Result<(), ExecutionLineageErrorV1> {
    let mut previous: Option<&str> = None;
    for name in names {
        if previous == Some(name) {
            return Err(ExecutionLineageErrorV1::DuplicateNamedEntry {
                category,
                name: name.to_string(),
            });
        }
        previous = Some(name);
    }
    Ok(())
}

fn validate_inputs(inputs: &ExecutionLineageInputsV1) -> Result<(), ExecutionLineageErrorV1> {
    require_non_empty(&inputs.source_repository)?;
    require_non_empty(&inputs.source_revision)?;
    inputs.source_tree_digest.validate()?;
    require_non_empty(&inputs.host_triple)?;
    require_non_empty(&inputs.target_triple)?;
    require_non_empty(&inputs.working_directory)?;
    if inputs.command_argv.is_empty()
        || inputs.command_argv[0].trim().is_empty()
    {
        return Err(ExecutionLineageErrorV1::EmptyCommand);
    }
    if inputs
        .nix_environment
        .as_ref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(ExecutionLineageErrorV1::EmptyField);
    }
    for feature in &inputs.feature_flags {
        require_non_empty(feature)?;
    }
    for value in &inputs.dependency_locks {
        value.validate()?;
    }
    for value in &inputs.toolchain {
        value.validate()?;
    }
    for value in &inputs.relevant_env {
        value.validate()?;
    }
    for value in &inputs.input_artifacts {
        value.validate()?;
    }
    Ok(())
}

fn inputs_are_canonical(inputs: &ExecutionLineageInputsV1) -> bool {
    sorted_unique_by_name(inputs.dependency_locks.iter().map(|value| value.name.as_str()))
        && sorted_unique_by_name(inputs.toolchain.iter().map(|value| value.name.as_str()))
        && sorted_unique_by_name(inputs.relevant_env.iter().map(|value| value.name.as_str()))
        && sorted_unique_by_name(inputs.input_artifacts.iter().map(|value| value.name.as_str()))
        && strictly_sorted_unique(inputs.feature_flags.iter().map(String::as_str))
}

fn sorted_unique_by_name<'a>(names: impl Iterator<Item = &'a str>) -> bool {
    strictly_sorted_unique(names)
}

fn strictly_sorted_unique<'a>(values: impl Iterator<Item = &'a str>) -> bool {
    let mut previous: Option<&str> = None;
    for value in values {
        if previous.is_some_and(|prior| prior >= value) {
            return false;
        }
        previous = Some(value);
    }
    true
}

fn require_non_empty(value: &str) -> Result<(), ExecutionLineageErrorV1> {
    if value.trim().is_empty() {
        Err(ExecutionLineageErrorV1::EmptyField)
    } else {
        Ok(())
    }
}

fn push_if_changed(
    fields: &mut Vec<ExecutionLineageFieldV1>,
    field: ExecutionLineageFieldV1,
    changed: bool,
) {
    if changed {
        fields.push(field);
    }
}

fn feed_named_digests(hasher: &mut blake3::Hasher, values: &[NamedDigestV1]) {
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        feed_str(hasher, &value.name);
        value.digest.feed(hasher);
    }
}

fn feed_named_values(hasher: &mut blake3::Hasher, values: &[NamedValueV1]) {
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        feed_str(hasher, &value.name);
        feed_str(hasher, &value.value);
    }
}

fn feed_strings(hasher: &mut blake3::Hasher, values: &[String]) {
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        feed_str(hasher, value);
    }
}

fn feed_option_str(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            feed_str(hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn feed_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: char) -> ContentDigestV1 {
        ContentDigestV1::new(ContentDigestAlgorithmV1::Sha256, seed.to_string().repeat(64))
            .unwrap()
    }

    fn inputs(reverse: bool) -> ExecutionLineageInputsV1 {
        let mut dependency_locks = vec![
            NamedDigestV1::new("flake.lock", digest('b')).unwrap(),
            NamedDigestV1::new("Cargo.lock", digest('a')).unwrap(),
        ];
        let mut toolchain = vec![
            NamedValueV1::new("rustc", "rustc 1.96.0").unwrap(),
            NamedValueV1::new("cargo", "cargo 1.96.0").unwrap(),
            NamedValueV1::new("nix", "nix 2.31").unwrap(),
        ];
        let mut relevant_env = vec![
            NamedValueV1::new("RUSTFLAGS", "-C target-cpu=x86-64-v3").unwrap(),
            NamedValueV1::new("OMP_NUM_THREADS", "1").unwrap(),
        ];
        let mut input_artifacts = vec![
            NamedDigestV1::new("dataset", digest('d')).unwrap(),
            NamedDigestV1::new("model-config", digest('c')).unwrap(),
        ];
        let mut feature_flags = vec!["humanoid".to_string(), "research".to_string()];
        if reverse {
            dependency_locks.reverse();
            toolchain.reverse();
            relevant_env.reverse();
            input_artifacts.reverse();
            feature_flags.reverse();
        }
        ExecutionLineageInputsV1 {
            source_repository: "Luminous-Dynamics/symthaea".into(),
            source_revision: "0123456789abcdef".into(),
            source_tree_digest: digest('e'),
            dependency_locks,
            toolchain,
            host_triple: "x86_64-unknown-linux-gnu".into(),
            target_triple: "x86_64-unknown-linux-gnu".into(),
            nix_environment: Some("/nix/store/example-symthaea-env".into()),
            feature_flags,
            working_directory: "/workspace/symthaea".into(),
            command_argv: vec![
                "cargo".into(),
                "test".into(),
                "-p".into(),
                "symthaea-humanoid".into(),
            ],
            relevant_env,
            input_artifacts,
        }
    }

    fn lineage() -> ExecutionLineageV1 {
        ExecutionLineageV1::seal(inputs(false)).unwrap()
    }

    #[test]
    fn canonical_order_produces_identical_lineage_digest() {
        let left = ExecutionLineageV1::seal(inputs(false)).unwrap();
        let right = ExecutionLineageV1::seal(inputs(true)).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.lineage_digest_hex, right.lineage_digest_hex);
    }

    #[test]
    fn exact_duplicate_feature_flags_are_canonicalized() {
        let mut candidate = inputs(false);
        candidate.feature_flags.push("humanoid".into());
        let sealed = ExecutionLineageV1::seal(candidate).unwrap();
        assert_eq!(sealed.inputs.feature_flags, vec!["humanoid", "research"]);
    }

    #[test]
    fn conflicting_duplicate_named_entries_fail_closed() {
        let mut duplicate_toolchain = inputs(false);
        duplicate_toolchain
            .toolchain
            .push(NamedValueV1::new("rustc", "different").unwrap());
        assert!(matches!(
            ExecutionLineageV1::seal(duplicate_toolchain),
            Err(ExecutionLineageErrorV1::DuplicateNamedEntry {
                category: "toolchain",
                ..
            })
        ));

        let mut duplicate_lock = inputs(false);
        duplicate_lock
            .dependency_locks
            .push(NamedDigestV1::new("Cargo.lock", digest('f')).unwrap());
        assert!(matches!(
            ExecutionLineageV1::seal(duplicate_lock),
            Err(ExecutionLineageErrorV1::DuplicateNamedEntry {
                category: "dependency lock",
                ..
            })
        ));

        let mut duplicate_env = inputs(false);
        duplicate_env
            .relevant_env
            .push(NamedValueV1::new("RUSTFLAGS", "different").unwrap());
        assert!(matches!(
            ExecutionLineageV1::seal(duplicate_env),
            Err(ExecutionLineageErrorV1::DuplicateNamedEntry {
                category: "relevant environment",
                ..
            })
        ));

        let mut duplicate_input = inputs(false);
        duplicate_input
            .input_artifacts
            .push(NamedDigestV1::new("dataset", digest('f')).unwrap());
        assert!(matches!(
            ExecutionLineageV1::seal(duplicate_input),
            Err(ExecutionLineageErrorV1::DuplicateNamedEntry {
                category: "input artifact",
                ..
            })
        ));
    }

    #[test]
    fn field_drift_changes_digest_and_is_categorized() {
        let prepared = lineage();
        let mut changed_inputs = inputs(false);
        changed_inputs.source_revision = "fedcba9876543210".into();
        changed_inputs
            .relevant_env
            .iter_mut()
            .find(|value| value.name == "OMP_NUM_THREADS")
            .unwrap()
            .value = "2".into();
        let observed = ExecutionLineageV1::seal(changed_inputs).unwrap();
        assert_ne!(prepared.lineage_digest_hex, observed.lineage_digest_hex);
        let drift = ExecutionLineageDriftV1::between(&prepared, &observed)
            .unwrap()
            .unwrap();
        assert_eq!(
            drift.changed_fields,
            vec![
                ExecutionLineageFieldV1::SourceRevision,
                ExecutionLineageFieldV1::RelevantEnvironment,
            ]
        );
    }

    #[test]
    fn same_lineage_guard_is_stable_and_commits_evidence() {
        let prepared = lineage();
        let mut guard = EvidenceLineageGuardV1::prepare(prepared.clone()).unwrap();
        assert_eq!(
            guard.assess(&prepared).unwrap(),
            EvidenceLineageDecisionV1::Stable
        );
        assert_eq!(guard.commit_evidence(&prepared).unwrap(), 1);
        assert_eq!(guard.commit_evidence(&prepared).unwrap(), 2);
    }

    #[test]
    fn drift_before_evidence_requires_explicit_reprepare() {
        let prepared = lineage();
        let mut changed_inputs = inputs(false);
        changed_inputs.target_triple = "aarch64-unknown-linux-gnu".into();
        let changed = ExecutionLineageV1::seal(changed_inputs).unwrap();
        let mut guard = EvidenceLineageGuardV1::prepare(prepared).unwrap();
        assert!(matches!(
            guard.assess(&changed).unwrap(),
            EvidenceLineageDecisionV1::ReprepareBeforeEvidence(_)
        ));
        assert_eq!(
            guard.commit_evidence(&changed),
            Err(ExecutionLineageErrorV1::ReprepareRequiredBeforeEvidence)
        );
        guard.reprepare(changed.clone()).unwrap();
        assert_eq!(guard.assess(&changed).unwrap(), EvidenceLineageDecisionV1::Stable);
        assert_eq!(guard.commit_evidence(&changed).unwrap(), 1);
    }

    #[test]
    fn drift_after_evidence_refuses_mixed_lineage() {
        let prepared = lineage();
        let mut changed_inputs = inputs(false);
        changed_inputs.source_tree_digest = digest('f');
        let changed = ExecutionLineageV1::seal(changed_inputs).unwrap();
        let mut guard = EvidenceLineageGuardV1::prepare(prepared.clone()).unwrap();
        guard.commit_evidence(&prepared).unwrap();
        assert!(matches!(
            guard.assess(&changed).unwrap(),
            EvidenceLineageDecisionV1::RefuseMixedLineageAfterEvidence(_)
        ));
        assert_eq!(
            guard.commit_evidence(&changed),
            Err(ExecutionLineageErrorV1::MixedLineageAfterEvidence)
        );
        assert_eq!(
            guard.reprepare(changed),
            Err(ExecutionLineageErrorV1::CannotReprepareAfterEvidence)
        );
    }

    #[test]
    fn digest_tampering_is_detected() {
        let mut value = lineage();
        value.lineage_digest_hex = "0".repeat(64);
        assert_eq!(
            value.validate(),
            Err(ExecutionLineageErrorV1::LineageDigestMismatch)
        );
    }
}
