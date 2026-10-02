// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Symthaea Evidence Plane
//!
//! A shared "evidence contract" for research/ablation harnesses: a declared
//! mechanism (what an experimental arm claims it does), measured counters
//! (what actually happened, instrumented at the call site), and a
//! hard-failing integrity check that the two agree.
//!
//! This crate did not originate as a from-scratch design. It generalizes a
//! pattern that already existed twice, independently, in this codebase, with
//! two different shapes:
//!
//! 1. `symthaea/examples/hdc_ltc_coupling_ablation.rs`'s `TemporalStateMode`
//!    (declared arm) + `CallCounts` (measured integer call counts) +
//!    `assert_mechanical_integrity()` (a hard `panic!`-on-mismatch check).
//!    This machinery is real and has already caught real bugs in that
//!    research arc (a mislabeled `no_engine` ablation arm; a metric that
//!    measured reconstruction instead of prediction).
//! 2. `symthaea-psych-bench`'s Butlin AE-2 runner
//!    (`benchmarks::butlin::ae2_empirical_runner`), which independently
//!    hand-rolled the same *kind* of check over float hook-fired
//!    *fractions* (e.g. `> 0.9` / `< 0.1` thresholds) instead of integer
//!    counts, and returns a `Result`-shaped report rather than panicking.
//!
//! `EvidenceCounters` (backed by `f64`, not `u64`) and `Expectation`
//! (threshold-based, not per-mode-hardcoded) are deliberately general enough
//! to express both of the above without favoring either shape. `check_integrity`
//! returns a `Result` for callers that want to fold violations into a report
//! (Butlin's style); `enforce_integrity` / `RunEvidence::enforce` panic on a
//! violation for callers that want the original hard-abort behavior
//! (`hdc_ltc_coupling_ablation.rs`'s style).
//!
//! `examples/hdc_ltc_coupling_ablation.rs` itself is deliberately left
//! untouched by this extraction (Phase 1 scope decision — it is a live,
//! actively-iterated research artifact with in-flight uncommitted changes;
//! migrating it onto this crate is a follow-up, not part of this phase).
//!
//! `config_hash` is a second, smaller piece of shared infrastructure pulled
//! out of the same audit: at least 3 independently-invented, mutually
//! inconsistent `DefaultHasher`-over-`Debug`-string formulas were found
//! across the codebase. This crate provides exactly one.
//!
//! See `SYMTHAEA_COGNITIVE_CORE_RECONCILIATION_PLAN_2026-07-28.md` (Phase 1)
//! and `SYMTHAEA_COGNITIVE_ARCHITECTURE_AUDIT_ADDENDUM_2026-07-28.md` (§0.1)
//! at the monorepo root for the audit trail behind this crate.

pub mod seed_plan;
pub mod task_validator;

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Caller-labeled identity for one evidence-bearing run.
///
/// Deliberately NOT derived from `std::time::SystemTime::now()` or any
/// randomness inside library code — determinism for tests requires the
/// caller supply an explicit label.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RunId(pub String);

impl RunId {
    pub fn new(label: impl Into<String>) -> Self {
        Self(label.into())
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for RunId {
    fn from(label: &str) -> Self {
        Self::new(label)
    }
}

impl From<String> for RunId {
    fn from(label: String) -> Self {
        Self::new(label)
    }
}

/// Canonical, single-implementation config-identity fingerprint.
///
/// Hashes the `Debug` representation of `config` via `DefaultHasher`. This
/// is an identity fingerprint for logging/deduplication/dashboards — it is
/// **not** a cryptographic hash and must never be used for anything
/// security-sensitive (no collision resistance, no stability guarantee
/// across Rust versions).
///
/// Other code in this workspace should call this instead of reinventing the
/// same `DefaultHasher`-over-`Debug`-string pattern locally (see the crate
/// doc comment for the audit finding this closes).
pub fn config_hash<T: fmt::Debug>(config: &T) -> String {
    let mut hasher = DefaultHasher::new();
    format!("{config:?}").hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Canonical execution identity for evidence admission and replay.
///
/// This v1 identity contains only computational inputs that can change what
/// was executed: source snapshot, dependency locks, toolchains, host/target,
/// Nix identity, feature selection, working directory, exact argv, allow-listed
/// environment, and immutable input digests.
///
/// Run timestamps, generated run IDs, wall-clock outcomes, and other runtime
/// bookkeeping are intentionally outside this identity. They may belong in an
/// evidence envelope, but must not silently create a new computational lineage.
///
/// Canonical identity of a validated repository source snapshot.
///
/// The inner value is private so callers cannot bypass canonicalization. The
/// serde representation remains a plain string for wire compatibility, while
/// deserialization and construction normalize hexadecimal to lowercase.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RepositorySourceSnapshotId(String);

impl RepositorySourceSnapshotId {
    pub const SCHEMA: &'static str = "symthaea.repository-source-snapshot.v2";

    pub fn parse(value: &str) -> Result<Self, String> {
        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("invalid repository_source_snapshot_id: {value:?}"));
        }
        Ok(Self(value.to_ascii_lowercase()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Serialize for RepositorySourceSnapshotId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for RepositorySourceSnapshotId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "ExecutionLineageV1Wire")]
pub struct ExecutionLineageV1 {
    pub source_repository: String,
    pub source_revision: String,
    pub source_tree: String,
    /// Exact identity of the validated repository source subject used for this execution.
    ///
    /// This is distinct from Git HEAD/tree metadata: staged, unstaged, deleted,
    /// and explicitly included ignored source bytes can all belong to the exact source subject.
    pub repository_source_snapshot_id: RepositorySourceSnapshotId,
    #[serde(deserialize_with = "deserialize_unique_string_map")]
    pub lock_digests: BTreeMap<String, String>,
    #[serde(deserialize_with = "deserialize_unique_string_map")]
    pub toolchain_versions: BTreeMap<String, String>,
    pub host_target: String,
    pub nix_identity: String,
    #[serde(deserialize_with = "deserialize_unique_string_set")]
    pub feature_flags: BTreeSet<String>,
    pub cwd: String,
    pub argv: Vec<String>,
    #[serde(deserialize_with = "deserialize_unique_string_map")]
    pub allowed_env: BTreeMap<String, String>,
    #[serde(deserialize_with = "deserialize_unique_string_map")]
    pub immutable_input_digests: BTreeMap<String, String>,
}

/// Serde-only wire form whose conversion validates the complete lineage.
///
/// Keeping this separate from the public representation means JSON/binary
/// inputs cannot bypass the semantic admission check merely by deserializing
/// into the same field shape.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionLineageV1Wire {
    pub source_repository: String,
    pub source_revision: String,
    pub source_tree: String,
    pub repository_source_snapshot_id: RepositorySourceSnapshotId,
    #[serde(deserialize_with = "deserialize_unique_string_map")]
    pub lock_digests: BTreeMap<String, String>,
    #[serde(deserialize_with = "deserialize_unique_string_map")]
    pub toolchain_versions: BTreeMap<String, String>,
    pub host_target: String,
    pub nix_identity: String,
    #[serde(deserialize_with = "deserialize_unique_string_set")]
    pub feature_flags: BTreeSet<String>,
    pub cwd: String,
    pub argv: Vec<String>,
    #[serde(deserialize_with = "deserialize_unique_string_map")]
    pub allowed_env: BTreeMap<String, String>,
    #[serde(deserialize_with = "deserialize_unique_string_map")]
    pub immutable_input_digests: BTreeMap<String, String>,
}

impl TryFrom<ExecutionLineageV1Wire> for ExecutionLineageV1 {
    type Error = String;

    fn try_from(wire: ExecutionLineageV1Wire) -> Result<Self, Self::Error> {
        let lineage = Self {
            source_repository: wire.source_repository,
            source_revision: wire.source_revision,
            source_tree: wire.source_tree,
            repository_source_snapshot_id: wire.repository_source_snapshot_id,
            lock_digests: wire.lock_digests,
            toolchain_versions: wire.toolchain_versions,
            host_target: wire.host_target,
            nix_identity: wire.nix_identity,
            feature_flags: wire.feature_flags,
            cwd: wire.cwd,
            argv: wire.argv,
            allowed_env: wire.allowed_env,
            immutable_input_digests: wire.immutable_input_digests,
        };
        lineage.validate()?;
        Ok(lineage)
    }
}

impl Serialize for ExecutionLineageV1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate().map_err(serde::ser::Error::custom)?;

        ExecutionLineageV1Wire {
            source_repository: self.source_repository.clone(),
            source_revision: self.source_revision.clone(),
            source_tree: self.source_tree.clone(),
            repository_source_snapshot_id: self.repository_source_snapshot_id.clone(),
            lock_digests: self.lock_digests.clone(),
            toolchain_versions: self.toolchain_versions.clone(),
            host_target: self.host_target.clone(),
            nix_identity: self.nix_identity.clone(),
            feature_flags: self.feature_flags.clone(),
            cwd: self.cwd.clone(),
            argv: self.argv.clone(),
            allowed_env: self.allowed_env.clone(),
            immutable_input_digests: self.immutable_input_digests.clone(),
        }
        .serialize(serializer)
    }
}

impl ExecutionLineageV1 {
    pub const DOMAIN_SEPARATOR: &'static [u8] = b"symthaea:execution-lineage:v1\n";

    /// Construct a lineage from untrusted named-entry sequences without first
    /// collapsing them into maps/sets. Duplicate names are rejected before
    /// canonical collection construction, preventing silent overwrite or
    /// deduplication at the admission boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw_entries(
        source_repository: String,
        source_revision: String,
        source_tree: String,
        repository_source_snapshot_id: String,
        lock_digests: Vec<(String, String)>,
        toolchain_versions: Vec<(String, String)>,
        host_target: String,
        nix_identity: String,
        feature_flags: Vec<String>,
        cwd: String,
        argv: Vec<String>,
        allowed_env: Vec<(String, String)>,
        immutable_input_digests: Vec<(String, String)>,
    ) -> Result<Self, String> {
        let lock_digests = unique_named_map("lock_digests", lock_digests)?;
        let toolchain_versions = unique_named_map("toolchain_versions", toolchain_versions)?;
        let allowed_env = unique_named_map("allowed_env", allowed_env)?;
        let immutable_input_digests =
            unique_named_map("immutable_input_digests", immutable_input_digests)?;
        let feature_flags = unique_named_set("feature_flags", feature_flags)?;

        let lineage = Self {
            source_repository,
            source_revision,
            source_tree,
            repository_source_snapshot_id: RepositorySourceSnapshotId::parse(
                &repository_source_snapshot_id,
            )?,
            lock_digests,
            toolchain_versions,
            host_target,
            nix_identity,
            feature_flags,
            cwd,
            argv,
            allowed_env,
            immutable_input_digests,
        };
        lineage.validate()?;
        Ok(lineage)
    }

    /// Validate semantic identifiers before a lineage is admitted.
    ///
    /// Digest values may be bare hexadecimal or explicitly prefixed
    /// (for example, "sha256:..." or "blake3:..."). Named collections are
    /// represented by BTreeMaps, so duplicate names cannot survive the typed
    /// representation; untrusted list inputs should reject duplicates before
    /// constructing the map.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in self
            .lock_digests
            .iter()
            .chain(self.immutable_input_digests.iter())
        {
            validate_name(name)?;
            validate_digest(value)?;
        }
        for (name, value) in &self.toolchain_versions {
            validate_name(name)?;
            if value.trim().is_empty() {
                return Err(format!("empty toolchain version for {name}"));
            }
        }
        for (name, value) in &self.allowed_env {
            validate_name(name)?;
            if value.contains('\0') {
                return Err(format!("NUL in environment value for {name}"));
            }
        }
        for (name, value) in [
            ("source_repository", self.source_repository.as_str()),
            ("source_revision", self.source_revision.as_str()),
            ("source_tree", self.source_tree.as_str()),
            ("host_target", self.host_target.as_str()),
            ("nix_identity", self.nix_identity.as_str()),
            ("cwd", self.cwd.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(format!("empty lineage field {name}"));
            }
        }
        validate_git_object_id("source_revision", &self.source_revision)?;
        validate_git_object_id("source_tree", &self.source_tree)?;
        if self
            .repository_source_snapshot_id
            .as_str()
            != self
                .repository_source_snapshot_id
                .as_str()
                .to_ascii_lowercase()
        {
            return Err("repository_source_snapshot_id must be canonical lowercase hex".into());
        }
        if self.argv.iter().any(|arg| arg.contains('\0')) {
            return Err("NUL in argv".into());
        }
        Ok(())
    }

    pub fn digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(Self::DOMAIN_SEPARATOR);
        self.write_canonical(&mut hasher);
        hasher.finalize().to_hex().to_string()
    }

    fn write_canonical(&self, hasher: &mut blake3::Hasher) {
        append_str(hasher, "source_repository", &self.source_repository);
        append_str(hasher, "source_revision", &self.source_revision);
        append_str(hasher, "source_tree", &self.source_tree);
        append_str(
            hasher,
            "repository_source_snapshot_schema",
            RepositorySourceSnapshotId::SCHEMA,
        );
        append_str(
            hasher,
            "repository_source_snapshot_id",
            self.repository_source_snapshot_id.as_str(),
        );
        append_map(hasher, "lock_digests", &self.lock_digests);
        append_map(hasher, "toolchain_versions", &self.toolchain_versions);
        append_str(hasher, "host_target", &self.host_target);
        append_str(hasher, "nix_identity", &self.nix_identity);
        append_set(hasher, "feature_flags", &self.feature_flags);
        append_str(hasher, "cwd", &self.cwd);
        append_sequence(hasher, "argv", &self.argv);
        append_map(hasher, "allowed_env", &self.allowed_env);
        append_map(
            hasher,
            "immutable_input_digests",
            &self.immutable_input_digests,
        );
    }
}

fn deserialize_unique_string_map<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<String, String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct UniqueMapVisitor;

    impl<'de> Visitor<'de> for UniqueMapVisitor {
        type Value = BTreeMap<String, String>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a map with unique string keys")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut result = BTreeMap::new();
            while let Some((key, value)) = map.next_entry::<String, String>()? {
                if result.insert(key.clone(), value).is_some() {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate map key {key:?}"
                    )));
                }
            }
            Ok(result)
        }
    }

    deserializer.deserialize_map(UniqueMapVisitor)
}

fn deserialize_unique_string_set<'de, D>(deserializer: D) -> Result<BTreeSet<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct UniqueSetVisitor;

    impl<'de> Visitor<'de> for UniqueSetVisitor {
        type Value = BTreeSet<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a sequence with unique string members")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut result = BTreeSet::new();
            while let Some(value) = seq.next_element::<String>()? {
                if !result.insert(value.clone()) {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate set member {value:?}"
                    )));
                }
            }
            Ok(result)
        }
    }

    deserializer.deserialize_seq(UniqueSetVisitor)
}

fn unique_named_map(
    field: &str,
    entries: Vec<(String, String)>,
) -> Result<BTreeMap<String, String>, String> {
    let mut map = BTreeMap::new();
    for (name, value) in entries {
        validate_name(&name)?;
        if map.insert(name.clone(), value).is_some() {
            return Err(format!("duplicate name {name:?} in {field}"));
        }
    }
    Ok(map)
}

fn unique_named_set(field: &str, entries: Vec<String>) -> Result<BTreeSet<String>, String> {
    let mut set = BTreeSet::new();
    for name in entries {
        validate_name(&name)?;
        if !set.insert(name.clone()) {
            return Err(format!("duplicate name {name:?} in {field}"));
        }
    }
    Ok(set)
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.trim().is_empty() || name.chars().any(|c| c == '\0' || c.is_control()) {
        Err(format!("invalid empty/control identifier: {name:?}"))
    } else {
        Ok(())
    }
}

fn validate_git_object_id(field: &str, value: &str) -> Result<(), String> {
    if !matches!(value.len(), 40 | 64)
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(format!(
            "invalid or non-canonical Git object identity for {field}: {value:?}"
        ));
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), String> {
    let payload = value.split_once(':').map_or(value, |(_, payload)| payload);
    if payload.len() < 16
        || payload.len() % 2 != 0
        || !payload
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(format!(
            "invalid or non-canonical digest syntax: {value:?}"
        ));
    }
    Ok(())
}

fn append_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn append_str(hasher: &mut blake3::Hasher, field: &str, value: &str) {
    append_bytes(hasher, field.as_bytes());
    append_bytes(hasher, value.as_bytes());
}

fn append_sequence(hasher: &mut blake3::Hasher, field: &str, values: &[String]) {
    append_bytes(hasher, field.as_bytes());
    hasher.update(&(values.len() as u64).to_be_bytes());
    for value in values {
        append_bytes(hasher, value.as_bytes());
    }
}

fn append_set(hasher: &mut blake3::Hasher, field: &str, values: &BTreeSet<String>) {
    append_bytes(hasher, field.as_bytes());
    hasher.update(&(values.len() as u64).to_be_bytes());
    for value in values {
        append_bytes(hasher, value.as_bytes());
    }
}

fn append_map(hasher: &mut blake3::Hasher, field: &str, values: &BTreeMap<String, String>) {
    append_bytes(hasher, field.as_bytes());
    hasher.update(&(values.len() as u64).to_be_bytes());
    for (key, value) in values {
        append_bytes(hasher, key.as_bytes());
        append_bytes(hasher, value.as_bytes());
    }
}

/// Admission decision when an execution lineage is compared with a prepared lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceLineageDecision {
    Stable,
    ReprepareBeforeEvidence,
    RefuseMixedLineageAfterEvidence,
}

/// Failure returned when claim-bearing evidence cannot be committed to the
/// prepared lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceLineageCommitError {
    InvalidCurrentLineage(String),
    LineageDecision(EvidenceLineageDecision),
}

impl fmt::Display for EvidenceLineageCommitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCurrentLineage(error) => {
                write!(
                    f,
                    "cannot commit evidence for invalid current lineage: {error}"
                )
            }
            Self::LineageDecision(decision) => {
                write!(
                    f,
                    "cannot commit evidence under lineage decision: {decision:?}"
                )
            }
        }
    }
}

impl std::error::Error for EvidenceLineageCommitError {}

/// Explicit guard against silently mixing evidence from different executions.
///
/// The guard never adopts a drifted lineage automatically. Callers must
/// deliberately prepare a new lineage before committing any new claim-bearing
/// evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceLineageGuardV1 {
    prepared_digest: String,
    evidence_committed: bool,
}

impl EvidenceLineageGuardV1 {
    /// Prepare an evidence guard only after validating the supplied lineage.
    ///
    /// Preparation is intentionally fallible so invalid lineage cannot reach
    /// the guard through a panic-based admission path.
    pub fn prepare(lineage: &ExecutionLineageV1) -> Result<Self, String> {
        lineage.validate()?;
        Ok(Self {
            prepared_digest: lineage.digest(),
            evidence_committed: false,
        })
    }

    /// Compatibility alias for callers that explicitly prefer a fallible
    /// constructor name.
    pub fn try_prepare(lineage: &ExecutionLineageV1) -> Result<Self, String> {
        Self::prepare(lineage)
    }

    /// Commit claim-bearing evidence only when the current lineage is valid
    /// and exactly matches the prepared lineage.
    ///
    /// The guard owns the phase transition so callers cannot accidentally
    /// mark evidence committed after a drifted lineage has already arrived.
    pub fn commit_evidence(
        &mut self,
        current: &ExecutionLineageV1,
    ) -> Result<(), EvidenceLineageCommitError> {
        current
            .validate()
            .map_err(EvidenceLineageCommitError::InvalidCurrentLineage)?;

        match self.check(current) {
            EvidenceLineageDecision::Stable => {
                self.evidence_committed = true;
                Ok(())
            }
            decision => Err(EvidenceLineageCommitError::LineageDecision(decision)),
        }
    }

    pub fn prepared_digest(&self) -> &str {
        &self.prepared_digest
    }

    pub fn check(&self, current: &ExecutionLineageV1) -> EvidenceLineageDecision {
        if self.prepared_digest == current.digest() {
            EvidenceLineageDecision::Stable
        } else if self.evidence_committed {
            EvidenceLineageDecision::RefuseMixedLineageAfterEvidence
        } else {
            EvidenceLineageDecision::ReprepareBeforeEvidence
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineagePerturbationResult {
    /// The declared dependency did not change and the lineage stayed stable.
    InvariantPreserved,
    /// The declared dependency changed and the lineage changed.
    ExpectedDependencyChanged,
    /// An undeclared dependency changed the lineage.
    UnexpectedCollateralChange,
    /// A declared dependency changed but the lineage stayed stable.
    UnexpectedInvariance,
}

pub fn qualify_lineage_perturbation(
    before: &ExecutionLineageV1,
    after: &ExecutionLineageV1,
    dependency_changed: bool,
) -> LineagePerturbationResult {
    let changed = before.digest() != after.digest();
    match (dependency_changed, changed) {
        (false, false) => LineagePerturbationResult::InvariantPreserved,
        (true, true) => LineagePerturbationResult::ExpectedDependencyChanged,
        (false, true) => LineagePerturbationResult::UnexpectedCollateralChange,
        (true, false) => LineagePerturbationResult::UnexpectedInvariance,
    }
}

/// A named bag of measured evidence values.
///
/// Backed by `f64` (not `u64`) so it can hold both integer call-counts
/// (cast up, e.g. `CallCounts::hdc_ltc_predict`) and float fractions or
/// thresholds (e.g. Butlin's `> 0.9` / `< 0.1` hook-fired-fraction checks)
/// under one type.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EvidenceCounters(HashMap<String, f64>);

impl EvidenceCounters {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set (overwrite) a named value.
    pub fn record(&mut self, name: impl Into<String>, value: f64) {
        self.0.insert(name.into(), value);
    }

    /// Increment a named value by `delta`, starting from `0.0` if absent.
    pub fn add(&mut self, name: impl Into<String>, delta: f64) {
        *self.0.entry(name.into()).or_insert(0.0) += delta;
    }

    /// Read a named value. Missing keys read as `0.0` — a counter that was
    /// never touched during a run is honestly zero, not an error.
    pub fn get(&self, name: &str) -> f64 {
        *self.0.get(name).unwrap_or(&0.0)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &f64)> {
        self.0.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// One requirement a declared mechanism places on a measured counter.
///
/// Generalizes `TemporalStateMode`'s per-arm requirements (each non-active
/// mechanism forbids every counter belonging to the OTHER mechanisms via
/// `MustBeZero`, the active one requires `MustBePositive`) and Butlin's
/// float-fraction thresholds (`MustExceed(0.9)` / `MustBeBelow(0.1)`) under
/// one type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Expectation {
    /// The measured value must be exactly `0.0` (a mechanism that should
    /// never have fired at all).
    MustBeZero,
    /// The measured value must be strictly greater than `0.0` (a mechanism
    /// that must have fired at least once).
    MustBePositive,
    /// The measured value must be strictly greater than the given
    /// threshold (e.g. a "fired in >90% of sampled cycles" fraction check).
    MustExceed(f64),
    /// The measured value must be strictly less than the given threshold
    /// (e.g. a "fired in <10% of sampled cycles" fraction check).
    MustBeBelow(f64),
}

impl Expectation {
    pub fn is_satisfied_by(&self, measured: f64) -> bool {
        match self {
            Expectation::MustBeZero => measured == 0.0,
            Expectation::MustBePositive => measured > 0.0,
            Expectation::MustExceed(threshold) => measured > *threshold,
            Expectation::MustBeBelow(threshold) => measured < *threshold,
        }
    }
}

impl fmt::Display for Expectation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expectation::MustBeZero => write!(f, "must be zero"),
            Expectation::MustBePositive => write!(f, "must be positive (> 0)"),
            Expectation::MustExceed(t) => write!(f, "must exceed {t}"),
            Expectation::MustBeBelow(t) => write!(f, "must be below {t}"),
        }
    }
}

/// One declared expectation that a measured value failed to satisfy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailedExpectation {
    pub name: String,
    pub expectation: Expectation,
    pub measured: f64,
}

impl fmt::Display for FailedExpectation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} (measured {})",
            self.name, self.expectation, self.measured
        )
    }
}

/// One or more declared expectations that the measured evidence violated.
///
/// Implements `std::error::Error` so it composes with `Result`-based
/// callers (Butlin's style); `enforce_integrity` / `RunEvidence::enforce`
/// turn this into a hard `panic!` for callers that want the original
/// abort-on-mismatch behavior (`hdc_ltc_coupling_ablation.rs`'s style).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IntegrityViolation {
    pub failures: Vec<FailedExpectation>,
}

impl fmt::Display for IntegrityViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "evidence-plane integrity check failed ({} violation(s)):",
            self.failures.len()
        )?;
        for failure in &self.failures {
            writeln!(f, "  - {failure}")?;
        }
        Ok(())
    }
}

impl std::error::Error for IntegrityViolation {}

/// Check declared expectations against measured evidence.
///
/// Returns `Ok(())` iff every named expectation in `declared` is satisfied
/// by the corresponding value in `measured` (a name with no declared
/// expectation is simply not checked). A name declared but never recorded
/// in `measured` reads as `0.0` (see `EvidenceCounters::get`).
pub fn check_integrity(
    declared: &HashMap<String, Expectation>,
    measured: &EvidenceCounters,
) -> Result<(), IntegrityViolation> {
    let mut failures: Vec<FailedExpectation> = declared
        .iter()
        .filter_map(|(name, expectation)| {
            let value = measured.get(name);
            if expectation.is_satisfied_by(value) {
                None
            } else {
                Some(FailedExpectation {
                    name: name.clone(),
                    expectation: *expectation,
                    measured: value,
                })
            }
        })
        .collect();
    // Deterministic ordering for reproducible error messages/tests --
    // `declared`'s HashMap iteration order is not stable.
    failures.sort_by(|a, b| a.name.cmp(&b.name));

    if failures.is_empty() {
        Ok(())
    } else {
        Err(IntegrityViolation { failures })
    }
}

/// Same check as [`check_integrity`], but panics with the violation's
/// `Display` output on failure — the hard-abort behavior
/// `hdc_ltc_coupling_ablation.rs`'s `assert_mechanical_integrity` originally
/// provided directly, generalized to work for any declared/measured pair.
pub fn enforce_integrity(declared: &HashMap<String, Expectation>, measured: &EvidenceCounters) {
    if let Err(violation) = check_integrity(declared, measured) {
        panic!("{violation}");
    }
}

/// A complete, exportable record of one evidence-bearing run: its identity,
/// its config fingerprint, what it declared, what it measured, and whether
/// the two agreed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEvidence {
    pub run_id: RunId,
    pub config_hash: String,
    /// Declared expectations, human-readable (`BTreeMap` for deterministic
    /// ordering in exported JSON/logs).
    pub declared: BTreeMap<String, Expectation>,
    pub measured: EvidenceCounters,
    pub satisfied: bool,
    pub violations: Vec<FailedExpectation>,
}

impl RunEvidence {
    /// Build a `RunEvidence` record, computing `config_hash` via
    /// [`config_hash`] and immediately checking `declared` against
    /// `measured` (populating `satisfied`/`violations`, never panicking).
    pub fn new<T: fmt::Debug>(
        run_id: RunId,
        config: &T,
        declared: BTreeMap<String, Expectation>,
        measured: EvidenceCounters,
    ) -> Self {
        let declared_map: HashMap<String, Expectation> =
            declared.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let (satisfied, violations) = match check_integrity(&declared_map, &measured) {
            Ok(()) => (true, Vec::new()),
            Err(violation) => (false, violation.failures),
        };
        Self {
            run_id,
            config_hash: config_hash(config),
            declared,
            measured,
            satisfied,
            violations,
        }
    }

    /// Panic with the recorded violations' `Display` output if this run's
    /// integrity check failed. A no-op if `satisfied` is `true`.
    pub fn enforce(&self) {
        if !self.satisfied {
            panic!(
                "{}",
                IntegrityViolation {
                    failures: self.violations.clone()
                }
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reproduces `TemporalStateMode::HdcLtc`'s requirement: the active
    /// mechanism's predict counter must be positive. A passing case.

    #[test]
    fn raw_named_entries_reject_duplicate_map_keys_before_canonicalization() {
        let result = ExecutionLineageV1::from_raw_entries(
            "repo".into(),
            "revision".into(),
            "tree".into(),
            "a".repeat(64),
            vec![
                ("cargo".into(), "sha256:0011223344556677".into()),
                ("cargo".into(), "sha256:8899aabbccddeeff".into()),
            ],
            vec![("rustc".into(), "1.96".into())],
            "host/target".into(),
            "nix".into(),
            vec!["feature".into()],
            "/work".into(),
            vec!["cargo".into(), "test".into()],
            vec![("RUSTFLAGS".into(), "-Dwarnings".into())],
            vec![("input".into(), "blake3:0011223344556677".into())],
        )
        .expect_err("duplicate lock name must fail");
        assert!(result.contains("duplicate name"));
    }

    #[test]
    fn raw_named_entries_reject_duplicate_set_members() {
        let result = ExecutionLineageV1::from_raw_entries(
            "repo".into(),
            "revision".into(),
            "tree".into(),
            "a".repeat(64),
            vec![("cargo".into(), "sha256:0011223344556677".into())],
            vec![("rustc".into(), "1.96".into())],
            "host/target".into(), "nix".into(), vec!["feature".into(), "feature".into()],
            "/work".into(), vec!["cargo".into(), "test".into()],
            vec![("RUSTFLAGS".into(), "-Dwarnings".into())],
            vec![("input".into(), "blake3:0011223344556677".into())],
        )
        .expect_err("duplicate feature must fail");
        assert!(result.contains("duplicate name"));
    }

    #[test]
    fn raw_named_entries_validate_before_digest_is_available() {
        let lineage = ExecutionLineageV1::from_raw_entries(
            "repo".into(),
            "revision".into(),
            "tree".into(),
            "a".repeat(64),
            vec![("cargo".into(), "sha256:0011223344556677".into())],
            vec![("rustc".into(), "1.96".into())],
            "host/target".into(),
            "nix".into(),
            vec!["feature".into()],
            "/work".into(),
            vec!["cargo".into(), "test".into()],
            vec![("RUSTFLAGS".into(), "-Dwarnings".into())],
            vec![("input".into(), "blake3:0011223344556677".into())],
        )
        .expect("valid raw entries");
        assert_eq!(
            lineage.lock_digests.get("cargo"),
            Some(&"sha256:0011223344556677".to_owned())
        );
    }

    fn lineage_fixture() -> ExecutionLineageV1 {
        ExecutionLineageV1 {
            source_repository: "github.com/Luminous-Dynamics/symthaea".into(),
            source_revision: "a".repeat(40),
            source_tree: "b".repeat(40),
            repository_source_snapshot_id: RepositorySourceSnapshotId::parse(&"A".repeat(64))
                .expect("valid snapshot id"),
            lock_digests: [(
                "Cargo.lock".into(),
                "sha256:00112233445566778899aabbccddeeff".into(),
            )]
            .into_iter()
            .collect(),
            toolchain_versions: [("rustc".into(), "1.96.0".into())].into_iter().collect(),
            host_target: "x86_64-unknown-linux-gnu".into(),
            nix_identity: "nixpkgs:deadbeef".into(),
            feature_flags: ["default".into()].into_iter().collect(),
            cwd: "/workspace/symthaea".into(),
            argv: vec![
                "cargo".into(),
                "test".into(),
                "-p".into(),
                "symthaea-evidence-plane".into(),
            ],
            allowed_env: [("RUST_BACKTRACE".into(), "0".into())]
                .into_iter()
                .collect(),
            immutable_input_digests: [(
                "fixture.json".into(),
                "blake3:00112233445566778899aabbccddeeff".into(),
            )]
            .into_iter()
            .collect(),
        }
    }

    #[test]
    fn execution_lineage_validation_rejects_invalid_git_object_identity() {
        let mut lineage = lineage_fixture();
        lineage.source_revision = "not-a-git-object".into();
        assert!(lineage.validate().is_err());

        lineage = lineage_fixture();
        lineage.source_tree = "1234".into();
        assert!(lineage.validate().is_err());
    }

    #[test]
    fn execution_lineage_validation_accepts_canonical_sha1_and_sha256_git_object_ids() {
        let mut lineage = lineage_fixture();
        lineage.source_revision = "c".repeat(40);
        lineage.source_tree = "d".repeat(64);
        assert!(lineage.validate().is_ok());
    }

    #[test]
    fn execution_lineage_validation_rejects_uppercase_git_object_ids() {
        let mut lineage = lineage_fixture();
        lineage.source_revision = "A".repeat(40);
        assert!(lineage.validate().is_err());

        lineage.source_revision = "c".repeat(40);
        lineage.source_tree = "D".repeat(64);
        assert!(lineage.validate().is_err());
    }

    #[test]
    fn execution_lineage_fixture_is_admissible() {
        assert!(lineage_fixture().validate().is_ok());
    }

    #[test]
    fn execution_lineage_serde_rejects_invalid_direct_value_on_serialize() {
        let mut lineage = lineage_fixture();
        lineage.source_repository.clear();

        let error = serde_json::to_value(&lineage)
            .expect_err("invalid direct lineage must not serialize");
        assert!(error.to_string().contains("empty lineage field source_repository"));
    }

    #[test]
    fn execution_lineage_serde_rejects_unknown_fields() {
        let mut lineage = serde_json::to_value(lineage_fixture()).expect("serialize fixture");
        lineage["unexpected_future_field"] = serde_json::Value::String("ignored".into());

        let error = serde_json::from_value::<ExecutionLineageV1>(lineage)
            .expect_err("unknown lineage fields must fail closed");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn execution_lineage_serde_rejects_semantically_invalid_lineage() {
        let mut lineage = serde_json::to_value(lineage_fixture()).expect("serialize fixture");
        lineage["source_repository"] = serde_json::Value::String("".into());

        let error = serde_json::from_value::<ExecutionLineageV1>(lineage)
            .expect_err("invalid lineage must be rejected during deserialization");
        assert!(
            error
                .to_string()
                .contains("empty lineage field source_repository")
        );
    }

    #[test]
    fn execution_lineage_serde_rejects_duplicate_map_keys() {
        let json = format!(
            r#"{{"source_repository":"repo","source_revision":"rev","source_tree":"tree","repository_source_snapshot_id":"{}","lock_digests":{{"Cargo.lock":"sha256:0011223344556677","Cargo.lock":"sha256:8899aabbccddeeff"}},"toolchain_versions":{{"rustc":"1.96.0"}},"host_target":"target","nix_identity":"nix","feature_flags":["default"],"cwd":"/work","argv":["cargo","test"],"allowed_env":{{"RUST_BACKTRACE":"0"}},"immutable_input_digests":{{"fixture":"sha256:0011223344556677"}}}}"#,
            "a".repeat(64)
        );
        let error = serde_json::from_str::<ExecutionLineageV1>(&json)
            .expect_err("duplicate JSON map keys must fail closed");
        assert!(error.to_string().contains("duplicate map key"));
    }

    #[test]
    fn execution_lineage_serde_rejects_duplicate_set_members() {
        let json = format!(
            r#"{{"source_repository":"repo","source_revision":"rev","source_tree":"tree","repository_source_snapshot_id":"{}","lock_digests":{{"Cargo.lock":"sha256:0011223344556677"}},"toolchain_versions":{{"rustc":"1.96.0"}},"host_target":"target","nix_identity":"nix","feature_flags":["default","default"],"cwd":"/work","argv":["cargo","test"],"allowed_env":{{"RUST_BACKTRACE":"0"}},"immutable_input_digests":{{"fixture":"sha256:0011223344556677"}}}}"#,
            "a".repeat(64)
        );
        let error = serde_json::from_str::<ExecutionLineageV1>(&json)
            .expect_err("duplicate feature members must fail closed");
        assert!(error.to_string().contains("duplicate set member"));
    }

    #[test]
    fn repository_source_snapshot_id_is_canonicalized_to_lowercase() {
        let id = RepositorySourceSnapshotId::parse(&"AB".repeat(32)).expect("valid snapshot id");
        assert_eq!(id.as_str(), &"ab".repeat(32));
        let json = serde_json::to_string(&id).expect("serialize snapshot id");
        assert_eq!(json, format!("\"{}\"", "ab".repeat(32)));
    }

    #[test]
    fn repository_source_snapshot_id_serde_canonicalizes_case() {
        let upper = format!("\"{}\"", "CD".repeat(32));
        let id: RepositorySourceSnapshotId =
            serde_json::from_str(&upper).expect("deserialize snapshot id");
        assert_eq!(id.as_str(), &"cd".repeat(32));
    }

    #[test]
    fn lineage_digest_is_case_canonical_for_snapshot_id() {
        let mut upper = lineage_fixture();
        let lower = lineage_fixture();
        upper.repository_source_snapshot_id =
            RepositorySourceSnapshotId::parse(&"A".repeat(64)).expect("valid snapshot id");
        assert_eq!(upper.digest(), lower.digest());
    }

    #[test]
    fn execution_lineage_serde_roundtrip_preserves_canonical_snapshot_id() {
        let lineage = lineage_fixture();
        let json = serde_json::to_string(&lineage).expect("serialize lineage");
        let restored: ExecutionLineageV1 =
            serde_json::from_str(&json).expect("deserialize lineage");
        assert_eq!(
            restored.repository_source_snapshot_id.as_str(),
            "a".repeat(64)
        );
        assert_eq!(restored.digest(), lineage.digest());
    }

    #[test]
    fn execution_lineage_validation_rejects_invalid_repository_source_snapshot_id() {
        assert!(RepositorySourceSnapshotId::parse("not-a-sha256").is_err());
    }

    #[test]
    fn lineage_digest_binds_repository_source_snapshot_schema() {
        let lineage = lineage_fixture();
        let current = lineage.digest();

        // Reproduce the pre-schema-binding canonical encoding. The new digest
        // must differ, proving the schema discriminator is actually committed.
        let mut legacy = blake3::Hasher::new();
        legacy.update(ExecutionLineageV1::DOMAIN_SEPARATOR);
        append_str(&mut legacy, "source_repository", &lineage.source_repository);
        append_str(&mut legacy, "source_revision", &lineage.source_revision);
        append_str(&mut legacy, "source_tree", &lineage.source_tree);
        append_str(
            &mut legacy,
            "repository_source_snapshot_id",
            lineage.repository_source_snapshot_id.as_str(),
        );
        append_map(&mut legacy, "lock_digests", &lineage.lock_digests);
        append_map(
            &mut legacy,
            "toolchain_versions",
            &lineage.toolchain_versions,
        );
        append_str(&mut legacy, "host_target", &lineage.host_target);
        append_str(&mut legacy, "nix_identity", &lineage.nix_identity);
        append_set(&mut legacy, "feature_flags", &lineage.feature_flags);
        append_str(&mut legacy, "cwd", &lineage.cwd);
        append_sequence(&mut legacy, "argv", &lineage.argv);
        append_map(&mut legacy, "allowed_env", &lineage.allowed_env);
        append_map(
            &mut legacy,
            "immutable_input_digests",
            &lineage.immutable_input_digests,
        );

        let legacy = legacy.finalize().to_hex().to_string();
        assert_ne!(current, legacy);
        assert_eq!(
            RepositorySourceSnapshotId::SCHEMA,
            "symthaea.repository-source-snapshot.v2"
        );
    }

    #[test]
    fn repository_source_snapshot_identity_changes_lineage_digest() {
        let base = lineage_fixture();
        let mut changed = base.clone();
        changed.repository_source_snapshot_id =
            RepositorySourceSnapshotId::parse(&"b".repeat(64)).expect("valid snapshot id");
        assert_ne!(base.digest(), changed.digest());
    }

    #[test]
    fn execution_lineage_guard_rejects_invalid_preparation() {
        let mut lineage = lineage_fixture();
        lineage
            .immutable_input_digests
            .insert("fixture.json".into(), "not-a-digest".into());
        assert!(EvidenceLineageGuardV1::try_prepare(&lineage).is_err());
    }

    #[test]
    fn execution_lineage_guard_commits_only_stable_valid_lineage() {
        let base = lineage_fixture();
        let mut guard = EvidenceLineageGuardV1::prepare(&base).expect("valid lineage preparation");

        guard
            .commit_evidence(&base)
            .expect("stable valid lineage should commit");

        assert_eq!(guard.check(&base), EvidenceLineageDecision::Stable);
    }

    #[test]
    fn execution_lineage_guard_does_not_commit_after_pre_evidence_drift() {
        let base = lineage_fixture();
        let mut changed = base.clone();
        changed.source_revision = "c".repeat(40);

        let mut guard = EvidenceLineageGuardV1::prepare(&base).expect("valid lineage preparation");
        let error = guard
            .commit_evidence(&changed)
            .expect_err("drifted lineage must not commit");

        assert_eq!(
            error,
            EvidenceLineageCommitError::LineageDecision(
                EvidenceLineageDecision::ReprepareBeforeEvidence
            )
        );
        assert_eq!(guard.check(&base), EvidenceLineageDecision::Stable);
    }

    #[test]
    fn execution_lineage_guard_rejects_invalid_current_lineage_before_commit() {
        let base = lineage_fixture();
        let mut invalid = base.clone();
        invalid
            .immutable_input_digests
            .insert("fixture.json".into(), "not-a-digest".into());

        let mut guard = EvidenceLineageGuardV1::prepare(&base).expect("valid lineage preparation");
        let error = guard
            .commit_evidence(&invalid)
            .expect_err("invalid current lineage must be rejected");

        assert!(matches!(
            error,
            EvidenceLineageCommitError::InvalidCurrentLineage(_)
        ));
        assert!(!guard.evidence_committed);
    }

    #[test]
    fn execution_lineage_guard_rejects_invalid_preparation() {
        let mut invalid = lineage_fixture();
        invalid
            .immutable_input_digests
            .insert("fixture.json".into(), "not-a-digest".into());

        assert!(EvidenceLineageGuardV1::prepare(&invalid).is_err());
    }

    #[test]
    fn execution_lineage_guard_requires_reprepare_before_evidence() {
        let base = lineage_fixture();
        let mut changed = base.clone();
        changed.source_revision = "def456".into();

        let guard = EvidenceLineageGuardV1::prepare(&base).expect("valid lineage preparation");
        assert_eq!(
            guard.check(&changed),
            EvidenceLineageDecision::ReprepareBeforeEvidence
        );
    }

    #[test]
    fn execution_lineage_guard_refuses_mixing_after_evidence() {
        let base = lineage_fixture();
        let mut changed = base.clone();
        changed.source_revision = "def456".into();

        let mut guard = EvidenceLineageGuardV1::prepare(&base).expect("valid lineage preparation");
        guard
            .commit_evidence(&base)
            .expect("stable lineage commits evidence");
        assert_eq!(
            guard.check(&changed),
            EvidenceLineageDecision::RefuseMixedLineageAfterEvidence
        );
    }

    #[test]
    fn execution_lineage_guard_allows_same_lineage_after_evidence() {
        let base = lineage_fixture();
        let mut guard =
            EvidenceLineageGuardV1::prepare(&base).expect("valid lineage preparation");
        guard
            .commit_evidence(&base)
            .expect("stable lineage commits evidence");
        assert_eq!(guard.check(&base), EvidenceLineageDecision::Stable);
    }

    #[test]
    fn execution_lineage_digest_is_insertion_order_independent() {
        let mut a = lineage_fixture();
        let mut b = lineage_fixture();
        a.lock_digests.insert("z".into(), "2".into());
        a.lock_digests.insert("a".into(), "1".into());
        b.lock_digests.insert("a".into(), "1".into());
        b.lock_digests.insert("z".into(), "2".into());
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn execution_lineage_validation_rejects_invalid_digest() {
        let mut lineage = lineage_fixture();
        lineage.immutable_input_digests.insert("fixture.json".into(), "not-a-digest".into());
        assert!(lineage.validate().is_err());
    }

    #[test]
    fn execution_lineage_validation_rejects_uppercase_digest_payload() {
        let mut lineage = lineage_fixture();
        lineage
            .immutable_input_digests
            .insert("uppercase".into(), "sha256:0011223344556677Aa".into());
        assert!(lineage.validate().is_err());

        lineage
            .immutable_input_digests
            .insert("uppercase".into(), "sha256:0011223344556677aa".into());
        assert!(lineage.validate().is_ok());
    }

    #[test]
    fn execution_lineage_validation_accepts_prefixed_digest() {
        let lineage = lineage_fixture();
        assert!(lineage.validate().is_ok());
        let mut prefixed = lineage.clone();
        prefixed.immutable_input_digests.insert(
            "other.bin".into(),
            "sha256:0123456789abcdef0123456789abcdef".into(),
        );
        assert!(prefixed.validate().is_ok());
    }

    #[test]
    fn execution_lineage_relevant_changes_change_identity() {
        let base = lineage_fixture();
        let mut revision = base.clone();
        revision.source_revision = "c".repeat(40);
        let mut lock = base.clone();
        lock.lock_digests
            .insert("Cargo.lock".into(), "lock999".into());
        let mut argv = base.clone();
        argv.argv.push("--nocapture".into());

        assert_ne!(base.digest(), revision.digest());
        assert_ne!(base.digest(), lock.digest());
        assert_ne!(base.digest(), argv.digest());
    }

    #[test]
    fn execution_lineage_qualification_distinguishes_all_perturbation_cases() {
        let base = lineage_fixture();
        let mut changed = base.clone();
        changed.source_revision = "def456".into();

        assert_eq!(
            qualify_lineage_perturbation(&base, &base, false),
            LineagePerturbationResult::InvariantPreserved
        );
        assert_eq!(
            qualify_lineage_perturbation(&base, &changed, true),
            LineagePerturbationResult::ExpectedDependencyChanged
        );
        assert_eq!(
            qualify_lineage_perturbation(&base, &changed, false),
            LineagePerturbationResult::UnexpectedCollateralChange
        );
        assert_eq!(
            qualify_lineage_perturbation(&base, &base, true),
            LineagePerturbationResult::UnexpectedInvariance
        );
    }

    #[test]
    fn execution_lineage_serde_round_trip_preserves_identity() {
        let lineage = lineage_fixture();
        let json = serde_json::to_string(&lineage).expect("serialize lineage");
        let restored: ExecutionLineageV1 =
            serde_json::from_str(&json).expect("deserialize lineage");
        assert_eq!(restored, lineage);
        assert_eq!(restored.digest(), lineage.digest());
    }

    #[test]
    fn hdc_ltc_style_positive_case_passes() {
        let mut declared = HashMap::new();
        declared.insert("hdc_ltc_predict".to_string(), Expectation::MustBePositive);
        declared.insert("static_updates".to_string(), Expectation::MustBeZero);

        let mut measured = EvidenceCounters::new();
        measured.add("hdc_ltc_predict", 1.0);
        measured.add("hdc_ltc_predict", 1.0);
        // static_updates never recorded -> reads as 0.0, satisfies MustBeZero.

        assert!(check_integrity(&declared, &measured).is_ok());
        assert_eq!(measured.get("hdc_ltc_predict"), 2.0);
    }

    /// Reproduces Butlin's `intervention_applied`-style float-fraction
    /// threshold checks (`> 0.9` / `< 0.1`) as a passing case.
    #[test]
    fn butlin_style_fraction_thresholds_pass() {
        let mut declared = HashMap::new();
        declared.insert(
            "baseline_hook_fired_fraction".to_string(),
            Expectation::MustExceed(0.9),
        );
        declared.insert(
            "target_hook_fired_fraction".to_string(),
            Expectation::MustBeBelow(0.1),
        );

        let mut measured = EvidenceCounters::new();
        measured.record("baseline_hook_fired_fraction", 1.0);
        measured.record("target_hook_fired_fraction", 0.0);

        assert!(check_integrity(&declared, &measured).is_ok());
    }

    /// The literal proof the "must ship with a hard-failing assertion"
    /// requirement is met, not just claimed: `.enforce()` on a genuinely
    /// unsatisfied run panics.
    #[test]
    #[should_panic(expected = "hdc_ltc_predict")]
    fn enforce_panics_on_violation() {
        let mut declared = BTreeMap::new();
        declared.insert("hdc_ltc_predict".to_string(), Expectation::MustBePositive);

        let measured = EvidenceCounters::new(); // hdc_ltc_predict never recorded -> 0.0

        let evidence = RunEvidence::new(RunId::new("test-run"), &"cfg", declared, measured);
        evidence.enforce();
    }

    /// The same failing case via the non-panicking `Result` form: no panic,
    /// and the returned violation names the exact failed expectation.
    #[test]
    fn check_integrity_returns_err_naming_the_failed_expectation() {
        let mut declared = HashMap::new();
        declared.insert("hdc_ltc_predict".to_string(), Expectation::MustBePositive);

        let measured = EvidenceCounters::new();

        let result = check_integrity(&declared, &measured);
        let violation = result.expect_err("expected a violation, got Ok");
        assert_eq!(violation.failures.len(), 1);
        assert_eq!(violation.failures[0].name, "hdc_ltc_predict");
        assert_eq!(
            violation.failures[0].expectation,
            Expectation::MustBePositive
        );
        assert_eq!(violation.failures[0].measured, 0.0);
    }

    /// `enforce_integrity` free function: same panic-on-violation contract
    /// as `RunEvidence::enforce`, exercised directly.
    #[test]
    #[should_panic(expected = "evidence-plane integrity check failed")]
    fn enforce_integrity_free_function_panics() {
        let mut declared = HashMap::new();
        declared.insert("permutation_ops".to_string(), Expectation::MustBeZero);

        let mut measured = EvidenceCounters::new();
        measured.record("permutation_ops", 3.0);

        enforce_integrity(&declared, &measured);
    }

    #[test]
    fn config_hash_is_deterministic() {
        #[derive(Debug)]
        struct Config {
            alpha: f32,
            mode: &'static str,
        }
        let a = Config {
            alpha: 0.3,
            mode: "Ema",
        };
        let b = Config {
            alpha: 0.3,
            mode: "Ema",
        };
        assert_eq!(config_hash(&a), config_hash(&b));
    }

    #[test]
    fn config_hash_is_sensitive_to_input_changes() {
        #[derive(Debug)]
        struct Config {
            alpha: f32,
        }
        let a = Config { alpha: 0.3 };
        let b = Config { alpha: 0.4 };
        assert_ne!(config_hash(&a), config_hash(&b));
    }

    #[test]
    fn run_evidence_construction_reports_satisfied_true_when_clean() {
        let mut declared = BTreeMap::new();
        declared.insert("hdc_ltc_predict".to_string(), Expectation::MustBePositive);
        declared.insert("static_updates".to_string(), Expectation::MustBeZero);

        let mut measured = EvidenceCounters::new();
        measured.record("hdc_ltc_predict", 12.0);

        let evidence = RunEvidence::new(RunId::new("hdc-ltc-run-1"), &"HdcLtc", declared, measured);

        assert!(evidence.satisfied);
        assert!(evidence.violations.is_empty());
        assert!(!evidence.config_hash.is_empty());
        // Must not panic.
        evidence.enforce();
    }

    #[test]
    fn run_evidence_serde_round_trip() {
        let mut declared = BTreeMap::new();
        declared.insert(
            "target_hook_fired_fraction".to_string(),
            Expectation::MustBeBelow(0.1),
        );

        let mut measured = EvidenceCounters::new();
        measured.record("target_hook_fired_fraction", 0.02);

        let evidence = RunEvidence::new(
            RunId::new("ae2-run-1"),
            &"AE-2:HOT-1:worm_spatial_updating",
            declared,
            measured,
        );

        let json = serde_json::to_string(&evidence).expect("serialize RunEvidence");
        let round_tripped: RunEvidence =
            serde_json::from_str(&json).expect("deserialize RunEvidence");

        assert_eq!(round_tripped.run_id, evidence.run_id);
        assert_eq!(round_tripped.config_hash, evidence.config_hash);
        assert_eq!(round_tripped.satisfied, evidence.satisfied);
        assert_eq!(round_tripped.measured, evidence.measured);
    }

    #[test]
    fn evidence_counters_missing_key_reads_as_zero() {
        let counters = EvidenceCounters::new();
        assert_eq!(counters.get("never_touched"), 0.0);
    }
}
