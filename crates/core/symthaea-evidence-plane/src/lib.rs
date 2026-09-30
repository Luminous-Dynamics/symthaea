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
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

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

/// Legacy config-identity fingerprint retained for compatibility.
///
/// This is intentionally process/toolchain dependent: Rust does not specify
/// the `DefaultHasher` algorithm as a stable serialization format. It is
/// suitable only for local diagnostics/deduplication, never for persisted
/// experiment identity or cross-version reproducibility.
pub fn config_hash<T: fmt::Debug>(config: &T) -> String {
    let mut hasher = DefaultHasher::new();
    format!("{config:?}").hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Version identifier for the stable semantic configuration identity contract.
///
/// Persist this alongside [`stable_config_hash`] when an evidence record needs
/// to declare which identity schema produced its digest.
pub const STABLE_CONFIG_IDENTITY_SCHEMA: &str = "symthaea:stable-config-identity:v1";

/// Stable semantic configuration identity.
///
/// The input is first converted to a JSON value, object keys are recursively
/// sorted, and the canonical JSON bytes are hashed with BLAKE3. This avoids
/// accidentally inheriting `HashMap` iteration order from a caller's
/// serialization implementation.
///
/// The schema/domain tag is part of the bytes being hashed. Changing this
/// canonicalization contract therefore requires an explicit version bump rather
/// than silently producing a different meaning under the same identity scheme.
///
/// This is an identity/determinism primitive, not an authorization primitive.
/// Use the evidence/provenance contracts for security-sensitive commitments.
pub fn stable_config_hash<T: Serialize>(config: &T) -> String {
    stable_config_hash_result(config)
        .expect("stable config serialization must produce a JSON value")
}

/// Fallible form of the stable semantic configuration identity.
///
/// Use this at persistence, ingestion, or externally supplied configuration
/// boundaries where serialization failure must remain an explicit validation
/// result rather than becoming a panic.
pub fn stable_config_hash_result<T: Serialize>(
    config: &T,
) -> Result<String, serde_json::Error> {
    const DOMAIN: &[u8] = b"symthaea:stable-config-identity:v1\0";
    let value = serde_json::to_value(config)?;
    let mut bytes = Vec::with_capacity(256);
    bytes.extend_from_slice(DOMAIN);
    write_canonical_json(&value, &mut bytes);
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Serialize a JSON value deterministically, recursively sorting object keys.
fn write_canonical_json(value: &serde_json::Value, out: &mut Vec<u8>) {
    match value {
        serde_json::Value::Null => out.extend_from_slice(b"null"),
        serde_json::Value::Bool(v) => out.extend_from_slice(if *v { b"true" } else { b"false" }),
        serde_json::Value::Number(v) => out.extend_from_slice(v.to_string().as_bytes()),
        serde_json::Value::String(v) => {
            out.extend_from_slice(serde_json::to_string(v).expect("string serialization").as_bytes())
        }
        serde_json::Value::Array(values) => {
            out.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 { out.push(b','); }
                write_canonical_json(value, out);
            }
            out.push(b']');
        }
        serde_json::Value::Object(values) => {
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
            out.push(b'{');
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index != 0 { out.push(b','); }
                out.extend_from_slice(serde_json::to_string(key).expect("object-key serialization").as_bytes());
                out.push(b':');
                write_canonical_json(value, out);
            }
            out.push(b'}');
        }
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
    /// Legacy/local fingerprint retained for serialized compatibility.
    pub config_hash: String,
    /// Stable semantic configuration identity, when the run was constructed
    /// through [`RunEvidence::new_stable`]. Historical records deserialize as
    /// `None` rather than receiving an invented identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_identity: Option<String>,
    /// Schema/domain identifier for [`config_identity`]. Kept separate from
    /// the digest so consumers cannot confuse an identity value with its
    /// canonicalization contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_identity_schema: Option<String>,
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
            config_identity: None,
            config_identity_schema: None,
            declared,
            measured,
            satisfied,
            violations,
        }
    }

    /// Build a `RunEvidence` record with the stable semantic configuration
    /// identity contract while retaining the legacy fingerprint for backward
    /// compatibility. This method is additive: old records remain readable,
    /// and legacy `config_hash` is never silently reinterpreted.
    pub fn new_stable<T: Serialize>(
        run_id: RunId,
        config: &T,
        declared: BTreeMap<String, Expectation>,
        measured: EvidenceCounters,
    ) -> Self {
        let mut evidence = Self::new(run_id, config, declared, measured);
        evidence.config_identity = Some(stable_config_hash(config));
        evidence.config_identity_schema = Some(STABLE_CONFIG_IDENTITY_SCHEMA.to_string());
        evidence
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

    #[test]
    fn stable_config_hash_is_deterministic() {
        let a = stable_config_hash(&(vec![100_u64, 101, 102], vec![1_u64, 2]));
        let b = stable_config_hash(&(vec![100_u64, 101, 102], vec![1_u64, 2]));
        assert_eq!(a, b);
        assert_eq!(a.len(), 64, "BLAKE3 hex digest must be 256 bits");
    }

    #[test]
    fn stable_config_hash_result_rejects_non_json_numbers() {
        #[derive(Serialize)]
        struct Unsupported {
            value: f64,
        }

        assert!(stable_config_hash_result(&Unsupported { value: f64::NAN }).is_err());
        assert!(stable_config_hash_result(&Unsupported { value: f64::INFINITY }).is_err());
    }
    #[test]
    fn stable_config_hash_changes_with_semantic_input() {
        let a = stable_config_hash(&(vec![100_u64, 101, 102], vec![1_u64, 2]));
        let b = stable_config_hash(&(vec![100_u64, 101, 103], vec![1_u64, 2]));
        assert_ne!(a, b);
    }

    #[test]
    fn stable_config_hash_is_domain_separated() {
        let value = serde_json::json!({"a": 1, "b": 2});
        let mut canonical = Vec::new();
        write_canonical_json(&value, &mut canonical);
        let raw = blake3::hash(&canonical).to_hex().to_string();
        assert_ne!(stable_config_hash(&value), raw);
    }

    #[test]
    fn canonical_json_sorts_object_keys_recursively() {
        let a = serde_json::json!({"outer": {"z": 1, "a": 2}, "first": [3, 4]});
        let b = serde_json::json!({"first": [3, 4], "outer": {"a": 2, "z": 1}});
        assert_eq!(stable_config_hash(&a), stable_config_hash(&b));
    }

    /// Reproduces `TemporalStateMode::HdcLtc`'s requirement: the active
    /// mechanism's predict counter must be positive. A passing case.
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
    fn run_evidence_stable_identity_is_explicit_and_versioned() {
        let mut declared = BTreeMap::new();
        declared.insert("hdc_ltc_predict".to_string(), Expectation::MustBePositive);
        let mut measured = EvidenceCounters::new();
        measured.record("hdc_ltc_predict", 1.0);

        let config = serde_json::json!({
            "mode": "HdcLtc",
            "seed": 42,
            "nested": {"z": 2, "a": 1}
        });
        let evidence = RunEvidence::new_stable(
            RunId::new("stable-identity-run"),
            &config,
            declared,
            measured,
        );

        assert_eq!(
            evidence.config_identity_schema.as_deref(),
            Some(STABLE_CONFIG_IDENTITY_SCHEMA)
        );
        assert_eq!(
            evidence.config_identity.as_deref(),
            Some(stable_config_hash(&config).as_str())
        );
        assert!(!evidence.config_hash.is_empty());
    }

    #[test]
    fn run_evidence_legacy_json_deserializes_without_inventing_identity() {
        let legacy = r#"{
            "run_id":"legacy-run",
            "config_hash":"legacy-fingerprint",
            "declared":{},
            "measured":{},
            "satisfied":true,
            "violations":[]
        }"#;

        let evidence: RunEvidence = serde_json::from_str(legacy).expect("legacy JSON must read");
        assert_eq!(evidence.config_hash, "legacy-fingerprint");
        assert!(evidence.config_identity.is_none());
        assert!(evidence.config_identity_schema.is_none());

        let reserialized = serde_json::to_string(&evidence).expect("legacy JSON must reserialize");
        assert!(!reserialized.contains("config_identity"));
        assert!(!reserialized.contains("config_identity_schema"));
    }

    #[test]
    fn evidence_counters_missing_key_reads_as_zero() {
        let counters = EvidenceCounters::new();
        assert_eq!(counters.get("never_touched"), 0.0);
    }
}
