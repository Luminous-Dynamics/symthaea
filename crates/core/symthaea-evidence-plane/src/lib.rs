// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Symthaea Evidence Plane
//!
//! Shared evidence contracts plus provenance-first scientific discovery primitives.

pub mod discovery_graph;
pub mod prospective;
pub mod seed_plan;
pub mod task_validator;

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RunId(pub String);

impl RunId {
    pub fn new(label: impl Into<String>) -> Self { Self(label.into()) }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.0) }
}
impl From<&str> for RunId {
    fn from(label: &str) -> Self { Self::new(label) }
}
impl From<String> for RunId {
    fn from(label: String) -> Self { Self::new(label) }
}

pub fn config_hash<T: fmt::Debug>(config: &T) -> String {
    let mut hasher = DefaultHasher::new();
    format!("{config:?}").hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EvidenceCounters(HashMap<String, f64>);

impl EvidenceCounters {
    pub fn new() -> Self { Self::default() }
    pub fn record(&mut self, name: impl Into<String>, value: f64) { self.0.insert(name.into(), value); }
    pub fn add(&mut self, name: impl Into<String>, delta: f64) { *self.0.entry(name.into()).or_insert(0.0) += delta; }
    pub fn get(&self, name: &str) -> f64 { *self.0.get(name).unwrap_or(&0.0) }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &f64)> { self.0.iter() }
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    pub fn len(&self) -> usize { self.0.len() }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Expectation { MustBeZero, MustBePositive, MustExceed(f64), MustBeBelow(f64) }

impl Expectation {
    pub fn is_satisfied_by(&self, measured: f64) -> bool {
        match self {
            Self::MustBeZero => measured == 0.0,
            Self::MustBePositive => measured > 0.0,
            Self::MustExceed(t) => measured > *t,
            Self::MustBeBelow(t) => measured < *t,
        }
    }
}
impl fmt::Display for Expectation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MustBeZero => write!(f, "must be zero"),
            Self::MustBePositive => write!(f, "must be positive (> 0)"),
            Self::MustExceed(t) => write!(f, "must exceed {t}"),
            Self::MustBeBelow(t) => write!(f, "must be below {t}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailedExpectation {
    pub name: String,
    pub expectation: Expectation,
    pub measured: f64,
}
impl fmt::Display for FailedExpectation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} (measured {})", self.name, self.expectation, self.measured)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IntegrityViolation { pub failures: Vec<FailedExpectation> }
impl fmt::Display for IntegrityViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "evidence-plane integrity check failed ({} violation(s)):", self.failures.len())?;
        for failure in &self.failures { writeln!(f, "  - {failure}")?; }
        Ok(())
    }
}
impl std::error::Error for IntegrityViolation {}

pub fn check_integrity(
    declared: &HashMap<String, Expectation>,
    measured: &EvidenceCounters,
) -> Result<(), IntegrityViolation> {
    let mut failures: Vec<FailedExpectation> = declared.iter().filter_map(|(name, expectation)| {
        let value = measured.get(name);
        (!expectation.is_satisfied_by(value)).then(|| FailedExpectation {
            name: name.clone(), expectation: *expectation, measured: value,
        })
    }).collect();
    failures.sort_by(|a, b| a.name.cmp(&b.name));
    if failures.is_empty() { Ok(()) } else { Err(IntegrityViolation { failures }) }
}

pub fn enforce_integrity(declared: &HashMap<String, Expectation>, measured: &EvidenceCounters) {
    if let Err(violation) = check_integrity(declared, measured) { panic!("{violation}"); }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEvidence {
    pub run_id: RunId,
    pub config_hash: String,
    pub declared: BTreeMap<String, Expectation>,
    pub measured: EvidenceCounters,
    pub satisfied: bool,
    pub violations: Vec<FailedExpectation>,
}
impl RunEvidence {
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
            Err(v) => (false, v.failures),
        };
        Self { run_id, config_hash: config_hash(config), declared, measured, satisfied, violations }
    }
    pub fn enforce(&self) {
        if !self.satisfied {
            panic!("{}", IntegrityViolation { failures: self.violations.clone() });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrity_passes_and_fails_deterministically() {
        let mut declared = HashMap::new();
        declared.insert("predict".into(), Expectation::MustBePositive);
        let mut measured = EvidenceCounters::new();
        measured.record("predict", 1.0);
        assert!(check_integrity(&declared, &measured).is_ok());
        measured.record("predict", 0.0);
        let error = check_integrity(&declared, &measured).unwrap_err();
        assert_eq!(error.failures[0].name, "predict");
    }

    #[test]
    #[should_panic(expected = "predict")]
    fn enforce_panics_on_violation() {
        let mut declared = HashMap::new();
        declared.insert("predict".into(), Expectation::MustBePositive);
        enforce_integrity(&declared, &EvidenceCounters::new());
    }

    #[test]
    fn run_evidence_serde_round_trip() {
        let mut declared = BTreeMap::new();
        declared.insert("target".into(), Expectation::MustBeBelow(0.1));
        let mut measured = EvidenceCounters::new();
        measured.record("target", 0.02);
        let evidence = RunEvidence::new(RunId::new("run-1"), &"cfg", declared, measured);
        let json = serde_json::to_string(&evidence).unwrap();
        let round_trip: RunEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip.run_id, evidence.run_id);
        assert_eq!(round_trip.satisfied, evidence.satisfied);
    }
}
