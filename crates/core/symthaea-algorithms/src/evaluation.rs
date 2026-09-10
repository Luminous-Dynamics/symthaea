//! Reproducible, non-authorizing evaluation receipts.
//!
//! Evaluation records measurements. It does not promote an implementation or make it callable.

use crate::{ContentId, ImplementationId, ProblemId, RegistryError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum EvaluationError {
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error("{field} must not be empty")]
    Empty { field: &'static str },
    #[error("objective measurement must be finite: {name}={value}")]
    NonFiniteObjective { name: String, value: f64 },
    #[error("duplicate objective name: {0}")]
    DuplicateObjective(String),
    #[error("correctness did not pass; performance evidence is ineligible for ranking")]
    CorrectnessNotPassed,
    #[error("receipt identity does not match its canonical fields")]
    IdentityMismatch,
}

fn require_text(field: &'static str, value: &str) -> Result<(), EvaluationError> {
    if value.trim().is_empty() {
        return Err(EvaluationError::Empty { field });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CorrectnessVerdict {
    Passed,
    Failed,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveMeasurement {
    pub name: String,
    pub direction: ObjectiveDirection,
    pub value: f64,
    pub unit: String,
}

impl ObjectiveMeasurement {
    pub fn new(
        name: impl Into<String>,
        direction: ObjectiveDirection,
        value: f64,
        unit: impl Into<String>,
    ) -> Result<Self, EvaluationError> {
        let name = name.into();
        let unit = unit.into();
        require_text("objective name", &name)?;
        require_text("objective unit", &unit)?;
        if !value.is_finite() {
            return Err(EvaluationError::NonFiniteObjective { name, value });
        }
        Ok(Self {
            name,
            direction,
            value,
            unit,
        })
    }
}

/// Exact evaluator/environment identity supplied by the caller.
///
/// These are content/provenance descriptors, not claims that the evaluator is independently
/// trustworthy. Higher evidence levels must establish that separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationContext {
    pub evaluator_id: ContentId,
    pub oracle_id: ContentId,
    pub input_profile_id: ContentId,
    pub environment_id: ContentId,
    pub source_revision: String,
    pub toolchain_profile: String,
    pub target_profile: String,
    pub seeds: Vec<u64>,
}

impl EvaluationContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        evaluator_id: ContentId,
        oracle_id: ContentId,
        input_profile_id: ContentId,
        environment_id: ContentId,
        source_revision: impl Into<String>,
        toolchain_profile: impl Into<String>,
        target_profile: impl Into<String>,
        mut seeds: Vec<u64>,
    ) -> Result<Self, EvaluationError> {
        let source_revision = source_revision.into();
        let toolchain_profile = toolchain_profile.into();
        let target_profile = target_profile.into();
        require_text("source revision", &source_revision)?;
        require_text("toolchain profile", &toolchain_profile)?;
        require_text("target profile", &target_profile)?;
        seeds.sort_unstable();
        seeds.dedup();
        Ok(Self {
            evaluator_id,
            oracle_id,
            input_profile_id,
            environment_id,
            source_revision,
            toolchain_profile,
            target_profile,
            seeds,
        })
    }

    pub fn content_id(&self) -> ContentId {
        let mut owned: Vec<Vec<u8>> = vec![
            self.evaluator_id.as_str().as_bytes().to_vec(),
            self.oracle_id.as_str().as_bytes().to_vec(),
            self.input_profile_id.as_str().as_bytes().to_vec(),
            self.environment_id.as_str().as_bytes().to_vec(),
            self.source_revision.as_bytes().to_vec(),
            self.toolchain_profile.as_bytes().to_vec(),
            self.target_profile.as_bytes().to_vec(),
        ];
        owned.extend(self.seeds.iter().map(|seed| seed.to_be_bytes().to_vec()));
        ContentId::derive(
            "symthaea.algorithm-evaluation-context.v1",
            owned.iter().map(Vec::as_slice),
        )
    }
}

/// Canonical result of one algorithm evaluation.
///
/// A receipt is historical evidence only. It is intentionally serializable but has no API that
/// converts it into promotion or runtime authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReceipt {
    pub id: ContentId,
    pub problem_id: ProblemId,
    pub implementation_id: ImplementationId,
    pub context: EvaluationContext,
    pub correctness: CorrectnessVerdict,
    pub correctness_evidence_id: ContentId,
    pub objectives: Vec<ObjectiveMeasurement>,
    pub evidence_run_id: Option<String>,
}

impl EvaluationReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        problem_id: ProblemId,
        implementation_id: ImplementationId,
        context: EvaluationContext,
        correctness: CorrectnessVerdict,
        correctness_evidence_id: ContentId,
        objectives: Vec<ObjectiveMeasurement>,
        evidence_run_id: Option<String>,
    ) -> Result<Self, EvaluationError> {
        if let Some(run_id) = &evidence_run_id {
            require_text("evidence run id", run_id)?;
        }

        let mut seen = BTreeMap::new();
        for objective in &objectives {
            if !objective.value.is_finite() {
                return Err(EvaluationError::NonFiniteObjective {
                    name: objective.name.clone(),
                    value: objective.value,
                });
            }
            if seen.insert(objective.name.clone(), ()).is_some() {
                return Err(EvaluationError::DuplicateObjective(objective.name.clone()));
            }
        }

        let id = Self::derive_id(
            &problem_id,
            &implementation_id,
            &context,
            correctness,
            &correctness_evidence_id,
            &objectives,
            evidence_run_id.as_deref(),
        );
        Ok(Self {
            id,
            problem_id,
            implementation_id,
            context,
            correctness,
            correctness_evidence_id,
            objectives,
            evidence_run_id,
        })
    }

    pub fn eligible_objectives(&self) -> Result<&[ObjectiveMeasurement], EvaluationError> {
        if self.correctness != CorrectnessVerdict::Passed {
            return Err(EvaluationError::CorrectnessNotPassed);
        }
        Ok(&self.objectives)
    }

    pub fn validate(&self) -> Result<(), EvaluationError> {
        let rebuilt = Self::new(
            self.problem_id.clone(),
            self.implementation_id.clone(),
            self.context.clone(),
            self.correctness,
            self.correctness_evidence_id.clone(),
            self.objectives.clone(),
            self.evidence_run_id.clone(),
        )?;
        if rebuilt.id != self.id {
            return Err(EvaluationError::IdentityMismatch);
        }
        Ok(())
    }

    fn derive_id(
        problem_id: &ProblemId,
        implementation_id: &ImplementationId,
        context: &EvaluationContext,
        correctness: CorrectnessVerdict,
        correctness_evidence_id: &ContentId,
        objectives: &[ObjectiveMeasurement],
        evidence_run_id: Option<&str>,
    ) -> ContentId {
        let correctness_tag = format!("{correctness:?}");
        let mut owned: Vec<Vec<u8>> = vec![
            problem_id.0.as_str().as_bytes().to_vec(),
            implementation_id.0.as_str().as_bytes().to_vec(),
            context.content_id().as_str().as_bytes().to_vec(),
            correctness_tag.as_bytes().to_vec(),
            correctness_evidence_id.as_str().as_bytes().to_vec(),
            evidence_run_id.unwrap_or("").as_bytes().to_vec(),
        ];
        for objective in objectives {
            owned.push(objective.name.as_bytes().to_vec());
            owned.push(format!("{:?}", objective.direction).into_bytes());
            owned.push(objective.value.to_bits().to_be_bytes().to_vec());
            owned.push(objective.unit.as_bytes().to_vec());
        }
        ContentId::derive(
            "symthaea.algorithm-evaluation-receipt.v1",
            owned.iter().map(Vec::as_slice),
        )
    }
}

/// Minimal bridge from the shared evidence plane. This preserves the evidence-plane run identity
/// without claiming that one counter set alone establishes semantic correctness.
#[cfg(feature = "evidence-plane")]
pub fn evidence_run_label(run: &symthaea_evidence_plane::RunEvidence) -> String {
    run.run_id.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AlgorithmId, ImplementationId, ProblemId};

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn context() -> EvaluationContext {
        EvaluationContext::new(
            cid("evaluator", "v1"),
            cid("oracle", "scalar-reference"),
            cid("inputs", "seeded-pairs"),
            cid("env", "x86_64-test"),
            "deadbeef",
            "rust-1.96.0",
            "x86_64-unknown-linux-gnu",
            vec![7, 3, 7],
        )
        .unwrap()
    }

    fn implementation_id() -> ImplementationId {
        ImplementationId(cid("impl", "candidate"))
    }

    fn problem_id() -> ProblemId {
        ProblemId(cid("problem", "hamming"))
    }

    #[test]
    fn context_seed_order_is_canonical() {
        let a = context();
        let b = EvaluationContext::new(
            a.evaluator_id.clone(),
            a.oracle_id.clone(),
            a.input_profile_id.clone(),
            a.environment_id.clone(),
            a.source_revision.clone(),
            a.toolchain_profile.clone(),
            a.target_profile.clone(),
            vec![3, 7],
        )
        .unwrap();
        assert_eq!(a.content_id(), b.content_id());
    }

    #[test]
    fn rejects_non_finite_measurements_before_ranking() {
        let err = ObjectiveMeasurement::new(
            "latency",
            ObjectiveDirection::Minimize,
            f64::NAN,
            "ns/op",
        )
        .unwrap_err();
        assert!(matches!(err, EvaluationError::NonFiniteObjective { .. }));
    }

    #[test]
    fn failed_correctness_has_no_rankable_objectives() {
        let receipt = EvaluationReceipt::new(
            problem_id(),
            implementation_id(),
            context(),
            CorrectnessVerdict::Failed,
            cid("correctness", "mismatch"),
            vec![ObjectiveMeasurement::new(
                "latency",
                ObjectiveDirection::Minimize,
                10.0,
                "ns/op",
            )
            .unwrap()],
            None,
        )
        .unwrap();
        assert_eq!(
            receipt.eligible_objectives().unwrap_err(),
            EvaluationError::CorrectnessNotPassed
        );
    }

    #[test]
    fn receipt_identity_changes_with_environment() {
        let objective = ObjectiveMeasurement::new(
            "latency",
            ObjectiveDirection::Minimize,
            10.0,
            "ns/op",
        )
        .unwrap();
        let a = EvaluationReceipt::new(
            problem_id(),
            implementation_id(),
            context(),
            CorrectnessVerdict::Passed,
            cid("correctness", "pass"),
            vec![objective.clone()],
            Some("run-1".into()),
        )
        .unwrap();

        let mut changed = context();
        changed.target_profile = "aarch64-unknown-linux-gnu".into();
        let b = EvaluationReceipt::new(
            problem_id(),
            implementation_id(),
            changed,
            CorrectnessVerdict::Passed,
            cid("correctness", "pass"),
            vec![objective],
            Some("run-1".into()),
        )
        .unwrap();
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn receipt_self_validation_detects_tampering() {
        let mut receipt = EvaluationReceipt::new(
            problem_id(),
            implementation_id(),
            context(),
            CorrectnessVerdict::Passed,
            cid("correctness", "pass"),
            vec![],
            None,
        )
        .unwrap();
        receipt.implementation_id = ImplementationId(cid("impl", "other"));
        assert_eq!(receipt.validate().unwrap_err(), EvaluationError::IdentityMismatch);
    }

    #[test]
    fn algorithm_id_type_stays_distinct_from_implementation_id() {
        let _algorithm = AlgorithmId(cid("algorithm", "a"));
        let _implementation = ImplementationId(cid("implementation", "a"));
        // Compile-time type separation is the assertion.
    }
}
