//! Strict compilation boundary from normalized scientific claims to prospective predictions.
//!
//! This module is deliberately narrower than an evidence adapter. It creates planning
//! artifacts only. In particular, it cannot create observations, replications, or
//! official criterion evidence, and it never accepts an observed outcome as input.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilationProvenance {
    pub source_payload_digest: String,
    pub model_lineage: Option<String>,
    pub knowledge_cutoff: Option<String>,
    pub exposure_cutoff: Option<String>,
}

impl CompilationProvenance {
    fn validate(&self) -> Result<(), PredictionCompilerError> {
        if self.source_payload_digest.trim().is_empty() {
            return Err(PredictionCompilerError::EmptyField("source_payload_digest"));
        }
        if self.model_lineage.as_deref().is_some_and(|v| v.trim().is_empty()) {
            return Err(PredictionCompilerError::EmptyField("model_lineage"));
        }
        match (&self.knowledge_cutoff, &self.exposure_cutoff) {
            (Some(knowledge), Some(exposure)) if knowledge.trim().is_empty() => {
                Err(PredictionCompilerError::EmptyField("knowledge_cutoff"))
            }
            (Some(_), Some(exposure)) if exposure.trim().is_empty() => {
                Err(PredictionCompilerError::EmptyField("exposure_cutoff"))
            }
            (Some(knowledge), Some(exposure)) if exposure < knowledge => {
                Err(PredictionCompilerError::ExposureBeforeKnowledgeCutoff),
            }
            (Some(_), None) | (None, Some(_)) => {
                Err(PredictionCompilerError::IncompleteCutoffPair)
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimForCompilation {
    pub claim_id: String,
    pub statement: String,
    pub mechanism_id: String,
    pub falsifiers: BTreeMap<String, String>,
    pub provenance: CompilationProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictionBinding {
    pub prediction_id: String,
    pub statement: String,
    pub falsifier_id: String,
    /// Explicit external-observation predicates. These describe what would count
    /// as an expected or falsifying outcome; they are not observations.
    pub expected_outcome_predicate: String,
    pub falsifying_outcome_predicate: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledPrediction {
    pub prediction_id: String,
    pub claim_id: String,
    pub mechanism_id: String,
    pub prediction: String,
    pub falsifier_id: String,
    pub falsifier: String,
    pub expected_outcome_predicate: String,
    pub falsifying_outcome_predicate: String,
    pub source_payload_digest: String,
    pub model_lineage: Option<String>,
    pub knowledge_cutoff: Option<String>,
    pub exposure_cutoff: Option<String>,
    pub identity_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictionCompilerError {
    EmptyField(&'static str),
    NoBindings,
    DuplicatePredictionId(String),
    UnknownFalsifier(String),
    ExposureBeforeKnowledgeCutoff,
    IncompleteCutoffPair,
}

pub fn compile(
    claim: &ClaimForCompilation,
    bindings: &[PredictionBinding],
) -> Result<Vec<CompiledPrediction>, PredictionCompilerError> {
    validate_claim(claim)?;
    if bindings.is_empty() {
        return Err(PredictionCompilerError::NoBindings);
    }

    let mut seen_predictions = BTreeSet::new();
    let mut compiled = Vec::with_capacity(bindings.len());

    for binding in bindings {
        validate_binding(binding)?;
        if !seen_predictions.insert(binding.prediction_id.clone()) {
            return Err(PredictionCompilerError::DuplicatePredictionId(
                binding.prediction_id.clone(),
            ));
        }
        let Some(falsifier) = claim.falsifiers.get(&binding.falsifier_id) else {
            return Err(PredictionCompilerError::UnknownFalsifier(
                binding.falsifier_id.clone(),
            ));
        };

        let identity_material = format!(
            "claim={}\nprediction={}\nfalsifier={}\nprediction_text={}\nfalsifier_text={}\nexpected={}\nfalsifying={}\nmechanism={}\nsource={}\nlineage={}\nknowledge={}\nexposure={}",
            claim.claim_id,
            binding.prediction_id,
            binding.falsifier_id,
            binding.statement,
            falsifier,
            binding.expected_outcome_predicate,
            binding.falsifying_outcome_predicate,
            claim.mechanism_id,
            claim.provenance.source_payload_digest,
            claim.provenance.model_lineage.as_deref().unwrap_or(""),
            claim.provenance.knowledge_cutoff.as_deref().unwrap_or(""),
            claim.provenance.exposure_cutoff.as_deref().unwrap_or(""),
        );
        let mut hasher = Sha256::new();
        hasher.update(identity_material.as_bytes());
        let identity_digest = format!("{:x}", hasher.finalize());

        compiled.push(CompiledPrediction {
            prediction_id: binding.prediction_id.clone(),
            claim_id: claim.claim_id.clone(),
            mechanism_id: claim.mechanism_id.clone(),
            prediction: binding.statement.clone(),
            falsifier_id: binding.falsifier_id.clone(),
            falsifier: falsifier.clone(),
            expected_outcome_predicate: binding.expected_outcome_predicate.clone(),
            falsifying_outcome_predicate: binding.falsifying_outcome_predicate.clone(),
            source_payload_digest: claim.provenance.source_payload_digest.clone(),
            model_lineage: claim.provenance.model_lineage.clone(),
            knowledge_cutoff: claim.provenance.knowledge_cutoff.clone(),
            exposure_cutoff: claim.provenance.exposure_cutoff.clone(),
            identity_digest,
        });
    }

    Ok(compiled)
}

fn validate_claim(claim: &ClaimForCompilation) -> Result<(), PredictionCompilerError> {
    require("claim_id", &claim.claim_id)?;
    require("statement", &claim.statement)?;
    require("mechanism_id", &claim.mechanism_id)?;
    if claim.falsifiers.is_empty() {
        return Err(PredictionCompilerError::EmptyField("falsifiers"));
    }
    claim.provenance.validate()
}

fn validate_binding(binding: &PredictionBinding) -> Result<(), PredictionCompilerError> {
    require("prediction_id", &binding.prediction_id)?;
    require("prediction", &binding.statement)?;
    require("falsifier_id", &binding.falsifier_id)?;
    require("expected_outcome_predicate", &binding.expected_outcome_predicate)?;
    require(
        "falsifying_outcome_predicate",
        &binding.falsifying_outcome_predicate,
    )
}

fn require(name: &'static str, value: &str) -> Result<(), PredictionCompilerError> {
    if value.trim().is_empty() {
        Err(PredictionCompilerError::EmptyField(name))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim() -> ClaimForCompilation {
        ClaimForCompilation {
            claim_id: "claim-1".into(),
            statement: "mechanism produces outcome".into(),
            mechanism_id: "mechanism-1".into(),
            falsifiers: BTreeMap::from([
                ("f-1".into(), "outcome fails".into()),
                ("f-2".into(), "replication fails".into()),
            ]),
            provenance: CompilationProvenance {
                source_payload_digest: "sha256:source".into(),
                model_lineage: Some("model-root-1".into()),
                knowledge_cutoff: Some("2026-09-20".into()),
                exposure_cutoff: Some("2026-09-28".into()),
            },
        }
    }

    fn binding(id: &str, falsifier_id: &str) -> PredictionBinding {
        PredictionBinding {
            prediction_id: id.into(),
            statement: format!("prediction {id}"),
            falsifier_id: falsifier_id.into(),
            expected_outcome_predicate: "externally observed criterion X".into(),
            falsifying_outcome_predicate: "externally observed criterion not-X".into(),
        }
    }

    #[test]
    fn compiles_explicit_prediction_to_falsifier_binding() {
        let out = compile(&claim(), &[binding("p-1", "f-1")]).unwrap();
        assert_eq!(out[0].falsifier, "outcome fails");
        assert!(!out[0].identity_digest.is_empty());
    }

    #[test]
    fn refuses_inferred_or_unknown_falsifier() {
        let err = compile(&claim(), &[binding("p-1", "not-declared")]).unwrap_err();
        assert_eq!(err, PredictionCompilerError::UnknownFalsifier("not-declared".into()));
    }

    #[test]
    fn refuses_duplicate_prediction_but_allows_shared_falsifier() {
        let err = compile(&claim(), &[binding("p-1", "f-1"), binding("p-1", "f-2")]).unwrap_err();
        assert_eq!(err, PredictionCompilerError::DuplicatePredictionId("p-1".into()));

        let out = compile(&claim(), &[binding("p-1", "f-1"), binding("p-2", "f-1")]).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn cutoff_pair_and_order_are_hard_constraints() {
        let mut c = claim();
        c.provenance.exposure_cutoff = None;
        assert_eq!(
            compile(&c, &[binding("p-1", "f-1")]).unwrap_err(),
            PredictionCompilerError::IncompleteCutoffPair
        );

        let mut c = claim();
        c.provenance.exposure_cutoff = Some("2026-09-19".into());
        assert_eq!(
            compile(&c, &[binding("p-1", "f-1")]).unwrap_err(),
            PredictionCompilerError::ExposureBeforeKnowledgeCutoff
        );
    }

    #[test]
    fn identity_is_deterministic_and_payload_sensitive() {
        let a = compile(&claim(), &[binding("p-1", "f-1")]).unwrap();
        let mut b = binding("p-1", "f-1");
        b.statement.push_str(" changed");
        let b = compile(&claim(), &[b]).unwrap();
        assert_eq!(a[0].identity_digest.len(), 64);
        assert_ne!(a[0].identity_digest, b[0].identity_digest);
    }

    #[test]
    fn observed_outcomes_cannot_enter_compiler() {
        // The input type contains only declarations. There is intentionally no
        // observed outcome field or promotion method in this module.
        let p = binding("p-1", "f-1");
        assert!(!p.expected_outcome_predicate.is_empty());
    }
}
