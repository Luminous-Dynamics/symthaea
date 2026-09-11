// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provenance-preserving adapter from external simulation evidence into
//! domain-neutral Symthaea discovery records.
//!
//! The central rule is intentionally strict: only a `SimulationResult` that
//! already satisfies `SimulationResult::is_engineering_evidence()` can cross
//! this boundary. Dry-run fixtures, legacy/unknown execution modes, failed
//! solves, empty metrics, and incomplete solver provenance are rejected rather
//! than relabeled as weaker scientific evidence.

#![forbid(unsafe_code)]

use symthaea_discovery::{
    CandidateId, DiscoveryError, Evaluation, EvidenceKind, EvidenceRef, FidelityLevel,
    Interval as DiscoveryInterval, ModelProvenance, Prediction,
    UncertaintyEstimate as DiscoveryUncertainty,
};
use symthaea_sim_bridge::{
    ExecutionMode, SimulationResult, UncertaintyEstimate as SimulationUncertainty,
};
use thiserror::Error;

/// Reviewable result of adapting one qualified external simulation.
///
/// Warnings remain warnings here instead of being copied into
/// `Prediction::assumptions`, which would change their scientific meaning.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptedSimulation {
    /// Exact simulator request identifier carried by the source result.
    pub request_id: String,
    /// Candidate-bound discovery evaluation containing one prediction per
    /// normalized simulation metric.
    pub evaluation: Evaluation,
    /// Adapter/solver warnings preserved without semantic relabeling.
    pub warnings: Vec<String>,
    /// Source confidence retained for auditability. Discovery ranking should
    /// use the explicit uncertainty carried by each prediction instead of
    /// treating this scalar as a substitute for evidence class or fidelity.
    pub source_confidence: f64,
}

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("candidate id cannot be empty")]
    InvalidCandidateId,

    #[error(
        "simulation result {request_id:?} is not qualified engineering evidence (mode={mode:?}, converged={converged}, metrics={metric_count})"
    )]
    NotEngineeringEvidence {
        request_id: String,
        mode: ExecutionMode,
        converged: bool,
        metric_count: usize,
    },

    #[error("qualified external simulation is missing required provenance field {field}")]
    MissingProvenance { field: &'static str },

    #[error("invalid discovery contract produced by simulation adapter: {0}")]
    Discovery(#[from] DiscoveryError),
}

/// Convert one already-qualified external solver result into discovery
/// predictions bound to `candidate_id`.
///
/// This function never runs a solver and never upgrades another execution mode
/// into external-simulation evidence. `SimulationResult::is_engineering_evidence`
/// is the authority boundary.
pub fn external_result_to_discovery(
    candidate_id: CandidateId,
    result: &SimulationResult,
) -> Result<AdaptedSimulation, AdapterError> {
    if candidate_id.0.trim().is_empty() {
        return Err(AdapterError::InvalidCandidateId);
    }

    if !result.is_engineering_evidence() {
        return Err(AdapterError::NotEngineeringEvidence {
            request_id: result.request_id.clone(),
            mode: result.evidence.mode,
            converged: result.converged,
            metric_count: result.metrics.len(),
        });
    }

    // is_engineering_evidence() has already established these fields are
    // present and non-blank. Keep explicit checked extraction here so future
    // changes to the source predicate fail closed rather than introducing an
    // unwrap-based trust dependency.
    let backend = required(result.evidence.backend.as_deref(), "backend")?;
    let solver_version = required(
        result.evidence.solver_version.as_deref(),
        "solver_version",
    )?;
    let input_digest = required(result.evidence.input_digest.as_deref(), "input_digest")?;
    let output_digest = required(result.evidence.output_digest.as_deref(), "output_digest")?;
    let parser_version = required(result.evidence.parser_version.as_deref(), "parser_version")?;

    let evidence = EvidenceRef {
        id: format!(
            "external-simulation:{}:{}:{}",
            result.request_id.trim(),
            input_digest,
            output_digest
        ),
        kind: EvidenceKind::ExternalSimulation,
        uri: None,
        digest: Some(output_digest.to_owned()),
        note: Some(format!(
            "backend={backend}; solver_version={solver_version}; parser_version={parser_version}; input_digest={input_digest}; output_digest={output_digest}"
        )),
    };
    evidence.validate()?;

    let model = ModelProvenance {
        name: backend.to_owned(),
        version: Some(solver_version.to_owned()),
        implementation_digest: None,
        input_digest: Some(input_digest.to_owned()),
        output_digest: Some(output_digest.to_owned()),
    };

    let mut predictions = Vec::with_capacity(result.metrics.len());
    for metric in &result.metrics {
        let uncertainty = metric.uncertainty.unwrap_or(result.uncertainty);
        let prediction = Prediction {
            metric: metric.name.clone(),
            value: metric.value,
            unit: metric.unit.clone(),
            uncertainty: convert_uncertainty(uncertainty)?,
            fidelity: FidelityLevel::ExternalSimulation,
            model: model.clone(),
            assumptions: Vec::new(),
            evidence: vec![evidence.clone()],
        };
        prediction.validate()?;
        predictions.push(prediction);
    }

    let evaluation = Evaluation {
        candidate_id,
        objectives: Vec::new(),
        constraints: Vec::new(),
        predictions,
        pareto_rank: None,
    };
    evaluation.validate()?;

    Ok(AdaptedSimulation {
        request_id: result.request_id.clone(),
        evaluation,
        warnings: result.warnings.clone(),
        source_confidence: result.confidence,
    })
}

fn required<'a>(value: Option<&'a str>, field: &'static str) -> Result<&'a str, AdapterError> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or(AdapterError::MissingProvenance { field })
}

fn convert_uncertainty(
    uncertainty: SimulationUncertainty,
) -> Result<DiscoveryUncertainty, AdapterError> {
    let mut converted = DiscoveryUncertainty::new(uncertainty.epistemic, uncertainty.aleatoric)?;
    if let Some(interval) = uncertainty.interval {
        converted = converted.with_interval(DiscoveryInterval::new(interval.lower, interval.upper)?);
    }
    Ok(converted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_sim_bridge::{
        Interval as SimulationInterval, SimulationEvidence, UncertaintyEstimate,
    };

    fn candidate() -> CandidateId {
        CandidateId::new("candidate:pv-001").unwrap()
    }

    fn qualified_result() -> SimulationResult {
        SimulationResult::converged("req:pv-001", 0.91)
            .with_uncertainty(
                UncertaintyEstimate::new(0.07, 0.02)
                    .with_interval(SimulationInterval::new(1.47, 1.53)),
            )
            .with_metric("band_gap", 1.50, "eV")
            .with_external_evidence(SimulationEvidence {
                mode: ExecutionMode::ExternalSolver,
                backend: Some("quantum-espresso".into()),
                solver_version: Some("7.4".into()),
                input_digest: Some("sha256:input".into()),
                output_digest: Some("sha256:output".into()),
                parser_version: Some("symthaea-qe-parser-v1".into()),
            })
    }

    #[test]
    fn qualified_external_result_maps_without_promoting_evidence_class() {
        let mut result = qualified_result();
        result.warnings.push("coarse k-point mesh".into());

        let adapted = external_result_to_discovery(candidate(), &result).unwrap();
        assert_eq!(adapted.request_id, "req:pv-001");
        assert_eq!(adapted.evaluation.predictions.len(), 1);
        assert_eq!(adapted.warnings, vec!["coarse k-point mesh"]);
        assert_eq!(adapted.source_confidence, 0.91);

        let prediction = &adapted.evaluation.predictions[0];
        assert_eq!(prediction.metric, "band_gap");
        assert_eq!(prediction.value, 1.50);
        assert_eq!(prediction.unit, "eV");
        assert_eq!(prediction.fidelity, FidelityLevel::ExternalSimulation);
        assert_eq!(prediction.evidence.len(), 1);
        assert_eq!(prediction.evidence[0].kind, EvidenceKind::ExternalSimulation);
        assert_eq!(prediction.evidence[0].digest.as_deref(), Some("sha256:output"));
        assert!(prediction.evidence[0]
            .note
            .as_deref()
            .unwrap()
            .contains("parser_version=symthaea-qe-parser-v1"));
        assert!(prediction.assumptions.is_empty());
    }

    #[test]
    fn dry_run_cannot_become_discovery_evidence() {
        let result = SimulationResult::dry_run("req:fixture", "fixture", 1.0)
            .with_metric("band_gap", 1.50, "eV");

        let error = external_result_to_discovery(candidate(), &result).unwrap_err();
        assert!(matches!(
            error,
            AdapterError::NotEngineeringEvidence {
                mode: ExecutionMode::DryRun,
                ..
            }
        ));
    }

    #[test]
    fn incomplete_external_provenance_is_rejected() {
        let result = SimulationResult::converged("req:incomplete", 0.9)
            .with_metric("band_gap", 1.5, "eV")
            .with_external_evidence(SimulationEvidence {
                mode: ExecutionMode::ExternalSolver,
                backend: Some("quantum-espresso".into()),
                solver_version: Some("7.4".into()),
                input_digest: Some("sha256:input".into()),
                output_digest: None,
                parser_version: Some("parser-v1".into()),
            });

        assert!(matches!(
            external_result_to_discovery(candidate(), &result),
            Err(AdapterError::NotEngineeringEvidence { .. })
        ));
    }

    #[test]
    fn metric_inherits_run_uncertainty_when_metric_has_none() {
        let adapted = external_result_to_discovery(candidate(), &qualified_result()).unwrap();
        let uncertainty = adapted.evaluation.predictions[0].uncertainty;
        assert_eq!(uncertainty.epistemic, 0.07);
        assert_eq!(uncertainty.aleatoric, 0.02);
        assert_eq!(uncertainty.interval.unwrap().lower, 1.47);
        assert_eq!(uncertainty.interval.unwrap().upper, 1.53);
    }

    #[test]
    fn explicit_metric_uncertainty_wins_over_run_uncertainty() {
        let mut result = qualified_result();
        result.metrics[0].uncertainty = Some(UncertaintyEstimate::new(0.01, 0.03));

        let adapted = external_result_to_discovery(candidate(), &result).unwrap();
        let uncertainty = adapted.evaluation.predictions[0].uncertainty;
        assert_eq!(uncertainty.epistemic, 0.01);
        assert_eq!(uncertainty.aleatoric, 0.03);
        assert!(uncertainty.interval.is_none());
    }

    #[test]
    fn public_tuple_constructor_cannot_bypass_candidate_id_check_at_adapter_boundary() {
        let invalid = CandidateId("   ".into());
        assert!(matches!(
            external_result_to_discovery(invalid, &qualified_result()),
            Err(AdapterError::InvalidCandidateId)
        ));
    }
}
