// SPDX-License-Identifier: AGPL-3.0-or-later
//! Support-qualified viability for regenerative genomes.
//!
//! A genome defines what a successor design must preserve. The base genome report
//! uses the closure model's direct dependency horizons. This module composes that
//! manifest with a validated flow-support graph so tooling, metrology, energy, or
//! other modeled prerequisites can conservatively tighten successor supportability.

use crate::{
    evaluate_supported_closure, RegenerativeClosureModel, RegenerativeFlowSupportV1,
    RegenerativeGenomeError, RegenerativeGenomeV1, RegenerativeHorizon,
    RegenerativeSupportedClosureError,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Support-qualified result for one genome requirement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeGenomeSupportedRequirementReport {
    /// Genome requirement identifier.
    pub requirement_id: String,
    /// Capability served by the requirement.
    pub capability_id: String,
    /// Bound baseline closure dependency.
    pub baseline_dependency_id: String,
    /// Whether the owning capability is essential.
    pub essential: bool,
    /// Conservative horizon after support-prerequisite propagation.
    pub conservative_horizon: RegenerativeHorizon,
    /// Root modeled dependencies establishing a finite conservative bound.
    pub limiting_dependency_ids: Vec<String>,
    /// Whether the requirement ultimately depends on an opaque external input
    /// whose lifetime is not represented in the closure model.
    pub externally_conditioned: bool,
}

/// Support-qualified viability report for one regenerative genome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeGenomeSupportedReport {
    /// Genome identifier.
    pub genome_id: String,
    /// Closure-model identifier.
    pub closure_model_id: String,
    /// Flow-support graph identifier.
    pub flow_support_id: String,
    /// Conservative horizon across essential genome requirements.
    pub essential_conservative_horizon: RegenerativeHorizon,
    /// Genome requirements establishing the finite system-level bound.
    pub limiting_requirement_ids: Vec<String>,
    /// Root modeled dependencies establishing the finite system-level bound.
    pub root_limiting_dependency_ids: Vec<String>,
    /// True only when every essential genome requirement has a fully modeled
    /// support path with no opaque external-input lifetime.
    pub fully_modeled_essential_support: bool,
    /// Essential requirements with externally conditioned support paths.
    pub externally_conditioned_requirement_ids: Vec<String>,
    /// Per-requirement support-qualified results.
    pub requirements: Vec<RegenerativeGenomeSupportedRequirementReport>,
}

/// Errors while composing genome and support-qualified closure evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeGenomeSupportedError {
    /// Genome does not validate against the supplied closure model.
    GenomeInvalid(RegenerativeGenomeError),
    /// Flow-support qualification/conservative closure evaluation failed.
    SupportedClosureInvalid(RegenerativeSupportedClosureError),
    /// A validated capability unexpectedly disappeared during composition.
    MissingValidatedCapability { capability_id: String },
    /// A validated dependency unexpectedly disappeared during composition.
    MissingValidatedDependency { dependency_id: String },
}

/// Evaluate one genome against support-qualified conservative closure bounds.
///
/// This function does not apply substitutions. Qualified substitution references
/// remain audit metadata until an explicit substitution derives a new closure model
/// and the genome is rebound to that model.
pub fn evaluate_genome_supported_closure(
    genome: &RegenerativeGenomeV1,
    model: &RegenerativeClosureModel,
    support: &RegenerativeFlowSupportV1,
) -> Result<RegenerativeGenomeSupportedReport, RegenerativeGenomeSupportedError> {
    genome
        .validate_against_model(model)
        .map_err(RegenerativeGenomeSupportedError::GenomeInvalid)?;
    let supported = evaluate_supported_closure(model, support)
        .map_err(RegenerativeGenomeSupportedError::SupportedClosureInvalid)?;

    let capabilities: BTreeMap<&str, bool> = model
        .capabilities
        .iter()
        .map(|capability| (capability.capability_id.as_str(), capability.essential))
        .collect();
    let dependency_support: BTreeMap<&str, _> = supported
        .dependencies
        .iter()
        .map(|dependency| (dependency.dependency_id.as_str(), dependency))
        .collect();

    let mut requirements = Vec::with_capacity(genome.requirements.len());
    let mut essential_limit: Option<u64> = None;
    for requirement in &genome.requirements {
        let essential = *capabilities
            .get(requirement.capability_id.as_str())
            .ok_or_else(|| RegenerativeGenomeSupportedError::MissingValidatedCapability {
                capability_id: requirement.capability_id.clone(),
            })?;
        let dependency = dependency_support
            .get(requirement.baseline_dependency_id.as_str())
            .ok_or_else(|| RegenerativeGenomeSupportedError::MissingValidatedDependency {
                dependency_id: requirement.baseline_dependency_id.clone(),
            })?;
        if essential {
            if let RegenerativeHorizon::FinitePeriods(periods) = dependency.conservative_horizon {
                essential_limit = Some(essential_limit.map_or(periods, |current| current.min(periods)));
            }
        }
        requirements.push(RegenerativeGenomeSupportedRequirementReport {
            requirement_id: requirement.requirement_id.clone(),
            capability_id: requirement.capability_id.clone(),
            baseline_dependency_id: requirement.baseline_dependency_id.clone(),
            essential,
            conservative_horizon: dependency.conservative_horizon,
            limiting_dependency_ids: dependency.limiting_dependency_ids.clone(),
            externally_conditioned: dependency.externally_conditioned,
        });
    }

    let essential_conservative_horizon = essential_limit.map_or(
        RegenerativeHorizon::IndefiniteUnderStaticModel,
        RegenerativeHorizon::FinitePeriods,
    );
    let mut limiting_requirement_ids = Vec::new();
    let mut root_limiters = BTreeSet::new();
    if let Some(limit) = essential_limit {
        for requirement in requirements.iter().filter(|requirement| requirement.essential) {
            if requirement.conservative_horizon == RegenerativeHorizon::FinitePeriods(limit) {
                limiting_requirement_ids.push(requirement.requirement_id.clone());
                root_limiters.extend(requirement.limiting_dependency_ids.iter().cloned());
            }
        }
    }

    let externally_conditioned_requirement_ids: Vec<String> = requirements
        .iter()
        .filter(|requirement| requirement.essential && requirement.externally_conditioned)
        .map(|requirement| requirement.requirement_id.clone())
        .collect();

    Ok(RegenerativeGenomeSupportedReport {
        genome_id: genome.genome_id.clone(),
        closure_model_id: model.model_id.clone(),
        flow_support_id: support.support_id.clone(),
        essential_conservative_horizon,
        limiting_requirement_ids,
        root_limiting_dependency_ids: root_limiters.into_iter().collect(),
        fully_modeled_essential_support: supported.fully_modeled_essential_support
            && externally_conditioned_requirement_ids.is_empty(),
        externally_conditioned_requirement_ids,
        requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DependencyGovernance, RegenerativeCapability, RegenerativeDependency,
        RegenerativeDependencyKind, RegenerativeFlowKindV1, RegenerativeFlowSupportClaimV1,
        RegenerativeGenomeRequirementV1, REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1,
        REGENERATIVE_GENOME_SCHEMA_V1,
    };

    fn model() -> RegenerativeClosureModel {
        RegenerativeClosureModel {
            model_id: "successor-support-v1".into(),
            period_duration_ms: 1,
            dependencies: vec![
                RegenerativeDependency {
                    dependency_id: "metrology".into(),
                    kind: RegenerativeDependencyKind::Metrology,
                    governance: DependencyGovernance::Ordinary,
                    demand_units_per_period: 1,
                    local_production_units_per_period: 1,
                    recycling_units_per_period: 0,
                    stockpile_units: 0,
                    unit_mass_grams: None,
                    evidence_binding: "dep:metrology".into(),
                },
                RegenerativeDependency {
                    dependency_id: "qualified-reactor-service".into(),
                    kind: RegenerativeDependencyKind::ExternalService,
                    governance: DependencyGovernance::SafeguardedExternal,
                    demand_units_per_period: 1,
                    local_production_units_per_period: 0,
                    recycling_units_per_period: 0,
                    stockpile_units: 30,
                    unit_mass_grams: None,
                    evidence_binding: "dep:reactor".into(),
                },
                RegenerativeDependency {
                    dependency_id: "successor-structure".into(),
                    kind: RegenerativeDependencyKind::Material,
                    governance: DependencyGovernance::Ordinary,
                    demand_units_per_period: 100,
                    local_production_units_per_period: 100,
                    recycling_units_per_period: 0,
                    stockpile_units: 0,
                    unit_mass_grams: Some(1_000),
                    evidence_binding: "dep:successor-structure".into(),
                },
            ],
            capabilities: vec![RegenerativeCapability {
                capability_id: "successor-platform".into(),
                essential: true,
                dependency_ids: BTreeSet::from(["successor-structure".into()]),
                evidence_binding: "cap:successor-platform".into(),
            }],
            evidence_binding: "model:successor-support-v1".into(),
        }
    }

    fn claim(
        dependency_id: &str,
        prerequisites: &[&str],
    ) -> RegenerativeFlowSupportClaimV1 {
        RegenerativeFlowSupportClaimV1 {
            dependency_id: dependency_id.into(),
            flow_kind: RegenerativeFlowKindV1::Production,
            capability_binding: format!("capability:{dependency_id}:production"),
            prerequisite_dependency_ids: prerequisites.iter().map(|value| (*value).into()).collect(),
            external_input_binding: None,
            metrology_binding: format!("metrology:{dependency_id}"),
            qualification_binding: format!("qualification:{dependency_id}"),
            bootstrap_binding: None,
        }
    }

    fn support() -> RegenerativeFlowSupportV1 {
        let mut claims = vec![
            claim("metrology", &["qualified-reactor-service"]),
            claim("successor-structure", &["metrology"]),
        ];
        claims.sort_by(|a, b| {
            (a.dependency_id.as_str(), a.flow_kind)
                .cmp(&(b.dependency_id.as_str(), b.flow_kind))
        });
        RegenerativeFlowSupportV1 {
            schema_version: REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1,
            support_id: "support:successor-v1".into(),
            closure_model_id: "successor-support-v1".into(),
            closure_model_evidence_binding: "model:successor-support-v1".into(),
            claims,
            evidence_binding: "flow-support:successor-v1".into(),
        }
    }

    fn genome() -> RegenerativeGenomeV1 {
        RegenerativeGenomeV1 {
            schema_version: REGENERATIVE_GENOME_SCHEMA_V1,
            genome_id: "manta-successor-v1".into(),
            lineage_parent_binding: None,
            closure_model_id: "successor-support-v1".into(),
            closure_model_evidence_binding: "model:successor-support-v1".into(),
            requirements: vec![RegenerativeGenomeRequirementV1 {
                requirement_id: "req-successor-structure".into(),
                capability_id: "successor-platform".into(),
                baseline_dependency_id: "successor-structure".into(),
                design_binding: "design:successor-structure".into(),
                metrology_profile_binding: "metrology-profile:successor-structure".into(),
                requalification_profile_binding: "requalification:successor-structure".into(),
                disassembly_profile_binding: "disassembly:successor-structure".into(),
                recovery_profile_binding: "recovery:successor-structure".into(),
                qualified_substitution_bindings: Vec::new(),
            }],
            evidence_binding: "genome:manta-successor-v1".into(),
        }
    }

    #[test]
    fn support_graph_tightens_an_otherwise_indefinite_genome_requirement() {
        let baseline = genome().evaluate_supportability(&model()).unwrap();
        assert_eq!(
            baseline.essential_horizon,
            RegenerativeHorizon::IndefiniteUnderStaticModel
        );

        let report = evaluate_genome_supported_closure(&genome(), &model(), &support()).unwrap();
        assert_eq!(
            report.essential_conservative_horizon,
            RegenerativeHorizon::FinitePeriods(30)
        );
        assert_eq!(
            report.limiting_requirement_ids,
            vec!["req-successor-structure"]
        );
        assert_eq!(
            report.root_limiting_dependency_ids,
            vec!["qualified-reactor-service"]
        );
        assert!(report.fully_modeled_essential_support);
    }

    #[test]
    fn externally_conditioned_support_is_visible_at_genome_requirement_level() {
        let mut support = support();
        let metrology = support
            .claims
            .iter_mut()
            .find(|claim| claim.dependency_id == "metrology")
            .unwrap();
        metrology.prerequisite_dependency_ids.clear();
        metrology.external_input_binding = Some("external-input:qualified-natural-flux".into());

        let report = evaluate_genome_supported_closure(&genome(), &model(), &support).unwrap();
        assert!(!report.fully_modeled_essential_support);
        assert_eq!(
            report.externally_conditioned_requirement_ids,
            vec!["req-successor-structure"]
        );
    }
}
