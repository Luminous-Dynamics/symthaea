// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative closure bounds after qualifying recurring local-flow support.
//!
//! The base regenerative closure model evaluates each dependency from its own
//! recurring flow and stockpile. This module adds a deliberately conservative
//! qualification view: if a claimed local flow depends on another modeled
//! dependency with a shorter finite horizon, that shorter horizon propagates to
//! the supported dependency. Opaque external inputs remain explicit uncertainty
//! rather than being silently treated as closed infrastructure.

use crate::{
    RegenerativeClosureError, RegenerativeClosureModel, RegenerativeFlowSupportError,
    RegenerativeFlowSupportV1, RegenerativeHorizon,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Conservative support result for one closure dependency.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeSupportedDependencyReport {
    /// Dependency identifier from the bound closure model.
    pub dependency_id: String,
    /// Guaranteed horizon under the static model after prerequisite propagation.
    ///
    /// This is a conservative bound, not a predicted failure time. A dependency
    /// may continue longer through residual inventory or alternate surviving flow.
    pub conservative_horizon: RegenerativeHorizon,
    /// Root dependencies establishing the finite conservative bound, if any.
    pub limiting_dependency_ids: Vec<String>,
    /// Whether this dependency's support chain reaches an opaque external input
    /// whose lifetime is not represented in the closure model.
    pub externally_conditioned: bool,
}

/// Derived support-qualified closure report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeSupportedClosureReport {
    /// Bound closure-model identifier.
    pub closure_model_id: String,
    /// Bound flow-support graph identifier.
    pub flow_support_id: String,
    /// Conservative horizon across dependencies of essential capabilities.
    pub essential_conservative_horizon: RegenerativeHorizon,
    /// Root dependencies establishing that finite essential bound.
    pub limiting_dependency_ids: Vec<String>,
    /// True only when every essential support path is represented by modeled
    /// dependencies rather than an opaque external-input binding.
    pub fully_modeled_essential_support: bool,
    /// Dependencies in essential support paths that ultimately rely on opaque
    /// external inputs without a modeled lifetime.
    pub externally_conditioned_essential_dependency_ids: Vec<String>,
    /// Per-dependency conservative support results, sorted by dependency ID.
    pub dependencies: Vec<RegenerativeSupportedDependencyReport>,
}

/// Errors while deriving support-qualified conservative bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeSupportedClosureError {
    /// The support graph is not valid for the supplied closure model.
    FlowSupportInvalid(RegenerativeFlowSupportError),
    /// The closure model could not evaluate one dependency horizon.
    ClosureModelInvalid(RegenerativeClosureError),
    /// Internal reference was unexpectedly absent after successful validation.
    MissingValidatedDependency { dependency_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SupportState {
    horizon: RegenerativeHorizon,
    limiting_dependency_ids: BTreeSet<String>,
    externally_conditioned: bool,
}

/// Evaluate conservative dependency and essential-system support bounds.
///
/// The support graph must first pass [`RegenerativeFlowSupportV1::validate_against_model`].
/// A finite prerequisite bound propagates downstream until a fixed point is
/// reached. `IndefiniteUnderStaticModel` behaves as infinity. Opaque external
/// inputs do not invent a numerical lifetime; they instead propagate the
/// `externally_conditioned` flag through downstream support paths.
pub fn evaluate_supported_closure(
    model: &RegenerativeClosureModel,
    support: &RegenerativeFlowSupportV1,
) -> Result<RegenerativeSupportedClosureReport, RegenerativeSupportedClosureError> {
    support
        .validate_against_model(model)
        .map_err(RegenerativeSupportedClosureError::FlowSupportInvalid)?;

    let mut states = BTreeMap::new();
    for dependency in &model.dependencies {
        let horizon = dependency
            .autonomous_horizon()
            .map_err(RegenerativeSupportedClosureError::ClosureModelInvalid)?;
        let limiting_dependency_ids = match horizon {
            RegenerativeHorizon::FinitePeriods(_) => {
                BTreeSet::from([dependency.dependency_id.clone()])
            }
            RegenerativeHorizon::IndefiniteUnderStaticModel => BTreeSet::new(),
        };
        states.insert(
            dependency.dependency_id.clone(),
            SupportState {
                horizon,
                limiting_dependency_ids,
                externally_conditioned: false,
            },
        );
    }

    for claim in &support.claims {
        if claim.external_input_binding.is_some() {
            states
                .get_mut(&claim.dependency_id)
                .ok_or_else(|| RegenerativeSupportedClosureError::MissingValidatedDependency {
                    dependency_id: claim.dependency_id.clone(),
                })?
                .externally_conditioned = true;
        }
    }

    // Horizon/uncertainty information only moves toward more conservative values.
    // At most N passes are needed for information to traverse a simple path of N
    // modeled dependencies; cycles cannot introduce a new, less conservative value.
    for _ in 0..model.dependencies.len().max(1) {
        let mut changed = false;
        for claim in &support.claims {
            for prerequisite_id in &claim.prerequisite_dependency_ids {
                let prerequisite = states.get(prerequisite_id).cloned().ok_or_else(|| {
                    RegenerativeSupportedClosureError::MissingValidatedDependency {
                        dependency_id: prerequisite_id.clone(),
                    }
                })?;
                let target = states.get_mut(&claim.dependency_id).ok_or_else(|| {
                    RegenerativeSupportedClosureError::MissingValidatedDependency {
                        dependency_id: claim.dependency_id.clone(),
                    }
                })?;

                if prerequisite.externally_conditioned && !target.externally_conditioned {
                    target.externally_conditioned = true;
                    changed = true;
                }

                match (target.horizon, prerequisite.horizon) {
                    (
                        RegenerativeHorizon::IndefiniteUnderStaticModel,
                        RegenerativeHorizon::FinitePeriods(_),
                    ) => {
                        target.horizon = prerequisite.horizon;
                        target.limiting_dependency_ids =
                            prerequisite.limiting_dependency_ids.clone();
                        changed = true;
                    }
                    (
                        RegenerativeHorizon::FinitePeriods(current),
                        RegenerativeHorizon::FinitePeriods(candidate),
                    ) if candidate < current => {
                        target.horizon = prerequisite.horizon;
                        target.limiting_dependency_ids =
                            prerequisite.limiting_dependency_ids.clone();
                        changed = true;
                    }
                    (
                        RegenerativeHorizon::FinitePeriods(current),
                        RegenerativeHorizon::FinitePeriods(candidate),
                    ) if candidate == current => {
                        let before = target.limiting_dependency_ids.len();
                        target
                            .limiting_dependency_ids
                            .extend(prerequisite.limiting_dependency_ids.iter().cloned());
                        changed |= target.limiting_dependency_ids.len() != before;
                    }
                    _ => {}
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut essential_horizon = RegenerativeHorizon::IndefiniteUnderStaticModel;
    let mut essential_limiters = BTreeSet::new();
    let mut externally_conditioned_essential = BTreeSet::new();

    for capability in model.capabilities.iter().filter(|capability| capability.essential) {
        for dependency_id in &capability.dependency_ids {
            let state = states.get(dependency_id).ok_or_else(|| {
                RegenerativeSupportedClosureError::MissingValidatedDependency {
                    dependency_id: dependency_id.clone(),
                }
            })?;
            if state.externally_conditioned {
                externally_conditioned_essential.insert(dependency_id.clone());
            }
            match (essential_horizon, state.horizon) {
                (
                    RegenerativeHorizon::IndefiniteUnderStaticModel,
                    RegenerativeHorizon::FinitePeriods(_),
                ) => {
                    essential_horizon = state.horizon;
                    essential_limiters = state.limiting_dependency_ids.clone();
                }
                (
                    RegenerativeHorizon::FinitePeriods(current),
                    RegenerativeHorizon::FinitePeriods(candidate),
                ) if candidate < current => {
                    essential_horizon = state.horizon;
                    essential_limiters = state.limiting_dependency_ids.clone();
                }
                (
                    RegenerativeHorizon::FinitePeriods(current),
                    RegenerativeHorizon::FinitePeriods(candidate),
                ) if candidate == current => {
                    essential_limiters.extend(state.limiting_dependency_ids.iter().cloned());
                }
                _ => {}
            }
        }
    }

    let dependencies = states
        .into_iter()
        .map(|(dependency_id, state)| RegenerativeSupportedDependencyReport {
            dependency_id,
            conservative_horizon: state.horizon,
            limiting_dependency_ids: state.limiting_dependency_ids.into_iter().collect(),
            externally_conditioned: state.externally_conditioned,
        })
        .collect();

    Ok(RegenerativeSupportedClosureReport {
        closure_model_id: model.model_id.clone(),
        flow_support_id: support.support_id.clone(),
        essential_conservative_horizon: essential_horizon,
        limiting_dependency_ids: essential_limiters.into_iter().collect(),
        fully_modeled_essential_support: externally_conditioned_essential.is_empty(),
        externally_conditioned_essential_dependency_ids: externally_conditioned_essential
            .into_iter()
            .collect(),
        dependencies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DependencyGovernance, RegenerativeCapability, RegenerativeDependency,
        RegenerativeDependencyKind, RegenerativeFlowKindV1, RegenerativeFlowSupportClaimV1,
    };

    fn model() -> RegenerativeClosureModel {
        RegenerativeClosureModel {
            model_id: "supported-forge-v1".into(),
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
                    dependency_id: "structural-material".into(),
                    kind: RegenerativeDependencyKind::Material,
                    governance: DependencyGovernance::Ordinary,
                    demand_units_per_period: 100,
                    local_production_units_per_period: 80,
                    recycling_units_per_period: 20,
                    stockpile_units: 0,
                    unit_mass_grams: Some(1_000),
                    evidence_binding: "dep:structure".into(),
                },
            ],
            capabilities: vec![RegenerativeCapability {
                capability_id: "persistent-ocean-infrastructure".into(),
                essential: true,
                dependency_ids: BTreeSet::from([
                    "metrology".into(),
                    "qualified-reactor-service".into(),
                    "structural-material".into(),
                ]),
                evidence_binding: "cap:persistent".into(),
            }],
            evidence_binding: "model:supported-forge-v1".into(),
        }
    }

    fn claim(
        dependency_id: &str,
        flow_kind: RegenerativeFlowKindV1,
        prerequisites: &[&str],
    ) -> RegenerativeFlowSupportClaimV1 {
        RegenerativeFlowSupportClaimV1 {
            dependency_id: dependency_id.into(),
            flow_kind,
            capability_binding: format!("capability:{dependency_id}:{flow_kind:?}"),
            prerequisite_dependency_ids: prerequisites.iter().map(|value| (*value).into()).collect(),
            external_input_binding: None,
            metrology_binding: format!("metrology:{dependency_id}:{flow_kind:?}"),
            qualification_binding: format!("qualification:{dependency_id}:{flow_kind:?}"),
            bootstrap_binding: None,
        }
    }

    fn support() -> RegenerativeFlowSupportV1 {
        let mut metrology = claim(
            "metrology",
            RegenerativeFlowKindV1::Production,
            &["qualified-reactor-service"],
        );
        metrology.bootstrap_binding = Some("bootstrap:metrology".into());
        let mut claims = vec![
            metrology,
            claim(
                "structural-material",
                RegenerativeFlowKindV1::Production,
                &["metrology"],
            ),
            claim(
                "structural-material",
                RegenerativeFlowKindV1::Recycling,
                &["metrology"],
            ),
        ];
        claims.sort_by(|a, b| {
            (a.dependency_id.as_str(), a.flow_kind)
                .cmp(&(b.dependency_id.as_str(), b.flow_kind))
        });
        RegenerativeFlowSupportV1 {
            schema_version: crate::REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1,
            support_id: "support:supported-forge-v1".into(),
            closure_model_id: "supported-forge-v1".into(),
            closure_model_evidence_binding: "model:supported-forge-v1".into(),
            claims,
            evidence_binding: "flow-support:supported-forge-v1".into(),
        }
    }

    #[test]
    fn finite_prerequisite_horizon_propagates_downstream() {
        let report = evaluate_supported_closure(&model(), &support()).unwrap();
        assert_eq!(
            report.essential_conservative_horizon,
            RegenerativeHorizon::FinitePeriods(30)
        );
        assert_eq!(
            report.limiting_dependency_ids,
            vec!["qualified-reactor-service"]
        );
        let structural = report
            .dependencies
            .iter()
            .find(|dependency| dependency.dependency_id == "structural-material")
            .unwrap();
        assert_eq!(
            structural.conservative_horizon,
            RegenerativeHorizon::FinitePeriods(30)
        );
        assert_eq!(
            structural.limiting_dependency_ids,
            vec!["qualified-reactor-service"]
        );
        assert!(report.fully_modeled_essential_support);
    }

    #[test]
    fn opaque_external_input_is_visible_instead_of_treated_as_closed() {
        let mut support = support();
        let metrology = support
            .claims
            .iter_mut()
            .find(|claim| claim.dependency_id == "metrology")
            .unwrap();
        metrology.prerequisite_dependency_ids.clear();
        metrology.external_input_binding = Some("external-input:qualified-natural-flux".into());
        let report = evaluate_supported_closure(&model(), &support).unwrap();
        assert!(!report.fully_modeled_essential_support);
        assert!(report
            .externally_conditioned_essential_dependency_ids
            .contains(&"metrology".to_string()));
        let structural = report
            .dependencies
            .iter()
            .find(|dependency| dependency.dependency_id == "structural-material")
            .unwrap();
        assert!(structural.externally_conditioned);
    }

    #[test]
    fn support_graph_must_validate_before_bounds_are_derived() {
        let mut support = support();
        support.claims.pop();
        assert!(matches!(
            evaluate_supported_closure(&model(), &support),
            Err(RegenerativeSupportedClosureError::FlowSupportInvalid(_))
        ));
    }
}
