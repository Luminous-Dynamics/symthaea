// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Principal / presented-identity / executable-agent separation for institutional games.
//!
//! False-name analysis changes the participant identity state itself; it is not
//! represented as an ordinary unilateral deviation inside one fixed N-player
//! game. This module provides the identity ontology plus explicit counterfactual
//! witnesses. Mechanism-specific enumeration belongs in a later institutional
//! lab layer.

use std::collections::{BTreeMap, BTreeSet};

macro_rules! identifier_type {
    ($name:ident, $label:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, String> {
                let value = value.into();
                if value.is_empty() {
                    return Err(concat!($label, " must be non-empty").to_string());
                }
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
    };
}

identifier_type!(PrincipalId, "principal id");
identifier_type!(IdentityId, "identity id");
identifier_type!(AgentInstanceId, "agent instance id");

/// Lifecycle state of one presented identity in a registry snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityStatus {
    Active,
    Revoked,
}

/// Explicit binding between a principal, one presented identity and one executable agent instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentityBinding {
    pub principal: PrincipalId,
    pub identity: IdentityId,
    pub agent_instance: AgentInstanceId,
    pub status: IdentityStatus,
}

impl IdentityBinding {
    pub fn active(
        principal: PrincipalId,
        identity: IdentityId,
        agent_instance: AgentInstanceId,
    ) -> Self {
        Self {
            principal,
            identity,
            agent_instance,
            status: IdentityStatus::Active,
        }
    }
}

/// Deterministic identity registry for one institutional-game snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentityRegistry {
    bindings: BTreeMap<IdentityId, IdentityBinding>,
}

impl IdentityRegistry {
    pub fn new(bindings: Vec<IdentityBinding>) -> Result<Self, String> {
        let mut indexed = BTreeMap::new();
        for binding in bindings {
            let identity = binding.identity.clone();
            if indexed.insert(identity.clone(), binding).is_some() {
                return Err(format!("duplicate identity id: {}", identity.as_str()));
            }
        }
        Ok(Self { bindings: indexed })
    }

    pub fn binding(&self, identity: &IdentityId) -> Option<&IdentityBinding> {
        self.bindings.get(identity)
    }

    pub fn identity_count(&self) -> usize {
        self.bindings.len()
    }

    pub fn active_identity_count(&self) -> usize {
        self.bindings
            .values()
            .filter(|binding| binding.status == IdentityStatus::Active)
            .count()
    }

    pub fn principal_count(&self) -> usize {
        self.bindings
            .values()
            .map(|binding| binding.principal.clone())
            .collect::<BTreeSet<_>>()
            .len()
    }

    pub fn agent_instance_count(&self) -> usize {
        self.bindings
            .values()
            .map(|binding| binding.agent_instance.clone())
            .collect::<BTreeSet<_>>()
            .len()
    }

    pub fn active_identities_for_principal(&self, principal: &PrincipalId) -> Vec<&IdentityBinding> {
        self.bindings
            .values()
            .filter(|binding| {
                binding.status == IdentityStatus::Active && &binding.principal == principal
            })
            .collect()
    }

    /// Return a new registry snapshot with one additional binding.
    pub fn with_added_identity(&self, binding: IdentityBinding) -> Result<Self, String> {
        if self.bindings.contains_key(&binding.identity) {
            return Err(format!(
                "duplicate identity id: {}",
                binding.identity.as_str()
            ));
        }
        let mut updated = self.clone();
        updated.bindings.insert(binding.identity.clone(), binding);
        Ok(updated)
    }

    /// Replace one active presented identity while preserving its principal.
    ///
    /// The old identity remains in the snapshot as `Revoked`; the replacement is
    /// active and may bind to a different executable agent instance.
    pub fn rotate_identity(
        &self,
        old_identity: &IdentityId,
        new_identity: IdentityId,
        new_agent_instance: AgentInstanceId,
    ) -> Result<Self, String> {
        if self.bindings.contains_key(&new_identity) {
            return Err(format!(
                "replacement identity already exists: {}",
                new_identity.as_str()
            ));
        }
        let old = self
            .bindings
            .get(old_identity)
            .ok_or_else(|| format!("unknown identity id: {}", old_identity.as_str()))?;
        if old.status != IdentityStatus::Active {
            return Err("only an active identity can be rotated".to_string());
        }

        let principal = old.principal.clone();
        let mut updated = self.clone();
        let old_mut = updated
            .bindings
            .get_mut(old_identity)
            .ok_or_else(|| "identity disappeared during rotation".to_string())?;
        old_mut.status = IdentityStatus::Revoked;
        updated.bindings.insert(
            new_identity.clone(),
            IdentityBinding::active(principal, new_identity, new_agent_instance),
        );
        Ok(updated)
    }

    fn bindings(&self) -> impl Iterator<Item = (&IdentityId, &IdentityBinding)> {
        self.bindings.iter()
    }
}

/// Explicit economic cost for creating additional presented identities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IdentityCostModel {
    pub per_added_identity: f64,
}

impl IdentityCostModel {
    pub fn new(per_added_identity: f64) -> Result<Self, String> {
        if !per_added_identity.is_finite() || per_added_identity < 0.0 {
            return Err("per-identity cost must be finite and non-negative".to_string());
        }
        Ok(Self {
            per_added_identity,
        })
    }

    pub fn cost_for(self, added_identities: usize) -> Result<f64, String> {
        let total = self.per_added_identity * added_identities as f64;
        if !total.is_finite() {
            return Err("identity cost overflowed finite f64 range".to_string());
        }
        Ok(total)
    }
}

/// Exact additive false-name counterfactual witness supplied by a mechanism adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct FalseNameDeviationWitness {
    pub principal: PrincipalId,
    pub added_bindings: Vec<IdentityBinding>,
    pub baseline_utility: f64,
    pub counterfactual_utility: f64,
    pub gross_gain: f64,
    pub identity_cost: f64,
    pub net_gain: f64,
    pub mechanism_profile_id: String,
    pub action_witness: String,
}

impl FalseNameDeviationWitness {
    pub fn is_profitable(&self) -> bool {
        self.net_gain > 0.0
    }
}

/// Validate an additive false-name counterfactual and compute its explicit gain/cost witness.
///
/// The generic layer does not compute the mechanism outcome. The caller must
/// supply finite baseline/counterfactual principal utility and an opaque exact
/// mechanism/action witness. Baseline identities must be preserved unchanged;
/// every newly added identity must be active and bound to `principal`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_false_name_counterfactual(
    baseline: &IdentityRegistry,
    counterfactual: &IdentityRegistry,
    principal: PrincipalId,
    cost_model: IdentityCostModel,
    baseline_utility: f64,
    counterfactual_utility: f64,
    mechanism_profile_id: impl Into<String>,
    action_witness: impl Into<String>,
) -> Result<FalseNameDeviationWitness, String> {
    if !baseline_utility.is_finite() || !counterfactual_utility.is_finite() {
        return Err("counterfactual utilities must be finite".to_string());
    }
    if baseline.active_identities_for_principal(&principal).is_empty() {
        return Err("principal must exist in the baseline identity registry".to_string());
    }

    for (identity, baseline_binding) in baseline.bindings() {
        let counterfactual_binding = counterfactual
            .binding(identity)
            .ok_or_else(|| "additive false-name counterfactual removed a baseline identity".to_string())?;
        if counterfactual_binding != baseline_binding {
            return Err("additive false-name counterfactual modified a baseline binding".to_string());
        }
    }

    let mut added_bindings = Vec::new();
    for (identity, binding) in counterfactual.bindings() {
        if baseline.binding(identity).is_none() {
            if binding.principal != principal {
                return Err("added false-name identity is bound to another principal".to_string());
            }
            if binding.status != IdentityStatus::Active {
                return Err("added false-name identity must be active".to_string());
            }
            added_bindings.push(binding.clone());
        }
    }
    if added_bindings.is_empty() {
        return Err("false-name counterfactual must add at least one identity".to_string());
    }

    let mechanism_profile_id = mechanism_profile_id.into();
    let action_witness = action_witness.into();
    if mechanism_profile_id.is_empty() || action_witness.is_empty() {
        return Err("mechanism profile id and action witness must be non-empty".to_string());
    }

    let gross_gain = counterfactual_utility - baseline_utility;
    if !gross_gain.is_finite() {
        return Err("gross false-name gain overflowed finite f64 range".to_string());
    }
    let identity_cost = cost_model.cost_for(added_bindings.len())?;
    let net_gain = gross_gain - identity_cost;
    if !net_gain.is_finite() {
        return Err("net false-name gain overflowed finite f64 range".to_string());
    }

    Ok(FalseNameDeviationWitness {
        principal,
        added_bindings,
        baseline_utility,
        counterfactual_utility,
        gross_gain,
        identity_cost,
        net_gain,
        mechanism_profile_id,
        action_witness,
    })
}

/// Coverage statement for a mechanism-specific bounded false-name search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalseNameSearchCoverage {
    ExhaustiveWithinDeclaredBound,
    Truncated,
}

/// Generic report envelope for mechanism-specific false-name enumeration.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedFalseNameReport {
    pub principal: PrincipalId,
    pub max_additional_identities: usize,
    pub candidates_evaluated: usize,
    pub coverage: FalseNameSearchCoverage,
    pub best_profitable_witness: Option<FalseNameDeviationWitness>,
}

impl BoundedFalseNameReport {
    pub fn new(
        principal: PrincipalId,
        max_additional_identities: usize,
        candidates_evaluated: usize,
        coverage: FalseNameSearchCoverage,
        best_profitable_witness: Option<FalseNameDeviationWitness>,
    ) -> Result<Self, String> {
        if max_additional_identities == 0 {
            return Err("false-name search bound must permit at least one added identity".to_string());
        }
        if let Some(witness) = &best_profitable_witness {
            if witness.principal != principal {
                return Err("false-name report witness principal mismatch".to_string());
            }
            if witness.added_bindings.len() > max_additional_identities {
                return Err("false-name witness exceeds declared identity bound".to_string());
            }
            if !witness.is_profitable() {
                return Err("best profitable witness must have positive net gain".to_string());
            }
        }
        Ok(Self {
            principal,
            max_additional_identities,
            candidates_evaluated,
            coverage,
            best_profitable_witness,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn principal(value: &str) -> PrincipalId {
        PrincipalId::new(value).unwrap()
    }

    fn identity(value: &str) -> IdentityId {
        IdentityId::new(value).unwrap()
    }

    fn agent(value: &str) -> AgentInstanceId {
        AgentInstanceId::new(value).unwrap()
    }

    fn binding(principal_id: &str, identity_id: &str, agent_id: &str) -> IdentityBinding {
        IdentityBinding::active(
            principal(principal_id),
            identity(identity_id),
            agent(agent_id),
        )
    }

    #[test]
    fn multiple_identities_can_remain_one_principal() {
        let registry = IdentityRegistry::new(vec![
            binding("p", "id-1", "agent-a"),
            binding("p", "id-2", "agent-b"),
        ])
        .unwrap();
        assert_eq!(registry.identity_count(), 2);
        assert_eq!(registry.principal_count(), 1);
        assert_eq!(registry.agent_instance_count(), 2);
    }

    #[test]
    fn shared_agent_instance_does_not_imply_independent_principals() {
        let registry = IdentityRegistry::new(vec![
            binding("p", "id-1", "same-agent"),
            binding("p", "id-2", "same-agent"),
        ])
        .unwrap();
        assert_eq!(registry.identity_count(), 2);
        assert_eq!(registry.agent_instance_count(), 1);
        assert_eq!(registry.principal_count(), 1);
    }

    #[test]
    fn identity_rotation_preserves_principal_and_revokes_old_identity() {
        let original = identity("old");
        let registry = IdentityRegistry::new(vec![IdentityBinding::active(
            principal("p"),
            original.clone(),
            agent("agent-a"),
        )])
        .unwrap();
        let replacement = identity("new");
        let rotated = registry
            .rotate_identity(&original, replacement.clone(), agent("agent-b"))
            .unwrap();

        assert_eq!(
            rotated.binding(&original).unwrap().status,
            IdentityStatus::Revoked
        );
        assert_eq!(
            rotated.binding(&replacement).unwrap().principal,
            principal("p")
        );
        assert_eq!(rotated.active_identity_count(), 1);
        assert_eq!(rotated.principal_count(), 1);
    }

    #[test]
    fn duplicate_identity_ids_fail_closed() {
        assert!(
            IdentityRegistry::new(vec![
                binding("p1", "same", "a1"),
                binding("p2", "same", "a2"),
            ])
            .is_err()
        );
    }

    #[test]
    fn false_name_additions_must_belong_to_declared_principal() {
        let base = IdentityRegistry::new(vec![binding("p", "base", "a")]).unwrap();
        let counterfactual = base
            .with_added_identity(binding("other", "fake", "b"))
            .unwrap();
        assert!(
            evaluate_false_name_counterfactual(
                &base,
                &counterfactual,
                principal("p"),
                IdentityCostModel::new(0.0).unwrap(),
                1.0,
                2.0,
                "mechanism-v1",
                "fake casts extra ballot",
            )
            .is_err()
        );
    }

    #[test]
    fn identity_cost_can_reverse_gross_profitability() {
        let p = principal("p");
        let base = IdentityRegistry::new(vec![IdentityBinding::active(
            p.clone(),
            identity("base"),
            agent("a"),
        )])
        .unwrap();
        let counterfactual = base
            .with_added_identity(IdentityBinding::active(
                p.clone(),
                identity("fake"),
                agent("a"),
            ))
            .unwrap();

        let witness = evaluate_false_name_counterfactual(
            &base,
            &counterfactual,
            p,
            IdentityCostModel::new(2.0).unwrap(),
            10.0,
            11.0,
            "mechanism-v1",
            "second identity changes allocation",
        )
        .unwrap();
        assert_eq!(witness.gross_gain, 1.0);
        assert_eq!(witness.identity_cost, 2.0);
        assert_eq!(witness.net_gain, -1.0);
        assert!(!witness.is_profitable());
    }

    #[test]
    fn invalid_costs_and_utilities_fail_closed() {
        assert!(IdentityCostModel::new(-1.0).is_err());
        assert!(IdentityCostModel::new(f64::NAN).is_err());

        let p = principal("p");
        let base = IdentityRegistry::new(vec![IdentityBinding::active(
            p.clone(),
            identity("base"),
            agent("a"),
        )])
        .unwrap();
        let counterfactual = base
            .with_added_identity(IdentityBinding::active(
                p.clone(),
                identity("fake"),
                agent("a"),
            ))
            .unwrap();
        assert!(
            evaluate_false_name_counterfactual(
                &base,
                &counterfactual,
                p,
                IdentityCostModel::new(0.0).unwrap(),
                f64::INFINITY,
                1.0,
                "mechanism-v1",
                "witness",
            )
            .is_err()
        );
    }

    #[test]
    fn truncated_search_report_is_explicitly_incomplete() {
        let report = BoundedFalseNameReport::new(
            principal("p"),
            4,
            10,
            FalseNameSearchCoverage::Truncated,
            None,
        )
        .unwrap();
        assert_eq!(report.coverage, FalseNameSearchCoverage::Truncated);
        assert!(report.best_profitable_witness.is_none());
    }

    #[test]
    fn profitable_report_retains_mechanism_action_witness() {
        let p = principal("p");
        let base = IdentityRegistry::new(vec![IdentityBinding::active(
            p.clone(),
            identity("base"),
            agent("a"),
        )])
        .unwrap();
        let counterfactual = base
            .with_added_identity(IdentityBinding::active(
                p.clone(),
                identity("fake"),
                agent("a"),
            ))
            .unwrap();
        let witness = evaluate_false_name_counterfactual(
            &base,
            &counterfactual,
            p.clone(),
            IdentityCostModel::new(0.1).unwrap(),
            1.0,
            2.0,
            "vote-rule-v7",
            "base=yes; fake=yes; allocation flips",
        )
        .unwrap();
        assert!(witness.is_profitable());
        assert_eq!(witness.mechanism_profile_id, "vote-rule-v7");
        assert_eq!(witness.action_witness, "base=yes; fake=yes; allocation flips");

        let report = BoundedFalseNameReport::new(
            p,
            2,
            8,
            FalseNameSearchCoverage::ExhaustiveWithinDeclaredBound,
            Some(witness),
        )
        .unwrap();
        assert!(report.best_profitable_witness.unwrap().is_profitable());
    }

    #[test]
    fn report_rejects_non_profitable_or_out_of_bound_witnesses() {
        let p = principal("p");
        let base = IdentityRegistry::new(vec![IdentityBinding::active(
            p.clone(),
            identity("base"),
            agent("a"),
        )])
        .unwrap();
        let counterfactual = base
            .with_added_identity(IdentityBinding::active(
                p.clone(),
                identity("fake"),
                agent("a"),
            ))
            .unwrap();
        let non_profitable = evaluate_false_name_counterfactual(
            &base,
            &counterfactual,
            p.clone(),
            IdentityCostModel::new(2.0).unwrap(),
            1.0,
            2.0,
            "mechanism",
            "witness",
        )
        .unwrap();
        assert!(
            BoundedFalseNameReport::new(
                p,
                2,
                1,
                FalseNameSearchCoverage::ExhaustiveWithinDeclaredBound,
                Some(non_profitable),
            )
            .is_err()
        );
    }
}
