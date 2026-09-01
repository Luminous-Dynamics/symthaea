// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed assurance-profile façades for canonical CogSec shadow scenarios.
//!
//! Early observer-only qualification must not claim protected-owner freshness.
//! Later owner-aware qualification may require owner-issued `ResourceVersion`,
//! but that stronger requirement must be explicit in the type used by the
//! scenario driver.

use crate::{ScenarioContract, feedback_input_v0, goal_no_eviction_v0, goal_with_eviction_v0};

/// Canonical scenario contract for observer-only shadow mode.
///
/// This profile guarantees that the scenario does **not** claim owner-issued
/// `ResourceVersion` coverage. Event-count expectations, required event kinds,
/// and the P0 denominator remain unchanged from the underlying scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObserverOnlyScenario(ScenarioContract);

impl ObserverOnlyScenario {
    fn new(mut contract: ScenarioContract) -> Self {
        contract.expected_manifest.require_resource_versions = false;
        Self(contract)
    }

    /// Borrow the independent scenario contract.
    pub fn contract(&self) -> &ScenarioContract {
        &self.0
    }

    /// Consume the profile wrapper and return the independent scenario contract.
    pub fn into_contract(self) -> ScenarioContract {
        self.0
    }
}

/// Canonical scenario contract for an owner-aware shadow/enforcement profile.
///
/// This profile requires owner-issued `ResourceVersion` evidence while keeping
/// the event-count expectations, required event kinds, and P0 denominator
/// identical to the corresponding observer-only scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnerAwareScenario(ScenarioContract);

impl OwnerAwareScenario {
    fn new(mut contract: ScenarioContract) -> Self {
        contract.expected_manifest.require_resource_versions = true;
        Self(contract)
    }

    /// Borrow the independent scenario contract.
    pub fn contract(&self) -> &ScenarioContract {
        &self.0
    }

    /// Consume the profile wrapper and return the independent scenario contract.
    pub fn into_contract(self) -> ScenarioContract {
        self.0
    }
}

/// Observer-only S0: one goal input while working memory has spare capacity.
pub fn observer_goal_no_eviction_v0() -> ObserverOnlyScenario {
    ObserverOnlyScenario::new(goal_no_eviction_v0())
}

/// Observer-only S1: one goal input that forces working-memory eviction.
pub fn observer_goal_with_eviction_v0() -> ObserverOnlyScenario {
    ObserverOnlyScenario::new(goal_with_eviction_v0())
}

/// Observer-only S2: one feedback input.
pub fn observer_feedback_input_v0() -> ObserverOnlyScenario {
    ObserverOnlyScenario::new(feedback_input_v0())
}

/// Owner-aware S0: one goal input while working memory has spare capacity.
pub fn owner_aware_goal_no_eviction_v0() -> OwnerAwareScenario {
    OwnerAwareScenario::new(goal_no_eviction_v0())
}

/// Owner-aware S1: one goal input that forces working-memory eviction.
pub fn owner_aware_goal_with_eviction_v0() -> OwnerAwareScenario {
    OwnerAwareScenario::new(goal_with_eviction_v0())
}

/// Owner-aware S2: one feedback input.
pub fn owner_aware_feedback_input_v0() -> OwnerAwareScenario {
    OwnerAwareScenario::new(feedback_input_v0())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_only_version_requirement_differs(
        observer: &ScenarioContract,
        owner_aware: &ScenarioContract,
    ) {
        assert_eq!(observer.scenario_id, owner_aware.scenario_id);
        assert_eq!(observer.expectations, owner_aware.expectations);
        assert_eq!(
            observer.expected_manifest.required_kinds,
            owner_aware.expected_manifest.required_kinds
        );
        assert_eq!(
            observer.expected_manifest.p0_observed_kinds,
            owner_aware.expected_manifest.p0_observed_kinds
        );
        assert!(!observer.expected_manifest.require_resource_versions);
        assert!(owner_aware.expected_manifest.require_resource_versions);
    }

    #[test]
    fn observer_only_canonical_profiles_never_claim_owner_versions() {
        for scenario in [
            observer_goal_no_eviction_v0().into_contract(),
            observer_goal_with_eviction_v0().into_contract(),
            observer_feedback_input_v0().into_contract(),
        ] {
            assert!(!scenario.expected_manifest.require_resource_versions);
        }
    }

    #[test]
    fn owner_aware_canonical_profiles_require_owner_versions() {
        for scenario in [
            owner_aware_goal_no_eviction_v0().into_contract(),
            owner_aware_goal_with_eviction_v0().into_contract(),
            owner_aware_feedback_input_v0().into_contract(),
        ] {
            assert!(scenario.expected_manifest.require_resource_versions);
        }
    }

    #[test]
    fn profile_conversion_does_not_change_counts_required_kinds_or_p0_denominator() {
        let observer = observer_goal_no_eviction_v0().into_contract();
        let owner_aware = owner_aware_goal_no_eviction_v0().into_contract();
        assert_only_version_requirement_differs(&observer, &owner_aware);

        let observer = observer_goal_with_eviction_v0().into_contract();
        let owner_aware = owner_aware_goal_with_eviction_v0().into_contract();
        assert_only_version_requirement_differs(&observer, &owner_aware);

        let observer = observer_feedback_input_v0().into_contract();
        let owner_aware = owner_aware_feedback_input_v0().into_contract();
        assert_only_version_requirement_differs(&observer, &owner_aware);
    }
}
