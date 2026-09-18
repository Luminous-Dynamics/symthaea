// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Independent authority dimensions. They deliberately do not collapse into
/// one scalar confidence value.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum AuthorityFacet {
    Provenance,
    Execution,
    Empirical,
    Causal,
    Formal,
    Replication,
    Independence,
}

impl AuthorityFacet {
    pub const ALL: [Self; 7] = [
        Self::Provenance,
        Self::Execution,
        Self::Empirical,
        Self::Causal,
        Self::Formal,
        Self::Replication,
        Self::Independence,
    ];
}

/// Authority available to ordinary, caller-constructible research records.
///
/// `Qualified` is deliberately absent. Qualification is a capability boundary,
/// not another value a caller may select. A later qualification layer must wrap
/// these records in a non-forgeable type after authenticating the exact
/// qualification lineage.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum AuthorityLevel {
    None,
    Declared,
    Bound,
}

/// Sparse multidimensional authority profile for ordinary research records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AuthorityProfile {
    levels: BTreeMap<AuthorityFacet, AuthorityLevel>,
}

impl AuthorityProfile {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn get(&self, facet: AuthorityFacet) -> AuthorityLevel {
        self.levels
            .get(&facet)
            .copied()
            .unwrap_or(AuthorityLevel::None)
    }

    pub fn with(mut self, facet: AuthorityFacet, level: AuthorityLevel) -> Self {
        if level == AuthorityLevel::None {
            self.levels.remove(&facet);
        } else {
            self.levels.insert(facet, level);
        }
        self
    }

    pub fn is_empty(&self) -> bool {
        AuthorityFacet::ALL
            .into_iter()
            .all(|facet| self.get(facet) == AuthorityLevel::None)
    }

    /// True when every facet in `self` is less than or equal to `ceiling`.
    pub fn is_within(&self, ceiling: &Self) -> bool {
        AuthorityFacet::ALL
            .into_iter()
            .all(|facet| self.get(facet) <= ceiling.get(facet))
    }

    /// Conservative adapter operation: clamp every facet to a declared ceiling.
    pub fn bounded_by(&self, ceiling: &Self) -> Self {
        let mut out = Self::empty();
        for facet in AuthorityFacet::ALL {
            let level = self.get(facet).min(ceiling.get(facet));
            out = out.with(facet, level);
        }
        out
    }

    /// Explicit evidence composition: retain the strongest authority actually
    /// present for each independent facet. This does not infer replication or
    /// independence from evidence count; those facets must exist explicitly.
    pub fn evidence_union(&self, other: &Self) -> Self {
        let mut out = Self::empty();
        for facet in AuthorityFacet::ALL {
            let level = self.get(facet).max(other.get(facet));
            out = out.with(facet, level);
        }
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceState {
    Pass,
    Fail,
    Unknown,
    Incomplete,
    NotApplicable,
    Invalid,
}

impl EvidenceState {
    pub const fn permits_positive_authority(self) -> bool {
        matches!(self, Self::Pass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_facets_are_none() {
        let profile = AuthorityProfile::empty();
        for facet in AuthorityFacet::ALL {
            assert_eq!(profile.get(facet), AuthorityLevel::None);
        }
    }

    #[test]
    fn bounded_adapter_cannot_invent_missing_facets() {
        let source = AuthorityProfile::empty()
            .with(AuthorityFacet::Provenance, AuthorityLevel::Bound)
            .with(AuthorityFacet::Empirical, AuthorityLevel::Declared);
        let ceiling = AuthorityProfile::empty()
            .with(AuthorityFacet::Provenance, AuthorityLevel::Declared)
            .with(AuthorityFacet::Empirical, AuthorityLevel::Bound)
            .with(AuthorityFacet::Causal, AuthorityLevel::Bound);

        let adapted = source.bounded_by(&ceiling);
        assert_eq!(adapted.get(AuthorityFacet::Provenance), AuthorityLevel::Declared);
        assert_eq!(adapted.get(AuthorityFacet::Empirical), AuthorityLevel::Declared);
        assert_eq!(adapted.get(AuthorityFacet::Causal), AuthorityLevel::None);
    }

    #[test]
    fn only_pass_can_carry_positive_authority() {
        assert!(EvidenceState::Pass.permits_positive_authority());
        for state in [
            EvidenceState::Fail,
            EvidenceState::Unknown,
            EvidenceState::Incomplete,
            EvidenceState::NotApplicable,
            EvidenceState::Invalid,
        ] {
            assert!(!state.permits_positive_authority());
        }
    }
}
