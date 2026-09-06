// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit source- and generation-qualified clock-domain types for spatial state.

use std::cmp::Ordering;
use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

use crate::SpatialValidationError;

/// Stable non-zero namespace for source-local clock-domain identities.
///
/// Independent sensors, simulators, processes, or external systems may reuse the
/// same local clock-domain identifier. The namespace prevents those unrelated
/// time bases from becoming comparable by accidental numeric collision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ClockNamespaceId(NonZeroU64);

impl ClockNamespaceId {
    /// Construct a non-zero clock namespace.
    pub fn new(value: u64) -> Result<Self, SpatialValidationError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(SpatialValidationError::ZeroId {
                kind: "clock-namespace",
            })
    }

    /// Return the numeric namespace identity.
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Source- and generation-qualified stable identity for one clock domain.
///
/// `generation` must advance whenever a clock is reset, wraps, changes epoch,
/// changes tick semantics/rate, or a source-local clock ID is reused. Numeric
/// ticks are comparable only when this complete identity is exactly equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClockDomainId {
    namespace: ClockNamespaceId,
    local_id: NonZeroU64,
    generation: NonZeroU64,
}

impl ClockDomainId {
    /// Construct a source- and generation-qualified clock-domain identity.
    pub fn new(
        namespace: ClockNamespaceId,
        local_id: u64,
        generation: u64,
    ) -> Result<Self, SpatialValidationError> {
        let local_id = NonZeroU64::new(local_id).ok_or(SpatialValidationError::ZeroId {
            kind: "clock-domain-local",
        })?;
        let generation = NonZeroU64::new(generation).ok_or(SpatialValidationError::ZeroId {
            kind: "clock-domain-generation",
        })?;
        Ok(Self {
            namespace,
            local_id,
            generation,
        })
    }

    /// Namespace that owns the source-local clock-domain identity.
    pub const fn namespace(self) -> ClockNamespaceId {
        self.namespace
    }

    /// Source-local clock-domain identifier.
    pub const fn local_id(self) -> u64 {
        self.local_id.get()
    }

    /// Semantic/reset generation of this clock domain.
    pub const fn generation(self) -> u64 {
        self.generation.get()
    }
}

/// Tick value qualified by the complete clock domain that gives it meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClockInstant {
    domain: ClockDomainId,
    tick: u64,
}

impl ClockInstant {
    /// Construct a clock-domain-qualified instant.
    pub const fn new(domain: ClockDomainId, tick: u64) -> Self {
        Self { domain, tick }
    }

    /// Complete source- and generation-qualified clock domain owning this tick.
    pub const fn domain(self) -> ClockDomainId {
        self.domain
    }

    /// Raw tick in the owning clock domain.
    pub const fn tick(self) -> u64 {
        self.tick
    }

    /// Compare two instants only when their complete clock domains are identical.
    ///
    /// Returns `None` across namespaces, local IDs, or generations. Cross-domain
    /// ordering requires a separately qualified synchronization transform.
    pub fn cmp_same_domain(self, other: Self) -> Option<Ordering> {
        (self.domain == other.domain).then_some(self.tick.cmp(&other.tick))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn namespace(value: u64) -> ClockNamespaceId {
        ClockNamespaceId::new(value).unwrap()
    }

    fn domain(namespace_id: u64, local_id: u64, generation: u64) -> ClockDomainId {
        ClockDomainId::new(namespace(namespace_id), local_id, generation).unwrap()
    }

    #[test]
    fn same_complete_clock_domain_ticks_are_comparable() {
        let domain = domain(1, 7, 1);
        let a = ClockInstant::new(domain, 10);
        let b = ClockInstant::new(domain, 11);
        assert_eq!(a.cmp_same_domain(b), Some(Ordering::Less));
    }

    #[test]
    fn same_local_clock_id_in_different_namespaces_is_not_comparable() {
        let a = ClockInstant::new(domain(1, 7, 1), 10);
        let b = ClockInstant::new(domain(2, 7, 1), 11);
        assert_eq!(a.cmp_same_domain(b), None);
    }

    #[test]
    fn clock_reset_generation_breaks_implicit_continuity() {
        let before_reset = ClockInstant::new(domain(1, 7, 1), u64::MAX);
        let after_reset = ClockInstant::new(domain(1, 7, 2), 0);
        assert_eq!(before_reset.cmp_same_domain(after_reset), None);
    }

    #[test]
    fn different_local_clock_domains_have_no_implicit_order() {
        let a = ClockInstant::new(domain(1, 1, 1), u64::MAX);
        let b = ClockInstant::new(domain(1, 2, 1), 0);
        assert_eq!(a.cmp_same_domain(b), None);
    }

    #[test]
    fn zero_clock_namespace_local_id_and_generation_are_rejected() {
        assert!(ClockNamespaceId::new(0).is_err());
        assert!(ClockDomainId::new(namespace(1), 0, 1).is_err());
        assert!(ClockDomainId::new(namespace(1), 1, 0).is_err());
    }
}
