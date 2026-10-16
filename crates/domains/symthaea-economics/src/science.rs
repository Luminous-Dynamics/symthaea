// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scientific classification primitives for economic statements.
//!
//! The central contract is that constraint, empirical, and normative claims
//! are different kinds of statements. No API in this module promotes one kind
//! into another.

/// Version of the economic-science constitutional boundary implemented by this
/// crate.
pub const ECONOMIC_SCIENCE_CONSTITUTION_VERSION: u16 = 1;

/// The epistemic kind of an economic statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatementKind {
    /// Accounting identity, conservation relation, or other declared hard
    /// model constraint.
    Constraint,
    /// Falsifiable proposition about the observed world.
    Empirical,
    /// Value judgment or objective supplied by people/governance.
    Normative,
}

/// Orthogonal channels through which a later qualified evaluator may assess an
/// empirical claim.
///
/// These values are deliberately not orderable and are not evidence authority.
/// A claim may have strong prospective evidence and weak mechanistic evidence,
/// or vice versa; neither channel silently dominates the other. `EmpiricalClaim`
/// itself cannot self-assign any of these channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceChannel {
    /// Mechanism-level evidence connecting the proposed causal process to
    /// measurable intermediate consequences.
    Mechanistic,
    /// Evaluation on historical observations available only after the modeled
    /// episode occurred.
    Retrospective,
    /// Evaluation on data excluded from fitting/calibration.
    OutOfSample,
    /// Prediction committed before the evaluated outcome became available.
    Prospective,
    /// Evidence from a controlled or otherwise explicitly identified
    /// intervention.
    Interventional,
    /// Independent reproduction or replication under a distinct evidence
    /// lineage.
    IndependentReplication,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scientific_statement_kinds_remain_distinct() {
        assert_ne!(StatementKind::Constraint, StatementKind::Empirical);
        assert_ne!(StatementKind::Empirical, StatementKind::Normative);
        assert_ne!(StatementKind::Constraint, StatementKind::Normative);
    }

    #[test]
    fn evidence_channels_are_distinct_dimensions_not_a_scalar_order() {
        let prospective = EvidenceChannel::Prospective;
        let interventional = EvidenceChannel::Interventional;
        let replication = EvidenceChannel::IndependentReplication;
        assert_ne!(prospective, interventional);
        assert_ne!(prospective, replication);
        assert_ne!(interventional, replication);
    }
}
