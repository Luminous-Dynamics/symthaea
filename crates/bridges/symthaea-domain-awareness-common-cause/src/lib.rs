// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Monotonic common-cause diversity gate for domain-awareness track assurance.
//!
//! This bridge can only retain or reduce an existing assurance level. It cannot
//! manufacture corroboration, identity, risk, intent, or physical authority.

#![deny(unsafe_code)]

use symthaea_domain_awareness_vision::{TrackAssuranceLevel, TrackAssuranceReport};
use symthaea_sensor_common_cause::CommonCauseDiversityReport;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommonCauseTrackIssue {
    CommonCauseRequiresFailClosed,
    CommonCauseReportContainsIssues,
    UnprofiledAcceptedSources {
        accepted: usize,
        profiled: usize,
    },
    PhysicalSourceCountMismatch {
        track_report: usize,
        common_cause_report: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommonCauseQualifiedTrackAssurance {
    pub original_level: TrackAssuranceLevel,
    pub qualified_level: TrackAssuranceLevel,
    pub physical_sources_reported_by_track: usize,
    pub physical_sources_checked_for_common_cause: usize,
    pub issues: Vec<CommonCauseTrackIssue>,
}

impl CommonCauseQualifiedTrackAssurance {
    pub fn was_downgraded(&self) -> bool {
        self.qualified_level < self.original_level
    }

    /// Assurance composition can never grant physical authority.
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Qualify an existing track-assurance result with common-cause diversity.
///
/// Levels below `Corroborated` are left unchanged because this bridge has no
/// evidence with which to improve them. Levels at or above `Corroborated` require
/// a structurally clean, non-fail-closed common-cause result, complete profiling,
/// and an exact physical-source-count match. Otherwise they are conservatively
/// reduced to `Persistent`.
pub fn qualify_track_assurance(
    track: &TrackAssuranceReport,
    common_cause: &CommonCauseDiversityReport,
) -> CommonCauseQualifiedTrackAssurance {
    let mut issues = Vec::new();
    if common_cause.requires_fail_closed {
        issues.push(CommonCauseTrackIssue::CommonCauseRequiresFailClosed);
    }
    if !common_cause.issues.is_empty() {
        issues.push(CommonCauseTrackIssue::CommonCauseReportContainsIssues);
    }
    if common_cause.profiled_physical_sources != common_cause.accepted_physical_sources {
        issues.push(CommonCauseTrackIssue::UnprofiledAcceptedSources {
            accepted: common_cause.accepted_physical_sources,
            profiled: common_cause.profiled_physical_sources,
        });
    }
    if track.independent_physical_sources != common_cause.accepted_physical_sources {
        issues.push(CommonCauseTrackIssue::PhysicalSourceCountMismatch {
            track_report: track.independent_physical_sources,
            common_cause_report: common_cause.accepted_physical_sources,
        });
    }

    let corroboration_supported = issues.is_empty();
    let qualified_level = if track.level >= TrackAssuranceLevel::Corroborated
        && !corroboration_supported
    {
        TrackAssuranceLevel::Persistent
    } else {
        track.level
    };

    CommonCauseQualifiedTrackAssurance {
        original_level: track.level,
        qualified_level,
        physical_sources_reported_by_track: track.independent_physical_sources,
        physical_sources_checked_for_common_cause: common_cause.accepted_physical_sources,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use symthaea_domain_awareness_vision::{TrackAssuranceReason, TrackAssuranceReport};
    use symthaea_sensor_common_cause::{
        CommonCauseDiversityReport, CommonCauseIssue, FaultDomainKind,
    };

    fn track(level: TrackAssuranceLevel, physical_sources: usize) -> TrackAssuranceReport {
        TrackAssuranceReport {
            level,
            fresh_usable_observations: 10,
            independent_physical_sources: physical_sources,
            independent_modalities: 2,
            evidence_span_ms: 1_000,
            reasons: Vec::<TrackAssuranceReason>::new(),
        }
    }

    fn diversity(physical_sources: usize, fail_closed: bool) -> CommonCauseDiversityReport {
        CommonCauseDiversityReport {
            policy_id: "diversity-v1".into(),
            accepted_physical_sources: physical_sources,
            profiled_physical_sources: physical_sources,
            distinct_domains: BTreeMap::from([
                (FaultDomainKind::Power, physical_sources),
                (FaultDomainKind::Network, physical_sources),
            ]),
            issues: Vec::new(),
            requires_fail_closed: fail_closed,
        }
    }

    #[test]
    fn diverse_corroborated_track_is_preserved() {
        let result = qualify_track_assurance(
            &track(TrackAssuranceLevel::Corroborated, 2),
            &diversity(2, false),
        );
        assert_eq!(result.qualified_level, TrackAssuranceLevel::Corroborated);
        assert!(!result.was_downgraded());
        assert!(!result.grants_physical_authority());
    }

    #[test]
    fn common_cause_failure_downgrades_corroboration_to_persistent() {
        let result = qualify_track_assurance(
            &track(TrackAssuranceLevel::Corroborated, 2),
            &diversity(2, true),
        );
        assert_eq!(result.qualified_level, TrackAssuranceLevel::Persistent);
        assert!(result.was_downgraded());
        assert!(result
            .issues
            .contains(&CommonCauseTrackIssue::CommonCauseRequiresFailClosed));
    }

    #[test]
    fn common_cause_failure_removes_identity_supported_assurance() {
        let result = qualify_track_assurance(
            &track(TrackAssuranceLevel::IdentityEvidenceSupported, 2),
            &diversity(2, true),
        );
        assert_eq!(result.qualified_level, TrackAssuranceLevel::Persistent);
    }

    #[test]
    fn source_count_mismatch_downgrades_even_when_diversity_report_claims_usable() {
        let result = qualify_track_assurance(
            &track(TrackAssuranceLevel::Corroborated, 3),
            &diversity(2, false),
        );
        assert_eq!(result.qualified_level, TrackAssuranceLevel::Persistent);
        assert!(result.issues.contains(
            &CommonCauseTrackIssue::PhysicalSourceCountMismatch {
                track_report: 3,
                common_cause_report: 2,
            }
        ));
    }

    #[test]
    fn hidden_issues_cannot_be_laundered_by_false_fail_closed_flag() {
        let mut malformed = diversity(2, false);
        malformed.issues.push(CommonCauseIssue::MissingProfile("camera-b".into()));
        let result = qualify_track_assurance(
            &track(TrackAssuranceLevel::Corroborated, 2),
            &malformed,
        );
        assert_eq!(result.qualified_level, TrackAssuranceLevel::Persistent);
        assert!(result
            .issues
            .contains(&CommonCauseTrackIssue::CommonCauseReportContainsIssues));
    }

    #[test]
    fn incomplete_profiling_downgrades_corroboration() {
        let mut malformed = diversity(2, false);
        malformed.profiled_physical_sources = 1;
        let result = qualify_track_assurance(
            &track(TrackAssuranceLevel::Corroborated, 2),
            &malformed,
        );
        assert_eq!(result.qualified_level, TrackAssuranceLevel::Persistent);
        assert!(result.issues.contains(
            &CommonCauseTrackIssue::UnprofiledAcceptedSources {
                accepted: 2,
                profiled: 1,
            }
        ));
    }

    #[test]
    fn bridge_never_upgrades_persistent_track() {
        let result = qualify_track_assurance(
            &track(TrackAssuranceLevel::Persistent, 2),
            &diversity(2, false),
        );
        assert_eq!(result.qualified_level, TrackAssuranceLevel::Persistent);
        assert!(!result.was_downgraded());
    }

    #[test]
    fn tentative_track_remains_tentative_even_with_diverse_sensors() {
        let result = qualify_track_assurance(
            &track(TrackAssuranceLevel::Tentative, 2),
            &diversity(2, false),
        );
        assert_eq!(result.qualified_level, TrackAssuranceLevel::Tentative);
    }
}
