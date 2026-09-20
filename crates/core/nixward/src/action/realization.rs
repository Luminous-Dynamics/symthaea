// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed admission between abstract planner actions and later command construction.
//!
//! `Admitted` means only semantic compatibility under this profile. It does not
//! mean command construction, authorization, execution, or verified post-state.

use crate::mind::ActionCategory;
use crate::observe::UnitInfo;

pub const ACTION_REALIZATION_PROFILE_V1: &str = "nixward-action-realizer-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetEvidenceSourceV1 {
    ExplicitSystem,
    SystemdInventory,
    Journal,
    PredictiveMonitor,
    UserInput,
    PackageInventory,
    NixOptionSchema,
    GenerationInventory,
    FlakeMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NixActionTargetV1 {
    System,
    SystemdUnit(String),
    JournalProducer(String),
    Metric(String),
    Package(String),
    NixOption(String),
    FlakeInput(String),
    SystemGeneration(u64),
    UserSupplied(String),
    Unknown(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObservationWindowV1 {
    observed_at_unix_secs: u64,
    valid_until_unix_secs: u64,
}

impl ObservationWindowV1 {
    pub fn new(
        observed_at_unix_secs: u64,
        valid_until_unix_secs: u64,
    ) -> Result<Self, TargetEvidenceErrorV1> {
        if valid_until_unix_secs < observed_at_unix_secs {
            return Err(TargetEvidenceErrorV1::InvalidObservationWindow);
        }
        Ok(Self {
            observed_at_unix_secs,
            valid_until_unix_secs,
        })
    }

    pub fn observed_at_unix_secs(&self) -> u64 {
        self.observed_at_unix_secs
    }

    pub fn valid_until_unix_secs(&self) -> u64 {
        self.valid_until_unix_secs
    }

    pub fn is_current_at(&self, now_unix_secs: u64) -> bool {
        self.observed_at_unix_secs <= now_unix_secs
            && now_unix_secs <= self.valid_until_unix_secs
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TargetCurrentnessV1 {
    Static,
    Observed(ObservationWindowV1),
    Unknown,
}

impl TargetCurrentnessV1 {
    pub fn is_current_at(&self, now_unix_secs: u64) -> bool {
        match self {
            Self::Static => true,
            Self::Observed(window) => window.is_current_at(now_unix_secs),
            Self::Unknown => false,
        }
    }
}

/// Source-typed target evidence. Fields are private so callers cannot relabel a
/// journal producer, metric, or user string as a systemd unit after the fact.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TargetEvidenceV1 {
    target: NixActionTargetV1,
    source: TargetEvidenceSourceV1,
    evidence_ref: Option<String>,
    currentness: TargetCurrentnessV1,
}

impl TargetEvidenceV1 {
    /// Explicit whole-system semantic target. This grants no authority.
    pub fn system() -> Self {
        Self {
            target: NixActionTargetV1::System,
            source: TargetEvidenceSourceV1::ExplicitSystem,
            evidence_ref: None,
            currentness: TargetCurrentnessV1::Static,
        }
    }

    /// Exact service identity sourced from structured systemd inventory.
    pub fn from_systemd_inventory(
        unit: &UnitInfo,
        evidence_ref: impl Into<String>,
        window: ObservationWindowV1,
    ) -> Result<Self, TargetEvidenceErrorV1> {
        if !is_exact_service_unit_identity(&unit.name) {
            return Err(TargetEvidenceErrorV1::InvalidSystemdUnit);
        }
        Ok(Self {
            target: NixActionTargetV1::SystemdUnit(unit.name.clone()),
            source: TargetEvidenceSourceV1::SystemdInventory,
            evidence_ref: Some(require_evidence_ref(evidence_ref)?),
            currentness: TargetCurrentnessV1::Observed(window),
        })
    }

    /// Journal producer identity; never promoted to SystemdUnit by name shape.
    pub fn journal_producer(
        producer: impl Into<String>,
        evidence_ref: impl Into<String>,
        window: ObservationWindowV1,
    ) -> Result<Self, TargetEvidenceErrorV1> {
        Ok(Self {
            target: NixActionTargetV1::JournalProducer(require_nonempty(producer)?),
            source: TargetEvidenceSourceV1::Journal,
            evidence_ref: Some(require_evidence_ref(evidence_ref)?),
            currentness: TargetCurrentnessV1::Observed(window),
        })
    }

    pub fn metric(
        metric: impl Into<String>,
        evidence_ref: impl Into<String>,
        window: ObservationWindowV1,
    ) -> Result<Self, TargetEvidenceErrorV1> {
        Ok(Self {
            target: NixActionTargetV1::Metric(require_nonempty(metric)?),
            source: TargetEvidenceSourceV1::PredictiveMonitor,
            evidence_ref: Some(require_evidence_ref(evidence_ref)?),
            currentness: TargetCurrentnessV1::Observed(window),
        })
    }

    pub fn user_supplied(display: impl Into<String>) -> Result<Self, TargetEvidenceErrorV1> {
        Ok(Self {
            target: NixActionTargetV1::UserSupplied(require_nonempty(display)?),
            source: TargetEvidenceSourceV1::UserInput,
            evidence_ref: None,
            currentness: TargetCurrentnessV1::Static,
        })
    }

    pub fn package(
        package: impl Into<String>,
        evidence_ref: impl Into<String>,
        window: ObservationWindowV1,
    ) -> Result<Self, TargetEvidenceErrorV1> {
        Ok(Self {
            target: NixActionTargetV1::Package(require_nonempty(package)?),
            source: TargetEvidenceSourceV1::PackageInventory,
            evidence_ref: Some(require_evidence_ref(evidence_ref)?),
            currentness: TargetCurrentnessV1::Observed(window),
        })
    }

    pub fn nix_option(
        option: impl Into<String>,
        evidence_ref: impl Into<String>,
    ) -> Result<Self, TargetEvidenceErrorV1> {
        Ok(Self {
            target: NixActionTargetV1::NixOption(require_nonempty(option)?),
            source: TargetEvidenceSourceV1::NixOptionSchema,
            evidence_ref: Some(require_evidence_ref(evidence_ref)?),
            currentness: TargetCurrentnessV1::Static,
        })
    }

    pub fn system_generation(
        generation: u64,
        evidence_ref: impl Into<String>,
        window: ObservationWindowV1,
    ) -> Result<Self, TargetEvidenceErrorV1> {
        if generation == 0 {
            return Err(TargetEvidenceErrorV1::InvalidGeneration);
        }
        Ok(Self {
            target: NixActionTargetV1::SystemGeneration(generation),
            source: TargetEvidenceSourceV1::GenerationInventory,
            evidence_ref: Some(require_evidence_ref(evidence_ref)?),
            currentness: TargetCurrentnessV1::Observed(window),
        })
    }

    pub fn unknown(display: impl Into<String>) -> Result<Self, TargetEvidenceErrorV1> {
        Ok(Self {
            target: NixActionTargetV1::Unknown(require_nonempty(display)?),
            source: TargetEvidenceSourceV1::UserInput,
            evidence_ref: None,
            currentness: TargetCurrentnessV1::Unknown,
        })
    }

    pub fn target(&self) -> &NixActionTargetV1 {
        &self.target
    }

    pub fn source(&self) -> TargetEvidenceSourceV1 {
        self.source
    }

    pub fn evidence_ref(&self) -> Option<&str> {
        self.evidence_ref.as_deref()
    }

    pub fn currentness(&self) -> &TargetCurrentnessV1 {
        &self.currentness
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetEvidenceErrorV1 {
    EmptyValue,
    EmptyEvidenceRef,
    InvalidObservationWindow,
    InvalidSystemdUnit,
    InvalidGeneration,
}

fn require_nonempty(value: impl Into<String>) -> Result<String, TargetEvidenceErrorV1> {
    let value = value.into();
    if value.trim().is_empty() {
        Err(TargetEvidenceErrorV1::EmptyValue)
    } else {
        Ok(value)
    }
}

fn require_evidence_ref(value: impl Into<String>) -> Result<String, TargetEvidenceErrorV1> {
    let value = value.into();
    if value.trim().is_empty() {
        Err(TargetEvidenceErrorV1::EmptyEvidenceRef)
    } else {
        Ok(value)
    }
}

fn is_exact_service_unit_identity(unit: &str) -> bool {
    if unit.is_empty()
        || unit.len() > 255
        || !unit.ends_with(".service")
        || unit.starts_with('-')
        || unit.contains('/')
        || unit.chars().any(char::is_whitespace)
        || unit.chars().any(|c| matches!(c, '*' | '?' | '[' | ']'))
    {
        return false;
    }

    unit.chars().all(|c| {
        c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '@' | ':' | '\\')
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RealizationDispositionV1 {
    Admitted,
    NeedsResolution,
    NeedsParameters,
    Inapplicable,
    Unsupported,
    StaleEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RealizationReasonV1 {
    ExactSystemdUnitForLifecycle,
    ExplicitSystemForGarbageCollection,
    ExplicitSystemForRebuild,
    JournalProducerRequiresInventoryResolution,
    UserTargetRequiresResolution,
    MetricIsNotServiceTarget,
    PackageOperationRequiresParameters,
    ConfigureRequiresPatchPlan,
    RollbackRequiresQualifiedProfile,
    UpdateRequiresExplicitProfile,
    TargetKindIncompatible,
    CustomActionUnsupported,
    EvidenceStale,
    EvidenceCurrentnessUnknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealizationOutcomeV1 {
    action: ActionCategory,
    target: TargetEvidenceV1,
    disposition: RealizationDispositionV1,
    reason: RealizationReasonV1,
    profile: &'static str,
}

impl RealizationOutcomeV1 {
    pub fn action(&self) -> &ActionCategory {
        &self.action
    }

    pub fn target(&self) -> &TargetEvidenceV1 {
        &self.target
    }

    pub fn disposition(&self) -> RealizationDispositionV1 {
        self.disposition
    }

    pub fn reason(&self) -> RealizationReasonV1 {
        self.reason
    }

    pub fn profile(&self) -> &'static str {
        self.profile
    }

    /// Semantic admission only; not command construction or authorization.
    pub fn is_admitted(&self) -> bool {
        self.disposition == RealizationDispositionV1::Admitted
    }
}

pub struct ActionRealizerV1;

impl ActionRealizerV1 {
    pub fn assess(
        action: ActionCategory,
        target: TargetEvidenceV1,
        now_unix_secs: u64,
    ) -> RealizationOutcomeV1 {
        if matches!(&action, ActionCategory::Custom(_)) {
            return outcome(
                action,
                target,
                RealizationDispositionV1::Unsupported,
                RealizationReasonV1::CustomActionUnsupported,
            );
        }

        if matches!(target.currentness(), TargetCurrentnessV1::Unknown) {
            return outcome(
                action,
                target,
                RealizationDispositionV1::NeedsResolution,
                RealizationReasonV1::EvidenceCurrentnessUnknown,
            );
        }

        if !target.currentness().is_current_at(now_unix_secs) {
            return outcome(
                action,
                target,
                RealizationDispositionV1::StaleEvidence,
                RealizationReasonV1::EvidenceStale,
            );
        }

        let (disposition, reason) = match (&action, target.target(), target.source()) {
            (
                ActionCategory::Enable | ActionCategory::Disable,
                NixActionTargetV1::SystemdUnit(_),
                TargetEvidenceSourceV1::SystemdInventory,
            ) => (
                RealizationDispositionV1::Admitted,
                RealizationReasonV1::ExactSystemdUnitForLifecycle,
            ),
            (
                ActionCategory::Enable | ActionCategory::Disable,
                NixActionTargetV1::JournalProducer(_),
                TargetEvidenceSourceV1::Journal,
            ) => (
                RealizationDispositionV1::NeedsResolution,
                RealizationReasonV1::JournalProducerRequiresInventoryResolution,
            ),
            (
                ActionCategory::Enable | ActionCategory::Disable,
                NixActionTargetV1::UserSupplied(_),
                TargetEvidenceSourceV1::UserInput,
            ) => (
                RealizationDispositionV1::NeedsResolution,
                RealizationReasonV1::UserTargetRequiresResolution,
            ),
            (
                ActionCategory::Enable | ActionCategory::Disable,
                NixActionTargetV1::Metric(_),
                TargetEvidenceSourceV1::PredictiveMonitor,
            ) => (
                RealizationDispositionV1::Inapplicable,
                RealizationReasonV1::MetricIsNotServiceTarget,
            ),
            (
                ActionCategory::GarbageCollect,
                NixActionTargetV1::System,
                TargetEvidenceSourceV1::ExplicitSystem,
            ) => (
                RealizationDispositionV1::Admitted,
                RealizationReasonV1::ExplicitSystemForGarbageCollection,
            ),
            (
                ActionCategory::Rebuild,
                NixActionTargetV1::System,
                TargetEvidenceSourceV1::ExplicitSystem,
            ) => (
                RealizationDispositionV1::Admitted,
                RealizationReasonV1::ExplicitSystemForRebuild,
            ),
            (
                ActionCategory::Install | ActionCategory::Remove,
                NixActionTargetV1::Package(_),
                TargetEvidenceSourceV1::PackageInventory,
            ) => (
                RealizationDispositionV1::NeedsParameters,
                RealizationReasonV1::PackageOperationRequiresParameters,
            ),
            (
                ActionCategory::Configure,
                NixActionTargetV1::NixOption(_),
                TargetEvidenceSourceV1::NixOptionSchema,
            ) => (
                RealizationDispositionV1::NeedsParameters,
                RealizationReasonV1::ConfigureRequiresPatchPlan,
            ),
            (
                ActionCategory::Rollback,
                NixActionTargetV1::SystemGeneration(_),
                TargetEvidenceSourceV1::GenerationInventory,
            ) => (
                RealizationDispositionV1::NeedsParameters,
                RealizationReasonV1::RollbackRequiresQualifiedProfile,
            ),
            (
                ActionCategory::Update,
                NixActionTargetV1::System,
                TargetEvidenceSourceV1::ExplicitSystem,
            ) => (
                RealizationDispositionV1::NeedsParameters,
                RealizationReasonV1::UpdateRequiresExplicitProfile,
            ),
            _ => (
                RealizationDispositionV1::Inapplicable,
                RealizationReasonV1::TargetKindIncompatible,
            ),
        };

        outcome(action, target, disposition, reason)
    }
}

fn outcome(
    action: ActionCategory,
    target: TargetEvidenceV1,
    disposition: RealizationDispositionV1,
    reason: RealizationReasonV1,
) -> RealizationOutcomeV1 {
    RealizationOutcomeV1 {
        action,
        target,
        disposition,
        reason,
        profile: ACTION_REALIZATION_PROFILE_V1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window() -> ObservationWindowV1 {
        ObservationWindowV1::new(100, 200).unwrap()
    }

    fn unit(name: &str) -> UnitInfo {
        UnitInfo {
            name: name.into(),
            load_state: "loaded".into(),
            active_state: "active".into(),
            sub_state: "running".into(),
            description: "fixture".into(),
        }
    }

    fn service_target() -> TargetEvidenceV1 {
        TargetEvidenceV1::from_systemd_inventory(
            &unit("nginx.service"),
            "obs:systemd:1",
            window(),
        )
        .unwrap()
    }

    #[test]
    fn observed_service_enable_is_admitted_but_not_authorized() {
        let result = ActionRealizerV1::assess(ActionCategory::Enable, service_target(), 150);
        assert!(result.is_admitted());
        assert_eq!(result.profile(), ACTION_REALIZATION_PROFILE_V1);
        assert_eq!(result.target().evidence_ref(), Some("obs:systemd:1"));
        assert_eq!(
            result.reason(),
            RealizationReasonV1::ExactSystemdUnitForLifecycle
        );
    }

    #[test]
    fn stale_systemd_inventory_fails_closed() {
        let result = ActionRealizerV1::assess(ActionCategory::Enable, service_target(), 201);
        assert_eq!(
            result.disposition(),
            RealizationDispositionV1::StaleEvidence
        );
        assert_eq!(result.reason(), RealizationReasonV1::EvidenceStale);
    }

    #[test]
    fn journal_producer_named_like_service_still_needs_resolution() {
        let target = TargetEvidenceV1::journal_producer(
            "nginx.service",
            "obs:journal:1",
            window(),
        )
        .unwrap();
        let result = ActionRealizerV1::assess(ActionCategory::Enable, target, 150);
        assert_eq!(
            result.disposition(),
            RealizationDispositionV1::NeedsResolution
        );
        assert_eq!(
            result.reason(),
            RealizationReasonV1::JournalProducerRequiresInventoryResolution
        );
    }

    #[test]
    fn metric_or_user_text_cannot_mint_service_identity() {
        let metric = TargetEvidenceV1::metric(
            "memory_used_pct",
            "obs:predictive:1",
            window(),
        )
        .unwrap();
        let metric_result = ActionRealizerV1::assess(ActionCategory::Enable, metric, 150);
        assert_eq!(
            metric_result.disposition(),
            RealizationDispositionV1::Inapplicable
        );

        let user = TargetEvidenceV1::user_supplied("nginx.service").unwrap();
        let user_result = ActionRealizerV1::assess(ActionCategory::Disable, user, 150);
        assert_eq!(
            user_result.disposition(),
            RealizationDispositionV1::NeedsResolution
        );
    }

    #[test]
    fn gc_and_rebuild_admit_only_explicit_system_target() {
        assert!(ActionRealizerV1::assess(
            ActionCategory::GarbageCollect,
            TargetEvidenceV1::system(),
            150,
        )
        .is_admitted());
        assert!(ActionRealizerV1::assess(
            ActionCategory::Rebuild,
            TargetEvidenceV1::system(),
            150,
        )
        .is_admitted());
        assert_eq!(
            ActionRealizerV1::assess(ActionCategory::GarbageCollect, service_target(), 150)
                .disposition(),
            RealizationDispositionV1::Inapplicable
        );
    }

    #[test]
    fn parameterized_domains_stop_before_command_construction() {
        let option = TargetEvidenceV1::nix_option(
            "services.nginx.enable",
            "schema:nixos-options:fixture",
        )
        .unwrap();
        assert_eq!(option.evidence_ref(), Some("schema:nixos-options:fixture"));
        assert_eq!(
            ActionRealizerV1::assess(ActionCategory::Configure, option, 150).disposition(),
            RealizationDispositionV1::NeedsParameters
        );

        let package = TargetEvidenceV1::package("ripgrep", "obs:packages:1", window()).unwrap();
        assert_eq!(
            ActionRealizerV1::assess(ActionCategory::Install, package, 150).disposition(),
            RealizationDispositionV1::NeedsParameters
        );

        let generation =
            TargetEvidenceV1::system_generation(42, "obs:generations:1", window()).unwrap();
        assert_eq!(
            ActionRealizerV1::assess(ActionCategory::Rollback, generation, 150).disposition(),
            RealizationDispositionV1::NeedsParameters
        );
    }

    #[test]
    fn semantically_wrong_target_is_inapplicable() {
        assert_eq!(
            ActionRealizerV1::assess(ActionCategory::Install, service_target(), 150).disposition(),
            RealizationDispositionV1::Inapplicable
        );
    }

    #[test]
    fn custom_and_unknown_never_admit() {
        let custom = ActionRealizerV1::assess(
            ActionCategory::Custom("restart-everything".into()),
            TargetEvidenceV1::system(),
            150,
        );
        assert_eq!(custom.disposition(), RealizationDispositionV1::Unsupported);

        let unknown = TargetEvidenceV1::unknown("nginx.service").unwrap();
        assert_eq!(
            ActionRealizerV1::assess(ActionCategory::Enable, unknown, 150).disposition(),
            RealizationDispositionV1::NeedsResolution
        );
    }

    #[test]
    fn invalid_window_and_empty_evidence_ref_fail_closed() {
        assert_eq!(
            ObservationWindowV1::new(200, 100),
            Err(TargetEvidenceErrorV1::InvalidObservationWindow)
        );
        assert_eq!(
            TargetEvidenceV1::metric("memory_used_pct", "", window()),
            Err(TargetEvidenceErrorV1::EmptyEvidenceRef)
        );
    }

    #[test]
    fn systemd_target_rejects_non_exact_unit_shapes() {
        for bad in [
            "--now.service",
            "*.service",
            "foo/bar.service",
            "not-a-service.timer",
            "white space.service",
        ] {
            assert_eq!(
                TargetEvidenceV1::from_systemd_inventory(
                    &unit(bad),
                    "obs:systemd:1",
                    window(),
                ),
                Err(TargetEvidenceErrorV1::InvalidSystemdUnit),
                "{bad} must not become a typed service identity"
            );
        }
    }
}
