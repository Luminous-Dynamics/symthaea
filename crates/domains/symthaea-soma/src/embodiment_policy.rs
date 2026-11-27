// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Default-deny presentation policy for Symthaea embodiments.
//!
//! This module is an output-permission decision point, not a renderer, actuator,
//! authentication system, or proof of consent provenance. A caller must validate
//! the event stream and authenticate the user's grant before constructing a
//! ConsentGrant. No output is permitted until a matching grant is installed.
//!
//! Sensor capture permissions are deliberately separate from presentation
//! permissions; granting a visual output never grants camera access.

use super::embodiment_event::{EpistemicStatus, EventKind, EventSource, PrivacyClass};
use super::embodiment_projection::{CanonicalCue, EmbodimentForm};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputChannel {
    Visual,
    Audio,
    Haptic,
    Spatial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputDestination {
    LocalPrivate,
    LocalShared,
    Remote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionMode {
    Dynamic,
    Static,
}

/// An explicit, bounded grant supplied by an authenticated consent boundary.
///
/// The policy validates scope, expiry, form, output channel, destination, and
/// intensity. It cannot establish who issued the grant or whether the user was
/// coerced; callers must authenticate and record that provenance separately.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsentGrant {
    pub session_id: String,
    pub clock_domain_id: String,
    pub channel: OutputChannel,
    pub destination: OutputDestination,
    pub allowed_forms: Vec<EmbodimentForm>,
    pub expires_at_monotonic_ns: String,
    pub max_intensity: f32,
}

/// A request for one output. Evaluating it never performs the output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresentationRequest {
    pub form: EmbodimentForm,
    pub channel: OutputChannel,
    pub destination: OutputDestination,
    /// Requested output amplitude, not the meaning or confidence of the cue.
    pub requested_intensity: f32,
}

/// A short-lived decision envelope for a renderer or output adapter.
///
/// It carries source identity and epistemic metadata so the adapter can retain
/// provenance. It is not a cryptographic receipt and is not authorization to
/// perform physical actuation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresentationPermit {
    pub session_id: String,
    pub clock_domain_id: String,
    pub event_id: String,
    pub issued_at_monotonic_ns: String,
    /// Minimum of event expiry and grant expiry at decision time.
    pub expires_at_monotonic_ns: String,
    pub requested_intensity: f32,
    pub sequence: u64,
    pub event_kind: EventKind,
    pub source: EventSource,
    pub epistemic_status: EpistemicStatus,
    pub privacy_class: PrivacyClass,
    pub form: EmbodimentForm,
    pub channel: OutputChannel,
    pub destination: OutputDestination,
    pub motion_mode: MotionMode,
    pub effective_intensity: f32,
    pub semantic_key: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    InvalidIdentifier,
    InvalidTimestamp,
    ClockDomainMismatch,
    SessionMismatch,
    GrantAlreadyExpired,
    InvalidIntensity,
    EmptyFormGrant,
    DuplicateFormGrant,
    CancellationCapacityReached,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenialReason {
    Paused,
    Cancelled,
    SessionMismatch,
    ClockDomainMismatch,
    InvalidTimestamp,
    EventObservedInFuture,
    EventExpired,
    NoMatchingGrant,
    GrantExpired,
    PermitExpired,
    FormNotGranted,
    RestrictedPrivacy,
    SensitiveDataNeedsPrivateDestination,
    InvalidIntensity,
}

/// Session-scoped, fail-closed output policy.
///
/// A new instance has no grants. Reconstructing it therefore returns the system
/// to the deny-all state. Callers should keep this policy in the same
/// synchronization domain as user pause/revocation controls.
#[derive(Debug, Clone)]
pub struct PresentationPolicy {
    session_id: String,
    clock_domain_id: String,
    grants: Vec<ConsentGrant>,
    cancelled_event_ids: HashSet<String>,
    paused: bool,
    reduced_motion: bool,
}

impl PresentationPolicy {
    pub fn new(
        session_id: impl Into<String>,
        clock_domain_id: impl Into<String>,
    ) -> Result<Self, PolicyError> {
        let session_id = session_id.into();
        let clock_domain_id = clock_domain_id.into();
        validate_identifier(&session_id)?;
        validate_identifier(&clock_domain_id)?;
        Ok(Self {
            session_id,
            clock_domain_id,
            grants: Vec::new(),
            cancelled_event_ids: HashSet::new(),
            paused: true,
            reduced_motion: true,
        })
    }

    /// Install an explicit grant. A grant is scoped to this session and clock,
    /// and must still be live at installation time.
    pub fn grant(
        &mut self,
        grant: ConsentGrant,
        now_monotonic_ns: &str,
        now_clock_domain_id: &str,
    ) -> Result<(), PolicyError> {
        if now_clock_domain_id != self.clock_domain_id {
            return Err(PolicyError::ClockDomainMismatch);
        }
        if grant.session_id != self.session_id {
            return Err(PolicyError::SessionMismatch);
        }
        if grant.clock_domain_id != self.clock_domain_id {
            return Err(PolicyError::ClockDomainMismatch);
        }
        validate_identifier(&grant.session_id)?;
        validate_identifier(&grant.clock_domain_id)?;

        let now = parse_monotonic_ns(now_monotonic_ns)?;
        let expires = parse_monotonic_ns(&grant.expires_at_monotonic_ns)?;
        if expires <= now {
            return Err(PolicyError::GrantAlreadyExpired);
        }
        if !grant.max_intensity.is_finite()
            || !(0.0..=1.0).contains(&grant.max_intensity)
        {
            return Err(PolicyError::InvalidIntensity);
        }
        if grant.allowed_forms.is_empty() {
            return Err(PolicyError::EmptyFormGrant);
        }
        for (index, form) in grant.allowed_forms.iter().enumerate() {
            if grant.allowed_forms[..index].contains(form) {
                return Err(PolicyError::DuplicateFormGrant);
            }
        }

        // Replace an earlier grant for the same output scope. A refresh cannot
        // accidentally retain a broader old permission.
        self.grants.retain(|existing| {
            existing.channel != grant.channel || existing.destination != grant.destination
        });
        self.grants.push(grant);
        Ok(())
    }

    /// Revoke all destinations for a channel immediately.
    pub fn revoke_channel(&mut self, channel: OutputChannel) -> bool {
        let before = self.grants.len();
        self.grants.retain(|grant| grant.channel != channel);
        self.grants.len() != before
    }

    /// Pause is fail-closed. Resume is explicit and does not create grants.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    pub fn set_reduced_motion(&mut self, reduced_motion: bool) {
        self.reduced_motion = reduced_motion;
    }

    /// Mark an event cancelled so no later request can present it.
    pub fn cancel_event(&mut self, event_id: &str) -> Result<(), PolicyError> {
        validate_identifier(event_id)?;
        if !self.cancelled_event_ids.contains(event_id)
            && self.cancelled_event_ids.len() >= 65_536
        {
            return Err(PolicyError::CancellationCapacityReached);
        }
        self.cancelled_event_ids.insert(event_id.to_owned());
        Ok(())
    }

    /// Decide whether one canonical cue may be presented.
    ///
    /// The caller must first run EmbodimentStreamValidator and must re-check
    /// cancellation immediately before handing the permit to a renderer. The
    /// method itself performs no I/O and never modifies the canonical cue.
    pub fn evaluate(
        &self,
        cue: &CanonicalCue,
        request: &PresentationRequest,
        now_monotonic_ns: &str,
        now_clock_domain_id: &str,
    ) -> Result<PresentationPermit, DenialReason> {
        if self.paused {
            return Err(DenialReason::Paused);
        }
        if cue.session_id != self.session_id {
            return Err(DenialReason::SessionMismatch);
        }
        if now_clock_domain_id != self.clock_domain_id
            || cue.clock_domain_id != self.clock_domain_id
        {
            return Err(DenialReason::ClockDomainMismatch);
        }
        if self.cancelled_event_ids.contains(&cue.event_id) {
            return Err(DenialReason::Cancelled);
        }

        let now = parse_monotonic_ns(now_monotonic_ns)
            .map_err(|_| DenialReason::InvalidTimestamp)?;
        let observed = parse_monotonic_ns(&cue.observed_at_monotonic_ns)
            .map_err(|_| DenialReason::InvalidTimestamp)?;
        let event_expires = parse_monotonic_ns(&cue.expires_at_monotonic_ns)
            .map_err(|_| DenialReason::InvalidTimestamp)?;
        if observed > now {
            return Err(DenialReason::EventObservedInFuture);
        }
        if event_expires <= now {
            return Err(DenialReason::EventExpired);
        }

        if !request.requested_intensity.is_finite()
            || !(0.0..=1.0).contains(&request.requested_intensity)
        {
            return Err(DenialReason::InvalidIntensity);
        }
        if cue.privacy_class == PrivacyClass::Restricted {
            return Err(DenialReason::RestrictedPrivacy);
        }
        if cue.privacy_class == PrivacyClass::LocalSensitive
            && request.destination != OutputDestination::LocalPrivate
        {
            return Err(DenialReason::SensitiveDataNeedsPrivateDestination);
        }

        let grant = self
            .grants
            .iter()
            .find(|grant| {
                grant.channel == request.channel && grant.destination == request.destination
            })
            .ok_or(DenialReason::NoMatchingGrant)?;

        let grant_expires = parse_monotonic_ns(&grant.expires_at_monotonic_ns)
            .map_err(|_| DenialReason::InvalidTimestamp)?;
        if grant_expires <= now {
            return Err(DenialReason::GrantExpired);
        }
        if !grant.allowed_forms.contains(&request.form) {
            return Err(DenialReason::FormNotGranted);
        }

        Ok(PresentationPermit {
            session_id: cue.session_id.clone(),
            clock_domain_id: cue.clock_domain_id.clone(),
            event_id: cue.event_id.clone(),
            issued_at_monotonic_ns: now_monotonic_ns.to_owned(),
            expires_at_monotonic_ns: event_expires.min(grant_expires).to_string(),
            requested_intensity: request.requested_intensity,
            sequence: cue.sequence,
            event_kind: cue.event_kind,
            source: cue.source.clone(),
            epistemic_status: cue.epistemic_status,
            privacy_class: cue.privacy_class,
            form: request.form,
            channel: request.channel,
            destination: request.destination,
            motion_mode: if request.channel == OutputChannel::Visual && self.reduced_motion {
                MotionMode::Static
            } else {
                MotionMode::Dynamic
            },
            // Clamp output amplitude without changing the canonical cue, its
            // semantics, confidence, or epistemic status.
            effective_intensity: request.requested_intensity.min(grant.max_intensity),
            semantic_key: cue.semantic.semantic_key().to_owned(),
        })
    }

    /// Revalidate a previously issued permit immediately before renderer
    /// submission. Pause, cancellation, grant revocation/narrowing, reduced
    /// motion, and expiry are re-applied at this final boundary.
    ///
    /// This is still a local policy check, not a cryptographic anti-replay
    /// token. Keep permits process-local and do not serialize them as proof.
    pub fn recheck_permit(
        &self,
        permit: &PresentationPermit,
        now_monotonic_ns: &str,
        now_clock_domain_id: &str,
    ) -> Result<PresentationPermit, DenialReason> {
        if self.paused {
            return Err(DenialReason::Paused);
        }
        if permit.session_id != self.session_id {
            return Err(DenialReason::SessionMismatch);
        }
        if now_clock_domain_id != self.clock_domain_id
            || permit.clock_domain_id != self.clock_domain_id
        {
            return Err(DenialReason::ClockDomainMismatch);
        }
        if self.cancelled_event_ids.contains(&permit.event_id) {
            return Err(DenialReason::Cancelled);
        }

        let now = parse_monotonic_ns(now_monotonic_ns)
            .map_err(|_| DenialReason::InvalidTimestamp)?;
        let issued = parse_monotonic_ns(&permit.issued_at_monotonic_ns)
            .map_err(|_| DenialReason::InvalidTimestamp)?;
        let permit_expires = parse_monotonic_ns(&permit.expires_at_monotonic_ns)
            .map_err(|_| DenialReason::InvalidTimestamp)?;
        if issued > now {
            return Err(DenialReason::EventObservedInFuture);
        }
        if permit_expires <= now {
            return Err(DenialReason::PermitExpired);
        }
        if !permit.requested_intensity.is_finite()
            || !(0.0..=1.0).contains(&permit.requested_intensity)
        {
            return Err(DenialReason::InvalidIntensity);
        }
        if permit.privacy_class == PrivacyClass::Restricted {
            return Err(DenialReason::RestrictedPrivacy);
        }
        if permit.privacy_class == PrivacyClass::LocalSensitive
            && permit.destination != OutputDestination::LocalPrivate
        {
            return Err(DenialReason::SensitiveDataNeedsPrivateDestination);
        }

        let grant = self
            .grants
            .iter()
            .find(|grant| {
                grant.channel == permit.channel && grant.destination == permit.destination
            })
            .ok_or(DenialReason::NoMatchingGrant)?;
        if !grant.allowed_forms.contains(&permit.form) {
            return Err(DenialReason::FormNotGranted);
        }
        let grant_expires = parse_monotonic_ns(&grant.expires_at_monotonic_ns)
            .map_err(|_| DenialReason::InvalidTimestamp)?;
        if grant_expires <= now {
            return Err(DenialReason::GrantExpired);
        }

        let mut refreshed = permit.clone();
        refreshed.expires_at_monotonic_ns = permit_expires.min(grant_expires).to_string();
        refreshed.effective_intensity =
            permit.requested_intensity.min(grant.max_intensity);
        refreshed.motion_mode =
            if permit.channel == OutputChannel::Visual && self.reduced_motion {
                MotionMode::Static
            } else {
                MotionMode::Dynamic
            };
        Ok(refreshed)
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn grant_count(&self) -> usize {
        self.grants.len()
    }
}

fn validate_identifier(value: &str) -> Result<(), PolicyError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(PolicyError::InvalidIdentifier);
    }
    Ok(())
}

fn parse_monotonic_ns(value: &str) -> Result<u64, PolicyError> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(PolicyError::InvalidTimestamp);
    }
    u64::from_str(value).map_err(|_| PolicyError::InvalidTimestamp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment_event::{EpistemicStatus, PrivacyClass};
    use crate::embodiment_projection::SemanticCue;

    const SESSION: &str = "session-1";
    const CLOCK: &str = "clock:dialogue-epoch-1";

    fn cue(privacy_class: PrivacyClass) -> CanonicalCue {
        CanonicalCue {
            session_id: SESSION.to_owned(),
            clock_domain_id: CLOCK.to_owned(),
            event_id: "event-1".to_owned(),
            sequence: 4,
            event_kind: EventKind::ConversationListening,
            semantic: SemanticCue::Listening { active: true },
            source: EventSource {
                component: "dialogue".to_owned(),
                instance_id: "dialogue-1".to_owned(),
                source_event_id: None,
                build_id: Some("build-1".to_owned()),
            },
            observed_at_monotonic_ns: "100".to_owned(),
            expires_at_monotonic_ns: "1000".to_owned(),
            epistemic_status: EpistemicStatus::Measured,
            confidence: None,
            evidence_refs: Vec::new(),
            privacy_class,
        }
    }

    fn request(channel: OutputChannel, destination: OutputDestination) -> PresentationRequest {
        PresentationRequest {
            form: EmbodimentForm::LivingConstellation,
            channel,
            destination,
            requested_intensity: 0.9,
        }
    }

    fn policy_with_grant(
        privacy_class: PrivacyClass,
        channel: OutputChannel,
        destination: OutputDestination,
    ) -> PresentationPolicy {
        let _ = privacy_class;
        let mut policy = PresentationPolicy::new(SESSION, CLOCK).unwrap();
        policy
            .grant(
                ConsentGrant {
                    session_id: SESSION.to_owned(),
                    clock_domain_id: CLOCK.to_owned(),
                    channel,
                    destination,
                    allowed_forms: vec![EmbodimentForm::LivingConstellation],
                    expires_at_monotonic_ns: "900".to_owned(),
                    max_intensity: 0.5,
                },
                "200",
                CLOCK,
            )
            .unwrap();
        policy.set_paused(false);
        policy
    }

    #[test]
    fn new_policy_denies_by_default() {
        let policy = PresentationPolicy::new(SESSION, CLOCK).unwrap();
        let result = policy.evaluate(
            &cue(PrivacyClass::Public),
            &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
            "200",
            CLOCK,
        );
        assert_eq!(result, Err(DenialReason::Paused));
        assert_eq!(policy.grant_count(), 0);
    }

    #[test]
    fn explicit_grant_permits_and_clamps_without_rewriting_semantics() {
        let policy = policy_with_grant(
            PrivacyClass::Public,
            OutputChannel::Visual,
            OutputDestination::LocalPrivate,
        );
        let canonical = cue(PrivacyClass::Public);
        let permit = policy
            .evaluate(
                &canonical,
                &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
                "200",
                CLOCK,
            )
            .unwrap();
        assert_eq!(permit.effective_intensity, 0.5);
        assert_eq!(permit.motion_mode, MotionMode::Static);
        assert_eq!(permit.event_id, canonical.event_id);
        assert_eq!(permit.semantic_key, canonical.semantic.semantic_key());
    }

    #[test]
    fn reduced_motion_can_be_disabled_only_as_an_explicit_policy_change() {
        let mut policy = policy_with_grant(
            PrivacyClass::Public,
            OutputChannel::Visual,
            OutputDestination::LocalPrivate,
        );
        policy.set_reduced_motion(false);
        let permit = policy
            .evaluate(
                &cue(PrivacyClass::Public),
                &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
                "200",
                CLOCK,
            )
            .unwrap();
        assert_eq!(permit.motion_mode, MotionMode::Dynamic);
    }

    #[test]
    fn local_sensitive_cues_cannot_be_sent_to_shared_or_remote_destinations() {
        let policy = policy_with_grant(
            PrivacyClass::LocalSensitive,
            OutputChannel::Visual,
            OutputDestination::Remote,
        );
        let result = policy.evaluate(
            &cue(PrivacyClass::LocalSensitive),
            &request(OutputChannel::Visual, OutputDestination::Remote),
            "200",
            CLOCK,
        );
        assert_eq!(
            result,
            Err(DenialReason::SensitiveDataNeedsPrivateDestination)
        );
    }

    #[test]
    fn restricted_cues_are_never_presented_by_this_policy_version() {
        let policy = policy_with_grant(
            PrivacyClass::Restricted,
            OutputChannel::Visual,
            OutputDestination::LocalPrivate,
        );
        assert_eq!(
            policy.evaluate(
                &cue(PrivacyClass::Restricted),
                &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
                "200",
                CLOCK,
            ),
            Err(DenialReason::RestrictedPrivacy)
        );
    }

    #[test]
    fn expired_events_and_expired_grants_are_rejected() {
        let policy = policy_with_grant(
            PrivacyClass::Public,
            OutputChannel::Visual,
            OutputDestination::LocalPrivate,
        );
        assert_eq!(
            policy.evaluate(
                &cue(PrivacyClass::Public),
                &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
                "1000",
                CLOCK,
            ),
            Err(DenialReason::EventExpired)
        );
        assert_eq!(
            policy.evaluate(
                &cue(PrivacyClass::Public),
                &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
                "900",
                CLOCK,
            ),
            Err(DenialReason::GrantExpired)
        );
    }

    #[test]
    fn cancellation_and_pause_are_immediate_deny_states() {
        let mut policy = policy_with_grant(
            PrivacyClass::Public,
            OutputChannel::Visual,
            OutputDestination::LocalPrivate,
        );
        policy.cancel_event("event-1").unwrap();
        assert_eq!(
            policy.evaluate(
                &cue(PrivacyClass::Public),
                &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
                "200",
                CLOCK,
            ),
            Err(DenialReason::Cancelled)
        );
        policy.set_paused(true);
        assert_eq!(
            policy.evaluate(
                &cue(PrivacyClass::Public),
                &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
                "200",
                CLOCK,
            ),
            Err(DenialReason::Paused)
        );
    }

    #[test]
    fn channel_revocation_removes_all_grants_for_that_channel() {
        let mut policy = policy_with_grant(
            PrivacyClass::Public,
            OutputChannel::Visual,
            OutputDestination::LocalPrivate,
        );
        assert!(policy.revoke_channel(OutputChannel::Visual));
        assert_eq!(policy.grant_count(), 0);
        assert_eq!(
            policy.evaluate(
                &cue(PrivacyClass::Public),
                &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
                "200",
                CLOCK,
            ),
            Err(DenialReason::NoMatchingGrant)
        );
    }

    #[test]
    fn wrong_clock_domain_is_rejected_at_evaluation() {
        let policy = policy_with_grant(
            PrivacyClass::Public,
            OutputChannel::Visual,
            OutputDestination::LocalPrivate,
        );
        assert_eq!(
            policy.evaluate(
                &cue(PrivacyClass::Public),
                &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
                "200",
                "clock:other-epoch",
            ),
            Err(DenialReason::ClockDomainMismatch)
        );
    }

    #[test]
    fn recheck_denies_permit_after_channel_revocation() {
        let mut policy = policy_with_grant(
            PrivacyClass::Public,
            OutputChannel::Visual,
            OutputDestination::LocalPrivate,
        );
        let permit = policy
            .evaluate(
                &cue(PrivacyClass::Public),
                &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
                "200",
                CLOCK,
            )
            .unwrap();
        policy.revoke_channel(OutputChannel::Visual);
        assert_eq!(
            policy.recheck_permit(&permit, "201", CLOCK),
            Err(DenialReason::NoMatchingGrant)
        );
    }

    #[test]
    fn recheck_applies_new_reduced_motion_state_before_submission() {
        let mut policy = policy_with_grant(
            PrivacyClass::Public,
            OutputChannel::Visual,
            OutputDestination::LocalPrivate,
        );
        policy.set_reduced_motion(false);
        let permit = policy
            .evaluate(
                &cue(PrivacyClass::Public),
                &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
                "200",
                CLOCK,
            )
            .unwrap();
        assert_eq!(permit.motion_mode, MotionMode::Dynamic);
        policy.set_reduced_motion(true);
        let refreshed = policy.recheck_permit(&permit, "201", CLOCK).unwrap();
        assert_eq!(refreshed.motion_mode, MotionMode::Static);
    }

    #[test]
    fn permit_expiry_is_the_earlier_of_event_and_grant_expiry() {
        let policy = policy_with_grant(
            PrivacyClass::Public,
            OutputChannel::Visual,
            OutputDestination::LocalPrivate,
        );
        let permit = policy
            .evaluate(
                &cue(PrivacyClass::Public),
                &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
                "200",
                CLOCK,
            )
            .unwrap();
        assert_eq!(permit.expires_at_monotonic_ns, "900");
        assert_eq!(
            policy.recheck_permit(&permit, "900", CLOCK),
            Err(DenialReason::PermitExpired)
        );
    }

    #[test]
    fn grant_cannot_be_installed_for_another_session_or_clock() {
        let mut policy = PresentationPolicy::new(SESSION, CLOCK).unwrap();
        let mut grant = ConsentGrant {
            session_id: "other-session".to_owned(),
            clock_domain_id: CLOCK.to_owned(),
            channel: OutputChannel::Visual,
            destination: OutputDestination::LocalPrivate,
            allowed_forms: vec![EmbodimentForm::LivingConstellation],
            expires_at_monotonic_ns: "900".to_owned(),
            max_intensity: 1.0,
        };
        assert_eq!(
            policy.grant(grant.clone(), "200", CLOCK),
            Err(PolicyError::SessionMismatch)
        );
        grant.session_id = SESSION.to_owned();
        grant.clock_domain_id = "clock:other".to_owned();
        assert_eq!(
            policy.grant(grant, "200", CLOCK),
            Err(PolicyError::ClockDomainMismatch)
        );
    }

    #[test]
    fn grant_replacement_does_not_accumulate_old_permissions() {
        let mut policy = policy_with_grant(
            PrivacyClass::Public,
            OutputChannel::Visual,
            OutputDestination::LocalPrivate,
        );
        policy
            .grant(
                ConsentGrant {
                    session_id: SESSION.to_owned(),
                    clock_domain_id: CLOCK.to_owned(),
                    channel: OutputChannel::Visual,
                    destination: OutputDestination::LocalPrivate,
                    allowed_forms: vec![EmbodimentForm::PuppetTheatre],
                    expires_at_monotonic_ns: "800".to_owned(),
                    max_intensity: 0.2,
                },
                "200",
                CLOCK,
            )
            .unwrap();
        assert_eq!(policy.grant_count(), 1);
        assert_eq!(
            policy.evaluate(
                &cue(PrivacyClass::Public),
                &request(OutputChannel::Visual, OutputDestination::LocalPrivate),
                "200",
                CLOCK,
            ),
            Err(DenialReason::FormNotGranted)
        );
    }
}
