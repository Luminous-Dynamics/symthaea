//! Symtropy/robotics-oriented admission contract over the direct Iroh lanes.
//!
//! The direct transport authenticates endpoints and packet bytes. This module
//! adds simulation semantics that QUIC cannot provide by itself: authority
//! epochs, bounded leases, stale-tick rejection, safety-stop latching, and
//! lane/delivery requirements. It intentionally contains no physics types so it
//! can move with [`crate::direct`] into a future shared Luminous transport crate.

use crate::{
    direct::{
        AuthenticatedDirectMessage, DirectDelivery, DirectLane, DirectSendReceipt, DirectTransport,
        DirectTransportError, ReliableDeliveryReceipt,
    },
    networking::{decode_bounded, encode_bounded},
};
use iroh::EndpointId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

pub const REALTIME_PROTOCOL_VERSION: u16 = 1;
pub const MAX_REALTIME_FRAME_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_REALTIME_PAYLOAD_BYTES: usize = 1024 * 1024;
pub const MAX_SAFETY_REASON_BYTES: usize = 4 * 1024;
pub const DEFAULT_MAX_REALTIME_SUBJECTS: usize = 65_536;
pub const DEFAULT_MAX_FUTURE_TICKS: u64 = 8;
pub const DEFAULT_TICK_REORDER_WINDOW: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RealtimeFrameKind {
    AuthorityLease,
    AuthorityRevoke,
    SafetyStop,
    SafetyResume,
    PlayerInput,
    StateDelta,
    Checkpoint,
    Telemetry,
    RoboticsCommand,
}

impl RealtimeFrameKind {
    pub const fn required_lane(self) -> DirectLane {
        match self {
            Self::AuthorityLease
            | Self::AuthorityRevoke
            | Self::SafetyStop
            | Self::SafetyResume => DirectLane::CONTROL,
            Self::PlayerInput => DirectLane::PLAYER_INPUT,
            Self::StateDelta | Self::Checkpoint => DirectLane::STATE_SNAPSHOT,
            Self::Telemetry => DirectLane::TELEMETRY,
            Self::RoboticsCommand => DirectLane::ROBOTICS,
        }
    }

    pub const fn required_delivery(self) -> DirectDelivery {
        match self {
            Self::PlayerInput | Self::StateDelta | Self::Telemetry => DirectDelivery::Datagram,
            Self::AuthorityLease
            | Self::AuthorityRevoke
            | Self::SafetyStop
            | Self::SafetyResume
            | Self::Checkpoint
            | Self::RoboticsCommand => DirectDelivery::Reliable,
        }
    }

    pub const fn is_actuating(self) -> bool {
        matches!(self, Self::PlayerInput | Self::RoboticsCommand)
    }

    pub const fn is_control(self) -> bool {
        matches!(
            self,
            Self::AuthorityLease | Self::AuthorityRevoke | Self::SafetyStop | Self::SafetyResume
        )
    }

    const fn requires_active_lease(self) -> bool {
        matches!(
            self,
            Self::PlayerInput
                | Self::StateDelta
                | Self::Checkpoint
                | Self::Telemetry
                | Self::RoboticsCommand
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityLeaseGrant {
    pub controller: EndpointId,
    pub valid_from_tick: u64,
    pub valid_until_tick: u64,
}

impl AuthorityLeaseGrant {
    fn validate(&self) -> Result<(), RealtimeFrameError> {
        if self.valid_until_tick < self.valid_from_tick {
            return Err(RealtimeFrameError::InvalidLeaseRange {
                valid_from_tick: self.valid_from_tick,
                valid_until_tick: self.valid_until_tick,
            });
        }
        Ok(())
    }
}

/// Versioned real-time frame encoded inside a signed [`DirectEnvelope`](crate::direct::DirectEnvelope).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealtimeFrame {
    pub protocol_version: u16,
    pub world_id: Uuid,
    pub subject_id: Uuid,
    pub authority_epoch: u64,
    pub tick: u64,
    pub valid_until_tick: u64,
    pub kind: RealtimeFrameKind,
    pub payload: Vec<u8>,
}

impl RealtimeFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        world_id: Uuid,
        subject_id: Uuid,
        authority_epoch: u64,
        tick: u64,
        valid_until_tick: u64,
        kind: RealtimeFrameKind,
        payload: Vec<u8>,
    ) -> Result<Self, RealtimeFrameError> {
        let frame = Self {
            protocol_version: REALTIME_PROTOCOL_VERSION,
            world_id,
            subject_id,
            authority_epoch,
            tick,
            valid_until_tick,
            kind,
            payload,
        };
        frame.validate()?;
        Ok(frame)
    }

    pub fn authority_lease(
        world_id: Uuid,
        subject_id: Uuid,
        authority_epoch: u64,
        tick: u64,
        grant: AuthorityLeaseGrant,
    ) -> Result<Self, RealtimeFrameError> {
        grant.validate()?;
        let valid_until_tick = grant.valid_until_tick;
        let payload = encode_bounded(&grant, MAX_REALTIME_PAYLOAD_BYTES)
            .map_err(RealtimeFrameError::Codec)?;
        Self::new(
            world_id,
            subject_id,
            authority_epoch,
            tick,
            valid_until_tick,
            RealtimeFrameKind::AuthorityLease,
            payload,
        )
    }

    pub fn authority_revoke(
        world_id: Uuid,
        subject_id: Uuid,
        authority_epoch: u64,
        tick: u64,
    ) -> Result<Self, RealtimeFrameError> {
        Self::new(
            world_id,
            subject_id,
            authority_epoch,
            tick,
            u64::MAX,
            RealtimeFrameKind::AuthorityRevoke,
            Vec::new(),
        )
    }

    pub fn safety_stop(
        world_id: Uuid,
        subject_id: Uuid,
        safety_epoch: u64,
        tick: u64,
        reason: impl Into<Vec<u8>>,
    ) -> Result<Self, RealtimeFrameError> {
        Self::new(
            world_id,
            subject_id,
            safety_epoch,
            tick,
            u64::MAX,
            RealtimeFrameKind::SafetyStop,
            reason.into(),
        )
    }

    pub fn safety_resume(
        world_id: Uuid,
        subject_id: Uuid,
        safety_epoch: u64,
        tick: u64,
        reason: impl Into<Vec<u8>>,
    ) -> Result<Self, RealtimeFrameError> {
        Self::new(
            world_id,
            subject_id,
            safety_epoch,
            tick,
            u64::MAX,
            RealtimeFrameKind::SafetyResume,
            reason.into(),
        )
    }

    pub fn encode(&self) -> Result<Vec<u8>, RealtimeFrameError> {
        self.validate()?;
        encode_bounded(self, MAX_REALTIME_FRAME_BYTES).map_err(RealtimeFrameError::Codec)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, RealtimeFrameError> {
        let frame: Self =
            decode_bounded(bytes, MAX_REALTIME_FRAME_BYTES).map_err(RealtimeFrameError::Codec)?;
        frame.validate()?;
        Ok(frame)
    }

    pub fn lease_grant(&self) -> Result<AuthorityLeaseGrant, RealtimeFrameError> {
        if self.kind != RealtimeFrameKind::AuthorityLease {
            return Err(RealtimeFrameError::NotAuthorityLease);
        }
        let grant: AuthorityLeaseGrant = decode_bounded(&self.payload, MAX_REALTIME_PAYLOAD_BYTES)
            .map_err(RealtimeFrameError::Codec)?;
        grant.validate()?;
        if grant.valid_until_tick != self.valid_until_tick {
            return Err(RealtimeFrameError::LeaseExpiryMismatch {
                frame: self.valid_until_tick,
                grant: grant.valid_until_tick,
            });
        }
        Ok(grant)
    }

    pub fn validate(&self) -> Result<(), RealtimeFrameError> {
        if self.protocol_version != REALTIME_PROTOCOL_VERSION {
            return Err(RealtimeFrameError::UnsupportedVersion {
                received: self.protocol_version,
            });
        }
        if self.world_id.is_nil() {
            return Err(RealtimeFrameError::NilWorldId);
        }
        if self.subject_id.is_nil() {
            return Err(RealtimeFrameError::NilSubjectId);
        }
        if self.authority_epoch == 0 {
            return Err(RealtimeFrameError::ZeroAuthorityEpoch);
        }
        if self.valid_until_tick < self.tick {
            return Err(RealtimeFrameError::InvalidTickRange {
                tick: self.tick,
                valid_until_tick: self.valid_until_tick,
            });
        }
        if self.payload.len() > MAX_REALTIME_PAYLOAD_BYTES {
            return Err(RealtimeFrameError::PayloadTooLarge {
                size: self.payload.len(),
                maximum: MAX_REALTIME_PAYLOAD_BYTES,
            });
        }
        match self.kind {
            RealtimeFrameKind::AuthorityLease => {
                self.lease_grant()?;
            }
            RealtimeFrameKind::AuthorityRevoke if !self.payload.is_empty() => {
                return Err(RealtimeFrameError::UnexpectedPayload { kind: self.kind });
            }
            RealtimeFrameKind::SafetyStop | RealtimeFrameKind::SafetyResume
                if self.payload.len() > MAX_SAFETY_REASON_BYTES =>
            {
                return Err(RealtimeFrameError::SafetyReasonTooLarge {
                    size: self.payload.len(),
                    maximum: MAX_SAFETY_REASON_BYTES,
                });
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum RealtimeFrameError {
    #[error("unsupported real-time protocol version {received}")]
    UnsupportedVersion { received: u16 },
    #[error("world ID must not be nil")]
    NilWorldId,
    #[error("subject ID must not be nil")]
    NilSubjectId,
    #[error("authority epoch zero is reserved")]
    ZeroAuthorityEpoch,
    #[error("frame tick {tick} exceeds expiry tick {valid_until_tick}")]
    InvalidTickRange { tick: u64, valid_until_tick: u64 },
    #[error("lease range {valid_from_tick}..={valid_until_tick} is invalid")]
    InvalidLeaseRange {
        valid_from_tick: u64,
        valid_until_tick: u64,
    },
    #[error("lease expiry mismatch: frame {frame}, grant {grant}")]
    LeaseExpiryMismatch { frame: u64, grant: u64 },
    #[error("frame is not an authority lease")]
    NotAuthorityLease,
    #[error("{kind:?} frames must not contain a payload")]
    UnexpectedPayload { kind: RealtimeFrameKind },
    #[error("real-time payload is {size} bytes; maximum is {maximum}")]
    PayloadTooLarge { size: usize, maximum: usize },
    #[error("safety reason is {size} bytes; maximum is {maximum}")]
    SafetyReasonTooLarge { size: usize, maximum: usize },
    #[error("real-time codec error: {0}")]
    Codec(String),
}

#[derive(Debug, Clone)]
pub struct RealtimeAdmissionPolicy {
    pub trusted_authority_issuers: HashSet<EndpointId>,
    pub trusted_safety_issuers: HashSet<EndpointId>,
    pub max_subjects: usize,
    pub max_future_ticks: u64,
    /// Small tolerated caller-clock regression. Admission still evaluates
    /// freshness against the retained high-water mark.
    pub max_clock_regression_ticks: u64,
    pub tick_reorder_window: u64,
}

impl Default for RealtimeAdmissionPolicy {
    fn default() -> Self {
        Self {
            trusted_authority_issuers: HashSet::new(),
            trusted_safety_issuers: HashSet::new(),
            max_subjects: DEFAULT_MAX_REALTIME_SUBJECTS,
            max_future_ticks: DEFAULT_MAX_FUTURE_TICKS,
            max_clock_regression_ticks: 0,
            tick_reorder_window: DEFAULT_TICK_REORDER_WINDOW,
        }
    }
}

impl RealtimeAdmissionPolicy {
    pub fn validate(&self) -> Result<(), RealtimeAdmissionError> {
        if self.max_subjects == 0 {
            return Err(RealtimeAdmissionError::InvalidPolicy(
                "max_subjects must be greater than zero",
            ));
        }
        if self.tick_reorder_window == 0 {
            return Err(RealtimeAdmissionError::InvalidPolicy(
                "tick_reorder_window must be greater than zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthorityLeaseSnapshot {
    pub epoch: u64,
    pub controller: Option<EndpointId>,
    pub valid_from_tick: u64,
    pub valid_until_tick: u64,
    pub revoked: bool,
}

#[derive(Debug, Clone, Copy)]
struct AuthorityLeaseState {
    epoch: u64,
    controller: Option<EndpointId>,
    valid_from_tick: u64,
    valid_until_tick: u64,
    revoked: bool,
}

impl From<AuthorityLeaseState> for AuthorityLeaseSnapshot {
    fn from(state: AuthorityLeaseState) -> Self {
        Self {
            epoch: state.epoch,
            controller: state.controller,
            valid_from_tick: state.valid_from_tick,
            valid_until_tick: state.valid_until_tick,
            revoked: state.revoked,
        }
    }
}

#[derive(Debug)]
struct TickWindow {
    width: u64,
    highest: u64,
    seen: HashSet<u64>,
}

impl TickWindow {
    fn new(width: u64) -> Self {
        Self {
            width: width.max(1),
            highest: 0,
            seen: HashSet::new(),
        }
    }

    fn insert(&mut self, tick: u64) -> bool {
        if self.highest > tick && self.highest - tick >= self.width {
            return false;
        }
        if !self.seen.insert(tick) {
            return false;
        }
        self.highest = self.highest.max(tick);
        let minimum = self.highest.saturating_sub(self.width - 1);
        self.seen.retain(|value| *value >= minimum);
        true
    }
}

#[derive(Debug)]
struct EpochTickWindow {
    epoch: u64,
    ticks: TickWindow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealtimeAdmissionOutcome {
    AuthorityLeaseInstalled,
    AuthorityLeaseStaged,
    AuthorityRevoked,
    SafetyStopLatched,
    SafetyStopCleared,
    OperationalFrameAccepted,
}

/// Stateful authority, safety, and tick admission guard.
#[derive(Debug)]
pub struct RealtimeAdmissionGuard {
    policy: RealtimeAdmissionPolicy,
    leases: HashMap<(Uuid, Uuid), AuthorityLeaseState>,
    pending_leases: HashMap<(Uuid, Uuid), AuthorityLeaseState>,
    safety_epochs: HashMap<(Uuid, Uuid), u64>,
    stopped: HashSet<(Uuid, Uuid)>,
    tick_windows: HashMap<(Uuid, Uuid, RealtimeFrameKind), EpochTickWindow>,
    world_ticks: HashMap<Uuid, u64>,
}

impl RealtimeAdmissionGuard {
    pub fn new(policy: RealtimeAdmissionPolicy) -> Result<Self, RealtimeAdmissionError> {
        policy.validate()?;
        Ok(Self {
            policy,
            leases: HashMap::new(),
            pending_leases: HashMap::new(),
            safety_epochs: HashMap::new(),
            stopped: HashSet::new(),
            tick_windows: HashMap::new(),
            world_ticks: HashMap::new(),
        })
    }

    pub fn policy(&self) -> &RealtimeAdmissionPolicy {
        &self.policy
    }

    pub fn replace_policy(
        &mut self,
        policy: RealtimeAdmissionPolicy,
    ) -> Result<(), RealtimeAdmissionError> {
        policy.validate()?;
        let current_subjects = self.subject_count();
        if policy.max_subjects < current_subjects {
            return Err(RealtimeAdmissionError::PolicyCapacityBelowState {
                current_subjects,
                maximum: policy.max_subjects,
            });
        }
        self.policy = policy;
        Ok(())
    }

    pub fn enroll_authority_issuer(&mut self, endpoint: EndpointId) {
        self.policy.trusted_authority_issuers.insert(endpoint);
    }

    pub fn enroll_safety_issuer(&mut self, endpoint: EndpointId) {
        self.policy.trusted_safety_issuers.insert(endpoint);
    }

    pub fn lease(&self, world_id: Uuid, subject_id: Uuid) -> Option<AuthorityLeaseSnapshot> {
        self.leases
            .get(&(world_id, subject_id))
            .copied()
            .map(Into::into)
    }

    pub fn pending_lease(
        &self,
        world_id: Uuid,
        subject_id: Uuid,
    ) -> Option<AuthorityLeaseSnapshot> {
        self.pending_leases
            .get(&(world_id, subject_id))
            .copied()
            .map(Into::into)
    }

    pub fn is_stopped(&self, world_id: Uuid, subject_id: Uuid) -> bool {
        self.stopped.contains(&(world_id, subject_id))
    }

    pub fn current_world_tick(&self, world_id: Uuid) -> Option<u64> {
        self.world_ticks.get(&world_id).copied()
    }

    pub fn subject_count(&self) -> usize {
        self.leases
            .keys()
            .chain(self.pending_leases.keys())
            .copied()
            .collect::<HashSet<_>>()
            .len()
    }

    /// Retire an expired or revoked subject after the application has finished
    /// any domain cleanup. A latched safety stop is intentionally not prunable.
    pub fn prune_subject(
        &mut self,
        world_id: Uuid,
        subject_id: Uuid,
        current_tick: u64,
    ) -> Result<bool, RealtimeAdmissionError> {
        let key = (world_id, subject_id);
        if self.stopped.contains(&key) {
            return Err(RealtimeAdmissionError::StoppedSubjectCannotBePruned);
        }
        if self.pending_leases.contains_key(&key) {
            return Err(RealtimeAdmissionError::PendingSubjectCannotBePruned);
        }
        let Some(lease) = self.leases.get(&key).copied() else {
            return Ok(false);
        };
        if !lease.revoked && current_tick <= lease.valid_until_tick {
            return Err(RealtimeAdmissionError::ActiveSubjectCannotBePruned {
                current_tick,
                valid_until_tick: lease.valid_until_tick,
            });
        }
        self.leases.remove(&key);
        self.pending_leases.remove(&key);
        self.safety_epochs.remove(&key);
        self.tick_windows
            .retain(|(world, subject, _), _| (*world, *subject) != key);
        let world_retained = self
            .leases
            .keys()
            .chain(self.pending_leases.keys())
            .any(|(world, _)| *world == world_id)
            || self.stopped.iter().any(|(world, _)| *world == world_id);
        if !world_retained {
            self.world_ticks.remove(&world_id);
        }
        Ok(true)
    }

    fn observe_world_tick(
        &mut self,
        world_id: Uuid,
        current_tick: u64,
    ) -> Result<u64, RealtimeAdmissionError> {
        let previous = self
            .world_ticks
            .get(&world_id)
            .copied()
            .unwrap_or(current_tick);
        if current_tick.saturating_add(self.policy.max_clock_regression_ticks) < previous {
            return Err(RealtimeAdmissionError::WorldClockRegressed {
                previous_tick: previous,
                received_tick: current_tick,
                tolerated_regression: self.policy.max_clock_regression_ticks,
            });
        }
        let high_water = previous.max(current_tick);
        self.world_ticks.insert(world_id, high_water);
        Ok(high_water)
    }

    pub fn admit_direct(
        &mut self,
        message: &AuthenticatedDirectMessage,
        current_tick: u64,
    ) -> Result<RealtimeAdmissionOutcome, RealtimeAdmissionError> {
        let frame = RealtimeFrame::decode(&message.payload)?;
        let expected_lane = frame.kind.required_lane();
        if message.lane != expected_lane {
            return Err(RealtimeAdmissionError::LaneMismatch {
                expected: expected_lane,
                received: message.lane,
            });
        }
        let expected_delivery = frame.kind.required_delivery();
        if message.delivery != expected_delivery {
            return Err(RealtimeAdmissionError::DeliveryMismatch {
                expected: expected_delivery,
                received: message.delivery,
            });
        }
        self.admit(message.author, frame, current_tick)
    }

    pub fn admit(
        &mut self,
        author: EndpointId,
        frame: RealtimeFrame,
        current_tick: u64,
    ) -> Result<RealtimeAdmissionOutcome, RealtimeAdmissionError> {
        frame.validate()?;
        let key = (frame.world_id, frame.subject_id);
        self.ensure_subject_capacity(key)?;
        let current_tick = self.observe_world_tick(frame.world_id, current_tick)?;
        if frame.kind.is_control() && frame.tick > current_tick {
            return Err(RealtimeAdmissionError::FutureControlFrame {
                kind: frame.kind,
                frame_tick: frame.tick,
                current_tick,
            });
        }
        if frame.tick > current_tick.saturating_add(self.policy.max_future_ticks) {
            return Err(RealtimeAdmissionError::FutureTick {
                frame_tick: frame.tick,
                current_tick,
                maximum_future_ticks: self.policy.max_future_ticks,
            });
        }
        if current_tick > frame.valid_until_tick {
            return Err(RealtimeAdmissionError::ExpiredTick {
                current_tick,
                valid_until_tick: frame.valid_until_tick,
            });
        }
        self.activate_pending_lease(key, current_tick);

        match frame.kind {
            RealtimeFrameKind::AuthorityLease => {
                self.install_lease(author, frame, key, current_tick)
            }
            RealtimeFrameKind::AuthorityRevoke => self.revoke_authority(author, frame, key),
            RealtimeFrameKind::SafetyStop => self.apply_safety_stop(author, frame, key),
            RealtimeFrameKind::SafetyResume => self.apply_safety_resume(author, frame, key),
            kind if kind.requires_active_lease() => self.admit_operational(author, frame, key),
            _ => Err(RealtimeAdmissionError::InvalidPolicy(
                "unhandled real-time frame kind",
            )),
        }
    }

    fn ensure_subject_capacity(&self, key: (Uuid, Uuid)) -> Result<(), RealtimeAdmissionError> {
        if !self.leases.contains_key(&key)
            && !self.pending_leases.contains_key(&key)
            && self.subject_count() >= self.policy.max_subjects
        {
            return Err(RealtimeAdmissionError::SubjectCapacityReached {
                maximum: self.policy.max_subjects,
            });
        }
        Ok(())
    }

    fn install_lease(
        &mut self,
        author: EndpointId,
        frame: RealtimeFrame,
        key: (Uuid, Uuid),
        current_tick: u64,
    ) -> Result<RealtimeAdmissionOutcome, RealtimeAdmissionError> {
        if !self.policy.trusted_authority_issuers.contains(&author) {
            return Err(RealtimeAdmissionError::UntrustedAuthorityIssuer { author });
        }
        let grant = frame.lease_grant()?;
        if frame.tick > grant.valid_until_tick {
            return Err(RealtimeAdmissionError::LeaseIssuedAfterExpiry {
                issue_tick: frame.tick,
                valid_until_tick: grant.valid_until_tick,
            });
        }
        let current_epoch = self.leases.get(&key).map(|lease| lease.epoch).unwrap_or(0);
        let pending_epoch = self
            .pending_leases
            .get(&key)
            .map(|lease| lease.epoch)
            .unwrap_or(0);
        let latest_epoch = current_epoch.max(pending_epoch);
        if frame.authority_epoch <= latest_epoch {
            return Err(RealtimeAdmissionError::AuthorityEpochNotAdvanced {
                current: latest_epoch,
                received: frame.authority_epoch,
            });
        }
        let state = AuthorityLeaseState {
            epoch: frame.authority_epoch,
            controller: Some(grant.controller),
            valid_from_tick: grant.valid_from_tick,
            valid_until_tick: grant.valid_until_tick,
            revoked: false,
        };
        if grant.valid_from_tick > current_tick {
            self.pending_leases.insert(key, state);
            Ok(RealtimeAdmissionOutcome::AuthorityLeaseStaged)
        } else {
            self.pending_leases.remove(&key);
            self.leases.insert(key, state);
            self.tick_windows
                .retain(|(world, subject, _), _| (*world, *subject) != key);
            Ok(RealtimeAdmissionOutcome::AuthorityLeaseInstalled)
        }
    }

    fn activate_pending_lease(&mut self, key: (Uuid, Uuid), current_tick: u64) {
        let Some(pending) = self.pending_leases.get(&key).copied() else {
            return;
        };
        if pending.valid_from_tick > current_tick {
            return;
        }
        self.pending_leases.remove(&key);
        self.leases.insert(key, pending);
        self.tick_windows
            .retain(|(world, subject, _), _| (*world, *subject) != key);
    }

    fn revoke_authority(
        &mut self,
        author: EndpointId,
        frame: RealtimeFrame,
        key: (Uuid, Uuid),
    ) -> Result<RealtimeAdmissionOutcome, RealtimeAdmissionError> {
        if !self.policy.trusted_authority_issuers.contains(&author) {
            return Err(RealtimeAdmissionError::UntrustedAuthorityIssuer { author });
        }
        let current_epoch = self
            .leases
            .get(&key)
            .map(|lease| lease.epoch)
            .unwrap_or(0)
            .max(
                self.pending_leases
                    .get(&key)
                    .map(|lease| lease.epoch)
                    .unwrap_or(0),
            );
        if frame.authority_epoch <= current_epoch {
            return Err(RealtimeAdmissionError::AuthorityEpochNotAdvanced {
                current: current_epoch,
                received: frame.authority_epoch,
            });
        }
        self.pending_leases.remove(&key);
        self.leases.insert(
            key,
            AuthorityLeaseState {
                epoch: frame.authority_epoch,
                controller: None,
                valid_from_tick: frame.tick,
                valid_until_tick: frame.tick,
                revoked: true,
            },
        );
        self.tick_windows
            .retain(|(world, subject, _), _| (*world, *subject) != key);
        Ok(RealtimeAdmissionOutcome::AuthorityRevoked)
    }

    fn apply_safety_stop(
        &mut self,
        author: EndpointId,
        frame: RealtimeFrame,
        key: (Uuid, Uuid),
    ) -> Result<RealtimeAdmissionOutcome, RealtimeAdmissionError> {
        if !self.leases.contains_key(&key) {
            return Err(RealtimeAdmissionError::NoAuthorityLease);
        }
        let safety_issuer = self.policy.trusted_safety_issuers.contains(&author);
        let controller = self
            .leases
            .get(&key)
            .filter(|lease| !lease.revoked)
            .and_then(|lease| lease.controller);
        if !safety_issuer && controller != Some(author) {
            return Err(RealtimeAdmissionError::UnauthorizedSafetyAuthor { author });
        }
        if !safety_issuer {
            let lease = self
                .leases
                .get(&key)
                .ok_or(RealtimeAdmissionError::NoAuthorityLease)?;
            if frame.authority_epoch != lease.epoch {
                return Err(RealtimeAdmissionError::AuthorityEpochMismatch {
                    expected: lease.epoch,
                    received: frame.authority_epoch,
                });
            }
        } else {
            self.advance_safety_epoch(key, frame.authority_epoch)?;
        }
        self.stopped.insert(key);
        Ok(RealtimeAdmissionOutcome::SafetyStopLatched)
    }

    fn apply_safety_resume(
        &mut self,
        author: EndpointId,
        frame: RealtimeFrame,
        key: (Uuid, Uuid),
    ) -> Result<RealtimeAdmissionOutcome, RealtimeAdmissionError> {
        if !self.leases.contains_key(&key) {
            return Err(RealtimeAdmissionError::NoAuthorityLease);
        }
        if !self.policy.trusted_safety_issuers.contains(&author) {
            return Err(RealtimeAdmissionError::UnauthorizedSafetyResume { author });
        }
        self.advance_safety_epoch(key, frame.authority_epoch)?;
        self.stopped.remove(&key);
        Ok(RealtimeAdmissionOutcome::SafetyStopCleared)
    }

    fn advance_safety_epoch(
        &mut self,
        key: (Uuid, Uuid),
        received: u64,
    ) -> Result<(), RealtimeAdmissionError> {
        let current = self.safety_epochs.get(&key).copied().unwrap_or(0);
        if received <= current {
            return Err(RealtimeAdmissionError::SafetyEpochNotAdvanced { current, received });
        }
        self.safety_epochs.insert(key, received);
        Ok(())
    }

    fn admit_operational(
        &mut self,
        author: EndpointId,
        frame: RealtimeFrame,
        key: (Uuid, Uuid),
    ) -> Result<RealtimeAdmissionOutcome, RealtimeAdmissionError> {
        let lease = self
            .leases
            .get(&key)
            .copied()
            .ok_or(RealtimeAdmissionError::NoAuthorityLease)?;
        if lease.revoked || lease.controller.is_none() {
            return Err(RealtimeAdmissionError::AuthorityRevoked);
        }
        let controller = lease
            .controller
            .ok_or(RealtimeAdmissionError::AuthorityRevoked)?;
        if controller != author {
            return Err(RealtimeAdmissionError::WrongController {
                expected: controller,
                received: author,
            });
        }
        if frame.authority_epoch != lease.epoch {
            return Err(RealtimeAdmissionError::AuthorityEpochMismatch {
                expected: lease.epoch,
                received: frame.authority_epoch,
            });
        }
        if frame.tick < lease.valid_from_tick || frame.tick > lease.valid_until_tick {
            return Err(RealtimeAdmissionError::TickOutsideLease {
                tick: frame.tick,
                valid_from_tick: lease.valid_from_tick,
                valid_until_tick: lease.valid_until_tick,
            });
        }
        if frame.kind.is_actuating() && self.stopped.contains(&key) {
            return Err(RealtimeAdmissionError::SafetyStopLatched);
        }

        let window_key = (frame.world_id, frame.subject_id, frame.kind);
        let epoch_window = self
            .tick_windows
            .entry(window_key)
            .or_insert_with(|| EpochTickWindow {
                epoch: frame.authority_epoch,
                ticks: TickWindow::new(self.policy.tick_reorder_window),
            });
        if epoch_window.epoch != frame.authority_epoch {
            epoch_window.epoch = frame.authority_epoch;
            epoch_window.ticks = TickWindow::new(self.policy.tick_reorder_window);
        }
        if !epoch_window.ticks.insert(frame.tick) {
            return Err(RealtimeAdmissionError::DuplicateOrStaleTick {
                tick: frame.tick,
                reorder_window: self.policy.tick_reorder_window,
            });
        }
        Ok(RealtimeAdmissionOutcome::OperationalFrameAccepted)
    }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum RealtimeAdmissionError {
    #[error("invalid real-time admission policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("invalid real-time frame: {0}")]
    InvalidFrame(#[from] RealtimeFrameError),
    #[error("direct lane mismatch: expected {expected:?}, received {received:?}")]
    LaneMismatch {
        expected: DirectLane,
        received: DirectLane,
    },
    #[error("direct delivery mismatch: expected {expected:?}, received {received:?}")]
    DeliveryMismatch {
        expected: DirectDelivery,
        received: DirectDelivery,
    },
    #[error(
        "world clock regressed from {previous_tick} to {received_tick}; tolerated regression is {tolerated_regression}"
    )]
    WorldClockRegressed {
        previous_tick: u64,
        received_tick: u64,
        tolerated_regression: u64,
    },
    #[error(
        "future control frame {kind:?} at tick {frame_tick} cannot execute at current tick {current_tick}"
    )]
    FutureControlFrame {
        kind: RealtimeFrameKind,
        frame_tick: u64,
        current_tick: u64,
    },
    #[error(
        "frame tick {frame_tick} is too far ahead of current tick {current_tick}; maximum lead is {maximum_future_ticks}"
    )]
    FutureTick {
        frame_tick: u64,
        current_tick: u64,
        maximum_future_ticks: u64,
    },
    #[error("frame expired at tick {valid_until_tick}; current tick is {current_tick}")]
    ExpiredTick {
        current_tick: u64,
        valid_until_tick: u64,
    },
    #[error("real-time subject capacity reached; maximum is {maximum}")]
    SubjectCapacityReached { maximum: usize },
    #[error("policy maximum {maximum} is below the {current_subjects} retained subjects")]
    PolicyCapacityBelowState {
        current_subjects: usize,
        maximum: usize,
    },
    #[error("endpoint {author} is not a trusted authority issuer")]
    UntrustedAuthorityIssuer { author: EndpointId },
    #[error("authority epoch must advance beyond {current}; received {received}")]
    AuthorityEpochNotAdvanced { current: u64, received: u64 },
    #[error("lease was issued at tick {issue_tick}, after expiry tick {valid_until_tick}")]
    LeaseIssuedAfterExpiry {
        issue_tick: u64,
        valid_until_tick: u64,
    },
    #[error("no authority lease exists for this subject")]
    NoAuthorityLease,
    #[error("authority has been revoked for this subject")]
    AuthorityRevoked,
    #[error("wrong controller: expected {expected}, received {received}")]
    WrongController {
        expected: EndpointId,
        received: EndpointId,
    },
    #[error("authority epoch mismatch: expected {expected}, received {received}")]
    AuthorityEpochMismatch { expected: u64, received: u64 },
    #[error("tick {tick} is outside lease {valid_from_tick}..={valid_until_tick}")]
    TickOutsideLease {
        tick: u64,
        valid_from_tick: u64,
        valid_until_tick: u64,
    },
    #[error("tick {tick} is duplicated or older than the {reorder_window}-tick reorder window")]
    DuplicateOrStaleTick { tick: u64, reorder_window: u64 },
    #[error("endpoint {author} is not authorized to issue a safety stop")]
    UnauthorizedSafetyAuthor { author: EndpointId },
    #[error("endpoint {author} is not authorized to resume after a safety stop")]
    UnauthorizedSafetyResume { author: EndpointId },
    #[error("safety epoch must advance beyond {current}; received {received}")]
    SafetyEpochNotAdvanced { current: u64, received: u64 },
    #[error("actuating traffic is blocked by the latched safety stop")]
    SafetyStopLatched,
    #[error("a subject with a latched safety stop cannot be pruned")]
    StoppedSubjectCannotBePruned,
    #[error("a subject with a staged authority lease cannot be pruned")]
    PendingSubjectCannotBePruned,
    #[error("active subject lease expires at {valid_until_tick}; current tick is {current_tick}")]
    ActiveSubjectCannotBePruned {
        current_tick: u64,
        valid_until_tick: u64,
    },
}

#[derive(Debug, Clone)]
pub enum RealtimeSendReceipt {
    Reliable(ReliableDeliveryReceipt),
    Datagram(DirectSendReceipt),
}

impl DirectTransport {
    /// Encode and send a real-time frame over its required lane and delivery
    /// primitive. The caller cannot accidentally place an authority lease on a
    /// datagram or a player-input frame on the gossip control plane.
    pub async fn send_realtime(
        &self,
        peer: EndpointId,
        frame: RealtimeFrame,
    ) -> Result<RealtimeSendReceipt, RealtimeSendError> {
        let lane = frame.kind.required_lane();
        let delivery = frame.kind.required_delivery();
        let payload = frame.encode()?;
        match delivery {
            DirectDelivery::Reliable => self
                .send_reliable(peer, lane, payload)
                .await
                .map(RealtimeSendReceipt::Reliable)
                .map_err(RealtimeSendError::Transport),
            DirectDelivery::Datagram => self
                .send_datagram(peer, lane, payload)
                .await
                .map(RealtimeSendReceipt::Datagram)
                .map_err(RealtimeSendError::Transport),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RealtimeSendError {
    #[error("invalid real-time frame: {0}")]
    Frame(#[from] RealtimeFrameError),
    #[error("direct transport failed: {0}")]
    Transport(DirectTransportError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::DirectEnvelope;
    use iroh::SecretKey;

    fn ids() -> (Uuid, Uuid) {
        (Uuid::from_u128(1), Uuid::from_u128(2))
    }

    fn policy(authority: EndpointId, safety: EndpointId) -> RealtimeAdmissionPolicy {
        RealtimeAdmissionPolicy {
            trusted_authority_issuers: HashSet::from([authority]),
            trusted_safety_issuers: HashSet::from([safety]),
            ..RealtimeAdmissionPolicy::default()
        }
    }

    #[test]
    fn authority_lease_binds_controller_epoch_and_tick_range() {
        let issuer = SecretKey::from_bytes(&[31u8; 32]).public();
        let safety = SecretKey::from_bytes(&[32u8; 32]).public();
        let controller = SecretKey::from_bytes(&[33u8; 32]).public();
        let (world, subject) = ids();
        let mut guard = RealtimeAdmissionGuard::new(policy(issuer, safety)).unwrap();
        let lease = RealtimeFrame::authority_lease(
            world,
            subject,
            1,
            10,
            AuthorityLeaseGrant {
                controller,
                valid_from_tick: 10,
                valid_until_tick: 100,
            },
        )
        .unwrap();
        assert_eq!(
            guard.admit(issuer, lease, 10).unwrap(),
            RealtimeAdmissionOutcome::AuthorityLeaseInstalled
        );
        let input = RealtimeFrame::new(
            world,
            subject,
            1,
            11,
            12,
            RealtimeFrameKind::PlayerInput,
            vec![1, 2, 3],
        )
        .unwrap();
        assert_eq!(
            guard.admit(controller, input, 11).unwrap(),
            RealtimeAdmissionOutcome::OperationalFrameAccepted
        );
    }

    #[test]
    fn stale_authority_and_wrong_controller_are_rejected() {
        let issuer = SecretKey::from_bytes(&[34u8; 32]).public();
        let safety = SecretKey::from_bytes(&[35u8; 32]).public();
        let controller = SecretKey::from_bytes(&[36u8; 32]).public();
        let attacker = SecretKey::from_bytes(&[37u8; 32]).public();
        let (world, subject) = ids();
        let mut guard = RealtimeAdmissionGuard::new(policy(issuer, safety)).unwrap();
        guard
            .admit(
                issuer,
                RealtimeFrame::authority_lease(
                    world,
                    subject,
                    2,
                    5,
                    AuthorityLeaseGrant {
                        controller,
                        valid_from_tick: 5,
                        valid_until_tick: 50,
                    },
                )
                .unwrap(),
                5,
            )
            .unwrap();
        let stale = RealtimeFrame::authority_revoke(world, subject, 2, 6).unwrap();
        assert!(matches!(
            guard.admit(issuer, stale, 6),
            Err(RealtimeAdmissionError::AuthorityEpochNotAdvanced { .. })
        ));
        let input = RealtimeFrame::new(
            world,
            subject,
            2,
            7,
            8,
            RealtimeFrameKind::PlayerInput,
            vec![9],
        )
        .unwrap();
        assert!(matches!(
            guard.admit(attacker, input, 7),
            Err(RealtimeAdmissionError::WrongController { .. })
        ));
    }

    #[test]
    fn safety_stop_blocks_actuation_until_trusted_resume() {
        let issuer = SecretKey::from_bytes(&[38u8; 32]).public();
        let safety = SecretKey::from_bytes(&[39u8; 32]).public();
        let controller = SecretKey::from_bytes(&[40u8; 32]).public();
        let (world, subject) = ids();
        let mut guard = RealtimeAdmissionGuard::new(policy(issuer, safety)).unwrap();
        guard
            .admit(
                issuer,
                RealtimeFrame::authority_lease(
                    world,
                    subject,
                    1,
                    1,
                    AuthorityLeaseGrant {
                        controller,
                        valid_from_tick: 1,
                        valid_until_tick: 100,
                    },
                )
                .unwrap(),
                1,
            )
            .unwrap();
        guard
            .admit(
                safety,
                RealtimeFrame::safety_stop(world, subject, 1, 2, b"operator stop".to_vec())
                    .unwrap(),
                2,
            )
            .unwrap();
        let command = RealtimeFrame::new(
            world,
            subject,
            1,
            3,
            3,
            RealtimeFrameKind::RoboticsCommand,
            vec![1],
        )
        .unwrap();
        assert_eq!(
            guard.admit(controller, command.clone(), 3),
            Err(RealtimeAdmissionError::SafetyStopLatched)
        );
        guard
            .admit(
                safety,
                RealtimeFrame::safety_resume(world, subject, 2, 3, b"clear".to_vec()).unwrap(),
                3,
            )
            .unwrap();
        assert!(guard.admit(controller, command, 3).is_ok());
    }

    #[test]
    fn direct_admission_rejects_wrong_lane_even_with_valid_signature() {
        let issuer_key = SecretKey::from_bytes(&[41u8; 32]);
        let safety = SecretKey::from_bytes(&[42u8; 32]).public();
        let controller = SecretKey::from_bytes(&[43u8; 32]).public();
        let (world, subject) = ids();
        let frame = RealtimeFrame::authority_lease(
            world,
            subject,
            1,
            1,
            AuthorityLeaseGrant {
                controller,
                valid_from_tick: 1,
                valid_until_tick: 10,
            },
        )
        .unwrap();
        let envelope = DirectEnvelope::sign(
            frame.encode().unwrap(),
            DirectLane::TELEMETRY,
            DirectDelivery::Reliable,
            &issuer_key,
            Uuid::from_u128(99),
            1,
            1_000,
            1_000,
        )
        .unwrap();
        let message = AuthenticatedDirectMessage::from(envelope);
        let mut guard = RealtimeAdmissionGuard::new(policy(issuer_key.public(), safety)).unwrap();
        assert!(matches!(
            guard.admit_direct(&message, 1),
            Err(RealtimeAdmissionError::LaneMismatch { .. })
        ));
    }
    #[test]
    fn world_clock_is_monotonic_and_control_frames_cannot_execute_early() {
        let issuer = SecretKey::from_bytes(&[36u8; 32]).public();
        let safety = SecretKey::from_bytes(&[37u8; 32]).public();
        let controller = SecretKey::from_bytes(&[38u8; 32]).public();
        let (world, subject) = ids();
        let mut guard = RealtimeAdmissionGuard::new(policy(issuer, safety)).unwrap();
        let future = RealtimeFrame::authority_lease(
            world,
            subject,
            1,
            11,
            AuthorityLeaseGrant {
                controller,
                valid_from_tick: 11,
                valid_until_tick: 20,
            },
        )
        .unwrap();
        assert!(matches!(
            guard.admit(issuer, future, 10),
            Err(RealtimeAdmissionError::FutureControlFrame { .. })
        ));
        assert_eq!(guard.current_world_tick(world), Some(10));
        assert!(matches!(
            guard.admit(
                issuer,
                RealtimeFrame::authority_revoke(world, subject, 2, 9).unwrap(),
                9,
            ),
            Err(RealtimeAdmissionError::WorldClockRegressed { .. })
        ));
    }

    #[test]
    fn expired_subjects_can_be_pruned_but_safety_stops_cannot() {
        let issuer = SecretKey::from_bytes(&[39u8; 32]).public();
        let safety = SecretKey::from_bytes(&[40u8; 32]).public();
        let controller = SecretKey::from_bytes(&[45u8; 32]).public();
        let (world, subject) = ids();
        let mut guard = RealtimeAdmissionGuard::new(policy(issuer, safety)).unwrap();
        let lease = RealtimeFrame::authority_lease(
            world,
            subject,
            1,
            1,
            AuthorityLeaseGrant {
                controller,
                valid_from_tick: 1,
                valid_until_tick: 2,
            },
        )
        .unwrap();
        guard.admit(issuer, lease, 1).unwrap();
        assert!(guard.prune_subject(world, subject, 3).unwrap());
        assert_eq!(guard.subject_count(), 0);
    }

    #[test]
    fn future_authority_handoff_is_staged_until_its_start_tick() {
        let issuer = SecretKey::from_bytes(&[46u8; 32]).public();
        let safety = SecretKey::from_bytes(&[47u8; 32]).public();
        let first = SecretKey::from_bytes(&[48u8; 32]).public();
        let second = SecretKey::from_bytes(&[49u8; 32]).public();
        let (world, subject) = ids();
        let mut guard = RealtimeAdmissionGuard::new(policy(issuer, safety)).unwrap();
        let initial = RealtimeFrame::authority_lease(
            world,
            subject,
            1,
            1,
            AuthorityLeaseGrant {
                controller: first,
                valid_from_tick: 1,
                valid_until_tick: 100,
            },
        )
        .unwrap();
        assert_eq!(
            guard.admit(issuer, initial, 1).unwrap(),
            RealtimeAdmissionOutcome::AuthorityLeaseInstalled
        );
        let handoff = RealtimeFrame::authority_lease(
            world,
            subject,
            2,
            10,
            AuthorityLeaseGrant {
                controller: second,
                valid_from_tick: 20,
                valid_until_tick: 200,
            },
        )
        .unwrap();
        assert_eq!(
            guard.admit(issuer, handoff, 10).unwrap(),
            RealtimeAdmissionOutcome::AuthorityLeaseStaged
        );
        assert_eq!(guard.lease(world, subject).unwrap().controller, Some(first));
        assert_eq!(
            guard.pending_lease(world, subject).unwrap().controller,
            Some(second)
        );

        let old_input = RealtimeFrame::new(
            world,
            subject,
            1,
            15,
            15,
            RealtimeFrameKind::PlayerInput,
            vec![1],
        )
        .unwrap();
        assert_eq!(
            guard.admit(first, old_input, 15).unwrap(),
            RealtimeAdmissionOutcome::OperationalFrameAccepted
        );
        let new_input = RealtimeFrame::new(
            world,
            subject,
            2,
            20,
            20,
            RealtimeFrameKind::PlayerInput,
            vec![2],
        )
        .unwrap();
        assert_eq!(
            guard.admit(second, new_input, 20).unwrap(),
            RealtimeAdmissionOutcome::OperationalFrameAccepted
        );
        assert_eq!(
            guard.lease(world, subject).unwrap().controller,
            Some(second)
        );
        assert!(guard.pending_lease(world, subject).is_none());
    }

    #[test]
    fn rejected_capacity_does_not_leak_world_clocks_and_prune_retires_world() {
        let issuer = SecretKey::from_bytes(&[50u8; 32]).public();
        let safety = SecretKey::from_bytes(&[51u8; 32]).public();
        let controller = SecretKey::from_bytes(&[52u8; 32]).public();
        let mut limited = policy(issuer, safety);
        limited.max_subjects = 1;
        let mut guard = RealtimeAdmissionGuard::new(limited).unwrap();
        let first_world = Uuid::from_u128(10);
        let first_subject = Uuid::from_u128(11);
        guard
            .admit(
                issuer,
                RealtimeFrame::authority_lease(
                    first_world,
                    first_subject,
                    1,
                    1,
                    AuthorityLeaseGrant {
                        controller,
                        valid_from_tick: 1,
                        valid_until_tick: 2,
                    },
                )
                .unwrap(),
                1,
            )
            .unwrap();

        let rejected_world = Uuid::from_u128(12);
        let rejected_subject = Uuid::from_u128(13);
        assert!(matches!(
            guard.admit(
                issuer,
                RealtimeFrame::authority_lease(
                    rejected_world,
                    rejected_subject,
                    1,
                    1,
                    AuthorityLeaseGrant {
                        controller,
                        valid_from_tick: 1,
                        valid_until_tick: 2,
                    },
                )
                .unwrap(),
                1,
            ),
            Err(RealtimeAdmissionError::SubjectCapacityReached { .. })
        ));
        assert_eq!(guard.current_world_tick(rejected_world), None);
        assert!(guard.prune_subject(first_world, first_subject, 3).unwrap());
        assert_eq!(guard.current_world_tick(first_world), None);
    }
}
