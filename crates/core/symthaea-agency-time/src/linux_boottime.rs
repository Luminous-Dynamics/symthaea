// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Linux suspend-aware live-time provider for bounded Agency deadlines.
//!
//! Deadline issuance and deadline verification are deliberately split:
//!
//! ```text
//! slow verifier plane -> LinuxBoottimeIssuer -> deadline
//! fast firewall       -> LinuxBoottimeVerifier -> current / reject
//! ```
//!
//! The verifier cannot mint, extend, or renew deadlines. Both handles share one
//! private runtime-time core so a verifier only accepts deadlines from the exact
//! runtime incarnation/profile it belongs to.

use std::fmt;
use std::fs::{self, File};
use std::os::unix::fs::MetadataExt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rustix::time::{clock_gettime, ClockId};

use crate::{
    DeadlineRejection, MonotonicLeaseDeadline, RuntimeIncarnationId, RuntimeTimeProfileId,
};

const NANOS_PER_SECOND: u64 = 1_000_000_000;
const LINUX_BOOTTIME_PROFILE_BYTES: [u8; 16] = *b"linux-boot-v1\0\0\0";
const TIME_NAMESPACE_PATH: &str = "/proc/self/ns/time";

/// Slow-plane Linux deadline issuer.
///
/// This handle owns the policy ceiling that permits minting a new live deadline.
/// It is intentionally not cloneable or serializable.
pub struct LinuxBoottimeIssuer {
    core: Arc<LinuxBoottimeCore>,
    max_lease_duration: Duration,
}

/// Fast-path Linux deadline verifier.
///
/// This type has no public constructor and no issuance API. It can only be
/// obtained from an issuer that owns the same private runtime-time core.
pub struct LinuxBoottimeVerifier {
    core: Arc<LinuxBoottimeCore>,
}

struct LinuxBoottimeCore {
    runtime_incarnation: RuntimeIncarnationId,
    profile_id: RuntimeTimeProfileId,
    process_id: u32,
    time_namespace: TimeNamespaceIdentity,
    /// Namespace FD retained solely to pin the exact nsfs object for the life
    /// of this runtime-time domain. This prevents namespace destruction/inode
    /// reuse from turning an unrelated future namespace into the same identity.
    _time_namespace_handle: File,
    state: Mutex<ProviderState>,
}

impl LinuxBoottimeIssuer {
    /// Create one new Linux boottime issuance domain.
    ///
    /// Reconstruction creates a fresh runtime incarnation even on the same boot.
    pub fn new(max_lease_duration: Duration) -> Result<Self, RuntimeTimeError> {
        if max_lease_duration.is_zero() {
            return Err(RuntimeTimeError::InvalidMaximumDuration);
        }
        duration_ns(max_lease_duration)?;

        let mut incarnation = [0u8; 32];
        getrandom::getrandom(&mut incarnation)
            .map_err(|_| RuntimeTimeError::RandomnessUnavailable)?;
        if incarnation == [0; 32] {
            return Err(RuntimeTimeError::RandomnessUnavailable);
        }

        let (time_namespace_handle, time_namespace) = open_current_time_namespace()?;
        let core = LinuxBoottimeCore {
            runtime_incarnation: RuntimeIncarnationId(incarnation),
            profile_id: RuntimeTimeProfileId(LINUX_BOOTTIME_PROFILE_BYTES),
            process_id: std::process::id(),
            time_namespace,
            _time_namespace_handle: time_namespace_handle,
            state: Mutex::new(ProviderState {
                last_coordinate_ns: boottime_ns()?,
                fault: None,
            }),
        };

        Ok(Self {
            core: Arc::new(core),
            max_lease_duration,
        })
    }

    pub const fn max_lease_duration(&self) -> Duration {
        self.max_lease_duration
    }

    pub fn runtime_incarnation(&self) -> RuntimeIncarnationId {
        self.core.runtime_incarnation
    }

    pub fn profile_id(&self) -> RuntimeTimeProfileId {
        self.core.profile_id
    }

    /// Obtain a verifier-only handle to this exact private runtime-time core.
    /// The returned handle cannot issue deadlines.
    pub fn verifier(&self) -> LinuxBoottimeVerifier {
        LinuxBoottimeVerifier {
            core: Arc::clone(&self.core),
        }
    }

    /// Issue one deadline from provider-owned `CLOCK_BOOTTIME` currentness.
    ///
    /// The caller supplies only a requested duration. The issuer rejects zero,
    /// widened, or overflowing requests instead of silently clamping them.
    pub fn issue_deadline(
        &self,
        requested_duration: Duration,
    ) -> Result<MonotonicLeaseDeadline, RuntimeTimeError> {
        if requested_duration.is_zero() {
            return Err(RuntimeTimeError::ZeroRequestedDuration);
        }
        if requested_duration > self.max_lease_duration {
            return Err(RuntimeTimeError::RequestedDurationExceedsMaximum);
        }

        let requested_ns = duration_ns(requested_duration)?;
        let now = self.core.observe_current()?;
        let deadline_coordinate = now
            .checked_add(requested_ns)
            .ok_or(RuntimeTimeError::DeadlineOverflow)?;

        Ok(MonotonicLeaseDeadline {
            runtime_incarnation: self.core.runtime_incarnation,
            profile_id: self.core.profile_id,
            deadline_coordinate,
        })
    }
}

impl LinuxBoottimeVerifier {
    pub fn runtime_incarnation(&self) -> RuntimeIncarnationId {
        self.core.runtime_incarnation
    }

    pub fn profile_id(&self) -> RuntimeTimeProfileId {
        self.core.profile_id
    }

    /// Re-evaluate one deadline against the live provider immediately.
    ///
    /// `Ok(())` is deliberately ephemeral call-site evidence, not a reusable
    /// positive token. The consuming firewall must call again at point of use.
    pub fn require_current(
        &self,
        deadline: &MonotonicLeaseDeadline,
    ) -> Result<(), DeadlineRejection> {
        if deadline.runtime_incarnation != self.core.runtime_incarnation {
            return Err(DeadlineRejection::RuntimeIncarnationMismatch);
        }
        if deadline.profile_id != self.core.profile_id {
            return Err(DeadlineRejection::ProfileMismatch);
        }

        let now = self
            .core
            .observe_current()
            .map_err(|_| DeadlineRejection::Indeterminate)?;
        if now >= deadline.deadline_coordinate {
            return Err(DeadlineRejection::Expired);
        }
        Ok(())
    }
}

impl LinuxBoottimeCore {
    fn observe_current(&self) -> Result<u64, RuntimeTimeError> {
        // Fork/inherited-process mismatch must be checked before touching the
        // inherited mutex. A child can inherit a mutex held by a vanished thread.
        if std::process::id() != self.process_id {
            return Err(RuntimeTimeError::ProviderFaultLatched);
        }

        let observed_namespace = current_time_namespace()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| RuntimeTimeError::ProviderStatePoisoned)?;

        if state.fault.is_some() {
            return Err(RuntimeTimeError::ProviderFaultLatched);
        }
        if observed_namespace != self.time_namespace {
            state.fault = Some(LatchedFault::TimeNamespaceChanged);
            return Err(RuntimeTimeError::ProviderFaultLatched);
        }

        let observed_coordinate = match boottime_ns() {
            Ok(value) => value,
            Err(error) => {
                state.fault = Some(LatchedFault::ClockInvalid);
                return Err(error);
            }
        };

        state.accept_coordinate(observed_coordinate)?;
        Ok(observed_coordinate)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TimeNamespaceIdentity {
    device: u64,
    inode: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LatchedFault {
    TimeNamespaceChanged,
    ClockRollback,
    ClockInvalid,
}

struct ProviderState {
    last_coordinate_ns: u64,
    fault: Option<LatchedFault>,
}

impl ProviderState {
    fn accept_coordinate(&mut self, observed_coordinate: u64) -> Result<(), RuntimeTimeError> {
        if self.fault.is_some() {
            return Err(RuntimeTimeError::ProviderFaultLatched);
        }
        if observed_coordinate < self.last_coordinate_ns {
            self.fault = Some(LatchedFault::ClockRollback);
            return Err(RuntimeTimeError::ProviderFaultLatched);
        }
        self.last_coordinate_ns = observed_coordinate;
        Ok(())
    }
}

fn namespace_identity(metadata: &fs::Metadata) -> TimeNamespaceIdentity {
    TimeNamespaceIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    }
}

fn open_current_time_namespace() -> Result<(File, TimeNamespaceIdentity), RuntimeTimeError> {
    let handle = File::open(TIME_NAMESPACE_PATH).map_err(RuntimeTimeError::TimeNamespaceRead)?;
    let metadata = handle
        .metadata()
        .map_err(RuntimeTimeError::TimeNamespaceRead)?;
    let identity = namespace_identity(&metadata);
    Ok((handle, identity))
}

fn current_time_namespace() -> Result<TimeNamespaceIdentity, RuntimeTimeError> {
    let metadata = fs::metadata(TIME_NAMESPACE_PATH).map_err(RuntimeTimeError::TimeNamespaceRead)?;
    Ok(namespace_identity(&metadata))
}

fn boottime_ns() -> Result<u64, RuntimeTimeError> {
    let value = clock_gettime(ClockId::Boottime);
    let seconds =
        u64::try_from(value.tv_sec).map_err(|_| RuntimeTimeError::ClockCoordinateInvalid)?;
    let nanos =
        u64::try_from(value.tv_nsec).map_err(|_| RuntimeTimeError::ClockCoordinateInvalid)?;
    if nanos >= NANOS_PER_SECOND {
        return Err(RuntimeTimeError::ClockCoordinateInvalid);
    }
    seconds
        .checked_mul(NANOS_PER_SECOND)
        .and_then(|value| value.checked_add(nanos))
        .ok_or(RuntimeTimeError::ClockCoordinateInvalid)
}

fn duration_ns(duration: Duration) -> Result<u64, RuntimeTimeError> {
    u64::try_from(duration.as_nanos()).map_err(|_| RuntimeTimeError::DurationOverflow)
}

/// Linux runtime-time construction/evaluation failure.
/// No variant carries a raw clock coordinate.
#[derive(Debug)]
pub enum RuntimeTimeError {
    InvalidMaximumDuration,
    ZeroRequestedDuration,
    RequestedDurationExceedsMaximum,
    DurationOverflow,
    DeadlineOverflow,
    RandomnessUnavailable,
    TimeNamespaceRead(std::io::Error),
    ClockCoordinateInvalid,
    ProviderStatePoisoned,
    ProviderFaultLatched,
}

impl fmt::Display for RuntimeTimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMaximumDuration => write!(
                f,
                "maximum runtime lease duration must be positive and representable"
            ),
            Self::ZeroRequestedDuration => {
                write!(f, "requested runtime lease duration must be positive")
            }
            Self::RequestedDurationExceedsMaximum => write!(
                f,
                "requested runtime lease duration exceeds configured maximum"
            ),
            Self::DurationOverflow => {
                write!(f, "runtime lease duration is not representable in nanoseconds")
            }
            Self::DeadlineOverflow => write!(f, "runtime lease deadline coordinate overflowed"),
            Self::RandomnessUnavailable => {
                write!(f, "runtime incarnation randomness is unavailable")
            }
            Self::TimeNamespaceRead(_) => {
                write!(f, "Linux time namespace identity is unavailable")
            }
            Self::ClockCoordinateInvalid => {
                write!(f, "Linux CLOCK_BOOTTIME returned an invalid coordinate")
            }
            Self::ProviderStatePoisoned => write!(f, "runtime-time provider state is poisoned"),
            Self::ProviderFaultLatched => write!(f, "runtime-time provider fault is latched"),
        }
    }
}

impl std::error::Error for RuntimeTimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TimeNamespaceRead(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issuer_mints_and_verifier_checks_same_domain_deadline() {
        let issuer = LinuxBoottimeIssuer::new(Duration::from_secs(1)).unwrap();
        let verifier = issuer.verifier();
        let deadline = issuer
            .issue_deadline(Duration::from_millis(500))
            .unwrap();

        assert_eq!(issuer.runtime_incarnation(), verifier.runtime_incarnation());
        assert_eq!(issuer.profile_id(), verifier.profile_id());
        assert_eq!(deadline.runtime_incarnation(), verifier.runtime_incarnation());
        assert!(verifier.require_current(&deadline).is_ok());
    }

    #[test]
    fn zero_and_widened_duration_requests_fail_closed() {
        assert!(matches!(
            LinuxBoottimeIssuer::new(Duration::ZERO),
            Err(RuntimeTimeError::InvalidMaximumDuration)
        ));

        let issuer = LinuxBoottimeIssuer::new(Duration::from_millis(10)).unwrap();
        assert!(matches!(
            issuer.issue_deadline(Duration::ZERO),
            Err(RuntimeTimeError::ZeroRequestedDuration)
        ));
        assert!(matches!(
            issuer.issue_deadline(Duration::from_millis(11)),
            Err(RuntimeTimeError::RequestedDurationExceedsMaximum)
        ));
    }

    #[test]
    fn same_profile_different_runtime_domain_rejects() {
        let first = LinuxBoottimeIssuer::new(Duration::from_secs(1)).unwrap();
        let second = LinuxBoottimeIssuer::new(Duration::from_secs(1)).unwrap();
        let deadline = first
            .issue_deadline(Duration::from_millis(500))
            .unwrap();

        assert_eq!(first.profile_id(), second.profile_id());
        assert_ne!(first.runtime_incarnation(), second.runtime_incarnation());
        assert_eq!(
            second.verifier().require_current(&deadline),
            Err(DeadlineRejection::RuntimeIncarnationMismatch)
        );
    }

    #[test]
    fn wrong_profile_and_expired_coordinate_are_distinct_rejections() {
        let issuer = LinuxBoottimeIssuer::new(Duration::from_secs(1)).unwrap();
        let verifier = issuer.verifier();

        let mut wrong_profile = issuer
            .issue_deadline(Duration::from_millis(500))
            .unwrap();
        wrong_profile.profile_id = RuntimeTimeProfileId([9; 16]);
        assert_eq!(
            verifier.require_current(&wrong_profile),
            Err(DeadlineRejection::ProfileMismatch)
        );

        let mut expired = issuer
            .issue_deadline(Duration::from_millis(500))
            .unwrap();
        expired.deadline_coordinate = 0;
        assert_eq!(
            verifier.require_current(&expired),
            Err(DeadlineRejection::Expired)
        );
    }

    #[test]
    fn rollback_latches_provider_state() {
        let mut state = ProviderState {
            last_coordinate_ns: 100,
            fault: None,
        };
        assert!(state.accept_coordinate(101).is_ok());
        assert!(matches!(
            state.accept_coordinate(99),
            Err(RuntimeTimeError::ProviderFaultLatched)
        ));
        assert_eq!(state.fault, Some(LatchedFault::ClockRollback));
        assert!(matches!(
            state.accept_coordinate(200),
            Err(RuntimeTimeError::ProviderFaultLatched)
        ));
    }

    #[test]
    fn namespace_handle_and_identity_are_bound_to_same_nsfs_object() {
        let (handle, identity) = open_current_time_namespace().unwrap();
        let handle_identity = namespace_identity(&handle.metadata().unwrap());
        let current_identity = current_time_namespace().unwrap();
        assert_eq!(identity, handle_identity);
        assert_eq!(identity, current_identity);
        assert_ne!(identity.inode, 0);
    }

    #[test]
    fn fixed_profile_is_versioned_and_not_runtime_selected() {
        let issuer = LinuxBoottimeIssuer::new(Duration::from_secs(1)).unwrap();
        assert_eq!(issuer.profile_id().as_bytes(), &LINUX_BOOTTIME_PROFILE_BYTES);
    }

    #[test]
    fn internal_duration_conversion_rejects_overflow() {
        assert!(matches!(
            duration_ns(Duration::MAX),
            Err(RuntimeTimeError::DurationOverflow)
        ));
    }
}
