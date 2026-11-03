// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Monotonic generation epochs for invalidating queued live musical output.
//!
//! This primitive fences application-owned output queues. It cannot recall
//! samples already copied into an in-flight device callback or submitted to
//! the operating system / physical audio device.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

/// Wire/protocol identity for the first output-epoch contract.
pub const OUTPUT_EPOCH_PROTOCOL_V1: &str = "mel-output-epoch-v1";

/// Failure to advance an output epoch without wrapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputEpochError {
    /// Reusing an earlier epoch could admit stale output, so the clock fails
    /// closed rather than wrapping at u64::MAX.
    Exhausted,
}

impl std::fmt::Display for OutputEpochError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exhausted => write!(f, "output generation epoch exhausted"),
        }
    }
}

impl std::error::Error for OutputEpochError {}

/// Shared, monotonic output-generation epoch.
///
/// The control thread advances this clock. The real-time callback reads it
/// with acquire semantics and admits only frames tagged with the exact current
/// epoch. Clones share the same atomic clock.
#[derive(Debug, Clone)]
pub struct OutputEpochClock {
    current: Arc<AtomicU64>,
}

impl Default for OutputEpochClock {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputEpochClock {
    /// Construct a new clock at epoch zero.
    pub fn new() -> Self {
        Self {
            current: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Read the currently authorized output epoch.
    pub fn current(&self) -> u64 {
        self.current.load(Ordering::Acquire)
    }

    /// Advance the epoch atomically, refusing integer wraparound.
    ///
    /// A frame tagged with the previous epoch becomes ineligible as soon as
    /// this operation's atomic update is visible to the output callback.
    pub fn invalidate(&self) -> Result<u64, OutputEpochError> {
        self.current
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |epoch| {
                epoch.checked_add(1)
            })
            .map(|previous| previous + 1)
            .map_err(|_| OutputEpochError::Exhausted)
    }

    /// Whether a queued frame belongs to the currently authorized epoch.
    pub fn accepts(&self, frame_epoch: u64) -> bool {
        self.current() == frame_epoch
    }
}

/// One stereo sample frame tagged with the generation that produced it.
///
/// The tag is attached before the rendered chunk is enqueued. A chunk that
/// began rendering before an invalidation therefore remains stale even if its
/// producer finishes and enqueues it afterward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochTaggedStereoFrame {
    pub epoch: u64,
    pub samples: [f32; 2],
}

impl EpochTaggedStereoFrame {
    pub fn new(epoch: u64, samples: [f32; 2]) -> Self {
        Self { epoch, samples }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_epoch_frame_is_accepted_unchanged() {
        let clock = OutputEpochClock::new();
        let frame = EpochTaggedStereoFrame::new(0, [0.25, -0.5]);

        assert!(clock.accepts(frame.epoch));
        assert_eq!(frame.samples, [0.25, -0.5]);
    }

    #[test]
    fn invalidation_rejects_every_prior_epoch() {
        let clock = OutputEpochClock::new();
        let old = EpochTaggedStereoFrame::new(0, [0.25, -0.5]);

        assert_eq!(clock.invalidate(), Ok(1));
        assert!(!clock.accepts(old.epoch));
        assert!(clock.accepts(1));
    }

    #[test]
    fn repeated_invalidation_never_reuses_an_epoch() {
        let clock = OutputEpochClock::new();

        assert_eq!(clock.invalidate(), Ok(1));
        assert_eq!(clock.invalidate(), Ok(2));
        assert_eq!(clock.invalidate(), Ok(3));
        assert!(!clock.accepts(0));
        assert!(!clock.accepts(1));
        assert!(!clock.accepts(2));
        assert!(clock.accepts(3));
    }

    #[test]
    fn render_started_before_invalidation_stays_stale_after_enqueue() {
        let clock = OutputEpochClock::new();
        let render_epoch = clock.current();

        // Model a render chunk that was already in flight when invalidation
        // occurred, but whose frames are enqueued afterward.
        assert_eq!(clock.invalidate(), Ok(1));
        let late_enqueued_frame = EpochTaggedStereoFrame::new(render_epoch, [1.0, 1.0]);

        assert!(!clock.accepts(late_enqueued_frame.epoch));
    }

    #[test]
    fn stale_frame_does_not_make_a_later_current_frame_stale() {
        let clock = OutputEpochClock::new();
        let stale = EpochTaggedStereoFrame::new(0, [1.0, 1.0]);
        assert_eq!(clock.invalidate(), Ok(1));
        let current = EpochTaggedStereoFrame::new(1, [0.0, 0.5]);

        assert!(!clock.accepts(stale.epoch));
        assert!(clock.accepts(current.epoch));
    }

    #[test]
    fn exhaustion_fails_closed_without_wrapping() {
        let clock = OutputEpochClock {
            current: Arc::new(AtomicU64::new(u64::MAX)),
        };

        assert_eq!(clock.invalidate(), Err(OutputEpochError::Exhausted));
        assert_eq!(clock.current(), u64::MAX);
    }
}
