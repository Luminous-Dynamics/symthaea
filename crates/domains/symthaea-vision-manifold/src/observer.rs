//! Capability-restricted wrapper for frozen temporal observation.
//!
//! This wrapper intentionally exposes no learning, memory, FEP action-selection,
//! geodesic, dilation, or arbitrary checkpoint-mutation APIs. It is a temporal
//! observer only; its outputs are not pose probabilities or physical truth.

use crate::manifold::VisionManifold;
use crate::types::{ManifoldState, VisionConfig, VisionTelemetry};

/// A vision manifold configured as a frozen, non-retaining, non-acting observer.
///
/// The wrapped manifold is private so consumers cannot call its broader
/// cognitive/action APIs. Construction enforces the observer-only profile before
/// the first frame is processed.
pub struct ObserverOnlyVisionManifold {
    inner: VisionManifold,
}

impl ObserverOnlyVisionManifold {
    /// Create a checked observer-only manifold for fixed image dimensions.
    pub fn try_new(
        config: VisionConfig,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let mut inner = VisionManifold::try_new(config, width, height)?;
        inner.enable_observer_only()?;
        Ok(Self { inner })
    }

    /// Restore only a checkpoint that satisfies observer-only invariants.
    pub fn from_checkpoint(
        config: VisionConfig,
        width: u32,
        height: u32,
        state: &ManifoldState,
    ) -> Result<Self, String> {
        let mut observer = Self::try_new(config, width, height)?;
        observer.inner.load_state(state)?;
        if !observer.inner.is_observer_only() || !observer.inner.is_learning_frozen() {
            return Err("restored checkpoint is not observer-only and frozen".to_string());
        }
        Ok(observer)
    }

    /// Observe one frame through the checked path.
    ///
    /// Invalid input returns an error and does not expose a previous telemetry
    /// snapshot as if it represented the rejected frame.
    pub fn observe_frame_checked(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        dt: f32,
    ) -> Result<VisionTelemetry, String> {
        self.inner
            .observe_frame_checked(pixels, width, height, channels, dt)
    }

    /// Read the latest temporal telemetry. This remains representation-level
    /// evidence and must not be interpreted as pose covariance or probability.
    pub fn telemetry(&self) -> &VisionTelemetry {
        self.inner.telemetry()
    }

    /// Number of observations accepted by this profile.
    pub fn frame_count(&self) -> u64 {
        self.inner.frame_count()
    }

    /// Export a checkpoint that retains no raw frame bytes.
    pub fn save_state(&self) -> ManifoldState {
        self.inner.save_state()
    }

    /// Confirm the observer profile is still active.
    pub fn is_observer_only(&self) -> bool {
        self.inner.is_observer_only() && self.inner.is_learning_frozen()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapper_constructs_frozen_and_does_not_retain_raw_frame_bytes() {
        let mut observer =
            ObserverOnlyVisionManifold::try_new(VisionConfig::default(), 32, 32).unwrap();
        assert!(observer.is_observer_only());

        let pixels = vec![96_u8; 32 * 32];
        let telemetry = observer
            .observe_frame_checked(&pixels, 32, 32, 1, 0.033)
            .unwrap();

        assert_eq!(telemetry.last_fep_action, "OBSERVER_ONLY_DISABLED");
        let state = observer.save_state();
        assert!(state.observer_only);
        assert!(state.learning_frozen);
        assert!(state.last_observed_frame.is_none());
        assert!(state.scene_memory.is_none());
        assert!(state.object_memory.is_none());
        assert!(state.working_memory.is_none());
    }

    #[test]
    fn wrapper_rejects_invalid_frame_without_advancing_observation_count() {
        let mut observer =
            ObserverOnlyVisionManifold::try_new(VisionConfig::default(), 32, 32).unwrap();
        let invalid = vec![0_u8; 3];

        assert!(observer
            .observe_frame_checked(&invalid, 32, 32, 1, 0.033)
            .is_err());
        assert_eq!(observer.frame_count(), 0);
    }

    #[test]
    fn wrapper_restores_only_an_observer_checkpoint() {
        let observer =
            ObserverOnlyVisionManifold::try_new(VisionConfig::default(), 32, 32).unwrap();
        let checkpoint = observer.save_state();
        let restored = ObserverOnlyVisionManifold::from_checkpoint(
            VisionConfig::default(),
            32,
            32,
            &checkpoint,
        )
        .unwrap();
        assert!(restored.is_observer_only());

        let mut ordinary = VisionManifold::try_new(VisionConfig::default(), 32, 32).unwrap();
        ordinary
            .observe_frame_checked(&vec![128_u8; 32 * 32], 32, 32, 1, 0.033)
            .unwrap();
        assert!(ObserverOnlyVisionManifold::from_checkpoint(
            VisionConfig::default(),
            32,
            32,
            &ordinary.save_state(),
        )
        .is_err());
    }
}
