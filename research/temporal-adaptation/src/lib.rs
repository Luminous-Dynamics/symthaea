//! Candidate-only adaptive pressure signals for non-fungible temporal resources.
//!
//! Each resource receives its own bounded shadow price. The prices are not
//! summed across units and do not replace hard feasibility checks in the
//! joint-plan selector. Subsystem policy generators may consult only the
//! coordinate corresponding to the resource they consume.

#![forbid(unsafe_code)]

use std::fmt;
use symthaea_temporal_joint_control::{ResourceDimension, ResourceVector};

const PPM: u128 = 1_000_000;
const MAX_OBSERVED_UTILIZATION_PPM: u64 = 1_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourcePriceConfig {
    /// Desired utilization of each resource, in parts per million of its cap.
    pub target_utilization_ppm: [u64; 12],
    /// EWMA weight assigned to the newest completed episode.
    pub smoothing_alpha_ppm: u64,
    /// Price learning rate per unit of normalized utilization error.
    pub step_size_ppm: u64,
    /// Maximum resource-specific shadow price.
    pub maximum_price_ppm: u64,
    /// Cap for measured utilization ratios. This bounds arithmetic and the
    /// influence of one malformed or catastrophic observation.
    pub maximum_observed_utilization_ppm: u64,
}

impl ResourcePriceConfig {
    pub fn validate(self) -> Result<(), PriceError> {
        if self.target_utilization_ppm.iter().any(|target| *target > 1_000_000)
            || self.smoothing_alpha_ppm == 0
            || self.smoothing_alpha_ppm > 1_000_000
            || self.step_size_ppm == 0
            || self.maximum_price_ppm == 0
            || self.target_utilization_ppm.iter().any(|target| {
                self.maximum_observed_utilization_ppm <= *target
            })
            || self.maximum_observed_utilization_ppm > MAX_OBSERVED_UTILIZATION_PPM
        {
            return Err(PriceError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourcePriceSnapshot {
    pub prices_ppm: [u64; 12],
    pub smoothed_utilization_ppm: [u64; 12],
    pub completed_episodes: u64,
    pub config: Option<ResourcePriceConfig>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourcePriceState {
    prices_ppm: [u64; 12],
    smoothed_utilization_ppm: [u64; 12],
    completed_episodes: u64,
    config: Option<ResourcePriceConfig>,
}

impl Default for ResourcePriceState {
    fn default() -> Self {
        Self {
            prices_ppm: [0; 12],
            smoothed_utilization_ppm: [0; 12],
            completed_episodes: 0,
            config: None,
        }
    }
}

impl ResourcePriceState {
    pub fn price_ppm(&self, dimension: ResourceDimension) -> u64 {
        let index = ResourceDimension::ALL
            .iter()
            .position(|candidate| *candidate == dimension)
            .expect("resource dimension is part of the fixed registry");
        self.prices_ppm[index]
    }

    pub fn smoothed_utilization_ppm(&self, dimension: ResourceDimension) -> u64 {
        let index = ResourceDimension::ALL
            .iter()
            .position(|candidate| *candidate == dimension)
            .expect("resource dimension is part of the fixed registry");
        self.smoothed_utilization_ppm[index]
    }

    pub const fn completed_episodes(&self) -> u64 {
        self.completed_episodes
    }

    pub const fn snapshot(&self) -> ResourcePriceSnapshot {
        ResourcePriceSnapshot {
            prices_ppm: self.prices_ppm,
            smoothed_utilization_ppm: self.smoothed_utilization_ppm,
            completed_episodes: self.completed_episodes,
            config: self.config,
        }
    }

    pub fn restore(
        snapshot: ResourcePriceSnapshot,
        config: ResourcePriceConfig,
    ) -> Result<Self, PriceError> {
        config.validate()?;
        let uninitialized = snapshot.completed_episodes == 0
            && snapshot.prices_ppm == [0; 12]
            && snapshot.smoothed_utilization_ppm == [0; 12]
            && snapshot.config.is_none();
        if snapshot.config != Some(config) && !uninitialized {
            return Err(PriceError::ConfigMismatch);
        }
        if snapshot.prices_ppm.iter().any(|price| *price > config.maximum_price_ppm)
            || snapshot.smoothed_utilization_ppm.iter().any(|value| {
                *value > config.maximum_observed_utilization_ppm
            })
        {
            return Err(PriceError::SnapshotOutOfRange);
        }
        Ok(Self {
            prices_ppm: snapshot.prices_ppm,
            smoothed_utilization_ppm: snapshot.smoothed_utilization_ppm,
            completed_episodes: snapshot.completed_episodes,
            config: Some(config),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourcePriceUpdate {
    pub dimension: ResourceDimension,
    pub measured_usage: u64,
    pub hard_budget: u64,
    pub instant_utilization_ppm: u64,
    pub smoothed_utilization_ppm: u64,
    pub previous_price_ppm: u64,
    pub updated_price_ppm: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourcePriceReceipt {
    pub completed_episode: u64,
    pub config: ResourcePriceConfig,
    pub updates: Vec<ResourcePriceUpdate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PriceError {
    InvalidConfig,
    ConfigMismatch,
    SnapshotOutOfRange,
    ArithmeticOverflow,
}

impl fmt::Display for PriceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "resource price configuration is invalid"),
            Self::ConfigMismatch => write!(f, "resource price configuration changed without an explicit state migration"),
            Self::SnapshotOutOfRange => write!(f, "persisted resource price snapshot exceeds configured bounds"),
            Self::ArithmeticOverflow => write!(f, "resource price update arithmetic overflowed"),
        }
    }
}

impl std::error::Error for PriceError {}

/// Update independent bounded resource prices from one completed episode.
///
/// This is a projected, sign-preserving subgradient-style update over normalized
/// utilization error. It is a control heuristic, not a proof of dual convergence.
/// Hard per-resource limits must still be enforced by select_joint_plan.
pub fn update_resource_prices(
    state: &mut ResourcePriceState,
    measured: ResourceVector,
    budget: ResourceVector,
    config: ResourcePriceConfig,
) -> Result<ResourcePriceReceipt, PriceError> {
    config.validate()?;
    if state.config.is_some_and(|active| active != config) {
        return Err(PriceError::ConfigMismatch);
    }
    let next_episode = state.completed_episodes
        .checked_add(1)
        .ok_or(PriceError::ArithmeticOverflow)?;
    let usage = measured.values();
    let limits = budget.values();
    let mut updates = Vec::with_capacity(ResourceDimension::ALL.len());

    for (index, dimension) in ResourceDimension::ALL.into_iter().enumerate() {
        let instant = utilization_ppm(
            usage[index],
            limits[index],
            config.maximum_observed_utilization_ppm,
        );
        let previous_smoothed = state.smoothed_utilization_ppm[index];
        let smoothed = if state.completed_episodes == 0 {
            instant
        } else {
            ewma(previous_smoothed, instant, config.smoothing_alpha_ppm)?
        };
        let previous_price = state.prices_ppm[index];
        let updated_price = update_one_price(
            previous_price,
            smoothed,
            config.target_utilization_ppm[index],
            config.step_size_ppm,
            config.maximum_price_ppm,
        )?;

        state.smoothed_utilization_ppm[index] = smoothed;
        state.prices_ppm[index] = updated_price;
        updates.push(ResourcePriceUpdate {
            dimension,
            measured_usage: usage[index],
            hard_budget: limits[index],
            instant_utilization_ppm: instant,
            smoothed_utilization_ppm: smoothed,
            previous_price_ppm: previous_price,
            updated_price_ppm: updated_price,
        });
    }

    state.completed_episodes = next_episode;
    state.config = Some(config);

    Ok(ResourcePriceReceipt {
        completed_episode: state.completed_episodes,
        config,
        updates,
    })
}

fn utilization_ppm(usage: u64, budget: u64, maximum: u64) -> u64 {
    if budget == 0 {
        return if usage == 0 { 0 } else { maximum };
    }
    let ratio = u128::from(usage)
        .saturating_mul(PPM)
        / u128::from(budget);
    u64::try_from(ratio.min(u128::from(maximum))).unwrap_or(maximum)
}

fn ewma(previous: u64, current: u64, alpha_ppm: u64) -> Result<u64, PriceError> {
    let old_weight = 1_000_000_u64
        .checked_sub(alpha_ppm)
        .ok_or(PriceError::InvalidConfig)?;
    let weighted = u128::from(previous)
        .checked_mul(u128::from(old_weight))
        .and_then(|value| {
            u128::from(current)
                .checked_mul(u128::from(alpha_ppm))
                .and_then(|new_value| value.checked_add(new_value))
        })
        .ok_or(PriceError::ArithmeticOverflow)?;
    let rounded = weighted
        .checked_add(PPM / 2)
        .ok_or(PriceError::ArithmeticOverflow)?
        / PPM;
    u64::try_from(rounded).map_err(|_| PriceError::ArithmeticOverflow)
}

fn update_one_price(
    previous: u64,
    utilization: u64,
    target: u64,
    step_size: u64,
    maximum: u64,
) -> Result<u64, PriceError> {
    if utilization == target {
        return Ok(previous.min(maximum));
    }
    let difference = utilization.abs_diff(target);
    let scaled = u128::from(step_size)
        .checked_mul(u128::from(difference))
        .ok_or(PriceError::ArithmeticOverflow)?;
    let mut delta = scaled
        .checked_add(PPM - 1)
        .ok_or(PriceError::ArithmeticOverflow)?
        / PPM;
    if delta == 0 {
        delta = 1;
    }
    let delta = u64::try_from(delta.min(u128::from(u64::MAX)))
        .map_err(|_| PriceError::ArithmeticOverflow)?;
    if utilization > target {
        Ok(previous.saturating_add(delta).min(maximum))
    } else {
        Ok(previous.saturating_sub(delta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ResourcePriceConfig {
        ResourcePriceConfig {
            target_utilization_ppm: [700_000; 12],
            smoothing_alpha_ppm: 1_000_000,
            step_size_ppm: 100_000,
            maximum_price_ppm: 1_000_000,
            maximum_observed_utilization_ppm: 2_000_000,
        }
    }

    #[test]
    fn over_target_utilization_increases_only_corresponding_price() {
        let mut state = ResourcePriceState::default();
        let receipt = update_resource_prices(
            &mut state,
            ResourceVector { observations: 9, ..ResourceVector::default() },
            ResourceVector { observations: 10, ..ResourceVector::default() },
            config(),
        ).unwrap();
        assert!(state.price_ppm(ResourceDimension::Observations) > 0);
        assert_eq!(state.price_ppm(ResourceDimension::PersistentBytes), 0);
        assert_eq!(receipt.updates.len(), 12);
    }

    #[test]
    fn under_target_utilization_decreases_price() {
        let mut state = ResourcePriceState {
            prices_ppm: [50_000; 12],
            ..ResourcePriceState::default()
        };
        update_resource_prices(
            &mut state,
            ResourceVector::default(),
            ResourceVector { observations: 10, ..ResourceVector::default() },
            config(),
        ).unwrap();
        assert!(state.price_ppm(ResourceDimension::Observations) < 50_000);
    }

    #[test]
    fn ewma_smooths_episode_to_episode_resource_pressure() {
        let mut state = ResourcePriceState::default();
        let mut cfg = config();
        cfg.smoothing_alpha_ppm = 500_000;
        let budget = ResourceVector { observations: 10, ..ResourceVector::default() };
        update_resource_prices(
            &mut state,
            ResourceVector { observations: 10, ..ResourceVector::default() },
            budget,
            cfg,
        ).unwrap();
        update_resource_prices(
            &mut state,
            ResourceVector::default(),
            budget,
            cfg,
        ).unwrap();
        assert_eq!(
            state.smoothed_utilization_ppm(ResourceDimension::Observations),
            500_000
        );
    }

    #[test]
    fn each_resource_can_have_its_own_utilization_target() {
        let mut state = ResourcePriceState::default();
        let mut cfg = config();
        cfg.target_utilization_ppm[0] = 900_000;
        cfg.target_utilization_ppm[6] = 500_000;
        let measured = ResourceVector {
            persistent_bytes: 800,
            observations: 800,
            ..ResourceVector::default()
        };
        let budget = ResourceVector {
            persistent_bytes: 1_000,
            observations: 1_000,
            ..ResourceVector::default()
        };
        update_resource_prices(&mut state, measured, budget, cfg).unwrap();
        assert_eq!(state.price_ppm(ResourceDimension::PersistentBytes), 0);
        assert!(state.price_ppm(ResourceDimension::Observations) > 0);
    }

    #[test]
    fn zero_budget_with_positive_usage_saturates_pressure_and_price() {
        let mut state = ResourcePriceState::default();
        let mut cfg = config();
        cfg.maximum_observed_utilization_ppm = 2_000_000;
        cfg.maximum_price_ppm = 10_000;
        update_resource_prices(
            &mut state,
            ResourceVector { energy_nj: 1, ..ResourceVector::default() },
            ResourceVector::default(),
            cfg,
        ).unwrap();
        assert_eq!(
            state.smoothed_utilization_ppm(ResourceDimension::EnergyNanojoules),
            2_000_000
        );
        assert_eq!(state.price_ppm(ResourceDimension::EnergyNanojoules), 10_000);
    }

    #[test]
    fn zero_usage_and_zero_budget_is_valid_and_decays_pressure() {
        let mut state = ResourcePriceState {
            prices_ppm: [100; 12],
            ..ResourcePriceState::default()
        };
        update_resource_prices(
            &mut state,
            ResourceVector::default(),
            ResourceVector::default(),
            config(),
        ).unwrap();
        assert_eq!(state.smoothed_utilization_ppm(ResourceDimension::EnergyNanojoules), 0);
        assert_eq!(state.price_ppm(ResourceDimension::EnergyNanojoules), 0);
    }

    #[test]
    fn price_saturates_at_configured_bound() {
        let mut state = ResourcePriceState {
            prices_ppm: [999_999; 12],
            ..ResourcePriceState::default()
        };
        update_resource_prices(
            &mut state,
            ResourceVector { observations: 100, ..ResourceVector::default() },
            ResourceVector { observations: 1, ..ResourceVector::default() },
            config(),
        ).unwrap();
        assert_eq!(state.price_ppm(ResourceDimension::Observations), 1_000_000);
    }

    #[test]
    fn update_is_deterministic_from_identical_state_and_episode_costs() {
        let mut left = ResourcePriceState::default();
        let mut right = ResourcePriceState::default();
        let usage = ResourceVector { compute_us: 80, energy_nj: 50, ..ResourceVector::default() };
        let budget = ResourceVector { compute_us: 100, energy_nj: 100, ..ResourceVector::default() };
        let a = update_resource_prices(&mut left, usage, budget, config()).unwrap();
        let b = update_resource_prices(&mut right, usage, budget, config()).unwrap();
        assert_eq!(a, b);
        assert_eq!(left, right);
    }

    #[test]
    fn snapshot_round_trip_preserves_all_adaptive_state() {
        let mut state = ResourcePriceState::default();
        update_resource_prices(
            &mut state,
            ResourceVector { observations: 9, ..ResourceVector::default() },
            ResourceVector { observations: 10, ..ResourceVector::default() },
            config(),
        ).unwrap();
        let restored = ResourcePriceState::restore(state.snapshot(), config()).unwrap();
        assert_eq!(restored, state);
    }

    #[test]
    fn snapshot_rejects_unannounced_configuration_change() {
        let mut state = ResourcePriceState::default();
        update_resource_prices(
            &mut state,
            ResourceVector::default(),
            ResourceVector::default(),
            config(),
        ).unwrap();
        let mut changed = config();
        changed.step_size_ppm += 1;
        let before = state;
        assert_eq!(
            update_resource_prices(&mut state, ResourceVector::default(), ResourceVector::default(), changed),
            Err(PriceError::ConfigMismatch)
        );
        assert_eq!(state, before);
        assert_eq!(
            ResourcePriceState::restore(state.snapshot(), changed),
            Err(PriceError::ConfigMismatch)
        );
    }

    #[test]
    fn out_of_range_snapshot_is_rejected() {
        let snapshot = ResourcePriceSnapshot {
            prices_ppm: [u64::MAX; 12],
            smoothed_utilization_ppm: [0; 12],
            completed_episodes: 1,
        };
        assert_eq!(
            ResourcePriceState::restore(snapshot, config()),
            Err(PriceError::SnapshotOutOfRange)
        );
    }

    #[test]
    fn invalid_config_and_episode_counter_overflow_fail_closed() {
        let mut state = ResourcePriceState::default();
        let mut invalid = config();
        invalid.target_utilization_ppm[0] = 1_000_001;
        assert_eq!(
            update_resource_prices(&mut state, ResourceVector::default(), ResourceVector::default(), invalid),
            Err(PriceError::InvalidConfig)
        );
        state.completed_episodes = u64::MAX;
        assert_eq!(
            update_resource_prices(&mut state, ResourceVector::default(), ResourceVector::default(), config()),
            Err(PriceError::ArithmeticOverflow)
        );
    }
}
