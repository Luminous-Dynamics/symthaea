// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reward functions for vehicle training episodes.
//!
//! These are NOT traditional RL rewards. They are the "prediction accuracy" signal
//! that feeds into the Active Inference loop. The vehicle's prior is a state of
//! safe, forward momentum. Deviations from that prior generate prediction error
//! (free energy), which the LTC controller learns to minimize.

use crate::types::{VehicleState, VehicleTask};

/// Core safety reward: penalizes dangerous states.
///
/// Returns 0.0 for catastrophic failure, 1.0 for perfectly safe.
pub fn safety_reward(state: &VehicleState) -> f64 {
    let mut reward = 1.0;

    // Tire slip penalty: exponential decay above 0.1 rad
    let max_slip = state.tire_slip_front.max(state.tire_slip_rear);
    if max_slip > 0.1 {
        reward *= (-3.0 * (max_slip - 0.1)).exp();
    }

    // Yaw rate penalty: should be small for stable driving
    reward *= (-0.5 * state.yaw_rate.abs()).exp();

    // Lateral velocity penalty: sideslip is dangerous
    reward *= (-0.3 * state.lateral_velocity.abs()).exp();

    // Backward motion is catastrophic
    if state.speed < 0.0 {
        reward *= 0.01;
    }

    reward.clamp(0.0, 1.0)
}

/// Speed tracking reward: how well the vehicle matches the target speed.
///
/// Returns 1.0 when at target speed, decays with error.
pub fn speed_reward(state: &VehicleState, target_speed: f64) -> f64 {
    let error = (state.speed - target_speed).abs();
    // Gaussian-like: exp(-error² / (2 * σ²)), σ = 2.0 m/s
    (-error * error / 8.0).exp()
}

/// Lane-keeping reward: penalizes lateral deviation from center.
///
/// Uses `position_y` as proxy for lateral offset from lane center.
/// Returns 1.0 at center, decays with offset.
pub fn lane_reward(state: &VehicleState) -> f64 {
    let offset = state.position_y.abs();
    // Lane half-width ~1.8m. Exponential decay.
    (-offset * offset / 3.24).exp() // σ = 1.8m → 2σ² = 6.48 → /3.24 per side
}

/// Heading alignment reward: penalizes deviation from road heading.
///
/// For straight-road scenarios, the target heading is 0.0 (east).
pub fn heading_reward(state: &VehicleState, target_heading: f64) -> f64 {
    let error = angle_diff(state.heading, target_heading).abs();
    (-2.0 * error * error).exp()
}

/// Smoothness reward: penalizes large acceleration and jerk.
///
/// Comfortable driving has low longitudinal and lateral acceleration.
pub fn smoothness_reward(state: &VehicleState) -> f64 {
    let ax = state.longitudinal_accel.abs();
    let ay = state.lateral_accel.abs();
    // Comfort threshold: 2 m/s² lon, 2 m/s² lat
    let lon_penalty = (-0.1 * (ax - 2.0).max(0.0).powi(2)).exp();
    let lat_penalty = (-0.1 * (ay - 2.0).max(0.0).powi(2)).exp();
    lon_penalty * lat_penalty
}

/// Follow-distance reward: how well the vehicle maintains target gap to lead vehicle.
///
/// Returns 1.0 when at exact target gap, decays with gap error (Gaussian, σ=10m).
pub fn follow_distance_reward(state: &VehicleState, target_gap: f64) -> f64 {
    let gap_error = (state.nearest_peer_distance - target_gap).abs();
    (-gap_error / 10.0).exp()
}

/// Combined episode reward for a given task.
///
/// Weighted combination of component rewards, tuned per task.
/// `target_heading` is the road heading at the vehicle's projected position (0.0 for straight roads).
pub fn episode_reward(
    state: &VehicleState,
    task: &VehicleTask,
    target_speed: f64,
    target_heading: f64,
) -> f64 {
    let safe = safety_reward(state);
    let speed = speed_reward(state, target_speed);
    let lane = lane_reward(state);
    let heading = heading_reward(state, target_heading);
    let smooth = smoothness_reward(state);

    match task {
        VehicleTask::LaneKeep => {
            // Safety dominates, then lane tracking, then speed, then comfort
            0.30 * safe + 0.25 * lane + 0.20 * speed + 0.15 * heading + 0.10 * smooth
        }
        VehicleTask::Follow { target_gap } => {
            // Safety + gap-keeping + speed matching for adaptive cruise
            let gap = follow_distance_reward(state, *target_gap);
            0.25 * safe + 0.30 * speed + 0.25 * gap + 0.10 * lane + 0.10 * smooth
        }
        VehicleTask::LaneChange => {
            // Heading and lane position change targets; safety still paramount
            0.35 * safe + 0.25 * smooth + 0.20 * speed + 0.10 * lane + 0.10 * heading
        }
        VehicleTask::EmergencyStop => {
            // Speed should go to zero; safety is everything
            0.40 * safe + 0.40 * speed + 0.10 * lane + 0.10 * smooth
        }
    }
}

/// Normalize angle difference to [-pi, pi].
fn angle_diff(a: f64, b: f64) -> f64 {
    let mut d = a - b;
    while d > std::f64::consts::PI {
        d -= 2.0 * std::f64::consts::PI;
    }
    while d < -std::f64::consts::PI {
        d += 2.0 * std::f64::consts::PI;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VehicleState;

    #[test]
    fn test_safety_reward_normal() {
        let state = VehicleState::cruising(13.4);
        let r = safety_reward(&state);
        assert!(r > 0.9, "Normal cruising should be safe: {r}");
    }

    #[test]
    fn test_safety_reward_sliding() {
        let mut state = VehicleState::cruising(13.4);
        state.tire_slip_front = 0.3;
        state.tire_slip_rear = 0.3;
        state.yaw_rate = 0.5;
        let r = safety_reward(&state);
        assert!(r < 0.5, "Sliding should have low safety reward: {r}");
    }

    #[test]
    fn test_speed_reward_at_target() {
        let state = VehicleState::cruising(13.4);
        let r = speed_reward(&state, 13.4);
        assert!((r - 1.0).abs() < 1e-6, "At target should be 1.0: {r}");
    }

    #[test]
    fn test_speed_reward_off_target() {
        let state = VehicleState::cruising(5.0);
        let r = speed_reward(&state, 13.4);
        assert!(r < 0.5, "Far from target should be low: {r}");
    }

    #[test]
    fn test_lane_reward_centered() {
        let state = VehicleState::cruising(13.4); // position_y = 0
        let r = lane_reward(&state);
        assert!((r - 1.0).abs() < 1e-6, "Centered should be 1.0: {r}");
    }

    #[test]
    fn test_lane_reward_offset() {
        let mut state = VehicleState::cruising(13.4);
        state.position_y = 1.8; // edge of lane
        let r = lane_reward(&state);
        assert!(r < 0.5, "At lane edge should be low: {r}");
    }

    #[test]
    fn test_heading_reward_aligned() {
        let state = VehicleState::cruising(13.4); // heading = 0
        let r = heading_reward(&state, 0.0);
        assert!((r - 1.0).abs() < 1e-6, "Aligned should be 1.0: {r}");
    }

    #[test]
    fn test_smoothness_reward_comfortable() {
        let state = VehicleState::cruising(13.4); // zero accel
        let r = smoothness_reward(&state);
        assert!((r - 1.0).abs() < 1e-6, "Zero accel should be smooth: {r}");
    }

    #[test]
    fn test_episode_reward_range() {
        let state = VehicleState::cruising(13.4);
        let r = episode_reward(&state, &VehicleTask::LaneKeep, 13.4, 0.0);
        assert!(r >= 0.0 && r <= 1.0, "Reward should be in [0,1]: {r}");
        assert!(r > 0.8, "Normal driving should score high: {r}");
    }

    #[test]
    fn test_episode_reward_emergency_stop() {
        let state = VehicleState::stopped();
        let r = episode_reward(&state, &VehicleTask::EmergencyStop, 0.0, 0.0);
        assert!(r > 0.8, "Stopped at target 0 should score high: {r}");
    }

    #[test]
    fn test_angle_diff_wraps() {
        let d = angle_diff(3.0, -3.0);
        assert!(d.abs() < std::f64::consts::PI + 0.01, "Should wrap: {d}");
    }

    #[test]
    fn test_follow_reward_on_target() {
        let mut state = VehicleState::cruising(13.4);
        state.nearest_peer_distance = 30.0;
        let r = follow_distance_reward(&state, 30.0);
        assert!((r - 1.0).abs() < 1e-6, "At target gap should be 1.0: {r}");
    }

    #[test]
    fn test_follow_reward_off_target() {
        let mut state = VehicleState::cruising(13.4);
        state.nearest_peer_distance = 60.0;
        let r = follow_distance_reward(&state, 30.0);
        assert!(r < 0.1, "Far from target gap should be low: {r}");
    }
}
