//! DMC-standard reward functions for humanoid tasks.
//!
//! Implements the reward formulations from the DeepMind Control Suite paper
//! (Tassa et al., 2018). Uses tolerance functions for smooth reward shaping.

use crate::types::{HumanoidCommand, HumanoidState, HumanoidTask};

/// Standing reward: head_height × uprightness.
///
/// Matches dm_control `humanoid.stand` reward:
/// - Head height target: 1.4m (tolerance bounds: [1.4, +inf), margin 0.35)
/// - Uprightness target: 0.9+ (tolerance bounds: [0.9, +inf), margin 0.19)
pub fn standing_reward(state: &HumanoidState) -> f64 {
    let height = tolerance(state.head_height, 1.4, f64::INFINITY, 0.35);
    let upright = tolerance(state.uprightness(), 0.9, f64::INFINITY, 0.19);
    height * upright
}

/// Small control reward: penalizes large torques.
///
/// Matches dm_control's control penalty formulation.
/// Returns a value in (0, 1] where 1.0 = zero torque.
pub fn small_control(cmd: &HumanoidCommand) -> f64 {
    let penalty: f64 = cmd
        .torques
        .iter()
        .map(|t| tolerance_quadratic(*t as f64, 0.0, 1.0))
        .sum();
    (4.0 + penalty / 21.0) / 5.0
}

/// Locomotion reward: horizontal speed matching target.
///
/// Returns high reward when horizontal speed matches `target_speed`.
pub fn locomotion_reward(state: &HumanoidState, target_speed: f64) -> f64 {
    let horizontal_speed = state.horizontal_speed();
    let margin = target_speed.max(0.5);
    tolerance_linear(horizontal_speed, target_speed, f64::INFINITY, margin)
}

/// Gait symmetry reward: incentivizes alternating foot contact.
///
/// Returns higher reward when feet are at different heights (one lifted, one grounded),
/// which indicates proper bipedal gait with swing/stance alternation.
/// Uses Gaussian tolerance: 1.0 when foot height difference >= 0.03m, decays below.
pub fn gait_symmetry_reward(state: &HumanoidState) -> f64 {
    let r_foot_z = state.extremities[8];
    let l_foot_z = state.extremities[11];
    let height_diff = (r_foot_z - l_foot_z).abs();
    // Proper gait lifts one foot ~0.05-0.15m while other is grounded
    // Tight margin (0.03m) penalizes shuffling harshly: 0.25 when both grounded
    tolerance(height_diff, 0.05, f64::INFINITY, 0.03)
}

/// Metabolic efficiency reward: penalizes sum of squared torques.
///
/// Biological walkers minimize energy expenditure (Cost of Transport).
/// This reward incentivizes passive dynamics — using gravity and momentum
/// instead of motor force. Returns 1.0 for zero torque, decays toward 0
/// for high torque.
pub fn metabolic_efficiency_reward(cmd: &HumanoidCommand) -> f64 {
    let sum_sq: f64 = cmd.torques.iter().map(|t| (*t as f64) * (*t as f64)).sum();
    // Normalize: 21 actuators, max torque 1.0 each → max sum_sq = 21
    let normalized = sum_sq / 21.0;
    (-2.0 * normalized).exp()
}

/// Clearance threshold: foot z above this = swing phase.
const CLEARANCE_THRESHOLD: f64 = 0.03;

/// Foot clearance reward: incentivizes proper foot lift during swing.
///
/// Swing phase (foot z > threshold): reward clearance near target 0.04-0.12m.
/// Stance phase (foot on ground): neutral 0.5 (no penalty for grounding).
pub fn clearance_reward(state: &HumanoidState) -> f64 {
    let foot_score = |z: f64| -> f64 {
        if z > CLEARANCE_THRESHOLD {
            tolerance(z, 0.04, 0.12, 0.03)
        } else {
            0.5
        }
    };
    0.5 * foot_score(state.extremities[8]) + 0.5 * foot_score(state.extremities[11])
}

/// Phasic metabolic cost: penalizes sustained activation, tolerates bursts.
///
/// Biological locomotion uses brief, high-amplitude muscle bursts during
/// stance phase (gastrocnemius pushoff ~200ms). This reward distinguishes
/// efficient burst patterns from wasteful sustained co-contraction.
///
/// Uses variance of squared torques across actuators: high variance = some
/// actuators active (bursting), others relaxed (efficient). Low variance
/// with high mean = co-contraction (inefficient).
pub fn phasic_metabolic_reward(cmd: &HumanoidCommand) -> f64 {
    let squares: Vec<f64> = cmd.torques.iter().map(|t| (*t as f64).powi(2)).collect();
    let mean_sq = squares.iter().sum::<f64>() / 21.0;
    let variance = squares.iter().map(|s| (s - mean_sq).powi(2)).sum::<f64>() / 21.0;

    // High mean + low variance = co-contraction → low reward
    // Low mean = efficient → high reward
    // High variance (burst pattern) gets partial credit even with higher mean
    let efficiency = (-2.0 * mean_sq).exp();
    let burst_bonus = (variance * 10.0).min(0.3); // up to 0.3 bonus for burst patterns
    (efficiency + burst_bonus).min(1.0)
}

/// Cost-of-transport efficiency reward: penalizes high power relative to speed.
///
/// Instantaneous: power = Σ|torque_i × velocity_i|, CoT_inst = power / (mass × max(speed, 0.1)).
/// Returns exp(-CoT_inst) — high when efficient, low when wasteful.
pub fn cot_efficiency_reward(state: &HumanoidState, cmd: &HumanoidCommand) -> f64 {
    let power: f64 = cmd
        .torques
        .iter()
        .zip(state.joint_velocities.iter())
        .map(|(t, v)| (*t as f64 * v).abs())
        .sum();
    let speed = state.horizontal_speed().max(0.1);
    let cot_inst = power / (70.0 * speed);
    (-cot_inst).exp()
}

/// Combined episode reward based on task with curriculum target speed.
///
/// - Stand: standing_reward × small_control
/// - Walk/Run: 0.30×stand + 0.28×locomotion + 0.10×symmetry + 0.12×metabolic + 0.10×clearance + 0.10×cot_eff
///
/// The additive formulation ensures a gradient signal exists even when
/// horizontal speed is far from target (multiplicative `stand × locomotion`
/// collapses to ~0 when either factor is small, starving the learner).
///
/// Clearance partially subsumes symmetry (proper foot lift implies alternation),
/// so symmetry weight is reduced. CoT efficiency overlaps metabolic (both penalize
/// torque), so metabolic drops 0.15→0.12. Standing drops 0.35→0.30 since CoT
/// provides complementary signal. Locomotion 0.30→0.28 to maintain sum=1.0.
///
/// The `target_speed` parameter allows the curriculum to ramp speed gradually
/// (e.g., Walk: 0→1 m/s, Run: 1→3 m/s).
pub fn episode_reward(
    state: &HumanoidState,
    cmd: &HumanoidCommand,
    task: &HumanoidTask,
    target_speed: f64,
) -> f64 {
    let stand = standing_reward(state) * small_control(cmd);
    match task {
        HumanoidTask::Stand => stand,
        HumanoidTask::Walk | HumanoidTask::Run => {
            let locomotion = locomotion_reward(state, target_speed);
            let symmetry = gait_symmetry_reward(state);
            let metabolic = phasic_metabolic_reward(cmd);
            let clearance = clearance_reward(state);
            let cot_eff = cot_efficiency_reward(state, cmd);
            0.30 * stand
                + 0.28 * locomotion
                + 0.10 * symmetry
                + 0.12 * metabolic
                + 0.10 * clearance
                + 0.10 * cot_eff
        }
    }
}

/// Tolerance function: returns 1.0 when value is within [lower, upper],
/// and decays smoothly outside with the given margin.
///
/// Uses Gaussian falloff: exp(-0.5 * ((x - bound) / margin)^2).
fn tolerance(value: f64, lower: f64, upper: f64, margin: f64) -> f64 {
    if value >= lower && value <= upper {
        1.0
    } else if margin <= 0.0 {
        0.0
    } else {
        let dist = if value < lower {
            lower - value
        } else {
            value - upper
        };
        (-0.5 * (dist / margin).powi(2)).exp()
    }
}

/// Tolerance for locomotion speed: linear falloff.
fn tolerance_linear(value: f64, lower: f64, upper: f64, margin: f64) -> f64 {
    if value >= lower && value <= upper {
        1.0
    } else if margin <= 0.0 {
        0.0
    } else {
        let dist = if value < lower {
            lower - value
        } else {
            value - upper
        };
        (1.0 - dist / margin).max(0.0)
    }
}

/// Quadratic tolerance: 1.0 at target, smooth decay.
fn tolerance_quadratic(value: f64, target: f64, margin: f64) -> f64 {
    if margin <= 0.0 {
        return if (value - target).abs() < 1e-10 {
            1.0
        } else {
            0.0
        };
    }
    let normalized = (value - target) / margin;
    (1.0 - normalized * normalized).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standing_reward_upright() {
        let state = HumanoidState::standing();
        let reward = standing_reward(&state);
        assert!(
            reward > 0.8,
            "Standing state should have high standing reward: {reward}"
        );
    }

    #[test]
    fn test_standing_reward_fallen() {
        let mut state = HumanoidState::standing();
        state.head_height = 0.3; // Fallen
        state.torso_vertical = [0.0, 0.0, 0.1]; // Nearly horizontal
        let reward = standing_reward(&state);
        assert!(
            reward < 0.1,
            "Fallen state should have low standing reward: {reward}"
        );
    }

    #[test]
    fn test_small_control_zero_torque() {
        let cmd = HumanoidCommand::zero();
        let reward = small_control(&cmd);
        assert!(
            (reward - 1.0).abs() < 0.01,
            "Zero torque should give max control reward: {reward}"
        );
    }

    #[test]
    fn test_small_control_max_torque() {
        let cmd = HumanoidCommand { torques: [1.0; 21] };
        let reward = small_control(&cmd);
        assert!(
            reward < 1.0,
            "Max torque should give reduced control reward: {reward}"
        );
        assert!(
            reward > 0.0,
            "Control reward should stay positive: {reward}"
        );
    }

    #[test]
    fn test_locomotion_reward_at_target() {
        let mut state = HumanoidState::standing();
        state.com_velocity = [1.0, 0.0, 0.0]; // 1 m/s forward
        let reward = locomotion_reward(&state, 1.0);
        assert!(
            reward > 0.8,
            "At target speed should give high reward: {reward}"
        );
    }

    #[test]
    fn test_locomotion_reward_stationary() {
        let state = HumanoidState::standing(); // Zero velocity
        let reward = locomotion_reward(&state, 1.0);
        assert!(
            reward < 0.5,
            "Stationary should give low locomotion reward: {reward}"
        );
    }

    #[test]
    fn test_episode_reward_stand() {
        let state = HumanoidState::standing();
        let cmd = HumanoidCommand::zero();
        let reward = episode_reward(&state, &cmd, &HumanoidTask::Stand, 0.0);
        assert!(
            reward > 0.5,
            "Standing correctly with zero torque should give good reward: {reward}"
        );
    }

    #[test]
    fn test_episode_reward_walk() {
        let mut state = HumanoidState::standing();
        state.com_velocity = [1.0, 0.0, 0.0];
        let cmd = HumanoidCommand::zero();
        let reward = episode_reward(&state, &cmd, &HumanoidTask::Walk, 1.0);
        assert!(
            reward > 0.0,
            "Walking at target speed should give positive reward: {reward}"
        );
    }

    #[test]
    fn test_episode_reward_walk_at_zero_speed() {
        let state = HumanoidState::standing();
        let cmd = HumanoidCommand::zero();
        // At curriculum start, target_speed=0, so standing still is "at target"
        let reward = episode_reward(&state, &cmd, &HumanoidTask::Walk, 0.0);
        assert!(
            reward > 0.5,
            "Walk with target_speed=0 should reward standing: {reward}"
        );
    }

    #[test]
    fn test_gait_symmetry_feet_alternating() {
        let mut state = HumanoidState::standing();
        // One foot lifted 0.1m, other on ground
        state.extremities[8] = 0.0; // right foot on ground
        state.extremities[11] = 0.1; // left foot lifted
        let reward = gait_symmetry_reward(&state);
        assert!(
            reward > 0.8,
            "Alternating feet should give high symmetry reward: {reward}"
        );
    }

    #[test]
    fn test_gait_symmetry_both_grounded() {
        let mut state = HumanoidState::standing();
        state.extremities[8] = 0.0;
        state.extremities[11] = 0.0;
        let reward = gait_symmetry_reward(&state);
        assert!(
            reward < 0.5,
            "Both feet on ground should give low symmetry reward: {reward}"
        );
    }

    #[test]
    fn test_metabolic_efficiency_zero_torque() {
        let cmd = HumanoidCommand::zero();
        let reward = metabolic_efficiency_reward(&cmd);
        assert!(
            (reward - 1.0).abs() < 0.01,
            "Zero torque should give max metabolic reward: {reward}"
        );
    }

    #[test]
    fn test_metabolic_efficiency_high_torque() {
        let cmd = HumanoidCommand { torques: [0.8; 21] };
        let reward = metabolic_efficiency_reward(&cmd);
        assert!(
            reward < 0.5,
            "High torque should give low metabolic reward: {reward}"
        );
        assert!(
            reward > 0.0,
            "Metabolic reward should stay positive: {reward}"
        );
    }

    #[test]
    fn test_phasic_metabolic_burst_pattern() {
        // Few actuators at high torque, rest at zero → burst pattern
        let mut cmd = HumanoidCommand::zero();
        cmd.torques[5] = 0.8; // right hip pushoff
        cmd.torques[6] = 0.6; // right knee
        let burst_reward = phasic_metabolic_reward(&cmd);

        // Uniform moderate torque → co-contraction
        let uniform_cmd = HumanoidCommand { torques: [0.4; 21] };
        let uniform_reward = phasic_metabolic_reward(&uniform_cmd);

        assert!(
            burst_reward > uniform_reward,
            "Burst pattern should reward higher than co-contraction: burst={burst_reward} uniform={uniform_reward}"
        );
    }

    #[test]
    fn test_phasic_metabolic_cocontraction() {
        // All actuators at moderate torque → co-contraction (low reward)
        let cmd = HumanoidCommand { torques: [0.5; 21] };
        let reward = phasic_metabolic_reward(&cmd);

        // Burst pattern: few active, rest zero
        let mut burst_cmd = HumanoidCommand::zero();
        burst_cmd.torques[5] = 0.8;
        burst_cmd.torques[11] = 0.8;
        let burst_reward = phasic_metabolic_reward(&burst_cmd);

        assert!(
            reward < burst_reward,
            "Co-contraction should reward lower than burst: cocontract={reward} burst={burst_reward}"
        );
    }

    #[test]
    fn test_phasic_metabolic_zero_torque() {
        let cmd = HumanoidCommand::zero();
        let reward = phasic_metabolic_reward(&cmd);
        assert!(
            (reward - 1.0).abs() < 0.01,
            "Zero torque should give max phasic metabolic reward: {reward}"
        );
    }

    #[test]
    fn test_clearance_reward_foot_lifted() {
        let mut state = HumanoidState::standing();
        state.extremities[8] = 0.06; // right foot lifted to 6cm (within target)
        state.extremities[11] = 0.0; // left foot on ground
        let reward = clearance_reward(&state);
        assert!(
            reward > 0.5,
            "One foot lifted in target range should give reward > 0.5: {reward}"
        );
    }

    #[test]
    fn test_clearance_reward_both_grounded() {
        let mut state = HumanoidState::standing();
        state.extremities[8] = 0.0;
        state.extremities[11] = 0.0;
        let reward = clearance_reward(&state);
        assert!(
            (reward - 0.5).abs() < 1e-10,
            "Both feet grounded should give neutral 0.5: {reward}"
        );
    }

    #[test]
    fn test_clearance_reward_shuffling() {
        let mut state = HumanoidState::standing();
        state.extremities[8] = 0.01; // barely lifted (below threshold)
        state.extremities[11] = 0.0;
        let reward = clearance_reward(&state);
        assert!(
            reward <= 0.5 + 1e-10,
            "Shuffling (below threshold) should give reward <= 0.5: {reward}"
        );
    }

    // ── CoT efficiency reward tests ──

    #[test]
    fn test_cot_efficiency_zero_torque() {
        let state = HumanoidState::standing();
        let cmd = HumanoidCommand::zero();
        let reward = cot_efficiency_reward(&state, &cmd);
        assert!(
            (reward - 1.0).abs() < 0.01,
            "Zero torque should give max CoT efficiency: {reward}"
        );
    }

    #[test]
    fn test_cot_efficiency_high_power() {
        let mut state = HumanoidState::standing();
        // Low speed (horizontal speed ≈ 0 → clamped to 0.1)
        state.com_velocity = [0.05, 0.0, 0.0];
        // High joint velocities + high torques = high power
        state.joint_velocities = [2.0; 21];
        let cmd = HumanoidCommand { torques: [0.8; 21] };
        let reward = cot_efficiency_reward(&state, &cmd);
        assert!(
            reward < 0.5,
            "High power + low speed should give low CoT efficiency: {reward}"
        );
    }

    #[test]
    fn test_tolerance_within_bounds() {
        assert!((tolerance(1.5, 1.0, 2.0, 0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_outside_bounds() {
        let val = tolerance(0.0, 1.0, 2.0, 0.5);
        assert!(val < 1.0);
        assert!(val > 0.0);
    }

    #[test]
    fn test_tolerance_gaussian_shape() {
        // Further from bounds = lower reward
        let near = tolerance(0.9, 1.0, 2.0, 0.5);
        let far = tolerance(0.0, 1.0, 2.0, 0.5);
        assert!(
            near > far,
            "Closer should give higher reward: near={near}, far={far}"
        );
    }
}
