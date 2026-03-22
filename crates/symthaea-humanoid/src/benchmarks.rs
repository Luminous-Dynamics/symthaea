//! DMC evaluation metrics and benchmark infrastructure.
//!
//! Implements standard DeepMind Control Suite evaluation:
//! - Episode return (mean reward over 1000 steps)
//! - Success metrics (standing, walking, running thresholds)
//! - Perturbation recovery metrics
//! - Comparison against published SAC/TD3 baselines

use crate::gait::GaitAnalyzer;
use crate::reward;
use crate::types::{HumanoidCommand, HumanoidState, HumanoidTask, NUM_ACTUATORS};

/// DMC benchmark result for a single evaluation run.
#[derive(Debug, Clone)]
pub struct DmcBenchmarkResult {
    /// Mean episode return (avg reward per step, in [0, 1]).
    pub mean_return: f64,
    /// Standard deviation of per-step rewards.
    pub return_std: f64,
    /// Fraction of steps with standing_reward > 0.5.
    pub standing_fraction: f64,
    /// Mean head height across episode.
    pub mean_head_height: f64,
    /// Mean uprightness across episode.
    pub mean_uprightness: f64,
    /// Mean horizontal speed.
    pub mean_horizontal_speed: f64,
    /// Whether the humanoid fell (head_height < 0.5 at any point).
    pub fell: bool,
    /// Number of steps before first fall (or total steps if no fall).
    pub steps_to_fall: usize,
    /// Total steps evaluated.
    pub total_steps: usize,
    /// Average foot clearance during swing phases (meters).
    pub avg_foot_clearance: f64,
    /// Minimum foot clearance across all strides (meters).
    pub min_foot_clearance: f64,
    /// Average stride length (meters).
    pub avg_stride_length: f64,
    /// Average cadence (steps per second).
    pub avg_cadence: f64,
    /// Gait asymmetry: |R-L|/(R+L), 0 = symmetric.
    pub gait_asymmetry: f64,
    /// Step regularity: exp(-CV) of step intervals.
    pub step_regularity: f64,
    /// Cost of Transport: energy / (mass × distance).
    pub cost_of_transport: f64,
}

/// Published baseline scores for DMC Humanoid tasks.
/// From Tassa et al., 2018 and Haarnoja et al., 2018 (SAC).
#[derive(Debug, Clone)]
pub struct BaselineScores {
    /// Task name.
    pub task: HumanoidTask,
    /// SAC baseline (Haarnoja et al., 2018).
    pub sac_return: f64,
    /// TD3 baseline (Fujimoto et al., 2018).
    pub td3_return: f64,
    /// D4PG baseline (Barth-Maron et al., 2018).
    pub d4pg_return: f64,
}

impl BaselineScores {
    /// Get published baseline scores for a task.
    pub fn for_task(task: &HumanoidTask) -> Self {
        match task {
            HumanoidTask::Stand => Self {
                task: *task,
                sac_return: 0.95,  // SAC achieves ~950/1000
                td3_return: 0.80,  // TD3 achieves ~800/1000
                d4pg_return: 0.90, // D4PG achieves ~900/1000
            },
            HumanoidTask::Walk => Self {
                task: *task,
                sac_return: 0.75,  // SAC achieves ~750/1000
                td3_return: 0.50,  // TD3 achieves ~500/1000
                d4pg_return: 0.70, // D4PG achieves ~700/1000
            },
            HumanoidTask::Run => Self {
                task: *task,
                sac_return: 0.45,  // SAC achieves ~450/1000
                td3_return: 0.30,  // TD3 achieves ~300/1000
                d4pg_return: 0.40, // D4PG achieves ~400/1000
            },
        }
    }
}

/// Evaluate a sequence of (state, command) pairs against a task.
pub fn evaluate_episode(
    states: &[HumanoidState],
    commands: &[HumanoidCommand],
    task: &HumanoidTask,
) -> DmcBenchmarkResult {
    let n = states.len().min(commands.len());
    if n == 0 {
        return DmcBenchmarkResult {
            mean_return: 0.0,
            return_std: 0.0,
            standing_fraction: 0.0,
            mean_head_height: 0.0,
            mean_uprightness: 0.0,
            mean_horizontal_speed: 0.0,
            fell: true,
            steps_to_fall: 0,
            total_steps: 0,
            avg_foot_clearance: 0.0,
            min_foot_clearance: 0.0,
            avg_stride_length: 0.0,
            avg_cadence: 0.0,
            gait_asymmetry: 0.0,
            step_regularity: 0.0,
            cost_of_transport: 0.0,
        };
    }

    let mut rewards = Vec::with_capacity(n);
    let mut standing_count = 0usize;
    let mut total_head_height = 0.0;
    let mut total_uprightness = 0.0;
    let mut total_speed = 0.0;
    let mut fell = false;
    let mut steps_to_fall = n;

    // Gait quality tracking
    let mut gait_analyzer = GaitAnalyzer::new();
    let mut horizontal_pos = [0.0f64; 2];
    let mut total_mechanical_energy = 0.0f64;
    let dt = 0.025; // standard DMC timestep

    for i in 0..n {
        let target_speed = match task {
            HumanoidTask::Stand => 0.0,
            HumanoidTask::Walk => 1.0,
            HumanoidTask::Run => 10.0,
        };
        let r = reward::episode_reward(&states[i], &commands[i], task, target_speed);
        rewards.push(r);

        let sr = reward::standing_reward(&states[i]);
        if sr > 0.5 {
            standing_count += 1;
        }

        total_head_height += states[i].head_height;
        total_uprightness += states[i].uprightness();
        total_speed += states[i].horizontal_speed();

        if !fell && states[i].head_height < 0.5 {
            fell = true;
            steps_to_fall = i;
        }

        // Gait quality tracking
        horizontal_pos[0] += states[i].root_linear_velocity[0] * dt;
        horizontal_pos[1] += states[i].root_linear_velocity[1] * dt;
        for j in 0..NUM_ACTUATORS {
            total_mechanical_energy +=
                (commands[i].torques[j] as f64 * states[i].joint_velocities[j]).abs() * dt;
        }
        gait_analyzer.update_with_position(&states[i], horizontal_pos, states[i].timestamp);
    }

    let mean_return: f64 = rewards.iter().sum::<f64>() / n as f64;
    let return_std = {
        let variance: f64 = rewards
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / n as f64;
        variance.sqrt()
    };

    let gait_summary = gait_analyzer.summary();
    let total_distance = (horizontal_pos[0].powi(2) + horizontal_pos[1].powi(2)).sqrt();
    let cost_of_transport = if total_distance > 0.01 {
        total_mechanical_energy / (70.0 * total_distance)
    } else {
        0.0
    };

    DmcBenchmarkResult {
        mean_return,
        return_std,
        standing_fraction: standing_count as f64 / n as f64,
        mean_head_height: total_head_height / n as f64,
        mean_uprightness: total_uprightness / n as f64,
        mean_horizontal_speed: total_speed / n as f64,
        fell,
        steps_to_fall,
        total_steps: n,
        avg_foot_clearance: gait_summary.avg_clearance,
        min_foot_clearance: gait_summary.min_clearance,
        avg_stride_length: gait_summary.avg_stride_length,
        avg_cadence: gait_summary.avg_cadence,
        gait_asymmetry: gait_summary.gait_asymmetry,
        step_regularity: gait_summary.step_regularity,
        cost_of_transport,
    }
}

/// Format a benchmark comparison table.
pub fn format_comparison(result: &DmcBenchmarkResult, task: &HumanoidTask) -> String {
    let baselines = BaselineScores::for_task(task);
    format!(
        "DMC Humanoid {:?} Benchmark\n\
         ═══════════════════════════════\n\
         HDC-LTC-FEP:  {:.3} (ours)\n\
         SAC:          {:.3}\n\
         TD3:          {:.3}\n\
         D4PG:         {:.3}\n\
         ───────────────────────────────\n\
         Standing %:   {:.1}%\n\
         Head Height:  {:.3}m\n\
         Uprightness:  {:.3}\n\
         Speed:        {:.3} m/s\n\
         Fell:         {}\n\
         Steps:        {}\n\
         ───────────────────────────────\n\
         Gait Quality:\n\
           Clearance:    {:.4}m (min {:.4}m)\n\
           Stride:       {:.3}m @ {:.1} steps/s\n\
           Asymmetry:    {:.4}\n\
           Regularity:   {:.3}\n\
           CoT:          {:.3}",
        task,
        result.mean_return,
        baselines.sac_return,
        baselines.td3_return,
        baselines.d4pg_return,
        result.standing_fraction * 100.0,
        result.mean_head_height,
        result.mean_uprightness,
        result.mean_horizontal_speed,
        result.fell,
        result.total_steps,
        result.avg_foot_clearance,
        result.min_foot_clearance,
        result.avg_stride_length,
        result.avg_cadence,
        result.gait_asymmetry,
        result.step_regularity,
        result.cost_of_transport,
    )
}

/// Format a LaTeX comparison table for publication.
///
/// Produces a standalone LaTeX tabular environment comparing HDC-LTC-FEP results
/// against published SAC/TD3/D4PG baselines for each task.
pub fn format_comparison_table(results: &[(HumanoidTask, DmcBenchmarkResult)]) -> String {
    let mut latex = String::new();
    latex.push_str("\\begin{table}[H]\n");
    latex.push_str("\\centering\n");
    latex.push_str("\\caption{DMC Humanoid benchmark: HDC-LTC-FEP vs.~published RL baselines.}\n");
    latex.push_str("\\label{tab:dmc-humanoid}\n");
    latex.push_str("\\begin{tabular}{lcccc}\n");
    latex.push_str("\\toprule\n");
    latex.push_str("\\textbf{Task} & \\textbf{HDC-LTC-FEP} & \\textbf{SAC} & \\textbf{TD3} & \\textbf{D4PG} \\\\\n");
    latex.push_str("\\midrule\n");

    for (task, result) in results {
        let baselines = BaselineScores::for_task(task);
        latex.push_str(&format!(
            "{:?} & {:.3} & {:.3} & {:.3} & {:.3} \\\\\n",
            task,
            result.mean_return,
            baselines.sac_return,
            baselines.td3_return,
            baselines.d4pg_return,
        ));
    }

    latex.push_str("\\bottomrule\n");
    latex.push_str("\\end{tabular}\n");

    // Add gait quality sub-table
    latex.push_str("\\vspace{0.5em}\n\n");
    latex.push_str("\\begin{tabular}{lccccc}\n");
    latex.push_str("\\toprule\n");
    latex.push_str("\\textbf{Task} & \\textbf{Standing\\%} & \\textbf{Fell?} & \\textbf{CoT} & \\textbf{Asymmetry} & \\textbf{Regularity} \\\\\n");
    latex.push_str("\\midrule\n");

    for (task, result) in results {
        latex.push_str(&format!(
            "{:?} & {:.1}\\% & {} & {:.2} & {:.4} & {:.3} \\\\\n",
            task,
            result.standing_fraction * 100.0,
            if result.fell { "Yes" } else { "No" },
            result.cost_of_transport,
            result.gait_asymmetry,
            result.step_regularity,
        ));
    }

    latex.push_str("\\bottomrule\n");
    latex.push_str("\\end{tabular}\n");
    latex.push_str("\\end{table}\n");
    latex
}

/// Format a JSON report for automated comparison.
pub fn format_json_report(results: &[(HumanoidTask, DmcBenchmarkResult)]) -> String {
    let mut json = String::from("{\n  \"benchmark\": \"dmc_humanoid\",\n  \"tasks\": [\n");

    for (i, (task, result)) in results.iter().enumerate() {
        let baselines = BaselineScores::for_task(task);
        json.push_str(&format!(
            "    {{\n\
             \x20     \"task\": \"{:?}\",\n\
             \x20     \"hdc_ltc_fep\": {:.4},\n\
             \x20     \"sac_baseline\": {:.4},\n\
             \x20     \"td3_baseline\": {:.4},\n\
             \x20     \"d4pg_baseline\": {:.4},\n\
             \x20     \"standing_fraction\": {:.4},\n\
             \x20     \"fell\": {},\n\
             \x20     \"mean_head_height\": {:.4},\n\
             \x20     \"mean_uprightness\": {:.4},\n\
             \x20     \"mean_horizontal_speed\": {:.4},\n\
             \x20     \"cost_of_transport\": {:.4},\n\
             \x20     \"gait_asymmetry\": {:.4},\n\
             \x20     \"step_regularity\": {:.4}\n\
             \x20   }}",
            task,
            result.mean_return,
            baselines.sac_return,
            baselines.td3_return,
            baselines.d4pg_return,
            result.standing_fraction,
            result.fell,
            result.mean_head_height,
            result.mean_uprightness,
            result.mean_horizontal_speed,
            result.cost_of_transport,
            result.gait_asymmetry,
            result.step_regularity,
        ));
        if i + 1 < results.len() {
            json.push(',');
        }
        json.push('\n');
    }

    json.push_str("  ]\n}\n");
    json
}

/// Perturbation recovery summary for JSON output.
pub fn format_perturbation_json(
    task: &HumanoidTask,
    perturbation_name: &str,
    result: &crate::perturbations::PerturbationBenchmarkResult,
) -> String {
    format!(
        "{{\n\
         \x20 \"task\": \"{:?}\",\n\
         \x20 \"perturbation\": \"{}\",\n\
         \x20 \"pre_reward\": {:.4},\n\
         \x20 \"min_reward\": {:.4},\n\
         \x20 \"final_reward\": {:.4},\n\
         \x20 \"recovery_steps\": {},\n\
         \x20 \"fell\": {},\n\
         \x20 \"min_tau\": {:.4}\n\
         }}",
        task,
        perturbation_name,
        result.pre_perturbation_reward,
        result.min_reward,
        result.final_reward,
        result.recovery_steps,
        result.fell,
        result.min_tau,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_standing_episode() {
        let states: Vec<HumanoidState> = (0..100).map(|_| HumanoidState::standing()).collect();
        let commands: Vec<HumanoidCommand> = (0..100).map(|_| HumanoidCommand::zero()).collect();

        let result = evaluate_episode(&states, &commands, &HumanoidTask::Stand);
        assert!(
            result.mean_return > 0.5,
            "Standing should score well: {}",
            result.mean_return
        );
        assert!(!result.fell);
        assert_eq!(result.total_steps, 100);
    }

    #[test]
    fn test_evaluate_fallen_episode() {
        let mut states: Vec<HumanoidState> = Vec::new();
        for i in 0..100 {
            let mut s = HumanoidState::standing();
            if i > 20 {
                s.head_height = 0.3;
                s.torso_vertical = [0.0, 0.0, 0.1];
            }
            states.push(s);
        }
        let commands: Vec<HumanoidCommand> = (0..100).map(|_| HumanoidCommand::zero()).collect();

        let result = evaluate_episode(&states, &commands, &HumanoidTask::Stand);
        assert!(result.fell);
        assert_eq!(result.steps_to_fall, 21);
    }

    #[test]
    fn test_evaluate_empty() {
        let result = evaluate_episode(&[], &[], &HumanoidTask::Stand);
        assert_eq!(result.total_steps, 0);
        assert!(result.fell);
    }

    #[test]
    fn test_baseline_scores() {
        let stand = BaselineScores::for_task(&HumanoidTask::Stand);
        assert!(stand.sac_return > stand.td3_return);

        let walk = BaselineScores::for_task(&HumanoidTask::Walk);
        assert!(walk.sac_return > 0.0);

        let run = BaselineScores::for_task(&HumanoidTask::Run);
        assert!(run.sac_return > 0.0);
    }

    #[test]
    fn test_format_comparison() {
        let result = DmcBenchmarkResult {
            mean_return: 0.85,
            return_std: 0.1,
            standing_fraction: 0.95,
            mean_head_height: 1.38,
            mean_uprightness: 0.97,
            mean_horizontal_speed: 0.1,
            fell: false,
            steps_to_fall: 1000,
            total_steps: 1000,
            avg_foot_clearance: 0.0,
            min_foot_clearance: 0.0,
            avg_stride_length: 0.0,
            avg_cadence: 0.0,
            gait_asymmetry: 0.0,
            step_regularity: 0.0,
            cost_of_transport: 0.0,
        };
        let output = format_comparison(&result, &HumanoidTask::Stand);
        assert!(output.contains("HDC-LTC-FEP"));
        assert!(output.contains("0.850"));
    }

    #[test]
    fn test_benchmark_gait_fields() {
        let states: Vec<HumanoidState> = (0..100).map(|_| HumanoidState::standing()).collect();
        let commands: Vec<HumanoidCommand> = (0..100).map(|_| HumanoidCommand::zero()).collect();

        let result = evaluate_episode(&states, &commands, &HumanoidTask::Stand);
        // All gait fields should be finite (even if zero for standing)
        assert!(result.avg_foot_clearance.is_finite());
        assert!(result.min_foot_clearance.is_finite());
        assert!(result.avg_stride_length.is_finite());
        assert!(result.avg_cadence.is_finite());
        assert!(result.gait_asymmetry.is_finite());
        assert!(result.step_regularity.is_finite());
        assert!(result.cost_of_transport.is_finite());
    }

    #[test]
    fn test_format_comparison_extended() {
        let result = DmcBenchmarkResult {
            mean_return: 0.75,
            return_std: 0.15,
            standing_fraction: 0.9,
            mean_head_height: 1.35,
            mean_uprightness: 0.95,
            mean_horizontal_speed: 0.8,
            fell: false,
            steps_to_fall: 1000,
            total_steps: 1000,
            avg_foot_clearance: 0.06,
            min_foot_clearance: 0.03,
            avg_stride_length: 0.65,
            avg_cadence: 1.8,
            gait_asymmetry: 0.05,
            step_regularity: 0.85,
            cost_of_transport: 3.2,
        };
        let output = format_comparison(&result, &HumanoidTask::Walk);
        assert!(
            output.contains("Gait Quality"),
            "Output should contain Gait Quality section"
        );
        assert!(
            output.contains("Clearance"),
            "Output should contain Clearance metric"
        );
        assert!(
            output.contains("Regularity"),
            "Output should contain Regularity metric"
        );
        assert!(output.contains("CoT"), "Output should contain CoT metric");
    }
}
