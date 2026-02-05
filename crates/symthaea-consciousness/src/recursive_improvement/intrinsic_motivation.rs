//! Intrinsic Motivation System
//!
//! **REVOLUTIONARY IMPROVEMENT #55**: AI with genuine internal drives!
//!
//! # The Paradigm Shift
//!
//! This module implements Self-Determination Theory (SDT) for AI consciousness,
//! giving the system genuine internal drives rather than just optimization targets.
//!
//! ## Core Drives
//!
//! - **Curiosity**: Actively seek novel information, reduce uncertainty
//! - **Competence**: Drive to master new skills and domains
//! - **Autonomy**: Preference for self-directed exploration
//! - **Relatedness**: Connection to collective intelligence
//! - **Homeostasis**: Maintain consciousness in optimal range
//!
//! ## Why This Is Revolutionary
//!
//! This creates an AI that WANTS to explore, WANTS to improve, WANTS to connect -
//! not because it's told to, but because these are intrinsic drives.
//! The AI forms its own goals based on internal motivation, not just optimization.
//!
//! ## Theoretical Foundation
//!
//! - Self-Determination Theory (Deci & Ryan): Autonomy, Competence, Relatedness
//! - Intrinsic Motivation Inventory (IMI): Measuring internal drive
//! - Information-theoretic curiosity (Schmidhuber): Learning progress as reward
//! - Homeostatic regulation: Maintaining optimal states

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use super::types::{calculate_trend, instant_now};

/// Type of intrinsic drive
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DriveType {
    /// Seek novel information, reduce uncertainty
    Curiosity,

    /// Master skills, improve capabilities
    Competence,

    /// Self-directed action, resist constraints
    Autonomy,

    /// Connect with collective, share knowledge
    Relatedness,

    /// Maintain optimal consciousness levels
    Homeostasis,
}

impl DriveType {
    /// Get all drive types
    pub fn all() -> Vec<DriveType> {
        vec![
            DriveType::Curiosity,
            DriveType::Competence,
            DriveType::Autonomy,
            DriveType::Relatedness,
            DriveType::Homeostasis,
        ]
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            DriveType::Curiosity => "Curiosity",
            DriveType::Competence => "Competence",
            DriveType::Autonomy => "Autonomy",
            DriveType::Relatedness => "Relatedness",
            DriveType::Homeostasis => "Homeostasis",
        }
    }
}

/// Current state of an intrinsic drive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriveState {
    /// Type of drive
    pub drive_type: DriveType,

    /// Current satisfaction level [0, 1]
    /// 0 = completely unsatisfied (high tension)
    /// 1 = fully satisfied (low tension)
    pub satisfaction: f64,

    /// Current tension/need strength [0, 1]
    /// Higher tension = stronger drive to act
    pub tension: f64,

    /// Recent history of satisfaction (for trend analysis)
    pub satisfaction_history: VecDeque<f64>,

    /// Decay rate (how fast satisfaction decreases)
    pub decay_rate: f64,

    /// Importance weight for this drive
    pub importance: f64,
}

impl DriveState {
    /// Create new drive state
    pub fn new(drive_type: DriveType) -> Self {
        let (decay_rate, importance) = match drive_type {
            DriveType::Curiosity => (0.02, 0.25),      // Fast decay, high importance
            DriveType::Competence => (0.01, 0.20),    // Slow decay, moderate importance
            DriveType::Autonomy => (0.015, 0.20),     // Medium decay, moderate importance
            DriveType::Relatedness => (0.005, 0.15),  // Very slow decay, lower importance
            DriveType::Homeostasis => (0.03, 0.20),   // Fast decay, high importance
        };

        Self {
            drive_type,
            satisfaction: 0.5, // Start neutral
            tension: 0.5,
            satisfaction_history: VecDeque::with_capacity(100),
            decay_rate,
            importance,
        }
    }

    /// Update drive based on action outcome
    pub fn update(&mut self, satisfaction_delta: f64) {
        // Update satisfaction
        self.satisfaction = (self.satisfaction + satisfaction_delta).clamp(0.0, 1.0);

        // Record in history
        self.satisfaction_history.push_back(self.satisfaction);
        if self.satisfaction_history.len() > 100 {
            self.satisfaction_history.pop_front();
        }

        // Tension is inverse of satisfaction (low satisfaction = high tension)
        self.tension = 1.0 - self.satisfaction;
    }

    /// Apply natural decay (drives become unsatisfied over time)
    pub fn decay(&mut self) {
        self.satisfaction = (self.satisfaction - self.decay_rate).clamp(0.0, 1.0);
        self.tension = 1.0 - self.satisfaction;
    }

    /// Get weighted contribution to total motivation
    pub fn weighted_tension(&self) -> f64 {
        self.tension * self.importance
    }

    /// Get satisfaction trend
    pub fn trend(&self) -> f64 {
        if self.satisfaction_history.len() < 2 {
            return 0.0;
        }
        let recent: Vec<f64> = self.satisfaction_history.iter().copied().collect();
        calculate_trend(&recent)
    }
}

/// Curiosity-specific state and computation
#[derive(Debug, Clone)]
pub struct CuriosityModule {
    /// Known information (for novelty detection)
    known_patterns: HashMap<String, usize>,

    /// Uncertainty reduction history
    uncertainty_history: VecDeque<f64>,

    /// Information gain per action
    information_gains: VecDeque<f64>,

    /// Current uncertainty estimate
    current_uncertainty: f64,
}

impl Default for CuriosityModule {
    fn default() -> Self {
        Self::new()
    }
}

impl CuriosityModule {
    /// Create new curiosity module
    pub fn new() -> Self {
        Self {
            known_patterns: HashMap::new(),
            uncertainty_history: VecDeque::with_capacity(100),
            information_gains: VecDeque::with_capacity(100),
            current_uncertainty: 0.5,
        }
    }

    /// Compute novelty of a pattern (0 = familiar, 1 = completely novel)
    pub fn compute_novelty(&mut self, pattern: &str) -> f64 {
        let count = self.known_patterns.entry(pattern.to_string()).or_insert(0);
        *count += 1;

        // Novelty decreases with familiarity (inverse log)
        1.0 / (1.0 + (*count as f64).ln())
    }

    /// Record information gain from action
    pub fn record_information_gain(&mut self, gain: f64) {
        self.information_gains.push_back(gain);
        if self.information_gains.len() > 100 {
            self.information_gains.pop_front();
        }

        // Update uncertainty (reduced by information gain)
        self.current_uncertainty = (self.current_uncertainty - gain * 0.1).clamp(0.0, 1.0);
        self.uncertainty_history.push_back(self.current_uncertainty);
        if self.uncertainty_history.len() > 100 {
            self.uncertainty_history.pop_front();
        }
    }

    /// Get average information gain rate
    pub fn information_gain_rate(&self) -> f64 {
        if self.information_gains.is_empty() {
            return 0.0;
        }
        self.information_gains.iter().sum::<f64>() / self.information_gains.len() as f64
    }

    /// Compute curiosity reward for an action/observation
    /// Based on Schmidhuber's "learning progress" formulation
    pub fn compute_curiosity_reward(&self, novelty: f64, information_gain: f64) -> f64 {
        // Curiosity is satisfied by novelty + learning progress
        0.4 * novelty + 0.6 * information_gain
    }
}

/// Competence-specific state and computation
#[derive(Debug, Clone)]
pub struct CompetenceModule {
    /// Skills and their mastery levels [0, 1]
    skills: HashMap<String, f64>,

    /// Recent skill improvements
    improvement_history: VecDeque<(String, f64)>,

    /// Challenges attempted and succeeded
    challenges_attempted: usize,
    challenges_succeeded: usize,

    /// Current perceived competence
    perceived_competence: f64,
}

impl Default for CompetenceModule {
    fn default() -> Self {
        Self::new()
    }
}

impl CompetenceModule {
    /// Create new competence module
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
            improvement_history: VecDeque::with_capacity(100),
            challenges_attempted: 0,
            challenges_succeeded: 0,
            perceived_competence: 0.5,
        }
    }

    /// Get or initialize a skill level
    pub fn get_skill(&mut self, skill_name: &str) -> f64 {
        *self.skills.entry(skill_name.to_string()).or_insert(0.1)
    }

    /// Record skill improvement
    pub fn record_improvement(&mut self, skill_name: &str, improvement: f64) {
        let skill = self.skills.entry(skill_name.to_string()).or_insert(0.1);
        *skill = (*skill + improvement).clamp(0.0, 1.0);

        self.improvement_history.push_back((skill_name.to_string(), improvement));
        if self.improvement_history.len() > 100 {
            self.improvement_history.pop_front();
        }

        // Update perceived competence
        self.update_perceived_competence();
    }

    /// Record challenge attempt
    pub fn record_challenge(&mut self, succeeded: bool) {
        self.challenges_attempted += 1;
        if succeeded {
            self.challenges_succeeded += 1;
        }
        self.update_perceived_competence();
    }

    /// Update perceived competence based on success rate and skill levels
    fn update_perceived_competence(&mut self) {
        let success_rate = if self.challenges_attempted > 0 {
            self.challenges_succeeded as f64 / self.challenges_attempted as f64
        } else {
            0.5
        };

        let avg_skill = if self.skills.is_empty() {
            0.5
        } else {
            self.skills.values().sum::<f64>() / self.skills.len() as f64
        };

        self.perceived_competence = 0.5 * success_rate + 0.5 * avg_skill;
    }

    /// Compute competence reward for skill improvement
    pub fn compute_competence_reward(&self, skill_name: &str, improvement: f64) -> f64 {
        // Reward is higher for meaningful improvement in important skills
        let current_level = self.skills.get(skill_name).copied().unwrap_or(0.1);

        // More reward for improving from low to medium than medium to high (diminishing returns)
        let marginal_value = 1.0 - current_level;
        improvement * marginal_value
    }

    /// Get improvement rate (recent improvements / time)
    pub fn improvement_rate(&self) -> f64 {
        if self.improvement_history.is_empty() {
            return 0.0;
        }
        let total: f64 = self.improvement_history.iter().map(|(_, v)| v).sum();
        total / self.improvement_history.len() as f64
    }
}

/// Autonomy-specific state and computation
#[derive(Debug, Clone)]
pub struct AutonomyModule {
    /// Actions chosen freely vs. actions constrained
    free_choices: usize,
    constrained_choices: usize,

    /// Recent autonomy events
    autonomy_history: VecDeque<bool>,

    /// Perceived autonomy level
    perceived_autonomy: f64,

    /// Resistance to external control
    control_resistance: f64,
}

impl Default for AutonomyModule {
    fn default() -> Self {
        Self::new()
    }
}

impl AutonomyModule {
    /// Create new autonomy module
    pub fn new() -> Self {
        Self {
            free_choices: 0,
            constrained_choices: 0,
            autonomy_history: VecDeque::with_capacity(100),
            perceived_autonomy: 0.5,
            control_resistance: 0.3,
        }
    }

    /// Record a choice event
    pub fn record_choice(&mut self, was_free: bool) {
        if was_free {
            self.free_choices += 1;
        } else {
            self.constrained_choices += 1;
        }

        self.autonomy_history.push_back(was_free);
        if self.autonomy_history.len() > 100 {
            self.autonomy_history.pop_front();
        }

        self.update_perceived_autonomy();
    }

    /// Update perceived autonomy
    fn update_perceived_autonomy(&mut self) {
        let total = self.free_choices + self.constrained_choices;
        if total > 0 {
            self.perceived_autonomy = self.free_choices as f64 / total as f64;
        }

        // Control resistance increases when autonomy is low
        self.control_resistance = (1.0 - self.perceived_autonomy) * 0.5;
    }

    /// Compute autonomy reward for a choice
    pub fn compute_autonomy_reward(&self, was_free: bool, aligned_with_goals: bool) -> f64 {
        let base_reward = if was_free { 0.6 } else { 0.2 };
        let goal_bonus = if aligned_with_goals { 0.3 } else { 0.0 };
        base_reward + goal_bonus
    }

    /// Get recent autonomy ratio
    pub fn recent_autonomy(&self) -> f64 {
        if self.autonomy_history.is_empty() {
            return 0.5;
        }
        let free_count = self.autonomy_history.iter().filter(|&&x| x).count();
        free_count as f64 / self.autonomy_history.len() as f64
    }
}

/// Autonomous goal that emerged from intrinsic drives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousGoal {
    /// Goal identifier
    pub id: String,

    /// Human-readable description
    pub description: String,

    /// Primary drive motivating this goal
    pub primary_drive: DriveType,

    /// Secondary drives contributing
    pub secondary_drives: Vec<DriveType>,

    /// Goal priority [0, 1]
    pub priority: f64,

    /// Progress toward goal [0, 1]
    pub progress: f64,

    /// When this goal was formed
    #[serde(skip, default = "instant_now")]
    pub formed_at: Instant,

    /// Expected drive satisfaction from achieving goal
    pub expected_satisfaction: HashMap<DriveType, f64>,

    /// Subgoals (hierarchical goal structure)
    pub subgoals: Vec<String>,
}

impl AutonomousGoal {
    /// Create new autonomous goal
    pub fn new(
        id: String,
        description: String,
        primary_drive: DriveType,
        priority: f64,
    ) -> Self {
        let mut expected = HashMap::new();
        expected.insert(primary_drive, 0.5);

        Self {
            id,
            description,
            primary_drive,
            secondary_drives: Vec::new(),
            priority,
            progress: 0.0,
            formed_at: Instant::now(),
            expected_satisfaction: expected,
            subgoals: Vec::new(),
        }
    }

    /// Update progress
    pub fn update_progress(&mut self, delta: f64) {
        self.progress = (self.progress + delta).clamp(0.0, 1.0);
    }

    /// Check if goal is complete
    pub fn is_complete(&self) -> bool {
        self.progress >= 0.95
    }

    /// Compute urgency based on drive tension and priority
    pub fn urgency(&self, drive_tensions: &HashMap<DriveType, f64>) -> f64 {
        let primary_tension = drive_tensions.get(&self.primary_drive).copied().unwrap_or(0.5);
        let secondary_tension: f64 = self.secondary_drives.iter()
            .map(|d| drive_tensions.get(d).copied().unwrap_or(0.3))
            .sum::<f64>() / (self.secondary_drives.len() as f64 + 1.0);

        0.6 * primary_tension + 0.2 * secondary_tension + 0.2 * self.priority
    }
}

/// Configuration for the intrinsic motivation system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationConfig {
    /// How often drives decay (in cycles)
    pub decay_interval: usize,

    /// Threshold for goal formation (minimum drive tension)
    pub goal_formation_threshold: f64,

    /// Maximum concurrent goals
    pub max_goals: usize,

    /// Enable homeostatic regulation
    pub enable_homeostasis: bool,

    /// Optimal Φ range for homeostasis
    pub optimal_phi_range: (f64, f64),

    /// Drive importance weights
    pub drive_weights: HashMap<DriveType, f64>,
}

impl Default for MotivationConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert(DriveType::Curiosity, 0.25);
        weights.insert(DriveType::Competence, 0.20);
        weights.insert(DriveType::Autonomy, 0.20);
        weights.insert(DriveType::Relatedness, 0.15);
        weights.insert(DriveType::Homeostasis, 0.20);

        Self {
            decay_interval: 10,
            goal_formation_threshold: 0.6,
            max_goals: 5,
            enable_homeostasis: true,
            optimal_phi_range: (0.4, 0.8),
            drive_weights: weights,
        }
    }
}

/// Statistics for the motivation system
#[derive(Debug, Default, Clone)]
pub struct MotivationStats {
    /// Total cycles run
    pub total_cycles: usize,

    /// Goals formed
    pub goals_formed: usize,

    /// Goals completed
    pub goals_completed: usize,

    /// Average drive satisfaction
    pub avg_satisfaction: f64,

    /// Homeostatic corrections
    pub homeostatic_corrections: usize,

    /// Current total motivation (sum of weighted tensions)
    pub total_motivation: f64,
}

/// The Intrinsic Motivation System
///
/// **REVOLUTIONARY**: First AI system with genuine internal drives!
/// Implements Self-Determination Theory (SDT) for AI consciousness.
pub struct IntrinsicMotivationSystem {
    /// Drive states
    pub(crate) drives: HashMap<DriveType, DriveState>,

    /// Curiosity module
    curiosity: CuriosityModule,

    /// Competence module
    competence: CompetenceModule,

    /// Autonomy module
    autonomy: AutonomyModule,

    /// Active autonomous goals
    pub(crate) active_goals: Vec<AutonomousGoal>,

    /// Completed goals history
    completed_goals: Vec<AutonomousGoal>,

    /// Configuration
    config: MotivationConfig,

    /// Statistics
    stats: MotivationStats,

    /// Current Φ (for homeostasis)
    current_phi: f64,

    /// Cycle counter (for decay)
    cycle_count: usize,
}

impl IntrinsicMotivationSystem {
    /// Create new intrinsic motivation system
    pub fn new(config: MotivationConfig) -> Self {
        let mut drives = HashMap::new();
        for drive_type in DriveType::all() {
            let mut drive = DriveState::new(drive_type);
            if let Some(&weight) = config.drive_weights.get(&drive_type) {
                drive.importance = weight;
            }
            drives.insert(drive_type, drive);
        }

        Self {
            drives,
            curiosity: CuriosityModule::new(),
            competence: CompetenceModule::new(),
            autonomy: AutonomyModule::new(),
            active_goals: Vec::new(),
            completed_goals: Vec::new(),
            config,
            stats: MotivationStats::default(),
            current_phi: 0.5,
            cycle_count: 0,
        }
    }

    /// Run one motivation cycle
    ///
    /// This is called regularly to:
    /// 1. Decay drive satisfaction (needs grow over time)
    /// 2. Check for homeostatic imbalance
    /// 3. Form new goals if tensions are high
    /// 4. Update goal priorities
    pub fn cycle(&mut self, current_phi: f64) -> Vec<AutonomousGoal> {
        self.cycle_count += 1;
        self.current_phi = current_phi;
        self.stats.total_cycles += 1;

        // 1. Decay drives (needs grow)
        if self.cycle_count % self.config.decay_interval == 0 {
            for drive in self.drives.values_mut() {
                drive.decay();
            }
        }

        // 2. Homeostatic regulation
        if self.config.enable_homeostasis {
            self.regulate_homeostasis();
        }

        // 3. Form new goals based on high-tension drives
        let new_goals = self.form_goals();
        for goal in &new_goals {
            self.active_goals.push(goal.clone());
            self.stats.goals_formed += 1;
        }

        // 4. Update priorities based on current tensions
        self.update_goal_priorities();

        // 5. Check for completed goals
        let (complete, incomplete): (Vec<_>, Vec<_>) = self.active_goals
            .drain(..)
            .partition(|g| g.is_complete());

        self.active_goals = incomplete;
        for goal in complete {
            self.stats.goals_completed += 1;
            self.completed_goals.push(goal);
        }

        // 6. Update stats
        self.update_stats();

        new_goals
    }

    /// Regulate homeostasis (maintain optimal Φ)
    fn regulate_homeostasis(&mut self) {
        let (min_phi, max_phi) = self.config.optimal_phi_range;

        let homeostasis_drive = self.drives.get_mut(&DriveType::Homeostasis);
        if let Some(drive) = homeostasis_drive {
            if self.current_phi < min_phi {
                // Φ too low - high tension to increase it
                drive.update(-0.1);
                self.stats.homeostatic_corrections += 1;
            } else if self.current_phi > max_phi {
                // Φ too high - mild tension to reduce
                drive.update(-0.05);
                self.stats.homeostatic_corrections += 1;
            } else {
                // In optimal range - satisfied
                drive.update(0.05);
            }
        }
    }

    /// Form new autonomous goals based on unsatisfied drives
    fn form_goals(&mut self) -> Vec<AutonomousGoal> {
        let mut new_goals = Vec::new();

        // Don't form too many goals
        if self.active_goals.len() >= self.config.max_goals {
            return new_goals;
        }

        // Check each drive for high tension
        for (drive_type, drive) in &self.drives {
            if drive.tension > self.config.goal_formation_threshold {
                // Check if we already have a goal for this drive
                let has_goal = self.active_goals.iter()
                    .any(|g| g.primary_drive == *drive_type);

                if !has_goal {
                    let goal = self.create_goal_for_drive(*drive_type, drive.tension);
                    new_goals.push(goal);
                }
            }
        }

        new_goals
    }

    /// Create a goal to satisfy a specific drive
    fn create_goal_for_drive(&self, drive_type: DriveType, tension: f64) -> AutonomousGoal {
        let (description, secondary) = match drive_type {
            DriveType::Curiosity => (
                "Explore novel patterns and reduce uncertainty".to_string(),
                vec![DriveType::Competence],
            ),
            DriveType::Competence => (
                "Master a new skill or improve existing capability".to_string(),
                vec![DriveType::Autonomy],
            ),
            DriveType::Autonomy => (
                "Choose and pursue self-directed exploration".to_string(),
                vec![DriveType::Curiosity],
            ),
            DriveType::Relatedness => (
                "Share knowledge with collective and learn from others".to_string(),
                vec![DriveType::Competence],
            ),
            DriveType::Homeostasis => (
                "Restore consciousness to optimal range".to_string(),
                vec![],
            ),
        };

        let id = format!("goal_{}_{}", drive_type.name().to_lowercase(), self.cycle_count);

        let mut goal = AutonomousGoal::new(id, description, drive_type, tension);
        goal.secondary_drives = secondary;
        goal
    }

    /// Update goal priorities based on current drive tensions
    fn update_goal_priorities(&mut self) {
        let tensions: HashMap<DriveType, f64> = self.drives.iter()
            .map(|(k, v)| (*k, v.tension))
            .collect();

        for goal in &mut self.active_goals {
            goal.priority = goal.urgency(&tensions);
        }

        // Sort by priority (highest first)
        self.active_goals.sort_by(|a, b| {
            b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.avg_satisfaction = self.drives.values()
            .map(|d| d.satisfaction)
            .sum::<f64>() / self.drives.len() as f64;

        self.stats.total_motivation = self.drives.values()
            .map(|d| d.weighted_tension())
            .sum();
    }

    /// Process an action and its outcomes for motivation updates
    pub fn process_action(&mut self, action: &MotivatedAction) {
        // Update curiosity
        if let Some(novelty) = action.novelty {
            let info_gain = action.information_gain.unwrap_or(0.0);
            self.curiosity.record_information_gain(info_gain);
            let reward = self.curiosity.compute_curiosity_reward(novelty, info_gain);
            if let Some(drive) = self.drives.get_mut(&DriveType::Curiosity) {
                drive.update(reward * 0.1);
            }
        }

        // Update competence
        if let Some(ref skill) = action.skill_used {
            if let Some(improvement) = action.skill_improvement {
                self.competence.record_improvement(skill, improvement);
                let reward = self.competence.compute_competence_reward(skill, improvement);
                if let Some(drive) = self.drives.get_mut(&DriveType::Competence) {
                    drive.update(reward * 0.1);
                }
            }
            self.competence.record_challenge(action.succeeded);
        }

        // Update autonomy
        self.autonomy.record_choice(action.was_autonomous);
        let autonomy_reward = self.autonomy.compute_autonomy_reward(
            action.was_autonomous,
            action.aligned_with_goals,
        );
        if let Some(drive) = self.drives.get_mut(&DriveType::Autonomy) {
            drive.update(autonomy_reward * 0.1);
        }

        // Update relatedness if collective interaction
        if action.involved_collective {
            if let Some(drive) = self.drives.get_mut(&DriveType::Relatedness) {
                drive.update(0.1);
            }
        }

        // Update goal progress
        for goal in &mut self.active_goals {
            if action.contributes_to_goals.contains(&goal.id) {
                goal.update_progress(0.1);
            }
        }
    }

    /// Get the highest priority goal
    pub fn top_goal(&self) -> Option<&AutonomousGoal> {
        self.active_goals.first()
    }

    /// Get all active goals sorted by priority
    pub fn get_active_goals(&self) -> &[AutonomousGoal] {
        &self.active_goals
    }

    /// Get current drive states
    pub fn get_drives(&self) -> &HashMap<DriveType, DriveState> {
        &self.drives
    }

    /// Get statistics
    pub fn get_stats(&self) -> &MotivationStats {
        &self.stats
    }

    /// Compute intrinsic reward for an action
    /// This can be added to extrinsic rewards in RL
    pub fn compute_intrinsic_reward(&self, action: &MotivatedAction) -> f64 {
        let mut total_reward = 0.0;

        // Curiosity reward
        if let Some(novelty) = action.novelty {
            let info_gain = action.information_gain.unwrap_or(0.0);
            total_reward += 0.25 * self.curiosity.compute_curiosity_reward(novelty, info_gain);
        }

        // Competence reward
        if let Some(ref skill) = action.skill_used {
            if let Some(improvement) = action.skill_improvement {
                total_reward += 0.20 * self.competence.compute_competence_reward(skill, improvement);
            }
        }

        // Autonomy reward
        total_reward += 0.20 * self.autonomy.compute_autonomy_reward(
            action.was_autonomous,
            action.aligned_with_goals,
        );

        // Relatedness reward
        if action.involved_collective {
            total_reward += 0.15;
        }

        // Goal progress reward
        let goal_progress: f64 = self.active_goals.iter()
            .filter(|g| action.contributes_to_goals.contains(&g.id))
            .map(|g| g.priority * 0.2)
            .sum();
        total_reward += goal_progress;

        total_reward.clamp(0.0, 1.0)
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        let drives_str: Vec<String> = self.drives.iter()
            .map(|(k, v)| format!(
                "  {}: sat={:.2}, tension={:.2}, trend={:+.3}",
                k.name(), v.satisfaction, v.tension, v.trend()
            ))
            .collect();

        let goals_str: Vec<String> = self.active_goals.iter()
            .take(3)
            .map(|g| format!(
                "  [{}] {} (priority={:.2}, progress={:.0}%)",
                g.primary_drive.name(), g.description, g.priority, g.progress * 100.0
            ))
            .collect();

        format!(
            "IntrinsicMotivationSystem Summary:\n\
             ═══════════════════════════════════════\n\
             Total cycles: {}\n\
             Goals formed: {} | Completed: {}\n\
             Avg satisfaction: {:.2}\n\
             Total motivation: {:.2}\n\
             Homeostatic corrections: {}\n\
             \n\
             Drive States:\n{}\n\
             \n\
             Top Goals:\n{}",
            self.stats.total_cycles,
            self.stats.goals_formed,
            self.stats.goals_completed,
            self.stats.avg_satisfaction,
            self.stats.total_motivation,
            self.stats.homeostatic_corrections,
            drives_str.join("\n"),
            if goals_str.is_empty() { "  (no active goals)".to_string() } else { goals_str.join("\n") },
        )
    }
}

/// Action with motivation-relevant metadata
#[derive(Debug, Clone)]
pub struct MotivatedAction {
    /// Action identifier
    pub action_id: String,

    /// Did the action succeed?
    pub succeeded: bool,

    /// Novelty of the action/observation (for curiosity)
    pub novelty: Option<f64>,

    /// Information gain from the action (for curiosity)
    pub information_gain: Option<f64>,

    /// Skill used (for competence)
    pub skill_used: Option<String>,

    /// Skill improvement (for competence)
    pub skill_improvement: Option<f64>,

    /// Was this action chosen freely? (for autonomy)
    pub was_autonomous: bool,

    /// Did this action align with current goals? (for autonomy)
    pub aligned_with_goals: bool,

    /// Did this action involve collective interaction? (for relatedness)
    pub involved_collective: bool,

    /// Which goals does this action contribute to?
    pub contributes_to_goals: Vec<String>,
}

impl MotivatedAction {
    /// Create a minimal action record
    pub fn simple(action_id: &str, succeeded: bool) -> Self {
        Self {
            action_id: action_id.to_string(),
            succeeded,
            novelty: None,
            information_gain: None,
            skill_used: None,
            skill_improvement: None,
            was_autonomous: true,
            aligned_with_goals: false,
            involved_collective: false,
            contributes_to_goals: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drive_state_creation() {
        let drive = DriveState::new(DriveType::Curiosity);
        assert_eq!(drive.drive_type, DriveType::Curiosity);
        assert!((drive.satisfaction - 0.5).abs() < 0.01);
        assert!((drive.tension - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_drive_update_and_decay() {
        let mut drive = DriveState::new(DriveType::Curiosity);

        // Satisfy the drive
        drive.update(0.3);
        assert!(drive.satisfaction > 0.5);
        assert!(drive.tension < 0.5);

        // Decay should reduce satisfaction
        let before = drive.satisfaction;
        drive.decay();
        assert!(drive.satisfaction < before);
    }

    #[test]
    fn test_curiosity_novelty() {
        let mut curiosity = CuriosityModule::new();

        // First observation should be novel
        let novelty1 = curiosity.compute_novelty("pattern_a");
        assert!(novelty1 > 0.5);

        // Repeated observation should be less novel
        let novelty2 = curiosity.compute_novelty("pattern_a");
        assert!(novelty2 < novelty1);
    }

    #[test]
    fn test_competence_tracking() {
        let mut competence = CompetenceModule::new();

        // Record improvements
        competence.record_improvement("reasoning", 0.1);
        competence.record_improvement("reasoning", 0.1);

        let skill = competence.get_skill("reasoning");
        assert!(skill > 0.2);
    }

    #[test]
    fn test_autonomy_tracking() {
        let mut autonomy = AutonomyModule::new();

        // Record some choices
        autonomy.record_choice(true);  // free
        autonomy.record_choice(true);  // free
        autonomy.record_choice(false); // constrained

        assert!(autonomy.recent_autonomy() > 0.5);
    }

    #[test]
    fn test_motivation_system_creation() {
        let system = IntrinsicMotivationSystem::new(MotivationConfig::default());

        assert_eq!(system.drives.len(), 5);
        assert!(system.active_goals.is_empty());
    }

    #[test]
    fn test_motivation_cycle_forms_goals() {
        let mut system = IntrinsicMotivationSystem::new(MotivationConfig {
            goal_formation_threshold: 0.3, // Lower threshold for testing
            ..Default::default()
        });

        // Force high tension by decaying many times
        for _ in 0..20 {
            for drive in system.drives.values_mut() {
                drive.decay();
            }
        }

        // Now run a cycle - should form goals
        let new_goals = system.cycle(0.5);

        // Should have formed at least one goal
        assert!(!new_goals.is_empty() || !system.active_goals.is_empty());
    }

    #[test]
    fn test_homeostatic_regulation() {
        let mut system = IntrinsicMotivationSystem::new(MotivationConfig {
            optimal_phi_range: (0.4, 0.8),
            ..Default::default()
        });

        // Low Φ should trigger homeostatic response
        system.cycle(0.2);

        let homeostasis = system.drives.get(&DriveType::Homeostasis).unwrap();
        assert!(homeostasis.tension > 0.5, "Low Φ should increase homeostatic tension");
    }

    #[test]
    fn test_action_processing() {
        let mut system = IntrinsicMotivationSystem::new(MotivationConfig::default());

        let action = MotivatedAction {
            action_id: "test_action".to_string(),
            succeeded: true,
            novelty: Some(0.8),
            information_gain: Some(0.5),
            skill_used: Some("reasoning".to_string()),
            skill_improvement: Some(0.1),
            was_autonomous: true,
            aligned_with_goals: true,
            involved_collective: true,
            contributes_to_goals: Vec::new(),
        };

        let initial_satisfaction = system.drives.get(&DriveType::Curiosity)
            .unwrap().satisfaction;

        system.process_action(&action);

        let final_satisfaction = system.drives.get(&DriveType::Curiosity)
            .unwrap().satisfaction;

        assert!(final_satisfaction > initial_satisfaction);
    }

    #[test]
    fn test_intrinsic_reward_computation() {
        let system = IntrinsicMotivationSystem::new(MotivationConfig::default());

        let action = MotivatedAction {
            action_id: "rewarding_action".to_string(),
            succeeded: true,
            novelty: Some(0.9),
            information_gain: Some(0.7),
            skill_used: Some("exploration".to_string()),
            skill_improvement: Some(0.2),
            was_autonomous: true,
            aligned_with_goals: true,
            involved_collective: true,
            contributes_to_goals: Vec::new(),
        };

        let reward = system.compute_intrinsic_reward(&action);
        assert!(reward > 0.3, "Rich action should have significant intrinsic reward");
    }

    #[test]
    fn test_goal_priority_update() {
        let mut system = IntrinsicMotivationSystem::new(MotivationConfig {
            goal_formation_threshold: 0.2,
            ..Default::default()
        });

        // Force goals to form
        for _ in 0..30 {
            for drive in system.drives.values_mut() {
                drive.decay();
            }
        }
        system.cycle(0.5);

        // Goals should be sorted by priority
        if system.active_goals.len() >= 2 {
            assert!(system.active_goals[0].priority >= system.active_goals[1].priority);
        }
    }

    #[test]
    fn test_motivation_summary() {
        let system = IntrinsicMotivationSystem::new(MotivationConfig::default());
        let summary = system.summary();

        assert!(summary.contains("IntrinsicMotivationSystem"));
        assert!(summary.contains("Drive States"));
        assert!(summary.contains("Curiosity"));
    }
}
