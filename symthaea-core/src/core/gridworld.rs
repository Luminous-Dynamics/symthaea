//! # GridWorld Domain Adapter
//!
//! A simple 2D grid world domain for validating the domain-agnostic generalization.
//! This demonstrates that Symthaea's infrastructure works across different domains,
//! not just consciousness.
//!
//! ## Domain Description
//!
//! - **State**: 2D position (x, y) in a bounded grid
//! - **Actions**: Up, Down, Left, Right
//! - **Goal**: Reach a target position
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::core::gridworld::{GridState, GridAction, GridGoal, GridWorldAdapter};
//! use symthaea::core::{State, Action, Goal, DomainAdapter, WorldModel};
//!
//! // Create a grid world
//! let adapter = GridWorldAdapter::new(10, 10);
//! let initial = adapter.initial_state();
//! let goal = GridGoal::new(9, 9);
//!
//! // Check available actions
//! let actions = adapter.available_actions(&initial);
//!
//! // Use with GenericWorldModel
//! let world_model = GenericWorldModel::<GridState, GridAction, _, _>::new(...);
//! ```

use std::fmt;

use crate::core::domain_traits::{
    Action, DomainAdapter, Goal, HdcEncodable, QualitySignal, State,
};
use crate::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ═══════════════════════════════════════════════════════════════════════════════
// GRID STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// A state in a 2D grid world.
#[derive(Debug, Clone, PartialEq)]
pub struct GridState {
    /// X coordinate (column)
    pub x: i32,
    /// Y coordinate (row)
    pub y: i32,
    /// Grid width bound
    pub width: i32,
    /// Grid height bound
    pub height: i32,
}

impl GridState {
    /// Create a new grid state.
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self { x, y, width, height }
    }

    /// Check if position is within bounds.
    pub fn is_valid(&self) -> bool {
        self.x >= 0 && self.x < self.width && self.y >= 0 && self.y < self.height
    }

    /// Apply an action and return the new state (clamped to bounds).
    pub fn apply(&self, action: &GridAction) -> Self {
        let (dx, dy) = action.delta();
        let new_x = (self.x + dx).clamp(0, self.width - 1);
        let new_y = (self.y + dy).clamp(0, self.height - 1);
        Self::new(new_x, new_y, self.width, self.height)
    }

    /// Manhattan distance to another state.
    pub fn manhattan_distance(&self, other: &Self) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }
}

impl State for GridState {
    fn to_features(&self) -> Vec<f64> {
        // Normalize coordinates to [0, 1]
        vec![
            self.x as f64 / (self.width - 1).max(1) as f64,
            self.y as f64 / (self.height - 1).max(1) as f64,
        ]
    }

    fn distance(&self, other: &Self) -> f64 {
        self.manhattan_distance(other) as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GRID ACTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Actions available in the grid world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GridAction {
    Up,
    Down,
    Left,
    Right,
    Stay,
}

impl GridAction {
    /// Get the (dx, dy) delta for this action.
    pub fn delta(&self) -> (i32, i32) {
        match self {
            GridAction::Up => (0, -1),
            GridAction::Down => (0, 1),
            GridAction::Left => (-1, 0),
            GridAction::Right => (1, 0),
            GridAction::Stay => (0, 0),
        }
    }

    /// All possible actions.
    pub fn all() -> Vec<GridAction> {
        vec![
            GridAction::Up,
            GridAction::Down,
            GridAction::Left,
            GridAction::Right,
            GridAction::Stay,
        ]
    }
}

impl Action for GridAction {
    fn action_id(&self) -> u64 {
        match self {
            GridAction::Up => 0,
            GridAction::Down => 1,
            GridAction::Left => 2,
            GridAction::Right => 3,
            GridAction::Stay => 4,
        }
    }

    fn describe(&self) -> String {
        format!("{:?}", self)
    }

    fn is_reversible(&self) -> bool {
        true
    }

    fn cost(&self) -> f64 {
        match self {
            GridAction::Stay => 0.0,
            _ => 1.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GRID GOAL
// ═══════════════════════════════════════════════════════════════════════════════

/// A goal to reach a specific grid position.
#[derive(Debug, Clone)]
pub struct GridGoal {
    /// Target X coordinate
    pub target_x: i32,
    /// Target Y coordinate
    pub target_y: i32,
    /// Reward for reaching the goal
    pub reward_value: f64,
}

impl GridGoal {
    /// Create a new grid goal.
    pub fn new(target_x: i32, target_y: i32) -> Self {
        Self {
            target_x,
            target_y,
            reward_value: 1.0,
        }
    }

    /// Create a goal with custom reward.
    pub fn with_reward(target_x: i32, target_y: i32, reward: f64) -> Self {
        Self {
            target_x,
            target_y,
            reward_value: reward,
        }
    }
}

impl Goal<GridState> for GridGoal {
    fn is_satisfied(&self, state: &GridState) -> bool {
        state.x == self.target_x && state.y == self.target_y
    }

    fn distance_to_goal(&self, state: &GridState) -> f64 {
        ((state.x - self.target_x).abs() + (state.y - self.target_y).abs()) as f64
    }

    fn reward(&self, state: &GridState) -> f64 {
        if self.is_satisfied(state) {
            self.reward_value
        } else {
            // Small negative reward proportional to distance
            -self.distance_to_goal(state) * 0.1
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HDC ENCODING
// ═══════════════════════════════════════════════════════════════════════════════

/// Lazy-static basis vectors for GridWorld HDC encoding.
mod grid_hdc_basis {
    use super::*;
    use std::sync::LazyLock;

    /// X-coordinate basis vector
    pub static X_BASIS: LazyLock<ContinuousHV> =
        LazyLock::new(|| ContinuousHV::random(HDC_DIMENSION, 1000));

    /// Y-coordinate basis vector
    pub static Y_BASIS: LazyLock<ContinuousHV> =
        LazyLock::new(|| ContinuousHV::random(HDC_DIMENSION, 1001));
}

impl HdcEncodable for GridState {
    type HyperVector = ContinuousHV;

    fn to_hv(&self) -> Self::HyperVector {
        // Encode position using permutation-based positional encoding.
        // Permute the X basis by x and Y basis by y to create unique signatures
        // for each position. This avoids the zero-scaling problem.
        let x_component = grid_hdc_basis::X_BASIS.permute(self.x as usize);
        let y_component = grid_hdc_basis::Y_BASIS.permute(self.y as usize);

        // Bind (multiply element-wise) x and y to create a unique position encoding
        // then add the original bases to preserve retrievability
        let bound = x_component.bind(&y_component);

        // Bundle with bases for hybrid encoding: position + context
        ContinuousHV::bundle(&[&bound, &grid_hdc_basis::X_BASIS, &grid_hdc_basis::Y_BASIS])
    }

    fn from_hv(hv: &Self::HyperVector) -> Option<Self> {
        // Decode by finding best matching position through brute force
        // (practical for small grids like 10x10)
        let mut best_sim = -1.0f32;
        let mut best_x = 0;
        let mut best_y = 0;

        for x in 0..10 {
            for y in 0..10 {
                let candidate = GridState::new(x, y, 10, 10);
                let candidate_hv = candidate.to_hv();
                let sim = hv.similarity(&candidate_hv);
                if sim > best_sim {
                    best_sim = sim;
                    best_x = x;
                    best_y = y;
                }
            }
        }

        Some(GridState::new(best_x, best_y, 10, 10))
    }

    fn semantic_similarity(&self, other: &Self) -> f64 {
        let self_hv = self.to_hv();
        let other_hv = other.to_hv();
        self_hv.similarity(&other_hv) as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DOMAIN ADAPTER
// ═══════════════════════════════════════════════════════════════════════════════

/// Quality signal measuring distance from goal.
pub struct GoalDistanceSignal {
    goal: GridGoal,
}

impl GoalDistanceSignal {
    pub fn new(goal: GridGoal) -> Self {
        Self { goal }
    }
}

impl QualitySignal<GridState> for GoalDistanceSignal {
    fn measure(&self, state: &GridState) -> f64 {
        // Invert distance so higher is better (1.0 when at goal)
        let max_dist = (state.width + state.height) as f64;
        let dist = self.goal.distance_to_goal(state);
        1.0 - (dist / max_dist)
    }

    fn name(&self) -> &'static str {
        "goal_distance"
    }
}

/// Adapter for the GridWorld domain.
pub struct GridWorldAdapter {
    width: i32,
    height: i32,
    start_x: i32,
    start_y: i32,
}

impl fmt::Debug for GridWorldAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GridWorldAdapter")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("start_x", &self.start_x)
            .field("start_y", &self.start_y)
            .finish()
    }
}

impl GridWorldAdapter {
    /// Create a new GridWorld adapter.
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            width,
            height,
            start_x: 0,
            start_y: 0,
        }
    }

    /// Create with custom starting position.
    pub fn with_start(width: i32, height: i32, start_x: i32, start_y: i32) -> Self {
        Self {
            width,
            height,
            start_x,
            start_y,
        }
    }
}

impl DomainAdapter<GridState, GridAction> for GridWorldAdapter {
    fn domain_name(&self) -> &'static str {
        "gridworld"
    }

    fn available_actions(&self, state: &GridState) -> Vec<GridAction> {
        let mut actions = vec![GridAction::Stay];

        if state.y > 0 {
            actions.push(GridAction::Up);
        }
        if state.y < self.height - 1 {
            actions.push(GridAction::Down);
        }
        if state.x > 0 {
            actions.push(GridAction::Left);
        }
        if state.x < self.width - 1 {
            actions.push(GridAction::Right);
        }

        actions
    }

    fn initial_state(&self) -> GridState {
        GridState::new(self.start_x, self.start_y, self.width, self.height)
    }

    fn quality_signals(&self) -> Vec<Box<dyn QualitySignal<GridState>>> {
        // Default: goal at bottom-right corner
        vec![Box::new(GoalDistanceSignal::new(GridGoal::new(
            self.width - 1,
            self.height - 1,
        )))]
    }

    fn is_valid_state(&self, state: &GridState) -> bool {
        state.is_valid() && state.width == self.width && state.height == self.height
    }

    fn is_valid_action(&self, state: &GridState, action: &GridAction) -> bool {
        let new_state = state.apply(action);
        new_state.is_valid()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTION PROVIDER (for dreaming support)
// ═══════════════════════════════════════════════════════════════════════════════

use crate::consciousness::recursive_improvement::world_model::ActionProvider;

impl ActionProvider<GridAction> for GridAction {
    fn all_actions() -> Vec<GridAction> {
        GridAction::all()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_state_creation() {
        let state = GridState::new(5, 5, 10, 10);
        assert_eq!(state.x, 5);
        assert_eq!(state.y, 5);
        assert!(state.is_valid());
    }

    #[test]
    fn test_grid_state_bounds() {
        let state = GridState::new(10, 10, 10, 10);
        assert!(!state.is_valid()); // Out of bounds

        let valid = GridState::new(9, 9, 10, 10);
        assert!(valid.is_valid());
    }

    #[test]
    fn test_grid_action_apply() {
        let state = GridState::new(5, 5, 10, 10);

        let up = state.apply(&GridAction::Up);
        assert_eq!(up.y, 4);

        let down = state.apply(&GridAction::Down);
        assert_eq!(down.y, 6);

        let left = state.apply(&GridAction::Left);
        assert_eq!(left.x, 4);

        let right = state.apply(&GridAction::Right);
        assert_eq!(right.x, 6);
    }

    #[test]
    fn test_grid_action_clamping() {
        let corner = GridState::new(0, 0, 10, 10);

        let up = corner.apply(&GridAction::Up);
        assert_eq!(up.y, 0); // Clamped

        let left = corner.apply(&GridAction::Left);
        assert_eq!(left.x, 0); // Clamped
    }

    #[test]
    fn test_grid_goal() {
        let state = GridState::new(5, 5, 10, 10);
        let goal = GridGoal::new(9, 9);

        assert!(!goal.is_satisfied(&state));
        assert_eq!(goal.distance_to_goal(&state), 8.0);

        let at_goal = GridState::new(9, 9, 10, 10);
        assert!(goal.is_satisfied(&at_goal));
        assert_eq!(goal.distance_to_goal(&at_goal), 0.0);
    }

    #[test]
    fn test_grid_state_trait() {
        let s1 = GridState::new(0, 0, 10, 10);
        let s2 = GridState::new(5, 5, 10, 10);

        // Distance should be Manhattan distance
        assert_eq!(s1.distance(&s2), 10.0);

        // Features should be normalized
        let features = s1.to_features();
        assert_eq!(features.len(), 2);
        assert_eq!(features[0], 0.0);
        assert_eq!(features[1], 0.0);
    }

    #[test]
    fn test_grid_hdc_encoding() {
        let s1 = GridState::new(0, 0, 10, 10);
        let s2 = GridState::new(9, 9, 10, 10);
        let s3 = GridState::new(0, 0, 10, 10);
        let s4 = GridState::new(1, 0, 10, 10); // Adjacent state

        // Same states should have high similarity
        let sim_same = s1.semantic_similarity(&s3);
        assert!(sim_same > 0.99, "Same state similarity: {}", sim_same);

        // Distant states should have lower similarity than same states
        let sim_distant = s1.semantic_similarity(&s2);
        assert!(
            sim_distant < sim_same,
            "Distant should be less similar: {} vs {}",
            sim_distant,
            sim_same
        );

        // Adjacent states should be more similar than distant states
        let sim_adjacent = s1.semantic_similarity(&s4);
        assert!(
            sim_adjacent > sim_distant,
            "Adjacent should be more similar than distant: {} vs {}",
            sim_adjacent,
            sim_distant
        );

        // Distant states should still be discriminable (significantly < 1.0)
        assert!(sim_distant < 0.95, "Distant states should be discriminable: {}", sim_distant);
    }

    #[test]
    fn test_grid_adapter() {
        let adapter = GridWorldAdapter::new(10, 10);
        let initial = adapter.initial_state();

        assert_eq!(initial.x, 0);
        assert_eq!(initial.y, 0);
        assert_eq!(adapter.domain_name(), "gridworld");

        // At corner, should only have right, down, and stay
        let actions = adapter.available_actions(&initial);
        assert!(actions.contains(&GridAction::Stay));
        assert!(actions.contains(&GridAction::Right));
        assert!(actions.contains(&GridAction::Down));
        assert!(!actions.contains(&GridAction::Up));
        assert!(!actions.contains(&GridAction::Left));
    }

    #[test]
    fn test_quality_signal() {
        let adapter = GridWorldAdapter::new(10, 10);
        let signals = adapter.quality_signals();
        assert_eq!(signals.len(), 1);

        let initial = adapter.initial_state();
        let quality = signals[0].measure(&initial);
        assert!(quality < 1.0); // Not at goal

        let at_goal = GridState::new(9, 9, 10, 10);
        let goal_quality = signals[0].measure(&at_goal);
        assert!((goal_quality - 1.0).abs() < 0.01); // At goal
    }

    #[test]
    fn test_action_provider() {
        let all = GridAction::all_actions();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&GridAction::Up));
        assert!(all.contains(&GridAction::Down));
        assert!(all.contains(&GridAction::Left));
        assert!(all.contains(&GridAction::Right));
        assert!(all.contains(&GridAction::Stay));
    }
}
