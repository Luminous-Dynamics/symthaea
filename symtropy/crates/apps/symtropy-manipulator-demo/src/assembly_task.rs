// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Assembly task: 9-phase pick-place cycle state machine.
//!
//! The arm picks an object from position A, places it at position B, and returns.
//! Cycle time is tracked to measure throughput impact of safety approaches.

/// Assembly phase in the pick-place cycle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AssemblyPhase {
    /// Move EE above pick position at approach height.
    ApproachPick,
    /// Lower EE to grasp height.
    Descend,
    /// Close gripper (hold for settle time).
    Grasp,
    /// Lift EE to approach height.
    Lift,
    /// Move EE above place position at approach height.
    Transit,
    /// Lower EE to grasp height at place position.
    DescendPlace,
    /// Open gripper.
    Release,
    /// Lift EE to approach height.
    Retract,
    /// Return to approach position above pick.
    ReturnToStart,
}

impl AssemblyPhase {
    /// Advance to the next phase, looping at the end.
    pub fn next(self) -> Self {
        match self {
            Self::ApproachPick => Self::Descend,
            Self::Descend => Self::Grasp,
            Self::Grasp => Self::Lift,
            Self::Lift => Self::Transit,
            Self::Transit => Self::DescendPlace,
            Self::DescendPlace => Self::Release,
            Self::Release => Self::Retract,
            Self::Retract => Self::ReturnToStart,
            Self::ReturnToStart => Self::ApproachPick,
        }
    }
}

/// Pick-place assembly task.
pub struct AssemblyTask {
    /// Pick location [x, y, z] on the table.
    pub pick_position: [f64; 3],
    /// Place location [x, y, z] on the table.
    pub place_position: [f64; 3],
    /// Height above table for approach/retract (meters).
    pub approach_height: f64,
    /// Height of object (grasp height, meters).
    pub grasp_height: f64,
    /// Current phase.
    pub phase: AssemblyPhase,
    /// Position tolerance for phase transition (meters).
    pub waypoint_tolerance: f64,
    /// Gripper settle time (seconds).
    pub gripper_settle_time: f64,
    /// Timer for timed phases (Grasp, Release).
    pub phase_timer: f64,
    /// Whether the task is paused (due to safety).
    pub paused: bool,
    /// Total completed cycles.
    pub completed_cycles: u32,
}

impl AssemblyTask {
    pub fn new() -> Self {
        Self {
            pick_position: [0.4, -0.3, 0.15],
            place_position: [0.4, 0.3, 0.15],
            approach_height: 0.30,
            grasp_height: 0.08,
            phase: AssemblyPhase::ApproachPick,
            waypoint_tolerance: 0.015,
            gripper_settle_time: 0.2,
            phase_timer: 0.0,
            paused: false,
            completed_cycles: 0,
        }
    }

    /// Get the current IK target position for this phase.
    pub fn current_target(&self) -> [f64; 3] {
        match self.phase {
            AssemblyPhase::ApproachPick | AssemblyPhase::ReturnToStart => [
                self.pick_position[0],
                self.pick_position[1],
                self.approach_height,
            ],
            AssemblyPhase::Descend | AssemblyPhase::Grasp => [
                self.pick_position[0],
                self.pick_position[1],
                self.grasp_height,
            ],
            AssemblyPhase::Lift => [
                self.pick_position[0],
                self.pick_position[1],
                self.approach_height,
            ],
            AssemblyPhase::Transit => [
                self.place_position[0],
                self.place_position[1],
                self.approach_height,
            ],
            AssemblyPhase::DescendPlace | AssemblyPhase::Release => [
                self.place_position[0],
                self.place_position[1],
                self.grasp_height,
            ],
            AssemblyPhase::Retract => [
                self.place_position[0],
                self.place_position[1],
                self.approach_height,
            ],
        }
    }

    /// Desired gripper opening for this phase (0.0 = closed, 1.0 = open).
    pub fn desired_gripper(&self) -> f64 {
        match self.phase {
            AssemblyPhase::Grasp
            | AssemblyPhase::Lift
            | AssemblyPhase::Transit
            | AssemblyPhase::DescendPlace => 0.0, // closed (holding object)
            _ => 1.0, // open
        }
    }

    /// Update the task state machine given the current EE position.
    /// Returns `true` if a full cycle just completed.
    pub fn update(&mut self, ee_position: &[f64; 3], dt: f64) -> bool {
        if self.paused {
            return false;
        }

        // For timed phases (Grasp, Release), tick the timer
        match self.phase {
            AssemblyPhase::Grasp | AssemblyPhase::Release => {
                self.phase_timer += dt;
                if self.phase_timer >= self.gripper_settle_time {
                    self.phase_timer = 0.0;
                    self.phase = self.phase.next();
                    return self.phase == AssemblyPhase::ApproachPick;
                }
                return false;
            }
            _ => {}
        }

        // For movement phases, check position tolerance
        let target = self.current_target();
        let dx = ee_position[0] - target[0];
        let dy = ee_position[1] - target[1];
        let dz = ee_position[2] - target[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist < self.waypoint_tolerance {
            let old_phase = self.phase;
            self.phase = self.phase.next();
            self.phase_timer = 0.0;

            // A full cycle completes when we return to ApproachPick from ReturnToStart
            if old_phase == AssemblyPhase::ReturnToStart {
                self.completed_cycles += 1;
                return true;
            }
        }

        false
    }

    /// Pause/unpause the task (for safety stops).
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_cycle_is_complete() {
        let mut phase = AssemblyPhase::ApproachPick;
        for _ in 0..9 {
            phase = phase.next();
        }
        assert_eq!(phase, AssemblyPhase::ApproachPick);
    }

    #[test]
    fn task_returns_sensible_targets() {
        let task = AssemblyTask::new();
        let target = task.current_target();
        assert!(target[2] > 0.0, "Target should be above ground");
        assert!(
            target[2] <= 0.30,
            "Target should be at or below approach height"
        );
    }

    #[test]
    fn gripper_closed_during_transit() {
        let mut task = AssemblyTask::new();
        task.phase = AssemblyPhase::Transit;
        assert_eq!(task.desired_gripper(), 0.0);
    }

    #[test]
    fn gripper_open_during_approach() {
        let task = AssemblyTask::new();
        assert_eq!(task.desired_gripper(), 1.0);
    }

    #[test]
    fn cycle_completes_on_return() {
        let mut task = AssemblyTask::new();
        task.phase = AssemblyPhase::ReturnToStart;
        let target = task.current_target();
        let completed = task.update(&target, 0.002);
        assert!(completed);
        assert_eq!(task.completed_cycles, 1);
        assert_eq!(task.phase, AssemblyPhase::ApproachPick);
    }

    #[test]
    fn timed_phase_waits() {
        let mut task = AssemblyTask::new();
        task.phase = AssemblyPhase::Grasp;
        let target = task.current_target();
        // First update: timer starts but hasn't elapsed
        assert!(!task.update(&target, 0.01));
        assert_eq!(task.phase, AssemblyPhase::Grasp);
        // After enough time: phase advances
        assert!(!task.update(&target, 0.25));
        assert_eq!(task.phase, AssemblyPhase::Lift);
    }

    #[test]
    fn paused_task_does_not_advance() {
        let mut task = AssemblyTask::new();
        task.phase = AssemblyPhase::ReturnToStart;
        task.set_paused(true);
        let target = task.current_target();
        assert!(!task.update(&target, 0.002));
        assert_eq!(task.phase, AssemblyPhase::ReturnToStart);
    }
}
