// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-step task sequencer for goal-directed phone control.
//!
//! A task is a sequence of steps, each with a visual goal and an action type.
//! The sequencer drives the PhoneBridge through each step, detecting state
//! transitions to know when to advance.

/// A single step in a multi-step task.
#[derive(Debug, Clone)]
pub struct TaskStep {
    /// Human-readable description of this step.
    pub description: String,
    /// How to find the target for this step.
    pub target: StepTarget,
    /// What to do when the target is found.
    pub action: StepAction,
    /// Maximum attempts before giving up on this step.
    pub max_attempts: u32,
}

/// How to identify the target for a task step.
#[derive(Debug, Clone)]
pub enum StepTarget {
    /// Find an app by name (uses known grid positions or saved templates).
    AppByName(String),
    /// Find a visual element at a known grid region.
    GridRegion {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },
    /// Use the highest-saliency working memory object (exploration).
    MostSalient,
    /// No target needed (e.g., for typing into the focused field).
    None,
}

/// What action to take when the target is found.
#[derive(Debug, Clone)]
pub enum StepAction {
    /// Tap the matched location.
    Tap,
    /// Type text (assumes a text field is focused).
    Type(String),
    /// Swipe down to scroll.
    ScrollDown,
    /// Press back.
    Back,
    /// Press home.
    Home,
    /// Wait for a state transition (observe only).
    WaitForTransition,
}

/// A complete multi-step task.
#[derive(Debug, Clone)]
pub struct Task {
    /// Task name for logging.
    pub name: String,
    /// Ordered sequence of steps.
    pub steps: Vec<TaskStep>,
}

impl Task {
    /// Parse a simple task description into steps.
    ///
    /// Supports patterns like:
    /// - "open youtube" → [find youtube, tap]
    /// - "open youtube and search NixOS" → [find youtube, tap, wait, tap search, type "NixOS"]
    /// - "open settings" → [find settings, tap]
    pub fn parse(description: &str) -> Self {
        let desc = description.to_lowercase();

        // Pattern: "open X and search Y"
        if let Some(rest) = desc.strip_prefix("open ") {
            if let Some((app, query)) = rest.split_once(" and search ") {
                return Self {
                    name: description.to_string(),
                    steps: vec![
                        TaskStep {
                            description: format!("Find {app}"),
                            target: StepTarget::AppByName(app.trim().to_string()),
                            action: StepAction::Tap,
                            max_attempts: 3,
                        },
                        TaskStep {
                            description: "Wait for app to open".to_string(),
                            target: StepTarget::None,
                            action: StepAction::WaitForTransition,
                            max_attempts: 1,
                        },
                        TaskStep {
                            description: "Tap search area".to_string(),
                            target: StepTarget::MostSalient,
                            action: StepAction::Tap,
                            max_attempts: 2,
                        },
                        TaskStep {
                            description: format!("Type \"{query}\""),
                            target: StepTarget::None,
                            action: StepAction::Type(query.trim().to_string()),
                            max_attempts: 1,
                        },
                    ],
                };
            }

            // Pattern: "open X"
            return Self {
                name: description.to_string(),
                steps: vec![TaskStep {
                    description: format!("Find {}", rest.trim()),
                    target: StepTarget::AppByName(rest.trim().to_string()),
                    action: StepAction::Tap,
                    max_attempts: 3,
                }],
            };
        }

        // Pattern: "find X"
        if let Some(app) = desc.strip_prefix("find ") {
            return Self {
                name: description.to_string(),
                steps: vec![TaskStep {
                    description: format!("Find {}", app.trim()),
                    target: StepTarget::AppByName(app.trim().to_string()),
                    action: StepAction::Tap,
                    max_attempts: 3,
                }],
            };
        }

        // Fallback: treat the whole thing as an app name
        Self {
            name: description.to_string(),
            steps: vec![TaskStep {
                description: format!("Find {desc}"),
                target: StepTarget::AppByName(desc.trim().to_string()),
                action: StepAction::Tap,
                max_attempts: 3,
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_open_app() {
        let task = Task::parse("open YouTube");
        assert_eq!(task.steps.len(), 1);
        assert!(matches!(task.steps[0].target, StepTarget::AppByName(ref s) if s == "youtube"));
    }

    #[test]
    fn test_parse_open_and_search() {
        let task = Task::parse("open YouTube and search NixOS");
        assert_eq!(task.steps.len(), 4);
        assert!(matches!(task.steps[0].action, StepAction::Tap));
        assert!(matches!(task.steps[1].action, StepAction::WaitForTransition));
        assert!(matches!(task.steps[2].action, StepAction::Tap));
        assert!(matches!(task.steps[3].action, StepAction::Type(ref s) if s == "nixos"));
    }

    #[test]
    fn test_parse_find_app() {
        let task = Task::parse("find Settings");
        assert_eq!(task.steps.len(), 1);
        assert!(matches!(task.steps[0].target, StepTarget::AppByName(ref s) if s == "settings"));
    }
}
