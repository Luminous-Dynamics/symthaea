// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reversible architecture-specific lesion adapters for GEOM-003.
//!
//! GEOM-003B1 uses the existing Global Workspace broadcast configuration
//! boundary. The lesion changes only `enable_broadcasting`; competition,
//! capacity, thresholds, decay, and duration remain untouched.

use std::fmt;

use symthaea_core::hdc::global_workspace::WorkspaceConfig;
use symthaea_geometric_interventions::ConditionMetadata;

/// Fixed experimental role for a workspace-broadcast run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceBroadcastRole {
    Intact,
    Lesion,
    Sham,
    Rescue,
}

/// One run specification emitted by the workspace-broadcast lesion adapter.
#[derive(Debug, Clone)]
pub struct WorkspaceBroadcastRunSpec {
    pub role: WorkspaceBroadcastRole,
    pub label: &'static str,
    pub metadata: ConditionMetadata,
    pub config: WorkspaceConfig,
    /// Number of explicit writes to `enable_broadcasting` used to construct
    /// this condition. Lesion and sham are matched at one write each.
    pub broadcast_field_writes: u8,
}

/// Complete matched quartet for the broadcast lesion.
#[derive(Debug, Clone)]
pub struct WorkspaceBroadcastLesionPlan {
    pub intact: WorkspaceBroadcastRunSpec,
    pub lesion: WorkspaceBroadcastRunSpec,
    pub sham: WorkspaceBroadcastRunSpec,
    pub rescue: WorkspaceBroadcastRunSpec,
}

/// Adapter failures that invalidate the lesion plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkspaceBroadcastLesionError {
    BaselineBroadcastingDisabled,
}

impl fmt::Display for WorkspaceBroadcastLesionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BaselineBroadcastingDisabled => write!(
                f,
                "workspace broadcast lesion requires an intact baseline with broadcasting enabled"
            ),
        }
    }
}

impl std::error::Error for WorkspaceBroadcastLesionError {}

fn config_matches_except_broadcasting(a: &WorkspaceConfig, b: &WorkspaceConfig) -> bool {
    a.max_capacity == b.max_capacity
        && a.entry_threshold.to_bits() == b.entry_threshold.to_bits()
        && a.decay_rate.to_bits() == b.decay_rate.to_bits()
        && a.winner_takes_all == b.winner_takes_all
        && a.max_duration == b.max_duration
}

/// Verify that two workspace configurations differ, if at all, only in the
/// broadcast-enable field.
pub fn differs_only_in_broadcasting(a: &WorkspaceConfig, b: &WorkspaceConfig) -> bool {
    config_matches_except_broadcasting(a, b)
}

fn run_spec(
    role: WorkspaceBroadcastRole,
    label: &'static str,
    metadata: &ConditionMetadata,
    config: WorkspaceConfig,
    broadcast_field_writes: u8,
) -> WorkspaceBroadcastRunSpec {
    WorkspaceBroadcastRunSpec {
        role,
        label,
        metadata: metadata.clone(),
        config,
        broadcast_field_writes,
    }
}

/// Build a matched intact/lesion/sham/rescue plan from one baseline.
///
/// Construction semantics:
/// - intact: exact baseline, no config write;
/// - lesion: one explicit write of `enable_broadcasting = false`;
/// - sham: one explicit write of `enable_broadcasting = true`;
/// - rescue: one explicit restoration write to `true`.
///
/// Because the baseline must already have broadcasting enabled, sham and rescue
/// are byte-equivalent in configuration to intact while still representing the
/// same config-field operation count as the lesion.
pub fn build_workspace_broadcast_lesion_plan(
    baseline: &WorkspaceConfig,
    metadata: ConditionMetadata,
) -> Result<WorkspaceBroadcastLesionPlan, WorkspaceBroadcastLesionError> {
    if !baseline.enable_broadcasting {
        return Err(WorkspaceBroadcastLesionError::BaselineBroadcastingDisabled);
    }

    let intact_config = baseline.clone();

    let mut lesion_config = baseline.clone();
    lesion_config.enable_broadcasting = false;

    let mut sham_config = baseline.clone();
    sham_config.enable_broadcasting = true;

    let mut rescue_config = baseline.clone();
    rescue_config.enable_broadcasting = true;

    debug_assert!(config_matches_except_broadcasting(
        &intact_config,
        &lesion_config
    ));
    debug_assert!(config_matches_except_broadcasting(
        &intact_config,
        &sham_config
    ));
    debug_assert!(config_matches_except_broadcasting(
        &intact_config,
        &rescue_config
    ));

    Ok(WorkspaceBroadcastLesionPlan {
        intact: run_spec(
            WorkspaceBroadcastRole::Intact,
            "workspace-broadcast-intact",
            &metadata,
            intact_config,
            0,
        ),
        lesion: run_spec(
            WorkspaceBroadcastRole::Lesion,
            "workspace-broadcast-lesion",
            &metadata,
            lesion_config,
            1,
        ),
        sham: run_spec(
            WorkspaceBroadcastRole::Sham,
            "workspace-broadcast-sham",
            &metadata,
            sham_config,
            1,
        ),
        rescue: run_spec(
            WorkspaceBroadcastRole::Rescue,
            "workspace-broadcast-rescue",
            &metadata,
            rescue_config,
            1,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::global_workspace::{GlobalWorkspace, WorkspaceContent};

    fn metadata() -> ConditionMetadata {
        ConditionMetadata::new("geom-003b1-workspace-broadcast-v1", 7, 1, 1)
    }

    fn same_config(a: &WorkspaceConfig, b: &WorkspaceConfig) -> bool {
        config_matches_except_broadcasting(a, b)
            && a.enable_broadcasting == b.enable_broadcasting
    }

    fn run_one_cycle(spec: &WorkspaceBroadcastRunSpec) -> (usize, usize) {
        let mut workspace = GlobalWorkspace::new(spec.config.clone());
        workspace.submit(WorkspaceContent::new(
            Vec::new(),
            0.95,
            "matched-probe".to_string(),
        ));
        let assessment = workspace.process();
        (
            assessment.conscious_contents.len(),
            assessment.broadcasts.len(),
        )
    }

    #[test]
    fn plan_changes_only_broadcast_enablement() {
        let baseline = WorkspaceConfig::default();
        let plan = build_workspace_broadcast_lesion_plan(&baseline, metadata())
            .expect("default workspace broadcasts");

        assert!(plan.intact.config.enable_broadcasting);
        assert!(!plan.lesion.config.enable_broadcasting);
        assert!(plan.sham.config.enable_broadcasting);
        assert!(plan.rescue.config.enable_broadcasting);

        assert!(differs_only_in_broadcasting(
            &plan.intact.config,
            &plan.lesion.config
        ));
        assert!(same_config(&plan.intact.config, &plan.sham.config));
        assert!(same_config(&plan.intact.config, &plan.rescue.config));
    }

    #[test]
    fn lesion_and_sham_have_matched_config_write_count() {
        let plan = build_workspace_broadcast_lesion_plan(&WorkspaceConfig::default(), metadata())
            .expect("default workspace broadcasts");

        assert_eq!(plan.intact.broadcast_field_writes, 0);
        assert_eq!(plan.lesion.broadcast_field_writes, 1);
        assert_eq!(plan.sham.broadcast_field_writes, 1);
        assert_eq!(plan.rescue.broadcast_field_writes, 1);
        assert_eq!(plan.lesion.metadata, plan.sham.metadata);
        assert_eq!(plan.lesion.metadata, plan.rescue.metadata);
    }

    #[test]
    fn one_cycle_probe_isolates_broadcasting_not_workspace_entry() {
        let plan = build_workspace_broadcast_lesion_plan(&WorkspaceConfig::default(), metadata())
            .expect("default workspace broadcasts");

        let intact = run_one_cycle(&plan.intact);
        let lesion = run_one_cycle(&plan.lesion);
        let sham = run_one_cycle(&plan.sham);
        let rescue = run_one_cycle(&plan.rescue);

        assert_eq!(intact.0, 1);
        assert_eq!(lesion.0, intact.0);
        assert_eq!(sham.0, intact.0);
        assert_eq!(rescue.0, intact.0);

        assert!(intact.1 > 0);
        assert_eq!(lesion.1, 0);
        assert_eq!(sham.1, intact.1);
        assert_eq!(rescue.1, intact.1);
    }

    #[test]
    fn already_disabled_baseline_is_not_a_valid_lesion() {
        let mut baseline = WorkspaceConfig::default();
        baseline.enable_broadcasting = false;

        assert!(matches!(
            build_workspace_broadcast_lesion_plan(&baseline, metadata()),
            Err(WorkspaceBroadcastLesionError::BaselineBroadcastingDisabled)
        ));
    }
}
