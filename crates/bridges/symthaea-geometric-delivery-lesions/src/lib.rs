// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Broadcast-path-matched recipient-delivery lesion for GEOM-003B2.
//!
//! Unlike GEOM-003B1, this intervention leaves workspace entry and broadcast
//! construction enabled in every condition. Every condition also registers and
//! invokes the same gate wrapper for the selected recipient. The lesion differs
//! only in whether that wrapper forwards the broadcast to the downstream
//! consumer.
//!
//! This removes broadcast construction, recipient lookup, and wrapper dispatch
//! as confounds. It does not claim total compute matching: a blocked downstream
//! consumer necessarily performs less work because it does not receive content.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::global_workspace::{GlobalWorkspace, WorkspaceConfig};
use symthaea_geometric_interventions::ConditionMetadata;

/// Fixed experimental role for one delivery-gate condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceDeliveryRole {
    Intact,
    Lesion,
    Sham,
    Rescue,
}

/// Whether the registered gate wrapper forwards to its downstream consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryGateMode {
    Forward,
    Block,
}

/// One run specification for a recipient-delivery intervention.
#[derive(Debug, Clone)]
pub struct WorkspaceDeliveryRunSpec {
    pub role: WorkspaceDeliveryRole,
    pub label: &'static str,
    pub metadata: ConditionMetadata,
    pub config: WorkspaceConfig,
    pub gate_mode: DeliveryGateMode,
}

/// Complete intact/lesion/sham/rescue plan.
#[derive(Debug, Clone)]
pub struct WorkspaceDeliveryLesionPlan {
    pub intact: WorkspaceDeliveryRunSpec,
    pub lesion: WorkspaceDeliveryRunSpec,
    pub sham: WorkspaceDeliveryRunSpec,
    pub rescue: WorkspaceDeliveryRunSpec,
}

/// Plan-construction failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkspaceDeliveryLesionError {
    BaselineBroadcastingDisabled,
}

impl std::fmt::Display for WorkspaceDeliveryLesionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BaselineBroadcastingDisabled => write!(
                f,
                "delivery-gate lesion requires broadcasting enabled in the baseline"
            ),
        }
    }
}

impl std::error::Error for WorkspaceDeliveryLesionError {}

/// Shared downstream consumer accepted by the delivery gate.
pub type SharedWorkspaceHandler = Arc<dyn Fn(&[BinaryHV]) + Send + Sync + 'static>;

/// Independent counters proving that the wrapper itself ran even when delivery
/// was blocked.
#[derive(Debug, Clone)]
pub struct DeliveryGateTelemetry {
    gate_invocations: Arc<AtomicUsize>,
    forwarded_deliveries: Arc<AtomicUsize>,
}

impl DeliveryGateTelemetry {
    pub fn gate_invocations(&self) -> usize {
        self.gate_invocations.load(Ordering::SeqCst)
    }

    pub fn forwarded_deliveries(&self) -> usize {
        self.forwarded_deliveries.load(Ordering::SeqCst)
    }
}

fn run_spec(
    role: WorkspaceDeliveryRole,
    label: &'static str,
    metadata: &ConditionMetadata,
    baseline: &WorkspaceConfig,
    gate_mode: DeliveryGateMode,
) -> WorkspaceDeliveryRunSpec {
    WorkspaceDeliveryRunSpec {
        role,
        label,
        metadata: metadata.clone(),
        config: baseline.clone(),
        gate_mode,
    }
}

/// Build a quartet in which the workspace configuration is byte-equivalent
/// across all four conditions and only delivery-gate mode changes.
pub fn build_workspace_delivery_lesion_plan(
    baseline: &WorkspaceConfig,
    metadata: ConditionMetadata,
) -> Result<WorkspaceDeliveryLesionPlan, WorkspaceDeliveryLesionError> {
    if !baseline.enable_broadcasting {
        return Err(WorkspaceDeliveryLesionError::BaselineBroadcastingDisabled);
    }

    Ok(WorkspaceDeliveryLesionPlan {
        intact: run_spec(
            WorkspaceDeliveryRole::Intact,
            "workspace-delivery-intact",
            &metadata,
            baseline,
            DeliveryGateMode::Forward,
        ),
        lesion: run_spec(
            WorkspaceDeliveryRole::Lesion,
            "workspace-delivery-lesion",
            &metadata,
            baseline,
            DeliveryGateMode::Block,
        ),
        sham: run_spec(
            WorkspaceDeliveryRole::Sham,
            "workspace-delivery-sham",
            &metadata,
            baseline,
            DeliveryGateMode::Forward,
        ),
        rescue: run_spec(
            WorkspaceDeliveryRole::Rescue,
            "workspace-delivery-rescue",
            &metadata,
            baseline,
            DeliveryGateMode::Forward,
        ),
    })
}

/// Register the same gate-wrapper structure in every condition.
///
/// The wrapper always increments `gate_invocations`. In `Forward` mode it also
/// increments `forwarded_deliveries` and invokes the downstream consumer. In
/// `Block` mode it returns without forwarding.
pub fn register_delivery_gate(
    workspace: &mut GlobalWorkspace,
    module: &str,
    mode: DeliveryGateMode,
    downstream: SharedWorkspaceHandler,
) -> DeliveryGateTelemetry {
    let gate_invocations = Arc::new(AtomicUsize::new(0));
    let forwarded_deliveries = Arc::new(AtomicUsize::new(0));

    let gate_counter = Arc::clone(&gate_invocations);
    let forwarded_counter = Arc::clone(&forwarded_deliveries);

    workspace.register_handler(
        module,
        Box::new(move |content| {
            gate_counter.fetch_add(1, Ordering::SeqCst);
            if mode == DeliveryGateMode::Forward {
                forwarded_counter.fetch_add(1, Ordering::SeqCst);
                downstream(content);
            }
        }),
    );

    DeliveryGateTelemetry {
        gate_invocations,
        forwarded_deliveries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::global_workspace::WorkspaceContent;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ProbeOutcome {
        workspace_entries: usize,
        broadcasts: usize,
        gate_invocations: usize,
        forwarded_deliveries: usize,
        downstream_deliveries: usize,
    }

    fn metadata() -> ConditionMetadata {
        ConditionMetadata::new("geom-003b2-workspace-delivery-v1", 11, 1, 1)
    }

    fn same_config(a: &WorkspaceConfig, b: &WorkspaceConfig) -> bool {
        a.max_capacity == b.max_capacity
            && a.entry_threshold.to_bits() == b.entry_threshold.to_bits()
            && a.decay_rate.to_bits() == b.decay_rate.to_bits()
            && a.enable_broadcasting == b.enable_broadcasting
            && a.winner_takes_all == b.winner_takes_all
            && a.max_duration == b.max_duration
    }

    fn run_one_cycle(spec: &WorkspaceDeliveryRunSpec) -> ProbeOutcome {
        let mut workspace = GlobalWorkspace::new(spec.config.clone());
        let downstream_deliveries = Arc::new(AtomicUsize::new(0));
        let downstream_counter = Arc::clone(&downstream_deliveries);
        let downstream: SharedWorkspaceHandler = Arc::new(move |_content| {
            downstream_counter.fetch_add(1, Ordering::SeqCst);
        });

        // `memory` is one of GlobalWorkspace's default broadcast recipients.
        let telemetry =
            register_delivery_gate(&mut workspace, "memory", spec.gate_mode, downstream);

        workspace.submit(WorkspaceContent::new(
            Vec::new(),
            0.95,
            "matched-delivery-probe".to_string(),
        ));
        let assessment = workspace.process();

        ProbeOutcome {
            workspace_entries: assessment.conscious_contents.len(),
            broadcasts: assessment.broadcasts.len(),
            gate_invocations: telemetry.gate_invocations(),
            forwarded_deliveries: telemetry.forwarded_deliveries(),
            downstream_deliveries: downstream_deliveries.load(Ordering::SeqCst),
        }
    }

    #[test]
    fn quartet_keeps_workspace_configuration_identical() {
        let baseline = WorkspaceConfig::default();
        let plan = build_workspace_delivery_lesion_plan(&baseline, metadata())
            .expect("default workspace broadcasts");

        assert!(same_config(&plan.intact.config, &plan.lesion.config));
        assert!(same_config(&plan.intact.config, &plan.sham.config));
        assert!(same_config(&plan.intact.config, &plan.rescue.config));
        assert_eq!(plan.intact.metadata, plan.lesion.metadata);
        assert_eq!(plan.intact.metadata, plan.sham.metadata);
        assert_eq!(plan.intact.metadata, plan.rescue.metadata);

        assert_eq!(plan.intact.gate_mode, DeliveryGateMode::Forward);
        assert_eq!(plan.lesion.gate_mode, DeliveryGateMode::Block);
        assert_eq!(plan.sham.gate_mode, DeliveryGateMode::Forward);
        assert_eq!(plan.rescue.gate_mode, DeliveryGateMode::Forward);
    }

    #[test]
    fn delivery_lesion_preserves_broadcast_path_until_forwarding_boundary() {
        let plan = build_workspace_delivery_lesion_plan(&WorkspaceConfig::default(), metadata())
            .expect("default workspace broadcasts");

        let intact = run_one_cycle(&plan.intact);
        let lesion = run_one_cycle(&plan.lesion);
        let sham = run_one_cycle(&plan.sham);
        let rescue = run_one_cycle(&plan.rescue);

        assert_eq!(intact.workspace_entries, 1);
        assert_eq!(lesion.workspace_entries, intact.workspace_entries);
        assert_eq!(sham.workspace_entries, intact.workspace_entries);
        assert_eq!(rescue.workspace_entries, intact.workspace_entries);

        assert!(intact.broadcasts > 0);
        assert_eq!(lesion.broadcasts, intact.broadcasts);
        assert_eq!(sham.broadcasts, intact.broadcasts);
        assert_eq!(rescue.broadcasts, intact.broadcasts);

        assert!(intact.gate_invocations > 0);
        assert_eq!(lesion.gate_invocations, intact.gate_invocations);
        assert_eq!(sham.gate_invocations, intact.gate_invocations);
        assert_eq!(rescue.gate_invocations, intact.gate_invocations);

        assert!(intact.forwarded_deliveries > 0);
        assert_eq!(lesion.forwarded_deliveries, 0);
        assert_eq!(sham.forwarded_deliveries, intact.forwarded_deliveries);
        assert_eq!(rescue.forwarded_deliveries, intact.forwarded_deliveries);

        assert_eq!(intact.downstream_deliveries, intact.forwarded_deliveries);
        assert_eq!(lesion.downstream_deliveries, 0);
        assert_eq!(sham.downstream_deliveries, intact.downstream_deliveries);
        assert_eq!(rescue.downstream_deliveries, intact.downstream_deliveries);
    }

    #[test]
    fn disabled_broadcast_baseline_is_rejected() {
        let mut baseline = WorkspaceConfig::default();
        baseline.enable_broadcasting = false;

        assert!(matches!(
            build_workspace_delivery_lesion_plan(&baseline, metadata()),
            Err(WorkspaceDeliveryLesionError::BaselineBroadcastingDisabled)
        ));
    }
}
