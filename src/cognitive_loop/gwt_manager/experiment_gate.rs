// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit experiment-only gate for the cognitive loop's built-in GWT handlers.
//!
//! The normal constructor installs `memory` and `perception` handlers directly.
//! This module changes nothing unless an experiment explicitly calls
//! [`CognitiveLoopService::install_gwt_builtin_delivery_gate`]. That call
//! replaces only those two named handlers with wrappers that preserve their
//! original enabled side effects while adding an externally owned atomic gate
//! and monotonic provenance counters.

use crate::cognitive_loop::CognitiveLoopService;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

impl CognitiveLoopService {
    /// Install an experiment-only delivery gate around the built-in GWT
    /// `memory` and `perception` handlers.
    ///
    /// Returns `false` without mutation when GWT is disabled. When it returns
    /// `true`, both existing named handlers have been replaced with wrappers
    /// whose enabled behavior is exactly the constructor's original side
    /// effect:
    ///
    /// - `memory` sets the shared memory-consolidation flag;
    /// - `perception` increments the shared perception-broadcast counter.
    ///
    /// Every wrapper invocation increments `handler_invocations` BEFORE reading
    /// `delivery_enabled`. If delivery is disabled, the wrapper increments
    /// `blocked_deliveries` and returns without performing the original side
    /// effect. Broadcast construction, recipient iteration, handler lookup, and
    /// wrapper invocation remain owned by the existing GWT implementation.
    ///
    /// The caller owns all three atomic handles, so no experiment state is added
    /// to `CognitiveLoopService` or `GwtManager`.
    pub fn install_gwt_builtin_delivery_gate(
        &mut self,
        delivery_enabled: Arc<AtomicBool>,
        handler_invocations: Arc<AtomicUsize>,
        blocked_deliveries: Arc<AtomicUsize>,
    ) -> bool {
        let memory_flag = Arc::clone(&self.consciousness.gwt_mgr.memory_flag);
        let perception_count = Arc::clone(&self.consciousness.gwt_mgr.perception_count);

        let Some(workspace) = self.consciousness.gwt_mgr.gwt.as_mut() else {
            return false;
        };

        let memory_enabled = Arc::clone(&delivery_enabled);
        let memory_invocations = Arc::clone(&handler_invocations);
        let memory_blocked = Arc::clone(&blocked_deliveries);
        workspace.register_handler(
            "memory",
            Box::new(move |_| {
                memory_invocations.fetch_add(1, Ordering::SeqCst);
                if memory_enabled.load(Ordering::SeqCst) {
                    memory_flag.store(true, Ordering::Relaxed);
                } else {
                    memory_blocked.fetch_add(1, Ordering::SeqCst);
                }
            }),
        );

        let perception_enabled = Arc::clone(&delivery_enabled);
        let perception_invocations = Arc::clone(&handler_invocations);
        let perception_blocked = Arc::clone(&blocked_deliveries);
        workspace.register_handler(
            "perception",
            Box::new(move |_| {
                perception_invocations.fetch_add(1, Ordering::SeqCst);
                if perception_enabled.load(Ordering::SeqCst) {
                    perception_count.fetch_add(1, Ordering::Relaxed);
                } else {
                    perception_blocked.fetch_add(1, Ordering::SeqCst);
                }
            }),
        );

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::CognitiveLoopConfig;

    fn service(enable_gwt: bool) -> CognitiveLoopService {
        let mut config = CognitiveLoopConfig::default();
        config.enable_gwt = enable_gwt;
        CognitiveLoopService::new(config).expect("valid cognitive loop")
    }

    fn install_gate(
        service: &mut CognitiveLoopService,
        enabled: bool,
    ) -> (Arc<AtomicBool>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let delivery_enabled = Arc::new(AtomicBool::new(enabled));
        let invocations = Arc::new(AtomicUsize::new(0));
        let blocked = Arc::new(AtomicUsize::new(0));
        assert!(service.install_gwt_builtin_delivery_gate(
            Arc::clone(&delivery_enabled),
            Arc::clone(&invocations),
            Arc::clone(&blocked),
        ));
        (delivery_enabled, invocations, blocked)
    }

    fn process_probe(service: &mut CognitiveLoopService) -> usize {
        let workspace = service
            .consciousness
            .gwt_mgr
            .gwt
            .as_mut()
            .expect("GWT enabled");
        workspace.submit_strategy(
            "GEOM delivery probe",
            1.0,
            Vec::new(),
            vec!["perception".to_string(), "memory".to_string()],
        );
        workspace.process().workspace_assessment.broadcasts.len()
    }

    #[test]
    fn enabled_gate_preserves_builtin_handler_side_effects() {
        let mut service = service(true);
        let (_enabled, invocations, blocked) = install_gate(&mut service, true);

        assert!(!service
            .consciousness
            .gwt_mgr
            .memory_flag
            .load(Ordering::Relaxed));
        assert_eq!(
            service
                .consciousness
                .gwt_mgr
                .perception_count
                .load(Ordering::Relaxed),
            0
        );

        assert!(process_probe(&mut service) > 0);
        assert!(invocations.load(Ordering::SeqCst) >= 2);
        assert_eq!(blocked.load(Ordering::SeqCst), 0);
        assert!(service
            .consciousness
            .gwt_mgr
            .memory_flag
            .load(Ordering::Relaxed));
        assert!(
            service
                .consciousness
                .gwt_mgr
                .perception_count
                .load(Ordering::Relaxed)
                > 0
        );
    }

    #[test]
    fn disabled_gate_records_invocation_but_blocks_builtin_side_effects() {
        let mut service = service(true);
        let (_enabled, invocations, blocked) = install_gate(&mut service, false);

        assert!(process_probe(&mut service) > 0);
        let invocation_count = invocations.load(Ordering::SeqCst);
        assert!(invocation_count >= 2);
        assert_eq!(blocked.load(Ordering::SeqCst), invocation_count);
        assert!(!service
            .consciousness
            .gwt_mgr
            .memory_flag
            .load(Ordering::Relaxed));
        assert_eq!(
            service
                .consciousness
                .gwt_mgr
                .perception_count
                .load(Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn re_enabling_same_gate_restores_delivery() {
        let mut service = service(true);
        let (enabled, invocations, blocked) = install_gate(&mut service, false);

        assert!(process_probe(&mut service) > 0);
        let invocations_after_block = invocations.load(Ordering::SeqCst);
        let blocked_after_block = blocked.load(Ordering::SeqCst);
        assert_eq!(blocked_after_block, invocations_after_block);

        enabled.store(true, Ordering::SeqCst);
        assert!(process_probe(&mut service) > 0);

        assert!(invocations.load(Ordering::SeqCst) > invocations_after_block);
        assert_eq!(blocked.load(Ordering::SeqCst), blocked_after_block);
        assert!(service
            .consciousness
            .gwt_mgr
            .memory_flag
            .load(Ordering::Relaxed));
        assert!(
            service
                .consciousness
                .gwt_mgr
                .perception_count
                .load(Ordering::Relaxed)
                > 0
        );
    }

    #[test]
    fn gate_installation_fails_closed_when_gwt_is_absent() {
        let mut service = service(false);
        let delivery_enabled = Arc::new(AtomicBool::new(false));
        let invocations = Arc::new(AtomicUsize::new(0));
        let blocked = Arc::new(AtomicUsize::new(0));

        assert!(!service.install_gwt_builtin_delivery_gate(
            Arc::clone(&delivery_enabled),
            Arc::clone(&invocations),
            Arc::clone(&blocked),
        ));
        assert_eq!(invocations.load(Ordering::SeqCst), 0);
        assert_eq!(blocked.load(Ordering::SeqCst), 0);
    }
}
