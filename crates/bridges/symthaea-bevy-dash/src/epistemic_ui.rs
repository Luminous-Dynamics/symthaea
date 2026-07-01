use crate::CognitiveStateResource;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use symthaea_epistemic_types::formal_watchdog::FormalWatchdog;
use symthaea_epistemic_types::global_ledger::{GlobalClaimStatus, GlobalEpistemicLedger};

#[derive(Resource, Default)]
pub struct EpistemicAuditResource {
    pub is_consistent: bool,
    pub last_violations: Vec<String>,
}

/// System to poll the Epistemic Watchdog and update the UI dashboard.
pub fn epistemic_dashboard_system(
    mut contexts: EguiContexts,
    mut audit_res: ResMut<EpistemicAuditResource>,
) {
    egui::Window::new("Formal Epistemic Audit").show(contexts.ctx_mut(), |ui| {
        if audit_res.is_consistent {
            ui.label(
                egui::RichText::new("✅ Formal Integrity Verified").color(egui::Color32::GREEN),
            );
        } else {
            ui.label(
                egui::RichText::new("❌ Formal Inconsistency Detected").color(egui::Color32::RED),
            );
            for violation in &audit_res.last_violations {
                ui.label(format!("- {}", violation));
            }
        }
    });
}
