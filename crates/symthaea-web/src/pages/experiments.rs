use leptos::prelude::*;

use crate::components::glass_panel::GlassPanel;

/// Tab 3: Consciousness validation experiments (psych-bench).
#[component]
pub fn ExperimentsPage() -> impl IntoView {
    view! {
        <GlassPanel title="Consciousness Validation Experiments">
            <p style="font-size: 0.82rem; color: var(--fg-dim); line-height: 1.6; margin-bottom: 1.5rem;">
                "Run psych-bench experiments to probe consciousness properties: "
                "IIT Phi measurement, Global Workspace access, Temporal Binding, "
                "Substrate Independence, and Metacognitive accuracy."
            </p>
        </GlassPanel>

        <div class="experiment-grid">
            <ExperimentCard
                title="IIT Phi Measurement"
                description="Compute integrated information across the 12-region causal graph. Measures how much the system is more than the sum of its parts."
            />
            <ExperimentCard
                title="Global Workspace Test"
                description="Verify that attended stimuli broadcast to all regions while unattended stimuli remain local. Tests the GWT workspace hypothesis."
            />
            <ExperimentCard
                title="Temporal Binding"
                description="Measure temporal integration window and binding coherence across sensory modalities. Tests unified experience over time."
            />
            <ExperimentCard
                title="Substrate Transfer"
                description="Switch substrate mid-run (Biological to Silicon to Quantum) and verify consciousness continuity. Tests Multiple Realizability."
            />
        </div>

        <div class="run-all-bar" style="margin-top: 1.5rem;">
            <button class="btn-action" disabled=true>
                "Run All Experiments"
            </button>
            <span style="font-size: 0.75rem; color: var(--fg-muted); font-style: italic;">
                "Requires SporeEngine worker connection"
            </span>
        </div>
    }
}

#[component]
fn ExperimentCard(title: &'static str, description: &'static str) -> impl IntoView {
    view! {
        <div class="experiment-card">
            <h3>{title}</h3>
            <p class="exp-desc">{description}</p>
            <button class="btn-action" disabled=true>
                "Run"
            </button>
            <div class="exp-result">"Awaiting execution..."</div>
        </div>
    }
}
