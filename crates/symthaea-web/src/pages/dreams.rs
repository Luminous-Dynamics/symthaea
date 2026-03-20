use leptos::prelude::*;

use crate::components::glass_panel::GlassPanel;
use crate::state::AppState;

/// Tab 4: Counterfactual dreaming and FEP exploration.
#[component]
pub fn DreamsPage() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let pe = state.prediction_error;

    view! {
        <GlassPanel title="Dream Engine">
            <p style="font-size: 0.82rem; color: var(--fg-dim); line-height: 1.6; margin-bottom: 1rem;">
                "Counterfactual learning through dreaming: the system replays experiences, "
                "perturbs them, and consolidates wisdom. FEP drives explore/exploit balance."
            </p>

            <div class="dream-stats-grid">
                <div class="dream-stat">
                    <div class="ds-val">"--"</div>
                    <div class="ds-label">"dream cycles"</div>
                </div>
                <div class="dream-stat">
                    <div class="ds-val">"--"</div>
                    <div class="ds-label">"wisdom entries"</div>
                </div>
                <div class="dream-stat">
                    <div class="ds-val">
                        {move || format!("{:.3}", pe.get())}
                    </div>
                    <div class="ds-label">"prediction error"</div>
                </div>
                <div class="dream-stat">
                    <div class="ds-val">"--"</div>
                    <div class="ds-label">"consolidations"</div>
                </div>
            </div>
        </GlassPanel>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
            <GlassPanel title="FEP State">
                <div class="fep-display" style="grid-template-columns: 1fr;">
                    <div class="fep-big">
                        <div class="fep-val">
                            {move || format!("{:.3}", pe.get())}
                        </div>
                        <div class="fep-label">"free energy"</div>
                        <div class="fep-mode exploring">"exploring"</div>
                    </div>
                </div>
            </GlassPanel>

            <GlassPanel title="Wisdom Journal">
                <div class="wisdom-journal">
                    <div class="wisdom-empty">
                        "No wisdom entries yet. Start a dream session to begin counterfactual learning."
                    </div>
                </div>
            </GlassPanel>
        </div>
    }
}
