use leptos::prelude::*;

use crate::state::AppState;

/// Displays the current integrated information (Phi) value
/// with color-coded feedback and honest confidence badge.
#[component]
pub fn PhiMeter() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let phi = state.consciousness_level;
    let confidence = state.honest_confidence;

    let color = move || {
        let p = phi.get();
        if p > 0.4 {
            "var(--solar-gold)"
        } else if p > 0.15 {
            "var(--leaf-green)"
        } else {
            "var(--lichen-grey)"
        }
    };

    view! {
        <div class="phi-meter">
            <div class="phi-big" style:color=color>
                {move || format!("{:.3}", phi.get())}
            </div>
            <div class="phi-label">"integrated information (Phi)"</div>
            <div class="confidence-badge">
                {move || format!("honest confidence: {:.2}", confidence.get())}
            </div>
        </div>
    }
}
