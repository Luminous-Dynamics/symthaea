use leptos::prelude::*;

use crate::state::AppState;

/// Neuromodulator bar chart: DA, NE, 5-HT, OT
#[component]
pub fn NeuroBars() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let neuromods = state.neuromods;

    let labels = ["DA", "NE", "5-HT", "OT"];
    let colors = [
        "var(--da-blue)",
        "var(--ne-red)",
        "var(--sht-green)",
        "var(--ot-pink)",
    ];

    view! {
        <div>
            {labels
                .iter()
                .zip(colors.iter())
                .enumerate()
                .map(|(i, (label, color))| {
                    let label = *label;
                    let color = *color;
                    view! {
                        <div class="neuro-bar-row">
                            <span class="nb-label">{label}</span>
                            <div class="nb-track">
                                <div
                                    class="nb-fill"
                                    style:width=move || {
                                        format!("{}%", (neuromods.get()[i] * 100.0) as u32)
                                    }
                                    style:background=color
                                />
                            </div>
                            <span class="nb-val">
                                {move || format!("{:.2}", neuromods.get()[i])}
                            </span>
                        </div>
                    }
                })
                .collect::<Vec<_>>()}
        </div>
    }
}
