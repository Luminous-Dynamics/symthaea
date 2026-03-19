use leptos::prelude::*;

/// Eight-point SVG radar chart for the Eight Harmonies.
///
/// For now renders a static placeholder; will be wired to live
/// harmony scores from the SporeEngine worker.
#[component]
pub fn HarmonyRadar() -> impl IntoView {
    // The eight harmonies: MC, RC, DC, WE, AR, EA, AM, SS
    let labels = [
        "MC", "RC", "DC", "WE", "AR", "EA", "AM", "SS",
    ];
    // Default values (will become reactive once worker is wired)
    let values = [0.5_f32; 8];

    let cx = 90.0_f32;
    let cy = 90.0_f32;
    let r = 70.0_f32;

    // Build polygon points
    let points: String = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let angle =
                (i as f32 / 8.0) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
            let x = cx + angle.cos() * r * v;
            let y = cy + angle.sin() * r * v;
            format!("{x:.1},{y:.1}")
        })
        .collect::<Vec<_>>()
        .join(" ");

    // Label positions (slightly outside the chart)
    let label_elems: Vec<_> = labels
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let angle =
                (i as f32 / 8.0) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
            let x = cx + angle.cos() * (r + 14.0);
            let y = cy + angle.sin() * (r + 14.0);
            (x, y, *label)
        })
        .collect();

    view! {
        <div class="harmony-radar">
            <svg viewBox="0 0 180 180" xmlns="http://www.w3.org/2000/svg">
                // Grid circles
                <circle cx={cx.to_string()} cy={cy.to_string()} r={(r * 0.33).to_string()} fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="0.5" />
                <circle cx={cx.to_string()} cy={cy.to_string()} r={(r * 0.66).to_string()} fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="0.5" />
                <circle cx={cx.to_string()} cy={cy.to_string()} r={r.to_string()} fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="0.5" />
                // Data polygon
                <polygon points={points} fill="rgba(126,200,160,0.15)" stroke="var(--leaf-green)" stroke-width="1.5" />
                // Labels
                {label_elems
                    .into_iter()
                    .map(|(x, y, label)| {
                        view! {
                            <text
                                x={x.to_string()}
                                y={y.to_string()}
                                text-anchor="middle"
                                dominant-baseline="central"
                                fill="var(--fg-muted)"
                                font-size="7"
                            >
                                {label}
                            </text>
                        }
                    })
                    .collect::<Vec<_>>()}
            </svg>
        </div>
    }
}
