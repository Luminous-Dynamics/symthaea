// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Interactive SVG Segrè chart — nuclear stability landscape.
//!
//! Renders a Z (rows) × N (columns) grid colored by binding energy per nucleon.
//! Click any cell for isotope details. Stability islands are highlighted.

use leptos::prelude::*;

/// Data for one isotope cell in the chart.
#[derive(Debug, Clone)]
struct NuclideCell {
    z: u16,
    n: u16,
    a: u16,
    ba: f64,           // binding energy per nucleon
    shell_corr: f64,   // shell correction
    is_island: bool,    // in predicted stability island
    symbol: String,
}

/// SEMF binding energy per nucleon (simplified, runs in browser).
fn semf_ba(a: u16, z: u16) -> f64 {
    if a < 2 || z < 1 || z >= a {
        return 0.0;
    }
    let a_f = a as f64;
    let z_f = z as f64;
    let vol = 15.56 * a_f;
    let surf = 17.23 * a_f.powf(2.0 / 3.0);
    let coul = 0.7 * z_f * (z_f - 1.0) / a_f.powf(1.0 / 3.0);
    let asym = 23.285 * (a_f - 2.0 * z_f).powi(2) / (4.0 * a_f);
    let n = a - z;
    let pair = if a % 2 != 0 {
        0.0
    } else if z % 2 == 0 && n % 2 == 0 {
        12.0 / a_f.sqrt()
    } else {
        -12.0 / a_f.sqrt()
    };
    ((vol - surf - coul - asym + pair) / a_f).max(0.0)
}

fn element_symbol(z: u16) -> &'static str {
    match z {
        100 => "Fm", 101 => "Md", 102 => "No", 103 => "Lr",
        104 => "Rf", 105 => "Db", 106 => "Sg", 107 => "Bh",
        108 => "Hs", 109 => "Mt", 110 => "Ds", 111 => "Rg",
        112 => "Cn", 113 => "Nh", 114 => "Fl", 115 => "Mc",
        116 => "Lv", 117 => "Ts", 118 => "Og", 119 => "119",
        120 => "120", _ => "?",
    }
}

/// Map B/A value to HSL color (blue=low → green=mid → red=high).
fn ba_to_color(ba: f64, min_ba: f64, max_ba: f64) -> String {
    if max_ba <= min_ba {
        return "hsl(240,50%,30%)".to_string();
    }
    let t = ((ba - min_ba) / (max_ba - min_ba)).clamp(0.0, 1.0);
    // Hue: 240 (blue) → 120 (green) → 0 (red)
    let hue = 240.0 * (1.0 - t);
    let lightness = 25.0 + 25.0 * t;
    format!("hsl({:.0},{:.0}%,{:.0}%)", hue, 60.0, lightness)
}

/// Interactive Segrè chart component.
#[component]
pub fn NuclearChart(
    /// Minimum Z to display
    #[prop(default = 100)]
    z_min: u16,
    /// Maximum Z to display
    #[prop(default = 120)]
    z_max: u16,
    /// Minimum N to display
    #[prop(default = 140)]
    n_min: u16,
    /// Maximum N to display
    #[prop(default = 190)]
    n_max: u16,
) -> impl IntoView {
    let (selected, set_selected) = signal(None::<NuclideCell>);

    // Pre-compute all cells
    let cells: Vec<NuclideCell> = (z_min..=z_max)
        .flat_map(|z| {
            (n_min..=n_max).map(move |n| {
                let a = z + n;
                let ba = semf_ba(a, z);
                NuclideCell {
                    z,
                    n,
                    a,
                    ba,
                    shell_corr: 0.0, // Simplified — full shell model too heavy for browser
                    is_island: z >= 114 && z <= 120 && n >= 178 && n <= 186,
                    symbol: element_symbol(z).to_string(),
                }
            })
        })
        .collect();

    let min_ba = cells.iter().map(|c| c.ba).fold(f64::INFINITY, f64::min);
    let max_ba = cells.iter().map(|c| c.ba).fold(f64::NEG_INFINITY, f64::max);

    let cell_size = 12;
    let n_range = (n_max - n_min + 1) as usize;
    let z_range = (z_max - z_min + 1) as usize;
    let svg_width = n_range * cell_size + 60;
    let svg_height = z_range * cell_size + 40;

    view! {
        <div class="glass-panel" style="overflow-x: auto;">
            <h3 style="font-size: 1rem; margin-bottom: 0.75rem;">
                "Nuclear Stability Landscape (Z=" {z_min} "-" {z_max} ", N=" {n_min} "-" {n_max} ")"
            </h3>

            <svg
                width=svg_width
                height=svg_height
                viewBox=format!("0 0 {} {}", svg_width, svg_height)
                style="font-family: monospace; font-size: 8px;"
            >
                // Y-axis labels (Z)
                {(z_min..=z_max).step_by(5).map(|z| {
                    let y = (z_max - z) as usize * cell_size + cell_size / 2 + 20;
                    view! {
                        <text x="0" y=y fill="var(--text-secondary)" font-size="8" dominant-baseline="middle">
                            {format!("{}", z)}
                        </text>
                    }
                }).collect::<Vec<_>>()}

                // X-axis labels (N)
                {(n_min..=n_max).step_by(10).map(|n| {
                    let x = (n - n_min) as usize * cell_size + cell_size / 2 + 40;
                    view! {
                        <text x=x y=format!("{}", svg_height - 5) fill="var(--text-secondary)" font-size="8" text-anchor="middle">
                            {format!("{}", n)}
                        </text>
                    }
                }).collect::<Vec<_>>()}

                // Cells
                {cells.iter().map(|cell| {
                    let x = (cell.n - n_min) as usize * cell_size + 40;
                    let y = (z_max - cell.z) as usize * cell_size + 20;
                    let color = ba_to_color(cell.ba, min_ba, max_ba);
                    let border = if cell.is_island { "stroke: var(--accent-emerald); stroke-width: 1.5;" } else { "" };
                    let cell_clone = cell.clone();

                    view! {
                        <rect
                            x=x
                            y=y
                            width=cell_size - 1
                            height=cell_size - 1
                            fill=color
                            style=border
                            on:click=move |_| set_selected.set(Some(cell_clone.clone()))
                        >
                            <title>{format!("{}-{} (Z={}, N={})\nB/A = {:.3} MeV", cell.symbol, cell.a, cell.z, cell.n, cell.ba)}</title>
                        </rect>
                    }
                }).collect::<Vec<_>>()}

                // Axis labels
                <text x=format!("{}", svg_width / 2) y="12" fill="var(--text-secondary)" font-size="10" text-anchor="middle">"Neutron Number (N) →"</text>
                <text x="8" y=format!("{}", svg_height / 2) fill="var(--text-secondary)" font-size="10" text-anchor="middle" transform=format!("rotate(-90, 8, {})", svg_height / 2)>"Z ↑"</text>
            </svg>

            // Detail panel
            {move || selected.get().map(|cell| view! {
                <div style="margin-top: 1rem; padding: 0.75rem; background: var(--bg-secondary); border-radius: 0.5rem;">
                    <h4 style="color: var(--accent-indigo);">
                        {format!("{}-{}", cell.symbol, cell.a)}
                        <span style="font-weight: normal; color: var(--text-secondary); margin-left: 0.5rem;">
                            {format!("(Z={}, N={})", cell.z, cell.n)}
                        </span>
                    </h4>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-top: 0.5rem; font-size: 0.875rem;">
                        <div>
                            <span style="color: var(--text-secondary);">"B/A: "</span>
                            <span style="color: var(--accent-emerald); font-weight: 600;">{format!("{:.3} MeV", cell.ba)}</span>
                        </div>
                        <div>
                            <span style="color: var(--text-secondary);">"Mass: "</span>
                            <span>{format!("A = {}", cell.a)}</span>
                        </div>
                        <div>
                            <span style="color: var(--text-secondary);">"Island: "</span>
                            <span style=if cell.is_island { "color: var(--accent-emerald);" } else { "" }>
                                {if cell.is_island { "YES" } else { "no" }}
                            </span>
                        </div>
                    </div>
                </div>
            })}
        </div>
    }
}
