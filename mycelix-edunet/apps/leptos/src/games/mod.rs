// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Educational games — interactive STEM simulations embedded in study pages.

pub mod shared;
pub mod foundation;
pub mod math;
pub mod physics;
pub mod chemistry;
pub mod universal;

use leptos::prelude::*;

/// Registry: which nodes have associated games.
pub fn has_game(node_id: &str) -> bool {
    game_type(node_id).is_some()
}

/// Get the game type for a node ID.
fn game_type(node_id: &str) -> Option<&'static str> {
    match node_id {
        // Functions (parabola explorer)
        "CAPS.Mathematics.Gr12.P1.FN" | "CAPS.Mathematics.Gr11.FN.1" |
        "CAPS.Mathematics.Gr10.FN.1" | "CAPS.Mathematics.Gr10.FN.2" |
        "CAPS.Mathematics.Gr11.FN.2" => Some("parabola"),
        // Calculus (tangent line explorer)
        "CAPS.Mathematics.Gr12.P1.CALC" => Some("tangent"),
        // Trigonometry (unit circle explorer)
        "CAPS.Mathematics.Gr12.P2.TRIG" | "CAPS.Mathematics.Gr11.TRIG.1" |
        "CAPS.Mathematics.Gr10.TRIG.1" | "CAPS.Mathematics.Gr10.TRIG.2" => Some("unit_circle"),
        // Analytical Geometry
        "CAPS.Mathematics.Gr12.P2.ANAG" | "CAPS.Mathematics.Gr11.ANAG.1" |
        "CAPS.Mathematics.Gr10.ANAG.1" => Some("analytical"),
        // Statistics
        "CAPS.Mathematics.Gr12.P2.STAT" | "CAPS.Mathematics.Gr11.STAT.1" |
        "CAPS.Mathematics.Gr10.STAT.1" => Some("stats"),
        // Projectile Motion
        "CAPS.PhysicalSciences.Gr12.P1.MECH2" | "CAPS.PhysicalSciences.Gr10.PHY.1" => Some("projectile"),
        // Circuit Explorer
        "CAPS.PhysicalSciences.Gr12.P1.ELEC1" | "CAPS.PhysicalSciences.Gr10.PHY.7" |
        "CAPS.PhysicalSciences.Gr10.PHY.6" => Some("circuits"),
        // Equilibrium Simulator
        "CAPS.PhysicalSciences.Gr12.P2.EQUIL" | "CAPS.PhysicalSciences.Gr11.CHM.3" => Some("equilibrium"),
        // Acid-Base Explorer
        "CAPS.PhysicalSciences.Gr12.P2.ACID" | "CAPS.PhysicalSciences.Gr11.CHM.4" => Some("acids"),
        // Financial Literacy (Budget Simulator)
        id if id.contains("FinancialLiteracy") || id.contains("FINLIT") => Some("budget"),
        // Cybersecurity (Password Strength)
        id if id.contains("Cybersecurity") || id.contains("CYBER") || id.contains("InfoSec") => Some("password"),
        // Philosophy / Critical Thinking (Fallacy Detector)
        id if id.contains("Philosophy") || id.contains("PHIL") || id.contains("CriticalThinking") || id.contains("CRITTHINK") => Some("fallacy"),
        // Foundation Phase (Gr1-6)
        id if id.contains("Gr1") || id.contains("Gr2") || id.contains("Grade1") || id.contains("Grade2") => Some("number_bonds"),
        id if (id.contains("Gr3") || id.contains("Gr4") || id.contains("Gr5") || id.contains("Grade3") || id.contains("Grade4") || id.contains("Grade5")) && (id.contains("NF") || id.contains("fraction") || id.contains("Fraction")) => Some("fraction_pizza"),
        id if (id.contains("Gr3") || id.contains("Gr4") || id.contains("Gr5") || id.contains("Grade3") || id.contains("Grade4") || id.contains("Grade5")) && (id.contains("OA") || id.contains("multiply") || id.contains("Multiply") || id.contains("times")) => Some("times_tables"),
        _ => None,
    }
}

/// Render the game component for a given node ID.
#[component]
pub fn GameContainer(node_id: String) -> impl IntoView {
    let game = game_type(&node_id);
    let id = node_id.clone();
    match game {
        Some("parabola") => view! { <math::parabola::ParabolaExplorer node_id=id /> }.into_any(),
        Some("tangent") => view! { <math::tangent_line::TangentLineExplorer node_id=id /> }.into_any(),
        Some("unit_circle") => view! { <math::unit_circle::UnitCircleExplorer node_id=id /> }.into_any(),
        Some("analytical") => view! { <math::analytical::AnalyticalGeometryExplorer node_id=id /> }.into_any(),
        Some("stats") => view! { <math::stats_explorer::StatsExplorer node_id=id /> }.into_any(),
        Some("projectile") => view! { <physics::projectile::ProjectileExplorer node_id=id /> }.into_any(),
        Some("circuits") => view! { <physics::circuits::CircuitExplorer node_id=id /> }.into_any(),
        Some("equilibrium") => view! { <chemistry::equilibrium::EquilibriumExplorer node_id=id /> }.into_any(),
        Some("acids") => view! { <chemistry::acids::AcidBaseExplorer node_id=id /> }.into_any(),
        Some("budget") => view! { <universal::budget_sim::BudgetSimulator node_id=id /> }.into_any(),
        Some("password") => view! { <universal::password_strength::PasswordStrengthGame node_id=id /> }.into_any(),
        Some("fallacy") => view! { <universal::fallacy_detector::FallacyDetector node_id=id /> }.into_any(),
        Some("number_bonds") => view! { <foundation::number_bonds::NumberBondsGame node_id=id /> }.into_any(),
        Some("fraction_pizza") => view! { <foundation::fraction_pizza::FractionPizzaGame node_id=id /> }.into_any(),
        Some("times_tables") => view! { <foundation::times_tables::TimesTablesGame node_id=id /> }.into_any(),
        _ => view! { <p style="color: var(--text-secondary)">"No interactive game available yet."</p> }.into_any(),
    }
}
