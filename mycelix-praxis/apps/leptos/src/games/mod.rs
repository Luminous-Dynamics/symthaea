// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Educational games — interactive STEM simulations embedded in study pages.

pub mod shared;
pub mod foundation;
pub mod intermediate;
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
        // Equation Explorer (Physics Discovery)
        "CAPS.PhysicalSciences.Gr12.P1.EQUATIONS" | "CAPS.PhysicalSciences.Gr11.P1.EQUATIONS" |
        "UNIVERSAL.PhysicsEquations" | "UNIVERSAL.EquationExplorer" => Some("equation_explorer"),
        // Newton's Gravitation
        "CAPS.PhysicalSciences.Gr11.P1.GRAV" | "CAPS.PhysicalSciences.Gr12.P1.GRAV" |
        "UNIVERSAL.Gravitation" => Some("gravity"),
        // Ideal Gas Law
        "CAPS.PhysicalSciences.Gr11.P1.GASES" | "CAPS.PhysicalSciences.Gr12.P2.GASES" |
        "UNIVERSAL.IdealGas" => Some("ideal_gas"),
        // Schwarzschild Radius / Black Holes
        "UNIVERSAL.BlackHoles" | "UNIVERSAL.Schwarzschild" => Some("schwarzschild"),
        // Coulomb's Law
        "CAPS.PhysicalSciences.Gr11.P1.ELEC2" | "UNIVERSAL.Coulomb" => Some("coulomb"),
        // Lorentz Factor / Special Relativity
        "CAPS.PhysicalSciences.Gr12.P1.REL" | "UNIVERSAL.Relativity" | "UNIVERSAL.Lorentz" => Some("lorentz"),
        // Hubble's Law / Cosmology
        "UNIVERSAL.Cosmology" | "UNIVERSAL.Hubble" => Some("hubble"),
        // Boltzmann Distribution
        "CAPS.PhysicalSciences.Gr12.P1.THERMO" | "UNIVERSAL.Boltzmann" | "UNIVERSAL.StatMech" => Some("boltzmann"),
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
        // Intermediate Phase (Gr6-9)
        id if (id.contains("Gr6") || id.contains("Grade6")) && (id.contains("integer") || id.contains("Integer") || id.contains("negative") || id.contains("NS")) => Some("integer_line"),
        id if (id.contains("Gr7") || id.contains("Gr8") || id.contains("Grade7") || id.contains("Grade8")) && (id.contains("ALG") || id.contains("algebra") || id.contains("equation") || id.contains("EE")) => Some("equation_balance"),
        id if id.contains("Pythag") || id.contains("pythag") || id.contains("hypotenuse") || (id.contains("Gr8") && id.contains("GEOM")) => Some("pythagoras"),
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
        Some("integer_line") => view! { <intermediate::integer_number_line::IntegerNumberLine node_id=id /> }.into_any(),
        Some("equation_balance") => view! { <intermediate::equation_balance::EquationBalance node_id=id /> }.into_any(),
        Some("pythagoras") => view! { <intermediate::pythagoras::PythagorasExplorer node_id=id /> }.into_any(),
        Some("equation_explorer") => view! { <physics::equation_explorer::EquationExplorer node_id=id /> }.into_any(),
        Some("gravity") => view! { <physics::gravity_explorer::GravityExplorer node_id=id /> }.into_any(),
        Some("ideal_gas") => view! { <physics::ideal_gas_explorer::IdealGasExplorer node_id=id /> }.into_any(),
        Some("schwarzschild") => view! { <physics::schwarzschild_explorer::SchwarzschildExplorer node_id=id /> }.into_any(),
        Some("coulomb") => view! { <physics::coulomb_explorer::CoulombExplorer node_id=id /> }.into_any(),
        Some("lorentz") => view! { <physics::lorentz_explorer::LorentzExplorer node_id=id /> }.into_any(),
        Some("hubble") => view! { <physics::hubble_explorer::HubbleExplorer node_id=id /> }.into_any(),
        Some("boltzmann") => view! { <physics::boltzmann_explorer::BoltzmannExplorer node_id=id /> }.into_any(),
        _ => view! { <p style="color: var(--text-secondary)">"No interactive game available yet."</p> }.into_any(),
    }
}
