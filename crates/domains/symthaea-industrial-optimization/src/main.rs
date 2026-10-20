// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic synthetic demo. All intervention costs and abatement values
//! below are invented test fixtures, not field measurements or industry averages.

use symthaea_industrial_optimization::{
    AbatementEstimate, Intervention, greedy_expected_per_dollar, optimize_robust_portfolio,
};

fn item(
    id: &str,
    source_group: &str,
    capex_usd: u64,
    floor: u64,
    expected: u64,
    ceiling: u64,
    eligible_for_planning: bool,
) -> Intervention {
    Intervention {
        id: id.to_owned(),
        source_group: source_group.to_owned(),
        capex_usd,
        abatement: AbatementEstimate {
            floor_tonnes_ch4_per_year: floor,
            expected_tonnes_ch4_per_year: expected,
            ceiling_tonnes_ch4_per_year: ceiling,
        },
        eligible_for_planning,
    }
}

fn print_portfolio(label: &str, portfolio: &symthaea_industrial_optimization::Portfolio) {
    println!("{label}");
    println!("  selected: {}", portfolio.selected_ids.join(", "));
    println!("  capital USD: {}", portfolio.capex_usd);
    println!(
        "  scenario CH4 abatement (floor / expected / ceiling): {} / {} / {} tonnes per year",
        portfolio.floor_tonnes_ch4_per_year,
        portfolio.expected_tonnes_ch4_per_year,
        portfolio.ceiling_tonnes_ch4_per_year
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SYNTHETIC BENCHMARK ONLY — NOT A SITE ESTIMATE OR OPERATING ADVICE");
    let candidates = vec![
        item("ldar-repair-event-a", "measured-source-event-a", 22_000, 2, 7, 15, true),
        item("dry-compressor-seal", "compressor-seals-unit-1", 88_000, 7, 12, 18, true),
        item("compressor-seal-repair", "compressor-seals-unit-1", 30_000, 2, 5, 9, true),
        item("pneumatic-controller-swap", "pneumatic-unit-4", 45_000, 3, 7, 11, true),
        item("tank-vapor-recovery", "tank-vents-battery-2", 72_000, 5, 10, 15, true),
        item("associated-gas-utilization", "low-pressure-source-7", 190_000, 12, 20, 32, true),
        item("pump-electrification", "pump-drive-unit-3", 115_000, 5, 11, 18, true),
        item("unqualified-capture-concept", "source-concept-only", 4_000, 40, 50, 60, false),
    ];

    let budget_usd = 250_000;
    let greedy = greedy_expected_per_dollar(&candidates, budget_usd)?;
    let robust = optimize_robust_portfolio(&candidates, budget_usd)?;

    println!("capital budget USD: {budget_usd}");
    print_portfolio("\nGreedy baseline (expected abatement per dollar):", &greedy);
    print_portfolio("\nExact robust baseline (scenario-floor abatement):", &robust);
    println!("\nInterpretation: the two strategies optimize different objectives.");
    println!("These outputs validate portfolio-selection mechanics only; they establish");
    println!("nothing about actual methane emissions, intervention efficacy, cost, or safety.");
    Ok(())
}
