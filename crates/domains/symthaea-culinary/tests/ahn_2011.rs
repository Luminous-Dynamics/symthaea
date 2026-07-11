//! GROUND-TRUTH TEST — reproduces the headline result of
//!
//!   Ahn, Ahnert, Bagrow & Barabási, "Flavor network and the principles of food
//!   pairing", Scientific Reports 1, 196 (2011).
//!
//! The paper's central, falsifiable claim (its Fig. 3): North American and Western
//! European cuisines pair ingredients that **share** flavor compounds more than
//! chance (ΔNc > 0), while East Asian and Southern European cuisines **avoid**
//! shared compounds (ΔNc < 0).
//!
//! If this test fails, either the flavor-vector representation or the ΔNc metric is
//! wrong — per CULINARY_PLAN_2026-07-09.md Phase 0, that means stop and fix the
//! foundation, not tune numbers to pass.

use symthaea_culinary::flavor_network::delta_nc_default;

fn delta(cuisine: &str) -> f64 {
    delta_nc_default(cuisine)
        .unwrap_or_else(|| panic!("cuisine {cuisine} missing from dataset"))
        .delta
}

#[test]
fn western_cuisines_pair_shared_compounds() {
    // The compound-sharing cuisines: ΔNc must be positive.
    let na = delta("NorthAmerican");
    let we = delta("WesternEuropean");
    assert!(na > 0.0, "NorthAmerican ΔNc should be > 0, got {na:+.3}");
    assert!(we > 0.0, "WesternEuropean ΔNc should be > 0, got {we:+.3}");
}

#[test]
fn east_asian_avoids_shared_compounds() {
    // The contrast cuisine: ΔNc must be negative.
    let ea = delta("EastAsian");
    assert!(ea < 0.0, "EastAsian ΔNc should be < 0, got {ea:+.3}");
}

#[test]
fn southern_european_avoids_shared_compounds() {
    // Mediterranean cooking is the other classic negative in the paper.
    let se = delta("SouthernEuropean");
    assert!(se < 0.0, "SouthernEuropean ΔNc should be < 0, got {se:+.3}");
}

#[test]
fn contrast_is_ordered() {
    // The headline contrast: North American shares more than East Asian.
    let na = delta("NorthAmerican");
    let ea = delta("EastAsian");
    assert!(
        na > ea,
        "NorthAmerican ({na:+.3}) should exceed EastAsian ({ea:+.3})"
    );
}

/// Not an assertion — prints the full ΔNc table for eyeballing against Fig. 3.
/// Run with `cargo test -p symthaea-culinary --test ahn_2011 -- --nocapture`.
#[test]
fn print_all_cuisines() {
    let cuisines = [
        "NorthAmerican",
        "WesternEuropean",
        "SouthernEuropean",
        "EasternEuropean",
        "NorthernEuropean",
        "LatinAmerican",
        "MiddleEastern",
        "African",
        "EastAsian",
        "SoutheastAsian",
        "SouthAsian",
    ];
    println!("\ncuisine            real     null    ΔNc    recipes");
    for c in cuisines {
        if let Some(r) = delta_nc_default(c) {
            println!(
                "{:18} {:6.2} {:7.2} {:+6.3} {:8}",
                c, r.real, r.null, r.delta, r.recipes_used
            );
        }
    }
}
