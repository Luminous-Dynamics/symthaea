//! Subnational (state/province/canton) tax examples.
//!
//! Run with: `cargo run --example subnational --features prover`

use mycelix_zk_tax::{
    FilingStatus, Jurisdiction, TaxBracketProver,
    subnational::{USState, SwissCanton, get_state_brackets, get_cantonal_brackets},
    proof::CombinedProofBuilder,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Subnational Tax Examples ===\n");

    let prover = TaxBracketProver::dev_mode();

    // ==========================================================================
    // Example 1: US State tax brackets
    // ==========================================================================
    println!("1. US State Tax Brackets for $100,000 income (Single, 2024)");

    let states = [
        (USState::CA, "California"),
        (USState::NY, "New York"),
        (USState::TX, "Texas"),
        (USState::FL, "Florida"),
        (USState::WA, "Washington"),
    ];

    for (state, name) in states {
        match get_state_brackets(state, 2024, FilingStatus::Single) {
            Ok(brackets) => {
                // Find bracket for $100k
                let bracket = brackets.iter().find(|b| 100_000 >= b.lower && 100_000 <= b.upper);
                if let Some(b) = bracket {
                    println!("   {}: {}% (Bracket: ${} - ${})",
                        name,
                        b.rate_bps as f64 / 100.0,
                        b.lower,
                        if b.upper == u64::MAX { "∞".to_string() } else { b.upper.to_string() }
                    );
                }
            }
            Err(_) => println!("   {}: No state income tax", name),
        }
    }

    // ==========================================================================
    // Example 2: Swiss Cantonal taxes
    // ==========================================================================
    println!("\n2. Swiss Cantonal Tax Rates for CHF 150,000 income");

    let cantons = [
        (SwissCanton::ZH, "Zürich"),
        (SwissCanton::ZG, "Zug"),
        (SwissCanton::GE, "Geneva"),
        (SwissCanton::BS, "Basel-Stadt"),
        (SwissCanton::VD, "Vaud"),
    ];

    for (canton, name) in cantons {
        let brackets = get_cantonal_brackets(canton, 2024, FilingStatus::Single)?;
        let bracket = brackets.iter().find(|b| 150_000 >= b.lower && 150_000 <= b.upper);
        if let Some(b) = bracket {
            println!("   {}: {}%", name, b.rate_bps as f64 / 100.0);
        }
    }

    // ==========================================================================
    // Example 3: Combined Federal + State proof
    // ==========================================================================
    println!("\n3. Combined Federal + State Proof (California, $120,000)");

    let combined_proof = CombinedProofBuilder::new(120_000, 2024)
        .federal(Jurisdiction::US, FilingStatus::Single)
        .state(USState::CA, FilingStatus::Single)
        .build_dev()?;

    println!("   Federal bracket: {} @ {}%",
        combined_proof.federal_proof.bracket_index,
        combined_proof.federal_proof.rate_bps as f64 / 100.0
    );
    println!("   State bracket: {} @ {}%",
        combined_proof.state_proof.bracket_index,
        combined_proof.state_proof.rate_bps as f64 / 100.0
    );
    println!("   Combined marginal: {}%",
        (combined_proof.federal_proof.rate_bps + combined_proof.state_proof.rate_bps) as f64 / 100.0
    );

    // ==========================================================================
    // Example 4: State comparison for same income
    // ==========================================================================
    println!("\n4. State Tax Comparison for $150,000 income");

    let high_tax_states = [USState::CA, USState::NY, USState::NJ, USState::OR, USState::MN];
    let no_tax_states = [USState::TX, USState::FL, USState::WA, USState::NV, USState::WY];

    println!("   High-tax states:");
    for state in high_tax_states {
        if let Ok(brackets) = get_state_brackets(state, 2024, FilingStatus::Single) {
            if let Some(b) = brackets.iter().find(|b| 150_000 >= b.lower && 150_000 <= b.upper) {
                println!("     {}: {}%", state.name(), b.rate_bps as f64 / 100.0);
            }
        }
    }

    println!("   No income tax states:");
    for state in no_tax_states {
        println!("     {}: 0%", state.name());
    }

    // ==========================================================================
    // Example 5: Tax burden analysis
    // ==========================================================================
    println!("\n5. Total Tax Burden Analysis ($200,000 income, Single)");

    // California resident
    let ca_federal = prover.prove(200_000, Jurisdiction::US, FilingStatus::Single, 2024)?;
    let ca_state = get_state_brackets(USState::CA, 2024, FilingStatus::Single)?;
    let ca_state_rate = ca_state.iter()
        .find(|b| 200_000 >= b.lower && 200_000 <= b.upper)
        .map(|b| b.rate_bps)
        .unwrap_or(0);

    // Texas resident
    let tx_federal = prover.prove(200_000, Jurisdiction::US, FilingStatus::Single, 2024)?;

    println!("   California:");
    println!("     Federal marginal: {}%", ca_federal.rate_bps as f64 / 100.0);
    println!("     State marginal: {}%", ca_state_rate as f64 / 100.0);
    println!("     Combined: {}%", (ca_federal.rate_bps + ca_state_rate) as f64 / 100.0);

    println!("   Texas:");
    println!("     Federal marginal: {}%", tx_federal.rate_bps as f64 / 100.0);
    println!("     State marginal: 0%");
    println!("     Combined: {}%", tx_federal.rate_bps as f64 / 100.0);

    println!("\n=== Subnational Example Complete ===");
    Ok(())
}
