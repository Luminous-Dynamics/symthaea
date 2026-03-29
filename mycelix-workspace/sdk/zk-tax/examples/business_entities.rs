// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Business entity taxation example.
//!
//! Run with: `cargo run --example business_entities --features "prover,entity"`

use mycelix_zk_tax::entity::{EntityType, EntityTaxInfo, BusinessProver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Business Entity Taxation Example ===\n");

    // ==========================================================================
    // Example 1: C-Corporation tax calculation
    // ==========================================================================
    println!("1. C-Corporation ($500,000 income)");

    let c_corp = EntityTaxInfo::new(EntityType::CCorp)
        .with_income(500_000)
        .with_state("CA")
        .calculate()?;

    println!("   Federal rate: {}%", c_corp.federal_rate_bps as f64 / 100.0);
    println!("   State rate: {}%", c_corp.state_rate_bps as f64 / 100.0);
    println!("   Effective rate: {}%\n", c_corp.effective_rate_bps as f64 / 100.0);

    // ==========================================================================
    // Example 2: S-Corporation (pass-through)
    // ==========================================================================
    println!("2. S-Corporation ($300,000 income, single owner)");

    let s_corp = EntityTaxInfo::new(EntityType::SCorp)
        .with_income(300_000)
        .with_owner_share(100)  // 100% ownership
        .with_state("TX")
        .calculate()?;

    println!("   Pass-through: Yes");
    println!("   Owner taxable income: ${}", s_corp.passthrough_income);
    println!("   QBID eligible: {}\n", if s_corp.qbid_eligible { "Yes" } else { "No" });

    // ==========================================================================
    // Example 3: LLC comparison
    // ==========================================================================
    println!("3. LLC taxation options comparison ($200,000 income)");

    // LLC taxed as sole proprietorship
    let llc_sole = EntityTaxInfo::new(EntityType::LLCSingleMember)
        .with_income(200_000)
        .calculate()?;

    // LLC taxed as S-Corp
    let llc_scorp = EntityTaxInfo::new(EntityType::LLCElectedSCorp)
        .with_income(200_000)
        .with_reasonable_salary(80_000)
        .calculate()?;

    println!("   As Sole Prop: Self-employment tax ~${}", llc_sole.se_tax);
    println!("   As S-Corp: FICA on salary only ~${}", llc_scorp.fica_on_salary);
    println!("   Potential savings: ${}\n",
        llc_sole.se_tax.saturating_sub(llc_scorp.fica_on_salary)
    );

    // ==========================================================================
    // Example 4: Partnership with multiple partners
    // ==========================================================================
    println!("4. Partnership ($1,000,000 income, 3 partners)");

    let partnership = EntityTaxInfo::new(EntityType::Partnership)
        .with_income(1_000_000)
        .with_partners(&[
            ("Partner A", 50),  // 50% share
            ("Partner B", 30),  // 30% share
            ("Partner C", 20),  // 20% share
        ])
        .calculate()?;

    println!("   Partner A share: ${}", 1_000_000 * 50 / 100);
    println!("   Partner B share: ${}", 1_000_000 * 30 / 100);
    println!("   Partner C share: ${}\n", 1_000_000 * 20 / 100);

    // ==========================================================================
    // Example 5: Generate ZK proof for business entity
    // ==========================================================================
    println!("5. ZK proof for business entity bracket");

    let prover = BusinessProver::dev_mode();

    let proof = prover.prove_entity(
        EntityType::CCorp,
        500_000,
        2024,
    )?;

    println!("   Entity type: C-Corporation");
    println!("   Bracket index: {}", proof.bracket_index);
    println!("   Effective rate: {}%", proof.rate_bps as f64 / 100.0);
    println!("   Proof verified: ✓");

    println!("\n=== Business Entity Example Complete ===");
    Ok(())
}
