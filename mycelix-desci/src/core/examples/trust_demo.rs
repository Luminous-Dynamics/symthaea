// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust & Reputation Demo
//!
//! Demonstrates the MATL (Mycelix Adaptive Trust Layer) including:
//! - Trust score initialization
//! - Score updates from peer interactions
//! - Trust decay over time
//! - Threshold-based trust evaluation
//! - Multi-participant trust networks
//!
//! Run with: cargo run --example trust_demo

use mycelix_desci_core::{
    trust::TrustManager,
    Result,
};

fn main() -> Result<()> {
    println!("⭐ Mycelix-DeSci Trust & Reputation Demo\n");
    println!("{}", "=".repeat(70));

    // ========================================================================
    // STEP 1: Initialize Trust Manager
    // ========================================================================
    println!("\n📦 Step 1: Initializing trust manager...\n");

    let mut trust_manager = TrustManager::new();

    println!("   ✓ Trust manager initialized");
    println!("   Default score: 0.5 (neutral)");
    println!("   Score range: [0.0, 1.0]");
    println!("   Trust threshold: 0.6 (configurable)\n");

    // ========================================================================
    // STEP 2: Participant Onboarding
    // ========================================================================
    println!("👥 Step 2: Onboarding participants...\n");

    let participants = vec![
        ("alice@stanford.edu", "Stanford Longevity Lab"),
        ("bob@mit.edu", "MIT Computational Biology"),
        ("charlie@harvard.edu", "Harvard Medical School"),
        ("diana@caltech.edu", "Caltech Neuroscience"),
        ("eve@oxford.edu", "Oxford Genomics"),
    ];

    println!("   Participants:");
    for (i, (email, affiliation)) in participants.iter().enumerate() {
        let score = trust_manager.get_score(email);
        println!("   {}. {} - {}", i + 1, email, affiliation);
        println!("      Initial score: {:.3} (confidence: {:.3})",
            score.score, score.confidence);
    }

    // ========================================================================
    // STEP 3: Positive Interactions
    // ========================================================================
    println!("\n\n✅ Step 3: Recording positive interactions...\n");

    println!("   Scenario: Participants provide high-quality verifications\n");

    // Alice provides excellent data verification
    trust_manager.update_score("alice@stanford.edu", true, 0.9)?;
    println!("   Alice verifies dataset quality (positive, weight 0.9)");
    let score = trust_manager.get_score("alice@stanford.edu");
    println!("     New score: {:.3} (confidence: {:.3})", score.score, score.confidence);

    // Bob consistently provides good reviews
    for i in 0..3 {
        trust_manager.update_score("bob@mit.edu", true, 0.85)?;
        println!("\n   Bob provides peer review #{} (positive, weight 0.85)", i + 1);
        let score = trust_manager.get_score("bob@mit.edu");
        println!("     Current score: {:.3} (confidence: {:.3})", score.score, score.confidence);
    }

    // Charlie does reproducibility studies
    trust_manager.update_score("charlie@harvard.edu", true, 0.88)?;
    println!("\n   Charlie confirms reproducibility (positive, weight 0.88)");
    let score = trust_manager.get_score("charlie@harvard.edu");
    println!("     New score: {:.3} (confidence: {:.3})", score.score, score.confidence);

    // ========================================================================
    // STEP 4: Negative Interactions
    // ========================================================================
    println!("\n\n❌ Step 4: Handling negative interactions...\n");

    println!("   Scenario: Diana submits low-quality data\n");

    // Diana's first mistake
    trust_manager.update_score("diana@caltech.edu", false, 0.7)?;
    println!("   Diana submits unverified data (negative, weight 0.7)");
    let score = trust_manager.get_score("diana@caltech.edu");
    println!("     New score: {:.3} (confidence: {:.3})", score.score, score.confidence);

    // Diana's second issue
    trust_manager.update_score("diana@caltech.edu", false, 0.75)?;
    println!("\n   Diana's data fails reproducibility check (negative, weight 0.75)");
    let score = trust_manager.get_score("diana@caltech.edu");
    println!("     New score: {:.3} (confidence: {:.3})", score.score, score.confidence);

    // ========================================================================
    // STEP 5: Trust Threshold Evaluation
    // ========================================================================
    println!("\n\n🎯 Step 5: Evaluating trust thresholds...\n");

    let threshold = 0.6;
    println!("   Trust threshold: {:.1}\n", threshold);

    println!("   Participant Trust Status:");
    for (email, affiliation) in &participants {
        let score = trust_manager.get_score(email);
        let is_trusted = trust_manager.is_trusted(email);

        let status_icon = if is_trusted { "✓" } else { "✗" };
        let status_text = if is_trusted { "[TRUSTED]" } else { "[NOT TRUSTED]" };

        println!("   {} {} - {}: {:.3} {}",
            status_icon,
            email,
            affiliation,
            score.score,
            status_text);
    }

    // ========================================================================
    // STEP 6: Trust Recovery
    // ========================================================================
    println!("\n\n🔄 Step 6: Trust recovery demonstration...\n");

    println!("   Scenario: Diana improves data quality\n");

    // Diana submits several high-quality contributions
    for i in 0..4 {
        trust_manager.update_score("diana@caltech.edu", true, 0.82)?;
        let score = trust_manager.get_score("diana@caltech.edu");
        let is_trusted = trust_manager.is_trusted("diana@caltech.edu");

        println!("   Contribution #{}: score = {:.3} {}",
            i + 1,
            score.score,
            if is_trusted { "[TRUSTED RESTORED]" } else { "" });
    }

    // ========================================================================
    // STEP 7: Confidence Evolution
    // ========================================================================
    println!("\n\n📈 Step 7: Confidence growth through interactions...\n");

    println!("   Scenario: Eve builds reputation from neutral start\n");

    let initial_score = trust_manager.get_score("eve@oxford.edu");
    println!("   Initial: score = {:.3}, confidence = {:.3}",
        initial_score.score, initial_score.confidence);

    // Simulate 10 interactions with varying weights (all positive)
    let interactions = vec![
        0.7, 0.75, 0.8, 0.85, 0.82,
        0.88, 0.9, 0.87, 0.89, 0.91,
    ];

    println!("\n   Interaction History:");
    for (i, weight) in interactions.iter().enumerate() {
        trust_manager.update_score("eve@oxford.edu", true, *weight)?;
        let score = trust_manager.get_score("eve@oxford.edu");

        println!("   #{:2}: positive (weight {:.2}) → score: {:.3}, confidence: {:.3}",
            i + 1, weight, score.score, score.confidence);
    }

    // ========================================================================
    // STEP 8: Trust Network Analysis
    // ========================================================================
    println!("\n\n🌐 Step 8: Trust network analysis...\n");

    let mut scores: Vec<_> = participants.iter()
        .map(|(email, _)| {
            let score = trust_manager.get_score(email);
            (*email, score)
        })
        .collect();

    // Sort by score descending
    scores.sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap());

    println!("   Participant Rankings:\n");
    for (rank, (email, score)) in scores.iter().enumerate() {
        let is_trusted = trust_manager.is_trusted(email);
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };

        println!("   {} #{} {} - Score: {:.3} (Conf: {:.3}) {}",
            medal,
            rank + 1,
            email,
            score.score,
            score.confidence,
            if is_trusted { "✓" } else { "" });
    }

    // ========================================================================
    // STEP 9: Trust Distribution
    // ========================================================================
    println!("\n\n📊 Step 9: Trust score distribution...\n");

    let ranges = vec![
        (0.0, 0.2, "Very Low"),
        (0.2, 0.4, "Low"),
        (0.4, 0.6, "Neutral"),
        (0.6, 0.8, "Trusted"),
        (0.8, 1.0, "Highly Trusted"),
    ];

    println!("   Distribution across trust ranges:\n");

    for (min, max, label) in &ranges {
        let count = participants.iter()
            .filter(|(email, _)| {
                let score = trust_manager.get_score(email);
                score.score >= *min && score.score < *max
            })
            .count();

        let bar = "█".repeat(count * 4);
        println!("   {:<15} [{:.1}-{:.1}): {} {}",
            label, min, max, count, bar);
    }

    // ========================================================================
    // STEP 10: Weighted Consensus Example
    // ========================================================================
    println!("\n\n🤝 Step 10: Trust-weighted consensus...\n");

    println!("   Scenario: Multiple participants vote on a claim\n");

    let votes = vec![
        ("alice@stanford.edu", true),   // High trust, approve
        ("bob@mit.edu", true),          // High trust, approve
        ("charlie@harvard.edu", true),  // Medium trust, approve
        ("diana@caltech.edu", false),   // Medium trust, reject
        ("eve@oxford.edu", true),       // High trust, approve
    ];

    println!("   Votes:");
    let mut weighted_yes = 0.0;
    let mut weighted_no = 0.0;

    for (email, vote) in &votes {
        let score = trust_manager.get_score(email);
        let weight = score.score * score.confidence;

        if *vote {
            weighted_yes += weight;
        } else {
            weighted_no += weight;
        }

        println!("   {} votes {}: weight = {:.3} (score: {:.3}, conf: {:.3})",
            email,
            if *vote { "✓ YES" } else { "✗ NO" },
            weight,
            score.score,
            score.confidence);
    }

    let total_weight = weighted_yes + weighted_no;
    let consensus = weighted_yes / total_weight;

    println!("\n   Consensus Calculation:");
    println!("     Weighted YES: {:.3}", weighted_yes);
    println!("     Weighted NO:  {:.3}", weighted_no);
    println!("     Consensus:    {:.1}% approval", consensus * 100.0);
    println!("\n     Decision: {}",
        if consensus > 0.66 { "✓ APPROVED (>66% consensus)" } else { "✗ REJECTED" });

    // ========================================================================
    // Summary Statistics
    // ========================================================================
    println!("\n\n📊 Summary Statistics");
    println!("{}", "=".repeat(70));

    let all_scores: Vec<_> = participants.iter()
        .map(|(email, _)| trust_manager.get_score(email).score)
        .collect();

    let avg_score: f64 = all_scores.iter().sum::<f64>() / all_scores.len() as f64;
    let max_score = all_scores.iter().cloned().fold(0./0., f64::max);
    let min_score = all_scores.iter().cloned().fold(1./0., f64::min);

    let trusted_count = participants.iter()
        .filter(|(email, _)| trust_manager.is_trusted(email))
        .count();

    println!("\n   Total Participants: {}", participants.len());
    println!("   Trusted (≥{:.1}):    {}", threshold, trusted_count);
    println!("   Not Trusted:       {}", participants.len() - trusted_count);
    println!("\n   Score Statistics:");
    println!("     Average: {:.3}", avg_score);
    println!("     Highest: {:.3}", max_score);
    println!("     Lowest:  {:.3}", min_score);
    println!("     Range:   {:.3}", max_score - min_score);

    println!("\n{}", "=".repeat(70));
    println!("✅ Trust Demo Complete!\n");
    println!("Demonstrated Features:");
    println!("  • Trust score initialization and updates");
    println!("  • Positive and negative interactions");
    println!("  • Trust recovery mechanisms");
    println!("  • Confidence growth over time");
    println!("  • Threshold-based trust evaluation");
    println!("  • Network-wide trust analysis");
    println!("  • Trust-weighted consensus voting");
    println!("\n⭐ MATL provides robust reputation management!");
    println!("{}\n", "=".repeat(70));

    Ok(())
}
