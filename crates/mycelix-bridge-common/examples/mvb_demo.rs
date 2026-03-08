//! Minimal Viable Bridge Demo
//!
//! Demonstrates the full Symthaea → Mycelix governance pipeline:
//!   C_unified → ConsciousnessProfile → ConsciousnessCredential → evaluate_governance → audit log
//!
//! Run: `cargo run --example mvb_demo` (from crates/mycelix-bridge-common/)

use mycelix_bridge_common::consciousness_profile::{
    evaluate_governance, requirement_for_basic, requirement_for_constitutional,
    requirement_for_proposal, requirement_for_voting, should_audit, ConsciousnessCredential,
    ConsciousnessProfile, GateAuditInput,
};

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

fn demo_agent(name: &str, c_unified: f64, identity: f64, reputation: f64, community: f64) {
    let now = now_us();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Agent: {}", name);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Step 1: Map C_unified → profile
    let profile = ConsciousnessProfile::from_unified_consciousness(
        c_unified, identity, reputation, community,
    );
    println!("\n  [1] Profile from C_unified = {:.2}", c_unified);
    println!("      identity:   {:.2} (25%)", profile.identity);
    println!("      reputation: {:.2} (25%)", profile.reputation);
    println!("      community:  {:.2} (30%)", profile.community);
    println!(
        "      engagement: {:.2} (20%) <- C_unified",
        profile.engagement
    );
    println!("      combined:   {:.3}", profile.combined_score());
    println!("      tier:       {:?}", profile.tier());

    // Step 2: Issue credential
    let cred = ConsciousnessCredential::from_unified_consciousness(
        format!("did:mycelix:{}", name),
        c_unified,
        identity,
        reputation,
        community,
        "did:mycelix:bridge_demo".into(),
        now,
    );
    println!("\n  [2] Credential issued");
    println!("      did:     {}", cred.did);
    println!("      tier:    {:?}", cred.tier);
    println!("      expires: {} (24h from now)", cred.expires_at);

    // Step 3: Evaluate against each governance level
    let requirements = [
        ("basic", requirement_for_basic()),
        ("proposal", requirement_for_proposal()),
        ("voting", requirement_for_voting()),
        ("constitutional", requirement_for_constitutional()),
    ];

    println!("\n  [3] Governance evaluation:");
    for (action, req) in &requirements {
        let result = evaluate_governance(&cred, req, now);
        let icon = if result.eligible { "Y" } else { "N" };
        let weight = if result.weight_bp > 0 {
            format!("weight={}bp", result.weight_bp)
        } else {
            "no weight".into()
        };
        println!(
            "      {:<16} [{}] {} (tier={:?})",
            action, icon, weight, result.tier
        );

        if !result.reasons.is_empty() {
            for reason in &result.reasons {
                println!("        reason: {}", reason);
            }
        }

        // Step 4: Audit decision
        let audit_required = should_audit(req, result.eligible, name.as_bytes(), action);
        if audit_required {
            let audit = GateAuditInput {
                action_name: format!("demo_{}", action),
                zome_name: "commons_bridge".into(),
                eligible: result.eligible,
                actual_tier: format!("{:?}", result.tier),
                required_tier: format!("{:?}", req.min_tier),
                weight_bp: result.weight_bp,
                correlation_id: Some(format!("mvb_demo:{}:{}", name, now)),
            };
            let json = serde_json::to_string(&audit).unwrap();
            println!("        audit: {}", json);
        }
    }
    println!();
}

fn main() {
    println!();
    println!("===========================================================");
    println!("  MINIMAL VIABLE BRIDGE DEMO");
    println!("  Symthaea C_unified -> Mycelix Governance Pipeline");
    println!("===========================================================");
    println!();

    // Agent 1: High consciousness, verified identity
    demo_agent("sophia", 0.75, 0.90, 0.70, 0.60);

    // Agent 2: Moderate consciousness, moderate everything
    demo_agent("explorer", 0.45, 0.50, 0.40, 0.35);

    // Agent 3: Low consciousness, minimal identity
    demo_agent("newcomer", 0.10, 0.20, 0.10, 0.05);

    // Agent 4: High consciousness but no identity verification
    demo_agent("anon_genius", 0.95, 0.05, 0.10, 0.10);

    println!("===========================================================");
    println!("  Key insight: Governance gates blast radius, not voice.");
    println!("  All agents can READ. Only action scope is gated.");
    println!("  C_unified affects only 20% of the combined score.");
    println!("===========================================================");
    println!();
}
