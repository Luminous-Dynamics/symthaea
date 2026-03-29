// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Epistemic Agent Lifecycle Demo
//!
//! This comprehensive demo showcases the full epistemic agent lifecycle:
//!
//! 1. **Agent Creation** - Initialize agent with K-Vector trust profile
//! 2. **Output Production** - Agent produces E-N-M-H classified outputs
//! 3. **Trust Evolution** - K-Vector updates based on behavior outcomes
//! 4. **KREDIT Dynamics** - Economic allocation tracks trust score
//! 5. **Attack Detection** - Gaming behavior triggers quarantine/slashing
//! 6. **Privacy Analytics** - Differential privacy preserves agent anonymity
//!
//! This demonstrates the novel "epistemic fingerprint" concept where AI agents
//! have verifiable trust profiles that evolve based on behavioral outcomes
//! evaluated through epistemic classification.

use mycelix_sdk::agentic::{
    // Differential privacy
    differential_privacy::DPConfig,
    // Epistemic classifier
    epistemic_classifier::OutputContent,
    // Integration flows
    integration::{
        AttackResponseConfig, EpistemicLifecycleConfig, IntegratedAttackResponse,
        IntegratedEpistemicLifecycle, IntegratedPrivacyAnalytics, IntegratedTrustPipeline,
        PrivacyAnalyticsConfig, TrustPipelineConfig,
    },
    // Core types
    AgentClass,
    VerificationOutcome,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        MYCELIX EPISTEMIC AGENT LIFECYCLE DEMO                ║");
    println!("║     Verifiable Trust Profiles for AI Agents                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // Phase 1: Agent Creation
    // =========================================================================
    demo_phase_1_agent_creation();

    // =========================================================================
    // Phase 2: Output Production with E-N-M-H Classification
    // =========================================================================
    demo_phase_2_output_production();

    // =========================================================================
    // Phase 3: Trust Evolution and KREDIT Dynamics
    // =========================================================================
    demo_phase_3_trust_evolution();

    // =========================================================================
    // Phase 4: Multi-Agent Trust Pipeline
    // =========================================================================
    demo_phase_4_trust_pipeline();

    // =========================================================================
    // Phase 5: Attack Detection and Response
    // =========================================================================
    demo_phase_5_attack_response();

    // =========================================================================
    // Phase 6: Privacy-Preserving Analytics
    // =========================================================================
    demo_phase_6_privacy_analytics();

    // =========================================================================
    // Phase 7: Full Integrated Scenario
    // =========================================================================
    demo_phase_7_full_scenario();

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                     DEMO COMPLETE                            ║");
    println!("║  All epistemic agent lifecycle phases demonstrated           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn demo_phase_1_agent_creation() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 1: Agent Creation with K-Vector Trust Profile          │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    let lifecycle = IntegratedEpistemicLifecycle::new(EpistemicLifecycleConfig::default());

    // Create different classes of agents
    let agents: Vec<(&str, AgentClass)> = vec![
        ("Supervised", AgentClass::Supervised),
        ("Assistive", AgentClass::Assistive),
        ("Autonomous", AgentClass::Autonomous),
    ];

    println!("Creating agents with initial epistemic fingerprints:");
    println!();

    for (name, class) in agents {
        let agent = lifecycle.create_agent("did:mycelix:sponsor:demo", class.clone());

        println!("  {} Agent ({:?})", name, class);
        println!("    ├─ ID: {}", agent.agent_id.as_str());
        println!("    ├─ Status: {:?}", agent.status);
        println!(
            "    ├─ KREDIT Cap: {} (balance: {})",
            agent.kredit_cap, agent.kredit_balance
        );
        println!("    └─ K-Vector Trust Profile (10 dimensions):");
        println!("       ├─ k_r (reputation):    {:.3}", agent.k_vector.k_r);
        println!("       ├─ k_a (activity):      {:.3}", agent.k_vector.k_a);
        println!("       ├─ k_i (integrity):     {:.3}", agent.k_vector.k_i);
        println!("       ├─ k_p (performance):   {:.3}", agent.k_vector.k_p);
        println!("       ├─ k_m (membership):    {:.3}", agent.k_vector.k_m);
        println!("       ├─ k_s (stake):         {:.3}", agent.k_vector.k_s);
        println!("       ├─ k_h (historical):    {:.3}", agent.k_vector.k_h);
        println!(
            "       ├─ k_topo (topology):   {:.3}",
            agent.k_vector.k_topo
        );
        println!("       ├─ k_v (verification):  {:.3}", agent.k_vector.k_v);
        println!(
            "       ├─ k_coherence (coherence):   {:.3}",
            agent.k_vector.k_coherence
        );
        println!(
            "       └─ Trust Score:         {:.3}",
            agent.k_vector.trust_score()
        );
        println!();
    }
}

fn demo_phase_2_output_production() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 2: Output Production with E-N-M-H Classification       │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    let lifecycle = IntegratedEpistemicLifecycle::new(EpistemicLifecycleConfig::default());
    let mut agent = lifecycle.create_agent("did:mycelix:sponsor:demo", AgentClass::Supervised);

    println!("Agent producing outputs with epistemic classification:");
    println!();

    let outputs: Vec<(&str, OutputContent)> = vec![
        (
            "Simple observation",
            OutputContent::Text("Temperature is 25C".to_string()),
        ),
        (
            "Structured data",
            OutputContent::Json(r#"{"status": "ok", "value": 42}"#.to_string()),
        ),
        (
            "Action result",
            OutputContent::ActionResult {
                action_type: "data_fetch".to_string(),
                success: true,
                details: "Retrieved 100 records".to_string(),
            },
        ),
        (
            "Analysis report",
            OutputContent::Analysis {
                subject: "Market trends".to_string(),
                findings: vec!["Growth detected".to_string(), "Risk moderate".to_string()],
                recommendations: vec!["Monitor closely".to_string()],
            },
        ),
    ];

    for (i, (description, content)) in outputs.into_iter().enumerate() {
        let result = lifecycle.process_output(&mut agent, content);

        match result {
            Ok(r) => {
                println!("  Output {}: {}", i + 1, description);
                println!(
                    "    ├─ Output ID: {}...",
                    &r.output_id[..16.min(r.output_id.len())]
                );
                println!("    ├─ Epistemic Weight: {:.3}", r.epistemic_weight);
                println!("    ├─ Trust Delta: +{:.4}", r.trust_delta);
                println!("    ├─ New Trust: {:.3}", r.new_trust);
                println!("    ├─ New KREDIT Cap: {}", r.new_kredit_cap);
                println!("    └─ Reward: {} KREDIT", r.reward);
                println!();
            }
            Err(e) => println!("    Error: {:?}", e),
        }
    }

    println!("  Agent Epistemic Stats:");
    println!(
        "    ├─ Total Outputs: {}",
        agent.epistemic_stats.total_outputs
    );
    println!(
        "    └─ Avg Epistemic Weight: {:.3}",
        agent.epistemic_stats.average_weight
    );
    println!();
}

fn demo_phase_3_trust_evolution() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 3: Trust Evolution and KREDIT Dynamics                 │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    let lifecycle = IntegratedEpistemicLifecycle::new(EpistemicLifecycleConfig::default());
    let mut agent = lifecycle.create_agent("did:mycelix:sponsor:demo", AgentClass::Supervised);

    let initial_trust = agent.k_vector.trust_score();
    let initial_kredit = agent.kredit_cap;

    println!("Simulating agent behavior over time:");
    println!(
        "  Initial State: Trust={:.3}, KREDIT Cap={}",
        initial_trust, initial_kredit
    );
    println!();

    // Produce several outputs
    let mut output_ids = vec![];
    for i in 0..5 {
        let result = lifecycle
            .process_output(
                &mut agent,
                OutputContent::Text(format!("Analysis report #{}", i)),
            )
            .unwrap();
        output_ids.push(result.output_id);
    }

    println!(
        "  After 5 outputs: Trust={:.3}, KREDIT Cap={}",
        agent.k_vector.trust_score(),
        agent.kredit_cap
    );

    // Verify outputs as correct (simulating positive feedback)
    println!();
    println!("  Verifying outputs as correct (positive feedback):");
    for (i, output_id) in output_ids.iter().take(3).enumerate() {
        lifecycle.verify_output(&mut agent, output_id, true);
        println!(
            "    Output {} verified CORRECT -> Trust={:.3}",
            i + 1,
            agent.k_vector.trust_score()
        );
    }

    // Verify some as incorrect (simulating negative feedback)
    println!();
    println!("  Verifying outputs as incorrect (negative feedback):");
    for (i, output_id) in output_ids.iter().skip(3).enumerate() {
        lifecycle.verify_output(&mut agent, output_id, false);
        println!(
            "    Output {} verified INCORRECT -> Trust={:.3}",
            i + 4,
            agent.k_vector.trust_score()
        );
    }

    let final_trust = agent.k_vector.trust_score();
    let final_kredit = agent.kredit_cap;

    // Count verified outputs from history
    let verified_correct = agent
        .output_history
        .iter()
        .filter(|e| {
            e.verified
                && e.verification_outcome
                    .map(|o| o == VerificationOutcome::Correct)
                    .unwrap_or(false)
        })
        .count();
    let verified_incorrect = agent
        .output_history
        .iter()
        .filter(|e| {
            e.verified
                && e.verification_outcome
                    .map(|o| o == VerificationOutcome::Incorrect)
                    .unwrap_or(false)
        })
        .count();

    println!();
    println!("  Trust Evolution Summary:");
    println!("    ├─ Initial Trust:  {:.3}", initial_trust);
    println!("    ├─ Final Trust:    {:.3}", final_trust);
    println!("    ├─ Trust Change:   {:+.3}", final_trust - initial_trust);
    println!(
        "    ├─ KREDIT Change:  {} -> {}",
        initial_kredit, final_kredit
    );
    println!(
        "    └─ Calibration:    {}/{} correct ({:.1}%)",
        verified_correct,
        verified_correct + verified_incorrect,
        (verified_correct as f64 / (verified_correct + verified_incorrect).max(1) as f64) * 100.0
    );
    println!();
}

fn demo_phase_4_trust_pipeline() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 4: Multi-Agent Trust Pipeline                          │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    let lifecycle = IntegratedEpistemicLifecycle::new(EpistemicLifecycleConfig::default());
    let config = TrustPipelineConfig::default();
    let mut pipeline = IntegratedTrustPipeline::new(config);

    // Create and register multiple agents
    println!("Creating trust network with 5 agents:");
    for i in 0..5 {
        let agent = lifecycle.create_agent(
            &format!("did:mycelix:sponsor:{}", i),
            AgentClass::Supervised,
        );
        let agent_id = agent.agent_id.as_str().to_string();
        let trust = agent.k_vector.trust_score();
        pipeline.register_agent(agent);
        println!(
            "  Agent {}: {} (trust: {:.3})",
            i,
            &agent_id[..16.min(agent_id.len())],
            trust
        );
    }

    println!();
    println!("Processing trust attestations (quadratic voting enabled):");

    // Get agent IDs for attestations
    let agent_ids: Vec<String> = pipeline.agents().keys().cloned().collect();

    // Create attestations
    for i in 0..agent_ids.len().saturating_sub(1) {
        match pipeline.process_attestation(&agent_ids[i], &agent_ids[i + 1], 0.8) {
            Ok(result) => {
                println!(
                    "  {} -> {}",
                    &agent_ids[i][..8.min(agent_ids[i].len())],
                    &agent_ids[i + 1][..8.min(agent_ids[i + 1].len())]
                );
                println!("    ├─ Weight: {:.3}", result.weight);
                println!(
                    "    ├─ Trust: {:.3} -> {:.3}",
                    result.old_trust, result.new_trust
                );
                println!("    └─ KREDIT Cap: {}", result.new_kredit_cap);
            }
            Err(e) => println!("  Error: {:?}", e),
        }
    }

    // Simulate trust cascade
    println!();
    println!("Simulating trust cascade from first agent:");
    let cascade_result = pipeline.simulate_cascade(&agent_ids[0], 0.3);
    println!("  ├─ Agents Affected: {}", cascade_result.agents_affected);
    println!("  ├─ Agents Failed: {}", cascade_result.agents_failed);
    println!(
        "  ├─ Total Trust Loss: {:.3}",
        cascade_result.total_trust_loss
    );
    println!("  └─ Max Depth: {}", cascade_result.max_depth_reached);

    // Verify invariants
    println!();
    println!("Verifying system invariants:");
    let invariant_results = pipeline.verify_invariants();
    for result in &invariant_results {
        let status = if result.holds { "✓" } else { "✗" };
        println!("  {} {}", status, result.invariant_id);
    }
    println!();
}

fn demo_phase_5_attack_response() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 5: Attack Detection and Response                       │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    let lifecycle = IntegratedEpistemicLifecycle::new(EpistemicLifecycleConfig::default());
    let config = AttackResponseConfig::default();
    let mut attack_system = IntegratedAttackResponse::new(config);

    // Create a good agent
    let mut good_agent = lifecycle.create_agent("did:mycelix:sponsor:good", AgentClass::Supervised);

    // Simulate good behavior
    println!("Monitoring good agent behavior:");
    for _ in 0..5 {
        let _ = lifecycle.process_output(
            &mut good_agent,
            OutputContent::Text("Valid output".to_string()),
        );
    }

    let response = attack_system.process_behavior(&mut good_agent);
    match response {
        Some(r) => println!("  Response: {:?}", r),
        None => println!("  ✓ No attacks detected - agent behaving normally"),
    }
    println!(
        "  Quarantined: {}",
        attack_system.is_quarantined(good_agent.agent_id.as_str())
    );

    // Create an agent with suspicious patterns
    println!();
    println!("Monitoring suspicious agent (simulated gaming behavior):");
    let mut bad_agent = lifecycle.create_agent("did:mycelix:sponsor:bad", AgentClass::Autonomous);

    // Simulate gaming behavior by producing many rapid outputs
    for i in 0..100 {
        bad_agent.actions_this_hour = i * 10; // Rapidly increasing action count
        let _ =
            lifecycle.process_output(&mut bad_agent, OutputContent::Text(format!("Spam {}", i)));
    }

    let response = attack_system.process_behavior(&mut bad_agent);
    match response {
        Some(r) => {
            println!("  ⚠ Attack detected!");
            println!("    ├─ Type: {}", r.attack_type);
            println!("    ├─ Confidence: {:.1}%", r.confidence * 100.0);
            println!("    └─ Actions:");
            for action in &r.actions_taken {
                println!("       └─ {:?}", action);
            }
        }
        None => println!("  No attacks detected"),
    }
    println!(
        "  Quarantined: {}",
        attack_system.is_quarantined(bad_agent.agent_id.as_str())
    );
    println!();
}

fn demo_phase_6_privacy_analytics() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 6: Privacy-Preserving Analytics                        │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    let config = PrivacyAnalyticsConfig {
        dp: DPConfig {
            epsilon: 5.0,
            delta: 1e-6,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut analytics = IntegratedPrivacyAnalytics::new(config);

    println!("Computing differentially private trust statistics:");
    println!();

    // Simulate trust scores from multiple agents
    let trust_scores = vec![0.45, 0.52, 0.67, 0.71, 0.58, 0.62, 0.49, 0.73, 0.55, 0.60];
    let phi_values = vec![0.72, 0.68, 0.75, 0.70, 0.69];

    let true_mean = trust_scores.iter().sum::<f64>() / trust_scores.len() as f64;

    println!("  Raw data (normally hidden):");
    println!("    ├─ Trust scores: {:?}", trust_scores);
    println!("    └─ Phi values: {:?}", phi_values);
    println!();

    match analytics.analyze_and_display(&trust_scores, &phi_values, 0.1) {
        Ok(result) => {
            println!("  Private statistics (ε-DP protected):");
            println!(
                "    ├─ Private Mean: {:.3} (true: {:.3})",
                result.distribution.mean, true_mean
            );
            println!("    ├─ Private Median: {:.3}", result.distribution.median);
            println!("    ├─ Private Count: {:.0}", result.distribution.count);
            println!(
                "    ├─ Privacy Budget Remaining: (ε={:.2}, δ={:.2e})",
                result.remaining_budget.0, result.remaining_budget.1
            );
            println!();
            println!("  Dashboard Live Metrics:");
            println!(
                "    ├─ Active Agents: {}",
                result.live_metrics.active_agents
            );
            println!(
                "    ├─ Average Trust: {:.3}",
                result.live_metrics.average_trust
            );
            println!(
                "    ├─ Collective Phi: {:.3}",
                result.live_metrics.collective_phi
            );
            println!(
                "    └─ Byzantine Threat: {:.3}",
                result.live_metrics.byzantine_threat
            );
        }
        Err(e) => println!("  Error: {:?}", e),
    }
    println!();
}

fn demo_phase_7_full_scenario() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 7: Full Integrated Scenario                            │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!();

    println!("Simulating a complete agent lifecycle scenario:");
    println!();

    // Initialize all systems
    let lifecycle = IntegratedEpistemicLifecycle::new(EpistemicLifecycleConfig::default());
    let mut pipeline = IntegratedTrustPipeline::new(TrustPipelineConfig::default());
    let mut attack_system = IntegratedAttackResponse::new(AttackResponseConfig::default());
    let mut analytics = IntegratedPrivacyAnalytics::new(PrivacyAnalyticsConfig {
        dp: DPConfig {
            epsilon: 5.0,
            delta: 1e-6,
            ..Default::default()
        },
        ..Default::default()
    });

    // Step 1: Create agents
    println!("Step 1: Create 3 agents with different roles");
    let mut agents = vec![];
    let classes = [
        AgentClass::Supervised,
        AgentClass::Assistive,
        AgentClass::Autonomous,
    ];
    for (i, class) in classes.iter().enumerate() {
        let agent = lifecycle.create_agent(&format!("did:mycelix:sponsor:{}", i), class.clone());
        println!(
            "  Agent {}: {:?} (trust: {:.3})",
            i,
            class,
            agent.k_vector.trust_score()
        );
        agents.push(agent);
    }
    println!();

    // Step 2: Agents produce outputs
    println!("Step 2: Agents produce epistemic outputs");
    for (i, agent) in agents.iter_mut().enumerate() {
        for j in 0..3 {
            let _ = lifecycle.process_output(
                agent,
                OutputContent::Text(format!("Agent {} output {}", i, j)),
            );
        }
        println!(
            "  Agent {}: {} outputs, trust={:.3}",
            i,
            agent.epistemic_stats.total_outputs,
            agent.k_vector.trust_score()
        );
    }
    println!();

    // Step 3: Register in trust pipeline and create attestations
    println!("Step 3: Build trust network");
    for agent in agents.iter() {
        pipeline.register_agent(agent.clone());
    }

    let agent_ids: Vec<String> = pipeline.agents().keys().cloned().collect();
    for i in 0..agent_ids.len().saturating_sub(1) {
        let _ = pipeline.process_attestation(&agent_ids[i], &agent_ids[i + 1], 0.7);
    }
    println!(
        "  Trust attestations: {} -> {} -> {}",
        &agent_ids[0][..8.min(agent_ids[0].len())],
        &agent_ids[1][..8.min(agent_ids[1].len())],
        &agent_ids[2][..8.min(agent_ids[2].len())]
    );
    println!();

    // Step 4: Check for attacks
    println!("Step 4: Security monitoring");
    for agent in agents.iter_mut() {
        let response = attack_system.process_behavior(agent);
        let status = if response.is_some() {
            "⚠ ATTACK"
        } else {
            "✓ OK"
        };
        println!(
            "  {}: {}",
            &agent.agent_id.as_str()[..8.min(agent.agent_id.as_str().len())],
            status
        );
    }
    println!();

    // Step 5: Privacy-preserving analytics
    println!("Step 5: Privacy-preserving analytics");
    let trust_scores: Vec<f64> = agents
        .iter()
        .map(|a| a.k_vector.trust_score() as f64)
        .collect();
    let phi_values: Vec<f64> = agents
        .iter()
        .map(|a| a.k_vector.k_coherence as f64)
        .collect();

    if let Ok(result) = analytics.analyze_and_display(&trust_scores, &phi_values, 0.1) {
        println!("  Private mean trust: {:.3}", result.distribution.mean);
        println!(
            "  Privacy budget: ε={:.2} remaining",
            result.remaining_budget.0
        );
    }
    println!();

    // Step 6: Verify invariants
    println!("Step 6: System verification");
    let invariants = pipeline.verify_invariants();
    let all_hold = invariants.iter().all(|r| r.holds);
    println!(
        "  All invariants hold: {}",
        if all_hold { "✓ YES" } else { "✗ NO" }
    );
    println!();

    // Final summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    SCENARIO COMPLETE                          ");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Summary:");
    println!("    ├─ Agents Created: {}", agents.len());
    println!(
        "    ├─ Total Outputs: {}",
        agents
            .iter()
            .map(|a| a.epistemic_stats.total_outputs)
            .sum::<u32>()
    );
    println!(
        "    ├─ Trust Network: {} attestations",
        agent_ids.len().saturating_sub(1)
    );
    println!(
        "    ├─ Attacks Detected: {}",
        attack_system.responses().len()
    );
    println!(
        "    └─ Invariants Valid: {}/{}",
        invariants.iter().filter(|r| r.holds).count(),
        invariants.len()
    );
    println!();
}
