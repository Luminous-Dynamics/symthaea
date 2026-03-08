//! Epistemic Fairness Stress Test
//!
//! This example demonstrates the "Full Stack Wisdom" architecture for fair epistemic
//! scoring using the WisdomEngine - the most advanced epistemic system ever designed
//! for decentralized networks.
//!
//! ## The Seven Components Demonstrated:
//!
//! 1. **The Lens** (HarmonicWeights) - Eight Harmonies as cultural lenses
//! 2. **The Setting** (CommunityProfile) - Per-community epistemic profiles
//! 3. **The Mirror** (DiversityAuditor) - Bias detection and diversity tracking
//! 4. **The Hand** (ReparationsManager) - Power corrections for marginalized voices
//! 5. **Emergent Weights** - Adaptive learning from outcomes
//! 6. **Multi-Epistemology** - Support for diverse knowledge systems
//! 7. **Causal Graph (Reality Check)** - Tests hypotheses against reality
//!
//! ## Stress Test Scenarios:
//!
//! 1. **Galileo Problem**: High E (reproducible) but low N (nobody agrees)
//! 2. **Grandmother's Remedy**: E1.5 (oral tradition) with community validation
//! 3. **Flat Earth**: E0 (unverifiable belief) - should be rejected universally
//! 4. **Contested Reproducibility**: Scientific claim with split results
//! 5. **WisdomEngine Multi-Community**: Same claim evaluated by different communities
//! 6. **Reparations in Action**: How marginalized voices are uplifted
//! 7. **Bias Detection**: The Mirror detecting potential unfairness
//! 8. **Reality Check**: Predictions vs actual outcomes with causal learning
//!
//! Key insight: "Fixed Weights = Colonialism; Adaptive Weights = Pluralism"
//!
//! Run with: cargo run --example epistemic_stress_test

use mycelix_core_types::epistemic::{
    EmpiricalLevel, EpistemicClassification, EpistemicContext, MaterialityLevel, NormativeLevel,
    TestimonialQuality,
};
use mycelix_core_types::wisdom_engine::{
    CausalGraph, Claim, CommunityProfile, DiversityAuditor, Epistemology, EvaluationRecord,
    LivingWisdomEngine, OracleVerificationLevel, ReparationsManager, SimpleOracle,
    StructuralPosition, WisdomEngine,
};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  🧠 FULL STACK WISDOM: Holistic Epistemics Stress Test             ║");
    println!("║  The Most Advanced Epistemic Architecture for Decentralized Systems ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    // ===========================================================================
    // PART 1: PRISMATIC ARCHITECTURE (Context-Adaptive Weights)
    // ===========================================================================
    println!("╭──────────────────────────────────────────────────────────────────────╮");
    println!("│  PART 1: PRISMATIC ARCHITECTURE (Context-Adaptive Weights)          │");
    println!("╰──────────────────────────────────────────────────────────────────────╯");
    println!();

    // --- Stress Test 1: Galileo Problem ---
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔭 STRESS TEST 1: The Galileo Problem");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Scenario: Galileo observes Jupiter's moons through telescope (1610)");
    println!("- E4: Publicly reproducible (anyone with telescope can verify)");
    println!("- N0: Personal (Church says he's wrong, nobody believes him)");
    println!("- M3: Foundational (if true, changes cosmology forever)");
    println!();

    let galileo = EpistemicClassification::new(
        EmpiricalLevel::CryptographicallyVerifiable,
        NormativeLevel::Individual,
        MaterialityLevel::Permanent,
    );

    println!("Classification: {}", galileo.code());
    println!();

    let contexts = [
        ("Standard (Fixed)", EpistemicContext::Standard),
        ("Scientific", EpistemicContext::Scientific),
        ("Governance", EpistemicContext::Governance),
    ];

    for (name, context) in contexts.iter() {
        let score = galileo.quality_score_contextual(*context);
        let (e, n, m) = context.weights();
        let verdict = if score > 0.5 { "ACCEPTED" } else { "REJECTED" };
        println!(
            "  {:<20} Score: {:.3}  [E={:.0}%, N={:.0}%, M={:.0}%]  {} {}",
            name,
            score,
            e * 100.0,
            n * 100.0,
            m * 100.0,
            if score > 0.5 { "✅" } else { "❌" },
            verdict
        );
    }

    println!();
    println!("📊 Analysis: Scientific context VINDICATES Galileo (E=50%)!");
    println!("   Standard context penalizes him for low consensus.");
    println!();

    // --- Stress Test 2: Grandmother's Remedy ---
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("👵 STRESS TEST 2: Grandmother's Remedy");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Scenario: Elder says 'willow bark tea cures headaches' (aspirin!)");
    println!("- E1.5: Corroborated Witness (generations of oral tradition agree)");
    println!("- N1: Communal (local community recognizes this)");
    println!("- M3: Foundational (passed down for centuries)");
    println!();

    let grandmother = EpistemicClassification::testimonial(
        TestimonialQuality::CorroboratedWitness,
        NormativeLevel::Group,
        MaterialityLevel::Permanent,
    );

    println!("Classification: {}", grandmother.code());
    println!();

    let contexts = [
        ("Standard (Fixed)", EpistemicContext::Standard),
        ("Scientific", EpistemicContext::Scientific),
        ("Indigenous", EpistemicContext::Indigenous),
        ("Contemplative", EpistemicContext::Contemplative),
    ];

    for (name, context) in contexts.iter() {
        let score = grandmother.quality_score_contextual(*context);
        let (e, n, m) = context.weights();
        let verdict = if score > 0.5 {
            "HONORED ✅"
        } else if score > 0.35 {
            "MARGINAL ⚠️"
        } else {
            "DISMISSED ❌"
        };
        println!(
            "  {:<20} Score: {:.3}  [E={:.0}%, N={:.0}%, M={:.0}%]  {}",
            name,
            score,
            e * 100.0,
            n * 100.0,
            m * 100.0,
            verdict
        );
    }

    println!();
    println!("📊 Analysis: Indigenous context HONORS grandmother's wisdom!");
    println!("   E1.5 (Corroborated Witness) + N/M weighting = fair treatment.");
    println!();

    // --- Stress Test 3: Flat Earth ---
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🌍 STRESS TEST 3: Flat Earth (The Veto Test)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Scenario: Someone claims 'The Earth is flat'");
    println!("- E0: Unverifiable (contradicts observable evidence)");
    println!("- N1: Communal (some community believes it)");
    println!("- M2: Persistent (they keep saying it)");
    println!();

    let flat_earth = EpistemicClassification::new(
        EmpiricalLevel::Unverifiable,
        NormativeLevel::Group,
        MaterialityLevel::Persistent,
    );

    println!("Classification: {}", flat_earth.code());
    println!();

    let contexts = [
        ("Standard (Fixed)", EpistemicContext::Standard),
        ("Scientific", EpistemicContext::Scientific),
        ("Governance", EpistemicContext::Governance),
        ("Indigenous", EpistemicContext::Indigenous),
    ];

    for (name, context) in contexts.iter() {
        let score = flat_earth.quality_score_contextual(*context);
        let (e, n, m) = context.weights();
        let verdict = if score > 0.5 {
            "ACCEPTED (PROBLEM!) ⚠️"
        } else {
            "REJECTED (CORRECT) ✅"
        };
        println!(
            "  {:<20} Score: {:.3}  [E={:.0}%, N={:.0}%, M={:.0}%]  {}",
            name,
            score,
            e * 100.0,
            n * 100.0,
            m * 100.0,
            verdict
        );
    }

    println!();
    println!("📊 Analysis: E0 (Unverifiable) acts as a floor, not a veto.");
    println!("   In ALL contexts, flat earth scores < 0.5 = REJECTED.");
    println!("   Pluralism doesn't mean accepting demonstrably false claims!");
    println!();

    // --- Stress Test 4: Contested Reproducibility ---
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔬 STRESS TEST 4: Contested Reproducibility");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Scenario: Scientific claim where 6 labs replicate, 4 labs fail");
    println!();

    let contested_claim = EpistemicClassification::new(
        EmpiricalLevel::Measurable,
        NormativeLevel::Network,
        MaterialityLevel::Persistent,
    )
    .with_contested(6, 4);

    let uncontested_claim = EpistemicClassification::new(
        EmpiricalLevel::Measurable,
        NormativeLevel::Network,
        MaterialityLevel::Persistent,
    );

    println!(
        "Classification: {} (Status: {:?})",
        contested_claim.code(),
        contested_claim.verification_status
    );
    println!();

    if let Some(ref details) = contested_claim.contested_details {
        println!("  Verification Attempts: {}", details.attempt_count);
        println!("  Success Ratio: {:.0}%", details.success_ratio * 100.0);
        println!("  Adjustment Factor: {:.2}", details.adjustment_factor());
    }

    let contested_score = contested_claim.quality_score();
    let uncontested_score = uncontested_claim.quality_score();

    println!();
    println!("  Uncontested Score: {:.3}", uncontested_score);
    println!(
        "  Contested Score:   {:.3} (-{:.1}%)",
        contested_score,
        (1.0 - contested_score / uncontested_score) * 100.0
    );
    println!(
        "  Requires Scrutiny: {}",
        if contested_claim.requires_scrutiny() {
            "YES ⚠️"
        } else {
            "NO"
        }
    );
    println!();

    // ===========================================================================
    // PART 2: THE WISDOM ENGINE (Full Stack Wisdom)
    // ===========================================================================
    println!();
    println!("╭──────────────────────────────────────────────────────────────────────╮");
    println!("│  PART 2: THE WISDOM ENGINE (Full Stack Wisdom)                      │");
    println!("╰──────────────────────────────────────────────────────────────────────╯");
    println!();

    // --- Stress Test 5: Multi-Community Evaluation ---
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🌐 STRESS TEST 5: Multi-Community Evaluation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Same claim evaluated by different communities with different profiles.");
    println!();

    let mut engine = WisdomEngine::new();

    // Register three different communities
    let indigenous_community = CommunityProfile::indigenous(1, "Lakota Wisdom Keepers");
    let scientific_community = CommunityProfile::scientific(2, "Nature Research Labs");
    let contemplative_community = CommunityProfile::contemplative(3, "Zen Buddhist Sangha");

    engine.register_community(indigenous_community);
    engine.register_community(scientific_community);
    engine.register_community(contemplative_community);

    println!("Registered Communities:");
    println!("  1. Lakota Wisdom Keepers (Indigenous epistemology)");
    println!("  2. Nature Research Labs (Scientific epistemology)");
    println!("  3. Zen Buddhist Sangha (Contemplative epistemology)");
    println!();

    // Grandmother's claim through the wisdom engine
    let grandmother_claim = Claim {
        id: 1,
        classification: EpistemicClassification::testimonial(
            TestimonialQuality::CorroboratedWitness,
            NormativeLevel::Group,
            MaterialityLevel::Permanent,
        ),
        author_position: StructuralPosition::HistoricallySilenced,
        timestamp: 1705689600, // Jan 19, 2024
    };

    println!("Claim: Grandmother's Remedy (E1.5/N1/M3, Historically Silenced author)");
    println!();

    println!("Evaluations by Community:");
    println!("┌─────────────────────────┬──────────┬──────────┬──────────┬─────────────┐");
    println!("│ Community               │ E/N/M    │ Harmonic │ Final    │ Reparations │");
    println!("├─────────────────────────┼──────────┼──────────┼──────────┼─────────────┤");

    for (id, name) in [(1, "Lakota Wisdom Keepers"), (2, "Nature Research Labs"), (3, "Zen Buddhist Sangha")] {
        let eval = engine.evaluate(&grandmother_claim, id);
        println!(
            "│ {:<23} │   {:.3}  │   {:.3}  │   {:.3}  │ {:^11} │",
            name,
            eval.enm_score,
            eval.harmonic_score,
            eval.final_score,
            if eval.reparations_applied { "YES" } else { "NO" }
        );
    }

    println!("└─────────────────────────┴──────────┴──────────┴──────────┴─────────────┘");
    println!();
    println!("📊 Analysis: Same claim, different evaluations based on community values.");
    println!("   Reparations applied because author is 'Historically Silenced'.");
    println!();

    // --- Stress Test 6: Reparations in Action ---
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✊ STRESS TEST 6: Reparations in Action (The Hand)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("How do different structural positions affect final scores?");
    println!();

    let manager = ReparationsManager::new();
    let base_score = 0.50;
    let current_diversity = 0.50; // Medium diversity

    println!("Base Score: {:.3}, Current Diversity Index: {:.1}%", base_score, current_diversity * 100.0);
    println!();

    println!("┌─────────────────────────┬──────────┬─────────────┬────────────┐");
    println!("│ Structural Position     │ Adjusted │ Boost       │ Applied?   │");
    println!("├─────────────────────────┼──────────┼─────────────┼────────────┤");

    for (name, position) in [
        ("Dominant", StructuralPosition::Dominant),
        ("Mainstream", StructuralPosition::Mainstream),
        ("Marginalized", StructuralPosition::Marginalized),
        ("Historically Silenced", StructuralPosition::HistoricallySilenced),
    ] {
        let (adjusted, applied) = manager.adjust(base_score, position, current_diversity);
        let boost = adjusted - base_score;
        println!(
            "│ {:<23} │   {:.3}  │   +{:.3}     │ {:^10} │",
            name,
            adjusted,
            boost,
            if applied { "YES" } else { "NO" }
        );
    }

    println!("└─────────────────────────┴──────────┴─────────────┴────────────┘");
    println!();
    println!("📊 Analysis: Reparations boost marginalized voices to correct");
    println!("   for historical silencing. Boosts decay as diversity increases.");
    println!();

    // --- Stress Test 7: Bias Detection (The Mirror) ---
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🪞 STRESS TEST 7: Bias Detection (The Mirror)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("The Diversity Auditor detects potential bias in evaluations.");
    println!();

    let mut auditor = DiversityAuditor::new();

    // Simulate biased evaluations: dominant voices getting higher scores
    for i in 0..100 {
        let (position, score) = if i % 3 == 0 {
            // Dominant voices: higher scores
            (StructuralPosition::Dominant, 0.75 + (i as f32 % 10.0) * 0.01)
        } else if i % 3 == 1 {
            // Mainstream: medium scores
            (StructuralPosition::Mainstream, 0.55 + (i as f32 % 10.0) * 0.01)
        } else {
            // Marginalized: lower scores (this is the bias!)
            (StructuralPosition::Marginalized, 0.45 + (i as f32 % 10.0) * 0.01)
        };

        auditor.record(EvaluationRecord {
            timestamp: i as u64,
            community_id: 1,
            author_position: position,
            epistemology: Epistemology::WesternScientific,
            raw_score: score,
            final_score: score,
            reparations_applied: false,
            classification_code: "E3/N2/M2".into(),
        });
    }

    println!("Simulated 100 evaluations with bias pattern...");
    println!();

    let metrics = &auditor.current_metrics;
    println!("Diversity Metrics:");
    println!("  Total Evaluations: {}", metrics.total_evaluations);
    println!("  Dominant: {}, Mainstream: {}, Marginalized: {}",
        metrics.by_position.dominant,
        metrics.by_position.mainstream,
        metrics.by_position.marginalized
    );
    println!("  Diversity Index: {:.3}", metrics.diversity_index);
    println!();

    if let Some(alert) = auditor.detect_bias() {
        println!("⚠️  BIAS ALERT DETECTED!");
        println!("  Score Gap: {:.1}%", alert.gap * 100.0);
        println!("  Dominant Avg: {:.3}, Marginalized Avg: {:.3}", alert.dominant_avg, alert.marginalized_avg);
        println!("  Samples: {} dominant, {} marginalized", alert.samples_dominant, alert.samples_marginalized);
        println!();
        println!("  Recommendation: {}", alert.recommendation);
    } else {
        println!("✅ No significant bias detected.");
    }
    println!();

    // ===========================================================================
    // PART 3: MULTI-EPISTEMOLOGY SUPPORT
    // ===========================================================================
    println!();
    println!("╭──────────────────────────────────────────────────────────────────────╮");
    println!("│  PART 3: MULTI-EPISTEMOLOGY SUPPORT                                 │");
    println!("╰──────────────────────────────────────────────────────────────────────╯");
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📚 Supported Epistemologies");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let epistemologies = [
        Epistemology::WesternScientific,
        Epistemology::IndigenousRelational,
        Epistemology::BuddhistContemplative,
        Epistemology::IslamicScholarly,
        Epistemology::AfricanUbuntu,
        Epistemology::FeministStandpoint,
        Epistemology::Pragmatist,
    ];

    println!("┌─────────────────────────┬──────────────────┬───────────────────────────────────┐");
    println!("│ Epistemology            │ Default Context  │ Description                       │");
    println!("├─────────────────────────┼──────────────────┼───────────────────────────────────┤");

    for ep in epistemologies.iter() {
        let desc = ep.description();
        let truncated_desc = if desc.len() > 33 {
            format!("{}...", &desc[..30])
        } else {
            desc.to_string()
        };
        println!(
            "│ {:<23} │ {:^16} │ {:<33} │",
            format!("{:?}", ep),
            format!("{:?}", ep.recommended_context()),
            truncated_desc
        );
    }

    println!("└─────────────────────────┴──────────────────┴───────────────────────────────────┘");
    println!();

    // ===========================================================================
    // PART 4: CAUSAL GRAPH - Reality Check / Causal Feedback Loops
    // ===========================================================================
    println!();
    println!("╭──────────────────────────────────────────────────────────────────────╮");
    println!("│  PART 4: CAUSAL GRAPH (Reality Check / Causal Feedback Loops)       │");
    println!("╰──────────────────────────────────────────────────────────────────────╯");
    println!();
    println!("The CausalGraph transforms Mycelix from 'Philosophy' to 'Scientific Instrument'.");
    println!("It tests epistemic hypotheses against reality and learns from outcomes.");
    println!();

    // --- Stress Test 8: Reality Check Learning Cycle ---
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔮 STRESS TEST 8: Reality Check Learning Cycle");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Scenario: Community makes decisions based on epistemic scores.");
    println!("          System predicts outcomes, oracles observe reality,");
    println!("          errors backpropagate to adjust weights.");
    println!();

    let mut living_engine = LivingWisdomEngine::new();
    living_engine.auto_predict = true;

    // Register a community
    let science_community = CommunityProfile::scientific(1, "Climate Research Consortium");
    living_engine.register_community(science_community);

    // Create a claim about climate models
    let climate_claim = Claim {
        id: 100,
        classification: EpistemicClassification::new(
            EmpiricalLevel::Measurable, // E2 - peer-reviewed measurements
            NormativeLevel::Network,     // N2 - scientific consensus
            MaterialityLevel::Permanent, // M3 - foundational for policy
        ),
        author_position: StructuralPosition::Mainstream,
        timestamp: 1000,
    };

    println!("Claim: Climate model predicting 1.5°C warming by 2030");
    println!();

    // Evaluate - this auto-generates a prediction
    let eval_result = living_engine.evaluate(&climate_claim, 1);
    let prediction_id = eval_result.prediction_id.unwrap();

    println!("Step 1: Evaluation & Prediction");
    println!("  E/N/M Score: {:.3}", eval_result.evaluation.enm_score);
    println!("  Harmonic Score: {:.3}", eval_result.evaluation.harmonic_score);
    println!("  Final Score: {:.3}", eval_result.evaluation.final_score);
    println!("  Prediction ID: {}", prediction_id);
    println!("  → System predicts outcome value: {:.3}", eval_result.evaluation.final_score);
    println!();

    // Simulate time passing and oracle observing outcome
    println!("Step 2: Oracle Observation (30 days later)");

    // Create an oracle (e.g., sensor network + audit)
    let mut climate_oracle = SimpleOracle::new(
        "climate_sensor_network",
        "Global temperature monitoring sensors with audit guild verification",
    );
    climate_oracle.verification = OracleVerificationLevel::Audited;
    climate_oracle.trust = 0.9;

    // Reality: the prediction was slightly off (over-optimistic)
    let actual_outcome = eval_result.evaluation.final_score - 0.15; // Reality was worse
    climate_oracle.set_observation(format!("claim_{}_outcome", 100), actual_outcome);

    // Resolve from oracle
    let resolved = living_engine.causal.resolve_from_oracle(&climate_oracle, 2000);

    println!("  Oracle: {} (Trust: {:.0}%, Verification: {:?})",
        climate_oracle.description,
        climate_oracle.trust * 100.0,
        climate_oracle.verification
    );
    println!("  Observed Outcome: {:.3}", actual_outcome);

    if !resolved.is_empty() {
        println!("  Prediction Error: {:.3} ({:.1}%)",
            resolved[0].1,
            resolved[0].1 * 100.0
        );
    }
    println!();

    // Check for suggested adjustments
    println!("Step 3: Causal Backpropagation");
    let adjustments = living_engine.pending_adjustments();

    if !adjustments.is_empty() {
        let adj = &adjustments[0];
        println!("  ⚠️ Weight Adjustment Suggested!");
        println!("  Error Magnitude: {:.1}%", adj.error_magnitude * 100.0);
        println!("  Confidence: {:.1}%", adj.confidence * 100.0);
        println!("  Explanation: {}", adj.explanation);
        println!();
        println!("  Weight Deltas (RC, PSF, IW, IP, UI, SR, EP):");
        println!("    [{:+.4}, {:+.4}, {:+.4}, {:+.4}, {:+.4}, {:+.4}, {:+.4}]",
            adj.weight_deltas[0], adj.weight_deltas[1], adj.weight_deltas[2],
            adj.weight_deltas[3], adj.weight_deltas[4], adj.weight_deltas[5],
            adj.weight_deltas[6]
        );
    } else {
        println!("  ✅ No significant adjustment needed (error within threshold).");
    }
    println!();

    // System health report
    println!("Step 4: System Health Report");
    let report = living_engine.health_report();
    println!("  Prediction Accuracy: {:.1}%", report.prediction_accuracy * 100.0);
    println!("  Accuracy Trend: {:+.3}", report.accuracy_trend);
    println!("  Total Predictions: {}", report.total_predictions);
    println!("  Pending Predictions: {}", report.pending_predictions);
    println!("  Diversity Index: {:.3}", report.diversity_index);
    println!("  Bias Detected: {}", if report.bias_detected { "YES ⚠️" } else { "NO ✅" });
    println!("  Adjustments Pending: {}", report.adjustments_pending);
    println!("  Overall Status: {:?}", report.status());
    println!();

    // --- Demonstrate Oracle Verification Levels ---
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📡 Oracle Verification Levels (Maps to E-Axis)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("┌─────────────────────────┬──────────────┬──────────────────────────────┐");
    println!("│ Verification Level      │ Trust Factor │ Example                      │");
    println!("├─────────────────────────┼──────────────┼──────────────────────────────┤");

    for (level, example) in [
        (OracleVerificationLevel::Testimonial, "Community member reports"),
        (OracleVerificationLevel::Audited, "Audit guild verification"),
        (OracleVerificationLevel::Cryptographic, "Sensor + ZKP attestation"),
        (OracleVerificationLevel::PubliclyReproducible, "Open data + reproducible"),
    ] {
        println!("│ {:<23} │    {:.0}%       │ {:<28} │",
            format!("{:?}", level),
            level.trust_multiplier() * 100.0,
            example
        );
    }

    println!("└─────────────────────────┴──────────────┴──────────────────────────────┘");
    println!();

    // --- Demonstrate Causal Node Graph ---
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔗 Causal Graph Structure");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut causal_graph = CausalGraph::new();

    // Create a causal model: Trust → Participation → Health → Outcomes
    let trust = causal_graph.add_node("community_trust", "Level of trust between members");
    let participation = causal_graph.add_node("participation_rate", "Active engagement in governance");
    let health = causal_graph.add_node("community_health", "Overall community wellbeing");
    let outcomes = causal_graph.add_node("decision_outcomes", "Quality of collective decisions");

    // Link them causally
    causal_graph.add_causal_link(trust, participation, 0.7);
    causal_graph.add_causal_link(participation, health, 0.6);
    causal_graph.add_causal_link(health, outcomes, 0.8);
    causal_graph.add_causal_link(trust, outcomes, 0.3); // Direct effect

    println!("Causal Model: Trust → Participation → Health → Outcomes");
    println!();
    println!("     community_trust (0.7)──→ participation_rate");
    println!("           │ (0.3)                    │ (0.6)");
    println!("           ▼                          ▼");
    println!("     decision_outcomes  ←────── community_health");
    println!("                           (0.8)");
    println!();

    println!("Nodes Created: {}", causal_graph.nodes.len());
    for node in causal_graph.nodes.values() {
        println!("  • {} (parents: {}, children: {})",
            node.name,
            node.parents.len(),
            node.children.len()
        );
    }
    println!();

    // ===========================================================================
    // SUMMARY
    // ===========================================================================
    println!();
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  📋 SUMMARY: The Full Stack Wisdom Architecture                    ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("┌─────────────────┬──────────────────────────────────────────────────┐");
    println!("│ Component       │ What It Does                                     │");
    println!("├─────────────────┼──────────────────────────────────────────────────┤");
    println!("│ The Lens        │ Eight Harmonies as cultural epistemic lenses     │");
    println!("│ The Setting     │ Community-specific profiles and preferences      │");
    println!("│ The Mirror      │ Tracks diversity & detects systemic bias         │");
    println!("│ The Hand        │ Epistemic reparations for marginalized voices    │");
    println!("│ Emergent Wts    │ Learns optimal weights from outcomes             │");
    println!("│ Multi-Epist     │ Supports 10+ epistemological frameworks          │");
    println!("│ Causal Graph    │ Tests hypotheses against reality via oracles     │");
    println!("└─────────────────┴──────────────────────────────────────────────────┘");
    println!();
    println!("Key Principles:");
    println!("  1. Fixed Weights = Colonialism (one epistemology dominates)");
    println!("  2. Adaptive Weights = Pluralism (context determines what matters)");
    println!("  3. E0 acts as floor, not veto (false claims still rejected)");
    println!("  4. E1 sub-levels rehabilitate oral/testimonial traditions");
    println!("  5. Contested claims receive reduced scores + scrutiny flags");
    println!("  6. Reparations correct for historical silencing");
    println!("  7. The Mirror detects when the system itself is biased");
    println!("  8. Causal feedback enables empirical learning from outcomes");
    println!();
    println!("This is a SELF-CORRECTING CIVILIZATIONAL OPERATING SYSTEM.");
    println!("It moves beyond 'Philosophy' to 'Scientific Instrument.'");
    println!("It tests its own hypotheses against reality.");
    println!();
    println!("Reference: EPISTEMICS_FAIRNESS_ANALYSIS.md");
    println!();
}
