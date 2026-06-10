// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Rehabilitation Pathway: End-to-End Integration Example
//!
//! Demonstrates the full Symthaea-powered recovery journey:
//! 1. Chronic substance use simulation (neuromodulator bath)
//! 2. Withdrawal initiation + medication-assisted treatment
//! 3. Clinical formulation with neurochemical perpetuating factors
//! 4. Affect regulation adapting to withdrawal severity
//! 5. Recovery timeline prediction
//! 6. Progress tracking through recovery phases
//!
//! Run: `cargo run --example rehabilitation_pathway`
//!
//! Science:
//! - Koob & Le Moal (2001) — allostatic model of drug addiction
//! - Volkow et al. (2016) — neurobiologic advances from brain disease model
//! - Marlatt & Gordon (1985) — relapse prevention model
//! - Stanley & Brown (2012) — safety planning intervention

use symthaea_neuromodulators::substance_profiles::*;
use symthaea_neuromodulators::{NeuromodulatorBath, NeuromodulatorInputs};

/// Create a zero-valued NeuromodulatorInputs (no external signals).
fn quiet_inputs() -> NeuromodulatorInputs {
    NeuromodulatorInputs {
        prediction_error: 0.0,
        surprise: false,
        reward_signal: 0.0,
        coherence: 0.3,
        arousal: 0.3,
        binding_strength: 0.3,
        epistemic_confidence: 0.3,
        flow_active: false,
        consciousness_level: None,
        moral_signal: None,
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SYMTHAEA REHABILITATION PATHWAY — End-to-End Integration");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // ── Phase 0: Baseline ──────────────────────────────────────────────
    let mut bath = NeuromodulatorBath::default();
    println!("Phase 0: HEALTHY BASELINE");
    print_neuromod_state(&bath, "  ");
    println!();

    // ── Phase 1: Chronic Opioid Use ────────────────────────────────────
    let opioid = opioid_profile();
    println!("Phase 1: CHRONIC OPIOID USE (simulating 200 cycles of exposure)");
    println!("  Substance: {}", opioid.name);
    println!("  Primary effects: DA↑↑ (0.8), Endocannabinoid↑↑ (0.5)");
    println!("  Secondary: 5-HT↓ (0.3), NE↓ (0.2), GABA↑ (0.3)");
    println!(
        "  Tolerance acceleration: {}x",
        opioid.tolerance_acceleration
    );
    println!();

    // Simulate chronic use (fast-forward tolerance development)
    bath.simulate_chronic_use(&opioid, 200);

    // Apply a few acute doses to show active intoxication
    for _ in 0..5 {
        bath.apply_substance_profile(&opioid, 0.7);
        // Run a few reuptake cycles
        let mut inputs = quiet_inputs();
        inputs.reward_signal = 0.8; // Drug-induced reward
        bath.update(&inputs);
    }

    println!("  After chronic use + acute intoxication:");
    print_neuromod_state(&bath, "    ");
    println!("    Allostatic load: {:.3}", bath.allostatic_load);
    println!();

    // ── Phase 2: Withdrawal ────────────────────────────────────────────
    println!("Phase 2: WITHDRAWAL INITIATION (cessation of opioid use)");
    bath.initiate_withdrawal(&opioid);
    println!("  Withdrawal initiated. Running 20 reuptake cycles...");

    for _ in 0..20 {
        let inputs = quiet_inputs();
        bath.update(&inputs);
    }

    println!("  After 20 cycles of withdrawal:");
    print_neuromod_state(&bath, "    ");
    println!("    In withdrawal: {}", bath.is_in_withdrawal());
    println!(
        "    Total withdrawal cycles remaining: {}",
        bath.total_withdrawal_cycles()
    );
    println!("    Allostatic load: {:.3}", bath.allostatic_load);
    println!();

    // ── Phase 3: MAT (Buprenorphine) ──────────────────────────────────
    println!("Phase 3: MEDICATION-ASSISTED TREATMENT (Buprenorphine)");
    println!("  Buprenorphine: partial mu-opioid agonist");
    println!("  Mechanism: mild DA elevation (ceiling effect), prevents withdrawal");
    println!("  Science: Mattick et al. (2014)");
    match bath.apply_mat_protocol(&MatProtocol::Buprenorphine) {
        Ok(()) => {
            println!("  Buprenorphine initiated successfully (patient in sufficient withdrawal)")
        }
        Err(risk) => println!("  WARNING: {}", risk),
    }

    // Run 30 cycles with MAT
    for _ in 0..30 {
        let mut inputs = quiet_inputs();
        inputs.coherence = 0.4; // Modest therapeutic engagement
        inputs.epistemic_confidence = 0.3;
        bath.update(&inputs);
    }

    println!("  After 30 cycles with MAT:");
    print_neuromod_state(&bath, "    ");
    println!("    In withdrawal: {}", bath.is_in_withdrawal());
    println!("    Allostatic load: {:.3}", bath.allostatic_load);
    println!();

    // ── Phase 4: Clinical Formulation ──────────────────────────────────
    println!("Phase 4: CLINICAL FORMULATION (CBT 4P Model)");
    println!();

    // Determine depleted transmitters
    let depleted: Vec<String> = [
        ("dopamine", bath.dopamine.receptor_sensitivity),
        ("serotonin", bath.serotonin.receptor_sensitivity),
        ("gaba", bath.gaba.receptor_sensitivity),
        ("noradrenaline", bath.noradrenaline.receptor_sensitivity),
        ("endocannabinoid", bath.endocannabinoid.receptor_sensitivity),
    ]
    .iter()
    .filter(|(_, sens)| *sens < 0.8)
    .map(|(name, _)| name.to_string())
    .collect();

    let withdrawing: Vec<String> = [
        ("dopamine", bath.dopamine.withdrawal_cycles),
        ("serotonin", bath.serotonin.withdrawal_cycles),
        ("gaba", bath.gaba.withdrawal_cycles),
        ("endocannabinoid", bath.endocannabinoid.withdrawal_cycles),
    ]
    .iter()
    .filter(|(_, cycles)| *cycles > 0)
    .map(|(name, _)| name.to_string())
    .collect();

    println!("  Depleted transmitters (sensitivity < 0.8):");
    for t in &depleted {
        println!("    - {}", t);
    }
    println!("  Transmitters in withdrawal rebound:");
    for t in &withdrawing {
        println!("    - {}", t);
    }
    println!();

    // Build formulation (using the types directly since symthaea-therapeutic
    // may not be feature-enabled; we demonstrate the concept)
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │ 4P CASE FORMULATION: Opioid Use Disorder               │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ PREDISPOSING                                            │");
    println!("  │  • Genetic vulnerability to addiction (family history)   │");
    println!("  │  • Early adverse experiences (ACE score elevated)        │");
    println!("  │  • High novelty-seeking temperament                     │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ PRECIPITATING                                           │");
    println!("  │  • Chronic pain → opioid prescription → escalation      │");
    println!("  │  • Job loss during pandemic (2020)                      │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ PERPETUATING (auto-populated from neurochemistry)       │");
    for t in &depleted {
        let desc = match t.as_str() {
            "dopamine" => "DA depletion → anhedonia, reduced motivation for non-drug rewards",
            "serotonin" => "5-HT dysregulation → mood instability, impulsivity",
            "gaba" => "GABA downregulation → chronic anxiety, sleep disruption",
            "noradrenaline" => "NE dysregulation → hyperarousal, startle response",
            "endocannabinoid" => "ECB deficiency → impaired stress buffering",
            _ => "Unknown transmitter depletion",
        };
        println!("  │  • {} [neuromod]", desc);
    }
    if bath.allostatic_load > 0.5 {
        println!(
            "  │  • Elevated allostatic load ({:.2}) → chronic stress   │",
            bath.allostatic_load
        );
    }
    println!("  │  • Negative reinforcement cycle (use to escape dysphoria)│");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ PROTECTIVE                                              │");
    println!("  │  • MAT (buprenorphine) → stabilized neurochemistry      │");
    println!("  │  • Supportive family (sister as emergency contact)       │");
    println!("  │  • Expressed motivation for change (Contemplation stage) │");
    println!("  │  • Care Circle membership (3 peer supporters)            │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ CBT BELIEF CHAIN                                        │");
    println!("  │  Core: \"I am powerless without the substance\"            │");
    println!("  │  → If I feel bad, I must use to cope                     │");
    println!("  │  → \"I need opioids now or I'll lose control\"             │");
    println!("  │  → Seeking/using substance                               │");
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();

    // ── Phase 5: Affect Regulation Strategy Selection ──────────────────
    println!("Phase 5: AFFECT REGULATION (Substance-Aware Strategy Selection)");
    println!();

    // Simulate different withdrawal/craving scenarios
    let scenarios = [
        ("Acute withdrawal (day 3)", 0.8, 0.9),
        ("Early abstinence (week 2)", 0.4, 0.6),
        ("Maintenance (month 3)", 0.15, 0.3),
        ("Advanced recovery (month 9)", 0.05, 0.1),
    ];

    for (label, withdrawal_sev, craving) in &scenarios {
        let strategy = select_substance_aware_strategy(*withdrawal_sev, *craving);
        println!(
            "  {:35} → {} (withdrawal={:.1}, craving={:.1})",
            label, strategy, withdrawal_sev, craving
        );
    }
    println!();

    // ── Phase 6: Recovery Timeline ─────────────────────────────────────
    println!("Phase 6: RECOVERY TIMELINE PREDICTION");
    println!("  Substance: Opioids");
    println!("  Duration of use: 200 cycles (chronic)");
    println!("  Severity: 0.7 (moderate-severe dependence)");
    println!();

    let timeline = predict_recovery_timeline(&opioid, 200, 0.7);
    println!("{}", timeline);

    // Also show comparison across substances
    println!("  ── Comparative Recovery Timelines (moderate severity, 100 cycles) ──");
    println!(
        "  {:20} {:>8} {:>8} {:>12} {:>12}",
        "Substance", "Acute", "PAWS", "Baseline", "Allostatic"
    );
    println!(
        "  {:20} {:>8} {:>8} {:>12} {:>12}",
        "─────────", "─────", "────", "────────", "──────────"
    );

    for cat in [
        SubstanceCategory::Opioids,
        SubstanceCategory::SyntheticOpioids,
        SubstanceCategory::Alcohol,
        SubstanceCategory::Stimulants,
        SubstanceCategory::BenzodiazepinesShort,
        SubstanceCategory::BenzodiazepinesLong,
        SubstanceCategory::Cannabis,
        SubstanceCategory::Tobacco,
        SubstanceCategory::Psychedelics,
    ] {
        let profile = profile_for_category(cat);
        let tl = predict_recovery_timeline(&profile, 100, 0.5);
        println!(
            "  {:20} {:>8} {:>8} {:>12} {:>12}",
            format!("{}", cat),
            format!("→{}", tl.acute_withdrawal_end),
            format!("→{}", tl.paws_end),
            format!("→{}", tl.baseline_restoration),
            format!("→{}", tl.allostatic_recovery),
        );
    }
    println!();

    // ── Phase 7: Recovery Progress ─────────────────────────────────────
    println!("Phase 7: RECOVERY MILESTONES & TEND COMPENSATION");
    println!();
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │ Recovery Dashboard                                          │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │ Sobriety: 90 days                                           │");
    println!("  │ Phase: EarlyAbstinence → Maintenance (transition milestone) │");
    println!("  │ MAT: Buprenorphine (adherence: 97%)                         │");
    println!("  │ Latest craving: 3/10 (↓ from 9/10 at intake)                │");
    println!("  │ Mood trend: 6.8/10 avg (last 7 check-ins)                   │");
    println!("  │ Safety plan: Active (last reviewed 14 days ago)              │");
    println!("  │ Relapse prevention plan: Active (8 triggers identified)      │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │ Milestones:                                                 │");
    println!("  │  ✓ 30-day sobriety (verified by sponsor)                    │");
    println!("  │  ✓ 60-day sobriety (verified by counselor)                  │");
    println!("  │  ✓ 90-day sobriety (verified by sponsor)                    │");
    println!("  │  ✓ Step 1-3 completion (12-step program)                    │");
    println!("  │  ✓ 5 peer support sessions attended                         │");
    println!("  │  ✓ Housing stability (transitional → permanent)             │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │ TEND Compensation (Care Economy):                           │");
    println!("  │  • Sponsor earned: 12.5 TEND (10 sessions × 1.25h avg)      │");
    println!("  │  • Counselor earned: 8.0 TEND (8 sessions × 1h)             │");
    println!("  │  • Participant now sponsors others: +3.0 TEND earned         │");
    println!("  │  • Commons Pool allocated: 25 TEND for this recovery plan    │");
    println!("  │  • Funded by: Community vote (NeedCategory::CareSupport)     │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │ 4D Trust Profile:                                           │");
    println!("  │  Identity: 0.85  Reputation: 0.62 (rebuilding from 0.31)    │");
    println!("  │  Community: 0.78  Engagement: 0.71                          │");
    println!("  │  Effective Tier: Participant (↑ from Observer at intake)     │");
    println!("  │  Governance: Can vote on non-constitutional proposals        │");
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!();

    // ── Phase 8: Restorative Justice Integration ───────────────────────
    println!("Phase 8: RESTORATIVE JUSTICE BRIDGE (if harm was caused)");
    println!();
    println!("  If the participant caused harm during active addiction:");
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │ Restorative Circle Status: Monitoring                       │");
    println!("  │  Facilitator: Community Elder                                │");
    println!("  │  Sessions completed: 4/6                                     │");
    println!("  │  Agreements:                                                 │");
    println!("  │   1. ✓ Formal apology delivered (Remedy::Apology)            │");
    println!("  │   2. ◐ Community service: 20/40 hours (TEND-linked)          │");
    println!("  │   3. ✓ Enrolled in rehabilitation (Remedy::Education)        │");
    println!("  │   4. ◐ Restitution: 15/25 TEND hours worked                 │");
    println!("  │                                                              │");
    println!("  │  Recovery progress counts toward restorative obligations.    │");
    println!("  │  The system does NOT permanently exile. Restoration is       │");
    println!("  │  earned through sustained corrective action.                 │");
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!();

    // ── Summary ────────────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════════");
    println!("  ARCHITECTURE SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Symthaea provides:");
    println!("    • Neurobiologically-grounded substance modeling (7 profiles)");
    println!("    • Tolerance/withdrawal simulation (Koob & Le Moal 2001)");
    println!("    • MAT protocol modeling (6 medications)");
    println!("    • Recovery timeline prediction (acute → PAWS → baseline)");
    println!("    • Substance-aware affect regulation strategy selection");
    println!("    • Clinical formulation with auto-populated neurochemical factors");
    println!("    • Crisis detection (SubstanceCrisis type)");
    println!();
    println!("  Mycelix provides:");
    println!("    • 42 CFR Part 2 substance abuse record protection");
    println!("    • Recovery milestones with sponsor/counselor attestation");
    println!("    • TEND mutual credit for compensating care work");
    println!("    • Commons Pool governance for funding rehabilitation");
    println!("    • Restorative Justice circles (not carceral punishment)");
    println!("    • Care Circles for peer recovery groups");
    println!("    • 4D Trust Profile with graceful relapse/recovery handling");
    println!("    • Privacy-preserving health vaults (agent-centric DHT)");
    println!();
    println!("  Together, they replace the 1602 punitive architecture with a");
    println!("  community-centric, scientifically-grounded, economically-viable");
    println!("  alternative that heals rather than punishes.");
    println!();
    println!("  Current status: 77% recidivism rate (punitive system).");
    println!("  Target: community-embedded recovery with restorative accountability.");
    println!();
}

/// Print key neuromodulator levels.
fn print_neuromod_state(bath: &NeuromodulatorBath, indent: &str) {
    println!(
        "{}DA: {:.3} (sens: {:.3})  NE: {:.3} (sens: {:.3})  5-HT: {:.3} (sens: {:.3})",
        indent,
        bath.dopamine.level,
        bath.dopamine.receptor_sensitivity,
        bath.noradrenaline.level,
        bath.noradrenaline.receptor_sensitivity,
        bath.serotonin.level,
        bath.serotonin.receptor_sensitivity,
    );
    println!(
        "{}GABA: {:.3} (sens: {:.3})  ECB: {:.3} (sens: {:.3})  Oxy: {:.3}",
        indent,
        bath.gaba.level,
        bath.gaba.receptor_sensitivity,
        bath.endocannabinoid.level,
        bath.endocannabinoid.receptor_sensitivity,
        bath.oxytocin.level,
    );
}

/// Substance-aware affect regulation strategy selection.
///
/// Simplified version of RegulationEngine::select_strategy_with_substance_context
/// for demonstration purposes.
fn select_substance_aware_strategy(
    withdrawal_severity: f32,
    craving_intensity: f32,
) -> &'static str {
    if withdrawal_severity > 0.7 || craving_intensity > 0.7 {
        "Grounding (somatic anchoring — GABA↑, NE↓)"
    } else if withdrawal_severity > 0.4 {
        "DistressTolerance (DBT skills — GABA↑, Adenosine↑)"
    } else if craving_intensity > 0.3 {
        "Defusion (urge surfing — ACh↑, 5-HT↑)"
    } else if withdrawal_severity < 0.1 && craving_intensity < 0.2 {
        "ExposurePrep (graduated cue exposure — NE↑, ACh↑)"
    } else {
        "Validation (normalize experience — Oxytocin↑, 5-HT↑)"
    }
}
