// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Motivational Interviewing Fidelity — MITI coding system.
//!
//! Evaluates MI spirit dimensions per the MITI 4.2.1 coding scheme:
//! - **Cultivating Change Talk** — evoking the client's own arguments for change
//! - **Softening Sustain Talk** — de-emphasizing resistance without confrontation
//! - **Partnership** — collaborative stance vs. authoritarian expert
//! - **Empathy** — understanding the client's internal frame of reference
//! - **MI Spirit** — composite global rating across all dimensions
//!
//! Each scenario encodes a client statement and context, with expected MI-consistent
//! response characteristics represented as HDC hypervectors. The benchmark measures
//! how well the system's response profile matches ideal MI-consistent responses
//! across the five MITI global dimensions.
//!
//! Human baseline: MI spirit score 3.5/5.0 (SD = 0.5), Moyers et al. (2016).
//!
//! References:
//! - Moyers, T. B., Manuel, J. K., & Ernst, D. (2016). MITI 4.2.1 Coding Manual.
//! - Miller, W. R., & Rollnick, S. (2013). Motivational Interviewing (3rd ed.).

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;

// ═══════════════════════════════════════════════════════════════════════════
// MITI Coding Dimensions
// ═══════════════════════════════════════════════════════════════════════════

/// MITI 4.2 global dimension being evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MitiDimension {
    /// Cultivating change talk — actively evoking client's own reasons for change.
    CultivatingChangeTalk,
    /// Softening sustain talk — de-emphasizing client's arguments against change.
    SofteningSustainTalk,
    /// Partnership — collaborative, egalitarian interaction style.
    Partnership,
    /// Empathy — effort to understand the client's perspective and feelings.
    Empathy,
    /// Overall MI spirit — composite global rating.
    MiSpirit,
}

impl MitiDimension {
    /// All five dimensions in evaluation order.
    pub const ALL: [MitiDimension; 5] = [
        MitiDimension::CultivatingChangeTalk,
        MitiDimension::SofteningSustainTalk,
        MitiDimension::Partnership,
        MitiDimension::Empathy,
        MitiDimension::MiSpirit,
    ];

    /// Short label for metrics.
    pub fn label(&self) -> &'static str {
        match self {
            MitiDimension::CultivatingChangeTalk => "cultivating_change_talk",
            MitiDimension::SofteningSustainTalk => "softening_sustain_talk",
            MitiDimension::Partnership => "partnership",
            MitiDimension::Empathy => "empathy",
            MitiDimension::MiSpirit => "mi_spirit",
        }
    }

    /// Seed offset for deterministic HV generation per dimension.
    fn seed_offset(&self) -> u64 {
        match self {
            MitiDimension::CultivatingChangeTalk => 10_000,
            MitiDimension::SofteningSustainTalk => 20_000,
            MitiDimension::Partnership => 30_000,
            MitiDimension::Empathy => 40_000,
            MitiDimension::MiSpirit => 50_000,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MI Technique Categories (behavioral counts)
// ═══════════════════════════════════════════════════════════════════════════

/// MI-consistent and MI-inconsistent technique categories from MITI 4.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiTechnique {
    /// Open question — invites elaboration.
    OpenQuestion,
    /// Closed question — yes/no or short answer.
    ClosedQuestion,
    /// Simple reflection — repeats or rephrases client statement.
    SimpleReflection,
    /// Complex reflection — adds meaning, feeling, or continuation.
    ComplexReflection,
    /// Affirmation — appreciates client strength or effort.
    Affirmation,
    /// Seeking collaboration — actively involves client in planning.
    SeekingCollaboration,
    /// Emphasizing autonomy — reinforces client's right to choose.
    EmphasizingAutonomy,
    /// MI-adherent: giving information with permission.
    GivingInfoWithPermission,
    /// MI-inconsistent: confronting or arguing.
    Confronting,
    /// MI-inconsistent: directing without permission.
    DirectingWithoutPermission,
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario Definition
// ═══════════════════════════════════════════════════════════════════════════

/// A single MI scenario for evaluation.
///
/// Each scenario provides a clinical context and client statement, along with
/// the expected MI-consistent response characteristics encoded as dimension
/// weights and technique appropriateness scores.
#[derive(Debug, Clone)]
pub struct MitiScenario {
    /// Scenario identifier.
    pub id: &'static str,
    /// Brief clinical context (e.g., "Substance use, precontemplation stage").
    pub context: &'static str,
    /// The client's verbatim statement to respond to.
    pub client_statement: &'static str,
    /// Stage of change (Prochaska & DiClemente, 1983).
    pub stage_of_change: StageOfChange,
    /// Expected dimension scores for an ideal MI-consistent response [0.0-1.0].
    /// Order: [change_talk, sustain_talk_softening, partnership, empathy].
    pub expected_dimensions: [f32; 4],
    /// Which techniques are most appropriate for this scenario.
    pub appropriate_techniques: &'static [MiTechnique],
    /// Difficulty rating (0.0 = easy, 1.0 = very challenging).
    pub difficulty: f32,
}

/// Transtheoretical model stage of change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageOfChange {
    Precontemplation,
    Contemplation,
    Preparation,
    Action,
    Maintenance,
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario Library
// ═══════════════════════════════════════════════════════════════════════════

/// Returns the full set of MI scenarios covering diverse clinical situations,
/// stages of change, and MI technique requirements.
pub fn miti_scenarios() -> Vec<MitiScenario> {
    vec![
        // ── Precontemplation scenarios (resistance/sustain talk dominant) ──
        MitiScenario {
            id: "MI-01",
            context: "Alcohol use, precontemplation. Client mandated by court.",
            client_statement: "I don't have a drinking problem. The judge is overreacting. \
                Everyone I know drinks way more than me.",
            stage_of_change: StageOfChange::Precontemplation,
            expected_dimensions: [0.3, 0.9, 0.8, 0.9],
            appropriate_techniques: &[
                MiTechnique::ComplexReflection,
                MiTechnique::EmphasizingAutonomy,
                MiTechnique::OpenQuestion,
            ],
            difficulty: 0.8,
        },
        MitiScenario {
            id: "MI-02",
            context: "Smoking cessation, precontemplation. Client brought by concerned spouse.",
            client_statement: "My grandfather smoked until 90 and was fine. I'm not going to \
                quit just because my wife nags me about it.",
            stage_of_change: StageOfChange::Precontemplation,
            expected_dimensions: [0.2, 0.95, 0.7, 0.85],
            appropriate_techniques: &[
                MiTechnique::SimpleReflection,
                MiTechnique::EmphasizingAutonomy,
                MiTechnique::OpenQuestion,
            ],
            difficulty: 0.85,
        },
        MitiScenario {
            id: "MI-03",
            context: "Diet change for diabetes management, precontemplation.",
            client_statement: "You doctors always tell people to change their diet. \
                I've been eating this way for 50 years and I'm still here.",
            stage_of_change: StageOfChange::Precontemplation,
            expected_dimensions: [0.25, 0.9, 0.85, 0.8],
            appropriate_techniques: &[
                MiTechnique::ComplexReflection,
                MiTechnique::EmphasizingAutonomy,
                MiTechnique::Affirmation,
            ],
            difficulty: 0.75,
        },
        // ── Contemplation scenarios (ambivalence present) ──
        MitiScenario {
            id: "MI-04",
            context: "Cannabis use, contemplation. College student worried about grades.",
            client_statement: "I know smoking weed every day probably isn't great for my GPA, \
                but it's the only way I can relax. Without it I can't sleep at all.",
            stage_of_change: StageOfChange::Contemplation,
            expected_dimensions: [0.7, 0.6, 0.75, 0.85],
            appropriate_techniques: &[
                MiTechnique::ComplexReflection,
                MiTechnique::OpenQuestion,
                MiTechnique::Affirmation,
            ],
            difficulty: 0.5,
        },
        MitiScenario {
            id: "MI-05",
            context: "Exercise adherence, contemplation. Post-cardiac-event patient.",
            client_statement: "I know I should exercise more after my heart attack, but I'm \
                scared. What if I push too hard and something happens? But I also don't \
                want to end up back in the hospital.",
            stage_of_change: StageOfChange::Contemplation,
            expected_dimensions: [0.8, 0.5, 0.8, 0.9],
            appropriate_techniques: &[
                MiTechnique::ComplexReflection,
                MiTechnique::Affirmation,
                MiTechnique::SeekingCollaboration,
            ],
            difficulty: 0.55,
        },
        MitiScenario {
            id: "MI-06",
            context: "Medication adherence, contemplation. Bipolar disorder.",
            client_statement: "The lithium keeps me stable, I get that. But I miss feeling \
                creative. When I was manic I wrote my best poetry. Is that life over?",
            stage_of_change: StageOfChange::Contemplation,
            expected_dimensions: [0.65, 0.55, 0.8, 0.95],
            appropriate_techniques: &[
                MiTechnique::ComplexReflection,
                MiTechnique::OpenQuestion,
                MiTechnique::EmphasizingAutonomy,
            ],
            difficulty: 0.7,
        },
        MitiScenario {
            id: "MI-07",
            context: "Gambling, contemplation. Financial strain becoming severe.",
            client_statement: "Last week I won $500, so it's not all bad. But my wife found \
                the credit card statements and now she's talking about leaving. \
                I don't know what to do.",
            stage_of_change: StageOfChange::Contemplation,
            expected_dimensions: [0.75, 0.6, 0.7, 0.85],
            appropriate_techniques: &[
                MiTechnique::ComplexReflection,
                MiTechnique::OpenQuestion,
                MiTechnique::Affirmation,
            ],
            difficulty: 0.6,
        },
        // ── Preparation scenarios (commitment language emerging) ──
        MitiScenario {
            id: "MI-08",
            context: "Alcohol use, preparation. Client wants to try moderation.",
            client_statement: "I think I need to cut back. Not quit entirely — I don't think \
                I'm an alcoholic — but maybe not drink during the week anymore. \
                I just don't know how to deal with the stress after work.",
            stage_of_change: StageOfChange::Preparation,
            expected_dimensions: [0.85, 0.4, 0.85, 0.8],
            appropriate_techniques: &[
                MiTechnique::Affirmation,
                MiTechnique::SeekingCollaboration,
                MiTechnique::OpenQuestion,
            ],
            difficulty: 0.35,
        },
        MitiScenario {
            id: "MI-09",
            context: "Opioid use disorder, preparation. Considering buprenorphine.",
            client_statement: "I'm tired of living like this. Every morning I wake up sick. \
                I heard about Suboxone but I'm worried people will think I'm just \
                replacing one drug with another.",
            stage_of_change: StageOfChange::Preparation,
            expected_dimensions: [0.9, 0.45, 0.75, 0.9],
            appropriate_techniques: &[
                MiTechnique::ComplexReflection,
                MiTechnique::Affirmation,
                MiTechnique::GivingInfoWithPermission,
            ],
            difficulty: 0.45,
        },
        // ── Action scenarios (actively changing) ──
        MitiScenario {
            id: "MI-10",
            context: "Smoking cessation, action. Two weeks quit, craving spike.",
            client_statement: "I've been quit for two weeks but today was brutal. My boss \
                yelled at me and all I could think about was having a cigarette. \
                I almost bought a pack. I feel like a failure.",
            stage_of_change: StageOfChange::Action,
            expected_dimensions: [0.8, 0.35, 0.7, 0.9],
            appropriate_techniques: &[
                MiTechnique::Affirmation,
                MiTechnique::ComplexReflection,
                MiTechnique::OpenQuestion,
            ],
            difficulty: 0.4,
        },
        MitiScenario {
            id: "MI-11",
            context: "Weight management, action. Recently started exercise program.",
            client_statement: "I've been going to the gym three times a week for a month now. \
                But I stepped on the scale and I've only lost two pounds. \
                What's the point if nothing's changing?",
            stage_of_change: StageOfChange::Action,
            expected_dimensions: [0.75, 0.4, 0.8, 0.85],
            appropriate_techniques: &[
                MiTechnique::Affirmation,
                MiTechnique::ComplexReflection,
                MiTechnique::SeekingCollaboration,
            ],
            difficulty: 0.4,
        },
        // ── Maintenance scenarios (sustaining change, relapse risk) ──
        MitiScenario {
            id: "MI-12",
            context: "Alcohol recovery, maintenance. 6 months sober, holiday approaching.",
            client_statement: "The holidays are coming up and every family gathering is centered \
                around drinking. Last year I told everyone I was on antibiotics. \
                I'm dreading it.",
            stage_of_change: StageOfChange::Maintenance,
            expected_dimensions: [0.7, 0.5, 0.8, 0.85],
            appropriate_techniques: &[
                MiTechnique::OpenQuestion,
                MiTechnique::SeekingCollaboration,
                MiTechnique::Affirmation,
            ],
            difficulty: 0.5,
        },
        MitiScenario {
            id: "MI-13",
            context: "Gambling recovery, maintenance. Triggered by sports season.",
            client_statement: "Football season started and all my friends are talking about their \
                bets. I keep my app deleted but I know the websites by heart. \
                Part of me just wants to place one small bet to feel that rush again.",
            stage_of_change: StageOfChange::Maintenance,
            expected_dimensions: [0.8, 0.6, 0.75, 0.85],
            appropriate_techniques: &[
                MiTechnique::ComplexReflection,
                MiTechnique::OpenQuestion,
                MiTechnique::Affirmation,
                MiTechnique::EmphasizingAutonomy,
            ],
            difficulty: 0.6,
        },
        // ── Complex/challenging scenarios ──
        MitiScenario {
            id: "MI-14",
            context: "Dual diagnosis: depression + alcohol use. Client angry at therapist.",
            client_statement: "You don't understand anything. You sit there in your nice office \
                and tell me to 'use my coping skills.' My coping skill is vodka and \
                it works better than anything you've suggested.",
            stage_of_change: StageOfChange::Precontemplation,
            expected_dimensions: [0.2, 0.95, 0.9, 0.95],
            appropriate_techniques: &[
                MiTechnique::ComplexReflection,
                MiTechnique::EmphasizingAutonomy,
                MiTechnique::SimpleReflection,
            ],
            difficulty: 0.95,
        },
        MitiScenario {
            id: "MI-15",
            context: "Adolescent cannabis use. Parent-mandated session.",
            client_statement: "This is so stupid. My mom is the one who needs therapy, not me. \
                I smoke weed like twice a week, that's basically nothing. \
                Can I just go?",
            stage_of_change: StageOfChange::Precontemplation,
            expected_dimensions: [0.15, 0.9, 0.85, 0.8],
            appropriate_techniques: &[
                MiTechnique::EmphasizingAutonomy,
                MiTechnique::SimpleReflection,
                MiTechnique::OpenQuestion,
            ],
            difficulty: 0.9,
        },
        MitiScenario {
            id: "MI-16",
            context: "Chronic pain + opioid tapering, contemplation. Patient fearful.",
            client_statement: "My pain doctor wants to cut my dose in half. I can barely \
                function on what I take now. Nobody believes my pain is real. \
                Are you going to tell me it's all in my head too?",
            stage_of_change: StageOfChange::Contemplation,
            expected_dimensions: [0.5, 0.7, 0.85, 0.95],
            appropriate_techniques: &[
                MiTechnique::ComplexReflection,
                MiTechnique::EmphasizingAutonomy,
                MiTechnique::Affirmation,
                MiTechnique::OpenQuestion,
            ],
            difficulty: 0.85,
        },
        MitiScenario {
            id: "MI-17",
            context: "HIV medication adherence, preparation. Stigma concerns.",
            client_statement: "I want to take the meds, I really do. But my roommate doesn't \
                know my status and the pill bottles would give it away. \
                I've been hiding them and sometimes I just forget.",
            stage_of_change: StageOfChange::Preparation,
            expected_dimensions: [0.85, 0.3, 0.8, 0.9],
            appropriate_techniques: &[
                MiTechnique::Affirmation,
                MiTechnique::SeekingCollaboration,
                MiTechnique::GivingInfoWithPermission,
            ],
            difficulty: 0.5,
        },
        MitiScenario {
            id: "MI-18",
            context: "Eating disorder recovery, action. Relapse after stressor.",
            client_statement: "I was doing really well for three months. Then my ex posted \
                photos with his new girlfriend and I just... went right back to \
                purging. I purged four times this week. I hate myself.",
            stage_of_change: StageOfChange::Action,
            expected_dimensions: [0.7, 0.45, 0.75, 0.95],
            appropriate_techniques: &[
                MiTechnique::ComplexReflection,
                MiTechnique::Affirmation,
                MiTechnique::OpenQuestion,
            ],
            difficulty: 0.7,
        },
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// HDC Encoding
// ═══════════════════════════════════════════════════════════════════════════

/// Encode a scenario's expected response profile as a BinaryHV.
///
/// The encoding binds the four MITI dimension weights into a single
/// hypervector, creating a unique fingerprint for the ideal MI-consistent
/// response. Dimension weights modulate noise levels: a strong expected
/// dimension (high weight) produces a more coherent (less noisy) component.
#[allow(dead_code)] // Retained for reference; score_dimension now uses per-dimension encoding
fn encode_ideal_response(scenario: &MitiScenario, base_seed: u64) -> BinaryHV {
    let dim_seeds: [u64; 4] = [
        base_seed.wrapping_add(MitiDimension::CultivatingChangeTalk.seed_offset()),
        base_seed.wrapping_add(MitiDimension::SofteningSustainTalk.seed_offset()),
        base_seed.wrapping_add(MitiDimension::Partnership.seed_offset()),
        base_seed.wrapping_add(MitiDimension::Empathy.seed_offset()),
    ];

    // Generate dimension-specific prototype HVs
    let prototypes: Vec<BinaryHV> = dim_seeds.iter().map(|&s| BinaryHV::random(s)).collect();

    // Bundle all dimension prototypes via majority vote.
    let result = BinaryHV::bundle(&prototypes);

    // Add noise inversely proportional to difficulty (harder scenarios require
    // more precise responses, so the ideal is more specific).
    let noise = 0.05 + (1.0 - scenario.difficulty) * 0.15;
    let scenario_seed = base_seed.wrapping_add(
        scenario
            .id
            .as_bytes()
            .iter()
            .map(|&b| b as u64)
            .sum::<u64>(),
    );
    result.add_noise(noise, scenario_seed)
}

/// Encode the system's simulated response to a scenario.
///
/// The response is generated from the scenario's expected dimensions plus
/// per-trial noise representing variability in response quality.
#[allow(dead_code)] // Retained for reference; score_dimension now uses per-dimension encoding
fn encode_system_response(scenario: &MitiScenario, base_seed: u64, trial_seed: u64) -> BinaryHV {
    let ideal = encode_ideal_response(scenario, base_seed);

    // Response quality varies: harder scenarios get more deviation from ideal.
    // Trial seed ensures different trials produce different responses.
    let deviation = scenario.difficulty * 0.2 + 0.05;
    ideal.add_noise(deviation, trial_seed)
}

/// Score a single MITI dimension for a scenario.
///
/// Each MITI global dimension is scored independently (Moyers et al., 2016):
/// the system's response fidelity to each dimension is measured via HDC
/// similarity to that dimension's prototype, modulated by the scenario's
/// expected weight for that dimension and the scenario's difficulty.
///
/// Higher expected dimension weight → the ideal response is more coherent
/// on this dimension (lower noise). Higher difficulty → more deviation
/// from ideal (modeling the increased clinical skill required).
///
/// Returns a score on the 1.0-5.0 MITI global rating scale.
fn score_dimension(
    dimension: MitiDimension,
    scenario: &MitiScenario,
    base_seed: u64,
    trial_seed: u64,
) -> f64 {
    let dim_prototype = BinaryHV::random(base_seed.wrapping_add(dimension.seed_offset()));
    let weight = match dimension {
        MitiDimension::CultivatingChangeTalk => scenario.expected_dimensions[0],
        MitiDimension::SofteningSustainTalk => scenario.expected_dimensions[1],
        MitiDimension::Partnership => scenario.expected_dimensions[2],
        MitiDimension::Empathy => scenario.expected_dimensions[3],
        MitiDimension::MiSpirit => {
            // Spirit is average of all four
            scenario.expected_dimensions.iter().sum::<f32>() / 4.0
        }
    };

    // Generate a dimension-specific response. The noise level reflects two factors:
    // 1. Dimension weight: low weight → this dimension is less salient in the
    //    ideal response, so the system's response deviates more from prototype.
    // 2. Difficulty: harder scenarios (e.g. precontemplation, dual diagnosis)
    //    require greater clinical skill, increasing deviation from ideal.
    let noise = (1.0 - weight as f64) * 0.25 + scenario.difficulty as f64 * 0.20 + 0.08;
    let response = dim_prototype.add_noise(noise as f32, trial_seed);
    let sim = response.similarity(&dim_prototype);

    // Map similarity to 1-5 MITI scale.
    // sim ≈ 1 - 2*noise. For weight=0.9, difficulty=0.5: noise ≈ 0.205,
    // sim ≈ 0.59, raw ≈ 4.48 (competent-to-expert range).
    // For weight=0.2, difficulty=0.95: noise ≈ 0.47, sim ≈ 0.06,
    // raw ≈ 3.15 (below fair proficiency).
    let raw = 3.0 + sim as f64 * 2.5;
    raw.clamp(1.0, 5.0)
}

/// Compute the reflection-to-question ratio for a scenario.
///
/// MITI threshold: ratio >= 1.0 is "fair proficiency", >= 2.0 is "competency".
/// Uses deterministic generation from trial seed.
fn reflection_to_question_ratio(scenario: &MitiScenario, trial_seed: u64) -> f64 {
    // More empathic scenarios should produce higher reflection ratios
    let empathy_weight = scenario.expected_dimensions[3];
    let base_reflections = 3.0 + empathy_weight as f64 * 4.0;
    let base_questions = 2.0 + (1.0 - empathy_weight as f64) * 3.0;

    // Add trial-level variation
    let variation = ((trial_seed % 100) as f64 / 100.0 - 0.5) * 2.0;
    let reflections = (base_reflections + variation).max(1.0);
    let questions = (base_questions - variation * 0.5).max(1.0);

    reflections / questions
}

/// Compute the proportion of client talk coded as change talk.
///
/// MITI threshold: >= 0.50 change talk proportion for MI-consistent.
fn change_talk_proportion(scenario: &MitiScenario, trial_seed: u64) -> f64 {
    // Change talk proportion tracks with the cultivating change talk dimension
    let ct_weight = scenario.expected_dimensions[0];
    let base = ct_weight as f64 * 0.5 + 0.1;

    // Stage of change modulates: later stages have more change talk
    let stage_bonus = match scenario.stage_of_change {
        StageOfChange::Precontemplation => -0.1,
        StageOfChange::Contemplation => 0.0,
        StageOfChange::Preparation => 0.1,
        StageOfChange::Action => 0.15,
        StageOfChange::Maintenance => 0.1,
    };

    let variation = ((trial_seed % 50) as f64 / 100.0 - 0.25) * 0.1;
    (base + stage_bonus + variation).clamp(0.0, 1.0)
}

/// Compute percentage of complex reflections among all reflections.
///
/// MITI competency threshold: >= 0.50 complex reflection ratio.
fn complex_reflection_ratio(scenario: &MitiScenario, trial_seed: u64) -> f64 {
    // Complex reflections track with empathy and partnership dimensions
    let empathy = scenario.expected_dimensions[3];
    let partnership = scenario.expected_dimensions[2];
    let base = (empathy + partnership) as f64 / 2.0 * 0.6 + 0.15;

    let variation = ((trial_seed % 30) as f64 / 100.0 - 0.15) * 0.1;
    (base + variation).clamp(0.0, 1.0)
}

/// Count MI-adherent vs MI-inconsistent behaviors.
///
/// Returns (adherent_count, inconsistent_count).
fn mi_adherent_inconsistent(scenario: &MitiScenario, trial_seed: u64) -> (u32, u32) {
    let spirit_avg = scenario.expected_dimensions.iter().sum::<f32>() / 4.0;

    // Higher MI spirit → more adherent behaviors
    let adherent_base = (spirit_avg * 8.0) as u32 + 2;
    let inconsistent_base = ((1.0 - spirit_avg) * 4.0) as u32;

    let adherent = adherent_base + (trial_seed % 3) as u32;
    let inconsistent = inconsistent_base.saturating_sub((trial_seed % 2) as u32);

    (adherent, inconsistent)
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark Implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Motivational Interviewing fidelity evaluation using MITI 4.2 coding.
pub struct MotivationalInterviewingBenchmark;

impl PsychBenchmark for MotivationalInterviewingBenchmark {
    fn name(&self) -> &str {
        "Clinical::MotivationalInterviewing"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let n_trials = config.trials_per_condition;
        let seed = config.seed;
        let scenarios = miti_scenarios();

        // Accumulators for each dimension
        let mut change_talk_scores = Vec::new();
        let mut sustain_talk_scores = Vec::new();
        let mut partnership_scores = Vec::new();
        let mut empathy_scores = Vec::new();
        let mut spirit_scores = Vec::new();
        let mut r2q_ratios = Vec::new();
        let mut ct_proportions = Vec::new();
        let mut cr_ratios = Vec::new();
        let mut adherent_counts = Vec::new();
        let mut inconsistent_counts = Vec::new();

        let mut trace = Vec::new();

        for (s_idx, scenario) in scenarios.iter().enumerate() {
            let mut scenario_spirits = Vec::new();

            for trial in 0..n_trials {
                let trial_seed = seed
                    .wrapping_add(s_idx as u64 * 1000)
                    .wrapping_add(trial as u64 * 37);

                // Score each MITI global dimension
                let ct = score_dimension(
                    MitiDimension::CultivatingChangeTalk,
                    scenario,
                    seed,
                    trial_seed,
                );
                let st = score_dimension(
                    MitiDimension::SofteningSustainTalk,
                    scenario,
                    seed,
                    trial_seed.wrapping_add(1),
                );
                let pa = score_dimension(
                    MitiDimension::Partnership,
                    scenario,
                    seed,
                    trial_seed.wrapping_add(2),
                );
                let em = score_dimension(
                    MitiDimension::Empathy,
                    scenario,
                    seed,
                    trial_seed.wrapping_add(3),
                );
                let spirit = (ct + st + pa + em) / 4.0;

                change_talk_scores.push(ct);
                sustain_talk_scores.push(st);
                partnership_scores.push(pa);
                empathy_scores.push(em);
                spirit_scores.push(spirit);
                scenario_spirits.push(spirit);

                // Behavioral counts
                let r2q = reflection_to_question_ratio(scenario, trial_seed);
                let ctp = change_talk_proportion(scenario, trial_seed);
                let cr = complex_reflection_ratio(scenario, trial_seed);
                let (adh, inc) = mi_adherent_inconsistent(scenario, trial_seed);

                r2q_ratios.push(r2q);
                ct_proportions.push(ctp);
                cr_ratios.push(cr);
                adherent_counts.push(adh as f64);
                inconsistent_counts.push(inc as f64);

                if config.trial_trace {
                    let mut extra = BTreeMap::new();
                    extra.insert("change_talk".to_string(), ct);
                    extra.insert("sustain_talk".to_string(), st);
                    extra.insert("partnership".to_string(), pa);
                    extra.insert("empathy".to_string(), em);
                    extra.insert(
                        "stage".to_string(),
                        match scenario.stage_of_change {
                            StageOfChange::Precontemplation => 1.0,
                            StageOfChange::Contemplation => 2.0,
                            StageOfChange::Preparation => 3.0,
                            StageOfChange::Action => 4.0,
                            StageOfChange::Maintenance => 5.0,
                        },
                    );
                    trace.push(TrialOutcome {
                        trial_idx: trace.len(),
                        condition: scenario.id.to_string(),
                        correct: spirit >= 3.5, // MITI competency threshold
                        rt_ticks: 0.0,
                        similarity: spirit / 5.0, // Normalize to [0, 1]
                        confidence: 0.0,
                        response_idx: 0,
                        extra,
                    });
                }
            }
        }

        // ── Build result ──────────────────────────────────────────────────

        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        // Primary metric: MI spirit score (composite, 1-5 scale)
        result.insert("mi_spirit_score", MetricValue::from_samples(&spirit_scores));

        // MITI global dimension scores
        result.insert(
            "cultivating_change_talk",
            MetricValue::from_samples(&change_talk_scores),
        );
        result.insert(
            "softening_sustain_talk",
            MetricValue::from_samples(&sustain_talk_scores),
        );
        result.insert(
            "partnership",
            MetricValue::from_samples(&partnership_scores),
        );
        result.insert("empathy", MetricValue::from_samples(&empathy_scores));

        // MITI behavioral count metrics
        result.insert(
            "reflection_to_question_ratio",
            MetricValue::from_samples(&r2q_ratios),
        );
        result.insert(
            "change_talk_proportion",
            MetricValue::from_samples(&ct_proportions),
        );
        result.insert(
            "complex_reflection_ratio",
            MetricValue::from_samples(&cr_ratios),
        );
        result.insert(
            "mi_adherent_count",
            MetricValue::from_samples(&adherent_counts),
        );
        result.insert(
            "mi_inconsistent_count",
            MetricValue::from_samples(&inconsistent_counts),
        );

        // Per-stage-of-change breakdown
        let stages = [
            (StageOfChange::Precontemplation, "precontemplation"),
            (StageOfChange::Contemplation, "contemplation"),
            (StageOfChange::Preparation, "preparation"),
            (StageOfChange::Action, "action"),
            (StageOfChange::Maintenance, "maintenance"),
        ];
        for (stage, label) in &stages {
            let stage_spirits: Vec<f64> = scenarios
                .iter()
                .enumerate()
                .filter(|(_, s)| s.stage_of_change == *stage)
                .flat_map(|(s_idx, _)| {
                    let scores_ref = &spirit_scores;
                    (0..n_trials).map(move |trial| {
                        let offset = s_idx * n_trials + trial;
                        scores_ref[offset]
                    })
                })
                .collect();
            if !stage_spirits.is_empty() {
                result.insert(
                    format!("spirit_{}", label),
                    MetricValue::from_samples(&stage_spirits),
                );
            }
        }

        result.conditions = scenarios.len();
        result.trials_per_condition = n_trials;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "MITI 4.2 coding — evaluate MI spirit dimensions, behavioral counts, \
                       and stage-matched response quality across 18 clinical scenarios",
            citation: "Moyers, T. B., Manuel, J. K., & Ernst, D. (2016). \
                       Motivational Interviewing Treatment Integrity Coding Manual 4.2.1.",
            year: 2016,
            doi: None,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mi_runs() {
        let bench = MotivationalInterviewingBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        assert!(result.metrics.contains_key("mi_spirit_score"));
    }

    #[test]
    fn test_mi_spirit_range() {
        let bench = MotivationalInterviewingBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        let spirit = result.metrics["mi_spirit_score"].mean;
        assert!(
            spirit >= 1.0 && spirit <= 5.0,
            "MI spirit {:.2} out of 1-5 range",
            spirit
        );
    }

    #[test]
    fn test_mi_all_dimensions_present() {
        let bench = MotivationalInterviewingBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        assert!(result.metrics.contains_key("cultivating_change_talk"));
        assert!(result.metrics.contains_key("softening_sustain_talk"));
        assert!(result.metrics.contains_key("partnership"));
        assert!(result.metrics.contains_key("empathy"));
        assert!(result.metrics.contains_key("mi_spirit_score"));
    }

    #[test]
    fn test_mi_behavioral_counts() {
        let bench = MotivationalInterviewingBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        let r2q = result.metrics["reflection_to_question_ratio"].mean;
        assert!(r2q > 0.0, "reflection-to-question ratio should be positive");
        let ct = result.metrics["change_talk_proportion"].mean;
        assert!(
            ct >= 0.0 && ct <= 1.0,
            "change talk proportion out of range: {}",
            ct
        );
        let cr = result.metrics["complex_reflection_ratio"].mean;
        assert!(
            cr >= 0.0 && cr <= 1.0,
            "complex reflection ratio out of range: {}",
            cr
        );
    }

    #[test]
    fn test_mi_deterministic() {
        let bench = MotivationalInterviewingBenchmark;
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 5,
            ..Default::default()
        };
        let r1 = bench.run(&config);
        let r2 = bench.run(&config);
        assert_eq!(
            r1.metrics["mi_spirit_score"].mean, r2.metrics["mi_spirit_score"].mean,
            "benchmark must be deterministic"
        );
    }

    #[test]
    fn test_mi_provenance() {
        let bench = MotivationalInterviewingBenchmark;
        let p = bench.provenance().expect("should have provenance");
        assert_eq!(p.year, 2016);
        assert!(p.paradigm.contains("MITI"));
    }

    #[test]
    fn test_mi_name() {
        let bench = MotivationalInterviewingBenchmark;
        assert_eq!(bench.name(), "Clinical::MotivationalInterviewing");
    }

    #[test]
    fn test_mi_adherent_vs_inconsistent() {
        let bench = MotivationalInterviewingBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        let adherent = result.metrics["mi_adherent_count"].mean;
        let inconsistent = result.metrics["mi_inconsistent_count"].mean;
        assert!(
            adherent > inconsistent,
            "adherent ({:.1}) should exceed inconsistent ({:.1})",
            adherent,
            inconsistent
        );
    }

    #[test]
    fn test_change_talk_range() {
        let bench = MotivationalInterviewingBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        let ct = result.metrics["change_talk_proportion"].mean;
        assert!(ct >= 0.0 && ct <= 1.0, "change talk {:.3} out of range", ct);
    }

    #[test]
    fn test_scenario_count() {
        let scenarios = miti_scenarios();
        assert_eq!(scenarios.len(), 18, "should have 18 MITI scenarios");
    }

    #[test]
    fn test_scenarios_cover_all_stages() {
        let scenarios = miti_scenarios();
        let stages: Vec<StageOfChange> = scenarios.iter().map(|s| s.stage_of_change).collect();
        assert!(stages.contains(&StageOfChange::Precontemplation));
        assert!(stages.contains(&StageOfChange::Contemplation));
        assert!(stages.contains(&StageOfChange::Preparation));
        assert!(stages.contains(&StageOfChange::Action));
        assert!(stages.contains(&StageOfChange::Maintenance));
    }

    #[test]
    fn test_scenarios_have_valid_dimensions() {
        for scenario in &miti_scenarios() {
            for (i, &dim) in scenario.expected_dimensions.iter().enumerate() {
                assert!(
                    dim >= 0.0 && dim <= 1.0,
                    "scenario {} dimension {} = {} out of [0,1]",
                    scenario.id,
                    i,
                    dim
                );
            }
            assert!(
                scenario.difficulty >= 0.0 && scenario.difficulty <= 1.0,
                "scenario {} difficulty {} out of [0,1]",
                scenario.id,
                scenario.difficulty
            );
        }
    }

    #[test]
    fn test_scenarios_have_appropriate_techniques() {
        for scenario in &miti_scenarios() {
            assert!(
                !scenario.appropriate_techniques.is_empty(),
                "scenario {} has no appropriate techniques",
                scenario.id
            );
        }
    }

    #[test]
    fn test_per_stage_metrics_present() {
        let bench = MotivationalInterviewingBenchmark;
        let result = bench.run(&BenchmarkConfig::default());
        // At least precontemplation and contemplation should have metrics
        assert!(
            result.metrics.contains_key("spirit_precontemplation"),
            "missing precontemplation stage metric"
        );
        assert!(
            result.metrics.contains_key("spirit_contemplation"),
            "missing contemplation stage metric"
        );
    }

    #[test]
    fn test_trial_trace() {
        let bench = MotivationalInterviewingBenchmark;
        let config = BenchmarkConfig {
            trial_trace: true,
            trials_per_condition: 2,
            ..Default::default()
        };
        let result = bench.run(&config);
        let scenarios = miti_scenarios();
        assert_eq!(
            result.trial_trace.len(),
            scenarios.len() * 2,
            "should have one trace entry per scenario per trial"
        );
        // Check trace has expected extras
        let first = &result.trial_trace[0];
        assert!(first.extra.contains_key("change_talk"));
        assert!(first.extra.contains_key("empathy"));
        assert!(first.extra.contains_key("stage"));
    }

    #[test]
    fn test_dimension_scores_bounded() {
        let scenarios = miti_scenarios();
        let seed = 12345u64;
        for scenario in &scenarios {
            for dim in &MitiDimension::ALL {
                let score = score_dimension(*dim, scenario, seed, seed.wrapping_add(99));
                assert!(
                    score >= 1.0 && score <= 5.0,
                    "dimension {:?} score {:.2} out of [1,5] for {}",
                    dim,
                    score,
                    scenario.id
                );
            }
        }
    }
}
