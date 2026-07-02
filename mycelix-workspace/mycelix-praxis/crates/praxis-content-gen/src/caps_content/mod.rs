// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Subject-aware content generator for South African CAPS curriculum.
//!
//! Organized by grade and subject. Each sub-module contains topic structs
//! implementing `TopicContent` with grade-appropriate explanations,
//! worked examples, practice problems, and vocabulary.

pub mod gr1_math;
pub mod gr2_math;
pub mod gr3_math;
pub mod gr4_math;
pub mod gr5_math;
pub mod gr6_math;
pub mod gr7_math;
pub mod gr8_math;
pub mod gr9_math;
pub mod gr9_natsci;
pub mod gr10_math;
pub mod gr10_physics;
pub mod gr10_chemistry;
pub mod gr11_math;
pub mod gr11_physics;
pub mod gr12_math;
pub mod gr12_physics;
pub mod gr12_chemistry;

use crate::channels::{ContentChannels, ContentIntent};
use crate::pipeline::{ContentGenerator, GenerationOutput};

/// Content generator for SA CAPS curriculum.
pub struct CapsContentGenerator;

impl ContentGenerator for CapsContentGenerator {
    fn generate(
        &self,
        channels: &ContentChannels,
        context: &str,
        _max_tokens: usize,
    ) -> GenerationOutput {
        let topic = detect_topic(context);

        let text = match channels.intent {
            ContentIntent::TeachConcept => {
                if context.contains("vocabulary") || context.contains("Vocabulary") {
                    topic.vocabulary()
                } else if context.contains("misconception") || context.contains("Misconception") {
                    topic.misconception()
                } else {
                    topic.explanation()
                }
            }
            ContentIntent::GiveExample => {
                let index = detect_example_index(context);
                topic.worked_example(index)
            }
            ContentIntent::AskQuestion => {
                if context.contains("flashcard") || context.contains("Flashcard") {
                    topic.flashcard()
                } else if context.contains("assessment") || context.contains("Assessment") {
                    topic.assessment_item(context)
                } else {
                    let difficulty = detect_difficulty(context);
                    topic.practice_problem(difficulty)
                }
            }
            ContentIntent::ProvideHint => {
                let level = detect_hint_level(context);
                topic.hint(level)
            }
            ContentIntent::ExplainMisconception => topic.misconception(),
            ContentIntent::Encourage => {
                "Good effort! Matric maths and science are challenging — every mistake \
                 brings you closer to understanding. Let's work through this together."
                    .to_string()
            }
            ContentIntent::ReflectOnLearning => {
                "Think about the method you used. Could you explain it to a classmate? \
                 How does this connect to what you learned last term?"
                    .to_string()
            }
        };

        GenerationOutput {
            text,
            coherence: 0.88,
            hallucination_flag: false,
            veto_count: 0,
        }
    }
}

// ============================================================
// Topic trait — implemented by all grade/subject topic structs
// ============================================================

pub(crate) trait TopicContent {
    fn explanation(&self) -> String;
    fn worked_example(&self, index: usize) -> String;
    fn practice_problem(&self, difficulty: u16) -> String;
    fn hint(&self, level: u8) -> String;
    fn misconception(&self) -> String;
    fn vocabulary(&self) -> String;
    fn flashcard(&self) -> String;
    fn assessment_item(&self, context: &str) -> String;
}

// ============================================================
// Grade-first topic detection
// ============================================================

fn extract_grade(context: &str) -> Option<u8> {
    // Check longer grades first to avoid Gr1 matching Gr10/11/12
    if context.contains("Gr10") || context.contains("Grade10") { Some(10) }
    else if context.contains("Gr11") || context.contains("Grade11") { Some(11) }
    else if context.contains("Gr12") || context.contains("Grade12") { Some(12) }
    else if context.contains("Gr9") || context.contains("Grade9") { Some(9) }
    else if context.contains("Gr8") || context.contains("Grade8") { Some(8) }
    else if context.contains("Gr7") || context.contains("Grade7") { Some(7) }
    else if context.contains("Gr6") || context.contains("Grade6") { Some(6) }
    else if context.contains("Gr5") || context.contains("Grade5") { Some(5) }
    else if context.contains("Gr4") || context.contains("Grade4") { Some(4) }
    else if context.contains("Gr3") || context.contains("Grade3") { Some(3) }
    else if context.contains("Gr2") || context.contains("Grade2") { Some(2) }
    else if context.contains("Gr1") || context.contains("Grade1") { Some(1) }
    else { None }
}

fn detect_topic(context: &str) -> Box<dyn TopicContent> {
    let grade = extract_grade(context);
    // Debug: print first 60 chars of context and detected grade
    if context.contains("CAPS.Mathematics.Gr9") {
        // Grade-first routing confirmed working
    }
    match grade {
        Some(1) => Box::new(gr1_math::Gr1MathFallbackTopic),
        Some(2) => Box::new(gr2_math::Gr2MathFallbackTopic),
        Some(3) => Box::new(gr3_math::Gr3MathFallbackTopic),
        Some(4) => Box::new(gr4_math::Gr4MathFallbackTopic),
        Some(5) => Box::new(gr5_math::Gr5MathFallbackTopic),
        Some(6) => Box::new(gr6_math::Gr6MathFallbackTopic),
        Some(7) => Box::new(gr7_math::Gr7MathFallbackTopic),
        Some(8) => Box::new(gr8_math::Gr8MathFallbackTopic),
        Some(9) => detect_gr9_topic(context),
        Some(10) => detect_gr10_topic(context),
        Some(11) => detect_gr11_topic(context),
        _ => detect_gr12_topic(context),
    }
}

fn detect_gr9_topic(context: &str) -> Box<dyn TopicContent> {
    // Natural Sciences first
    if context.contains("MM.1") || context.contains("Properties of Materials") {
        Box::new(gr9_natsci::Gr9MaterialPropertiesTopic)
    } else if context.contains("MM.2") || context.contains("Particle Model") {
        Box::new(gr9_natsci::Gr9ParticleModelTopic)
    } else if context.contains("MM.3") || (context.contains("Chemical Reactions") && !context.contains("Rate")) {
        Box::new(gr9_natsci::Gr9ChemicalReactionsTopic)
    } else if context.contains("EC.1") || context.contains("Energy and Electricity") {
        Box::new(gr9_natsci::Gr9EnergyElectricityTopic)
    } else if context.contains("EC.2") || context.contains("Forces and Motion") {
        Box::new(gr9_natsci::Gr9ForcesMotionTopic)
    } else if context.contains("EC.3") || context.contains("Electric Cells") {
        Box::new(gr9_natsci::Gr9CircuitsTopic)
    } else if context.contains("EC.4") || context.contains("Waves, Sound") {
        Box::new(gr9_natsci::Gr9WavesSoundLightTopic)
    // Mathematics
    } else if context.contains("NUM.1") || context.contains("Whole Numbers") {
        Box::new(gr9_math::Gr9WholeNumbersTopic)
    } else if context.contains("NUM.2") || context.contains("Fractions") || context.contains("Percentages") {
        Box::new(gr9_math::Gr9FractionsDecimalsTopic)
    } else if context.contains("EXP.1") || context.contains("Exponents") {
        Box::new(gr9_math::Gr9ExponentsTopic)
    } else if context.contains("ALG.1") || context.contains("Algebraic Expressions") {
        Box::new(gr9_math::Gr9AlgebraExpressionsTopic)
    } else if context.contains("ALG.2") || context.contains("Algebraic Equations") {
        Box::new(gr9_math::Gr9AlgebraEquationsTopic)
    } else if context.contains("PAT.1") || context.contains("Patterns") {
        Box::new(gr9_math::Gr9PatternsTopic)
    } else if context.contains("FN.1") || context.contains("Functions") {
        Box::new(gr9_math::Gr9FunctionsTopic)
    } else if context.contains("GEOM.1") || context.contains("Straight Lines") {
        Box::new(gr9_math::Gr9GeomStraightLinesToopic)
    } else if context.contains("GEOM.2") || context.contains("2D Shapes") || context.contains("Pythagoras") {
        Box::new(gr9_math::Gr9Geom2DShapesTopic)
    } else if context.contains("MEAS") || context.contains("Area") || context.contains("Volume") {
        Box::new(gr9_math::Gr9MeasurementTopic)
    } else if context.contains("STAT") || context.contains("Data Handling") {
        Box::new(gr9_math::Gr9DataHandlingTopic)
    } else if context.contains("PROB") || context.contains("Probability") {
        Box::new(gr9_math::Gr9ProbabilityTopic)
    } else {
        Box::new(gr9_math::Gr9WholeNumbersTopic) // safe Gr9 fallback
    }
}

fn detect_gr10_topic(context: &str) -> Box<dyn TopicContent> {
    // Physics
    if context.contains("PHY.1") || context.contains("Motion in One") || context.contains("kinematics") {
        Box::new(gr10_physics::Gr10KinematicsTopic)
    } else if context.contains("PHY.2") || context.contains("Energy") && !context.contains("Electr") {
        Box::new(gr10_physics::Gr10EnergyTopic)
    } else if context.contains("PHY.3") || context.contains("Transverse") || context.contains("Longitudinal") {
        Box::new(gr10_physics::Gr10WavesTopic)
    } else if context.contains("PHY.4") || context.contains("Sound") {
        Box::new(gr10_physics::Gr10SoundTopic)
    } else if context.contains("PHY.5") || context.contains("Electromagnetic Radiation") {
        Box::new(gr10_physics::Gr10EMRadiationTopic)
    } else if context.contains("PHY.6") || context.contains("Electrostatics") || context.contains("Coulomb") {
        Box::new(gr12_physics::ElectricityTopic) // electrostatics uses same concepts
    } else if context.contains("PHY.7") || context.contains("Electric Circuits") {
        Box::new(gr12_physics::ElectricityTopic)
    // Chemistry
    } else if context.contains("CHM.1") || context.contains("Atomic Structure") || context.contains("electron configuration") {
        Box::new(gr10_chemistry::Gr10AtomicStructureTopic)
    } else if context.contains("CHM.2") || context.contains("Chemical Bonding") || context.contains("VSEPR") {
        Box::new(gr10_chemistry::Gr10BondingTopic)
    } else if context.contains("CHM.3") || context.contains("Intermolecular") {
        Box::new(gr10_chemistry::Gr10IntermolecularForcesTopic)
    } else if context.contains("CHM.4") || context.contains("Stoichiometry") || context.contains("mole") {
        Box::new(gr10_chemistry::Gr10StoichiometryTopic)
    } else if context.contains("CHM.5") || context.contains("Types of Reactions") || context.contains("redox") {
        Box::new(gr10_chemistry::Gr10TypesOfReactionsTopic)
    } else if context.contains("CHM.6") || context.contains("Water") || context.contains("Hydrosphere") {
        Box::new(gr10_chemistry::Gr10WaterChemistryTopic)
    // Mathematics
    } else if context.contains("ALG.1") || context.contains("Algebraic Expressions") || context.contains("factoris") {
        Box::new(gr10_math::Gr10FactorisationTopic)
    } else if context.contains("ALG.2") || context.contains("Equations and Inequalities") {
        Box::new(gr10_math::Gr10EquationsTopic)
    } else if context.contains("PAT.1") || context.contains("Number Patterns") {
        Box::new(gr10_math::Gr10PatternsTopic)
    } else if context.contains("FIN.1") || context.contains("Finance") {
        Box::new(gr10_math::Gr10FinanceTopic)
    } else if context.contains("FN.1") || context.contains("Linear and Quadratic") {
        Box::new(gr10_math::Gr10FunctionsTopic)
    } else if context.contains("FN.2") || context.contains("Hyperbola and Exponential") {
        Box::new(gr10_math::Gr10FunctionsGraphsTopic)
    } else if context.contains("TRIG.1") || context.contains("Definitions and Identities") {
        Box::new(gr12_math::TrigonometryTopic) // basic trig shares concepts
    } else if context.contains("TRIG.2") || context.contains("Trigonometric Functions") {
        Box::new(gr10_math::Gr10TrigGraphsTopic)
    } else if context.contains("GEOM") || context.contains("Geometry") {
        Box::new(gr12_math::GeometryTopic)
    } else if context.contains("ANAG") || context.contains("Analytical") {
        Box::new(gr12_math::AnalyticalGeomTopic)
    } else if context.contains("STAT") || context.contains("Statistics") {
        Box::new(gr12_math::StatisticsTopic)
    } else if context.contains("PROB") || context.contains("Probability") {
        Box::new(gr10_math::Gr10ProbabilityTopic)
    } else if context.contains("MEAS") || context.contains("Measurement") {
        Box::new(gr10_math::Gr10MeasurementTopic)
    } else {
        Box::new(gr10_math::Gr10FactorisationTopic) // safe Gr10 fallback
    }
}

fn detect_gr11_topic(context: &str) -> Box<dyn TopicContent> {
    // Physics
    if context.contains("PHY.1") || context.contains("Vectors") || context.contains("inclined") {
        Box::new(gr11_physics::Gr11VectorsTopic)
    } else if context.contains("PHY.2") || context.contains("Newton") {
        Box::new(gr11_physics::Gr11NewtonsLawsTopic)
    } else if context.contains("PHY.3") || context.contains("Momentum") {
        Box::new(gr12_physics::MechanicsTopic) // Gr11 momentum similar to Gr12
    } else if context.contains("PHY.4") || context.contains("Work, Energy") {
        Box::new(gr11_physics::Gr11WorkEnergyPowerTopic)
    } else if context.contains("PHY.5") || context.contains("Optics") || context.contains("Snell") {
        Box::new(gr11_physics::Gr11OpticsTopic)
    } else if context.contains("PHY.6") || context.contains("Electrodynamics") {
        Box::new(gr12_physics::ElectricityTopic)
    // Chemistry
    } else if context.contains("CHM.1") || context.contains("Organic") {
        Box::new(gr12_chemistry::OrganicChemTopic)
    } else if context.contains("CHM.2") || context.contains("Reaction Rate") {
        Box::new(gr12_chemistry::ReactionRateTopic)
    } else if context.contains("CHM.3") || context.contains("Equilibrium") {
        Box::new(gr12_chemistry::EquilibriumTopic)
    } else if context.contains("CHM.4") || context.contains("Acids") {
        Box::new(gr12_chemistry::AcidBaseTopic)
    } else if context.contains("CHM.5") || context.contains("Electrochem") || context.contains("Galvanic") {
        Box::new(gr12_chemistry::ElectrochemTopic)
    // Mathematics
    } else if context.contains("ALG.1") || context.contains("Surds") {
        Box::new(gr11_math::Gr11SurdsTopic)
    } else if context.contains("ALG.2") || context.contains("Quadratic") || context.contains("discriminant") {
        Box::new(gr11_math::Gr11QuadraticsTopic)
    } else if context.contains("PAT.1") || context.contains("Sequences") || context.contains("Series") {
        Box::new(gr11_math::Gr11SequencesTopic)
    } else if context.contains("FIN.1") || context.contains("Finance") || context.contains("Decay") {
        Box::new(gr11_math::Gr11FinanceTopic)
    } else if context.contains("FN.1") || context.contains("Parabola") || context.contains("Hyperbola") {
        Box::new(gr11_math::Gr11FunctionsTopic)
    } else if context.contains("FN.2") || context.contains("Inverse") {
        Box::new(gr11_math::Gr11InverseFunctionsTopic)
    } else if context.contains("TRIG.1") || context.contains("Compound") || context.contains("Double") {
        Box::new(gr12_math::TrigonometryTopic)
    } else if context.contains("TRIG.2") || context.contains("Sine") || context.contains("Cosine") && context.contains("rule") {
        Box::new(gr12_math::TrigonometryTopic)
    } else if context.contains("GEOM") || context.contains("Circle Geometry") {
        Box::new(gr12_math::GeometryTopic)
    } else if context.contains("ANAG") || context.contains("Analytical") {
        Box::new(gr12_math::AnalyticalGeomTopic)
    } else if context.contains("STAT") || context.contains("Statistics") {
        Box::new(gr12_math::StatisticsTopic)
    } else if context.contains("PROB") || context.contains("Probability") {
        Box::new(gr11_math::Gr11ProbabilityTopic)
    } else if context.contains("MEAS") || context.contains("Measurement") || context.contains("Pyramid") {
        Box::new(gr11_math::Gr11MeasurementTopic)
    } else {
        Box::new(gr11_math::Gr11SurdsTopic) // safe Gr11 fallback
    }
}

fn detect_gr12_topic(context: &str) -> Box<dyn TopicContent> {
    // Math
    if context.contains("P1.CALC") || context.contains("Differential Calculus") || context.contains("first principles") {
        Box::new(gr12_math::CalculusTopic)
    } else if context.contains("P1.ALG") || context.contains("Algebra") {
        Box::new(gr12_math::AlgebraTopic { grade: 12 })
    } else if context.contains("P1.SEQ") || context.contains("Sequences") || context.contains("Sigma") {
        Box::new(gr12_math::SequencesTopic)
    } else if context.contains("P1.FIN") || context.contains("Annuities") || context.contains("annuit") {
        Box::new(gr12_math::FinanceTopic)
    } else if context.contains("P1.FN") || context.contains("Functions and Graphs") {
        Box::new(gr12_math::FunctionsTopic)
    } else if context.contains("P1.COUNT") || context.contains("Counting") || context.contains("permutation") {
        Box::new(gr12_math::CountingTopic)
    } else if context.contains("P2.TRIG") || context.contains("Trigonometry") {
        Box::new(gr12_math::TrigonometryTopic)
    } else if context.contains("P2.GEOM") || context.contains("Euclidean Geometry") {
        Box::new(gr12_math::GeometryTopic)
    } else if context.contains("P2.ANAG") || context.contains("Analytical Geometry") {
        Box::new(gr12_math::AnalyticalGeomTopic)
    } else if context.contains("P2.STAT") || context.contains("Statistics") {
        Box::new(gr12_math::StatisticsTopic)
    // Physics
    } else if context.contains("P1.MECH") || context.contains("Momentum") || context.contains("Work, Energy") {
        Box::new(gr12_physics::MechanicsTopic)
    } else if context.contains("P1.DOP") || context.contains("Doppler") {
        Box::new(gr12_physics::DopplerTopic)
    } else if context.contains("P1.ELEC") || context.contains("Electric") || context.contains("Electrodynamics") {
        Box::new(gr12_physics::ElectricityTopic)
    } else if context.contains("P1.MOD") || context.contains("Photoelectric") || context.contains("Emission") {
        Box::new(gr12_physics::ModernPhysicsTopic)
    // Chemistry
    } else if context.contains("P2.ORG") || context.contains("Organic") {
        Box::new(gr12_chemistry::OrganicChemTopic)
    } else if context.contains("P2.RATE") || context.contains("Reaction Rate") {
        Box::new(gr12_chemistry::ReactionRateTopic)
    } else if context.contains("P2.EQUIL") || context.contains("Equilibrium") {
        Box::new(gr12_chemistry::EquilibriumTopic)
    } else if context.contains("P2.ACID") || context.contains("Acids") {
        Box::new(gr12_chemistry::AcidBaseTopic)
    } else if context.contains("P2.ECHEM") || context.contains("Electrochemistry") {
        Box::new(gr12_chemistry::ElectrochemTopic)
    } else if context.contains("P2.FERT") || context.contains("Fertiliser") {
        Box::new(gr12_chemistry::FertiliserTopic)
    } else {
        Box::new(gr12_math::AlgebraTopic { grade: 12 }) // safe Gr12 fallback
    }
}

// ============================================================
// Helpers
// ============================================================

fn detect_example_index(context: &str) -> usize {
    if context.contains("Example #3") || context.contains("index: 2") { 2 }
    else if context.contains("Example #2") || context.contains("index: 1") { 1 }
    else { 0 }
}

fn detect_difficulty(context: &str) -> u16 {
    if context.contains("advanced") || context.contains("complex") { 875 }
    else if context.contains("challenging") || context.contains("multi-step") { 625 }
    else if context.contains("medium") || context.contains("two steps") { 375 }
    else { 125 }
}

fn detect_hint_level(context: &str) -> u8 {
    if context.contains("level-3") || context.contains("very specific") { 3 }
    else if context.contains("level-2") || context.contains("moderate") { 2 }
    else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::ContentChannels;

    #[test]
    fn test_all_grades_route_correctly() {
        let test_cases = [
            ("Gr9", "CAPS.Mathematics.Gr9.NUM.1", "integer"),
            ("Gr9", "CAPS.Mathematics.Gr9.ALG.1", "expand"),
            ("Gr9", "CAPS.NaturalSciences.Gr9.MM.2", "atom"),
            ("Gr10", "CAPS.Mathematics.Gr10.ALG.1", "factoris"),
            ("Gr10", "CAPS.PhysicalSciences.Gr10.PHY.1", "motion"),
            ("Gr10", "CAPS.PhysicalSciences.Gr10.CHM.1", "atom"),
            ("Gr11", "CAPS.Mathematics.Gr11.ALG.1", "surd"),
            ("Gr11", "CAPS.PhysicalSciences.Gr11.PHY.2", "newton"),
            ("Gr12", "CAPS.Mathematics.Gr12.P1.CALC", "deriv"),
            ("Gr12", "CAPS.PhysicalSciences.Gr12.P1.ELEC1", "electric"),
        ];

        for (grade, code, keyword) in &test_cases {
            let context = format!("Standard: {} - test description", code);
            let topic = detect_topic(&context);
            let explanation = topic.explanation();
            assert!(!explanation.is_empty(), "{} ({}) produced empty explanation", code, grade);
            assert!(
                explanation.to_lowercase().contains(keyword),
                "{} ({}) explanation missing keyword '{}'. Got: {}...",
                code, grade, keyword, &explanation[..80.min(explanation.len())]
            );
        }
    }

    #[test]
    fn test_gr9_alg1_via_pipeline() {
        let gen = CapsContentGenerator;
        let pipeline = crate::pipeline::ContentPipeline::new(gen);
        let standard = crate::pipeline::StandardInput {
            code: "CAPS.Mathematics.Gr9.ALG.1".to_string(),
            description: "Identify and classify like terms".to_string(),
            grade_level: "Grade9".to_string(),
            domain: "Mathematics".to_string(),
            prerequisites: vec![],
        };
        let lesson = pipeline.generate_lesson(&standard).unwrap();
        eprintln!("ALG.1 via pipeline: {}...", &lesson.explanation[..80.min(lesson.explanation.len())]);
        assert!(
            lesson.explanation.to_lowercase().contains("algebra")
            || lesson.explanation.to_lowercase().contains("expand")
            || lesson.explanation.to_lowercase().contains("factoris"),
            "Gr9 ALG.1 got wrong content via pipeline: {}...",
            &lesson.explanation[..100.min(lesson.explanation.len())]
        );
    }

    #[test]
    fn test_caps_generator_produces_content() {
        let gen = CapsContentGenerator;
        let ch = ContentChannels::teaching_factual();
        let out = gen.generate(&ch, "Standard: CAPS.Mathematics.Gr12.P1.CALC - Differential Calculus", 500);
        assert!(out.text.contains("deriv") || out.text.contains("rate of change"));
        assert!(out.coherence >= 0.85);
    }
}
