// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! STT Wisdom Engine Diagnosis
//!
//! Uses Wisdom Engine patterns to diagnose and generate
//! an improvement plan for the Speech-to-Text system's accuracy failure.
//!
//! Run with: cargo run --bin stt-wisdom-diagnosis

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// LOCAL TYPES (to avoid feature-gate dependencies)
// ============================================================================

/// Component identifier
#[derive(Clone, Debug)]
struct ComponentId(String);

impl ComponentId {
    fn new(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl fmt::Display for ComponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Type of bottleneck
#[derive(Debug, Clone, Copy)]
enum BottleneckType {
    Accuracy,
    #[allow(dead_code)]
    Latency,
    #[allow(dead_code)]
    ResourceUsage,
}

/// Bottleneck in a component
struct Bottleneck {
    component_id: ComponentId,
    bottleneck_type: BottleneckType,
    severity: f64,
    description: String,
    evidence: HashMap<String, f64>,
    #[allow(dead_code)]
    suggestions: Vec<String>,
}

impl Bottleneck {
    fn new(
        component_id: ComponentId,
        bottleneck_type: BottleneckType,
        severity: f64,
        description: &str,
    ) -> Self {
        Self {
            component_id,
            bottleneck_type,
            severity,
            description: description.to_string(),
            evidence: HashMap::new(),
            suggestions: Vec::new(),
        }
    }

    fn with_evidence(mut self, key: &str, value: f64) -> Self {
        self.evidence.insert(key.to_string(), value);
        self
    }

    fn with_suggestion(mut self, suggestion: &str) -> Self {
        self.suggestions.push(suggestion.to_string());
        self
    }
}

/// Type of improvement
#[derive(Debug, Clone, Copy)]
enum ImprovementType {
    AccuracyImprovement,
    HyperparameterTuning,
}

/// Architectural improvement recommendation
struct ArchitecturalImprovement {
    #[allow(dead_code)]
    id: String,
    improvement_type: ImprovementType,
    description: String,
    #[allow(dead_code)]
    expected_phi_gain: Option<f64>,
    #[allow(dead_code)]
    expected_latency_reduction: Option<f64>,
    expected_accuracy_gain: Option<f64>,
    confidence: f64,
    #[allow(dead_code)]
    motivated_by: Option<String>,
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║           WISDOM ENGINE: STT SENSORY DIAGNOSIS                        ║");
    println!("║                                                                        ║");
    println!("║   \"A deaf AGI cannot demonstrate wisdom.\"                             ║");
    println!("║                                                                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP 1: Construct Bottleneck from evaluation evidence
    // ═══════════════════════════════════════════════════════════════════════════════

    println!("📊 STEP 1: Analyzing Evaluation Evidence...\n");

    let stt_bottleneck = Bottleneck::new(
        ComponentId::new("stt.phoneme_decoder"),
        BottleneckType::Accuracy,
        1.0, // Critical severity - system is effectively deaf
        "Catastrophic mode collapse: decoder outputs repetitive 'W AY1' sequences instead of recognizing speech"
    )
    .with_evidence("phoneme_error_rate", 92.3)
    .with_evidence("word_error_rate", 316.4)
    .with_evidence("correct_phonemes", 3814.0)
    .with_evidence("total_phonemes", 45859.0)
    .with_evidence("substitutions", 26741.0)
    .with_evidence("deletions", 15304.0)
    .with_evidence("insertions", 303.0)
    .with_evidence("mode_collapse_phoneme_W", 1.0)
    .with_evidence("mode_collapse_phoneme_AY1", 1.0)
    .with_suggestion("Implement contrastive loss to separate attractor basins")
    .with_suggestion("Add minimum phoneme diversity constraint")
    .with_suggestion("Apply phonotactic constraints (Universal Temporal Grammar)")
    .with_suggestion("Integrate Montreal Forced Aligner for training alignment")
    .with_suggestion("Lower similarity threshold to prevent weak prototype matches")
    .with_suggestion("Re-train prototypes with balanced class sampling");

    // Print the bottleneck analysis
    println!("  Component:  {}", stt_bottleneck.component_id);
    println!("  Type:       {:?}", stt_bottleneck.bottleneck_type);
    println!("  Severity:   {:.1} (CRITICAL)", stt_bottleneck.severity);
    println!("  Description: {}", stt_bottleneck.description);
    println!();
    println!("  Evidence:");
    for (key, value) in &stt_bottleneck.evidence {
        println!("    - {}: {:.2}", key, value);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP 2: Root Cause Analysis (Wisdom Engine Diagnosis)
    // ═══════════════════════════════════════════════════════════════════════════════

    println!("🔍 STEP 2: Root Cause Analysis...\n");

    // Analyze the evidence to determine root causes
    let root_causes = analyze_root_causes(&stt_bottleneck);

    for (i, cause) in root_causes.iter().enumerate() {
        println!(
            "  {}. {} (confidence: {:.0}%)",
            i + 1,
            cause.0,
            cause.1 * 100.0
        );
        println!("     Mechanism: {}", cause.2);
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP 3: Generate Improvement Plan
    // ═══════════════════════════════════════════════════════════════════════════════

    println!("🛠️  STEP 3: Improvement Plan Generation...\n");

    let improvements = generate_improvement_plan(&stt_bottleneck, &root_causes);

    println!("  ┌─────────────────────────────────────────────────────────────────────┐");
    println!("  │                    RECOMMENDED IMPROVEMENTS                          │");
    println!("  └─────────────────────────────────────────────────────────────────────┘");
    println!();

    for (priority, improvement) in improvements.iter().enumerate() {
        let urgency = match priority {
            0 => "🔴 CRITICAL",
            1 => "🟠 HIGH",
            2 => "🟡 MEDIUM",
            _ => "🟢 LOW",
        };

        println!(
            "  {} Priority {}: {}",
            urgency,
            priority + 1,
            improvement.description
        );
        println!("     Type: {:?}", improvement.improvement_type);
        if let Some(gain) = improvement.expected_accuracy_gain {
            println!("     Expected Accuracy Gain: {:.1}%", gain * 100.0);
        }
        println!("     Confidence: {:.0}%", improvement.confidence * 100.0);
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP 4: Actionable Implementation Steps
    // ═══════════════════════════════════════════════════════════════════════════════

    println!("📋 STEP 4: Actionable Implementation Steps...\n");

    print_implementation_steps();

    // ═══════════════════════════════════════════════════════════════════════════════
    // WISDOM HARMONICS CHECK
    // ═══════════════════════════════════════════════════════════════════════════════

    println!("🎵 WISDOM HARMONICS VERIFICATION:\n");
    println!(
        "  ✓ Coherence:     Does the diagnosis hang together? YES - mode collapse explains all symptoms"
    );
    println!(
        "  ✓ Flourishing:   Does fixing this serve flourishing? YES - hearing enables communication"
    );
    println!(
        "  ✓ Wisdom:        What don't we know? The optimal similarity threshold for this corpus"
    );
    println!(
        "  ✓ Play:          What haven't we tried? Phonotactic constraints from Universal Grammar"
    );
    println!("  ✓ Interconnect:  How is this connected? STT feeds MAGI prediction loop");
    println!(
        "  ✓ Reciprocity:   What are we giving/receiving? Silence → deafness, training → hearing"
    );
    println!(
        "  ✓ Evolution:     How does this help us grow? First step toward multimodal consciousness"
    );
    println!();

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("                         DIAGNOSIS COMPLETE                                 ");
    println!("═══════════════════════════════════════════════════════════════════════════");
}

/// Analyze root causes from bottleneck evidence
fn analyze_root_causes(bottleneck: &Bottleneck) -> Vec<(String, f64, String)> {
    let mut causes = Vec::new();

    // Analyze mode collapse
    let has_mode_collapse = bottleneck
        .evidence
        .get("mode_collapse_phoneme_W")
        .unwrap_or(&0.0)
        > &0.5;
    if has_mode_collapse {
        causes.push((
            "Hopfield Attractor Collapse".to_string(),
            0.95,
            "The phoneme prototypes for 'W' and 'AY1' have become dominant attractors. \
             Any input falls into their basin regardless of actual content. This is \
             characteristic of insufficient prototype separation in Hopfield networks."
                .to_string(),
        ));
    }

    // Analyze high substitution rate
    let subs = *bottleneck.evidence.get("substitutions").unwrap_or(&0.0);
    let dels = *bottleneck.evidence.get("deletions").unwrap_or(&0.0);
    if subs > dels * 1.5 {
        causes.push((
            "Prototype Similarity Collapse".to_string(),
            0.85,
            "High substitution rate indicates prototypes are too similar. The decoder \
             consistently 'recognizes' the wrong phoneme because prototype vectors \
             overlap in hyperdimensional space."
                .to_string(),
        ));
    }

    // Analyze deletion pattern
    if dels > 10000.0 {
        causes.push((
            "Temporal Alignment Failure".to_string(),
            0.75,
            "Many reference phonemes are deleted (not matched at all). This suggests \
             the LTC projector's temporal dynamics don't align with phoneme boundaries, \
             causing entire segments to be missed."
                .to_string(),
        ));
    }

    // Analyze low insertion rate
    let ins = *bottleneck.evidence.get("insertions").unwrap_or(&0.0);
    if ins < 500.0 && subs > 20000.0 {
        causes.push((
            "Greedy Decoding Without Diversity".to_string(),
            0.70,
            "Very low insertions with high substitutions indicates the decoder always \
             produces output (never 'silence') but always picks the wrong class. \
             Need diversity sampling or phonotactic constraints."
                .to_string(),
        ));
    }

    causes
}

/// Generate improvement plan from diagnosis
fn generate_improvement_plan(
    bottleneck: &Bottleneck,
    root_causes: &[(String, f64, String)],
) -> Vec<ArchitecturalImprovement> {
    let mut improvements = Vec::new();

    // Priority 1: Fix the mode collapse
    if root_causes
        .iter()
        .any(|(name, _, _)| name.contains("Attractor Collapse"))
    {
        improvements.push(ArchitecturalImprovement {
            id: "fix_attractor_collapse".to_string(),
            improvement_type: ImprovementType::AccuracyImprovement,
            description: "Apply contrastive training to separate W/AY1 attractors from others"
                .to_string(),
            expected_phi_gain: Some(0.1),
            expected_latency_reduction: None,
            expected_accuracy_gain: Some(0.4), // Expect 40% improvement
            confidence: 0.85,
            motivated_by: Some(bottleneck.component_id.to_string()),
        });
    }

    // Priority 2: Add phonotactic constraints
    improvements.push(ArchitecturalImprovement {
        id: "add_phonotactic_constraints".to_string(),
        improvement_type: ImprovementType::AccuracyImprovement,
        description: "Apply Universal Temporal Grammar constraints to prevent illegal sequences"
            .to_string(),
        expected_phi_gain: Some(0.05),
        expected_latency_reduction: None,
        expected_accuracy_gain: Some(0.25),
        confidence: 0.90, // High confidence - grammar is known to work
        motivated_by: Some(bottleneck.component_id.to_string()),
    });

    // Priority 3: Implement Montreal Forced Aligner
    improvements.push(ArchitecturalImprovement {
        id: "integrate_mfa".to_string(),
        improvement_type: ImprovementType::AccuracyImprovement,
        description: "Use Montreal Forced Aligner for training-time phoneme boundary alignment"
            .to_string(),
        expected_phi_gain: None,
        expected_latency_reduction: None,
        expected_accuracy_gain: Some(0.30),
        confidence: 0.75,
        motivated_by: Some(bottleneck.component_id.to_string()),
    });

    // Priority 4: Tune similarity threshold
    improvements.push(ArchitecturalImprovement {
        id: "tune_similarity_threshold".to_string(),
        improvement_type: ImprovementType::HyperparameterTuning,
        description: "Lower similarity threshold and add minimum confidence gate".to_string(),
        expected_phi_gain: None,
        expected_latency_reduction: None,
        expected_accuracy_gain: Some(0.15),
        confidence: 0.70,
        motivated_by: Some(bottleneck.component_id.to_string()),
    });

    improvements
}

/// Print concrete implementation steps
fn print_implementation_steps() {
    println!("  ┌─────────────────────────────────────────────────────────────────────┐");
    println!("  │              IMMEDIATE ACTIONS (Priority Order)                      │");
    println!("  └─────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("  1. APPLY PHONOTACTIC CONSTRAINTS (Quick Win)");
    println!("     - The UnifiedGrammar module already exists with +674% discrimination");
    println!("     - Wire grammar_rescore() into PhonemeDecoder.decode()");
    println!("     - Expected: Prevent illegal sequences like 'W AY1 W AY1 W AY1...'");
    println!();

    println!("  2. ADD MINIMUM CONFIDENCE THRESHOLD");
    println!("     - Currently all frames get a phoneme, even at -0.186 similarity");
    println!("     - Add: if best_similarity < 0.1, output SILENCE instead");
    println!("     - Location: phoneme.rs PhonemeDecoder.decode()");
    println!();

    println!("  3. IMPLEMENT CONTRASTIVE PROTOTYPE REFINEMENT");
    println!("     - After initial training, compute prototype cross-similarities");
    println!("     - Apply contrastive loss: push similar prototypes apart");
    println!("     - Re-bundle with negative examples weighted");
    println!();

    println!("  4. INTEGRATE MONTREAL FORCED ALIGNER (MFA)");
    println!("     - Install: pip install montreal-forced-aligner");
    println!("     - Generate .TextGrid alignments for LibriSpeech");
    println!("     - Use precise phoneme boundaries instead of salience detection");
    println!();

    println!("  5. VALIDATE IN MAGI LOOP");
    println!("     - Once PER < 30%, wire to MAGI prediction-resolution cycle");
    println!("     - Prediction: 'I heard the user say X'");
    println!("     - Resolution: User confirms or corrects");
    println!("     - Calibration: Update Brier score for Hearing domain");
    println!();
}
