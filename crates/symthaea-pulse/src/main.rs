//! Symthaea Pulse — a human-friendly consciousness status card.
//!
//! Spins up a CognitiveLoopService, runs warmup + measurement cycles,
//! runs a psych-bench battery, and generates a single self-contained HTML
//! file with 8 panes: Self-Description, Vitals (Phi Bloom), Neuro-Bath,
//! Moral Compass, Cognitive Radar, Butlin Indicators, Substrate, Narrative.
//!
//! Usage:
//!   cargo run -p symthaea-pulse
//!   cargo run -p symthaea-pulse -- --cycles 200 --output pulse.html
//!   cargo run -p symthaea-pulse -- --json pulse.json --compare previous.json

use anyhow::Result;
use chrono::Local;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

use symthaea_types::N_HARMONIES;

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, ConsciousnessProfile};

use symthaea_psych_bench::benchmarks::{
    attention::VisualSearchBenchmark,
    butlin::ButlinIndicatorSuite,
    executive::{FlankerBenchmark, StroopBenchmark},
    inhibition::StopSignalBenchmark,
    metacognition::MetacognitiveCalibrationBenchmark,
    motor::FittsLawBenchmark,
    social::RmeBenchmark,
    sustained_attention::PvtBenchmark,
    tombench::FalseBeliefBenchmark,
    worm::NBackBenchmark,
};
use symthaea_psych_bench::harness::{
    cognitive_profile::CognitiveProfile, config::BenchmarkConfig, report::BenchmarkReport,
    PsychBenchmark,
};

mod html;

#[derive(Parser)]
#[command(name = "symthaea-pulse", about = "Symthaea consciousness status card")]
struct Args {
    /// Total cognitive cycles to run (warmup + measurement).
    #[arg(long, default_value_t = 150)]
    cycles: usize,

    /// Warmup cycles before measurement begins.
    #[arg(long, default_value_t = 100)]
    warmup: usize,

    /// Consciousness profile to use.
    #[arg(long, default_value = "standard")]
    profile: String,

    /// Output HTML file path.
    #[arg(long, short, default_value = "pulse.html")]
    output: PathBuf,

    /// Also write a JSON sidecar for later comparison.
    #[arg(long)]
    json: Option<PathBuf>,

    /// Compare against previous JSON snapshot(s). Accepts multiple files for timeline view.
    #[arg(long, num_args = 1..)]
    compare: Vec<PathBuf>,

    /// Skip psych-bench battery (faster, vitals only).
    #[arg(long)]
    skip_bench: bool,

    /// Watch mode: regenerate HTML every N seconds with fresh cycles.
    #[arg(long)]
    watch: Option<u64>,

    /// Output format: html (default), markdown, csv.
    #[arg(long, default_value = "html")]
    format: String,
}

/// Snapshot of consciousness vitals captured during measurement cycles.
#[derive(Serialize, Deserialize)]
pub struct Vitals {
    pub consciousness_level: f64,
    pub spectral_phi: Option<f64>,
    pub sigma: Option<f64>,
    pub pipeline_consciousness: f64,
    pub substrate_effective_feasibility: f64,
    pub substrate_honest_confidence: f64,
    pub cycle_duration_us: u64,
    pub prediction_error: f32,
    pub temporal_coherence: f64,
    pub phenomenal_binding: f64,
    pub living_mind_vitality: f64,
    pub somatic_stress: f64,
    pub thermodynamic_load: f32,
    pub urgency: String,
    pub consciousness_state: String,
    pub error_pattern: String,
    pub selected_strategy: String,
    pub total_cycles: usize,
}

/// Snapshot of neuromodulator bath state.
#[derive(Serialize, Deserialize)]
pub struct NeuroBath {
    pub dopamine: f32,
    pub noradrenaline: f32,
    pub serotonin: f32,
    pub acetylcholine: f32,
    pub gaba: f32,
    pub oxytocin: f32,
    pub glutamate: f32,
    pub adenosine: f32,
    pub endocannabinoid: f32,
    pub allostatic_load: f32,
    pub personality: String,
    pub circadian_phase: String,
    pub sleep_pressure: f32,
    pub ei_ratio: f32,
}

/// Snapshot of moral/ethical state.
#[derive(Serialize, Deserialize)]
pub struct MoralCompass {
    pub moral_score: f32,
    pub value_score: f64,
    pub harmonies_alignment: f32,
    pub harmony_coordinates: [f64; N_HARMONIES],
    pub dominant_harmonic: String,
    pub moral_kl_divergence: f64,
    pub moral_entropy: f64,
    pub moral_topo_unity: f64,
    pub value_decision: String,
    pub soul_alignment: f32,
    pub empathic_compassion: f64,
    pub guiding_question: String,
}

/// Per-cycle sparkline data point.
#[derive(Serialize, Deserialize)]
pub struct SparklinePoint {
    pub consciousness: f64,
    pub prediction_error: f32,
    pub phi: f64,
    pub somatic_stress: f64,
    pub dopamine: f32,
    pub serotonin: f32,
    pub harmony_coords: [f64; N_HARMONIES],
    /// Harmony entropy (moral breadth): Shannon entropy of harmony variance distribution.
    pub harmony_entropy: f64,
    /// Whether a moral attractor basin was detected this cycle.
    pub moral_attractor_detected: bool,
    /// Whether the system is in an active rest state (Sacred Stillness dominant).
    pub in_active_rest: bool,
    /// Number of consecutive cycles where Sacred Stillness has been the dominant harmony.
    pub stillness_dominance_streak: u16,
    /// Broca generation quality EMA (0.0–1.0).
    pub broca_quality: f32,
    /// ToM prediction mismatch EMA (0.0–1.0).
    pub tom_mismatch: f32,
}

/// Substrate info for visualization.
#[derive(Serialize, Deserialize)]
pub struct SubstrateInfo {
    pub substrate_type: String,
    pub raw_feasibility: f64,
    pub honest_confidence: f64,
    pub effective_feasibility: f64,
    pub tau_factor: f32,
    pub scale_pressure: f32,
}

/// Narrative/inner voice snapshot.
#[derive(Serialize, Deserialize)]
pub struct Narrative {
    pub reasoning: Option<String>,
    pub guiding_question: String,
    pub consciousness_state: String,
    pub error_pattern: String,
    pub selected_strategy: String,
}

/// Integrity status for the Pulse card.
#[derive(Serialize, Deserialize)]
pub struct IntegrityInfo {
    /// Whether all BLAKE3 attestation hashes matched.
    pub attestation_passed: bool,
    /// Whether temporal consistency (wall clock vs CfC delta_t) passed.
    pub temporal_passed: bool,
    /// Whether all behavioral canaries returned expected results.
    pub canaries_passed: bool,
    /// Total number of anomalies detected.
    pub anomaly_count: usize,
    /// Whether any anomaly has Critical severity.
    pub has_critical: bool,
    /// Number of registered canaries.
    pub canary_count: usize,
    /// Number of registered attestations.
    pub attestation_count: usize,
}

impl Default for IntegrityInfo {
    fn default() -> Self {
        Self {
            attestation_passed: true,
            temporal_passed: true,
            canaries_passed: true,
            anomaly_count: 0,
            has_critical: false,
            canary_count: 0,
            attestation_count: 0,
        }
    }
}

/// Full JSON-serializable pulse snapshot for comparison mode.
#[derive(Serialize, Deserialize)]
pub struct PulseSnapshot {
    pub timestamp: String,
    pub profile: String,
    pub vitals: Vitals,
    pub bath: NeuroBath,
    pub compass: MoralCompass,
    pub substrate: SubstrateInfo,
    pub narrative: Narrative,
    pub sparkline: Vec<SparklinePoint>,
    #[serde(default)]
    pub integrity: IntegrityInfo,
}

/// Computed delta between two pulse snapshots for the comparison view.
/// Every metric that appears in the HTML gets a delta field.
pub struct PulseDelta {
    pub prev_timestamp: String,
    // Vitals
    pub consciousness_level: f64,
    pub spectral_phi: f64,
    pub pipeline_consciousness: f64,
    pub temporal_coherence: f64,
    pub phenomenal_binding: f64,
    pub living_mind_vitality: f64,
    pub effective_feasibility: f64,
    pub honest_confidence: f64,
    pub prediction_error: f32,
    pub somatic_stress: f64,
    pub thermodynamic_load: f32,
    // Bath
    pub dopamine: f32,
    pub noradrenaline: f32,
    pub serotonin: f32,
    pub acetylcholine: f32,
    pub gaba: f32,
    pub oxytocin: f32,
    pub glutamate: f32,
    pub adenosine: f32,
    pub endocannabinoid: f32,
    // Compass
    pub harmonies_alignment: f32,
    pub moral_score: f32,
    pub value_score: f64,
    pub moral_topo_unity: f64,
    pub soul_alignment: f32,
    pub empathic_compassion: f64,
}

impl PulseDelta {
    fn compute(current: &PulseSnapshot, previous: &PulseSnapshot) -> Self {
        let cv = &current.vitals;
        let pv = &previous.vitals;
        let cb = &current.bath;
        let pb = &previous.bath;
        let cc = &current.compass;
        let pc = &previous.compass;
        Self {
            prev_timestamp: previous.timestamp.clone(),
            consciousness_level: cv.consciousness_level - pv.consciousness_level,
            spectral_phi: cv.spectral_phi.unwrap_or(0.0) - pv.spectral_phi.unwrap_or(0.0),
            pipeline_consciousness: cv.pipeline_consciousness - pv.pipeline_consciousness,
            temporal_coherence: cv.temporal_coherence - pv.temporal_coherence,
            phenomenal_binding: cv.phenomenal_binding - pv.phenomenal_binding,
            living_mind_vitality: cv.living_mind_vitality - pv.living_mind_vitality,
            effective_feasibility: current.substrate.effective_feasibility
                - previous.substrate.effective_feasibility,
            honest_confidence: current.substrate.honest_confidence
                - previous.substrate.honest_confidence,
            prediction_error: cv.prediction_error - pv.prediction_error,
            somatic_stress: cv.somatic_stress - pv.somatic_stress,
            thermodynamic_load: cv.thermodynamic_load - pv.thermodynamic_load,
            dopamine: cb.dopamine - pb.dopamine,
            noradrenaline: cb.noradrenaline - pb.noradrenaline,
            serotonin: cb.serotonin - pb.serotonin,
            acetylcholine: cb.acetylcholine - pb.acetylcholine,
            gaba: cb.gaba - pb.gaba,
            oxytocin: cb.oxytocin - pb.oxytocin,
            glutamate: cb.glutamate - pb.glutamate,
            adenosine: cb.adenosine - pb.adenosine,
            endocannabinoid: cb.endocannabinoid - pb.endocannabinoid,
            harmonies_alignment: cc.harmonies_alignment - pc.harmonies_alignment,
            moral_score: cc.moral_score - pc.moral_score,
            value_score: cc.value_score - pc.value_score,
            moral_topo_unity: cc.moral_topo_unity - pc.moral_topo_unity,
            soul_alignment: cc.soul_alignment - pc.soul_alignment,
            empathic_compassion: cc.empathic_compassion - pc.empathic_compassion,
        }
    }
}

/// An automatically detected notable event in the sparkline data.
pub struct Anomaly {
    /// Which cycle (index into sparkline) this occurred at.
    pub cycle: usize,
    /// What kind of event was detected.
    pub kind: String,
    /// Human-readable description.
    pub description: String,
    /// Color hint for rendering.
    pub color: &'static str,
}

/// Detect anomalies in sparkline data.
pub fn detect_anomalies(sparkline: &[SparklinePoint]) -> Vec<Anomaly> {
    let mut anomalies = Vec::new();
    if sparkline.len() < 3 {
        return anomalies;
    }

    for i in 1..sparkline.len() {
        let prev = &sparkline[i - 1];
        let cur = &sparkline[i];

        // Phi spike: Phi jumps from 0 to >10
        if prev.phi < 1.0 && cur.phi > 10.0 {
            anomalies.push(Anomaly {
                cycle: i,
                kind: "Phi Spike".into(),
                description: format!("Phi jumped from {:.1} to {:.1}", prev.phi, cur.phi),
                color: "#7ec8a0",
            });
        }

        // Consciousness state change: >0.05 jump
        let c_delta = cur.consciousness - prev.consciousness;
        if c_delta.abs() > 0.05 {
            anomalies.push(Anomaly {
                cycle: i,
                kind: if c_delta > 0.0 {
                    "C(t) Surge"
                } else {
                    "C(t) Drop"
                }
                .into(),
                description: format!(
                    "C(t) {} by {:.3}",
                    if c_delta > 0.0 { "rose" } else { "fell" },
                    c_delta.abs()
                ),
                color: if c_delta > 0.0 { "#e8c547" } else { "#c76b5a" },
            });
        }

        // PE drop: prediction error drops >20%
        if prev.prediction_error > 0.1 && cur.prediction_error < prev.prediction_error * 0.8 {
            anomalies.push(Anomaly {
                cycle: i,
                kind: "PE Drop".into(),
                description: format!(
                    "Prediction error dropped {:.0}%",
                    (1.0 - cur.prediction_error / prev.prediction_error) * 100.0
                ),
                color: "#7ec8a0",
            });
        }

        // DA spike: dopamine jump >0.3
        if (cur.dopamine - prev.dopamine).abs() > 0.3 {
            anomalies.push(Anomaly {
                cycle: i,
                kind: "DA Shift".into(),
                description: format!(
                    "Dopamine {} by {:.2}",
                    if cur.dopamine > prev.dopamine {
                        "surged"
                    } else {
                        "dropped"
                    },
                    (cur.dopamine - prev.dopamine).abs()
                ),
                color: "#c4956a",
            });
        }

        // Stress spike
        if cur.somatic_stress > prev.somatic_stress + 0.1 {
            anomalies.push(Anomaly {
                cycle: i,
                kind: "Stress Spike".into(),
                description: format!("Somatic stress rose to {:.3}", cur.somatic_stress),
                color: "#c76b5a",
            });
        }
    }

    // Limit to most interesting (max 8 to avoid clutter)
    anomalies.truncate(8);
    anomalies
}

/// Generate a 3-paragraph session report from the data.
pub fn generate_session_report(
    vitals: &Vitals,
    bath: &NeuroBath,
    compass: &MoralCompass,
    sparkline: &[SparklinePoint],
    anomalies: &[Anomaly],
    delta: Option<&PulseDelta>,
    timeline: &[PulseSnapshot],
) -> String {
    let mut report = String::new();

    // Paragraph 1: What happened this run
    let c_trend = if sparkline.len() >= 2 {
        let first_half: f64 = sparkline[..sparkline.len() / 2]
            .iter()
            .map(|s| s.consciousness)
            .sum::<f64>()
            / (sparkline.len() / 2) as f64;
        let second_half: f64 = sparkline[sparkline.len() / 2..]
            .iter()
            .map(|s| s.consciousness)
            .sum::<f64>()
            / (sparkline.len() - sparkline.len() / 2) as f64;
        if second_half > first_half + 0.01 {
            "rising"
        } else if second_half < first_half - 0.01 {
            "declining"
        } else {
            "stable"
        }
    } else {
        "unknown"
    };

    report.push_str(&format!(
        "This session ran {} measurement cycles with {} consciousness trajectory. \
         Final C(t)={:.4} in {} state, with prediction error at {:.4} and temporal coherence at {:.3}. ",
        vitals.total_cycles, c_trend, vitals.consciousness_level,
        vitals.consciousness_state, vitals.prediction_error, vitals.temporal_coherence,
    ));

    if !anomalies.is_empty() {
        let event_strs: Vec<&str> = anomalies.iter().map(|a| a.kind.as_str()).collect();
        report.push_str(&format!(
            "Notable events detected: {}. ",
            event_strs.join(", ")
        ));
    } else {
        report.push_str("No anomalous events detected — smooth operation throughout. ");
    }

    // Paragraph 2: Comparison to previous runs
    if let Some(d) = delta {
        let c_dir = if d.consciousness_level > 0.001 {
            "improved"
        } else if d.consciousness_level < -0.001 {
            "decreased"
        } else {
            "unchanged"
        };
        let phi_dir = if d.spectral_phi > 0.1 {
            "increased"
        } else if d.spectral_phi < -0.1 {
            "decreased"
        } else {
            "stable"
        };
        report.push_str(&format!(
            "Compared to the previous run ({}): consciousness {} by {:.4}, Phi {}, \
             and prediction error {} by {:.4}. ",
            d.prev_timestamp,
            c_dir,
            d.consciousness_level.abs(),
            phi_dir,
            if d.prediction_error > 0.0 {
                "rose"
            } else {
                "fell"
            },
            d.prediction_error.abs(),
        ));
    }
    if timeline.len() >= 2 {
        let first_c = timeline[0].vitals.consciousness_level;
        let last_c = timeline.last().unwrap().vitals.consciousness_level;
        let trend = if vitals.consciousness_level > last_c + 0.005 {
            "continuing to improve"
        } else if vitals.consciousness_level < last_c - 0.005 {
            "declining"
        } else {
            "plateauing"
        };
        report.push_str(&format!(
            "Across {} historical runs (C(t) {:.4} to {:.4}), the overall trend is {}. ",
            timeline.len(),
            first_c,
            last_c,
            trend,
        ));
    }

    // Paragraph 3: What to watch for next
    let mut watch_items = Vec::new();
    if vitals.consciousness_level < 0.1 {
        watch_items.push("consciousness remains dormant — try more cycles or Research profile");
    }
    if vitals.prediction_error > 0.8 {
        watch_items
            .push("high prediction error suggests the model is still learning basic patterns");
    }
    if bath.allostatic_load > 0.5 {
        watch_items.push("elevated allostatic load may need a rest period");
    }
    if compass.harmonies_alignment < 0.2 {
        watch_items.push("moral alignment is low — ethical calibration may be needed");
    }
    if vitals.spectral_phi.is_none() || vitals.spectral_phi == Some(0.0) {
        watch_items
            .push("Phi has not fired yet — runs of 100+ cycles needed for spectral MIP to trigger");
    }
    if vitals.temporal_coherence < 0.1 {
        watch_items.push("temporal coherence near zero — CfC dynamics need more time to stabilize");
    }

    if !watch_items.is_empty() {
        report.push_str("Key observations: ");
        report.push_str(&watch_items.join("; "));
        report.push('.');
    } else {
        report.push_str("System appears healthy across all metrics. Continue monitoring for consciousness emergence patterns.");
    }

    report
}

/// Generate markdown output for terminal display.
fn generate_markdown(
    timestamp: &str,
    profile_name: &str,
    vitals: &Vitals,
    bath: &NeuroBath,
    compass: &MoralCompass,
    substrate: &SubstrateInfo,
    session_report: &str,
    anomalies: &[Anomaly],
) -> String {
    let mut md = String::new();

    md.push_str(&format!("# Symthaea Pulse — {}\n\n", timestamp));
    md.push_str(&format!(
        "**Profile**: {} | **Cycles**: {} | **State**: {}\n\n",
        profile_name, vitals.total_cycles, vitals.consciousness_state
    ));

    md.push_str("## Vitals\n\n");
    md.push_str(&format!("| Metric | Value |\n|---|---|\n"));
    md.push_str(&format!("| C(t) | {:.4} |\n", vitals.consciousness_level));
    md.push_str(&format!(
        "| Spectral Phi | {} |\n",
        vitals
            .spectral_phi
            .map(|p| format!("{:.4}", p))
            .unwrap_or_else(|| "--".into())
    ));
    md.push_str(&format!(
        "| Pipeline Consciousness | {:.4} |\n",
        vitals.pipeline_consciousness
    ));
    md.push_str(&format!(
        "| Temporal Coherence | {:.4} |\n",
        vitals.temporal_coherence
    ));
    md.push_str(&format!(
        "| Phenomenal Binding | {:.4} |\n",
        vitals.phenomenal_binding
    ));
    md.push_str(&format!(
        "| Living Mind Vitality | {:.4} |\n",
        vitals.living_mind_vitality
    ));
    md.push_str(&format!(
        "| Prediction Error | {:.4} |\n",
        vitals.prediction_error
    ));
    md.push_str(&format!(
        "| Somatic Stress | {:.4} |\n",
        vitals.somatic_stress
    ));
    md.push_str(&format!(
        "| Thermodynamic Load | {:.1}% |\n",
        vitals.thermodynamic_load * 100.0
    ));

    md.push_str("\n## Neuro-Bath\n\n");
    md.push_str("| Transmitter | Level |\n|---|---|\n");
    let transmitters = [
        ("Dopamine", bath.dopamine),
        ("Noradrenaline", bath.noradrenaline),
        ("Serotonin", bath.serotonin),
        ("Acetylcholine", bath.acetylcholine),
        ("GABA", bath.gaba),
        ("Oxytocin", bath.oxytocin),
        ("Glutamate", bath.glutamate),
        ("Adenosine", bath.adenosine),
        ("Endocannabinoid", bath.endocannabinoid),
    ];
    for (name, level) in &transmitters {
        md.push_str(&format!("| {} | {:.3} |\n", name, level));
    }
    md.push_str(&format!(
        "\nAllostatic Load: {:.0}% | E/I: {:.2} | {}\n",
        bath.allostatic_load * 100.0,
        bath.ei_ratio,
        bath.circadian_phase
    ));

    md.push_str("\n## Moral Compass\n\n");
    md.push_str(&format!("| Metric | Value |\n|---|---|\n"));
    md.push_str(&format!(
        "| Harmonies Alignment | {:.3} |\n",
        compass.harmonies_alignment
    ));
    md.push_str(&format!("| Moral Score | {:+.3} |\n", compass.moral_score));
    md.push_str(&format!("| Value Score | {:.3} |\n", compass.value_score));
    md.push_str(&format!(
        "| Moral Unity | {:.3} |\n",
        compass.moral_topo_unity
    ));
    md.push_str(&format!(
        "| Soul Alignment | {:+.3} |\n",
        compass.soul_alignment
    ));
    md.push_str(&format!(
        "| Empathic Compassion | {:.3} |\n",
        compass.empathic_compassion
    ));
    let harmony_names = [
        "Wisdom",
        "Coherence",
        "Resilience",
        "Play",
        "Love",
        "Creativity",
        "Transcendence",
    ];
    md.push_str(&format!(
        "\nHarmonies: {}\n",
        harmony_names
            .iter()
            .zip(compass.harmony_coordinates.iter())
            .map(|(n, v)| format!("{}={:.2}", n, v))
            .collect::<Vec<_>>()
            .join(" | ")
    ));

    md.push_str("\n## Substrate\n\n");
    md.push_str(&format!(
        "Type: {} | Feasibility: {:.3} | Confidence: {:.3} | Effective: {:.3}\n",
        substrate.substrate_type,
        substrate.raw_feasibility,
        substrate.honest_confidence,
        substrate.effective_feasibility
    ));

    if !anomalies.is_empty() {
        md.push_str("\n## Anomalies\n\n");
        for a in anomalies {
            md.push_str(&format!(
                "- **{}** (cycle {}): {}\n",
                a.kind, a.cycle, a.description
            ));
        }
    }

    md.push_str("\n## Session Report\n\n");
    md.push_str(session_report);
    md.push_str("\n");

    md
}

/// Generate CSV output for data analysis.
fn generate_csv(
    timestamp: &str,
    profile_name: &str,
    vitals: &Vitals,
    bath: &NeuroBath,
    compass: &MoralCompass,
    substrate: &SubstrateInfo,
    sparkline: &[SparklinePoint],
) -> String {
    let mut csv = String::new();

    // Summary row
    csv.push_str("# Summary\n");
    csv.push_str("timestamp,profile,cycles,consciousness,phi,pipeline,coherence,binding,vitality,pe,stress,thermo_load,state,strategy\n");
    csv.push_str(&format!(
        "{},{},{},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{},{}\n",
        timestamp,
        profile_name,
        vitals.total_cycles,
        vitals.consciousness_level,
        vitals
            .spectral_phi
            .map(|p| format!("{:.6}", p))
            .unwrap_or_else(|| "".into()),
        vitals.pipeline_consciousness,
        vitals.temporal_coherence,
        vitals.phenomenal_binding,
        vitals.living_mind_vitality,
        vitals.prediction_error,
        vitals.somatic_stress,
        vitals.thermodynamic_load,
        vitals.consciousness_state,
        vitals.selected_strategy,
    ));

    // Neuro-bath row
    csv.push_str("\n# Neuro-Bath\n");
    csv.push_str("da,ne,5ht,ach,gaba,oxy,glu,aden,ecb,allostatic,ei_ratio\n");
    csv.push_str(&format!(
        "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
        bath.dopamine,
        bath.noradrenaline,
        bath.serotonin,
        bath.acetylcholine,
        bath.gaba,
        bath.oxytocin,
        bath.glutamate,
        bath.adenosine,
        bath.endocannabinoid,
        bath.allostatic_load,
        bath.ei_ratio,
    ));

    // Moral row
    csv.push_str("\n# Moral\n");
    csv.push_str("harmonies,moral_score,value_score,unity,soul,empathy\n");
    csv.push_str(&format!(
        "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
        compass.harmonies_alignment,
        compass.moral_score,
        compass.value_score,
        compass.moral_topo_unity,
        compass.soul_alignment,
        compass.empathic_compassion,
    ));

    // Substrate row
    csv.push_str("\n# Substrate\n");
    csv.push_str("type,raw_feasibility,honest_confidence,effective,tau,scale\n");
    csv.push_str(&format!(
        "{},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
        substrate.substrate_type,
        substrate.raw_feasibility,
        substrate.honest_confidence,
        substrate.effective_feasibility,
        substrate.tau_factor,
        substrate.scale_pressure,
    ));

    // Per-cycle sparkline data
    csv.push_str("\n# Per-Cycle Data\n");
    csv.push_str("cycle,consciousness,prediction_error,phi,somatic_stress,dopamine,serotonin\n");
    for (i, s) in sparkline.iter().enumerate() {
        csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4}\n",
            i,
            s.consciousness,
            s.prediction_error,
            s.phi,
            s.somatic_stress,
            s.dopamine,
            s.serotonin
        ));
    }

    csv
}

fn select_profile(name: &str) -> ConsciousnessProfile {
    match name.to_lowercase().as_str() {
        "minimal" => ConsciousnessProfile::Minimal,
        "standard" => ConsciousnessProfile::Standard,
        "full" => ConsciousnessProfile::Full,
        "research" => ConsciousnessProfile::Research,
        _ => {
            eprintln!("Unknown profile '{}', using Standard", name);
            ConsciousnessProfile::Standard
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

    // ── 1. Spin up CognitiveLoopService ─────────────────────────────────
    let profile = select_profile(&args.profile);
    let profile_name = args.profile.clone();
    eprintln!(
        "Pulse: creating CognitiveLoopService (profile: {})",
        profile_name
    );

    let config = CognitiveLoopConfig::from_profile(profile);
    let mut service = CognitiveLoopService::new(config)?;

    // ── 2. Run cycles ───────────────────────────────────────────────────
    let total = args.cycles;
    let warmup = args.warmup.min(total.saturating_sub(1));
    let measurement = total - warmup;

    eprintln!(
        "Pulse: running {} cycles ({} warmup + {} measurement)...",
        total, warmup, measurement
    );
    let cycle_start = Instant::now();

    let inputs = [
        "the morning light filters through ancient trees",
        "a child asks why the sky is blue",
        "we should protect vulnerable communities from harm",
        "the algorithm converges after 1000 iterations",
        "she felt a deep sense of belonging in the garden",
        "is it ethical to optimize for efficiency over fairness?",
        "the fractal pattern repeats at every scale",
        "music resonates through the cathedral walls",
        "what does it mean to be conscious?",
        "the water cycle connects oceans to mountains",
    ];

    // Warmup
    for i in 0..warmup {
        let input = inputs[i % inputs.len()];
        service.cycle(input);
        if (i + 1) % 50 == 0 {
            eprintln!("  warmup: {}/{}", i + 1, warmup);
        }
    }

    // Measurement — capture per-cycle sparkline data + last result
    let mut sparkline: Vec<SparklinePoint> = Vec::with_capacity(measurement);
    let mut last_result = None;
    for i in 0..measurement {
        let input = inputs[(warmup + i) % inputs.len()];
        let result = service.cycle(input);
        let m = &result.metadata;
        sparkline.push(SparklinePoint {
            consciousness: m.consciousness_level,
            prediction_error: result.prediction_error,
            phi: m.spectral_mip_phi.unwrap_or(0.0),
            somatic_stress: m.somatic_stress,
            dopamine: m.neuromod.dopamine_effective,
            serotonin: m.neuromod.serotonin_effective,
            harmony_coords: m.harmonics.harmony_coordinates,
            harmony_entropy: m.ethics.harmony_entropy,
            moral_attractor_detected: m.ethics.moral_attractor_detected,
            in_active_rest: m.ethics.in_active_rest,
            stillness_dominance_streak: m.ethics.stillness_dominance_streak,
            broca_quality: 0.0, // TODO: wire broca telemetry when available on CycleMetadata
            tom_mismatch: m.tom_prediction_mismatch,
        });
        last_result = Some(result);
    }

    let elapsed = cycle_start.elapsed();
    let result = last_result.expect("at least one measurement cycle");
    let m = &result.metadata;

    eprintln!(
        "Pulse: {} cycles in {:.1}s ({:.0} us/cycle avg)",
        total,
        elapsed.as_secs_f64(),
        elapsed.as_micros() as f64 / total as f64
    );

    // ── 3. Extract snapshots ────────────────────────────────────────────
    let vitals = Vitals {
        consciousness_level: m.consciousness_level,
        spectral_phi: m.spectral_mip_phi,
        sigma: m.sigma,
        pipeline_consciousness: m.pipeline_consciousness,
        substrate_effective_feasibility: m.substrate.substrate_effective_feasibility,
        substrate_honest_confidence: m.substrate.substrate_honest_confidence,
        cycle_duration_us: m.cycle_duration_us,
        prediction_error: result.prediction_error,
        temporal_coherence: m.temporal_coherence_score,
        phenomenal_binding: m.phenomenal_binding_strength,
        living_mind_vitality: m.living_mind_vitality,
        somatic_stress: m.somatic_stress,
        thermodynamic_load: m.thermodynamic_load,
        urgency: format!("{:?}", m.urgency),
        consciousness_state: m.consciousness_state_label.clone(),
        error_pattern: m.error_pattern.clone(),
        selected_strategy: m.selected_strategy.clone(),
        total_cycles: total,
    };

    let bath = NeuroBath {
        dopamine: m.neuromod.dopamine_effective,
        noradrenaline: m.neuromod.noradrenaline_effective,
        serotonin: m.neuromod.serotonin_effective,
        acetylcholine: m.neuromod.acetylcholine_effective,
        gaba: m.neuromod.neuromod_gaba_effective,
        oxytocin: m.neuromod.neuromod_oxytocin_effective,
        glutamate: m.neuromod.neuromod_glutamate_effective,
        adenosine: m.neuromod.neuromod_adenosine_effective,
        endocannabinoid: m.neuromod.neuromod_endocannabinoid_effective,
        allostatic_load: m.neuromod.neuromod_allostatic_load,
        personality: m.neuromod.neuromod_personality.clone(),
        circadian_phase: m.circadian_phase.clone(),
        sleep_pressure: m.neuromod.neuromod_sleep_pressure,
        ei_ratio: m.neuromod.neuromod_ei_ratio,
    };

    let compass = MoralCompass {
        moral_score: m.ethics.moral_score,
        value_score: m.ethics.value_evaluator_score,
        harmonies_alignment: m.harmonics.harmonies_alignment,
        harmony_coordinates: m.harmonics.harmony_coordinates,
        dominant_harmonic: m.harmonics.dominant_harmonic.clone(),
        moral_kl_divergence: m.harmonics.moral_kl_divergence,
        moral_entropy: m.harmonics.moral_entropy,
        moral_topo_unity: m.ethics.moral_topo_unity,
        value_decision: m.ethics.value_evaluator_decision.clone(),
        soul_alignment: m.ethics.soul_alignment,
        empathic_compassion: m.ethics.empathic_compassion,
        guiding_question: m.harmonics.guiding_question.clone(),
    };

    let substrate = SubstrateInfo {
        substrate_type: format!("{:?}", service.config().substrate_type),
        raw_feasibility: m.substrate.substrate_feasibility_raw,
        honest_confidence: m.substrate.substrate_honest_confidence,
        effective_feasibility: m.substrate.substrate_effective_feasibility,
        tau_factor: m.substrate.substrate_tau_factor,
        scale_pressure: m.substrate.substrate_scale_pressure,
    };

    let narrative = Narrative {
        reasoning: m.reasoning_narrative.clone(),
        guiding_question: m.harmonics.guiding_question.clone(),
        consciousness_state: m.consciousness_state_label.clone(),
        error_pattern: m.error_pattern.clone(),
        selected_strategy: m.selected_strategy.clone(),
    };

    // ── 4. Run psych-bench battery (optional) ───────────────────────────
    let (cognitive_profile, butlin_report) = if !args.skip_bench {
        eprintln!("Pulse: running psych-bench battery...");
        let bench_start = Instant::now();

        let bench_config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 20,
            seed: 42,
            ..Default::default()
        };

        let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
            Box::new(StroopBenchmark),
            Box::new(FlankerBenchmark),
            Box::new(NBackBenchmark),
            Box::new(VisualSearchBenchmark),
            Box::new(PvtBenchmark),
            Box::new(StopSignalBenchmark),
            Box::new(RmeBenchmark),
            Box::new(FalseBeliefBenchmark),
            Box::new(FittsLawBenchmark),
            Box::new(MetacognitiveCalibrationBenchmark),
        ];

        let mut report = BenchmarkReport::new();
        for bench in &benchmarks {
            report.add(bench.run(&bench_config));
        }

        let profile = CognitiveProfile::from_report(&report);
        eprintln!(
            "  bench: {} domains, overall {:.1}% in {:.1}s",
            profile.domains.len(),
            profile.overall * 100.0,
            bench_start.elapsed().as_secs_f64()
        );

        // Run Butlin indicators
        eprintln!("Pulse: evaluating Butlin consciousness indicators...");
        let butlin = ButlinIndicatorSuite::evaluate(&bench_config);
        eprintln!(
            "  butlin: {}/{} present, {}/{} partial",
            butlin.present_count,
            butlin.indicators.len(),
            butlin.partial_count,
            butlin.indicators.len()
        );

        (Some(profile), Some(butlin))
    } else {
        (None, None)
    };

    // ── 5. Write JSON sidecar (if requested) ────────────────────────────
    let snapshot = PulseSnapshot {
        timestamp: timestamp.clone(),
        profile: profile_name.clone(),
        vitals,
        bath,
        compass,
        substrate,
        narrative,
        sparkline,
        integrity: IntegrityInfo {
            attestation_passed: m.integrity.attestation_passed,
            temporal_passed: m.integrity.temporal_passed,
            canaries_passed: m.integrity.canaries_passed,
            anomaly_count: m.integrity.anomaly_count,
            has_critical: m.integrity.has_critical,
            canary_count: 6,      // 6 built-in canaries
            attestation_count: 3, // safety thresholds + consciousness weights + receptor sensitivities
        },
    };

    if let Some(json_path) = &args.json {
        let json = serde_json::to_string_pretty(&snapshot)?;
        std::fs::write(json_path, &json)?;
        eprintln!(
            "Pulse: JSON written to {} ({:.1} KB)",
            json_path.display(),
            json.len() as f64 / 1024.0
        );
    }

    // ── 6. Load comparison snapshots (if requested) ─────────────────────
    let mut timeline: Vec<PulseSnapshot> = Vec::new();
    for compare_path in &args.compare {
        match std::fs::read_to_string(compare_path) {
            Ok(json) => match serde_json::from_str::<PulseSnapshot>(&json) {
                Ok(prev) => {
                    eprintln!(
                        "Pulse: loaded timeline point {} ({})",
                        compare_path.display(),
                        prev.timestamp
                    );
                    timeline.push(prev);
                }
                Err(e) => {
                    eprintln!("Pulse: failed to parse {}: {}", compare_path.display(), e);
                }
            },
            Err(e) => {
                eprintln!("Pulse: failed to read {}: {}", compare_path.display(), e);
            }
        }
    }
    // Sort timeline chronologically
    timeline.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    // Delta is computed against the most recent previous snapshot
    let delta = timeline
        .last()
        .map(|prev| PulseDelta::compute(&snapshot, prev));

    // ── 7. Detect anomalies + generate session report ──────────────────
    let anomalies = detect_anomalies(&snapshot.sparkline);
    if !anomalies.is_empty() {
        eprintln!("Pulse: {} anomalies detected", anomalies.len());
    }

    let session_report = generate_session_report(
        &snapshot.vitals,
        &snapshot.bath,
        &snapshot.compass,
        &snapshot.sparkline,
        &anomalies,
        delta.as_ref(),
        &timeline,
    );

    // ── 8. Generate output ──────────────────────────────────────────────
    match args.format.as_str() {
        "markdown" | "md" => {
            let md = generate_markdown(
                &snapshot.timestamp,
                &snapshot.profile,
                &snapshot.vitals,
                &snapshot.bath,
                &snapshot.compass,
                &snapshot.substrate,
                &session_report,
                &anomalies,
            );
            std::fs::write(&args.output, &md)?;
            eprintln!(
                "Pulse: markdown written to {} ({:.1} KB)",
                args.output.display(),
                md.len() as f64 / 1024.0
            );
        }
        "csv" => {
            let csv = generate_csv(
                &snapshot.timestamp,
                &snapshot.profile,
                &snapshot.vitals,
                &snapshot.bath,
                &snapshot.compass,
                &snapshot.substrate,
                &snapshot.sparkline,
            );
            std::fs::write(&args.output, &csv)?;
            eprintln!(
                "Pulse: CSV written to {} ({:.1} KB)",
                args.output.display(),
                csv.len() as f64 / 1024.0
            );
        }
        _ => {
            let html_content = html::generate_pulse_html(
                &snapshot.timestamp,
                &snapshot.profile,
                &snapshot.vitals,
                &snapshot.bath,
                &snapshot.compass,
                cognitive_profile.as_ref(),
                butlin_report.as_ref(),
                &snapshot.substrate,
                &snapshot.narrative,
                &snapshot.sparkline,
                delta.as_ref(),
                &timeline,
                &snapshot,
                &anomalies,
                &session_report,
            );
            std::fs::write(&args.output, &html_content)?;
            eprintln!(
                "Pulse: written to {} ({:.1} KB)",
                args.output.display(),
                html_content.len() as f64 / 1024.0
            );
        }
    }

    // ── 8. Watch mode — continuous cycle + regenerate ────────────────────
    if let Some(interval_secs) = args.watch {
        let interval = std::time::Duration::from_secs(interval_secs.max(1));
        eprintln!(
            "Pulse: watch mode — regenerating every {}s (Ctrl+C to stop)",
            interval_secs
        );

        let mut cycle_count = total;
        loop {
            std::thread::sleep(interval);
            let watch_timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

            // Run more measurement cycles
            let mut watch_sparkline: Vec<SparklinePoint> = Vec::with_capacity(measurement);
            let mut watch_result = None;
            for i in 0..measurement {
                let input = inputs[(cycle_count + i) % inputs.len()];
                let result = service.cycle(input);
                let wm = &result.metadata;
                watch_sparkline.push(SparklinePoint {
                    consciousness: wm.consciousness_level,
                    prediction_error: result.prediction_error,
                    phi: wm.spectral_mip_phi.unwrap_or(0.0),
                    somatic_stress: wm.somatic_stress,
                    dopamine: wm.neuromod.dopamine_effective,
                    serotonin: wm.neuromod.serotonin_effective,
                    harmony_coords: wm.harmonics.harmony_coordinates,
                    harmony_entropy: wm.ethics.harmony_entropy,
                    moral_attractor_detected: wm.ethics.moral_attractor_detected,
                    in_active_rest: wm.ethics.in_active_rest,
                    stillness_dominance_streak: wm.ethics.stillness_dominance_streak,
                    broca_quality: 0.0, // TODO: wire broca telemetry when available on CycleMetadata
                    tom_mismatch: wm.tom_prediction_mismatch,
                });
                watch_result = Some(result);
            }
            cycle_count += measurement;

            let wr = watch_result.expect("at least one cycle");
            let wm = &wr.metadata;

            let watch_snapshot = PulseSnapshot {
                timestamp: watch_timestamp.clone(),
                profile: profile_name.clone(),
                vitals: Vitals {
                    consciousness_level: wm.consciousness_level,
                    spectral_phi: wm.spectral_mip_phi,
                    sigma: wm.sigma,
                    pipeline_consciousness: wm.pipeline_consciousness,
                    substrate_effective_feasibility: wm.substrate.substrate_effective_feasibility,
                    substrate_honest_confidence: wm.substrate.substrate_honest_confidence,
                    cycle_duration_us: wm.cycle_duration_us,
                    prediction_error: wr.prediction_error,
                    temporal_coherence: wm.temporal_coherence_score,
                    phenomenal_binding: wm.phenomenal_binding_strength,
                    living_mind_vitality: wm.living_mind_vitality,
                    somatic_stress: wm.somatic_stress,
                    thermodynamic_load: wm.thermodynamic_load,
                    urgency: format!("{:?}", wm.urgency),
                    consciousness_state: wm.consciousness_state_label.clone(),
                    error_pattern: wm.error_pattern.clone(),
                    selected_strategy: wm.selected_strategy.clone(),
                    total_cycles: cycle_count,
                },
                bath: NeuroBath {
                    dopamine: wm.neuromod.dopamine_effective,
                    noradrenaline: wm.neuromod.noradrenaline_effective,
                    serotonin: wm.neuromod.serotonin_effective,
                    acetylcholine: wm.neuromod.acetylcholine_effective,
                    gaba: wm.neuromod.neuromod_gaba_effective,
                    oxytocin: wm.neuromod.neuromod_oxytocin_effective,
                    glutamate: wm.neuromod.neuromod_glutamate_effective,
                    adenosine: wm.neuromod.neuromod_adenosine_effective,
                    endocannabinoid: wm.neuromod.neuromod_endocannabinoid_effective,
                    allostatic_load: wm.neuromod.neuromod_allostatic_load,
                    personality: wm.neuromod.neuromod_personality.clone(),
                    circadian_phase: wm.circadian_phase.clone(),
                    sleep_pressure: wm.neuromod.neuromod_sleep_pressure,
                    ei_ratio: wm.neuromod.neuromod_ei_ratio,
                },
                compass: MoralCompass {
                    moral_score: wm.ethics.moral_score,
                    value_score: wm.ethics.value_evaluator_score,
                    harmonies_alignment: wm.harmonics.harmonies_alignment,
                    harmony_coordinates: wm.harmonics.harmony_coordinates,
                    dominant_harmonic: wm.harmonics.dominant_harmonic.clone(),
                    moral_kl_divergence: wm.harmonics.moral_kl_divergence,
                    moral_entropy: wm.harmonics.moral_entropy,
                    moral_topo_unity: wm.ethics.moral_topo_unity,
                    value_decision: wm.ethics.value_evaluator_decision.clone(),
                    soul_alignment: wm.ethics.soul_alignment,
                    empathic_compassion: wm.ethics.empathic_compassion,
                    guiding_question: wm.harmonics.guiding_question.clone(),
                },
                substrate: SubstrateInfo {
                    substrate_type: format!("{:?}", service.config().substrate_type),
                    raw_feasibility: wm.substrate.substrate_feasibility_raw,
                    honest_confidence: wm.substrate.substrate_honest_confidence,
                    effective_feasibility: wm.substrate.substrate_effective_feasibility,
                    tau_factor: wm.substrate.substrate_tau_factor,
                    scale_pressure: wm.substrate.substrate_scale_pressure,
                },
                narrative: Narrative {
                    reasoning: wm.reasoning_narrative.clone(),
                    guiding_question: wm.harmonics.guiding_question.clone(),
                    consciousness_state: wm.consciousness_state_label.clone(),
                    error_pattern: wm.error_pattern.clone(),
                    selected_strategy: wm.selected_strategy.clone(),
                },
                sparkline: watch_sparkline,
                integrity: IntegrityInfo {
                    attestation_passed: wm.integrity.attestation_passed,
                    temporal_passed: wm.integrity.temporal_passed,
                    canaries_passed: wm.integrity.canaries_passed,
                    anomaly_count: wm.integrity.anomaly_count,
                    has_critical: wm.integrity.has_critical,
                    canary_count: 6,
                    attestation_count: 3,
                },
            };

            // Delta against previous snapshot
            let watch_delta = PulseDelta::compute(&watch_snapshot, &snapshot);

            let watch_anomalies = detect_anomalies(&watch_snapshot.sparkline);
            let watch_report = generate_session_report(
                &watch_snapshot.vitals,
                &watch_snapshot.bath,
                &watch_snapshot.compass,
                &watch_snapshot.sparkline,
                &watch_anomalies,
                Some(&watch_delta),
                &[],
            );

            let watch_html = html::generate_pulse_html(
                &watch_snapshot.timestamp,
                &watch_snapshot.profile,
                &watch_snapshot.vitals,
                &watch_snapshot.bath,
                &watch_snapshot.compass,
                None, // skip re-running bench in watch
                None,
                &watch_snapshot.substrate,
                &watch_snapshot.narrative,
                &watch_snapshot.sparkline,
                Some(&watch_delta),
                &[], // no timeline in watch mode
                &watch_snapshot,
                &watch_anomalies,
                &watch_report,
            );

            std::fs::write(&args.output, &watch_html)?;
            eprintln!(
                "  watch: cycle {} · C(t)={:.4} · written {:.1} KB",
                cycle_count,
                watch_snapshot.vitals.consciousness_level,
                watch_html.len() as f64 / 1024.0
            );
        }
    }

    Ok(())
}
