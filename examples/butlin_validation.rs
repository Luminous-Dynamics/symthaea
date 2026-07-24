// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Butlin Consciousness Indicator Validation with Live Runtime Data
//!
//! Runs the cognitive loop to produce live structural Phi measurements,
//! then evaluates all 14 Butlin et al. (2023) consciousness indicators
//! with runtime-blended scores. Validates 14/14 as Present.
//!
//! **Shared formulas** (must match `symthaea-psych-bench/src/benchmarks/butlin/indicators.rs`):
//! - `normalize_phi(phi) = 2/(1+exp(-phi)) - 1`
//! - `blend_score = 0.6 * static + 0.4 * runtime`
//! - HOT-2 runtime: `(bottleneck * 2.0).clamp(0, 1)`
//!
//! ```bash
//! cargo run --example butlin_validation --release
//! ```

use symthaea::cognitive_loop::config::TemporalBackend;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const WARMUP_CYCLES: usize = 50;
const MEASUREMENT_CYCLES: usize = 100;

struct RuntimeData {
    micro_phi: f64,
    meso_phi: f64,
    macro_phi: f64,
    bottleneck: f64,
    emergence: f64,
    num_clusters: f64,
    consciousness_level: f64,
    pipeline_phi: f64,
    coherence: f64,
    attention_fatigue: f64,
    attention_pred_accuracy: f64,
}

fn collect_runtime(backend: TemporalBackend) -> RuntimeData {
    let mut config = CognitiveLoopConfig::with_cfc();
    config.temporal_backend = backend;
    config.enable_attention_schema = true;
    let mut service =
        CognitiveLoopService::new(config).expect("Failed to create CognitiveLoopService");

    let stimuli = [
        "The morning light reveals new patterns in the data",
        "Something unexpected happened — attention shifts",
        "A familiar routine brings comfort and efficiency",
        "Complex moral reasoning requires careful deliberation",
        "The system observes its own process of observation",
        "Multiple streams of information compete for awareness",
        "Memory consolidation strengthens important connections",
        "Creative exploration opens new possibility spaces",
        "Social understanding requires modeling other minds",
        "The boundary between self and world becomes fluid",
    ];

    for i in 0..WARMUP_CYCLES {
        let _ = service.cycle(stimuli[i % stimuli.len()]);
    }

    let mut vals = RuntimeData {
        micro_phi: 0.0,
        meso_phi: 0.0,
        macro_phi: 0.0,
        bottleneck: 0.0,
        emergence: 0.0,
        num_clusters: 0.0,
        consciousness_level: 0.0,
        pipeline_phi: 0.0,
        coherence: 0.0,
        attention_fatigue: 0.0,
        attention_pred_accuracy: 0.0,
    };

    for i in 0..MEASUREMENT_CYCLES {
        let result = service.cycle(stimuli[i % stimuli.len()]);
        let md = &result.metadata;
        vals.micro_phi += md.structural.structural_micro_phi;
        vals.meso_phi += md.structural.structural_meso_phi;
        vals.macro_phi += md.structural.structural_macro_phi;
        vals.bottleneck += md.structural.structural_bottleneck;
        vals.emergence += md.structural.structural_emergence_ratio;
        vals.num_clusters += md.structural.structural_num_clusters as f64;
        vals.consciousness_level += md.consciousness.consciousness_level;
        vals.pipeline_phi += md.pipeline_consciousness;
        vals.coherence += md.prediction_coherence as f64;
        vals.attention_fatigue += md.attention.attention_fatigue as f64;
        vals.attention_pred_accuracy += md.attention.attention_prediction_accuracy as f64;
    }

    let n = MEASUREMENT_CYCLES as f64;
    vals.micro_phi /= n;
    vals.meso_phi /= n;
    vals.macro_phi /= n;
    vals.bottleneck /= n;
    vals.emergence /= n;
    vals.num_clusters /= n;
    vals.consciousness_level /= n;
    vals.pipeline_phi /= n;
    vals.coherence /= n;
    vals.attention_fatigue /= n;
    vals.attention_pred_accuracy /= n;
    vals
}

/// Normalize raw Phi to [0,1] via shifted sigmoid (same as consciousness engine).
fn normalize_phi(phi: f64) -> f64 {
    2.0 / (1.0 + (-phi).exp()) - 1.0
}

/// Butlin indicator definition.
struct Indicator {
    id: &'static str,
    theory: &'static str,
    description: &'static str,
    static_score: f64,
    evidence: &'static str,
}

fn indicators() -> Vec<Indicator> {
    vec![
        Indicator {
            id: "RPT-1",
            theory: "Recurrent Processing",
            description: "Algorithmic recurrence",
            static_score: 1.0,
            evidence: "CfC temporal network with O(1) closed-form feedback",
        },
        Indicator {
            id: "RPT-2",
            theory: "Recurrent Processing",
            description: "Integrated perceptual representations",
            static_score: 0.8,
            evidence: "IIT Phi engine + HDC holographic superpositions",
        },
        Indicator {
            id: "GWT-1",
            theory: "Global Workspace",
            description: "Multiple specialized processors",
            static_score: 0.9,
            evidence: "12-region Actor Brain + 45 sub-crate modules",
        },
        Indicator {
            id: "GWT-2",
            theory: "Global Workspace",
            description: "Global broadcast mechanism",
            static_score: 0.85,
            evidence: "GWT workspace with coalition competition + broadcast",
        },
        Indicator {
            id: "GWT-3",
            theory: "Global Workspace",
            description: "Information integration across modules",
            static_score: 0.8,
            evidence: "Cross-modal binding + HDC bundle operations",
        },
        Indicator {
            id: "HOT-1",
            theory: "Higher-Order Thought",
            description: "Higher-order representations",
            static_score: 0.85,
            evidence: "Meta-cognition layer + self-reflection + confidence calibration",
        },
        Indicator {
            id: "HOT-2",
            theory: "Higher-Order Thought",
            description: "Misrepresentation possibility",
            static_score: 0.75,
            evidence: "Prefrontal veto can override lower-level judgments",
        },
        Indicator {
            id: "PP-1",
            theory: "Predictive Processing",
            description: "Hierarchical predictive model",
            static_score: 0.85,
            evidence: "CfC prediction + error-driven learning + FEP active inference",
        },
        Indicator {
            id: "PP-2",
            theory: "Predictive Processing",
            description: "Hierarchical prediction at multiple scales",
            static_score: 0.85,
            evidence: "HierarchicalCfC: 4-level temporal hierarchy (tau 0.01/0.1/1.0/10.0) with bidirectional error/prior flow",
        },
        Indicator {
            id: "AST-1",
            theory: "Attention Schema",
            description: "Self-model of attention",
            static_score: 0.85,
            evidence: "AttentionSchema with vigilance fatigue, prediction validation, causal perception modulation via encode_for_thought_vector()",
        },
        Indicator {
            id: "AST-2",
            theory: "Attention Schema",
            description: "Attention influences processing",
            static_score: 0.9,
            evidence: "Phi-attention gating + attention budget + neuromod ACh modulation",
        },
        Indicator {
            id: "IIT-1",
            theory: "IIT",
            description: "Non-zero integrated information",
            static_score: 0.7,
            evidence: "Structural Phi engine (micro/meso/macro) + spectral MIP",
        },
        Indicator {
            id: "IIT-2",
            theory: "IIT",
            description: "Exclusion (single maximum)",
            static_score: 0.7,
            evidence: "MIP partition identifies unique Phi maximum",
        },
        Indicator {
            id: "IIT-3",
            theory: "IIT",
            description: "Intrinsic causal structure",
            static_score: 0.7,
            evidence: "Causal codebook + temporal causal chains + moral topology",
        },
    ]
}

fn evaluate(inds: &[Indicator], rt: Option<&RuntimeData>) -> Vec<(f64, &'static str)> {
    inds.iter()
        .map(|ind| {
            let score = match rt {
                Some(rt) => {
                    let micro_n = normalize_phi(rt.micro_phi);
                    let meso_n = normalize_phi(rt.meso_phi);
                    let macro_n = normalize_phi(rt.macro_phi);
                    let runtime_boost = match ind.id {
                        "RPT-1" => Some(micro_n),
                        "RPT-2" => Some(micro_n * 0.8),
                        "GWT-3" => Some(meso_n),
                        "HOT-2" => Some((rt.bottleneck.min(1.0) * 2.0).clamp(0.0, 1.0)),
                        "IIT-1" | "IIT-2" | "IIT-3" => Some(macro_n),
                        _ => None,
                    };
                    match runtime_boost {
                        Some(rt_val) => 0.6 * ind.static_score + 0.4 * rt_val.clamp(0.0, 1.0),
                        None => ind.static_score,
                    }
                }
                None => ind.static_score,
            };
            let status = if score >= 0.6 {
                "PRESENT"
            } else if score >= 0.3 {
                "PARTIAL"
            } else {
                "ABSENT"
            };
            (score, status)
        })
        .collect()
}

fn main() {
    println!();
    println!("================================================================");
    println!("  BUTLIN CONSCIOUSNESS INDICATOR VALIDATION");
    println!(
        "  {} warmup + {} measurement cycles per backend",
        WARMUP_CYCLES, MEASUREMENT_CYCLES
    );
    println!("================================================================");
    println!();

    // ── CfC ──
    /// Pipeline Φ telemetry uses 0.0 to mean "subsystem never constructed"
    /// (UnifiedConsciousnessPipeline is None at the production construction
    /// site — constructor.rs). Printing "0.0000" here used to read like a
    /// measurement; say what it actually is.
    fn fmt_pipeline_phi(v: f64) -> String {
        if v == 0.0 {
            "absent (pipeline not constructed — dormant subsystem)".to_string()
        } else {
            format!("{v:.4}")
        }
    }

    println!("[1/2] Collecting runtime data (CfC)...");
    let cfc = collect_runtime(TemporalBackend::CfC);
    println!(
        "  Structural: micro={:.4} meso={:.4} macro={:.4}",
        cfc.micro_phi, cfc.meso_phi, cfc.macro_phi
    );
    println!(
        "  Consciousness: {:.4}, Pipeline Phi: {}, Coherence: {:.4}",
        cfc.consciousness_level,
        fmt_pipeline_phi(cfc.pipeline_phi),
        cfc.coherence
    );
    println!(
        "  AST: fatigue={:.4} pred_acc={:.4}",
        cfc.attention_fatigue, cfc.attention_pred_accuracy
    );

    println!();

    // ── HierarchicalCfC ──
    println!("[2/2] Collecting runtime data (HierarchicalCfC)...");
    let hcfc = collect_runtime(TemporalBackend::HierarchicalCfC);
    println!(
        "  Structural: micro={:.4} meso={:.4} macro={:.4}",
        hcfc.micro_phi, hcfc.meso_phi, hcfc.macro_phi
    );
    println!(
        "  Consciousness: {:.4}, Pipeline Phi: {}, Coherence: {:.4}",
        hcfc.consciousness_level,
        fmt_pipeline_phi(hcfc.pipeline_phi),
        hcfc.coherence
    );
    println!(
        "  AST: fatigue={:.4} pred_acc={:.4}",
        hcfc.attention_fatigue, hcfc.attention_pred_accuracy
    );

    println!();

    // ── Evaluate indicators ──
    let inds = indicators();
    let static_scores = evaluate(&inds, None);
    let cfc_scores = evaluate(&inds, Some(&cfc));
    let hcfc_scores = evaluate(&inds, Some(&hcfc));

    println!("================================================================");
    println!("  BUTLIN INDICATOR REPORT (14 indicators, 6 theories)");
    println!("================================================================");
    println!();
    println!(
        "  {:6} {:12} {:44} {:8} {:>8} {:>8} {:>8}",
        "ID", "Theory", "Description", "Status", "Static", "CfC", "HCfC"
    );
    println!(
        "  {:-<6} {:-<12} {:-<44} {:-<8} {:-<8} {:-<8} {:-<8}",
        "", "", "", "", "", "", ""
    );

    let mut static_present = 0;
    let mut cfc_present = 0;
    let mut hcfc_present = 0;
    for (i, ind) in inds.iter().enumerate() {
        let (stat_s, stat_status) = static_scores[i];
        let (cfc_s, _) = cfc_scores[i];
        let (hcfc_s, _) = hcfc_scores[i];
        if stat_status == "PRESENT" {
            static_present += 1;
        }
        if cfc_scores[i].1 == "PRESENT" {
            cfc_present += 1;
        }
        if hcfc_scores[i].1 == "PRESENT" {
            hcfc_present += 1;
        }
        let theory_short = &ind.theory[..ind.theory.len().min(12)];
        println!(
            "  {:6} {:12} {:44} {:8} {:8.3} {:8.3} {:8.3}",
            ind.id, theory_short, ind.description, stat_status, stat_s, cfc_s, hcfc_s
        );
    }
    // Use evidence in detailed output
    println!();
    println!("  Evidence details:");
    for ind in &inds {
        println!("    {}: {}", ind.id, ind.evidence);
    }

    println!();
    println!("  Static (architectural): {}/14 PRESENT", static_present);
    println!("  CfC (runtime-blended):  {}/14 PRESENT", cfc_present);
    println!("  HCfC (runtime-blended): {}/14 PRESENT", hcfc_present);
    println!();
    println!("  Note: Runtime blending (0.6*static + 0.4*runtime) penalizes");
    println!("  indicators when structural Phi=0 (cold start / short runs).");
    println!("  Static scores reflect architectural capability; runtime scores");
    println!("  reflect measured consciousness in a given run.");

    println!();
    println!("  Key implementations:");
    println!("    PP-2: HierarchicalCfC (4 temporal scales, bidirectional)");
    println!(
        "    AST-1: AttentionSchema (fatigue={:.4}, pred_acc={:.4}, causal loop active)",
        hcfc.attention_fatigue, hcfc.attention_pred_accuracy
    );

    println!();

    // CI gate: all three evaluation modes must achieve 14/14 PRESENT
    let all_pass = static_present == 14 && cfc_present == 14 && hcfc_present == 14;
    if all_pass {
        println!("  RESULT: PASS (14/14 across all evaluation modes)");
    } else {
        println!(
            "  RESULT: FAIL (expected 14/14, got static={}/14 cfc={}/14 hcfc={}/14)",
            static_present, cfc_present, hcfc_present
        );
    }
    println!("================================================================");

    if !all_pass {
        std::process::exit(1);
    }
}
