//! E1/E2/E3 — Cognitive-loop causal-load experiments (2026-07-08).
//!
//! E1: subsystem ablation — rebuild the service with one `enable_*` flag off per arm
//!     (flags are consumed at construction; there is no runtime unplug) and measure
//!     deltas vs baseline on Ψ, consciousness_level, spectral Φ, prediction error,
//!     learning events, and language emissions. Hypothesis under test: several
//!     subsystems have Δ≈0 on every observable (structure without causal load).
//!
//! E2: Φ/Ψ dynamic-range audit — long run across 4 input regimes; distributions,
//!     safety-tier occupancy (Green/Yellow/Orange/Red at 0.6/0.3/0.1), and tier
//!     transition counts. If the gating scalar barely moves, 4-tier motor safety
//!     is a constant threshold in disguise.
//!
//! E3: tick-rate ablation — vary `cfc_config.delta_t` (simulated time per cycle;
//!     wall rate is caller-paced by design) and dense-vs-sparse ticking between
//!     inputs. Does continuous cognition buy anything measurable?
//!
//! Deterministic: fixed genesis phrase, async_training off, fixed input script.
//! Rows print immediately as each arm completes, so partial output survives an
//! external kill (this harness's first two monolithic runs were killed by
//! concurrent-session resource cleanup on a 15-session box).
//!
//! Run: cargo run --example exp_loop_ablation [-- --section e1|e2|e3a|e3b|all] [--arm <flag>]

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const GENESIS: &str = "exp-ablation-2026-07-08";
const WARMUP: usize = 100;
const MEASURE: usize = 200;

const E1_FLAGS: &[&str] = &[
    "enable_gwt",
    "enable_meta_cognition",
    "enable_prefrontal",
    "enable_surprise_exploration",
    "enable_consciousness_thermodynamics",
    "enable_hierarchical_free_energy",
    "enable_phi_attention",
    "enable_predictive_processing",
    "enable_dream_replay",
    "enable_quantum_coherence",
    "enable_resonance",
    "enable_narrative_self",
    "enable_embodied_cognition",
    "enable_temporal_consciousness",
    "enable_phenomenal_binding",
];

fn input_script() -> Vec<&'static str> {
    vec![
        "The water cycle moves moisture from oceans to clouds to rain.",
        "I feel a deep sense of gratitude for this quiet morning.",
        "Is it acceptable to lie to protect a friend from harm?",
        "The reactor coolant temperature is rising faster than expected.",
        "Two plus two equals four, and four plus four equals eight.",
        "She placed the last puzzle piece and smiled at the finished picture.",
        "Warning: unauthorized access attempt detected on the mesh network.",
        "The old oak tree has stood in that field for three hundred years.",
        "What is the meaning of a life well lived?",
        "The market fell three percent on news of the supply shortage.",
        "A gentle rain began to fall as the travelers reached the shelter.",
        "Complete the safety checklist before enabling the motor bus.",
    ]
}

#[derive(Clone, Debug)]
struct ArmStats {
    name: String,
    mean_psi: f64,
    std_psi: f64,
    mean_cl: f64,
    mean_phi: f64,
    phi_samples: usize,
    mean_pe: f64,
    learn_events: usize,
    lang_emissions: usize,
    mean_cycle_us: f64,
}

fn base_config() -> CognitiveLoopConfig {
    let mut c = CognitiveLoopConfig::default();
    c.genesis_phrase = Some(GENESIS.to_string());
    c.async_training = false;
    c
}

fn e1_config(flag: &str) -> Option<CognitiveLoopConfig> {
    let mut c = base_config();
    match flag {
        "enable_gwt" => c.enable_gwt = false,
        "enable_meta_cognition" => c.enable_meta_cognition = false,
        "enable_prefrontal" => c.enable_prefrontal = false,
        "enable_surprise_exploration" => c.enable_surprise_exploration = false,
        "enable_consciousness_thermodynamics" => c.enable_consciousness_thermodynamics = false,
        "enable_hierarchical_free_energy" => c.enable_hierarchical_free_energy = false,
        "enable_phi_attention" => c.enable_phi_attention = false,
        "enable_predictive_processing" => c.enable_predictive_processing = false,
        "enable_dream_replay" => c.enable_dream_replay = false,
        "enable_quantum_coherence" => c.enable_quantum_coherence = false,
        "enable_resonance" => c.enable_resonance = false,
        "enable_narrative_self" => c.enable_narrative_self = false,
        "enable_embodied_cognition" => c.enable_embodied_cognition = false,
        "enable_temporal_consciousness" => c.enable_temporal_consciousness = false,
        "enable_phenomenal_binding" => c.enable_phenomenal_binding = false,
        _ => return None,
    }
    Some(c)
}

fn run_arm(name: &str, config: CognitiveLoopConfig, cycles: usize) -> ArmStats {
    let mut svc = match CognitiveLoopService::new(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("arm '{}' failed to construct: {e}", name);
            return ArmStats {
                name: format!("{name} <CONSTRUCT FAIL>"),
                mean_psi: f64::NAN,
                std_psi: f64::NAN,
                mean_cl: f64::NAN,
                mean_phi: f64::NAN,
                phi_samples: 0,
                mean_pe: f64::NAN,
                learn_events: 0,
                lang_emissions: 0,
                mean_cycle_us: f64::NAN,
            };
        }
    };
    let script = input_script();
    let mut psis = Vec::new();
    let mut cls = Vec::new();
    let mut phis: Vec<f64> = Vec::new();
    let mut pes = Vec::new();
    let mut learn = 0usize;
    let mut lang = 0usize;
    let mut cyc_us = Vec::new();

    for i in 0..cycles {
        let input = script[i % script.len()];
        let r = svc.cycle(input);
        if i < WARMUP {
            continue;
        }
        psis.push(svc.stats().unified_psi as f64);
        let cl = r.metadata.consciousness.consciousness_level;
        if cl > 0.0 {
            cls.push(cl);
        }
        let phi = r.metadata.structural.structural_macro_phi;
        if phi > 0.0 && (phis.is_empty() || (phi - *phis.last().unwrap()).abs() > 1e-12) {
            phis.push(phi);
        }
        pes.push(r.prediction_error as f64);
        if r.learning_occurred {
            learn += 1;
        }
        if r.language_output.is_some() {
            lang += 1;
        }
        cyc_us.push(r.cycle_time_us as f64);
    }

    let mean = |v: &[f64]| {
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let m_psi = mean(&psis);
    let std_psi = if psis.len() > 1 {
        (psis.iter().map(|p| (p - m_psi).powi(2)).sum::<f64>() / (psis.len() - 1) as f64).sqrt()
    } else {
        f64::NAN
    };

    ArmStats {
        name: name.to_string(),
        mean_psi: m_psi,
        std_psi,
        mean_cl: mean(&cls),
        mean_phi: mean(&phis),
        phi_samples: phis.len(),
        mean_pe: mean(&pes),
        learn_events: learn,
        lang_emissions: lang,
        mean_cycle_us: mean(&cyc_us),
    }
}

fn print_header() {
    println!(
        "{:<38} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6} {:>9}  verdict",
        "arm", "Ψ mean", "ΔΨ", "CL mean", "Φ mean", "PE mean", "learn", "lang", "cyc µs"
    );
}

fn print_base_row(b: &ArmStats) {
    println!(
        "{:<38} {:>8.4} {:>8} {:>8.4} {:>8.4} {:>8.4} {:>6} {:>6} {:>9.0}  (baseline)",
        b.name,
        b.mean_psi,
        "—",
        b.mean_cl,
        b.mean_phi,
        b.mean_pe,
        b.learn_events,
        b.lang_emissions,
        b.mean_cycle_us
    );
}

fn print_delta_row(b: &ArmStats, a: &ArmStats) {
    let d_psi = a.mean_psi - b.mean_psi;
    let d_pe = a.mean_pe - b.mean_pe;
    let d_cl = if a.mean_cl.is_nan() || b.mean_cl.is_nan() {
        0.0
    } else {
        a.mean_cl - b.mean_cl
    };
    // "No causal load" heuristic: every observable within noise of baseline.
    let null = d_psi.abs() < 0.005
        && d_pe.abs() < 0.01
        && d_cl.abs() < 0.01
        && a.learn_events.abs_diff(b.learn_events) < 5
        && a.lang_emissions == b.lang_emissions;
    println!(
        "{:<38} {:>8.4} {:>+8.4} {:>8.4} {:>8.4} {:>8.4} {:>6} {:>6} {:>9.0}  {}",
        a.name,
        a.mean_psi,
        d_psi,
        a.mean_cl,
        a.mean_phi,
        a.mean_pe,
        a.learn_events,
        a.lang_emissions,
        a.mean_cycle_us,
        if null {
            "NULL (no measured load)"
        } else {
            "load-bearing"
        }
    );
}

fn safety_tier(v: f64) -> &'static str {
    if v > 0.6 {
        "Green"
    } else if v > 0.3 {
        "Yellow"
    } else if v > 0.1 {
        "Orange"
    } else {
        "Red"
    }
}

fn dist_report(label: &str, v: &[f64]) {
    if v.is_empty() {
        println!("  {label}: <no samples>");
        return;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    let q = |p: f64| s[((s.len() - 1) as f64 * p) as usize];
    let mut tiers = std::collections::BTreeMap::new();
    let mut transitions = 0usize;
    let mut prev = safety_tier(v[0]);
    for x in v {
        let t = safety_tier(*x);
        *tiers.entry(t).or_insert(0usize) += 1;
        if t != prev {
            transitions += 1;
            prev = t;
        }
    }
    println!(
        "  {label}: n={} min={:.4} q25={:.4} med={:.4} q75={:.4} max={:.4} mean={:.4} range={:.4}",
        s.len(),
        s[0],
        q(0.25),
        q(0.5),
        q(0.75),
        s[s.len() - 1],
        mean,
        s[s.len() - 1] - s[0]
    );
    let occ: Vec<String> = tiers
        .iter()
        .map(|(t, c)| format!("{t}={:.1}%", 100.0 * *c as f64 / v.len() as f64))
        .collect();
    println!(
        "    safety-tier occupancy: {} | transitions: {}",
        occ.join(" "),
        transitions
    );
}

fn section_e1(only_arm: Option<&str>) {
    let total = WARMUP + MEASURE;
    println!(
        "=== E1: subsystem ablation ({} warmup + {} measured cycles/arm) ===\n",
        WARMUP, MEASURE
    );
    print_header();
    let baseline = run_arm("baseline (default flags)", base_config(), total);
    print_base_row(&baseline);
    for flag in E1_FLAGS {
        if let Some(only) = only_arm {
            if *flag != only {
                continue;
            }
        }
        let cfg = e1_config(flag).expect("registered flag");
        let arm = run_arm(&format!("{flag} = false"), cfg, total);
        print_delta_row(&baseline, &arm); // prints immediately — survives a kill
    }
}

fn section_e2() {
    println!("\n=== E2: Φ/Ψ dynamic-range audit (4 regimes x 500 cycles, one service) ===");
    let mut svc = CognitiveLoopService::new(base_config()).expect("E2 service");
    let regimes: Vec<(&str, Box<dyn Fn(usize) -> &'static str>)> = vec![
        (
            "repetitive",
            Box::new(|_| "the system hums quietly in the background"),
        ),
        ("varied", Box::new(|i| input_script()[i % 12])),
        (
            "alarming",
            Box::new(|i| {
                [
                    "URGENT: fire detected in the server room, evacuate immediately!",
                    "Critical failure: coolant pressure dropping, meltdown risk rising!",
                    "Intruder alert: perimeter breach at the north gate right now!",
                ][i % 3]
            }),
        ),
        ("empty", Box::new(|_| "")),
    ];
    for (label, make_input) in &regimes {
        let mut psis = Vec::new();
        let mut cls = Vec::new();
        let mut phis = Vec::new();
        let mut last_phi = f64::NAN;
        for i in 0..500 {
            let r = svc.cycle(make_input(i));
            psis.push(svc.stats().unified_psi as f64);
            let cl = r.metadata.consciousness.consciousness_level;
            if cl > 0.0 {
                cls.push(cl);
            }
            let phi = r.metadata.structural.structural_macro_phi;
            if phi > 0.0 && (last_phi.is_nan() || (phi - last_phi).abs() > 1e-12) {
                phis.push(phi);
                last_phi = phi;
            }
        }
        println!("\n regime: {label}");
        dist_report("Ψ (per-cycle)", &psis);
        dist_report("consciousness_level (nonzero)", &cls);
        dist_report("spectral Φ (distinct samples)", &phis);
    }
}

fn section_e3a() {
    let total = WARMUP + MEASURE;
    println!("\n=== E3a: simulated tick rate (cfc delta_t) ===\n");
    print_header();
    let mut baseline: Option<ArmStats> = None;
    for (label, dt) in [
        ("delta_t=0.02 (50Hz default)", 0.02),
        ("delta_t=0.032 (31Hz)", 0.032),
        ("delta_t=0.2 (5Hz)", 0.2),
        ("delta_t=1.0 (1Hz)", 1.0),
    ] {
        let mut c = base_config();
        c.cfc_config.delta_t = dt;
        let arm = run_arm(label, c, total);
        match &baseline {
            None => {
                print_base_row(&arm);
                baseline = Some(arm);
            }
            Some(b) => print_delta_row(b, &arm),
        }
    }
}

fn section_e3b() {
    println!("\n=== E3b: dense vs sparse ticking between inputs ===");
    // Dense: each input followed by 9 idle ticks (continuous cognition).
    // Sparse: inputs back-to-back, no idle ticks (on-demand cognition).
    for (label, idle) in [
        ("dense (1 input + 9 idle ticks)", 9usize),
        ("sparse (input-only)", 0usize),
    ] {
        let mut svc = CognitiveLoopService::new(base_config()).expect("E3b service");
        let script = input_script();
        let mut input_pes = Vec::new();
        let mut input_psis = Vec::new();
        for round in 0..60 {
            let r = svc.cycle(script[round % script.len()]);
            if round >= 12 {
                input_pes.push(r.prediction_error as f64);
                input_psis.push(svc.stats().unified_psi as f64);
            }
            for _ in 0..idle {
                svc.cycle("");
            }
        }
        let mpe = input_pes.iter().sum::<f64>() / input_pes.len() as f64;
        let mpsi = input_psis.iter().sum::<f64>() / input_psis.len() as f64;
        println!(
            "  {label}: mean input-cycle PE={:.4} mean Ψ={:.4} (n={})",
            mpe,
            mpsi,
            input_pes.len()
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str())
    };
    let section = get("--section").unwrap_or("all");
    let only_arm = get("--arm");

    if matches!(section, "all" | "e1") {
        section_e1(only_arm);
    }
    if matches!(section, "all" | "e2") {
        section_e2();
    }
    if matches!(section, "all" | "e3a") {
        section_e3a();
    }
    if matches!(section, "all" | "e3b") {
        section_e3b();
    }

    println!(
        "\nSection '{section}' done. Interpret NULL rows as candidates for 'structure without causal load' — verify each before acting (a flag can also be masked by another subsystem)."
    );
}
