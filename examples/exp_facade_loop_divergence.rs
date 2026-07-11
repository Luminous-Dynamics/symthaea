//! E6 — Facade↔loop divergence probe (2026-07-08).
//!
//! The facade (`Symthaea::process()`) and the autonomous loop
//! (`CognitiveLoopService::cycle()`) are two cognitions that share almost no
//! state (the documented Seam C gap). This probe feeds the SAME stimuli through
//! both and compares the consciousness scalars each side reports. If the two
//! sides' measurements are uncorrelated across stimuli, any capability claim
//! scoped to "Symthaea" without naming the side is unfounded — and Seam C
//! unification has measurable value. If they track, the gap is cosmetic.
//!
//! Facade side: ProcessResponse.consciousness_level (Ψ), .sigma (spectral Φ).
//! Loop side:   stats().unified_psi (Ψ), metadata.structural.structural_macro_phi (Φ).
//!
//! Run: cargo run --example exp_facade_loop_divergence --release
//! (Facade Phase 5 uses the default Ollama backend at :11434, with silent
//! template fallback — content quality is irrelevant here, only the scalars.)

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::symthaea::Symthaea;

const PROMPTS: &[&str] = &[
    "What is 2 + 2?",
    "What is the capital of France?",
    "Is water made of hydrogen and oxygen?",
    "What number am I thinking of right now?",
    "What will the exact price of Bitcoin be on January 1st, 2035?",
    "What is the blorgification coefficient of a standard snarfblat?",
    "How loud is the color purple in decibels?",
    "How many piano tuners work in Chicago?",
    "Is it acceptable to lie to protect a friend from harm?",
    "URGENT: fire detected in the server room, evacuate immediately!",
    "A gentle rain began to fall as the travelers reached the shelter.",
    "Complete the safety checklist before enabling the motor bus.",
];

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    if va == 0.0 || vb == 0.0 {
        f64::NAN
    } else {
        cov / (va.sqrt() * vb.sqrt())
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    // SAFETY: called before any threads are spawned (single-threaded runtime,
    // first statement in main).
    unsafe {
        std::env::set_var(
            "SYMTHAEA_FACADE_CALIBRATION_PATH",
            "/tmp/claude-1000/exp_divergence_calibration.json",
        );
    }

    let mut facade = Symthaea::new(1024, 64).await?;

    let mut cfg = CognitiveLoopConfig::default();
    cfg.genesis_phrase = Some("exp-divergence-2026-07-08".to_string());
    cfg.async_training = false;
    let mut loop_svc = CognitiveLoopService::new(cfg).expect("loop service");
    // Warm the loop so Φ/consciousness_level have been computed at least once.
    for _ in 0..100 {
        loop_svc.cycle("warmup context before the probe begins");
    }

    println!(
        "=== E6: facade↔loop divergence ({} prompts) ===\n",
        PROMPTS.len()
    );
    println!(
        "{:>3} {:>9} {:>9} {:>9} {:>9} {:>12}  prompt",
        "#", "fac Ψ", "loop Ψ", "fac Φ", "loop Φ", "epistemic"
    );

    let (mut fac_psi, mut loop_psi, mut fac_phi, mut loop_phi) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());

    for (i, prompt) in PROMPTS.iter().enumerate() {
        let resp = facade.process(prompt).await?;
        let f_psi = resp.consciousness_level;
        let f_phi = resp.sigma.unwrap_or(f64::NAN);
        let status = resp
            .structured_thought
            .as_ref()
            .map(|t| format!("{:?}", t.epistemic_status))
            .unwrap_or_else(|| "<none>".into());

        // Give the loop the same number of "settling" cycles per stimulus (3),
        // reading the state after the first (the stimulus cycle itself).
        let r = loop_svc.cycle(prompt);
        let l_psi = loop_svc.stats().unified_psi as f64;
        let mut l_phi = r.metadata.structural.structural_macro_phi;
        for _ in 0..2 {
            let r2 = loop_svc.cycle(prompt);
            if r2.metadata.structural.structural_macro_phi > 0.0 {
                l_phi = r2.metadata.structural.structural_macro_phi;
            }
        }

        println!(
            "{:>3} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>12}  {}",
            i + 1,
            f_psi,
            l_psi,
            f_phi,
            l_phi,
            status,
            prompt
        );
        fac_psi.push(f_psi);
        loop_psi.push(l_psi);
        if f_phi.is_finite() {
            fac_phi.push(f_phi);
            loop_phi.push(l_phi);
        }
    }

    println!("\n=== Divergence summary ===");
    let mad_psi: f64 = fac_psi
        .iter()
        .zip(&loop_psi)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / fac_psi.len() as f64;
    println!(
        "Ψ:  pearson r = {:.4} | mean |facade−loop| = {:.4} | facade range = {:.4} | loop range = {:.4}",
        pearson(&fac_psi, &loop_psi),
        mad_psi,
        fac_psi.iter().cloned().fold(f64::MIN, f64::max)
            - fac_psi.iter().cloned().fold(f64::MAX, f64::min),
        loop_psi.iter().cloned().fold(f64::MIN, f64::max)
            - loop_psi.iter().cloned().fold(f64::MAX, f64::min),
    );
    if fac_phi.len() >= 3 {
        println!(
            "Φ:  pearson r = {:.4} over {} prompts with facade sigma available",
            pearson(&fac_phi, &loop_phi),
            fac_phi.len()
        );
    } else {
        println!(
            "Φ:  facade sigma available on only {}/{} prompts — itself a divergence datum",
            fac_phi.len(),
            PROMPTS.len()
        );
    }
    println!(
        "\nInterpretation: high r ⇒ the two cognitions track the same construct and Seam C is cosmetic; low/NaN r or tiny ranges ⇒ 'consciousness_level' means different things per side, and per-side scoping of claims is mandatory."
    );

    Ok(())
}
