// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end proof of the Tier 1.2 CLS threshold-phenotype promotion path
//! (`DISCOVERY_AND_SELF_IMPROVEMENT_PLAN_2026-07-06.md`):
//!
//!   evolve (tiny) -> gate (fresh-seed re-eval) -> promote (the real
//!   `scripts/cls_promote_candidate.sh`) -> construct a REAL
//!   `CognitiveLoopService` with `SYMTHAEA_THRESHOLD_OVERRIDES_PATH` set ->
//!   assert the LIVE instance's effective thresholds match the promoted
//!   phenotype, not compile-time defaults.
//!
//! This is the acceptance criterion for Tier 1.2: "an evolved threshold
//! phenotype runs in a live loop via the promotion path." Every step uses the
//! real production code paths (`cls_evolution_harness::evaluate_with_cls`
//! against a REAL `CognitiveLoopService`, the real promote shell script, the
//! real `ThresholdOverrides::from_env()` constructor wiring) — nothing here
//! is a mock of the promotion mechanism itself. Only the evolution population
//! size / generation count / cycle count are shrunk for test speed (per the
//! task's explicit allowance).
//!
//! Requires: `cargo test --release --features neuroevolution --test cls_threshold_promotion_e2e`
//! (release strongly recommended — this constructs multiple real
//! `CognitiveLoopService` instances and runs real cognitive cycles).

#![cfg(feature = "neuroevolution")]

use symthaea::cognitive_loop::cls_evolution_harness::{
    CandidateProvenance, ClsFitness, EVOLUTION_INPUTS, FRESH_INPUTS, PromotionReady,
    current_git_sha, evaluate_with_cls,
};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea_core::genesis::GenesisSeed;
use symthaea_neuroevolution::{
    NeuralGenome, ThresholdPhenotype,
    threshold_genome::{decode_thresholds, evaluate_threshold_fitness},
};

fn unique_temp_dir(label: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{label}_{}_{nanos}", std::process::id()))
}

#[test]
fn evolved_threshold_phenotype_reaches_a_live_service_via_promotion_path() {
    // ── Step 1: tiny/fast "evolution" — pop 3, single generation, 2 cycles
    // each. Small enough to run in a test suite, still a real search over
    // real genome-decoded phenotypes scored by the REAL CLS harness (never a
    // proxy) — the same `evaluate_with_cls` function `evolve_cls.rs` uses. ──
    let eval_cycles = 2;
    let genesis = GenesisSeed::from_phrase("cls-promotion-e2e-test");
    let candidates: Vec<ThresholdPhenotype> = (0..3)
        .map(|i| {
            let genome = NeuralGenome::from_genesis(&genesis, &format!("e2e-org-{i}"));
            decode_thresholds(&genome.hv)
        })
        .collect();

    let defaults = ThresholdPhenotype::default();
    let default_fitness = evaluate_with_cls(&defaults, EVOLUTION_INPUTS, eval_cycles);

    let mut best: Option<(ThresholdPhenotype, ClsFitness)> = None;
    for pheno in &candidates {
        let mut fitness = evaluate_with_cls(pheno, EVOLUTION_INPUTS, eval_cycles);
        fitness.threshold_consistency = evaluate_threshold_fitness(pheno);
        let is_better = best
            .as_ref()
            .map(|(_, f)| fitness.composite() > f.composite())
            .unwrap_or(true);
        if is_better {
            best = Some((pheno.clone(), fitness));
        }
    }
    let (best_pheno, best_fitness) = best.expect("at least one candidate evaluated");

    // This test is only meaningful if the "evolved" phenotype actually
    // differs from compile-time defaults — otherwise a passing assertion at
    // the end wouldn't prove promotion did anything.
    assert_ne!(
        best_pheno, defaults,
        "evolved candidate must differ from defaults for this test to prove anything"
    );

    // ── Write candidate + provenance, same shape `evolve_cls.rs` writes,
    // to an isolated temp dir (never a path the live system reads). ────────
    let candidate_dir = unique_temp_dir("cls_e2e_candidate");
    std::fs::create_dir_all(&candidate_dir).unwrap();
    let phenotype_path = candidate_dir.join("candidate-phenotype.json");
    std::fs::write(
        &phenotype_path,
        serde_json::to_string_pretty(&best_pheno).unwrap(),
    )
    .unwrap();

    let provenance = CandidateProvenance {
        created_at_utc: chrono::Utc::now().to_rfc3339(),
        git_sha: current_git_sha(),
        pop_size: candidates.len(),
        generations: 1,
        eval_cycles,
        genesis_seed_phrase: "cls-promotion-e2e-test".to_string(),
        evolution_input_count: EVOLUTION_INPUTS.len(),
        default_fitness,
        final_fitness: best_fitness.clone(),
    };
    std::fs::write(
        candidate_dir.join("provenance.json"),
        serde_json::to_string_pretty(&provenance).unwrap(),
    )
    .unwrap();

    // ── Step 2: gate — re-evaluate on FRESH seeds, disjoint from
    // EVOLUTION_INPUTS, using the same real harness. ────────────────────────
    let consistency = evaluate_threshold_fitness(&best_pheno);
    assert!(
        consistency >= 0.5,
        "candidate too internally inconsistent for a meaningful gate test: {consistency}"
    );

    let mut fresh_fitness = evaluate_with_cls(&best_pheno, FRESH_INPUTS, eval_cycles);
    fresh_fitness.threshold_consistency = consistency;

    // Tolerance is generous here deliberately: this test's job is to prove
    // the PIPELINE moves data end-to-end, not to validate the real gate's
    // statistical strictness (`cls_promotion_gate.rs`'s default 10%
    // tolerance) under eval_cycles=2, where composite scores are inherently
    // noisy. The real gate's tolerance is exercised by actual evolution runs
    // with realistic cycle counts.
    let tolerance = 1.0;
    let recorded_composite = provenance.final_fitness.composite();
    let fresh_composite = fresh_fitness.composite();
    let required = recorded_composite * (1.0 - tolerance);
    assert!(
        fresh_composite >= required,
        "gate should pass with tolerance=1.0 (recorded={recorded_composite}, fresh={fresh_composite})"
    );

    let promotion_ready = PromotionReady {
        candidate_phenotype_path: phenotype_path
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .to_string(),
        created_at_utc: chrono::Utc::now().to_rfc3339(),
        gate_git_sha: current_git_sha(),
        recorded_fitness: provenance.final_fitness.clone(),
        fresh_fitness,
        fresh_input_count: FRESH_INPUTS.len(),
        eval_cycles,
        tolerance,
        passed: true,
    };
    std::fs::write(
        candidate_dir.join("PROMOTION_READY.json"),
        serde_json::to_string_pretty(&promotion_ready).unwrap(),
    )
    .unwrap();

    // ── Step 3: promote via the REAL script. In production this is always a
    // deliberate human action with the confirm flag typed by hand; here the
    // test supplies it explicitly to prove the mechanism, exactly as a human
    // operator would invoke it. ─────────────────────────────────────────────
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let active_root = unique_temp_dir("cls_e2e_active");
    std::fs::create_dir_all(&active_root).unwrap();
    let cls_data_dir = active_root.join("cls-thresholds");

    let script = repo_root.join("scripts/cls_promote_candidate.sh");
    let output = std::process::Command::new("bash")
        .arg(&script)
        .arg(&candidate_dir)
        .arg("--i-understand-this-is-live")
        .env("CLS_DATA_DIR", cls_data_dir.to_string_lossy().to_string())
        .current_dir(repo_root)
        .output()
        .expect("failed to run cls_promote_candidate.sh");

    assert!(
        output.status.success(),
        "promote script failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let active_path = cls_data_dir.join("active/threshold-overrides-active.json");
    assert!(
        active_path.exists(),
        "promote script should have written {active_path:?}"
    );

    // ── Step 4: construct a REAL live CognitiveLoopService with the env var
    // pointed at the promoted file — exactly what an operator would do on
    // restart. ───────────────────────────────────────────────────────────────
    // SAFETY: single-threaded test process section; no other thread reads
    // this env var concurrently within this test binary's lifetime for this
    // key.
    unsafe {
        std::env::set_var("SYMTHAEA_THRESHOLD_OVERRIDES_PATH", &active_path);
    }
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default())
        .expect("CognitiveLoopService::new should succeed with promoted overrides set");
    unsafe {
        std::env::remove_var("SYMTHAEA_THRESHOLD_OVERRIDES_PATH");
    }

    // ── Step 5 (the acceptance criterion): the LIVE instance's effective
    // thresholds match the PROMOTED phenotype's values, not compile-time
    // defaults. ──────────────────────────────────────────────────────────────
    let overrides = service.threshold_overrides();
    assert_eq!(
        overrides.fep_surprise_scale(),
        best_pheno.fep_surprise_scale
    );
    assert_eq!(overrides.fep_lr_decay(), best_pheno.fep_lr_decay);
    assert_eq!(
        overrides.dream_base_interval(),
        best_pheno.dream_base_interval
    );
    assert_eq!(
        overrides.dream_min_interval(),
        best_pheno.dream_min_interval
    );
    assert_eq!(
        overrides.neuromod_d2_baseline(),
        best_pheno.neuromod_d2_baseline
    );
    assert_eq!(
        overrides.neuromod_ne_phasic_threshold(),
        best_pheno.neuromod_ne_phasic_threshold
    );
    assert_eq!(
        overrides.neuromod_arousal_ema_decay(),
        best_pheno.neuromod_arousal_ema_decay
    );
    assert_eq!(
        overrides.homeostasis_recalibrate_high(),
        best_pheno.homeostasis_recalibrate_high
    );
    assert_eq!(
        overrides.homeostasis_recalibrate_low(),
        best_pheno.homeostasis_recalibrate_low
    );
    assert_eq!(
        overrides.neuromod_ema_alpha(),
        best_pheno.neuromod_ema_alpha
    );
    assert_eq!(
        overrides.frustration_dampen_threshold(),
        best_pheno.frustration_dampen_threshold
    );
    assert_eq!(
        overrides.engagement_low_threshold(),
        best_pheno.engagement_low_threshold
    );
    assert_eq!(
        overrides.flow_exploration_increment(),
        best_pheno.flow_exploration_increment
    );
    assert_eq!(overrides.coherence_low(), best_pheno.coherence_low);
    assert_eq!(
        overrides.arousal_trap_threshold(),
        best_pheno.arousal_trap_threshold
    );
    assert_eq!(
        overrides.self_model_weight_high(),
        best_pheno.self_model_weight_high
    );
    assert_eq!(
        overrides.homeostasis_pull_cruise(),
        best_pheno.homeostasis_pull_cruise
    );
    assert_eq!(
        overrides.confidence_crash_threshold(),
        best_pheno.confidence_crash_threshold
    );

    // Since best_pheno != defaults (asserted above) and every override field
    // now equals best_pheno's value, the live service is provably running
    // the promoted phenotype, not compile-time defaults.

    std::fs::remove_dir_all(&candidate_dir).ok();
    std::fs::remove_dir_all(&active_root).ok();
    drop(service);
}
