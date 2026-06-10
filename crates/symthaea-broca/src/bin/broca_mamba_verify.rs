// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! broca-mamba-verify: Verification suite for the mamba-130m integration.
//!
//! Runs 10 sequential checks proving the Mamba SSM model loads, generates,
//! and integrates with the HDC projection pipeline.
//!
//! Usage:
//!   broca-mamba-verify [--model state-spaces/mamba-130m]
//!
//! Exit code 0 = all checks pass.

use std::process;
use std::time::Instant;

use symthaea_broca::mamba::MambaWrapper;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".parse().unwrap()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let model_id = parse_model_arg(&args);

    println!("=== Broca Mamba Verification ===");
    println!("Model: {model_id}\n");

    let mut passed = 0usize;
    let mut failed = 0usize;
    let total = 10;

    // Check 1: Load model
    let start = Instant::now();
    print!("[1/10] Load model... ");
    let mut wrapper = match MambaWrapper::load(&model_id, symthaea_broca::mamba::best_device()) {
        Ok(w) => {
            println!("PASS ({:.1}s)", start.elapsed().as_secs_f32());
            passed += 1;
            w
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
            eprintln!("\nFATAL: Cannot load model, aborting remaining checks.");
            print_summary(passed, failed, total);
            process::exit(1);
        }
    };

    // Check 2: Dimensions
    print!("[2/10] Dimensions... ");
    let d_ok = wrapper.d_model() == 768;
    let n_ok = wrapper.n_layer() == 24;
    let v_ok = wrapper.vocab_size() >= 50257;
    if d_ok && n_ok && v_ok {
        println!(
            "PASS (d_model={}, n_layer={}, vocab={})",
            wrapper.d_model(),
            wrapper.n_layer(),
            wrapper.vocab_size()
        );
        passed += 1;
    } else {
        println!(
            "FAIL (d_model={}, n_layer={}, vocab={})",
            wrapper.d_model(),
            wrapper.n_layer(),
            wrapper.vocab_size()
        );
        failed += 1;
    }

    // Check 3: Forward pass
    print!("[3/10] Forward pass... ");
    let start = Instant::now();
    match wrapper.forward_one_token(0) {
        Ok(logits) => {
            let len_ok = logits.len() >= 50257;
            let finite_ok = logits.iter().all(|x| x.is_finite());
            if len_ok && finite_ok {
                println!(
                    "PASS (logits={}, {:.1}ms)",
                    logits.len(),
                    start.elapsed().as_secs_f32() * 1000.0
                );
                passed += 1;
            } else {
                println!("FAIL (len_ok={len_ok}, finite_ok={finite_ok})");
                failed += 1;
            }
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // Check 4: Context injection
    print!("[4/10] Context injection... ");
    wrapper.reset();
    let context = vec![0.1f32; 768];
    match wrapper.inject_initial_context(&context) {
        Ok(()) => {
            println!("PASS");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // Check 5: State scaling
    print!("[5/10] State scaling... ");
    wrapper.reset();
    let _ = wrapper.forward_one_token(0); // Get non-zero state
    let scale_down = wrapper.scale_hidden_states(0.5);
    let scale_up = wrapper.scale_hidden_states(2.0);
    if scale_down.is_ok() && scale_up.is_ok() {
        println!("PASS (0.5x then 2.0x)");
        passed += 1;
    } else {
        println!("FAIL");
        failed += 1;
    }

    // Check 6: Encode/decode round-trip
    print!("[6/10] Encode/decode... ");
    let test_text = "Hello, world!";
    match wrapper.encode(test_text) {
        Ok(ids) => match wrapper.decode(&ids) {
            Ok(decoded) => {
                if decoded.trim() == test_text {
                    println!("PASS ({} tokens)", ids.len());
                    passed += 1;
                } else {
                    println!("FAIL (roundtrip mismatch: {decoded:?})");
                    failed += 1;
                }
            }
            Err(e) => {
                println!("FAIL (decode): {e}");
                failed += 1;
            }
        },
        Err(e) => {
            println!("FAIL (encode): {e}");
            failed += 1;
        }
    }

    // Check 7: Reset produces different logits
    print!("[7/10] Reset... ");
    wrapper.reset();
    let logits_pre = wrapper.forward_one_token(42).unwrap_or_default();
    wrapper.reset();
    let logits_post = wrapper.forward_one_token(42).unwrap_or_default();
    // After reset, same input should produce same logits (determinism)
    let same = logits_pre
        .iter()
        .zip(logits_post.iter())
        .all(|(a, b)| (a - b).abs() < 1e-5);
    // But after running more tokens, different state should yield different output
    let _ = wrapper.forward_one_token(100);
    let logits_diff = wrapper.forward_one_token(42).unwrap_or_default();
    let differs = logits_pre
        .iter()
        .zip(logits_diff.iter())
        .any(|(a, b)| (a - b).abs() > 1e-3);
    if same && differs {
        println!("PASS (reset restores, context diverges)");
        passed += 1;
    } else {
        println!("FAIL (same={same}, differs={differs})");
        failed += 1;
    }

    // Check 8: Embedding extraction
    print!("[8/10] Embedding... ");
    match wrapper.embedding_vector(42) {
        Ok(emb) => {
            let is_768 = emb.len() == 768;
            let non_zero = emb.iter().any(|x| x.abs() > 1e-10);
            if is_768 && non_zero {
                println!("PASS (768D, non-zero)");
                passed += 1;
            } else {
                println!("FAIL (len={}, non_zero={non_zero})", emb.len());
                failed += 1;
            }
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // Check 9: Generate 10 tokens from prompt
    print!("[9/10] Generate 10 tokens... ");
    wrapper.reset();
    let start = Instant::now();
    match generate_tokens(&mut wrapper, "The meaning of life is", 10) {
        Ok((ids, text)) => {
            let all_ascii = text.chars().all(|c| c.is_ascii() || c.is_alphanumeric());
            if ids.len() == 10 && all_ascii {
                println!(
                    "PASS ({:.0}ms, {:?})",
                    start.elapsed().as_secs_f32() * 1000.0,
                    text.trim()
                );
                passed += 1;
            } else {
                println!("FAIL (len={}, ascii={all_ascii}, text={text:?})", ids.len());
                failed += 1;
            }
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // Check 10: E2E pipeline (ThoughtChannels → HDC → projection → Mamba → text)
    print!("[10/10] E2E pipeline... ");
    wrapper.reset();
    let start = Instant::now();
    match run_e2e_pipeline(&mut wrapper) {
        Ok(text) => {
            println!(
                "PASS ({:.0}ms, {:?})",
                start.elapsed().as_secs_f32() * 1000.0,
                text.trim()
            );
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    println!();
    print_summary(passed, failed, total);

    if failed > 0 {
        process::exit(1);
    }
}

fn generate_tokens(
    wrapper: &mut MambaWrapper,
    prompt: &str,
    count: usize,
) -> Result<(Vec<u32>, String), String> {
    let prompt_ids = wrapper.encode(prompt).map_err(|e| format!("encode: {e}"))?;

    let mut last_logits = vec![];
    for &token_id in &prompt_ids {
        last_logits = wrapper
            .forward_one_token(token_id)
            .map_err(|e| format!("forward: {e}"))?;
    }

    let mut generated = Vec::new();
    for _ in 0..count {
        let next_id = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        generated.push(next_id);
        last_logits = wrapper
            .forward_one_token(next_id)
            .map_err(|e| format!("forward: {e}"))?;
    }

    let text = wrapper
        .decode(&generated)
        .map_err(|e| format!("decode: {e}"))?;
    Ok((generated, text))
}

fn run_e2e_pipeline(wrapper: &mut MambaWrapper) -> Result<String, String> {
    use symthaea_broca::encoder::{ThoughtChannels, ThoughtLanguageEncoder};
    use symthaea_broca::projection::HdcSsmProjection;
    use symthaea_core::genesis::GenesisSeed;

    let genesis = GenesisSeed::from_phrase("broca-mamba-verify");

    // 1. Create thought channels
    let mut channels = ThoughtChannels::with_intent(1); // Answer
    channels.set_epistemic(0.0); // Certain
    channels.set_emotion(0.5, 0.5, 0.6);

    // 2. Encode to HDC
    let encoder = ThoughtLanguageEncoder::new(&genesis);
    let thought_hv = encoder.encode(&channels);

    // 3. Project HDC → SSM
    let projection = HdcSsmProjection::new(&genesis, 16384, 256, 768);
    let ssm_context = projection.project_to_ssm(&thought_hv);

    // 4. Inject into Mamba
    wrapper
        .inject_initial_context(&ssm_context)
        .map_err(|e| format!("inject: {e}"))?;

    // 5. Generate text
    let (_, text) = generate_tokens(wrapper, "", 5).or_else(|_| {
        // If empty prompt fails, try with BOS
        let eos_id = wrapper.eos_token_id();
        let mut logits = wrapper
            .forward_one_token(eos_id)
            .map_err(|e| format!("forward: {e}"))?;
        let mut ids = Vec::new();
        for _ in 0..5 {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            ids.push(next);
            logits = wrapper
                .forward_one_token(next)
                .map_err(|e| format!("forward: {e}"))?;
        }
        let text = wrapper.decode(&ids).map_err(|e| format!("decode: {e}"))?;
        Ok::<_, String>((ids, text))
    })?;

    if text.is_empty() {
        return Err("E2E produced empty text".to_string());
    }

    Ok(text)
}

fn print_summary(passed: usize, failed: usize, total: usize) {
    println!("--- Summary ---");
    println!("  Passed: {passed}/{total}");
    println!("  Failed: {failed}/{total}");
    if failed == 0 {
        println!("  Status: ALL CHECKS PASSED");
    } else {
        println!("  Status: SOME CHECKS FAILED");
    }
}

fn parse_model_arg(args: &[String]) -> String {
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                if let Some(val) = args.get(i + 1) {
                    return val.clone();
                }
            }
            "--help" | "-h" => {
                eprintln!("Usage: broca-mamba-verify [--model MODEL_ID]");
                eprintln!();
                eprintln!("Options:");
                eprintln!(
                    "  --model, -m MODEL   HuggingFace model ID (default: state-spaces/mamba-130m)"
                );
                process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    "state-spaces/mamba-130m".to_string()
}
