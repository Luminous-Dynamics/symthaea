// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phone Agent — Symthaea's first embodied task on the Pixel 8 Pro.
//!
//! Interactive perception-action loop:
//! 1. Capture screen → vision manifold (P1-P8) → scene understanding
//! 2. Propose action based on visual state + task goal
//! 3. User confirms or rejects
//! 4. Execute via ADB → observe result → loop
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example phone_agent --features vision-manifold,phone
//! ```

#[cfg(not(feature = "vision-manifold"))]
fn main() {
    eprintln!("Requires: --features vision-manifold");
}

#[cfg(feature = "vision-manifold")]
fn main() {
    use std::io::{self, BufRead, Write};
    use std::time::Instant;

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║     Symthaea — Phone Agent (Embodied Cognition)     ║");
    println!("║                                                     ║");
    println!("║  She sees the screen, reasons about it, and         ║");
    println!("║  proposes actions. You approve or reject.           ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();

    // Initialize phone bridge at 128×128 for better icon discrimination.
    // 64×64 runs at 52Hz but can't distinguish icons. 128×128 runs at
    // ~10Hz but each icon occupies 4-8 patches — enough to tell YouTube
    // (red play button) from Settings (blue gear) from Soma (green fractal).
    let mut phone = symthaea_phone_embodiment::PhoneBridge::with_resolution(
        "41201FDJG000UM",
        1008,
        2244,
        128,
        128,
    );

    // Check device connectivity
    if !phone.adb().is_connected() {
        eprintln!("ERROR: Pixel not connected via ADB. Check USB cable.");
        std::process::exit(1);
    }
    println!("[OK] Pixel 8 Pro connected via ADB\n");

    // Simulated consciousness level — in a full cognitive loop this comes
    // from the consciousness engine. For the demo we use a fixed moderate value.
    let phi: f64 = 0.65; // Green safety level → full control allowed
    println!("[Phi] Consciousness level: {:.2} (Green — full control)", phi);

    // Parse args for mode
    let args: Vec<String> = std::env::args().collect();
    let max_steps: u32 = args
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let interactive = args.iter().any(|a| a == "--interactive");
    let task: Option<String> = args
        .iter()
        .position(|a| a == "--task")
        .and_then(|i| args.get(i + 1))
        .cloned();

    // Load goal template if task specified.
    // Strategy: first capture a reference frame, then learn the template
    // directly from the live screen patches (avoids resolution mismatch).
    let goal_hv: Option<symthaea_core::hdc::ContinuousHV> = task.as_ref().and_then(|task_name| {
        // Known app positions on the Pixel home screen (128×128, 16×16 grid).
        // YouTube icon center: approximately grid row=5, col=8 (spans 2×2 patches)
        let known_positions: &[(&str, usize, usize, usize, usize)] = &[
            // (name, row_start, col_start, rows, cols)
            ("youtube",     4, 8, 2, 2),
            ("settings",    8, 8, 2, 2),
            ("clock",       4, 12, 2, 2),
            ("spotify",     8, 0, 2, 2),
            ("authenticator", 4, 0, 2, 2),
        ];

        let name_lower = task_name.to_lowercase();
        if let Some(&(_, row, col, rows, cols)) = known_positions.iter().find(|(n, _, _, _, _)| *n == name_lower.as_str()) {
            // Capture a reference frame first to learn the template from
            println!("[Goal] Learning template from live screen at grid ({row},{col})...");
            if phone.capture_and_observe(0.033).is_ok() {
                if let Some(hv) = phone.learn_template_from_region(row, col, rows, cols) {
                    println!("[Goal] Learned template from {rows}×{cols} patches");
                    return Some(hv);
                }
            }
            println!("[Goal] Failed to learn from screen. Falling back to saliency.");
            None
        } else {
            // Try file-based template
            let template_path = std::path::Path::new("data/phone-templates")
                .join(format!("{}.png", name_lower.replace(' ', "_")));
            if template_path.exists() {
                match phone.learn_template_from_file(&template_path) {
                    Ok(hv) => {
                        println!("[Goal] Loaded visual template: {}", template_path.display());
                        Some(hv)
                    }
                    Err(e) => {
                        eprintln!("[Goal] Template load failed: {e}");
                        None
                    }
                }
            } else {
                println!("[Goal] Unknown target \"{task_name}\". Using saliency exploration.");
                None
            }
        }
    });

    if let Some(ref task_name) = task {
        println!("[Task] \"find {}\"", task_name);
        if goal_hv.is_some() {
            println!("[Strategy] Visual template matching (exploitation)\n");
        } else {
            println!("[Strategy] Saliency-driven exploration (no template)\n");
        }
    } else if interactive {
        println!("[Mode] Interactive — Symthaea proposes, you confirm\n");
    } else {
        println!("[Mode] Autonomous — Symthaea acts on her own ({max_steps} steps)\n");
    }

    // Disable confirmation mode for autonomous operation.
    // Safety is enforced by Phi-gated actions (NRC 4-tier model).
    phone.set_confirmation_mode(interactive);

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for step in 1..=max_steps {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Step {step}/{max_steps}: Perceiving...");

        // 1. Perceive
        let t0 = Instant::now();
        let tel = match phone.capture_and_observe(0.033) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  Capture failed: {e}");
                continue;
            }
        };
        let perceive_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // 2. Report what she sees
        println!("  [{:.0}ms] PE={:.3} ImgSurp={:.3} Coh={:.3} motion={:.3}",
            perceive_ms, tel.prediction_error, tel.imagination_surprise,
            tel.manifold_coherence, tel.motion_surprise);
        println!("  WM={}/4  SG={}edges",
            tel.working_memory_load, tel.scene_graph_edges);

        // 3. Working memory — what she's attending to
        let wm_summary = phone.working_memory_summary();
        if !wm_summary.is_empty() {
            print!("  Attending:");
            for (track_id, sal, sx, sy) in &wm_summary {
                print!(" #{track_id}@({sx},{sy})[{sal:.2}]");
            }
            println!();
        }

        // 4. Scene description
        let desc = phone.describe_scene();
        if !desc.is_empty() {
            print!("  Scene:");
            for (s, r, o) in desc.iter().take(3) {
                print!(" {s} {r} {o};");
            }
            println!();
        }

        // 5. Propose action (goal-directed if template available, else saliency)
        let action = if let Some(ref ghv) = goal_hv {
            phone.propose_goal_action(phi, ghv, 0.3)
        } else {
            phone.propose_action(phi)
        };
        let match_info = phone.last_match_similarity()
            .map(|s| format!(" [MATCH sim={s:.3}]"))
            .unwrap_or_default();
        println!("  ACTION: {} (phi_req={:.2}){match_info}", action.label(), action.required_phi());

        if interactive {
            // Interactive mode: ask for confirmation
            print!("  [y/n/q] > ");
            stdout.flush().unwrap();
            let mut input = String::new();
            stdin.lock().read_line(&mut input).unwrap();
            let input = input.trim().to_lowercase();
            match input.as_str() {
                "y" | "yes" => {
                    match phone.confirm_and_execute() {
                        Ok(()) => println!("  [OK] Executed."),
                        Err(e) => println!("  [ERR] {e}"),
                    }
                }
                "q" | "quit" => {
                    println!("\nSymthaea returns to stillness.");
                    break;
                }
                _ => println!("  Skipped."),
            }
        } else {
            // Autonomous mode: execute directly
            if action.is_mutating() {
                match phone.execute_action(&action) {
                    Ok(()) => println!("  [EXEC] {}", action.label()),
                    Err(e) => println!("  [ERR] {e}"),
                }
            } else {
                println!("  [OBSERVE] No mutating action needed.");
            }
        }

        // Pause for screen to settle, then detect state transition
        std::thread::sleep(std::time::Duration::from_millis(800));

        // Re-observe after action to detect state transition
        if action.is_mutating() {
            if phone.capture_and_observe(0.033).is_ok() {
                let (transitioned, confidence) = phone.detect_state_transition();
                if transitioned {
                    println!("  [TRANSITION] Screen changed (confidence={confidence:.3})");
                    // If we had a goal and the screen changed, we likely succeeded
                    if goal_hv.is_some() {
                        println!("  [SUCCESS] Goal-directed action caused state transition");
                    }
                } else {
                    println!("  [STABLE] No significant screen change");
                }
            }
        }
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Agent complete after {max_steps} steps.");
    println!("Final scene:");
    for (s, r, o) in phone.describe_scene().iter().take(5) {
        println!("  {s} {r} {o}");
    }
}
