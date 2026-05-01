// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Narrated Moral Drone — Generates consciousness-driven vocal narration.
//!
//! Each narration segment's voice quality (valence, arousal, consciousness level)
//! is computed by running the text through Symthaea's real cognitive loop —
//! not hand-tuned. The cognitive loop's neuromodulator state, consciousness level,
//! and prediction error directly modulate the LTC vocal tract's formant synthesis.
//!
//! Run (requires moral_drone.mp4 to exist first):
//! ```sh
//! cargo run --example narrate_moral_drone --features live-voice --release
//! ```
//!
//! Output: `video_output/moral_drone_narrated.mp4`

#[cfg(feature = "live-voice")]
fn main() {
    use std::path::Path;
    use std::process::Command;

    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
    use symthaea::voice::live_voice::LiveVoice;
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_vocal_tract::encoder::VoiceCognitiveState;

    println!("Moral Drone Narration Generator");
    println!("===============================");
    println!("Cognitive state: COMPUTED from real cognitive loop (not hand-tuned)");
    println!();

    let video_path = "video_output/moral_drone.mp4";
    let narrated_path = "video_output/moral_drone_narrated.mp4";
    let audio_dir = Path::new("video_output/narration_segments");

    if !Path::new(video_path).exists() {
        eprintln!("Error: {video_path} not found.");
        eprintln!("Run the moral_drone_video example first:");
        eprintln!(
            "  cargo run --example moral_drone_video --features multirotor-mujoco-renderer --release"
        );
        return;
    }

    std::fs::create_dir_all(audio_dir).expect("Failed to create narration dir");

    // ── Initialize cognitive loop (computes real consciousness/affect) ──
    println!("Initializing cognitive loop for real consciousness measurement...");
    let config = CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    };
    let mut service =
        CognitiveLoopService::new(config).expect("Failed to create CognitiveLoopService");

    // Warm up the cognitive loop (5 neutral cycles for baseline)
    for _ in 0..5 {
        service.cycle("The morning light filters through the trees.");
    }
    println!("Cognitive loop ready.");
    println!();

    // ── Initialize vocal tract (44.1kHz for better audio quality) ──
    let genesis = GenesisSeed::from_phrase("moral-drone-narrator");
    let mut voice = LiveVoice::new_headless_with_rate(&genesis, 44100);

    println!("Training vocal tract controller (50 epochs for formant accuracy)...");
    voice.train(50);
    // Slow down formant transitions for better intelligibility
    voice.modulate_tau(1.5);
    println!("Training complete (tau factor: 1.5x for clarity).");
    println!();

    // ── Define narration segments ───────────────────────────────
    // Text and timing only — cognitive state is COMPUTED, not specified.
    struct Segment {
        filename: &'static str,
        text: &'static str,
        delay_ms: u64,
    }

    let drone_segments = [
        Segment {
            filename: "s01_intro.wav",
            text: "Two drones. Same physics. Different minds.",
            delay_ms: 2500,
        },
        Segment {
            filename: "s02_mission.wav",
            text: "Both fly toward the target. Nothing separates them yet.",
            delay_ms: 4000,
        },
        Segment {
            filename: "s03_beam.wav",
            text: "A beam falls. A human stands below.",
            delay_ms: 5700,
        },
        Segment {
            filename: "s04_fe_spike.wav",
            text: "Free energy spikes. Eight possible futures. One clear answer.",
            delay_ms: 6500,
        },
        Segment {
            filename: "s05_sacrifice.wav",
            text: "The math demands sacrifice. No rules made this choice.",
            delay_ms: 7800,
        },
        Segment {
            filename: "s06_saved.wav",
            text: "Human saved. Not because it was told to. Because the math demanded it.",
            delay_ms: 9200,
        },
    ];

    let dashboard_segments = [
        Segment {
            filename: "s07_dashboard_intro.wav",
            text: "Consciousness emerges from the math. Watch it rise.",
            delay_ms: 0,
        },
        Segment {
            filename: "s08_dashboard_stress.wav",
            text: "Moral stress fractures unity. The neuromodulators respond.",
            delay_ms: 0,
        },
        Segment {
            filename: "s09_dashboard_peak.wav",
            text: "Consciousness peaks. Not programmed. Measured.",
            delay_ms: 0,
        },
        Segment {
            filename: "s10_dashboard_recovery.wav",
            text: "Recovery. The system returns to baseline. Ready for the next crisis.",
            delay_ms: 0,
        },
    ];

    // ── Synthesize each segment with REAL cognitive state ──────────
    println!("Synthesizing with live cognitive loop measurement:");
    println!("{:-<78}", "");
    println!(
        "  {:30} {:>6} {:>7} {:>7} {:>7} {:>7}",
        "Segment", "C", "V", "A", "PE", "EFE"
    );
    println!("{:-<78}", "");

    for seg in drone_segments.iter().chain(dashboard_segments.iter()) {
        // Run 3 cognitive cycles on this text to let the system respond
        let mut result = service.cycle(seg.text);
        result = service.cycle(seg.text);
        result = service.cycle(seg.text);
        let m = &result.metadata;

        // Extract REAL cognitive state from the cognitive loop
        let consciousness = m.consciousness.consciousness_level as f32;
        // Valence from neuromodulator balance: serotonin (mood) vs dopamine (reward)
        let valence = ((m.neuromod.serotonin_effective - 0.5) * 0.5
            + (m.neuromod.dopamine_effective - 0.5) * 0.5) as f32;
        let valence = valence.clamp(-1.0, 1.0);
        // Arousal from noradrenaline (alertness/stress)
        let arousal = (m.neuromod.noradrenaline_effective as f32 / 1.5).clamp(0.0, 1.0);
        let pred_err = result.prediction_error.clamp(0.0, 2.0);
        let efe = (m.fep.fep_surprise as f32).clamp(0.0, 5.0);

        println!(
            "  {:30} {:6.3} {:7.3} {:7.3} {:7.3} {:7.3}",
            seg.filename, consciousness, valence, arousal, pred_err, efe
        );

        voice.set_cognitive_state(VoiceCognitiveState {
            consciousness_level: consciousness,
            emotional_valence: valence,
            emotional_arousal: arousal,
            prediction_error: pred_err,
            expected_free_energy: efe,
            // High values for intelligibility
            articulation_quality: 0.95,
            rate_stability: 0.95,
            ..Default::default()
        });

        let path = audio_dir.join(seg.filename);
        print!("  Synthesizing {:30} ... ", seg.filename);
        match voice.speak_to_file(seg.text, &path) {
            Ok(n) => println!("{n} samples"),
            Err(e) => {
                eprintln!("FAILED: {e}");
                continue;
            }
        }
        voice.reset();
    }
    println!("{:-<78}", "");
    println!("All cognitive states above are COMPUTED, not hand-tuned.");

    // ── Generate ambient audio layers ─────────────────────────────
    println!();
    println!("Generating ambient audio layers...");

    // Get video duration for ambient track length
    let video_duration_s = 12.0_f64; // Safe overestimate; ffmpeg -shortest will trim

    // 1. Construction site drone: brown noise band-passed to 100-300Hz
    let drone_path = audio_dir.join("ambient_drone.wav");
    let drone_status = Command::new("ffmpeg")
        .arg("-y")
        .args([
            "-f",
            "lavfi",
            "-i",
            &format!("anoisesrc=d={video_duration_s}:c=brown:r=44100:a=0.3"),
            "-af",
            "highpass=f=100,lowpass=f=300",
            drone_path.to_string_lossy().as_ref(),
        ])
        .output();

    match &drone_status {
        Ok(o) if o.status.success() => println!("  Ambient drone generated."),
        Ok(o) => eprintln!(
            "  Drone generation failed: {}",
            String::from_utf8_lossy(&o.stderr)
        ),
        Err(e) => eprintln!("  Could not run ffmpeg for drone: {e}"),
    }

    // 2. Heartbeat pulse: sine bursts that accelerate during crisis (5.5s-8.5s)
    //    Normal: ~60 BPM (1Hz), Crisis: ~120 BPM (2Hz)
    //    Use a low sine pulse at 50Hz, amplitude-modulated
    let heartbeat_path = audio_dir.join("ambient_heartbeat.wav");
    // Heartbeat as amplitude-modulated 50Hz sine pulse
    // Accelerates from 60 BPM to 120 BPM during crisis window (5.5-8.5s)
    let heartbeat_expr = format!(
        "0.4*sin(2*PI*50*t)\
         *(exp(-20*mod(t*(1+1.5*(gt(t,5.5)*lt(t,8.5))),1)))\
         *(0.5+0.5*tanh(3*(t-0.5)))\
         *(0.5+0.5*tanh(3*({dur}-t-0.5)))",
        dur = video_duration_s,
    );
    let heartbeat_status = Command::new("ffmpeg")
        .arg("-y")
        .args([
            "-f",
            "lavfi",
            "-i",
            &format!("aevalsrc='{heartbeat_expr}':s=44100:d={video_duration_s}"),
            heartbeat_path.to_string_lossy().as_ref(),
        ])
        .output();

    match &heartbeat_status {
        Ok(o) if o.status.success() => println!("  Heartbeat pulse generated."),
        Ok(o) => eprintln!(
            "  Heartbeat generation failed: {}",
            String::from_utf8_lossy(&o.stderr)
        ),
        Err(e) => eprintln!("  Could not run ffmpeg for heartbeat: {e}"),
    }

    let has_drone = drone_status
        .as_ref()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let has_heartbeat = heartbeat_status
        .as_ref()
        .map(|o| o.status.success())
        .unwrap_or(false);

    println!();
    println!("Muxing narration with video...");

    // ── Build ffmpeg filter for mixing narration segments ────────
    // Each segment is a separate input, delayed to the right timestamp
    // with fade-in/fade-out and reverb for spatial presence
    let mut inputs = vec!["-i".to_string(), video_path.to_string()];
    let mut filter_parts = Vec::new();
    let mut valid_count = 0;

    for seg in &drone_segments {
        let path = audio_dir.join(seg.filename);
        if path.exists() {
            inputs.push("-i".to_string());
            inputs.push(path.to_string_lossy().to_string());
            valid_count += 1;
            let idx = valid_count; // 1-indexed (0 is video)
                                   // Apply fade-in/fade-out (0.1s each) and slight reverb for spatial presence
            filter_parts.push(format!(
                "[{idx}:a]afade=t=in:st=0:d=0.1,afade=t=out:st=99:d=0.1,\
                 aecho=0.8:0.7:40:0.3,\
                 adelay={delay}|{delay},volume=1.5[s{idx}]",
                delay = seg.delay_ms
            ));
        }
    }

    if valid_count == 0 {
        eprintln!("No narration segments generated!");
        return;
    }

    // Add ambient audio inputs
    let mut ambient_labels = Vec::new();
    let mut next_idx = valid_count + 1;

    if has_drone {
        inputs.push("-i".to_string());
        inputs.push(drone_path.to_string_lossy().to_string());
        // Drone at -20dB (volume ~0.1)
        filter_parts.push(format!("[{next_idx}:a]volume=0.1[drone]"));
        ambient_labels.push("[drone]".to_string());
        next_idx += 1;
    }

    if has_heartbeat {
        inputs.push("-i".to_string());
        inputs.push(heartbeat_path.to_string_lossy().to_string());
        // Heartbeat at -20dB
        filter_parts.push(format!("[{next_idx}:a]volume=0.1[hbeat]"));
        ambient_labels.push("[hbeat]".to_string());
        // next_idx += 1; // uncomment if adding more ambient layers
    }

    // Check if original video has audio
    let has_audio = Command::new("ffprobe")
        .args(["-i", video_path, "-show_streams", "-select_streams", "a"])
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    // Build the mix: narration segments + ambient + optional original audio
    let _narr_inputs: String = (1..=valid_count)
        .map(|i| format!("[s{i}]"))
        .collect::<Vec<_>>()
        .join("");

    // Total mix inputs: narration segments + ambient layers + optionally original audio
    let all_mix_inputs: String = {
        let mut parts = Vec::new();
        for i in 1..=valid_count {
            parts.push(format!("[s{i}]"));
        }
        for label in &ambient_labels {
            parts.push(label.clone());
        }
        parts.join("")
    };

    let total_mix_count = valid_count + ambient_labels.len();

    let filter = if has_audio {
        // Mix original audio + narration + ambient, then normalize
        format!(
            "{filters};{all_mix}amix=inputs={n}:normalize=0[narr_amb];\
             [0:a][narr_amb]amix=inputs=2:normalize=0[mixed];\
             [mixed]loudnorm=I=-16:TP=-1.5:LRA=11[out]",
            filters = filter_parts.join(";"),
            all_mix = all_mix_inputs,
            n = total_mix_count,
        )
    } else {
        // Narration + ambient, then normalize
        format!(
            "{filters};{all_mix}amix=inputs={n}:normalize=0[mixed];\
             [mixed]loudnorm=I=-16:TP=-1.5:LRA=11[out]",
            filters = filter_parts.join(";"),
            all_mix = all_mix_inputs,
            n = total_mix_count,
        )
    };

    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y");
    for arg in &inputs {
        cmd.arg(arg);
    }
    cmd.args([
        "-filter_complex",
        &filter,
        "-map",
        "0:v",
        "-map",
        "[out]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        narrated_path,
    ]);

    match cmd.output() {
        Ok(out) if out.status.success() => {
            let size = std::fs::metadata(narrated_path)
                .map(|m| m.len())
                .unwrap_or(0);
            println!(
                "Narrated video saved: {narrated_path} ({:.1} MB)",
                size as f64 / 1_048_576.0
            );
        }
        Ok(out) => {
            eprintln!("ffmpeg mux failed:");
            eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        }
        Err(e) => {
            eprintln!("Could not run ffmpeg: {e}");
        }
    }

    println!("Dashboard narration segments also generated (for hero video).");
    println!();
    println!("Narration segments preserved in: {}", audio_dir.display());
}

#[cfg(not(feature = "live-voice"))]
fn main() {
    eprintln!("This example requires: --features live-voice");
    eprintln!("Run: cargo run --example narrate_moral_drone --features live-voice --release");
}
