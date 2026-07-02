// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[cfg(all(feature = "vision-manifold", feature = "swarm", feature = "muse"))]
use hound::WavReader;
#[cfg(all(feature = "vision-manifold", feature = "swarm", feature = "muse"))]
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, imageops};
#[cfg(all(feature = "vision-manifold", feature = "swarm", feature = "muse"))]
use std::fs;
#[cfg(all(feature = "vision-manifold", feature = "swarm", feature = "muse"))]
use std::time::{Instant, SystemTime, UNIX_EPOCH};
#[cfg(all(feature = "vision-manifold", feature = "swarm", feature = "muse"))]
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, SwarmEvent};
#[cfg(all(feature = "vision-manifold", feature = "swarm", feature = "muse"))]
use symthaea_core::core::ContinuousHV;
#[cfg(all(feature = "vision-manifold", feature = "swarm", feature = "muse"))]
use symthaea_muse::audio_analyzer::AudioSurpriseMeter;
#[cfg(all(feature = "vision-manifold", feature = "swarm", feature = "muse"))]
use symthaea_muse::mel_extractor::MelConfig;
#[cfg(all(feature = "vision-manifold", feature = "swarm", feature = "muse"))]
use symthaea_swarm::SwarmStateMsg;

fn main() {
    #[cfg(all(feature = "vision-manifold", feature = "swarm", feature = "muse"))]
    {
        println!("🎵 Initializing Musical Geodesic (Thermodynamic Visualizer)...");
        let mut config = CognitiveLoopConfig::default();
        config.enable_vision_manifold = true;

        let mut node_a = CognitiveLoopService::new(config.clone()).unwrap();
        let id_a = node_a.node_id().unwrap();
        let mut node_b = CognitiveLoopService::new(config).unwrap();

        // 1. Build High-Res Dictionary from Blueprints
        println!("📂 Inducting Concept Art blueprints...");
        let concept_dir = "/srv/luminous-dynamics/Concept Art";
        let mut visual_dictionary: Vec<(ContinuousHV, ImageBuffer<Rgb<u8>, Vec<u8>>)> = Vec::new();

        if let Ok(entries) = fs::read_dir(concept_dir) {
            let mut paths: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
            paths.sort();
            for path in paths.iter().take(15) {
                if let Ok(img) = image::open(path) {
                    let resized = img.resize_exact(64, 64, imageops::FilterType::Lanczos3);
                    let rgb_crunch = resized.to_rgb8();
                    node_b.inject_vision_frame(rgb_crunch.into_raw());
                    node_b.set_vision_free_energy_override(0.9);
                    let _ = node_b.cycle("learn");
                    if let Some(hv) = node_b.consciousness_hv() {
                        let high_res = img.resize_exact(512, 512, imageops::FilterType::CatmullRom);
                        visual_dictionary.push((hv, high_res.to_rgb8()));
                    }
                }
            }
        }
        println!("✨ Loaded {} blueprint landmarks.", visual_dictionary.len());

        // 2. Analyze the Audio Entropy
        let audio_path = "gallery/music/turbulent.wav";
        println!("🎧 Analyzing musical entropy: {} ...", audio_path);

        let mut reader = WavReader::open(audio_path).expect("Failed to open wav");
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s: Result<f32, _>| s.unwrap())
                .collect(),
            hound::SampleFormat::Int => {
                let max = (1 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s: Result<i32, _>| s.unwrap() as f32 / max)
                    .collect()
            }
        };

        let mut meter = AudioSurpriseMeter::new(MelConfig {
            sample_rate: spec.sample_rate,
            ..Default::default()
        });
        let report = meter.analyze(&samples);
        println!(
            "📈 Audio Analysis Complete. Mean Entropy: {:.3} | Max Surprise: {:.3}",
            report.mean_entropy, report.max_surprise
        );

        // Find the peak surprise index to trigger the "Shock"
        let (peak_idx, &peak_val) = report
            .surprise_curve
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();

        println!(
            "💥 Peak Surprise detected at frame {} (val: {:.3})",
            peak_idx, peak_val
        );

        // 3. Trigger Swarm SOS based on Audio Peak
        println!("📡 Node A encountering musical shock -> Broadcasting SOS...");
        node_a.set_vision_free_energy_override(peak_val);
        let _ = node_a.cycle("musical-shock");

        let sos_msg = SwarmStateMsg {
            node_id: id_a,
            local_phi: 0.9,
            consciousness_hv: node_a.consciousness_hv().unwrap(),
            intent_hv: node_a.last_intent_hv().unwrap(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };
        node_b
            .swarm_manager_mut()
            .inject_event(SwarmEvent::FullStateUpdate(sos_msg));
        for _ in 0..50 {
            let _ = node_b.cycle("sync-swarm");
        }

        // 4. Generate the Musical Geodesic (12 frames)
        println!("🎭 Dreaming a 12-frame trajectory synced to entropy peaks...");
        let start_dream = Instant::now();

        // We use the collaborative imagine helper, which now uses the O(1) CfC projection
        match node_b.collaborative_imagine_future(&id_a, 12) {
            Ok(movie) => {
                std::fs::create_dir_all("dream_captures_musical").unwrap();

                for (i, hv_frame) in movie.trajectory.iter().enumerate() {
                    let traj_dim = hv_frame.values.len();

                    // Match mathematical coordinates to Blueprint Vocabulary
                    let mut distances: Vec<(f32, usize)> = visual_dictionary
                        .iter()
                        .enumerate()
                        .map(|(idx, (dict_hv, _))| {
                            // DILATION SYNC: Ensure landmark matches the trajectory's resolution
                            let synced_hv = if dict_hv.values.len() != traj_dim {
                                dict_hv.dilate(traj_dim)
                            } else {
                                dict_hv.clone()
                            };
                            (hv_frame.similarity(&synced_hv), idx)
                        })
                        .collect();
                    distances.sort_by(|a, b| b.0.total_cmp(&a.0));

                    let (sim1, idx1) = distances[0];
                    let img1 = &visual_dictionary[idx1].1;

                    let mut final_img = img1.clone();

                    if distances.len() > 1 && distances[1].0 > 0.35 {
                        let (sim2, idx2) = distances[1];
                        let img2 = &visual_dictionary[idx2].1;
                        let alpha = sim2 / (sim1 + sim2);

                        for (x, y, pixel) in final_img.enumerate_pixels_mut() {
                            let p2 = img2.get_pixel(x, y);
                            pixel[0] =
                                ((pixel[0] as f32 * (1.0 - alpha)) + (p2[0] as f32 * alpha)) as u8;
                            pixel[1] =
                                ((pixel[1] as f32 * (1.0 - alpha)) + (p2[1] as f32 * alpha)) as u8;
                            pixel[2] =
                                ((pixel[2] as f32 * (1.0 - alpha)) + (p2[2] as f32 * alpha)) as u8;
                        }
                    }

                    let path = format!("dream_captures_musical/frame_{:02}.png", i);
                    final_img.save(&path).unwrap();
                    println!("💾 Saved Frame {:02} | Primary Sim: {:.3}", i, sim1);
                }
                println!(
                    "\n✅ Musical Geodesic complete in {:?}. Blueprints synced to audio entropy.",
                    start_dream.elapsed()
                );
            }
            Err(e) => println!("❌ Geodesic failed: {:?}", e),
        }
    }
}
