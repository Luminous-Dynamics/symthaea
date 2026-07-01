// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Quick diagnostic for voice synthesis
use symthaea::voice::{ArticulatorySynthesizer, FormantVocoder, LTCPacing};

fn main() {
    let mut synth = ArticulatorySynthesizer::new();
    let mut vocoder = FormantVocoder::new();
    let pacing = LTCPacing::default();

    let arpabet = "HH AH0 L OW1";
    let frames = synth.synthesize_arpabet(arpabet, &pacing);

    println!("Generated {} frames", frames.len());
    if let Some(f) = frames.first() {
        println!(
            "First frame: f0={:.1} f1={:.1} f2={:.1} f3={:.1} energy={:.3} voicing={:.1}",
            f.f0, f.f1, f.f2, f.f3, f.energy, f.voicing
        );
    }
    if let Some(f) = frames.get(frames.len() / 2) {
        println!(
            "Mid frame: f0={:.1} f1={:.1} f2={:.1} f3={:.1} energy={:.3} voicing={:.1}",
            f.f0, f.f1, f.f2, f.f3, f.energy, f.voicing
        );
    }

    let samples = vocoder.synthesize(&frames);
    println!("\nGenerated {} samples", samples.len());

    let non_zero = samples.iter().filter(|&&s| s.abs() > 0.0001).count();
    println!(
        "Non-zero samples: {} ({:.1}%)",
        non_zero,
        non_zero as f32 / samples.len() as f32 * 100.0
    );

    let max = samples.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    println!("Max amplitude: {:.6}", max);
    println!("RMS: {:.6}", rms);
}
