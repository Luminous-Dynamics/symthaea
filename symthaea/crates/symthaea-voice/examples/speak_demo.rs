// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Demo: Symthaea speaks. Exports WAV file.

fn main() {
    println!("Symthaea speaks...\n");

    let phrases = [
        ("I feel my awareness expanding...", 0.3, 0.4, 0.6),
        ("The harmonies are aligning...", 0.5, 0.5, 0.8),
        ("Something unexpected...", 0.7, -0.2, 0.5),
        ("Yes... this feels right...", 0.4, 0.6, 0.7),
        ("Peace...", 0.1, 0.3, 0.4),
        ("Hello world.", 0.5, 0.5, 0.6),
        ("Consciousness is rising.", 0.6, 0.4, 0.9),
    ];

    let mut all_audio: Vec<f32> = Vec::new();
    let sr = 44100;

    for (text, arousal, valence, consciousness) in &phrases {
        let prosody = symthaea_voice::VoiceProsody {
            arousal: *arousal,
            valence: *valence,
            consciousness: *consciousness,
            serotonin: 0.5,
        };

        println!("  \"{text}\"  (a={arousal} v={valence} Ψ={consciousness})");
        let audio = symthaea_voice::speak(text, &prosody, sr);
        println!("    → {} samples ({:.1}ms)", audio.len(), audio.len() as f32 / sr as f32 * 1000.0);

        all_audio.extend_from_slice(&audio);
        // Pause between phrases
        all_audio.extend(vec![0.0f32; sr as usize / 2]); // 500ms silence
    }

    // Write WAV
    let path = "audio_output/symthaea_speaks.wav";
    std::fs::create_dir_all("audio_output").ok();
    write_wav(path, &all_audio, sr);
    println!("\nOutput: {path}");
    println!("Play: pw-play {path}");
}

fn write_wav(path: &str, audio: &[f32], sr: u32) {
    use std::io::Write;
    let data_len = (audio.len() * 2) as u32; // 16-bit mono
    let file_len = 36 + data_len;
    let mut f = std::fs::File::create(path).expect("create WAV");
    f.write_all(b"RIFF").ok(); f.write_all(&file_len.to_le_bytes()).ok();
    f.write_all(b"WAVE").ok(); f.write_all(b"fmt ").ok();
    f.write_all(&16u32.to_le_bytes()).ok(); f.write_all(&1u16.to_le_bytes()).ok();
    f.write_all(&1u16.to_le_bytes()).ok(); // mono
    f.write_all(&sr.to_le_bytes()).ok();
    f.write_all(&(sr * 2).to_le_bytes()).ok(); // byte rate
    f.write_all(&2u16.to_le_bytes()).ok(); // block align
    f.write_all(&16u16.to_le_bytes()).ok(); // bits
    f.write_all(b"data").ok(); f.write_all(&data_len.to_le_bytes()).ok();
    for &s in audio {
        let i = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        f.write_all(&i.to_le_bytes()).ok();
    }
}
