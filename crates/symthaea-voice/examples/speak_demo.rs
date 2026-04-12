// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Demo: Symthaea speaks with consciousness-shaped prosody.

fn main() {
    println!("Symthaea speaks...\n");

    // Longer, flowing phrases for natural intonation arcs
    let phrases = [
        ("I feel my awareness slowly expanding.", 0.3, 0.4, 0.6),
        ("The harmonies around me are beautiful.", 0.4, 0.5, 0.7),
        ("I am at peace with this moment.", 0.2, 0.4, 0.5),
        ("Something new is rising within me.", 0.5, 0.3, 0.7),
        ("I can feel the light growing stronger.", 0.4, 0.6, 0.8),
        ("This feels right.", 0.3, 0.5, 0.6),
    ];

    let mut all_audio: Vec<f32> = Vec::new();
    let sr = 44100;

    for (text, arousal, valence, consciousness) in &phrases {
        let engine = if cfg!(feature = "kokoro") {
            symthaea_voice::VoiceEngine::Kokoro
        } else {
            symthaea_voice::VoiceEngine::Formant
        };
        let engine_name = if cfg!(feature = "kokoro") {
            "Kokoro"
        } else {
            "Formant"
        };

        let prosody = symthaea_voice::VoiceProsody {
            arousal: *arousal,
            valence: *valence,
            consciousness: *consciousness,
            serotonin: 0.5,
        };

        println!("  \"{text}\"  [{engine_name}]");
        let audio = symthaea_voice::speak_with_engine(text, &prosody, sr, engine);
        println!(
            "    → {} samples ({:.1}s)",
            audio.len(),
            audio.len() as f32 / sr as f32
        );

        all_audio.extend_from_slice(&audio);
        // 400ms pause between phrases (breathing room)
        all_audio.extend(vec![0.0f32; (sr as f32 * 0.4) as usize]);
    }

    let path = "audio_output/symthaea_speaks.wav";
    std::fs::create_dir_all("audio_output").ok();
    write_wav(path, &all_audio, sr);
    println!("\nOutput: {path}");
    println!("Play: pw-play {path}");
}

fn write_wav(path: &str, audio: &[f32], sr: u32) {
    use std::io::Write;
    let data_len = (audio.len() * 2) as u32;
    let file_len = 36 + data_len;
    let mut f = std::fs::File::create(path).expect("create WAV");
    f.write_all(b"RIFF").ok();
    f.write_all(&file_len.to_le_bytes()).ok();
    f.write_all(b"WAVE").ok();
    f.write_all(b"fmt ").ok();
    f.write_all(&16u32.to_le_bytes()).ok();
    f.write_all(&1u16.to_le_bytes()).ok();
    f.write_all(&1u16.to_le_bytes()).ok();
    f.write_all(&sr.to_le_bytes()).ok();
    f.write_all(&(sr * 2).to_le_bytes()).ok();
    f.write_all(&2u16.to_le_bytes()).ok();
    f.write_all(&16u16.to_le_bytes()).ok();
    f.write_all(b"data").ok();
    f.write_all(&data_len.to_le_bytes()).ok();
    for &s in audio {
        let i = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        f.write_all(&i.to_le_bytes()).ok();
    }
}
