//! Diagnostic probe for the wistful-shuffle harshness investigation.
//!
//! Part 1: which notes fall through the strict sample-pick window?
//! Part 2: which notes SOUND during each measured harsh window (replicates
//! muse's Timeline math — swing then rubato — from public score data)?
fn main() {
    use symthaea_muse::instruments::Instrument;
    use symthaea_muse::vcsl::VcslLibrary;
    use symthaea_music_theory::*;

    // Both inputs are local-only (data/ is gitignored): the CC0 sample
    // libraries and a spec saved from Muse Studio. Skip cleanly when absent.
    let Some(lib) = VcslLibrary::load(std::path::Path::new("data/samples/vcsl")) else {
        eprintln!("sample library data/samples/vcsl not found — skipping");
        return;
    };
    let Ok(spec_json) = std::fs::read_to_string("data/specs/wistful-shuffle.json") else {
        eprintln!(
            "saved spec data/specs/wistful-shuffle.json not found (save one in Muse Studio) — skipping"
        );
        return;
    };
    let spec: CompositionSpec = serde_json::from_str(&spec_json).unwrap();
    let intent = MusicalIntent {
        valence: -0.4,
        arousal: 0.35,
        energy: 0.45,
        tonic: pitch::PitchClass::new(9),
        bars: 4,
        seed: 3,
    };
    let score = compose_with_spec(&intent, &spec);

    // ---- Part 1: strict-window fallthrough per role ----
    for (role, instr) in [
        (VoiceRole::Melody, Instrument::Violin),
        (VoiceRole::Harmony, Instrument::Harp),
        (VoiceRole::Bass, Instrument::Cello),
        (VoiceRole::CounterMelody, Instrument::Cello),
    ] {
        let mut fails = std::collections::BTreeMap::new();
        let notes = score.voice(role);
        for n in &notes {
            if !lib.can_play(instr, n.pitch.midi() as f32, n.velocity) {
                *fails.entry(n.pitch.midi()).or_insert(0) += 1;
            }
        }
        println!(
            "{role:?} ({instr:?}): {} notes, fallthrough: {fails:?}",
            notes.len()
        );
    }

    // ---- Part 2: what sounds in each harsh window ----
    let spb = 60.0 / score.tempo_bpm as f64;
    let melody = score.voice(VoiceRole::Melody);
    let mut events: Vec<(f64, f64)> = Vec::new();
    let last_idx = melody.len().saturating_sub(1);
    for (i, n) in melody.iter().enumerate() {
        let onset = n.onset.beats();
        let end = (n.onset + n.duration).beats();
        match n.emphasis {
            score::Emphasis::PhraseStart => events.push((onset, 0.10 * spb)),
            score::Emphasis::Climax => events.push((end, 0.18 * spb)),
            score::Emphasis::Cadential => {
                let amount = if i == last_idx { 0.9 } else { 0.35 };
                events.push((end, amount * spb));
            }
            score::Emphasis::Normal => {}
        }
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let swing = spec.texture.swing as f64;
    let swing_beat = |b: f64| {
        let whole = b.floor();
        let frac = b - whole;
        let warped = if frac <= 0.5 {
            frac * (swing / 0.5)
        } else {
            swing + (frac - 0.5) * ((1.0 - swing) / 0.5)
        };
        whole + warped
    };
    let secs = |beat: f64| -> f64 {
        let b = swing_beat(beat);
        b * spb
            + events
                .iter()
                .take_while(|(e, _)| *e <= b + 1e-9)
                .map(|(_, d)| d)
                .sum::<f64>()
    };

    let max_intensity = melody
        .iter()
        .map(|n| n.section_intensity)
        .fold(0.0f32, f32::max);

    let windows = [
        (20.0, 22.0),
        (36.0, 40.0),
        (48.0, 56.0),
        (72.0, 84.0),
        (96.0, 98.0),
    ];
    for (w0, w1) in windows {
        println!("\n=== window {w0}-{w1}s ===");
        for role in [
            VoiceRole::Melody,
            VoiceRole::Harmony,
            VoiceRole::Bass,
            VoiceRole::CounterMelody,
        ] {
            let mut active: Vec<String> = Vec::new();
            for n in &score.voice(role) {
                let s = secs(n.onset.beats());
                let e = secs((n.onset + n.duration).beats());
                if s < w1 && e > w0 {
                    let doubling = role == VoiceRole::Melody
                        && (n.section_intensity - max_intensity).abs() < 1e-6;
                    active.push(format!(
                        "m{} v{:.2}{}",
                        n.pitch.midi(),
                        n.velocity,
                        if doubling { "+dbl" } else { "" }
                    ));
                }
            }
            if !active.is_empty() {
                println!("  {role:?}: {}", active.join(", "));
            }
        }
    }
}
