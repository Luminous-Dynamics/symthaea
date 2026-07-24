// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The formal listening test — the reviewer's protocol, runnable:
//!
//! > "Take four tango hooks, four nocturne hooks, four march hooks.
//! > Strip away everything except the lead melody. Play them to
//! > listeners in random order. Ask only: which style is this? If people
//! > consistently identify them above chance, you've built something
//! > real. If not, the decomposition tells you exactly which layer still
//! > needs work."
//!
//! This is the instrument that verifies the engine's measured diversity
//! is PERCEIVED. Two honesty rules beyond the obvious:
//! - **drums are stripped** (a march's kit would answer for you), and
//! - **every melody renders on the same instrument** (piano) — timbre
//!   must not leak the answer; only the melodic thought may.
//!
//! Usage:
//! ```bash
//! cargo run --release -p symthaea-muse --features studio --bin listening_test -- \
//!     generate audio_output/listening_test
//! # listen to the clips in any order; then:
//! cargo run ... --bin listening_test -- \
//!     score audio_output/listening_test "1=Tango,2=Nocturne,3=March,..."
//! ```
//!
//! `generate` draws a fresh, time-derived seed by default -- every
//! invocation produces a genuinely different 32-clip set (different
//! per-style seeds AND a different shuffle order), not the same fixed
//! clips every run. Pass an explicit seed as a third argument
//! (`generate <out_dir> <seed>`) to get a REPRODUCIBLE set instead --
//! e.g. to re-share a specific set that surfaced a bug.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::Path;

use symthaea_music_theory::{MusicalIntent, Style, compose_with_spec};

/// Eight maximally-contrasting styles (diversity plan Phase 1): the original
/// three plus five more chosen to span dance/nocturne/march/blues/process-
/// music/flamenco/choral/ambient territory. All eight verified present in
/// `symthaea_music_theory::Style` and composing without panics.
const STYLES: [Style; 8] = [
    Style::Tango,
    Style::Nocturne,
    Style::March,
    Style::Blues,
    Style::Minimalism,
    Style::Flamenco,
    Style::SacredChoral,
    Style::Ambient,
];
const SEEDS_PER_STYLE: usize = 4;
const SAMPLE_RATE: u32 = 48_000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("generate") => {
            let out = args
                .get(2)
                .cloned()
                .unwrap_or_else(|| "audio_output/listening_test".into());
            // A time-derived default means every unparameterized invocation
            // produces a genuinely fresh set -- the original hardcoded
            // seeds (`11 + k*17`) and shuffle constant made EVERY run,
            // forever, generate the exact same 32 clips in the exact same
            // order. Pass an explicit seed to get a reproducible set back.
            let base_seed = match args.get(3) {
                Some(s) => s.parse::<u64>().expect("seed must be a u64"),
                None => std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system clock before UNIX epoch")
                    .as_nanos() as u64,
            };
            generate(Path::new(&out), base_seed);
        }
        Some("score") => {
            let dir = args.get(2).expect("score <dir> <guesses>");
            let guesses = args.get(3).expect("score <dir> <guesses>");
            score(Path::new(dir), guesses);
        }
        _ => {
            eprintln!("usage: listening_test generate [out_dir] [seed] | score <dir> <guesses>");
            std::process::exit(2);
        }
    }
}

/// Deterministic shuffle (LCG Fisher-Yates) so a test SET is reproducible
/// FOR A GIVEN SEED -- `generate`'s default seed still varies per run (see
/// `main`), this only guarantees re-running with the SAME seed reproduces
/// the SAME set.
fn shuffle<T>(items: &mut [T], mut state: u64) {
    for i in (1..items.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = ((state >> 33) as usize) % (i + 1);
        items.swap(i, j);
    }
}

fn generate(out: &Path, base_seed: u64) {
    std::fs::create_dir_all(out).expect("create out dir");
    let mut clips: Vec<(Style, usize, u64)> = Vec::new();
    for style in STYLES {
        for k in 0..SEEDS_PER_STYLE {
            clips.push((
                style,
                k,
                base_seed
                    .wrapping_add(11)
                    .wrapping_add((k as u64).wrapping_mul(17)),
            ));
        }
    }
    shuffle(&mut clips, base_seed ^ 0xC1A5_51F1);

    let mut key: Vec<(String, String)> = Vec::new();
    for (i, (style, k, seed)) in clips.iter().enumerate() {
        let mut spec = style.spec();
        // The honesty rules: no kit, one shared timbre for every clip.
        spec.texture.drums = symthaea_music_theory::DrumPolicy::None;
        spec.ensemble_pool = vec![["piano".into(), "piano".into(), "piano".into()]];
        // Each style's 4 clips deliberately span all three arousal tiers
        // (calm/medium/busy) plus a valence swing, instead of every clip
        // using the same fixed arousal=0.5/energy=0.5/valence=0.0.
        // `CompositionSpec::motif` picks its template bank purely from
        // `intent.arousal` (see spec.rs) -- with a single fixed arousal,
        // `generate` only ever touched each style's `motifs_medium` bank
        // (typically 2-3 hand-authored templates), silently narrowing the
        // test down to a fraction of what each style can actually
        // produce, on top of the seed-reuse bug fixed separately.
        let (arousal, energy, valence) = match k {
            0 => (0.15, 0.25, 0.5),  // calm, bright
            1 => (0.5, 0.5, 0.0),    // medium, neutral
            2 => (0.85, 0.85, -0.5), // busy, dark
            _ => (0.5, 0.6, 0.5),    // medium again, opposite valence
        };
        let intent = MusicalIntent {
            seed: *seed,
            valence,
            arousal,
            energy,
            ..Default::default()
        };
        let mut score = compose_with_spec(&intent, &spec);
        // Strip to the lead melody: the question is whether the melodic
        // THOUGHT carries the style.
        score
            .notes
            .retain(|n| n.role == symthaea_music_theory::VoiceRole::Melody);
        // Trim the leading silence this stripping introduces: every piece
        // opens with a real, 2-bar instrumental intro (`prepend_intro` in
        // composer.rs), but that intro is composed ENTIRELY of Harmony/
        // Bass/accompaniment content by design -- it carries zero melody
        // notes. In a full rendering that's a real, audible intro; here,
        // after stripping to melody-only, it becomes 4-14+ seconds of true
        // silence before the first note a listener can judge (found via a
        // real signal-analysis pass on a rendered set: every clip had
        // lead-in dead air matching this exact mechanism, one clip
        // extreme enough to look like a render failure). Shift every
        // remaining note left so the first melody note starts at beat 0.
        if let Some(first_onset) = score
            .notes
            .iter()
            .map(|n| n.onset)
            .min_by(|a, b| a.beats().total_cmp(&b.beats()))
            && first_onset.beats() > 1e-9
        {
            for n in &mut score.notes {
                n.onset = n.onset.saturating_sub(first_onset);
            }
            score.total_beats = score.total_beats.saturating_sub(first_onset);
        }

        let name = format!("clip_{:02}.wav", i + 1);
        let state = symthaea_muse::MusicalState::default();
        let wav: Vec<u8> = render(&score, &spec, *seed, &state)
            .unwrap_or_else(|| panic!("render failed for clip {}", i + 1));
        std::fs::write(out.join(&name), &wav).expect("write clip");
        println!("{name}  ({} KB)", wav.len() / 1024);
        key.push((name, format!("{style:?}")));
    }

    let key_json =
        serde_json::to_string_pretty(&key.iter().cloned().collect::<HashMap<String, String>>())
            .unwrap();
    std::fs::write(out.join("answer_key.json"), key_json).expect("write key");

    let style_names: Vec<String> = STYLES.iter().map(|s| format!("{s:?}")).collect();
    let mut sheet = std::fs::File::create(out.join("README.md")).expect("sheet");
    writeln!(
        sheet,
        "# Muse listening test — which style is this?\n\n\
         {} clips, lead melody only, all rendered on the same piano —\n\
         no accompaniment, no drums, no timbre hints. The styles present:\n\
         **{}** ({} clips each, shuffled).\n\n\
         Listen in any order (the opening 20-30 seconds is enough), write\n\
         down one style per clip, and DO NOT open `answer_key.json` until\n\
         you're done. Then score yourself:\n\n\
         ```bash\n\
         cargo run --release -p symthaea-muse --features studio \\\n\
             --bin listening_test -- score <this dir> \\\n\
             \"1={},2={},3={},...\"\n\
         ```\n\n\
         Optionally attach your confidence per guess (0-100) with `@N` —\n\
         e.g. `\"1={}@95,2={}@60,...\"`. A bare guess\n\
         defaults to 100%. Confidence isn't scored for correctness, but a\n\
         well-calibrated listener should show higher confidence when right\n\
         than when wrong — that gap tells us something a bare accuracy\n\
         number can't.\n\n\
         Chance is 1 in {}. If identification is consistently above chance,\n\
         the styles' melodic thought is PERCEIVED, not just measured.\n\n\
         This set's seed was `{base_seed}` — pass it as a third argument\n\
         (`generate <out_dir> {base_seed}`) to reproduce this exact set;\n\
         omit it to get a fresh one every time.",
        clips.len(),
        style_names.join(", "),
        SEEDS_PER_STYLE,
        style_names[0],
        style_names[1],
        style_names[2],
        style_names[0],
        style_names[1],
        STYLES.len(),
    )
    .unwrap();
    println!(
        "\nWrote {} clips + README.md + answer_key.json to {} (seed {base_seed})",
        clips.len(),
        out.display()
    );
}

fn render(
    score: &symthaea_music_theory::Score,
    spec: &symthaea_music_theory::CompositionSpec,
    seed: u64,
    state: &symthaea_muse::MusicalState,
) -> Option<Vec<u8>> {
    // Prefer the soundfont path (same as the Studio); it requires the
    // environment the launcher provides.
    if symthaea_muse::fluid_render::available().is_some() {
        let path =
            std::env::temp_dir().join(format!("listening_test_{}_{seed}.mid", std::process::id()));
        symthaea_muse::midi_export::export_performance_midi(score, spec, seed, state, &path)
            .ok()?;
        let wav = symthaea_muse::fluid_render::render_midi_to_wav(&path, SAMPLE_RATE, None);
        let _ = std::fs::remove_file(&path);
        return wav;
    }
    eprintln!("(fluidsynth unavailable — run via muse-studio-launch's env for the soundfont path)");
    None
}

/// Guesses may optionally carry a confidence: `"1=Tango@95,2=Nocturne,..."`.
/// A bare guess (no `@N`) is treated as 100% confident — the format stays
/// backward compatible with every test set already scored this way.
fn parse_guess(part: &str) -> Option<(usize, String, u32)> {
    let mut kv = part.trim().splitn(2, '=');
    let (idx, rest) = (kv.next()?, kv.next()?);
    let idx = idx.trim().parse::<usize>().ok()?;
    let (guess, conf) = match rest.rsplit_once('@') {
        Some((g, c)) => (
            g.trim(),
            c.trim().parse::<u32>().unwrap_or(100).clamp(0, 100),
        ),
        None => (rest.trim(), 100),
    };
    Some((idx, guess.to_string(), conf))
}

fn score(dir: &Path, guesses: &str) {
    let key: HashMap<String, String> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("answer_key.json")).expect("answer key"),
    )
    .expect("key json");
    let mut correct = 0usize;
    let mut total = 0usize;
    // Style -> (hits, attempts), for the per-style breakdown.
    let mut per_style: HashMap<String, (usize, usize)> = HashMap::new();
    // Confusion pairs: (truth, guess) -> count, truth != guess only.
    let mut confusions: HashMap<(String, String), usize> = HashMap::new();
    let mut confidence_sum = 0u64;
    let mut confidence_when_right = 0u64;
    let mut confidence_when_wrong = 0u64;
    let mut wrong = 0usize;
    for part in guesses.split(',') {
        let Some((idx, guess, conf)) = parse_guess(part) else {
            continue;
        };
        let name = format!("clip_{idx:02}.wav");
        let Some(truth) = key.get(&name) else {
            continue;
        };
        total += 1;
        let hit = truth.eq_ignore_ascii_case(&guess);
        println!(
            "{name}: guessed {guess:10} @{conf:3}% truth {truth:10} {}",
            if hit { "✓" } else { "✗" }
        );
        confidence_sum += conf as u64;
        let entry = per_style.entry(truth.clone()).or_insert((0, 0));
        entry.1 += 1;
        if hit {
            correct += 1;
            entry.0 += 1;
            confidence_when_right += conf as u64;
        } else {
            wrong += 1;
            confidence_when_wrong += conf as u64;
            *confusions
                .entry((truth.clone(), guess.to_string()))
                .or_insert(0) += 1;
        }
    }
    let p_chance = 1.0 / STYLES.len() as f64;
    // One-sided binomial tail: P(X >= correct | n=total, p=chance).
    let p_value: f64 = (correct..=total)
        .map(|k| {
            let mut c = 1.0f64;
            for i in 0..k {
                c = c * (total - i) as f64 / (i + 1) as f64;
            }
            c * p_chance.powi(k as i32) * (1.0 - p_chance).powi((total - k) as i32)
        })
        .sum();
    println!(
        "\n{}/{} correct ({:.0}%) — chance would be {:.0}%.",
        correct,
        total,
        100.0 * correct as f64 / total.max(1) as f64,
        100.0 * p_chance
    );
    println!(
        "one-sided binomial p = {p_value:.4} {}",
        if p_value < 0.05 {
            "— above chance: the melodic thought is PERCEIVED."
        } else {
            "— not above chance: the decomposition says which layer to work on next."
        }
    );
    if total > 0 {
        println!(
            "\nAverage confidence: {:.0}% overall; {:.0}% when right, {:.0}% when wrong{}.",
            confidence_sum as f64 / total as f64,
            if correct > 0 {
                confidence_when_right as f64 / correct as f64
            } else {
                0.0
            },
            if wrong > 0 {
                confidence_when_wrong as f64 / wrong as f64
            } else {
                0.0
            },
            if correct > 0
                && wrong > 0
                && confidence_when_right as f64 / correct as f64
                    > confidence_when_wrong as f64 / wrong as f64
            {
                " — well-calibrated: you knew what you didn't know"
            } else {
                ""
            }
        );
    }
    let mut styles: Vec<&String> = per_style.keys().collect();
    styles.sort();
    println!("\nPer-style breakdown:");
    for style in styles {
        let (hits, attempts) = per_style[style];
        println!(
            "  {style:10} {hits}/{attempts} ({:.0}%)",
            100.0 * hits as f64 / attempts.max(1) as f64
        );
    }
    if !confusions.is_empty() {
        println!("\nConfusions (truth heard as guess):");
        let mut pairs: Vec<_> = confusions.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        for ((truth, guess), n) in pairs {
            println!("  {truth} heard as {guess}: {n}");
        }
    }
}
