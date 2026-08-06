// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Measure what two styles have IDENTICAL, to locate perceptual sameness.
//!
//! Structural diversity and perceptual diversity are different things. Two
//! pieces can differ in form, progression, voice count and pitch content and
//! still land as "the same house texture" because the channels a listener
//! actually keys on — tempo, meter, key, register, density, articulation,
//! dynamics — are pinned across styles.
//!
//! This probe does not judge music. It reports, per style, the objective
//! parameters that determine those channels, so a shared value can be traced
//! to the code that pins it. Ears remain the arbiter of whether changing one
//! made an audible difference; this only says where to look.
//!
//! Deliberately reports the FIRST 20 SECONDS separately from the whole piece.
//! That is the window in which a listener decides what they are hearing, and
//! it is where a distinct premise has to declare itself.
//!
//! Usage:
//!   cargo run --release -p symthaea-muse --features theory \
//!       --example style_sameness_probe -- [seed] [bars]

use std::collections::BTreeMap;

use symthaea_music_theory::composer::compose_styled;
use symthaea_music_theory::score::{Score, VoiceRole};
use symthaea_music_theory::{MusicalIntent, PitchClass, Style};

/// Styles to compare. Fugue vs Passacaglia is the assigned contrast; the rest
/// are Baroque-family reference points so a shared value can be told apart
/// from a two-style coincidence.
const STYLES: &[Style] = &[
    Style::Fugue,
    Style::Passacaglia,
    Style::BaroqueSuite,
    Style::RenaissancePolyphony,
    Style::Nocturne,
];

struct Profile {
    style: &'static str,
    tempo_bpm: f32,
    meter: u8,
    key: String,
    /// Distinct role classes across the whole piece — again roles, not parts.
    voices: usize,
    notes: usize,
    /// Notes per beat over the whole piece.
    density: f64,
    /// Notes per beat in the opening window.
    density_open: f64,
    /// Semitone span of all pitches, and of the opening window.
    range_semitones: i32,
    range_open: i32,
    /// Mean/spread of velocity — the dynamic channel.
    velocity_mean: f32,
    velocity_sd: f32,
    /// Mean note length in beats — the articulation channel (short = detached
    /// vocabulary, long = sustained).
    duration_mean: f64,
    /// Distinct inter-onset intervals, quantised — the rhythmic vocabulary.
    /// A style with 2 distinct IOIs speaks in a much narrower rhythm than one
    /// with 8, regardless of what its form is called.
    distinct_iois: usize,
    /// How many ROLE CLASSES sound in the opening window. Not voices: a
    /// `ScoreNote` carries `VoiceRole` (Melody/Harmony/Bass/CounterMelody) and
    /// no part identity, so two contrapuntal lines sharing a role are
    /// indistinguishable here, as is one line whose role changes.
    roles_open: usize,
}

fn beats_per_second(score: &Score) -> f64 {
    score.tempo_bpm as f64 / 60.0
}

fn profile(style: Style, intent: &MusicalIntent, open_secs: f64) -> Profile {
    let score = compose_styled(intent, style);
    let bps = beats_per_second(&score);
    let open_beats = open_secs * bps;

    let notes = &score.notes;
    let onset_beats = |n: &symthaea_music_theory::score::ScoreNote| n.onset.beats();
    let open: Vec<_> = notes
        .iter()
        .filter(|n| onset_beats(n) < open_beats)
        .collect();

    let total_beats = notes
        .iter()
        .map(|n| onset_beats(n) + n.duration.beats())
        .fold(0.0f64, f64::max)
        .max(1.0);

    let midi = |n: &symthaea_music_theory::score::ScoreNote| n.pitch.midi() as i32;
    let span = |v: &[&symthaea_music_theory::score::ScoreNote]| -> i32 {
        match (
            v.iter().map(|n| midi(n)).min(),
            v.iter().map(|n| midi(n)).max(),
        ) {
            (Some(lo), Some(hi)) => hi - lo,
            _ => 0,
        }
    };
    let all: Vec<_> = notes.iter().collect();

    let vmean = if notes.is_empty() {
        0.0
    } else {
        notes.iter().map(|n| n.velocity).sum::<f32>() / notes.len() as f32
    };
    let vsd = if notes.len() < 2 {
        0.0
    } else {
        (notes
            .iter()
            .map(|n| (n.velocity - vmean).powi(2))
            .sum::<f32>()
            / notes.len() as f32)
            .sqrt()
    };

    // Inter-onset intervals, quantised to 1/16 beat, over the melody voice —
    // the line a listener follows.
    let mut lead: Vec<f64> = notes
        .iter()
        .filter(|n| n.role == VoiceRole::Melody)
        .map(onset_beats)
        .collect();
    lead.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut iois: BTreeMap<i64, usize> = BTreeMap::new();
    for w in lead.windows(2) {
        let d = ((w[1] - w[0]) * 16.0).round() as i64;
        if d > 0 {
            *iois.entry(d).or_default() += 1;
        }
    }

    let mut roles_open: Vec<VoiceRole> = open.iter().map(|n| n.role).collect();
    roles_open.sort_by_key(|r| format!("{r:?}"));
    roles_open.dedup();
    let mut roles_all: Vec<VoiceRole> = notes.iter().map(|n| n.role).collect();
    roles_all.sort_by_key(|r| format!("{r:?}"));
    roles_all.dedup();

    Profile {
        style: Box::leak(format!("{style:?}").into_boxed_str()),
        tempo_bpm: score.tempo_bpm,
        meter: score.meter,
        key: format!("{:?} {:?}", score.key.tonic, score.key.tonality),
        voices: roles_all.len(),
        notes: notes.len(),
        density: notes.len() as f64 / total_beats,
        density_open: open.len() as f64 / open_beats.max(1.0),
        range_semitones: span(&all),
        range_open: span(&open),
        velocity_mean: vmean,
        velocity_sd: vsd,
        duration_mean: if notes.is_empty() {
            0.0
        } else {
            notes.iter().map(|n| n.duration.beats()).sum::<f64>() / notes.len() as f64
        },
        distinct_iois: iois.len(),
        roles_open: roles_open.len(),
    }
}

/// Report any parameter that is identical across ALL probed styles. Those are
/// the house template: no matter which premise the caller asks for, the
/// listener gets the same value.
fn report_pinned(ps: &[Profile]) {
    let same = |f: &dyn Fn(&Profile) -> String| -> Option<String> {
        let first = f(&ps[0]);
        ps.iter().all(|p| f(p) == first).then_some(first)
    };
    println!(
        "\nPINNED ACROSS ALL {} STYLES (the house template):",
        ps.len()
    );
    let checks: [(&str, &dyn Fn(&Profile) -> String); 6] = [
        ("tempo_bpm", &|p: &Profile| format!("{:.1}", p.tempo_bpm)),
        ("meter", &|p: &Profile| p.meter.to_string()),
        ("key", &|p: &Profile| p.key.clone()),
        ("role classes (all)", &|p: &Profile| p.voices.to_string()),
        ("role classes in opening", &|p: &Profile| {
            p.roles_open.to_string()
        }),
        ("distinct melody IOIs", &|p: &Profile| {
            p.distinct_iois.to_string()
        }),
    ];
    let mut any = false;
    for (name, f) in checks {
        if let Some(v) = same(f) {
            println!("  {name:22} = {v}   <-- identical everywhere");
            any = true;
        }
    }
    if !any {
        println!("  (none — every probed channel differs somewhere)");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let bars: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let open_secs = 20.0;

    let intent = MusicalIntent {
        valence: 0.0,
        arousal: 0.5,
        energy: 0.5,
        bars,
        seed,
        tonic: PitchClass::C,
    };

    println!(
        "seed={seed} bars={bars} opening-window={open_secs:.0}s  \
         (identical intent for every style — differences below are the STYLE's doing)"
    );

    let ps: Vec<Profile> = STYLES
        .iter()
        .map(|&s| profile(s, &intent, open_secs))
        .collect();

    println!(
        "\n{:<22} {:>6} {:>6} {:>10} {:>4} {:>7} {:>8} {:>6} {:>6} {:>7} {:>6} {:>5} {:>6}",
        "style",
        "tempo",
        "meter",
        "key",
        "vox",
        "notes",
        "dens",
        "d/20s",
        "range",
        "r/20s",
        "vel",
        "IOIs",
        "vox20"
    );
    for p in &ps {
        println!(
            "{:<22} {:>6.1} {:>6} {:>10} {:>4} {:>7} {:>8.2} {:>6.2} {:>6} {:>7} {:>6.2} {:>5} {:>6}",
            p.style,
            p.tempo_bpm,
            p.meter,
            p.key,
            p.voices,
            p.notes,
            p.density,
            p.density_open,
            p.range_semitones,
            p.range_open,
            p.velocity_mean,
            p.distinct_iois,
            p.roles_open
        );
    }

    println!("\nvelocity spread (dynamics) and mean note length (articulation):");
    for p in &ps {
        println!(
            "  {:<22} vel {:.3} +/- {:.3}   mean note {:.2} beats",
            p.style, p.velocity_mean, p.velocity_sd, p.duration_mean
        );
    }

    report_pinned(&ps);

    println!(
        "\nFUGUE vs PASSACAGLIA — the assigned contrast, opening 20s:\n\
         \x20 A fugue should open with ONE voice stating a subject; a passacaglia\n\
         \x20 should open with an unmistakable repeating bass. If both report the\n\
         \x20 same voices-in-opening, neither premise is declaring itself."
    );
    for p in ps
        .iter()
        .filter(|p| p.style == "Fugue" || p.style == "Passacaglia")
    {
        println!(
            "  {:<14} role classes in first 20s: {}   density: {:.2} notes/beat   range: {} semitones",
            p.style, p.roles_open, p.density_open, p.range_open
        );
    }

    // "Roles within 20s" is too coarse to judge an exposition: at 93 BPM a
    // legitimate fughetta would have all three entries inside that window. The
    // sharper question is WHEN THE SECOND ROLE ENTERS — a staggered opening
    // leaves the first line alone for a bar or more; a homophonic one starts
    // everything at once.
    //
    // MEASURED vs INFERRED, stated deliberately:
    //   Measured:   three ROLE CLASSES enter at 0.00, 2.58 and 5.16 seconds.
    //   Supported:  the arrangement has stable staggered role entry.
    //   NOT shown:  independent voices restating a fugue subject. That needs a
    //               real part identity on ScoreNote (a `PartId` distinct from
    //               `VoiceRole`) plus a subject-recurrence check. Until those
    //               exist this is `role_entry_stagger`, and calling it a
    //               "textbook staggered exposition" — as an earlier report of
    //               this probe did — promotes a measurement into a claim it
    //               does not establish.
    println!("\nROLE-ENTRY STAGGER — first onset of each ROLE CLASS, in seconds:");
    for &style in STYLES {
        let score = compose_styled(&intent, style);
        let spb = 60.0 / score.tempo_bpm as f64;
        let mut first: BTreeMap<String, f64> = BTreeMap::new();
        for n in &score.notes {
            let t = n.onset.beats() * spb;
            first
                .entry(format!("{:?}", n.role))
                .and_modify(|e| *e = e.min(t))
                .or_insert(t);
        }
        let mut times: Vec<(String, f64)> = first.into_iter().collect();
        times.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let stagger = times
            .get(1)
            .map(|(_, t)| *t - times[0].1)
            .unwrap_or(f64::NAN);
        let entries: Vec<String> = times.iter().map(|(r, t)| format!("{r} @{t:.2}s")).collect();
        println!(
            "  {:<22} gap to 2nd role:  {:>5.2}s   [{}]",
            format!("{style:?}"),
            stagger,
            entries.join(", ")
        );
    }
    println!(
        "\n  A gap of 0.00s means every ROLE starts together. That rules out a\n\
         \x20 staggered opening; it does not by itself confirm one, because two\n\
         \x20 lines can share a role."
    );
}
