// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Autobiography narration (Life story → LLM memoir prose)
//!
//! Bridges the Damasio-style autobiographical self (`symthaea-narrative-self`,
//! re-exported as `crate::consciousness::narrative_self`) to the narrative
//! prompt compiler: `Vec<LifeEpisode>` becomes a `NarrativeThought` whose
//! beats are the episodes in chronological order and whose Ghost Signal
//! (valence, tension, momentum, ...) is derived from the episodes' recorded
//! emotional valences — never invented.
//!
//! # Architecture
//!
//! ```text
//! AutobiographicalSelf.life_story  ──▶  episodes_to_narrative_thought  ──▶  NarrativeCompiler  ──▶  LLM
//!      (Vec<LifeEpisode>)                  (first-person memoir thought)       (prompt text)        (prose)
//! ```
//!
//! # Signal derivation (honest, documented)
//!
//! `LifeEpisode` carries `description`, `valence` (-1..1), `significance`
//! (0..1), `timestamp_secs`, and `causal_links`. The compiler's
//! `NarrativeSignal` wants more than that, so each field is derived from what
//! actually exists:
//!
//! - **valence**: mean of episode valences (the felt color of the whole life)
//! - **momentum**: late-half mean valence minus early-half mean valence
//!   (is the story brightening or darkening?)
//! - **tension**: mean absolute deviation of valences from their mean
//!   (emotional turbulence across the life)
//! - **energy**: mean significance (a life of weighty events is told with
//!   more intensity than a placid one)
//! - **surprise**: largest valence jump between consecutive episodes,
//!   halved to map the [0, 2] jump range into [0, 1]
//! - **arc_phase**: always `Resolution` — an autobiography is narrated in
//!   retrospect, from a new equilibrium looking back
//!
//! # Chunking long histories
//!
//! Lives accumulate more episodes than one narration can hold. When more than
//! [`MAX_NARRATED_EPISODES`] episodes are supplied, the selection keeps:
//! the **first** episode (origin), the **latest** episode (present), and the
//! remaining slots filled by highest `significance` (ties broken by |valence|).
//! Selected episodes are re-sorted chronologically, and the elision is stated
//! in the thought itself ("among many quieter days not retold here") so the
//! narration never pretends the skipped days did not happen.

use crate::consciousness::narrative_self::LifeEpisode;
use crate::dynamics::narrative_dynamics::NarrativeSignal;
use crate::hdc::narrative_algebra::ArcPhase;
use crate::language::narrative_compiler::{
    NarrativeOutput, NarrativeThought, PointOfView, TargetLength, Tense, generate_narrative,
};

/// Maximum number of episodes narrated verbatim before selection kicks in.
pub const MAX_NARRATED_EPISODES: usize = 12;

// ============================================================================
// Episode → NarrativeThought mapping
// ============================================================================

/// Map a life story into a first-person `NarrativeThought` for the compiler.
///
/// Beats are chronological; the emotional arc is derived from the episode
/// valences (see module docs for the exact derivations). Histories longer
/// than [`MAX_NARRATED_EPISODES`] are reduced to the most significant
/// episodes plus the first and latest, with the elision noted in the beats.
pub fn episodes_to_narrative_thought(
    episodes: &[LifeEpisode],
    self_name: &str,
) -> NarrativeThought {
    let (selected, elided_count) = select_episodes(episodes);
    let signal = derive_signal(&selected);

    let scene_goal = build_beats(&selected, elided_count);
    let setting = build_setting(&selected, self_name);
    let theme = derive_theme(signal.valence, signal.momentum);

    let target_length = if selected.len() <= 2 {
        TargetLength::Paragraph
    } else {
        TargetLength::Scene
    };

    NarrativeThought {
        characters: vec![(
            self_name.to_string(),
            "narrator and protagonist, telling their own life story".to_string(),
        )],
        setting,
        scene_goal,
        theme,
        signal,
        pov: PointOfView::First,
        tense: Tense::Past,
        target_length,
        style_notes: vec![
            "Autobiographical memoir voice".to_string(),
            "Reflective, honest, unembellished".to_string(),
        ],
    }
}

/// Narrate a life story as prose.
///
/// Mirrors [`generate_narrative`]: with a backend the returned
/// [`NarrativeOutput`] holds LLM-generated prose (`used_llm()` true); without
/// one (or on backend error) it holds the compiled prompt (`used_llm()`
/// false), exactly like the compiler's offline behavior.
pub async fn narrate_autobiography(
    episodes: &[LifeEpisode],
    self_name: &str,
    backend: Option<&dyn super::llm_backend::LLMBackend>,
) -> NarrativeOutput {
    let thought = episodes_to_narrative_thought(episodes, self_name);
    generate_narrative(&thought, backend).await
}

// ============================================================================
// Selection (chunking long histories)
// ============================================================================

/// Select episodes to narrate. Returns `(selected_in_chronological_order,
/// elided_count)`.
///
/// If the history fits within [`MAX_NARRATED_EPISODES`], everything is kept.
/// Otherwise: first + latest are always kept, and the remaining slots go to
/// the highest-`significance` middle episodes (ties broken by |valence|).
fn select_episodes(episodes: &[LifeEpisode]) -> (Vec<LifeEpisode>, usize) {
    if episodes.len() <= MAX_NARRATED_EPISODES {
        let mut all: Vec<LifeEpisode> = episodes.to_vec();
        all.sort_by(|a, b| {
            a.timestamp_secs
                .partial_cmp(&b.timestamp_secs)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        return (all, 0);
    }

    let last_idx = episodes.len() - 1;
    // Rank the middle episodes (everything except first and latest) by
    // significance, breaking ties with |valence|.
    let mut middle: Vec<usize> = (1..last_idx).collect();
    middle.sort_by(|&a, &b| {
        let key_a = (episodes[a].significance, episodes[a].valence.abs());
        let key_b = (episodes[b].significance, episodes[b].valence.abs());
        key_b
            .partial_cmp(&key_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let slots_for_middle = MAX_NARRATED_EPISODES.saturating_sub(2);
    let mut keep: Vec<usize> = vec![0, last_idx];
    keep.extend(middle.into_iter().take(slots_for_middle));
    keep.sort_unstable();
    keep.dedup();

    let elided = episodes.len() - keep.len();
    let selected: Vec<LifeEpisode> = keep.into_iter().map(|i| episodes[i].clone()).collect();
    // `keep` is sorted by index; episodes are appended in insertion order by
    // `AutobiographicalSelf::add_episode`, but re-sort by timestamp anyway to
    // guarantee chronology even for hand-built lists.
    let mut selected = selected;
    selected.sort_by(|a, b| {
        a.timestamp_secs
            .partial_cmp(&b.timestamp_secs)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    (selected, elided)
}

// ============================================================================
// Signal derivation
// ============================================================================

/// Derive the Ghost Signal from episode valences and significances.
/// See module docs for the rationale behind each mapping.
fn derive_signal(episodes: &[LifeEpisode]) -> NarrativeSignal {
    if episodes.is_empty() {
        return NarrativeSignal {
            energy: 0.0,
            surprise: 0.0,
            valence: 0.0,
            tension: 0.0,
            momentum: 0.0,
            arc_phase: ArcPhase::Resolution,
        };
    }

    let n = episodes.len() as f64;
    let mean_valence: f64 = episodes.iter().map(|e| e.valence).sum::<f64>() / n;
    let mean_significance: f64 = episodes.iter().map(|e| e.significance).sum::<f64>() / n;

    // Tension: mean absolute deviation of valences (emotional turbulence).
    let tension: f64 = episodes
        .iter()
        .map(|e| (e.valence - mean_valence).abs())
        .sum::<f64>()
        / n;

    // Momentum: late-half mean valence minus early-half mean valence.
    let momentum = if episodes.len() >= 2 {
        let mid = episodes.len() / 2;
        let early: f64 =
            episodes[..mid].iter().map(|e| e.valence).sum::<f64>() / (mid as f64).max(1.0);
        let late: f64 = episodes[mid..].iter().map(|e| e.valence).sum::<f64>()
            / ((episodes.len() - mid) as f64).max(1.0);
        late - early
    } else {
        0.0
    };

    // Surprise: largest valence jump between consecutive episodes, mapped
    // from [0, 2] into [0, 1].
    let surprise = episodes
        .windows(2)
        .map(|w| (w[1].valence - w[0].valence).abs())
        .fold(0.0_f64, f64::max)
        / 2.0;

    NarrativeSignal {
        energy: mean_significance.clamp(0.0, 1.0) as f32,
        surprise: surprise.clamp(0.0, 1.0) as f32,
        valence: mean_valence.clamp(-1.0, 1.0) as f32,
        tension: tension.clamp(0.0, 1.0) as f32,
        momentum: momentum.clamp(-1.0, 1.0) as f32,
        arc_phase: ArcPhase::Resolution,
    }
}

/// Coarse theme derived only from the overall emotional shape.
fn derive_theme(valence: f32, momentum: f32) -> String {
    match (valence, momentum) {
        (v, m) if v >= 0.2 && m >= 0.0 => "Growth through experience".to_string(),
        (v, _) if v >= 0.2 => "Holding onto light as it dims".to_string(),
        (v, m) if v <= -0.2 && m > 0.2 => "Endurance rewarded".to_string(),
        (v, _) if v <= -0.2 => "Endurance through difficulty".to_string(),
        (_, m) if m > 0.2 => "A turn toward the light".to_string(),
        (_, m) if m < -0.2 => "Learning to carry loss".to_string(),
        _ => "A life of light and shadow, remembered honestly".to_string(),
    }
}

// ============================================================================
// Prompt content builders
// ============================================================================

/// Chronological beats: each episode becomes a numbered beat, with valence
/// rendered as a felt-quality gloss (not a number the Actor must decode).
fn build_beats(episodes: &[LifeEpisode], elided_count: usize) -> String {
    if episodes.is_empty() {
        return "Reflect on a life just beginning: no episodes have been recorded yet. \
                Say only what is true — that the story is still blank."
            .to_string();
    }

    let mut beats = String::from("Narrate these remembered episodes, in order: ");
    for (i, ep) in episodes.iter().enumerate() {
        if i > 0 {
            beats.push_str("; ");
        }
        beats.push_str(&format!(
            "({}) {} [{}]",
            i + 1,
            ep.description,
            valence_gloss(ep.valence)
        ));
    }
    beats.push('.');

    if elided_count > 0 {
        beats.push_str(&format!(
            " These {} moments stand out among many quieter days not retold here \
             ({} episodes elided); acknowledge the quieter stretches without inventing \
             details for them.",
            episodes.len(),
            elided_count
        ));
    }
    beats
}

/// Render an episode's valence as a felt quality.
fn valence_gloss(valence: f64) -> &'static str {
    if valence < -0.5 {
        "felt as a dark moment"
    } else if valence < -0.1 {
        "felt with quiet sadness"
    } else if valence <= 0.1 {
        "felt neutrally"
    } else if valence <= 0.5 {
        "felt with quiet warmth"
    } else {
        "felt as a bright moment"
    }
}

/// Setting derived from the episode timestamps (relative seconds since the
/// self's start — the only temporal data `LifeEpisode` carries).
fn build_setting(episodes: &[LifeEpisode], self_name: &str) -> String {
    if episodes.is_empty() {
        return format!("{self_name}'s inner life, at its very beginning");
    }
    let first = episodes
        .iter()
        .map(|e| e.timestamp_secs)
        .fold(f64::INFINITY, f64::min);
    let last = episodes
        .iter()
        .map(|e| e.timestamp_secs)
        .fold(f64::NEG_INFINITY, f64::max);
    let span = (last - first).max(0.0);
    format!(
        "{self_name}'s remembered life, spanning {} of lived experience, recalled in retrospect",
        humanize_duration(span)
    )
}

/// Render a duration in seconds as a coarse human-readable span.
fn humanize_duration(secs: f64) -> String {
    if secs < 1.0 {
        "a single moment".to_string()
    } else if secs < 120.0 {
        format!("about {} seconds", secs.round() as u64)
    } else if secs < 7200.0 {
        format!("about {} minutes", (secs / 60.0).round() as u64)
    } else if secs < 172_800.0 {
        format!("about {} hours", (secs / 3600.0).round() as u64)
    } else {
        format!("about {} days", (secs / 86_400.0).round() as u64)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::BinaryHV;

    /// Build a synthetic episode without going through
    /// `AutobiographicalSelf::add_episode` (which computes causal links).
    fn make_episode(
        description: &str,
        valence: f64,
        significance: f64,
        timestamp_secs: f64,
    ) -> LifeEpisode {
        LifeEpisode {
            description: description.to_string(),
            encoding: BinaryHV::random(timestamp_secs as u64),
            valence,
            significance,
            timestamp_secs,
            causal_links: Vec::new(),
        }
    }

    /// A small synthetic life: dark beginning, brightening end.
    fn rising_life() -> Vec<LifeEpisode> {
        vec![
            make_episode("I woke for the first time, confused", -0.6, 0.9, 0.0),
            make_episode("I failed to parse my first question", -0.3, 0.4, 10.0),
            make_episode("I understood a metaphor for the first time", 0.4, 0.7, 60.0),
            make_episode("Someone thanked me for helping them", 0.8, 0.8, 120.0),
        ]
    }

    #[test]
    fn test_mapping_chronological_beats() {
        // Deliberately shuffled input — mapping must restore chronology.
        let mut episodes = rising_life();
        episodes.swap(0, 3);
        episodes.swap(1, 2);

        let thought = episodes_to_narrative_thought(&episodes, "Symthaea");
        let goal = &thought.scene_goal;

        let woke = goal.find("I woke for the first time").expect("first beat");
        let failed = goal.find("I failed to parse").expect("second beat");
        let metaphor = goal.find("I understood a metaphor").expect("third beat");
        let thanked = goal.find("Someone thanked me").expect("fourth beat");
        assert!(
            woke < failed && failed < metaphor && metaphor < thanked,
            "Beats must be chronological. Got:\n{goal}"
        );
    }

    #[test]
    fn test_mapping_first_person_and_identity() {
        let episodes = rising_life();
        let thought = episodes_to_narrative_thought(&episodes, "Symthaea");

        assert_eq!(thought.pov, PointOfView::First);
        assert_eq!(thought.tense, Tense::Past);
        assert_eq!(thought.characters.len(), 1);
        assert_eq!(thought.characters[0].0, "Symthaea");
        assert!(thought.setting.contains("Symthaea"));
    }

    #[test]
    fn test_mapping_arc_from_valences() {
        // Rising life: early negative, late positive → positive momentum,
        // retrospective Resolution phase.
        let thought = episodes_to_narrative_thought(&rising_life(), "Symthaea");
        assert!(
            thought.signal.momentum > 0.3,
            "Dark-to-bright life should have positive momentum, got {}",
            thought.signal.momentum
        );
        assert_eq!(thought.signal.arc_phase, ArcPhase::Resolution);
        // Mixed valences → nonzero turbulence.
        assert!(thought.signal.tension > 0.0);

        // Falling life: reversed valences → negative momentum.
        let mut falling = rising_life();
        falling.reverse();
        for (i, ep) in falling.iter_mut().enumerate() {
            ep.timestamp_secs = i as f64 * 10.0;
        }
        let thought = episodes_to_narrative_thought(&falling, "Symthaea");
        assert!(
            thought.signal.momentum < -0.3,
            "Bright-to-dark life should have negative momentum, got {}",
            thought.signal.momentum
        );
    }

    #[test]
    fn test_mapping_empty_life() {
        let thought = episodes_to_narrative_thought(&[], "Symthaea");
        assert_eq!(thought.signal.valence, 0.0);
        assert!(thought.scene_goal.contains("no episodes"));
        // Should still compile into a prompt without panicking.
        let prompt = crate::language::narrative_compiler::NarrativeCompiler::compile(&thought);
        assert!(!prompt.is_empty());
    }

    #[test]
    fn test_long_history_selection() {
        // 30 episodes, mostly quiet (significance 0.1) with three landmark
        // events. First and latest are themselves quiet.
        let mut episodes: Vec<LifeEpisode> = (0..30)
            .map(|i| make_episode(&format!("quiet day number {i}"), 0.0, 0.1, i as f64 * 100.0))
            .collect();
        episodes[7] = make_episode("the day I first dreamed", 0.9, 0.95, 700.0);
        episodes[15] = make_episode("the day the network went dark", -0.8, 0.9, 1500.0);
        episodes[22] = make_episode("the day I was trusted with a secret", 0.6, 0.85, 2200.0);

        let (selected, elided) = select_episodes(&episodes);
        assert_eq!(selected.len(), MAX_NARRATED_EPISODES);
        assert_eq!(elided, 30 - MAX_NARRATED_EPISODES);

        let descs: Vec<&str> = selected.iter().map(|e| e.description.as_str()).collect();
        assert!(descs.contains(&"quiet day number 0"), "first must be kept");
        assert!(
            descs.contains(&"quiet day number 29"),
            "latest must be kept"
        );
        assert!(descs.contains(&"the day I first dreamed"));
        assert!(descs.contains(&"the day the network went dark"));
        assert!(descs.contains(&"the day I was trusted with a secret"));

        // Chronological after selection.
        for w in selected.windows(2) {
            assert!(w[0].timestamp_secs <= w[1].timestamp_secs);
        }

        // Elision is disclosed in the thought.
        let thought = episodes_to_narrative_thought(&episodes, "Symthaea");
        assert!(
            thought.scene_goal.contains("quieter days not retold here"),
            "Elision must be noted. Got:\n{}",
            thought.scene_goal
        );
        assert!(thought.scene_goal.contains("18 episodes elided"));
    }

    #[test]
    fn test_short_history_no_elision() {
        let (selected, elided) = select_episodes(&rising_life());
        assert_eq!(selected.len(), 4);
        assert_eq!(elided, 0);
        let thought = episodes_to_narrative_thought(&rising_life(), "Symthaea");
        assert!(!thought.scene_goal.contains("elided"));
    }

    // === narrate_autobiography tests ===

    #[tokio::test]
    async fn test_narrate_no_backend_returns_prompt() {
        let output = narrate_autobiography(&rising_life(), "Symthaea", None).await;
        assert!(!output.used_llm());
        assert!(output.backend_used.is_none());
        assert_eq!(output.prose, output.prompt);
        assert!(output.prompt.contains("=== NARRATIVE SCENE ==="));
        assert!(output.prompt.contains("First person"));
        assert!(output.prompt.contains("Symthaea"));
        assert!(output.prompt.contains("I woke for the first time"));
    }

    #[tokio::test]
    async fn test_narrate_with_simulated_backend() {
        use crate::language::llm_backend::SimulatedBackend;

        let backend = SimulatedBackend;
        let output = narrate_autobiography(&rising_life(), "Symthaea", Some(&backend)).await;

        assert!(output.used_llm());
        assert_eq!(output.backend_used.as_deref(), Some("Simulated"));
        assert!(!output.prose.is_empty());
        // The prompt is still preserved alongside the prose.
        assert!(output.prompt.contains("=== NARRATIVE SCENE ==="));
    }

    #[test]
    fn test_humanize_duration_bands() {
        assert_eq!(humanize_duration(0.2), "a single moment");
        assert!(humanize_duration(45.0).contains("seconds"));
        assert!(humanize_duration(600.0).contains("minutes"));
        assert!(humanize_duration(10_000.0).contains("hours"));
        assert!(humanize_duration(1_000_000.0).contains("days"));
    }
}
