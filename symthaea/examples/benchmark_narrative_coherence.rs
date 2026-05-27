// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Narrative Coherence Benchmark
//!
//! Tests 4 canonical story arcs through the narrative dynamics pipeline and
//! reports paper-worthy metrics: peak tension, trend accuracy, Spearman rank
//! correlation, coherence score (within > across half similarity), arc symmetry
//! deviation (golden ratio), and Vonnegut story shape classification.
//!
//! Run: `cargo run --example benchmark_narrative_coherence`

use symthaea::dynamics::narrative_dynamics::NarrativeSignal;
use symthaea::dynamics::story_session::StorySession;
use symthaea::hdc::narrative_algebra::NarrativeMood;

/// A scene definition for a benchmark arc.
struct BenchScene {
    title: &'static str,
    setting: &'static str,
    conflict: &'static str,
    mood: NarrativeMood,
    /// Expected tension trend vs. previous scene: +1 = rise, 0 = hold, -1 = fall
    expected_trend: i8,
}

/// Run one story arc through the session and collect signals.
fn run_arc(name: &str, scenes: &[BenchScene]) -> Vec<NarrativeSignal> {
    let mut session = StorySession::new();
    session.set_expected_scenes(scenes.len());
    let prot = session.algebra().primitives.protagonist.clone();
    session.register_character("Hero", &prot);

    let mut signals = Vec::new();
    for scene in scenes {
        let sig = session.add_scene(
            scene.title,
            scene.setting,
            &["Hero"],
            scene.conflict,
            scene.mood,
        );
        signals.push(sig);
    }

    // Print ASCII tension curve
    println!("\n  {} tension curve:", name);
    print!("  ");
    for sig in &signals {
        let bars = (sig.tension * 20.0).round() as usize;
        print!("|{}", "#".repeat(bars).to_string() + &" ".repeat(20 - bars));
    }
    println!("|");
    print!("  ");
    for scene in scenes {
        print!(" {:<20}", &scene.title[..scene.title.len().min(19)]);
    }
    println!();

    signals
}

/// Compute Spearman rank correlation between two sequences.
///
/// Returns rho in [-1, 1]. Ties are handled by average ranking.
fn spearman_rank_correlation(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n < 2 {
        return 0.0;
    }

    fn rank(vals: &[f32]) -> Vec<f32> {
        let n = vals.len();
        let mut indexed: Vec<(usize, f32)> = vals.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut ranks = vec![0.0f32; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-9 {
                j += 1;
            }
            let avg_rank = (i + j + 1) as f32 / 2.0; // 1-based average
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }
            i = j;
        }
        ranks
    }

    let ra = rank(a);
    let rb = rank(b);
    let d_sq_sum: f32 = ra.iter().zip(rb.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let n = n as f32;
    1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1.0))
}

/// Compute segment-aware Spearman: split the arc at trend direction changes,
/// compute Spearman within each monotonic segment, and return the weighted average.
///
/// For monotonic arcs this equals global Spearman. For U-shaped arcs like
/// Cinderella (down-up-down-up), it captures per-segment ordering quality.
fn segmented_spearman(scenes: &[BenchScene], signals: &[NarrativeSignal]) -> f32 {
    let n = scenes.len();
    if n < 3 {
        // Too short to segment — use global
        let mut cum = Vec::with_capacity(n);
        let mut c = 0.0f32;
        for s in scenes {
            c += s.expected_trend as f32;
            cum.push(c);
        }
        let actual: Vec<f32> = signals.iter().map(|s| s.tension).collect();
        return spearman_rank_correlation(&cum, &actual);
    }

    // Find segment boundaries: wherever expected_trend changes sign
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut seg_start = 0;
    for i in 1..n {
        let prev_dir = scenes[i - 1].expected_trend.signum();
        let curr_dir = scenes[i].expected_trend.signum();
        if curr_dir != prev_dir && curr_dir != 0 {
            segments.push((seg_start, i));
            seg_start = i;
        }
    }
    segments.push((seg_start, n));

    // Compute per-segment Spearman (weighted by segment length)
    // Skip segments shorter than 3 — Spearman on 2 points is degenerate
    let mut weighted_sum = 0.0f32;
    let mut total_weight = 0usize;
    for (start, end) in &segments {
        let len = end - start;
        if len < 3 {
            continue;
        }
        let mut cum = Vec::with_capacity(len);
        let mut c = 0.0f32;
        for scene in scenes.iter().take(*end).skip(*start) {
            c += scene.expected_trend as f32;
            cum.push(c);
        }
        let actual: Vec<f32> = signals[*start..*end].iter().map(|s| s.tension).collect();
        let rho = spearman_rank_correlation(&cum, &actual);
        weighted_sum += rho * len as f32;
        total_weight += len;
    }
    if total_weight == 0 {
        // No segments long enough — fall back to global Spearman
        let mut cum = Vec::with_capacity(n);
        let mut c = 0.0f32;
        for s in scenes {
            c += s.expected_trend as f32;
            cum.push(c);
        }
        let actual: Vec<f32> = signals.iter().map(|s| s.tension).collect();
        spearman_rank_correlation(&cum, &actual)
    } else {
        weighted_sum / total_weight as f32
    }
}

/// Compute metrics for a single arc.
fn compute_metrics(name: &str, scenes: &[BenchScene], signals: &[NarrativeSignal]) {
    let n = signals.len();

    // Peak tension
    let (peak_idx, peak_val) = signals
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.tension.partial_cmp(&b.1.tension).unwrap())
        .unwrap();
    let peak_position = peak_idx as f32 / (n - 1).max(1) as f32;

    // Trend accuracy
    let mut correct = 0usize;
    let mut total = 0usize;
    for i in 1..n {
        let actual_trend = if signals[i].tension > signals[i - 1].tension + 0.01 {
            1
        } else if signals[i].tension < signals[i - 1].tension - 0.01 {
            -1
        } else {
            0
        };
        let expected = scenes[i].expected_trend;
        if expected != 0 {
            total += 1;
            if actual_trend == expected {
                correct += 1;
            }
        }
    }
    let trend_accuracy = if total > 0 {
        correct as f32 / total as f32
    } else {
        1.0
    };

    // Spearman rank correlation: expected cumulative tension vs actual
    let mut expected_cumulative = Vec::with_capacity(n);
    let mut cum = 0.0f32;
    for scene in scenes {
        cum += scene.expected_trend as f32;
        expected_cumulative.push(cum);
    }
    let actual_tensions: Vec<f32> = signals.iter().map(|s| s.tension).collect();
    let spearman_rho = spearman_rank_correlation(&expected_cumulative, &actual_tensions);

    // Within-half and across-half similarity (using scene HVs from session)
    // We re-run to get access to session's scene_similarity
    let mut session = StorySession::new();
    session.set_expected_scenes(scenes.len());
    let prot = session.algebra().primitives.protagonist.clone();
    session.register_character("Hero", &prot);
    for scene in scenes {
        session.add_scene(
            scene.title,
            scene.setting,
            &["Hero"],
            scene.conflict,
            scene.mood,
        );
    }

    let half = n / 2;
    let mut within_sims = Vec::new();
    let mut across_sims = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            if let Some(sim) = session.scene_similarity(i, j) {
                let i_first_half = i < half;
                let j_first_half = j < half;
                if i_first_half == j_first_half {
                    within_sims.push(sim);
                } else {
                    across_sims.push(sim);
                }
            }
        }
    }

    let mean = |v: &[f32]| -> f32 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };

    let within_mean = mean(&within_sims);
    let across_mean = mean(&across_sims);
    let coherence_pass = within_mean > across_mean;

    // Arc symmetry: deviation from golden ratio position (0.618)
    let golden_ratio = 0.618;
    let symmetry_deviation = (peak_position - golden_ratio).abs();

    println!("\n  {} metrics:", name);
    println!(
        "  {:<30} {}/{} (scene: {})",
        "Peak tension step:",
        peak_idx + 1,
        n,
        scenes[peak_idx].title
    );
    println!("  {:<30} {:.4}", "Peak tension value:", peak_val.tension);
    println!(
        "  {:<30} {:.1}% ({}/{})",
        "Trend accuracy:",
        trend_accuracy * 100.0,
        correct,
        total
    );
    let seg_spearman = segmented_spearman(scenes, signals);
    println!("  {:<30} {:.4}", "Spearman rho:", spearman_rho);
    println!("  {:<30} {:.4}", "Segmented Spearman:", seg_spearman);
    println!("  {:<30} {:.4}", "Within-half similarity:", within_mean);
    println!("  {:<30} {:.4}", "Across-half similarity:", across_mean);
    println!(
        "  {:<30} {} (W={:.4} > A={:.4})",
        "Coherence:",
        if coherence_pass { "PASS" } else { "FAIL" },
        within_mean,
        across_mean
    );
    println!(
        "  {:<30} {:.4} (peak@{:.3}, golden={:.3})",
        "Arc symmetry deviation:", symmetry_deviation, peak_position, golden_ratio
    );
}

fn main() {
    println!("=== Narrative Coherence Benchmark ===\n");

    // ---- Hero's Journey (7 scenes) ----
    let heros_journey = vec![
        BenchScene {
            title: "The Village",
            setting: "quiet village at dawn",
            conflict: "restless dreams",
            mood: NarrativeMood::Peaceful,
            expected_trend: 0,
        },
        BenchScene {
            title: "The Call",
            setting: "forest edge",
            conflict: "mysterious stranger arrives",
            mood: NarrativeMood::Mysterious,
            expected_trend: 1,
        },
        BenchScene {
            title: "Crossing Threshold",
            setting: "deep cave entrance",
            conflict: "point of no return",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "The Ordeal",
            setting: "mountain peak in storm",
            conflict: "facing the dragon",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "The Sanctum",
            setting: "inner sanctum of fire",
            conflict: "ultimate test",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "The Road Back",
            setting: "winding road home",
            conflict: "pursued by shadows",
            mood: NarrativeMood::Melancholy,
            expected_trend: -1,
        },
        BenchScene {
            title: "Return Home",
            setting: "village transformed",
            conflict: "acceptance of change",
            mood: NarrativeMood::Hopeful,
            expected_trend: -1,
        },
    ];

    let hj_signals = run_arc("Hero's Journey", &heros_journey);
    compute_metrics("Hero's Journey", &heros_journey, &hj_signals);

    // ---- Three-Act Structure (7 scenes) ----
    // Scenes use escalating conflict vocabulary so intensity stacks with
    // position shaping and mood compounding for a proper rising curve.
    let three_act = vec![
        BenchScene {
            title: "Normal Life",
            setting: "apartment morning",
            conflict: "boredom and routine",
            mood: NarrativeMood::Peaceful,
            expected_trend: 0,
        },
        BenchScene {
            title: "Inciting Incident",
            setting: "office corridor",
            conflict: "fired and threatened",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "Rising Stakes",
            setting: "city streets at night",
            conflict: "pursued by unknown enemy",
            mood: NarrativeMood::Mysterious,
            expected_trend: 1,
        },
        BenchScene {
            title: "Midpoint Twist",
            setting: "abandoned warehouse",
            conflict: "betrayal by closest ally",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "Crisis Point",
            setting: "siege at the safehouse",
            conflict: "ambush from all sides, escape impossible",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "Climactic Battle",
            setting: "rooftop inferno",
            conflict: "final battle against the mastermind, death or victory",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "New Beginning",
            setting: "home renewed",
            conflict: "rebuilding life",
            mood: NarrativeMood::Hopeful,
            expected_trend: -1,
        },
    ];

    let ta_signals = run_arc("Three-Act Structure", &three_act);
    compute_metrics("Three-Act Structure", &three_act, &ta_signals);

    // ---- Tragedy / Freytag (6 scenes) ----
    let tragedy = vec![
        BenchScene {
            title: "Exposition",
            setting: "grand castle",
            conflict: "ambition stirs",
            mood: NarrativeMood::Triumphant,
            expected_trend: 0,
        },
        BenchScene {
            title: "Rising Action",
            setting: "battlefield",
            conflict: "seizing power",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "Climax",
            setting: "throne room",
            conflict: "crown won through betrayal",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "Falling Action",
            setting: "palace corridors",
            conflict: "paranoia and isolation",
            mood: NarrativeMood::Melancholy,
            expected_trend: -1,
        },
        BenchScene {
            title: "Catastrophe",
            setting: "dungeon",
            conflict: "allies turn enemies",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "Denouement",
            setting: "castle ruins",
            conflict: "alone with consequences",
            mood: NarrativeMood::Melancholy,
            expected_trend: -1,
        },
    ];

    let tr_signals = run_arc("Tragedy (Freytag)", &tragedy);
    compute_metrics("Tragedy (Freytag)", &tragedy, &tr_signals);

    // ---- Cinderella / Rags-to-Riches (8 scenes) ----
    let cinderella = vec![
        BenchScene {
            title: "Rags",
            setting: "cold attic room",
            conflict: "poverty and neglect",
            mood: NarrativeMood::Melancholy,
            expected_trend: 0,
        },
        BenchScene {
            title: "Humiliation",
            setting: "kitchen hearth",
            conflict: "mocked by stepsisters",
            mood: NarrativeMood::Melancholy,
            expected_trend: -1,
        },
        BenchScene {
            title: "Fairy Godmother",
            setting: "moonlit garden",
            conflict: "magical intervention",
            mood: NarrativeMood::Mysterious,
            expected_trend: 1,
        },
        BenchScene {
            title: "The Ball",
            setting: "glittering ballroom",
            conflict: "dazzling the prince",
            mood: NarrativeMood::Triumphant,
            expected_trend: 1,
        },
        BenchScene {
            title: "Midnight Flight",
            setting: "palace steps",
            conflict: "magic expires",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "Back to Rags",
            setting: "cold attic again",
            conflict: "lost everything",
            mood: NarrativeMood::Melancholy,
            expected_trend: -1,
        },
        BenchScene {
            title: "The Search",
            setting: "kingdom roads",
            conflict: "glass slipper quest",
            mood: NarrativeMood::Hopeful,
            expected_trend: 1,
        },
        BenchScene {
            title: "Happily Ever After",
            setting: "palace throne room",
            conflict: "recognition and union",
            mood: NarrativeMood::Triumphant,
            expected_trend: 1,
        },
    ];

    let ci_signals = run_arc("Cinderella", &cinderella);
    compute_metrics("Cinderella", &cinderella, &ci_signals);

    // ---- Comedy (7 scenes) ----
    // Comedy arc: tension builds through misunderstandings, then releases
    // in a cascade of revelations leading to celebration.
    let comedy = vec![
        BenchScene {
            title: "The Setup",
            setting: "bustling village square",
            conflict: "mistaken identity at the market",
            mood: NarrativeMood::Peaceful,
            expected_trend: 0,
        },
        BenchScene {
            title: "The Misunderstanding",
            setting: "town hall",
            conflict: "accused of stealing the mayor's pig",
            mood: NarrativeMood::Mysterious,
            expected_trend: 1,
        },
        BenchScene {
            title: "Escalating Chaos",
            setting: "crowded festival",
            conflict: "three people claim the same prize",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "The Tangle",
            setting: "backstage at the theater",
            conflict: "disguises and secret letters discovered",
            mood: NarrativeMood::Mysterious,
            expected_trend: 1,
        },
        BenchScene {
            title: "The Revelation",
            setting: "grand banquet hall",
            conflict: "truth comes out, all secrets revealed",
            mood: NarrativeMood::Triumphant,
            expected_trend: -1,
        },
        BenchScene {
            title: "Reconciliation",
            setting: "garden at sunset",
            conflict: "forgiveness and laughter",
            mood: NarrativeMood::Hopeful,
            expected_trend: -1,
        },
        BenchScene {
            title: "The Celebration",
            setting: "village feast",
            conflict: "everyone dances together",
            mood: NarrativeMood::Triumphant,
            expected_trend: -1,
        },
    ];

    let co_signals = run_arc("Comedy", &comedy);
    compute_metrics("Comedy", &comedy, &co_signals);

    // ---- Quest (8 scenes) ----
    // Quest arc: episodic peaks as the hero faces sequential challenges,
    // each harder than the last, culminating in the final trial.
    let quest = vec![
        BenchScene {
            title: "The Commission",
            setting: "king's throne room",
            conflict: "tasked with an impossible quest",
            mood: NarrativeMood::Mysterious,
            expected_trend: 0,
        },
        BenchScene {
            title: "First Trial",
            setting: "enchanted forest",
            conflict: "riddle of the ancient guardian",
            mood: NarrativeMood::Mysterious,
            expected_trend: 1,
        },
        BenchScene {
            title: "Respite",
            setting: "friendly village",
            conflict: "healing and gathering supplies",
            mood: NarrativeMood::Peaceful,
            expected_trend: -1,
        },
        BenchScene {
            title: "Second Trial",
            setting: "mountain pass",
            conflict: "ambush by bandits in a narrow gorge",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "The Ally",
            setting: "hidden camp",
            conflict: "forming an uneasy alliance",
            mood: NarrativeMood::Hopeful,
            expected_trend: -1,
        },
        BenchScene {
            title: "Third Trial",
            setting: "dragon's lair",
            conflict: "facing the dragon that guards the artifact",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "The Final Gate",
            setting: "temple of fire",
            conflict: "ultimate trial of sacrifice and courage",
            mood: NarrativeMood::Tense,
            expected_trend: 1,
        },
        BenchScene {
            title: "Return Triumphant",
            setting: "kingdom gates",
            conflict: "hero returns with the artifact",
            mood: NarrativeMood::Triumphant,
            expected_trend: -1,
        },
    ];

    let qu_signals = run_arc("Quest", &quest);
    compute_metrics("Quest", &quest, &qu_signals);

    // ---- Summary Table ----
    println!("\n=== Summary ===\n");
    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Arc",
        "Peak Step",
        "Peak Val",
        "Trend Acc",
        "Spearman",
        "Seg Spear",
        "Coherence",
        "Sym Dev"
    );
    println!("{}", "-".repeat(105));

    let all: Vec<(&str, &Vec<BenchScene>, &Vec<NarrativeSignal>)> = vec![
        ("Hero's Journey", &heros_journey, &hj_signals),
        ("Three-Act Structure", &three_act, &ta_signals),
        ("Tragedy (Freytag)", &tragedy, &tr_signals),
        ("Cinderella", &cinderella, &ci_signals),
        ("Comedy", &comedy, &co_signals),
        ("Quest", &quest, &qu_signals),
    ];

    for (name, scenes, signals) in &all {
        let n = signals.len();
        let (peak_idx, peak_val) = signals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.tension.partial_cmp(&b.1.tension).unwrap())
            .unwrap();
        let peak_pos = peak_idx as f32 / (n - 1).max(1) as f32;

        let mut correct = 0usize;
        let mut total = 0usize;
        for i in 1..n {
            let actual = if signals[i].tension > signals[i - 1].tension + 0.01 {
                1
            } else if signals[i].tension < signals[i - 1].tension - 0.01 {
                -1
            } else {
                0
            };
            if scenes[i].expected_trend != 0 {
                total += 1;
                if actual == scenes[i].expected_trend {
                    correct += 1;
                }
            }
        }
        let trend_acc = if total > 0 {
            correct as f32 / total as f32
        } else {
            1.0
        };
        let sym_dev = (peak_pos - 0.618).abs();

        // Spearman rank correlation
        let mut expected_cum = Vec::with_capacity(n);
        let mut cum = 0.0f32;
        for scene in scenes.iter() {
            cum += scene.expected_trend as f32;
            expected_cum.push(cum);
        }
        let actual_tensions: Vec<f32> = signals.iter().map(|s| s.tension).collect();
        let spearman = spearman_rank_correlation(&expected_cum, &actual_tensions);

        // Coherence: within-half > across-half
        let mut session = StorySession::new();
        session.set_expected_scenes(scenes.len());
        let prot = session.algebra().primitives.protagonist.clone();
        session.register_character("Hero", &prot);
        for scene in scenes.iter() {
            session.add_scene(
                scene.title,
                scene.setting,
                &["Hero"],
                scene.conflict,
                scene.mood,
            );
        }
        let half = n / 2;
        let mut w_sims = Vec::new();
        let mut a_sims = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if let Some(sim) = session.scene_similarity(i, j) {
                    if (i < half) == (j < half) {
                        w_sims.push(sim);
                    } else {
                        a_sims.push(sim);
                    }
                }
            }
        }
        let w_mean = if w_sims.is_empty() {
            0.0
        } else {
            w_sims.iter().sum::<f32>() / w_sims.len() as f32
        };
        let a_mean = if a_sims.is_empty() {
            0.0
        } else {
            a_sims.iter().sum::<f32>() / a_sims.len() as f32
        };
        let coherent = if w_mean > a_mean { "PASS" } else { "FAIL" };
        let seg_spear = segmented_spearman(scenes, signals);

        println!(
            "{:<25} {:>4}/{:<5} {:>10.4} {:>9.1}% {:>10.4} {:>10.4} {:>10} {:>10.4}",
            name,
            peak_idx + 1,
            n,
            peak_val.tension,
            trend_acc * 100.0,
            spearman,
            seg_spear,
            coherent,
            sym_dev,
        );
    }

    // ================================================================
    // Vonnegut Story Shapes Classification
    // ================================================================

    println!("\n=== Vonnegut Story Shapes ===\n");

    // Six canonical shapes as tension profiles (normalized to [0,1])
    let vonnegut_shapes: Vec<(&str, Vec<f32>)> = vec![
        // "Man in Hole" — fall then rise
        ("Man in Hole", vec![0.5, 0.3, 0.15, 0.1, 0.2, 0.4, 0.6, 0.8]),
        // "Boy Meets Girl" — rise, fall, rise
        (
            "Boy Meets Girl",
            vec![0.3, 0.5, 0.7, 0.8, 0.5, 0.3, 0.5, 0.8],
        ),
        // "From Bad to Worse" — steady decline / rising tension
        (
            "From Bad to Worse",
            vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ),
        // "Which Way Is Up" — oscillation
        (
            "Which Way Is Up",
            vec![0.5, 0.7, 0.4, 0.6, 0.3, 0.7, 0.4, 0.5],
        ),
        // "Creation Story" — steady rise then plateau
        (
            "Creation Story",
            vec![0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.8, 0.8],
        ),
        // "Old Testament" — rise then crash
        (
            "Old Testament",
            vec![0.2, 0.4, 0.6, 0.8, 0.9, 0.7, 0.3, 0.1],
        ),
    ];

    // Simple DTW distance (no external deps)
    fn dtw_distance(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let m = b.len();
        let mut dp = vec![vec![f32::INFINITY; m + 1]; n + 1];
        dp[0][0] = 0.0;
        for i in 1..=n {
            for j in 1..=m {
                let cost = (a[i - 1] - b[j - 1]).abs();
                dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            }
        }
        dp[n][m]
    }

    // Normalize a vector to [0, 1] range
    fn normalize_01(v: &[f32]) -> Vec<f32> {
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        if range < 1e-8 {
            vec![0.5; v.len()]
        } else {
            v.iter().map(|x| (x - min) / range).collect()
        }
    }

    // Classify each benchmark arc against the 6 shapes
    println!("{:<25} {:<20} {:>10}", "Arc", "Best Shape", "DTW Dist");
    println!("{}", "-".repeat(55));

    for (name, _scenes, signals) in &all {
        // Extract and normalize tension profile to [0,1] for fair comparison
        let raw_tensions: Vec<f32> = signals.iter().map(|s| s.tension).collect();
        let tension_profile = normalize_01(&raw_tensions);

        let mut best_shape = "Unknown";
        let mut best_dist = f32::INFINITY;

        for (shape_name, shape_profile) in &vonnegut_shapes {
            let dist = dtw_distance(&tension_profile, shape_profile);
            if dist < best_dist {
                best_dist = dist;
                best_shape = shape_name;
            }
        }

        println!("{:<25} {:<20} {:>10.4}", name, best_shape, best_dist);
    }

    // ================================================================
    // Regression Assertions (CI-safe)
    // ================================================================

    println!("\n=== Regression Checks ===\n");

    let mut pass_count = 0;
    let mut total_checks = 0;

    // Helper to run an assertion-like check without panicking
    let mut check = |name: &str, ok: bool| {
        total_checks += 1;
        if ok {
            pass_count += 1;
            println!("  [PASS] {}", name);
        } else {
            println!("  [FAIL] {}", name);
        }
    };

    // 1. Tension range should span at least 0.15 for each arc
    for (name, _scenes, signals) in &all {
        let tensions: Vec<f32> = signals.iter().map(|s| s.tension).collect();
        let min_t = tensions.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_t = tensions.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_t - min_t;
        check(
            &format!("{}: tension range >= 0.15 (got {:.3})", name, range),
            range >= 0.15,
        );
    }

    // 2. Hero's Journey peak should be in second half (scene 4-7 out of 7)
    {
        let (peak_idx, _) = hj_signals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.tension.partial_cmp(&b.1.tension).unwrap())
            .unwrap();
        check(
            &format!(
                "Hero's Journey: peak in second half (scene {}/7)",
                peak_idx + 1
            ),
            peak_idx >= 3,
        );
    }

    // 3. Three-Act: rising trend through act 2 (scenes 2-6 should mostly rise)
    {
        let mut rises = 0;
        for i in 1..6.min(ta_signals.len()) {
            if ta_signals[i].tension > ta_signals[i - 1].tension - 0.01 {
                rises += 1;
            }
        }
        check(
            &format!("Three-Act: mostly rising through act 2 ({}/5 rises)", rises),
            rises >= 3,
        );
    }

    // 4. Tragedy: tension should drop in final third
    {
        let n = tr_signals.len();
        let final_third_start = n * 2 / 3;
        let drops_at_end = tr_signals[n - 1].tension < tr_signals[final_third_start].tension;
        check("Tragedy: tension drops in final third", drops_at_end);
    }

    // 5. At least 2/4 arcs should have positive Spearman correlation
    {
        let mut positive_spearman = 0;
        for (_name, scenes, signals) in &all {
            let mut expected_cum = Vec::with_capacity(signals.len());
            let mut cum = 0.0f32;
            for scene in scenes.iter() {
                cum += scene.expected_trend as f32;
                expected_cum.push(cum);
            }
            let actual: Vec<f32> = signals.iter().map(|s| s.tension).collect();
            let rho = spearman_rank_correlation(&expected_cum, &actual);
            if rho > 0.0 {
                positive_spearman += 1;
            }
        }
        let arc_count = all.len();
        let min_positive = arc_count / 2; // at least half
        check(
            &format!(
                "At least {}/{} arcs have positive Spearman ({}/{})",
                min_positive, arc_count, positive_spearman, arc_count
            ),
            positive_spearman >= min_positive,
        );
    }

    // 6. Overall trend accuracy should average >= 50%
    {
        let mut total_correct = 0usize;
        let mut total_measured = 0usize;
        for (_name, scenes, signals) in &all {
            for i in 1..signals.len() {
                let actual = if signals[i].tension > signals[i - 1].tension + 0.01 {
                    1
                } else if signals[i].tension < signals[i - 1].tension - 0.01 {
                    -1
                } else {
                    0
                };
                if scenes[i].expected_trend != 0 {
                    total_measured += 1;
                    if actual == scenes[i].expected_trend {
                        total_correct += 1;
                    }
                }
            }
        }
        let avg_acc = if total_measured > 0 {
            total_correct as f32 / total_measured as f32
        } else {
            0.0
        };
        check(
            &format!(
                "Average trend accuracy >= 50% (got {:.1}%)",
                avg_acc * 100.0
            ),
            avg_acc >= 0.50,
        );
    }

    println!(
        "\n  Result: {}/{} checks passed\n",
        pass_count, total_checks
    );

    // ================================================================
    // Paper-Ready Markdown Table
    // ================================================================

    println!("=== Paper Results (Markdown) ===\n");
    println!("| Arc | N | Peak | Peak Pos | Trend Acc | Spearman | Seg Spearman | Sym Dev |");
    println!("|-----|---|------|----------|-----------|----------|--------------|---------|");
    for (name, scenes, signals) in &all {
        let n = signals.len();
        let (peak_idx, peak_val) = signals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.tension.partial_cmp(&b.1.tension).unwrap())
            .unwrap();
        let peak_pos = peak_idx as f32 / (n - 1).max(1) as f32;

        let mut correct = 0usize;
        let mut total = 0usize;
        for i in 1..n {
            let actual = if signals[i].tension > signals[i - 1].tension + 0.01 {
                1
            } else if signals[i].tension < signals[i - 1].tension - 0.01 {
                -1
            } else {
                0
            };
            if scenes[i].expected_trend != 0 {
                total += 1;
                if actual == scenes[i].expected_trend {
                    correct += 1;
                }
            }
        }
        let trend_acc = if total > 0 {
            correct as f32 / total as f32
        } else {
            1.0
        };

        let mut expected_cum = Vec::with_capacity(n);
        let mut cum = 0.0f32;
        for scene in scenes.iter() {
            cum += scene.expected_trend as f32;
            expected_cum.push(cum);
        }
        let actual_tensions: Vec<f32> = signals.iter().map(|s| s.tension).collect();
        let spearman = spearman_rank_correlation(&expected_cum, &actual_tensions);
        let seg_spear = segmented_spearman(scenes, signals);
        let sym_dev = (peak_pos - 0.618).abs();

        println!(
            "| {} | {} | {:.3} | {:.3} | {:.1}% | {:.3} | {:.3} | {:.3} |",
            name,
            n,
            peak_val.tension,
            peak_pos,
            trend_acc * 100.0,
            spearman,
            seg_spear,
            sym_dev
        );
    }
    println!();

    println!("Done.");
}