//! Narrative Coherence Benchmark
//!
//! Tests 3 canonical story arcs through the narrative dynamics pipeline and
//! reports paper-worthy metrics: peak tension, trend accuracy, within/across
//! half similarity, and arc symmetry deviation (golden ratio).
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

    // Within-half and across-half similarity (using scene HVs from session)
    // We re-run to get access to session's scene_similarity
    let mut session = StorySession::new();
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

    // Arc symmetry: deviation from golden ratio position (0.618)
    let golden_ratio = 0.618;
    let symmetry_deviation = (peak_position - golden_ratio).abs();

    println!("\n  {} metrics:", name);
    println!("  {:<30} {}", "Peak tension step:", format!("{}/{} (scene: {})", peak_idx + 1, n, scenes[peak_idx].title));
    println!("  {:<30} {:.4}", "Peak tension value:", peak_val.tension);
    println!("  {:<30} {:.1}% ({}/{})", "Trend accuracy:", trend_accuracy * 100.0, correct, total);
    println!("  {:<30} {:.4}", "Within-half similarity:", within_mean);
    println!("  {:<30} {:.4}", "Across-half similarity:", across_mean);
    println!("  {:<30} {:.4} (peak@{:.3}, golden={:.3})", "Arc symmetry deviation:", symmetry_deviation, peak_position, golden_ratio);
}

fn main() {
    println!("=== Narrative Coherence Benchmark ===\n");

    // ---- Hero's Journey (7 scenes) ----
    let heros_journey = vec![
        BenchScene { title: "The Village", setting: "quiet village at dawn", conflict: "restless dreams", mood: NarrativeMood::Peaceful, expected_trend: 0 },
        BenchScene { title: "The Call", setting: "forest edge", conflict: "mysterious stranger arrives", mood: NarrativeMood::Mysterious, expected_trend: 1 },
        BenchScene { title: "Crossing Threshold", setting: "deep cave entrance", conflict: "point of no return", mood: NarrativeMood::Tense, expected_trend: 1 },
        BenchScene { title: "The Ordeal", setting: "mountain peak in storm", conflict: "facing the dragon", mood: NarrativeMood::Tense, expected_trend: 1 },
        BenchScene { title: "The Sanctum", setting: "inner sanctum of fire", conflict: "ultimate test", mood: NarrativeMood::Tense, expected_trend: 1 },
        BenchScene { title: "The Road Back", setting: "winding road home", conflict: "pursued by shadows", mood: NarrativeMood::Melancholy, expected_trend: -1 },
        BenchScene { title: "Return Home", setting: "village transformed", conflict: "acceptance of change", mood: NarrativeMood::Hopeful, expected_trend: -1 },
    ];

    let hj_signals = run_arc("Hero's Journey", &heros_journey);
    compute_metrics("Hero's Journey", &heros_journey, &hj_signals);

    // ---- Three-Act Structure (7 scenes) ----
    let three_act = vec![
        BenchScene { title: "Normal Life", setting: "apartment morning", conflict: "boredom and routine", mood: NarrativeMood::Peaceful, expected_trend: 0 },
        BenchScene { title: "Inciting Incident", setting: "office", conflict: "fired unexpectedly", mood: NarrativeMood::Tense, expected_trend: 1 },
        BenchScene { title: "Rising Stakes", setting: "city streets at night", conflict: "discovers conspiracy", mood: NarrativeMood::Mysterious, expected_trend: 1 },
        BenchScene { title: "Midpoint Twist", setting: "confrontation alley", conflict: "ally revealed as traitor", mood: NarrativeMood::Tense, expected_trend: 1 },
        BenchScene { title: "Crisis Point", setting: "crisis meeting", conflict: "all options exhausted", mood: NarrativeMood::Tense, expected_trend: 1 },
        BenchScene { title: "Climactic Battle", setting: "rooftop showdown", conflict: "final confrontation", mood: NarrativeMood::Tense, expected_trend: 1 },
        BenchScene { title: "New Beginning", setting: "home renewed", conflict: "rebuilding life", mood: NarrativeMood::Hopeful, expected_trend: -1 },
    ];

    let ta_signals = run_arc("Three-Act Structure", &three_act);
    compute_metrics("Three-Act Structure", &three_act, &ta_signals);

    // ---- Tragedy / Freytag (6 scenes) ----
    let tragedy = vec![
        BenchScene { title: "Exposition", setting: "grand castle", conflict: "ambition stirs", mood: NarrativeMood::Triumphant, expected_trend: 0 },
        BenchScene { title: "Rising Action", setting: "battlefield", conflict: "seizing power", mood: NarrativeMood::Tense, expected_trend: 1 },
        BenchScene { title: "Climax", setting: "throne room", conflict: "crown won through betrayal", mood: NarrativeMood::Tense, expected_trend: 1 },
        BenchScene { title: "Falling Action", setting: "palace corridors", conflict: "paranoia and isolation", mood: NarrativeMood::Melancholy, expected_trend: -1 },
        BenchScene { title: "Catastrophe", setting: "dungeon", conflict: "allies turn enemies", mood: NarrativeMood::Tense, expected_trend: 1 },
        BenchScene { title: "Denouement", setting: "castle ruins", conflict: "alone with consequences", mood: NarrativeMood::Melancholy, expected_trend: -1 },
    ];

    let tr_signals = run_arc("Tragedy (Freytag)", &tragedy);
    compute_metrics("Tragedy (Freytag)", &tragedy, &tr_signals);

    // ---- Summary Table ----
    println!("\n=== Summary ===\n");
    println!("{:<25} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Arc", "Peak Step", "Peak Val", "Trend Acc", "W-Half Sim", "Symmetry Dev");
    println!("{}", "-".repeat(85));

    let all = [
        ("Hero's Journey", &heros_journey, &hj_signals),
        ("Three-Act Structure", &three_act, &ta_signals),
        ("Tragedy (Freytag)", &tragedy, &tr_signals),
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
        let trend_acc = if total > 0 { correct as f32 / total as f32 } else { 1.0 };
        let sym_dev = (peak_pos - 0.618).abs();

        // Quick within-half mean (re-run session)
        let mut session = StorySession::new();
        let prot = session.algebra().primitives.protagonist.clone();
        session.register_character("Hero", &prot);
        for scene in scenes.iter() {
            session.add_scene(scene.title, scene.setting, &["Hero"], scene.conflict, scene.mood);
        }
        let half = n / 2;
        let mut w_sims = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if let Some(sim) = session.scene_similarity(i, j) {
                    if (i < half) == (j < half) {
                        w_sims.push(sim);
                    }
                }
            }
        }
        let w_mean = if w_sims.is_empty() { 0.0 } else { w_sims.iter().sum::<f32>() / w_sims.len() as f32 };

        println!(
            "{:<25} {:>5}/{:<6} {:>12.4} {:>11.1}% {:>12.4} {:>12.4}",
            name,
            peak_idx + 1,
            n,
            peak_val.tension,
            trend_acc * 100.0,
            w_mean,
            sym_dev,
        );
    }

    println!("\nDone.");
}
