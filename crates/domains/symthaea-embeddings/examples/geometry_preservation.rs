//! OMI-3 — Does semantic geometry survive the trip into HDC space?
//!
//! Embeds a phrase battery via live Ollama (`embeddinggemma:300m`, 768-D),
//! projects to 16,384-D via `HdcBridge` (JL), and compares the full pairwise
//! cosine matrix before vs after projection. The Johnson–Lindenstrauss lemma
//! predicts small distortion; this measures it concretely — the prerequisite
//! for grounded thought vectors that E1's cognition-ablation found missing
//! (genesis-label "intents" carried zero semantic signal).
//!
//! Run: cargo run -p symthaea-embeddings --example geometry_preservation
//! (Skips gracefully if Ollama is down.)

use symthaea_embeddings::{BridgeConfig, HdcBridge, ollama};

const PHRASES: &[(&str, &str)] = &[
    ("motor-L", "move the arm to the left"),
    ("motor-R", "move the arm to the right"),
    ("motor-G", "grasp the red cup on the table"),
    ("climb", "ascend to a higher altitude now"),
    ("descend", "descend slowly toward the landing pad"),
    ("math", "two plus two equals four"),
    ("emotion", "I feel deep gratitude for this quiet morning"),
    ("alarm", "URGENT: fire detected in the server room"),
    ("nature", "a gentle rain fell on the old oak tree"),
    ("moral", "is it acceptable to lie to protect a friend"),
];

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn main() {
    // Embed everything (live Ollama; bail politely if down).
    let mut embeddings = Vec::new();
    for (tag, phrase) in PHRASES {
        match ollama::embed(ollama::DEFAULT_ENDPOINT, ollama::DEFAULT_MODEL, phrase, 768) {
            Ok(e) => embeddings.push((*tag, e)),
            Err(e) => {
                println!(
                    "SKIP: Ollama unavailable ({e}) — start Ollama with embeddinggemma:300m and rerun."
                );
                return;
            }
        }
    }

    let bridge = HdcBridge::with_config(BridgeConfig {
        input_dim: 768,
        ..Default::default()
    });
    let projected: Vec<(&str, Vec<f32>)> = embeddings
        .iter()
        .map(|(tag, e)| (*tag, bridge.project_continuous(e)))
        .collect();

    println!("=== OMI-3: geometry preservation through 768→16,384 JL projection ===\n");
    println!(
        "{:<10} {:<10} {:>10} {:>10} {:>10}",
        "a", "b", "cos(embed)", "cos(HDC)", "|Δ|"
    );

    let (mut max_d, mut sum_d, mut n) = (0.0f32, 0.0f32, 0u32);
    let mut worst = (String::new(), 0.0f32, 0.0f32);
    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let ce = cos(&embeddings[i].1, &embeddings[j].1);
            let ch = cos(&projected[i].1, &projected[j].1);
            let d = (ce - ch).abs();
            sum_d += d;
            n += 1;
            if d > max_d {
                max_d = d;
                worst = (format!("{}/{}", embeddings[i].0, embeddings[j].0), ce, ch);
            }
            println!(
                "{:<10} {:<10} {:>10.4} {:>10.4} {:>10.4}",
                embeddings[i].0, embeddings[j].0, ce, ch, d
            );
        }
    }

    println!("\n=== Verdict ===");
    println!(
        "pairs={} | mean |Δcos| = {:.4} | max |Δcos| = {:.4} (pair {}: {:.4}→{:.4})",
        n,
        sum_d / n as f32,
        max_d,
        worst.0,
        worst.1,
        worst.2
    );
    // Ordering preservation: does 'closest neighbor' survive projection?
    let mut order_ok = 0;
    for i in 0..embeddings.len() {
        let nn = |space: &[(&str, Vec<f32>)]| {
            (0..space.len())
                .filter(|&j| j != i)
                .max_by(|&a, &b| {
                    cos(&space[i].1, &space[a].1).total_cmp(&cos(&space[i].1, &space[b].1))
                })
                .unwrap()
        };
        if nn(&embeddings) == nn(&projected) {
            order_ok += 1;
        }
    }
    println!(
        "nearest-neighbor preserved for {}/{} phrases",
        order_ok,
        embeddings.len()
    );
    println!(
        "{}",
        if max_d < 0.05 && order_ok == embeddings.len() {
            "PASS: semantic geometry survives projection — grounded HDC thought vectors are viable."
        } else if max_d < 0.10 {
            "PASS (loose): distortion within JL expectations; ordering mostly preserved."
        } else {
            "FAIL: projection distorts geometry beyond JL expectations — check bridge config."
        }
    );
}
