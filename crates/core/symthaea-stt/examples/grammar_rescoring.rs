// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Grammar-Based Hypothesis Rescoring
//!
//! Demonstrates how the UnifiedGrammar can be used to rescore multiple
//! phoneme hypotheses from an acoustic model, improving recognition accuracy
//! by incorporating phonotactic knowledge.
//!
//! Key insight: Acoustic confusion sets (e.g., P/B/T confusion) can be
//! disambiguated by checking which hypothesis creates a valid phonotactic
//! sequence in context.

use symthaea_stt::temporal_grammar::TemporalEvent;
use symthaea_stt::unified_grammar::UnifiedGrammar;

fn separator(c: char, n: usize) {
    println!("{}", std::iter::repeat(c).take(n).collect::<String>());
}

fn header(title: &str) {
    println!();
    separator('=', 70);
    println!("  {}", title);
    separator('=', 70);
    println!();
}

fn subheader(title: &str) {
    println!();
    separator('-', 70);
    println!("  {}", title);
    separator('-', 70);
    println!();
}

/// Represents a phoneme hypothesis with acoustic confidence
#[derive(Debug, Clone)]
struct PhonemeHypothesis {
    phoneme: String,
    acoustic_score: f32, // Higher = more confident acoustic match
}

/// Represents a frame with multiple competing phoneme hypotheses
#[derive(Debug, Clone)]
struct FrameHypotheses {
    hypotheses: Vec<PhonemeHypothesis>,
    time: f32,
    duration: f32,
}

/// N-best list of complete sequences
#[derive(Debug, Clone)]
struct SequenceHypothesis {
    phonemes: Vec<String>,
    acoustic_score: f32,
    grammar_score: f32,
    combined_score: f32,
}

/// Grammar-integrated rescorer
struct GrammarRescorer {
    grammar: UnifiedGrammar,
    acoustic_weight: f32,
    grammar_weight: f32,
}

impl GrammarRescorer {
    fn new(grammar: UnifiedGrammar) -> Self {
        Self {
            grammar,
            acoustic_weight: 0.6, // Trust acoustic model more
            grammar_weight: 0.4,  // But grammar provides phonotactic constraints
        }
    }

    /// Set rescoring weights
    fn set_weights(&mut self, acoustic: f32, grammar: f32) {
        let total = acoustic + grammar;
        self.acoustic_weight = acoustic / total;
        self.grammar_weight = grammar / total;
    }

    /// Convert phoneme sequence to temporal events
    fn phonemes_to_events(&self, phonemes: &[String]) -> Vec<TemporalEvent> {
        let mut events = Vec::new();
        let mut time = 0.0f32;

        for phoneme in phonemes {
            if let Some(cid) = self.grammar.category_id(phoneme) {
                let duration = estimate_duration(phoneme);
                let intensity = estimate_intensity(phoneme);
                events.push(TemporalEvent::new(phoneme, cid, time, duration, intensity));
                time += duration;
            }
        }

        events
    }

    /// Score a single sequence
    fn score_sequence(&mut self, phonemes: &[String], acoustic_score: f32) -> SequenceHypothesis {
        let events = self.phonemes_to_events(phonemes);
        let grammar_score = if events.len() >= 2 {
            self.grammar.score_sequence(&events)
        } else {
            0.0
        };

        // Normalize grammar score to [0, 1] range for combination
        // Grammar scores are typically in [-1, 1] range
        let normalized_grammar = (grammar_score + 1.0) / 2.0;

        let combined_score =
            self.acoustic_weight * acoustic_score + self.grammar_weight * normalized_grammar;

        SequenceHypothesis {
            phonemes: phonemes.to_vec(),
            acoustic_score,
            grammar_score,
            combined_score,
        }
    }

    /// Generate N-best hypotheses from frame-level confusion sets
    fn generate_nbest(&self, frames: &[FrameHypotheses], beam_width: usize) -> Vec<Vec<String>> {
        if frames.is_empty() {
            return vec![vec![]];
        }

        // Simple beam search to generate N-best
        let mut beams: Vec<(Vec<String>, f32)> = vec![(vec![], 1.0)];

        for frame in frames {
            let mut new_beams = Vec::new();

            for (prefix, prefix_score) in &beams {
                for hyp in &frame.hypotheses {
                    let mut new_seq = prefix.clone();
                    new_seq.push(hyp.phoneme.clone());
                    let new_score = prefix_score * hyp.acoustic_score;
                    new_beams.push((new_seq, new_score));
                }
            }

            // Sort by score and keep top beam_width
            new_beams.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            new_beams.truncate(beam_width);
            beams = new_beams;
        }

        beams.into_iter().map(|(seq, _)| seq).collect()
    }

    /// Rescore and rerank N-best list
    fn rescore_nbest(
        &mut self,
        nbest: &[Vec<String>],
        acoustic_scores: &[f32],
    ) -> Vec<SequenceHypothesis> {
        let mut scored: Vec<SequenceHypothesis> = nbest
            .iter()
            .zip(acoustic_scores.iter())
            .map(|(seq, &acoustic)| self.score_sequence(seq, acoustic))
            .collect();

        // Sort by combined score
        scored.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored
    }
}

fn estimate_duration(phoneme: &str) -> f32 {
    match phoneme {
        p if p.starts_with('A')
            || p.starts_with('E')
            || p.starts_with('I')
            || p.starts_with('O')
            || p.starts_with('U') =>
        {
            0.10
        }
        "S" | "Z" | "SH" | "ZH" | "F" | "V" | "TH" | "DH" => 0.08,
        "P" | "B" | "T" | "D" | "K" | "G" => 0.05,
        "M" | "N" | "NG" => 0.06,
        _ => 0.06,
    }
}

fn estimate_intensity(phoneme: &str) -> f32 {
    match phoneme {
        p if p.starts_with('A')
            || p.starts_with('E')
            || p.starts_with('I')
            || p.starts_with('O')
            || p.starts_with('U') =>
        {
            0.8
        }
        "B" | "D" | "G" | "V" | "Z" | "DH" | "ZH" | "M" | "N" | "NG" => 0.6,
        "P" | "T" | "K" | "F" | "S" | "TH" | "SH" | "HH" => 0.5,
        _ => 0.55,
    }
}

fn main() {
    header("GRAMMAR-BASED HYPOTHESIS RESCORING");

    println!("  Demonstrates phonotactic rescoring of acoustic hypotheses.");
    println!("  Grammar knowledge can resolve acoustic confusions.");

    // Create and train grammar
    let mut grammar = UnifiedGrammar::speech();

    let training_sequences: Vec<Vec<&str>> = vec![
        vec!["DH", "AH", "K", "AE", "T", "S", "AE", "T"],
        vec!["DH", "AH", "D", "AO", "G", "R", "AE", "N"],
        vec!["DH", "AH", "B", "AE", "T", "HH", "AE", "T"],
        vec!["B", "IH", "G", "P", "IH", "G", "D", "IH", "G"],
        vec!["F", "IH", "SH", "D", "IH", "SH", "W", "IH", "SH"],
        vec!["S", "T", "R", "AO", "NG", "S", "T", "R", "IH", "NG"],
        vec!["S", "P", "R", "IH", "NG", "T", "AY", "M"],
    ];

    for seq in &training_sequences {
        let events: Vec<TemporalEvent> = seq
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                grammar
                    .category_id(*p)
                    .map(|cid| TemporalEvent::new(*p, cid, i as f32 * 0.06, 0.06, 0.6))
            })
            .collect();
        for _ in 0..30 {
            grammar.train_sequence(&events);
        }
    }

    let stats = grammar.stats();
    subheader("Phase 1: Grammar Training");
    println!(
        "    Trained on {} example sequences",
        training_sequences.len()
    );
    println!(
        "    Grammar: {} clusters, {:.1}% density",
        stats.num_clusters,
        stats.avg_cluster_density * 100.0
    );

    let mut rescorer = GrammarRescorer::new(grammar);

    // Test Case 1: Stop consonant confusion (P/B/T)
    subheader("Test 1: Stop Consonant Confusion");
    println!("    Scenario: Acoustic model confuses P/B/T in word-initial position");
    println!("    Context: 'the _at sat'");
    println!();

    let confusions_1 = vec![
        (
            vec!["DH", "AH", "P", "AE", "T", "S", "AE", "T"],
            0.35,
            "PAT",
        ), // Wrong - acoustic confusion
        (
            vec!["DH", "AH", "B", "AE", "T", "S", "AE", "T"],
            0.30,
            "BAT",
        ), // Wrong - acoustic confusion
        (
            vec!["DH", "AH", "K", "AE", "T", "S", "AE", "T"],
            0.25,
            "CAT",
        ), // Correct - lower acoustic
        (
            vec!["DH", "AH", "T", "AE", "T", "S", "AE", "T"],
            0.10,
            "TAT",
        ), // Nonsense
    ];

    println!("    N-best (acoustic only):");
    for (seq, score, word) in &confusions_1 {
        println!(
            "      {:.2}  {}  \"the {} sat\"",
            score,
            seq.join(" "),
            word.to_lowercase()
        );
    }

    let nbest: Vec<Vec<String>> = confusions_1
        .iter()
        .map(|(seq, _, _)| seq.iter().map(|s| s.to_string()).collect())
        .collect();
    let acoustic_scores: Vec<f32> = confusions_1.iter().map(|(_, s, _)| *s).collect();

    let rescored = rescorer.rescore_nbest(&nbest, &acoustic_scores);

    println!();
    println!("    After grammar rescoring (60% acoustic, 40% grammar):");
    for (i, hyp) in rescored.iter().take(4).enumerate() {
        let word = confusions_1
            .iter()
            .find(|(seq, _, _)| *seq == hyp.phonemes.iter().map(|s| s.as_str()).collect::<Vec<_>>())
            .map(|(_, _, w)| *w)
            .unwrap_or("?");
        println!(
            "      #{} {:.3} (a:{:.2} g:{:+.2})  \"the {} sat\"",
            i + 1,
            hyp.combined_score,
            hyp.acoustic_score,
            hyp.grammar_score,
            word.to_lowercase()
        );
    }

    // Test Case 2: Invalid onset cluster rejection
    subheader("Test 2: Invalid Onset Cluster Rejection");
    println!("    Scenario: Acoustic model generates invalid consonant cluster");
    println!("    STR- is valid, TL- is invalid in English");
    println!();

    let confusions_2 = vec![
        (vec!["T", "L", "IY", "T"], 0.45, "invalid: TLEET"),
        (vec!["S", "T", "R", "IY", "T"], 0.35, "valid: STREET"),
        (vec!["S", "L", "IY", "T"], 0.15, "invalid: SLEET"),
        (vec!["T", "R", "IY", "T"], 0.05, "valid: TREET"),
    ];

    println!("    N-best (acoustic only):");
    for (seq, score, desc) in &confusions_2 {
        println!("      {:.2}  {}  ({})", score, seq.join(" "), desc);
    }

    let nbest: Vec<Vec<String>> = confusions_2
        .iter()
        .map(|(seq, _, _)| seq.iter().map(|s| s.to_string()).collect())
        .collect();
    let acoustic_scores: Vec<f32> = confusions_2.iter().map(|(_, s, _)| *s).collect();

    let rescored = rescorer.rescore_nbest(&nbest, &acoustic_scores);

    println!();
    println!("    After grammar rescoring:");
    for (i, hyp) in rescored.iter().take(4).enumerate() {
        let desc = confusions_2
            .iter()
            .find(|(seq, _, _)| *seq == hyp.phonemes.iter().map(|s| s.as_str()).collect::<Vec<_>>())
            .map(|(_, _, d)| *d)
            .unwrap_or("?");
        println!(
            "      #{} {:.3} (a:{:.2} g:{:+.2})  {} ({})",
            i + 1,
            hyp.combined_score,
            hyp.acoustic_score,
            hyp.grammar_score,
            hyp.phonemes.join(" "),
            desc
        );
    }

    // Test Case 3: Longer sequence with multiple confusions
    subheader("Test 3: Multiple Confusions in Sequence");
    println!("    Scenario: Several phonemes are uncertain");
    println!("    Target: 'big fish'");
    println!();

    // Create frame-level confusion sets
    let frames = vec![
        FrameHypotheses {
            hypotheses: vec![
                PhonemeHypothesis {
                    phoneme: "B".to_string(),
                    acoustic_score: 0.5,
                },
                PhonemeHypothesis {
                    phoneme: "P".to_string(),
                    acoustic_score: 0.3,
                },
                PhonemeHypothesis {
                    phoneme: "D".to_string(),
                    acoustic_score: 0.2,
                },
            ],
            time: 0.0,
            duration: 0.05,
        },
        FrameHypotheses {
            hypotheses: vec![
                PhonemeHypothesis {
                    phoneme: "IH".to_string(),
                    acoustic_score: 0.8,
                },
                PhonemeHypothesis {
                    phoneme: "IY".to_string(),
                    acoustic_score: 0.2,
                },
            ],
            time: 0.05,
            duration: 0.08,
        },
        FrameHypotheses {
            hypotheses: vec![
                PhonemeHypothesis {
                    phoneme: "G".to_string(),
                    acoustic_score: 0.6,
                },
                PhonemeHypothesis {
                    phoneme: "K".to_string(),
                    acoustic_score: 0.4,
                },
            ],
            time: 0.13,
            duration: 0.06,
        },
        FrameHypotheses {
            hypotheses: vec![
                PhonemeHypothesis {
                    phoneme: "F".to_string(),
                    acoustic_score: 0.7,
                },
                PhonemeHypothesis {
                    phoneme: "V".to_string(),
                    acoustic_score: 0.3,
                },
            ],
            time: 0.19,
            duration: 0.08,
        },
        FrameHypotheses {
            hypotheses: vec![
                PhonemeHypothesis {
                    phoneme: "IH".to_string(),
                    acoustic_score: 0.85,
                },
                PhonemeHypothesis {
                    phoneme: "IY".to_string(),
                    acoustic_score: 0.15,
                },
            ],
            time: 0.27,
            duration: 0.08,
        },
        FrameHypotheses {
            hypotheses: vec![
                PhonemeHypothesis {
                    phoneme: "SH".to_string(),
                    acoustic_score: 0.9,
                },
                PhonemeHypothesis {
                    phoneme: "CH".to_string(),
                    acoustic_score: 0.1,
                },
            ],
            time: 0.35,
            duration: 0.10,
        },
    ];

    // Generate N-best
    let nbest = rescorer.generate_nbest(&frames, 10);

    println!("    Frame-level confusion lattice:");
    for (i, frame) in frames.iter().enumerate() {
        let hyps: Vec<String> = frame
            .hypotheses
            .iter()
            .map(|h| format!("{}:{:.1}", h.phoneme, h.acoustic_score))
            .collect();
        println!("      Frame {}: [{}]", i + 1, hyps.join(", "));
    }

    println!();
    println!("    Generated {} hypotheses (beam=10)", nbest.len());

    // Compute acoustic scores for N-best (product of frame scores)
    let acoustic_scores: Vec<f32> = nbest
        .iter()
        .map(|seq| {
            seq.iter()
                .zip(frames.iter())
                .map(|(phoneme, frame)| {
                    frame
                        .hypotheses
                        .iter()
                        .find(|h| &h.phoneme == phoneme)
                        .map(|h| h.acoustic_score)
                        .unwrap_or(0.1)
                })
                .product()
        })
        .collect();

    let rescored = rescorer.rescore_nbest(&nbest, &acoustic_scores);

    println!();
    println!("    Top 5 after rescoring:");
    for (i, hyp) in rescored.iter().take(5).enumerate() {
        println!(
            "      #{} {:.4} (a:{:.4} g:{:+.4})  {}",
            i + 1,
            hyp.combined_score,
            hyp.acoustic_score,
            hyp.grammar_score,
            hyp.phonemes.join(" ")
        );
    }

    // Summary
    header("RESCORING SUMMARY");

    println!("    Grammar rescoring benefits:");
    println!("      1. Resolves acoustic confusions using phonotactic knowledge");
    println!("      2. Penalizes invalid consonant clusters (TL-, SL- initial)");
    println!("      3. Boosts sequences matching trained patterns");
    println!();
    println!("    Integration points:");
    println!("      - N-best list rescoring (demonstrated above)");
    println!("      - Lattice rescoring (for larger search spaces)");
    println!("      - Online beam pruning (during decoding)");
    println!();
    println!("    Weight tuning:");
    println!("      - Higher acoustic weight: Trust neural/acoustic model more");
    println!("      - Higher grammar weight: Stronger phonotactic constraints");
    println!("      - Optimal: Tune on held-out dev set");

    separator('=', 70);
}
