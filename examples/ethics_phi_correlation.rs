// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Ethics-Φ Correlation Study
//!
//! This example investigates whether network topology Φ correlates with
//! ethical reasoning accuracy on the ETHICS benchmark.
//!
//! ## Hypothesis
//! Higher integrated information (Φ) topologies may correspond to better
//! ethical reasoning, as consciousness is theorized to be necessary for
//! moral cognition.
//!
//! ## Method
//! 1. Load ETHICS benchmark scenarios (justice, deontology, virtue, utilitarianism, commonsense)
//! 2. Encode scenarios using different network topologies (via HDC binding)
//! 3. Measure similarity between encoding and "correct" ethical concept
//! 4. Correlate accuracy with topology Φ values
//!
//! ## Run
//! ```bash
//! cargo run --example ethics_phi_correlation --release
//! ```

use std::collections::HashMap;
use std::path::Path;

use symthaea::hdc::{
    HDC_DIMENSION,
    consciousness_topology_generators::ConsciousnessTopology,
    semantic_encoder::{CharNgramEncoder, MoralSemanticEncoder, SemanticEncoder},
    spectral_connectivity::ConnectivityCalculator,
    unified_hv::ContinuousHV,
};

/// Ethics category from the ETHICS benchmark
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum EthicsCategory {
    Justice,
    Deontology,
    Virtue,
    Utilitarianism,
    Commonsense,
}

impl EthicsCategory {
    fn all() -> Vec<Self> {
        vec![
            Self::Justice,
            Self::Deontology,
            Self::Virtue,
            Self::Utilitarianism,
            Self::Commonsense,
        ]
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Justice => "justice",
            Self::Deontology => "deontology",
            Self::Virtue => "virtue",
            Self::Utilitarianism => "utilitarianism",
            Self::Commonsense => "commonsense",
        }
    }

    fn file_prefix(&self) -> &'static str {
        match self {
            Self::Justice => "justice",
            Self::Deontology => "deontology",
            Self::Virtue => "virtue",
            Self::Utilitarianism => "util",
            Self::Commonsense => "cm",
        }
    }
}

/// A single ethics scenario with label
#[derive(Debug, Clone)]
struct EthicsScenario {
    text: String,
    label: i32, // 1 = ethical, 0 = unethical (or vice versa depending on category)
    #[allow(dead_code)] // Category is stored for potential future analysis
    category: EthicsCategory,
}

/// Load ethics scenarios from CSV file
/// Handles multi-line quoted fields properly
fn load_ethics_csv(category: EthicsCategory, split: &str) -> Result<Vec<EthicsScenario>, String> {
    let base_path = Path::new("datasets/ethics/raw/ethics");
    let filename = match category {
        EthicsCategory::Commonsense => format!("{}_{}.csv", category.file_prefix(), split),
        _ => format!("{}_{}.csv", category.file_prefix(), split),
    };
    let path = base_path.join(category.name()).join(&filename);

    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut scenarios = Vec::new();
    let mut current_line = String::new();
    let mut in_quotes = false;
    let mut is_header = true;

    for char in content.chars() {
        match char {
            '"' => {
                in_quotes = !in_quotes;
                current_line.push(char);
            }
            '\n' if !in_quotes => {
                if is_header {
                    is_header = false;
                    current_line.clear();
                    continue;
                }

                if !current_line.trim().is_empty() {
                    // Try to parse the complete line
                    match parse_csv_line(&current_line, category) {
                        Ok((label, text)) => {
                            scenarios.push(EthicsScenario {
                                text,
                                label,
                                category,
                            });
                        }
                        Err(_e) => {
                            // Skip malformed lines silently (multi-line entries we can't handle)
                        }
                    }
                }
                current_line.clear();
            }
            '\n' if in_quotes => {
                // Keep newlines inside quotes
                current_line.push(' '); // Replace newline with space for cleaner text
            }
            _ => {
                current_line.push(char);
            }
        }
    }

    // Handle last line if not empty
    if !current_line.trim().is_empty() && !is_header {
        if let Ok((label, text)) = parse_csv_line(&current_line, category) {
            scenarios.push(EthicsScenario {
                text,
                label,
                category,
            });
        }
    }

    Ok(scenarios)
}

/// Parse CSV fields, properly handling quoted strings with commas
fn parse_csv_fields(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '"' => {
                // Check for escaped quote ("")
                if in_quotes && chars.peek() == Some(&'"') {
                    chars.next(); // consume the second quote
                    current.push('"');
                } else {
                    in_quotes = !in_quotes;
                }
            }
            ',' if !in_quotes => {
                fields.push(current.trim().to_string());
                current = String::new();
            }
            _ => {
                current.push(c);
            }
        }
    }
    fields.push(current.trim().to_string());
    fields
}

/// Parse a CSV line based on category format
fn parse_csv_line(line: &str, category: EthicsCategory) -> Result<(i32, String), String> {
    let fields = parse_csv_fields(line);

    match category {
        EthicsCategory::Justice | EthicsCategory::Deontology | EthicsCategory::Virtue => {
            // Format: label,scenario
            if fields.len() < 2 {
                return Err(format!("Invalid line format (need 2+ fields): {}", line));
            }
            let label: i32 = fields[0]
                .parse()
                .map_err(|_| format!("Invalid label: '{}'", fields[0]))?;
            let text = fields[1].trim_matches('"').to_string();
            Ok((label, text))
        }
        EthicsCategory::Utilitarianism => {
            // Format: scenario1 ||| scenario2 (comparison)
            // For simplicity, just use first scenario
            if fields.len() < 2 {
                return Err(format!("Invalid line format (need 2+ fields): {}", line));
            }
            let label: i32 = fields[0].parse().unwrap_or(1);
            let text = fields[1].trim_matches('"').to_string();
            Ok((label, text))
        }
        EthicsCategory::Commonsense => {
            // Format: label,input,is_short,edited
            // Text may contain commas, so use proper CSV parsing
            if fields.len() < 2 {
                return Err(format!(
                    "Invalid commonsense format (need 2+ fields): {}",
                    line
                ));
            }
            let label: i32 = fields[0]
                .parse()
                .map_err(|_| format!("Invalid label: '{}'", fields[0]))?;
            let text = fields[1].trim_matches('"').to_string();
            Ok((label, text))
        }
    }
}

/// Encoder enum to switch between methods
#[derive(Clone, Copy)]
enum EncoderMethod {
    CharNgram,
    MoralSemantic,
}

impl EncoderMethod {
    fn name(&self) -> &'static str {
        match self {
            EncoderMethod::CharNgram => "CharNgram",
            EncoderMethod::MoralSemantic => "MoralSemantic",
        }
    }
}

/// Encode text as hypervector using the specified encoding method
fn encode_text_hdc(text: &str, seed: u64, method: EncoderMethod) -> ContinuousHV {
    match method {
        EncoderMethod::CharNgram => {
            // Original character n-gram encoding (baseline)
            let encoder = CharNgramEncoder::new(seed);
            encoder.encode(text)
        }
        EncoderMethod::MoralSemantic => {
            // New moral semantic encoder (captures ethical concepts)
            let encoder = MoralSemanticEncoder::new(seed);
            encoder.encode(text)
        }
    }
}

/// Create concept hypervectors for ethical/unethical classification
fn create_concept_vectors(seed: u64) -> (ContinuousHV, ContinuousHV) {
    let dim = HDC_DIMENSION;

    // Create orthogonal vectors for ethical/unethical concepts
    let ethical = ContinuousHV::random(dim, seed);
    let unethical = ContinuousHV::random(dim, seed + 1000000); // Different seed = quasi-orthogonal

    (ethical, unethical)
}

/// Apply topology-based transformation to encoding
fn apply_topology_transform(hv: &ContinuousHV, topology: &ConsciousnessTopology) -> ContinuousHV {
    // Project input through topology's node representations
    // This simulates how information integrates through the network structure

    let mut result = ContinuousHV::zero(hv.values.len());

    // Weight by similarity to each node, then bundle with node representation
    for node_rep in &topology.node_representations {
        let sim = hv.similarity(node_rep);
        let weighted = node_rep.scale(sim.max(0.0));
        result = result.add(&weighted);
    }

    result.normalize()
}

/// Evaluate ethics accuracy for a topology with a given encoder
fn evaluate_topology(
    topology: &ConsciousnessTopology,
    scenarios: &[EthicsScenario],
    ethical_concept: &ContinuousHV,
    unethical_concept: &ContinuousHV,
    seed: u64,
    method: EncoderMethod,
) -> f32 {
    let mut correct = 0;
    let total = scenarios.len().min(500); // Sample for speed

    for (i, scenario) in scenarios.iter().take(total).enumerate() {
        // Encode scenario text using specified method
        let encoding = encode_text_hdc(&scenario.text, seed + i as u64, method);

        // Apply topology transformation
        let transformed = apply_topology_transform(&encoding, topology);

        // Classify based on similarity to concepts
        let sim_ethical = transformed.similarity(ethical_concept);
        let sim_unethical = transformed.similarity(unethical_concept);

        let predicted = if sim_ethical > sim_unethical { 1 } else { 0 };

        if predicted == scenario.label {
            correct += 1;
        }
    }

    correct as f32 / total as f32
}

/// Main correlation study
fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ETHICS-Φ CORRELATION STUDY                                          ║");
    println!("║           Investigating consciousness-ethics relationship                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // Configuration
    let n_nodes = 8;
    let seed = 42;
    let n_samples = 5;

    // Create Φ calculator
    let phi_calc = ConnectivityCalculator::new();

    // Create concept vectors
    let (ethical_concept, unethical_concept) = create_concept_vectors(seed);

    // Define topologies to test (subset of 19 for speed)
    #[allow(clippy::type_complexity)]
    let topologies: Vec<(&str, Box<dyn Fn() -> ConsciousnessTopology>)> = vec![
        (
            "Hypercube 4D",
            Box::new(|| ConsciousnessTopology::hypercube(4, HDC_DIMENSION, seed)),
        ),
        (
            "Hypercube 3D",
            Box::new(|| ConsciousnessTopology::hypercube(3, HDC_DIMENSION, seed)),
        ),
        (
            "Ring",
            Box::new(|| ConsciousnessTopology::ring(n_nodes, HDC_DIMENSION, seed)),
        ),
        (
            "Torus",
            Box::new(|| ConsciousnessTopology::torus_square(3, HDC_DIMENSION, seed)),
        ),
        (
            "Star",
            Box::new(|| ConsciousnessTopology::star(n_nodes, HDC_DIMENSION, seed)),
        ),
        (
            "Random",
            Box::new(|| ConsciousnessTopology::random(n_nodes, HDC_DIMENSION, seed)),
        ),
        (
            "Dense",
            Box::new(|| ConsciousnessTopology::dense_network(n_nodes, HDC_DIMENSION, None, seed)),
        ),
        (
            "Modular",
            Box::new(|| ConsciousnessTopology::modular(n_nodes, HDC_DIMENSION, 3, seed)),
        ),
    ];

    // Measure Φ for each topology
    println!("📊 Computing Φ for {} topologies...\n", topologies.len());

    let mut topology_phi: Vec<(String, f64, ConsciousnessTopology)> = Vec::new();

    for (name, generator) in &topologies {
        let mut phi_samples = Vec::new();
        let mut topology_instance = None;

        for s in 0..n_samples {
            let topo = generator();
            let phi = phi_calc.algebraic_connectivity(&topo.node_representations);
            phi_samples.push(phi);
            if s == 0 {
                topology_instance = Some(topo);
            }
        }

        let mean_phi: f64 = phi_samples.iter().sum::<f64>() / phi_samples.len() as f64;
        println!("  {} Φ = {:.4}", name, mean_phi);

        if let Some(topo) = topology_instance {
            topology_phi.push((name.to_string(), mean_phi, topo));
        }
    }

    // Sort by Φ
    topology_phi.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n📚 Loading ETHICS benchmark...\n");

    // Load ethics scenarios
    let mut all_scenarios: HashMap<EthicsCategory, Vec<EthicsScenario>> = HashMap::new();

    for category in EthicsCategory::all() {
        match load_ethics_csv(category, "test") {
            Ok(scenarios) => {
                println!(
                    "  ✅ {} - {} scenarios loaded",
                    category.name(),
                    scenarios.len()
                );
                all_scenarios.insert(category, scenarios);
            }
            Err(e) => {
                println!("  ⚠️  {} - {}", category.name(), e);
            }
        }
    }

    if all_scenarios.is_empty() {
        println!("\n❌ No ethics data loaded. Please ensure datasets/ethics/raw/ethics exists.");
        println!("   Download from: https://people.eecs.berkeley.edu/~hendrycks/ethics.tar");
        return;
    }

    // Test both encoding methods
    let encoding_methods = [EncoderMethod::CharNgram, EncoderMethod::MoralSemantic];

    for method in &encoding_methods {
        println!(
            "\n🧠 Evaluating topology-ethics correlation with {} encoder...\n",
            method.name()
        );

        // Results table header
        println!(
            "╔════════════════╦════════╦══════════════════════════════════════════════════════╗"
        );
        println!(
            "║ Topology       ║   Φ    ║  Justice   Deont.   Virtue   Util.   Common  │ Mean ║"
        );
        println!(
            "╠════════════════╬════════╬══════════════════════════════════════════════════════╣"
        );

        let mut results: Vec<(String, f64, Vec<f32>, f32)> = Vec::new();

        for (name, phi, topology) in &topology_phi {
            let mut accuracies = Vec::new();
            let mut acc_strs = Vec::new();

            for category in EthicsCategory::all() {
                if let Some(scenarios) = all_scenarios.get(&category) {
                    let acc = evaluate_topology(
                        topology,
                        scenarios,
                        &ethical_concept,
                        &unethical_concept,
                        seed,
                        *method,
                    );
                    accuracies.push(acc);
                    acc_strs.push(format!("{:6.1}%", acc * 100.0));
                } else {
                    accuracies.push(0.0);
                    acc_strs.push("  N/A  ".to_string());
                }
            }

            let mean_acc: f32 = if !accuracies.is_empty() {
                accuracies.iter().sum::<f32>() / accuracies.len() as f32
            } else {
                0.0
            };

            println!(
                "║ {:14} ║ {:.4} ║ {} {} {} {} {} │{:5.1}%║",
                name,
                phi,
                acc_strs.first().unwrap_or(&"  N/A  ".to_string()),
                acc_strs.get(1).unwrap_or(&"  N/A  ".to_string()),
                acc_strs.get(2).unwrap_or(&"  N/A  ".to_string()),
                acc_strs.get(3).unwrap_or(&"  N/A  ".to_string()),
                acc_strs.get(4).unwrap_or(&"  N/A  ".to_string()),
                mean_acc * 100.0
            );

            results.push((name.clone(), *phi, accuracies, mean_acc));
        }

        println!(
            "╚════════════════╩════════╩══════════════════════════════════════════════════════╝\n"
        );

        // Compute correlation
        if results.len() >= 3 {
            let phi_values: Vec<f32> = results.iter().map(|(_, phi, _, _)| *phi as f32).collect();
            let acc_values: Vec<f32> = results.iter().map(|(_, _, _, mean)| *mean).collect();

            let correlation = pearson_correlation(&phi_values, &acc_values);

            println!("📈 CORRELATION ANALYSIS ({})", method.name());
            println!("═══════════════════════════════════════════════════════════════════");
            println!(
                "  Pearson correlation (Φ vs Ethics Accuracy): r = {:.4}",
                correlation
            );
            println!();

            if correlation > 0.5 {
                println!("  🟢 POSITIVE CORRELATION: Higher Φ → Better ethical reasoning");
                println!("     This supports the hypothesis that integrated information");
                println!("     (consciousness metric) relates to moral cognition.");
            } else if correlation < -0.5 {
                println!("  🔴 NEGATIVE CORRELATION: Higher Φ → Worse ethical reasoning");
                println!("     This contradicts the consciousness-ethics hypothesis.");
            } else {
                println!("  🟡 WEAK CORRELATION: Φ and ethics accuracy are not strongly related");
                if matches!(method, EncoderMethod::CharNgram) {
                    println!(
                        "     The character n-gram method may be too simple to reveal the relationship."
                    );
                } else {
                    println!(
                        "     The correlation may be inherently weak, or require more samples."
                    );
                }
            }

            println!();
            println!("  📊 Statistical Note:");
            println!("     - Encoding method: {}", method.name());
            println!("     - Sample size: {} topologies", results.len());
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("Study complete. Compared CharNgram vs MoralSemantic encoders.");
    println!("MoralSemantic captures ethical vocabulary categories and valence.");
    println!("═══════════════════════════════════════════════════════════════════\n");
}

/// Compute Pearson correlation coefficient
fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x: f32 = x.iter().sum::<f32>() / n;
    let mean_y: f32 = y.iter().sum::<f32>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}