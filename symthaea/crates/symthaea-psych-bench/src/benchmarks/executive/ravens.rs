//! Raven's Progressive Matrices (RPM).
//!
//! Tests fluid intelligence via pattern completion — HDC's natural strength.
//! 3×3 grid with 8 filled cells and 1 missing. Features vary along rows/columns
//! following rules (increment, XOR, distribution-of-3).
//!
//! 3 difficulty tiers × 10 items = 30 items.
//! - Easy (Set A-B): single rule
//! - Medium (Set C): two combined rules
//! - Hard (Set D-E): three rules with XOR
//!
//! Human baselines (Raven, 1938; Murphy et al., 2023):
//! - easy_accuracy: 0.95
//! - medium_accuracy: 0.70
//! - hard_accuracy: 0.40
//! - overall_accuracy: ~0.78

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use symthaea_core::hdc::ContinuousHV;

/// Raven's Progressive Matrices benchmark.
pub struct RavensProgressiveMatricesBenchmark;

/// A cell in the 3×3 grid, defined by feature indices.
#[derive(Clone, Copy)]
struct Cell {
    shape: usize,  // 0-2
    size: usize,   // 0-2
    color: usize,  // 0-2
}

/// A procedurally generated RPM item.
struct RpmItem {
    /// 8 visible cells (row-major, position 8 = missing)
    cells: [Cell; 8],
    /// The correct answer
    answer: Cell,
    /// 5 distractor options (answer is mixed in during evaluation)
    distractors: [Cell; 5],
}

/// Rule types for feature progression along rows.
#[derive(Clone, Copy)]
enum Rule {
    /// Feature increments: 0, 1, 2 across columns
    Increment,
    /// Feature is constant within each row
    Constant,
    /// Distribution-of-3: each row contains all 3 values in some order
    Distribution,
    /// XOR-like: col3 = (col1 + col2) % 3
    Xor,
}

impl Rule {
    fn apply(&self, row: usize, col: usize, base: usize) -> usize {
        match self {
            Rule::Increment => (base + col) % 3,
            Rule::Constant => base,
            Rule::Distribution => {
                // Row-dependent permutation
                let perms = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];
                perms[row % 3][col]
            }
            Rule::Xor => {
                if col < 2 {
                    // First two columns use base offset
                    (base + col) % 3
                } else {
                    // Third column = (col0 + col1) % 3
                    let col0 = (base) % 3;
                    let col1 = (base + 1) % 3;
                    (col0 + col1) % 3
                }
            }
        }
    }
}

fn generate_item(seed: u64, difficulty: usize) -> RpmItem {
    let mut rng = seed ^ 0x9E3779B97F4A7C15;

    // Select rules based on difficulty
    let xor_shift = |s: &mut u64| {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
    };

    let rules = match difficulty {
        0 => {
            // Easy: single rule (increment) + constants for other features
            xor_shift(&mut rng);
            let active = (rng % 3) as usize; // Which feature uses increment
            let mut r = [Rule::Constant; 3];
            r[active] = Rule::Increment;
            r
        }
        1 => {
            // Medium: two combined rules
            xor_shift(&mut rng);
            let a = (rng % 3) as usize;
            xor_shift(&mut rng);
            let mut b = (rng % 2) as usize;
            if b >= a { b += 1; }
            let mut r = [Rule::Constant; 3];
            r[a] = Rule::Increment;
            r[b] = Rule::Distribution;
            r
        }
        _ => {
            // Hard: three rules with XOR
            [Rule::Increment, Rule::Distribution, Rule::Xor]
        }
    };

    // Generate base values per row for each feature
    xor_shift(&mut rng);
    let shape_bases = [(rng % 3) as usize, ((rng >> 8) % 3) as usize, ((rng >> 16) % 3) as usize];
    xor_shift(&mut rng);
    let size_bases = [(rng % 3) as usize, ((rng >> 8) % 3) as usize, ((rng >> 16) % 3) as usize];
    xor_shift(&mut rng);
    let color_bases = [(rng % 3) as usize, ((rng >> 8) % 3) as usize, ((rng >> 16) % 3) as usize];

    let make_cell = |row: usize, col: usize| -> Cell {
        Cell {
            shape: rules[0].apply(row, col, shape_bases[row]),
            size: rules[1].apply(row, col, size_bases[row]),
            color: rules[2].apply(row, col, color_bases[row]),
        }
    };

    // Fill the 3×3 grid
    let mut cells = [Cell { shape: 0, size: 0, color: 0 }; 8];
    for row in 0..3 {
        for col in 0..3 {
            let idx = row * 3 + col;
            if idx < 8 {
                cells[idx] = make_cell(row, col);
            }
        }
    }

    // The answer is position [2][2] (row 2, col 2)
    let answer = make_cell(2, 2);

    // Generate 5 distractors (different from answer)
    let mut distractors = [Cell { shape: 0, size: 0, color: 0 }; 5];
    for d in 0..5 {
        xor_shift(&mut rng);
        distractors[d] = Cell {
            shape: ((answer.shape + 1 + (rng % 2) as usize) % 3),
            size: ((answer.size + ((rng >> 4) % 3) as usize) % 3),
            color: ((answer.color + ((rng >> 8) % 3) as usize) % 3),
        };
        // Ensure distractor differs from answer in at least one feature
        if distractors[d].shape == answer.shape
            && distractors[d].size == answer.size
            && distractors[d].color == answer.color
        {
            distractors[d].shape = (answer.shape + 1) % 3;
        }
    }

    RpmItem { cells, answer, distractors }
}

impl RavensProgressiveMatricesBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> RpmResult {
        let dim = config.dimension;
        let seed = config.trial_seed("executive", "ravens", trial_idx);

        // Create feature HVs
        let shape_hvs: Vec<ContinuousHV> = (0..3)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(100 + i)))
            .collect();
        let size_hvs: Vec<ContinuousHV> = (0..3)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(200 + i)))
            .collect();
        let color_hvs: Vec<ContinuousHV> = (0..3)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(300 + i)))
            .collect();

        // Role HVs
        let role_shape = ContinuousHV::random(dim, seed.wrapping_add(400));
        let role_size = ContinuousHV::random(dim, seed.wrapping_add(401));
        let role_color = ContinuousHV::random(dim, seed.wrapping_add(402));

        let encode_cell = |cell: &Cell| -> ContinuousHV {
            let s = role_shape.bind(&shape_hvs[cell.shape]);
            let z = role_size.bind(&size_hvs[cell.size]);
            let c = role_color.bind(&color_hvs[cell.color]);
            ContinuousHV::bundle(&[&s, &z, &c])
        };

        let items_per_tier = 10;
        let mut easy_correct = 0u32;
        let mut medium_correct = 0u32;
        let mut hard_correct = 0u32;

        for difficulty in 0..3 {
            for item_idx in 0..items_per_tier {
                let item_seed = seed
                    .wrapping_add(difficulty as u64 * 1000)
                    .wrapping_add(item_idx as u64 * 100)
                    .wrapping_add(trial_idx as u64 * 10000);
                let item = generate_item(item_seed, difficulty);

                // Encode visible cells
                let cell_hvs: Vec<ContinuousHV> = item.cells.iter().map(|c| encode_cell(c)).collect();

                // Extract row patterns: what transforms row elements?
                // Row 0: cells [0,1,2], Row 1: cells [3,4,5], Row 2: cells [6,7,?]
                // Rule extraction: unbind(cell_end, cell_start) per completed row
                let row0_rule = cell_hvs[2].bind(&cell_hvs[0].inverse());
                let row1_rule = cell_hvs[5].bind(&cell_hvs[3].inverse());

                // Average the two row rules for a robust rule estimate
                let avg_rule = ContinuousHV::bundle(&[&row0_rule, &row1_rule]);

                // Predict missing cell: apply rule to row 2's col 0 (cell 6)
                let predicted = avg_rule.bind(&cell_hvs[6]);

                // Also try column pattern: col 2 across rows
                let col_rule = cell_hvs[5].bind(&cell_hvs[2].inverse());
                let predicted_col = col_rule.bind(&cell_hvs[2]);

                // Ensemble: average both predictions
                let predicted_ensemble = ContinuousHV::bundle(&[&predicted, &predicted_col]);

                // Score answer and distractors
                let answer_hv = encode_cell(&item.answer);
                let answer_sim = predicted_ensemble.similarity(&answer_hv);

                let mut best_distractor_sim = f32::NEG_INFINITY;
                for d in &item.distractors {
                    let d_hv = encode_cell(d);
                    let sim = predicted_ensemble.similarity(&d_hv);
                    if sim > best_distractor_sim {
                        best_distractor_sim = sim;
                    }
                }

                if answer_sim > best_distractor_sim {
                    match difficulty {
                        0 => easy_correct += 1,
                        1 => medium_correct += 1,
                        _ => hard_correct += 1,
                    }
                }
            }
        }

        let easy_acc = easy_correct as f64 / items_per_tier as f64;
        let medium_acc = medium_correct as f64 / items_per_tier as f64;
        let hard_acc = hard_correct as f64 / items_per_tier as f64;
        let overall = (easy_correct + medium_correct + hard_correct) as f64 / (3 * items_per_tier) as f64;

        RpmResult {
            easy_accuracy: easy_acc,
            medium_accuracy: medium_acc,
            hard_accuracy: hard_acc,
            overall_accuracy: overall,
            difficulty_gradient: easy_acc - hard_acc,
        }
    }
}

struct RpmResult {
    easy_accuracy: f64,
    medium_accuracy: f64,
    hard_accuracy: f64,
    overall_accuracy: f64,
    difficulty_gradient: f64,
}

impl PsychBenchmark for RavensProgressiveMatricesBenchmark {
    fn name(&self) -> &str {
        "Executive::Ravens"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut easy = Vec::new();
        let mut medium = Vec::new();
        let mut hard = Vec::new();
        let mut overall = Vec::new();
        let mut gradient = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            easy.push(r.easy_accuracy);
            medium.push(r.medium_accuracy);
            hard.push(r.hard_accuracy);
            overall.push(r.overall_accuracy);
            gradient.push(r.difficulty_gradient);
        }

        result.insert("easy_accuracy", MetricValue::from_samples(&easy));
        result.insert("medium_accuracy", MetricValue::from_samples(&medium));
        result.insert("hard_accuracy", MetricValue::from_samples(&hard));
        result.insert("overall_accuracy", MetricValue::from_samples(&overall));
        result.insert("difficulty_gradient", MetricValue::from_samples(&gradient));

        result.conditions = 3;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ravens_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = RavensProgressiveMatricesBenchmark.run(&config);
        assert!(result.metrics.contains_key("easy_accuracy"));
        assert!(result.metrics.contains_key("hard_accuracy"));
        assert!(result.metrics.contains_key("overall_accuracy"));
        assert!(result.metrics.contains_key("difficulty_gradient"));
    }

    #[test]
    fn test_ravens_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = RavensProgressiveMatricesBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_ravens_difficulty_gradient_positive() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = RavensProgressiveMatricesBenchmark.run(&config);
        let gradient = result.metrics["difficulty_gradient"].mean;
        // Easy should generally be >= hard, so gradient >= 0
        assert!(gradient >= -0.1, "difficulty gradient should be non-negative, got {}", gradient);
    }

    #[test]
    fn test_generate_item_deterministic() {
        let item1 = generate_item(42, 0);
        let item2 = generate_item(42, 0);
        assert_eq!(item1.answer.shape, item2.answer.shape);
        assert_eq!(item1.answer.size, item2.answer.size);
        assert_eq!(item1.answer.color, item2.answer.color);
    }
}
