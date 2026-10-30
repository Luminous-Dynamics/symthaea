//! CMAQ-008 residual-path ablation using the real CfC world-model implementation.
//!
//! Compares the unchanged absolute-next-state objective against an identity-preserving
//! residual objective: train the CfC output on delta-state and add the current state
//! through an explicit persistence skip. This is a synthetic diagnostic, not a SOTA claim.

use std::process;
use symthaea::dynamics::crate_world_model::{HierarchicalCfCWorldModel, WorldModelConfig};

const REPORT_SCHEMA: &str = "cmaq-008-residual-ablation-v0.1";
const DIM: usize = 16;
const HIDDEN: usize = 24;
const TRAIN_SAMPLES: usize = 128;
const EVAL_SAMPLES: usize = 64;
const EPOCHS: usize = 12;
const LEARNING_RATE: f32 = 0.001;
const DATA_SEED: u64 = 0xC0A8_2026_1029;
const MODEL_SEEDS: [u64; 5] = [
    0x5EED_0001,
    0x5EED_0002,
    0x5EED_0003,
    0x5EED_0004,
    0x5EED_0005,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1_u32 << 24) as f32)
    }

    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Law {
    Source,
    ActionEffectReversal,
}

impl Law {
    const ALL: [Self; 2] = [Self::Source, Self::ActionEffectReversal];

    fn name(self) -> &'static str {
        match self {
            Self::Source => "source_holdout",
            Self::ActionEffectReversal => "action_effect_reversal",
        }
    }

    fn action_coefficient(self) -> f32 {
        match self {
            Self::Source => 0.25,
            Self::ActionEffectReversal => -0.25,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Experience {
    state: Vec<f32>,
    action: Vec<f32>,
    next_state: Vec<f32>,
}

fn transition(state: &[f32], action: &[f32], law: Law) -> Vec<f32> {
    (0..DIM)
        .map(|i| {
            let neighbour = state[(i + 1) % DIM];
            0.55 * state[i]
                + law.action_coefficient() * action[i]
                + 0.06 * neighbour * action[i]
        })
        .collect()
}

fn dataset(seed: u64, count: usize, law: Law) -> Vec<Experience> {
    let mut rng = SplitMix64::new(seed);
    (0..count)
        .map(|_| {
            let state = (0..DIM).map(|_| rng.signed()).collect::<Vec<_>>();
            let action = (0..DIM).map(|_| rng.signed()).collect::<Vec<_>>();
            let next_state = transition(&state, &action, law);
            Experience {
                state,
                action,
                next_state,
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Objective {
    AbsoluteNextState,
    ResidualDelta,
}

impl Objective {
    const ALL: [Self; 2] = [Self::AbsoluteNextState, Self::ResidualDelta];

    fn name(self) -> &'static str {
        match self {
            Self::AbsoluteNextState => "absolute_next_state",
            Self::ResidualDelta => "identity_residual_delta",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Metrics {
    mse: f64,
    mae: f64,
    persistence_mse: f64,
    zero_mse: f64,
    sample_count: usize,
}

impl Metrics {
    fn evaluate(
        model: &HierarchicalCfCWorldModel,
        data: &[Experience],
        objective: Objective,
    ) -> Result<Self, String> {
        let mut squared_error = 0.0_f64;
        let mut absolute_error = 0.0_f64;
        let mut persistence_squared_error = 0.0_f64;
        let mut zero_squared_error = 0.0_f64;
        let mut count = 0_usize;

        for row in data {
            let prediction = match objective {
                Objective::AbsoluteNextState => model.predict(&row.state, &row.action, 1.0),
                Objective::ResidualDelta => {
                    model.predict_next_state_residual(&row.state, &row.action, 1.0)
                }
            }
            .map_err(|error| format!("world-model prediction failed: {error}"))?;

            if prediction.len() != row.next_state.len() {
                return Err(format!(
                    "prediction dimension mismatch: got {}, expected {}",
                    prediction.len(),
                    row.next_state.len()
                ));
            }
            for i in 0..DIM {
                let error = prediction[i] as f64 - row.next_state[i] as f64;
                let persistence_error = row.state[i] as f64 - row.next_state[i] as f64;
                let zero_error = row.next_state[i] as f64;
                squared_error += error * error;
                absolute_error += error.abs();
                persistence_squared_error += persistence_error * persistence_error;
                zero_squared_error += zero_error * zero_error;
                count += 1;
            }
        }

        if count == 0 {
            return Err("evaluation dataset must not be empty".to_owned());
        }
        let denominator = count as f64;
        Ok(Self {
            mse: squared_error / denominator,
            mae: absolute_error / denominator,
            persistence_mse: persistence_squared_error / denominator,
            zero_mse: zero_squared_error / denominator,
            sample_count: data.len(),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct SeedResult {
    seed: u64,
    final_epoch_train_mse: f64,
    training_steps: u64,
    regimes: [Metrics; 2],
}

fn run_seed(seed: u64, objective: Objective) -> Result<SeedResult, String> {
    let training = dataset(DATA_SEED, TRAIN_SAMPLES, Law::Source);
    let evaluations = [
        dataset(DATA_SEED ^ 0x1111_2222_3333_4444, EVAL_SAMPLES, Law::Source),
        dataset(
            DATA_SEED ^ 0xAAAA_BBBB_CCCC_DDDD,
            EVAL_SAMPLES,
            Law::ActionEffectReversal,
        ),
    ];

    let config = WorldModelConfig {
        hdc_dim: DIM,
        hidden_dim: HIDDEN,
        learning_rate: LEARNING_RATE,
        momentum: 0.9,
        consciousness_decay: 0.99,
    };
    let mut model = HierarchicalCfCWorldModel::new_with_seed(config, seed)
        .map_err(|error| format!("seeded model construction failed: {error}"))?;

    let mut final_epoch_loss = 0.0_f64;
    for _ in 0..EPOCHS {
        final_epoch_loss = 0.0;
        for row in &training {
            let loss = match objective {
                Objective::AbsoluteNextState => {
                    model.train_step(&row.state, &row.action, &row.next_state, 1.0)
                }
                Objective::ResidualDelta => {
                    model.train_step_residual(&row.state, &row.action, &row.next_state, 1.0)
                }
            }
            .map_err(|error| format!("world-model training failed: {error}"))?;
            final_epoch_loss += loss as f64;
        }
        final_epoch_loss /= TRAIN_SAMPLES as f64;
    }

    let mut regimes = [Metrics::default(); 2];
    for (index, law) in Law::ALL.into_iter().enumerate() {
        regimes[index] = Metrics::evaluate(&model, &evaluations[index], objective)
            .map_err(|error| format!("{} evaluation failed: {error}", law.name()))?;
    }
    Ok(SeedResult {
        seed,
        final_epoch_train_mse: final_epoch_loss,
        training_steps: model.training_steps,
        regimes,
    })
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn mean_ci95(values: &[f64]) -> (f64, f64, f64) {
    let average = mean(values);
    if values.len() < 2 {
        return (average, average, average);
    }
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let half_width = 1.96 * (variance / values.len() as f64).sqrt();
    (average, average - half_width, average + half_width)
}

#[derive(Clone, Debug, PartialEq)]
struct Aggregate {
    mse: f64,
    mae: f64,
    persistence_mse: f64,
    zero_mse: f64,
    mse_to_persistence_ratio: f64,
    mse_delta_vs_persistence: f64,
    mse_delta_vs_persistence_ci95_low: f64,
    mse_delta_vs_persistence_ci95_high: f64,
}

fn aggregate(results: &[SeedResult], law_index: usize) -> Aggregate {
    let rows = results.iter().map(|row| row.regimes[law_index]).collect::<Vec<_>>();
    let mses = rows.iter().map(|row| row.mse).collect::<Vec<_>>();
    let persistence = rows.iter().map(|row| row.persistence_mse).collect::<Vec<_>>();
    let (delta, low, high) = mean_ci95(
        &rows
            .iter()
            .map(|row| row.mse - row.persistence_mse)
            .collect::<Vec<_>>(),
    );
    let mse = mean(&mses);
    let persistence_mse = mean(&persistence);
    Aggregate {
        mse,
        mae: mean(&rows.iter().map(|row| row.mae).collect::<Vec<_>>()),
        persistence_mse,
        zero_mse: mean(&rows.iter().map(|row| row.zero_mse).collect::<Vec<_>>()),
        mse_to_persistence_ratio: mse / persistence_mse.max(f64::MIN_POSITIVE),
        mse_delta_vs_persistence: delta,
        mse_delta_vs_persistence_ci95_low: low,
        mse_delta_vs_persistence_ci95_high: high,
    }
}

#[derive(Clone, Debug)]
struct ModeResults {
    objective: Objective,
    seeds: Vec<SeedResult>,
}

fn run() -> Result<Vec<ModeResults>, String> {
    let mut reports = Vec::with_capacity(Objective::ALL.len());
    for objective in Objective::ALL {
        let results = MODEL_SEEDS
            .iter()
            .map(|seed| run_seed(*seed, objective))
            .collect::<Result<Vec<_>, _>>()?;
        if results.iter().any(|row| {
            !row.final_epoch_train_mse.is_finite()
                || row.training_steps != (TRAIN_SAMPLES * EPOCHS) as u64
                || row.regimes.iter().any(|metric| {
                    !metric.mse.is_finite()
                        || !metric.mae.is_finite()
                        || !metric.persistence_mse.is_finite()
                        || metric.mse < 0.0
                })
        }) {
            return Err(format!(
                "non-finite metrics or unexpected steps for {}",
                objective.name()
            ));
        }
        reports.push(ModeResults {
            objective,
            seeds: results,
        });
    }
    Ok(reports)
}

fn render(reports: &[ModeResults]) -> String {
    let mut out = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"{}\",\n",
            "  \"model\": \"HierarchicalCfCWorldModel\",\n",
            "  \"dimensions\": {{\"hdc\": {}, \"hidden\": {}}},\n",
            "  \"training\": {{\"samples\": {}, \"epochs\": {}, \"seeds\": {}, \"learning_rate\": {:.6}}},\n",
            "  \"evaluation_samples_per_regime_per_seed\": {},\n",
            "  \"regimes\": [\"source_holdout\", \"action_effect_reversal\"],\n",
            "  \"objectives\": [\n"
        ),
        REPORT_SCHEMA, DIM, HIDDEN, TRAIN_SAMPLES, EPOCHS, MODEL_SEEDS.len(), LEARNING_RATE, EVAL_SAMPLES
    );

    for (mode_index, mode) in reports.iter().enumerate() {
        let source = aggregate(&mode.seeds, 0);
        let reversal = aggregate(&mode.seeds, 1);
        out.push_str(&format!(
            concat!(
                "    {{\"objective\":\"{}\",",
                "\"training_steps_per_seed\":{},",
                "\"source\":{{\"mse\":{:.8},\"mae\":{:.8},\"persistence_mse\":{:.8},\"zero_mse\":{:.8},",
                "\"mse_to_persistence_ratio\":{:.8},\"mse_delta_vs_persistence\":{:.8},",
                "\"mse_delta_vs_persistence_ci95_low\":{:.8},\"mse_delta_vs_persistence_ci95_high\":{:.8}}},",
                "\"action_effect_reversal\":{{\"mse\":{:.8},\"mae\":{:.8},\"persistence_mse\":{:.8},\"zero_mse\":{:.8},",
                "\"mse_to_persistence_ratio\":{:.8},\"mse_delta_vs_persistence\":{:.8},",
                "\"mse_delta_vs_persistence_ci95_low\":{:.8},\"mse_delta_vs_persistence_ci95_high\":{:.8}}},",
                "\"per_seed\":["
            ),
            mode.objective.name(),
            (TRAIN_SAMPLES * EPOCHS),
            source.mse, source.mae, source.persistence_mse, source.zero_mse,
            source.mse_to_persistence_ratio, source.mse_delta_vs_persistence,
            source.mse_delta_vs_persistence_ci95_low, source.mse_delta_vs_persistence_ci95_high,
            reversal.mse, reversal.mae, reversal.persistence_mse, reversal.zero_mse,
            reversal.mse_to_persistence_ratio, reversal.mse_delta_vs_persistence,
            reversal.mse_delta_vs_persistence_ci95_low, reversal.mse_delta_vs_persistence_ci95_high
        ));
        for (seed_index, row) in mode.seeds.iter().enumerate() {
            out.push_str(&format!(
                "{{\"seed\":\"{:016x}\",\"final_epoch_train_mse\":{:.8},\"source_mse\":{:.8},\"reversal_mse\":{:.8}}}",
                row.seed, row.final_epoch_train_mse, row.regimes[0].mse, row.regimes[1].mse
            ));
            if seed_index + 1 != mode.seeds.len() {
                out.push(',');
            }
        }
        out.push_str("]}");
        if mode_index + 1 != reports.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n}\n");
    out
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("CMAQ-008 residual-path ablation: real CfC model; compare absolute next-state learning with identity-preserving residual delta learning.");
        return;
    }
    if !args.is_empty() {
        eprintln!("unexpected argument {}; use --help", args[0]);
        process::exit(2);
    }
    match run() {
        Ok(reports) => print!("{}", render(&reports)),
        Err(error) => {
            eprintln!("{error}");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_tape_is_paired_across_learning_objectives() {
        for law in Law::ALL {
            assert_eq!(
                dataset(DATA_SEED, 12, law),
                dataset(DATA_SEED, 12, law)
            );
        }
    }

    #[test]
    fn both_objectives_replay_same_seed() {
        for objective in Objective::ALL {
            let first = run_seed(MODEL_SEEDS[0], objective).unwrap();
            let second = run_seed(MODEL_SEEDS[0], objective).unwrap();
            assert_eq!(first, second);
        }
    }

    #[test]
    fn report_contains_paired_objectives_and_finite_metrics() {
        let reports = run().unwrap();
        assert_eq!(reports.len(), 2);
        let output = render(&reports);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(
            parsed["schema_version"],
            "cmaq-008-residual-ablation-v0.1"
        );
        assert_eq!(parsed["objectives"].as_array().unwrap().len(), 2);
        for objective in parsed["objectives"].as_array().unwrap() {
            for regime in ["source", "action_effect_reversal"] {
                for metric in [
                    "mse",
                    "mae",
                    "persistence_mse",
                    "zero_mse",
                    "mse_to_persistence_ratio",
                    "mse_delta_vs_persistence",
                    "mse_delta_vs_persistence_ci95_low",
                    "mse_delta_vs_persistence_ci95_high",
                ] {
                    assert!(objective[regime][metric].is_finite());
                }
            }
        }
    }
}
