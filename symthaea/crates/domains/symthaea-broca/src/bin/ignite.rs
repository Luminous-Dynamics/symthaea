use std::error::Error;
use symthaea_broca::generator::{BrocaConfig, BrocaGenerator};
use symthaea_broca::training::{TrainingConfig, TrainingDataset, train_with_adam};
use symthaea_core::genesis::GenesisSeed;

fn main() -> Result<(), Box<dyn Error>> {
    println!("🔥 Igniting Symthaea Brain Substrate Matrix...");

    let genesis = GenesisSeed::from_phrase("symthaea-fused-node-2026");
    let mut generator = BrocaGenerator::new_4k(&genesis, BrocaConfig::default());

    println!("Loading epistemic JSONL training pairs...");
    let dataset = TrainingDataset::from_jsonl("data/train-epistemic-v1.jsonl")?;

    let config = TrainingConfig {
        epochs: 1,
        learning_rate: 0.01,
        bptt_window: 16,
        use_adam: true,
        progress: true,
        train_network: true,
        network_lr_scale: 0.3,
        embedding_target_norm: 128.0,
        ..Default::default()
    };

    println!("Executing teacher-forced BPTT weight optimization phase...");
    let (_metrics, _, _, _, _) = train_with_adam(&mut generator, &dataset, &config, None);

    println!("Matrix weight ignition complete. Checkpoint locked to disk.");
    Ok(())
}