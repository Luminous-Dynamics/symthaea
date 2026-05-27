use std::path::PathBuf;
use symthaea_broca::generator::{BrocaConfig, BrocaGenerator};
use symthaea_broca::tokenizer::BpeTokenizer;
use symthaea_broca::training::{TrainingConfig, TrainingDataset, train};
use symthaea_core::genesis::GenesisSeed;

fn main() {
    let genesis = GenesisSeed::from_phrase("symthaea-clean-core-v1");
    let mut generator = BrocaGenerator::new_4k(&genesis, BrocaConfig::default());

    println!("Loading clean NSM training data...");
    let mut dataset =
        TrainingDataset::from_jsonl("crates/symthaea-broca/data/train-nsm-v1.jsonl").unwrap();
    dataset.tokenize_all(generator.tokenizer());

    let config = TrainingConfig {
        epochs: 5,
        learning_rate: 0.01,
        bptt_window: 16,
        train_network: true,
        use_adam: true,
        progress: true,
        ..Default::default()
    };

    println!(
        "Training fresh core checkpoint for {} epochs...",
        config.epochs
    );
    let metrics = train(&mut generator, &dataset, &config);

    let final_loss = metrics.last().map(|m| m.avg_loss).unwrap_or(0.0);
    let out_path = PathBuf::from("data/models/broca-clean-v1.bin");
    std::fs::create_dir_all(out_path.parent().unwrap()).unwrap();

    generator
        .save_checkpoint(&out_path, metrics.len(), final_loss, None, None, None)
        .unwrap();
    println!(
        "Successfully saved pure checkpoint to {}",
        out_path.display()
    );
}