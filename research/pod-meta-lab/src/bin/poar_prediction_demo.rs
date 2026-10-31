use pod_meta_lab::graph::{
    encode_prediction_artifact_v1, generate_poar_graph_suite, DatasetSplit,
    PoarPredictionIdentityV1, ProofPacket,
};
use std::{env, fs, path::Path};

fn split(value: &str) -> Result<DatasetSplit, String> {
    match value {
        "dev" => Ok(DatasetSplit::Development),
        "cal" => Ok(DatasetSplit::Calibration),
        "test" => Ok(DatasetSplit::Test),
        _ => Err("split must be dev, cal, or test".into()),
    }
}

fn run() -> Result<(), String> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 5 {
        return Err(format!("usage: {} <seed> <dev|cal|test> <planner-sha256> <predictions-out>", args[0]));
    }
    let seed = args[1].parse::<u64>().map_err(|e| format!("invalid seed: {e}"))?;
    let split = split(&args[2])?;
    let planner_sha256 = &args[3];
    if planner_sha256.len() != 64
        || !planner_sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("planner digest must be canonical lowercase SHA-256".into());
    }
    let suite = generate_poar_graph_suite(seed, split);
    let packets = suite.planner_cases().iter().map(ProofPacket::abstain_all).collect::<Vec<_>>();
    let identity = PoarPredictionIdentityV1 {
        planner_sha256: planner_sha256.clone(),
        model_id: "all-abstain-smoke-baseline".into(),
        checkpoint_sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
        prompt_sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
        run_config_sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
        seed,
        split,
    };
    let artifact = encode_prediction_artifact_v1(&identity, &packets)?;
    fs::write(Path::new(&args[4]), artifact.as_bytes())
        .map_err(|e| format!("failed to write prediction artifact: {e}"))?;
    println!("prediction_schema=POAR-PRED/1");
    println!("prediction_model=all-abstain-smoke-baseline");
    println!("prediction_cases={}", packets.len());
    println!("prediction_claims={}", packets.iter().map(|packet| packet.claim_dispositions.len()).sum::<usize>());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("poar-prediction-demo: {error}");
        std::process::exit(1);
    }
}
