use contamination_panel_bench::{
    ADJUDICATION_REPORT_SCHEMA, AdjudicationRecordV1, ChallengeManifestV1, DetectorRunV1,
    causal_analysis::{ReplicatedCausalStudyV1, analyze_replicated_causal_study},
    exposure_transfer::{analyze_exposure_transfer, ExposureTransferStudyV1},
    DetectorSpecV1, PublicChallengePacketV1, SealedChallengeKeyV1,
    audit_blinded_adjudications, detectors::{evaluate_detector_run, score_public_packet},
    ftd::{
        empirical_reference_p_values, ftd_weighted_cauchy_bh, FeaturePValuesV1,
        FtdWeightPolicyV1, ReferenceTailV1, ScoreObservationV1,
    },
    generate_challenge_bundle, GeneratorConfigV1,
    intervention::{
        RandomizedInterventionDesignV1, RandomizedInterventionManifestV1,
        audit_intervention, randomize_intervention,
    },
    training_receipt::validate_paired_training_execution_receipt,
};
use std::{
    env,
    error::Error,
    fs,
    io::{self, BufRead, Read},
    path::Path,
    process,
};

fn decode_hex(input: &str) -> Result<Vec<u8>, String> {
    if input.len() % 2 != 0 {
        return Err("seed hex must contain an even number of characters".to_owned());
    }
    let mut output = Vec::with_capacity(input.len() / 2);
    for pair in input.as_bytes().chunks_exact(2) {
        let high = hex_nibble(pair[0])?;
        let low = hex_nibble(pair[1])?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> Result<u8, String> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err("seed_hex contains a non-hexadecimal character".to_owned()),
    }
}

fn write_bytes(path: &str, contents: &[u8]) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, contents)?;
    Ok(())
}

fn generate(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 7 {
        return Err(
            "usage: contamination-panel-bench generate <generation-id> <pairs-per-tier> <seed-hex> <public-packet.json> <sealed-key.json> <manifest.json>"
                .into(),
        );
    }
    let pairs_per_tier = args[2].parse::<u32>()?;
    let seed = decode_hex(&args[3]).map_err(io::Error::other)?;
    let config = GeneratorConfigV1 {
        generation_id: args[1].clone(),
        pairs_per_tier,
    };
    let bundle = generate_challenge_bundle(&config, &seed).map_err(io::Error::other)?;

    write_bytes(&args[4], &bundle.public_packet_bytes)?;
    write_bytes(&args[5], &bundle.sealed_key_bytes)?;
    write_bytes(&args[6], &bundle.manifest_bytes)?;

    println!("public_packet_sha256={}", bundle.public_packet_sha256);
    println!("sealed_key_sha256={}", bundle.sealed_key_sha256);
    println!("seed_commitment_sha256={}", bundle.seed_commitment_sha256);
    println!("WARNING: keep the sealed key and seed outside reviewer/model access.");
    Ok(())
}

fn score_builtins(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 4 {
        return Err(
            "usage: contamination-panel-bench score <public-packet.json> <detector-spec.json> <detector-run.json>"
                .into(),
        );
    }
    let packet: PublicChallengePacketV1 = serde_json::from_slice(&fs::read(&args[1])?)?;
    let spec: DetectorSpecV1 = serde_json::from_slice(&fs::read(&args[2])?)?;
    let run = score_public_packet(&packet, &spec).map_err(io::Error::other)?;
    let bytes = serde_json::to_vec_pretty(&run)?;
    write_bytes(&args[3], &bytes)?;
    println!(
        "detector_id={} pairs_scored={} public_packet_sha256={}",
        run.detector_id,
        run.scores.len(),
        run.public_packet_sha256
    );
    Ok(())
}

fn evaluate_run(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 6 {
        return Err(
            "usage: contamination-panel-bench evaluate <public-packet.json> <sealed-key.json> <manifest.json> <detector-run.json> <evaluation-report.json>"
                .into(),
        );
    }
    let packet: PublicChallengePacketV1 = serde_json::from_slice(&fs::read(&args[1])?)?;
    let key: SealedChallengeKeyV1 = serde_json::from_slice(&fs::read(&args[2])?)?;
    let manifest: ChallengeManifestV1 = serde_json::from_slice(&fs::read(&args[3])?)?;
    let run: DetectorRunV1 = serde_json::from_slice(&fs::read(&args[4])?)?;
    let report = evaluate_detector_run(&packet, &key, &manifest, &run).map_err(io::Error::other)?;
    let bytes = serde_json::to_vec_pretty(&report)?;
    write_bytes(&args[5], &bytes)?;
    println!(
        "detector_id={} tiers={} pairs={} report_schema={}",
        report.detector_id,
        report.metrics_by_tier.len(),
        report.pair_count,
        report.schema
    );
    Ok(())
}

fn ftd_pvalues(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 7 {
        return Err(
            "usage: contamination-panel-bench ftd-pvalues <feature-id> <greater|less> <reference-scores.json> <test-scores.json> <seed-hex> <feature-pvalues.json>"
                .into(),
        );
    }
    let direction = match args[2].as_str() {
        "greater" => ReferenceTailV1::Greater,
        "less" => ReferenceTailV1::Less,
        _ => return Err("tail direction must be greater or less".into()),
    };
    let reference_scores: Vec<f64> = serde_json::from_slice(&fs::read(&args[3])?)?;
    let test_scores: Vec<ScoreObservationV1> = serde_json::from_slice(&fs::read(&args[4])?)?;
    let seed = decode_hex(&args[5]).map_err(io::Error::other)?;
    let p_values = empirical_reference_p_values(
        &args[1],
        direction,
        &reference_scores,
        &test_scores,
        &seed,
    )
    .map_err(io::Error::other)?;
    let bytes = serde_json::to_vec_pretty(&p_values)?;
    write_bytes(&args[6], &bytes)?;
    println!(
        "feature_id={} direction={:?} reference_count={} test_count={}",
        p_values.feature_id,
        p_values.direction,
        p_values.reference_count,
        p_values.p_values.len()
    );
    Ok(())
}

fn ftd_fuse(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 6 {
        return Err(
            "usage: contamination-panel-bench ftd-fuse <feature-pvalues.json> <alpha> <paper|official> <temperature-milli|none> <report.json>"
                .into(),
        );
    }
    let feature_p_values: Vec<FeaturePValuesV1> =
        serde_json::from_slice(&fs::read(&args[1])?)?;
    let alpha = args[2].parse::<f64>()?;
    let (weight_policy, temperature) = match args[3].as_str() {
        "paper" if args[4] == "none" => {
            (FtdWeightPolicyV1::PaperNormalizedRejectionCounts, None)
        }
        "official" if args[4] != "none" => (
            FtdWeightPolicyV1::OfficialRepositorySoftmaxSharpened,
            Some(args[4].parse::<u16>()?),
        ),
        "paper" => return Err("paper policy requires temperature=none".into()),
        "official" => return Err("official policy requires an integer temperature in thousandths".into()),
        _ => return Err("weight policy must be paper or official".into()),
    };
    let report = ftd_weighted_cauchy_bh(
        &feature_p_values,
        alpha,
        weight_policy,
        temperature,
    )
    .map_err(io::Error::other)?;
    let bytes = serde_json::to_vec_pretty(&report)?;
    write_bytes(&args[5], &bytes)?;
    println!(
        "hypotheses={} selected_clean={} features={} policy={:?}",
        report.hypothesis_count,
        report.retained_as_clean_count,
        report.feature_count,
        report.weight_policy
    );
    println!("FDR guarantee status: {}", report.nominal_fdr_guarantee);
    Ok(())
}

fn randomize_intervention_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 5 {
        return Err(
            "usage: contamination-panel-bench randomize-intervention <design.json> <seed-hex> <manifest.json> <audit-report.json>"
                .into(),
        );
    }
    let design: RandomizedInterventionDesignV1 = serde_json::from_slice(&fs::read(&args[1])?)?;
    let seed = decode_hex(&args[2]).map_err(io::Error::other)?;
    let manifest = randomize_intervention(&design, &seed).map_err(io::Error::other)?;
    let audit = audit_intervention(&design, &manifest, &seed).map_err(io::Error::other)?;
    write_bytes(&args[3], &serde_json::to_vec_pretty(&manifest)?)?;
    write_bytes(&args[4], &serde_json::to_vec_pretty(&audit)?)?;
    println!(
        "experiment={} replicate={} exposed_clusters={}/{} exposed_items={} matched_tokens={} audit_valid={}",
        manifest.experiment_id,
        manifest.replicate_id,
        manifest.exposed_cluster_count,
        manifest.total_cluster_count,
        manifest.exposed_item_count,
        manifest.matched_placebo_token_count,
        audit.valid
    );
    println!("claim boundary: {}", manifest.claim_boundary);
    Ok(())
}

fn audit_intervention_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 5 {
        return Err(
            "usage: contamination-panel-bench audit-intervention <design.json> <manifest.json> <seed-hex> <audit-report.json>"
                .into(),
        );
    }
    let design: RandomizedInterventionDesignV1 = serde_json::from_slice(&fs::read(&args[1])?)?;
    let manifest: RandomizedInterventionManifestV1 =
        serde_json::from_slice(&fs::read(&args[2])?)?;
    let seed = decode_hex(&args[3]).map_err(io::Error::other)?;
    let report = audit_intervention(&design, &manifest, &seed).map_err(io::Error::other)?;
    write_bytes(&args[4], &serde_json::to_vec_pretty(&report)?)?;
    println!(
        "valid={} replayed={} token_budget_match={} source_cluster_integrity={}",
        report.valid,
        report.randomization_replayed,
        report.token_budget_match,
        report.source_cluster_integrity
    );
    Ok(())
}

fn analyze_causal_study_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 3 {
        return Err(
            "usage: contamination-panel-bench analyze-causal-study <replicated-causal-study.json> <analysis-report.json>"
                .into(),
        );
    }
    let study: ReplicatedCausalStudyV1 = serde_json::from_slice(&fs::read(&args[1])?)?;
    let report = analyze_replicated_causal_study(&study).map_err(io::Error::other)?;
    write_bytes(&args[2], &serde_json::to_vec_pretty(&report)?)?;
    println!(
        "replicates={} primary_cluster_effect={:.3}pp localization_p={:.5} confirmatory_eligible={} status={}",
        report.independent_model_pair_replicate_count,
        report.primary_cluster_weighted_effect_permille.mean / 10.0,
        report.localization_randomization_p_value,
        report.confirmatory_eligible,
        report.analysis_status
    );
    println!("claim boundary: {}", report.claim_boundary);
    Ok(())
}


fn read_file_bounded(path: &str, maximum_bytes: u64) -> Result<Vec<u8>, Box<dyn Error>> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > maximum_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("input file {path} exceeds {maximum_bytes} bytes"),
        )
        .into());
    }
    let file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(maximum_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("input file {path} grew beyond {maximum_bytes} bytes while reading"),
        )
        .into());
    }
    Ok(bytes)
}

fn analyze_exposure_transfer_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 4 {
        return Err(
            "usage: contamination-panel-bench analyze-exposure-transfer <replicated-causal-study.json> <exposure-transfer-study.json> <transfer-report.json>"
                .into(),
        );
    }
    const MAX_STUDY_BYTES: u64 = 256 * 1024 * 1024;
    let base_bytes = read_file_bounded(&args[1], MAX_STUDY_BYTES)?;
    let transfer_bytes = read_file_bounded(&args[2], MAX_STUDY_BYTES)?;
    let base_study: ReplicatedCausalStudyV1 = serde_json::from_slice(&base_bytes)?;
    let transfer_study: ExposureTransferStudyV1 = serde_json::from_slice(&transfer_bytes)?;
    let report = analyze_exposure_transfer(&base_study, &transfer_study)
        .map_err(io::Error::other)?;
    write_bytes(&args[3], &serde_json::to_vec_pretty(&report)?)?;
    println!(
        "replicates={} families={} relations={} contrasts={} confirmatory_eligible={} status={}",
        report.independent_model_pair_replicate_count,
        report.family_count,
        report.transfer_relation_effects.len(),
        report.predeclared_contrasts.len(),
        report.confirmatory_eligible,
        report.analysis_status
    );
    println!("claim boundary: {}", report.claim_boundary);
    Ok(())
}

fn audit_training_receipt_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 6 {
        return Err(
            "usage: contamination-panel-bench audit-training-receipt <design.json> <intervention-manifest.json> <seed-hex> <paired-training-receipt.json> <audit-report.json>"
                .into(),
        );
    }
    let design: RandomizedInterventionDesignV1 = serde_json::from_slice(&fs::read(&args[1])?)?;
    let manifest: RandomizedInterventionManifestV1 =
        serde_json::from_slice(&fs::read(&args[2])?)?;
    let seed = decode_hex(&args[3]).map_err(io::Error::other)?;
    let receipt_bytes = fs::read(&args[4])?;
    let report = validate_paired_training_execution_receipt(
        &design,
        &manifest,
        &seed,
        &receipt_bytes,
    ).map_err(io::Error::other)?;
    write_bytes(&args[5], &serde_json::to_vec_pretty(&report)?)?;
    println!(
        "valid={} execution_consistent={} ready_for_external_verification={} independently_verified={}",
        report.valid,
        report.execution_consistent,
        report.ready_for_independent_attestation_verification,
        report.independently_verified
    );
    println!("attestation status: {}", report.external_attestation_verification);
    Ok(())
}

fn audit_adjudications(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 4 {
        return Err(
            "usage: contamination-panel-bench audit-adjudications <public-packet.json> <adjudications.jsonl> <report.json>"
                .into(),
        );
    }
    let packet_bytes = fs::read(&args[1])?;
    let packet: PublicChallengePacketV1 = serde_json::from_slice(&packet_bytes)?;
    let file = fs::File::open(&args[2])?;
    let mut records = Vec::new();
    for (line_number, line) in io::BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: AdjudicationRecordV1 = serde_json::from_str(&line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid adjudication JSON at line {}: {error}", line_number + 1),
            )
        })?;
        records.push(record);
    }

    let report = audit_blinded_adjudications(&packet, &records).map_err(io::Error::other)?;
    if report.schema != ADJUDICATION_REPORT_SCHEMA {
        return Err("internal report schema mismatch".into());
    }
    let report_bytes = serde_json::to_vec_pretty(&report)?;
    write_bytes(&args[3], &report_bytes)?;
    println!(
        "valid={} pairs={} disputed={} insufficient_evidence={}",
        report.valid,
        report.pair_count,
        report.disputed_pair_count,
        report.insufficient_evidence_pair_count
    );
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("generate") => generate(&args),
        Some("score") => score_builtins(&args),
        Some("evaluate") => evaluate_run(&args),
        Some("ftd-pvalues") => ftd_pvalues(&args),
        Some("ftd-fuse") => ftd_fuse(&args),
        Some("randomize-intervention") => randomize_intervention_cli(&args),
        Some("audit-intervention") => audit_intervention_cli(&args),
        Some("audit-training-receipt") => audit_training_receipt_cli(&args),
        Some("analyze-causal-study") => analyze_causal_study_cli(&args),
        Some("analyze-exposure-transfer") => analyze_exposure_transfer_cli(&args),
        Some("audit-adjudications") => audit_adjudications(&args),
        _ => Err(
            "commands: generate | score | evaluate | randomize-intervention | audit-intervention | audit-training-receipt | analyze-causal-study | analyze-exposure-transfer | ftd-pvalues | ftd-fuse | audit-adjudications\nRun a command without the required arguments to see its usage."
                .into(),
        ),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(2);
    }
}
