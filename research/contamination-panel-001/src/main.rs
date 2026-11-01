use contamination_panel_bench::{
    ADJUDICATION_REPORT_SCHEMA, AdjudicationRecordV1, ChallengeManifestV1, DetectorRunV1,
    causal_analysis::{ReplicatedCausalStudyV1, analyze_replicated_causal_study},
    fresh_instance::{
        FreshChallengeBundleV1, FreshChallengeConfigV1, FreshChallengeManifestV1,
        FreshPublicChallengePacketV1, SealedFreshOraclePacketV1,
        audit_fresh_challenge_bundle, generate_fresh_challenge_bundle,
    },
    fresh_scoring::{FreshScoreReportV1, score_fresh_submission, verify_fresh_score_report_integrity},
    search_time::{
        SearchTimeExposureTraceV1, SearchTimeExposureFindingsV1,
        SEARCH_MAX_FINDINGS_BYTES, SEARCH_MAX_TRACE_BYTES,
        audit_search_time_trace, audit_search_time_findings,
    },
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

fn read_bounded(path: &str, maximum_bytes: usize, label: &str) -> io::Result<Vec<u8>> {
    let file = fs::File::open(path)?;
    let limit = u64::try_from(maximum_bytes)
        .map_err(|_| io::Error::other(format!("{label} byte limit exceeds u64")))?;
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() > maximum_bytes {
        return Err(io::Error::other(format!(
            "{label} exceeds the {} byte safety bound",
            maximum_bytes
        )));
    }
    Ok(bytes)
}

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

fn verify_fresh_score_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 2 {
        return Err(
            "usage: contamination-panel-bench verify-fresh-score <score-report.json>"
                .into(),
        );
    }
    let report: FreshScoreReportV1 = serde_json::from_slice(&fs::read(&args[1])?)?;
    verify_fresh_score_report_integrity(&report).map_err(io::Error::other)?;
    println!(
        "report_valid=true run={} accuracy={:.1}% coverage={:.1}% independently_verified={}",
        report.run_id,
        report.accuracy_over_all_permille / 10.0,
        report.coverage_permille / 10.0,
        report.independently_verified
    );
    Ok(())
}

fn audit_search_time_trace_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 3 {
        return Err(
            "usage: contamination-panel-bench audit-search-time-trace <trace.json> <audit-report.json>"
                .into(),
        );
    }
    let trace_bytes = read_bounded(&args[1], SEARCH_MAX_TRACE_BYTES, "search-time trace")?;
    let trace: SearchTimeExposureTraceV1 = serde_json::from_slice(&trace_bytes)?;
    let report = audit_search_time_trace(&trace).map_err(io::Error::other)?;
    write_bytes(&args[2], &serde_json::to_vec_pretty(&report)?)?;
    println!(
        "valid={} events={} allowed={} blocked={} coverage={} complete_mediation_verified={}",
        report.valid,
        report.event_count,
        report.allowed_event_count,
        report.blocked_event_count,
        report.coverage_status,
        report.complete_mediation_verified
    );
    Ok(())
}

fn audit_search_time_findings_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 4 {
        return Err(
            "usage: contamination-panel-bench audit-search-time-findings <trace.json> <findings.json> <audit-report.json>"
                .into(),
        );
    }
    let trace_bytes = fs::read(&args[1])?;
    if trace_bytes.len() > SEARCH_MAX_TRACE_BYTES {
        return Err("search-time trace exceeds 256 MiB safety bound".into());
    }
    let trace: SearchTimeExposureTraceV1 = serde_json::from_slice(&trace_bytes)?;
    let trace_audit = audit_search_time_trace(&trace).map_err(io::Error::other)?;
    if !trace_audit.valid {
        return Err("search-time trace failed integrity audit".into());
    }
    let findings_bytes = read_bounded(&args[2], SEARCH_MAX_FINDINGS_BYTES, "search-time findings")?;
    let findings: SearchTimeExposureFindingsV1 =
        serde_json::from_slice(&findings_bytes)?;
    let report = audit_search_time_findings(&trace, &findings).map_err(io::Error::other)?;
    write_bytes(&args[3], &serde_json::to_vec_pretty(&report)?)?;
    println!(
        "valid={} metadata={} context={} answer={} unknown={}",
        report.valid,
        report.metadata_only_count,
        report.context_bearing_count,
        report.answer_bearing_count,
        report.unknown_count
    );
    println!("Classification evidence is evaluator-supplied; causal influence is not established.");
    Ok(())
}

fn score_fresh_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 6 {
        return Err(
            "usage: contamination-panel-bench score-fresh <public-packet.json> <sealed-oracle.json> <manifest.json> <answer-submission.json> <score-report.json>"
                .into(),
        );
    }
    let bundle = FreshChallengeBundleV1 {
        public_packet: serde_json::from_slice::<FreshPublicChallengePacketV1>(&fs::read(&args[1])?)?,
        sealed_oracle_packet: serde_json::from_slice::<SealedFreshOraclePacketV1>(&fs::read(&args[2])?)?,
        manifest: serde_json::from_slice::<FreshChallengeManifestV1>(&fs::read(&args[3])?)?,
    };
    let submission_bytes = fs::read(&args[4])?;
    let report = score_fresh_submission(&bundle, &submission_bytes).map_err(io::Error::other)?;
    write_bytes(&args[5], &serde_json::to_vec_pretty(&report)?)?;
    println!(
        "run={} correct={}/{} accuracy={:.1}% coverage={:.1}% invalid={} abstained={}",
        report.run_id,
        report.correct_count,
        report.item_count,
        report.accuracy_over_all_permille / 10.0,
        report.coverage_permille / 10.0,
        report.invalid_format_count,
        report.abstained_count
    );
    println!(
        "execution receipt status: {}",
        report.execution_receipt_verification
    );
    Ok(())
}

fn generate_fresh_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 6 {
        return Err(
            "usage: contamination-panel-bench generate-fresh <config.json> <seed-hex> <public-packet.json> <sealed-oracle.json> <manifest.json>"
                .into(),
        );
    }
    let config: FreshChallengeConfigV1 = serde_json::from_slice(&fs::read(&args[1])?)?;
    let seed = decode_hex(&args[2]).map_err(io::Error::other)?;
    let bundle = generate_fresh_challenge_bundle(&config, &seed).map_err(io::Error::other)?;
    write_bytes(&args[3], &serde_json::to_vec_pretty(&bundle.public_packet)?)?;
    write_bytes(&args[4], &serde_json::to_vec_pretty(&bundle.sealed_oracle_packet)?)?;
    write_bytes(&args[5], &serde_json::to_vec_pretty(&bundle.manifest)?)?;
    println!(
        "batch={} items={} public_sha256={} manifest_sha256={}",
        bundle.manifest.batch_id,
        bundle.manifest.item_count,
        bundle.manifest.public_packet_sha256,
        bundle.manifest.manifest_sha256
    );
    println!("The sealed oracle and seed are evaluator-only; do not publish them.");
    Ok(())
}

fn audit_fresh_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 7 {
        return Err(
            "usage: contamination-panel-bench audit-fresh <config.json> <seed-hex> <public-packet.json> <sealed-oracle.json> <manifest.json> <audit.json>"
                .into(),
        );
    }
    let config: FreshChallengeConfigV1 = serde_json::from_slice(&fs::read(&args[1])?)?;
    let seed = decode_hex(&args[2]).map_err(io::Error::other)?;
    let bundle = FreshChallengeBundleV1 {
        public_packet: serde_json::from_slice::<FreshPublicChallengePacketV1>(&fs::read(&args[3])?)?,
        sealed_oracle_packet: serde_json::from_slice::<SealedFreshOraclePacketV1>(&fs::read(&args[4])?)?,
        manifest: serde_json::from_slice::<FreshChallengeManifestV1>(&fs::read(&args[5])?)?,
    };
    let report = audit_fresh_challenge_bundle(&config, &seed, &bundle).map_err(io::Error::other)?;
    write_bytes(&args[6], &serde_json::to_vec_pretty(&report)?)?;
    println!(
        "valid={} regenerated={} nonsingular={} answers_exact={} freeze_authenticated={}",
        report.valid,
        report.deterministic_regeneration_matches,
        report.all_matrices_are_nonsingular,
        report.all_answers_recompute_exactly,
        report.independently_verified_checkpoint_freeze
    );
    println!("claim boundary: {}", report.claim_boundary);
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
        Some("generate-fresh") => generate_fresh_cli(&args),
        Some("audit-fresh") => audit_fresh_cli(&args),
        Some("score-fresh") => score_fresh_cli(&args),
        Some("verify-fresh-score") => verify_fresh_score_cli(&args),
        Some("audit-search-time-trace") => audit_search_time_trace_cli(&args),
        Some("audit-search-time-findings") => audit_search_time_findings_cli(&args),
        Some("audit-adjudications") => audit_adjudications(&args),
        _ => Err(
            "commands: generate | score | evaluate | randomize-intervention | audit-intervention | audit-training-receipt | analyze-causal-study | generate-fresh | audit-fresh | score-fresh | verify-fresh-score | audit-search-time-trace | audit-search-time-findings | ftd-pvalues | ftd-fuse | audit-adjudications\nRun a command without the required arguments to see its usage."
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
