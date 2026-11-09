// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//
// Dependency-free, reproducible HME selector microbenchmark.
// Run with:
//   cargo bench -p symthaea-memory --bench hme_hierarchical_select
// Optional environment:
//   HME_BENCH_RECORDS=4096 HME_BENCH_DIM=16384 HME_BENCH_GROUP_SIZE=64
//   HME_BENCH_TOP_K=10 HME_BENCH_REPEATS=21 HME_BENCH_MODE=box|ball
//
// This harness measures CPU/algorithm behavior and logical candidate payload
// reads. It cannot establish physical disk I/O, page faults, cache residency,
// or production compression ratios.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::hint::black_box;
use std::time::Instant;

use symthaea_core::hdc::ContinuousHV;
use symthaea_memory::hme::{
    ContinuousHVGroupEnvelope, ExactScoreRecord, GroupEnvelopeMode,
    GroupIntervalCoverageCounts, GroupMemberScoreIntervalRecord,
    GroupPartitionCoverageReceipt, GroupScoreIntervalRecord,
    HmeEligibilityScopeBinding, VerifiedMemberResidualCertificate,
    estimate_group_envelope_memory, member_id_set_digest,
    select_group_refinement_set, select_group_refinement_set_with_partition_receipt,
    select_hierarchical_refinement_set_with_partition_receipt, stable_exact_top_k,
};

#[derive(Clone, Copy)]
struct Config {
    records: usize,
    dimension: usize,
    group_size: usize,
    top_k: usize,
    repeats: usize,
    mode: GroupEnvelopeMode,
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{name} must be a positive integer, got {value:?}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("cannot read {name}: {error}")),
    }
}

fn env_mode() -> Result<GroupEnvelopeMode, String> {
    let value = env::var("HME_BENCH_MODE").unwrap_or_else(|_| "box".to_owned());
    match value.to_ascii_lowercase().as_str() {
        "box" | "box_and_ball" => Ok(GroupEnvelopeMode::BoxAndBall),
        "ball" | "ball_only" => Ok(GroupEnvelopeMode::BallOnly),
        "quantized" | "quantized_box_and_ball" => Err(
            "QuantizedBoxAndBall remains selector-gated pending #11243; benchmark box or ball".to_owned(),
        ),
        _ => Err(format!("HME_BENCH_MODE must be 'box' or 'ball', got {value:?}")),
    }
}

fn config() -> Result<Config, String> {
    let cfg = Config {
        records: env_usize("HME_BENCH_RECORDS", 512)?,
        dimension: env_usize("HME_BENCH_DIM", 512)?,
        group_size: env_usize("HME_BENCH_GROUP_SIZE", 32)?,
        top_k: env_usize("HME_BENCH_TOP_K", 10)?,
        repeats: env_usize("HME_BENCH_REPEATS", 21)?,
        mode: env_mode()?,
    };
    if cfg.records == 0 || cfg.dimension == 0 || cfg.group_size == 0 || cfg.top_k == 0 || cfg.repeats == 0 {
        return Err("all HME_BENCH_* values must be positive".to_owned());
    }
    if cfg.top_k > cfg.records {
        return Err("HME_BENCH_TOP_K must not exceed HME_BENCH_RECORDS".to_owned());
    }
    if cfg.dimension > 1_048_576 || cfg.records > 10_000_000 {
        return Err("benchmark dimensions exceed the safety cap".to_owned());
    }
    cfg.records
        .checked_mul(cfg.dimension)
        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "vector payload byte calculation overflowed usize".to_owned())?;
    Ok(cfg)
}

fn scope_binding() -> HmeEligibilityScopeBinding {
    HmeEligibilityScopeBinding {
        principal_scope_digest: [0x11; 32],
        policy_digest: [0x22; 32],
        policy_generation: 1,
        source_ledger_generation: 1,
        bitemporal_query_digest: [0x33; 32],
    }
}

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn make_member(
    group_index: usize,
    member_index: usize,
    group_count: usize,
    dimension: usize,
    state: &mut u32,
) -> ContinuousHV {
    let mut values = Vec::with_capacity(dimension);
    let rank = if group_count <= 1 {
        0.0
    } else {
        group_index as f32 / (group_count - 1) as f32
    };
    let leading = 0.94 - 1.88 * rank;
    for coordinate in 0..dimension {
        let random = (xorshift(state) as f64 / u32::MAX as f64) as f32;
        let centered_noise = (random - 0.5) * 0.02;
        let member_shift = (member_index as f32 - 0.5) * 0.00002;
        let value = if coordinate == 0 {
            leading + centered_noise * 0.1 + member_shift
        } else {
            centered_noise + member_shift * ((coordinate % 7) as f32 / 7.0)
        };
        values.push(value.clamp(-1.0, 1.0));
    }
    ContinuousHV::from_vec(values)
}

fn quantized_reconstruction(original: &ContinuousHV) -> (ContinuousHV, f32) {
    let reconstruction = ContinuousHV::from_vec(
        original
            .values
            .iter()
            .map(|value| ((*value * 1024.0).round() / 1024.0).clamp(-1.0, 1.0))
            .collect(),
    );
    let residual_squared = original
        .values
        .iter()
        .zip(reconstruction.values.iter())
        .map(|(a, b)| {
            let delta = *a as f64 - *b as f64;
            delta * delta
        })
        .sum::<f64>();
    let residual = residual_squared.sqrt();
    // The verifier is authoritative. This conservative margin keeps the
    // benchmark from relying on a nearest-rounded residual estimate.
    let radius = (residual * 1.01 + 1.0e-7) as f32;
    (reconstruction, radius)
}

fn percentile_ns(samples: &[u128], percentile: usize) -> u128 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let index = ((sorted.len() - 1) * percentile + 99) / 100;
    sorted[index.min(sorted.len() - 1)]
}

fn timed_samples<F>(repeats: usize, mut operation: F) -> Vec<u128>
where
    F: FnMut(),
{
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        operation();
        samples.push(start.elapsed().as_nanos());
    }
    samples
}

fn main() {
    if let Err(error) = run() {
        eprintln!("hme_hierarchical_select: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let cfg = config()?;
    let group_count = cfg.records / cfg.group_size
        + usize::from(cfg.records % cfg.group_size != 0);
    let query = {
        let mut values = vec![0.0f32; cfg.dimension];
        values[0] = 1.0;
        ContinuousHV::from_vec(values)
    };

    let mut state = 0x484D_4531u32;
    let mut groups: Vec<Vec<(String, ContinuousHV)>> = Vec::with_capacity(group_count);
    let mut reconstructions: Vec<Vec<(String, ContinuousHV)>> = Vec::with_capacity(group_count);
    let mut originals_by_id: BTreeMap<String, ContinuousHV> = BTreeMap::new();
    let mut all_ids = Vec::with_capacity(cfg.records);

    let dataset_start = Instant::now();
    for group_index in 0..group_count {
        let start_index = group_index * cfg.group_size;
        let member_count = cfg.group_size.min(cfg.records - start_index);
        let mut members = Vec::with_capacity(member_count);
        let mut reconstructed_members = Vec::with_capacity(member_count);
        for member_index in 0..member_count {
            let global_index = start_index + member_index;
            let id = format!("hme-{global_index:012}");
            let original = make_member(
                group_index,
                member_index,
                group_count,
                cfg.dimension,
                &mut state,
            );
            let (reconstruction, _) = quantized_reconstruction(&original);
            all_ids.push(id.clone());
            originals_by_id.insert(id.clone(), original.clone());
            members.push((id.clone(), original));
            reconstructed_members.push((id, reconstruction));
        }
        groups.push(members);
        reconstructions.push(reconstructed_members);
    }
    let dataset_build_ns = dataset_start.elapsed().as_nanos();

    let query_scope = scope_binding();
    let envelope_start = Instant::now();
    let mut envelopes = Vec::with_capacity(group_count);
    for members in &groups {
        envelopes.push(
            ContinuousHVGroupEnvelope::from_original_members_with_mode(
                1,
                1,
                members,
                cfg.mode,
            )
            .map_err(|error| format!("group envelope build failed: {error}"))?,
        );
    }
    let envelope_build_ns = envelope_start.elapsed().as_nanos();

    let receipt_start = Instant::now();
    let member_set_digest = member_id_set_digest(&all_ids)
        .map_err(|error| format!("eligible member digest failed: {error}"))?;
    let partition_receipt = GroupPartitionCoverageReceipt::from_group_envelopes(
        &envelopes,
        cfg.records,
        member_set_digest,
        query_scope,
    )
    .map_err(|error| format!("partition receipt build failed: {error}"))?;
    let partition_receipt_build_ns = receipt_start.elapsed().as_nanos();

    let group_record_start = Instant::now();
    let mut group_records = Vec::with_capacity(group_count);
    for envelope in &envelopes {
        group_records.push(
            GroupScoreIntervalRecord::from_group_envelope(
                envelope,
                &query,
                1,
                1,
                envelope.member_set_digest(),
                envelope.content_digest(),
                query_scope,
            )
            .map_err(|error| format!("group score interval failed: {error}"))?,
        );
    }
    let group_record_build_ns = group_record_start.elapsed().as_nanos();

    let coverage = GroupIntervalCoverageCounts {
        eligible_records: cfg.records,
        covered_records: cfg.records,
        interval_groups: group_count,
        expected_manifest_generation: 1,
        expected_eligibility_generation: 1,
        query_digest: symthaea_memory::hme::continuous_hv_query_digest(&query),
        scope_binding: query_scope,
        unbounded_unopened_groups: 0,
    };

    let original_payload_bytes = cfg.records
        .checked_mul(cfg.dimension)
        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "original payload byte calculation overflowed".to_owned())?;
    let metadata_estimate = estimate_group_envelope_memory(
        cfg.dimension,
        cfg.records,
        cfg.group_size,
        24,
        cfg.mode,
    )
    .map_err(|error| format!("metadata estimate failed: {error}"))?;

    for _ in 0..3 {
        black_box(
            select_group_refinement_set(&group_records, cfg.top_k, coverage)
                .expect("defensive selector warmup must succeed"),
        );
        black_box(
            select_group_refinement_set_with_partition_receipt(
                &group_records,
                cfg.top_k,
                coverage,
                &partition_receipt,
            )
            .expect("receipt-backed selector warmup must succeed"),
        );
    }

    let defensive_samples = timed_samples(cfg.repeats, || {
        let plan = select_group_refinement_set(
            black_box(&group_records),
            cfg.top_k,
            coverage,
        )
        .expect("defensive selector must succeed");
        black_box(plan);
    });
    let receipt_samples = timed_samples(cfg.repeats, || {
        let plan = select_group_refinement_set_with_partition_receipt(
            black_box(&group_records),
            cfg.top_k,
            coverage,
            black_box(&partition_receipt),
        )
        .expect("receipt-backed selector must succeed");
        black_box(plan);
    });

    // Admission-time receipts are generated once from original payloads and a
    // deterministic 1/1024 reconstruction. Query-time interval materialization
    // below receives only the compact reconstruction plus the opaque receipt.
    let certificate_start = Instant::now();
    let transform_profile_digest = [0x71; 32];
    let mut certificates = Vec::with_capacity(cfg.records);
    for group_index in 0..group_count {
        let mut group_certificates = Vec::with_capacity(groups[group_index].len());
        for ((id, original), (_, reconstruction)) in groups[group_index]
            .iter()
            .zip(reconstructions[group_index].iter())
        {
            let (_, residual_radius) = quantized_reconstruction(original);
            group_certificates.push(
                VerifiedMemberResidualCertificate::verify_at_ingest(
                    &group_records[group_index],
                    id.clone(),
                    original,
                    reconstruction,
                    residual_radius,
                    transform_profile_digest,
                )
                .map_err(|error| format!("residual admission failed for {id}: {error}"))?,
            );
        }
        certificates.push(group_certificates);
    }
    let certificate_build_ns = certificate_start.elapsed().as_nanos();

    let exact_oracle: Vec<ExactScoreRecord<String>> = originals_by_id
        .iter()
        .map(|(id, original)| ExactScoreRecord {
            id: id.clone(),
            score: query.similarity_full_resolution(original),
        })
        .collect();
    let expected_top_k = stable_exact_top_k(&exact_oracle, cfg.top_k)
        .map_err(|error| format!("full scan oracle failed: {error}"))?;

    let mut group_select_samples = Vec::with_capacity(cfg.repeats);
    let mut member_interval_samples = Vec::with_capacity(cfg.repeats);
    let mut hierarchy_samples = Vec::with_capacity(cfg.repeats);
    let mut exact_rerank_samples = Vec::with_capacity(cfg.repeats);
    let mut last_group_plan = None;
    let mut last_hierarchical_plan = None;

    for iteration in 0..(cfg.repeats + 3) {
        let measure = iteration >= 3;
        let group_start = Instant::now();
        let group_plan = select_group_refinement_set_with_partition_receipt(
            &group_records,
            cfg.top_k,
            coverage,
            &partition_receipt,
        )
        .map_err(|error| format!("group selection failed: {error}"))?;
        if measure {
            group_select_samples.push(group_start.elapsed().as_nanos());
        }

        let selected_identities: BTreeSet<[u8; 32]> = group_plan
            .refinement_groups
            .iter()
            .map(|identity| identity.member_set_digest)
            .collect();
        let selected: BTreeMap<[u8; 32], usize> = group_records
            .iter()
            .enumerate()
            .filter(|(_, group)| {
                selected_identities.contains(&group.identity().member_set_digest)
            })
            .map(|(index, group)| (group.identity().member_set_digest, index))
            .collect();

        let member_start = Instant::now();
        let mut member_records: Vec<GroupMemberScoreIntervalRecord> =
            Vec::with_capacity(group_plan.refinement_record_count);
        for group_identity in &group_plan.refinement_groups {
            let group_index = *selected
                .get(&group_identity.member_set_digest)
                .ok_or_else(|| "selected group was absent from group directory".to_owned())?;
            for (((id, _original), (_, reconstruction)), certificate) in groups[group_index]
                .iter()
                .zip(reconstructions[group_index].iter())
                .zip(certificates[group_index].iter())
            {
                member_records.push(
                    GroupMemberScoreIntervalRecord::from_verified_certificate(
                        &group_records[group_index],
                        &query,
                        reconstruction,
                        certificate,
                        transform_profile_digest,
                    )
                    .map_err(|error| format!("member interval failed for {id}: {error}"))?,
                );
            }
        }
        if measure {
            member_interval_samples.push(member_start.elapsed().as_nanos());
        }

        let hierarchy_start = Instant::now();
        let plan = select_hierarchical_refinement_set_with_partition_receipt(
            &group_records,
            &member_records,
            cfg.top_k,
            coverage,
            &partition_receipt,
        )
        .map_err(|error| format!("hierarchical selection failed: {error}"))?;
        if measure {
            hierarchy_samples.push(hierarchy_start.elapsed().as_nanos());
        }

        let exact_start = Instant::now();
        let mut exact_scores = Vec::with_capacity(plan.member_plan.refinement_ids.len());
        for id in &plan.member_plan.refinement_ids {
            let original = originals_by_id
                .get(id)
                .ok_or_else(|| format!("exact refinement payload missing for {id}"))?;
            exact_scores.push(ExactScoreRecord {
                id: id.clone(),
                score: query.similarity_full_resolution(black_box(original)),
            });
        }
        let actual_top_k = stable_exact_top_k(&exact_scores, cfg.top_k)
            .map_err(|error| format!("refined exact rerank failed: {error}"))?;
        if measure {
            exact_rerank_samples.push(exact_start.elapsed().as_nanos());
        }
        if actual_top_k != expected_top_k {
            return Err(format!(
                "exact top-k mismatch: expected {expected_top_k:?}, received {actual_top_k:?}"
            ));
        }
        last_group_plan = Some(group_plan);
        last_hierarchical_plan = Some(plan);
    }

    let group_plan = last_group_plan.ok_or_else(|| "no benchmark iteration ran".to_owned())?;
    let hierarchy_plan = last_hierarchical_plan.ok_or_else(|| "no benchmark iteration ran".to_owned())?;
    let exact_payload_reads = hierarchy_plan.member_plan.refinement_ids.len();
    let exact_payload_bytes = exact_payload_reads
        .checked_mul(cfg.dimension)
        .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "exact payload byte calculation overflowed".to_owned())?;
    let avoided_bytes = original_payload_bytes.saturating_sub(exact_payload_bytes);
    let avoided_pct = if original_payload_bytes == 0 {
        0.0
    } else {
        avoided_bytes as f64 * 100.0 / original_payload_bytes as f64
    };
    let full_numeric_per_group = cfg.dimension * 12 + 4;
    let ball_numeric_per_group = cfg.dimension * 4 + 4;

    println!("schema_version=hme_hierarchical_select_bench_v1");
    println!("records={}", cfg.records);
    println!("dimension={}", cfg.dimension);
    println!("group_size={}", cfg.group_size);
    println!("envelope_mode={:?}", cfg.mode);
    println!("groups={group_count}");
    println!("top_k={}", cfg.top_k);
    println!("repeats={}", cfg.repeats);
    println!("warmup_iterations=3");
    println!("exact_top_k_agreement=true");
    println!("expected_top_k={:?}", expected_top_k);
    println!("refinement_groups={}", group_plan.refinement_groups.len());
    println!("refinement_members={}", hierarchy_plan.group_plan.refinement_record_count);
    println!("final_exact_payload_reads={exact_payload_reads}");
    println!("full_scan_payload_reads={}", cfg.records);
    println!("logical_original_payload_bytes_full_scan={original_payload_bytes}");
    println!("logical_original_payload_bytes_exact_refine={exact_payload_bytes}");
    println!("logical_original_payload_bytes_avoided={avoided_bytes}");
    println!("logical_payload_bytes_avoided_pct={avoided_pct:.3}");
    println!("physical_io_measured=false");
    println!("physical_io_note=requires immutable-segment integration and device/page telemetry");
    println!("group_partition_receipt_query_verification=O(G_log_G)");
    println!("receipt_selector_p50_ns={}", percentile_ns(&receipt_samples, 50));
    println!("receipt_selector_p95_ns={}", percentile_ns(&receipt_samples, 95));
    println!("receipt_selector_p99_ns={}", percentile_ns(&receipt_samples, 99));
    println!("defensive_selector_p50_ns={}", percentile_ns(&defensive_samples, 50));
    println!("defensive_selector_p95_ns={}", percentile_ns(&defensive_samples, 95));
    println!("defensive_selector_p99_ns={}", percentile_ns(&defensive_samples, 99));
    println!("stage1_group_select_p50_ns={}", percentile_ns(&group_select_samples, 50));
    println!("stage1_group_select_p95_ns={}", percentile_ns(&group_select_samples, 95));
    println!("stage1_group_select_p99_ns={}", percentile_ns(&group_select_samples, 99));
    println!("member_interval_materialization_p50_ns={}", percentile_ns(&member_interval_samples, 50));
    println!("member_interval_materialization_p95_ns={}", percentile_ns(&member_interval_samples, 95));
    println!("member_interval_materialization_p99_ns={}", percentile_ns(&member_interval_samples, 99));
    println!("hierarchical_selector_p50_ns={}", percentile_ns(&hierarchy_samples, 50));
    println!("hierarchical_selector_p95_ns={}", percentile_ns(&hierarchy_samples, 95));
    println!("hierarchical_selector_p99_ns={}", percentile_ns(&hierarchy_samples, 99));
    println!("exact_rerank_p50_ns={}", percentile_ns(&exact_rerank_samples, 50));
    println!("exact_rerank_p95_ns={}", percentile_ns(&exact_rerank_samples, 95));
    println!("exact_rerank_p99_ns={}", percentile_ns(&exact_rerank_samples, 99));
    println!("dataset_generation_ns={dataset_build_ns}");
    println!("group_envelope_build_ns={envelope_build_ns}");
    println!("partition_receipt_build_ns={partition_receipt_build_ns}");
    println!("group_interval_build_ns={group_record_build_ns}");
    println!("member_residual_certificate_build_ns={certificate_build_ns}");
    println!("known_group_metadata_bytes={}", metadata_estimate.known_metadata_bytes);
    println!("box_ball_numeric_bytes_per_group={full_numeric_per_group}");
    println!("ball_only_numeric_bytes_per_group={ball_numeric_per_group}");
    println!("reconstruction_representation=unpacked_f32_quantization_grid_1_over_1024");
    println!("score_contract=continuous_hv_cosine_full_f32_group8_v1");
    println!("benchmark_status=authored_not_executed_in_current_environment");

    Ok(())
}
