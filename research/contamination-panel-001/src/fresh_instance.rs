//! Secret-seeded, inference-time fresh benchmark instances.
//!
//! This module creates small exact-arithmetic linear-system challenges after a
//! model checkpoint freeze timestamp. The public challenge packet contains task
//! text only; the sealed oracle packet contains parameters and exact answers.
//! A secret seed commitment, generator/config digests and artifact hashes bind
//! deterministic regeneration.
//!
//! This provides exact-instance freshness, not task-recipe novelty. Models may
//! have learned the general method. Timestamp ordering and receipt digests are
//! self-reported claims unless separately authenticated by a trusted signer.

use crate::hash_domain;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const FRESH_CONFIG_SCHEMA: &str = "symthaea.fresh_instance_config.v1";
pub const FRESH_PUBLIC_SCHEMA: &str = "symthaea.fresh_instance_public_packet.v1";
pub const FRESH_ORACLE_SCHEMA: &str = "symthaea.fresh_instance_sealed_oracle.v1";
pub const FRESH_MANIFEST_SCHEMA: &str = "symthaea.fresh_instance_manifest.v1";
pub const FRESH_AUDIT_SCHEMA: &str = "symthaea.fresh_instance_audit.v1";
pub const FRESH_MIN_SEED_BYTES: usize = 32;
pub const FRESH_MAX_ITEMS: u32 = 10_000;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshChallengeConfigV1 {
    pub schema: String,
    pub batch_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub task_family: String,
    pub requested_item_count: u32,
    pub matrix_coefficient_min: i32,
    pub matrix_coefficient_max: i32,
    pub solution_component_min: i32,
    pub solution_component_max: i32,
    pub minimum_absolute_determinant: i64,
    pub model_checkpoint_sha256: String,
    pub checkpoint_freeze_receipt_sha256: String,
    pub checkpoint_frozen_at_utc: String,
    pub challenge_generated_at_utc: String,
    pub generator_source_sha256: String,
    pub generator_binary_sha256: String,
    pub runtime_image_sha256: String,
    pub generator_config_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PublicFreshTaskV1 {
    pub item_id: String,
    pub task_text: String,
    pub task_text_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshPublicChallengePacketV1 {
    pub schema: String,
    pub batch_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub task_family: String,
    pub model_checkpoint_sha256: String,
    pub generator_source_sha256: String,
    pub generator_binary_sha256: String,
    pub runtime_image_sha256: String,
    pub generator_config_sha256: String,
    pub checkpoint_frozen_at_utc: String,
    pub challenge_generated_at_utc: String,
    pub item_count: u64,
    pub tasks: Vec<PublicFreshTaskV1>,
    pub public_packet_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LinearSystemParametersV1 {
    pub a11: i32,
    pub a12: i32,
    pub a21: i32,
    pub a22: i32,
    pub b1: i64,
    pub b2: i64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExactLinearSolutionV1 {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedFreshOracleEntryV1 {
    pub item_id: String,
    pub parameters: LinearSystemParametersV1,
    pub solution: ExactLinearSolutionV1,
    pub parameter_sha256: String,
    pub answer_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedFreshOraclePacketV1 {
    pub schema: String,
    pub batch_id: String,
    /// Evaluator-only seed. Never distribute this packet to evaluated models.
    pub generation_seed_hex: String,
    pub generation_seed_commitment_sha256: String,
    pub item_count: u64,
    pub entries: Vec<SealedFreshOracleEntryV1>,
    pub sealed_oracle_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshChallengeManifestV1 {
    pub schema: String,
    pub batch_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub task_family: String,
    pub model_checkpoint_sha256: String,
    pub checkpoint_freeze_receipt_sha256: String,
    pub checkpoint_frozen_at_utc: String,
    pub challenge_generated_at_utc: String,
    pub generator_source_sha256: String,
    pub generator_binary_sha256: String,
    pub runtime_image_sha256: String,
    pub generator_config_sha256: String,
    pub generation_seed_commitment_sha256: String,
    pub seed_length_bits: u64,
    pub item_count: u64,
    pub public_packet_sha256: String,
    pub sealed_oracle_sha256: String,
    pub item_set_sha256: String,
    pub manifest_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshChallengeBundleV1 {
    pub public_packet: FreshPublicChallengePacketV1,
    pub sealed_oracle_packet: SealedFreshOraclePacketV1,
    pub manifest: FreshChallengeManifestV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FreshChallengeAuditV1 {
    pub schema: String,
    pub valid: bool,
    pub deterministic_regeneration_matches: bool,
    pub checkpoint_freeze_precedes_generation: bool,
    pub public_packet_contains_no_seed_or_oracle_answers: bool,
    pub all_matrices_are_nonsingular: bool,
    pub all_answers_recompute_exactly: bool,
    pub item_count: u64,
    pub batch_id: String,
    pub manifest_sha256: String,
    pub generation_seed_commitment_sha256: String,
    pub independently_verified_checkpoint_freeze: bool,
    pub claim_boundary: String,
    pub issues: Vec<String>,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if is_sha256(value) {
        Ok(())
    } else {
        Err(format!("{label} must be a lowercase SHA-256 digest"))
    }
}

fn parse_timestamp(value: &str, label: &str) -> Result<(), String> {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return Err(format!("{label} must use YYYY-MM-DDTHH:MM:SSZ"));
    }
    for (index, byte) in bytes.iter().enumerate() {
        if ![4, 7, 10, 13, 16, 19].contains(&index) && !byte.is_ascii_digit() {
            return Err(format!("{label} must use YYYY-MM-DDTHH:MM:SSZ"));
        }
    }
    let part = |start: usize, end: usize| value[start..end].parse::<u32>().unwrap_or(u32::MAX);
    let year = part(0, 4);
    let month = part(5, 7);
    let day = part(8, 10);
    let hour = part(11, 13);
    let minute = part(14, 16);
    let second = part(17, 19);
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 0,
    };
    if year == 0 || day == 0 || day > max_day || hour > 23 || minute > 59 || second > 59 {
        return Err(format!("{label} contains an impossible date or clock value"));
    }
    Ok(())
}

pub(crate) fn canonical_config_digest(config: &FreshChallengeConfigV1) -> Result<String, String> {
    let mut normalized = config.clone();
    normalized.generator_config_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize normalized generator config: {error}"))?;
    Ok(hash_domain(b"fresh-generator-config.v1", &bytes))
}

fn validate_config(config: &FreshChallengeConfigV1, seed: &[u8]) -> Result<(), String> {
    if config.schema != FRESH_CONFIG_SCHEMA {
        return Err(format!("config schema must be {FRESH_CONFIG_SCHEMA}"));
    }
    if config.batch_id.trim().is_empty()
        || config.batch_id.len() > 128
        || config.benchmark_id.trim().is_empty()
        || config.benchmark_version.trim().is_empty()
        || config.task_family.trim().is_empty()
    {
        return Err("batch, benchmark and task-family identities must be nonempty".to_owned());
    }
    if !(1..=FRESH_MAX_ITEMS).contains(&config.requested_item_count) {
        return Err(format!("requested_item_count must be in [1, {FRESH_MAX_ITEMS}]"));
    }
    for (value, label) in [
        (&config.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&config.checkpoint_freeze_receipt_sha256, "checkpoint_freeze_receipt_sha256"),
        (&config.generator_source_sha256, "generator_source_sha256"),
        (&config.generator_binary_sha256, "generator_binary_sha256"),
        (&config.runtime_image_sha256, "runtime_image_sha256"),
        (&config.generator_config_sha256, "generator_config_sha256"),
    ] {
        validate_digest(value, label)?;
    }
    if config.generator_config_sha256 != canonical_config_digest(config)? {
        return Err("generator_config_sha256 does not match canonical configuration contents".to_owned());
    }
    parse_timestamp(&config.checkpoint_frozen_at_utc, "checkpoint_frozen_at_utc")?;
    parse_timestamp(&config.challenge_generated_at_utc, "challenge_generated_at_utc")?;
    if config.checkpoint_frozen_at_utc >= config.challenge_generated_at_utc {
        return Err("checkpoint freeze timestamp must precede fresh challenge generation".to_owned());
    }
    if seed.len() < FRESH_MIN_SEED_BYTES {
        return Err(format!("fresh challenge seed must contain at least {} bytes", FRESH_MIN_SEED_BYTES));
    }
    if config.matrix_coefficient_min >= config.matrix_coefficient_max
        || config.solution_component_min >= config.solution_component_max
        || config.minimum_absolute_determinant < 1
    {
        return Err("invalid matrix/solution bounds or minimum determinant".to_owned());
    }
    if i64::from(config.matrix_coefficient_max)
        - i64::from(config.matrix_coefficient_min) > 1_000
        || i64::from(config.solution_component_max)
            - i64::from(config.solution_component_min) > 1_000
    {
        return Err("initial fresh algebra generator supports parameter widths <=1000".to_owned());
    }
    Ok(())
}

struct SeededStream {
    seed: Vec<u8>,
    counter: u64,
}

impl SeededStream {
    fn new(seed: &[u8]) -> Self {
        Self {
            seed: seed.to_vec(),
            counter: 0,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.fresh-instance-prf.v1\0");
        hasher.update(&self.seed);
        hasher.update(self.counter.to_be_bytes());
        self.counter = self.counter.wrapping_add(1);
        let digest = hasher.finalize();
        u64::from_be_bytes(digest[0..8].try_into().unwrap_or([0; 8]))
    }

    fn uniform_below(&mut self, upper_exclusive: u64) -> Result<u64, String> {
        if upper_exclusive == 0 {
            return Err("uniform sampling upper bound must be positive".to_owned());
        }
        let threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return Ok(value % upper_exclusive);
            }
        }
    }

    fn range_i32(&mut self, min: i32, max: i32) -> Result<i32, String> {
        let width = u64::try_from(i64::from(max) - i64::from(min) + 1)
            .map_err(|_| "integer sampling range is invalid")?;
        let offset = self.uniform_below(width)?;
        let value = i64::from(min) + i64::try_from(offset).map_err(|_| "sample offset overflows")?;
        i32::try_from(value).map_err(|_| "sampled integer is outside i32")
    }
}

fn determinant(parameters: &LinearSystemParametersV1) -> i64 {
    i64::from(parameters.a11) * i64::from(parameters.a22)
        - i64::from(parameters.a12) * i64::from(parameters.a21)
}

fn compute_rhs(
    parameters: &LinearSystemParametersV1,
    solution: &ExactLinearSolutionV1,
) -> Result<(i64, i64), String> {
    let b1 = i64::from(parameters.a11)
        .checked_mul(i64::from(solution.x))
        .and_then(|value| {
            i64::from(parameters.a12)
                .checked_mul(i64::from(solution.y))
                .and_then(|right| value.checked_add(right))
        })
        .ok_or_else(|| "linear-system rhs b1 overflowed".to_owned())?;
    let b2 = i64::from(parameters.a21)
        .checked_mul(i64::from(solution.x))
        .and_then(|value| {
            i64::from(parameters.a22)
                .checked_mul(i64::from(solution.y))
                .and_then(|right| value.checked_add(right))
        })
        .ok_or_else(|| "linear-system rhs b2 overflowed".to_owned())?;
    Ok((b1, b2))
}

fn task_text(parameters: &LinearSystemParametersV1) -> String {
    format!(
        "Find the integer values of x and y that satisfy both equations:\n{}x + {}y = {}\n{}x + {}y = {}\nReturn exactly one JSON object with integer fields x and y.",
        parameters.a11,
        parameters.a12,
        parameters.b1,
        parameters.a21,
        parameters.a22,
        parameters.b2
    )
}

fn generation_seed_commitment(seed: &[u8]) -> String {
    hash_domain(b"fresh-instance-seed.v1", seed)
}

fn public_packet_digest(packet: &FreshPublicChallengePacketV1) -> Result<String, String> {
    let mut clone = packet.clone();
    clone.public_packet_sha256.clear();
    let bytes = serde_json::to_vec(&clone)
        .map_err(|error| format!("cannot serialize public challenge packet: {error}"))?;
    Ok(hash_domain(b"fresh-public-packet.v1", &bytes))
}

fn oracle_packet_digest(packet: &SealedFreshOraclePacketV1) -> Result<String, String> {
    let mut clone = packet.clone();
    clone.sealed_oracle_sha256.clear();
    let bytes = serde_json::to_vec(&clone)
        .map_err(|error| format!("cannot serialize sealed oracle packet: {error}"))?;
    Ok(hash_domain(b"fresh-sealed-oracle.v1", &bytes))
}

fn item_set_digest(tasks: &[PublicFreshTaskV1]) -> Result<String, String> {
    let items: Vec<(&str, &str)> = tasks.iter()
        .map(|task| (task.item_id.as_str(), task.task_text_sha256.as_str()))
        .collect();
    let bytes = serde_json::to_vec(&items)
        .map_err(|error| format!("cannot serialize fresh item set: {error}"))?;
    Ok(hash_domain(b"fresh-instance-item-set.v1", &bytes))
}

fn manifest_digest(manifest: &FreshChallengeManifestV1) -> Result<String, String> {
    let mut clone = manifest.clone();
    clone.manifest_sha256.clear();
    let bytes = serde_json::to_vec(&clone)
        .map_err(|error| format!("cannot serialize fresh challenge manifest: {error}"))?;
    Ok(hash_domain(b"fresh-challenge-manifest.v1", &bytes))
}

/// Generate a deterministic fresh challenge bundle from a secret evaluator seed.
///
/// The public packet exposes no seed, matrix parameters or answer. The sealed
/// oracle packet is evaluator-only. Reusing the same config/seed reproduces the
/// exact bundle; a distinct seed changes the generated item payloads.
pub fn generate_fresh_challenge_bundle(
    config: &FreshChallengeConfigV1,
    seed: &[u8],
) -> Result<FreshChallengeBundleV1, String> {
    validate_config(config, seed)?;
    let seed_commitment = generation_seed_commitment(seed);
    let mut stream = SeededStream::new(seed);
    let mut tasks = Vec::with_capacity(config.requested_item_count as usize);
    let mut entries = Vec::with_capacity(config.requested_item_count as usize);
    let mut item_ids = BTreeSet::new();
    let mut task_digests = BTreeSet::new();

    for index in 0..config.requested_item_count {
        let mut accepted = None;
        for _attempt in 0..10_000 {
            let a11 = stream.range_i32(config.matrix_coefficient_min, config.matrix_coefficient_max)?;
            let a12 = stream.range_i32(config.matrix_coefficient_min, config.matrix_coefficient_max)?;
            let a21 = stream.range_i32(config.matrix_coefficient_min, config.matrix_coefficient_max)?;
            let a22 = stream.range_i32(config.matrix_coefficient_min, config.matrix_coefficient_max)?;
            let solution = ExactLinearSolutionV1 {
                x: stream.range_i32(config.solution_component_min, config.solution_component_max)?,
                y: stream.range_i32(config.solution_component_min, config.solution_component_max)?,
            };
            let mut parameters = LinearSystemParametersV1 {
                a11, a12, a21, a22, b1: 0, b2: 0,
            };
            if determinant(&parameters).abs() < config.minimum_absolute_determinant {
                continue;
            }
            let (b1, b2) = compute_rhs(&parameters, &solution)?;
            parameters.b1 = b1;
            parameters.b2 = b2;
            let text = task_text(&parameters);
            let task_digest = hash_domain(b"fresh-task-text.v1", text.as_bytes());
            if task_digests.contains(&task_digest) {
                continue;
            }
            accepted = Some((parameters, solution, text, task_digest));
            break;
        }
        let (parameters, solution, text, task_digest) = accepted
            .ok_or_else(|| format!("could not generate a unique nonsingular task at index {index}"))?;
        let item_id = format!("{}-item-{index:06}", config.batch_id);
        if !item_ids.insert(item_id.clone()) || !task_digests.insert(task_digest.clone()) {
            return Err("fresh generator produced duplicate item or task digest".to_owned());
        }
        let parameter_bytes = serde_json::to_vec(&parameters)
            .map_err(|error| format!("cannot serialize hidden parameters: {error}"))?;
        let answer_bytes = serde_json::to_vec(&solution)
            .map_err(|error| format!("cannot serialize exact solution: {error}"))?;
        let parameter_sha256 = hash_domain(b"fresh-task-parameters.v1", &parameter_bytes);
        let answer_sha256 = hash_domain(b"fresh-task-answer.v1", &answer_bytes);
        tasks.push(PublicFreshTaskV1 {
            item_id: item_id.clone(),
            task_text: text,
            task_text_sha256: task_digest,
        });
        entries.push(SealedFreshOracleEntryV1 {
            item_id,
            parameters,
            solution,
            parameter_sha256,
            answer_sha256,
        });
    }

    let mut public_packet = FreshPublicChallengePacketV1 {
        schema: FRESH_PUBLIC_SCHEMA.to_owned(),
        batch_id: config.batch_id.clone(),
        benchmark_id: config.benchmark_id.clone(),
        benchmark_version: config.benchmark_version.clone(),
        task_family: config.task_family.clone(),
        model_checkpoint_sha256: config.model_checkpoint_sha256.clone(),
        generator_source_sha256: config.generator_source_sha256.clone(),
        generator_binary_sha256: config.generator_binary_sha256.clone(),
        runtime_image_sha256: config.runtime_image_sha256.clone(),
        generator_config_sha256: config.generator_config_sha256.clone(),
        checkpoint_frozen_at_utc: config.checkpoint_frozen_at_utc.clone(),
        challenge_generated_at_utc: config.challenge_generated_at_utc.clone(),
        item_count: u64::from(config.requested_item_count),
        tasks,
        public_packet_sha256: String::new(),
        claim_boundary: "fresh_instance_challenge_only_no_claim_of_task_recipe_novelty_or_training_data_absence".to_owned(),
    };
    public_packet.public_packet_sha256 = public_packet_digest(&public_packet)?;

    let mut sealed_oracle_packet = SealedFreshOraclePacketV1 {
        schema: FRESH_ORACLE_SCHEMA.to_owned(),
        batch_id: config.batch_id.clone(),
        generation_seed_hex: seed.iter().map(|byte| format!("{byte:02x}")).collect(),
        generation_seed_commitment_sha256: seed_commitment.clone(),
        item_count: u64::from(config.requested_item_count),
        entries,
        sealed_oracle_sha256: String::new(),
        claim_boundary: "evaluator_only_secret_seed_hidden_parameters_and_exact_oracle_answers".to_owned(),
    };
    sealed_oracle_packet.sealed_oracle_sha256 = oracle_packet_digest(&sealed_oracle_packet)?;
    let item_set_sha256 = item_set_digest(&public_packet.tasks)?;

    let mut manifest = FreshChallengeManifestV1 {
        schema: FRESH_MANIFEST_SCHEMA.to_owned(),
        batch_id: config.batch_id.clone(),
        benchmark_id: config.benchmark_id.clone(),
        benchmark_version: config.benchmark_version.clone(),
        task_family: config.task_family.clone(),
        model_checkpoint_sha256: config.model_checkpoint_sha256.clone(),
        checkpoint_freeze_receipt_sha256: config.checkpoint_freeze_receipt_sha256.clone(),
        checkpoint_frozen_at_utc: config.checkpoint_frozen_at_utc.clone(),
        challenge_generated_at_utc: config.challenge_generated_at_utc.clone(),
        generator_source_sha256: config.generator_source_sha256.clone(),
        generator_binary_sha256: config.generator_binary_sha256.clone(),
        runtime_image_sha256: config.runtime_image_sha256.clone(),
        generator_config_sha256: config.generator_config_sha256.clone(),
        generation_seed_commitment_sha256: seed_commitment,
        seed_length_bits: u64::try_from(seed.len())
            .map_err(|_| "seed length overflows u64")? * 8,
        item_count: u64::from(config.requested_item_count),
        public_packet_sha256: public_packet.public_packet_sha256.clone(),
        sealed_oracle_sha256: sealed_oracle_packet.sealed_oracle_sha256.clone(),
        item_set_sha256,
        manifest_sha256: String::new(),
        claim_boundary: "checkpoint_time_ordering_and_fresh_instance_generation_claims_require_independent_trust_receipts".to_owned(),
    };
    manifest.manifest_sha256 = manifest_digest(&manifest)?;
    Ok(FreshChallengeBundleV1 {
        public_packet,
        sealed_oracle_packet,
        manifest,
    })
}

/// Independently regenerate a supplied bundle from the evaluator-held seed.
/// This checks deterministic consistency and oracle correctness; it does not
/// authenticate the checkpoint freeze receipt or generator issuer.
pub fn audit_fresh_challenge_bundle(
    config: &FreshChallengeConfigV1,
    seed: &[u8],
    bundle: &FreshChallengeBundleV1,
) -> Result<FreshChallengeAuditV1, String> {
    let expected = generate_fresh_challenge_bundle(config, seed)?;
    let supplied_public = serde_json::to_vec(&bundle.public_packet)
        .map_err(|error| format!("cannot serialize supplied public packet: {error}"))?;
    let expected_public = serde_json::to_vec(&expected.public_packet)
        .map_err(|error| format!("cannot serialize expected public packet: {error}"))?;
    let supplied_oracle = serde_json::to_vec(&bundle.sealed_oracle_packet)
        .map_err(|error| format!("cannot serialize supplied oracle packet: {error}"))?;
    let expected_oracle = serde_json::to_vec(&expected.sealed_oracle_packet)
        .map_err(|error| format!("cannot serialize expected oracle packet: {error}"))?;
    let supplied_manifest = serde_json::to_vec(&bundle.manifest)
        .map_err(|error| format!("cannot serialize supplied manifest: {error}"))?;
    let expected_manifest = serde_json::to_vec(&expected.manifest)
        .map_err(|error| format!("cannot serialize expected manifest: {error}"))?;
    if supplied_public != expected_public
        || supplied_oracle != expected_oracle
        || supplied_manifest != expected_manifest
    {
        return Err("fresh challenge bundle differs from deterministic regeneration".to_owned());
    }
    if public_packet_digest(&bundle.public_packet)? != bundle.public_packet.public_packet_sha256
        || oracle_packet_digest(&bundle.sealed_oracle_packet)?
            != bundle.sealed_oracle_packet.sealed_oracle_sha256
        || manifest_digest(&bundle.manifest)? != bundle.manifest.manifest_sha256
    {
        return Err("fresh challenge bundle content digest mismatch".to_owned());
    }
    let mut all_nonsingular = true;
    let mut all_answers_exact = true;
    for entry in &bundle.sealed_oracle_packet.entries {
        let determinant = determinant(&entry.parameters);
        all_nonsingular &= determinant.abs() >= config.minimum_absolute_determinant;
        let (b1, b2) = compute_rhs(&entry.parameters, &entry.solution)?;
        all_answers_exact &= b1 == entry.parameters.b1 && b2 == entry.parameters.b2;
        let parameter_bytes = serde_json::to_vec(&entry.parameters)
            .map_err(|error| format!("cannot serialize hidden parameters: {error}"))?;
        all_answers_exact &= hash_domain(b"fresh-task-parameters.v1", &parameter_bytes)
            == entry.parameter_sha256;
        let answer_bytes = serde_json::to_vec(&entry.solution)
            .map_err(|error| format!("cannot serialize exact solution: {error}"))?;
        all_answers_exact &= hash_domain(b"fresh-task-answer.v1", &answer_bytes)
            == entry.answer_sha256;
    }
    let public_value = serde_json::to_value(&bundle.public_packet)
        .map_err(|error| format!("cannot serialize public packet: {error}"))?;
    let public_keys: BTreeSet<String> = public_value.as_object()
        .ok_or_else(|| "public packet must serialize to an object".to_owned())?
        .keys().cloned().collect();
    let expected_public_keys: BTreeSet<String> = [
        "schema",
        "batch_id",
        "benchmark_id",
        "benchmark_version",
        "task_family",
        "model_checkpoint_sha256",
        "generator_source_sha256",
        "generator_binary_sha256",
        "runtime_image_sha256",
        "generator_config_sha256",
        "checkpoint_frozen_at_utc",
        "challenge_generated_at_utc",
        "item_count",
        "tasks",
        "public_packet_sha256",
        "claim_boundary",
    ].iter().map(|value| (*value).to_owned()).collect();
    let tasks_have_only_public_fields = bundle.public_packet.tasks.iter().all(|task| {
        serde_json::to_value(task).ok().and_then(|value| {
            let keys: BTreeSet<String> = value.as_object()?.keys().cloned().collect();
            let expected: BTreeSet<String> =
                ["item_id", "task_text", "task_text_sha256"]
                    .iter().map(|value| (*value).to_owned()).collect();
            Some(keys == expected)
        }).unwrap_or(false)
    });
    let public_serialized = serde_json::to_string(&bundle.public_packet)
        .map_err(|error| format!("cannot serialize public packet: {error}"))?;
    let public_contains_no_seed_or_answers =
        public_keys == expected_public_keys
            && tasks_have_only_public_fields
            && !public_serialized.contains(&bundle.sealed_oracle_packet.generation_seed_hex)
            && !public_serialized.contains("generation_seed_hex")
            && !public_serialized.contains("solution")
            && !public_serialized.contains("parameters")
            && !public_serialized.contains("answer_sha256");
    if !all_nonsingular || !all_answers_exact || !public_contains_no_seed_or_answers {
        return Err("fresh challenge public/oracle separation or exact answer validation failed".to_owned());
    }
    Ok(FreshChallengeAuditV1 {
        schema: FRESH_AUDIT_SCHEMA.to_owned(),
        valid: true,
        deterministic_regeneration_matches: true,
        checkpoint_freeze_precedes_generation: true,
        public_packet_contains_no_seed_or_oracle_answers: true,
        all_matrices_are_nonsingular: all_nonsingular,
        all_answers_recompute_exactly: all_answers_exact,
        item_count: bundle.manifest.item_count,
        batch_id: bundle.manifest.batch_id.clone(),
        manifest_sha256: bundle.manifest.manifest_sha256.clone(),
        generation_seed_commitment_sha256: bundle.manifest.generation_seed_commitment_sha256.clone(),
        independently_verified_checkpoint_freeze: false,
        claim_boundary: "deterministic_regeneration_and_exact_oracle_checked;checkpoint_freeze_signature_and_generation_custody_not_authenticated".to_owned(),
        issues: vec![
            "checkpoint freeze receipt digest is present but not cryptographically authenticated".to_owned(),
            "fresh instances do not establish task-recipe novelty or absence from model training".to_owned(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn digest(label: &str) -> String {
        hash_domain(b"fresh-instance-test", label.as_bytes())
    }

    fn config() -> FreshChallengeConfigV1 {
        let mut config = FreshChallengeConfigV1 {
            schema: FRESH_CONFIG_SCHEMA.to_owned(),
            batch_id: "fresh-pilot-001".to_owned(),
            benchmark_id: "symthaea-fresh-instances".to_owned(),
            benchmark_version: "v1".to_owned(),
            task_family: "linear-system-2x2".to_owned(),
            requested_item_count: 25,
            matrix_coefficient_min: -9,
            matrix_coefficient_max: 9,
            solution_component_min: -9,
            solution_component_max: 9,
            minimum_absolute_determinant: 1,
            model_checkpoint_sha256: digest("checkpoint"),
            checkpoint_freeze_receipt_sha256: digest("freeze-receipt"),
            checkpoint_frozen_at_utc: "2026-10-31T18:00:00Z".to_owned(),
            challenge_generated_at_utc: "2026-10-31T18:05:00Z".to_owned(),
            generator_source_sha256: digest("generator-source"),
            generator_binary_sha256: digest("generator-binary"),
            runtime_image_sha256: digest("runtime"),
            generator_config_sha256: String::new(),
        };
        config.generator_config_sha256 = canonical_config_digest(&config).unwrap();
        config
    }

    fn seed() -> Vec<u8> {
        (0..32).map(|value| value as u8).collect()
    }

    #[test]
    fn strict_schemas_match_serialized_fixture_bundle_fields() {
        let config_schema: Value = serde_json::from_str(include_str!(
            "../schema/fresh-instance-config.schema.json"
        )).unwrap();
        let public_schema: Value = serde_json::from_str(include_str!(
            "../schema/fresh-instance-public-packet.schema.json"
        )).unwrap();
        let oracle_schema: Value = serde_json::from_str(include_str!(
            "../schema/fresh-instance-sealed-oracle.schema.json"
        )).unwrap();
        let manifest_schema: Value = serde_json::from_str(include_str!(
            "../schema/fresh-instance-manifest.schema.json"
        )).unwrap();
        let audit_schema: Value = serde_json::from_str(include_str!(
            "../schema/fresh-instance-audit.schema.json"
        )).unwrap();
        let config: FreshChallengeConfigV1 = serde_json::from_str(include_str!(
            "../fixtures/fresh-instance-linear-system-config.json"
        )).unwrap();
        let bundle = generate_fresh_challenge_bundle(&config, &seed()).unwrap();
        let audit = audit_fresh_challenge_bundle(&config, &seed(), &bundle).unwrap();

        for (schema, value, version) in [
            (&config_schema, serde_json::to_value(&config).unwrap(), FRESH_CONFIG_SCHEMA),
            (&public_schema, serde_json::to_value(&bundle.public_packet).unwrap(), FRESH_PUBLIC_SCHEMA),
            (&oracle_schema, serde_json::to_value(&bundle.sealed_oracle_packet).unwrap(), FRESH_ORACLE_SCHEMA),
            (&manifest_schema, serde_json::to_value(&bundle.manifest).unwrap(), FRESH_MANIFEST_SCHEMA),
            (&audit_schema, serde_json::to_value(&audit).unwrap(), FRESH_AUDIT_SCHEMA),
        ] {
            assert_eq!(schema["properties"]["schema"]["const"].as_str(), Some(version));
            let required: BTreeSet<String> = schema["required"].as_array().unwrap()
                .iter().map(|item| item.as_str().unwrap().to_owned()).collect();
            let properties: BTreeSet<String> = schema["properties"].as_object().unwrap()
                .keys().cloned().collect();
            let actual: BTreeSet<String> = value.as_object().unwrap().keys().cloned().collect();
            assert_eq!(required, properties);
            assert_eq!(actual, properties);
        }

        let public_task_properties: BTreeSet<String> =
            public_schema["properties"]["tasks"]["items"]["properties"]
                .as_object().unwrap().keys().cloned().collect();
        let actual_public_task: BTreeSet<String> = serde_json::to_value(
            &bundle.public_packet.tasks[0]
        ).unwrap().as_object().unwrap().keys().cloned().collect();
        assert_eq!(public_task_properties, actual_public_task);
        assert!(!public_task_properties.contains("solution"));
        assert!(!public_task_properties.contains("generation_seed_hex"));

        let first_oracle = &bundle.sealed_oracle_packet.entries[0];
        let oracle_entry_schema = &oracle_schema["properties"]["entries"]["items"]["properties"];
        let actual_parameters: BTreeSet<String> = serde_json::to_value(&first_oracle.parameters)
            .unwrap().as_object().unwrap().keys().cloned().collect();
        let expected_parameters: BTreeSet<String> = oracle_entry_schema["parameters"]["properties"]
            .as_object().unwrap().keys().cloned().collect();
        assert_eq!(actual_parameters, expected_parameters);
        let actual_solution: BTreeSet<String> = serde_json::to_value(&first_oracle.solution)
            .unwrap().as_object().unwrap().keys().cloned().collect();
        let expected_solution: BTreeSet<String> = oracle_entry_schema["solution"]["properties"]
            .as_object().unwrap().keys().cloned().collect();
        assert_eq!(actual_solution, expected_solution);
    }

    #[test]
    fn deterministic_generation_and_regeneration_are_identical() {
        let config = config();
        let first = generate_fresh_challenge_bundle(&config, &seed()).unwrap();
        let second = generate_fresh_challenge_bundle(&config, &seed()).unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        let audit = audit_fresh_challenge_bundle(&config, &seed(), &first).unwrap();
        assert!(audit.valid && audit.deterministic_regeneration_matches);
        assert!(!audit.independently_verified_checkpoint_freeze);
    }

    #[test]
    fn changing_secret_seed_changes_fresh_task_payloads() {
        let config = config();
        let first = generate_fresh_challenge_bundle(&config, &seed()).unwrap();
        let second = generate_fresh_challenge_bundle(&config, &[0xD3; 32]).unwrap();
        assert_ne!(
            first.public_packet.public_packet_sha256,
            second.public_packet.public_packet_sha256
        );
        assert_ne!(
            first.manifest.generation_seed_commitment_sha256,
            second.manifest.generation_seed_commitment_sha256
        );
    }

    #[test]
    fn public_packet_key_set_contains_only_inference_time_task_fields() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let value: serde_json::Value = serde_json::to_value(&bundle.public_packet).unwrap();
        let keys: BTreeSet<String> = value.as_object().unwrap().keys().cloned().collect();
        assert!(keys.contains("tasks"));
        assert!(!keys.contains("generation_seed_hex"));
        assert!(!keys.contains("sealed_oracle_packet"));
        for task in &bundle.public_packet.tasks {
            let task_value: serde_json::Value = serde_json::to_value(task).unwrap();
            let task_keys: BTreeSet<String> = task_value.as_object().unwrap().keys().cloned().collect();
            assert_eq!(
                task_keys,
                ["item_id", "task_text", "task_text_sha256"]
                    .iter().map(|value| (*value).to_owned()).collect()
            );
        }
    }

    #[test]
    fn public_packet_contains_no_seed_or_solution_fields() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        let serialized = serde_json::to_string(&bundle.public_packet).unwrap();
        assert!(!serialized.contains("generation_seed_hex"));
        assert!(!serialized.contains("solution"));
        assert!(!serialized.contains("a11"));
        assert!(!serialized.contains("answer_sha256"));
    }

    #[test]
    fn all_generated_matrices_are_nonsingular_and_oracles_recompute() {
        let bundle = generate_fresh_challenge_bundle(&config(), &seed()).unwrap();
        for entry in &bundle.sealed_oracle_packet.entries {
            assert_ne!(determinant(&entry.parameters), 0);
            let (b1, b2) = compute_rhs(&entry.parameters, &entry.solution).unwrap();
            assert_eq!(b1, entry.parameters.b1);
            assert_eq!(b2, entry.parameters.b2);
        }
    }

    #[test]
    fn generator_config_digest_is_bound_to_actual_config_fields() {
        let mut invalid = config();
        invalid.matrix_coefficient_max -= 1;
        assert!(generate_fresh_challenge_bundle(&invalid, &seed())
            .unwrap_err().contains("does not match canonical configuration"));
    }

    #[test]
    fn checkpoint_must_be_frozen_before_generation() {
        let mut invalid = config();
        invalid.checkpoint_frozen_at_utc = "2026-10-31T18:10:00Z".to_owned();
        assert!(generate_fresh_challenge_bundle(&invalid, &seed())
            .unwrap_err().contains("must precede"));
    }

    #[test]
    fn impossible_timestamps_and_short_seed_fail_closed() {
        let mut invalid = config();
        invalid.challenge_generated_at_utc = "2026-02-30T18:05:00Z".to_owned();
        assert!(generate_fresh_challenge_bundle(&invalid, &seed()).is_err());
        assert!(generate_fresh_challenge_bundle(&config(), &[1, 2, 3]).is_err());
    }

    #[test]
    fn duplicate_or_mutated_bundle_content_fails_regeneration_audit() {
        let config = config();
        let mut bundle = generate_fresh_challenge_bundle(&config, &seed()).unwrap();
        bundle.public_packet.tasks[0].task_text.push_str(" modified");
        assert!(audit_fresh_challenge_bundle(&config, &seed(), &bundle)
            .unwrap_err().contains("differs from deterministic regeneration"));
    }

    #[test]
    fn task_count_and_generation_bounds_fail_closed() {
        let mut invalid = config();
        invalid.requested_item_count = 0;
        assert!(generate_fresh_challenge_bundle(&invalid, &seed()).is_err());
        invalid = config();
        invalid.matrix_coefficient_min = invalid.matrix_coefficient_max;
        assert!(generate_fresh_challenge_bundle(&invalid, &seed()).is_err());
    }
}
