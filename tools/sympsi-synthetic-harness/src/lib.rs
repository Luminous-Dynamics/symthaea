#![forbid(unsafe_code)]

//! Offline synthetic qualification only. This crate is not an ESP detector,
//! does not ingest human data, and must never score real AIA/psi observations.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");
const FOUR_SEED_DOMAIN: &str = "sympsi.synthetic.v0.1";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: String,
    pub synthetic_only: bool,
    pub seed: u64,
    pub alpha: f64,
    pub null_campaigns: u64,
    pub power_campaigns: u64,
    pub optional_stopping_looks: usize,
    pub confidence_z: f64,
    pub profiles: std::collections::BTreeMap<String, ProfileSpec>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSpec {
    pub choice_count: usize,
    pub analysis_trials: usize,
    pub smallest_effect_increment: f64,
    pub planted_hit_rate: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct NullCalibration {
    pub campaigns: u64,
    pub analysis_trials: usize,
    pub critical_hits: usize,
    pub empirical_false_positive_count: u64,
    pub empirical_false_positive_rate: f64,
    pub empirical_false_positive_wilson_99: Interval,
    pub exact_discrete_type_i_rate: f64,
    pub nominal_alpha: f64,
    pub monte_carlo_within_four_standard_errors: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct PowerCalibration {
    pub campaigns: u64,
    pub assumed_hit_rate: f64,
    pub absolute_lift_over_chance: f64,
    pub exact_binomial_power: f64,
    pub empirical_rejection_count: u64,
    pub empirical_power: f64,
    pub empirical_power_wilson_99: Interval,
    pub monte_carlo_within_four_standard_errors: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct SequentialLookCalibration {
    pub campaigns: u64,
    pub looks: usize,
    pub alpha_per_look: f64,
    pub empirical_familywise_false_positive_count: u64,
    pub empirical_familywise_false_positive_rate: f64,
    pub empirical_familywise_false_positive_wilson_99: Interval,
    pub nominal_fixed_sample_alpha: f64,
    pub inflation_multiple: f64,
    pub demonstrates_optional_stopping_inflation: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct PlantedEffect {
    pub planted_hit_rate: f64,
    pub synthetic_trials: usize,
    pub observed_hits: usize,
    pub observed_hit_rate: f64,
    pub critical_hits: usize,
    pub rejects_null: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ControlResult {
    pub id: String,
    pub expected_disposition: String,
    pub actual_disposition: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReceiptBody {
    pub receipt_schema_version: String,
    pub run_id: String,
    pub tool_version: String,
    pub git_head: String,
    pub synthetic_only: bool,
    pub measurement_class: String,
    pub claim_ceiling: String,
    pub profile: String,
    pub choice_count: usize,
    pub chance_rate: f64,
    pub smallest_effect_increment: f64,
    pub smallest_effect_hit_rate: f64,
    pub alpha: f64,
    pub manifest_sha256: String,
    pub seed_ids: Vec<String>,
    pub null_calibration: NullCalibration,
    pub power_calibration: PowerCalibration,
    pub sequential_look_calibration: SequentialLookCalibration,
    pub planted_effect: PlantedEffect,
    pub controls: Vec<ControlResult>,
    pub all_controls_passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReceiptEnvelope {
    pub receipt: ReceiptBody,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrialReceipt {
    pub trial_id: String,
    pub independent_unit_id: String,
    pub target_index: usize,
    pub response_index: usize,
    pub response_payload: String,
    pub response_commit_sequence: u64,
    pub target_reveal_sequence: u64,
    pub target_manifest_sha256: String,
    pub response_digest_sha256: String,
    pub status: String,
    pub analyst_can_read_target: bool,
    pub runner_can_read_target: bool,
    pub side_channel_target_hint: Option<usize>,
    pub synthetic_only: bool,
}

#[derive(Clone, Debug)]
pub struct ProtocolGuard {
    pub stopping_policy: String,
    pub posthoc_exclusions: usize,
    pub expected_trial_count: usize,
    pub assumes_independent_units: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Rejection {
    pub code: &'static str,
}

/// Immutable inputs for deriving a target from a verified public beacon.
///
/// `verified_randomness` must come from an independently pinned successful
/// beacon verifier. The context is not evidence of chronology or verification.
#[derive(Clone, Copy, Debug)]
pub struct TargetMappingContext<'a> {
    pub protocol_id: &'a str,
    pub campaign_id: &'a str,
    pub trial_id: &'a str,
    pub chain_hash: &'a [u8; 32],
    pub scheme_id: &'a str,
    pub response_commitment: &'a [u8; 32],
    pub round: u64,
    pub verified_randomness: &'a [u8; 32],
}

/// Derive a 4- or 5-way target from *already verified* public beacon randomness.
///
/// This function does not fetch or verify a beacon, authenticate a timestamp,
/// or prove the response preceded the beacon. Those are separate trust gates.
pub fn derive_target_index_from_verified_beacon(
    choice_count: usize,
    context: &TargetMappingContext<'_>,
) -> Result<usize, String> {
    if choice_count != 4 && choice_count != 5 {
        return Err("only four-choice and five-choice targets are supported".into());
    }
    if context.protocol_id.trim().is_empty()
        || context.campaign_id.trim().is_empty()
        || context.trial_id.trim().is_empty()
        || context.scheme_id.trim().is_empty()
        || context.round == 0
    {
        return Err("protocol, campaign, trial, scheme and nonzero round are required".into());
    }

    let threshold = two_to_256_mod(choice_count);
    let mut counter = 0_u32;
    loop {
        let digest = target_mapping_digest(context, counter)?;
        if !digest_is_below_small_integer(&digest, threshold) {
            return Ok(digest_modulo(&digest, choice_count));
        }
        counter = counter
            .checked_add(1)
            .ok_or_else(|| "target mapping rejection counter exhausted".to_string())?;
    }
}

fn target_mapping_digest(
    context: &TargetMappingContext<'_>,
    counter: u32,
) -> Result<[u8; 32], String> {
    let mut bytes = b"sympsi.aia.target.v1\0".to_vec();
    append_length_prefixed(&mut bytes, context.protocol_id.as_bytes())?;
    append_length_prefixed(&mut bytes, context.campaign_id.as_bytes())?;
    append_length_prefixed(&mut bytes, context.trial_id.as_bytes())?;
    append_length_prefixed(&mut bytes, context.chain_hash)?;
    append_length_prefixed(&mut bytes, context.scheme_id.as_bytes())?;
    bytes.extend_from_slice(&context.round.to_be_bytes());
    append_length_prefixed(&mut bytes, context.response_commitment)?;
    append_length_prefixed(&mut bytes, context.verified_randomness)?;
    bytes.extend_from_slice(&counter.to_be_bytes());
    Ok(Sha256::digest(&bytes).into())
}

fn append_length_prefixed(output: &mut Vec<u8>, field: &[u8]) -> Result<(), String> {
    let length = u32::try_from(field.len())
        .map_err(|_| "target-mapping field exceeds u32 length".to_string())?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(field);
    Ok(())
}

fn two_to_256_mod(modulus: usize) -> usize {
    (0..256).fold(1 % modulus, |remainder, _| (remainder * 2) % modulus)
}

fn digest_is_below_small_integer(digest: &[u8; 32], threshold: usize) -> bool {
    digest[..31].iter().all(|byte| *byte == 0) && usize::from(digest[31]) < threshold
}

fn digest_modulo(digest: &[u8; 32], modulus: usize) -> usize {
    digest.iter().fold(0usize, |remainder, byte| {
        (remainder * 256 + usize::from(*byte)) % modulus
    })
}

pub fn run_profile(
    profile_name: &str,
    git_head: &str,
    manifest_bytes: &[u8],
) -> Result<ReceiptEnvelope, String> {
    validate_git_head(git_head)?;
    let manifest: Manifest = serde_json::from_slice(manifest_bytes)
        .map_err(|error| format!("manifest parse failed: {error}"))?;
    validate_manifest(&manifest)?;

    let profile = manifest
        .profiles
        .get(profile_name)
        .ok_or_else(|| format!("unknown profile: {profile_name}"))?;
    validate_profile(profile)?;

    let choice_count = profile.choice_count;
    let chance_rate = 1.0 / choice_count as f64;
    let sesoi_rate = chance_rate + profile.smallest_effect_increment;
    let manifest_sha256 = sha256_hex(manifest_bytes);
    let n = profile.analysis_trials;

    let null_distribution = BinomialDistribution::new(n, chance_rate)?;
    let critical_hits = null_distribution.critical_hits(manifest.alpha)?;
    let exact_type_i = null_distribution.upper_tail(critical_hits);

    let null_seed = derive_seed(manifest.seed, "null-calibration");
    let mut null_rng = SplitMix64::new(null_seed);
    let mut false_positives = 0_u64;
    for _ in 0..manifest.null_campaigns {
        let hits = null_distribution.sample(&mut null_rng);
        if hits >= critical_hits {
            false_positives += 1;
        }
    }
    let empirical_false_positive_rate = false_positives as f64 / manifest.null_campaigns as f64;
    let null_interval = wilson_interval(
        false_positives,
        manifest.null_campaigns,
        manifest.confidence_z,
    );
    let null_mc_tolerance = 4.0
        * (exact_type_i * (1.0 - exact_type_i) / manifest.null_campaigns as f64).sqrt()
        + 1.0 / manifest.null_campaigns as f64;
    let null_mc_pass = (empirical_false_positive_rate - exact_type_i).abs() <= null_mc_tolerance;

    let power_distribution = BinomialDistribution::new(n, sesoi_rate)?;
    let exact_power = power_distribution.upper_tail(critical_hits);
    let power_seed = derive_seed(manifest.seed, "sesoi-power-calibration");
    let mut power_rng = SplitMix64::new(power_seed);
    let mut power_rejections = 0_u64;
    for _ in 0..manifest.power_campaigns {
        if power_distribution.sample(&mut power_rng) >= critical_hits {
            power_rejections += 1;
        }
    }
    let empirical_power = power_rejections as f64 / manifest.power_campaigns as f64;
    let power_interval = wilson_interval(
        power_rejections,
        manifest.power_campaigns,
        manifest.confidence_z,
    );
    let power_mc_tolerance = 4.0
        * (exact_power * (1.0 - exact_power) / manifest.power_campaigns as f64).sqrt()
        + 1.0 / manifest.power_campaigns as f64;
    let power_mc_pass = (empirical_power - exact_power).abs() <= power_mc_tolerance;

    let sequential_seed = derive_seed(manifest.seed, "ten-look-optional-stopping-null");
    let sequential_look_calibration = simulate_optional_stopping_null(
        n,
        chance_rate,
        manifest.alpha,
        manifest.null_campaigns,
        manifest.optional_stopping_looks,
        sequential_seed,
        manifest.confidence_z,
    )?;

    let planted_seed = derive_seed(manifest.seed, "strong-planted-effect");
    let mut planted_rng = SplitMix64::new(planted_seed);
    let planted_distribution = BinomialDistribution::new(n, profile.planted_hit_rate)?;
    let planted_hits = planted_distribution.sample(&mut planted_rng);
    let planted = PlantedEffect {
        planted_hit_rate: profile.planted_hit_rate,
        synthetic_trials: n,
        observed_hits: planted_hits,
        observed_hit_rate: planted_hits as f64 / n as f64,
        critical_hits,
        rejects_null: planted_hits >= critical_hits,
    };

    let mut controls = run_controls(
        choice_count,
        &manifest_sha256,
        profile_name,
        critical_hits,
        &null_distribution,
        &manifest,
    );
    controls.push(control(
        "ten_look_optional_stopping_inflation",
        "SEQUENTIAL_NULL_FWER_EXCEEDS_FIXED_ALPHA",
        if sequential_look_calibration.demonstrates_optional_stopping_inflation {
            "SEQUENTIAL_NULL_FWER_EXCEEDS_FIXED_ALPHA"
        } else {
            "SEQUENTIAL_NULL_FWER_NOT_DEMONSTRABLY_INFLATED"
        }
        .into(),
        "Under synthetic null outcomes, ten unadjusted fixed-alpha looks inflate campaign-level false positives; this is a researcher-flexibility diagnostic, not a psi test.",
    ));
    let all_controls_passed = controls.iter().all(|control| control.passed)
        && null_mc_pass
        && power_mc_pass
        && planted.rejects_null
        && sequential_look_calibration.demonstrates_optional_stopping_inflation;

    let run_id = sha256_hex(
        format!(
            "{FOUR_SEED_DOMAIN}\0{profile_name}\0{}\0{}\0{}",
            manifest.seed,
            manifest_sha256,
            git_head.to_ascii_lowercase()
        )
        .as_bytes(),
    );
    let receipt = ReceiptBody {
        receipt_schema_version: "sympsi.synthetic-receipt.v0.1".into(),
        run_id,
        tool_version: TOOL_VERSION.into(),
        git_head: git_head.to_ascii_lowercase(),
        synthetic_only: true,
        measurement_class: "SIMULATED".into(),
        claim_ceiling: "HARNESS_QUALIFICATION_ONLY_NOT_PSI_EVIDENCE".into(),
        profile: profile_name.into(),
        choice_count,
        chance_rate,
        smallest_effect_increment: profile.smallest_effect_increment,
        smallest_effect_hit_rate: sesoi_rate,
        alpha: manifest.alpha,
        manifest_sha256,
        seed_ids: vec![
            "null-calibration".into(),
            "sesoi-power-calibration".into(),
            "ten-look-optional-stopping-null".into(),
            "strong-planted-effect".into(),
            "leakage-and-custody-fixtures".into(),
            "label-permutation".into(),
        ],
        null_calibration: NullCalibration {
            campaigns: manifest.null_campaigns,
            analysis_trials: n,
            critical_hits,
            empirical_false_positive_count: false_positives,
            empirical_false_positive_rate,
            empirical_false_positive_wilson_99: null_interval,
            exact_discrete_type_i_rate: exact_type_i,
            nominal_alpha: manifest.alpha,
            monte_carlo_within_four_standard_errors: null_mc_pass,
        },
        power_calibration: PowerCalibration {
            campaigns: manifest.power_campaigns,
            assumed_hit_rate: sesoi_rate,
            absolute_lift_over_chance: profile.smallest_effect_increment,
            exact_binomial_power: exact_power,
            empirical_rejection_count: power_rejections,
            empirical_power,
            empirical_power_wilson_99: power_interval,
            monte_carlo_within_four_standard_errors: power_mc_pass,
        },
        sequential_look_calibration,
        planted_effect: planted,
        controls,
        all_controls_passed,
    };
    let canonical_body = serde_json::to_vec(&receipt)
        .map_err(|error| format!("receipt serialization failed: {error}"))?;
    let receipt_sha256 = sha256_hex(&canonical_body);
    Ok(ReceiptEnvelope {
        receipt,
        receipt_sha256,
    })
}

fn validate_git_head(git_head: &str) -> Result<(), String> {
    if git_head.len() != 40 || !git_head.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("git head must be an exact 40-character hexadecimal commit SHA".into());
    }
    Ok(())
}

fn validate_manifest(manifest: &Manifest) -> Result<(), String> {
    if manifest.schema_version != "sympsi.synthetic-manifest.v0.1" {
        return Err("unsupported manifest schema version".into());
    }
    if !manifest.synthetic_only {
        return Err("manifest must explicitly declare synthetic_only=true".into());
    }
    if !(0.0..0.05).contains(&manifest.alpha) {
        return Err("alpha must be greater than 0 and below 0.05".into());
    }
    if manifest.null_campaigns < 10_000 || manifest.power_campaigns < 1_000 {
        return Err("simulation campaign counts are below qualification minimums".into());
    }
    if !(2..=25).contains(&manifest.optional_stopping_looks) {
        return Err("optional-stopping diagnostic must use between 2 and 25 looks".into());
    }
    if !(1.96..=3.5).contains(&manifest.confidence_z) {
        return Err("confidence z-score is outside the supported range".into());
    }
    if manifest.profiles.len() != 2
        || !manifest.profiles.contains_key("four-choice")
        || !manifest.profiles.contains_key("five-choice")
    {
        return Err("manifest must contain exactly four-choice and five-choice profiles".into());
    }
    Ok(())
}

fn validate_profile(profile: &ProfileSpec) -> Result<(), String> {
    if profile.choice_count != 4 && profile.choice_count != 5 {
        return Err("only explicit four-choice and five-choice profiles are supported".into());
    }
    if profile.analysis_trials < 100 {
        return Err("analysis trial count is too small".into());
    }
    if !(0.0..0.1).contains(&profile.smallest_effect_increment) {
        return Err("smallest effect increment must be positive and below 0.1".into());
    }
    let chance = 1.0 / profile.choice_count as f64;
    if !(chance + profile.smallest_effect_increment..1.0).contains(&profile.planted_hit_rate) {
        return Err("planted hit rate must exceed the SESOI and remain below 1".into());
    }
    Ok(())
}

fn run_controls(
    choices: usize,
    manifest_sha256: &str,
    profile_name: &str,
    critical_hits: usize,
    null_distribution: &BinomialDistribution,
    manifest: &Manifest,
) -> Vec<ControlResult> {
    let expected_count = 24;
    let valid = make_trial_fixture(
        expected_count,
        choices,
        manifest_sha256,
        false,
        false,
        false,
    );
    let guard = ProtocolGuard {
        stopping_policy: "fixed_count".into(),
        posthoc_exclusions: 0,
        expected_trial_count: expected_count,
        assumes_independent_units: true,
    };
    let valid_result = validate_trial_bundle(&valid, choices, manifest_sha256, &guard);

    let leak = make_trial_fixture(expected_count, choices, manifest_sha256, false, true, false);
    let leak_result = validate_trial_bundle(&leak, choices, manifest_sha256, &guard);

    let mut duplicate = valid.clone();
    duplicate[1].trial_id = duplicate[0].trial_id.clone();
    let duplicate_result = validate_trial_bundle(&duplicate, choices, manifest_sha256, &guard);

    let mut altered_manifest = valid.clone();
    altered_manifest[0].target_manifest_sha256 = "0".repeat(64);
    let manifest_result =
        validate_trial_bundle(&altered_manifest, choices, manifest_sha256, &guard);

    let mut blinded_violation = valid.clone();
    blinded_violation[0].analyst_can_read_target = true;
    let blindness_result =
        validate_trial_bundle(&blinded_violation, choices, manifest_sha256, &guard);

    let mut runner_blindness_violation = valid.clone();
    runner_blindness_violation[0].runner_can_read_target = true;
    let runner_blindness_result = validate_trial_bundle(
        &runner_blindness_violation,
        choices,
        manifest_sha256,
        &guard,
    );

    let mut missing_receipt = valid.clone();
    missing_receipt.pop();
    let missing_result = validate_trial_bundle(&missing_receipt, choices, manifest_sha256, &guard);

    let mut response_tamper = valid.clone();
    response_tamper[0].response_digest_sha256 = "0".repeat(64);
    let response_tamper_result =
        validate_trial_bundle(&response_tamper, choices, manifest_sha256, &guard);

    let mut chronology_tamper = valid.clone();
    chronology_tamper[0].target_reveal_sequence = chronology_tamper[0].response_commit_sequence;
    let chronology_result =
        validate_trial_bundle(&chronology_tamper, choices, manifest_sha256, &guard);

    let optional_stopping = ProtocolGuard {
        stopping_policy: "optional_stopping".into(),
        posthoc_exclusions: 1,
        ..guard.clone()
    };
    let stopping_result =
        validate_trial_bundle(&valid, choices, manifest_sha256, &optional_stopping);

    let repeated_units =
        make_trial_fixture(expected_count, choices, manifest_sha256, true, false, false);
    let cluster_result = validate_trial_bundle(&repeated_units, choices, manifest_sha256, &guard);

    let permutation_pass = label_permutation_is_invariant(choices, manifest.seed, profile_name);
    let target_order_permutation_pass =
        target_order_permutation_recomputes(choices, manifest.seed, profile_name);
    let target_mapping_vectors_pass = target_mapping_known_answer_vectors();

    vec![
        control(
            "valid_bundle",
            "ACCEPT_VALID_SYNTHETIC_BUNDLE",
            disposition(valid_result, "ACCEPT_VALID_SYNTHETIC_BUNDLE"),
            "Valid, fixed-count, independent-unit synthetic receipts are accepted.",
        ),
        control(
            "direct_target_leak",
            "REJECT_KNOWN_DIRECT_TARGET_CHANNEL",
            disposition(leak_result, "REJECT_KNOWN_DIRECT_TARGET_CHANNEL"),
            "A deliberately planted side-channel field equal to the hidden target is detected. This is not a general unknown-leak detector.",
        ),
        control(
            "duplicate_trial_id",
            "REJECT_DUPLICATE_TRIAL_ID",
            disposition(duplicate_result, "REJECT_DUPLICATE_TRIAL_ID"),
            "Duplicate trial identifiers are rejected.",
        ),
        control(
            "target_manifest_tamper",
            "REJECT_TARGET_MANIFEST_MISMATCH",
            disposition(manifest_result, "REJECT_TARGET_MANIFEST_MISMATCH"),
            "A trial bound to a different target manifest is rejected.",
        ),
        control(
            "blinding_violation",
            "REJECT_TARGET_VISIBILITY",
            disposition(blindness_result, "REJECT_TARGET_VISIBILITY"),
            "An analyst-visible hidden target is rejected.",
        ),
        control(
            "runner_blinding_violation",
            "REJECT_TARGET_VISIBILITY",
            disposition(runner_blindness_result, "REJECT_TARGET_VISIBILITY"),
            "The inference runner must not have access to hidden target labels.",
        ),
        control(
            "missing_trial_receipt",
            "REJECT_TRIAL_COUNT_MISMATCH",
            disposition(missing_result, "REJECT_TRIAL_COUNT_MISMATCH"),
            "A missing member of the fixed trial set is rejected, not silently replaced.",
        ),
        control(
            "response_digest_tamper",
            "REJECT_RESPONSE_DIGEST_MISMATCH",
            disposition(response_tamper_result, "REJECT_RESPONSE_DIGEST_MISMATCH"),
            "A response digest that no longer matches the committed synthetic payload is rejected.",
        ),
        control(
            "response_target_chronology",
            "REJECT_RESPONSE_AFTER_TARGET",
            disposition(chronology_result, "REJECT_RESPONSE_AFTER_TARGET"),
            "A target reveal sequence that is not strictly after response commitment is rejected.",
        ),
        control(
            "optional_stopping",
            "REJECT_OPTIONAL_STOPPING",
            disposition(stopping_result, "REJECT_OPTIONAL_STOPPING"),
            "Outcome-dependent stopping or exclusions are rejected.",
        ),
        control(
            "repeated_units_under_iid",
            "REJECT_REPEATED_UNIT_UNDER_IID_ASSUMPTION",
            disposition(cluster_result, "REJECT_REPEATED_UNIT_UNDER_IID_ASSUMPTION"),
            "Repeated participant/session units cannot enter the iid binomial analysis path.",
        ),
        control(
            "label_permutation",
            "PRESERVE_SCORE_UNDER_BIJECTIVE_RECODING",
            if permutation_pass {
                "PRESERVE_SCORE_UNDER_BIJECTIVE_RECODING"
            } else {
                "PERMUTATION_INVARIANCE_BROKEN"
            }
            .into(),
            "A bijective relabeling applied to both targets and responses preserves the hit count.",
        ),
        control(
            "target_order_permutation",
            "PERMUTED_TARGET_COUNTS_AND_SCORE_RECOMPUTED",
            if target_order_permutation_pass {
                "PERMUTED_TARGET_COUNTS_AND_SCORE_RECOMPUTED"
            } else {
                "PERMUTATION_SCORE_RECOMPUTATION_FAILED"
            }
            .into(),
            "A deterministic shuffle preserves target-label counts and the separately recomputed score is identical.",
        ),
        control(
            "target_mapping_known_answer",
            "MATCH_FROZEN_FOUR_AND_FIVE_CHOICE_VECTORS",
            if target_mapping_vectors_pass {
                "MATCH_FROZEN_FOUR_AND_FIVE_CHOICE_VECTORS"
            } else {
                "TARGET_MAPPING_VECTOR_MISMATCH"
            }
            .into(),
            "Domain-separated SHA-256 mapping with rejection sampling matches independent known-answer vectors.",
        ),
        control(
            "critical_value",
            "EXACT_DISCRETE_TYPE_I_NOT_ABOVE_ALPHA",
            if null_distribution.upper_tail(critical_hits) <= manifest.alpha {
                "EXACT_DISCRETE_TYPE_I_NOT_ABOVE_ALPHA"
            } else {
                "CRITICAL_VALUE_INVALID"
            }
            .into(),
            "The one-sided exact-binomial critical value controls the discrete null tail at or below alpha.",
        ),
    ]
}

fn control(id: &str, expected: &str, actual: String, detail: &str) -> ControlResult {
    ControlResult {
        id: id.into(),
        expected_disposition: expected.into(),
        passed: actual == expected,
        actual_disposition: actual,
        detail: detail.into(),
    }
}

fn disposition(result: Result<(), Rejection>, accepted: &str) -> String {
    match result {
        Ok(()) => accepted.into(),
        Err(rejection) => rejection.code.into(),
    }
}

fn make_trial_fixture(
    count: usize,
    choices: usize,
    manifest_sha256: &str,
    repeated_units: bool,
    direct_leak: bool,
    analyst_visible: bool,
) -> Vec<TrialReceipt> {
    (0..count)
        .map(|index| {
            let trial_id = format!("synthetic-trial-{index:04}");
            let target_index = index % choices;
            let response_index = (index + 1) % choices;
            let response_payload = format!("{trial_id}|{response_index}|fixture-v0.1");
            TrialReceipt {
                trial_id,
                independent_unit_id: if repeated_units {
                    "same-participant-session".into()
                } else {
                    format!("participant-session-{index:04}")
                },
                target_index,
                response_index,
                response_payload: response_payload.clone(),
                response_commit_sequence: (index as u64) * 2,
                target_reveal_sequence: (index as u64) * 2 + 1,
                target_manifest_sha256: manifest_sha256.into(),
                response_digest_sha256: sha256_hex(response_payload.as_bytes()),
                status: "VALID".into(),
                analyst_can_read_target: analyst_visible,
                runner_can_read_target: false,
                side_channel_target_hint: if direct_leak {
                    Some(target_index)
                } else {
                    None
                },
                synthetic_only: true,
            }
        })
        .collect()
}

fn validate_trial_bundle(
    trials: &[TrialReceipt],
    choices: usize,
    expected_manifest_sha256: &str,
    guard: &ProtocolGuard,
) -> Result<(), Rejection> {
    if guard.stopping_policy != "fixed_count" || guard.posthoc_exclusions != 0 {
        return Err(Rejection {
            code: "REJECT_OPTIONAL_STOPPING",
        });
    }
    if trials.len() != guard.expected_trial_count {
        return Err(Rejection {
            code: "REJECT_TRIAL_COUNT_MISMATCH",
        });
    }

    let mut trial_ids = BTreeSet::new();
    let mut independent_units = BTreeSet::new();
    for trial in trials {
        if !trial_ids.insert(trial.trial_id.as_str()) {
            return Err(Rejection {
                code: "REJECT_DUPLICATE_TRIAL_ID",
            });
        }
        if trial.status != "VALID" || !trial.synthetic_only {
            return Err(Rejection {
                code: "REJECT_UNKNOWN_OR_NON_SYNTHETIC_STATUS",
            });
        }
        if trial.target_manifest_sha256 != expected_manifest_sha256 {
            return Err(Rejection {
                code: "REJECT_TARGET_MANIFEST_MISMATCH",
            });
        }
        if trial.target_index >= choices || trial.response_index >= choices {
            return Err(Rejection {
                code: "REJECT_CHOICE_OUT_OF_RANGE",
            });
        }
        if trial.response_commit_sequence >= trial.target_reveal_sequence {
            return Err(Rejection {
                code: "REJECT_RESPONSE_AFTER_TARGET",
            });
        }
        let expected_payload = format!("{}|{}|fixture-v0.1", trial.trial_id, trial.response_index);
        if trial.response_payload != expected_payload {
            return Err(Rejection {
                code: "REJECT_RESPONSE_PAYLOAD_MISMATCH",
            });
        }
        if !is_sha256_hex(&trial.response_digest_sha256) {
            return Err(Rejection {
                code: "REJECT_RESPONSE_DIGEST_INVALID",
            });
        }
        if sha256_hex(trial.response_payload.as_bytes()) != trial.response_digest_sha256 {
            return Err(Rejection {
                code: "REJECT_RESPONSE_DIGEST_MISMATCH",
            });
        }
        if trial.analyst_can_read_target || trial.runner_can_read_target {
            return Err(Rejection {
                code: "REJECT_TARGET_VISIBILITY",
            });
        }
        if trial
            .side_channel_target_hint
            .is_some_and(|hint| hint == trial.target_index)
        {
            return Err(Rejection {
                code: "REJECT_KNOWN_DIRECT_TARGET_CHANNEL",
            });
        }
        if trial.independent_unit_id.trim().is_empty() {
            return Err(Rejection {
                code: "REJECT_EMPTY_INDEPENDENCE_UNIT",
            });
        }
        if guard.assumes_independent_units
            && !independent_units.insert(trial.independent_unit_id.as_str())
        {
            return Err(Rejection {
                code: "REJECT_REPEATED_UNIT_UNDER_IID_ASSUMPTION",
            });
        }
    }
    Ok(())
}

fn target_mapping_known_answer_vectors() -> bool {
    let chain_hash = [
        0x52, 0xdb, 0x9b, 0xa7, 0x0e, 0x0c, 0xc0, 0xf6, 0xea, 0xf7, 0x80, 0x3d, 0xd0, 0x74, 0x47,
        0xa1, 0xf5, 0x47, 0x77, 0x35, 0xfd, 0x3f, 0x66, 0x17, 0x92, 0xba, 0x94, 0x60, 0x0c, 0x84,
        0xe9, 0x71,
    ];
    let randomness = [
        0xfe, 0x29, 0x0b, 0xec, 0xa1, 0x08, 0x72, 0xef, 0x2f, 0xb1, 0x64, 0xd2, 0xaa, 0x44, 0x42,
        0xde, 0x45, 0x66, 0x18, 0x3e, 0xc5, 0x1c, 0x56, 0xff, 0x3c, 0xd6, 0x03, 0xd9, 0x30, 0xe5,
        0x4f, 0xdd,
    ];
    let response_commitment = Sha256::digest(b"sympsi-response-commitment-000042").into();
    let context = TargetMappingContext {
        protocol_id: "sympsi-aia-01-v0.1",
        campaign_id: "campaign-0001",
        trial_id: "trial-000042",
        chain_hash: &chain_hash,
        scheme_id: "bls-unchained-g1-rfc9380",
        response_commitment: &response_commitment,
        round: 1000,
        verified_randomness: &randomness,
    };
    let expected_digest = [
        0xcd, 0x4e, 0xd4, 0xb3, 0xa8, 0xab, 0x1d, 0xa6, 0x18, 0x5e, 0xa3, 0xa5, 0x35, 0xbd, 0x6b,
        0x87, 0xb7, 0x18, 0x62, 0xf6, 0x06, 0xa9, 0xd0, 0x3c, 0xc5, 0xa3, 0x88, 0xe8, 0xb8, 0x54,
        0xbd, 0xe6,
    ];
    let actual_digest = target_mapping_digest(&context, 0);
    let four = derive_target_index_from_verified_beacon(4, &context);
    let five = derive_target_index_from_verified_beacon(5, &context);
    actual_digest == Ok(expected_digest) && four == Ok(2) && five == Ok(2)
}

fn target_order_permutation_recomputes(choices: usize, seed: u64, profile: &str) -> bool {
    let count = choices * 37;
    let mut targets: Vec<usize> = (0..count).map(|index| index % choices).collect();
    let responses: Vec<usize> = (0..count).map(|index| (index + 1) % choices).collect();
    let mut rng = SplitMix64::new(derive_seed(seed, &format!("target-order-{profile}")));
    for index in (1..targets.len()).rev() {
        let other = rng.below(index + 1);
        targets.swap(index, other);
    }

    let mut counts = vec![0usize; choices];
    for target in &targets {
        counts[*target] += 1;
    }
    let recomputed_hits = (0..count)
        .filter(|index| targets[*index] == responses[*index])
        .count();
    let independent_hits = (0..count)
        .map(|index| usize::from(targets[index] == responses[index]))
        .sum::<usize>();
    counts.iter().all(|count| *count == 37) && recomputed_hits == independent_hits
}

fn label_permutation_is_invariant(choices: usize, seed: u64, profile: &str) -> bool {
    let count = choices * 31;
    let targets: Vec<usize> = (0..count).map(|index| index % choices).collect();
    let responses: Vec<usize> = (0..count)
        .map(|index| {
            if index % 5 == 0 {
                index % choices
            } else {
                (index + 1) % choices
            }
        })
        .collect();
    let original_hits = (0..count)
        .filter(|index| targets[*index] == responses[*index])
        .count();

    let mut rng = SplitMix64::new(derive_seed(seed, &format!("label-permutation-{profile}")));
    let mut mapping: Vec<usize> = (0..choices).collect();
    for index in (1..choices).rev() {
        let other = rng.below(index + 1);
        mapping.swap(index, other);
    }
    let recoded_hits = (0..count)
        .filter(|index| mapping[targets[*index]] == mapping[responses[*index]])
        .count();
    original_hits == recoded_hits
}

#[derive(Clone)]
struct BinomialDistribution {
    n: usize,
    cdf: Vec<f64>,
    upper_tails: Vec<f64>,
}

impl BinomialDistribution {
    fn new(n: usize, p: f64) -> Result<Self, String> {
        if n == 0 || !(0.0..1.0).contains(&p) {
            return Err("binomial distribution requires n>0 and 0<p<1".into());
        }
        let q = 1.0 - p;
        let mode = (((n + 1) as f64) * p).floor() as usize;
        let log_probability = log_factorial(n) - log_factorial(mode) - log_factorial(n - mode)
            + mode as f64 * p.ln()
            + (n - mode) as f64 * q.ln();
        let mut masses = vec![0.0_f64; n + 1];
        masses[mode] = log_probability.exp();

        for k in (1..=mode).rev() {
            masses[k - 1] = masses[k] * (k as f64 / (n - k + 1) as f64) * (q / p);
        }
        for k in mode..n {
            masses[k + 1] = masses[k] * ((n - k) as f64 / (k + 1) as f64) * (p / q);
        }

        let total: f64 = masses.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err("binomial normalization failed".into());
        }
        for mass in &mut masses {
            *mass /= total;
        }

        let mut cdf = vec![0.0_f64; n + 1];
        let mut running = 0.0;
        for (index, mass) in masses.iter().enumerate() {
            running += mass;
            cdf[index] = running;
        }
        cdf[n] = 1.0;

        let mut upper_tails = vec![0.0_f64; n + 1];
        running = 0.0;
        for index in (0..=n).rev() {
            running += masses[index];
            upper_tails[index] = running;
        }
        Ok(Self {
            n,
            cdf,
            upper_tails,
        })
    }

    fn critical_hits(&self, alpha: f64) -> Result<usize, String> {
        (0..=self.n)
            .find(|&hits| self.upper_tail(hits) <= alpha)
            .ok_or_else(|| "no exact-binomial critical value exists".into())
    }

    fn upper_tail(&self, hits: usize) -> f64 {
        if hits > self.n {
            0.0
        } else {
            self.upper_tails[hits]
        }
    }

    fn sample(&self, rng: &mut SplitMix64) -> usize {
        let value = rng.unit_interval();
        self.cdf
            .partition_point(|cumulative| *cumulative < value)
            .min(self.n)
    }
}

fn log_factorial(n: usize) -> f64 {
    (2..=n).map(|value| (value as f64).ln()).sum()
}

fn simulate_optional_stopping_null(
    total_trials: usize,
    chance_rate: f64,
    alpha_per_look: f64,
    campaigns: u64,
    looks: usize,
    seed: u64,
    confidence_z: f64,
) -> Result<SequentialLookCalibration, String> {
    if looks < 2 || looks > total_trials {
        return Err("sequential diagnostic requires 2..=total_trials looks".into());
    }

    let base_block = total_trials / looks;
    let remainder = total_trials % looks;
    let mut block_distributions = Vec::with_capacity(looks);
    let mut critical_hits = Vec::with_capacity(looks);
    let mut cumulative_trials = 0usize;

    for look in 0..looks {
        let block_size = base_block + if look < remainder { 1 } else { 0 };
        let block_distribution = BinomialDistribution::new(block_size, chance_rate)?;
        cumulative_trials += block_size;
        let cumulative_distribution = BinomialDistribution::new(cumulative_trials, chance_rate)?;
        critical_hits.push(cumulative_distribution.critical_hits(alpha_per_look)?);
        block_distributions.push(block_distribution);
    }

    let mut rng = SplitMix64::new(seed);
    let mut false_positives = 0_u64;
    for _ in 0..campaigns {
        let mut cumulative_hits = 0usize;
        let mut crossed_at_least_one_look = false;
        for (look, distribution) in block_distributions.iter().enumerate() {
            cumulative_hits += distribution.sample(&mut rng);
            if cumulative_hits >= critical_hits[look] {
                crossed_at_least_one_look = true;
                break;
            }
        }
        if crossed_at_least_one_look {
            false_positives += 1;
        }
    }

    let rate = false_positives as f64 / campaigns as f64;
    Ok(SequentialLookCalibration {
        campaigns,
        looks,
        alpha_per_look,
        empirical_familywise_false_positive_count: false_positives,
        empirical_familywise_false_positive_rate: rate,
        empirical_familywise_false_positive_wilson_99: wilson_interval(
            false_positives,
            campaigns,
            confidence_z,
        ),
        nominal_fixed_sample_alpha: alpha_per_look,
        inflation_multiple: rate / alpha_per_look,
        demonstrates_optional_stopping_inflation: rate > alpha_per_look * 1.5,
    })
}

fn wilson_interval(successes: u64, trials: u64, z: f64) -> Interval {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    Interval {
        lower: (center - margin).max(0.0),
        upper: (center + margin).min(1.0),
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn derive_seed(master: u64, domain: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(FOUR_SEED_DOMAIN.as_bytes());
    hasher.update([0]);
    hasher.update(master.to_be_bytes());
    hasher.update([0]);
    hasher.update(domain.as_bytes());
    let digest = hasher.finalize();
    u64::from_be_bytes(digest[..8].try_into().expect("fixed digest prefix"))
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit_interval(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn below(&mut self, upper: usize) -> usize {
        let upper = upper as u64;
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % upper) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_bytes() -> Vec<u8> {
        include_bytes!("../fixtures/manifest-v0.1.json").to_vec()
    }

    #[test]
    fn ten_unadjusted_null_looks_inflate_familywise_false_positives() {
        for (trials, chance) in [(7300, 0.25), (6300, 0.20)] {
            let result = simulate_optional_stopping_null(
                trials,
                chance,
                0.005,
                100_000,
                10,
                20261115,
                2.5758293035489004,
            )
            .unwrap();
            assert!(result.empirical_familywise_false_positive_rate > 0.0075);
            assert!(result.inflation_multiple > 1.5);
            assert!(result.demonstrates_optional_stopping_inflation);
        }
    }

    #[test]
    fn manifest_with_synthetic_only_false_is_rejected() {
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes()).unwrap();
        let mut invalid = manifest;
        invalid.synthetic_only = false;
        assert!(validate_manifest(&invalid).is_err());
    }

    #[test]
    fn full_five_choice_profile_emits_deterministic_synthetic_receipt() {
        let bytes = manifest_bytes();
        let head = "a".repeat(40);
        let first = run_profile("five-choice", &head, &bytes).unwrap();
        let second = run_profile("five-choice", &head, &bytes).unwrap();
        assert!(first.receipt.all_controls_passed);
        assert_eq!(first.receipt_sha256, second.receipt_sha256);
        assert_eq!(first.receipt.profile, "five-choice");
        assert_eq!(first.receipt.chance_rate, 0.20);
        assert!(first.receipt.synthetic_only);
        assert_eq!(first.receipt.measurement_class, "SIMULATED");
    }

    #[test]
    fn manifest_is_explicitly_synthetic_and_has_both_choice_profiles() {
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes()).unwrap();
        validate_manifest(&manifest).unwrap();
        assert_eq!(manifest.profiles["four-choice"].choice_count, 4);
        assert_eq!(manifest.profiles["five-choice"].choice_count, 5);
    }

    #[test]
    fn four_and_five_choice_baselines_are_not_interchangeable() {
        assert_eq!(1.0 / 4.0, 0.25);
        assert_eq!(1.0 / 5.0, 0.20);
        assert_ne!(1.0 / 4.0, 1.0 / 5.0);
    }

    #[test]
    fn binomial_distribution_normalizes_and_has_expected_critical_values() {
        let four = BinomialDistribution::new(7300, 0.25).unwrap();
        let five = BinomialDistribution::new(6300, 0.20).unwrap();
        assert!((four.cdf[7300] - 1.0).abs() < 1e-12);
        assert!((five.cdf[6300] - 1.0).abs() < 1e-12);
        let four_critical = four.critical_hits(0.005).unwrap();
        let five_critical = five.critical_hits(0.005).unwrap();
        assert_eq!(four_critical, 1922);
        assert_eq!(five_critical, 1343);
        assert!(four.upper_tail(four_critical) <= 0.005);
        assert!(five.upper_tail(five_critical) <= 0.005);
    }

    #[test]
    fn deterministic_rng_and_binomial_sampling_replay() {
        let distribution = BinomialDistribution::new(6300, 0.20).unwrap();
        let mut first = SplitMix64::new(12345);
        let mut second = SplitMix64::new(12345);
        let a: Vec<_> = (0..100).map(|_| distribution.sample(&mut first)).collect();
        let b: Vec<_> = (0..100).map(|_| distribution.sample(&mut second)).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn valid_synthetic_bundle_is_accepted() {
        let digest = "a".repeat(64);
        let trials = make_trial_fixture(24, 5, &digest, false, false, false);
        let guard = ProtocolGuard {
            stopping_policy: "fixed_count".into(),
            posthoc_exclusions: 0,
            expected_trial_count: 24,
            assumes_independent_units: true,
        };
        assert!(validate_trial_bundle(&trials, 5, &digest, &guard).is_ok());
    }

    #[test]
    fn duplicate_trial_ids_are_rejected() {
        let digest = "a".repeat(64);
        let mut trials = make_trial_fixture(24, 4, &digest, false, false, false);
        trials[1].trial_id = trials[0].trial_id.clone();
        let guard = ProtocolGuard {
            stopping_policy: "fixed_count".into(),
            posthoc_exclusions: 0,
            expected_trial_count: 24,
            assumes_independent_units: true,
        };
        assert_eq!(
            validate_trial_bundle(&trials, 4, &digest, &guard)
                .unwrap_err()
                .code,
            "REJECT_DUPLICATE_TRIAL_ID"
        );
    }

    #[test]
    fn direct_target_channel_is_detected() {
        let digest = "a".repeat(64);
        let trials = make_trial_fixture(24, 5, &digest, false, true, false);
        let guard = ProtocolGuard {
            stopping_policy: "fixed_count".into(),
            posthoc_exclusions: 0,
            expected_trial_count: 24,
            assumes_independent_units: true,
        };
        assert_eq!(
            validate_trial_bundle(&trials, 5, &digest, &guard)
                .unwrap_err()
                .code,
            "REJECT_KNOWN_DIRECT_TARGET_CHANNEL"
        );
    }

    #[test]
    fn target_manifest_mismatch_is_rejected() {
        let digest = "a".repeat(64);
        let mut trials = make_trial_fixture(24, 5, &digest, false, false, false);
        trials[2].target_manifest_sha256 = "b".repeat(64);
        let guard = ProtocolGuard {
            stopping_policy: "fixed_count".into(),
            posthoc_exclusions: 0,
            expected_trial_count: 24,
            assumes_independent_units: true,
        };
        assert_eq!(
            validate_trial_bundle(&trials, 5, &digest, &guard)
                .unwrap_err()
                .code,
            "REJECT_TARGET_MANIFEST_MISMATCH"
        );
    }

    #[test]
    fn response_after_target_is_rejected() {
        let digest = "a".repeat(64);
        let mut trials = make_trial_fixture(24, 5, &digest, false, false, false);
        trials[3].target_reveal_sequence = trials[3].response_commit_sequence;
        let guard = ProtocolGuard {
            stopping_policy: "fixed_count".into(),
            posthoc_exclusions: 0,
            expected_trial_count: 24,
            assumes_independent_units: true,
        };
        assert_eq!(
            validate_trial_bundle(&trials, 5, &digest, &guard)
                .unwrap_err()
                .code,
            "REJECT_RESPONSE_AFTER_TARGET"
        );
    }

    #[test]
    fn target_visibility_is_rejected() {
        let digest = "a".repeat(64);
        let trials = make_trial_fixture(24, 5, &digest, false, false, true);
        let guard = ProtocolGuard {
            stopping_policy: "fixed_count".into(),
            posthoc_exclusions: 0,
            expected_trial_count: 24,
            assumes_independent_units: true,
        };
        assert_eq!(
            validate_trial_bundle(&trials, 5, &digest, &guard)
                .unwrap_err()
                .code,
            "REJECT_TARGET_VISIBILITY"
        );
    }

    #[test]
    fn missing_trial_receipt_is_rejected() {
        let digest = "a".repeat(64);
        let mut trials = make_trial_fixture(24, 5, &digest, false, false, false);
        trials.pop();
        let guard = ProtocolGuard {
            stopping_policy: "fixed_count".into(),
            posthoc_exclusions: 0,
            expected_trial_count: 24,
            assumes_independent_units: true,
        };
        assert_eq!(
            validate_trial_bundle(&trials, 5, &digest, &guard)
                .unwrap_err()
                .code,
            "REJECT_TRIAL_COUNT_MISMATCH"
        );
    }

    #[test]
    fn response_digest_tampering_is_rejected() {
        let digest = "a".repeat(64);
        let mut trials = make_trial_fixture(24, 5, &digest, false, false, false);
        trials[4].response_digest_sha256 = "0".repeat(64);
        let guard = ProtocolGuard {
            stopping_policy: "fixed_count".into(),
            posthoc_exclusions: 0,
            expected_trial_count: 24,
            assumes_independent_units: true,
        };
        assert_eq!(
            validate_trial_bundle(&trials, 5, &digest, &guard)
                .unwrap_err()
                .code,
            "REJECT_RESPONSE_DIGEST_MISMATCH"
        );
    }

    #[test]
    fn response_payload_cannot_disagree_with_scored_choice() {
        let digest = "a".repeat(64);
        let mut trials = make_trial_fixture(24, 5, &digest, false, false, false);
        trials[4].response_payload = "payload-for-a-different-choice".into();
        let guard = ProtocolGuard {
            stopping_policy: "fixed_count".into(),
            posthoc_exclusions: 0,
            expected_trial_count: 24,
            assumes_independent_units: true,
        };
        assert_eq!(
            validate_trial_bundle(&trials, 5, &digest, &guard)
                .unwrap_err()
                .code,
            "REJECT_RESPONSE_PAYLOAD_MISMATCH"
        );
    }

    #[test]
    fn target_mapping_matches_independent_four_and_five_choice_vectors() {
        assert!(target_mapping_known_answer_vectors());
    }

    #[test]
    fn target_mapping_rejects_unsupported_choice_count() {
        let chain_hash = [0x52_u8; 32];
        let randomness = [0xfe_u8; 32];
        let commitment = [0x01_u8; 32];
        let context = TargetMappingContext {
            protocol_id: "protocol-v1",
            campaign_id: "campaign-1",
            trial_id: "trial-1",
            chain_hash: &chain_hash,
            scheme_id: "scheme-a",
            response_commitment: &commitment,
            round: 1,
            verified_randomness: &randomness,
        };
        assert!(derive_target_index_from_verified_beacon(3, &context).is_err());
    }

    #[test]
    fn target_mapping_rejects_zero_digest_for_five_choice_only() {
        let zero = [0_u8; 32];
        assert!(!digest_is_below_small_integer(&zero, two_to_256_mod(4)));
        assert!(digest_is_below_small_integer(&zero, two_to_256_mod(5)));
    }

    #[test]
    fn target_mapping_is_domain_and_round_separated() {
        let chain_hash = [0x52_u8; 32];
        let randomness = [0xfe_u8; 32];
        let first_context = TargetMappingContext {
            protocol_id: "protocol-v1",
            campaign_id: "campaign-1",
            trial_id: "trial-1",
            chain_hash: &chain_hash,
            scheme_id: "scheme-a",
            response_commitment: &[0x01_u8; 32],
            round: 100,
            verified_randomness: &randomness,
        };
        let second_context = TargetMappingContext {
            round: 101,
            ..first_context
        };
        let third_context = TargetMappingContext {
            response_commitment: &[0x02_u8; 32],
            ..first_context
        };
        let first = target_mapping_digest(&first_context, 0).unwrap();
        let second = target_mapping_digest(&second_context, 0).unwrap();
        let third = target_mapping_digest(&third_context, 0).unwrap();
        assert_ne!(first, second);
        assert_ne!(first, third);
    }

    #[test]
    fn target_order_permutation_preserves_marginals_and_recomputes_score() {
        assert!(target_order_permutation_recomputes(
            4,
            20261115,
            "four-choice"
        ));
        assert!(target_order_permutation_recomputes(
            5,
            20261115,
            "five-choice"
        ));
    }

    #[test]
    fn optional_stopping_and_posthoc_exclusions_are_rejected() {
        let digest = "a".repeat(64);
        let trials = make_trial_fixture(24, 5, &digest, false, false, false);
        let guard = ProtocolGuard {
            stopping_policy: "optional_stopping".into(),
            posthoc_exclusions: 1,
            expected_trial_count: 24,
            assumes_independent_units: true,
        };
        assert_eq!(
            validate_trial_bundle(&trials, 5, &digest, &guard)
                .unwrap_err()
                .code,
            "REJECT_OPTIONAL_STOPPING"
        );
    }

    #[test]
    fn repeated_units_cannot_be_analyzed_as_iid() {
        let digest = "a".repeat(64);
        let trials = make_trial_fixture(24, 5, &digest, true, false, false);
        let guard = ProtocolGuard {
            stopping_policy: "fixed_count".into(),
            posthoc_exclusions: 0,
            expected_trial_count: 24,
            assumes_independent_units: true,
        };
        assert_eq!(
            validate_trial_bundle(&trials, 5, &digest, &guard)
                .unwrap_err()
                .code,
            "REJECT_REPEATED_UNIT_UNDER_IID_ASSUMPTION"
        );
    }

    #[test]
    fn bijective_label_recoding_preserves_hit_count() {
        assert!(label_permutation_is_invariant(4, 42, "four-choice"));
        assert!(label_permutation_is_invariant(5, 42, "five-choice"));
    }

    #[test]
    fn git_head_must_be_exact_commit_sha() {
        assert!(validate_git_head(&"a".repeat(40)).is_ok());
        assert!(validate_git_head("main").is_err());
        assert!(validate_git_head(&"a".repeat(39)).is_err());
    }

    #[test]
    fn receipt_hash_is_deterministic_for_fixed_body() {
        let data = b"fixed synthetic receipt body";
        assert_eq!(sha256_hex(data), sha256_hex(data));
        assert_eq!(sha256_hex(data).len(), 64);
    }
}
