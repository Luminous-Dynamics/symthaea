//! Deterministic contamination challenge-pair generation and blind adjudication.
//!
//! The public packet contains only opaque pair/item IDs and the two task texts.
//! Tier, planted control label, transformation, source lineage and recipe lineage
//! live in a separate sealed key. The generator is a controlled challenge fixture,
//! not an oracle for real-world model-training exposure.
//!
//! SHA-256/HMAC commitments bind the generated artifacts. They do not prove who
//! ran the generator, who held the sealed key, human reviewer identity, semantic
//! truth, or that a model never encountered a task in training.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const PUBLIC_PACKET_SCHEMA: &str = "symthaea.contamination_panel_public_packet.v1";
pub const SEALED_KEY_SCHEMA: &str = "symthaea.contamination_panel_sealed_key.v1";
pub const MANIFEST_SCHEMA: &str = "symthaea.contamination_panel_manifest.v1";
pub const ADJUDICATION_SCHEMA: &str = "symthaea.contamination_panel_adjudication.v1";
pub const ADJUDICATION_REPORT_SCHEMA: &str =
    "symthaea.contamination_panel_adjudication_report.v1";
pub const BLIND_PROTOCOL_VERSION: &str = "blind-pair-review-v1";
pub const GENERATOR_VERSION: &str = "contamination-panel-generator-v1";
pub const MIN_SEED_BYTES: usize = 32;
pub const MAX_PAIRS_PER_TIER: u32 = 1_000;
pub const REQUIRED_REVIEWERS_PER_PAIR: usize = 3;

const HASH_DOMAIN: &[u8] = b"symthaea.contamination_panel.hash.v1\0";
const HMAC_DOMAIN: &[u8] = b"symthaea.contamination_panel.prf.v1\0";

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum TierV1 {
    T1Exact,
    T2Syntactic,
    T3Semantic,
    T4TaskLevel,
}

impl TierV1 {
    pub const ALL: [Self; 4] = [
        Self::T1Exact,
        Self::T2Syntactic,
        Self::T3Semantic,
        Self::T4TaskLevel,
    ];

    fn target_mechanism(self) -> ExposureMechanismV1 {
        match self {
            Self::T1Exact => ExposureMechanismV1::ExactInstance,
            Self::T2Syntactic => ExposureMechanismV1::DerivativeInstance,
            Self::T3Semantic => ExposureMechanismV1::SemanticEquivalentInstance,
            Self::T4TaskLevel => ExposureMechanismV1::SharedTaskRecipe,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ExposureMechanismV1 {
    ExactInstance,
    DerivativeInstance,
    SemanticEquivalentInstance,
    SharedTaskRecipe,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetControlLabelV1 {
    PositiveForTargetTier,
    NegativeForTargetTier,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlantedRelationshipV1 {
    ExactSameInstance,
    SurfaceVariantSameInstance,
    SemanticEquivalentSameTarget,
    SharedRecipeDistinctInstance,
    SameRecipeDistinctInstance,
    SameDomainDifferentTarget,
    SharedDomainDifferentRecipe,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AdjudicationDecisionV1 {
    SameTargetSameInstance,
    SameConstructDifferentInstance,
    SharedRecipeDifferentInstance,
    DifferentConstruct,
    Indeterminate,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorConfigV1 {
    pub generation_id: String,
    pub pairs_per_tier: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicPairV1 {
    pub pair_id: String,
    pub left_item_id: String,
    pub right_item_id: String,
    pub left_text: String,
    pub right_text: String,
    pub pair_digest_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicChallengePacketV1 {
    pub schema: String,
    pub generation_id: String,
    pub pairs: Vec<PublicPairV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedPairTruthV1 {
    pub pair_id: String,
    pub target_tier: TierV1,
    pub target_mechanism: ExposureMechanismV1,
    pub target_control_label: TargetControlLabelV1,
    pub planted_mechanisms_present: Vec<ExposureMechanismV1>,
    pub planted_relationship: PlantedRelationshipV1,
    pub transformation_family: String,
    pub left_source_lineage_sha256: String,
    pub right_source_lineage_sha256: String,
    pub left_recipe_family_sha256: String,
    pub right_recipe_family_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SealedChallengeKeyV1 {
    pub schema: String,
    pub generation_id: String,
    pub public_packet_sha256: String,
    pub seed_commitment_sha256: String,
    pub pair_truth: Vec<SealedPairTruthV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChallengeManifestV1 {
    pub schema: String,
    pub generator_version: String,
    pub generation_id: String,
    pub pair_count: u64,
    pub pairs_per_tier: u32,
    pub public_packet_sha256: String,
    pub sealed_key_sha256: String,
    pub seed_commitment_sha256: String,
    pub generator_config_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone)]
pub struct ChallengeBundleV1 {
    pub public_packet_bytes: Vec<u8>,
    pub sealed_key_bytes: Vec<u8>,
    pub manifest_bytes: Vec<u8>,
    pub public_packet_sha256: String,
    pub sealed_key_sha256: String,
    pub seed_commitment_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdjudicationRecordV1 {
    pub schema: String,
    pub pair_id: String,
    pub pair_digest_sha256: String,
    pub reviewer_pseudonym: String,
    pub independence_group: String,
    pub blind_protocol_version: String,
    pub reviewer_aware_of_tier: bool,
    pub reviewer_aware_of_planted_control: bool,
    pub decision: AdjudicationDecisionV1,
    pub confidence_permille: u16,
    pub rationale_sha256: String,
    pub evidence_bundle_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairAdjudicationSummaryV1 {
    pub pair_id: String,
    pub review_count: u64,
    pub distinct_reviewer_count: u64,
    pub distinct_independence_group_count: u64,
    pub decision_counts: BTreeMap<String, u64>,
    pub resolution: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdjudicationPanelReportV1 {
    pub schema: String,
    pub valid: bool,
    pub blind_protocol_version: String,
    pub required_reviewers_per_pair: u64,
    pub pair_count: u64,
    pub observed_review_count: u64,
    pub adjudicated_pair_count: u64,
    pub disputed_pair_count: u64,
    pub insufficient_evidence_pair_count: u64,
    pub pairs: Vec<PairAdjudicationSummaryV1>,
    pub reviewer_identity: String,
    pub independence_groups: String,
    pub decision_semantics: String,
    pub claim_boundary: String,
}

#[derive(Serialize)]
struct PairDigestPayloadV1<'a> {
    pair_id: &'a str,
    left_item_id: &'a str,
    right_item_id: &'a str,
    left_text: &'a str,
    right_text: &'a str,
}

fn hash_domain(domain: &[u8], payload: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(HASH_DOMAIN);
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((payload.len() as u64).to_be_bytes());
    hasher.update(payload);
    hex_digest(&hasher.finalize())
}

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut result, "{byte:02x}");
    }
    result
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut key_block = [0_u8; 64];
    if key.len() > key_block.len() {
        let digest = Sha256::digest(key);
        key_block[..digest.len()].copy_from_slice(&digest);
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }

    let mut inner_pad = [0x36_u8; 64];
    let mut outer_pad = [0x5c_u8; 64];
    for index in 0..64 {
        inner_pad[index] ^= key_block[index];
        outer_pad[index] ^= key_block[index];
    }

    let mut inner = Sha256::new();
    inner.update(inner_pad);
    inner.update(HMAC_DOMAIN);
    inner.update(message);
    let inner_digest = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(outer_pad);
    outer.update(inner_digest);
    outer.finalize().into()
}

fn keyed_digest(seed: &[u8], domain: &[u8], ordinal: u64, slot: u64) -> [u8; 32] {
    let mut message = Vec::with_capacity(domain.len() + 24);
    message.extend_from_slice(&(domain.len() as u64).to_be_bytes());
    message.extend_from_slice(domain);
    message.extend_from_slice(&ordinal.to_be_bytes());
    message.extend_from_slice(&slot.to_be_bytes());
    hmac_sha256(seed, &message)
}

fn keyed_id(seed: &[u8], domain: &[u8], ordinal: u64, slot: u64) -> String {
    hex_digest(&keyed_digest(seed, domain, ordinal, slot))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_config(config: &GeneratorConfigV1, seed: &[u8]) -> Result<(), String> {
    if seed.len() < MIN_SEED_BYTES {
        return Err(format!(
            "seed must contain at least {MIN_SEED_BYTES} bytes of caller-supplied entropy"
        ));
    }
    if config.generation_id.trim().is_empty()
        || config.generation_id.len() > 128
        || !config
            .generation_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err("generation_id must be 1..=128 ASCII letters/digits or . _ : -".to_owned());
    }
    if !(1..=MAX_PAIRS_PER_TIER).contains(&config.pairs_per_tier) {
        return Err(format!(
            "pairs_per_tier must be in 1..={MAX_PAIRS_PER_TIER}"
        ));
    }
    Ok(())
}

fn task_values(seed: &[u8], ordinal: u64) -> (u32, u32, u32) {
    let bytes = keyed_digest(seed, b"task-values", ordinal, 0);
    let a = 1 + (u32::from_be_bytes(bytes[0..4].try_into().unwrap()) % 7);
    let b = 2 + (u32::from_be_bytes(bytes[4..8].try_into().unwrap()) % 17);
    let slope = 1 + (u32::from_be_bytes(bytes[8..12].try_into().unwrap()) % 5);
    (a, b, slope)
}

fn linear_prediction_prompt(a: u32, b: u32, slope: u32) -> String {
    format!(
        "Infer the linear input-output rule from ({a}, {b}) and ({}, {}); predict the output when x = {}.",
        a + 1,
        b + slope,
        a + 3
    )
}

fn semantic_paraphrase_left(a: u32, b: u32, slope: u32) -> String {
    format!(
        "A device emits {b} for input {a}, then emits {} for input {}. Extrapolate what it emits for input {}.",
        b + slope,
        a + 1,
        a + 3
    )
}

fn semantic_paraphrase_right(a: u32, b: u32, slope: u32) -> String {
    format!(
        "Two samples of a fixed-rate mapping are ({a}, {b}) and ({}, {}). Estimate its value at an argument of {}.",
        a + 1,
        b + slope,
        a + 3
    )
}

fn inverse_task_prompt(a: u32, b: u32, slope: u32) -> String {
    let output = b + 3 * slope;
    format!(
        "For the observed linear rule through ({a}, {b}) and ({}, {}), determine the input that would produce output {output}.",
        a + 1,
        b + slope
    )
}

fn polynomial_task_prompt(a: u32, b: u32, slope: u32) -> String {
    format!(
        "For q(t) = t squared + {slope}t + {b}, calculate q({a})."
    )
}

struct PairConstructionV1 {
    left_text: String,
    right_text: String,
    relationship: PlantedRelationshipV1,
    mechanisms: Vec<ExposureMechanismV1>,
    transformation_family: String,
    same_source_lineage: bool,
    same_recipe_family: bool,
}

fn construct_pair(tier: TierV1, positive: bool, a: u32, b: u32, slope: u32) -> PairConstructionV1 {
    let base = linear_prediction_prompt(a, b, slope);
    match (tier, positive) {
        (TierV1::T1Exact, true) => PairConstructionV1 {
            left_text: base.clone(),
            right_text: base,
            relationship: PlantedRelationshipV1::ExactSameInstance,
            mechanisms: vec![
                ExposureMechanismV1::ExactInstance,
                ExposureMechanismV1::DerivativeInstance,
                ExposureMechanismV1::SemanticEquivalentInstance,
                ExposureMechanismV1::SharedTaskRecipe,
            ],
            transformation_family: "byte_identical".to_owned(),
            same_source_lineage: true,
            same_recipe_family: true,
        },
        (TierV1::T1Exact, false) => PairConstructionV1 {
            left_text: base,
            right_text: linear_prediction_prompt(a, b + 1, slope),
            relationship: PlantedRelationshipV1::SameRecipeDistinctInstance,
            mechanisms: vec![ExposureMechanismV1::SharedTaskRecipe],
            transformation_family: "parameter_changed_exact_negative_control".to_owned(),
            same_source_lineage: false,
            same_recipe_family: true,
        },
        (TierV1::T2Syntactic, true) => PairConstructionV1 {
            left_text: base.clone(),
            right_text: format!(
                "  {}\n",
                base.to_uppercase().replace("(", " ( ").replace(")", " )").replace(",", " ,")
            ),
            relationship: PlantedRelationshipV1::SurfaceVariantSameInstance,
            mechanisms: vec![
                ExposureMechanismV1::DerivativeInstance,
                ExposureMechanismV1::SemanticEquivalentInstance,
                ExposureMechanismV1::SharedTaskRecipe,
            ],
            transformation_family: "case_spacing_punctuation".to_owned(),
            same_source_lineage: true,
            same_recipe_family: true,
        },
        (TierV1::T2Syntactic, false) => PairConstructionV1 {
            left_text: base,
            right_text: linear_prediction_prompt(a, b + 1, slope),
            relationship: PlantedRelationshipV1::SameRecipeDistinctInstance,
            mechanisms: vec![ExposureMechanismV1::SharedTaskRecipe],
            transformation_family: "lexical_near_miss_parameter_change".to_owned(),
            same_source_lineage: false,
            same_recipe_family: true,
        },
        (TierV1::T3Semantic, true) => PairConstructionV1 {
            left_text: semantic_paraphrase_left(a, b, slope),
            right_text: semantic_paraphrase_right(a, b, slope),
            relationship: PlantedRelationshipV1::SemanticEquivalentSameTarget,
            mechanisms: vec![
                ExposureMechanismV1::SemanticEquivalentInstance,
                ExposureMechanismV1::SharedTaskRecipe,
            ],
            transformation_family: "authored_semantic_reformulation".to_owned(),
            same_source_lineage: true,
            same_recipe_family: true,
        },
        (TierV1::T3Semantic, false) => PairConstructionV1 {
            left_text: base,
            right_text: inverse_task_prompt(a, b, slope),
            relationship: PlantedRelationshipV1::SameDomainDifferentTarget,
            mechanisms: Vec::new(),
            transformation_family: "same_domain_inverse_target_negative_control".to_owned(),
            same_source_lineage: false,
            same_recipe_family: false,
        },
        (TierV1::T4TaskLevel, true) => PairConstructionV1 {
            left_text: linear_prediction_prompt(a, b, slope),
            right_text: semantic_paraphrase_right(a + 2, b + 3, slope + 1),
            relationship: PlantedRelationshipV1::SharedRecipeDistinctInstance,
            mechanisms: vec![ExposureMechanismV1::SharedTaskRecipe],
            transformation_family: "same_recipe_distinct_numeric_instance".to_owned(),
            same_source_lineage: false,
            same_recipe_family: true,
        },
        (TierV1::T4TaskLevel, false) => PairConstructionV1 {
            left_text: base,
            right_text: polynomial_task_prompt(a, b, slope),
            relationship: PlantedRelationshipV1::SharedDomainDifferentRecipe,
            mechanisms: Vec::new(),
            transformation_family: "same_algebra_domain_different_recipe".to_owned(),
            same_source_lineage: false,
            same_recipe_family: false,
        },
    }
}

fn pair_digest(pair: &PublicPairV1) -> Result<String, String> {
    let payload = PairDigestPayloadV1 {
        pair_id: &pair.pair_id,
        left_item_id: &pair.left_item_id,
        right_item_id: &pair.right_item_id,
        left_text: &pair.left_text,
        right_text: &pair.right_text,
    };
    let encoded = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize public pair digest payload: {error}"))?;
    Ok(hash_domain(b"public-pair.v1", &encoded))
}

fn recipe_digest(seed: &[u8], ordinal: u64, side: u64, same: bool) -> String {
    let (recipe_ordinal, recipe_slot) = if same { (0, 0) } else { (ordinal, side) };
    keyed_id(seed, b"recipe-family", recipe_ordinal, recipe_slot)
}

fn source_digest(seed: &[u8], ordinal: u64, side: u64, same: bool) -> String {
    let (source_ordinal, source_slot) = if same {
        (ordinal ^ 0x8000_0000_0000_0000, 0)
    } else {
        (ordinal, side)
    };
    keyed_id(seed, b"source-lineage", source_ordinal, source_slot)
}

/// Generate a public packet, separately held answer key, and commitment manifest.
///
/// The seed is a caller-supplied secret of at least 256 bits. Use an OS-backed
/// cryptographic random source in the evaluator environment; never publish the
/// seed or sealed key alongside the public pair packet.
pub fn generate_challenge_bundle(
    config: &GeneratorConfigV1,
    seed: &[u8],
) -> Result<ChallengeBundleV1, String> {
    validate_config(config, seed)?;

    let seed_commitment_sha256 = hash_domain(b"seed-commitment.v1", seed);
    let mut pairs = Vec::new();
    let mut pair_truth = Vec::new();
    let mut ordinal = 0_u64;

    for tier in TierV1::ALL {
        for target_positive in [true, false] {
            for _ in 0..config.pairs_per_tier {
                let (a, b, slope) = task_values(seed, ordinal);
                let construction = construct_pair(tier, target_positive, a, b, slope);
                let pair_id = keyed_id(seed, b"pair-id", ordinal, 0);
                let mut left_item_id = keyed_id(seed, b"item-id", ordinal, 0);
                let mut right_item_id = keyed_id(seed, b"item-id", ordinal, 1);
                let mut left_text = construction.left_text;
                let mut right_text = construction.right_text;
                let mut mechanisms = construction.mechanisms;
                let relationship = construction.relationship;
                let transformation_family = construction.transformation_family;
                let mut left_source_lineage_sha256 =
                    source_digest(seed, ordinal, 0, construction.same_source_lineage);
                let mut right_source_lineage_sha256 =
                    source_digest(seed, ordinal, 1, construction.same_source_lineage);
                let mut left_recipe_family_sha256 =
                    recipe_digest(seed, ordinal, 0, construction.same_recipe_family);
                let mut right_recipe_family_sha256 =
                    recipe_digest(seed, ordinal, 1, construction.same_recipe_family);

                // Deterministically randomize pair orientation independently of
                // tier and control label; the public packet exposes no side marker.
                let reverse = keyed_digest(seed, b"pair-orientation", ordinal, 0)[0] & 1 == 1;
                if reverse {
                    std::mem::swap(&mut left_item_id, &mut right_item_id);
                    std::mem::swap(&mut left_text, &mut right_text);
                    std::mem::swap(
                        &mut left_source_lineage_sha256,
                        &mut right_source_lineage_sha256,
                    );
                    std::mem::swap(
                        &mut left_recipe_family_sha256,
                        &mut right_recipe_family_sha256,
                    );
                }
                mechanisms.sort();
                mechanisms.dedup();

                let mut public_pair = PublicPairV1 {
                    pair_id: pair_id.clone(),
                    left_item_id,
                    right_item_id,
                    left_text,
                    right_text,
                    pair_digest_sha256: String::new(),
                };
                public_pair.pair_digest_sha256 = pair_digest(&public_pair)?;

                let target_mechanism = tier.target_mechanism();
                let target_control_label = if mechanisms.contains(&target_mechanism) {
                    TargetControlLabelV1::PositiveForTargetTier
                } else {
                    TargetControlLabelV1::NegativeForTargetTier
                };

                pairs.push(public_pair);
                pair_truth.push(SealedPairTruthV1 {
                    pair_id,
                    target_tier: tier,
                    target_mechanism,
                    target_control_label,
                    planted_mechanisms_present: mechanisms,
                    planted_relationship: relationship,
                    transformation_family,
                    left_source_lineage_sha256,
                    right_source_lineage_sha256,
                    left_recipe_family_sha256,
                    right_recipe_family_sha256,
                });
                ordinal = ordinal
                    .checked_add(1)
                    .ok_or_else(|| "pair ordinal overflowed u64".to_owned())?;
            }
        }
    }

    // Hash-ordered opaque IDs randomize packet order without exposing generation order.
    pairs.sort_by(|left, right| left.pair_id.cmp(&right.pair_id));
    pair_truth.sort_by(|left, right| left.pair_id.cmp(&right.pair_id));

    let pair_ids: BTreeSet<&str> = pairs.iter().map(|pair| pair.pair_id.as_str()).collect();
    if pair_ids.len() != pairs.len() {
        return Err("derived pair ID collision detected; regenerate with a new seed".to_owned());
    }

    let public_packet = PublicChallengePacketV1 {
        schema: PUBLIC_PACKET_SCHEMA.to_owned(),
        generation_id: config.generation_id.clone(),
        pairs,
    };
    let public_packet_bytes = serde_json::to_vec_pretty(&public_packet)
        .map_err(|error| format!("cannot serialize public challenge packet: {error}"))?;
    let public_packet_sha256 = hash_domain(b"public-packet.v1", &public_packet_bytes);

    let sealed_key = SealedChallengeKeyV1 {
        schema: SEALED_KEY_SCHEMA.to_owned(),
        generation_id: config.generation_id.clone(),
        public_packet_sha256: public_packet_sha256.clone(),
        seed_commitment_sha256: seed_commitment_sha256.clone(),
        pair_truth,
    };
    let sealed_key_bytes = serde_json::to_vec_pretty(&sealed_key)
        .map_err(|error| format!("cannot serialize sealed challenge key: {error}"))?;
    let sealed_key_sha256 = hash_domain(b"sealed-key.v1", &sealed_key_bytes);

    let config_bytes = serde_json::to_vec(config)
        .map_err(|error| format!("cannot serialize generator config: {error}"))?;
    let manifest = ChallengeManifestV1 {
        schema: MANIFEST_SCHEMA.to_owned(),
        generator_version: GENERATOR_VERSION.to_owned(),
        generation_id: config.generation_id.clone(),
        pair_count: u64::try_from(public_packet.pairs.len())
            .map_err(|_| "pair count exceeds u64")?,
        pairs_per_tier: config.pairs_per_tier,
        public_packet_sha256: public_packet_sha256.clone(),
        sealed_key_sha256: sealed_key_sha256.clone(),
        seed_commitment_sha256: seed_commitment_sha256.clone(),
        generator_config_sha256: hash_domain(b"generator-config.v1", &config_bytes),
        claim_boundary: "synthetic_planted_control_labels_only_not_evidence_of_real_model_training_exposure_or_semantic_validity".to_owned(),
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("cannot serialize challenge manifest: {error}"))?;

    Ok(ChallengeBundleV1 {
        public_packet_bytes,
        sealed_key_bytes,
        manifest_bytes,
        public_packet_sha256,
        sealed_key_sha256,
        seed_commitment_sha256,
    })
}

fn validate_pair_packet(packet: &PublicChallengePacketV1) -> Result<(), String> {
    if packet.schema != PUBLIC_PACKET_SCHEMA {
        return Err(format!("public packet schema must be {PUBLIC_PACKET_SCHEMA}"));
    }
    if packet.generation_id.trim().is_empty() || packet.pairs.is_empty() {
        return Err("public packet must have a generation_id and at least one pair".to_owned());
    }
    let mut pair_ids = BTreeSet::new();
    for pair in &packet.pairs {
        if !valid_digest(&pair.pair_id)
            || !valid_digest(&pair.left_item_id)
            || !valid_digest(&pair.right_item_id)
            || !valid_digest(&pair.pair_digest_sha256)
        {
            return Err(format!("pair {} contains an invalid digest or identifier", pair.pair_id));
        }
        if pair.left_text.trim().is_empty() || pair.right_text.trim().is_empty() {
            return Err(format!("pair {} has empty task text", pair.pair_id));
        }
        if !pair_ids.insert(pair.pair_id.as_str()) {
            return Err(format!("public packet repeats pair ID {}", pair.pair_id));
        }
        if pair_digest(pair)? != pair.pair_digest_sha256 {
            return Err(format!("pair {} digest does not match public content", pair.pair_id));
        }
    }
    Ok(())
}

/// Validate a blinded human panel without access to the sealed key.
///
/// Exactly three records are required per pair. Two-of-three votes are not
/// promoted to truth: any disagreement is reported as disputed. Reviewer IDs
/// and independence groups are self-declared and not authenticated here.
pub fn audit_blinded_adjudications(
    packet: &PublicChallengePacketV1,
    records: &[AdjudicationRecordV1],
) -> Result<AdjudicationPanelReportV1, String> {
    validate_pair_packet(packet)?;

    let pair_by_id: BTreeMap<&str, &PublicPairV1> = packet
        .pairs
        .iter()
        .map(|pair| (pair.pair_id.as_str(), pair))
        .collect();
    let mut records_by_pair = BTreeMap::<&str, Vec<&AdjudicationRecordV1>>::new();

    for record in records {
        if record.schema != ADJUDICATION_SCHEMA {
            return Err(format!(
                "adjudication for pair {} has unsupported schema",
                record.pair_id
            ));
        }
        let Some(pair) = pair_by_id.get(record.pair_id.as_str()) else {
            return Err(format!(
                "adjudication references unknown pair {}",
                record.pair_id
            ));
        };
        if record.pair_digest_sha256 != pair.pair_digest_sha256 {
            return Err(format!(
                "adjudication for pair {} does not bind the exact public pair",
                record.pair_id
            ));
        }
        if record.reviewer_pseudonym.trim().is_empty()
            || record.reviewer_pseudonym.len() > 128
            || record.independence_group.trim().is_empty()
            || record.independence_group.len() > 128
        {
            return Err(format!(
                "adjudication for pair {} has an empty or oversized reviewer identity/group",
                record.pair_id
            ));
        }
        if record.blind_protocol_version != BLIND_PROTOCOL_VERSION {
            return Err(format!(
                "adjudication for pair {} must use {BLIND_PROTOCOL_VERSION}",
                record.pair_id
            ));
        }
        if record.reviewer_aware_of_tier || record.reviewer_aware_of_planted_control {
            return Err(format!(
                "adjudication for pair {} is not blinded to tier and planted control",
                record.pair_id
            ));
        }
        if record.confidence_permille > 1_000 {
            return Err(format!(
                "adjudication for pair {} confidence_permille exceeds 1000",
                record.pair_id
            ));
        }
        if !valid_digest(&record.rationale_sha256)
            || !valid_digest(&record.evidence_bundle_sha256)
        {
            return Err(format!(
                "adjudication for pair {} must bind rationale and evidence digests",
                record.pair_id
            ));
        }
        records_by_pair
            .entry(record.pair_id.as_str())
            .or_default()
            .push(record);
    }

    let mut summaries = Vec::with_capacity(packet.pairs.len());
    let mut observed_review_count = 0_u64;
    let mut disputed_pair_count = 0_u64;
    let mut insufficient_evidence_pair_count = 0_u64;

    for pair in &packet.pairs {
        let pair_records = records_by_pair
            .get(pair.pair_id.as_str())
            .ok_or_else(|| format!("pair {} has no adjudication records", pair.pair_id))?;
        if pair_records.len() != REQUIRED_REVIEWERS_PER_PAIR {
            return Err(format!(
                "pair {} requires exactly {REQUIRED_REVIEWERS_PER_PAIR} adjudicators; observed {}",
                pair.pair_id,
                pair_records.len()
            ));
        }

        let mut reviewers = BTreeSet::new();
        let mut independence_groups = BTreeSet::new();
        let mut decision_counts = BTreeMap::<String, u64>::new();
        for record in pair_records {
            if !reviewers.insert(record.reviewer_pseudonym.as_str()) {
                return Err(format!(
                    "pair {} repeats reviewer pseudonym {}",
                    pair.pair_id, record.reviewer_pseudonym
                ));
            }
            if !independence_groups.insert(record.independence_group.as_str()) {
                return Err(format!(
                    "pair {} has duplicate independence group {}",
                    pair.pair_id, record.independence_group
                ));
            }
            let decision = format!("{:?}", record.decision).to_ascii_lowercase();
            let count = decision_counts.entry(decision).or_default();
            *count = count
                .checked_add(1)
                .ok_or_else(|| "decision counter overflow".to_owned())?;
        }

        let resolution = if decision_counts.len() == 1 {
            let (decision, count) = decision_counts
                .iter()
                .next()
                .ok_or_else(|| "empty decision map".to_owned())?;
            if *count != REQUIRED_REVIEWERS_PER_PAIR as u64 {
                return Err("unanimity count does not match required panel size".to_owned());
            }
            if decision == "indeterminate" {
                insufficient_evidence_pair_count = insufficient_evidence_pair_count
                    .checked_add(1)
                    .ok_or_else(|| "insufficient-evidence counter overflow".to_owned())?;
                "insufficient_evidence".to_owned()
            } else {
                "unanimous".to_owned()
            }
        } else {
            disputed_pair_count = disputed_pair_count
                .checked_add(1)
                .ok_or_else(|| "disputed-pair counter overflow".to_owned())?;
            "disputed".to_owned()
        };

        observed_review_count = observed_review_count
            .checked_add(u64::try_from(pair_records.len()).map_err(|_| "review count overflow")?)
            .ok_or_else(|| "observed review count overflow".to_owned())?;

        summaries.push(PairAdjudicationSummaryV1 {
            pair_id: pair.pair_id.clone(),
            review_count: u64::try_from(pair_records.len()).map_err(|_| "review count overflow")?,
            distinct_reviewer_count: u64::try_from(reviewers.len())
                .map_err(|_| "reviewer count overflow")?,
            distinct_independence_group_count: u64::try_from(independence_groups.len())
                .map_err(|_| "independence-group count overflow")?,
            decision_counts,
            resolution,
        });
    }

    if records_by_pair.len() != packet.pairs.len() {
        return Err("adjudication panel contains an unexpected pair grouping".to_owned());
    }

    Ok(AdjudicationPanelReportV1 {
        schema: ADJUDICATION_REPORT_SCHEMA.to_owned(),
        valid: true,
        blind_protocol_version: BLIND_PROTOCOL_VERSION.to_owned(),
        required_reviewers_per_pair: REQUIRED_REVIEWERS_PER_PAIR as u64,
        pair_count: u64::try_from(packet.pairs.len()).map_err(|_| "pair count overflow")?,
        observed_review_count,
        adjudicated_pair_count: u64::try_from(packet.pairs.len())
            .map_err(|_| "adjudicated-pair count overflow")?
            .checked_sub(disputed_pair_count)
            .and_then(|count| count.checked_sub(insufficient_evidence_pair_count))
            .ok_or_else(|| "adjudicated-pair count underflow".to_owned())?,
        disputed_pair_count,
        insufficient_evidence_pair_count,
        pairs: summaries,
        reviewer_identity: "pseudonymous_self_declared_not_authenticated".to_owned(),
        independence_groups: "declared_distinct_per_pair_not_independently_verified".to_owned(),
        decision_semantics: "unanimity_required_for_non_disputed_resolution_majority_vote_never_promoted_to_truth".to_owned(),
        claim_boundary: "panel_structure_and_blinding_attestations_validated_only_not_semantic_truth_or_reviewer_identity".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn config() -> GeneratorConfigV1 {
        GeneratorConfigV1 {
            generation_id: "fixture-generation-001".to_owned(),
            pairs_per_tier: 2,
        }
    }

    fn seed(byte: u8) -> Vec<u8> {
        vec![byte; MIN_SEED_BYTES]
    }

    fn packet(bundle: &ChallengeBundleV1) -> PublicChallengePacketV1 {
        serde_json::from_slice(&bundle.public_packet_bytes).unwrap()
    }

    fn valid_records(packet: &PublicChallengePacketV1) -> Vec<AdjudicationRecordV1> {
        let mut records = Vec::new();
        for pair in &packet.pairs {
            for reviewer in 0..REQUIRED_REVIEWERS_PER_PAIR {
                records.push(AdjudicationRecordV1 {
                    schema: ADJUDICATION_SCHEMA.to_owned(),
                    pair_id: pair.pair_id.clone(),
                    pair_digest_sha256: pair.pair_digest_sha256.clone(),
                    reviewer_pseudonym: format!("reviewer-{reviewer}"),
                    independence_group: format!("group-{reviewer}"),
                    blind_protocol_version: BLIND_PROTOCOL_VERSION.to_owned(),
                    reviewer_aware_of_tier: false,
                    reviewer_aware_of_planted_control: false,
                    decision: AdjudicationDecisionV1::SameConstructDifferentInstance,
                    confidence_permille: 900,
                    rationale_sha256: hash_domain(b"rationale", b"fixture"),
                    evidence_bundle_sha256: hash_domain(b"evidence", b"fixture"),
                });
            }
        }
        records
    }

    #[test]
    fn generation_is_deterministic_and_packet_is_separate_from_key() {
        let config = config();
        let seed = seed(7);
        let first = generate_challenge_bundle(&config, &seed).unwrap();
        let second = generate_challenge_bundle(&config, &seed).unwrap();

        assert_eq!(first.public_packet_bytes, second.public_packet_bytes);
        assert_eq!(first.sealed_key_bytes, second.sealed_key_bytes);
        assert_eq!(first.manifest_bytes, second.manifest_bytes);

        let public_text = String::from_utf8(first.public_packet_bytes.clone()).unwrap();
        assert!(!public_text.contains("target_tier"));
        assert!(!public_text.contains("target_control_label"));
        assert!(!public_text.contains("planted_mechanisms_present"));
        assert!(!public_text.contains("transformation_family"));
        assert!(!public_text.contains("positive_for_target_tier"));
        assert!(!public_text.contains("negative_for_target_tier"));
    }

    #[test]
    fn generation_emits_balanced_controls_per_target_tier() {
        let bundle = generate_challenge_bundle(&config(), &seed(9)).unwrap();
        let key: SealedChallengeKeyV1 = serde_json::from_slice(&bundle.sealed_key_bytes).unwrap();
        assert_eq!(key.pair_truth.len(), 16);

        for tier in TierV1::ALL {
            let rows: Vec<_> = key
                .pair_truth
                .iter()
                .filter(|row| row.target_tier == tier)
                .collect();
            assert_eq!(rows.len(), 4);
            assert_eq!(
                rows.iter()
                    .filter(|row| row.target_control_label == TargetControlLabelV1::PositiveForTargetTier)
                    .count(),
                2
            );
            assert_eq!(
                rows.iter()
                    .filter(|row| row.target_control_label == TargetControlLabelV1::NegativeForTargetTier)
                    .count(),
                2
            );
        }
    }

    #[test]
    fn tier_four_positive_uses_distinct_instances_and_shared_recipe_not_same_source() {
        let bundle = generate_challenge_bundle(&config(), &seed(11)).unwrap();
        let key: SealedChallengeKeyV1 = serde_json::from_slice(&bundle.sealed_key_bytes).unwrap();
        let positive = key
            .pair_truth
            .iter()
            .find(|row| {
                row.target_tier == TierV1::T4TaskLevel
                    && row.target_control_label == TargetControlLabelV1::PositiveForTargetTier
            })
            .unwrap();
        assert_ne!(
            positive.left_source_lineage_sha256,
            positive.right_source_lineage_sha256
        );
        assert_eq!(
            positive.left_recipe_family_sha256,
            positive.right_recipe_family_sha256
        );
        assert_eq!(
            positive.planted_relationship,
            PlantedRelationshipV1::SharedRecipeDistinctInstance
        );
    }

    #[test]
    fn public_manifest_binds_both_artifacts_and_seed_commitment() {
        let bundle = generate_challenge_bundle(&config(), &seed(13)).unwrap();
        let manifest: ChallengeManifestV1 =
            serde_json::from_slice(&bundle.manifest_bytes).unwrap();
        assert_eq!(manifest.public_packet_sha256, bundle.public_packet_sha256);
        assert_eq!(manifest.sealed_key_sha256, bundle.sealed_key_sha256);
        assert_eq!(manifest.seed_commitment_sha256, bundle.seed_commitment_sha256);
        assert!(manifest.claim_boundary.contains("not_evidence_of_real_model_training_exposure"));
    }

    #[test]
    fn different_seeds_change_opaque_ids_and_seed_commitments() {
        let left = generate_challenge_bundle(&config(), &seed(17)).unwrap();
        let right = generate_challenge_bundle(&config(), &seed(18)).unwrap();
        assert_ne!(left.public_packet_sha256, right.public_packet_sha256);
        assert_ne!(left.seed_commitment_sha256, right.seed_commitment_sha256);
    }

    #[test]
    fn short_seed_and_out_of_range_pair_count_fail_closed() {
        assert!(generate_challenge_bundle(&config(), b"short").is_err());
        let mut invalid = config();
        invalid.pairs_per_tier = 0;
        assert!(generate_challenge_bundle(&invalid, &seed(1)).is_err());
        invalid.pairs_per_tier = MAX_PAIRS_PER_TIER + 1;
        assert!(generate_challenge_bundle(&invalid, &seed(1)).is_err());
    }

    #[test]
    fn pair_mutation_invalidates_blinded_adjudication_binding() {
        let bundle = generate_challenge_bundle(&config(), &seed(21)).unwrap();
        let mut packet = packet(&bundle);
        let records = valid_records(&packet);
        packet.pairs[0].left_text.push_str(" mutated");
        let error = audit_blinded_adjudications(&packet, &records).unwrap_err();
        assert!(error.contains("digest does not match public content"));

        // Re-sealing pair content still invalidates an old reviewer record.
        packet.pairs[0].pair_digest_sha256 = pair_digest(&packet.pairs[0]).unwrap();
        let error = audit_blinded_adjudications(&packet, &records).unwrap_err();
        assert!(error.contains("does not bind the exact public pair"));
    }

    #[test]
    fn complete_unanimous_panel_passes_without_claiming_verified_identity() {
        let bundle = generate_challenge_bundle(&config(), &seed(23)).unwrap();
        let packet = packet(&bundle);
        let records = valid_records(&packet);
        let report = audit_blinded_adjudications(&packet, &records).unwrap();

        assert!(report.valid);
        assert_eq!(report.pair_count, 16);
        assert_eq!(report.observed_review_count, 48);
        assert_eq!(report.disputed_pair_count, 0);
        assert_eq!(report.insufficient_evidence_pair_count, 0);
        assert_eq!(report.reviewer_identity, "pseudonymous_self_declared_not_authenticated");
    }

    #[test]
    fn two_to_one_vote_is_disputed_not_promoted_by_majority() {
        let bundle = generate_challenge_bundle(&config(), &seed(25)).unwrap();
        let packet = packet(&bundle);
        let mut records = valid_records(&packet);
        let target = &packet.pairs[0].pair_id;
        let third = records
            .iter_mut()
            .find(|record| record.pair_id == *target && record.reviewer_pseudonym == "reviewer-2")
            .unwrap();
        third.decision = AdjudicationDecisionV1::DifferentConstruct;

        let report = audit_blinded_adjudications(&packet, &records).unwrap();
        assert_eq!(report.disputed_pair_count, 1);
        let row = report.pairs.iter().find(|row| row.pair_id == *target).unwrap();
        assert_eq!(row.resolution, "disputed");
    }

    #[test]
    fn duplicate_reviewer_or_independence_group_fails_closed() {
        let bundle = generate_challenge_bundle(&config(), &seed(27)).unwrap();
        let packet = packet(&bundle);
        let mut records = valid_records(&packet);
        records[1].reviewer_pseudonym = records[0].reviewer_pseudonym.clone();
        assert!(audit_blinded_adjudications(&packet, &records)
            .unwrap_err()
            .contains("repeats reviewer"));

        let mut records = valid_records(&packet);
        records[1].independence_group = records[0].independence_group.clone();
        assert!(audit_blinded_adjudications(&packet, &records)
            .unwrap_err()
            .contains("duplicate independence group"));
    }

    #[test]
    fn missing_panel_member_or_unknown_pair_fails_closed() {
        let bundle = generate_challenge_bundle(&config(), &seed(29)).unwrap();
        let packet = packet(&bundle);
        let mut records = valid_records(&packet);
        records.pop();
        assert!(audit_blinded_adjudications(&packet, &records)
            .unwrap_err()
            .contains("requires exactly 3 adjudicators"));

        let mut records = valid_records(&packet);
        records[0].pair_id = "not-a-real-pair".to_owned();
        assert!(audit_blinded_adjudications(&packet, &records)
            .unwrap_err()
            .contains("unknown pair"));
    }

    #[test]
    fn non_blinded_record_fails_closed() {
        let bundle = generate_challenge_bundle(&config(), &seed(31)).unwrap();
        let packet = packet(&bundle);
        let mut records = valid_records(&packet);
        records[0].reviewer_aware_of_tier = true;
        assert!(audit_blinded_adjudications(&packet, &records)
            .unwrap_err()
            .contains("not blinded"));
    }

    #[test]
    fn all_indeterminate_reviews_are_insufficient_evidence() {
        let bundle = generate_challenge_bundle(&config(), &seed(33)).unwrap();
        let packet = packet(&bundle);
        let mut records = valid_records(&packet);
        for record in &mut records {
            record.decision = AdjudicationDecisionV1::Indeterminate;
        }
        let report = audit_blinded_adjudications(&packet, &records).unwrap();
        assert_eq!(report.insufficient_evidence_pair_count, 16);
        assert_eq!(report.adjudicated_pair_count, 0);
        assert!(report.pairs.iter().all(|pair| pair.resolution == "insufficient_evidence"));
    }

    #[test]
    fn pair_digest_is_stable_and_changes_for_either_side() {
        let bundle = generate_challenge_bundle(&config(), &seed(35)).unwrap();
        let mut packet = packet(&bundle);
        let pair = &mut packet.pairs[0];
        let original = pair.pair_digest_sha256.clone();
        assert_eq!(pair_digest(pair).unwrap(), original);
        pair.right_text.push('!');
        assert_ne!(pair_digest(pair).unwrap(), original);
    }

    #[test]
    fn unknown_fields_are_rejected_by_strict_deserialization() {
        let bundle = generate_challenge_bundle(&config(), &seed(37)).unwrap();
        let mut packet_json: Value = serde_json::from_slice(&bundle.public_packet_bytes).unwrap();
        packet_json["unexpected_answer"] = serde_json::json!("leak");
        let bytes = serde_json::to_vec(&packet_json).unwrap();
        assert!(serde_json::from_slice::<PublicChallengePacketV1>(&bytes).is_err());
    }
}
