//! Verify externally signed PITB component-coverage attestations.
//!
//! This binary authenticates a coverage claim; it does not itself qualify a
//! component graph or authorize confirmatory analysis. The run config must pin
//! the exact trust-store digest, and the trust-store root must be managed outside
//! the outcome-producing experiment.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use anyhow::{Context, Result, bail};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const UPSTREAM_COMMIT: &str = "1581d4cbdd69a0faf34b2089614fe2a814316a6a";
pub const EXPECTED_PARTITION_DIGEST: &str =
    "e7ac417556cbd47cee1ddd51b98f9684fadfc8cd097587ae3beacc4dfb59f970";
const SPLIT_SEED: &str = "pitb-v0-openai-math-2026-10-19";
const SPLIT_ALGORITHM: &str = concat!(
    "sha256(seed || NUL || family_id), sort by digest then family ID, ",
    "floor 70% train and floor 85% train+validation"
);
const RECEIPT_SCHEMA: &str = "pitb.component-coverage-receipt.v1";
const TRUST_SCHEMA: &str = "pitb.coverage-reviewer-trust-store.v1";
const DOMAIN: &str = "PITB_COMPONENT_COVERAGE_RECEIPT_V1";

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Inventory {
    schema_version: String,
    source_partition_digest_sha256: String,
    coverage_status: String,
    coverage_receipt_sha256: Option<String>,
    edges: Vec<Edge>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Edge {
    edge_id: String,
    source_family_id: String,
    target_family_id: String,
    relation_kind: String,
    assessment_status: String,
    evidence_sha256: String,
    reviewer_ids: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: String,
    upstream_commit: String,
    source_partition_digest_sha256: String,
    relationship_inventory_edges_sha256: String,
    review_methodology_sha256: String,
    trust_store_sha256: String,
    attestation: String,
    signatures: Vec<ReviewerSignature>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReviewerSignature {
    reviewer_id: String,
    key_id: String,
    signature_hex: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TrustStore {
    schema_version: String,
    reviewers: Vec<TrustedReviewer>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TrustedReviewer {
    reviewer_id: String,
    key_id: String,
    organization_id: String,
    role: String,
    public_key_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedComponent {
    pub component_id: String,
    pub split_names: Vec<String>,
    pub family_ids: Vec<String>,
    pub hard_edge_ids: Vec<String>,
    pub confirmed_edge_ids: Vec<String>,
    pub unresolved_edge_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct VerificationReport {
    pub schema_version: &'static str,
    pub verification_status: &'static str,
    pub cryptographically_verified: bool,
    pub resampling_qualification: &'static str,
    pub upstream_commit: String,
    pub source_partition_digest_sha256: String,
    pub relationship_inventory_edges_sha256: String,
    pub relationship_inventory_sha256: String,
    pub coverage_receipt_sha256: String,
    pub review_methodology_sha256: String,
    pub trust_store_sha256: String,
    pub verified_reviewer_ids: Vec<String>,
    pub distinct_reviewer_organization_count: usize,
    pub limitation: &'static str,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn partition_digest(families: &BTreeMap<String, String>) -> String {
    let mut material = String::new();
    for (family_id, split) in families {
        material.push_str(family_id);
        material.push('\t');
        material.push_str(split);
        material.push('\n');
    }
    sha256_hex(material.as_bytes())
}

pub fn validate_frozen_partition(bytes: &[u8]) -> Result<BTreeMap<String, String>> {
    let doc: Value = serde_json::from_slice(bytes).context("parse frozen split JSON")?;
    if doc["schema_version"].as_str()
        != Some("symthaea.proof-insight-transfer.splits.v0")
        || doc["upstream"]["commit"].as_str() != Some(UPSTREAM_COMMIT)
    {
        bail!("wrong frozen split schema or upstream commit");
    }
    let plan = &doc["split_plan"];
    if plan["seed"].as_str() != Some(SPLIT_SEED)
        || plan["algorithm"].as_str() != Some(SPLIT_ALGORITHM)
        || plan["train_percent"].as_u64() != Some(70)
        || plan["validation_percent"].as_u64() != Some(15)
        || plan["test_percent"].as_u64() != Some(15)
        || plan["counts"]["train"].as_u64() != Some(260)
        || plan["counts"]["validation"].as_u64() != Some(56)
        || plan["counts"]["test"].as_u64() != Some(56)
        || plan["partition_digest_sha256"].as_str() != Some(EXPECTED_PARTITION_DIGEST)
    {
        bail!("frozen split plan differs from the registered PITB partition");
    }
    let rows = doc["families"]
        .as_array()
        .context("frozen split families must be an array")?;
    if rows.len() != 372 {
        bail!("frozen split must contain exactly 372 families");
    }
    let mut families = BTreeMap::new();
    for row in rows {
        let family_id = row["family_id"].as_str().context("missing family_id")?;
        let split = row["split"].as_str().context("missing family split")?;
        if family_id.trim().is_empty()
            || !matches!(split, "train" | "validation" | "test")
            || families.insert(family_id.to_owned(), split.to_owned()).is_some()
        {
            bail!("blank/duplicate family or invalid split assignment");
        }
    }
    let mut ranked: Vec<(String, String)> = families
        .keys()
        .map(|id| (sha256_hex(format!("{SPLIT_SEED}\0{id}").as_bytes()), id.clone()))
        .collect();
    ranked.sort_unstable();
    let train_end = ranked.len() * 70 / 100;
    let validation_end = ranked.len() * 85 / 100;
    let expected: BTreeMap<String, String> = ranked
        .into_iter()
        .enumerate()
        .map(|(index, (_, id))| {
            let split = if index < train_end {
                "train"
            } else if index < validation_end {
                "validation"
            } else {
                "test"
            };
            (id, split.to_owned())
        })
        .collect();
    if families != expected || partition_digest(&families) != EXPECTED_PARTITION_DIGEST {
        bail!("frozen family assignments or partition digest do not reproduce");
    }
    let counts = (
        families.values().filter(|split| split.as_str() == "train").count(),
        families.values().filter(|split| split.as_str() == "validation").count(),
        families.values().filter(|split| split.as_str() == "test").count(),
    );
    if counts != (260, 56, 56) {
        bail!("frozen split counts must be 260/56/56");
    }
    Ok(families)
}

fn canonical_edges_digest(edges: &[Edge], families: &BTreeMap<String, String>) -> Result<String> {
    let mut canonical = edges.to_vec();
    let mut edge_ids = BTreeSet::new();
    let mut pairs = BTreeSet::new();
    for edge in &mut canonical {
        if edge.edge_id.trim().is_empty() || !edge_ids.insert(edge.edge_id.clone()) {
            bail!("edge IDs must be nonblank and unique");
        }
        if edge.source_family_id == edge.target_family_id
            || !families.contains_key(&edge.source_family_id)
            || !families.contains_key(&edge.target_family_id)
        {
            bail!("self-edge or edge references an unknown frozen family");
        }
        if !is_sha256(&edge.evidence_sha256) {
            bail!("edge evidence digest must be SHA-256");
        }
        if !matches!(
            edge.relation_kind.as_str(),
            "exact_duplicate"
                | "shared_source_artifact"
                | "formal_dependency"
                | "reused_lemma"
                | "equivalent_statement"
                | "conceptual_similarity"
                | "citation_only"
        ) || !matches!(
            edge.assessment_status.as_str(),
            "confirmed" | "dismissed" | "unresolved"
        ) {
            bail!("unregistered edge relation or assessment status");
        }
        let mut reviewer_set = BTreeSet::new();
        if edge.reviewer_ids.iter().any(|id| id.trim().is_empty())
            || edge.reviewer_ids.iter().any(|id| !reviewer_set.insert(id.clone()))
        {
            bail!("edge reviewer IDs must be nonblank and unique");
        }
        if edge.assessment_status != "unresolved" && edge.reviewer_ids.len() < 2 {
            bail!("resolved edges require at least two reviewer IDs");
        }
        let (left, right) = if edge.source_family_id < edge.target_family_id {
            (&edge.source_family_id, &edge.target_family_id)
        } else {
            (&edge.target_family_id, &edge.source_family_id)
        };
        if !pairs.insert(format!("{left}\0{right}\0{}", edge.relation_kind)) {
            bail!("duplicate unordered family pair/relation");
        }
        edge.reviewer_ids.sort();
    }
    canonical.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
    Ok(sha256_hex(&serde_json::to_vec(&canonical)?))
}

fn is_hard_relation(relation_kind: &str) -> bool {
    matches!(
        relation_kind,
        "exact_duplicate"
            | "shared_source_artifact"
            | "formal_dependency"
            | "reused_lemma"
            | "equivalent_statement"
    )
}

/// Independently reconstruct the exact connected-component partition and edge
/// rollups from the authenticated relationship inventory. A manifest's own
/// self-consistency is not enough: its components must match this graph.
pub fn expected_components_from_inventory(
    inventory_bytes: &[u8],
    families: &BTreeMap<String, String>,
) -> Result<Vec<ExpectedComponent>> {
    let inventory: Inventory =
        serde_json::from_slice(inventory_bytes).context("parse signed relationship inventory")?;
    if inventory.schema_version != "pitb.family-leakage-edges.v0"
        || inventory.source_partition_digest_sha256 != partition_digest(families)
        || inventory.coverage_status != "independently_reviewed_complete"
    {
        bail!("component reconstruction requires the complete inventory for the exact frozen partition");
    }
    canonical_edges_digest(&inventory.edges, families)?;

    let mut adjacency: BTreeMap<String, BTreeSet<String>> = families
        .keys()
        .map(|family_id| (family_id.clone(), BTreeSet::new()))
        .collect();
    for edge in &inventory.edges {
        if is_hard_relation(&edge.relation_kind) && edge.assessment_status != "dismissed" {
            adjacency
                .get_mut(&edge.source_family_id)
                .context("missing source family in component graph")?
                .insert(edge.target_family_id.clone());
            adjacency
                .get_mut(&edge.target_family_id)
                .context("missing target family in component graph")?
                .insert(edge.source_family_id.clone());
        }
    }

    let mut seen = BTreeSet::new();
    let mut expected = Vec::new();
    for start in families.keys() {
        if !seen.insert(start.clone()) {
            continue;
        }
        let mut queue = VecDeque::from([start.clone()]);
        let mut members = BTreeSet::from([start.clone()]);
        while let Some(family_id) = queue.pop_front() {
            for neighbor in &adjacency[&family_id] {
                if seen.insert(neighbor.clone()) {
                    members.insert(neighbor.clone());
                    queue.push_back(neighbor.clone());
                }
            }
        }

        let family_ids: Vec<String> = members.into_iter().collect();
        let split_names: Vec<String> = family_ids
            .iter()
            .map(|family_id| families[family_id].clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let member_set: BTreeSet<&str> = family_ids.iter().map(String::as_str).collect();
        let related_edges: Vec<&Edge> = inventory
            .edges
            .iter()
            .filter(|edge| {
                is_hard_relation(&edge.relation_kind)
                    && edge.assessment_status != "dismissed"
                    && member_set.contains(edge.source_family_id.as_str())
                    && member_set.contains(edge.target_family_id.as_str())
            })
            .collect();
        let sorted_ids = |mut ids: Vec<String>| {
            ids.sort();
            ids
        };
        let mut material = family_ids.join("\n");
        material.push('\n');
        expected.push(ExpectedComponent {
            component_id: format!("component-{}", sha256_hex(material.as_bytes())),
            split_names,
            family_ids,
            hard_edge_ids: sorted_ids(
                related_edges.iter().map(|edge| edge.edge_id.clone()).collect(),
            ),
            confirmed_edge_ids: sorted_ids(
                related_edges
                    .iter()
                    .filter(|edge| edge.assessment_status == "confirmed")
                    .map(|edge| edge.edge_id.clone())
                    .collect(),
            ),
            unresolved_edge_ids: sorted_ids(
                related_edges
                    .iter()
                    .filter(|edge| edge.assessment_status == "unresolved")
                    .map(|edge| edge.edge_id.clone())
                    .collect(),
            ),
        });
    }
    expected.sort_by(|left, right| left.family_ids.cmp(&right.family_ids));
    Ok(expected)
}

fn signing_message(
    upstream_commit: &str,
    partition_digest: &str,
    edges_digest: &str,
    methodology_digest: &str,
    trust_store_digest: &str,
) -> Vec<u8> {
    format!(
        concat!(
            "{0}\n{1}\n{2}\n{3}\n",
            "{4}\n{5}\n{6}\n",
            "complete_relationship_inventory\n"
        ),
        DOMAIN,
        RECEIPT_SCHEMA,
        upstream_commit,
        partition_digest,
        edges_digest,
        methodology_digest,
        trust_store_digest
    )
    .into_bytes()
}

pub fn verify_receipt(
    inventory_bytes: &[u8],
    receipt_bytes: &[u8],
    trust_store_bytes: &[u8],
    methodology_bytes: &[u8],
    expected_trust_store_digest: &str,
    expected_methodology_digest: &str,
    expected_partition_digest: &str,
    families: &BTreeMap<String, String>,
) -> Result<VerificationReport> {
    let inventory: Inventory =
        serde_json::from_slice(inventory_bytes).context("parse relationship inventory")?;
    if inventory.schema_version != "pitb.family-leakage-edges.v0"
        || inventory.source_partition_digest_sha256 != expected_partition_digest
        || partition_digest(families) != inventory.source_partition_digest_sha256
    {
        bail!("relationship inventory schema or frozen partition digest mismatch");
    }
    if inventory.coverage_status != "independently_reviewed_complete" {
        bail!("inventory must explicitly claim independently reviewed complete coverage");
    }
    let receipt: Receipt =
        serde_json::from_slice(receipt_bytes).context("parse signed coverage receipt")?;
    if sha256_hex(receipt_bytes)
        != inventory.coverage_receipt_sha256.as_deref().unwrap_or_default()
    {
        bail!("receipt bytes do not match inventory coverage_receipt_sha256");
    }
    if receipt.schema_version != RECEIPT_SCHEMA
        || receipt.upstream_commit != UPSTREAM_COMMIT
        || receipt.source_partition_digest_sha256 != inventory.source_partition_digest_sha256
        || receipt.attestation != "complete_relationship_inventory"
    {
        bail!("receipt schema, upstream, partition, or attestation claim is invalid");
    }

    let edges_digest = canonical_edges_digest(&inventory.edges, families)?;
    if methodology_bytes.is_empty() {
        bail!("review methodology document must not be empty");
    }
    if !is_sha256(&receipt.relationship_inventory_edges_sha256)
        || receipt.relationship_inventory_edges_sha256 != edges_digest
    {
        bail!("signed relationship-edge payload digest mismatch");
    }
    let methodology_digest = sha256_hex(methodology_bytes);
    if !is_sha256(expected_methodology_digest)
        || expected_methodology_digest != methodology_digest
        || !is_sha256(&receipt.review_methodology_sha256)
        || receipt.review_methodology_sha256 != methodology_digest
    {
        bail!("signed review-methodology digest mismatch");
    }
    let trust_store_digest = sha256_hex(trust_store_bytes);
    if !is_sha256(expected_trust_store_digest)
        || expected_trust_store_digest != trust_store_digest
        || receipt.trust_store_sha256 != trust_store_digest
    {
        bail!("trust store does not match both the frozen run-config digest and signed receipt");
    }

    let trust: TrustStore =
        serde_json::from_slice(trust_store_bytes).context("parse reviewer trust store")?;
    if trust.schema_version != TRUST_SCHEMA {
        bail!("unsupported reviewer trust-store schema");
    }
    if trust.reviewers.len() < 2 {
        bail!("reviewer trust store must contain at least two trusted reviewers");
    }
    let mut trusted_by_id = BTreeMap::new();
    let mut public_keys: BTreeSet<[u8; 32]> = BTreeSet::new();
    let mut trusted_key_ids = BTreeSet::new();
    for reviewer in &trust.reviewers {
        if reviewer.reviewer_id.trim().is_empty()
            || reviewer.key_id.trim().is_empty()
            || reviewer.organization_id.trim().is_empty()
            || reviewer.role != "coverage_auditor"
        {
            bail!("trusted reviewers require nonblank IDs/org and role coverage_auditor");
        }
        if !is_sha256(&reviewer.public_key_hex)
            || !trusted_key_ids.insert(reviewer.key_id.as_str())
        {
            bail!("trusted reviewer key IDs and public keys must be valid and unique");
        }
        let public_key_bytes =
            hex::decode(&reviewer.public_key_hex).context("decode trusted reviewer public key")?;
        let public_key_array: [u8; 32] = public_key_bytes
            .try_into()
            .map_err(|_| anyhow::anyhow!("Ed25519 public key must contain exactly 32 bytes"))?;
        if !public_keys.insert(public_key_array) {
            bail!("trusted reviewer public keys must be unique by decoded bytes");
        }
        if trusted_by_id
            .insert(reviewer.reviewer_id.as_str(), reviewer)
            .is_some()
        {
            bail!("duplicate reviewer ID in trust store");
        }
    }

    let message = signing_message(
        &receipt.upstream_commit,
        &receipt.source_partition_digest_sha256,
        &receipt.relationship_inventory_edges_sha256,
        &receipt.review_methodology_sha256,
        &receipt.trust_store_sha256,
    );
    let mut signer_ids = BTreeSet::new();
    let mut signer_keys = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    for endorsed in &receipt.signatures {
        if !signer_ids.insert(endorsed.reviewer_id.as_str())
            || !signer_keys.insert(endorsed.key_id.as_str())
        {
            bail!("receipt signer IDs and key IDs must each be unique");
        }
        let reviewer = trusted_by_id
            .get(endorsed.reviewer_id.as_str())
            .context("receipt contains an untrusted reviewer ID")?;
        if reviewer.key_id != endorsed.key_id {
            bail!("receipt key ID does not match the trusted key for reviewer");
        }
        let public_key_bytes = hex::decode(&reviewer.public_key_hex)
            .context("decode trusted reviewer public key")?;
        let public_key_array: [u8; 32] = public_key_bytes
            .try_into()
            .map_err(|_| anyhow::anyhow!("Ed25519 public key must contain exactly 32 bytes"))?;
        let verifying_key =
            VerifyingKey::from_bytes(&public_key_array).context("invalid Ed25519 public key")?;
        let signature_bytes =
            hex::decode(&endorsed.signature_hex).context("decode Ed25519 signature")?;
        let signature =
            Signature::from_slice(&signature_bytes).context("invalid Ed25519 signature length")?;
        verifying_key
            .verify_strict(&message, &signature)
            .context("coverage receipt signature verification failed")?;
        organizations.insert(reviewer.organization_id.as_str());
    }
    if signer_ids.len() < 2 || organizations.len() < 2 {
        bail!("coverage requires at least two trusted signers from distinct organizations");
    }

    Ok(VerificationReport {
        schema_version: "pitb.coverage-receipt-verification.v1",
        verification_status: "verified_signature_set",
        cryptographically_verified: true,
        // Authentication is necessary but not sufficient: the analysis binary
        // must consume and validate this report before any confirmatory claim.
        resampling_qualification: "requires_component_and_analyzer_validation",
        upstream_commit: receipt.upstream_commit,
        source_partition_digest_sha256: receipt.source_partition_digest_sha256,
        relationship_inventory_edges_sha256: edges_digest,
        relationship_inventory_sha256: sha256_hex(inventory_bytes),
        coverage_receipt_sha256: sha256_hex(receipt_bytes),
        review_methodology_sha256: methodology_digest,
        trust_store_sha256: trust_store_digest,
        verified_reviewer_ids: signer_ids.into_iter().map(str::to_owned).collect(),
        distinct_reviewer_organization_count: organizations.len(),
        limitation: concat!(
            "Signature verification authenticates endorsements under the pinned trust store; ",
            "it does not independently prove that the review was competent or exhaustive."
        ),
    })
}

#[cfg(test)]
pub mod test_support {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    const TEST_SPLITS: &[u8] =
        include_bytes!("../../../../data/benchmarks/proof-insight-transfer-v0/splits.json");
    const TEST_METHODOLOGY: &[u8] = include_bytes!(
        "../../../../data/benchmarks/proof-insight-transfer-v0/coverage-review-methodology.md"
    );

    pub struct FullPartitionFixture {
        pub split_bytes: Vec<u8>,
        pub inventory_bytes: Vec<u8>,
        pub receipt_bytes: Vec<u8>,
        pub trust_store_bytes: Vec<u8>,
        pub methodology_bytes: Vec<u8>,
        pub partition_digest: String,
        pub inventory_digest: String,
        pub receipt_digest: String,
        pub trust_store_digest: String,
        pub methodology_digest: String,
        pub families: BTreeMap<String, String>,
    }

    pub fn full_partition_fixture() -> Result<FullPartitionFixture> {
        let split_bytes = TEST_SPLITS.to_vec();
        let methodology_bytes = TEST_METHODOLOGY.to_vec();
        let families = validate_frozen_partition(&split_bytes)?;
        let edge_bytes = serde_json::to_vec(&Vec::<Edge>::new())?;
        let edge_digest = sha256_hex(&edge_bytes);
        let methodology_digest = sha256_hex(&methodology_bytes);

        // These deterministic keys are test-only. They cannot stand in for the
        // externally provisioned reviewer trust root used by a real experiment.
        let keys = [
            SigningKey::from_bytes(&[31_u8; 32]),
            SigningKey::from_bytes(&[47_u8; 32]),
        ];
        let trust = TrustStore {
            schema_version: TRUST_SCHEMA.to_owned(),
            reviewers: vec![
                TrustedReviewer {
                    reviewer_id: "synthetic-auditor-a".to_owned(),
                    key_id: "synthetic-key-a".to_owned(),
                    organization_id: "synthetic-org-a".to_owned(),
                    role: "coverage_auditor".to_owned(),
                    public_key_hex: hex::encode(keys[0].verifying_key().as_bytes()),
                },
                TrustedReviewer {
                    reviewer_id: "synthetic-auditor-b".to_owned(),
                    key_id: "synthetic-key-b".to_owned(),
                    organization_id: "synthetic-org-b".to_owned(),
                    role: "coverage_auditor".to_owned(),
                    public_key_hex: hex::encode(keys[1].verifying_key().as_bytes()),
                },
            ],
        };
        let trust_store_bytes = serde_json::to_vec_pretty(&trust)?;
        let trust_store_digest = sha256_hex(&trust_store_bytes);
        let partition = partition_digest(&families);
        let message = signing_message(
            UPSTREAM_COMMIT,
            &partition,
            &edge_digest,
            &methodology_digest,
            &trust_store_digest,
        );
        let receipt = Receipt {
            schema_version: RECEIPT_SCHEMA.to_owned(),
            upstream_commit: UPSTREAM_COMMIT.to_owned(),
            source_partition_digest_sha256: partition.clone(),
            relationship_inventory_edges_sha256: edge_digest,
            review_methodology_sha256: methodology_digest.clone(),
            trust_store_sha256: trust_store_digest.clone(),
            attestation: "complete_relationship_inventory".to_owned(),
            signatures: keys
                .iter()
                .enumerate()
                .map(|(index, key)| ReviewerSignature {
                    reviewer_id: format!(
                        "synthetic-auditor-{}",
                        if index == 0 { "a" } else { "b" }
                    ),
                    key_id: format!("synthetic-key-{}", if index == 0 { "a" } else { "b" }),
                    signature_hex: hex::encode(key.sign(&message).to_bytes()),
                })
                .collect(),
        };
        let receipt_bytes = serde_json::to_vec_pretty(&receipt)?;
        let receipt_digest = sha256_hex(&receipt_bytes);
        let inventory = Inventory {
            schema_version: "pitb.family-leakage-edges.v0".to_owned(),
            source_partition_digest_sha256: partition.clone(),
            coverage_status: "independently_reviewed_complete".to_owned(),
            coverage_receipt_sha256: Some(receipt_digest.clone()),
            edges: vec![],
        };
        let inventory_bytes = serde_json::to_vec_pretty(&inventory)?;
        let inventory_digest = sha256_hex(&inventory_bytes);

        Ok(FullPartitionFixture {
            split_bytes,
            inventory_bytes,
            receipt_bytes,
            trust_store_bytes,
            methodology_bytes,
            partition_digest: partition,
            inventory_digest,
            receipt_digest,
            trust_store_digest,
            methodology_digest,
            families,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn digest(byte: char) -> String {
        std::iter::repeat(byte).take(64).collect()
    }

    fn test_edges() -> Vec<Edge> {
        vec![Edge {
            edge_id: "edge-1".to_owned(),
            source_family_id: "001".to_owned(),
            target_family_id: "002".to_owned(),
            relation_kind: "formal_dependency".to_owned(),
            assessment_status: "confirmed".to_owned(),
            evidence_sha256: digest('a'),
            reviewer_ids: vec!["auditor-a".to_owned(), "auditor-b".to_owned()],
        }]
    }

    fn test_families() -> BTreeMap<String, String> {
        BTreeMap::from([
            ("001".to_owned(), "train".to_owned()),
            ("002".to_owned(), "train".to_owned()),
            ("003".to_owned(), "test".to_owned()),
        ])
    }

    fn signed_fixture(
        organizations: [&str; 2],
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, String, String, BTreeMap<String, String>) {
        let families = test_families();
        let test_partition_digest = partition_digest(&families);
        let mut edges = test_edges();
        let edges_digest = canonical_edges_digest(&edges, &families).unwrap();
        edges[0].reviewer_ids.sort();
        let methodology = b"synthetic review methodology v1\n".to_vec();
        let method_digest = sha256_hex(&methodology);
        let keys = [SigningKey::from_bytes(&[7_u8; 32]), SigningKey::from_bytes(&[9_u8; 32])];
        let reviewers = vec![
            TrustedReviewer {
                reviewer_id: "reviewer-a".to_owned(),
                key_id: "key-a".to_owned(),
                organization_id: organizations[0].to_owned(),
                role: "coverage_auditor".to_owned(),
                public_key_hex: hex::encode(keys[0].verifying_key().as_bytes()),
            },
            TrustedReviewer {
                reviewer_id: "reviewer-b".to_owned(),
                key_id: "key-b".to_owned(),
                organization_id: organizations[1].to_owned(),
                role: "coverage_auditor".to_owned(),
                public_key_hex: hex::encode(keys[1].verifying_key().as_bytes()),
            },
        ];
        let trust = TrustStore {
            schema_version: TRUST_SCHEMA.to_owned(),
            reviewers,
        };
        let trust_bytes = serde_json::to_vec_pretty(&trust).unwrap();
        let trust_digest = sha256_hex(&trust_bytes);
        let message = signing_message(
            UPSTREAM_COMMIT,
            &test_partition_digest,
            &edges_digest,
            &method_digest,
            &trust_digest,
        );
        let receipt = Receipt {
            schema_version: RECEIPT_SCHEMA.to_owned(),
            upstream_commit: UPSTREAM_COMMIT.to_owned(),
            source_partition_digest_sha256: test_partition_digest.clone(),
            relationship_inventory_edges_sha256: edges_digest,
            review_methodology_sha256: method_digest,
            trust_store_sha256: trust_digest.clone(),
            attestation: "complete_relationship_inventory".to_owned(),
            signatures: keys
                .iter()
                .enumerate()
                .map(|(index, key)| ReviewerSignature {
                    reviewer_id: format!("reviewer-{}", if index == 0 { "a" } else { "b" }),
                    key_id: format!("key-{}", if index == 0 { "a" } else { "b" }),
                    signature_hex: hex::encode(key.sign(&message).to_bytes()),
                })
                .collect(),
        };
        let receipt_bytes = serde_json::to_vec_pretty(&receipt).unwrap();
        let inventory = Inventory {
            schema_version: "pitb.family-leakage-edges.v0".to_owned(),
            source_partition_digest_sha256: test_partition_digest.clone(),
            coverage_status: "independently_reviewed_complete".to_owned(),
            coverage_receipt_sha256: Some(sha256_hex(&receipt_bytes)),
            edges,
        };
        (
            serde_json::to_vec_pretty(&inventory).unwrap(),
            receipt_bytes,
            trust_bytes,
            methodology,
            trust_digest,
            test_partition_digest,
            families,
        )
    }

    #[test]
    fn full_frozen_partition_fixture_verifies_without_granting_analyzer_qualification() {
        let fixture = test_support::full_partition_fixture().unwrap();
        assert_eq!(fixture.families.len(), 372);
        let report = verify_receipt(
            &fixture.inventory_bytes,
            &fixture.receipt_bytes,
            &fixture.trust_store_bytes,
            &fixture.methodology_bytes,
            &fixture.trust_store_digest,
            &fixture.methodology_digest,
            &fixture.partition_digest,
            &fixture.families,
        )
        .unwrap();
        assert!(report.cryptographically_verified);
        assert_eq!(report.relationship_inventory_sha256, fixture.inventory_digest);
        assert_eq!(report.coverage_receipt_sha256, fixture.receipt_digest);
        assert_eq!(report.trust_store_sha256, fixture.trust_store_digest);
        assert_eq!(report.review_methodology_sha256, fixture.methodology_digest);
        assert_eq!(
            report.resampling_qualification,
            "requires_component_and_analyzer_validation"
        );
    }

    #[test]
    fn valid_two_organization_receipt_verifies_but_does_not_qualify_analysis() {
        let (inventory, receipt, trust, method, trust_digest, partition_digest, families) =
            signed_fixture(["research-lab-a", "research-lab-b"]);
        let report = verify_receipt(
            &inventory,
            &receipt,
            &trust,
            &method,
            &trust_digest,
            &sha256_hex(&method),
            &partition_digest,
            &families,
        )
        .unwrap();
        assert!(report.cryptographically_verified);
        assert_eq!(report.verified_reviewer_ids.len(), 2);
        assert_eq!(report.distinct_reviewer_organization_count, 2);
        assert_eq!(
            report.resampling_qualification,
            "requires_component_and_analyzer_validation"
        );
    }

    #[test]
    fn same_organization_cannot_supply_independent_endorsements() {
        let (inventory, receipt, trust, method, trust_digest, partition_digest, families) =
            signed_fixture(["same-org", "same-org"]);
        let error = verify_receipt(
            &inventory,
            &receipt,
            &trust,
            &method,
            &trust_digest,
            &sha256_hex(&method),
            &partition_digest,
            &families,
        )
        .unwrap_err();
        assert!(error.to_string().contains("distinct organizations"));
    }

    #[test]
    fn edge_mutation_invalidates_signed_payload_digest() {
        let (inventory_bytes, receipt, trust, method, trust_digest, partition_digest, families) =
            signed_fixture(["org-a", "org-b"]);
        let mut inventory: Value = serde_json::from_slice(&inventory_bytes).unwrap();
        inventory["edges"][0]["target_family_id"] = Value::String("003".to_owned());
        let mutated_inventory = serde_json::to_vec_pretty(&inventory).unwrap();
        let error = verify_receipt(
            &mutated_inventory,
            &receipt,
            &trust,
            &method,
            &trust_digest,
            &sha256_hex(&method),
            &partition_digest,
            &families,
        )
        .unwrap_err();
        assert!(error.to_string().contains("signed relationship-edge payload digest"));
    }

    #[test]
    fn changed_trust_store_is_rejected_even_with_an_old_receipt() {
        let (inventory, receipt, mut trust, method, trust_digest, partition_digest, families) =
            signed_fixture(["org-a", "org-b"]);
        trust.push(b' ');
        let error = verify_receipt(
            &inventory,
            &receipt,
            &trust,
            &method,
            &trust_digest,
            &sha256_hex(&method),
            &partition_digest,
            &families,
        )
        .unwrap_err();
        assert!(error.to_string().contains("trust store does not match"));
    }

    #[test]
    fn changed_methodology_is_rejected_even_when_old_receipt_is_used() {
        let (inventory, receipt, trust, method, trust_digest, partition_digest, families) =
            signed_fixture(["org-a", "org-b"]);
        let expected_methodology_digest = sha256_hex(&method);
        let mut changed_methodology = method.clone();
        changed_methodology.push(b' ');
        let error = verify_receipt(
            &inventory,
            &receipt,
            &trust,
            &changed_methodology,
            &trust_digest,
            &expected_methodology_digest,
            &partition_digest,
            &families,
        )
        .unwrap_err();
        assert!(error.to_string().contains("signed review-methodology digest mismatch"));
    }

    #[test]
    fn incomplete_inventory_cannot_be_promoted_by_a_signature() {
        let (inventory_bytes, receipt, trust, method, trust_digest, partition_digest, families) =
            signed_fixture(["org-a", "org-b"]);
        let mut inventory: Value = serde_json::from_slice(&inventory_bytes).unwrap();
        inventory["coverage_status"] = Value::String("incomplete".to_owned());
        inventory["coverage_receipt_sha256"] = Value::Null;
        let error = verify_receipt(
            &serde_json::to_vec(&inventory).unwrap(),
            &receipt,
            &trust,
            &method,
            &trust_digest,
            &sha256_hex(&method),
            &partition_digest,
            &families,
        )
        .unwrap_err();
        assert!(error.to_string().contains("inventory must explicitly claim"));
    }
}
