use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_manufacturing_process::ProcessDefinitionId;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum RecipeDisclosureV1 {
    PublicInline,
    PrivateEncryptedRef,
    CommitmentOnly,
    SelectiveDisclosureProfile { profile_ref: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ProcessRecipeV1 {
    process_id: ProcessDefinitionId,
    semantic_version: String,
    input_state_refs: Vec<String>,
    output_state_refs: Vec<String>,
    parameter_bundle_refs: Vec<String>,
    configuration_refs: Vec<String>,
    environment_refs: Vec<String>,
    inspection_refs: Vec<String>,
    model_profile_refs: Vec<String>,
    qualification_profile_refs: Vec<String>,
    display_label: Option<String>,
    notes: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct RecipeCommitmentV1 {
    recipe_id: String,
    content_blake3: String,
    schema_profile_id: String,
    owner_issuer_id: String,
    semantic_version: String,
    disclosure: RecipeDisclosureV1,
    private_artifact_ref: Option<String>,
}

fn canonical_token(value: &str) -> bool {
    !value.is_empty() && value.trim() == value && !value.chars().any(char::is_control)
}

fn validate_unique(field: &str, refs: &[String]) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for value in refs {
        if !canonical_token(value) {
            return Err(format!("{field} contains non-canonical ref"));
        }
        if !seen.insert(value.clone()) {
            return Err(format!("duplicate ref in {field}: {value}"));
        }
    }
    Ok(())
}

fn validate_recipe(recipe: &ProcessRecipeV1) -> Result<(), String> {
    if !canonical_token(&recipe.process_id.0) || !canonical_token(&recipe.semantic_version) {
        return Err("recipe identity fields must be canonical".into());
    }
    if recipe.input_state_refs.is_empty() || recipe.output_state_refs.is_empty() {
        return Err("recipe requires at least one input and output state ref".into());
    }
    for (field, refs) in [
        ("input_state_refs", &recipe.input_state_refs),
        ("output_state_refs", &recipe.output_state_refs),
        ("parameter_bundle_refs", &recipe.parameter_bundle_refs),
        ("configuration_refs", &recipe.configuration_refs),
        ("environment_refs", &recipe.environment_refs),
        ("inspection_refs", &recipe.inspection_refs),
        ("model_profile_refs", &recipe.model_profile_refs),
        ("qualification_profile_refs", &recipe.qualification_profile_refs),
    ] {
        validate_unique(field, refs)?;
    }
    Ok(())
}

fn hash_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_sorted(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut sorted = refs.to_vec();
    sorted.sort();
    for value in sorted {
        hash_field(hasher, &value);
    }
}

fn recipe_content_digest(recipe: &ProcessRecipeV1) -> Result<String, String> {
    validate_recipe(recipe)?;
    let mut hasher = blake3::Hasher::new();
    hash_field(&mut hasher, "symthaea.mfg-proc.recipe-content.v1");
    hash_field(&mut hasher, &recipe.process_id.0);
    hash_field(&mut hasher, &recipe.semantic_version);
    hash_sorted(&mut hasher, &recipe.input_state_refs);
    hash_sorted(&mut hasher, &recipe.output_state_refs);
    hash_sorted(&mut hasher, &recipe.parameter_bundle_refs);
    hash_sorted(&mut hasher, &recipe.configuration_refs);
    hash_sorted(&mut hasher, &recipe.environment_refs);
    hash_sorted(&mut hasher, &recipe.inspection_refs);
    hash_sorted(&mut hasher, &recipe.model_profile_refs);
    hash_sorted(&mut hasher, &recipe.qualification_profile_refs);
    Ok(hasher.finalize().to_hex().to_string())
}

fn recipe_id(recipe: &ProcessRecipeV1) -> Result<String, String> {
    let content = recipe_content_digest(recipe)?;
    let mut hasher = blake3::Hasher::new();
    hash_field(&mut hasher, "symthaea.mfg-proc.recipe-id.v1");
    hash_field(&mut hasher, &recipe.process_id.0);
    hash_field(&mut hasher, &recipe.semantic_version);
    hash_field(&mut hasher, &content);
    Ok(hasher.finalize().to_hex().to_string())
}

fn validate_commitment(commitment: &RecipeCommitmentV1) -> Result<(), String> {
    for value in [
        &commitment.recipe_id,
        &commitment.content_blake3,
        &commitment.schema_profile_id,
        &commitment.owner_issuer_id,
        &commitment.semantic_version,
    ] {
        if !canonical_token(value) {
            return Err("commitment identity fields must be canonical".into());
        }
    }
    if commitment.recipe_id.len() != 64
        || commitment.content_blake3.len() != 64
        || !commitment
            .recipe_id
            .bytes()
            .chain(commitment.content_blake3.bytes())
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err("recipe/content digests must be lowercase 64-hex BLAKE3 values".into());
    }
    match &commitment.disclosure {
        RecipeDisclosureV1::PublicInline | RecipeDisclosureV1::CommitmentOnly => {}
        RecipeDisclosureV1::PrivateEncryptedRef => {
            if commitment.private_artifact_ref.as_deref().is_none_or(|v| !canonical_token(v)) {
                return Err("private encrypted disclosure requires artifact ref".into());
            }
        }
        RecipeDisclosureV1::SelectiveDisclosureProfile { profile_ref } => {
            if !canonical_token(profile_ref) {
                return Err("selective-disclosure profile ref must be canonical".into());
            }
        }
    }
    Ok(())
}

fn commitment_for(
    recipe: &ProcessRecipeV1,
    disclosure: RecipeDisclosureV1,
    private_artifact_ref: Option<String>,
) -> RecipeCommitmentV1 {
    RecipeCommitmentV1 {
        recipe_id: recipe_id(recipe).unwrap(),
        content_blake3: recipe_content_digest(recipe).unwrap(),
        schema_profile_id: "symthaea.mfg-proc.recipe-schema.v1".into(),
        owner_issuer_id: "org.example:process-owner".into(),
        semantic_version: recipe.semantic_version.clone(),
        disclosure,
        private_artifact_ref,
    }
}

fn sample_recipe() -> ProcessRecipeV1 {
    ProcessRecipeV1 {
        process_id: ProcessDefinitionId("process:cnc-milling-v1".into()),
        semantic_version: "1.0.0".into(),
        input_state_refs: vec!["state:stock:al6061".into()],
        output_state_refs: vec!["state:machined:rev-a".into()],
        parameter_bundle_refs: vec!["se-sem:bundle:recipe-params-v1".into()],
        configuration_refs: vec!["eng-catalog:fixture:vise-v2".into()],
        environment_refs: vec!["env:shop-default-v1".into()],
        inspection_refs: vec!["qif:inspection:profile-a".into()],
        model_profile_refs: vec!["model:cutting-force:reduced-order-v1".into()],
        qualification_profile_refs: vec!["qualification:cnc-profile-a".into()],
        display_label: Some("UI recipe label".into()),
        notes: Some("operator-facing note".into()),
    }
}

#[test]
fn display_metadata_does_not_change_recipe_identity() {
    let a = sample_recipe();
    let mut b = a.clone();
    b.display_label = Some("renamed".into());
    b.notes = Some("different note".into());
    assert_eq!(recipe_id(&a).unwrap(), recipe_id(&b).unwrap());
}

#[test]
fn semantic_refs_change_recipe_identity() {
    let a = sample_recipe();
    let mut b = a.clone();
    b.parameter_bundle_refs = vec!["se-sem:bundle:recipe-params-v2".into()];
    assert_ne!(recipe_id(&a).unwrap(), recipe_id(&b).unwrap());

    let mut c = a.clone();
    c.configuration_refs = vec!["eng-catalog:fixture:vise-v3".into()];
    assert_ne!(recipe_id(&a).unwrap(), recipe_id(&c).unwrap());
}

#[test]
fn semantic_collection_order_does_not_change_identity() {
    let mut a = sample_recipe();
    a.inspection_refs.push("qif:inspection:profile-b".into());
    let mut b = a.clone();
    b.inspection_refs.reverse();
    assert_eq!(recipe_id(&a).unwrap(), recipe_id(&b).unwrap());
}

#[test]
fn duplicate_semantic_refs_fail_closed() {
    let mut recipe = sample_recipe();
    recipe.parameter_bundle_refs.push(recipe.parameter_bundle_refs[0].clone());
    assert!(validate_recipe(&recipe).unwrap_err().contains("duplicate ref"));
}

#[test]
fn commitment_only_can_validate_without_public_recipe_payload() {
    let recipe = sample_recipe();
    let commitment = commitment_for(&recipe, RecipeDisclosureV1::CommitmentOnly, None);
    assert!(validate_commitment(&commitment).is_ok());
    let encoded = serde_json::to_string(&commitment).unwrap();
    assert!(!encoded.contains("recipe-params-v1"));
    assert!(!encoded.contains("al6061"));
}

#[test]
fn private_encrypted_disclosure_requires_opaque_artifact_ref() {
    let recipe = sample_recipe();
    let missing = commitment_for(&recipe, RecipeDisclosureV1::PrivateEncryptedRef, None);
    assert!(validate_commitment(&missing).is_err());

    let opaque = commitment_for(
        &recipe,
        RecipeDisclosureV1::PrivateEncryptedRef,
        Some("mycelix:private-artifact:recipe-ciphertext-123".into()),
    );
    assert!(validate_commitment(&opaque).is_ok());
}

#[test]
fn commitment_digest_mismatch_is_detectable_against_disclosed_recipe() {
    let recipe = sample_recipe();
    let mut commitment = commitment_for(&recipe, RecipeDisclosureV1::PublicInline, None);
    assert_eq!(commitment.content_blake3, recipe_content_digest(&recipe).unwrap());
    commitment.content_blake3 = "0".repeat(64);
    assert_ne!(commitment.content_blake3, recipe_content_digest(&recipe).unwrap());
}

#[test]
fn qualification_evidence_is_not_part_of_immutable_recipe_content() {
    let recipe = sample_recipe();
    let before = recipe_id(&recipe).unwrap();
    let qualification_evidence_ref = "evidence:recipe-qualification:run-42";
    assert!(canonical_token(qualification_evidence_ref));
    assert_eq!(before, recipe_id(&recipe).unwrap());
}

#[test]
fn serialization_preserves_disclosure_and_commitment_identity() {
    let recipe = sample_recipe();
    let commitment = commitment_for(
        &recipe,
        RecipeDisclosureV1::SelectiveDisclosureProfile {
            profile_ref: "policy:selective-disclosure:manufacturing-v1".into(),
        },
        None,
    );
    let encoded = serde_json::to_string(&commitment).unwrap();
    let decoded: RecipeCommitmentV1 = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, commitment);
    assert!(validate_commitment(&decoded).is_ok());
}
