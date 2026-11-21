use crate::{canonical_token, hash_field, validate_digest, ManufacturingContractError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_manufacturing_process::ProcessDefinitionId;

const RECIPE_CONTENT_DOMAIN: &str = "symthaea-manufacturing-contracts::recipe-content-v1";
const RECIPE_ID_DOMAIN: &str = "symthaea-manufacturing-contracts::recipe-id-v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecipeDisclosureV1 {
    PublicInline,
    PrivateEncryptedRef,
    CommitmentOnly,
    SelectiveDisclosureProfile { profile_ref: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessRecipeV1 {
    pub process_id: ProcessDefinitionId,
    pub semantic_version: String,
    pub input_state_refs: Vec<String>,
    pub output_state_refs: Vec<String>,
    pub parameter_bundle_refs: Vec<String>,
    pub configuration_refs: Vec<String>,
    pub environment_refs: Vec<String>,
    pub inspection_refs: Vec<String>,
    pub model_profile_refs: Vec<String>,
    pub qualification_profile_refs: Vec<String>,
    pub display_label: Option<String>,
    pub notes: Option<String>,
}

impl ProcessRecipeV1 {
    pub fn validate(&self) -> Result<(), ManufacturingContractError> {
        canonical_token("recipe.process_id", &self.process_id.0)?;
        canonical_token("recipe.semantic_version", &self.semantic_version)?;
        if self.input_state_refs.is_empty() || self.output_state_refs.is_empty() {
            return Err(ManufacturingContractError::Invalid(
                "recipe requires at least one input and one output state ref",
            ));
        }
        for (field, refs) in [
            ("recipe.input_state_refs", &self.input_state_refs),
            ("recipe.output_state_refs", &self.output_state_refs),
            ("recipe.parameter_bundle_refs", &self.parameter_bundle_refs),
            ("recipe.configuration_refs", &self.configuration_refs),
            ("recipe.environment_refs", &self.environment_refs),
            ("recipe.inspection_refs", &self.inspection_refs),
            ("recipe.model_profile_refs", &self.model_profile_refs),
            (
                "recipe.qualification_profile_refs",
                &self.qualification_profile_refs,
            ),
        ] {
            validate_unique(field, refs)?;
        }
        Ok(())
    }

    pub fn content_digest(&self) -> Result<String, ManufacturingContractError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, RECIPE_CONTENT_DOMAIN);
        hash_field(&mut hasher, &self.process_id.0);
        hash_field(&mut hasher, &self.semantic_version);
        for refs in [
            &self.input_state_refs,
            &self.output_state_refs,
            &self.parameter_bundle_refs,
            &self.configuration_refs,
            &self.environment_refs,
            &self.inspection_refs,
            &self.model_profile_refs,
            &self.qualification_profile_refs,
        ] {
            let mut sorted = refs.clone();
            sorted.sort();
            for value in sorted {
                hash_field(&mut hasher, &value);
            }
            hash_field(&mut hasher, "--");
        }
        Ok(hasher.finalize().to_hex().to_string())
    }

    pub fn recipe_id(&self) -> Result<String, ManufacturingContractError> {
        let content = self.content_digest()?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, RECIPE_ID_DOMAIN);
        hash_field(&mut hasher, &self.process_id.0);
        hash_field(&mut hasher, &self.semantic_version);
        hash_field(&mut hasher, &content);
        Ok(hasher.finalize().to_hex().to_string())
    }

    pub fn commitment(
        &self,
        schema_profile_id: String,
        owner_issuer_id: String,
        disclosure: RecipeDisclosureV1,
        private_artifact_ref: Option<String>,
    ) -> Result<RecipeCommitmentV1, ManufacturingContractError> {
        let commitment = RecipeCommitmentV1 {
            recipe_id: self.recipe_id()?,
            content_blake3: self.content_digest()?,
            schema_profile_id,
            owner_issuer_id,
            semantic_version: self.semantic_version.clone(),
            disclosure,
            private_artifact_ref,
        };
        commitment.validate()?;
        Ok(commitment)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecipeCommitmentV1 {
    pub recipe_id: String,
    pub content_blake3: String,
    pub schema_profile_id: String,
    pub owner_issuer_id: String,
    pub semantic_version: String,
    pub disclosure: RecipeDisclosureV1,
    pub private_artifact_ref: Option<String>,
}

impl RecipeCommitmentV1 {
    pub fn validate(&self) -> Result<(), ManufacturingContractError> {
        validate_digest("recipe_commitment.recipe_id", &self.recipe_id)?;
        validate_digest(
            "recipe_commitment.content_blake3",
            &self.content_blake3,
        )?;
        canonical_token(
            "recipe_commitment.schema_profile_id",
            &self.schema_profile_id,
        )?;
        canonical_token(
            "recipe_commitment.owner_issuer_id",
            &self.owner_issuer_id,
        )?;
        canonical_token(
            "recipe_commitment.semantic_version",
            &self.semantic_version,
        )?;
        match &self.disclosure {
            RecipeDisclosureV1::PublicInline | RecipeDisclosureV1::CommitmentOnly => {}
            RecipeDisclosureV1::PrivateEncryptedRef => {
                let Some(reference) = &self.private_artifact_ref else {
                    return Err(ManufacturingContractError::Invalid(
                        "private encrypted disclosure requires artifact ref",
                    ));
                };
                canonical_token("recipe_commitment.private_artifact_ref", reference)?;
            }
            RecipeDisclosureV1::SelectiveDisclosureProfile { profile_ref } => {
                canonical_token("recipe_commitment.selective_profile_ref", profile_ref)?;
            }
        }
        Ok(())
    }

    pub fn matches_recipe(
        &self,
        recipe: &ProcessRecipeV1,
    ) -> Result<bool, ManufacturingContractError> {
        self.validate()?;
        Ok(self.recipe_id == recipe.recipe_id()?
            && self.content_blake3 == recipe.content_digest()?)
    }
}

fn validate_unique(
    field: &'static str,
    refs: &[String],
) -> Result<(), ManufacturingContractError> {
    let mut seen = BTreeSet::new();
    for value in refs {
        canonical_token(field, value)?;
        if !seen.insert(value.clone()) {
            return Err(ManufacturingContractError::DuplicateReference {
                field,
                value: value.clone(),
            });
        }
    }
    Ok(())
}
