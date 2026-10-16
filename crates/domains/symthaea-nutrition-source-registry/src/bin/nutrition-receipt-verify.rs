// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent consistency checker for serialized nutrition decision receipts.
//!
//! This binary recomputes receipt hashes and input identity. It deliberately does
//! not call the primary assessment evaluator and does not re-evaluate eligibility
//! semantics. Hash consistency is not proof of publisher identity, legal permission
//! or truth.

use serde::de::{DeserializeSeed, Error as DeError, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;
use symthaea_nutrition_source_registry::{
    NutritionCanonicalizationProfileV1, NutritionFoodFactV1,
    NutritionRegistryAssessmentV1, NutritionSourceManifestV1, SourceUseRequestV1,
    NUTRITION_EVALUATOR_CONTRACT_VERSION, NUTRITION_REGISTRY_RECEIPT_SCHEMA_VERSION,
};

#[derive(Debug, Deserialize)]
struct ReplayBundle {
    manifest: NutritionSourceManifestV1,
    fact: NutritionFoodFactV1,
    evaluation_date: String,
    use_request: SourceUseRequestV1,
    assessment: NutritionRegistryAssessmentV1,
}

#[derive(Serialize)]
struct InputEnvelope<'a> {
    receipt_schema_version: u16,
    canonicalization_profile: NutritionCanonicalizationProfileV1,
    evaluation_date: &'a str,
    manifest: &'a NutritionSourceManifestV1,
    fact: &'a NutritionFoodFactV1,
    use_request: &'a SourceUseRequestV1,
    supplied_artifact_sha256: Option<String>,
}

/// Recursively deserializes JSON while rejecting duplicate property names.
/// serde_json::Value alone does not preserve duplicate object members, so the
/// check must happen before converting the value into typed structs.
struct StrictJsonValue;

impl<'de> DeserializeSeed<'de> for StrictJsonValue {
    type Value = serde_json::Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = serde_json::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object property names")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: DeError,
    {
        serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .ok_or_else(|| E::custom("non-finite number is not valid JSON"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(serde_json::Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(serde_json::Value::String(value))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Null)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        StrictJsonValue.deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(StrictJsonValue)? {
            values.push(value);
        }
        Ok(serde_json::Value::Array(values))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut seen = BTreeSet::new();
        let mut values = serde_json::Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(A::Error::custom(format!(
                    "duplicate JSON property name {key:?}"
                )));
            }
            let value = object.next_value_seed(StrictJsonValue)?;
            values.insert(key, value);
        }
        Ok(serde_json::Value::Object(values))
    }
}

fn parse_strict_json(raw: &str) -> Result<serde_json::Value, serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_str(raw);
    let value = StrictJsonValue.deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(value)
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn canonical_sha256<T: Serialize>(value: &T) -> Result<String, io::Error> {
    let bytes = serde_json_canonicalizer::to_vec(value)
        .map_err(|error| invalid_data(format!("JCS canonicalization failed: {error}")))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn compare_digest(
    label: &str,
    reported: Option<&str>,
    computed: &str,
) -> Result<(), io::Error> {
    if reported != Some(computed) {
        return Err(invalid_data(format!("{label} digest mismatch")));
    }
    Ok(())
}

fn verify_bundle(
    bundle: &ReplayBundle,
    supplied_artifact_sha256: Option<String>,
) -> Result<(), io::Error> {
    let receipt = &bundle.assessment;

    if receipt.receipt_schema_version != NUTRITION_REGISTRY_RECEIPT_SCHEMA_VERSION {
        return Err(invalid_data("unsupported receipt schema version"));
    }
    if receipt.canonicalization_profile != NutritionCanonicalizationProfileV1::Rfc8785JcsV1 {
        return Err(invalid_data("unsupported canonicalization profile"));
    }
    if receipt.evaluator_contract_version != NUTRITION_EVALUATOR_CONTRACT_VERSION {
        return Err(invalid_data("unsupported evaluator contract version"));
    }
    if receipt.source_id != bundle.manifest.source_id
        || receipt.source_record_id != bundle.fact.source_record_id
        || bundle.fact.source_id != bundle.manifest.source_id
        || receipt.evaluation_date != bundle.evaluation_date
        || receipt.source_manifest_schema_version != bundle.manifest.schema_version
        || receipt.food_fact_schema_version != bundle.fact.schema_version
    {
        return Err(invalid_data("receipt identity fields do not match supplied inputs"));
    }
    if receipt.supplied_artifact_sha256 != supplied_artifact_sha256 {
        return Err(invalid_data(
            "supplied artifact bytes do not match the digest recorded in the receipt",
        ));
    }

    let manifest_digest = canonical_sha256(&bundle.manifest)?;
    let fact_digest = canonical_sha256(&bundle.fact)?;
    let use_request_digest = canonical_sha256(&bundle.use_request)?;
    compare_digest(
        "manifest",
        receipt.manifest_sha256.as_deref(),
        &manifest_digest,
    )?;
    compare_digest(
        "food fact",
        receipt.food_fact_sha256.as_deref(),
        &fact_digest,
    )?;
    compare_digest(
        "use request",
        receipt.use_request_sha256.as_deref(),
        &use_request_digest,
    )?;

    let envelope = InputEnvelope {
        receipt_schema_version: receipt.receipt_schema_version,
        canonicalization_profile: receipt.canonicalization_profile,
        evaluation_date: &bundle.evaluation_date,
        manifest: &bundle.manifest,
        fact: &bundle.fact,
        use_request: &bundle.use_request,
        supplied_artifact_sha256,
    };
    let input_digest = canonical_sha256(&envelope)?;
    compare_digest("input envelope", receipt.input_sha256.as_deref(), &input_digest)?;

    let mut receipt_without_own_digest = (*receipt).clone();
    receipt_without_own_digest.decision_sha256 = None;
    let decision_digest = canonical_sha256(&receipt_without_own_digest)?;
    compare_digest(
        "decision receipt",
        receipt.decision_sha256.as_deref(),
        &decision_digest,
    )?;

    Ok(())
}

fn load_bundle(path: &Path) -> Result<ReplayBundle, Box<dyn Error>> {
    let raw = fs::read_to_string(path)?;
    let value = parse_strict_json(&raw)?;
    Ok(serde_json::from_value(value)?)
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if !(2..=3).contains(&args.len()) {
        return Err(invalid_data(
            "usage: nutrition-receipt-verify <bundle.json> [exact-artifact-bytes-file]",
        )
        .into());
    }

    let bundle = load_bundle(Path::new(&args[1]))?;
    let supplied_artifact_sha256 = match args.get(2) {
        Some(path) => {
            let bytes = fs::read(path)?;
            Some(hex::encode(Sha256::digest(bytes)))
        }
        None => None,
    };

    verify_bundle(&bundle, supplied_artifact_sha256)?;
    println!(
        "Receipt hash consistency verified. Assessment eligibility was not re-evaluated; source authenticity, permission validity and nutrition truth are not certified."
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("nutrition receipt verification failed: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_strict_json, verify_bundle, ReplayBundle};
    use symthaea_nutrition_source_registry::{
        assess_nutrition_source, EvidenceClass, Eligibility, FoodGeographicMatch,
        FoodIdentityConfidence, FoodMatchQuality, NutritionFoodFactV1,
        NutritionSourceManifestV1, NutrientUnit,
        PermissionDecision, PermissionEvidenceBasisV1, PermissionEvidenceLifecycleV1,
        PermissionEvidenceV1, SourceLifecycle, SourcePermissionReferencesV1, SourcePermissionsV1,
        SourceRightV1, SourceUseRequestV1, NutritionTransformationStepV1,
        sha256_hex,
    };

    const EVALUATION_DATE: &str = "2026-10-15";
    const TEST_ARTIFACT: &[u8] = b"synthetic nutrition source artifact v1";

    fn manifest() -> NutritionSourceManifestV1 {
        let evidence = [
            ("license:assessment-clause", SourceRightV1::Assessment),
            ("license:cache-clause", SourceRightV1::Cache),
            ("license:transformation-clause", SourceRightV1::Transformation),
            ("license:redistribution-clause", SourceRightV1::Redistribution),
            ("license:commercial-use-clause", SourceRightV1::CommercialUse),
            ("license:public-fixture-clause", SourceRightV1::PublicFixture),
        ]
        .into_iter()
        .map(|(evidence_id, right)| PermissionEvidenceV1 {
            evidence_id: evidence_id.to_owned(),
            source_id: "fixture:food-table-v1".to_owned(),
            right,
            basis: PermissionEvidenceBasisV1::LicenseTerms,
            evidence_uri: "https://example.invalid/license".to_owned(),
            reviewed_on: "2026-01-01".to_owned(),
            valid_until: "2027-01-01".to_owned(),
            lifecycle: PermissionEvidenceLifecycleV1::Active,
        })
        .collect();

        NutritionSourceManifestV1 {
            schema_version: 1,
            source_id: "fixture:food-table-v1".to_owned(),
            publisher: "Synthetic test publisher".to_owned(),
            dataset_title: "Synthetic food composition fixture".to_owned(),
            dataset_edition: "v1".to_owned(),
            source_uri: "https://example.invalid/food-table".to_owned(),
            terms_uri: "https://example.invalid/terms".to_owned(),
            license_id: "synthetic-fixture-only".to_owned(),
            permission_references: SourcePermissionReferencesV1 {
                use_for_assessment: Some("license:assessment-clause".to_owned()),
                cache: Some("license:cache-clause".to_owned()),
                transform: Some("license:transformation-clause".to_owned()),
                redistribute: Some("license:redistribution-clause".to_owned()),
                commercial_use: Some("license:commercial-use-clause".to_owned()),
                public_fixture: Some("license:public-fixture-clause".to_owned()),
            },
            permission_evidence: evidence,
            permissions: SourcePermissionsV1 {
                use_for_assessment: PermissionDecision::AllowedByLicense,
                cache: PermissionDecision::AllowedByLicense,
                transform: PermissionDecision::AllowedByLicense,
                redistribute: PermissionDecision::AllowedByLicense,
                commercial_use: PermissionDecision::AllowedByLicense,
                public_fixture: PermissionDecision::AllowedByLicense,
            },
            artifact_sha256: Some(sha256_hex(TEST_ARTIFACT)),
            retrieved_on: "2026-01-02".to_owned(),
            reviewed_on: "2026-01-01".to_owned(),
            review_valid_until: "2027-01-01".to_owned(),
            lifecycle: SourceLifecycle::Active,
        }
    }

    fn fact() -> NutritionFoodFactV1 {
        NutritionFoodFactV1 {
            schema_version: 1,
            source_id: "fixture:food-table-v1".to_owned(),
            source_record_id: "row-001".to_owned(),
            source_record_citation: "Synthetic fixture row 001".to_owned(),
            food_id: "food:test-leaf".to_owned(),
            food_description: "Synthetic leafy food".to_owned(),
            food_taxon_ref: Some("taxon:synthetic-leaf".to_owned()),
            cultivar_id: None,
            identity_confidence: FoodIdentityConfidence::Reviewed,
            food_state_id: "cooked_edible".to_owned(),
            match_quality: FoodMatchQuality::ExactReviewedMatch,
            geographic_match: FoodGeographicMatch::SouthAfricanLocal,
            evidence_class: EvidenceClass::SyntheticFixture,
            nutrient_id: "iron".to_owned(),
            nutrient_definition_ref: "nutrient-definition:iron-v1".to_owned(),
            unit: NutrientUnit::Milligrams,
            value_scaled: Some(1250),
            value_scale: 1000,
            basis_grams: 100,
            quality_notes: Vec::new(),
            transformation_applied: false,
            transformation_lineage: Vec::<NutritionTransformationStepV1>::new(),
        }
    }

    fn use_request() -> SourceUseRequestV1 {
        SourceUseRequestV1 {
            retain_source_data: true,
            transform_record: false,
            commercial_use: false,
            public_fixture: false,
        }
    }

    fn valid_bundle() -> (ReplayBundle, Option<String>) {
        let manifest = manifest();
        let fact = fact();
        let use_request = use_request();
        let assessment = assess_nutrition_source(
            &manifest,
            &fact,
            EVALUATION_DATE,
            &use_request,
            Some(TEST_ARTIFACT),
        );
        (
            ReplayBundle {
                manifest,
                fact,
                evaluation_date: EVALUATION_DATE.to_owned(),
                use_request,
                assessment,
            },
            Some(sha256_hex(TEST_ARTIFACT)),
        )
    }

    #[test]
    fn rejects_duplicate_top_level_property_names() {
        assert!(parse_strict_json(r#"{"x":1,"x":2}"#).is_err());
    }

    #[test]
    fn rejects_duplicate_nested_property_names() {
        assert!(parse_strict_json(r#"{"outer":{"x":1,"x":2}}"#).is_err());
    }

    #[test]
    fn accepts_unicode_and_null_without_normalizing_strings() {
        let parsed = parse_strict_json(r#"{"food":"Maïs 🥬","missing":null}"#)
            .expect("valid strict JSON parses");
        assert_eq!(parsed["food"], "Maïs 🥬");
        assert!(parsed["missing"].is_null());
    }

    #[test]
    fn verifies_receipt_emitted_by_assessor() {
        let (bundle, artifact_hash) = valid_bundle();
        assert!(verify_bundle(&bundle, artifact_hash).is_ok());
    }

    #[test]
    fn rejects_changed_outcome_when_decision_digest_is_stale() {
        let (mut bundle, artifact_hash) = valid_bundle();
        bundle.assessment.source_eligibility = Eligibility::Blocked;
        let error = verify_bundle(&bundle, artifact_hash).expect_err("tampered outcome must fail");
        assert!(error.to_string().contains("decision receipt digest mismatch"));
    }

    #[test]
    fn rejects_changed_fact_when_component_digest_is_stale() {
        let (mut bundle, artifact_hash) = valid_bundle();
        bundle.fact.value_scaled = Some(1251);
        let error = verify_bundle(&bundle, artifact_hash).expect_err("changed fact must fail");
        assert!(error.to_string().contains("food fact digest mismatch"));
    }

    #[test]
    fn rejects_missing_artifact_bytes_when_receipt_records_them() {
        let (bundle, _) = valid_bundle();
        let error = verify_bundle(&bundle, None).expect_err("artifact bytes must be provided");
        assert!(error.to_string().contains("supplied artifact bytes"));
    }
}
