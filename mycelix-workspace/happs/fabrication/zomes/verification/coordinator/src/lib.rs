// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verification Coordinator Zome
//!
//! Functions for submitting verifications, safety claims, and
//! bridging to the Knowledge hApp for epistemic scoring.

use hdk::prelude::*;
use verification_integrity::*;
use fabrication_common::*;

use std::cell::RefCell;
use std::collections::HashMap;

const EPISTEMIC_CACHE_TTL_MICROS: i64 = 300_000_000; // 5 min

thread_local! {
    static CONFIG: RefCell<Option<FabricationConfig>> = const { RefCell::new(None) };
    static EPISTEMIC_CACHE: RefCell<HashMap<String, (i64, ClaimEpistemic)>> = RefCell::new(HashMap::new());
}

fn get_config() -> FabricationConfig {
    CONFIG.with(|c| {
        c.borrow_mut()
            .get_or_insert_with(|| {
                dna_info()
                    .map(|info| FabricationConfig::from_properties_or_default(info.modifiers.properties.bytes()))
                    .unwrap_or_default()
            })
            .clone()
    })
}

// =============================================================================
// RATE LIMITING
// =============================================================================

fn rate_limit_anchor(agent: &AgentPubKey) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("rate_limit:{}", agent).into_bytes(),
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

fn enforce_rate_limit(caller: &AgentPubKey) -> ExternResult<()> {
    let cfg = get_config();
    let max_ops = cfg.rate_limit_max_ops as usize;
    let window_micros = cfg.rate_limit_window_secs as i64 * 1_000_000;

    let anchor = rate_limit_anchor(caller)?;
    let links = get_links(
        LinkQuery::try_new(anchor.clone(), LinkTypes::RateLimitBucket)?,
        GetStrategy::default(),
    )?;

    let now = sys_time()?;
    let window_start = now.as_micros() - window_micros;

    let recent_count = links
        .iter()
        .filter(|l| l.timestamp.as_micros() >= window_start)
        .count();

    if recent_count >= max_ops {
        return Err(FabricationError::RateLimited {
            max_ops: cfg.rate_limit_max_ops,
            window_secs: cfg.rate_limit_window_secs,
        }.to_wasm_error());
    }

    create_link(anchor.clone(), anchor, LinkTypes::RateLimitBucket, ())?;
    Ok(())
}

fn rate_limit_caller() -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    enforce_rate_limit(&agent)
}

/// Fetch epistemic classification from Knowledge hApp with cache.
/// Falls back to default values if the Knowledge hApp is unreachable.
fn fetch_epistemic(claim_text: &str, claim_type_key: &str) -> ClaimEpistemic {
    let now = sys_time().map(|t| t.as_micros()).unwrap_or(0);

    // Check cache by claim_type_key (not full text)
    let cached = EPISTEMIC_CACHE.with(|c| {
        c.borrow().get(claim_type_key).and_then(|(ts, ep)| {
            if now - ts < EPISTEMIC_CACHE_TTL_MICROS { Some(ep.clone()) } else { None }
        })
    });
    if let Some(ep) = cached { return ep; }

    let ep = match call(
        CallTargetCell::OtherRole("mycelix-knowledge".into()),
        ZomeName::from("epistemic"),
        FunctionName::from("classify_claim"),
        None,
        claim_text,
    ) {
        Ok(ZomeCallResponse::Ok(bytes)) => {
            bytes.decode().unwrap_or_else(|_| default_epistemic())
        }
        _ => default_epistemic(),
    };
    EPISTEMIC_CACHE.with(|c| {
        c.borrow_mut().insert(claim_type_key.to_string(), (now, ep.clone()));
    });
    ep
}

fn default_epistemic() -> ClaimEpistemic {
    ClaimEpistemic { empirical: 0.5, normative: 0.3, mythic: 0.2 }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitVerificationInput {
    pub design_hash: ActionHash,
    pub verification_type: VerificationType,
    pub result: VerificationResult,
    pub evidence: Vec<ActionHash>,
    pub credentials: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitClaimInput {
    pub design_hash: ActionHash,
    pub claim_type: SafetyClaimType,
    pub claim_text: String,
    pub supporting_evidence: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerificationSummary {
    pub design_hash: ActionHash,
    pub total_verifications: u32,
    pub passed: u32,
    pub failed: u32,
    pub claims_count: u32,
    pub average_confidence: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct EpistemicScore {
    pub empirical: f32,
    pub normative: f32,
    pub mythic: f32,
    pub overall_confidence: f32,
}

#[hdk_extern]
pub fn submit_verification(input: SubmitVerificationInput) -> ExternResult<Record> {
    rate_limit_caller()?;
    let verifier = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let verification = DesignVerification {
        design_hash: input.design_hash.clone(),
        verification_type: input.verification_type,
        result: input.result,
        evidence: input.evidence,
        verifier: verifier.clone(),
        verifier_credentials: input.credentials,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::DesignVerification(verification))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Verification,
        event_type: FabricationEventType::VerificationSubmitted,
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    create_link(input.design_hash, hash.clone(), LinkTypes::DesignToVerifications, ())?;
    create_link(verifier, hash.clone(), LinkTypes::VerifierToVerifications, ())?;

    get(hash.clone(), GetOptions::default())?.ok_or(FabricationError::not_found("Verification", &hash))
}

/// Internal: get all verifications for a design (used by summary/score functions)
fn get_design_verifications_all(design_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(LinkQuery::try_new(design_hash, LinkTypes::DesignToVerifications)?, GetStrategy::default())?;
    let mut results = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                results.push(record);
            }
        }
    }
    Ok(results)
}

#[hdk_extern]
pub fn get_design_verifications(input: HashPaginationInput) -> ExternResult<PaginatedResponse<Record>> {
    let items = get_design_verifications_all(input.hash)?;
    Ok(paginate(items, input.pagination.as_ref()))
}

#[hdk_extern]
pub fn get_verification_summary(design_hash: ActionHash) -> ExternResult<VerificationSummary> {
    let verifications = get_design_verifications_all(design_hash.clone())?;
    let claims = get_design_claims_all(design_hash.clone())?;

    let mut passed = 0u32;
    let mut failed = 0u32;
    let mut confidence_sum = 0.0f32;

    for record in &verifications {
        if let Some(v) = record.entry().to_app_option::<DesignVerification>().ok().flatten() {
            match v.result {
                VerificationResult::Passed { confidence, .. } => {
                    passed += 1;
                    confidence_sum += confidence;
                }
                VerificationResult::Failed { .. } => failed += 1,
                VerificationResult::ConditionalPass { confidence, .. } => {
                    passed += 1;
                    confidence_sum += confidence * 0.8;
                }
                _ => {}
            }
        }
    }

    let total = verifications.len() as u32;
    let avg_confidence = if passed > 0 { confidence_sum / passed as f32 } else { 0.0 };

    Ok(VerificationSummary {
        design_hash,
        total_verifications: total,
        passed,
        failed,
        claims_count: claims.len() as u32,
        average_confidence: avg_confidence,
    })
}

#[hdk_extern]
pub fn submit_safety_claim(input: SubmitClaimInput) -> ExternResult<Record> {
    rate_limit_caller()?;
    let author = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Fetch epistemic scores from Knowledge hApp (falls back to defaults)
    let claim_type_key = format!("{:?}", input.claim_type);
    let epistemic = fetch_epistemic(&input.claim_text, &claim_type_key);

    let claim = SafetyClaim {
        design_hash: input.design_hash.clone(),
        claim_type: input.claim_type,
        claim_text: input.claim_text,
        epistemic,
        supporting_evidence: input.supporting_evidence,
        knowledge_claim_hash: None,
        author,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::SafetyClaim(claim))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Verification,
        event_type: FabricationEventType::ClaimSubmitted,
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    create_link(input.design_hash, hash.clone(), LinkTypes::DesignToClaims, ())?;

    get(hash.clone(), GetOptions::default())?.ok_or(FabricationError::not_found("SafetyClaim", &hash))
}

/// Internal: get all claims for a design (used by summary/score functions)
fn get_design_claims_all(design_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(LinkQuery::try_new(design_hash, LinkTypes::DesignToClaims)?, GetStrategy::default())?;
    let mut results = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                results.push(record);
            }
        }
    }
    Ok(results)
}

#[hdk_extern]
pub fn get_design_claims(input: HashPaginationInput) -> ExternResult<PaginatedResponse<Record>> {
    let items = get_design_claims_all(input.hash)?;
    Ok(paginate(items, input.pagination.as_ref()))
}

#[hdk_extern]
pub fn get_epistemic_score(design_hash: ActionHash) -> ExternResult<EpistemicScore> {
    let claims = get_design_claims_all(design_hash)?;

    let mut e_sum = 0.0f32;
    let mut n_sum = 0.0f32;
    let mut m_sum = 0.0f32;
    let mut count = 0;

    for record in claims {
        if let Some(claim) = record.entry().to_app_option::<SafetyClaim>().ok().flatten() {
            e_sum += claim.epistemic.empirical;
            n_sum += claim.epistemic.normative;
            m_sum += claim.epistemic.mythic;
            count += 1;
        }
    }

    let count_f = count.max(1) as f32;
    Ok(EpistemicScore {
        empirical: e_sum / count_f,
        normative: n_sum / count_f,
        mythic: m_sum / count_f,
        overall_confidence: (e_sum + n_sum) / (2.0 * count_f),
    })
}

/// Return the default epistemic values applied when no Knowledge hApp score is available.
pub fn default_epistemic_values() -> (f32, f32, f32) {
    let ep = default_epistemic();
    (ep.empirical, ep.normative, ep.mythic)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn test_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn test_agent_key() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    // ── 1. SubmitVerificationInput serde roundtrip ────────────────────────────

    #[test]
    fn test_submit_verification_input_serde() {
        let input = SubmitVerificationInput {
            design_hash: test_action_hash(),
            verification_type: VerificationType::StructuralAnalysis,
            result: VerificationResult::Passed {
                confidence: 0.95,
                notes: "All finite-element checks passed".to_string(),
            },
            evidence: vec![ActionHash::from_raw_36(vec![1u8; 36])],
            credentials: vec!["PE License #12345".to_string()],
        };

        let json = serde_json::to_string(&input)
            .expect("SubmitVerificationInput should serialize to JSON");
        let restored: SubmitVerificationInput = serde_json::from_str(&json)
            .expect("SubmitVerificationInput should deserialize from JSON");

        // Verify structural fields survive the roundtrip.
        assert_eq!(restored.design_hash, input.design_hash);
        assert_eq!(restored.verification_type, input.verification_type);
        assert_eq!(restored.credentials, input.credentials);
        assert_eq!(restored.evidence.len(), 1);

        // Verify the result variant and its inner values.
        match restored.result {
            VerificationResult::Passed { confidence, ref notes } => {
                assert!((confidence - 0.95).abs() < f32::EPSILON);
                assert_eq!(notes, "All finite-element checks passed");
            }
            other => panic!("Expected Passed variant, got {:?}", other),
        }
    }

    // ── 2. SubmitClaimInput serde roundtrip ───────────────────────────────────

    #[test]
    fn test_submit_safety_claim_input_serde() {
        let input = SubmitClaimInput {
            design_hash: test_action_hash(),
            claim_type: SafetyClaimType::LoadCapacity("Supports 50kg static load".to_string()),
            claim_text: "Bracket rated for 50 kg at SWL 3:1 safety factor".to_string(),
            supporting_evidence: vec!["FEA report v2.1".to_string(), "Physical test #7".to_string()],
        };

        let json = serde_json::to_string(&input)
            .expect("SubmitClaimInput should serialize to JSON");
        let restored: SubmitClaimInput = serde_json::from_str(&json)
            .expect("SubmitClaimInput should deserialize from JSON");

        assert_eq!(restored.design_hash, input.design_hash);
        assert_eq!(restored.claim_text, input.claim_text);
        assert_eq!(restored.supporting_evidence, input.supporting_evidence);
        assert_eq!(
            restored.claim_type,
            SafetyClaimType::LoadCapacity("Supports 50kg static load".to_string())
        );
    }

    // ── 3. VerificationType — all variants roundtrip ──────────────────────────

    #[test]
    fn test_verification_type_all_variants_serde() {
        let variants = vec![
            VerificationType::StructuralAnalysis,
            VerificationType::MaterialCompatibility,
            VerificationType::PrintabilityTest,
            VerificationType::SafetyReview,
            VerificationType::FoodSafeCertification,
            VerificationType::MedicalCertification,
            VerificationType::CommunityReview,
        ];

        for variant in &variants {
            let json = serde_json::to_string(variant)
                .unwrap_or_else(|e| panic!("Failed to serialize {:?}: {}", variant, e));
            let restored: VerificationType = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("Failed to deserialize {:?}: {}", json, e));
            assert_eq!(
                &restored, variant,
                "VerificationType::{:?} did not survive serde roundtrip",
                variant
            );
        }
    }

    // ── 4. SafetyClaimType — all variants roundtrip ───────────────────────────

    #[test]
    fn test_safety_claim_type_variants_serde() {
        let variants = vec![
            SafetyClaimType::LoadCapacity("Supports 80kg".to_string()),
            SafetyClaimType::MaterialSafety("Food-safe in PETG".to_string()),
            SafetyClaimType::DimensionalAccuracy("Fits M8 bolt".to_string()),
            SafetyClaimType::TemperatureRange("Safe to 80°C".to_string()),
            SafetyClaimType::ChemicalResistance("Resistant to IPA".to_string()),
            SafetyClaimType::Custom("Outdoor UV rating 10yr".to_string()),
        ];

        for variant in &variants {
            let json = serde_json::to_string(variant)
                .unwrap_or_else(|e| panic!("Failed to serialize {:?}: {}", variant, e));
            let restored: SafetyClaimType = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("Failed to deserialize {:?}: {}", json, e));
            assert_eq!(
                &restored, variant,
                "SafetyClaimType::{:?} did not survive serde roundtrip",
                variant
            );
        }
    }

    // ── 5. default_epistemic_values matches hard-coded submit_safety_claim defaults ──

    #[test]
    fn test_claim_epistemic_defaults() {
        let (empirical, normative, mythic) = default_epistemic_values();

        // Values must match the ClaimEpistemic hard-coded in submit_safety_claim.
        assert!(
            (empirical - 0.5).abs() < f32::EPSILON,
            "Default empirical should be 0.5, got {}",
            empirical
        );
        assert!(
            (normative - 0.3).abs() < f32::EPSILON,
            "Default normative should be 0.3, got {}",
            normative
        );
        assert!(
            (mythic - 0.2).abs() < f32::EPSILON,
            "Default mythic should be 0.2, got {}",
            mythic
        );

        // They must also be valid for ClaimEpistemic (all in 0.0..=1.0).
        assert!(empirical >= 0.0 && empirical <= 1.0);
        assert!(normative >= 0.0 && normative <= 1.0);
        assert!(mythic >= 0.0 && mythic <= 1.0);

        // Serde roundtrip of ClaimEpistemic using the defaults.
        let epistemic = ClaimEpistemic { empirical, normative, mythic };
        let json = serde_json::to_string(&epistemic)
            .expect("ClaimEpistemic should serialize");
        let restored: ClaimEpistemic = serde_json::from_str(&json)
            .expect("ClaimEpistemic should deserialize");
        assert!((restored.empirical - empirical).abs() < f32::EPSILON);
        assert!((restored.normative - normative).abs() < f32::EPSILON);
        assert!((restored.mythic - mythic).abs() < f32::EPSILON);

        // Unused variable suppression — agent key and action hash constructors
        // are available in coordinator tests via hdk::prelude::*.
        let _ = test_action_hash();
        let _ = test_agent_key();
    }

    #[test]
    fn test_default_epistemic_function() {
        let ep = default_epistemic();
        assert!((ep.empirical - 0.5).abs() < f32::EPSILON);
        assert!((ep.normative - 0.3).abs() < f32::EPSILON);
        assert!((ep.mythic - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_epistemic_cache_ttl_constant() {
        assert_eq!(EPISTEMIC_CACHE_TTL_MICROS, 300_000_000);
        // 5 minutes in micros
        assert_eq!(EPISTEMIC_CACHE_TTL_MICROS / 1_000_000, 300);
    }
}
