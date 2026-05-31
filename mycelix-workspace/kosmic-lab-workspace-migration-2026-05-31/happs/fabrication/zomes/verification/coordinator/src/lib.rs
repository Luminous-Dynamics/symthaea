// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Verification Coordinator Zome
//!
//! Functions for submitting verifications, safety claims, and
//! bridging to the Knowledge hApp for epistemic scoring.

use fabrication_common::*;
use hdk::prelude::*;
use verification_integrity::*;

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
    create_link(
        input.design_hash,
        hash.clone(),
        LinkTypes::DesignToVerifications,
        (),
    )?;
    create_link(
        verifier,
        hash.clone(),
        LinkTypes::VerifierToVerifications,
        (),
    )?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[hdk_extern]
pub fn get_design_verifications(design_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(design_hash, LinkTypes::DesignToVerifications)?,
        GetStrategy::default(),
    )?;
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
pub fn get_verification_summary(design_hash: ActionHash) -> ExternResult<VerificationSummary> {
    let verifications = get_design_verifications(design_hash.clone())?;
    let claims = get_design_claims(design_hash.clone())?;

    let mut passed = 0u32;
    let mut failed = 0u32;
    let mut confidence_sum = 0.0f32;

    for record in &verifications {
        if let Some(v) = record
            .entry()
            .to_app_option::<DesignVerification>()
            .ok()
            .flatten()
        {
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
    let avg_confidence = if passed > 0 {
        confidence_sum / passed as f32
    } else {
        0.0
    };

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
    let author = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Default epistemic scores (would be calculated by Knowledge hApp)
    let epistemic = ClaimEpistemic {
        empirical: 0.5,
        normative: 0.3,
        mythic: 0.2,
    };

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
    create_link(
        input.design_hash,
        hash.clone(),
        LinkTypes::DesignToClaims,
        (),
    )?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[hdk_extern]
pub fn get_design_claims(design_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(design_hash, LinkTypes::DesignToClaims)?,
        GetStrategy::default(),
    )?;
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
pub fn get_epistemic_score(design_hash: ActionHash) -> ExternResult<EpistemicScore> {
    let claims = get_design_claims(design_hash)?;

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
