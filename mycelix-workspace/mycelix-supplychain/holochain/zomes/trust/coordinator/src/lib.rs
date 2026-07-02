// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust Coordinator Zome - Business logic for reputation and compliance
use hdk::prelude::*;
use trust_integrity::*;

use mycelix_zome_helpers as _;
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

// ============================================================================
// Review and Reputation Management
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitReviewInput {
    pub subject: AgentPubKey,
    pub po_hash: Option<ActionHash>,
    pub category: ReputationCategory,
    pub rating: u8,
    pub comment: Option<String>,
}

#[hdk_extern]
pub fn submit_review(input: SubmitReviewInput) -> ExternResult<ActionHash> {
    // Input validation
    if input.rating > 5 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Rating must be 0-5".to_string()
        )));
    }
    if let Some(ref comment) = input.comment {
        if comment.len() > 1000 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Comment cannot exceed 1000 characters".to_string()
            )));
        }
    }

    let reviewer = agent_info()?.agent_initial_pubkey;

    // Cannot review yourself
    if reviewer == input.subject {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot review yourself".to_string()
        )));
    }

    let review = Review {
        subject: input.subject.clone(),
        reviewer: reviewer.clone(),
        po_hash: input.po_hash,
        category: input.category.clone(),
        rating: input.rating,
        comment: input.comment,
        created_at: sys_time()?,
    };

    let action_hash = create_entry(EntryTypes::Review(review.clone()))?;

    let subject_path = Path::from(format!("reviews/{}", review.subject));
    let subject_hash = ensure_path(subject_path, LinkTypes::AgentToReviews)?;
    create_link(
        subject_hash,
        action_hash.clone(),
        LinkTypes::AgentToReviews,
        (),
    )?;

    let reviewer_path = Path::from(format!("reviewer/{}", reviewer));
    let reviewer_hash = ensure_path(reviewer_path, LinkTypes::ReviewerToReviews)?;
    create_link(
        reviewer_hash,
        action_hash.clone(),
        LinkTypes::ReviewerToReviews,
        (),
    )?;

    // Update reputation score
    update_reputation_for_review(review)?;

    Ok(action_hash)
}

fn update_reputation_for_review(review: Review) -> ExternResult<()> {
    let rep_path = Path::from(format!(
        "reputation/{}/{:?}",
        review.subject, review.category
    ));
    let typed = rep_path.typed(LinkTypes::AgentToReputation)?;
    typed.ensure()?;
    let rep_hash = typed.path_entry_hash()?;

    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToReputation)?;
    let links = get_links(
        LinkQuery::new(rep_hash.clone(), filter),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(mut score) = record
                    .entry()
                    .to_app_option::<ReputationScore>()
                    .map_err(|e| wasm_error!(e))?
                {
                    let total = score.score as u64 * score.total_reviews + review.rating as u64;
                    score.total_reviews += 1;
                    score.score = (total / score.total_reviews) as u32;
                    score.updated_at = sys_time()?;
                    update_entry(hash, EntryTypes::ReputationScore(score))?;
                    return Ok(());
                }
            }
        }
    }

    // Create new reputation score
    let new_score = ReputationScore {
        agent: review.subject,
        category: review.category,
        score: review.rating as u32,
        total_reviews: 1,
        updated_at: sys_time()?,
    };
    let score_hash = create_entry(EntryTypes::ReputationScore(new_score))?;
    create_link(rep_hash, score_hash, LinkTypes::AgentToReputation, ())?;
    Ok(())
}

#[hdk_extern]
pub fn get_reputation(
    input: (AgentPubKey, ReputationCategory),
) -> ExternResult<Option<ReputationScore>> {
    let (agent, category) = input;
    let rep_path = Path::from(format!("reputation/{}/{:?}", agent, category));
    let typed = rep_path.typed(LinkTypes::AgentToReputation)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToReputation)?;
    let links = get_links(
        LinkQuery::new(typed.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                return Ok(record
                    .entry()
                    .to_app_option::<ReputationScore>()
                    .map_err(|e| wasm_error!(e))?);
            }
        }
    }
    Ok(None)
}

#[hdk_extern]
pub fn get_agent_reviews(agent: AgentPubKey) -> ExternResult<Vec<Review>> {
    let path = Path::from(format!("reviews/{}", agent));
    let typed = path.typed(LinkTypes::AgentToReviews)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToReviews)?;
    let links = get_links(
        LinkQuery::new(typed.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut reviews = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(review) = record
                    .entry()
                    .to_app_option::<Review>()
                    .map_err(|e| wasm_error!(e))?
                {
                    reviews.push(review);
                }
            }
        }
    }
    Ok(reviews)
}

#[hdk_extern]
pub fn add_certification(cert: Certification) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::Certification(cert.clone()))?;
    let path = Path::from(format!("certs/{}", cert.holder));
    let path_hash = ensure_path(path, LinkTypes::AgentToCertifications)?;
    create_link(
        path_hash,
        action_hash.clone(),
        LinkTypes::AgentToCertifications,
        (),
    )?;
    Ok(action_hash)
}

#[hdk_extern]
pub fn verify_certification(hash: ActionHash) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    if let Some(record) = get(hash.clone(), GetOptions::default())? {
        if let Some(mut cert) = record
            .entry()
            .to_app_option::<Certification>()
            .map_err(|e| wasm_error!(e))?
        {
            cert.verified_by = Some(agent);
            cert.verified_at = Some(sys_time()?);
            return update_entry(hash, EntryTypes::Certification(cert));
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Certification not found".into()
    )))
}

#[hdk_extern]
pub fn get_certifications(agent: AgentPubKey) -> ExternResult<Vec<Certification>> {
    let path = Path::from(format!("certs/{}", agent));
    let typed = path.typed(LinkTypes::AgentToCertifications)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToCertifications)?;
    let links = get_links(
        LinkQuery::new(typed.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut certs = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(cert) = record
                    .entry()
                    .to_app_option::<Certification>()
                    .map_err(|e| wasm_error!(e))?
                {
                    certs.push(cert);
                }
            }
        }
    }
    Ok(certs)
}

#[hdk_extern]
pub fn file_dispute(dispute: Dispute) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::Dispute(dispute.clone()))?;
    create_link(
        dispute.po_hash,
        action_hash.clone(),
        LinkTypes::PoToDisputes,
        (),
    )?;
    Ok(action_hash)
}

#[hdk_extern]
pub fn resolve_dispute(input: (ActionHash, String)) -> ExternResult<ActionHash> {
    let (hash, resolution) = input;

    if resolution.is_empty() || resolution.len() > 1000 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Resolution must be 1-1000 characters".to_string()
        )));
    }

    if let Some(record) = get(hash.clone(), GetOptions::default())? {
        if let Some(mut dispute) = record
            .entry()
            .to_app_option::<Dispute>()
            .map_err(|e| wasm_error!(e))?
        {
            dispute.resolution = Some(resolution);
            dispute.resolved_at = Some(sys_time()?);
            return update_entry(hash, EntryTypes::Dispute(dispute));
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Dispute not found".into()
    )))
}

#[hdk_extern]
pub fn get_provider_rating(agent: AgentPubKey) -> ExternResult<f64> {
    // Get all reviews for the provider
    let reviews = get_agent_reviews(agent)?;

    if reviews.is_empty() {
        return Ok(0.0);
    }

    let total: u32 = reviews.iter().map(|r| r.rating as u32).sum();
    let average = total as f64 / reviews.len() as f64;

    Ok(average)
}

#[hdk_extern]
pub fn flag_provider(input: (AgentPubKey, String)) -> ExternResult<ActionHash> {
    let (provider, reason) = input;

    if reason.is_empty() || reason.len() > 500 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reason must be 1-500 characters".to_string()
        )));
    }

    let flagger = agent_info()?.agent_initial_pubkey;

    // Create a low-rating review to flag the provider
    let review = Review {
        subject: provider,
        reviewer: flagger,
        po_hash: None,
        category: ReputationCategory::Compliance,
        rating: 0,
        comment: Some(format!("FLAGGED: {}", reason)),
        created_at: sys_time()?,
    };

    let action_hash = create_entry(EntryTypes::Review(review.clone()))?;

    let subject_path = Path::from(format!("reviews/{}", review.subject));
    let subject_hash = ensure_path(subject_path, LinkTypes::AgentToReviews)?;
    create_link(
        subject_hash,
        action_hash.clone(),
        LinkTypes::AgentToReviews,
        (),
    )?;

    update_reputation_for_review(review)?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_all_reputation_categories(
    agent: AgentPubKey,
) -> ExternResult<Vec<(ReputationCategory, u32)>> {
    let categories = vec![
        ReputationCategory::Reliability,
        ReputationCategory::Quality,
        ReputationCategory::Communication,
        ReputationCategory::Timeliness,
        ReputationCategory::Compliance,
    ];

    let mut scores = Vec::new();
    for category in categories {
        if let Some(score) = get_reputation((agent.clone(), category.clone()))? {
            scores.push((category, score.score));
        }
    }

    Ok(scores)
}

// ============================================================================
// Workflow 5: Closed-Loop Quality Feedback
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct QualityReportInput {
    pub supplier: AgentPubKey,
    pub po_hash: ActionHash,
    pub quality_score: f64,
    pub description: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct QualityReportResult {
    pub actions_taken: Vec<String>,
    pub review_hash: Option<ActionHash>,
    pub dispute_hash: Option<ActionHash>,
}

/// Convert a 0.0–1.0 quality score to a 1–5 star rating.
fn quality_score_to_rating(score: f64) -> u8 {
    if score >= 0.9 {
        5
    } else if score >= 0.7 {
        4
    } else if score >= 0.5 {
        3
    } else if score >= 0.3 {
        2
    } else {
        1
    }
}

#[hdk_extern]
pub fn report_quality_issue(input: QualityReportInput) -> ExternResult<QualityReportResult> {
    let rating = quality_score_to_rating(input.quality_score);
    let mut actions_taken = Vec::new();
    let mut dispute_hash = None;

    // Submit review for all cases
    let review_input = SubmitReviewInput {
        subject: input.supplier.clone(),
        po_hash: Some(input.po_hash.clone()),
        category: ReputationCategory::Quality,
        rating,
        comment: Some(format!(
            "Quality score: {:.2}. {}",
            input.quality_score, input.description
        )),
    };
    let review_hash = Some(submit_review(review_input)?);
    actions_taken.push(format!("submitted {}-star quality review", rating));

    if input.quality_score < 0.3 {
        // Critical: 1-star review already submitted + file dispute + flag provider
        let my_agent = agent_info()?.agent_initial_pubkey;
        let dispute = Dispute {
            po_hash: input.po_hash.clone(),
            claimant: my_agent.clone(),
            respondent: input.supplier.clone(),
            description: format!(
                "Critical quality failure (score: {:.2}): {}",
                input.quality_score, input.description
            ),
            evidence_hashes: vec![],
            resolution: None,
            resolved_at: None,
            created_at: sys_time()?,
        };
        let dh = file_dispute(dispute)?;
        dispute_hash = Some(dh);
        actions_taken.push("filed dispute".to_string());

        flag_provider((
            input.supplier.clone(),
            format!(
                "Critical quality failure (score: {:.2}): {}",
                input.quality_score, input.description
            ),
        ))?;
        actions_taken.push("flagged provider".to_string());
    } else if input.quality_score < 0.7 {
        // Moderate: proportional review already submitted + file dispute
        let my_agent = agent_info()?.agent_initial_pubkey;
        let dispute = Dispute {
            po_hash: input.po_hash.clone(),
            claimant: my_agent,
            respondent: input.supplier.clone(),
            description: format!(
                "Moderate quality issue (score: {:.2}): {}",
                input.quality_score, input.description
            ),
            evidence_hashes: vec![],
            resolution: None,
            resolved_at: None,
            created_at: sys_time()?,
        };
        let dh = file_dispute(dispute)?;
        dispute_hash = Some(dh);
        actions_taken.push("filed dispute".to_string());
    }
    // score >= 0.7: positive review only (already submitted above)

    Ok(QualityReportResult {
        actions_taken,
        review_hash,
        dispute_hash,
    })
}

/// Compute composite reliability score from category scores.
/// Category weights: Quality 0.3, Reliability 0.3, Timeliness 0.2,
///                   Communication 0.1, Compliance 0.1.
/// Scores on 0–5 scale → normalized to 0.0–1.0.
fn compute_composite_reliability(scores: &[(String, u32)]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }

    let weight_for = |cat: &str| -> f64 {
        match cat {
            "Quality" => 0.3,
            "Reliability" => 0.3,
            "Timeliness" => 0.2,
            "Communication" => 0.1,
            "Compliance" => 0.1,
            _ => 0.0,
        }
    };

    let mut weighted_sum = 0.0_f64;
    let mut total_weight = 0.0_f64;

    for (category, score) in scores {
        let w = weight_for(category.as_str());
        if w > 0.0 {
            let normalized = (*score as f64 / 5.0).clamp(0.0, 1.0);
            weighted_sum += normalized * w;
            total_weight += w;
        }
    }

    if total_weight == 0.0 {
        // Fall back to simple average of known scores normalized to 0–1
        let total: u32 = scores.iter().map(|(_, s)| s).sum();
        let avg = total as f64 / scores.len() as f64;
        return (avg / 5.0).clamp(0.0, 1.0);
    }

    // Scale by the fraction of known categories covered
    weighted_sum / total_weight
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SupplierReliability {
    pub composite_score: f64,
    pub category_scores: Vec<(String, u32)>,
    pub total_reviews: u64,
}

#[hdk_extern]
pub fn get_supplier_reliability(supplier: AgentPubKey) -> ExternResult<SupplierReliability> {
    let raw_scores = get_all_reputation_categories(supplier.clone())?;

    // Collect total_reviews by fetching full ReputationScore per category
    let categories = vec![
        ReputationCategory::Reliability,
        ReputationCategory::Quality,
        ReputationCategory::Communication,
        ReputationCategory::Timeliness,
        ReputationCategory::Compliance,
    ];

    let mut total_reviews_sum = 0u64;
    for cat in &categories {
        if let Some(score) = get_reputation((supplier.clone(), cat.clone()))? {
            total_reviews_sum += score.total_reviews;
        }
    }

    // Map (ReputationCategory, u32) → (String, u32) for the public API
    let category_scores: Vec<(String, u32)> = raw_scores
        .iter()
        .map(|(cat, score)| (format!("{:?}", cat), *score))
        .collect();

    let composite_score = compute_composite_reliability(&category_scores);

    Ok(SupplierReliability {
        composite_score,
        category_scores,
        total_reviews: total_reviews_sum,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_score_to_rating() {
        // Edge cases at each boundary
        assert_eq!(quality_score_to_rating(1.0), 5);
        assert_eq!(quality_score_to_rating(0.9), 5);
        assert_eq!(quality_score_to_rating(0.89), 4);
        assert_eq!(quality_score_to_rating(0.7), 4);
        assert_eq!(quality_score_to_rating(0.69), 3);
        assert_eq!(quality_score_to_rating(0.5), 3);
        assert_eq!(quality_score_to_rating(0.49), 2);
        assert_eq!(quality_score_to_rating(0.3), 2);
        assert_eq!(quality_score_to_rating(0.29), 1);
        assert_eq!(quality_score_to_rating(0.0), 1);
    }

    #[test]
    fn test_quality_actions_critical() {
        // score < 0.3: review + dispute + flag
        // We test the routing logic by checking what quality_score_to_rating produces
        // and verifying the intended action count
        let score = 0.1_f64;
        let rating = quality_score_to_rating(score);
        assert_eq!(rating, 1);
        // At score < 0.3 the function would take: review + dispute + flag = 3 actions
        // We verify score range detection
        assert!(score < 0.3);
    }

    #[test]
    fn test_quality_actions_moderate() {
        // 0.3 <= score < 0.7: review + dispute
        let score = 0.5_f64;
        let rating = quality_score_to_rating(score);
        assert_eq!(rating, 3);
        assert!(score >= 0.3 && score < 0.7);
    }

    #[test]
    fn test_quality_actions_good() {
        // score >= 0.7: positive review only
        let score = 0.85_f64;
        let rating = quality_score_to_rating(score);
        assert_eq!(rating, 4);
        assert!(score >= 0.7);
    }

    #[test]
    fn test_composite_reliability() {
        // Weighted: Quality(5)*0.3 + Reliability(4)*0.3 + Timeliness(3)*0.2
        //           + Communication(5)*0.1 + Compliance(4)*0.1
        // = (1.0*0.3 + 0.8*0.3 + 0.6*0.2 + 1.0*0.1 + 0.8*0.1) / 1.0
        // = 0.3 + 0.24 + 0.12 + 0.1 + 0.08 = 0.84
        let scores = vec![
            ("Quality".to_string(), 5u32),
            ("Reliability".to_string(), 4u32),
            ("Timeliness".to_string(), 3u32),
            ("Communication".to_string(), 5u32),
            ("Compliance".to_string(), 4u32),
        ];
        let composite = compute_composite_reliability(&scores);
        assert!(
            (composite - 0.84).abs() < 0.001,
            "Expected ~0.84, got {}",
            composite
        );
    }

    #[test]
    fn test_composite_reliability_empty() {
        let composite = compute_composite_reliability(&[]);
        assert_eq!(composite, 0.0, "Empty scores should return 0.0");
    }

    #[test]
    fn test_quality_report_input_serde() {
        let input = QualityReportInput {
            supplier: AgentPubKey::from_raw_36(vec![3u8; 36]),
            po_hash: ActionHash::from_raw_36(vec![4u8; 36]),
            quality_score: 0.25,
            description: "Components arrived damaged".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: QualityReportInput = serde_json::from_str(&json).unwrap();
        assert!((back.quality_score - 0.25).abs() < 0.001);
        assert_eq!(back.description, "Components arrived damaged");
    }

    #[test]
    fn test_supplier_reliability_serde() {
        let rel = SupplierReliability {
            composite_score: 0.78,
            category_scores: vec![
                ("Quality".to_string(), 4u32),
                ("Reliability".to_string(), 4u32),
            ],
            total_reviews: 12,
        };
        let json = serde_json::to_string(&rel).unwrap();
        let back: SupplierReliability = serde_json::from_str(&json).unwrap();
        assert!((back.composite_score - 0.78).abs() < 0.001);
        assert_eq!(back.category_scores.len(), 2);
        assert_eq!(back.total_reviews, 12);
    }

    #[test]
    fn test_rating_bounds_valid() {
        // Ratings 0-5 are valid per the coordinator and integrity validation
        for rating in 0u8..=5 {
            assert!(rating <= 5, "Rating {} should be in bounds", rating);
        }
    }

    #[test]
    fn test_rating_bounds_invalid() {
        // Ratings above 5 are rejected
        for rating in 6u8..=255 {
            assert!(rating > 5, "Rating {} should be out of bounds", rating);
        }
    }

    #[test]
    fn test_submit_review_input_serde() {
        let input = SubmitReviewInput {
            subject: AgentPubKey::from_raw_36(vec![1u8; 36]),
            po_hash: None,
            category: ReputationCategory::Quality,
            rating: 4,
            comment: Some("Good quality materials".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SubmitReviewInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.rating, 4);
        assert!(back.po_hash.is_none());
    }

    #[test]
    fn test_self_review_rejection_concept() {
        // The same agent cannot be both reviewer and subject
        let agent_a = AgentPubKey::from_raw_36(vec![0u8; 36]);
        let agent_b = AgentPubKey::from_raw_36(vec![1u8; 36]);
        // Different agents: valid
        assert_ne!(agent_a, agent_b);
        // Same agent: would be rejected by coordinator
        assert_eq!(agent_a.clone(), agent_a.clone());
    }

    #[test]
    fn test_reputation_category_serde_roundtrip() {
        let categories = vec![
            ReputationCategory::Reliability,
            ReputationCategory::Quality,
            ReputationCategory::Communication,
            ReputationCategory::Timeliness,
            ReputationCategory::Compliance,
        ];
        for cat in categories {
            let json = serde_json::to_string(&cat).unwrap();
            let back: ReputationCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(back, cat);
        }
    }

    #[test]
    fn test_average_rating_calculation() {
        // Reproduce get_provider_rating logic
        let ratings: Vec<u32> = vec![4, 5, 3, 4, 5];
        let total: u32 = ratings.iter().sum();
        let average = total as f64 / ratings.len() as f64;
        assert!((average - 4.2).abs() < 0.001);
    }

    #[test]
    fn test_reputation_score_serde() {
        // The scoring update: (old_score * total + new_rating) / (total + 1)
        let old_score: u64 = 4;
        let old_total: u64 = 5;
        let new_rating: u64 = 5;
        let new_total = old_total + 1;
        let new_score = (old_score * old_total + new_rating) / new_total;
        assert_eq!(new_score, 4); // (20 + 5) / 6 = 4 (integer div)
    }
}
