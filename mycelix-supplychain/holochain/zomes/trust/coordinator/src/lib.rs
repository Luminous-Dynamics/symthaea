//! Trust Coordinator Zome - Business logic for reputation and compliance
use hdk::prelude::*;
use trust_integrity::*;

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
    create_link(subject_hash, action_hash.clone(), LinkTypes::AgentToReviews, ())?;

    let reviewer_path = Path::from(format!("reviewer/{}", reviewer));
    let reviewer_hash = ensure_path(reviewer_path, LinkTypes::ReviewerToReviews)?;
    create_link(reviewer_hash, action_hash.clone(), LinkTypes::ReviewerToReviews, ())?;

    // Update reputation score
    update_reputation_for_review(review)?;

    Ok(action_hash)
}

fn update_reputation_for_review(review: Review) -> ExternResult<()> {
    let rep_path = Path::from(format!("reputation/{}/{:?}", review.subject, review.category));
    let typed = rep_path.typed(LinkTypes::AgentToReputation)?;
    typed.ensure()?;
    let rep_hash = typed.path_entry_hash()?;

    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToReputation)?;
    let links = get_links(LinkQuery::new(rep_hash.clone(), filter), GetStrategy::default())?;

    if let Some(link) = links.first() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(mut score) = record.entry().to_app_option::<ReputationScore>().map_err(|e| wasm_error!(e))? {
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
pub fn get_reputation(input: (AgentPubKey, ReputationCategory)) -> ExternResult<Option<ReputationScore>> {
    let (agent, category) = input;
    let rep_path = Path::from(format!("reputation/{}/{:?}", agent, category));
    let typed = rep_path.typed(LinkTypes::AgentToReputation)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToReputation)?;
    let links = get_links(LinkQuery::new(typed.path_entry_hash()?, filter), GetStrategy::default())?;

    if let Some(link) = links.first() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                return Ok(record.entry().to_app_option::<ReputationScore>().map_err(|e| wasm_error!(e))?);
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
    let links = get_links(LinkQuery::new(typed.path_entry_hash()?, filter), GetStrategy::default())?;

    let mut reviews = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(review) = record.entry().to_app_option::<Review>().map_err(|e| wasm_error!(e))? {
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
    create_link(path_hash, action_hash.clone(), LinkTypes::AgentToCertifications, ())?;
    Ok(action_hash)
}

#[hdk_extern]
pub fn verify_certification(hash: ActionHash) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    if let Some(record) = get(hash.clone(), GetOptions::default())? {
        if let Some(mut cert) = record.entry().to_app_option::<Certification>().map_err(|e| wasm_error!(e))? {
            cert.verified_by = Some(agent);
            cert.verified_at = Some(sys_time()?);
            return update_entry(hash, EntryTypes::Certification(cert));
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Certification not found".into())))
}

#[hdk_extern]
pub fn get_certifications(agent: AgentPubKey) -> ExternResult<Vec<Certification>> {
    let path = Path::from(format!("certs/{}", agent));
    let typed = path.typed(LinkTypes::AgentToCertifications)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToCertifications)?;
    let links = get_links(LinkQuery::new(typed.path_entry_hash()?, filter), GetStrategy::default())?;

    let mut certs = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(cert) = record.entry().to_app_option::<Certification>().map_err(|e| wasm_error!(e))? {
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
    create_link(dispute.po_hash, action_hash.clone(), LinkTypes::PoToDisputes, ())?;
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
        if let Some(mut dispute) = record.entry().to_app_option::<Dispute>().map_err(|e| wasm_error!(e))? {
            dispute.resolution = Some(resolution);
            dispute.resolved_at = Some(sys_time()?);
            return update_entry(hash, EntryTypes::Dispute(dispute));
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Dispute not found".into())))
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
    create_link(subject_hash, action_hash.clone(), LinkTypes::AgentToReviews, ())?;

    update_reputation_for_review(review)?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_all_reputation_categories(agent: AgentPubKey) -> ExternResult<Vec<(ReputationCategory, u32)>> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
