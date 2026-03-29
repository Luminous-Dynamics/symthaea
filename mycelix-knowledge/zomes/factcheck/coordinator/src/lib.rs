// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fact-Check Coordinator Zome
//!
//! Provides a simple fact-checking API for external hApps (Media, Governance, Justice)
//! to verify statements against the knowledge graph.
//!
//! ## Main Functions
//! - `fact_check`: Simple statement verification
//! - `query_claims_by_subject`: Find claims about a subject
//! - `query_claims_by_topic`: Find claims by topic
//! - `batch_fact_check`: Check multiple statements efficiently
//!
//! ## Verdicts
//! Results include a verdict (True, MostlyTrue, Mixed, etc.) along with
//! supporting/contradicting claims, confidence scores, and suggested actions.
//!
//! Updated to use HDK 0.6 patterns (LinkQuery, GetStrategy, Anchor pattern)

use hdk::prelude::*;
use factcheck_integrity::*;

/// Helper to get an anchor entry hash for link bases
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

// ============================================================================
// INPUT TYPES
// ============================================================================

/// Input for fact checking a statement
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FactCheckInput {
    /// Source hApp making the request
    pub source_happ: String,

    /// Statement to fact-check
    pub statement: String,

    /// Additional context
    pub context: Option<String>,

    /// Subject of the statement
    pub subject: Option<String>,

    /// Topics/tags
    pub topics: Option<Vec<String>>,

    /// Minimum empirical level (default: 0.0)
    pub min_e: Option<f64>,

    /// Minimum normative level (default: 0.0)
    pub min_n: Option<f64>,
}

/// Input for querying claims by subject
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubjectQueryInput {
    /// Subject to search for
    pub subject: String,

    /// Minimum empirical level
    pub min_e: f64,

    /// Minimum normative level
    pub min_n: f64,

    /// Maximum results
    pub limit: u32,
}

/// Input for querying claims by topic
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TopicQueryInput {
    /// Topics to search for
    pub topics: Vec<String>,

    /// Minimum empirical level
    pub min_e: f64,

    /// Minimum normative level
    pub min_n: f64,

    /// Maximum results
    pub limit: u32,
}

/// Detailed claim information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClaimDetails {
    /// The claim record
    pub claim: Record,

    /// Evidence records
    pub evidence: Vec<Record>,

    /// Relationship records
    pub relationships: Vec<Record>,

    /// Credibility assessment
    pub credibility: f64,

    /// Verification market status
    pub verification_status: Option<String>,
}

// ============================================================================
// MAIN FACT CHECK FUNCTION
// ============================================================================

/// Simple fact check - main API for external hApps
///
/// Takes a statement and returns a structured verdict with supporting evidence.
#[hdk_extern]
pub fn fact_check(input: FactCheckInput) -> ExternResult<FactCheckResult> {
    let agent = agent_info()?.agent_initial_pubkey;
    let start_time = sys_time()?;

    // Create the request entry
    let request = FactCheckRequest {
        id: format!("fcr_{}_{}", input.source_happ, start_time.as_micros()),
        source_happ: input.source_happ.clone(),
        statement: input.statement.clone(),
        context: input.context.clone(),
        subject: input.subject.clone(),
        topics: input.topics.clone().unwrap_or_default(),
        min_epistemic_e: input.min_e.unwrap_or(0.0),
        min_epistemic_n: input.min_n.unwrap_or(0.0),
        requester: agent,
        requested_at: start_time,
        status: FactCheckStatus::Processing,
    };

    // Store the request
    let request_hash = create_entry(EntryTypes::FactCheckRequest(request.clone()))?;

    // Index by source hApp using anchor pattern
    let happ_anchor = format!("happ:{}", input.source_happ);
    create_entry(&EntryTypes::Anchor(Anchor(happ_anchor.clone())))?;
    create_link(anchor_hash(&happ_anchor)?, request_hash.clone(), LinkTypes::HappToFactCheck, ())?;

    // Index by subject if provided
    if let Some(ref subject) = input.subject {
        let subject_anchor = format!("subject:{}", subject);
        create_entry(&EntryTypes::Anchor(Anchor(subject_anchor.clone())))?;
        create_link(anchor_hash(&subject_anchor)?, request_hash.clone(), LinkTypes::SubjectToFactCheck, ())?;
    }

    // Index by topics
    for topic in input.topics.clone().unwrap_or_default() {
        let topic_anchor = format!("topic:{}", topic);
        create_entry(&EntryTypes::Anchor(Anchor(topic_anchor.clone())))?;
        create_link(anchor_hash(&topic_anchor)?, request_hash.clone(), LinkTypes::TopicToFactCheck, ())?;
    }

    // Perform the fact check
    let (verdict, confidence, supporting, contradicting, related) =
        analyze_statement(&input.statement, &input.subject, &input.topics)?;

    let end_time = sys_time()?;
    let processing_time_ms = (end_time.as_micros() - start_time.as_micros()) / 1000;

    // Calculate aggregate epistemic position
    let aggregate_epistemic = calculate_aggregate_epistemic(&supporting, &contradicting);

    // Calculate overall credibility
    let credibility_score = calculate_credibility(&supporting, &contradicting);

    // Calculate source diversity
    let source_diversity = calculate_diversity(&supporting, &contradicting, &related);

    // Generate suggested actions
    let suggested_actions = generate_suggestions(&verdict, &supporting, &contradicting);

    // Create the result
    let result = FactCheckResult {
        id: format!("fcresult_{}_{}", input.source_happ, end_time.as_micros()),
        request_id: request.id,
        statement: input.statement,
        verdict,
        verdict_confidence: confidence,
        supporting_claims: supporting,
        contradicting_claims: contradicting,
        related_claims: related,
        aggregate_epistemic,
        credibility_score,
        source_diversity,
        evidence_count: 0, // Will be populated in full implementation
        processed_at: end_time,
        processing_time_ms: processing_time_ms as u64,
        suggested_actions,
    };

    // Store the result
    let result_hash = create_entry(EntryTypes::FactCheckResult(result.clone()))?;

    // Link request to result
    create_link(request_hash, result_hash, LinkTypes::RequestToResult, ())?;

    Ok(result)
}

/// Analyze a statement against the knowledge graph
fn analyze_statement(
    statement: &str,
    subject: &Option<String>,
    topics: &Option<Vec<String>>,
) -> ExternResult<(FactCheckVerdict, f64, Vec<ClaimSummary>, Vec<ClaimSummary>, Vec<ClaimSummary>)>
{
    let mut all_claims: Vec<ClaimSummary> = Vec::new();
    let statement_lower = statement.to_lowercase();

    // Query by subject if provided
    if let Some(subj) = subject {
        let subject_claims = query_claims_by_subject(SubjectQueryInput {
            subject: subj.clone(),
            min_e: 0.0,
            min_n: 0.0,
            limit: 50,
        })?;
        all_claims.extend(subject_claims);
    }

    // Query by topics if provided
    if let Some(topic_list) = topics {
        if !topic_list.is_empty() {
            let topic_claims = query_claims_by_topic(TopicQueryInput {
                topics: topic_list.clone(),
                min_e: 0.0,
                min_n: 0.0,
                limit: 50,
            })?;

            // Deduplicate
            for claim in topic_claims {
                if !all_claims.iter().any(|c| c.claim_id == claim.claim_id) {
                    all_claims.push(claim);
                }
            }
        }
    }

    // If no subject or topics provided, extract keywords from statement
    if all_claims.is_empty() && subject.is_none() && topics.is_none() {
        // Extract key words (simple approach: words longer than 4 chars)
        let keywords: Vec<String> = statement_lower
            .split_whitespace()
            .filter(|w| w.len() > 4)
            .map(|s| s.to_string())
            .take(5)
            .collect();

        if !keywords.is_empty() {
            let keyword_claims = query_claims_by_topic(TopicQueryInput {
                topics: keywords,
                min_e: 0.0,
                min_n: 0.0,
                limit: 50,
            })?;
            all_claims.extend(keyword_claims);
        }
    }

    // Classify claims into supporting, contradicting, and related
    let mut supporting: Vec<ClaimSummary> = Vec::new();
    let mut contradicting: Vec<ClaimSummary> = Vec::new();
    let mut related: Vec<ClaimSummary> = Vec::new();

    for mut claim in all_claims {
        let content_lower = claim.content_snippet.to_lowercase();

        // Simple classification based on content analysis
        let (relationship, _is_supporting) = classify_claim_relationship(
            &statement_lower,
            &content_lower,
        );

        claim.relationship = relationship.clone();

        match relationship {
            ClaimRelationship::DirectSupport | ClaimRelationship::IndirectSupport => {
                supporting.push(claim);
            }
            ClaimRelationship::DirectContradiction | ClaimRelationship::IndirectContradiction => {
                contradicting.push(claim);
            }
            _ => {
                related.push(claim);
            }
        }
    }

    // Calculate verdict based on evidence
    let (verdict, confidence) = calculate_verdict(&supporting, &contradicting, &related);

    Ok((verdict, confidence, supporting, contradicting, related))
}

/// Classify the relationship between a claim and the statement being checked
fn classify_claim_relationship(
    statement: &str,
    claim_content: &str,
) -> (ClaimRelationship, bool) {
    // Check for negation patterns
    let negation_words = ["not", "never", "false", "incorrect", "wrong", "untrue", "deny", "denies"];
    let statement_has_negation = negation_words.iter().any(|w| statement.contains(w));
    let claim_has_negation = negation_words.iter().any(|w| claim_content.contains(w));

    // Check for affirmation patterns
    let affirmation_words = ["true", "correct", "verified", "confirmed", "proven", "shows", "demonstrates"];
    let claim_has_affirmation = affirmation_words.iter().any(|w| claim_content.contains(w));

    // Extract key concepts (simple word overlap)
    let statement_words: Vec<&str> = statement.split_whitespace().collect();
    let claim_words: Vec<&str> = claim_content.split_whitespace().collect();
    let overlap: Vec<&str> = statement_words.iter()
        .filter(|w| w.len() > 3 && claim_words.contains(w))
        .copied()
        .collect();

    let overlap_ratio = if statement_words.is_empty() {
        0.0
    } else {
        overlap.len() as f64 / statement_words.len() as f64
    };

    // Determine relationship
    if overlap_ratio > 0.3 {
        // High overlap - check for contradiction or support
        if statement_has_negation != claim_has_negation {
            // One has negation, other doesn't - likely contradiction
            (ClaimRelationship::DirectContradiction, false)
        } else if claim_has_affirmation {
            // Claim affirms something with high overlap
            (ClaimRelationship::DirectSupport, true)
        } else if overlap_ratio > 0.5 {
            // Very high overlap without negation differences
            (ClaimRelationship::IndirectSupport, true)
        } else {
            (ClaimRelationship::Context, false)
        }
    } else if overlap_ratio > 0.1 {
        // Moderate overlap
        if statement_has_negation != claim_has_negation {
            (ClaimRelationship::IndirectContradiction, false)
        } else {
            (ClaimRelationship::Related, false)
        }
    } else {
        // Low overlap - just related by topic
        (ClaimRelationship::Related, false)
    }
}

/// Calculate verdict based on supporting and contradicting claims
fn calculate_verdict(
    supporting: &[ClaimSummary],
    contradicting: &[ClaimSummary],
    _related: &[ClaimSummary],
) -> (FactCheckVerdict, f64) {
    let total_evidence = supporting.len() + contradicting.len();

    if total_evidence == 0 {
        return (FactCheckVerdict::InsufficientEvidence, 0.3);
    }

    // Calculate weighted support/contradiction scores
    let support_score: f64 = supporting.iter()
        .map(|c| c.credibility * c.relevance_score * c.epistemic.empirical)
        .sum();
    let contradict_score: f64 = contradicting.iter()
        .map(|c| c.credibility * c.relevance_score * c.epistemic.empirical)
        .sum();

    let total_score = support_score + contradict_score;
    if total_score == 0.0 {
        return (FactCheckVerdict::InsufficientEvidence, 0.4);
    }

    let support_ratio = support_score / total_score;
    let confidence = calculate_evidence_confidence(supporting, contradicting);

    // Determine verdict based on support ratio and confidence
    let verdict = if contradicting.is_empty() && !supporting.is_empty() {
        if support_ratio > 0.9 && confidence > 0.7 {
            FactCheckVerdict::True
        } else {
            FactCheckVerdict::MostlyTrue
        }
    } else if supporting.is_empty() && !contradicting.is_empty() {
        if support_ratio < 0.1 && confidence > 0.7 {
            FactCheckVerdict::False
        } else {
            FactCheckVerdict::MostlyFalse
        }
    } else if support_ratio > 0.75 {
        FactCheckVerdict::MostlyTrue
    } else if support_ratio < 0.25 {
        FactCheckVerdict::MostlyFalse
    } else {
        FactCheckVerdict::Mixed
    };

    (verdict, confidence)
}

/// Calculate confidence in the evidence
fn calculate_evidence_confidence(
    supporting: &[ClaimSummary],
    contradicting: &[ClaimSummary],
) -> f64 {
    let all_claims: Vec<&ClaimSummary> = supporting.iter()
        .chain(contradicting.iter())
        .collect();

    if all_claims.is_empty() {
        return 0.3;
    }

    // Average credibility weighted by relevance
    let total_weight: f64 = all_claims.iter().map(|c| c.relevance_score).sum();
    if total_weight == 0.0 {
        return 0.4;
    }

    let weighted_credibility: f64 = all_claims.iter()
        .map(|c| c.credibility * c.relevance_score)
        .sum::<f64>() / total_weight;

    // Average empirical level
    let avg_empirical: f64 = all_claims.iter()
        .map(|c| c.epistemic.empirical)
        .sum::<f64>() / all_claims.len() as f64;

    // Boost confidence for more evidence (diminishing returns)
    let evidence_bonus = (all_claims.len() as f64 / 10.0).min(0.2);

    (weighted_credibility * 0.5 + avg_empirical * 0.3 + evidence_bonus + 0.2).min(1.0)
}

/// Calculate aggregate epistemic position from claims
fn calculate_aggregate_epistemic(
    supporting: &[ClaimSummary],
    contradicting: &[ClaimSummary],
) -> EpistemicPosition {
    let all_claims: Vec<&ClaimSummary> = supporting.iter().chain(contradicting.iter()).collect();

    if all_claims.is_empty() {
        return EpistemicPosition {
            empirical: 0.5,
            normative: 0.5,
            mythic: 0.5,
        };
    }

    let total_relevance: f64 = all_claims.iter().map(|c| c.relevance_score).sum();

    let weighted_e: f64 = all_claims
        .iter()
        .map(|c| c.epistemic.empirical * c.relevance_score)
        .sum();
    let weighted_n: f64 = all_claims
        .iter()
        .map(|c| c.epistemic.normative * c.relevance_score)
        .sum();
    let weighted_m: f64 = all_claims
        .iter()
        .map(|c| c.epistemic.mythic * c.relevance_score)
        .sum();

    EpistemicPosition {
        empirical: weighted_e / total_relevance,
        normative: weighted_n / total_relevance,
        mythic: weighted_m / total_relevance,
    }
}

/// Calculate overall credibility score
fn calculate_credibility(
    supporting: &[ClaimSummary],
    contradicting: &[ClaimSummary],
) -> f64 {
    let all_claims: Vec<&ClaimSummary> = supporting.iter().chain(contradicting.iter()).collect();

    if all_claims.is_empty() {
        return 0.5;
    }

    let total_relevance: f64 = all_claims.iter().map(|c| c.relevance_score).sum();
    let weighted_cred: f64 = all_claims
        .iter()
        .map(|c| c.credibility * c.relevance_score)
        .sum();

    weighted_cred / total_relevance
}

/// Calculate source diversity
fn calculate_diversity(
    supporting: &[ClaimSummary],
    contradicting: &[ClaimSummary],
    related: &[ClaimSummary],
) -> f64 {
    let all_sources: Vec<Option<&String>> = supporting
        .iter()
        .chain(contradicting.iter())
        .chain(related.iter())
        .map(|c| c.source.as_ref())
        .collect();

    if all_sources.is_empty() {
        return 0.0;
    }

    // Count unique sources
    let mut unique_sources: Vec<&String> = vec![];
    for source in all_sources.iter().flatten() {
        if !unique_sources.contains(source) {
            unique_sources.push(source);
        }
    }

    // Diversity is ratio of unique sources to total
    let total = all_sources.len() as f64;
    let unique = unique_sources.len() as f64;

    (unique / total).min(1.0)
}

/// Generate suggested actions based on verdict
fn generate_suggestions(
    verdict: &FactCheckVerdict,
    supporting: &[ClaimSummary],
    contradicting: &[ClaimSummary],
) -> Vec<SuggestedAction> {
    let mut suggestions = vec![];

    match verdict {
        FactCheckVerdict::InsufficientEvidence => {
            suggestions.push(SuggestedAction {
                action_type: SuggestedActionType::SubmitClaim,
                description: "Submit a claim with evidence to help verify this statement".to_string(),
                priority: 1,
            });
        }
        FactCheckVerdict::Mixed => {
            // Suggest verification markets for key claims
            for claim in supporting.iter().chain(contradicting.iter()).take(3) {
                suggestions.push(SuggestedAction {
                    action_type: SuggestedActionType::RequestVerificationMarket {
                        claim_id: claim.claim_id.clone(),
                    },
                    description: format!(
                        "Request verification market for claim: {}",
                        claim.content_snippet
                    ),
                    priority: 2,
                });
            }
        }
        FactCheckVerdict::Unverifiable => {
            suggestions.push(SuggestedAction {
                action_type: SuggestedActionType::AddContext,
                description: "This statement may be subjective - consider adding context".to_string(),
                priority: 2,
            });
        }
        _ => {}
    }

    suggestions
}

// ============================================================================
// QUERY FUNCTIONS
// ============================================================================

/// Query claims by subject
#[hdk_extern]
pub fn query_claims_by_subject(input: SubjectQueryInput) -> ExternResult<Vec<ClaimSummary>> {
    // Query claims zome by searching tags that match the subject
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("claims"),
        FunctionName::from("get_claims_by_tag"),
        None,
        input.subject.clone(),
    )?;

    let claims: Vec<Record> = match response {
        ZomeCallResponse::Ok(bytes) => bytes.decode()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
        ZomeCallResponse::NetworkError(e) | ZomeCallResponse::CountersigningSession(e) => {
            debug!("Failed to query claims zome: {:?}", e);
            return Ok(vec![]);
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) | ZomeCallResponse::AuthenticationFailed(_, _) => {
            debug!("Unauthorized to query claims zome");
            return Ok(vec![]);
        }
    };

    // Filter and convert to ClaimSummary
    let mut summaries: Vec<ClaimSummary> = Vec::new();
    let now = sys_time()?;

    for record in claims {
        // Try to deserialize entry bytes
        if let Some(Entry::App(app_entry)) = record.entry().as_option() {
            let entry_bytes = app_entry.bytes();
            if let Ok(claim_data) = serde_json::from_slice::<serde_json::Value>(&entry_bytes) {
                // Extract epistemic values
                let classification = claim_data.get("classification").cloned().unwrap_or_default();
                let e = classification.get("empirical").and_then(|v| v.as_f64()).unwrap_or(0.5);
                let n = classification.get("normative").and_then(|v| v.as_f64()).unwrap_or(0.5);
                let m = classification.get("mythic").and_then(|v| v.as_f64()).unwrap_or(0.5);

                // Filter by epistemic requirements
                if e >= input.min_e && n >= input.min_n {
                    let claim_id = claim_data.get("id").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                    let content = claim_data.get("content").and_then(|v| v.as_str()).unwrap_or_default();
                    let confidence = claim_data.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.5);

                    // Create snippet from content (first 200 chars)
                    let snippet = if content.len() > 200 {
                        format!("{}...", &content[..200])
                    } else {
                        content.to_string()
                    };

                    summaries.push(ClaimSummary {
                        claim_id,
                        content_snippet: snippet,
                        epistemic: EpistemicPosition {
                            empirical: e,
                            normative: n,
                            mythic: m,
                        },
                        credibility: confidence,
                        relevance_score: calculate_relevance(&input.subject, content),
                        relationship: ClaimRelationship::Related,
                        source: claim_data.get("author").and_then(|v| v.as_str()).map(|s| s.to_string()),
                        created_at: now,
                    });
                }
            }
        }

        // Respect limit
        if summaries.len() >= input.limit as usize {
            break;
        }
    }

    // Sort by relevance
    summaries.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));

    Ok(summaries)
}

/// Calculate relevance score based on subject match
fn calculate_relevance(subject: &str, content: &str) -> f64 {
    let subject_lower = subject.to_lowercase();
    let content_lower = content.to_lowercase();

    // Simple relevance: check for exact match, partial match, or keyword presence
    if content_lower.contains(&subject_lower) {
        // Boost for exact phrase match
        0.9
    } else {
        // Check for individual keywords
        let keywords: Vec<&str> = subject_lower.split_whitespace().collect();
        let matched = keywords.iter().filter(|kw| content_lower.contains(*kw)).count();
        if keywords.is_empty() {
            0.5
        } else {
            0.3 + (0.6 * matched as f64 / keywords.len() as f64)
        }
    }
}

/// Query claims by topic
#[hdk_extern]
pub fn query_claims_by_topic(input: TopicQueryInput) -> ExternResult<Vec<ClaimSummary>> {
    let mut all_summaries: Vec<ClaimSummary> = Vec::new();
    let mut seen_claim_ids: Vec<String> = Vec::new();
    let now = sys_time()?;

    // Query claims zome for each topic
    for topic in &input.topics {
        let response = call(
            CallTargetCell::Local,
            ZomeName::from("claims"),
            FunctionName::from("get_claims_by_tag"),
            None,
            topic.clone(),
        );

        let claims: Vec<Record> = match response {
            Ok(ZomeCallResponse::Ok(bytes)) => bytes.decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
            Ok(_) | Err(_) => {
                debug!("Failed to query claims for topic {}", topic);
                continue;
            }
        };

        for record in claims {
            // Try to deserialize entry bytes
            if let Some(Entry::App(app_entry)) = record.entry().as_option() {
                let entry_bytes = app_entry.bytes();
                if let Ok(claim_data) = serde_json::from_slice::<serde_json::Value>(&entry_bytes) {
                    let claim_id = claim_data.get("id").and_then(|v| v.as_str()).unwrap_or_default().to_string();

                    // Deduplicate claims across topics
                    if seen_claim_ids.contains(&claim_id) {
                        continue;
                    }
                    seen_claim_ids.push(claim_id.clone());

                    // Extract epistemic values
                    let classification = claim_data.get("classification").cloned().unwrap_or_default();
                    let e = classification.get("empirical").and_then(|v| v.as_f64()).unwrap_or(0.5);
                    let n = classification.get("normative").and_then(|v| v.as_f64()).unwrap_or(0.5);
                    let m = classification.get("mythic").and_then(|v| v.as_f64()).unwrap_or(0.5);

                    // Filter by epistemic requirements
                    if e >= input.min_e && n >= input.min_n {
                        let content = claim_data.get("content").and_then(|v| v.as_str()).unwrap_or_default();
                        let confidence = claim_data.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.5);

                        // Create snippet from content (first 200 chars)
                        let snippet = if content.len() > 200 {
                            format!("{}...", &content[..200])
                        } else {
                            content.to_string()
                        };

                        // Calculate relevance based on topic match count
                        let tags: Vec<String> = claim_data.get("tags")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|t| t.as_str().map(String::from)).collect())
                            .unwrap_or_default();

                        let topic_match_count = input.topics.iter()
                            .filter(|t| tags.iter().any(|tag| tag.to_lowercase() == t.to_lowercase()))
                            .count();

                        let relevance = if input.topics.is_empty() {
                            0.5
                        } else {
                            0.3 + (0.7 * topic_match_count as f64 / input.topics.len() as f64)
                        };

                        all_summaries.push(ClaimSummary {
                            claim_id,
                            content_snippet: snippet,
                            epistemic: EpistemicPosition {
                                empirical: e,
                                normative: n,
                                mythic: m,
                            },
                            credibility: confidence,
                            relevance_score: relevance,
                            relationship: ClaimRelationship::Related,
                            source: claim_data.get("author").and_then(|v| v.as_str()).map(|s| s.to_string()),
                            created_at: now,
                        });
                    }
                }
            }

            // Respect limit
            if all_summaries.len() >= input.limit as usize {
                break;
            }
        }

        if all_summaries.len() >= input.limit as usize {
            break;
        }
    }

    // Sort by relevance, then by credibility
    all_summaries.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.credibility.partial_cmp(&a.credibility).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Truncate to limit
    all_summaries.truncate(input.limit as usize);

    Ok(all_summaries)
}

/// Get detailed information about a claim
#[hdk_extern]
pub fn get_claim_details(claim_id: String) -> ExternResult<Option<ClaimDetails>> {
    // 1. Get the claim from claims zome
    let claim_response = call(
        CallTargetCell::Local,
        ZomeName::from("claims"),
        FunctionName::from("get_claim"),
        None,
        claim_id.clone(),
    )?;

    let claim_record: Option<Record> = match claim_response {
        ZomeCallResponse::Ok(bytes) => bytes
            .decode()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
        ZomeCallResponse::NetworkError(e) | ZomeCallResponse::CountersigningSession(e) => {
            debug!("Failed to get claim from claims zome: {:?}", e);
            return Ok(None);
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) | ZomeCallResponse::AuthenticationFailed(_, _) => {
            debug!("Unauthorized to get claim from claims zome");
            return Ok(None);
        }
    };

    let claim_record = match claim_record {
        Some(record) => record,
        None => return Ok(None),
    };

    // 2. Get evidence from claims zome
    let evidence_response = call(
        CallTargetCell::Local,
        ZomeName::from("claims"),
        FunctionName::from("get_claim_evidence"),
        None,
        claim_id.clone(),
    )?;

    let evidence: Vec<Record> = match evidence_response {
        ZomeCallResponse::Ok(bytes) => bytes
            .decode()
            .unwrap_or_default(),
        _ => {
            debug!("Failed to get evidence from claims zome");
            vec![]
        }
    };

    // 3. Get relationships from graph zome
    let outgoing_response = call(
        CallTargetCell::Local,
        ZomeName::from("graph"),
        FunctionName::from("get_outgoing_relationships"),
        None,
        claim_id.clone(),
    )?;

    let incoming_response = call(
        CallTargetCell::Local,
        ZomeName::from("graph"),
        FunctionName::from("get_incoming_relationships"),
        None,
        claim_id.clone(),
    )?;

    let mut relationships: Vec<Record> = match outgoing_response {
        ZomeCallResponse::Ok(bytes) => bytes.decode().unwrap_or_default(),
        _ => {
            debug!("Failed to get outgoing relationships");
            vec![]
        }
    };

    if let ZomeCallResponse::Ok(bytes) = incoming_response {
        if let Ok(incoming) = bytes.decode::<Vec<Record>>() {
            relationships.extend(incoming);
        }
    }

    // 4. Calculate credibility based on claim data and evidence
    let credibility = calculate_claim_credibility(&claim_record, &evidence)?;

    // 5. Get verification status from markets_integration zome
    let markets_response = call(
        CallTargetCell::Local,
        ZomeName::from("markets_integration"),
        FunctionName::from("get_claim_evidence"),
        None,
        claim_id.clone(),
    )?;

    let verification_status = match markets_response {
        ZomeCallResponse::Ok(bytes) => {
            let market_records: Vec<Record> = bytes.decode().unwrap_or_default();
            if market_records.is_empty() {
                None
            } else {
                // Check if any market has resolved
                let mut status = "pending".to_string();
                for record in market_records {
                    if let Some(Entry::App(app_entry)) = record.entry().as_option() {
                        let entry_bytes = app_entry.bytes();
                        if let Ok(evidence) = serde_json::from_slice::<serde_json::Value>(&entry_bytes) {
                            if let Some(evidence_type) = evidence.get("evidence_type") {
                                if evidence_type.get("PredictionResolved").is_some() {
                                    status = "verified".to_string();
                                    break;
                                }
                            }
                        }
                    }
                }
                Some(status)
            }
        }
        _ => {
            debug!("Failed to get market evidence");
            None
        }
    };

    Ok(Some(ClaimDetails {
        claim: claim_record,
        evidence,
        relationships,
        credibility,
        verification_status,
    }))
}

/// Calculate credibility based on claim and evidence
fn calculate_claim_credibility(claim_record: &Record, evidence: &[Record]) -> ExternResult<f64> {
    // Base credibility from claim confidence
    let base_credibility = if let Some(Entry::App(app_entry)) = claim_record.entry().as_option() {
        let entry_bytes = app_entry.bytes();
        if let Ok(claim) = serde_json::from_slice::<serde_json::Value>(&entry_bytes) {
            claim.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.5)
        } else {
            0.5
        }
    } else {
        0.5
    };

    // Boost credibility based on evidence count and strength
    if evidence.is_empty() {
        return Ok(base_credibility * 0.8); // Penalize claims with no evidence
    }

    let mut total_strength = 0.0;
    let mut count = 0;

    for record in evidence {
        if let Some(Entry::App(app_entry)) = record.entry().as_option() {
            let entry_bytes = app_entry.bytes();
            if let Ok(ev) = serde_json::from_slice::<serde_json::Value>(&entry_bytes) {
                if let Some(strength) = ev.get("strength").and_then(|v| v.as_f64()) {
                    total_strength += strength;
                    count += 1;
                }
            }
        }
    }

    let avg_strength = if count > 0 {
        total_strength / count as f64
    } else {
        0.5
    };

    // Combine base credibility with evidence strength
    let combined = base_credibility * 0.6 + avg_strength * 0.4;

    // Boost for multiple pieces of evidence (up to 20% boost for 5+ pieces)
    let evidence_boost = (count as f64 / 5.0).min(1.0) * 0.2;

    Ok((combined + evidence_boost).min(1.0))
}

// ============================================================================
// BATCH OPERATIONS
// ============================================================================

/// Batch fact check multiple statements
///
/// More efficient than calling fact_check multiple times as it can
/// share query results across statements.
#[hdk_extern]
pub fn batch_fact_check(inputs: Vec<FactCheckInput>) -> ExternResult<Vec<FactCheckResult>> {
    let mut results = Vec::with_capacity(inputs.len());

    for input in inputs {
        let result = fact_check(input)?;
        results.push(result);
    }

    Ok(results)
}

// ============================================================================
// HISTORY AND ANALYTICS
// ============================================================================

/// Get fact check history for a hApp
#[hdk_extern]
pub fn get_happ_fact_checks(source_happ: String) -> ExternResult<Vec<Record>> {
    let happ_anchor = format!("happ:{}", source_happ);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&happ_anchor)?, LinkTypes::HappToFactCheck)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Ok(target) = ActionHash::try_from(link.target) {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

/// Get fact checks for a subject
#[hdk_extern]
pub fn get_subject_fact_checks(subject: String) -> ExternResult<Vec<Record>> {
    let subject_anchor = format!("subject:{}", subject);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&subject_anchor)?, LinkTypes::SubjectToFactCheck)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Ok(target) = ActionHash::try_from(link.target) {
            if let Some(record) = get(target, GetOptions::default())? {
                records.push(record);
            }
        }
    }

    Ok(records)
}

/// Get result for a fact check request
#[hdk_extern]
pub fn get_fact_check_result(request_hash: ActionHash) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(request_hash, LinkTypes::RequestToResult)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Ok(target) = ActionHash::try_from(link.target.clone()) {
            return get(target, GetOptions::default());
        }
    }

    Ok(None)
}

// ============================================================================
// TEMPLATES
// ============================================================================

/// Create a fact check template
#[hdk_extern]
pub fn create_template(template: FactCheckTemplate) -> ExternResult<ActionHash> {
    create_entry(EntryTypes::FactCheckTemplate(template))
}

/// Get all templates
#[hdk_extern]
pub fn get_templates(_: ()) -> ExternResult<Vec<Record>> {
    // In a full implementation, we'd have an index of all templates
    Ok(vec![])
}
