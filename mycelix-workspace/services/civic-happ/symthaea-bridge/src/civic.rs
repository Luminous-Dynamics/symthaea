//! Holochain client for civic-happ
//!
//! Provides access to civic knowledge and agent reputation zomes.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Civic knowledge search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeResult {
    pub action_hash: String,
    pub title: String,
    pub content: String,
    pub domain: String,
    pub knowledge_type: String,
    pub geographic_scope: Option<String>,
    pub source: Option<String>,
    pub contact_phone: Option<String>,
    pub address: Option<String>,
    pub keywords: Vec<String>,
}

/// Trust score for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustScore {
    pub quality: f32,
    pub consistency: f32,
    pub reputation: f32,
    pub composite: f32,
    pub confidence: f32,
    pub positive_count: u32,
    pub negative_count: u32,
}

/// Client for interacting with civic-happ
pub struct CivicClient {
    #[allow(dead_code)]
    conductor_url: String,
    // In a full implementation, this would hold the actual Holochain client
    // For now, we use HTTP fallback via the conductor's app interface
    http_client: reqwest::Client,
}

impl CivicClient {
    /// Connect to Holochain conductor
    pub async fn connect(conductor_url: &str) -> Result<Self> {
        // For now, verify the conductor is reachable
        // Full implementation would use holochain_client::AppWebsocket
        let http_client = reqwest::Client::new();

        // Try a simple connection check
        // The actual Holochain WebSocket connection would be established here
        debug!("Attempting connection to Holochain at {}", conductor_url);

        Ok(Self {
            conductor_url: conductor_url.to_string(),
            http_client,
        })
    }

    /// Search for civic knowledge
    pub async fn search_knowledge(
        &self,
        question: &str,
        location: Option<&str>,
        domain: Option<&str>,
    ) -> Result<Vec<KnowledgeResult>> {
        // Extract keywords from question
        let keywords = self.extract_keywords(question);

        // Detect domain if not provided
        let detected_domain = domain.map(String::from).or_else(|| self.detect_domain(&keywords));

        debug!(
            "Searching civic knowledge: domain={:?}, location={:?}, keywords={:?}",
            detected_domain, location, keywords
        );

        // In production, this would call the Holochain zome directly
        // For now, return mock data that demonstrates the system
        self.mock_search(detected_domain.as_deref(), location, &keywords).await
    }

    /// Record a reputation event for an agent
    pub async fn record_reputation_event(
        &self,
        agent_pubkey: &str,
        event_type: &str,
        context: Option<&str>,
        conversation_id: Option<&str>,
    ) -> Result<String> {
        debug!(
            "Recording reputation event: agent={}, type={}, conv={:?}",
            agent_pubkey, event_type, conversation_id
        );

        // In production, this would call the agent_reputation zome
        // For now, return a mock action hash
        Ok(format!("mock_event_{}", uuid::Uuid::new_v4()))
    }

    /// Get trust score for an agent
    pub async fn get_trust_score(&self, agent_pubkey: &str) -> Result<Option<TrustScore>> {
        debug!("Getting trust score for agent: {}", agent_pubkey);

        // In production, this would call the agent_reputation zome
        // For now, return mock data
        if agent_pubkey.starts_with("uhCAk") {
            Ok(Some(TrustScore {
                quality: 0.85,
                consistency: 0.78,
                reputation: 0.82,
                composite: 0.82, // 0.4*0.85 + 0.3*0.78 + 0.3*0.82
                confidence: 0.65,
                positive_count: 45,
                negative_count: 8,
            }))
        } else {
            Ok(None)
        }
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    fn extract_keywords(&self, text: &str) -> Vec<String> {
        let stop_words: std::collections::HashSet<&str> = [
            "i", "me", "my", "the", "a", "an", "is", "are", "was", "were",
            "do", "does", "did", "have", "has", "had", "can", "could", "would",
            "should", "will", "what", "where", "when", "how", "who", "which",
            "for", "to", "of", "in", "on", "at", "by", "with", "about", "help",
            "need", "want", "get", "find", "looking",
        ].into_iter().collect();

        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .filter(|w| w.len() > 2 && !stop_words.contains(w))
            .map(String::from)
            .collect()
    }

    fn detect_domain(&self, keywords: &[String]) -> Option<String> {
        let domain_keywords: &[(&str, &[&str])] = &[
            ("benefits", &["benefits", "snap", "food", "stamps", "welfare", "medicaid", "tanf", "wic", "assistance"]),
            ("permits", &["permit", "license", "building", "business", "construction", "zoning"]),
            ("tax", &["tax", "taxes", "irs", "refund", "deduction", "property"]),
            ("voting", &["vote", "voting", "election", "ballot", "registration", "poll"]),
            ("justice", &["court", "legal", "lawyer", "appeal", "custody", "bail", "ticket"]),
            ("housing", &["housing", "rent", "eviction", "landlord", "tenant", "section", "hud"]),
            ("employment", &["job", "employment", "unemployment", "work", "hired", "fired", "wages"]),
            ("education", &["school", "education", "college", "scholarship", "financial", "aid", "tuition"]),
            ("health", &["health", "hospital", "doctor", "insurance", "medical", "clinic", "mental"]),
            ("emergency", &["emergency", "disaster", "fire", "flood", "shelter", "fema", "crisis"]),
        ];

        for (domain, domain_words) in domain_keywords {
            if keywords.iter().any(|kw| domain_words.contains(&kw.as_str())) {
                return Some(domain.to_string());
            }
        }

        None
    }

    async fn mock_search(
        &self,
        domain: Option<&str>,
        location: Option<&str>,
        keywords: &[String],
    ) -> Result<Vec<KnowledgeResult>> {
        // Mock knowledge base for demonstration
        let mock_knowledge = vec![
            KnowledgeResult {
                action_hash: "uhCEk_mock_1".to_string(),
                title: "SNAP Benefits Eligibility".to_string(),
                content: "To qualify for SNAP benefits, your household income must be at or below 130% of the federal poverty level. A household of 1 can earn up to $1,580/month. Apply at your local SNAP office or online.".to_string(),
                domain: "benefits".to_string(),
                knowledge_type: "eligibility_rule".to_string(),
                geographic_scope: Some("US".to_string()),
                source: Some("USDA FNS".to_string()),
                contact_phone: Some("1-800-221-5689".to_string()),
                address: None,
                keywords: vec!["snap".to_string(), "food".to_string(), "benefits".to_string()],
            },
            KnowledgeResult {
                action_hash: "uhCEk_mock_2".to_string(),
                title: "Building Permit Requirements".to_string(),
                content: "Building permits are required for new construction, additions, and major renovations. Typical processing time is 2-4 weeks. Fees vary by project scope.".to_string(),
                domain: "permits".to_string(),
                knowledge_type: "process".to_string(),
                geographic_scope: location.map(String::from),
                source: Some("City Planning Dept".to_string()),
                contact_phone: Some("555-123-4567".to_string()),
                address: Some("123 City Hall Plaza".to_string()),
                keywords: vec!["permit".to_string(), "building".to_string(), "construction".to_string()],
            },
            KnowledgeResult {
                action_hash: "uhCEk_mock_3".to_string(),
                title: "Voter Registration".to_string(),
                content: "You can register to vote online, by mail, or in person at your county elections office. The deadline is 30 days before election day in most states.".to_string(),
                domain: "voting".to_string(),
                knowledge_type: "process".to_string(),
                geographic_scope: Some("US".to_string()),
                source: Some("Vote.gov".to_string()),
                contact_phone: Some("1-866-OUR-VOTE".to_string()),
                address: None,
                keywords: vec!["vote".to_string(), "register".to_string(), "election".to_string()],
            },
            KnowledgeResult {
                action_hash: "uhCEk_mock_4".to_string(),
                title: "Emergency Shelter Services".to_string(),
                content: "If you need emergency shelter, call 211 for local resources. Most shelters provide meals, showers, and case management services.".to_string(),
                domain: "emergency".to_string(),
                knowledge_type: "resource".to_string(),
                geographic_scope: location.map(String::from),
                source: Some("United Way 211".to_string()),
                contact_phone: Some("211".to_string()),
                address: None,
                keywords: vec!["shelter".to_string(), "emergency".to_string(), "homeless".to_string()],
            },
            KnowledgeResult {
                action_hash: "uhCEk_mock_5".to_string(),
                title: "Free Tax Preparation (VITA)".to_string(),
                content: "The Volunteer Income Tax Assistance (VITA) program offers free tax prep for people earning $60,000 or less. Find a site at irs.gov/vita.".to_string(),
                domain: "tax".to_string(),
                knowledge_type: "resource".to_string(),
                geographic_scope: Some("US".to_string()),
                source: Some("IRS".to_string()),
                contact_phone: Some("1-800-906-9887".to_string()),
                address: None,
                keywords: vec!["tax".to_string(), "free".to_string(), "vita".to_string()],
            },
        ];

        // Filter by domain if specified
        let filtered: Vec<KnowledgeResult> = mock_knowledge
            .into_iter()
            .filter(|k| {
                // Match domain
                if let Some(d) = domain {
                    if k.domain != d {
                        return false;
                    }
                }
                // Match keywords
                if !keywords.is_empty() {
                    let has_match = keywords.iter().any(|kw| {
                        k.keywords.iter().any(|k_kw| k_kw.contains(kw)) ||
                        k.title.to_lowercase().contains(kw) ||
                        k.content.to_lowercase().contains(kw)
                    });
                    if !has_match {
                        return false;
                    }
                }
                true
            })
            .collect();

        Ok(filtered)
    }
}
