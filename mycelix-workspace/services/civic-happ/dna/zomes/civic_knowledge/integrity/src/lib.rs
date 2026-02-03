//! Civic Knowledge Integrity Zome
//!
//! Defines entry types for civic knowledge that can be queried by Symthaea AI agents.
//! This enables decentralized storage of civic information on the DHT.
//!
//! ## Entry Types
//!
//! - `CivicKnowledge`: A piece of civic knowledge (FAQ, eligibility rule, contact info)
//! - `KnowledgeUpdate`: An update to existing knowledge
//! - `KnowledgeValidation`: A validation from a civic agent or authority
//!
//! ## Design Philosophy
//!
//! Knowledge is categorized by civic domain (benefits, permits, voting, etc.) and
//! can be validated by trusted civic authorities. This allows Symthaea agents to
//! provide accurate, up-to-date information to citizens.

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

/// Civic domains (matches Symthaea domains)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CivicDomain {
    /// SNAP, Medicaid, TANF, housing assistance
    Benefits,
    /// Business licenses, building permits
    Permits,
    /// Tax questions, credits, free filing
    Tax,
    /// Voter registration, polling, ballots
    Voting,
    /// Courts, legal aid, expungement
    Justice,
    /// Affordable housing, tenant rights
    Housing,
    /// Jobs, unemployment, workplace rights
    Employment,
    /// Schools, financial aid, GED
    Education,
    /// Healthcare, clinics, mental health
    Health,
    /// Crisis resources, disaster assistance
    Emergency,
    /// General government questions
    General,
}

impl Default for CivicDomain {
    fn default() -> Self {
        Self::General
    }
}

/// Knowledge type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeType {
    /// Frequently asked question with answer
    Faq,
    /// Eligibility rule for a benefit or program
    EligibilityRule,
    /// Contact information (phone, address, hours)
    ContactInfo,
    /// Process or procedure description
    Process,
    /// Legal requirement or regulation
    Regulation,
    /// Resource or service available
    Resource,
    /// Form or document information
    Form,
    /// Deadline or date-based information
    Deadline,
}

impl Default for KnowledgeType {
    fn default() -> Self {
        Self::Faq
    }
}

/// A piece of civic knowledge stored on the DHT
#[hdk_entry_helper]
#[derive(Clone)]
pub struct CivicKnowledge {
    /// The civic domain this knowledge belongs to
    pub domain: CivicDomain,
    /// Type of knowledge
    pub knowledge_type: KnowledgeType,
    /// Short title/question
    pub title: String,
    /// Detailed content/answer
    pub content: String,
    /// Geographic scope (e.g., "Texas", "Austin", "78701")
    pub geographic_scope: Option<String>,
    /// Keywords for search
    pub keywords: Vec<String>,
    /// Source of this information
    pub source: Option<String>,
    /// When this information was last verified
    pub last_verified: Option<u64>,
    /// Expiration timestamp (if applicable)
    pub expires_at: Option<u64>,
    /// Related links
    pub links: Vec<String>,
    /// Contact phone number (if applicable)
    pub contact_phone: Option<String>,
    /// Address (if applicable)
    pub address: Option<String>,
}

/// An update to existing civic knowledge
#[hdk_entry_helper]
#[derive(Clone)]
pub struct KnowledgeUpdate {
    /// Hash of the original CivicKnowledge entry being updated
    pub original_hash: ActionHash,
    /// The updated content
    pub updated_content: String,
    /// Reason for update
    pub reason: String,
    /// Who submitted this update
    pub submitter_note: Option<String>,
}

/// A validation of civic knowledge by an authority
#[hdk_entry_helper]
#[derive(Clone)]
pub struct KnowledgeValidation {
    /// Hash of the CivicKnowledge entry being validated
    pub knowledge_hash: ActionHash,
    /// Whether the knowledge is validated as accurate
    pub is_valid: bool,
    /// Authority type (e.g., "government_agency", "subject_expert", "community")
    pub authority_type: String,
    /// Notes from the validator
    pub notes: Option<String>,
    /// Timestamp of validation
    pub validated_at: u64,
}

/// Link types for the civic knowledge zome
#[hdk_link_types]
pub enum CivicKnowledgeLinkTypes {
    /// Links knowledge entries by domain
    DomainToKnowledge,
    /// Links knowledge entries by geographic scope
    GeoToKnowledge,
    /// Links knowledge to its updates
    KnowledgeToUpdate,
    /// Links knowledge to its validations
    KnowledgeToValidation,
    /// Links for keyword search
    KeywordToKnowledge,
    /// Links from agent to their authored knowledge
    AgentToKnowledge,
}

/// Anchor entry for path-based linking
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Anchor(pub String);

/// Entry types for this zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    CivicKnowledge(CivicKnowledge),
    #[entry_type(visibility = "public")]
    KnowledgeUpdate(KnowledgeUpdate),
    #[entry_type(visibility = "public")]
    KnowledgeValidation(KnowledgeValidation),
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
}

/// Genesis validation - always succeeds for this zome
#[hdk_extern]
pub fn genesis_self_check(_: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Validate agent joining the network
#[hdk_extern]
pub fn validate(_: Op) -> ExternResult<ValidateCallbackResult> {
    // TODO: Add proper validation rules
    // - CivicKnowledge: Validate content length, required fields
    // - KnowledgeUpdate: Validate original_hash exists
    // - KnowledgeValidation: Validate authority_type
    Ok(ValidateCallbackResult::Valid)
}
