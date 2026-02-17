//! # Mycelix Governance - Sweettest Integration Tests
//!
//! Uses pre-built .dna bundle instead of linking zome crates directly,
//! avoiding duplicate `__num_entry_types`/`__num_link_types` symbol collisions
//! from multiple integrity zomes.
//!
//! ## Running Tests
//!
//! ```bash
//! # Build the DNA first
//! cd mycelix-governance && hc dna pack dna/
//!
//! # Run tests (require DNA bundle)
//! cargo test --test sweettest_integration -- --ignored
//! ```

use holochain::sweettest::*;
use holochain::prelude::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types for deserialization (avoids importing zome crates)
// ============================================================================

// --- Proposals integrity types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Proposal {
    pub id: String,
    pub title: String,
    pub description: String,
    pub proposal_type: ProposalType,
    pub author: String,
    pub status: ProposalStatus,
    pub actions: String,
    pub discussion_url: Option<String>,
    pub voting_starts: Timestamp,
    pub voting_ends: Timestamp,
    pub created: Timestamp,
    pub updated: Timestamp,
    pub version: u32,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ProposalType {
    Standard,
    Emergency,
    Constitutional,
    Parameter,
    Funding,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ProposalStatus {
    Draft,
    Active,
    Ended,
    Approved,
    Signed,
    Rejected,
    Executed,
    Cancelled,
    Failed,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DiscussionContribution {
    pub id: String,
    pub proposal_id: String,
    pub contributor: String,
    pub content: String,
    pub harmony_tags: Vec<String>,
    pub stance: Option<Stance>,
    pub parent_id: Option<String>,
    pub created_at: Timestamp,
    pub edited: bool,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Stance {
    Support,
    Oppose,
    Neutral,
    Amend,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DiscussionReflection {
    pub id: String,
    pub proposal_id: String,
    pub timestamp: Timestamp,
    pub contributor_count: u64,
    pub contribution_count: u64,
    pub avg_contributions_per_participant: f64,
    pub max_thread_depth: u8,
    pub harmony_coverage: Vec<HarmonyPresenceProposals>,
    pub harmony_diversity: f64,
    pub absent_harmonies: Vec<String>,
    pub support_count: u64,
    pub oppose_count: u64,
    pub neutral_count: u64,
    pub amend_count: u64,
    pub preliminary_sentiment: f64,
    pub voice_concentration: f64,
    pub cross_camp_engagement: f64,
    pub substantiveness_score: f64,
    pub discussion_saturated: bool,
    pub unaddressed_concerns: Vec<String>,
    pub ready_for_vote: bool,
    pub readiness_reasoning: String,
    pub summary: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HarmonyPresenceProposals {
    pub harmony: String,
    pub presence: f64,
    pub example_contribution_id: Option<String>,
}

// --- Voting integrity types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Vote {
    pub id: String,
    pub proposal_id: String,
    pub voter: String,
    pub choice: VoteChoice,
    pub weight: f64,
    pub reason: Option<String>,
    pub delegated: bool,
    pub delegator: Option<String>,
    pub voted_at: Timestamp,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum VoteChoice {
    For,
    Against,
    Abstain,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VoteTally {
    pub proposal_id: String,
    pub votes_for: f64,
    pub votes_against: f64,
    pub abstentions: f64,
    pub total_weight: f64,
    pub quorum_reached: bool,
    pub approved: bool,
    pub tallied_at: Timestamp,
    pub final_tally: bool,
}

// --- Constitution integrity types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Charter {
    pub id: String,
    pub version: u32,
    pub preamble: String,
    pub articles: String,
    pub rights: Vec<String>,
    pub amendment_process: String,
    pub adopted: Timestamp,
    pub last_amended: Option<Timestamp>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Amendment {
    pub id: String,
    pub charter_version: u32,
    pub amendment_type: AmendmentType,
    pub article: Option<String>,
    pub original_text: Option<String>,
    pub new_text: String,
    pub rationale: String,
    pub proposer: String,
    pub proposal_id: String,
    pub status: ConstitutionAmendmentStatus,
    pub created: Timestamp,
    pub ratified: Option<Timestamp>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AmendmentType {
    AddArticle,
    ModifyArticle,
    RemoveArticle,
    AddRight,
    ModifyRight,
    RemoveRight,
    ModifyPreamble,
    ModifyProcess,
}

/// Constitution amendment status (distinct from proposals AmendmentStatus)
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ConstitutionAmendmentStatus {
    Draft,
    Deliberation,
    Voting,
    Ratified,
    Rejected,
    Withdrawn,
}

/// Input mirror for propose_amendment
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProposeAmendmentInput {
    pub amendment_type: AmendmentType,
    pub article: Option<String>,
    pub original_text: Option<String>,
    pub new_text: String,
    pub rationale: String,
    pub proposer_did: String,
    pub proposal_id: String,
}

/// Input mirror for set_parameter
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SetParameterInput {
    pub name: String,
    pub value: String,
    pub value_type: ParameterType,
    pub description: String,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
    pub proposal_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ParameterType {
    Integer,
    Float,
    Boolean,
    String,
    Duration,
    Percentage,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GovernanceParameter {
    pub name: String,
    pub value: String,
    pub value_type: ParameterType,
    pub description: String,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
    pub updated: Timestamp,
    pub changed_by_proposal: Option<String>,
}

// --- Execution integrity types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Timelock {
    pub id: String,
    pub proposal_id: String,
    pub actions: String,
    pub started: Timestamp,
    pub expires: Timestamp,
    pub status: TimelockStatus,
    pub cancellation_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TimelockStatus {
    Pending,
    Ready,
    Executed,
    Cancelled,
    Failed,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Execution {
    pub id: String,
    pub timelock_id: String,
    pub proposal_id: String,
    pub executor: String,
    pub status: ExecutionStatus,
    pub result: Option<String>,
    pub error: Option<String>,
    pub executed_at: Timestamp,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ExecutionStatus {
    Success,
    Failed,
    PartialSuccess,
}

/// Input mirror for create_timelock
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateTimelockInput {
    pub proposal_id: String,
    pub actions: String,
    pub duration_hours: u32,
}

/// Input mirror for execute_timelock
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExecuteTimelockInput {
    pub timelock_id: String,
    pub executor_did: String,
}

// --- Councils integrity types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Council {
    pub id: String,
    pub name: String,
    pub purpose: String,
    pub council_type: CouncilType,
    pub parent_council_id: Option<String>,
    pub phi_threshold: f64,
    pub quorum: f64,
    pub supermajority: f64,
    pub can_spawn_children: bool,
    pub max_delegation_depth: u8,
    pub signing_committee_id: Option<String>,
    pub status: CouncilStatus,
    pub created_at: Timestamp,
    pub last_activity: Timestamp,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CouncilType {
    Root,
    Domain { domain: String },
    Regional { region: String },
    WorkingGroup { focus: String, expires: Option<Timestamp> },
    Advisory,
    Emergency { expires: Timestamp },
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CouncilStatus {
    Active,
    Dormant,
    Dissolved,
    Suspended,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CouncilMembership {
    pub id: String,
    pub council_id: String,
    pub member_did: String,
    pub role: MemberRole,
    pub phi_score: f64,
    pub voting_weight: f64,
    pub can_delegate: bool,
    pub status: MembershipStatus,
    pub joined_at: Timestamp,
    pub last_participation: Timestamp,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MemberRole {
    Member,
    Facilitator,
    Steward,
    Observer,
    Delegate { from_council: String },
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MembershipStatus {
    Active,
    OnLeave,
    Suspended,
    Removed,
}

// --- Coordinator input types ---

/// Input mirror for cast_vote
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CastVoteInput {
    pub proposal_id: String,
    pub voter_did: String,
    pub choice: VoteChoice,
    pub reason: Option<String>,
}

/// Input mirror for create_council
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateCouncilInput {
    pub name: String,
    pub purpose: String,
    pub council_type: CouncilType,
    pub parent_council_id: Option<String>,
    pub phi_threshold: f64,
    pub quorum: f64,
    pub supermajority: f64,
    pub can_spawn_children: bool,
    pub max_delegation_depth: u8,
    #[serde(default)]
    pub signing_committee_id: Option<String>,
}

/// Input mirror for join_council
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct JoinCouncilInput {
    pub council_id: String,
    pub member_did: String,
    pub role: MemberRole,
    pub phi_score: f64,
}

/// Input mirror for update_proposal_status
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateStatusInput {
    pub proposal_id: String,
    pub new_status: ProposalStatus,
}

/// Input mirror for add_contribution
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AddContributionInput {
    pub proposal_id: String,
    pub contributor_did: String,
    pub content: String,
    pub harmony_tags: Option<Vec<String>>,
    pub stance: Option<Stance>,
    pub parent_id: Option<String>,
}

// ============================================================================
// Threshold-Signing Mirror Types
// ============================================================================

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DkgPhase {
    Registration,
    Dealing,
    Verification,
    Complete,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SigningCommittee {
    pub id: String,
    pub name: String,
    pub threshold: u32,
    pub min_members: u32,
    pub max_members: u32,
    pub phase: DkgPhase,
    pub epoch: u32,
    pub public_key: Option<Vec<u8>>,
    pub public_commitments: Vec<Vec<u8>>,
    pub qualified_members: Vec<String>,
    pub created_at: Timestamp,
    pub created_by: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CommitteeMember {
    pub committee_id: String,
    pub participant_id: u32,
    pub agent: String,
    pub member_did: String,
    pub trust_score: f64,
    pub public_share: Option<Vec<u8>>,
    pub vss_commitment: Option<Vec<u8>>,
    pub deal_submitted: bool,
    pub qualified: bool,
    pub registered_at: Timestamp,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ThresholdSignature {
    pub committee_id: String,
    pub content_hash: Vec<u8>,
    pub combined_signature: Vec<u8>,
    pub signer_count: u32,
    pub signer_ids: Vec<u32>,
    pub verified: bool,
    pub created_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateCommitteeInput {
    pub name: String,
    pub threshold: u32,
    pub min_members: u32,
    pub max_members: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterMemberInput {
    pub committee_id: String,
    pub member_did: String,
    pub trust_score: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MarkTimelockReadyInput {
    pub timelock_id: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateParameterInput {
    pub parameter: String,
    pub value: String,
    #[serde(default)]
    pub proposal_id: Option<String>,
}

// ============================================================================
// Test Utilities
// ============================================================================

/// Path to the pre-built DNA bundle
fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_governance_dna.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load governance DNA bundle")
}

/// Decode an entry from a Record into a concrete type via MessagePack deserialization.
fn decode_entry<T: serde::de::DeserializeOwned>(record: &Record) -> Option<T> {
    match record.entry().as_option()? {
        Entry::App(bytes) => {
            let sb = SerializedBytes::from(bytes.to_owned());
            rmp_serde::from_slice(sb.bytes()).ok()
        }
        _ => None,
    }
}

fn now_timestamp() -> Timestamp {
    Timestamp::from_micros(chrono::Utc::now().timestamp_micros())
}

fn make_proposal(id: &str, author_did: &str, proposal_type: ProposalType) -> Proposal {
    let now = now_timestamp();
    let one_day_micros: i64 = 86400 * 1_000_000;
    let voting_starts = now;
    let voting_ends = Timestamp::from_micros(now.as_micros() + one_day_micros);

    Proposal {
        id: id.to_string(),
        title: format!("Test Proposal: {}", id),
        description: "This is a comprehensive test proposal for the governance system.".to_string(),
        proposal_type,
        author: author_did.to_string(),
        status: ProposalStatus::Draft,
        actions: serde_json::json!({"action": "test"}).to_string(),
        discussion_url: Some("https://discourse.mycelix.net/proposals/test".to_string()),
        voting_starts,
        voting_ends,
        created: now,
        updated: now,
        version: 1,
    }
}

fn make_cast_vote_input(proposal_id: &str, voter_did: &str, choice: VoteChoice) -> CastVoteInput {
    CastVoteInput {
        proposal_id: proposal_id.to_string(),
        voter_did: voter_did.to_string(),
        choice,
        reason: Some("Test vote".to_string()),
    }
}

fn make_contribution_input(proposal_id: &str, contributor_did: &str) -> AddContributionInput {
    AddContributionInput {
        proposal_id: proposal_id.to_string(),
        contributor_did: contributor_did.to_string(),
        content: "Discussion contribution for governance test.".to_string(),
        harmony_tags: Some(vec!["SacredReciprocity".to_string()]),
        stance: Some(Stance::Support),
        parent_id: None,
    }
}

fn make_charter(id: &str) -> Charter {
    let now = now_timestamp();
    Charter {
        id: id.to_string(),
        version: 1,
        preamble: "We the participants of Mycelix commit to decentralized governance.".to_string(),
        articles: serde_json::json!([
            {"number": 1, "title": "Governance Structure", "text": "All decisions through proposals."},
            {"number": 2, "title": "Rights", "text": "Every participant has voice."}
        ]).to_string(),
        rights: vec![
            "Right to propose".to_string(),
            "Right to vote".to_string(),
            "Right to information".to_string(),
        ],
        amendment_process: "Requires 2/3 supermajority and 7-day deliberation.".to_string(),
        adopted: now,
        last_amended: None,
    }
}

// ============================================================================
// Proposal Tests
// ============================================================================

#[cfg(test)]
mod proposal_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_and_get_proposal() {
        println!("=== test_create_and_get_proposal ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let author_did = format!("did:mycelix:{}", agent);
        let proposal = make_proposal("MIP-001", &author_did, ProposalType::Standard);

        // Create proposal
        let proposal_record: Record = conductor
            .call(&cell.zome("proposals"), "create_proposal", proposal.clone())
            .await;

        let created: Proposal = decode_entry(&proposal_record).expect("No entry found");

        assert_eq!(created.id, "MIP-001", "Proposal ID must match");
        assert_eq!(created.author, author_did, "Author must match");
        assert_eq!(created.version, 1, "Version should be 1");

        // Get proposal by ID
        let retrieved: Option<Record> = conductor
            .call(&cell.zome("proposals"), "get_proposal", "MIP-001".to_string())
            .await;

        assert!(retrieved.is_some(), "Proposal should be retrievable");

        let retrieved_proposal: Proposal =
            decode_entry(&retrieved.unwrap()).expect("Failed to decode");
        assert_eq!(retrieved_proposal.id, "MIP-001");

        println!("=== test_create_and_get_proposal PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_active_proposals() {
        println!("=== test_get_active_proposals ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let author_did = format!("did:mycelix:{}", agent);

        // Create multiple proposals
        for i in 0..3 {
            let proposal = make_proposal(
                &format!("MIP-{:03}", i + 10),
                &author_did,
                ProposalType::Standard,
            );

            let _: Record = conductor
                .call(&cell.zome("proposals"), "create_proposal", proposal)
                .await;
        }

        // Get active proposals
        let active_proposals: Vec<Record> = conductor
            .call(&cell.zome("proposals"), "get_active_proposals", ())
            .await;

        assert!(
            active_proposals.len() >= 3,
            "Should have at least 3 active proposals"
        );

        println!("=== test_get_active_proposals PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_proposals_by_author() {
        println!("=== test_get_proposals_by_author ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app1 = conductor
            .setup_app("app-alice", &[dna.clone()])
            .await
            .unwrap();
        let app2 = conductor
            .setup_app("app-bob", &[dna.clone()])
            .await
            .unwrap();
        let alice_cell = app1.cells()[0].clone();
        let bob_cell = app2.cells()[0].clone();
        let alice_agent = app1.agent().clone();
        let bob_agent = app2.agent().clone();

        let alice_did = format!("did:mycelix:{}", alice_agent);
        let bob_did = format!("did:mycelix:{}", bob_agent);

        // Alice creates 2 proposals
        for i in 0..2 {
            let proposal = make_proposal(
                &format!("MIP-A{}", i),
                &alice_did,
                ProposalType::Standard,
            );
            let _: Record = conductor
                .call(
                    &alice_cell.zome("proposals"),
                    "create_proposal",
                    proposal,
                )
                .await;
        }

        // Bob creates 1 proposal
        let bob_proposal = make_proposal("MIP-B0", &bob_did, ProposalType::Standard);
        let _: Record = conductor
            .call(&bob_cell.zome("proposals"), "create_proposal", bob_proposal)
            .await;

        // Get Alice's proposals
        let alice_proposals: Vec<Record> = conductor
            .call(
                &alice_cell.zome("proposals"),
                "get_proposals_by_author",
                alice_did.clone(),
            )
            .await;

        assert!(alice_proposals.len() >= 2, "Alice should have 2+ proposals");

        for record in &alice_proposals {
            let p: Proposal = decode_entry(record).expect("decode");
            assert_eq!(p.author, alice_did, "All should be by Alice");
        }

        println!("=== test_get_proposals_by_author PASSED ===\n");
    }
}

// ============================================================================
// Discussion Tests
// ============================================================================

#[cfg(test)]
mod discussion_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_add_contribution_to_proposal() {
        println!("=== test_add_contribution_to_proposal ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let author_did = format!("did:mycelix:{}", agent);
        let proposal = make_proposal("MIP-D01", &author_did, ProposalType::Standard);

        let _: Record = conductor
            .call(&cell.zome("proposals"), "create_proposal", proposal.clone())
            .await;

        let contribution = make_contribution_input("MIP-D01", &author_did);

        let contrib_record: Record = conductor
            .call(
                &cell.zome("proposals"),
                "add_contribution",
                contribution.clone(),
            )
            .await;

        let created: DiscussionContribution =
            decode_entry(&contrib_record).expect("No entry found");

        assert_eq!(created.proposal_id, "MIP-D01");
        assert_eq!(created.contributor, author_did);

        println!("=== test_add_contribution_to_proposal PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_discussion() {
        println!("=== test_get_discussion ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let author_did = format!("did:mycelix:{}", agent);
        let proposal = make_proposal("MIP-D02", &author_did, ProposalType::Standard);

        let _: Record = conductor
            .call(&cell.zome("proposals"), "create_proposal", proposal)
            .await;

        // Add multiple contributions
        for _i in 0..3 {
            let contribution = make_contribution_input("MIP-D02", &author_did);

            let _: Record = conductor
                .call(&cell.zome("proposals"), "add_contribution", contribution)
                .await;
        }

        let discussion: Vec<Record> = conductor
            .call(
                &cell.zome("proposals"),
                "get_discussion",
                "MIP-D02".to_string(),
            )
            .await;

        assert!(
            discussion.len() >= 3,
            "Should have at least 3 contributions"
        );

        println!("=== test_get_discussion PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_reflect_on_discussion() {
        println!("=== test_reflect_on_discussion ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let author_did = format!("did:mycelix:{}", agent);
        let proposal = make_proposal("MIP-D03", &author_did, ProposalType::Standard);

        let _: Record = conductor
            .call(&cell.zome("proposals"), "create_proposal", proposal)
            .await;

        for _i in 0..2 {
            let contribution = make_contribution_input("MIP-D03", &author_did);
            let _: Record = conductor
                .call(&cell.zome("proposals"), "add_contribution", contribution)
                .await;
        }

        let reflection_record: Record = conductor
            .call(
                &cell.zome("proposals"),
                "reflect_on_discussion",
                "MIP-D03".to_string(),
            )
            .await;

        let reflection: DiscussionReflection =
            decode_entry(&reflection_record).expect("No entry found");

        assert_eq!(reflection.proposal_id, "MIP-D03");
        assert!(!reflection.summary.is_empty(), "Summary should not be empty");

        println!("=== test_reflect_on_discussion PASSED ===\n");
    }
}

// ============================================================================
// Voting Tests
// ============================================================================

#[cfg(test)]
mod voting_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_cast_vote() {
        println!("=== test_cast_vote ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let voter_did = format!("did:mycelix:{}", agent);

        // Create proposal first
        let proposal = make_proposal("MIP-V01", &voter_did, ProposalType::Standard);
        let _: Record = conductor
            .call(&cell.zome("proposals"), "create_proposal", proposal)
            .await;

        // Cast vote
        let vote_input = make_cast_vote_input("MIP-V01", &voter_did, VoteChoice::For);

        let vote_record: Record = conductor
            .call(&cell.zome("voting"), "cast_vote", vote_input)
            .await;

        let created_vote: Vote = decode_entry(&vote_record).expect("No entry found");

        assert_eq!(created_vote.proposal_id, "MIP-V01");
        assert_eq!(created_vote.voter, voter_did);
        assert_eq!(created_vote.choice, VoteChoice::For);

        println!("=== test_cast_vote PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_votes_for_proposal() {
        println!("=== test_get_votes_for_proposal ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app1 = conductor
            .setup_app("app-voter1", &[dna.clone()])
            .await
            .unwrap();
        let app2 = conductor
            .setup_app("app-voter2", &[dna.clone()])
            .await
            .unwrap();
        let app3 = conductor
            .setup_app("app-voter3", &[dna.clone()])
            .await
            .unwrap();
        let cell1 = app1.cells()[0].clone();
        let cell2 = app2.cells()[0].clone();
        let cell3 = app3.cells()[0].clone();
        let agent1 = app1.agent().clone();
        let agent2 = app2.agent().clone();
        let agent3 = app3.agent().clone();

        let did1 = format!("did:mycelix:{}", agent1);
        let did2 = format!("did:mycelix:{}", agent2);
        let did3 = format!("did:mycelix:{}", agent3);

        // Create proposal
        let proposal = make_proposal("MIP-V02", &did1, ProposalType::Standard);
        let _: Record = conductor
            .call(&cell1.zome("proposals"), "create_proposal", proposal)
            .await;

        // Each agent casts a vote
        let choices = [VoteChoice::For, VoteChoice::For, VoteChoice::Against];
        let cells = [&cell1, &cell2, &cell3];
        let dids = [&did1, &did2, &did3];

        for ((cell, did), choice) in cells.iter().zip(dids.iter()).zip(choices.iter()) {
            let vote_input = make_cast_vote_input("MIP-V02", did, choice.clone());
            let _: Record = conductor
                .call(&cell.zome("voting"), "cast_vote", vote_input)
                .await;
        }

        // Get all votes
        let votes: Vec<Record> = conductor
            .call(
                &cell1.zome("voting"),
                "get_proposal_votes",
                "MIP-V02".to_string(),
            )
            .await;

        assert!(votes.len() >= 3, "Should have at least 3 votes");

        println!("=== test_get_votes_for_proposal PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_tally_votes() {
        println!("=== test_tally_votes ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app1 = conductor
            .setup_app("app-t1", &[dna.clone()])
            .await
            .unwrap();
        let app2 = conductor
            .setup_app("app-t2", &[dna.clone()])
            .await
            .unwrap();
        let app3 = conductor
            .setup_app("app-t3", &[dna.clone()])
            .await
            .unwrap();
        let cell1 = app1.cells()[0].clone();
        let cell2 = app2.cells()[0].clone();
        let cell3 = app3.cells()[0].clone();
        let agent1 = app1.agent().clone();
        let agent2 = app2.agent().clone();
        let agent3 = app3.agent().clone();

        let did1 = format!("did:mycelix:{}", agent1);
        let did2 = format!("did:mycelix:{}", agent2);
        let did3 = format!("did:mycelix:{}", agent3);

        // Create proposal
        let proposal = make_proposal("MIP-V03", &did1, ProposalType::Standard);
        let _: Record = conductor
            .call(&cell1.zome("proposals"), "create_proposal", proposal)
            .await;

        // Cast votes: 2 For, 1 Against
        let votes_data = [
            (&cell1, &did1, VoteChoice::For),
            (&cell2, &did2, VoteChoice::For),
            (&cell3, &did3, VoteChoice::Against),
        ];

        for (cell, did, choice) in votes_data.iter() {
            let vote_input = make_cast_vote_input("MIP-V03", did, choice.clone());
            let _: Record = conductor
                .call(&cell.zome("voting"), "cast_vote", vote_input)
                .await;
        }

        // Tally votes
        let tally_record: Record = conductor
            .call(
                &cell1.zome("voting"),
                "tally_votes",
                "MIP-V03".to_string(),
            )
            .await;

        let tally: VoteTally = decode_entry(&tally_record).expect("Failed to decode tally");

        assert_eq!(tally.proposal_id, "MIP-V03");
        assert!(tally.votes_for >= 2.0, "Should have at least 2 For votes");
        assert!(
            tally.votes_against >= 1.0,
            "Should have at least 1 Against vote"
        );

        println!(
            "  Tally: For={}, Against={}, Abstain={}",
            tally.votes_for, tally.votes_against, tally.abstentions
        );
        println!("=== test_tally_votes PASSED ===\n");
    }
}

// ============================================================================
// Integration Flow Tests
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_complete_governance_flow() {
        println!("=== test_complete_governance_flow ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app1 = conductor
            .setup_app("app-flow1", &[dna.clone()])
            .await
            .unwrap();
        let app2 = conductor
            .setup_app("app-flow2", &[dna.clone()])
            .await
            .unwrap();
        let app3 = conductor
            .setup_app("app-flow3", &[dna.clone()])
            .await
            .unwrap();
        let cell1 = app1.cells()[0].clone();
        let cell2 = app2.cells()[0].clone();
        let cell3 = app3.cells()[0].clone();
        let agent1 = app1.agent().clone();
        let agent2 = app2.agent().clone();
        let agent3 = app3.agent().clone();

        let did1 = format!("did:mycelix:{}", agent1);
        let did2 = format!("did:mycelix:{}", agent2);
        let did3 = format!("did:mycelix:{}", agent3);

        // 1. Create proposal
        let proposal = make_proposal("MIP-FLOW", &did1, ProposalType::Standard);
        let _: Record = conductor
            .call(&cell1.zome("proposals"), "create_proposal", proposal)
            .await;
        println!("  1. Proposal created: MIP-FLOW");

        // 2. Add discussion contributions
        for _i in 0..2 {
            let contribution = make_contribution_input("MIP-FLOW", &did1);
            let _: Record = conductor
                .call(&cell1.zome("proposals"), "add_contribution", contribution)
                .await;
        }
        println!("  2. Discussion contributions added");

        // 3. Cast votes from all agents
        let cells = [&cell1, &cell2, &cell3];
        let dids = [&did1, &did2, &did3];
        for (cell, did) in cells.iter().zip(dids.iter()) {
            let vote_input = make_cast_vote_input("MIP-FLOW", did, VoteChoice::For);
            let _: Record = conductor
                .call(&cell.zome("voting"), "cast_vote", vote_input)
                .await;
        }
        println!("  3. Votes cast from all agents");

        // 4. Tally votes
        let tally_record: Record = conductor
            .call(
                &cell1.zome("voting"),
                "tally_votes",
                "MIP-FLOW".to_string(),
            )
            .await;

        let tally: VoteTally = decode_entry(&tally_record).expect("Failed to decode tally");
        assert!(tally.votes_for >= 3.0, "Should have 3 For votes");
        println!("  4. Votes tallied - unanimous approval");

        // 5. Verify final state
        let final_proposal: Option<Record> = conductor
            .call(
                &cell1.zome("proposals"),
                "get_proposal",
                "MIP-FLOW".to_string(),
            )
            .await;

        assert!(final_proposal.is_some(), "Proposal should exist");
        println!("  5. Final verification complete");
        println!("=== test_complete_governance_flow PASSED ===\n");
    }
}

// ============================================================================
// Constitution Tests
// ============================================================================

#[cfg(test)]
mod constitution_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_create_and_get_charter() {
        println!("=== test_create_and_get_charter ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        let charter = make_charter("charter-v1");

        // Create charter
        let charter_record: Record = conductor
            .call(&cell.zome("constitution"), "create_charter", charter.clone())
            .await;

        let created: Charter = decode_entry(&charter_record).expect("Failed to decode charter");
        assert_eq!(created.id, "charter-v1", "Charter ID must match");
        assert_eq!(created.version, 1, "Version should be 1");
        assert!(!created.preamble.is_empty(), "Preamble should not be empty");
        assert_eq!(created.rights.len(), 3, "Should have 3 rights");

        // Get current charter
        let current: Option<Record> = conductor
            .call(&cell.zome("constitution"), "get_current_charter", ())
            .await;

        assert!(current.is_some(), "Current charter should exist");

        let current_charter: Charter =
            decode_entry(&current.unwrap()).expect("Failed to decode");
        assert_eq!(current_charter.id, "charter-v1");

        println!("=== test_create_and_get_charter PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_propose_amendment() {
        println!("=== test_propose_amendment ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();
        let proposer_did = format!("did:mycelix:{}", agent);

        // Create charter first
        let charter = make_charter("charter-amend");
        let _: Record = conductor
            .call(&cell.zome("constitution"), "create_charter", charter)
            .await;

        // Propose amendment
        let amendment_input = ProposeAmendmentInput {
            amendment_type: AmendmentType::ModifyPreamble,
            article: None,
            original_text: Some("We the participants of Mycelix commit to decentralized governance.".to_string()),
            new_text: "We the sovereign participants of Mycelix commit to consciousness-first decentralized governance.".to_string(),
            rationale: "Better reflects our philosophical foundations.".to_string(),
            proposer_did: proposer_did.clone(),
            proposal_id: "MIP-AMEND-01".to_string(),
        };

        let amendment_record: Record = conductor
            .call(&cell.zome("constitution"), "propose_amendment", amendment_input)
            .await;

        let amendment: Amendment =
            decode_entry(&amendment_record).expect("Failed to decode amendment");

        assert!(amendment.id.starts_with("amendment:"), "Amendment ID should have prefix");
        assert_eq!(amendment.proposer, proposer_did, "Proposer must match");
        assert_eq!(amendment.charter_version, 1, "Should amend version 1");

        println!("=== test_propose_amendment PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_set_and_get_parameter() {
        println!("=== test_set_and_get_parameter ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Set a governance parameter
        let param_input = SetParameterInput {
            name: "voting_quorum".to_string(),
            value: "0.33".to_string(),
            value_type: ParameterType::Percentage,
            description: "Minimum participation for valid vote".to_string(),
            min_value: Some("0.1".to_string()),
            max_value: Some("1.0".to_string()),
            proposal_id: None,
        };

        let param_record: Record = conductor
            .call(&cell.zome("constitution"), "set_parameter", param_input)
            .await;

        let param: GovernanceParameter =
            decode_entry(&param_record).expect("Failed to decode parameter");
        assert_eq!(param.name, "voting_quorum");
        assert_eq!(param.value, "0.33");

        // Retrieve parameter
        let retrieved: Option<Record> = conductor
            .call(
                &cell.zome("constitution"),
                "get_parameter",
                "voting_quorum".to_string(),
            )
            .await;

        assert!(retrieved.is_some(), "Parameter should be retrievable");

        println!("=== test_set_and_get_parameter PASSED ===\n");
    }
}

// ============================================================================
// Council Tests
// ============================================================================

#[cfg(test)]
mod council_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_council() {
        println!("=== test_create_council ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        let input = CreateCouncilInput {
            name: "Root Governance Council".to_string(),
            purpose: "Top-level governance decisions for the Mycelix network.".to_string(),
            council_type: CouncilType::Root,
            parent_council_id: None,
            phi_threshold: 0.3,
            quorum: 0.5,
            supermajority: 0.67,
            can_spawn_children: true,
            max_delegation_depth: 3,
        };

        let council_record: Record = conductor
            .call(&cell.zome("councils"), "create_council", input)
            .await;

        let council: Council =
            decode_entry(&council_record).expect("Failed to decode council");

        assert!(council.id.starts_with("council-"), "Council ID should have prefix");
        assert_eq!(council.name, "Root Governance Council");
        assert_eq!(council.status, CouncilStatus::Active);
        assert_eq!(council.phi_threshold, 0.3);
        assert!(council.can_spawn_children);

        println!("=== test_create_council PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_join_council_and_get_members() {
        println!("=== test_join_council_and_get_members ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();
        let member_did = format!("did:mycelix:{}", agent);

        // Create council
        let council_input = CreateCouncilInput {
            name: "Treasury Council".to_string(),
            purpose: "Manage network treasury and funding.".to_string(),
            council_type: CouncilType::Domain { domain: "treasury".to_string() },
            parent_council_id: None,
            phi_threshold: 0.4,
            quorum: 0.5,
            supermajority: 0.67,
            can_spawn_children: false,
            max_delegation_depth: 2,
        };

        let council_record: Record = conductor
            .call(&cell.zome("councils"), "create_council", council_input)
            .await;

        let council: Council =
            decode_entry(&council_record).expect("Failed to decode council");

        // Join council
        let join_input = JoinCouncilInput {
            council_id: council.id.clone(),
            member_did: member_did.clone(),
            role: MemberRole::Member,
            phi_score: 0.7,
        };

        let membership_record: Record = conductor
            .call(&cell.zome("councils"), "join_council", join_input)
            .await;

        let membership: CouncilMembership =
            decode_entry(&membership_record).expect("Failed to decode membership");

        assert_eq!(membership.council_id, council.id);
        assert_eq!(membership.member_did, member_did);
        assert_eq!(membership.role, MemberRole::Member);

        // Get council members
        let members: Vec<Record> = conductor
            .call(
                &cell.zome("councils"),
                "get_council_members",
                council.id.clone(),
            )
            .await;

        assert!(members.len() >= 1, "Should have at least 1 member");

        println!("=== test_join_council_and_get_members PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_all_councils() {
        println!("=== test_get_all_councils ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Create two councils
        for name in ["Technical Council", "Ethics Council"] {
            let input = CreateCouncilInput {
                name: name.to_string(),
                purpose: format!("{} governance.", name),
                council_type: CouncilType::Advisory,
                parent_council_id: None,
                phi_threshold: 0.2,
                quorum: 0.5,
                supermajority: 0.67,
                can_spawn_children: false,
                max_delegation_depth: 1,
            };

            let _: Record = conductor
                .call(&cell.zome("councils"), "create_council", input)
                .await;
        }

        let all_councils: Vec<Record> = conductor
            .call(&cell.zome("councils"), "get_all_councils", ())
            .await;

        assert!(all_councils.len() >= 2, "Should have at least 2 councils");

        println!("=== test_get_all_councils PASSED ===\n");
    }
}

// ============================================================================
// Execution Tests
// ============================================================================

#[cfg(test)]
mod execution_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_timelock() {
        println!("=== test_create_timelock ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        let timelock_input = CreateTimelockInput {
            proposal_id: "MIP-EXEC-01".to_string(),
            actions: serde_json::json!([
                {"type": "UpdateParameter", "parameter": "voting_quorum", "value": "0.4"}
            ]).to_string(),
            duration_hours: 48,
        };

        let timelock_record: Record = conductor
            .call(&cell.zome("execution"), "create_timelock", timelock_input)
            .await;

        let timelock: Timelock =
            decode_entry(&timelock_record).expect("Failed to decode timelock");

        assert!(timelock.id.starts_with("timelock:"), "Timelock ID should have prefix");
        assert_eq!(timelock.proposal_id, "MIP-EXEC-01");
        assert_eq!(timelock.status, TimelockStatus::Pending);
        assert!(timelock.expires > timelock.started, "Expires must be after started");

        println!("=== test_create_timelock PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_proposal_timelock() {
        println!("=== test_get_proposal_timelock ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Create timelock
        let timelock_input = CreateTimelockInput {
            proposal_id: "MIP-EXEC-02".to_string(),
            actions: serde_json::json!([{"type": "EmitEvent", "event": "test"}]).to_string(),
            duration_hours: 24,
        };

        let _: Record = conductor
            .call(&cell.zome("execution"), "create_timelock", timelock_input)
            .await;

        // Retrieve it
        let retrieved: Option<Record> = conductor
            .call(
                &cell.zome("execution"),
                "get_proposal_timelock",
                "MIP-EXEC-02".to_string(),
            )
            .await;

        assert!(retrieved.is_some(), "Timelock should be retrievable");

        let timelock: Timelock =
            decode_entry(&retrieved.unwrap()).expect("Failed to decode");
        assert_eq!(timelock.proposal_id, "MIP-EXEC-02");

        println!("=== test_get_proposal_timelock PASSED ===\n");
    }

    // ========================================================================
    // Phase 5 Tests: Threshold-Signing, Execution Pipeline, Constitution
    // ========================================================================

    #[tokio::test(flavor = "multi_thread")]
    async fn test_create_signing_committee() {
        println!("=== test_create_signing_committee ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        let input = CreateCommitteeInput {
            name: "Treasury Signers".to_string(),
            threshold: 2,
            min_members: 3,
            max_members: 5,
        };

        let record: Record = conductor
            .call(&cell.zome("threshold_signing"), "create_committee", input)
            .await;

        let committee: SigningCommittee =
            decode_entry(&record).expect("Failed to decode committee");
        assert_eq!(committee.name, "Treasury Signers");
        assert_eq!(committee.threshold, 2);
        assert_eq!(committee.min_members, 3);
        assert_eq!(committee.max_members, 5);
        assert_eq!(committee.phase, DkgPhase::Registration);
        assert_eq!(committee.epoch, 1);
        assert!(committee.public_key.is_none());

        println!("=== test_create_signing_committee PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_register_committee_member() {
        println!("=== test_register_committee_member ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Create committee first
        let committee_input = CreateCommitteeInput {
            name: "Test Signers".to_string(),
            threshold: 2,
            min_members: 2,
            max_members: 5,
        };
        let committee_record: Record = conductor
            .call(&cell.zome("threshold_signing"), "create_committee", committee_input)
            .await;
        let committee: SigningCommittee = decode_entry(&committee_record).unwrap();

        // Register a member
        let member_input = RegisterMemberInput {
            committee_id: committee.id.clone(),
            member_did: format!("did:mycelix:{}", app.agent_pubkey()),
            trust_score: 0.85,
        };
        let member_record: Record = conductor
            .call(&cell.zome("threshold_signing"), "register_member", member_input)
            .await;

        let member: CommitteeMember = decode_entry(&member_record).expect("decode member");
        assert_eq!(member.committee_id, committee.id);
        assert!(!member.deal_submitted);
        assert!(!member.qualified);
        assert_eq!(member.trust_score, 0.85);

        println!("=== test_register_committee_member PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_mark_timelock_ready() {
        println!("=== test_mark_timelock_ready ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Create a timelock
        let create_input = CreateTimelockInput {
            proposal_id: "MIP-READY-01".to_string(),
            actions: serde_json::json!([{"type": "EmitEvent", "event": "test"}]).to_string(),
            duration_hours: 1,
        };
        let _: Record = conductor
            .call(&cell.zome("execution"), "create_timelock", create_input)
            .await;

        // Get it to find the ID
        let retrieved: Option<Record> = conductor
            .call(
                &cell.zome("execution"),
                "get_proposal_timelock",
                "MIP-READY-01".to_string(),
            )
            .await;
        let timelock: Timelock = decode_entry(&retrieved.unwrap()).unwrap();
        assert_eq!(timelock.status, TimelockStatus::Pending);

        // Mark as ready
        let ready_input = MarkTimelockReadyInput {
            timelock_id: timelock.id.clone(),
        };
        let ready_record: Record = conductor
            .call(&cell.zome("execution"), "mark_timelock_ready", ready_input)
            .await;

        let ready_timelock: Timelock = decode_entry(&ready_record).expect("decode ready timelock");
        assert_eq!(ready_timelock.status, TimelockStatus::Ready);
        assert_eq!(ready_timelock.proposal_id, "MIP-READY-01");

        println!("=== test_mark_timelock_ready PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_update_parameter_preserves_type() {
        println!("=== test_update_parameter_preserves_type ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Set a parameter with Float type
        let set_input = SetParameterInput {
            name: "quorum_threshold".to_string(),
            value: "\"0.5\"".to_string(),
            value_type: ParameterType::Float,
            description: "Minimum quorum for standard proposals".to_string(),
            min_value: Some("\"0.1\"".to_string()),
            max_value: Some("\"1.0\"".to_string()),
            proposal_id: None,
        };
        let _: Record = conductor
            .call(&cell.zome("constitution"), "set_parameter", set_input)
            .await;

        // Update the value via update_parameter (which should preserve Float type)
        let update_input = UpdateParameterInput {
            parameter: "quorum_threshold".to_string(),
            value: "\"0.6\"".to_string(),
            proposal_id: Some("MIP-PARAM-01".to_string()),
        };
        let updated_record: Record = conductor
            .call(&cell.zome("constitution"), "update_parameter", update_input)
            .await;

        let param: GovernanceParameter = decode_entry(&updated_record).expect("decode param");
        assert_eq!(param.name, "quorum_threshold");
        assert_eq!(param.value, "\"0.6\"");
        // Key assertion: type should be preserved as Float, not reset to String
        assert_eq!(param.value_type, ParameterType::Float);

        println!("=== test_update_parameter_preserves_type PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_proposal_status_requires_draft() {
        println!("=== test_proposal_status_requires_draft ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Create a proposal (must start as Draft per Phase 4 integrity validation)
        let author_did = format!("did:mycelix:{}", app.agent_pubkey());
        let proposal = make_proposal("MIP-DRAFT-01", ProposalType::Standard, &author_did);
        assert_eq!(proposal.status, ProposalStatus::Draft, "Helper should create Draft proposals");

        let record: Record = conductor
            .call(&cell.zome("proposals"), "create_proposal", proposal)
            .await;

        let created: Proposal = decode_entry(&record).expect("decode proposal");
        assert_eq!(created.status, ProposalStatus::Draft);

        println!("=== test_proposal_status_requires_draft PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_all_committees() {
        println!("=== test_get_all_committees ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-governance", &[dna.clone()])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Create two committees
        for name in ["Committee A", "Committee B"] {
            let input = CreateCommitteeInput {
                name: name.to_string(),
                threshold: 2,
                min_members: 3,
                max_members: 5,
            };
            let _: Record = conductor
                .call(&cell.zome("threshold_signing"), "create_committee", input)
                .await;
        }

        let all: Vec<Record> = conductor
            .call(&cell.zome("threshold_signing"), "get_all_committees", ())
            .await;

        assert!(all.len() >= 2, "Should have at least 2 committees");

        println!("=== test_get_all_committees PASSED ===\n");
    }
}

// ============================================================================
// Unit Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_proposal_type_serialization() {
        let types = vec![
            ProposalType::Standard,
            ProposalType::Emergency,
            ProposalType::Constitutional,
            ProposalType::Parameter,
            ProposalType::Funding,
        ];

        for pt in types {
            let json = serde_json::to_string(&pt).expect("serialize");
            let deserialized: ProposalType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(pt, deserialized);
        }
    }

    #[test]
    fn test_vote_choice_serialization() {
        let choices = vec![VoteChoice::For, VoteChoice::Against, VoteChoice::Abstain];

        for choice in choices {
            let json = serde_json::to_string(&choice).expect("serialize");
            let deserialized: VoteChoice = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(choice, deserialized);
        }
    }

    #[test]
    fn test_council_type_serialization() {
        let types = vec![
            CouncilType::Root,
            CouncilType::Domain {
                domain: "treasury".to_string(),
            },
            CouncilType::Advisory,
        ];

        for ct in types {
            let json = serde_json::to_string(&ct).expect("serialize");
            let deserialized: CouncilType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(ct, deserialized);
        }
    }

    #[test]
    fn test_stance_serialization() {
        let stances = vec![Stance::Support, Stance::Oppose, Stance::Neutral, Stance::Amend];

        for stance in stances {
            let json = serde_json::to_string(&stance).expect("serialize");
            let deserialized: Stance = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(stance, deserialized);
        }
    }

    #[test]
    fn test_dkg_phase_serialization() {
        let phases = vec![
            DkgPhase::Registration,
            DkgPhase::Dealing,
            DkgPhase::Verification,
            DkgPhase::Complete,
        ];

        for phase in phases {
            let json = serde_json::to_string(&phase).expect("serialize");
            let deserialized: DkgPhase = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(phase, deserialized);
        }
    }

    #[test]
    fn test_timelock_status_includes_ready() {
        let statuses = vec![
            TimelockStatus::Pending,
            TimelockStatus::Ready,
            TimelockStatus::Executed,
            TimelockStatus::Cancelled,
            TimelockStatus::Failed,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("serialize");
            let deserialized: TimelockStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(status, deserialized);
        }
    }

    #[test]
    fn test_proposal_status_includes_signed() {
        let statuses = vec![
            ProposalStatus::Draft,
            ProposalStatus::Active,
            ProposalStatus::Ended,
            ProposalStatus::Approved,
            ProposalStatus::Signed,
            ProposalStatus::Rejected,
            ProposalStatus::Executed,
            ProposalStatus::Cancelled,
            ProposalStatus::Failed,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("serialize");
            let deserialized: ProposalStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(status, deserialized);
        }
    }

    #[test]
    fn test_execution_status_partial_success() {
        let statuses = vec![
            ExecutionStatus::Success,
            ExecutionStatus::Failed,
            ExecutionStatus::PartialSuccess,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).expect("serialize");
            let deserialized: ExecutionStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(status, deserialized);
        }
    }

    #[test]
    fn test_amendment_type_all_8_variants() {
        let types = vec![
            AmendmentType::AddArticle,
            AmendmentType::ModifyArticle,
            AmendmentType::RemoveArticle,
            AmendmentType::AddRight,
            AmendmentType::ModifyRight,
            AmendmentType::RemoveRight,
            AmendmentType::ModifyPreamble,
            AmendmentType::ModifyProcess,
        ];

        assert_eq!(types.len(), 8, "Must have all 8 amendment types");
        for at in types {
            let json = serde_json::to_string(&at).expect("serialize");
            let deserialized: AmendmentType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(at, deserialized);
        }
    }

    #[test]
    fn test_constitution_amendment_status_all_variants() {
        let statuses = vec![
            ConstitutionAmendmentStatus::Draft,
            ConstitutionAmendmentStatus::Deliberation,
            ConstitutionAmendmentStatus::Voting,
            ConstitutionAmendmentStatus::Ratified,
            ConstitutionAmendmentStatus::Rejected,
            ConstitutionAmendmentStatus::Withdrawn,
        ];

        assert_eq!(statuses.len(), 6, "Must have all 6 amendment status variants");
        for status in statuses {
            let json = serde_json::to_string(&status).expect("serialize");
            let deserialized: ConstitutionAmendmentStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(status, deserialized);
        }
    }

    #[test]
    fn test_create_committee_input_serialization() {
        let input = CreateCommitteeInput {
            name: "Test Committee".to_string(),
            threshold: 3,
            min_members: 5,
            max_members: 10,
        };

        let json = serde_json::to_string(&input).expect("serialize");
        let deserialized: CreateCommitteeInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.name, "Test Committee");
        assert_eq!(deserialized.threshold, 3);
        assert_eq!(deserialized.min_members, 5);
        assert_eq!(deserialized.max_members, 10);
    }
}
