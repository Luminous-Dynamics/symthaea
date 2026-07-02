// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Energy Bridge Integrity Zome
//!
//! Entry types for Terra Atlas integration, investment tracking,
//! and regenerative exit coordination.

use hdi::prelude::*;
use mycelix_bridge_entry_types::CrossClusterNotification;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Energy project query from external systems (Terra Atlas)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProjectQuery {
    pub id: String,
    pub project_id: String,
    pub source: String, // "terra-atlas" or hApp ID
    pub query_type: ProjectQueryType,
    pub queried_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ProjectQueryType {
    ProjectDetails,
    InvestmentStatus,
    CommunityReadiness,
    RegenerativeProgress,
}

/// Energy project reference from Terra Atlas
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TerraAtlasProject {
    pub id: String,
    pub terra_atlas_id: String,
    pub name: String,
    pub project_type: EnergyType,
    pub location: GeoLocation,
    pub capacity_mw: f64,
    pub total_investment: u64,
    pub current_investment: u64,
    pub status: ProjectStatus,
    pub regenerative_progress: f64, // 0.0 - 1.0
    pub synced_at: Timestamp,
    /// Symthaea Phi consciousness score (0.0–1.0). None if not yet assessed.
    pub phi_score: Option<f64>,
    /// Eight Harmonies alignment (0.0–1.0). None if not yet assessed.
    pub harmony_alignment: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EnergyType {
    Solar,
    Wind,
    Hydro,
    Geothermal,
    Nuclear,
    Storage,
    Mixed,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub region: String,
    pub country: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ProjectStatus {
    Discovery,
    Funding,
    Development,
    Operational,
    Transitioning,
    CommunityOwned,
}

/// Investment record for cross-hApp tracking
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct InvestmentRecord {
    pub id: String,
    pub project_id: String,
    pub investor_did: String,
    pub amount: u64,
    pub currency: String,
    pub source_happ: String, // e.g., "mycelix-finance"
    pub investment_type: InvestmentType,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum InvestmentType {
    Equity,
    Loan,
    Grant,
    CommunityShare,
}

/// Regenerative exit milestone
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RegenerativeMilestone {
    pub id: String,
    pub project_id: String,
    pub milestone_type: MilestoneType,
    pub community_readiness: f64,
    pub operator_certification: bool,
    pub financial_sustainability: f64,
    pub achieved_at: Option<Timestamp>,
    pub verified_by: Option<String>, // DID of verifier
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MilestoneType {
    CommunityFormation,
    OperatorTraining,
    FinancialIndependence,
    GovernanceEstablished,
    FullTransition,
}

/// Energy event for broadcasting
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EnergyBridgeEvent {
    pub id: String,
    pub event_type: EnergyEventType,
    pub project_id: String,
    pub payload: String,
    pub source: String,
    pub timestamp: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EnergyEventType {
    ProjectDiscovered,
    InvestmentReceived,
    MilestoneAchieved,
    TransitionInitiated,
    CommunityOwnershipComplete,
    ProductionUpdate,
    StatusChanged,
    SyncPending,
    CertificateCollateralRegistered,
    ConsciousnessAssessed,
    PledgeSubmitted,
    PledgeWithdrawn,
    MatchProposed,
    MatchAccepted,
    MatchRejected,
    ImpactReported,
}

/// Production metrics for energy projects
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProductionRecord {
    pub id: String,
    pub project_id: String,
    pub terra_atlas_id: String,
    pub period_start: Timestamp,
    pub period_end: Timestamp,
    pub energy_generated_mwh: f64,
    pub capacity_factor: f64,        // 0.0-1.0 (actual vs theoretical max)
    pub revenue_generated: u64,      // in smallest currency unit
    pub currency: String,
    pub grid_injection_mwh: f64,     // energy sold to grid
    pub self_consumption_mwh: f64,   // energy used locally
    pub recorded_at: Timestamp,
    pub verified_by: Option<String>, // DID of verifier (e.g., grid operator)
}

/// Consciousness assessment from Symthaea for a project
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConsciousnessAssessment {
    pub id: String,
    pub project_id: String,              // Links to TerraAtlasProject
    pub scorer_did: String,              // Symthaea instance DID
    pub phi_score: f64,                  // Φ integrated information [0,1]
    pub harmony_alignment: f64,          // Eight Harmonies composite [0,1]
    pub per_harmony_scores: String,      // JSON HashMap<String, f64>
    pub care_activation: f64,            // CARE system activation [0,1]
    pub meta_awareness: f64,             // Meta-awareness level [0,1]
    pub assessment_cycle: u64,           // Symthaea cycle number
    pub assessed_at: Timestamp,
}

/// Allocation pledge — a trust-weighted commitment of resources toward a project
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AllocationPledge {
    pub id: String,
    pub pledger_did: String,
    pub project_id: String,
    pub amount: u64,                        // In smallest currency unit
    pub currency: String,                   // "TEND" | "SAP" | community currency ID
    pub trust_score: f64,                   // Pledger's trust profile combined_score [0,1]
    pub trust_tier: String,                 // Tier name for audit trail
    pub harmony_intent: String,             // Which harmony does this serve
    pub status: PledgeStatus,
    pub pledged_at: Timestamp,
    pub expires_at: Timestamp,              // 24h default (matches credential TTL)
    pub matched_at: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PledgeStatus {
    Pending,
    Matched,
    Fulfilled,
    Expired,
    Withdrawn,
}

/// Allocation match — a confirmed bilateral match between a pledge and an asset
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AllocationMatch {
    pub id: String,
    pub pledge_id: String,
    pub project_id: String,
    pub pledger_did: String,
    pub holder_did: String,
    pub amount: u64,
    pub match_score: f64,               // Composite trust-weighted score [0,1]
    pub trust_weight: f64,              // Pledger's trust contribution to score
    pub amount_fit: f64,                // How well pledge fills funding gap [0,1]
    pub harmony_alignment: f64,         // Harmony evaluation of pledge intent [0,1]
    pub community_proximity: f64,       // Geographic/community closeness [0,1]
    pub status: MatchStatus,
    pub proposed_at: Timestamp,
    pub resolved_at: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MatchStatus {
    Proposed,
    Accepted,
    Rejected,
    Completed,
}

/// QOL impact record — real-world impact data for a project
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ImpactRecord {
    pub id: String,
    pub project_id: String,
    pub reporter_did: String,
    pub period_start: Timestamp,
    pub period_end: Timestamp,
    pub co2_avoided_tonnes: f64,
    pub jobs_created: u32,
    pub community_trust_delta: f64,         // [-1, +1]
    pub energy_access_households: u32,
    pub biodiversity_index_delta: f64,
    pub verification_evidence: Option<String>,  // IPFS hash or GPS-stamped proof
    pub verified_by: Option<String>,            // Verifier DID
    pub recorded_at: Timestamp,
}

/// Reputation feedback after a match completes — both parties rate the outcome
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MatchFeedback {
    pub id: String,
    pub match_id: String,
    pub project_id: String,
    pub rater_did: String,
    pub rated_did: String,
    pub role: FeedbackRole,
    pub outcome_score: f64,          // Did the match deliver? [0,1]
    pub harmony_fulfilled: f64,      // Was harmony intent honored? [0,1]
    pub would_match_again: bool,
    pub comment: Option<String>,
    pub rated_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FeedbackRole {
    Pledger,
    Holder,
}

/// Performance bond — community-backed guarantee for project delivery
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PerformanceBond {
    pub id: String,
    pub project_id: String,
    pub guarantor_did: String,
    pub amount: u64,
    pub currency: String,
    pub conditions: Vec<String>,
    pub status: BondStatus,
    pub staked_at: Timestamp,
    pub expires_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BondStatus {
    Active,
    ClaimFiled,
    Released,
    Forfeited,
}

/// Mutual insurance pool — participants share risk across projects
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct InsurancePool {
    pub pool_id: String,
    pub pool_type: PoolType,
    pub total_capital: u64,
    pub claim_count: u32,
    pub max_payout_per_claim: u64,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PoolType {
    ProjectCompletion,
    GenerationShortfall,
    EquipmentFailure,
}

/// Risk assessment — trust-gated evaluation feeding project scores
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RiskAssessment {
    pub id: String,
    pub project_id: String,
    pub assessor_did: String,
    pub overall_risk: f64,
    pub technology_risk: f64,
    pub regulatory_risk: f64,
    pub market_risk: f64,
    pub construction_risk: f64,
    pub methodology: String,
    pub assessed_at: Timestamp,
}

/// Regulatory permit tracking
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Permit {
    pub permit_id: String,
    pub project_id: String,
    pub permit_type: PermitType,
    pub authority: String,
    pub status: PermitStatus,
    pub applied_at: Timestamp,
    pub decision_at: Option<Timestamp>,
    pub expires_at: Option<Timestamp>,
    pub conditions: Vec<String>,
    pub evidence_cid: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PermitType {
    FERC,
    NRC,
    StateUtility,
    LocalZoning,
    Environmental,
    BuildingCode,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PermitStatus {
    Applied,
    UnderReview,
    Approved,
    Denied,
    Expired,
    Revoked,
}

/// Compliance verification record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ComplianceCheck {
    pub id: String,
    pub project_id: String,
    pub checker_did: String,
    pub rule_id: String,
    pub rule_description: String,
    pub compliant: bool,
    pub evidence: String,
    pub checked_at: Timestamp,
}

/// Immutable audit entry (BLAKE3-signed)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AuditEntry {
    pub id: String,
    pub project_id: String,
    pub action: String,
    pub actor_did: String,
    pub details: String,
    pub timestamp: Timestamp,
    pub signature_hash: String,
}

/// Construction task with dependency tracking
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConstructionTask {
    pub task_id: String,
    pub project_id: String,
    pub name: String,
    pub description: String,
    pub predecessors: Vec<String>,
    pub duration_days: u32,
    pub assigned_to: Option<String>,
    pub status: TaskStatus,
    pub planned_start: Timestamp,
    pub planned_end: Timestamp,
    pub actual_start: Option<Timestamp>,
    pub actual_end: Option<Timestamp>,
    pub budget: u64,
    pub actual_cost: u64,
    pub percent_complete: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TaskStatus {
    NotStarted,
    InProgress,
    Completed,
    Blocked,
}

/// Change order for construction scope modifications
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ChangeOrder {
    pub change_id: String,
    pub project_id: String,
    pub description: String,
    pub cost_impact: i64,
    pub schedule_impact_days: i32,
    pub requested_by: String,
    pub approved_by: Option<String>,
    pub status: ChangeStatus,
    pub requested_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ChangeStatus {
    Requested,
    Approved,
    Rejected,
    Implemented,
}

/// Pending sync record for bidirectional sync to Terra Atlas
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PendingSyncRecord {
    pub id: String,
    pub sync_type: SyncType,
    pub target_system: String,       // "terra-atlas"
    pub payload: String,             // JSON payload for external sync service
    pub created_at: Timestamp,
    pub synced_at: Option<Timestamp>,
    pub retry_count: u32,
    pub last_error: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SyncType {
    InvestmentUpdate,
    ProductionMetrics,
    MilestoneProgress,
    StatusChange,
    TransitionProgress,
    ConsciousnessScore,
    ImpactMetrics,
}

/// Regenerative transition tracking
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TransitionRecord {
    pub id: String,
    pub project_id: String,
    pub terra_atlas_id: String,
    pub from_ownership_pct: f64,
    pub to_ownership_pct: f64,
    pub community_did: String,
    pub reserve_account_balance: u64,
    pub conditions_met: Vec<String>,
    pub conditions_pending: Vec<String>,
    pub status: TransitionStatus,
    pub initiated_at: Timestamp,
    pub completed_at: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransitionStatus {
    Proposed,
    ConditionsReview,
    InProgress,
    Completed,
    Cancelled,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    ProjectQuery(ProjectQuery),
    TerraAtlasProject(TerraAtlasProject),
    InvestmentRecord(InvestmentRecord),
    RegenerativeMilestone(RegenerativeMilestone),
    EnergyBridgeEvent(EnergyBridgeEvent),
    ProductionRecord(ProductionRecord),
    PendingSyncRecord(PendingSyncRecord),
    TransitionRecord(TransitionRecord),
    ConsciousnessAssessment(ConsciousnessAssessment),
    AllocationPledge(AllocationPledge),
    AllocationMatch(AllocationMatch),
    ImpactRecord(ImpactRecord),
    MatchFeedback(MatchFeedback),
    Permit(Permit),
    ComplianceCheck(ComplianceCheck),
    AuditEntry(AuditEntry),
    ConstructionTask(ConstructionTask),
    ChangeOrder(ChangeOrder),
    PerformanceBond(PerformanceBond),
    InsurancePool(InsurancePool),
    RiskAssessment(RiskAssessment),
    Notification(CrossClusterNotification),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllProjects,
    ProjectToInvestments,
    DidToInvestments,
    ProjectToMilestones,
    RecentEvents,
    ByEnergyType,
    ProjectToProduction,
    PendingSyncs,
    ProjectToTransitions,
    ActiveTransitions,
    ProjectToAssessments,
    AssetToPledges,
    PledgerToPledges,
    AssetToMatches,
    PledgeToMatch,
    AssetToImpacts,
    MatchToFeedback,
    ProjectToPermits,
    ProjectToCompliance,
    ProjectToAudit,
    ProjectToTasks,
    ProjectToChangeOrders,
    ProjectToBonds,
    PoolToMembers,
    ProjectToRiskAssessments,
    AgentToNotification,
    AllNotifications,
    NotificationSubscription,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. })
        | FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => {
            validate_entry(&app_entry)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_entry(entry: &EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::ProjectQuery(query) => {
            if query.project_id.is_empty() && query.source.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Project ID or source required".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::TerraAtlasProject(project) => {
            if project.capacity_mw <= 0.0 {
                return Ok(ValidateCallbackResult::Invalid("Capacity must be positive".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::InvestmentRecord(record) => {
            if !record.investor_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Investor must have valid DID".into()));
            }
            if record.amount == 0 {
                return Ok(ValidateCallbackResult::Invalid("Amount must be positive".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::RegenerativeMilestone(milestone) => {
            if milestone.community_readiness < 0.0 || milestone.community_readiness > 1.0 {
                return Ok(ValidateCallbackResult::Invalid("Community readiness must be 0.0-1.0".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::ProductionRecord(record) => {
            if record.energy_generated_mwh < 0.0 {
                return Ok(ValidateCallbackResult::Invalid("Energy generated must be non-negative".into()));
            }
            if record.capacity_factor < 0.0 || record.capacity_factor > 1.0 {
                return Ok(ValidateCallbackResult::Invalid("Capacity factor must be 0.0-1.0".into()));
            }
            if record.grid_injection_mwh + record.self_consumption_mwh > record.energy_generated_mwh * 1.01 {
                return Ok(ValidateCallbackResult::Invalid("Grid + self consumption cannot exceed generated".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::PendingSyncRecord(record) => {
            if record.target_system.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Target system required".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::TransitionRecord(record) => {
            if record.to_ownership_pct < record.from_ownership_pct {
                return Ok(ValidateCallbackResult::Invalid("Transition must increase community ownership".into()));
            }
            if record.to_ownership_pct > 100.0 {
                return Ok(ValidateCallbackResult::Invalid("Ownership cannot exceed 100%".into()));
            }
            if !record.community_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Community must have valid DID".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::PerformanceBond(bond) => {
            if !bond.guarantor_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Guarantor must have valid DID".into()));
            }
            if bond.amount == 0 {
                return Ok(ValidateCallbackResult::Invalid("Bond amount must be positive".into()));
            }
            if bond.conditions.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Bond must have at least one condition".into()));
            }
            if bond.expires_at <= bond.staked_at {
                return Ok(ValidateCallbackResult::Invalid("Expiry must be after stake time".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::InsurancePool(pool) => {
            if pool.pool_id.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Pool ID required".into()));
            }
            if pool.max_payout_per_claim == 0 {
                return Ok(ValidateCallbackResult::Invalid("Max payout must be positive".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::RiskAssessment(risk) => {
            if !risk.assessor_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Assessor must have valid DID".into()));
            }
            if risk.project_id.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Project ID required".into()));
            }
            for score in [risk.overall_risk, risk.technology_risk, risk.regulatory_risk, risk.market_risk, risk.construction_risk] {
                if !score.is_finite() || score < 0.0 || score > 1.0 {
                    return Ok(ValidateCallbackResult::Invalid("Risk scores must be 0.0-1.0".into()));
                }
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::Permit(p) => {
            if p.project_id.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Project ID required".into()));
            }
            if p.authority.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Authority required".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::ComplianceCheck(c) => {
            if !c.checker_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Checker must have valid DID".into()));
            }
            if c.project_id.is_empty() || c.rule_id.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Project ID and rule ID required".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::AuditEntry(a) => {
            if !a.actor_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Actor must have valid DID".into()));
            }
            if a.project_id.is_empty() || a.action.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Project ID and action required".into()));
            }
            if a.signature_hash.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Signature hash required".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::ConstructionTask(t) => {
            if t.project_id.is_empty() || t.name.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Project ID and task name required".into()));
            }
            if t.duration_days == 0 {
                return Ok(ValidateCallbackResult::Invalid("Duration must be positive".into()));
            }
            if !t.percent_complete.is_finite() || t.percent_complete < 0.0 || t.percent_complete > 1.0 {
                return Ok(ValidateCallbackResult::Invalid("Percent complete must be 0.0-1.0".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::ChangeOrder(c) => {
            if c.project_id.is_empty() || c.description.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Project ID and description required".into()));
            }
            if !c.requested_by.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Requester must have valid DID".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::MatchFeedback(fb) => {
            if !fb.rater_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Rater must have valid DID".into()));
            }
            if !fb.rated_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Rated party must have valid DID".into()));
            }
            if fb.rater_did == fb.rated_did {
                return Ok(ValidateCallbackResult::Invalid("Cannot rate yourself".into()));
            }
            if fb.match_id.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Match ID required".into()));
            }
            if !fb.outcome_score.is_finite() || fb.outcome_score < 0.0 || fb.outcome_score > 1.0 {
                return Ok(ValidateCallbackResult::Invalid("Outcome score must be 0.0-1.0".into()));
            }
            if !fb.harmony_fulfilled.is_finite() || fb.harmony_fulfilled < 0.0 || fb.harmony_fulfilled > 1.0 {
                return Ok(ValidateCallbackResult::Invalid("Harmony fulfilled must be 0.0-1.0".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::AllocationMatch(m) => {
            if m.pledge_id.is_empty() || m.project_id.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Pledge ID and project ID required".into()));
            }
            if !m.pledger_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Pledger must have valid DID".into()));
            }
            if !m.holder_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Holder must have valid DID".into()));
            }
            if m.amount == 0 {
                return Ok(ValidateCallbackResult::Invalid("Match amount must be positive".into()));
            }
            if !m.match_score.is_finite() || m.match_score < 0.0 || m.match_score > 1.0 {
                return Ok(ValidateCallbackResult::Invalid("Match score must be 0.0-1.0".into()));
            }
            if !m.trust_weight.is_finite() || m.trust_weight < 0.0 || m.trust_weight > 1.0 {
                return Ok(ValidateCallbackResult::Invalid("Trust weight must be 0.0-1.0".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::ImpactRecord(impact) => {
            if !impact.reporter_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Reporter must have valid DID".into()));
            }
            if impact.project_id.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Project ID required".into()));
            }
            if !impact.co2_avoided_tonnes.is_finite() || impact.co2_avoided_tonnes < 0.0 {
                return Ok(ValidateCallbackResult::Invalid("CO2 avoided must be non-negative".into()));
            }
            if !impact.community_trust_delta.is_finite() || impact.community_trust_delta < -1.0 || impact.community_trust_delta > 1.0 {
                return Ok(ValidateCallbackResult::Invalid("Community trust delta must be -1.0 to 1.0".into()));
            }
            if !impact.biodiversity_index_delta.is_finite() {
                return Ok(ValidateCallbackResult::Invalid("Biodiversity index must be finite".into()));
            }
            if impact.period_end <= impact.period_start {
                return Ok(ValidateCallbackResult::Invalid("Period end must be after period start".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::AllocationPledge(pledge) => {
            if !pledge.pledger_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Pledger must have valid DID".into()));
            }
            if pledge.project_id.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Project ID required".into()));
            }
            if pledge.amount == 0 {
                return Ok(ValidateCallbackResult::Invalid("Pledge amount must be positive".into()));
            }
            if pledge.currency.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Currency required".into()));
            }
            if !pledge.trust_score.is_finite() || pledge.trust_score < 0.0 || pledge.trust_score > 1.0 {
                return Ok(ValidateCallbackResult::Invalid("Trust score must be 0.0-1.0".into()));
            }
            if pledge.harmony_intent.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Harmony intent required".into()));
            }
            if pledge.expires_at <= pledge.pledged_at {
                return Ok(ValidateCallbackResult::Invalid("Expiry must be after pledge time".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::ConsciousnessAssessment(assessment) => {
            if !assessment.scorer_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("Scorer must have valid DID".into()));
            }
            if assessment.project_id.is_empty() {
                return Ok(ValidateCallbackResult::Invalid("Project ID required".into()));
            }
            if !assessment.phi_score.is_finite() || assessment.phi_score < 0.0 || assessment.phi_score > 1.0 {
                return Ok(ValidateCallbackResult::Invalid("Phi score must be 0.0-1.0".into()));
            }
            if !assessment.harmony_alignment.is_finite() || assessment.harmony_alignment < 0.0 || assessment.harmony_alignment > 1.0 {
                return Ok(ValidateCallbackResult::Invalid("Harmony alignment must be 0.0-1.0".into()));
            }
            if !assessment.care_activation.is_finite() || assessment.care_activation < 0.0 || assessment.care_activation > 1.0 {
                return Ok(ValidateCallbackResult::Invalid("Care activation must be 0.0-1.0".into()));
            }
            if !assessment.meta_awareness.is_finite() || assessment.meta_awareness < 0.0 || assessment.meta_awareness > 1.0 {
                return Ok(ValidateCallbackResult::Invalid("Meta awareness must be 0.0-1.0".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::Notification(n) => {
            mycelix_bridge_entry_types::validate_notification(&n)
                .map(|()| ValidateCallbackResult::Valid)
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // =========================================================================
    // Test Helpers
    // =========================================================================

    fn create_test_timestamp() -> Timestamp {
        Timestamp::from_micros(1704067200000000) // 2024-01-01 00:00:00 UTC
    }

    fn valid_geo_location() -> GeoLocation {
        GeoLocation {
            latitude: 37.7749,
            longitude: -122.4194,
            region: "California".to_string(),
            country: "USA".to_string(),
        }
    }

    // =========================================================================
    // EnergyType Enum Tests
    // =========================================================================

    #[test]
    fn test_energy_type_variants() {
        let types = vec![
            EnergyType::Solar,
            EnergyType::Wind,
            EnergyType::Hydro,
            EnergyType::Geothermal,
            EnergyType::Nuclear,
            EnergyType::Storage,
            EnergyType::Mixed,
        ];
        assert_eq!(types.len(), 7);
    }

    #[test]
    fn test_energy_type_equality() {
        assert_eq!(EnergyType::Solar, EnergyType::Solar);
        assert_ne!(EnergyType::Solar, EnergyType::Wind);
    }

    // =========================================================================
    // ProjectStatus Enum Tests
    // =========================================================================

    #[test]
    fn test_project_status_variants() {
        let statuses = vec![
            ProjectStatus::Discovery,
            ProjectStatus::Funding,
            ProjectStatus::Development,
            ProjectStatus::Operational,
            ProjectStatus::Transitioning,
            ProjectStatus::CommunityOwned,
        ];
        assert_eq!(statuses.len(), 6);
    }

    #[test]
    fn test_project_status_lifecycle() {
        let lifecycle = [
            ProjectStatus::Discovery,
            ProjectStatus::Funding,
            ProjectStatus::Development,
            ProjectStatus::Operational,
            ProjectStatus::Transitioning,
            ProjectStatus::CommunityOwned,
        ];
        for i in 0..lifecycle.len() - 1 {
            assert_ne!(lifecycle[i], lifecycle[i + 1]);
        }
    }

    // =========================================================================
    // InvestmentType Enum Tests
    // =========================================================================

    #[test]
    fn test_investment_type_variants() {
        let types = vec![
            InvestmentType::Equity,
            InvestmentType::Loan,
            InvestmentType::Grant,
            InvestmentType::CommunityShare,
        ];
        assert_eq!(types.len(), 4);
    }

    // =========================================================================
    // MilestoneType Enum Tests
    // =========================================================================

    #[test]
    fn test_milestone_type_variants() {
        let types = vec![
            MilestoneType::CommunityFormation,
            MilestoneType::OperatorTraining,
            MilestoneType::FinancialIndependence,
            MilestoneType::GovernanceEstablished,
            MilestoneType::FullTransition,
        ];
        assert_eq!(types.len(), 5);
    }

    // =========================================================================
    // EnergyEventType Enum Tests
    // =========================================================================

    #[test]
    fn test_energy_event_type_variants() {
        let types = vec![
            EnergyEventType::ProjectDiscovered,
            EnergyEventType::InvestmentReceived,
            EnergyEventType::MilestoneAchieved,
            EnergyEventType::TransitionInitiated,
            EnergyEventType::CommunityOwnershipComplete,
            EnergyEventType::ProductionUpdate,
            EnergyEventType::StatusChanged,
            EnergyEventType::SyncPending,
            EnergyEventType::CertificateCollateralRegistered,
        ];
        assert_eq!(types.len(), 9);
    }

    // =========================================================================
    // SyncType Enum Tests
    // =========================================================================

    #[test]
    fn test_sync_type_variants() {
        let types = vec![
            SyncType::InvestmentUpdate,
            SyncType::ProductionMetrics,
            SyncType::MilestoneProgress,
            SyncType::StatusChange,
            SyncType::TransitionProgress,
        ];
        assert_eq!(types.len(), 5);
    }

    // =========================================================================
    // TransitionStatus Enum Tests
    // =========================================================================

    #[test]
    fn test_transition_status_variants() {
        let statuses = vec![
            TransitionStatus::Proposed,
            TransitionStatus::ConditionsReview,
            TransitionStatus::InProgress,
            TransitionStatus::Completed,
            TransitionStatus::Cancelled,
        ];
        assert_eq!(statuses.len(), 5);
    }

    // =========================================================================
    // GeoLocation Tests
    // =========================================================================

    #[test]
    fn test_geo_location_valid() {
        let location = valid_geo_location();
        assert!(location.latitude >= -90.0 && location.latitude <= 90.0);
        assert!(location.longitude >= -180.0 && location.longitude <= 180.0);
    }

    #[test]
    fn test_geo_location_equator() {
        let location = GeoLocation {
            latitude: 0.0,
            longitude: 0.0,
            region: "Gulf of Guinea".to_string(),
            country: "International Waters".to_string(),
        };
        assert_eq!(location.latitude, 0.0);
        assert_eq!(location.longitude, 0.0);
    }

    // =========================================================================
    // TerraAtlasProject Tests
    // =========================================================================

    fn valid_terra_atlas_project() -> TerraAtlasProject {
        TerraAtlasProject {
            id: "project:TA-2024-001:123456".to_string(),
            terra_atlas_id: "TA-2024-001".to_string(),
            name: "Sahara Solar Farm".to_string(),
            project_type: EnergyType::Solar,
            location: valid_geo_location(),
            capacity_mw: 500.0,
            total_investment: 100_000_000,
            current_investment: 50_000_000,
            status: ProjectStatus::Funding,
            regenerative_progress: 0.0,
            synced_at: create_test_timestamp(),
            phi_score: None,
            harmony_alignment: None,
        }
    }

    #[test]
    fn test_terra_atlas_project_valid() {
        let project = valid_terra_atlas_project();
        assert!(project.capacity_mw > 0.0);
        assert!(!project.terra_atlas_id.is_empty());
    }

    #[test]
    fn test_terra_atlas_project_zero_capacity_invalid() {
        let project = TerraAtlasProject {
            capacity_mw: 0.0,
            ..valid_terra_atlas_project()
        };
        assert!(project.capacity_mw <= 0.0);
    }

    #[test]
    fn test_terra_atlas_project_negative_capacity_invalid() {
        let project = TerraAtlasProject {
            capacity_mw: -50.0,
            ..valid_terra_atlas_project()
        };
        assert!(project.capacity_mw <= 0.0);
    }

    #[test]
    fn test_terra_atlas_project_regenerative_progress() {
        let project = TerraAtlasProject {
            regenerative_progress: 0.5,
            ..valid_terra_atlas_project()
        };
        assert!(project.regenerative_progress >= 0.0 && project.regenerative_progress <= 1.0);
    }

    #[test]
    fn test_terra_atlas_project_funding_status() {
        let project = valid_terra_atlas_project();
        let funding_percentage = (project.current_investment as f64 / project.total_investment as f64) * 100.0;
        assert_eq!(funding_percentage, 50.0);
    }

    // =========================================================================
    // InvestmentRecord Tests
    // =========================================================================

    fn valid_investment_record() -> InvestmentRecord {
        InvestmentRecord {
            id: "invest:project1:did:mycelix:investor1:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            investor_did: "did:mycelix:investor1".to_string(),
            amount: 50000,
            currency: "USD".to_string(),
            source_happ: "mycelix-finance".to_string(),
            investment_type: InvestmentType::Equity,
            created_at: create_test_timestamp(),
        }
    }

    #[test]
    fn test_investment_record_valid_did() {
        let record = valid_investment_record();
        assert!(record.investor_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_investment_record_positive_amount() {
        let record = valid_investment_record();
        assert!(record.amount > 0);
    }

    #[test]
    fn test_investment_record_invalid_did() {
        let record = InvestmentRecord {
            investor_did: "investor123".to_string(),
            ..valid_investment_record()
        };
        assert!(!record.investor_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_investment_record_zero_amount_invalid() {
        let record = InvestmentRecord {
            amount: 0,
            ..valid_investment_record()
        };
        assert!(record.amount == 0);
    }

    #[test]
    fn test_investment_record_all_types() {
        let types = vec![
            InvestmentType::Equity,
            InvestmentType::Loan,
            InvestmentType::Grant,
            InvestmentType::CommunityShare,
        ];
        for inv_type in types {
            let record = InvestmentRecord {
                investment_type: inv_type.clone(),
                ..valid_investment_record()
            };
            assert_eq!(record.investment_type, inv_type);
        }
    }

    // =========================================================================
    // RegenerativeMilestone Tests
    // =========================================================================

    fn valid_regenerative_milestone() -> RegenerativeMilestone {
        RegenerativeMilestone {
            id: "milestone:project1:CommunityFormation:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            milestone_type: MilestoneType::CommunityFormation,
            community_readiness: 0.75,
            operator_certification: true,
            financial_sustainability: 0.8,
            achieved_at: Some(create_test_timestamp()),
            verified_by: Some("did:mycelix:verifier1".to_string()),
        }
    }

    #[test]
    fn test_regenerative_milestone_valid_readiness() {
        let milestone = valid_regenerative_milestone();
        assert!(milestone.community_readiness >= 0.0 && milestone.community_readiness <= 1.0);
    }

    #[test]
    fn test_regenerative_milestone_over_1_invalid() {
        let milestone = RegenerativeMilestone {
            community_readiness: 1.5,
            ..valid_regenerative_milestone()
        };
        assert!(milestone.community_readiness > 1.0);
    }

    #[test]
    fn test_regenerative_milestone_negative_invalid() {
        let milestone = RegenerativeMilestone {
            community_readiness: -0.1,
            ..valid_regenerative_milestone()
        };
        assert!(milestone.community_readiness < 0.0);
    }

    #[test]
    fn test_regenerative_milestone_achieved() {
        let milestone = valid_regenerative_milestone();
        assert!(milestone.achieved_at.is_some());
        assert!(milestone.verified_by.is_some());
    }

    #[test]
    fn test_regenerative_milestone_pending() {
        let milestone = RegenerativeMilestone {
            achieved_at: None,
            verified_by: None,
            ..valid_regenerative_milestone()
        };
        assert!(milestone.achieved_at.is_none());
    }

    // =========================================================================
    // ProductionRecord Tests
    // =========================================================================

    fn valid_production_record() -> ProductionRecord {
        ProductionRecord {
            id: "prod:project1:123456:789".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            terra_atlas_id: "TA-2024-001".to_string(),
            period_start: create_test_timestamp(),
            period_end: Timestamp::from_micros(1706745600000000),
            energy_generated_mwh: 1500.0,
            capacity_factor: 0.25,
            revenue_generated: 150000,
            currency: "USD".to_string(),
            grid_injection_mwh: 1200.0,
            self_consumption_mwh: 300.0,
            recorded_at: Timestamp::from_micros(1706832000000000),
            verified_by: Some("did:mycelix:grid_operator".to_string()),
        }
    }

    #[test]
    fn test_production_record_valid() {
        let record = valid_production_record();
        assert!(record.energy_generated_mwh >= 0.0);
        assert!(record.capacity_factor >= 0.0 && record.capacity_factor <= 1.0);
    }

    #[test]
    fn test_production_record_negative_energy_invalid() {
        let record = ProductionRecord {
            energy_generated_mwh: -100.0,
            ..valid_production_record()
        };
        assert!(record.energy_generated_mwh < 0.0);
    }

    #[test]
    fn test_production_record_capacity_factor_over_1_invalid() {
        let record = ProductionRecord {
            capacity_factor: 1.5,
            ..valid_production_record()
        };
        assert!(record.capacity_factor > 1.0);
    }

    #[test]
    fn test_production_record_capacity_factor_negative_invalid() {
        let record = ProductionRecord {
            capacity_factor: -0.1,
            ..valid_production_record()
        };
        assert!(record.capacity_factor < 0.0);
    }

    #[test]
    fn test_production_record_grid_plus_self_not_exceed_generated() {
        let record = valid_production_record();
        let total_usage = record.grid_injection_mwh + record.self_consumption_mwh;
        // Allow 1% tolerance as stated in validation
        assert!(total_usage <= record.energy_generated_mwh * 1.01);
    }

    #[test]
    fn test_production_record_exceeds_generated_invalid() {
        let record = ProductionRecord {
            energy_generated_mwh: 1000.0,
            grid_injection_mwh: 800.0,
            self_consumption_mwh: 300.0, // 800 + 300 = 1100 > 1000
            ..valid_production_record()
        };
        let total = record.grid_injection_mwh + record.self_consumption_mwh;
        assert!(total > record.energy_generated_mwh * 1.01);
    }

    // =========================================================================
    // PendingSyncRecord Tests
    // =========================================================================

    fn valid_pending_sync_record() -> PendingSyncRecord {
        PendingSyncRecord {
            id: "sync:ProductionMetrics:project1:123456".to_string(),
            sync_type: SyncType::ProductionMetrics,
            target_system: "terra-atlas".to_string(),
            payload: r#"{"energy_mwh": 1500}"#.to_string(),
            created_at: create_test_timestamp(),
            synced_at: None,
            retry_count: 0,
            last_error: None,
        }
    }

    #[test]
    fn test_pending_sync_record_valid() {
        let record = valid_pending_sync_record();
        assert!(!record.target_system.is_empty());
    }

    #[test]
    fn test_pending_sync_record_empty_target_invalid() {
        let record = PendingSyncRecord {
            target_system: "".to_string(),
            ..valid_pending_sync_record()
        };
        assert!(record.target_system.is_empty());
    }

    #[test]
    fn test_pending_sync_record_not_synced() {
        let record = valid_pending_sync_record();
        assert!(record.synced_at.is_none());
    }

    #[test]
    fn test_pending_sync_record_synced() {
        let record = PendingSyncRecord {
            synced_at: Some(Timestamp::from_micros(1706918400000000)),
            ..valid_pending_sync_record()
        };
        assert!(record.synced_at.is_some());
    }

    #[test]
    fn test_pending_sync_record_with_retry() {
        let record = PendingSyncRecord {
            retry_count: 3,
            last_error: Some("Connection timeout".to_string()),
            ..valid_pending_sync_record()
        };
        assert!(record.retry_count > 0);
        assert!(record.last_error.is_some());
    }

    // =========================================================================
    // TransitionRecord Tests
    // =========================================================================

    fn valid_transition_record() -> TransitionRecord {
        TransitionRecord {
            id: "transition:project1:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            terra_atlas_id: "TA-2024-001".to_string(),
            from_ownership_pct: 25.0,
            to_ownership_pct: 50.0,
            community_did: "did:mycelix:community1".to_string(),
            reserve_account_balance: 100000,
            conditions_met: vec!["CommunityReadiness".to_string(), "FinancialSustainability".to_string()],
            conditions_pending: vec!["GovernanceMaturity".to_string()],
            status: TransitionStatus::InProgress,
            initiated_at: create_test_timestamp(),
            completed_at: None,
        }
    }

    #[test]
    fn test_transition_record_valid() {
        let record = valid_transition_record();
        assert!(record.to_ownership_pct > record.from_ownership_pct);
        assert!(record.community_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_transition_record_decrease_invalid() {
        let record = TransitionRecord {
            from_ownership_pct: 50.0,
            to_ownership_pct: 25.0,
            ..valid_transition_record()
        };
        assert!(record.to_ownership_pct < record.from_ownership_pct);
    }

    #[test]
    fn test_transition_record_over_100_invalid() {
        let record = TransitionRecord {
            to_ownership_pct: 110.0,
            ..valid_transition_record()
        };
        assert!(record.to_ownership_pct > 100.0);
    }

    #[test]
    fn test_transition_record_invalid_did() {
        let record = TransitionRecord {
            community_did: "community123".to_string(),
            ..valid_transition_record()
        };
        assert!(!record.community_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_transition_record_to_100_percent() {
        let record = TransitionRecord {
            from_ownership_pct: 75.0,
            to_ownership_pct: 100.0,
            status: TransitionStatus::InProgress,
            ..valid_transition_record()
        };
        assert_eq!(record.to_ownership_pct, 100.0);
    }

    #[test]
    fn test_transition_record_completed() {
        let record = TransitionRecord {
            status: TransitionStatus::Completed,
            completed_at: Some(Timestamp::from_micros(1709510400000000)),
            ..valid_transition_record()
        };
        assert!(record.completed_at.is_some());
        assert_eq!(record.status, TransitionStatus::Completed);
    }

    // =========================================================================
    // ProjectQuery Tests
    // =========================================================================

    fn valid_project_query() -> ProjectQuery {
        ProjectQuery {
            id: "query:project1:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            source: "terra-atlas".to_string(),
            query_type: ProjectQueryType::ProjectDetails,
            queried_at: create_test_timestamp(),
        }
    }

    #[test]
    fn test_project_query_valid() {
        let query = valid_project_query();
        assert!(!query.project_id.is_empty() || !query.source.is_empty());
    }

    #[test]
    fn test_project_query_empty_both_invalid() {
        let query = ProjectQuery {
            project_id: "".to_string(),
            source: "".to_string(),
            ..valid_project_query()
        };
        assert!(query.project_id.is_empty() && query.source.is_empty());
    }

    #[test]
    fn test_project_query_all_types() {
        let types = vec![
            ProjectQueryType::ProjectDetails,
            ProjectQueryType::InvestmentStatus,
            ProjectQueryType::CommunityReadiness,
            ProjectQueryType::RegenerativeProgress,
        ];
        for query_type in types {
            let query = ProjectQuery {
                query_type: query_type.clone(),
                ..valid_project_query()
            };
            assert_eq!(query.query_type, query_type);
        }
    }

    // =========================================================================
    // Anchor Tests
    // =========================================================================

    #[test]
    fn test_anchor_creation() {
        let anchor = Anchor("all_projects".to_string());
        assert_eq!(anchor.0, "all_projects");
    }

    #[test]
    fn test_anchor_equality() {
        let anchor1 = Anchor("pending_syncs".to_string());
        let anchor2 = Anchor("pending_syncs".to_string());
        let anchor3 = Anchor("all_projects".to_string());

        assert_eq!(anchor1, anchor2);
        assert_ne!(anchor1, anchor3);
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_serialization_terra_atlas_project() {
        let project = valid_terra_atlas_project();
        let result = serde_json::to_string(&project);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_investment_record() {
        let record = valid_investment_record();
        let result = serde_json::to_string(&record);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_production_record() {
        let record = valid_production_record();
        let result = serde_json::to_string(&record);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_transition_record() {
        let record = valid_transition_record();
        let result = serde_json::to_string(&record);
        assert!(result.is_ok());
    }

    // =========================================================================
    // ConsciousnessAssessment Tests
    // =========================================================================

    fn valid_consciousness_assessment() -> ConsciousnessAssessment {
        ConsciousnessAssessment {
            id: "assess:project1:did:mycelix:symthaea1:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            scorer_did: "did:mycelix:symthaea1".to_string(),
            phi_score: 0.72,
            harmony_alignment: 0.85,
            per_harmony_scores: r#"{"ResonantCoherence":0.9,"PanSentientFlourishing":0.8}"#.to_string(),
            care_activation: 0.65,
            meta_awareness: 0.70,
            assessment_cycle: 42000,
            assessed_at: create_test_timestamp(),
        }
    }

    #[test]
    fn test_consciousness_assessment_valid() {
        let assessment = valid_consciousness_assessment();
        assert!(assessment.scorer_did.starts_with("did:mycelix:"));
        assert!(assessment.phi_score >= 0.0 && assessment.phi_score <= 1.0);
        assert!(assessment.harmony_alignment >= 0.0 && assessment.harmony_alignment <= 1.0);
        assert!(assessment.care_activation >= 0.0 && assessment.care_activation <= 1.0);
        assert!(assessment.meta_awareness >= 0.0 && assessment.meta_awareness <= 1.0);
    }

    #[test]
    fn test_consciousness_assessment_invalid_did() {
        let assessment = ConsciousnessAssessment {
            scorer_did: "symthaea1".to_string(),
            ..valid_consciousness_assessment()
        };
        assert!(!assessment.scorer_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_consciousness_assessment_phi_out_of_range() {
        let assessment = ConsciousnessAssessment {
            phi_score: 1.5,
            ..valid_consciousness_assessment()
        };
        assert!(assessment.phi_score > 1.0);
    }

    #[test]
    fn test_consciousness_assessment_negative_phi() {
        let assessment = ConsciousnessAssessment {
            phi_score: -0.1,
            ..valid_consciousness_assessment()
        };
        assert!(assessment.phi_score < 0.0);
    }

    #[test]
    fn test_consciousness_assessment_harmony_out_of_range() {
        let assessment = ConsciousnessAssessment {
            harmony_alignment: 1.2,
            ..valid_consciousness_assessment()
        };
        assert!(assessment.harmony_alignment > 1.0);
    }

    #[test]
    fn test_consciousness_assessment_care_out_of_range() {
        let assessment = ConsciousnessAssessment {
            care_activation: -0.5,
            ..valid_consciousness_assessment()
        };
        assert!(assessment.care_activation < 0.0);
    }

    #[test]
    fn test_consciousness_assessment_meta_out_of_range() {
        let assessment = ConsciousnessAssessment {
            meta_awareness: 2.0,
            ..valid_consciousness_assessment()
        };
        assert!(assessment.meta_awareness > 1.0);
    }

    #[test]
    fn test_consciousness_assessment_nan_phi() {
        let assessment = ConsciousnessAssessment {
            phi_score: f64::NAN,
            ..valid_consciousness_assessment()
        };
        assert!(!assessment.phi_score.is_finite());
    }

    #[test]
    fn test_consciousness_assessment_infinity_harmony() {
        let assessment = ConsciousnessAssessment {
            harmony_alignment: f64::INFINITY,
            ..valid_consciousness_assessment()
        };
        assert!(!assessment.harmony_alignment.is_finite());
    }

    #[test]
    fn test_consciousness_assessment_empty_project_invalid() {
        let assessment = ConsciousnessAssessment {
            project_id: "".to_string(),
            ..valid_consciousness_assessment()
        };
        assert!(assessment.project_id.is_empty());
    }

    #[test]
    fn test_consciousness_assessment_boundary_values() {
        let assessment = ConsciousnessAssessment {
            phi_score: 0.0,
            harmony_alignment: 1.0,
            care_activation: 0.0,
            meta_awareness: 1.0,
            ..valid_consciousness_assessment()
        };
        assert!(assessment.phi_score >= 0.0 && assessment.phi_score <= 1.0);
        assert!(assessment.harmony_alignment >= 0.0 && assessment.harmony_alignment <= 1.0);
    }

    #[test]
    fn test_consciousness_assessment_per_harmony_json() {
        let assessment = valid_consciousness_assessment();
        let parsed: Result<HashMap<String, f64>, _> = serde_json::from_str(&assessment.per_harmony_scores);
        assert!(parsed.is_ok());
        let scores = parsed.unwrap();
        assert!(scores.contains_key("ResonantCoherence"));
    }

    #[test]
    fn test_serialization_consciousness_assessment() {
        let assessment = valid_consciousness_assessment();
        let result = serde_json::to_string(&assessment);
        assert!(result.is_ok());
        let roundtrip: Result<ConsciousnessAssessment, _> = serde_json::from_str(&result.unwrap());
        assert!(roundtrip.is_ok());
        assert_eq!(roundtrip.unwrap().phi_score, 0.72);
    }

    #[test]
    fn test_consciousness_assessment_cycle_tracking() {
        let a1 = ConsciousnessAssessment {
            assessment_cycle: 1000,
            ..valid_consciousness_assessment()
        };
        let a2 = ConsciousnessAssessment {
            assessment_cycle: 2000,
            ..valid_consciousness_assessment()
        };
        assert!(a2.assessment_cycle > a1.assessment_cycle);
    }

    #[test]
    fn test_energy_event_type_consciousness_assessed() {
        let event_type = EnergyEventType::ConsciousnessAssessed;
        let json = serde_json::to_string(&event_type);
        assert!(json.is_ok());
        assert_eq!(json.unwrap(), "\"ConsciousnessAssessed\"");
    }

    #[test]
    fn test_sync_type_trust_score() {
        let sync_type = SyncType::ConsciousnessScore;
        let json = serde_json::to_string(&sync_type);
        assert!(json.is_ok());
        assert_eq!(json.unwrap(), "\"ConsciousnessScore\"");
    }

    // =========================================================================
    // AllocationPledge Tests
    // =========================================================================

    fn valid_allocation_pledge() -> AllocationPledge {
        AllocationPledge {
            id: "pledge:project1:did:mycelix:alice:123456".to_string(),
            pledger_did: "did:mycelix:alice".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            amount: 40,
            currency: "TEND".to_string(),
            trust_score: 0.55,
            trust_tier: "Citizen".to_string(),
            harmony_intent: "Ecological Reciprocity — supporting community energy access".to_string(),
            status: PledgeStatus::Pending,
            pledged_at: create_test_timestamp(),
            expires_at: Timestamp::from_micros(1704067200000000 + 86_400_000_000), // +24h
            matched_at: None,
        }
    }

    #[test]
    fn test_allocation_pledge_valid() {
        let pledge = valid_allocation_pledge();
        assert!(pledge.pledger_did.starts_with("did:mycelix:"));
        assert!(pledge.amount > 0);
        assert!(pledge.trust_score >= 0.0 && pledge.trust_score <= 1.0);
        assert!(!pledge.harmony_intent.is_empty());
        assert!(pledge.expires_at > pledge.pledged_at);
    }

    #[test]
    fn test_allocation_pledge_invalid_did() {
        let pledge = AllocationPledge {
            pledger_did: "alice".to_string(),
            ..valid_allocation_pledge()
        };
        assert!(!pledge.pledger_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_allocation_pledge_zero_amount() {
        let pledge = AllocationPledge {
            amount: 0,
            ..valid_allocation_pledge()
        };
        assert_eq!(pledge.amount, 0);
    }

    #[test]
    fn test_allocation_pledge_consciousness_out_of_range() {
        let pledge = AllocationPledge {
            trust_score: 1.5,
            ..valid_allocation_pledge()
        };
        assert!(pledge.trust_score > 1.0);
    }

    #[test]
    fn test_allocation_pledge_negative_consciousness() {
        let pledge = AllocationPledge {
            trust_score: -0.1,
            ..valid_allocation_pledge()
        };
        assert!(pledge.trust_score < 0.0);
    }

    #[test]
    fn test_allocation_pledge_nan_consciousness() {
        let pledge = AllocationPledge {
            trust_score: f64::NAN,
            ..valid_allocation_pledge()
        };
        assert!(!pledge.trust_score.is_finite());
    }

    #[test]
    fn test_allocation_pledge_empty_project() {
        let pledge = AllocationPledge {
            project_id: "".to_string(),
            ..valid_allocation_pledge()
        };
        assert!(pledge.project_id.is_empty());
    }

    #[test]
    fn test_allocation_pledge_empty_currency() {
        let pledge = AllocationPledge {
            currency: "".to_string(),
            ..valid_allocation_pledge()
        };
        assert!(pledge.currency.is_empty());
    }

    #[test]
    fn test_allocation_pledge_empty_harmony_intent() {
        let pledge = AllocationPledge {
            harmony_intent: "".to_string(),
            ..valid_allocation_pledge()
        };
        assert!(pledge.harmony_intent.is_empty());
    }

    #[test]
    fn test_allocation_pledge_expiry_before_pledge() {
        let pledge = AllocationPledge {
            pledged_at: Timestamp::from_micros(1704067200000000 + 86_400_000_000),
            expires_at: create_test_timestamp(), // Before pledged_at
            ..valid_allocation_pledge()
        };
        assert!(pledge.expires_at <= pledge.pledged_at);
    }

    #[test]
    fn test_allocation_pledge_status_lifecycle() {
        let statuses = vec![
            PledgeStatus::Pending,
            PledgeStatus::Matched,
            PledgeStatus::Fulfilled,
            PledgeStatus::Expired,
            PledgeStatus::Withdrawn,
        ];
        assert_eq!(statuses.len(), 5);
    }

    #[test]
    fn test_allocation_pledge_matched() {
        let pledge = AllocationPledge {
            status: PledgeStatus::Matched,
            matched_at: Some(Timestamp::from_micros(1704067200000000 + 3600_000_000)),
            ..valid_allocation_pledge()
        };
        assert!(pledge.matched_at.is_some());
        assert_eq!(pledge.status, PledgeStatus::Matched);
    }

    #[test]
    fn test_allocation_pledge_tend_currency() {
        let pledge = valid_allocation_pledge();
        assert_eq!(pledge.currency, "TEND");
        assert!(pledge.amount <= 40); // TEND balance cap
    }

    #[test]
    fn test_serialization_allocation_pledge() {
        let pledge = valid_allocation_pledge();
        let json = serde_json::to_string(&pledge).unwrap();
        let back: AllocationPledge = serde_json::from_str(&json).unwrap();
        assert_eq!(back.pledger_did, "did:mycelix:alice");
        assert_eq!(back.amount, 40);
        assert_eq!(back.status, PledgeStatus::Pending);
    }

    #[test]
    fn test_pledge_status_serialization() {
        let status = PledgeStatus::Matched;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"Matched\"");
    }

    #[test]
    fn test_energy_event_type_pledge_submitted() {
        let event = EnergyEventType::PledgeSubmitted;
        let json = serde_json::to_string(&event).unwrap();
        assert_eq!(json, "\"PledgeSubmitted\"");
    }

    // =========================================================================
    // AllocationMatch Tests
    // =========================================================================

    fn valid_allocation_match() -> AllocationMatch {
        AllocationMatch {
            id: "match:project1:pledge1:123456".to_string(),
            pledge_id: "pledge:project1:did:mycelix:alice:111".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            pledger_did: "did:mycelix:alice".to_string(),
            holder_did: "did:mycelix:developer_bob".to_string(),
            amount: 30,
            match_score: 0.75, // 0.40*0.65 + 0.30*0.90 + 0.20*0.85 + 0.10*0.50
            trust_weight: 0.65,
            amount_fit: 0.90,
            harmony_alignment: 0.85,
            community_proximity: 0.50,
            status: MatchStatus::Proposed,
            proposed_at: create_test_timestamp(),
            resolved_at: None,
        }
    }

    #[test]
    fn test_allocation_match_valid() {
        let m = valid_allocation_match();
        assert!(m.pledger_did.starts_with("did:mycelix:"));
        assert!(m.holder_did.starts_with("did:mycelix:"));
        assert!(m.match_score >= 0.0 && m.match_score <= 1.0);
        assert!(m.trust_weight >= 0.0 && m.trust_weight <= 1.0);
        assert!(m.amount > 0);
    }

    #[test]
    fn test_allocation_match_score_components() {
        let m = valid_allocation_match();
        // Verify the 4 scoring components are valid
        assert!(m.trust_weight >= 0.0 && m.trust_weight <= 1.0);
        assert!(m.amount_fit >= 0.0 && m.amount_fit <= 1.0);
        assert!(m.harmony_alignment >= 0.0 && m.harmony_alignment <= 1.0);
        assert!(m.community_proximity >= 0.0 && m.community_proximity <= 1.0);
        // Composite should be weighted average of components
        let expected = 0.40 * m.trust_weight + 0.30 * m.amount_fit
            + 0.20 * m.harmony_alignment + 0.10 * m.community_proximity;
        assert!((m.match_score - expected).abs() < 0.01);
    }

    #[test]
    fn test_allocation_match_invalid_pledger_did() {
        let m = AllocationMatch { pledger_did: "alice".to_string(), ..valid_allocation_match() };
        assert!(!m.pledger_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_allocation_match_invalid_holder_did() {
        let m = AllocationMatch { holder_did: "bob".to_string(), ..valid_allocation_match() };
        assert!(!m.holder_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_allocation_match_zero_amount() {
        let m = AllocationMatch { amount: 0, ..valid_allocation_match() };
        assert_eq!(m.amount, 0);
    }

    #[test]
    fn test_allocation_match_score_out_of_range() {
        let m = AllocationMatch { match_score: 1.5, ..valid_allocation_match() };
        assert!(m.match_score > 1.0);
    }

    #[test]
    fn test_allocation_match_nan_score() {
        let m = AllocationMatch { match_score: f64::NAN, ..valid_allocation_match() };
        assert!(!m.match_score.is_finite());
    }

    #[test]
    fn test_allocation_match_status_lifecycle() {
        let statuses = vec![MatchStatus::Proposed, MatchStatus::Accepted, MatchStatus::Rejected, MatchStatus::Completed];
        assert_eq!(statuses.len(), 4);
    }

    #[test]
    fn test_allocation_match_accepted() {
        let m = AllocationMatch {
            status: MatchStatus::Accepted,
            resolved_at: Some(Timestamp::from_micros(1704067200000000 + 3600_000_000)),
            ..valid_allocation_match()
        };
        assert_eq!(m.status, MatchStatus::Accepted);
        assert!(m.resolved_at.is_some());
    }

    #[test]
    fn test_serialization_allocation_match() {
        let m = valid_allocation_match();
        let json = serde_json::to_string(&m).unwrap();
        let back: AllocationMatch = serde_json::from_str(&json).unwrap();
        assert_eq!(back.match_score, 0.75);
        assert_eq!(back.status, MatchStatus::Proposed);
    }

    // =========================================================================
    // ImpactRecord Tests
    // =========================================================================

    fn valid_impact_record() -> ImpactRecord {
        ImpactRecord {
            id: "impact:project1:did:mycelix:reporter:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            reporter_did: "did:mycelix:reporter".to_string(),
            period_start: create_test_timestamp(),
            period_end: Timestamp::from_micros(1704067200000000 + 2_592_000_000_000), // +30 days
            co2_avoided_tonnes: 267.5,
            jobs_created: 12,
            community_trust_delta: 0.14,
            energy_access_households: 500,
            biodiversity_index_delta: 0.03,
            verification_evidence: Some("ipfs://QmSolar123...".to_string()),
            verified_by: Some("did:mycelix:grid_verifier".to_string()),
            recorded_at: Timestamp::from_micros(1706745600000000),
        }
    }

    #[test]
    fn test_impact_record_valid() {
        let impact = valid_impact_record();
        assert!(impact.reporter_did.starts_with("did:mycelix:"));
        assert!(impact.co2_avoided_tonnes >= 0.0);
        assert!(impact.community_trust_delta >= -1.0 && impact.community_trust_delta <= 1.0);
        assert!(impact.period_end > impact.period_start);
    }

    #[test]
    fn test_impact_record_invalid_reporter_did() {
        let impact = ImpactRecord { reporter_did: "reporter".to_string(), ..valid_impact_record() };
        assert!(!impact.reporter_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_impact_record_negative_co2() {
        let impact = ImpactRecord { co2_avoided_tonnes: -10.0, ..valid_impact_record() };
        assert!(impact.co2_avoided_tonnes < 0.0);
    }

    #[test]
    fn test_impact_record_trust_out_of_range() {
        let impact = ImpactRecord { community_trust_delta: 1.5, ..valid_impact_record() };
        assert!(impact.community_trust_delta > 1.0);
    }

    #[test]
    fn test_impact_record_trust_negative() {
        let impact = ImpactRecord { community_trust_delta: -0.3, ..valid_impact_record() };
        assert!(impact.community_trust_delta >= -1.0 && impact.community_trust_delta <= 0.0);
    }

    #[test]
    fn test_impact_record_nan_biodiversity() {
        let impact = ImpactRecord { biodiversity_index_delta: f64::NAN, ..valid_impact_record() };
        assert!(!impact.biodiversity_index_delta.is_finite());
    }

    #[test]
    fn test_impact_record_period_end_before_start() {
        let impact = ImpactRecord {
            period_start: Timestamp::from_micros(1706745600000000),
            period_end: create_test_timestamp(), // Before start
            ..valid_impact_record()
        };
        assert!(impact.period_end <= impact.period_start);
    }

    #[test]
    fn test_impact_record_unverified() {
        let impact = ImpactRecord {
            verification_evidence: None,
            verified_by: None,
            ..valid_impact_record()
        };
        assert!(impact.verification_evidence.is_none());
        assert!(impact.verified_by.is_none());
    }

    #[test]
    fn test_impact_record_qol_composite() {
        let impact = valid_impact_record();
        // Net Humanity Benefit: weighted composite of QOL metrics
        let nhb = (impact.co2_avoided_tonnes / 1000.0) * 0.3
            + (impact.jobs_created as f64 / 100.0) * 0.25
            + impact.community_trust_delta * 0.25
            + (impact.energy_access_households as f64 / 1000.0) * 0.2;
        assert!(nhb > 0.0); // Positive project should have positive NHB
    }

    #[test]
    fn test_serialization_impact_record() {
        let impact = valid_impact_record();
        let json = serde_json::to_string(&impact).unwrap();
        let back: ImpactRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.co2_avoided_tonnes, 267.5);
        assert_eq!(back.jobs_created, 12);
        assert_eq!(back.energy_access_households, 500);
    }

    #[test]
    fn test_match_status_serialization() {
        let s = MatchStatus::Accepted;
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, "\"Accepted\"");
    }

    #[test]
    fn test_impact_event_type() {
        let e = EnergyEventType::ImpactReported;
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(json, "\"ImpactReported\"");
    }

    #[test]
    fn test_sync_type_impact_metrics() {
        let s = SyncType::ImpactMetrics;
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, "\"ImpactMetrics\"");
    }

    // =========================================================================
    // MatchFeedback Tests
    // =========================================================================

    fn valid_match_feedback() -> MatchFeedback {
        MatchFeedback {
            id: "feedback:match1:Pledger:123456".to_string(),
            match_id: "match:project1:pledge1:111".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            rater_did: "did:mycelix:alice".to_string(),
            rated_did: "did:mycelix:developer_bob".to_string(),
            role: FeedbackRole::Pledger,
            outcome_score: 0.85,
            harmony_fulfilled: 0.90,
            would_match_again: true,
            comment: Some("Project delivered on ecological promises".to_string()),
            rated_at: create_test_timestamp(),
        }
    }

    #[test]
    fn test_match_feedback_valid() {
        let fb = valid_match_feedback();
        assert!(fb.rater_did.starts_with("did:mycelix:"));
        assert!(fb.rated_did.starts_with("did:mycelix:"));
        assert_ne!(fb.rater_did, fb.rated_did);
        assert!(fb.outcome_score >= 0.0 && fb.outcome_score <= 1.0);
        assert!(fb.harmony_fulfilled >= 0.0 && fb.harmony_fulfilled <= 1.0);
    }

    #[test]
    fn test_match_feedback_self_rate_invalid() {
        let fb = MatchFeedback {
            rated_did: "did:mycelix:alice".to_string(), // same as rater
            ..valid_match_feedback()
        };
        assert_eq!(fb.rater_did, fb.rated_did);
    }

    #[test]
    fn test_match_feedback_outcome_out_of_range() {
        let fb = MatchFeedback { outcome_score: 1.5, ..valid_match_feedback() };
        assert!(fb.outcome_score > 1.0);
    }

    #[test]
    fn test_match_feedback_harmony_nan() {
        let fb = MatchFeedback { harmony_fulfilled: f64::NAN, ..valid_match_feedback() };
        assert!(!fb.harmony_fulfilled.is_finite());
    }

    #[test]
    fn test_match_feedback_roles() {
        let pledger = FeedbackRole::Pledger;
        let holder = FeedbackRole::Holder;
        assert_ne!(pledger, holder);
    }

    #[test]
    fn test_match_feedback_no_comment() {
        let fb = MatchFeedback { comment: None, ..valid_match_feedback() };
        assert!(fb.comment.is_none());
    }

    #[test]
    fn test_serialization_match_feedback() {
        let fb = valid_match_feedback();
        let json = serde_json::to_string(&fb).unwrap();
        let back: MatchFeedback = serde_json::from_str(&json).unwrap();
        assert_eq!(back.outcome_score, 0.85);
        assert!(back.would_match_again);
        assert_eq!(back.role, FeedbackRole::Pledger);
    }

    #[test]
    fn test_feedback_role_serialization() {
        let r = FeedbackRole::Holder;
        let json = serde_json::to_string(&r).unwrap();
        assert_eq!(json, "\"Holder\"");
    }
}
