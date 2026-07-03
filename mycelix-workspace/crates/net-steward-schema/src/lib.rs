use ed25519_dalek::Verifier;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InfrastructureNode {
    pub node_id: String,
    pub hostname: Option<String>,
    pub node_kind: NodeKind,
    pub management_state: ManagementState,
    pub owner_did: Option<String>,
    pub site_id: Option<String>,
    pub observed_at_unix_ms: u64,
    // Provenance
    pub source_collector: String,
    pub confidence: f32,
    pub staleness_ms: u64,
    pub evidence_hash: Option<String>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeKind {
    Router,
    Switch,
    AccessPoint,
    Server,
    Workstation,
    Phone,
    Iot,
    Service,
    Container,
    VirtualMachine,
    VirtualBridge,
    Unknown,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ManagementState {
    Managed,
    Unmanaged,
    Ignored,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NetworkEdge {
    pub source_node_id: String,
    pub target_node_id: String,
    pub edge_kind: EdgeKind,
    pub confidence: f32,
    pub evidence_refs: Vec<String>,
    // Provenance
    pub source_collector: String,
    pub staleness_ms: u64,
    pub evidence_hash: Option<String>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EdgeKind {
    Lldp,
    DhcpLease,
    Route,
    WireGuardPeer,
    XeniaSession,
    ServiceDependency,
    VirtualBridgeLink,
    ContainerLink,
    VlanTag,
    Inferred,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigState {
    pub node_id: String,
    pub desired_generation: Option<String>,
    pub observed_generation: Option<String>,
    pub drift_status: DriftStatus,
    pub rollback_generation: Option<String>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DriftStatus {
    InSync,
    DriftDetected,
    Unknown,
    Unmanaged,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InfrastructureReceipt {
    pub receipt_id: String,
    pub parent_hash: Option<String>,
    pub actor_did: String,
    pub target_node_id: String,
    pub action_kind: ActionKind,
    pub requested_at_unix_ms: u64,
    pub approved_by: Vec<String>,
    pub evidence_hashes: Vec<String>,
    pub cryptographic_signature: Option<String>,
    pub rollback_ref: Option<String>,
    pub chronicle_status: ChronicleStatus,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionKind {
    ApplyConfig,
    RollbackConfig,
    VerifyIdentity,
    EmergencyOverride,
    ConsoleSession,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChronicleStatus {
    Committed,
    Pending,
    Failed,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SafetyVerdict {
    Safe,
    Warning,
    Blocked,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProofStatus {
    NotPresent,
    CommitmentOnly,
    SimulatedEnvelope,
    VerificationUnavailable,
    Verified,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HumanReadableIncidentSummary {
    pub incident_id: String,
    pub safety_verdict: SafetyVerdict,
    pub safety_violations: Vec<String>,
    pub root_cause: String,
    pub affected_services: Vec<String>,
    pub affected_users: Vec<String>,
    pub blast_radius_score: f32,
    pub confidence: f32,
    pub recommended_action: String,
    pub rollback_path: Option<String>,
    pub safety_proof: Option<Vec<u8>>,
    pub safety_commitment: Option<[u8; 32]>,
    pub proof_status: ProofStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObservedTopologySnapshot {
    pub snapshot_id: String,
    pub observed_at_unix_ms: u64,
    pub nodes: Vec<InfrastructureNode>,
    pub edges: Vec<NetworkEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigDriftReport {
    pub report_id: String,
    pub checked_at_unix_ms: u64,
    pub node_id: String,
    pub drift_status: DriftStatus,
    pub diff_closure: Option<String>,
    pub systemd_unit_delta: Vec<String>,
    pub firewall_delta: Vec<String>,
    pub service_delta: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RollbackPlan {
    pub plan_id: String,
    pub target_node_id: String,
    pub current_generation: Option<String>,
    pub rollback_generation: Option<String>,
    pub expected_changes: Vec<ConfigDelta>,
    pub risk_level: RiskLevel,
    pub requires_approval: bool,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfigDelta {
    pub component: String,
    pub delta_desc: String,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DaemonVersion {
    pub name: String,
    pub version: String,
    pub mode: String,
    pub mutation_enabled: bool,
    pub proof_mode: String,
    pub bind: String,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CollectorKind {
    LocalHostProc,
    NixosProfile,
    OpenWrtLease,
    OpnSenseApi,
    Wireguard,
    BridgeLink,
    XeniaSession,
    MycelixIdentity,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CollectorOutput {
    pub nodes: Vec<InfrastructureNode>,
    pub edges: Vec<NetworkEdge>,
    pub evidence: Vec<EvidenceArtifact>,
    pub warnings: Vec<CollectorWarning>,
    pub collected_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvidenceArtifact {
    pub artifact_id: String,
    pub source_collector: String,
    pub raw_payload: String,
    pub hash_commitment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CollectorWarning {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CollectorError {
    AccessDenied(String),
    CommandFailed(String),
    ParseError(String),
    Unknown(String),
}

impl std::fmt::Display for CollectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for CollectorError {}

// --- Security Witness Schemas ---

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SecurityEvent {
    pub event_id: String,
    pub node_id: String,
    pub observed_at_unix_ms: u64,
    pub event_kind: SecurityEventKind,
    pub severity: Severity,
    pub confidence: f32,
    pub source_collector: String,
    pub evidence_hash: String,
    pub related_process: Option<ProcessRef>,
    pub related_identity: Option<String>,
    pub related_network_edge: Option<String>,
    pub recommended_action: Vec<RecommendedAction>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecurityEventKind {
    SuspiciousProcess,
    UnexpectedListeningPort,
    PrivilegeChange,
    NewSuidBinary,
    SystemdUnitChanged,
    FirewallRuleChanged,
    UnknownDeviceObserved,
    UnusualOutboundConnection,
    XeniaSessionStarted,
    ConfigDriftDetected,
    IdentityRiskObserved,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProcessRef {
    pub pid: u32,
    pub process_name: String,
    pub exe_path: String,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecommendedAction {
    GenerateIsolationPlan,
    GenerateRollbackPlan,
    RequestXeniaAdminSession,
    RequestUserConfirmation,
    CreateEvidenceCapsule,
    ExportIncidentBundle,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SecurityPosture {
    pub node_id: String,
    pub active_systemd_units: Vec<String>,
    pub open_ports: Vec<u16>,
    pub active_users: Vec<String>,
    pub firewall_policy_hash: String,
    pub nixos_generation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnownGoodBaseline {
    pub node_id: String,
    pub desired_generation: String,
    pub expected_services: Vec<String>,
    pub expected_open_ports: Vec<u16>,
    pub expected_users: Vec<String>,
    pub expected_systemd_units: Vec<String>,
    pub expected_firewall_policy_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IncidentCapsuleManifest {
    pub capsule_id: String,
    pub exported_at_unix_ms: u64,
    pub target_event_id: String,
    pub topology_snapshot: ObservedTopologySnapshot,
    pub posture_snapshot: SecurityPosture,
    pub baseline_snapshot: KnownGoodBaseline,
    pub rollback_plan: RollbackPlan,
    pub cryptographic_receipt_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IncidentCapsule {
    pub capsule_id: String,
    pub exported_at_unix_ms: u64,
    pub target_event_id: String,
    pub topology_snapshot: ObservedTopologySnapshot,
    pub security_events: Vec<SecurityEvent>,
    pub posture_snapshot: SecurityPosture,
    pub baseline_snapshot: KnownGoodBaseline,
    pub evidence_ledger: Vec<InfrastructureReceipt>,
    pub rollback_plan: RollbackPlan,
    pub cryptographic_receipt_hash: String,
}

// --- Local Authorization Envelope (Execution Plans Disabled) ---

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionIntent {
    pub intent_id: String,
    pub target_node_id: String,
    pub requested_capability: OperatorCapability,
    pub reason: String,
    pub evidence_refs: Vec<String>,
    pub rollback_plan: RollbackPlan,
    pub expiration_unix_ms: u64,
    pub approved: bool,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OperatorCapability {
    NixosSwitch,
    WireguardPeerRevoke,
    FirewallRuleUpdate,
    InterfaceIsolate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApprovalRequest {
    pub request_id: String,
    pub intent: ActionIntent,
    pub requesting_operator_did: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApprovalReceipt {
    pub approval_id: String,
    pub request_id: String,
    pub authorizing_signature: String,
    pub approved_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionPlan {
    pub plan_id: String,
    pub intent_id: String,
    pub execution_steps: Vec<String>,
    pub expected_target_state_hash: String,
}

// --- v0.3-alpha.4: Blast-Radius Preview ---
//
// A BlastRadiusPreview is produced by the dry-run executor BEFORE any
// operation is approved.  It is purely a read-only display artifact: the
// operator sees it, decides whether to proceed, and signs an
// ApprovalEnvelope.  The executor itself never writes to the network.

/// Risk tier for an operation's potential impact.
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BlastRadiusRiskTier {
    /// No active services affected; safe to proceed without escalation.
    Negligible,
    /// One or more services briefly interrupted; operator awareness required.
    Low,
    /// Multiple services or data paths affected; peer-witnessed approval required.
    Moderate,
    /// Core infrastructure impacted; quorum of independent witnesses required.
    High,
    /// Catastrophic scope — federation-wide or security-critical; blocked in lab mode.
    Critical,
}

/// A single service (or network path) that the planned operation would touch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AffectedService {
    /// Logical name of the service, e.g. "wireguard-peer-mesh" or "nixos-config-store".
    pub service_name: String,
    /// Owning node identifier.
    pub node_id: String,
    /// Whether the service would be interrupted (vs. merely reconfigured).
    pub interruption_expected: bool,
    /// Human-readable description of the specific impact.
    pub impact_description: String,
    /// Whether a rollback restores this service automatically.
    pub rollback_restores: bool,
}

/// The blast-radius preview produced for an OperationIntent before any
/// approval is sought.  Operators review this before signing an
/// ApprovalEnvelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BlastRadiusPreview {
    /// Stable identifier for this preview (linked to the intent it describes).
    pub preview_id: String,
    /// The intent this preview was generated from.
    pub intent_id: String,
    /// Human-readable one-line summary shown at the top of the UI card.
    pub summary: String,
    /// Computed risk tier.
    pub risk_tier: BlastRadiusRiskTier,
    /// Numeric score 0.0–1.0 (preserved for backward compat with HumanReadableIncidentSummary).
    pub blast_radius_score: f32,
    /// Services that would be touched by the operation.
    pub affected_services: Vec<AffectedService>,
    /// Node IDs in the transitive impact radius (may extend beyond direct targets).
    pub transitive_node_ids: Vec<String>,
    /// Estimated recovery time in seconds if rollback is needed.
    pub estimated_recovery_seconds: Option<u32>,
    /// Whether a rollback plan is already attached to the linked intent.
    pub rollback_plan_attached: bool,
    /// Whether independent-witness approval is required given the risk tier.
    pub requires_witness_approval: bool,
    /// Unix-ms timestamp when this preview was generated.
    pub generated_at_unix_ms: u64,
    /// Optional cryptographic commitment over the preview fields (for audit).
    pub preview_commitment: Option<[u8; 32]>,
}

// --- Federated Posture, Attack Path & Verification Schemas ---

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeerNodeStatus {
    pub peer_id: String,
    pub node_id: String,
    pub display_name: String,
    pub peer_kind: PeerKind,
    pub posture_summary: PostureSummary,
    pub trust_status: PeerTrustStatus,
    pub last_seen_unix_ms: u64,
    pub staleness_ms: u64,
    pub evidence_refs: Vec<String>,
    pub capsule_refs: Vec<String>,
    pub claimed_by: String,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PeerKind {
    Gateway,
    Server,
    Workstation,
    IoT,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PostureSummary {
    Healthy,
    Degraded,
    Critical,
    Unknown,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PeerTrustStatus {
    LocalSelf,
    SignedPeer,
    UnsignedPeer,
    StalePeer,
    ConflictingClaims,
    Quarantined,
    FederatedConsensus,
    VerifiedBoundFresh,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttackPathNode {
    pub node_id: String,
    pub display_name: String,
    pub step_description: String,
    pub evidence_hash: String,
    pub confidence: f32,
    pub truth_layer: String,
    pub source_collector: String,
    pub recommended_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttackPathEdge {
    pub source_node_id: String,
    pub target_node_id: String,
    pub relationship: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IncidentVerificationResult {
    pub capsule_id: String,
    pub schema_version: String,
    pub hashes_valid: bool,
    pub evidence_ledger_valid: bool,
    pub rollback_plan_valid: bool,
    pub security_events_valid: bool,
    pub proof_status: ProofStatus,
    pub mutation_claims_found: bool,
    pub result_passed: bool,
    pub verification_summary: String,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ClaimStatus {
    Valid,
    Expired,
    Revoked,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SignatureScheme {
    Ed25519,
    Secp256k1,
    Dilithium5,
    SimulatedEnvelope,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ClaimVerificationStatus {
    Unverified,
    VerifiedSignature,
    InvalidSignature,
    TrustedIssuer,
    ExpiredClaim,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeerPostureClaim {
    pub claim_id: String,
    pub issuer_did: String,
    pub subject_node_id: String,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub posture_summary: PostureSummary,
    pub topology_refs: Vec<String>,
    pub security_event_refs: Vec<String>,
    pub evidence_refs: Vec<String>,
    pub capsule_refs: Vec<String>,
    pub claim_status: ClaimStatus,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EncodingProfile {
    NetStewardCanonicalJsonV1,
    NetStewardDeterministicCborV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClaimSignature {
    pub issuer_did: String,
    pub signature: String,
    pub signature_scheme: SignatureScheme,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClaimEnvelope<T> {
    pub schema_version: String,
    pub encoding_profile: EncodingProfile,
    pub envelope_id: String,
    pub payload: T,
    pub payload_hash: String,
    pub issuer_did: String,
    pub signature: String,
    pub signature_scheme: SignatureScheme,
    pub verification_status: ClaimVerificationStatus,
    #[serde(default)]
    pub signatures: Option<Vec<ClaimSignature>>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrustTier {
    Tier1Trusted,
    Tier2Observer,
    Tier3Untrusted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WitnessAuthority {
    pub issuer_did: String,
    pub authority_domain: Option<String>,
    pub device_id: Option<String>,
    pub trust_tier: TrustTier,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConsensusPolicy {
    pub policy_version: String,
    pub min_signers: usize,
    pub max_claims_per_request: usize,
    pub max_payload_bytes: usize,
    pub require_unique_dids: bool,
    pub require_active_binding: bool,
    pub require_scope_valid: bool,
    pub allow_simulated_signatures: bool,
    pub allow_expired_claims: bool,
    pub allow_revoked_signers: bool,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConflictKind {
    PortStateConflict,
    GenerationConflict,
    PostureSeverityConflict,
    TopologyEdgeConflict,
    IdentityBindingConflict,
    EvidenceHashConflict,
    FreshnessConflict,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConflictingClaimRef {
    pub claim_id: String,
    pub issuer_did: String,
    pub value_summary: String,
    pub trust_status: PeerTrustStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostureConflict {
    pub conflict_id: String,
    pub subject_node_id: String,
    pub conflict_kind: ConflictKind,
    pub claims: Vec<ConflictingClaimRef>,
    pub recommended_display_status: PeerTrustStatus,
    pub operator_summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeerTelemetryReport {
    pub federation_mode: String,
    pub transport_available: bool,
    pub claims_fetched: u32,
    pub claims_accepted: u32,
    pub claims_rejected: u32,
    pub claims_stale: u32,
    pub conflict_count: u32,
    pub conflicts: Vec<PostureConflict>,
    pub peers: Vec<PeerNodeStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FederationStatus {
    pub transport: String,
    pub conductor_available: bool,
    pub dna_installed: bool,
    pub role_id: String,
    pub zome: String,
    pub identity_binding: String,
    pub claims_fetched: u32,
    pub claims_rejected: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CapabilityScope {
    TopologyObserve,
    SecurityPostureObserve,
    NixosGenerationObserve,
    XeniaSessionObserve,
    EvidencePublish,
    IncidentCapsulePublish,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DidAgentBinding {
    pub issuer_did: String,
    pub agent_pubkey: String,
    pub device_id: Option<String>,
    pub public_key_multibase: String,
    pub signature_scheme: SignatureScheme,
    pub claim_scopes: Vec<CapabilityScope>,
    pub valid_from_unix_ms: u64,
    pub expires_at_unix_ms: Option<u64>,
    pub revoked: bool,
    pub evidence_refs: Vec<String>,
}

pub trait DidBindingResolver {
    fn resolve_binding(&self, issuer_did: &str) -> Result<Option<DidAgentBinding>, String>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClaimEncodingError {
    EncodingFailure(String),
}

pub trait CanonicalClaimBytes {
    fn canonical_bytes(&self) -> Result<Vec<u8>, ClaimEncodingError>;
}

impl CanonicalClaimBytes for PeerPostureClaim {
    fn canonical_bytes(&self) -> Result<Vec<u8>, ClaimEncodingError> {
        let serialized = format!(
            "version:claim_envelope_v0.1|hash:{}|did:{}|issued:{}|expires:{}|subject:{}|posture:{:?}",
            self.claim_id,
            self.issuer_did,
            self.issued_at_unix_ms,
            self.expires_at_unix_ms,
            self.subject_node_id,
            self.posture_summary
        );
        Ok(serialized.into_bytes())
    }
}

pub trait ClaimSignatureVerifier {
    fn verify_envelope_signature(
        &self,
        envelope: &ClaimEnvelope<PeerPostureClaim>,
        binding: Option<&DidAgentBinding>,
    ) -> ClaimVerificationStatus;
}

pub struct Ed25519ClaimVerifier;

impl ClaimSignatureVerifier for Ed25519ClaimVerifier {
    fn verify_envelope_signature(
        &self,
        envelope: &ClaimEnvelope<PeerPostureClaim>,
        binding: Option<&DidAgentBinding>,
    ) -> ClaimVerificationStatus {
        if envelope.signature_scheme == SignatureScheme::SimulatedEnvelope {
            return ClaimVerificationStatus::InvalidSignature;
        }

        let Some(binding) = binding else {
            return ClaimVerificationStatus::InvalidSignature;
        };

        if binding.revoked {
            return ClaimVerificationStatus::InvalidSignature;
        }

        let Ok(public_key_bytes) = hex::decode(&binding.agent_pubkey) else {
            return ClaimVerificationStatus::InvalidSignature;
        };

        let Ok(public_key_arr) = public_key_bytes.try_into() else {
            return ClaimVerificationStatus::InvalidSignature;
        };

        let Ok(verifying_key) = ed25519_dalek::VerifyingKey::from_bytes(&public_key_arr) else {
            return ClaimVerificationStatus::InvalidSignature;
        };

        let Ok(signature_bytes) = hex::decode(&envelope.signature) else {
            return ClaimVerificationStatus::InvalidSignature;
        };

        let Ok(signature_arr) = signature_bytes.try_into() else {
            return ClaimVerificationStatus::InvalidSignature;
        };

        let signature = ed25519_dalek::Signature::from_bytes(&signature_arr);

        let Ok(claim_bytes) = envelope.payload.canonical_bytes() else {
            return ClaimVerificationStatus::InvalidSignature;
        };

        if verifying_key.verify(&claim_bytes, &signature).is_ok() {
            ClaimVerificationStatus::VerifiedSignature
        } else {
            ClaimVerificationStatus::InvalidSignature
        }
    }
}

pub struct SimulatedClaimVerifier;

impl ClaimSignatureVerifier for SimulatedClaimVerifier {
    fn verify_envelope_signature(
        &self,
        envelope: &ClaimEnvelope<PeerPostureClaim>,
        binding: Option<&DidAgentBinding>,
    ) -> ClaimVerificationStatus {
        if envelope.signature_scheme == SignatureScheme::SimulatedEnvelope {
            if let Some(binding) = binding {
                if !binding.revoked
                    && envelope.verification_status == ClaimVerificationStatus::VerifiedSignature
                {
                    return ClaimVerificationStatus::VerifiedSignature;
                }
            }
        }
        ClaimVerificationStatus::InvalidSignature
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OperationKind {
    GenerateRollbackPlan,
    ValidateRollbackPlan,
    PreviewServiceRestart,
    PreviewFirewallChange,
    ExportIncidentCapsule,
    RequestOperatorApproval,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperationIntent {
    pub intent_id: String,
    pub actor_did: String,
    pub target_node_id: String,
    pub operation_kind: OperationKind,
    pub reason: String,
    pub evidence_refs: Vec<String>,
    pub rollback_plan_ref: Option<String>,
    pub expires_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApprovalEnvelope {
    pub intent_id: String,
    pub approver_did: String,
    pub signature: String,
    pub signature_scheme: SignatureScheme,
    pub timestamp_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionResult {
    pub success: bool,
    pub output: String,
}

pub trait OperationExecutor {
    fn dry_run(&self, intent: &OperationIntent) -> Result<ExecutionPlan, String>;
    fn apply(&self, intent: &OperationIntent) -> Result<ExecutionResult, String>;
}
