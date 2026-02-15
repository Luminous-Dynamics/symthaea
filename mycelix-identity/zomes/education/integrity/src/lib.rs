//! Legacy Bridge: Academic Credential Integrity Zome
//!
//! Enhanced W3C Verifiable Credentials for educational institutions with:
//! - DNS-DID verification with DNSSEC validation
//! - ZK commitments for selective disclosure
//! - Mandatory revocation registry integration
//! - Batch CSV import support
//!
//! Part of the Mycelix Identity system.

use hdi::prelude::*;
use sha2::{Digest, Sha256};

// ============================================================================
// Academic Credential Types
// ============================================================================

/// Academic credential degree types
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DegreeType {
    /// High school diploma or equivalent
    HighSchool,
    /// Associate's degree (2-year)
    Associate,
    /// Bachelor's degree (4-year)
    Bachelor,
    /// Master's degree
    Master,
    /// Doctoral degree (PhD, EdD, etc.)
    Doctorate,
    /// Professional degree (JD, MD, etc.)
    Professional,
    /// Certificate program completion
    Certificate,
    /// Diploma (non-degree credential)
    Diploma,
    /// Microcredential or badge
    Microcredential,
    /// Course completion
    CourseCompletion,
}

/// Credential status for academic credentials
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AcademicCredentialStatus {
    /// Credential is valid and verified
    Valid,
    /// Credential is pending verification
    Pending,
    /// Credential has been revoked
    Revoked,
    /// Credential has expired
    Expired,
    /// Credential is suspended (temporary)
    Suspended,
}

/// Academic Credential - W3C VC 2.0 compliant with Mycelix enhancements
///
/// Key enhancements over base VerifiableCredential:
/// - `zk_commitment`: SHA-256 hash enabling selective disclosure
/// - `revocation_registry_id`: Mandatory revocation registry reference
/// - `dns_did`: DNS-DID verification data
/// - `achievement_metadata`: Rich academic achievement data
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AcademicCredential {
    // =========== W3C VC 2.0 Standard Fields ===========

    /// JSON-LD context (required: `https://www.w3.org/ns/credentials/v2`)
    #[serde(rename = "@context")]
    pub context: Vec<String>,

    /// Unique credential identifier (URN format)
    pub id: String,

    /// Credential types (must include "VerifiableCredential", "AcademicCredential")
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,

    /// Issuing institution
    pub issuer: InstitutionalIssuer,

    /// When the credential becomes valid (ISO 8601)
    #[serde(rename = "validFrom")]
    pub valid_from: String,

    /// When the credential expires (optional, ISO 8601)
    #[serde(rename = "validUntil")]
    pub valid_until: Option<String>,

    /// The academic achievement claims
    #[serde(rename = "credentialSubject")]
    pub credential_subject: AcademicSubject,

    /// Cryptographic proof
    pub proof: AcademicProof,

    // =========== Mycelix Enhancements ===========

    /// ZK commitment: SHA-256(credential_id || subject_id || nonce)
    /// Enables selective disclosure without revealing full credential
    pub zk_commitment: Vec<u8>,

    /// Nonce used in ZK commitment (kept private, shared only for verification)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commitment_nonce: Option<Vec<u8>>,

    /// MANDATORY: Revocation registry reference (not optional!)
    pub revocation_registry_id: String,

    /// Revocation list index for this credential
    pub revocation_index: u32,

    /// DNS-DID verification data
    pub dns_did: DnsDid,

    /// Achievement metadata
    pub achievement: AchievementMetadata,

    /// Mycelix schema reference
    pub mycelix_schema_id: String,

    /// Creation timestamp
    pub mycelix_created: Timestamp,

    /// Legacy system import reference (if imported via CSV)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub legacy_import_ref: Option<String>,
}

impl AcademicCredential {
    /// Compute ZK commitment from credential data
    pub fn compute_commitment(credential_id: &str, subject_id: &str, nonce: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(credential_id.as_bytes());
        hasher.update(b"|");
        hasher.update(subject_id.as_bytes());
        hasher.update(b"|");
        hasher.update(nonce);
        hasher.finalize().to_vec()
    }

    /// Verify ZK commitment matches credential data
    pub fn verify_commitment(&self, nonce: &[u8]) -> bool {
        let expected = Self::compute_commitment(&self.id, &self.credential_subject.id, nonce);
        self.zk_commitment == expected
    }
}

/// Institutional Issuer - Rich institution metadata
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct InstitutionalIssuer {
    /// Institution's DID (dns: or key: method)
    pub id: String,

    /// Institution name
    pub name: String,

    /// Types (e.g., "University", "College", "TrainingProvider")
    #[serde(rename = "type")]
    pub issuer_type: Vec<String>,

    /// Institution's image/logo URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,

    /// Institution location
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<InstitutionLocation>,

    /// Accreditation information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accreditation: Option<Vec<Accreditation>>,
}

/// Institution location
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct InstitutionLocation {
    /// Country code (ISO 3166-1 alpha-2)
    pub country: String,
    /// State/Province/Region
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// City
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
}

/// Accreditation record
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Accreditation {
    /// Accrediting body
    pub accreditor: String,
    /// Accreditation type
    pub accreditation_type: String,
    /// Valid from date
    pub valid_from: String,
    /// Valid until date
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<String>,
}

/// Academic credential subject
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct AcademicSubject {
    /// Subject's DID
    pub id: String,

    /// Legal name (may be hashed for privacy)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Name hash for privacy-preserving verification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name_hash: Option<Vec<u8>>,

    /// Date of birth (may be hashed or omitted)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub birth_date: Option<String>,

    /// Student ID (may be omitted)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub student_id: Option<String>,
}

/// Academic proof with enhanced verification
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct AcademicProof {
    /// Proof type
    #[serde(rename = "type")]
    pub proof_type: String,

    /// Creation timestamp
    pub created: String,

    /// Verification method (DID URL)
    #[serde(rename = "verificationMethod")]
    pub verification_method: String,

    /// Proof purpose
    #[serde(rename = "proofPurpose")]
    pub proof_purpose: String,

    /// Signature value (multibase encoded)
    #[serde(rename = "proofValue")]
    pub proof_value: String,

    /// Cryptosuite used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cryptosuite: Option<String>,

    /// Domain binding (prevents replay attacks)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,

    /// Challenge binding (for presentations)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenge: Option<String>,

    /// Algorithm identifier (multicodec u16) for PQC signature dispatch.
    ///
    /// When present, verifiers can use this to determine the cryptographic
    /// algorithm (e.g. Ed25519=0xED, ML-DSA-65=0x1206, Hybrid=0xF101)
    /// without parsing the signature bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<u16>,
}

/// Rich achievement metadata
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct AchievementMetadata {
    /// Degree type
    pub degree_type: DegreeType,

    /// Degree name (e.g., "Bachelor of Science")
    pub degree_name: String,

    /// Field of study / Major
    pub field_of_study: String,

    /// Minor(s) if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minors: Option<Vec<String>>,

    /// Conferral date
    pub conferral_date: String,

    /// GPA (optional, privacy-sensitive)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpa: Option<f32>,

    /// Honors/distinctions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub honors: Option<Vec<String>>,

    /// CIP code (Classification of Instructional Programs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cip_code: Option<String>,

    /// Credits earned
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credits_earned: Option<u32>,
}

// ============================================================================
// DNS-DID Verification Types
// ============================================================================

/// DNS-DID verification data with DNSSEC validation
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct DnsDid {
    /// The DNS domain (e.g., "registrar.university.edu")
    pub domain: String,

    /// DID document identifier
    pub did: String,

    /// DNS TXT record location (e.g., "_did.registrar.university.edu")
    pub txt_record: String,

    /// DNSSEC validation status
    pub dnssec: DnssecStatus,

    /// Last verification timestamp
    pub last_verified: Timestamp,

    /// Verification chain (for audit trail)
    pub verification_chain: Vec<DnsVerificationRecord>,
}

/// DNSSEC validation status
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum DnssecStatus {
    /// DNSSEC validated (AD flag set)
    Validated,
    /// DNSSEC signatures present but not validated
    Insecure,
    /// DNSSEC validation failed
    Invalid,
    /// Domain has no DNSSEC records
    Unsigned,
    /// DNSSEC status not checked
    Unknown,
}

/// DNS verification audit record
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct DnsVerificationRecord {
    /// Verification timestamp
    pub timestamp: Timestamp,
    /// DNS resolver used
    pub resolver: String,
    /// DNSSEC status at verification time
    pub dnssec_status: DnssecStatus,
    /// TTL of the TXT record
    pub ttl_seconds: u32,
}

/// DNS-DID linkage proof for Certificate Transparency
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct DnsDidLinkageProof {
    /// Domain being proved
    pub domain: String,
    /// DID being linked
    pub did: String,
    /// Certificate Transparency log entry (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ct_log_entry: Option<String>,
    /// Proof timestamp
    pub timestamp: Timestamp,
    /// Signature over the linkage
    pub signature: Vec<u8>,
}

// ============================================================================
// Legacy Bridge Import Types
// ============================================================================

/// Legacy system import batch
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct LegacyBridgeImport {
    /// Unique import batch ID
    pub batch_id: String,

    /// Importing institution's DID
    pub institution_did: String,

    /// Source system name
    pub source_system: String,

    /// Import timestamp
    pub import_timestamp: Timestamp,

    /// Total credentials in batch
    pub total_credentials: u32,

    /// Successfully imported
    pub imported_count: u32,

    /// Failed imports
    pub failed_count: u32,

    /// Import status
    pub status: ImportStatus,

    /// SHA-256 hash of source CSV for audit
    pub source_hash: Vec<u8>,

    /// Error details for failed imports
    pub errors: Vec<ImportError>,
}

/// Import batch status
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum ImportStatus {
    /// Import in progress
    InProgress,
    /// Import completed successfully
    Completed,
    /// Import completed with errors
    CompletedWithErrors,
    /// Import failed
    Failed,
    /// Import rolled back
    RolledBack,
}

/// Import error record
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ImportError {
    /// Row number in CSV (1-indexed)
    pub row: u32,
    /// Field name with error
    pub field: String,
    /// Error message
    pub message: String,
    /// Error code
    pub code: String,
}

/// CSV field mapping for legacy import
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct CsvFieldMapping {
    /// CSV column name -> Credential field
    pub student_id: String,
    pub first_name: String,
    pub last_name: String,
    pub degree_name: String,
    pub major: String,
    pub conferral_date: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpa: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub honors: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minor: Option<String>,
}

// ============================================================================
// Revocation Registry Types
// ============================================================================

/// Academic credential revocation request
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AcademicRevocationRequest {
    /// Credential to revoke
    pub credential_id: String,

    /// Requesting authority's DID
    pub requester_did: String,

    /// Revocation reason
    pub reason: RevocationReason,

    /// Detailed explanation
    pub explanation: String,

    /// Supporting evidence
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Vec<String>>,

    /// Request timestamp
    pub requested_at: Timestamp,

    /// Request status
    pub status: RevocationRequestStatus,
}

/// Reasons for academic credential revocation
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum RevocationReason {
    /// Academic fraud (cheating, plagiarism)
    AcademicFraud,
    /// Degree rescinded by institution
    DegreeRescinded,
    /// Issued in error
    IssuedInError,
    /// Holder request (voluntary revocation)
    HolderRequest,
    /// Court order
    CourtOrder,
    /// Institution closure/merger
    InstitutionChange,
    /// Other (must provide explanation)
    Other,
}

/// Revocation request processing status
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum RevocationRequestStatus {
    Pending,
    UnderReview,
    Approved,
    Rejected,
    Executed,
}

// ============================================================================
// Epistemic Claim Integration (DKG Bridge)
// ============================================================================

/// Reference linking an academic credential to its epistemic claim in the DKG.
///
/// When an academic credential is published to the knowledge graph, this entry
/// records the mapping between the credential and its E3-level epistemic claim.
/// The epistemic position is fixed at E3/N2/M3 for institutionally-verified
/// academic credentials (cryptographic proof, network consensus, immutable record).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EpistemicClaimReference {
    /// The credential ID (urn:uuid:...)
    pub credential_id: String,
    /// Action hash of the credential entry
    pub credential_hash: ActionHash,
    /// Claim ID assigned by the knowledge bridge
    pub claim_id: String,
    /// Source hApp identifier
    pub source_happ: String,
    /// Subject (student DID) - the claim subject
    pub subject: String,
    /// Predicate used in the knowledge graph triple
    pub predicate: String,
    /// Object value (achievement metadata as JSON)
    pub object: String,
    /// Empirical axis score (E3 = 0.8 for cryptographic proof)
    pub epistemic_e: f64,
    /// Normative axis score (N2 = 0.7 for network consensus)
    pub epistemic_n: f64,
    /// Materiality axis score (M3 = 0.9 for immutable on-chain record)
    pub epistemic_m: f64,
    /// Timestamp of claim publication
    pub published_at: Timestamp,
}

// ============================================================================
// Entry Types and Links
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
#[allow(clippy::large_enum_variant)] // HDK entry types require inline variants
pub enum EntryTypes {
    AcademicCredential(AcademicCredential),
    LegacyBridgeImport(LegacyBridgeImport),
    AcademicRevocationRequest(AcademicRevocationRequest),
    EpistemicClaimReference(EpistemicClaimReference),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Institution to credentials they've issued
    InstitutionToCredential,
    /// Subject to their credentials
    SubjectToCredential,
    /// Domain to DNS-DID verification
    DomainToDidVerification,
    /// Credential to revocation request
    CredentialToRevocation,
    /// Import batch to credentials
    ImportToCredential,
    /// Revocation registry to credentials
    RevocationRegistryToCredential,
    /// Credential to its epistemic claim in the DKG
    CredentialToEpistemicClaim,
    /// Subject to their epistemic claims
    SubjectToEpistemicClaim,
}

// ============================================================================
// Validation
// ============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::AcademicCredential(cred) => {
                    validate_create_academic_credential(EntryCreationAction::Create(action), cred)
                }
                EntryTypes::LegacyBridgeImport(import) => {
                    validate_create_legacy_import(EntryCreationAction::Create(action), import)
                }
                EntryTypes::AcademicRevocationRequest(req) => {
                    validate_create_revocation_request(EntryCreationAction::Create(action), req)
                }
                EntryTypes::EpistemicClaimReference(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry { app_entry, action, .. } => match app_entry {
                EntryTypes::AcademicCredential(_) => Ok(ValidateCallbackResult::Invalid(
                    "Academic credentials cannot be updated (immutable)".into(),
                )),
                EntryTypes::LegacyBridgeImport(import) => {
                    validate_update_legacy_import(action, import)
                }
                EntryTypes::AcademicRevocationRequest(req) => {
                    validate_update_revocation_request(action, req)
                }
                EntryTypes::EpistemicClaimReference(_) => Ok(ValidateCallbackResult::Invalid(
                    "Epistemic claim references cannot be updated (immutable)".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            if tag.0.len() > 1024 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds maximum length of 1024 bytes".into(),
                ));
            }
            match link_type {
                LinkTypes::InstitutionToCredential => Ok(ValidateCallbackResult::Valid),
                LinkTypes::SubjectToCredential => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DomainToDidVerification => Ok(ValidateCallbackResult::Valid),
                LinkTypes::CredentialToRevocation => Ok(ValidateCallbackResult::Valid),
                LinkTypes::ImportToCredential => Ok(ValidateCallbackResult::Valid),
                LinkTypes::RevocationRegistryToCredential => Ok(ValidateCallbackResult::Valid),
                LinkTypes::CredentialToEpistemicClaim => Ok(ValidateCallbackResult::Valid),
                LinkTypes::SubjectToEpistemicClaim => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the link creator can delete their links".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can update their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

/// Validate academic credential creation
fn validate_create_academic_credential(
    _action: EntryCreationAction,
    cred: AcademicCredential,
) -> ExternResult<ValidateCallbackResult> {
    // ===== W3C VC 2.0 Compliance =====

    // Must include W3C credentials context
    if !cred.context.iter().any(|c| c.contains("credentials")) {
        return Ok(ValidateCallbackResult::Invalid(
            "Credential must include W3C credentials context".into(),
        ));
    }

    // Must include required types
    if !cred.credential_type.contains(&"VerifiableCredential".to_string()) {
        return Ok(ValidateCallbackResult::Invalid(
            "Credential type must include 'VerifiableCredential'".into(),
        ));
    }
    if !cred.credential_type.contains(&"AcademicCredential".to_string()) {
        return Ok(ValidateCallbackResult::Invalid(
            "Credential type must include 'AcademicCredential'".into(),
        ));
    }

    // ===== Issuer Validation =====

    // Issuer must be a valid DID
    if !cred.issuer.id.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer must be a valid DID".into(),
        ));
    }

    // Issuer name required
    if cred.issuer.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer name is required".into(),
        ));
    }

    // ===== Subject Validation =====

    // Subject must have valid DID
    if !cred.credential_subject.id.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject must have a valid DID".into(),
        ));
    }

    // ===== Proof Validation =====

    if cred.proof.proof_type.is_empty() || cred.proof.proof_value.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Credential must have valid proof".into(),
        ));
    }

    if cred.proof.proof_purpose != "assertionMethod" {
        return Ok(ValidateCallbackResult::Invalid(
            "Credential proof purpose must be 'assertionMethod'".into(),
        ));
    }

    // ===== Mycelix Enhancements =====

    // ZK commitment required (32 bytes for SHA-256)
    if cred.zk_commitment.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "ZK commitment must be 32 bytes (SHA-256)".into(),
        ));
    }

    // MANDATORY: Revocation registry must be specified
    if cred.revocation_registry_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Revocation registry ID is MANDATORY for academic credentials".into(),
        ));
    }

    // DNS-DID domain must be valid
    if cred.dns_did.domain.is_empty() || !cred.dns_did.domain.contains('.') {
        return Ok(ValidateCallbackResult::Invalid(
            "DNS-DID domain must be a valid domain name".into(),
        ));
    }

    // Achievement validation
    if cred.achievement.degree_name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Degree name is required".into(),
        ));
    }

    if cred.achievement.field_of_study.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Field of study is required".into(),
        ));
    }

    // GPA validation if present
    if let Some(gpa) = cred.achievement.gpa {
        if !(0.0..=4.0).contains(&gpa) {
            return Ok(ValidateCallbackResult::Invalid(
                "GPA must be between 0.0 and 4.0".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate legacy import creation
fn validate_create_legacy_import(
    _action: EntryCreationAction,
    import: LegacyBridgeImport,
) -> ExternResult<ValidateCallbackResult> {
    // Institution must be valid DID
    if !import.institution_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Institution must be a valid DID".into(),
        ));
    }

    // Source system required
    if import.source_system.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source system name is required".into(),
        ));
    }

    // Source hash required (32 bytes for SHA-256)
    if import.source_hash.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hash must be 32 bytes (SHA-256)".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate legacy import update
fn validate_update_legacy_import(
    action: Update,
    import: LegacyBridgeImport,
) -> ExternResult<ValidateCallbackResult> {
    if !import.institution_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Institution must be a valid DID".into(),
        ));
    }

    // Fetch original to enforce invariants
    let original_record = must_get_valid_record(action.original_action_address.clone())?;
    let original: LegacyBridgeImport = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original import entry not found".into()
        )))?;

    // Immutable fields
    if import.batch_id != original.batch_id {
        return Ok(ValidateCallbackResult::Invalid(
            "Import batch ID cannot be changed".into(),
        ));
    }
    if import.institution_did != original.institution_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Institution DID cannot be changed".into(),
        ));
    }
    if import.source_system != original.source_system {
        return Ok(ValidateCallbackResult::Invalid(
            "Source system cannot be changed".into(),
        ));
    }
    if import.source_hash != original.source_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hash cannot be changed".into(),
        ));
    }
    if import.total_credentials != original.total_credentials {
        return Ok(ValidateCallbackResult::Invalid(
            "Total credentials count cannot be changed".into(),
        ));
    }

    // Counts must be monotonically non-decreasing
    if import.imported_count < original.imported_count {
        return Ok(ValidateCallbackResult::Invalid(
            "Imported count cannot decrease".into(),
        ));
    }
    if import.failed_count < original.failed_count {
        return Ok(ValidateCallbackResult::Invalid(
            "Failed count cannot decrease".into(),
        ));
    }

    // State machine: InProgress → Completed/CompletedWithErrors/Failed, Failed → RolledBack
    let valid = match (&original.status, &import.status) {
        (ImportStatus::InProgress, ImportStatus::Completed)
        | (ImportStatus::InProgress, ImportStatus::CompletedWithErrors)
        | (ImportStatus::InProgress, ImportStatus::Failed)
        | (ImportStatus::Failed, ImportStatus::RolledBack) => true,
        (a, b) if a == b => true,
        _ => false,
    };

    if !valid {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid import status transition".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate revocation request creation
fn validate_create_revocation_request(
    _action: EntryCreationAction,
    req: AcademicRevocationRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Requester must be valid DID
    if !req.requester_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester must be a valid DID".into(),
        ));
    }

    // Explanation required
    if req.explanation.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Revocation explanation is required".into(),
        ));
    }

    // For "Other" reason, explanation must be detailed
    if req.reason == RevocationReason::Other && req.explanation.len() < 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "For 'Other' reason, explanation must be at least 50 characters".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate revocation request update
fn validate_update_revocation_request(
    action: Update,
    req: AcademicRevocationRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Fetch original to enforce state transitions
    let original_record = must_get_valid_record(action.original_action_address.clone())?;
    let original: AcademicRevocationRequest = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original revocation request not found".into()
        )))?;

    // Immutable fields
    if req.credential_id != original.credential_id {
        return Ok(ValidateCallbackResult::Invalid(
            "Credential ID cannot be changed".into(),
        ));
    }
    if req.requester_did != original.requester_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester DID cannot be changed".into(),
        ));
    }

    // State machine: Pending → UnderReview → Approved/Rejected, Approved → Executed
    let valid = match (&original.status, &req.status) {
        (RevocationRequestStatus::Pending, RevocationRequestStatus::UnderReview)
        | (RevocationRequestStatus::UnderReview, RevocationRequestStatus::Approved)
        | (RevocationRequestStatus::UnderReview, RevocationRequestStatus::Rejected)
        | (RevocationRequestStatus::Approved, RevocationRequestStatus::Executed) => true,
        (a, b) if a == b => true,
        _ => false,
    };

    if !valid {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid revocation request status transition".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a ZK commitment for selective disclosure
pub fn create_zk_commitment(credential_id: &str, subject_id: &str, nonce: &[u8]) -> Vec<u8> {
    AcademicCredential::compute_commitment(credential_id, subject_id, nonce)
}

/// Validate DNSSEC status is acceptable
pub fn is_dnssec_acceptable(status: &DnssecStatus) -> bool {
    matches!(status, DnssecStatus::Validated | DnssecStatus::Insecure)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zk_commitment_computation() {
        let nonce = b"test_nonce_12345678901234567890";
        let commitment = create_zk_commitment("urn:uuid:123", "did:student:test", nonce);

        assert_eq!(commitment.len(), 32); // SHA-256 output

        // Same inputs should produce same output
        let commitment2 = create_zk_commitment("urn:uuid:123", "did:student:test", nonce);
        assert_eq!(commitment, commitment2);

        // Different inputs should produce different output
        let commitment3 = create_zk_commitment("urn:uuid:456", "did:student:test", nonce);
        assert_ne!(commitment, commitment3);
    }

    #[test]
    fn test_degree_types_serialization() {
        let types = vec![
            DegreeType::HighSchool,
            DegreeType::Associate,
            DegreeType::Bachelor,
            DegreeType::Master,
            DegreeType::Doctorate,
            DegreeType::Professional,
            DegreeType::Certificate,
            DegreeType::Diploma,
            DegreeType::Microcredential,
            DegreeType::CourseCompletion,
        ];

        for degree_type in types {
            let json = serde_json::to_string(&degree_type).unwrap();
            let deserialized: DegreeType = serde_json::from_str(&json).unwrap();
            assert_eq!(format!("{:?}", degree_type), format!("{:?}", deserialized));
        }
    }

    #[test]
    fn test_dnssec_status_acceptable() {
        assert!(is_dnssec_acceptable(&DnssecStatus::Validated));
        assert!(is_dnssec_acceptable(&DnssecStatus::Insecure));
        assert!(!is_dnssec_acceptable(&DnssecStatus::Invalid));
        assert!(!is_dnssec_acceptable(&DnssecStatus::Unsigned));
        assert!(!is_dnssec_acceptable(&DnssecStatus::Unknown));
    }

    #[test]
    fn test_dnssec_status_serialization() {
        let statuses = vec![
            DnssecStatus::Validated,
            DnssecStatus::Insecure,
            DnssecStatus::Invalid,
            DnssecStatus::Unsigned,
            DnssecStatus::Unknown,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let deserialized: DnssecStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(format!("{:?}", status), format!("{:?}", deserialized));
        }
    }

    #[test]
    fn test_revocation_reasons_serialization() {
        let reasons = vec![
            RevocationReason::AcademicFraud,
            RevocationReason::DegreeRescinded,
            RevocationReason::IssuedInError,
            RevocationReason::HolderRequest,
            RevocationReason::CourtOrder,
            RevocationReason::InstitutionChange,
            RevocationReason::Other,
        ];

        for reason in reasons {
            let json = serde_json::to_string(&reason).unwrap();
            let deserialized: RevocationReason = serde_json::from_str(&json).unwrap();
            assert_eq!(format!("{:?}", reason), format!("{:?}", deserialized));
        }
    }

    #[test]
    fn test_import_status_serialization() {
        let statuses = vec![
            ImportStatus::InProgress,
            ImportStatus::Completed,
            ImportStatus::CompletedWithErrors,
            ImportStatus::Failed,
            ImportStatus::RolledBack,
        ];

        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let deserialized: ImportStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(format!("{:?}", status), format!("{:?}", deserialized));
        }
    }

    #[test]
    fn test_achievement_metadata_serialization() {
        let achievement = AchievementMetadata {
            degree_type: DegreeType::Master,
            degree_name: "Master of Science".to_string(),
            field_of_study: "Data Science".to_string(),
            minors: Some(vec!["Statistics".to_string()]),
            conferral_date: "2025-12-15".to_string(),
            gpa: Some(3.95),
            honors: Some(vec!["Summa Cum Laude".to_string()]),
            cip_code: Some("11.0701".to_string()),
            credits_earned: Some(36),
        };

        let json = serde_json::to_string(&achievement).unwrap();
        let deserialized: AchievementMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.gpa, Some(3.95));
        assert_eq!(deserialized.minors.as_ref().unwrap().len(), 1);
        assert_eq!(deserialized.cip_code, Some("11.0701".to_string()));
        assert_eq!(deserialized.credits_earned, Some(36));
    }

    #[test]
    fn test_achievement_metadata_minimal() {
        let achievement = AchievementMetadata {
            degree_type: DegreeType::Certificate,
            degree_name: "Professional Certificate".to_string(),
            field_of_study: "Project Management".to_string(),
            minors: None,
            conferral_date: "2026-01-15".to_string(),
            gpa: None,
            honors: None,
            cip_code: None,
            credits_earned: None,
        };

        let json = serde_json::to_string(&achievement).unwrap();

        // Optional fields should not appear when None
        assert!(!json.contains("minors"));
        assert!(!json.contains("gpa"));
        assert!(!json.contains("honors"));
        assert!(!json.contains("cip_code"));
        assert!(!json.contains("credits_earned"));
    }

    #[test]
    fn test_institutional_issuer_serialization() {
        let issuer = InstitutionalIssuer {
            id: "did:dns:mit.edu".to_string(),
            name: "Massachusetts Institute of Technology".to_string(),
            issuer_type: vec!["Organization".to_string(), "EducationalOrganization".to_string()],
            image: Some("https://mit.edu/logo.png".to_string()),
            location: Some(InstitutionLocation {
                country: "US".to_string(),
                region: Some("Massachusetts".to_string()),
                city: Some("Cambridge".to_string()),
            }),
            accreditation: None,
        };

        let json = serde_json::to_string(&issuer).unwrap();
        assert!(json.contains("mit.edu"));
        assert!(json.contains("Cambridge"));
    }

    #[test]
    fn test_academic_subject_serialization() {
        let subject = AcademicSubject {
            id: "did:student:test:STU001".to_string(),
            name: Some("Jane Doe".to_string()),
            name_hash: Some(vec![1, 2, 3, 4, 5, 6, 7, 8]),
            birth_date: None,
            student_id: Some("STU001".to_string()),
        };

        let json = serde_json::to_string(&subject).unwrap();
        let deserialized: AcademicSubject = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, subject.id);
        assert_eq!(deserialized.name, Some("Jane Doe".to_string()));
    }

    #[test]
    fn test_academic_proof_serialization() {
        let proof = AcademicProof {
            proof_type: "DataIntegrityProof".to_string(),
            created: "2026-01-28T00:00:00Z".to_string(),
            verification_method: "did:dns:university.edu#keys-1".to_string(),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: "zBase64Signature".to_string(),
            cryptosuite: Some("eddsa-rdfc-2022".to_string()),
            domain: None,
            challenge: None,
            algorithm: None,
        };

        let json = serde_json::to_string(&proof).unwrap();

        // Check W3C VC 2.0 field names
        assert!(json.contains("\"type\""));
        assert!(json.contains("verificationMethod"));
        assert!(json.contains("proofPurpose"));
        assert!(json.contains("proofValue"));
        // algorithm: None is skipped in serialization
        assert!(!json.contains("algorithm"));
    }

    #[test]
    fn test_academic_proof_with_pqc_algorithm() {
        let proof = AcademicProof {
            proof_type: "DataIntegrityProof".to_string(),
            created: "2026-01-28T00:00:00Z".to_string(),
            verification_method: "did:dns:university.edu#keys-2".to_string(),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: "zPQCSignatureBase58".to_string(),
            cryptosuite: Some("dilithium-rdfc-2026".to_string()),
            domain: None,
            challenge: None,
            algorithm: Some(0x1206), // ML-DSA-65
        };

        let json = serde_json::to_string(&proof).unwrap();
        assert!(json.contains("\"algorithm\":4614")); // 0x1206 = 4614 decimal

        // Round-trip
        let restored: AcademicProof = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.algorithm, Some(0x1206));
    }

    #[test]
    fn test_academic_proof_backward_compat_no_algorithm() {
        // Old proofs without algorithm field should still deserialize
        let json = r#"{
            "type": "DataIntegrityProof",
            "created": "2025-06-01T00:00:00Z",
            "verificationMethod": "did:dns:old-uni.edu#key-1",
            "proofPurpose": "assertionMethod",
            "proofValue": "zOldSignature"
        }"#;

        let proof: AcademicProof = serde_json::from_str(json).unwrap();
        assert_eq!(proof.proof_type, "DataIntegrityProof");
        assert!(proof.algorithm.is_none());
        assert!(proof.cryptosuite.is_none());
    }

    #[test]
    fn test_csv_field_mapping_serialization() {
        let mapping = CsvFieldMapping {
            student_id: "student_id".to_string(),
            first_name: "first_name".to_string(),
            last_name: "last_name".to_string(),
            degree_name: "degree_name".to_string(),
            major: "major".to_string(),
            conferral_date: "conferral_date".to_string(),
            gpa: Some("gpa".to_string()),
            honors: Some("honors".to_string()),
            minor: Some("minor".to_string()),
        };

        let json = serde_json::to_string(&mapping).unwrap();
        let deserialized: CsvFieldMapping = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.student_id, mapping.student_id);
        assert_eq!(deserialized.gpa, Some("gpa".to_string()));
    }

    #[test]
    fn test_import_error_serialization() {
        let error = ImportError {
            row: 5,
            field: "gpa".to_string(),
            message: "Invalid GPA value".to_string(),
            code: "INVALID_GPA".to_string(),
        };

        let json = serde_json::to_string(&error).unwrap();
        let deserialized: ImportError = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.row, 5);
        assert_eq!(deserialized.field, "gpa");
    }
}
