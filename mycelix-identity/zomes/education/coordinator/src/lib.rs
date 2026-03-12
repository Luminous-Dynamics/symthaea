//! Legacy Bridge: Academic Credential Coordinator Zome
//!
//! Provides functions for:
//! - Issuing academic credentials with W3C VC 2.0 compliance
//! - CSV batch import from legacy systems
//! - DNS-DID verification
//! - Revocation management
//!
//! Part of the Mycelix Identity system.

use education_integrity::*;
use hdk::prelude::*;

// ============================================================================
// Credential Issuance
// ============================================================================

/// Input for creating a new academic credential
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateAcademicCredentialInput {
    /// Institution details
    pub issuer: InstitutionalIssuer,
    /// Subject (student) details
    pub subject: AcademicSubject,
    /// Achievement details
    pub achievement: AchievementMetadata,
    /// DNS-DID verification data
    pub dns_did: DnsDid,
    /// Revocation registry ID
    pub revocation_registry_id: String,
    /// Valid from date (ISO 8601)
    pub valid_from: String,
    /// Valid until date (optional)
    pub valid_until: Option<String>,
    /// Proof data
    pub proof: AcademicProof,
}

/// Output from credential creation
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateAcademicCredentialOutput {
    /// Created credential action hash
    pub action_hash: ActionHash,
    /// Credential ID
    pub credential_id: String,
    /// Revocation index assigned
    pub revocation_index: u32,
}

/// Create a new academic credential
#[hdk_extern]
pub fn create_academic_credential(
    input: CreateAcademicCredentialInput,
) -> ExternResult<CreateAcademicCredentialOutput> {
    // Verify caller is the issuing institution
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);
    if input.issuer.id != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the issuing institution can create academic credentials".into()
        )));
    }

    // Generate credential ID
    let credential_id = generate_credential_id(&input.issuer.id, &input.subject.id)?;

    // Generate ZK commitment
    let nonce = generate_nonce()?;
    let zk_commitment = create_zk_commitment(&credential_id, &input.subject.id, &nonce);

    // Get next revocation index (simplified - in production, coordinate with registry)
    let revocation_index = get_next_revocation_index(&input.revocation_registry_id)?;

    // Build W3C VC 2.0 compliant credential
    let credential = AcademicCredential {
        context: vec![
            "https://www.w3.org/ns/credentials/v2".to_string(),
            "https://w3id.org/vc/status-list/2021/v1".to_string(),
            "https://purl.imsglobal.org/spec/clr/v2p0/context.json".to_string(),
        ],
        id: credential_id.clone(),
        credential_type: vec![
            "VerifiableCredential".to_string(),
            "AcademicCredential".to_string(),
            format!("{:?}Credential", input.achievement.degree_type),
        ],
        issuer: input.issuer.clone(),
        valid_from: input.valid_from,
        valid_until: input.valid_until,
        credential_subject: input.subject.clone(),
        proof: input.proof,
        zk_commitment,
        commitment_nonce: Some(nonce),
        revocation_registry_id: input.revocation_registry_id,
        revocation_index,
        dns_did: input.dns_did,
        achievement: input.achievement,
        mycelix_schema_id: "mycelix:schema:education:academic:v1".to_string(),
        mycelix_created: sys_time()?,
        legacy_import_ref: None,
    };

    // Create the entry
    let action_hash = create_entry(EntryTypes::AcademicCredential(credential))?;

    // Create links using agent pubkey as anchor
    // Institution -> Credential (using path-based anchor)
    let institution_anchor = anchor_for_did(&input.issuer.id)?;
    create_link(
        institution_anchor,
        action_hash.clone(),
        LinkTypes::InstitutionToCredential,
        (),
    )?;

    // Subject -> Credential
    let subject_anchor = anchor_for_did(&input.subject.id)?;
    create_link(
        subject_anchor,
        action_hash.clone(),
        LinkTypes::SubjectToCredential,
        (),
    )?;

    Ok(CreateAcademicCredentialOutput {
        action_hash,
        credential_id,
        revocation_index,
    })
}

/// Get academic credential by action hash
#[hdk_extern]
pub fn get_academic_credential(
    action_hash: ActionHash,
) -> ExternResult<Option<AcademicCredential>> {
    match get(action_hash, GetOptions::default())? {
        Some(record) => {
            let credential: AcademicCredential = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Entry not found".to_string())))?;
            Ok(Some(credential))
        }
        None => Ok(None),
    }
}

/// Get credentials by institution DID
#[hdk_extern]
pub fn get_credentials_by_institution(institution_did: String) -> ExternResult<Vec<ActionHash>> {
    let anchor = anchor_for_did(&institution_did)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::InstitutionToCredential)?,
        GetStrategy::default(),
    )?;

    Ok(links
        .into_iter()
        .filter_map(|l| ActionHash::try_from(l.target).ok())
        .collect())
}

/// Get credentials by subject DID
#[hdk_extern]
pub fn get_credentials_by_subject(subject_did: String) -> ExternResult<Vec<ActionHash>> {
    let anchor = anchor_for_did(&subject_did)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::SubjectToCredential)?,
        GetStrategy::default(),
    )?;

    Ok(links
        .into_iter()
        .filter_map(|l| ActionHash::try_from(l.target).ok())
        .collect())
}

// ============================================================================
// Legacy CSV Import
// ============================================================================

/// Input for starting a legacy import
#[derive(Serialize, Deserialize, Debug)]
pub struct StartLegacyImportInput {
    /// Institution's DID
    pub institution_did: String,
    /// Source system name
    pub source_system: String,
    /// SHA-256 hash of CSV file
    pub source_hash: Vec<u8>,
    /// Expected total credentials
    pub total_credentials: u32,
}

/// Start a new legacy import batch
#[hdk_extern]
pub fn start_legacy_import(input: StartLegacyImportInput) -> ExternResult<String> {
    let batch_id = generate_batch_id()?;

    let import = LegacyBridgeImport {
        batch_id: batch_id.clone(),
        institution_did: input.institution_did,
        source_system: input.source_system,
        import_timestamp: sys_time()?,
        total_credentials: input.total_credentials,
        imported_count: 0,
        failed_count: 0,
        status: ImportStatus::InProgress,
        source_hash: input.source_hash,
        errors: vec![],
    };

    create_entry(EntryTypes::LegacyBridgeImport(import))?;

    Ok(batch_id)
}

/// Input for a single CSV row import
#[derive(Serialize, Deserialize, Debug)]
pub struct ImportCredentialFromCsvInput {
    /// Import batch ID
    pub batch_id: String,
    /// Row number (for error reporting)
    pub row_number: u32,
    /// Parsed CSV data
    pub student_id: String,
    pub first_name: String,
    pub last_name: String,
    pub degree_name: String,
    pub major: String,
    pub conferral_date: String,
    pub gpa: Option<f32>,
    pub honors: Option<Vec<String>>,
}

/// Import result for a single credential
#[derive(Serialize, Deserialize, Debug)]
pub struct ImportCredentialResult {
    pub row_number: u32,
    pub success: bool,
    pub credential_id: Option<String>,
    pub action_hash: Option<ActionHash>,
    pub error: Option<ImportError>,
}

/// Import a single credential from CSV data
/// Note: In production, this would be called by an off-chain CLI tool
#[hdk_extern]
pub fn import_credential_from_csv(
    input: ImportCredentialFromCsvInput,
) -> ExternResult<ImportCredentialResult> {
    // Validate and sanitize required fields
    let validate_field = |value: &str, name: &str, max_len: usize| -> Option<ImportError> {
        if value.is_empty() {
            return Some(ImportError {
                row: input.row_number,
                field: name.to_string(),
                message: format!("{} is required", name),
                code: "MISSING_REQUIRED".to_string(),
            });
        }
        if value.len() > max_len {
            return Some(ImportError {
                row: input.row_number,
                field: name.to_string(),
                message: format!("{} exceeds maximum length of {} bytes", name, max_len),
                code: "FIELD_TOO_LONG".to_string(),
            });
        }
        if value.contains('\0') {
            return Some(ImportError {
                row: input.row_number,
                field: name.to_string(),
                message: format!("{} contains invalid null bytes", name),
                code: "INVALID_CONTENT".to_string(),
            });
        }
        None
    };

    // Validate required fields with length limits
    let validations = [
        validate_field(&input.student_id, "student_id", 128),
        validate_field(&input.first_name, "first_name", 256),
        validate_field(&input.last_name, "last_name", 256),
        validate_field(&input.degree_name, "degree_name", 512),
    ];

    if let Some(validation) = validations.into_iter().flatten().next() {
        return Ok(ImportCredentialResult {
            row_number: input.row_number,
            success: false,
            credential_id: None,
            action_hash: None,
            error: Some(validation),
        });
    }

    // Validate major field length
    if input.major.len() > 512 {
        return Ok(ImportCredentialResult {
            row_number: input.row_number,
            success: false,
            credential_id: None,
            action_hash: None,
            error: Some(ImportError {
                row: input.row_number,
                field: "major".to_string(),
                message: "major exceeds maximum length of 512 bytes".to_string(),
                code: "FIELD_TOO_LONG".to_string(),
            }),
        });
    }

    // Validate GPA range if present
    if let Some(gpa) = input.gpa {
        if !(0.0..=5.0).contains(&gpa) {
            return Ok(ImportCredentialResult {
                row_number: input.row_number,
                success: false,
                credential_id: None,
                action_hash: None,
                error: Some(ImportError {
                    row: input.row_number,
                    field: "gpa".to_string(),
                    message: "GPA must be between 0.0 and 5.0".to_string(),
                    code: "INVALID_RANGE".to_string(),
                }),
            });
        }
    }

    let degree_type = parse_degree_type(&input.degree_name);

    // Build a subject DID from student_id (legacy import pattern)
    let subject_did = format!("did:mycelix:legacy:{}", input.student_id);

    // Build the credential creation input from CSV row data
    let now = sys_time()?;
    let now_str = format!("{}", now.as_micros());
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);
    let batch_ref = format!("did:mycelix:batch:{}", input.batch_id);

    let credential_input = CreateAcademicCredentialInput {
        issuer: InstitutionalIssuer {
            id: caller_did.clone(),
            name: "Legacy Import".to_string(),
            issuer_type: vec!["LegacyImportSource".to_string()],
            image: None,
            location: None,
            accreditation: None,
        },
        subject: AcademicSubject {
            id: subject_did,
            name: Some(format!("{} {}", input.first_name, input.last_name)),
            name_hash: None,
            birth_date: None,
            student_id: Some(input.student_id.clone()),
        },
        achievement: AchievementMetadata {
            degree_type,
            degree_name: input.degree_name,
            field_of_study: input.major,
            minors: None,
            conferral_date: input.conferral_date.clone(),
            gpa: input.gpa,
            honors: input.honors,
            cip_code: None,
            credits_earned: None,
        },
        dns_did: DnsDid {
            domain: "legacy-import.local".to_string(),
            did: batch_ref.clone(),
            txt_record: String::new(),
            dnssec: DnssecStatus::Unknown,
            last_verified: now,
            verification_chain: vec![],
        },
        revocation_registry_id: format!("registry:batch:{}", input.batch_id),
        valid_from: input.conferral_date,
        valid_until: None,
        proof: AcademicProof {
            proof_type: "LegacyImport".to_string(),
            created: now_str,
            verification_method: format!("{}#import-key", batch_ref),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: "legacy-import-no-cryptographic-proof".to_string(),
            cryptosuite: None,
            domain: None,
            challenge: None,
            algorithm: None,
        },
    };

    match create_academic_credential(credential_input) {
        Ok(output) => Ok(ImportCredentialResult {
            row_number: input.row_number,
            success: true,
            credential_id: Some(output.credential_id),
            action_hash: Some(output.action_hash),
            error: None,
        }),
        Err(e) => Ok(ImportCredentialResult {
            row_number: input.row_number,
            success: false,
            credential_id: None,
            action_hash: None,
            error: Some(ImportError {
                row: input.row_number,
                field: "credential_creation".to_string(),
                message: format!("Failed to create credential: {}", e),
                code: "CREATION_FAILED".to_string(),
            }),
        }),
    }
}

/// Update import batch status
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateImportStatusInput {
    pub batch_id: String,
    pub imported_count: u32,
    pub failed_count: u32,
    pub status: ImportStatus,
    pub new_errors: Vec<ImportError>,
}

#[hdk_extern]
pub fn update_import_status(input: UpdateImportStatusInput) -> ExternResult<()> {
    // In production, would fetch and update the import record
    // For now, this is a placeholder
    debug!("Updating import batch {} status", input.batch_id);
    Ok(())
}

// ============================================================================
// DNS-DID Verification
// ============================================================================

/// Input for recording DNS-DID verification result
#[derive(Serialize, Deserialize, Debug)]
pub struct RecordDnsDidVerificationInput {
    /// Domain to verify
    pub domain: String,
    /// Expected DID
    pub expected_did: String,
    /// DNSSEC status from verification
    pub dnssec_status: DnssecStatus,
    /// Resolved DID (if any)
    pub resolved_did: Option<String>,
}

/// Result of DNS-DID verification
#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyDnsDidResult {
    pub verified: bool,
    pub dnssec_status: DnssecStatus,
    pub resolved_did: Option<String>,
    pub verification_record: DnsVerificationRecord,
    pub error: Option<String>,
}

/// Record DNS-DID verification result
/// Note: Actual DNS resolution happens off-chain; this records the result
#[hdk_extern]
pub fn record_dns_did_verification(
    input: RecordDnsDidVerificationInput,
) -> ExternResult<VerifyDnsDidResult> {
    let verified = input.resolved_did.as_ref() == Some(&input.expected_did);

    let verification_record = DnsVerificationRecord {
        timestamp: sys_time()?,
        resolver: "1.1.1.1".to_string(), // Would be actual resolver used
        dnssec_status: input.dnssec_status.clone(),
        ttl_seconds: 3600,
    };

    Ok(VerifyDnsDidResult {
        verified,
        dnssec_status: input.dnssec_status,
        resolved_did: input.resolved_did,
        verification_record,
        error: if verified {
            None
        } else {
            Some("DID mismatch".to_string())
        },
    })
}

// ============================================================================
// Revocation Management
// ============================================================================

/// Request credential revocation
#[derive(Serialize, Deserialize, Debug)]
pub struct RequestRevocationInput {
    pub credential_id: String,
    pub reason: RevocationReason,
    pub explanation: String,
    pub evidence: Option<Vec<String>>,
}

#[hdk_extern]
pub fn request_academic_revocation(input: RequestRevocationInput) -> ExternResult<ActionHash> {
    let agent_info = agent_info()?;
    let requester_did = format!("did:key:{}", agent_info.agent_initial_pubkey);

    let request = AcademicRevocationRequest {
        credential_id: input.credential_id.clone(),
        requester_did,
        reason: input.reason,
        explanation: input.explanation,
        evidence: input.evidence,
        requested_at: sys_time()?,
        status: RevocationRequestStatus::Pending,
    };

    let action_hash = create_entry(EntryTypes::AcademicRevocationRequest(request))?;

    // Link credential to revocation request using anchor
    let credential_anchor = anchor_for_credential(&input.credential_id)?;
    create_link(
        credential_anchor,
        action_hash.clone(),
        LinkTypes::CredentialToRevocation,
        (),
    )?;

    Ok(action_hash)
}

/// Verify ZK commitment
#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyCommitmentInput {
    pub credential_id: String,
    pub subject_id: String,
    pub nonce: Vec<u8>,
    pub expected_commitment: Vec<u8>,
}

#[hdk_extern]
pub fn verify_zk_commitment(input: VerifyCommitmentInput) -> ExternResult<bool> {
    let computed = create_zk_commitment(&input.credential_id, &input.subject_id, &input.nonce);
    Ok(computed == input.expected_commitment)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create an anchor hash for a DID (for linking)
fn anchor_for_did(did: &str) -> ExternResult<EntryHash> {
    // Use a simple path-based anchor
    let path = Path::from(format!("did_anchor/{}", did));
    path.path_entry_hash()
}

/// Create an anchor hash for a credential ID
fn anchor_for_credential(credential_id: &str) -> ExternResult<EntryHash> {
    let path = Path::from(format!("credential_anchor/{}", credential_id));
    path.path_entry_hash()
}

/// Generate unique credential ID
fn generate_credential_id(issuer_did: &str, subject_did: &str) -> ExternResult<String> {
    let timestamp = sys_time()?.as_micros();
    let agent = agent_info()?.agent_initial_pubkey;

    // Create deterministic but unique ID
    let mut hasher = sha2::Sha256::new();
    use sha2::Digest;
    hasher.update(issuer_did.as_bytes());
    hasher.update(subject_did.as_bytes());
    hasher.update(timestamp.to_le_bytes());
    hasher.update(agent.get_raw_39());
    let hash = hasher.finalize();

    // Use first 16 bytes as UUID-like identifier
    Ok(format!(
        "urn:uuid:{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        hash[0], hash[1], hash[2], hash[3],
        hash[4], hash[5],
        hash[6], hash[7],
        hash[8], hash[9],
        hash[10], hash[11], hash[12], hash[13], hash[14], hash[15]
    ))
}

/// Generate random nonce for ZK commitment
fn generate_nonce() -> ExternResult<Vec<u8>> {
    // In production, use cryptographically secure random
    let timestamp = sys_time()?.as_micros();
    let agent = agent_info()?.agent_initial_pubkey;

    let mut hasher = sha2::Sha256::new();
    use sha2::Digest;
    hasher.update(b"nonce:");
    hasher.update(timestamp.to_le_bytes());
    hasher.update(agent.get_raw_39());

    Ok(hasher.finalize().to_vec())
}

/// Generate import batch ID
fn generate_batch_id() -> ExternResult<String> {
    let timestamp = sys_time()?.as_micros();
    let agent = agent_info()?.agent_initial_pubkey;

    let mut hasher = sha2::Sha256::new();
    use sha2::Digest;
    hasher.update(b"batch:");
    hasher.update(timestamp.to_le_bytes());
    hasher.update(agent.get_raw_39());
    let hash = hasher.finalize();

    let bytes: [u8; 8] = hash[0..8].try_into().map_err(|_| {
        wasm_error!(WasmErrorInner::Guest(
            "Hash slice conversion failed".to_string()
        ))
    })?;
    Ok(format!("batch-{:016x}", u64::from_le_bytes(bytes)))
}

/// Get next revocation index (simplified)
fn get_next_revocation_index(_registry_id: &str) -> ExternResult<u32> {
    // In production, query the revocation registry for next available index
    // For now, use timestamp-based index
    let timestamp = sys_time()?.as_micros();
    Ok((timestamp % u32::MAX as i64) as u32)
}

// ============================================================================
// DKG Integration - Epistemic Claim Publication
// ============================================================================

/// Epistemic position for institutionally-verified academic credentials.
/// E3 (Cryptographic) - signed by institution with verifiable proof
/// N2 (Network)       - accredited institution + registrar consensus
/// M3 (Immutable)     - permanent on-chain record
const ACADEMIC_EPISTEMIC_E: f64 = 0.8;
const ACADEMIC_EPISTEMIC_N: f64 = 0.7;
const ACADEMIC_EPISTEMIC_M: f64 = 0.9;

/// Input for publishing a credential as an epistemic claim
#[derive(Serialize, Deserialize, Debug)]
pub struct PublishCredentialAsClaimInput {
    /// Action hash of the credential to publish
    pub credential_action_hash: ActionHash,
}

/// Output from publishing a credential as an epistemic claim
#[derive(Serialize, Deserialize, Debug)]
pub struct PublishCredentialAsClaimOutput {
    /// The claim reference action hash
    pub claim_ref_hash: ActionHash,
    /// The claim ID in the knowledge graph
    pub claim_id: String,
    /// Epistemic classification (E-N-M position)
    pub epistemic_position: EpistemicPosition,
}

/// Epistemic position summary
#[derive(Serialize, Deserialize, Debug)]
pub struct EpistemicPosition {
    pub empirical: f64,
    pub normative: f64,
    pub materiality: f64,
    pub label: String,
}

/// Publish an academic credential as an E3 epistemic claim in the knowledge graph.
///
/// Transforms the credential's achievement metadata into a subject-predicate-object
/// triple and stores a local `EpistemicClaimReference` linking the credential to its
/// claim in the DKG. The claim is classified at E3/N2/M3 (cryptographic proof,
/// network consensus, immutable record).
///
/// In a multi-hApp deployment, this would call the knowledge bridge zome via
/// `hdk::prelude::call()` to register the external claim. In single-hApp mode,
/// it records the reference locally for later bridge sync.
#[hdk_extern]
pub fn publish_credential_as_epistemic_claim(
    input: PublishCredentialAsClaimInput,
) -> ExternResult<PublishCredentialAsClaimOutput> {
    // Retrieve the credential
    let credential = get_academic_credential(input.credential_action_hash.clone())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Credential not found".to_string())))?;

    // Build the knowledge graph triple
    let subject = credential.credential_subject.id.clone();
    let predicate = "hasAcademicAchievement".to_string();

    // Serialize achievement metadata as the claim object (omit PII)
    let object = serde_json::json!({
        "degree_type": format!("{:?}", credential.achievement.degree_type),
        "degree_name": credential.achievement.degree_name,
        "field_of_study": credential.achievement.field_of_study,
        "conferral_date": credential.achievement.conferral_date,
        "issuer_name": credential.issuer.name,
        "issuer_did": credential.issuer.id,
        "credential_id": credential.id,
        "schema": credential.mycelix_schema_id,
    })
    .to_string();

    let now = sys_time()?;

    // Generate a deterministic claim ID
    let claim_id = format!("ext:mycelix-identity:{}:{}", subject, now.as_micros());

    // Store the epistemic claim reference locally
    let claim_ref = EpistemicClaimReference {
        credential_id: credential.id.clone(),
        credential_hash: input.credential_action_hash.clone(),
        claim_id: claim_id.clone(),
        source_happ: "mycelix-identity".to_string(),
        subject: subject.clone(),
        predicate: predicate.clone(),
        object: object.clone(),
        epistemic_e: ACADEMIC_EPISTEMIC_E,
        epistemic_n: ACADEMIC_EPISTEMIC_N,
        epistemic_m: ACADEMIC_EPISTEMIC_M,
        published_at: now,
    };

    let claim_ref_hash = create_entry(EntryTypes::EpistemicClaimReference(claim_ref))?;

    // Link credential → epistemic claim
    let credential_anchor = anchor_for_credential(&credential.id)?;
    create_link(
        credential_anchor,
        claim_ref_hash.clone(),
        LinkTypes::CredentialToEpistemicClaim,
        (),
    )?;

    // Link subject → epistemic claim (for querying all claims by student)
    let subject_anchor = anchor_for_did(&subject)?;
    create_link(
        subject_anchor,
        claim_ref_hash.clone(),
        LinkTypes::SubjectToEpistemicClaim,
        (),
    )?;

    // Attempt cross-zome call to knowledge bridge if available.
    // This uses hdk::prelude::call() to the knowledge_bridge zome's
    // register_external_claim function. If the zome isn't installed
    // in the same DNA, the call fails gracefully and the local
    // reference is still stored for later bridge sync.
    let bridge_input = serde_json::json!({
        "source_happ": "mycelix-identity",
        "subject": subject,
        "predicate": predicate,
        "object": object,
        "epistemic_e": ACADEMIC_EPISTEMIC_E,
        "epistemic_n": ACADEMIC_EPISTEMIC_N,
        "epistemic_m": ACADEMIC_EPISTEMIC_M,
    });

    // Best-effort bridge call - don't fail if bridge zome unavailable
    match call(
        CallTargetCell::Local,
        ZomeName::from("knowledge_bridge"),
        FunctionName::from("register_external_claim"),
        None,
        bridge_input,
    ) {
        Ok(ZomeCallResponse::Ok(_)) => {
            debug!(
                "Successfully registered credential {} as epistemic claim in knowledge bridge",
                credential.id
            );
        }
        Ok(other) => {
            debug!(
                "Knowledge bridge call returned non-OK: {:?} (claim stored locally)",
                other
            );
        }
        Err(e) => {
            debug!(
                "Knowledge bridge not available: {} (claim stored locally for later sync)",
                e
            );
        }
    }

    Ok(PublishCredentialAsClaimOutput {
        claim_ref_hash,
        claim_id,
        epistemic_position: EpistemicPosition {
            empirical: ACADEMIC_EPISTEMIC_E,
            normative: ACADEMIC_EPISTEMIC_N,
            materiality: ACADEMIC_EPISTEMIC_M,
            label: "E3/N2/M3 - Cryptographic·Network·Immutable".to_string(),
        },
    })
}

/// Get epistemic claims for a credential
#[hdk_extern]
pub fn get_epistemic_claims_for_credential(
    credential_id: String,
) -> ExternResult<Vec<EpistemicClaimReference>> {
    let credential_anchor = anchor_for_credential(&credential_id)?;
    let links = get_links(
        LinkQuery::try_new(credential_anchor, LinkTypes::CredentialToEpistemicClaim)?,
        GetStrategy::default(),
    )?;

    let mut claims = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(claim_ref) = record
                .entry()
                .to_app_option::<EpistemicClaimReference>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                claims.push(claim_ref);
            }
        }
    }

    Ok(claims)
}

/// Get all epistemic claims for a subject (student DID)
#[hdk_extern]
pub fn get_epistemic_claims_for_subject(
    subject_did: String,
) -> ExternResult<Vec<EpistemicClaimReference>> {
    let subject_anchor = anchor_for_did(&subject_did)?;
    let links = get_links(
        LinkQuery::try_new(subject_anchor, LinkTypes::SubjectToEpistemicClaim)?,
        GetStrategy::default(),
    )?;

    let mut claims = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(claim_ref) = record
                .entry()
                .to_app_option::<EpistemicClaimReference>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                claims.push(claim_ref);
            }
        }
    }

    Ok(claims)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse degree name to DegreeType
fn parse_degree_type(degree_name: &str) -> DegreeType {
    let lower = degree_name.to_lowercase();

    if lower.contains("bachelor") || lower.contains("b.s.") || lower.contains("b.a.") {
        DegreeType::Bachelor
    } else if lower.contains("master") || lower.contains("m.s.") || lower.contains("m.a.") {
        DegreeType::Master
    } else if lower.contains("doctor") || lower.contains("ph.d") || lower.contains("phd") {
        DegreeType::Doctorate
    } else if lower.contains("associate") || lower.contains("a.s.") || lower.contains("a.a.") {
        DegreeType::Associate
    } else if lower.contains("certificate") {
        DegreeType::Certificate
    } else if lower.contains("diploma") || lower.contains("high school") {
        DegreeType::HighSchool
    } else if lower.contains("j.d.") || lower.contains("m.d.") || lower.contains("professional") {
        DegreeType::Professional
    } else {
        DegreeType::Diploma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_degree_type ---

    #[test]
    fn parse_bachelor_variants() {
        assert_eq!(
            parse_degree_type("Bachelor of Science"),
            DegreeType::Bachelor
        );
        assert_eq!(
            parse_degree_type("B.S. in Computer Science"),
            DegreeType::Bachelor
        );
        assert_eq!(
            parse_degree_type("B.A. English Literature"),
            DegreeType::Bachelor
        );
        assert_eq!(parse_degree_type("BACHELOR"), DegreeType::Bachelor);
    }

    #[test]
    fn parse_master_variants() {
        assert_eq!(parse_degree_type("Master of Arts"), DegreeType::Master);
        assert_eq!(parse_degree_type("M.S. Physics"), DegreeType::Master);
        assert_eq!(parse_degree_type("M.A. in History"), DegreeType::Master);
    }

    #[test]
    fn parse_doctorate_variants() {
        assert_eq!(
            parse_degree_type("Doctor of Philosophy"),
            DegreeType::Doctorate
        );
        assert_eq!(
            parse_degree_type("Ph.D. in Mathematics"),
            DegreeType::Doctorate
        );
        assert_eq!(
            parse_degree_type("PhD Computer Science"),
            DegreeType::Doctorate
        );
    }

    #[test]
    fn parse_associate() {
        assert_eq!(
            parse_degree_type("Associate of Science"),
            DegreeType::Associate
        );
        assert_eq!(parse_degree_type("A.S. Nursing"), DegreeType::Associate);
        assert_eq!(
            parse_degree_type("A.A. Liberal Arts"),
            DegreeType::Associate
        );
    }

    #[test]
    fn parse_certificate() {
        assert_eq!(
            parse_degree_type("Certificate in Data Analytics"),
            DegreeType::Certificate
        );
    }

    #[test]
    fn parse_high_school() {
        assert_eq!(
            parse_degree_type("High School Diploma"),
            DegreeType::HighSchool
        );
        assert_eq!(
            parse_degree_type("Diploma from High School"),
            DegreeType::HighSchool
        );
    }

    #[test]
    fn parse_professional() {
        assert_eq!(parse_degree_type("J.D. Law"), DegreeType::Professional);
        assert_eq!(parse_degree_type("M.D. Medicine"), DegreeType::Professional);
        assert_eq!(
            parse_degree_type("Professional Degree"),
            DegreeType::Professional
        );
    }

    #[test]
    fn parse_unknown_defaults_to_diploma() {
        assert_eq!(parse_degree_type("Something Random"), DegreeType::Diploma);
        assert_eq!(parse_degree_type(""), DegreeType::Diploma);
    }

    #[test]
    fn parse_case_insensitive() {
        assert_eq!(parse_degree_type("bachelor"), DegreeType::Bachelor);
        assert_eq!(parse_degree_type("MASTER"), DegreeType::Master);
        assert_eq!(parse_degree_type("doctorate"), DegreeType::Doctorate);
    }
}
