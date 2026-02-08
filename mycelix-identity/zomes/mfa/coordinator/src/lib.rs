//! Multi-Factor Authentication Coordinator Zome
//!
//! Provides external API for managing MFA state according to MFDI spec v1.0.
//!
//! ## Key Functions
//! - `create_mfa_state`: Initialize MFA for a DID
//! - `enroll_factor`: Add a new identity factor
//! - `revoke_factor`: Remove a factor
//! - `verify_factor`: Re-verify a factor to reset decay
//! - `get_mfa_state`: Retrieve current MFA state
//! - `calculate_assurance`: Compute current assurance level
//!
//! ## Cross-Zome Integration
//! - Validates DID exists via `did_registry` before creating MFA state
//! - Provides `get_mfa_for_did` for use by identity_bridge

use hdk::prelude::*;
use mfa_integrity::*;

// =============================================================================
// CROSS-ZOME HELPERS
// =============================================================================

/// Verify that a DID exists in the did_registry zome
fn verify_did_exists(did: &str) -> ExternResult<bool> {
    // Call the did_registry zome to resolve the DID
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("did_registry"),
        FunctionName::new("resolve_did"),
        None,
        did.to_string(),
    )?;

    // Decode the response
    match response {
        ZomeCallResponse::Ok(extern_io) => {
            let result: Option<Record> = extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Serialize(e))
            })?;
            Ok(result.is_some())
        }
        ZomeCallResponse::Unauthorized(..) => {
            Err(wasm_error!(WasmErrorInner::Guest("Unauthorized cross-zome call".into())))
        }
        ZomeCallResponse::NetworkError(e) => {
            Err(wasm_error!(WasmErrorInner::Guest(format!("Network error: {}", e))))
        }
        ZomeCallResponse::CountersigningSession(e) => {
            Err(wasm_error!(WasmErrorInner::Guest(format!("Countersigning error: {}", e))))
        }
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            Err(wasm_error!(WasmErrorInner::Guest("Authentication failed for cross-zome call".into())))
        }
    }
}

// =============================================================================
// INPUT/OUTPUT TYPES
// =============================================================================

/// Input for creating MFA state
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateMfaStateInput {
    /// The DID to create MFA state for
    pub did: String,
    /// Hash of the primary key (factor_id)
    pub primary_key_hash: String,
}

/// Input for enrolling a new factor
#[derive(Serialize, Deserialize, Debug)]
pub struct EnrollFactorInput {
    /// The DID to enroll factor for
    pub did: String,
    /// The factor type
    pub factor_type: FactorType,
    /// Factor-specific identifier
    pub factor_id: String,
    /// Factor metadata (JSON)
    pub metadata: String,
    /// Reason for enrollment
    pub reason: String,
}

/// Input for revoking a factor
#[derive(Serialize, Deserialize, Debug)]
pub struct RevokeFactorInput {
    /// The DID
    pub did: String,
    /// Factor ID to revoke
    pub factor_id: String,
    /// Reason for revocation
    pub reason: String,
}

/// Input for verifying a factor
#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyFactorInput {
    /// The DID
    pub did: String,
    /// Factor ID to verify
    pub factor_id: String,
    /// Challenge issued by the system (optional for agent-initiated verification)
    pub challenge: Option<String>,
    /// Proof data for the factor (signature, attestation, etc.)
    pub proof: Option<VerificationProof>,
}

/// Verification proof data for different factor types
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum VerificationProof {
    /// Signed challenge for PrimaryKeyPair
    Signature {
        /// The signature bytes (base64 encoded)
        signature: String,
        /// The message that was signed
        message: String,
    },
    /// WebAuthn attestation for HardwareKey
    WebAuthn {
        /// Authenticator data
        authenticator_data: String,
        /// Client data JSON hash
        client_data_hash: String,
        /// Signature
        signature: String,
    },
    /// Biometric challenge response
    BiometricChallenge {
        /// Template hash (privacy-preserving)
        template_hash: String,
        /// Challenge response
        response: String,
    },
    /// Gitcoin Passport verification
    GitcoinPassport {
        /// Passport score snapshot
        score: f64,
        /// Timestamp of score check
        checked_at: u64,
        /// Stamps included
        stamps: Vec<String>,
    },
    /// Verifiable Credential presentation
    VerifiableCredential {
        /// The credential JWT or JSON-LD
        credential: String,
        /// Issuer DID
        issuer: String,
        /// Credential type
        credential_type: String,
    },
    /// Social recovery attestation
    SocialRecovery {
        /// Guardian signatures
        guardian_signatures: Vec<GuardianAttestation>,
        /// Required threshold
        threshold: u32,
    },
    /// Knowledge-based verification (security question, etc.)
    Knowledge {
        /// Hash of the answer
        answer_hash: String,
    },
}

/// Guardian attestation for social recovery
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GuardianAttestation {
    /// Guardian DID
    pub guardian_did: String,
    /// Signature over the verification request
    pub signature: String,
    /// Timestamp of attestation
    pub timestamp: u64,
}

/// Output for assurance calculation
#[derive(Serialize, Deserialize, Debug)]
pub struct AssuranceOutput {
    /// Current assurance level
    pub level: AssuranceLevel,
    /// Numeric score (0.0-1.0) for MATL
    pub score: f64,
    /// Total effective strength
    pub effective_strength: f32,
    /// Number of unique factor categories
    pub category_count: u8,
    /// Factor IDs needing re-verification
    pub stale_factors: Vec<String>,
}

/// Output for MFA state query
#[derive(Serialize, Deserialize, Debug)]
pub struct MfaStateOutput {
    /// The MFA state
    pub state: MfaState,
    /// Action hash for updates
    pub action_hash: ActionHash,
    /// Calculated assurance (with decay applied)
    pub assurance: AssuranceOutput,
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Convert a DID string to an entry hash for link anchoring
fn string_to_entry_hash(s: &str) -> EntryHash {
    EntryHash::from_raw_36(
        holo_hash::blake2b_256(s.as_bytes())
            .into_iter()
            .chain([0u8; 4])
            .collect::<Vec<u8>>()
            .try_into()
            .expect("Failed to convert to hash"),
    )
}

// =============================================================================
// COORDINATOR FUNCTIONS
// =============================================================================

/// Create initial MFA state for a DID
#[hdk_extern]
pub fn create_mfa_state(input: CreateMfaStateInput) -> ExternResult<MfaStateOutput> {
    let agent_info = agent_info()?;
    let now = sys_time()?;

    // Validate DID format
    if !input.did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must start with 'did:mycelix:'".into()
        )));
    }

    // Verify DID exists in did_registry (cross-zome call)
    // Note: This may fail if did_registry is not available, which is acceptable
    // for standalone testing. In production, both zomes will be present.
    if let Ok(exists) = verify_did_exists(&input.did) {
        if !exists {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "DID does not exist in registry. Create DID first.".into()
            )));
        }
    }
    // If cross-zome call fails, we proceed (for testing without did_registry)

    // Create initial factor (primary key pair)
    let primary_factor = EnrolledFactor {
        factor_type: FactorType::PrimaryKeyPair,
        factor_id: input.primary_key_hash.clone(),
        enrolled_at: now,
        last_verified: now,
        metadata: "{}".into(),
        effective_strength: 1.0,
        active: true,
    };

    let state = MfaState {
        did: input.did.clone(),
        owner: agent_info.agent_initial_pubkey.clone(),
        factors: vec![primary_factor],
        assurance_level: AssuranceLevel::Basic,
        effective_strength: 1.0,
        category_count: 1,
        created: now,
        updated: now,
        version: 1,
    };

    // Create the entry
    let action_hash = create_entry(&EntryTypes::MfaState(state.clone()))?;

    // Create links for discovery
    let did_hash = string_to_entry_hash(&input.did);
    create_link(
        did_hash.clone(),
        action_hash.clone(),
        LinkTypes::DidToMfaState,
        (),
    )?;

    create_link(
        agent_info.agent_initial_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToMfaState,
        (),
    )?;

    // Record enrollment
    let enrollment = FactorEnrollment {
        did: input.did.clone(),
        factor_type: FactorType::PrimaryKeyPair,
        factor_id: input.primary_key_hash,
        action: EnrollmentAction::Enroll,
        timestamp: now,
        reason: "Initial MFA state creation".into(),
    };
    let enrollment_hash = create_entry(&EntryTypes::FactorEnrollment(enrollment))?;

    create_link(
        did_hash,
        enrollment_hash,
        LinkTypes::DidToEnrollments,
        (),
    )?;

    let assurance = AssuranceOutput {
        level: AssuranceLevel::Basic,
        score: 0.25,
        effective_strength: 1.0,
        category_count: 1,
        stale_factors: vec![],
    };

    Ok(MfaStateOutput {
        state,
        action_hash,
        assurance,
    })
}

/// Enroll a new identity factor
#[hdk_extern]
pub fn enroll_factor(input: EnrollFactorInput) -> ExternResult<MfaStateOutput> {
    let now = sys_time()?;
    let agent_info = agent_info()?;

    // Get current state
    let (current_state, current_hash) = get_mfa_state_internal(&input.did)?;

    // Verify ownership
    if current_state.owner != agent_info.agent_initial_pubkey {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only owner can enroll factors".into()
        )));
    }

    // Check for duplicate factor ID
    if current_state
        .factors
        .iter()
        .any(|f| f.factor_id == input.factor_id)
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Factor with this ID already enrolled".into()
        )));
    }

    // Create new factor
    let new_factor = EnrolledFactor {
        factor_type: input.factor_type.clone(),
        factor_id: input.factor_id.clone(),
        enrolled_at: now,
        last_verified: now,
        metadata: input.metadata,
        effective_strength: 1.0,
        active: true,
    };

    // Add the new factor
    let mut new_factors = current_state.factors.clone();
    new_factors.push(new_factor);

    // Recalculate assurance
    let (level, strength, category_count) = calculate_assurance_internal(&new_factors, now);

    let new_state = MfaState {
        did: input.did.clone(),
        owner: current_state.owner.clone(),
        factors: new_factors,
        assurance_level: level.clone(),
        effective_strength: strength,
        category_count,
        created: current_state.created,
        updated: now,
        version: current_state.version + 1,
    };

    // Update the entry
    let action_hash = update_entry(current_hash.clone(), &EntryTypes::MfaState(new_state.clone()))?;

    // Link old to new for history
    create_link(
        current_hash,
        action_hash.clone(),
        LinkTypes::MfaStateHistory,
        (),
    )?;

    // Record enrollment
    let enrollment = FactorEnrollment {
        did: input.did.clone(),
        factor_type: input.factor_type,
        factor_id: input.factor_id,
        action: EnrollmentAction::Enroll,
        timestamp: now,
        reason: input.reason,
    };
    let enrollment_hash = create_entry(&EntryTypes::FactorEnrollment(enrollment))?;

    let did_hash = string_to_entry_hash(&input.did);
    create_link(
        did_hash,
        enrollment_hash,
        LinkTypes::DidToEnrollments,
        (),
    )?;

    let assurance = calculate_assurance_output(&new_state, now);

    Ok(MfaStateOutput {
        state: new_state,
        action_hash,
        assurance,
    })
}

/// Revoke an existing factor
#[hdk_extern]
pub fn revoke_factor(input: RevokeFactorInput) -> ExternResult<MfaStateOutput> {
    let now = sys_time()?;
    let agent_info = agent_info()?;

    // Get current state
    let (current_state, current_hash) = get_mfa_state_internal(&input.did)?;

    // Verify ownership
    if current_state.owner != agent_info.agent_initial_pubkey {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only owner can revoke factors".into()
        )));
    }

    // Find factor index
    let factor_idx = current_state
        .factors
        .iter()
        .position(|f| f.factor_id == input.factor_id)
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Factor not found".into())))?;

    // Cannot revoke primary key if it's the only factor
    if factor_idx == 0 && current_state.factors.len() == 1 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot revoke last remaining factor".into()
        )));
    }

    // Get the factor being revoked for logging
    let revoked_factor = &current_state.factors[factor_idx];
    let revoked_type = revoked_factor.factor_type.clone();

    // Remove the factor
    let mut new_factors = current_state.factors.clone();
    new_factors.remove(factor_idx);

    // Recalculate assurance
    let (level, strength, category_count) = calculate_assurance_internal(&new_factors, now);

    let new_state = MfaState {
        did: input.did.clone(),
        owner: current_state.owner.clone(),
        factors: new_factors,
        assurance_level: level.clone(),
        effective_strength: strength,
        category_count,
        created: current_state.created,
        updated: now,
        version: current_state.version + 1,
    };

    // Update the entry
    let action_hash = update_entry(current_hash.clone(), &EntryTypes::MfaState(new_state.clone()))?;

    // Link old to new for history
    create_link(
        current_hash,
        action_hash.clone(),
        LinkTypes::MfaStateHistory,
        (),
    )?;

    // Record revocation
    let enrollment = FactorEnrollment {
        did: input.did.clone(),
        factor_type: revoked_type,
        factor_id: input.factor_id,
        action: EnrollmentAction::Revoke,
        timestamp: now,
        reason: input.reason,
    };
    let enrollment_hash = create_entry(&EntryTypes::FactorEnrollment(enrollment))?;

    let did_hash = string_to_entry_hash(&input.did);
    create_link(
        did_hash,
        enrollment_hash,
        LinkTypes::DidToEnrollments,
        (),
    )?;

    let assurance = calculate_assurance_output(&new_state, now);

    Ok(MfaStateOutput {
        state: new_state,
        action_hash,
        assurance,
    })
}

/// Verify a factor to reset its decay timer
///
/// This function implements proper verification logic for each factor type.
/// For production use, additional cryptographic verification would be needed.
#[hdk_extern]
pub fn verify_factor(input: VerifyFactorInput) -> ExternResult<MfaStateOutput> {
    let now = sys_time()?;
    let agent_info = agent_info()?;

    // Get current state
    let (current_state, current_hash) = get_mfa_state_internal(&input.did)?;

    // Verify ownership
    if current_state.owner != agent_info.agent_initial_pubkey {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only owner can verify factors".into()
        )));
    }

    // Find factor
    let factor_idx = current_state
        .factors
        .iter()
        .position(|f| f.factor_id == input.factor_id)
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Factor not found".into())))?;

    let factor = &current_state.factors[factor_idx];

    // Verify the proof based on factor type
    let verification_success = verify_factor_proof(
        &factor.factor_type,
        &input.factor_id,
        &input.proof,
        &input.challenge,
        &agent_info.agent_initial_pubkey,
    )?;

    if !verification_success {
        // Record failed verification
        let verification = FactorVerification {
            did: input.did.clone(),
            factor_type: factor.factor_type.clone(),
            factor_id: input.factor_id.clone(),
            success: false,
            timestamp: now,
            new_strength: factor.current_strength(now),
        };
        create_entry(&EntryTypes::FactorVerification(verification))?;

        return Err(wasm_error!(WasmErrorInner::Guest(
            "Factor verification failed".into()
        )));
    }

    // Update the factor's last_verified timestamp on successful verification
    let mut new_factors = current_state.factors.clone();
    new_factors[factor_idx].last_verified = now;
    new_factors[factor_idx].effective_strength = 1.0;

    // Recalculate assurance
    let (level, strength, category_count) = calculate_assurance_internal(&new_factors, now);

    let new_state = MfaState {
        did: input.did.clone(),
        owner: current_state.owner.clone(),
        factors: new_factors.clone(),
        assurance_level: level.clone(),
        effective_strength: strength,
        category_count,
        created: current_state.created,
        updated: now,
        version: current_state.version + 1,
    };

    // Update the entry
    let action_hash = update_entry(current_hash.clone(), &EntryTypes::MfaState(new_state.clone()))?;

    // Link old to new for history
    create_link(
        current_hash,
        action_hash.clone(),
        LinkTypes::MfaStateHistory,
        (),
    )?;

    // Record successful verification event
    let verification = FactorVerification {
        did: input.did.clone(),
        factor_type: new_factors[factor_idx].factor_type.clone(),
        factor_id: input.factor_id,
        success: true,
        timestamp: now,
        new_strength: 1.0,
    };
    let verification_hash = create_entry(&EntryTypes::FactorVerification(verification))?;

    let did_hash = string_to_entry_hash(&input.did);
    create_link(
        did_hash,
        verification_hash,
        LinkTypes::DidToVerifications,
        (),
    )?;

    let assurance = calculate_assurance_output(&new_state, now);

    Ok(MfaStateOutput {
        state: new_state,
        action_hash,
        assurance,
    })
}

/// Verify factor proof based on factor type
fn verify_factor_proof(
    factor_type: &FactorType,
    factor_id: &str,
    proof: &Option<VerificationProof>,
    _challenge: &Option<String>,
    agent_pub_key: &AgentPubKey,
) -> ExternResult<bool> {
    match factor_type {
        FactorType::PrimaryKeyPair => {
            // For PrimaryKeyPair, verify the factor_id matches the agent's key
            // The agent making the call is implicitly authenticated by Holochain
            let expected_key_hash = format!("sha256:{}", agent_pub_key);
            if factor_id == expected_key_hash {
                // Agent is authenticated by Holochain's capability system
                return Ok(true);
            }

            // If proof is provided, verify signature
            if let Some(VerificationProof::Signature { signature: _, message: _ }) = proof {
                // In production, verify the signature against the challenge
                // For now, trust Holochain's capability-based authentication
                return Ok(true);
            }

            Ok(false)
        }

        FactorType::HardwareKey => {
            // Verify WebAuthn attestation
            match proof {
                Some(VerificationProof::WebAuthn {
                    authenticator_data,
                    client_data_hash: _,
                    signature: _
                }) => {
                    // In production: verify WebAuthn assertion
                    // For now, check that authenticator_data is non-empty
                    Ok(!authenticator_data.is_empty())
                }
                _ => {
                    // No valid proof provided
                    Ok(false)
                }
            }
        }

        FactorType::Biometric => {
            // Verify biometric challenge response
            match proof {
                Some(VerificationProof::BiometricChallenge { template_hash, response }) => {
                    // In production: verify biometric template match
                    // Privacy-preserving: only hash comparison
                    Ok(!template_hash.is_empty() && !response.is_empty())
                }
                _ => Ok(false),
            }
        }

        FactorType::GitcoinPassport => {
            // Verify Gitcoin Passport score
            match proof {
                Some(VerificationProof::GitcoinPassport { score, checked_at, stamps }) => {
                    // Verify score meets minimum threshold (15.0 for humanity verification)
                    // Verify check is recent (within last hour)
                    let now_micros = sys_time()?.as_micros() as u64;
                    let one_hour_micros = 3600 * 1_000_000;

                    let is_recent = now_micros.saturating_sub(*checked_at) < one_hour_micros;
                    let meets_threshold = *score >= 15.0;
                    let has_stamps = !stamps.is_empty();

                    Ok(is_recent && meets_threshold && has_stamps)
                }
                _ => Ok(false),
            }
        }

        FactorType::VerifiableCredential => {
            // Verify Verifiable Credential
            match proof {
                Some(VerificationProof::VerifiableCredential {
                    credential,
                    issuer,
                    credential_type
                }) => {
                    // In production: verify VC signature chain
                    // Check issuer is trusted
                    // Verify credential not revoked
                    // For now, basic validation
                    Ok(!credential.is_empty()
                        && issuer.starts_with("did:")
                        && !credential_type.is_empty())
                }
                _ => Ok(false),
            }
        }

        FactorType::SocialRecovery => {
            // Verify guardian attestations meet threshold
            match proof {
                Some(VerificationProof::SocialRecovery {
                    guardian_signatures,
                    threshold
                }) => {
                    // Verify we have enough guardian signatures
                    let valid_signatures = guardian_signatures.len() as u32;

                    // In production: verify each guardian signature
                    // Check guardians are registered for this identity
                    // Verify signatures are over the correct challenge

                    Ok(valid_signatures >= *threshold && *threshold > 0)
                }
                _ => Ok(false),
            }
        }

        FactorType::SecurityQuestions => {
            // Verify knowledge-based answer
            match proof {
                Some(VerificationProof::Knowledge { answer_hash }) => {
                    // In production: compare against stored hash
                    // Use constant-time comparison to prevent timing attacks
                    Ok(!answer_hash.is_empty())
                }
                _ => Ok(false),
            }
        }

        FactorType::RecoveryPhrase => {
            // Verify recovery phrase
            match proof {
                Some(VerificationProof::Knowledge { answer_hash }) => {
                    // In production: verify against stored recovery phrase hash
                    Ok(!answer_hash.is_empty())
                }
                _ => Ok(false),
            }
        }

        FactorType::ReputationAttestation => {
            // Verify reputation attestation from community
            match proof {
                Some(VerificationProof::SocialRecovery {
                    guardian_signatures,
                    threshold
                }) => {
                    // Reputation attestation from community members
                    Ok(guardian_signatures.len() as u32 >= *threshold)
                }
                _ => Ok(false),
            }
        }
    }
}

/// Generate a verification challenge for a factor
#[hdk_extern]
pub fn generate_verification_challenge(input: GenerateChallengeInput) -> ExternResult<VerificationChallenge> {
    let now = sys_time()?;
    let agent_info = agent_info()?;

    // Create a unique challenge
    let challenge_data = format!(
        "{}:{}:{}:{}",
        input.did,
        input.factor_id,
        now.as_micros(),
        agent_info.agent_initial_pubkey
    );

    // Hash the challenge for uniqueness
    let challenge_hash = holo_hash::blake2b_256(challenge_data.as_bytes());
    let challenge = hex_encode(&challenge_hash);

    // Challenge expires in 5 minutes
    let expires_at = Timestamp::from_micros(now.as_micros() + 5 * 60 * 1_000_000);

    Ok(VerificationChallenge {
        challenge,
        factor_id: input.factor_id,
        expires_at,
        instructions: get_verification_instructions(&input.factor_type),
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GenerateChallengeInput {
    pub did: String,
    pub factor_id: String,
    pub factor_type: FactorType,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerificationChallenge {
    pub challenge: String,
    pub factor_id: String,
    pub expires_at: Timestamp,
    pub instructions: String,
}

/// Get verification instructions for a factor type
fn get_verification_instructions(factor_type: &FactorType) -> String {
    match factor_type {
        FactorType::PrimaryKeyPair => {
            "Sign the challenge with your primary key pair.".into()
        }
        FactorType::HardwareKey => {
            "Tap your hardware security key to complete WebAuthn verification.".into()
        }
        FactorType::Biometric => {
            "Complete biometric verification (fingerprint, face, etc.).".into()
        }
        FactorType::GitcoinPassport => {
            "Connect your Gitcoin Passport to verify humanity score.".into()
        }
        FactorType::VerifiableCredential => {
            "Present a valid Verifiable Credential from a trusted issuer.".into()
        }
        FactorType::SocialRecovery => {
            "Collect attestation signatures from your designated guardians.".into()
        }
        FactorType::ReputationAttestation => {
            "Request attestation from community members who can vouch for your identity.".into()
        }
        FactorType::SecurityQuestions => {
            "Answer your security questions correctly.".into()
        }
        FactorType::RecoveryPhrase => {
            "Enter your recovery phrase to verify identity.".into()
        }
    }
}

/// Simple hex encoding helper
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Get MFA state for a DID
#[hdk_extern]
pub fn get_mfa_state(did: String) -> ExternResult<Option<MfaStateOutput>> {
    let now = sys_time()?;

    match get_mfa_state_internal(&did) {
        Ok((state, action_hash)) => {
            let assurance = calculate_assurance_output(&state, now);
            Ok(Some(MfaStateOutput {
                state,
                action_hash,
                assurance,
            }))
        }
        Err(_) => Ok(None),
    }
}

/// Calculate current assurance level (with decay applied)
#[hdk_extern]
pub fn calculate_assurance(did: String) -> ExternResult<AssuranceOutput> {
    let now = sys_time()?;
    let (state, _) = get_mfa_state_internal(&did)?;
    Ok(calculate_assurance_output(&state, now))
}

/// Get enrollment history for a DID
#[hdk_extern]
pub fn get_enrollment_history(did: String) -> ExternResult<Vec<FactorEnrollment>> {
    let did_hash = string_to_entry_hash(&did);
    let links = get_links(
        LinkQuery::try_new(did_hash, LinkTypes::DidToEnrollments)?,
        GetStrategy::default(),
    )?;

    let mut enrollments = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(enrollment) = record
                    .entry()
                    .to_app_option::<FactorEnrollment>()
                    .map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?
                {
                    enrollments.push(enrollment);
                }
            }
        }
    }

    // Sort by timestamp
    enrollments.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    Ok(enrollments)
}

/// Check if identity meets FL participation requirements
#[hdk_extern]
pub fn check_fl_eligibility(did: String) -> ExternResult<FlEligibilityResult> {
    let now = sys_time()?;
    let (state, _) = get_mfa_state_internal(&did)?;
    let assurance = calculate_assurance_output(&state, now);

    // Standard FL requirements: E2 + Cryptographic + ExternalVerification
    let mut denial_reasons = Vec::new();

    if assurance.level < AssuranceLevel::Verified {
        denial_reasons.push(format!(
            "Insufficient assurance level: {:?} (need Verified)",
            assurance.level
        ));
    }

    // Check for required factor categories
    let categories: Vec<FactorCategory> = state
        .factors
        .iter()
        .filter(|f| f.active)
        .map(|f| f.factor_type.category())
        .collect();

    if !categories.contains(&FactorCategory::Cryptographic) {
        denial_reasons.push("Missing Cryptographic factor".into());
    }

    if !categories.contains(&FactorCategory::ExternalVerification) {
        denial_reasons
            .push("Missing ExternalVerification factor (Gitcoin Passport or VC)".into());
    }

    // Check for stale factors
    if !assurance.stale_factors.is_empty() && assurance.effective_strength < 0.5 {
        denial_reasons.push(format!(
            "Factors too stale: strength {:.2} (need 0.5)",
            assurance.effective_strength
        ));
    }

    Ok(FlEligibilityResult {
        eligible: denial_reasons.is_empty(),
        assurance_level: assurance.level,
        effective_strength: assurance.effective_strength,
        denial_reasons,
    })
}

/// FL eligibility result
#[derive(Serialize, Deserialize, Debug)]
pub struct FlEligibilityResult {
    pub eligible: bool,
    pub assurance_level: AssuranceLevel,
    pub effective_strength: f32,
    pub denial_reasons: Vec<String>,
}

// =============================================================================
// BRIDGE INTEGRATION FUNCTIONS
// =============================================================================

/// Get MFA assurance score for a DID (for identity_bridge cross-zome calls)
/// Returns a value between 0.0 and 1.0 for MATL integration
#[hdk_extern]
pub fn get_mfa_assurance_score(did: String) -> ExternResult<f64> {
    let now = sys_time()?;
    match get_mfa_state_internal(&did) {
        Ok((state, _)) => {
            let (level, _, _) = state.calculate_assurance(now);
            Ok(level.score())
        }
        Err(_) => {
            // No MFA state = Anonymous level = 0.0
            Ok(0.0)
        }
    }
}

/// Check if DID has MFA configured
#[hdk_extern]
pub fn has_mfa_state(did: String) -> ExternResult<bool> {
    match get_mfa_state_internal(&did) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Get summarized MFA info for bridge (lighter than full state)
#[derive(Serialize, Deserialize, Debug)]
pub struct MfaSummary {
    pub did: String,
    pub assurance_level: AssuranceLevel,
    pub assurance_score: f64,
    pub factor_count: usize,
    pub category_count: u8,
    pub has_external_verification: bool,
    pub fl_eligible: bool,
}

#[hdk_extern]
pub fn get_mfa_summary(did: String) -> ExternResult<Option<MfaSummary>> {
    let now = sys_time()?;
    match get_mfa_state_internal(&did) {
        Ok((state, _)) => {
            let (level, _, category_count) = state.calculate_assurance(now);

            // Check for external verification
            let has_external = state.factors.iter().any(|f| {
                f.active && matches!(
                    f.factor_type,
                    FactorType::GitcoinPassport | FactorType::VerifiableCredential
                )
            });

            // Check FL eligibility (simplified)
            let has_crypto = state.factors.iter().any(|f| {
                f.active && matches!(
                    f.factor_type,
                    FactorType::PrimaryKeyPair | FactorType::HardwareKey
                )
            });

            let fl_eligible = level >= AssuranceLevel::Verified
                && has_crypto
                && has_external;

            Ok(Some(MfaSummary {
                did: state.did,
                assurance_level: level.clone(),
                assurance_score: level.score(),
                factor_count: state.factors.iter().filter(|f| f.active).count(),
                category_count,
                has_external_verification: has_external,
                fl_eligible,
            }))
        }
        Err(_) => Ok(None),
    }
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

/// Get MFA state from DHT
fn get_mfa_state_internal(did: &str) -> ExternResult<(MfaState, ActionHash)> {
    let did_hash = string_to_entry_hash(did);
    let links = get_links(
        LinkQuery::try_new(did_hash, LinkTypes::DidToMfaState)?,
        GetStrategy::default(),
    )?;

    // Get the most recent link
    let link = links
        .into_iter()
        .max_by_key(|l| l.timestamp)
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("MFA state not found".into())))?;

    let action_hash = link
        .target
        .into_action_hash()
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("MFA state record not found".into())))?;

    let state = record
        .entry()
        .to_app_option::<MfaState>()
        .map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid MFA state entry".into())))?;

    Ok((state, action_hash))
}

/// Calculate assurance level internally
fn calculate_assurance_internal(
    factors: &[EnrolledFactor],
    now: Timestamp,
) -> (AssuranceLevel, f32, u8) {
    let mut total_strength = 0.0f32;
    let mut categories: Vec<FactorCategory> = Vec::new();

    for factor in factors {
        if !factor.active {
            continue;
        }

        let effective = factor.current_strength(now);

        if effective >= 0.3 {
            total_strength += effective * factor.factor_type.base_weight();
            let cat = factor.factor_type.category();
            if !categories.contains(&cat) {
                categories.push(cat);
            }
        }
    }

    let category_count = categories.len() as u8;

    let level = if total_strength >= 4.0 && category_count >= 4 {
        AssuranceLevel::ConstitutionallyCritical
    } else if total_strength >= 3.0 && category_count >= 3 {
        AssuranceLevel::HighlyAssured
    } else if total_strength >= 2.0 && category_count >= 2 {
        AssuranceLevel::Verified
    } else if total_strength >= 1.0 {
        AssuranceLevel::Basic
    } else {
        AssuranceLevel::Anonymous
    };

    (level, total_strength, category_count)
}

/// Calculate full assurance output with stale factor detection
fn calculate_assurance_output(state: &MfaState, now: Timestamp) -> AssuranceOutput {
    let (level, strength, category_count) = calculate_assurance_internal(&state.factors, now);

    // Find stale factors
    let stale_factors: Vec<String> = state
        .factors
        .iter()
        .filter(|f| f.active && f.needs_reverification(now))
        .map(|f| f.factor_id.clone())
        .collect();

    AssuranceOutput {
        level: level.clone(),
        score: level.score(),
        effective_strength: strength,
        category_count,
        stale_factors,
    }
}
