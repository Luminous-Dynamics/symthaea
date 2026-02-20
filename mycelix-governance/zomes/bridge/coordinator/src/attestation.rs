use super::*;

// =============================================================================
// AUTHENTICATED CONSCIOUSNESS ATTESTATION (V2)
// =============================================================================

/// Record an authenticated consciousness attestation (preferred over record_consciousness_snapshot).
///
/// The agent signs a message of the form:
///   `symthaea-consciousness-attestation:v1:{agent_did}:{consciousness_level:.6}:{cycle_id}:{captured_at_us}`
/// using their Holochain agent key. The signature is Ed25519-verified against
/// the caller's public key before the entry is committed.
#[hdk_extern]
pub fn record_consciousness_attestation(input: RecordConsciousnessAttestationInput) -> ExternResult<Record> {
    // Validate inputs
    if input.consciousness_level < 0.0 || input.consciousness_level > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Consciousness level must be between 0.0 and 1.0".into()
        )));
    }
    if input.signature.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Attestation signature must be non-empty".into()
        )));
    }
    if input.cycle_id == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cycle ID must be > 0".into()
        )));
    }

    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Reconstruct the signed message and verify Ed25519 signature.
    // Message format matches ConsciousnessAttestationRecord::sign_message() in Symthaea.
    let signed_message = format!(
        "symthaea-consciousness-attestation:v1:{}:{:.6}:{}:{}",
        agent_did, input.consciousness_level, input.cycle_id, input.captured_at_us,
    );
    let signature = Signature::from(
        <[u8; 64]>::try_from(input.signature.as_slice()).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest(
                "Signature must be exactly 64 bytes (Ed25519)".into()
            ))
        })?,
    );
    let valid = verify_signature_raw(
        agent_info.agent_initial_pubkey.clone(),
        signature,
        signed_message.into_bytes(),
    )?;
    if !valid {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Ed25519 signature verification failed — attestation rejected".into()
        )));
    }

    let attestation = ConsciousnessAttestation {
        agent_did: agent_did.clone(),
        consciousness_level: input.consciousness_level,
        cycle_id: input.cycle_id,
        captured_at: now,
        signature: input.signature,
        source: "symthaea".to_string(),
        consciousness_vector: input.consciousness_vector,
    };

    let action_hash = create_entry(&EntryTypes::ConsciousnessAttestation(attestation))?;

    // Link from agent to attestation
    let agent_anchor = format!("agent:{}", agent_did);
    create_entry(&EntryTypes::Anchor(Anchor(agent_anchor.clone())))?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToAttestations,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Attestation not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordConsciousnessAttestationInput {
    pub consciousness_level: f64,
    pub cycle_id: u64,
    /// Microseconds since Unix epoch when the consciousness level was captured.
    /// Must match the value used in the signed message.
    pub captured_at_us: u64,
    /// Ed25519 signature (64 bytes) over the attestation message.
    pub signature: Vec<u8>,
    /// Optional C-Vector (v2 attestations). Signature still binds to the scalar
    /// `consciousness_level` for backward-compatible verification.
    #[serde(default)]
    pub consciousness_vector: Option<ConsciousnessVectorEntry>,
}

/// Verify consciousness gate using attested consciousness level (preferred).
///
/// Falls back to legacy ConsciousnessSnapshot if no attestation exists.
/// Returns a v2 result with provenance tracking.
#[hdk_extern]
pub fn verify_consciousness_gate_v2(input: VerifyGateInput) -> ExternResult<GateVerificationResultV2> {
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);
    let required_consciousness = input.action_type.consciousness_gate();

    // Phase 1: Try to find latest ConsciousnessAttestation
    if let Some((consciousness_level, consciousness_vector, _record)) = get_latest_agent_attestation(&agent_did)? {
        // Use C-Vector composite if available, otherwise fall back to scalar
        let effective_level = match &consciousness_vector {
            Some(cv) => cv.composite(),
            None => consciousness_level,
        };
        let mut passed = effective_level >= required_consciousness;
        let mut failure_reason = if passed {
            None
        } else {
            Some(format!(
                "Attested consciousness {:.2} below gate {:.2} for {:?}",
                effective_level, required_consciousness, input.action_type
            ))
        };

        // Per-dimension C-Vector gates (checked in addition to composite gate)
        if passed {
            if let Some(cv) = &consciousness_vector {
                let config = get_current_consciousness_config()?;

                // Constitutional actions require minimum true_phi if configured
                if input.action_type == GovernanceActionType::Constitutional {
                    if let Some(min_phi) = config.min_true_phi_constitutional {
                        if cv.best_phi() < min_phi {
                            passed = false;
                            failure_reason = Some(format!(
                                "True phi {:.2} below constitutional minimum {:.2}",
                                cv.best_phi(), min_phi
                            ));
                        }
                    }
                }

                // Voting actions require minimum coherence if configured
                if matches!(input.action_type, GovernanceActionType::Voting | GovernanceActionType::Constitutional) {
                    if let Some(min_coh) = config.min_coherence_voting {
                        if let Some(coh) = cv.coherence {
                            if coh < min_coh {
                                passed = false;
                                failure_reason = Some(format!(
                                    "Coherence {:.2} below voting minimum {:.2}",
                                    coh, min_coh
                                ));
                            }
                        }
                    }
                }
            }
        }

        return Ok(GateVerificationResultV2 {
            passed,
            consciousness_level: Some(effective_level),
            required_consciousness,
            provenance: ConsciousnessProvenance::Attested,
            action_type: input.action_type,
            failure_reason,
        });
    }

    // Phase 2: Fall back to legacy ConsciousnessSnapshot
    if let Some((_record, snapshot)) = get_latest_agent_snapshot(&agent_did)? {
        let consciousness_level = snapshot.consciousness_level;
        let passed = consciousness_level >= required_consciousness;
        let failure_reason = if passed {
            None
        } else {
            Some(format!(
                "Snapshot consciousness {:.2} below gate {:.2} for {:?}",
                consciousness_level, required_consciousness, input.action_type
            ))
        };
        return Ok(GateVerificationResultV2 {
            passed,
            consciousness_level: Some(consciousness_level),
            required_consciousness,
            provenance: ConsciousnessProvenance::Snapshot,
            action_type: input.action_type,
            failure_reason,
        });
    }

    // Phase 3: No consciousness data available
    Ok(GateVerificationResultV2 {
        passed: false,
        consciousness_level: None,
        required_consciousness,
        provenance: ConsciousnessProvenance::Unavailable,
        action_type: input.action_type,
        failure_reason: Some("No consciousness data available".to_string()),
    })
}

/// Gate verification result with provenance tracking (v2).
#[derive(Serialize, Deserialize, Debug)]
pub struct GateVerificationResultV2 {
    pub passed: bool,
    /// None if no consciousness data available
    pub consciousness_level: Option<f64>,
    pub required_consciousness: f64,
    /// How the consciousness value was obtained
    pub provenance: ConsciousnessProvenance,
    pub action_type: GovernanceActionType,
    pub failure_reason: Option<String>,
}
