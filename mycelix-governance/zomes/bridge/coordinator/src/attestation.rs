use super::*;

// =============================================================================
// AUTHENTICATED PHI ATTESTATION (V2)
// =============================================================================

/// Record an authenticated Phi attestation (preferred over record_consciousness_snapshot).
///
/// The agent signs a message of the form:
///   `symthaea-phi-attestation:v1:{agent_did}:{phi:.6}:{cycle_id}:{captured_at_us}`
/// using their Holochain agent key. The signature is Ed25519-verified against
/// the caller's public key before the entry is committed.
#[hdk_extern]
pub fn record_phi_attestation(input: RecordPhiAttestationInput) -> ExternResult<Record> {
    // Validate inputs
    if input.phi < 0.0 || input.phi > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Phi must be between 0.0 and 1.0".into()
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
    // Message format matches PhiAttestationRecord::sign_message() in Symthaea.
    let signed_message = format!(
        "symthaea-phi-attestation:v1:{}:{:.6}:{}:{}",
        agent_did, input.phi, input.cycle_id, input.captured_at_us,
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

    let attestation = PhiAttestation {
        agent_did: agent_did.clone(),
        phi: input.phi,
        cycle_id: input.cycle_id,
        captured_at: now,
        signature: input.signature,
        source: "symthaea".to_string(),
    };

    let action_hash = create_entry(&EntryTypes::PhiAttestation(attestation))?;

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
pub struct RecordPhiAttestationInput {
    pub phi: f64,
    pub cycle_id: u64,
    /// Microseconds since Unix epoch when the Phi was captured.
    /// Must match the value used in the signed message.
    pub captured_at_us: u64,
    /// Ed25519 signature (64 bytes) over the attestation message.
    pub signature: Vec<u8>,
}

/// Verify consciousness gate using attested Phi (preferred).
///
/// Falls back to legacy ConsciousnessSnapshot if no attestation exists.
/// Returns a v2 result with provenance tracking.
#[hdk_extern]
pub fn verify_consciousness_gate_v2(input: VerifyGateInput) -> ExternResult<GateVerificationResultV2> {
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);
    let required_phi = input.action_type.phi_threshold();

    // Phase 1: Try to find latest PhiAttestation
    if let Some((phi, _record)) = get_latest_agent_attestation(&agent_did)? {
        let passed = phi >= required_phi;
        let failure_reason = if passed {
            None
        } else {
            Some(format!(
                "Attested Φ {:.2} below threshold {:.2} for {:?}",
                phi, required_phi, input.action_type
            ))
        };
        return Ok(GateVerificationResultV2 {
            passed,
            phi: Some(phi),
            required_phi,
            provenance: PhiProvenance::Attested,
            action_type: input.action_type,
            failure_reason,
        });
    }

    // Phase 2: Fall back to legacy ConsciousnessSnapshot
    if let Some((_record, snapshot)) = get_latest_agent_snapshot(&agent_did)? {
        let phi = snapshot.phi;
        let passed = phi >= required_phi;
        let failure_reason = if passed {
            None
        } else {
            Some(format!(
                "Snapshot Φ {:.2} below threshold {:.2} for {:?}",
                phi, required_phi, input.action_type
            ))
        };
        return Ok(GateVerificationResultV2 {
            passed,
            phi: Some(phi),
            required_phi,
            provenance: PhiProvenance::Snapshot,
            action_type: input.action_type,
            failure_reason,
        });
    }

    // Phase 3: No Phi data available
    Ok(GateVerificationResultV2 {
        passed: false,
        phi: None,
        required_phi,
        provenance: PhiProvenance::Unavailable,
        action_type: input.action_type,
        failure_reason: Some("No consciousness data available".to_string()),
    })
}

/// Gate verification result with provenance tracking (v2).
#[derive(Serialize, Deserialize, Debug)]
pub struct GateVerificationResultV2 {
    pub passed: bool,
    /// None if no Phi data available
    pub phi: Option<f64>,
    pub required_phi: f64,
    /// How the Phi value was obtained
    pub provenance: PhiProvenance,
    pub action_type: GovernanceActionType,
    pub failure_reason: Option<String>,
}
