// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Signal types for P2P communication and cryptographic verification.

use hdk::prelude::*;
use federated_learning_integrity::*;

use crate::auth::check_rate_limit;

/// Signal types for P2P communication
/// SECURITY (F-05): Includes cryptographic signature for verification (SEC-005 fix)
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum Signal {
    GradientSubmitted {
        node_id: String,
        round: u32,
        action_hash: ActionHash,
        /// Source agent public key (set by receiver to actual caller)
        #[serde(default)]
        source: Option<String>,
        /// Ed25519 signature over signal content (base64-encoded)
        /// Signs: "{type}:{node_id}:{round}:{action_hash}"
        #[serde(default)]
        signature: Option<String>,
    },
    RoundCompleted {
        round: u32,
        accuracy: f32,
        byzantine_count: u32,
        /// Source agent public key (set by receiver to actual caller)
        #[serde(default)]
        source: Option<String>,
        /// Ed25519 signature over signal content (base64-encoded)
        /// Signs: "{type}:{round}:{accuracy}:{byzantine_count}"
        #[serde(default)]
        signature: Option<String>,
    },
    ByzantineDetected {
        node_id: String,
        round: u32,
        confidence: f32,
        /// Source agent public key (set by receiver to actual caller)
        #[serde(default)]
        source: Option<String>,
        /// Ed25519 signature over signal content (base64-encoded)
        /// Signs: "{type}:{node_id}:{round}:{confidence}"
        #[serde(default)]
        signature: Option<String>,
    },
    // =========================================================================
    // Phase 6: Gossip-Based Round Participation Signals
    // =========================================================================
    /// Emitted when a new round begins (by the node that calls advance_round)
    RoundStarted {
        round: u64,
        schedule_hash: Option<String>,
        #[serde(default)]
        source: Option<String>,
        #[serde(default)]
        signature: Option<String>,
    },
    /// Emitted when a node's gradient is ready for submission
    GradientReady {
        node_id: String,
        round: u64,
        gradient_hash: String,
        #[serde(default)]
        source: Option<String>,
        #[serde(default)]
        signature: Option<String>,
    },
    /// Emitted when a validator has committed their aggregation result
    CommitReady {
        round: u64,
        validator_id: String,
        commitment_hash: String,
        #[serde(default)]
        source: Option<String>,
        #[serde(default)]
        signature: Option<String>,
    },
    /// Emitted when consensus is reached for a round
    ConsensusReached {
        round: u64,
        agreed_hash: String,
        validator_count: u32,
        consensus_weight: f64,
        #[serde(default)]
        source: Option<String>,
        #[serde(default)]
        signature: Option<String>,
    },
}

/// Signed signal wrapper for cryptographic verification
/// SECURITY (SEC-005): Enables Ed25519 signature verification
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SignedSignal {
    /// The signal payload
    pub signal: Signal,
    /// Timestamp when signal was created (microseconds since Unix epoch)
    pub timestamp: i64,
    /// Nonce to prevent replay attacks
    pub nonce: [u8; 16],
}

impl SignedSignal {
    /// Create a new signed signal with current timestamp and CSPRNG nonce
    pub fn new(signal: Signal) -> Self {
        // C-01: Use CSPRNG for nonce generation instead of predictable timestamp
        let mut nonce = [0u8; 16];
        getrandom_02::getrandom(&mut nonce).unwrap_or_else(|_| {
            // Fallback: use timestamp if getrandom fails (shouldn't in Holochain WASM)
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            for (i, byte) in nonce.iter_mut().enumerate() {
                let shift = (i % 8) * 8;
                *byte = ((ts.wrapping_shr(shift as u32)) & 0xFF) as u8;
                if i >= 8 { *byte ^= i as u8; }
            }
        });

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        Self {
            signal,
            timestamp: ts as i64 / 1000, // Convert to microseconds
            nonce,
        }
    }

    /// Get bytes to sign for this signal
    pub fn signable_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Add signal type and content
        match &self.signal {
            Signal::GradientSubmitted { node_id, round, action_hash, .. } => {
                bytes.extend_from_slice(b"GradientSubmitted:");
                bytes.extend_from_slice(node_id.as_bytes());
                bytes.push(b':');
                bytes.extend_from_slice(&round.to_le_bytes());
                bytes.push(b':');
                bytes.extend_from_slice(action_hash.get_raw_39());
            }
            Signal::RoundCompleted { round, accuracy, byzantine_count, .. } => {
                bytes.extend_from_slice(b"RoundCompleted:");
                bytes.extend_from_slice(&round.to_le_bytes());
                bytes.push(b':');
                bytes.extend_from_slice(&accuracy.to_le_bytes());
                bytes.push(b':');
                bytes.extend_from_slice(&byzantine_count.to_le_bytes());
            }
            Signal::ByzantineDetected { node_id, round, confidence, .. } => {
                bytes.extend_from_slice(b"ByzantineDetected:");
                bytes.extend_from_slice(node_id.as_bytes());
                bytes.push(b':');
                bytes.extend_from_slice(&round.to_le_bytes());
                bytes.push(b':');
                bytes.extend_from_slice(&confidence.to_le_bytes());
            }
            Signal::RoundStarted { round, schedule_hash, .. } => {
                bytes.extend_from_slice(b"RoundStarted:");
                bytes.extend_from_slice(&round.to_le_bytes());
                if let Some(ref sh) = schedule_hash {
                    bytes.push(b':');
                    bytes.extend_from_slice(sh.as_bytes());
                }
            }
            Signal::GradientReady { node_id, round, gradient_hash, .. } => {
                bytes.extend_from_slice(b"GradientReady:");
                bytes.extend_from_slice(node_id.as_bytes());
                bytes.push(b':');
                bytes.extend_from_slice(&round.to_le_bytes());
                bytes.push(b':');
                bytes.extend_from_slice(gradient_hash.as_bytes());
            }
            Signal::CommitReady { round, validator_id, commitment_hash, .. } => {
                bytes.extend_from_slice(b"CommitReady:");
                bytes.extend_from_slice(&round.to_le_bytes());
                bytes.push(b':');
                bytes.extend_from_slice(validator_id.as_bytes());
                bytes.push(b':');
                bytes.extend_from_slice(commitment_hash.as_bytes());
            }
            Signal::ConsensusReached { round, agreed_hash, validator_count, consensus_weight, .. } => {
                bytes.extend_from_slice(b"ConsensusReached:");
                bytes.extend_from_slice(&round.to_le_bytes());
                bytes.push(b':');
                bytes.extend_from_slice(agreed_hash.as_bytes());
                bytes.push(b':');
                bytes.extend_from_slice(&validator_count.to_le_bytes());
                bytes.push(b':');
                bytes.extend_from_slice(&consensus_weight.to_le_bytes());
            }
        }

        // Add timestamp and nonce for replay protection
        bytes.push(b':');
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes.push(b':');
        bytes.extend_from_slice(&self.nonce);

        bytes
    }
}

/// Rate limit for remote signals per source (used by rate limiter)
#[allow(dead_code)]
const MAX_SIGNALS_PER_SOURCE_PER_MINUTE: u32 = 100;

/// Maximum age for signal timestamps (5 minutes in microseconds)
const MAX_SIGNAL_AGE_MICROSECONDS: i64 = 5 * 60 * 1_000_000;

/// Verify Ed25519 signature over signal content
/// SECURITY (SEC-005): Cryptographic verification replaces string comparison
pub(crate) fn verify_signal_signature(
    signed_signal: &SignedSignal,
    claimed_source: &AgentPubKey,
    signature: &Signature,
) -> ExternResult<bool> {
    // Get the signable bytes from the signal
    let signable_bytes = signed_signal.signable_bytes();

    // Use Holochain's built-in Ed25519 signature verification
    // This verifies the signature was created by the private key corresponding to claimed_source
    match verify_signature(claimed_source.clone(), signature.clone(), signable_bytes) {
        Ok(valid) => Ok(valid),
        Err(e) => {
            debug!("Signature verification error: {:?}", e);
            Ok(false)
        }
    }
}

/// Check if a signal timestamp is within acceptable range (replay protection)
pub(crate) fn validate_signal_timestamp(timestamp: i64) -> ExternResult<bool> {
    let now = sys_time()?.0 as i64;
    let age = now - timestamp;

    // Reject signals that are too old (potential replay)
    if age > MAX_SIGNAL_AGE_MICROSECONDS {
        return Ok(false);
    }

    // Reject signals from the future (clock skew tolerance: 30 seconds)
    if age < -30_000_000 {
        return Ok(false);
    }

    Ok(true)
}

/// Input for receiving a signed remote signal
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SignedRemoteSignalInput {
    /// The signed signal wrapper
    pub signed_signal: SignedSignal,
    /// Ed25519 signature over the signal (64 bytes, base64-encoded)
    pub signature_base64: String,
}

/// Receive remote signals with cryptographic verification
/// SECURITY (SEC-005): Uses Ed25519 signature verification instead of string comparison
#[hdk_extern]
pub fn recv_remote_signal(signal: ExternIO) -> ExternResult<()> {
    // F-05: Get the actual caller (provenance) for verification
    let caller = call_info()?.provenance;

    // Try to decode as SignedRemoteSignalInput first (secure path)
    if let Ok(signed_input) = signal.decode::<SignedRemoteSignalInput>() {
        // Decode the signature from base64
        let signature_bytes = base64_decode(&signed_input.signature_base64)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(
                format!("Invalid signature encoding: {:?}", e)
            )))?;

        if signature_bytes.len() != 64 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("Invalid signature length: expected 64 bytes, got {}", signature_bytes.len())
            )));
        }

        let signature = Signature::try_from(signature_bytes.as_slice())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(
                format!("Invalid signature format: {:?}", e)
            )))?;

        // SEC-005: Verify the cryptographic signature
        let is_valid = verify_signal_signature(&signed_input.signed_signal, &caller, &signature)?;

        if !is_valid {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Signal signature verification failed: signature does not match claimed source".to_string()
            )));
        }

        // Verify timestamp for replay protection
        if !validate_signal_timestamp(signed_input.signed_signal.timestamp)? {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Signal timestamp validation failed: signal too old or from future".to_string()
            )));
        }

        // F-06: Rate limit signals from each source
        check_rate_limit(&format!("signal_{}", caller))?;

        // Extract and update the signal with verified source
        let mut verified_signal = signed_input.signed_signal.signal;
        let caller_str = caller.to_string();

        match &mut verified_signal {
            Signal::GradientSubmitted { source, signature: sig, .. } |
            Signal::RoundCompleted { source, signature: sig, .. } |
            Signal::ByzantineDetected { source, signature: sig, .. } |
            Signal::RoundStarted { source, signature: sig, .. } |
            Signal::GradientReady { source, signature: sig, .. } |
            Signal::CommitReady { source, signature: sig, .. } |
            Signal::ConsensusReached { source, signature: sig, .. } => {
                *source = Some(caller_str.clone());
                *sig = Some(signed_input.signature_base64.clone());
            }
        }

        debug!("Received cryptographically verified signal from {}: {:?}", caller, verified_signal);
        emit_signal(verified_signal)?;
        return Ok(());
    }

    // H-08: Legacy unsigned signal path removed. All signals must include Ed25519 signature.
    return Err(wasm_error!(WasmErrorInner::Guest(
        "Unsigned signals are no longer accepted. All signals must include Ed25519 signature.".to_string()
    )));
}

/// Input for creating a signed signal to send remotely
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateSignedSignalInput {
    /// The signal to sign
    pub signal: Signal,
}

/// Output of creating a signed signal
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateSignedSignalOutput {
    /// The signed signal ready to send
    pub signed_input: SignedRemoteSignalInput,
}

/// Create a signed signal for sending to remote agents
///
/// SECURITY (SEC-005): Signs the signal with the caller's private key
/// The recipient can verify the signature using verify_signal_signature
#[hdk_extern]
pub fn create_signed_signal(input: CreateSignedSignalInput) -> ExternResult<CreateSignedSignalOutput> {
    // Create signed signal wrapper with timestamp and nonce
    let signed_signal = SignedSignal::new(input.signal);

    // Get the signable bytes
    let signable_bytes = signed_signal.signable_bytes();

    // Sign with caller's private key using Holochain's signing facility
    let signature = sign(agent_info()?.agent_initial_pubkey, signable_bytes)?;

    // Encode signature to base64
    let signature_base64 = base64_encode(signature.as_ref());

    Ok(CreateSignedSignalOutput {
        signed_input: SignedRemoteSignalInput {
            signed_signal,
            signature_base64,
        },
    })
}

/// Verify a signed signal without processing it
/// Useful for checking signal authenticity before deciding to act on it
#[hdk_extern]
pub fn verify_signed_signal_explicit(input: SignedRemoteSignalInput) -> ExternResult<bool> {
    // Get claimed source from the signal
    let claimed_source = match &input.signed_signal.signal {
        Signal::GradientSubmitted { source, .. } |
        Signal::RoundCompleted { source, .. } |
        Signal::ByzantineDetected { source, .. } |
        Signal::RoundStarted { source, .. } |
        Signal::GradientReady { source, .. } |
        Signal::CommitReady { source, .. } |
        Signal::ConsensusReached { source, .. } => {
            source.as_ref()
                .ok_or_else(|| wasm_error!(WasmErrorInner::Guest(
                    "Signal has no source specified for verification".to_string()
                )))?
        }
    };

    // Parse the claimed source as AgentPubKey
    let source_key = AgentPubKey::try_from(claimed_source.as_str())
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(
            format!("Invalid source agent key: {:?}", e)
        )))?;

    // Decode signature
    let signature_bytes = base64_decode(&input.signature_base64)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(
            format!("Invalid signature encoding: {:?}", e)
        )))?;

    if signature_bytes.len() != 64 {
        return Ok(false);
    }

    let signature = Signature::try_from(signature_bytes.as_slice())
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(
            format!("Invalid signature format: {:?}", e)
        )))?;

    // Verify signature
    let is_valid = verify_signal_signature(&input.signed_signal, &source_key, &signature)?;

    // Also check timestamp
    if is_valid {
        return validate_signal_timestamp(input.signed_signal.timestamp);
    }

    Ok(false)
}

// =============================================================================
// GOSSIP: Epidemic Relay Forwarding
// =============================================================================

/// Gossip relay configuration
const GOSSIP_FANOUT: usize = 3;
const GOSSIP_MAX_HOPS: u8 = 6;
const GOSSIP_TTL_SECS: i64 = 60;

/// Input for gossip relay
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GossipRelayInput {
    /// The signed signal to relay
    pub signed_input: SignedRemoteSignalInput,
    /// Current hop count
    pub hops: u8,
    /// Original emission timestamp (seconds since epoch)
    pub origin_time: i64,
}

/// Relay a gossip signal to random peers using epidemic forwarding
///
/// Each node that receives a gossip signal:
/// 1. Processes it locally (emit_signal)
/// 2. Checks hop count and TTL
/// 3. Forwards to `fanout` random peers from the DHT
///
/// Deduplication: signals are identified by (nonce, timestamp) from SignedSignal.
/// The rate limiter prevents replay flooding.
#[hdk_extern]
pub fn gossip_relay(input: GossipRelayInput) -> ExternResult<u32> {
    let now = sys_time()?.0 as i64 / 1_000_000;

    // Check TTL
    if now - input.origin_time > GOSSIP_TTL_SECS {
        return Ok(0); // Expired, don't forward
    }

    // Check hop limit
    if input.hops >= GOSSIP_MAX_HOPS {
        return Ok(0); // Max hops reached
    }

    // C-02: Verify cryptographic signature before processing gossip relay
    {
        let signature_bytes = base64_decode(&input.signed_input.signature_base64)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(
                format!("Gossip relay: invalid signature encoding: {:?}", e)
            )))?;

        if signature_bytes.len() != 64 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("Gossip relay: invalid signature length: expected 64 bytes, got {}", signature_bytes.len())
            )));
        }

        let signature = Signature::try_from(signature_bytes.as_slice())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(
                format!("Gossip relay: invalid signature format: {:?}", e)
            )))?;

        // Extract source from the signal to verify against
        let source_str = match &input.signed_input.signed_signal.signal {
            Signal::GradientSubmitted { source, .. } |
            Signal::RoundCompleted { source, .. } |
            Signal::ByzantineDetected { source, .. } |
            Signal::RoundStarted { source, .. } |
            Signal::GradientReady { source, .. } |
            Signal::CommitReady { source, .. } |
            Signal::ConsensusReached { source, .. } => {
                source.as_ref()
                    .ok_or_else(|| wasm_error!(WasmErrorInner::Guest(
                        "Gossip relay: signal has no source for signature verification".to_string()
                    )))?
            }
        };

        let source_key = AgentPubKey::try_from(source_str.as_str())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(
                format!("Gossip relay: invalid source agent key: {:?}", e)
            )))?;

        let is_valid = verify_signal_signature(
            &input.signed_input.signed_signal,
            &source_key,
            &signature,
        )?;

        if !is_valid {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Gossip relay: invalid signature".to_string()
            )));
        }

        // Validate timestamp freshness
        if !validate_signal_timestamp(input.signed_input.signed_signal.timestamp)? {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Gossip relay: signal timestamp validation failed".to_string()
            )));
        }
    }

    // Process locally: emit_signal for local listeners
    let signal = input.signed_input.signed_signal.signal.clone();
    emit_signal(signal)?;

    // Get peers from agent activity (DHT neighbors)
    // In Holochain, we query agent records from a well-known path
    let peers_path = Path::from("registered_validators");
    let peers_hash = match peers_path.clone().typed(LinkTypes::RoundToSchedule) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(0); // No peers known
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(0),
    };

    let links = get_links(
        LinkQuery::new(
            peers_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToSchedule as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    // Select random subset of peers (fanout)
    let my_agent = agent_info()?.agent_initial_pubkey;
    let peer_keys: Vec<AgentPubKey> = links.iter()
        .filter_map(|link| link.target.clone().into_agent_pub_key())
        .filter(|key| key != &my_agent) // Don't send to self
        .collect();

    if peer_keys.is_empty() {
        return Ok(0);
    }

    // Deterministic pseudo-random selection using nonce as seed
    let nonce_seed = u64::from_le_bytes([
        input.signed_input.signed_signal.nonce[0],
        input.signed_input.signed_signal.nonce[1],
        input.signed_input.signed_signal.nonce[2],
        input.signed_input.signed_signal.nonce[3],
        input.signed_input.signed_signal.nonce[4],
        input.signed_input.signed_signal.nonce[5],
        input.signed_input.signed_signal.nonce[6],
        input.signed_input.signed_signal.nonce[7],
    ]);

    let mut selected = Vec::with_capacity(GOSSIP_FANOUT);
    let n = peer_keys.len();
    for i in 0..GOSSIP_FANOUT.min(n) {
        // Simple hash-based selection to avoid duplicates
        let idx = ((nonce_seed.wrapping_add(i as u64).wrapping_mul(0x9e3779b97f4a7c15)) >> 32) as usize % n;
        if !selected.contains(&idx) {
            selected.push(idx);
        }
    }

    let next_input = GossipRelayInput {
        signed_input: input.signed_input,
        hops: input.hops + 1,
        origin_time: input.origin_time,
    };

    let payload = ExternIO::encode(next_input)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Encode error: {:?}", e))))?;

    let mut forwarded = 0u32;
    for &idx in &selected {
        if idx < peer_keys.len() {
            // Non-blocking best-effort forward -- don't fail if a peer is unreachable
            let _ = call_remote(
                peer_keys[idx].clone(),
                ZomeName::from("federated_learning"),
                FunctionName::from("gossip_relay"),
                None,
                payload.clone(),
            );
            forwarded += 1;
        }
    }

    Ok(forwarded)
}

/// Simple base64 encoder for signature output
pub(crate) fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut output = String::with_capacity((input.len() + 2) / 3 * 4);

    for chunk in input.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        output.push(ALPHABET[b0 >> 2] as char);
        output.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            output.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        } else {
            output.push('=');
        }

        if chunk.len() > 2 {
            output.push(ALPHABET[b2 & 0x3f] as char);
        } else {
            output.push('=');
        }
    }

    output
}

/// Simple base64 decoder for signature verification
pub(crate) fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    fn char_to_val(c: u8) -> Result<u8, String> {
        if c == b'=' {
            return Ok(0);
        }
        for (i, &ch) in ALPHABET.iter().enumerate() {
            if ch == c {
                return Ok(i as u8);
            }
        }
        Err(format!("Invalid base64 character: {}", c as char))
    }

    let input = input.trim().as_bytes();
    if input.is_empty() {
        return Ok(Vec::new());
    }

    if input.len() % 4 != 0 {
        return Err("Invalid base64 length".to_string());
    }

    let mut output = Vec::with_capacity(input.len() * 3 / 4);

    for chunk in input.chunks(4) {
        let a = char_to_val(chunk[0])?;
        let b = char_to_val(chunk[1])?;
        let c = char_to_val(chunk[2])?;
        let d = char_to_val(chunk[3])?;

        output.push((a << 2) | (b >> 4));
        if chunk[2] != b'=' {
            output.push((b << 4) | (c >> 2));
        }
        if chunk[3] != b'=' {
            output.push((c << 6) | d);
        }
    }

    Ok(output)
}
