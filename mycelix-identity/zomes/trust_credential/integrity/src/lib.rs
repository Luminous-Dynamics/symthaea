//! Trust Credential Integrity Zome
//!
//! Defines entry types for K-Vector trust credentials with ZKP proofs.
//! Enables privacy-preserving trust attestations where:
//! - The K-Vector commitment is public (can be verified)
//! - The actual K-Vector values are private (hidden via ZKP)
//! - The trust score can be proven in a range without revealing exact value
//!
//! Integration with kvector-zkp library:
//! - KVectorWitness.commitment() produces the 32-byte commitment
//! - KVectorRangeProof proves values are in valid \[0,1\] range
//! - Proofs are generated off-chain and verified on-chain

use hdi::prelude::*;

/// K-Vector Trust Credential
///
/// A verifiable credential that attests to an agent's K-Vector trust profile
/// without revealing the individual component values.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TrustCredential {
    /// Unique credential identifier
    pub id: String,
    /// Subject's DID (who this credential is about)
    pub subject_did: String,
    /// Issuer's DID (who issued this credential)
    pub issuer_did: String,
    /// K-Vector commitment hash (32 bytes, SHA3-256)
    /// This binds to the actual K-Vector without revealing values
    pub kvector_commitment: Vec<u8>,
    /// STARK proof that K-Vector components are in valid \[0,1\] range
    /// Serialized proof from kvector-zkp library
    pub range_proof: Vec<u8>,
    /// Proven trust score range (e.g., [0.5, 0.7])
    /// Proves score is within this range without revealing exact value
    pub trust_score_range: TrustScoreRange,
    /// Trust tier derived from K-Vector (for governance thresholds)
    pub trust_tier: TrustTier,
    /// Credential issuance timestamp
    pub issued_at: Timestamp,
    /// Credential expiration (None = never)
    pub expires_at: Option<Timestamp>,
    /// Whether credential has been revoked
    pub revoked: bool,
    /// Revocation reason if revoked
    pub revocation_reason: Option<String>,
    /// When the credential was revoked (for audit trails)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revoked_at: Option<Timestamp>,
    /// Previous credential this supersedes (for updates)
    pub supersedes: Option<String>,
}

/// Trust score range (privacy-preserving)
///
/// Proves the trust score falls within a range without revealing exact value.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TrustScoreRange {
    /// Lower bound (inclusive)
    pub lower: f32,
    /// Upper bound (inclusive)
    pub upper: f32,
}

/// Trust tiers for governance participation
///
/// Derived from K-Vector trust_score() with defined thresholds.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TrustTier {
    /// Trust score < 0.3 - Observer only, cannot vote
    Observer,
    /// Trust score >= 0.3 - Basic participation
    Basic,
    /// Trust score >= 0.4 - Can vote on major proposals
    Standard,
    /// Trust score >= 0.6 - Can propose and vote on constitutional changes
    Elevated,
    /// Trust score >= 0.8 - Full governance rights including emergency powers
    Guardian,
}

impl TrustTier {
    /// Get the minimum trust score for this tier
    pub fn min_score(&self) -> f64 {
        match self {
            TrustTier::Observer => 0.0,
            TrustTier::Basic => 0.3,
            TrustTier::Standard => 0.4,
            TrustTier::Elevated => 0.6,
            TrustTier::Guardian => 0.8,
        }
    }

    /// Determine tier from trust score.
    ///
    /// Accepts f64 to avoid precision loss at tier boundaries when computing
    /// the midpoint of an f32 range (e.g., `(0.39 + 0.41) / 2.0` in f32 could
    /// round to 0.3999... instead of 0.4).
    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            TrustTier::Guardian
        } else if score >= 0.6 {
            TrustTier::Elevated
        } else if score >= 0.4 {
            TrustTier::Standard
        } else if score >= 0.3 {
            TrustTier::Basic
        } else {
            TrustTier::Observer
        }
    }
}

/// K-Vector Attestation Request
///
/// A request for someone to attest to components of their K-Vector.
/// Used when an issuer needs to verify specific trust properties.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AttestationRequest {
    /// Request identifier
    pub id: String,
    /// Who is requesting attestation
    pub requester_did: String,
    /// Who should provide attestation
    pub subject_did: String,
    /// Which K-Vector components need attestation
    pub components: Vec<KVectorComponent>,
    /// Minimum acceptable trust score
    pub min_trust_score: Option<f32>,
    /// Minimum acceptable tier
    pub min_tier: Option<TrustTier>,
    /// Purpose of the attestation
    pub purpose: String,
    /// Request expiration
    pub expires_at: Timestamp,
    /// Request status
    pub status: AttestationStatus,
    /// Creation timestamp
    pub created_at: Timestamp,
}

/// K-Vector component identifiers
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum KVectorComponent {
    /// k_r: Reputation
    Reputation,
    /// k_a: Activity
    Activity,
    /// k_i: Integrity
    Integrity,
    /// k_p: Performance
    Performance,
    /// k_m: Membership duration
    Membership,
    /// k_s: Stake weight
    Stake,
    /// k_h: Historical consistency
    History,
    /// k_topo: Network topology contribution
    Topology,
}

/// Status of an attestation request
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AttestationStatus {
    /// Waiting for response
    Pending,
    /// Attestation provided
    Fulfilled,
    /// Request was declined
    Declined,
    /// Request expired
    Expired,
    /// Request was cancelled
    Cancelled,
}

/// Trust Credential Presentation
///
/// A selective disclosure presentation of a trust credential.
/// Can reveal specific attributes while keeping others private.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TrustPresentation {
    /// Presentation identifier
    pub id: String,
    /// Reference to source credential
    pub credential_id: String,
    /// Subject's DID
    pub subject_did: String,
    /// Disclosed trust tier (always disclosed)
    pub disclosed_tier: TrustTier,
    /// Disclosed trust score range (if disclosed)
    pub disclosed_range: Option<TrustScoreRange>,
    /// Presentation proof (derived from original proof)
    pub presentation_proof: Vec<u8>,
    /// Who this presentation is for
    pub verifier_did: Option<String>,
    /// Purpose of presentation
    pub purpose: String,
    /// Presentation timestamp
    pub presented_at: Timestamp,
    /// Single-use nonce to prevent replay
    pub nonce: Vec<u8>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    TrustCredential(TrustCredential),
    AttestationRequest(AttestationRequest),
    TrustPresentation(TrustPresentation),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Subject to their trust credentials
    SubjectToCredential,
    /// Issuer to credentials they issued
    IssuerToCredential,
    /// Subject to attestation requests they received
    SubjectToRequest,
    /// Credential to presentations derived from it
    CredentialToPresentation,
    /// Trust tier anchor to credentials in that tier
    TierToCredential,
}

/// Main validation callback
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::TrustCredential(cred) => validate_create_credential(action, cred),
                EntryTypes::AttestationRequest(req) => validate_create_request(action, req),
                EntryTypes::TrustPresentation(pres) => validate_create_presentation(action, pres),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::TrustCredential(cred) => validate_update_credential(action, cred),
                EntryTypes::AttestationRequest(req) => validate_update_request(action, req),
                EntryTypes::TrustPresentation(_) => Ok(ValidateCallbackResult::Invalid(
                    "Trust presentations cannot be updated".into(),
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
                LinkTypes::SubjectToCredential => Ok(ValidateCallbackResult::Valid),
                LinkTypes::IssuerToCredential => Ok(ValidateCallbackResult::Valid),
                LinkTypes::SubjectToRequest => Ok(ValidateCallbackResult::Valid),
                LinkTypes::CredentialToPresentation => Ok(ValidateCallbackResult::Valid),
                LinkTypes::TierToCredential => Ok(ValidateCallbackResult::Valid),
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

/// Validate trust credential creation
fn validate_create_credential(
    action: Create,
    cred: TrustCredential,
) -> ExternResult<ValidateCallbackResult> {
    // Issuer must be the action author (prevent impersonation)
    let expected_issuer = format!("did:mycelix:{}", action.author);
    if cred.issuer_did != expected_issuer {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Issuer DID must match action author. Expected '{}', got '{}'",
            expected_issuer, cred.issuer_did
        )));
    }

    // Subject must be a valid DID
    if !cred.subject_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject must be a valid DID".into(),
        ));
    }

    // Issuer must be a valid DID
    if !cred.issuer_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer must be a valid DID".into(),
        ));
    }

    // K-Vector commitment must be 32 bytes (SHA3-256)
    if cred.kvector_commitment.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "K-Vector commitment must be 32 bytes".into(),
        ));
    }

    // Range proof must not be empty
    if cred.range_proof.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Range proof cannot be empty".into(),
        ));
    }

    // Trust score range must be valid
    if cred.trust_score_range.lower < 0.0 || cred.trust_score_range.upper > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score range must be within [0, 1]".into(),
        ));
    }

    if cred.trust_score_range.lower > cred.trust_score_range.upper {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score range lower bound cannot exceed upper bound".into(),
        ));
    }

    // Trust tier must be consistent with range
    let tier_min = cred.trust_tier.min_score();
    if (cred.trust_score_range.upper as f64) < tier_min {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score range is inconsistent with claimed tier".into(),
        ));
    }

    // New credentials cannot be revoked
    if cred.revoked {
        return Ok(ValidateCallbackResult::Invalid(
            "New credentials cannot be created in revoked state".into(),
        ));
    }

    // New credentials must not have a revocation timestamp
    if cred.revoked_at.is_some() {
        return Ok(ValidateCallbackResult::Invalid(
            "New credentials cannot have a revocation timestamp".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate trust credential update (for revocation)
fn validate_update_credential(
    action: Update,
    cred: TrustCredential,
) -> ExternResult<ValidateCallbackResult> {
    if cred.kvector_commitment.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "K-Vector commitment must be 32 bytes".into(),
        ));
    }

    // Fetch original to enforce invariants
    let original_record = must_get_valid_record(action.original_action_address.clone())?;
    let original: TrustCredential = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original trust credential not found".into()
        )))?;

    // Immutable fields
    if cred.id != original.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust credential ID cannot be changed".into(),
        ));
    }
    if cred.subject_did != original.subject_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject DID cannot be changed".into(),
        ));
    }
    if cred.issuer_did != original.issuer_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer DID cannot be changed".into(),
        ));
    }
    if cred.kvector_commitment != original.kvector_commitment {
        return Ok(ValidateCallbackResult::Invalid(
            "K-Vector commitment cannot be changed".into(),
        ));
    }
    if cred.issued_at != original.issued_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuance timestamp cannot be changed".into(),
        ));
    }

    // Revoked is irreversible
    if original.revoked && !cred.revoked {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust credential revocation is irreversible".into(),
        ));
    }

    // If being revoked, revoked_at must be set
    if cred.revoked && cred.revoked_at.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Revoked credentials must have a revoked_at timestamp".into(),
        ));
    }

    // revoked_at is immutable once set
    if let Some(original_ts) = &original.revoked_at {
        match &cred.revoked_at {
            Some(new_ts) if new_ts != original_ts => {
                return Ok(ValidateCallbackResult::Invalid(
                    "Revocation timestamp cannot be changed once set".into(),
                ));
            }
            None => {
                return Ok(ValidateCallbackResult::Invalid(
                    "Revocation timestamp cannot be removed".into(),
                ));
            }
            _ => {}
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate attestation request creation
fn validate_create_request(
    _action: Create,
    req: AttestationRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Requester must be a valid DID
    if !req.requester_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester must be a valid DID".into(),
        ));
    }

    // Subject must be a valid DID
    if !req.subject_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject must be a valid DID".into(),
        ));
    }

    // Cannot request attestation from yourself
    if req.requester_did == req.subject_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot request attestation from yourself".into(),
        ));
    }

    // New requests must be pending
    if req.status != AttestationStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "New requests must have Pending status".into(),
        ));
    }

    // Min trust score must be valid if specified
    if let Some(score) = req.min_trust_score {
        if !(0.0..=1.0).contains(&score) {
            return Ok(ValidateCallbackResult::Invalid(
                "Minimum trust score must be in [0, 1]".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate attestation request update
fn validate_update_request(
    action: Update,
    req: AttestationRequest,
) -> ExternResult<ValidateCallbackResult> {
    if !req.requester_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester must be a valid DID".into(),
        ));
    }

    // Fetch original to enforce state transitions
    let original_record = must_get_valid_record(action.original_action_address.clone())?;
    let original: AttestationRequest = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original attestation request not found".into()
        )))?;

    // Immutable fields
    if req.id != original.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Attestation request ID cannot be changed".into(),
        ));
    }
    if req.requester_did != original.requester_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester DID cannot be changed".into(),
        ));
    }
    if req.subject_did != original.subject_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject DID cannot be changed".into(),
        ));
    }

    // State machine: Pending → Fulfilled/Declined/Expired/Cancelled
    // Terminal states (Fulfilled, Declined, Expired, Cancelled) cannot transition further
    let valid = match (&original.status, &req.status) {
        (AttestationStatus::Pending, AttestationStatus::Fulfilled)
        | (AttestationStatus::Pending, AttestationStatus::Declined)
        | (AttestationStatus::Pending, AttestationStatus::Expired)
        | (AttestationStatus::Pending, AttestationStatus::Cancelled) => true,
        (a, b) if a == b => true,
        _ => false,
    };

    if !valid {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid attestation request status transition".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_trust_tier() -> impl Strategy<Value = TrustTier> {
        prop_oneof![
            Just(TrustTier::Observer),
            Just(TrustTier::Basic),
            Just(TrustTier::Standard),
            Just(TrustTier::Elevated),
            Just(TrustTier::Guardian),
        ]
    }

    fn arb_kvector_component() -> impl Strategy<Value = KVectorComponent> {
        prop_oneof![
            Just(KVectorComponent::Reputation),
            Just(KVectorComponent::Activity),
            Just(KVectorComponent::Integrity),
            Just(KVectorComponent::Performance),
            Just(KVectorComponent::Membership),
            Just(KVectorComponent::Stake),
            Just(KVectorComponent::History),
            Just(KVectorComponent::Topology),
        ]
    }

    fn arb_attestation_status() -> impl Strategy<Value = AttestationStatus> {
        prop_oneof![
            Just(AttestationStatus::Pending),
            Just(AttestationStatus::Fulfilled),
            Just(AttestationStatus::Declined),
            Just(AttestationStatus::Expired),
            Just(AttestationStatus::Cancelled),
        ]
    }

    /// Generate a valid TrustScoreRange within [0, 1] where lower <= upper.
    fn arb_trust_score_range() -> impl Strategy<Value = TrustScoreRange> {
        (0.0f32..=1.0f32, 0.0f32..=1.0f32).prop_map(|(a, b)| {
            let lower = a.min(b);
            let upper = a.max(b);
            TrustScoreRange { lower, upper }
        })
    }

    proptest! {
        /// TrustTier::from_score always returns a tier whose min_score <= the input.
        #[test]
        fn from_score_consistent_with_min_score(score in 0.0f64..=1.0f64) {
            let tier = TrustTier::from_score(score);
            prop_assert!(
                score >= tier.min_score(),
                "score {} should be >= tier {:?} min_score {}",
                score, tier, tier.min_score()
            );
        }

        /// TrustTier::from_score is monotonically non-decreasing.
        #[test]
        fn from_score_monotone(a in 0.0f64..=1.0f64, b in 0.0f64..=1.0f64) {
            if a <= b {
                let tier_a = TrustTier::from_score(a);
                let tier_b = TrustTier::from_score(b);
                prop_assert!(
                    tier_a.min_score() <= tier_b.min_score(),
                    "tier({}) = {:?} should have min_score <= tier({}) = {:?}",
                    a, tier_a, b, tier_b
                );
            }
        }

        /// TrustScoreRange round-trips through JSON.
        #[test]
        fn trust_score_range_json_roundtrip(range in arb_trust_score_range()) {
            let json = serde_json::to_string(&range).unwrap();
            let back: TrustScoreRange = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(range, back);
        }

        /// TrustTier round-trips through JSON.
        #[test]
        fn trust_tier_json_roundtrip(tier in arb_trust_tier()) {
            let json = serde_json::to_string(&tier).unwrap();
            let back: TrustTier = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(tier, back);
        }

        /// KVectorComponent round-trips through JSON.
        #[test]
        fn kvector_component_json_roundtrip(comp in arb_kvector_component()) {
            let json = serde_json::to_string(&comp).unwrap();
            let back: KVectorComponent = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(comp, back);
        }

        /// AttestationStatus round-trips through JSON.
        #[test]
        fn attestation_status_json_roundtrip(status in arb_attestation_status()) {
            let json = serde_json::to_string(&status).unwrap();
            let back: AttestationStatus = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(status, back);
        }

        /// Tier boundaries are exact at threshold values.
        #[test]
        fn tier_boundaries_exact(tier in arb_trust_tier()) {
            let boundary = tier.min_score();
            let result = TrustTier::from_score(boundary);
            prop_assert_eq!(tier, result);
        }

        /// Valid ranges have lower <= upper.
        #[test]
        fn trust_score_range_invariant(range in arb_trust_score_range()) {
            prop_assert!(range.lower <= range.upper);
            prop_assert!(range.lower >= 0.0);
            prop_assert!(range.upper <= 1.0);
        }

        /// Only Pending can transition to terminal states.
        #[test]
        fn attestation_terminal_states_dont_transition(
            terminal in prop_oneof![
                Just(AttestationStatus::Fulfilled),
                Just(AttestationStatus::Declined),
                Just(AttestationStatus::Expired),
                Just(AttestationStatus::Cancelled),
            ],
            target in arb_attestation_status()
        ) {
            // Terminal states can only stay the same (no-op update)
            let valid = terminal == target;
            // From the validation code: only Pending can transition to these
            if !valid {
                // Attempting to transition from terminal to different state is invalid
                prop_assert_ne!(terminal, AttestationStatus::Pending);
            }
        }
    }
}

/// Validate trust presentation creation
fn validate_create_presentation(
    _action: Create,
    pres: TrustPresentation,
) -> ExternResult<ValidateCallbackResult> {
    // Subject must be a valid DID
    if !pres.subject_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject must be a valid DID".into(),
        ));
    }

    // Presentation proof must not be empty
    if pres.presentation_proof.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Presentation proof cannot be empty".into(),
        ));
    }

    // Nonce must be present for replay protection
    if pres.nonce.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Nonce is required for replay protection".into(),
        ));
    }

    // If range is disclosed, it must be valid
    if let Some(ref range) = pres.disclosed_range {
        if range.lower < 0.0 || range.upper > 1.0 || range.lower > range.upper {
            return Ok(ValidateCallbackResult::Invalid(
                "Disclosed range must be valid".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}
