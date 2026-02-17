//! Credential Wallet Integrity Zome
//!
//! Defines entry types for storing verifiable credentials (VCs) issued
//! by any cluster (Civic, Commons, or external issuers). Credentials
//! are stored privately; selective presentation happens via personal_bridge.

use hdi::prelude::*;
use serde::{Deserialize, Serialize};
use personal_types::CredentialType;

/// A verifiable credential stored in the agent's wallet.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct StoredCredential {
    /// Type of credential.
    pub credential_type: CredentialType,
    /// The credential payload (JSON-LD, JWT, or custom format).
    pub credential_data: String,
    /// DID or agent key of the issuer.
    pub issuer: String,
    /// When the credential was issued.
    pub issued_at: Timestamp,
    /// When the credential expires (None = no expiry).
    pub expires_at: Option<Timestamp>,
    /// Whether this credential has been revoked by the holder.
    pub revoked: bool,
}

/// A proof derived from a stored credential for selective presentation.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CredentialProof {
    /// Hash of the source credential.
    pub credential_hash: ActionHash,
    /// The proof payload (e.g., ZK proof, signed subset).
    pub proof_data: String,
    /// What was proven (human-readable description).
    pub claim: String,
    /// When this proof was generated.
    pub created_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    StoredCredential(StoredCredential),
    CredentialProof(CredentialProof),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToCredentials,
    AgentToProofs,
    CredentialTypeToCredential,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, action: _ }) => match app_entry {
            EntryTypes::StoredCredential(cred) => validate_credential(&cred),
            EntryTypes::CredentialProof(proof) => validate_proof(&proof),
        },
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::StoredCredential(cred) => validate_credential(&cred),
            EntryTypes::CredentialProof(proof) => validate_proof(&proof),
        },
        FlatOp::StoreEntry(_) => Ok(ValidateCallbackResult::Valid),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_credential(cred: &StoredCredential) -> ExternResult<ValidateCallbackResult> {
    if cred.credential_data.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "StoredCredential credential_data cannot be empty".into(),
        ));
    }
    if cred.credential_data.len() > 65536 {
        return Ok(ValidateCallbackResult::Invalid(
            "StoredCredential credential_data must be <= 65536 bytes".into(),
        ));
    }
    if cred.issuer.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "StoredCredential issuer cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_proof(proof: &CredentialProof) -> ExternResult<ValidateCallbackResult> {
    if proof.proof_data.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "CredentialProof proof_data cannot be empty".into(),
        ));
    }
    if proof.proof_data.len() > 32768 {
        return Ok(ValidateCallbackResult::Invalid(
            "CredentialProof proof_data must be <= 32768 bytes".into(),
        ));
    }
    if proof.claim.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "CredentialProof claim cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_credential(ctype: CredentialType) -> StoredCredential {
        StoredCredential {
            credential_type: ctype,
            credential_data: r#"{"vc":"test"}"#.into(),
            issuer: "did:key:z6MkTest".into(),
            issued_at: Timestamp::from_micros(0),
            expires_at: None,
            revoked: false,
        }
    }

    fn make_proof() -> CredentialProof {
        CredentialProof {
            credential_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            proof_data: r#"{"zk_proof":"abc123"}"#.into(),
            claim: "holder has valid identity credential".into(),
            created_at: Timestamp::from_micros(0),
        }
    }

    #[test]
    fn valid_identity_credential_passes() {
        let c = make_credential(CredentialType::Identity);
        assert!(matches!(validate_credential(&c).unwrap(), ValidateCallbackResult::Valid));
    }

    #[test]
    fn valid_fl_credential_passes() {
        let c = make_credential(CredentialType::FederatedLearning);
        assert!(matches!(validate_credential(&c).unwrap(), ValidateCallbackResult::Valid));
    }

    #[test]
    fn empty_credential_data_rejected() {
        let mut c = make_credential(CredentialType::Identity);
        c.credential_data = String::new();
        match validate_credential(&c).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn oversized_credential_data_rejected() {
        let mut c = make_credential(CredentialType::Identity);
        c.credential_data = "x".repeat(65537);
        match validate_credential(&c).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("65536")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn empty_issuer_rejected() {
        let mut c = make_credential(CredentialType::Identity);
        c.issuer = String::new();
        match validate_credential(&c).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn valid_proof_passes() {
        let p = make_proof();
        assert!(matches!(validate_proof(&p).unwrap(), ValidateCallbackResult::Valid));
    }

    #[test]
    fn empty_proof_data_rejected() {
        let mut p = make_proof();
        p.proof_data = String::new();
        match validate_proof(&p).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn oversized_proof_data_rejected() {
        let mut p = make_proof();
        p.proof_data = "x".repeat(32769);
        match validate_proof(&p).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("32768")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn empty_claim_rejected() {
        let mut p = make_proof();
        p.claim = String::new();
        match validate_proof(&p).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn credential_serde_roundtrip() {
        let c = make_credential(CredentialType::Governance);
        let json = serde_json::to_string(&c).unwrap();
        let back: StoredCredential = serde_json::from_str(&json).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn proof_serde_roundtrip() {
        let p = make_proof();
        let json = serde_json::to_string(&p).unwrap();
        let back: CredentialProof = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }
}
