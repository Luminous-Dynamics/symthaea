# Mycelix-Attest: Decentralized Identity Attestation

## Vision & Design Document

**Version**: 1.0.0
**Created**: December 30, 2025
**Status**: Design Phase
**Priority**: Tier 1 - High Synergy

---

## Executive Summary

Mycelix-Attest is the identity layer for the entire Mycelix ecosystem. It provides self-sovereign identity management with progressive disclosure, allowing agents to prove claims about themselves without revealing unnecessary information. Built on W3C Verifiable Credentials and integrated with Praxis's credential system, Attest becomes the foundation for trust across all hApps.

### Why Attest?

Identity is foundational:
- **Marketplace**: Verify seller is who they claim to be
- **HealthVault**: Confirm provider's medical license
- **Arbiter**: Check arbitrator qualifications
- **Voting (Agora)**: Ensure one-person-one-vote

Current identity systems are either:
- **Centralized**: Single points of failure, privacy nightmares
- **Anonymous**: Enable fraud and Sybil attacks
- **Binary**: All-or-nothing disclosure

Attest provides **progressive identity**: reveal only what's needed, when it's needed, to whom it's needed.

---

## Core Principles

### 1. Self-Sovereignty
You control your identity. No platform, government, or organization owns your data.

### 2. Minimal Disclosure
Prove only what's necessary. "I'm over 18" doesn't require revealing your birthdate.

### 3. Credential Portability
Credentials issued in one context are verifiable everywhere—across hApps, networks, even blockchains.

### 4. Revocable Trust
Attestations can be revoked. Trust is not permanent.

### 5. Progressive Identity
Build identity over time through accumulated attestations from diverse sources.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Identity Sources                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Government│ │University│ │ Employer │ │  Peers   │ │  Self    │  │
│  │(KYC/AML) │ │(Degrees) │ │(Employment│ │(Vouch)  │ │(Claims)  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│       └────────────┴────────────┴────────────┴────────────┘        │
│                              ↓                                       │
├─────────────────────────────────────────────────────────────────────┤
│                          Attest hApp                                 │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     Coordinator Zomes                            ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │ Identity │ │Credential│ │Disclosure│ │Reputation│           ││
│  │  │ Manager  │ │ Manager  │ │ Protocol │ │ Linker   │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │  Vouch   │ │ Recovery │ │   ZKP    │ │  Issuer  │           ││
│  │  │  System  │ │  Module  │ │Generator │ │ Registry │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │                     Integrity Zomes                              ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │ Identity │ │Credential│ │ Vouch    │ │ Reveal   │           ││
│  │  │ Anchor   │ │  Entry   │ │  Entry   │ │  Log     │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                       Consuming hApps                                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │HealthV  │ │Marketplace│ │ Arbiter │ │  Agora  │ │  All    │      │
│  │(Provider)│ │ (Seller)  │ │(Arbitr) │ │(Voter)  │ │ Others  │      │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Core Entry Types

```rust
/// Identity anchor - the root of an agent's identity
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct IdentityAnchor {
    /// Primary agent public key
    pub agent: AgentPubKey,
    /// DID document root
    pub did_document: DIDDocument,
    /// Recovery configuration
    pub recovery_config: RecoveryConfig,
    /// Linked external identities
    pub external_links: Vec<ExternalIdentityLink>,
    /// Identity creation timestamp
    pub created_at: Timestamp,
    /// Last updated
    pub updated_at: Timestamp,
    /// Identity version (for key rotation)
    pub version: u32,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct DIDDocument {
    /// DID identifier (did:mycelix:...)
    pub id: String,
    /// Verification methods (keys)
    pub verification_method: Vec<VerificationMethod>,
    /// Authentication methods
    pub authentication: Vec<String>,
    /// Assertion methods (for signing credentials)
    pub assertion_method: Vec<String>,
    /// Key agreement (for encryption)
    pub key_agreement: Vec<String>,
    /// Service endpoints
    pub service: Vec<ServiceEndpoint>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationMethod {
    pub id: String,
    pub r#type: String, // Ed25519VerificationKey2020, etc.
    pub controller: String,
    pub public_key_multibase: String,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalIdentityLink {
    /// External platform/system
    pub platform: String,
    /// Identifier on that platform
    pub identifier: String,
    /// Proof of link
    pub proof: ExternalLinkProof,
    /// Linked timestamp
    pub linked_at: Timestamp,
    /// Is this link verified?
    pub verified: bool,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ExternalLinkProof {
    /// Signed message from external account
    SignedMessage { signature: String, message: String },
    /// DNS record verification
    DNSRecord { record_type: String, value: String },
    /// OAuth token (verified by oracle)
    OAuthVerified { verifier: AgentPubKey },
    /// Out-of-band verification
    OutOfBand { verifier: AgentPubKey, method: String },
}

/// W3C Verifiable Credential
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerifiableCredential {
    /// Unique credential ID
    pub id: String,
    /// Credential type(s)
    pub credential_type: Vec<String>,
    /// Who issued this credential
    pub issuer: CredentialIssuer,
    /// Who this credential is about
    pub subject: CredentialSubject,
    /// When issued
    pub issuance_date: Timestamp,
    /// When it expires (if applicable)
    pub expiration_date: Option<Timestamp>,
    /// The actual claims
    pub credential_claims: Vec<CredentialClaim>,
    /// Cryptographic proof
    pub proof: CredentialProof,
    /// Revocation status
    pub revocation: RevocationStatus,
    /// Epistemic classification
    pub epistemic: EpistemicClaim,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct CredentialIssuer {
    /// Issuer's DID
    pub id: String,
    /// Issuer's Holochain agent key
    pub agent: AgentPubKey,
    /// Issuer's name (optional)
    pub name: Option<String>,
    /// Issuer type
    pub issuer_type: IssuerType,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum IssuerType {
    /// Government agency
    Government { jurisdiction: String },
    /// Educational institution
    Educational { accreditation: String },
    /// Professional body
    Professional { domain: String },
    /// Employer
    Employer { organization: String },
    /// Individual (peer attestation)
    Individual,
    /// Mycelix system (automated)
    System,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct CredentialSubject {
    /// Subject's DID
    pub id: String,
    /// Subject's Holochain agent key
    pub agent: AgentPubKey,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct CredentialClaim {
    /// Claim type
    pub claim_type: String,
    /// Claim value (may be structured)
    pub value: ClaimValue,
    /// Evidence supporting this claim
    pub evidence: Vec<ClaimEvidence>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ClaimValue {
    /// String value
    String(String),
    /// Numeric value
    Number(f64),
    /// Boolean
    Boolean(bool),
    /// Date
    Date(String),
    /// Complex structure
    Object(String), // JSON
    /// Commitment (for ZKP)
    Commitment(String),
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimEvidence {
    /// Evidence type
    pub evidence_type: String,
    /// Reference (URL, hash, etc.)
    pub reference: String,
    /// Verification method
    pub verification: String,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct CredentialProof {
    /// Proof type (Ed25519Signature2020, etc.)
    pub proof_type: String,
    /// Created timestamp
    pub created: Timestamp,
    /// Verification method used
    pub verification_method: String,
    /// Proof purpose
    pub proof_purpose: String,
    /// The proof value (signature)
    pub proof_value: String,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum RevocationStatus {
    /// Not revoked
    Active,
    /// Revoked by issuer
    Revoked {
        revoked_at: Timestamp,
        reason: String,
        revocation_proof: String,
    },
    /// Suspended (temporarily)
    Suspended {
        suspended_at: Timestamp,
        reason: String,
    },
}

/// Peer vouching (web of trust)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Vouch {
    /// Who is vouching
    pub voucher: AgentPubKey,
    /// Who is being vouched for
    pub vouchee: AgentPubKey,
    /// What is being vouched
    pub vouch_type: VouchType,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Context for this vouch
    pub context: String,
    /// When created
    pub created_at: Timestamp,
    /// Expiry (if applicable)
    pub expires_at: Option<Timestamp>,
    /// Signature
    pub signature: Signature,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum VouchType {
    /// "I know this person"
    Identity,
    /// "This person is skilled at X"
    Skill(String),
    /// "I've worked with this person"
    ProfessionalRelationship,
    /// "This person is trustworthy"
    Trustworthiness,
    /// "This person belongs to our community"
    CommunityMembership(String),
    /// Custom vouch type
    Custom(String),
}

/// Disclosure request from a verifier
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DisclosureRequest {
    /// Unique request ID
    pub request_id: String,
    /// Who is requesting
    pub verifier: AgentPubKey,
    /// What is being requested
    pub requested_claims: Vec<RequestedClaim>,
    /// Purpose of the request
    pub purpose: String,
    /// Deadline for response
    pub deadline: Timestamp,
    /// Is selective disclosure allowed?
    pub allows_selective: bool,
    /// Is ZKP proof acceptable?
    pub accepts_zkp: bool,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestedClaim {
    /// Claim type requested
    pub claim_type: String,
    /// Is this claim required or optional?
    pub required: bool,
    /// Acceptable issuers (if specified)
    pub acceptable_issuers: Option<Vec<String>>,
    /// ZKP predicate (if accepting ZKP)
    pub zkp_predicate: Option<ZKPPredicate>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ZKPPredicate {
    /// Value equals
    Equals(ClaimValue),
    /// Value greater than
    GreaterThan(f64),
    /// Value less than
    LessThan(f64),
    /// Value in set
    InSet(Vec<ClaimValue>),
    /// Value not in set
    NotInSet(Vec<ClaimValue>),
}

/// Disclosure response from identity holder
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DisclosureResponse {
    /// Request this responds to
    pub request: ActionHash,
    /// Who is responding
    pub responder: AgentPubKey,
    /// Disclosed credentials/claims
    pub disclosures: Vec<Disclosure>,
    /// Declined claims (with reason)
    pub declined: Vec<DeclinedClaim>,
    /// Response timestamp
    pub responded_at: Timestamp,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum Disclosure {
    /// Full credential disclosure
    FullCredential(ActionHash),
    /// Selective claim disclosure
    SelectiveClaim {
        credential: ActionHash,
        claims: Vec<String>,
    },
    /// Zero-knowledge proof
    ZKProof {
        claim_type: String,
        predicate: ZKPPredicate,
        proof: String,
    },
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct DeclinedClaim {
    pub claim_type: String,
    pub reason: DeclineReason,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum DeclineReason {
    NotAvailable,
    InsufficientPurpose,
    TooSensitive,
    RequiresAdditionalVerification,
    Custom(String),
}
```

### Link Types

```rust
#[hdk_link_types]
pub enum LinkTypes {
    // Identity organization
    AgentToIdentity,
    IdentityToCredential,
    IdentityToVouch,
    IdentityToExternalLink,

    // Credential indexing
    IssuerToCredential,
    CredentialTypeToCredential,
    SubjectToCredential,

    // Vouching
    VoucherToVouch,
    VoucheeToVouch,
    VouchTypeToVouch,

    // Disclosure
    VerifierToRequest,
    ResponderToResponse,
    RequestToResponse,

    // Issuer registry
    IssuerTypeToIssuer,
    AccreditedIssuers,
}
```

---

## Identity Levels

### Progressive Identity Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Identity Trust Pyramid                            │
│                                                                      │
│                         ╱╲                                           │
│                        ╱  ╲ L5: Sovereign                            │
│                       ╱    ╲ (Full legal identity, KYC'd)            │
│                      ╱──────╲                                        │
│                     ╱        ╲ L4: Verified                          │
│                    ╱          ╲ (Government-issued credentials)      │
│                   ╱────────────╲                                     │
│                  ╱              ╲ L3: Attested                       │
│                 ╱                ╲ (Institutional credentials)       │
│                ╱──────────────────╲                                  │
│               ╱                    ╲ L2: Vouched                     │
│              ╱                      ╲ (Peer attestations)            │
│             ╱────────────────────────╲                               │
│            ╱                          ╲ L1: Pseudonymous             │
│           ╱                            ╲ (Agent key only)            │
│          ╱──────────────────────────────╲                            │
│         ╱                                ╲ L0: Anonymous             │
│        ╱                                  ╲ (No identity binding)    │
│       ╱────────────────────────────────────╲                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Level Requirements

| Level | Requirements | Use Cases |
|-------|--------------|-----------|
| L0 | None | Reading public data |
| L1 | Agent key | Basic Mycelix participation |
| L2 | 3+ vouches from L2+ agents | Marketplace selling, commenting |
| L3 | Institutional credential | Professional services |
| L4 | Government ID verification | Financial services, voting |
| L5 | Full KYC + biometric | High-value transactions |

---

## Zome Specifications

### 1. Identity Manager Zome

```rust
// identity_manager/src/lib.rs

/// Create identity anchor (one per agent)
#[hdk_extern]
pub fn create_identity() -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_latest_pubkey;

    // Check if identity already exists
    if get_identity(&agent).is_ok() {
        return Err(WasmError::Guest("Identity already exists".into()));
    }

    // Generate DID
    let did = format!("did:mycelix:{}", agent.to_string());

    // Create DID document
    let did_document = DIDDocument {
        id: did.clone(),
        verification_method: vec![
            VerificationMethod {
                id: format!("{}#key-1", did),
                r#type: "Ed25519VerificationKey2020".to_string(),
                controller: did.clone(),
                public_key_multibase: encode_public_key(&agent),
            }
        ],
        authentication: vec![format!("{}#key-1", did)],
        assertion_method: vec![format!("{}#key-1", did)],
        key_agreement: vec![],
        service: vec![],
    };

    // Default recovery config
    let recovery_config = RecoveryConfig {
        method: RecoveryMethod::SocialRecovery {
            guardians: vec![],
            threshold: 0,
        },
        backup_key: None,
    };

    let identity = IdentityAnchor {
        agent: agent.clone(),
        did_document,
        recovery_config,
        external_links: vec![],
        created_at: sys_time()?,
        updated_at: sys_time()?,
        version: 1,
    };

    let hash = create_entry(&EntryTypes::IdentityAnchor(identity))?;

    // Index
    create_link(
        agent_identity_path(&agent).path_entry_hash()?,
        hash.clone(),
        LinkTypes::AgentToIdentity,
        (),
    )?;

    Ok(hash)
}

/// Link external identity
#[hdk_extern]
pub fn link_external_identity(input: LinkExternalInput) -> ExternResult<()> {
    let agent = agent_info()?.agent_latest_pubkey;
    let mut identity = get_identity(&agent)?;

    // Verify the link proof
    let verified = verify_external_link_proof(&input.platform, &input.identifier, &input.proof)?;

    let link = ExternalIdentityLink {
        platform: input.platform,
        identifier: input.identifier,
        proof: input.proof,
        linked_at: sys_time()?,
        verified,
    };

    identity.external_links.push(link);
    identity.updated_at = sys_time()?;

    update_identity(&identity)?;

    Ok(())
}

/// Get identity level for an agent
#[hdk_extern]
pub fn get_identity_level(agent: AgentPubKey) -> ExternResult<IdentityLevel> {
    // Check if identity exists
    let identity = match get_identity(&agent) {
        Ok(i) => i,
        Err(_) => return Ok(IdentityLevel::L0Anonymous),
    };

    // Count credentials and vouches
    let credentials = get_agent_credentials(&agent)?;
    let vouches = get_vouches_for(&agent)?;

    // Check for government ID (L4+)
    let has_gov_id = credentials.iter().any(|c| {
        matches!(c.issuer.issuer_type, IssuerType::Government { .. })
            && c.credential_type.contains(&"GovernmentID".to_string())
    });

    // Check for KYC (L5)
    let has_kyc = credentials.iter().any(|c| {
        c.credential_type.contains(&"KYCVerification".to_string())
    });

    if has_kyc {
        return Ok(IdentityLevel::L5Sovereign);
    }

    if has_gov_id {
        return Ok(IdentityLevel::L4Verified);
    }

    // Check for institutional credentials (L3)
    let has_institutional = credentials.iter().any(|c| {
        matches!(
            c.issuer.issuer_type,
            IssuerType::Educational { .. } | IssuerType::Professional { .. }
        )
    });

    if has_institutional {
        return Ok(IdentityLevel::L3Attested);
    }

    // Count valid vouches from L2+ agents
    let valid_vouches: usize = vouches
        .iter()
        .filter(|v| {
            let voucher_level = get_identity_level(v.voucher.clone())
                .unwrap_or(IdentityLevel::L0Anonymous);
            voucher_level >= IdentityLevel::L2Vouched
        })
        .count();

    if valid_vouches >= 3 {
        return Ok(IdentityLevel::L2Vouched);
    }

    Ok(IdentityLevel::L1Pseudonymous)
}

/// Rotate keys
#[hdk_extern]
pub fn rotate_keys(input: KeyRotationInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_latest_pubkey;
    let mut identity = get_identity(&agent)?;

    // Verify old key signature
    verify_signature(&input.old_key_signature, &agent)?;

    // Add new verification method
    let new_key_id = format!("{}#key-{}", identity.did_document.id, identity.version + 1);

    identity.did_document.verification_method.push(
        VerificationMethod {
            id: new_key_id.clone(),
            r#type: input.new_key_type,
            controller: identity.did_document.id.clone(),
            public_key_multibase: input.new_public_key,
        }
    );

    // Update authentication methods
    identity.did_document.authentication = vec![new_key_id.clone()];
    identity.did_document.assertion_method = vec![new_key_id];

    identity.version += 1;
    identity.updated_at = sys_time()?;

    // Create new entry (history preserved)
    let hash = create_entry(&EntryTypes::IdentityAnchor(identity))?;

    Ok(hash)
}
```

### 2. Credential Manager Zome

```rust
// credential_manager/src/lib.rs

/// Issue a verifiable credential
#[hdk_extern]
pub fn issue_credential(input: IssueCredentialInput) -> ExternResult<ActionHash> {
    let issuer_agent = agent_info()?.agent_latest_pubkey;

    // Get issuer's identity
    let issuer_identity = get_identity(&issuer_agent)?;

    // Verify issuer is authorized (for institutional issuance)
    if let Some(required_issuer_type) = input.required_issuer_type {
        verify_issuer_authorization(&issuer_agent, &required_issuer_type)?;
    }

    // Build issuer info
    let issuer = CredentialIssuer {
        id: issuer_identity.did_document.id.clone(),
        agent: issuer_agent.clone(),
        name: input.issuer_name,
        issuer_type: input.issuer_type,
    };

    // Build subject info
    let subject = CredentialSubject {
        id: format!("did:mycelix:{}", input.subject.to_string()),
        agent: input.subject.clone(),
    };

    // Create credential ID
    let credential_id = format!(
        "urn:uuid:{}",
        generate_uuid()
    );

    // Determine epistemic level based on issuer type
    let epistemic_level = match &issuer.issuer_type {
        IssuerType::Government { .. } => EmpiricalLevel::E4PublicRepro,
        IssuerType::Educational { .. } => EmpiricalLevel::E3Cryptographic,
        IssuerType::Professional { .. } => EmpiricalLevel::E3Cryptographic,
        IssuerType::Employer { .. } => EmpiricalLevel::E2PrivateVerify,
        IssuerType::Individual => EmpiricalLevel::E1Testimonial,
        IssuerType::System => EmpiricalLevel::E3Cryptographic,
    };

    // Create proof
    let proof = create_credential_proof(
        &credential_id,
        &issuer_identity.did_document,
        &input.claims,
    )?;

    let credential = VerifiableCredential {
        id: credential_id,
        credential_type: input.credential_types,
        issuer,
        subject,
        issuance_date: sys_time()?,
        expiration_date: input.expiration_date,
        credential_claims: input.claims,
        proof,
        revocation: RevocationStatus::Active,
        epistemic: EpistemicClaim::new(
            format!("Credential: {}", input.credential_types.join(", ")),
            epistemic_level,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
        ),
    };

    let hash = create_entry(&EntryTypes::VerifiableCredential(credential.clone()))?;

    // Index by issuer
    create_link(
        issuer_credentials_path(&issuer_agent).path_entry_hash()?,
        hash.clone(),
        LinkTypes::IssuerToCredential,
        (),
    )?;

    // Index by subject
    create_link(
        subject_credentials_path(&input.subject).path_entry_hash()?,
        hash.clone(),
        LinkTypes::SubjectToCredential,
        (),
    )?;

    // Index by type
    for cred_type in &input.credential_types {
        create_link(
            credential_type_path(cred_type).path_entry_hash()?,
            hash.clone(),
            LinkTypes::CredentialTypeToCredential,
            (),
        )?;
    }

    // Notify subject
    send_signal_to_agent(&input.subject, Signal::CredentialIssued(hash.clone()))?;

    // Register with Praxis if educational
    if matches!(issuer.issuer_type, IssuerType::Educational { .. }) {
        bridge_call::<()>(
            "praxis",
            "register_credential",
            hash.clone(),
        )?;
    }

    Ok(hash)
}

/// Revoke a credential
#[hdk_extern]
pub fn revoke_credential(input: RevokeCredentialInput) -> ExternResult<()> {
    let issuer = agent_info()?.agent_latest_pubkey;

    // Get credential
    let mut credential = get_credential(&input.credential)?;

    // Verify caller is issuer
    if credential.issuer.agent != issuer {
        return Err(WasmError::Guest("Only issuer can revoke".into()));
    }

    // Create revocation proof
    let revocation_proof = create_revocation_proof(&input.credential, &input.reason)?;

    // Update status
    credential.revocation = RevocationStatus::Revoked {
        revoked_at: sys_time()?,
        reason: input.reason,
        revocation_proof,
    };

    // Update entry
    update_credential(&credential)?;

    // Notify subject
    send_signal_to_agent(
        &credential.subject.agent,
        Signal::CredentialRevoked(input.credential),
    )?;

    Ok(())
}

/// Verify a credential
#[hdk_extern]
pub fn verify_credential(credential_hash: ActionHash) -> ExternResult<VerificationResult> {
    let credential = get_credential(&credential_hash)?;

    // Check revocation status
    if !matches!(credential.revocation, RevocationStatus::Active) {
        return Ok(VerificationResult {
            valid: false,
            reason: "Credential has been revoked".to_string(),
            issuer_verified: false,
            signature_valid: false,
            not_expired: false,
        });
    }

    // Check expiration
    let not_expired = match credential.expiration_date {
        Some(expiry) => sys_time()? < expiry,
        None => true,
    };

    if !not_expired {
        return Ok(VerificationResult {
            valid: false,
            reason: "Credential has expired".to_string(),
            issuer_verified: false,
            signature_valid: false,
            not_expired: false,
        });
    }

    // Verify issuer
    let issuer_identity = get_identity(&credential.issuer.agent)?;
    let issuer_verified = issuer_identity.did_document.id == credential.issuer.id;

    // Verify signature
    let signature_valid = verify_credential_proof(&credential)?;

    let valid = issuer_verified && signature_valid && not_expired;

    Ok(VerificationResult {
        valid,
        reason: if valid { "Valid".to_string() } else { "Verification failed".to_string() },
        issuer_verified,
        signature_valid,
        not_expired,
    })
}

/// Get all credentials for an agent
#[hdk_extern]
pub fn get_my_credentials() -> ExternResult<Vec<VerifiableCredential>> {
    let agent = agent_info()?.agent_latest_pubkey;
    get_agent_credentials(&agent)
}
```

### 3. Disclosure Protocol Zome

```rust
// disclosure_protocol/src/lib.rs

/// Request disclosure from an agent
#[hdk_extern]
pub fn request_disclosure(input: DisclosureRequestInput) -> ExternResult<ActionHash> {
    let verifier = agent_info()?.agent_latest_pubkey;

    let request = DisclosureRequest {
        request_id: generate_request_id(),
        verifier: verifier.clone(),
        requested_claims: input.requested_claims,
        purpose: input.purpose,
        deadline: input.deadline,
        allows_selective: input.allows_selective,
        accepts_zkp: input.accepts_zkp,
    };

    let hash = create_entry(&EntryTypes::DisclosureRequest(request))?;

    // Notify target
    send_signal_to_agent(&input.target, Signal::DisclosureRequested(hash.clone()))?;

    Ok(hash)
}

/// Respond to disclosure request
#[hdk_extern]
pub fn respond_to_disclosure(input: DisclosureResponseInput) -> ExternResult<ActionHash> {
    let responder = agent_info()?.agent_latest_pubkey;

    // Get request
    let request = get_disclosure_request(&input.request)?;

    // Check deadline
    if sys_time()? > request.deadline {
        return Err(WasmError::Guest("Request deadline passed".into()));
    }

    // Process each requested claim
    let mut disclosures = vec![];
    let mut declined = vec![];

    for requested in &request.requested_claims {
        match find_matching_credential(&responder, requested)? {
            Some((credential_hash, claims)) => {
                // Determine disclosure type
                if request.accepts_zkp && requested.zkp_predicate.is_some() {
                    // Generate ZKP
                    let proof = generate_zkp(
                        &credential_hash,
                        &requested.claim_type,
                        requested.zkp_predicate.as_ref().unwrap(),
                    )?;
                    disclosures.push(Disclosure::ZKProof {
                        claim_type: requested.claim_type.clone(),
                        predicate: requested.zkp_predicate.clone().unwrap(),
                        proof,
                    });
                } else if request.allows_selective && claims.len() < get_credential(&credential_hash)?.credential_claims.len() {
                    // Selective disclosure
                    disclosures.push(Disclosure::SelectiveClaim {
                        credential: credential_hash,
                        claims,
                    });
                } else {
                    // Full credential
                    disclosures.push(Disclosure::FullCredential(credential_hash));
                }
            }
            None => {
                if requested.required {
                    declined.push(DeclinedClaim {
                        claim_type: requested.claim_type.clone(),
                        reason: DeclineReason::NotAvailable,
                    });
                }
            }
        }
    }

    let response = DisclosureResponse {
        request: input.request,
        responder,
        disclosures,
        declined,
        responded_at: sys_time()?,
    };

    let hash = create_entry(&EntryTypes::DisclosureResponse(response))?;

    // Notify verifier
    send_signal_to_agent(&request.verifier, Signal::DisclosureReceived(hash.clone()))?;

    // Log disclosure
    log_disclosure(&input.request, &hash)?;

    Ok(hash)
}

/// Generate zero-knowledge proof for a claim
fn generate_zkp(
    credential: &ActionHash,
    claim_type: &str,
    predicate: &ZKPPredicate,
) -> ExternResult<String> {
    let cred = get_credential(credential)?;

    // Find the claim
    let claim = cred.credential_claims
        .iter()
        .find(|c| c.claim_type == claim_type)
        .ok_or(WasmError::Guest("Claim not found".into()))?;

    // Generate proof based on predicate type
    match predicate {
        ZKPPredicate::GreaterThan(threshold) => {
            // Prove value > threshold without revealing value
            if let ClaimValue::Number(value) = &claim.value {
                let proof = create_range_proof(*value, *threshold, f64::MAX)?;
                Ok(proof)
            } else {
                Err(WasmError::Guest("Claim is not numeric".into()))
            }
        }
        ZKPPredicate::LessThan(threshold) => {
            if let ClaimValue::Number(value) = &claim.value {
                let proof = create_range_proof(*value, f64::MIN, *threshold)?;
                Ok(proof)
            } else {
                Err(WasmError::Guest("Claim is not numeric".into()))
            }
        }
        ZKPPredicate::Equals(expected) => {
            // Prove equality without revealing value
            let proof = create_equality_proof(&claim.value, expected)?;
            Ok(proof)
        }
        ZKPPredicate::InSet(set) => {
            // Prove membership in set
            let proof = create_set_membership_proof(&claim.value, set)?;
            Ok(proof)
        }
        ZKPPredicate::NotInSet(set) => {
            // Prove non-membership
            let proof = create_set_non_membership_proof(&claim.value, set)?;
            Ok(proof)
        }
    }
}
```

### 4. Vouch System Zome

```rust
// vouch_system/src/lib.rs

/// Vouch for another agent
#[hdk_extern]
pub fn vouch_for(input: VouchInput) -> ExternResult<ActionHash> {
    let voucher = agent_info()?.agent_latest_pubkey;

    // Can't vouch for yourself
    if voucher == input.vouchee {
        return Err(WasmError::Guest("Cannot vouch for yourself".into()));
    }

    // Check voucher's identity level
    let voucher_level = get_identity_level(voucher.clone())?;
    if voucher_level < IdentityLevel::L1Pseudonymous {
        return Err(WasmError::Guest("Must have identity to vouch".into()));
    }

    // Check for existing vouch
    if has_existing_vouch(&voucher, &input.vouchee, &input.vouch_type)? {
        return Err(WasmError::Guest("Already vouched for this".into()));
    }

    // Create vouch
    let vouch = Vouch {
        voucher: voucher.clone(),
        vouchee: input.vouchee.clone(),
        vouch_type: input.vouch_type,
        confidence: input.confidence.clamp(0.0, 1.0),
        context: input.context,
        created_at: sys_time()?,
        expires_at: input.expires_at,
        signature: sign_vouch(&input)?,
    };

    let hash = create_entry(&EntryTypes::Vouch(vouch.clone()))?;

    // Index by voucher
    create_link(
        voucher_path(&voucher).path_entry_hash()?,
        hash.clone(),
        LinkTypes::VoucherToVouch,
        (),
    )?;

    // Index by vouchee
    create_link(
        vouchee_path(&input.vouchee).path_entry_hash()?,
        hash.clone(),
        LinkTypes::VoucheeToVouch,
        (),
    )?;

    // Notify vouchee
    send_signal_to_agent(&input.vouchee, Signal::ReceivedVouch(hash.clone()))?;

    // Update vouchee's reputation via Bridge
    let voucher_trust = bridge_call::<f64>(
        "bridge",
        "get_cross_happ_reputation",
        voucher.clone(),
    )?;

    let vouch_impact = vouch.confidence * voucher_trust * 0.01; // Small positive impact

    bridge_call::<()>(
        "bridge",
        "apply_reputation_impact",
        ReputationImpact {
            agent: input.vouchee,
            happ: "attest".to_string(),
            impact: vouch_impact,
            reason: format!("Vouch received from trusted agent"),
        },
    )?;

    Ok(hash)
}

/// Revoke a vouch
#[hdk_extern]
pub fn revoke_vouch(vouch_hash: ActionHash) -> ExternResult<()> {
    let voucher = agent_info()?.agent_latest_pubkey;

    // Get vouch
    let vouch = get_vouch(&vouch_hash)?;

    // Verify caller is voucher
    if vouch.voucher != voucher {
        return Err(WasmError::Guest("Only voucher can revoke".into()));
    }

    // Mark as revoked
    delete_entry(vouch_hash)?;

    // Negative reputation impact to vouchee
    bridge_call::<()>(
        "bridge",
        "apply_reputation_impact",
        ReputationImpact {
            agent: vouch.vouchee.clone(),
            happ: "attest".to_string(),
            impact: -0.02,
            reason: "Vouch revoked".to_string(),
        },
    )?;

    // Notify vouchee
    send_signal_to_agent(&vouch.vouchee, Signal::VouchRevoked(vouch_hash))?;

    Ok(())
}

/// Get web of trust path between two agents
#[hdk_extern]
pub fn get_trust_path(from: AgentPubKey, to: AgentPubKey) -> ExternResult<TrustPath> {
    // BFS to find path through vouch network
    let mut visited: HashSet<AgentPubKey> = HashSet::new();
    let mut queue: VecDeque<(AgentPubKey, Vec<AgentPubKey>)> = VecDeque::new();

    visited.insert(from.clone());
    queue.push_back((from.clone(), vec![from.clone()]));

    while let Some((current, path)) = queue.pop_front() {
        if current == to {
            // Calculate path trust (product of confidences)
            let trust = calculate_path_trust(&path)?;
            return Ok(TrustPath {
                from,
                to,
                path,
                trust_level: trust,
                hops: path.len() - 1,
            });
        }

        // Get all vouches from current agent
        let vouches = get_vouches_from(&current)?;

        for vouch in vouches {
            if !visited.contains(&vouch.vouchee) {
                visited.insert(vouch.vouchee.clone());
                let mut new_path = path.clone();
                new_path.push(vouch.vouchee.clone());
                queue.push_back((vouch.vouchee, new_path));
            }
        }
    }

    // No path found
    Ok(TrustPath {
        from,
        to,
        path: vec![],
        trust_level: 0.0,
        hops: 0,
    })
}
```

---

## ZKP Capabilities

### Supported Predicates

| Predicate | Example Use Case |
|-----------|------------------|
| Age >= 18 | Adult content access |
| Age >= 21 | Alcohol purchase |
| Country in {US, CA, EU} | Regional services |
| Citizenship = X | Voting rights |
| Income >= $50,000 | Loan qualification |
| Degree = Bachelor's | Job application |
| License = Active | Professional services |

### Implementation (Simplified)

```rust
// Using bulletproofs for range proofs
fn create_range_proof(value: f64, min: f64, max: f64) -> ExternResult<String> {
    // Convert to u64 for bulletproofs
    let value_u64 = (value * 100.0) as u64;
    let min_u64 = (min * 100.0) as u64;
    let max_u64 = (max * 100.0) as u64;

    // Generate range proof
    // This would use a bulletproofs library
    let proof = bulletproofs::RangeProof::prove(
        value_u64,
        min_u64,
        max_u64,
    )?;

    Ok(base64::encode(proof.to_bytes()))
}

// Using commitment schemes for equality
fn create_equality_proof(value: &ClaimValue, expected: &ClaimValue) -> ExternResult<String> {
    // Pedersen commitment
    let commitment = pedersen_commit(value)?;

    // ZK proof that commitment opens to expected value
    let proof = zkp::equality_proof(commitment, expected)?;

    Ok(base64::encode(proof))
}
```

---

## Cross-hApp Integration

### HealthVault Integration
```rust
// Verify provider credentials
let disclosure = bridge_call::<DisclosureResponse>(
    "attest",
    "request_disclosure",
    DisclosureRequest {
        requested_claims: vec![
            RequestedClaim {
                claim_type: "MedicalLicense".to_string(),
                required: true,
                acceptable_issuers: Some(vec!["did:mycelix:medical-board".to_string()]),
                zkp_predicate: None,
            }
        ],
        purpose: "Verify medical provider for patient access".to_string(),
        ..
    },
)?;
```

### Agora Integration
```rust
// Verify voter eligibility
let eligible = bridge_call::<bool>(
    "attest",
    "check_identity_level",
    CheckLevelInput {
        agent: voter,
        required_level: IdentityLevel::L4Verified,
    },
)?;
```

### Marketplace Integration
```rust
// Get seller trust level
let trust_path = bridge_call::<TrustPath>(
    "attest",
    "get_trust_path",
    (buyer, seller),
)?;
```

---

## Recovery Mechanisms

### Social Recovery
```rust
pub struct SocialRecoveryConfig {
    /// Trusted guardians
    pub guardians: Vec<AgentPubKey>,
    /// Number required to recover
    pub threshold: u32,
    /// Recovery delay (hours)
    pub delay_hours: u32,
}

/// Initiate recovery
#[hdk_extern]
pub fn initiate_recovery(new_key: AgentPubKey) -> ExternResult<ActionHash> {
    // Create recovery request
    // Notify guardians
    // Wait for threshold approvals
    // After delay, transfer identity to new key
}

/// Guardian approves recovery
#[hdk_extern]
pub fn approve_recovery(recovery_request: ActionHash) -> ExternResult<()> {
    let guardian = agent_info()?.agent_latest_pubkey;

    // Verify guardian is in recovery config
    // Add approval
    // Check if threshold met
}
```

---

## Success Metrics

| Metric | 1 Year | 3 Years |
|--------|--------|---------|
| Identities created | 5,000 | 100,000 |
| Credentials issued | 10,000 | 500,000 |
| Vouches created | 20,000 | 1,000,000 |
| ZKP verifications | 5,000 | 200,000 |
| Identity recoveries | 50 | 1,000 |
| Cross-hApp verifications | 50,000 | 5,000,000 |

---

*"You are not your data. Your identity is yours to reveal or conceal."*
