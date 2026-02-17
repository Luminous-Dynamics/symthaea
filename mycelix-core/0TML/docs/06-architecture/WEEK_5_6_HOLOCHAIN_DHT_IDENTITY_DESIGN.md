# Week 5-6: Holochain DHT Integration for Decentralized Identity

**Date**: November 11, 2025
**Version**: v1.0
**Status**: Design Phase → Implementation

---

## Executive Summary

This document describes the integration of the Multi-Factor Decentralized Identity System with Holochain's Distributed Hash Table (DHT) to enable fully decentralized, peer-to-peer identity resolution without centralized infrastructure.

**Key Goals:**
1. **Decentralized DID Resolution** - Store and resolve DIDs via Holochain DHT
2. **Distributed Identity Storage** - Identity factors, credentials, and signals on DHT
3. **Cross-Network Reputation Tracking** - Aggregate reputation across Holochain networks
4. **Guardian Network Validation** - Store and validate guardian graphs on DHT

**Why Holochain DHT for Identity?**
- **No Central Authority**: Peer-to-peer resolution, no DNS-like servers
- **Agent-Centric**: Each agent controls their own identity data
- **Validation Rules**: DHT validates identity operations via consensus
- **Scalability**: Linear scaling per agent (not global bottleneck)
- **Privacy**: Private by default, selective disclosure

---

## Architecture Overview

### System Components

```
┌────────────────────────────────────────────────────────────────┐
│                    Identity Coordinator                         │
│  (Python - Zero-TrustML Coordinator)                           │
│                                                                 │
│  - Node registration with identity                             │
│  - Reputation with identity boost                              │
│  - Byzantine resistance (45% BFT)                              │
└────────────────────┬───────────────────────────────────────────┘
                     │ Python Client (via WebSocket)
┌────────────────────┴───────────────────────────────────────────┐
│           Holochain Identity DHT (New DNA)                     │
│                                                                 │
│  Zomes:                                                        │
│  - did_registry    - DID creation and resolution              │
│  - identity_store  - Factors, credentials, signals            │
│  - reputation_sync - Cross-network reputation                 │
│  - guardian_graph  - Guardian network validation              │
│                                                                 │
│  Entry Types:                                                  │
│  - DIDDocument      - W3C DID Document                        │
│  - IdentityFactors  - Multi-factor verification data          │
│  - IdentitySignals  - Trust signals (assurance, Sybil, etc)  │
│  - ReputationEntry  - Reputation scores with provenance       │
│  - GuardianNetwork  - Social recovery graph                   │
└─────────────────────────────────────────────────────────────────┘
                     │ DHT Gossip
┌────────────────────┴───────────────────────────────────────────┐
│                    Peer Validation Network                      │
│                                                                 │
│  All nodes participate in:                                     │
│  - DID document validation                                     │
│  - Identity factor verification                                │
│  - Guardian graph cartel detection                             │
│  - Reputation score aggregation                                │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow: Node Registration with DHT Identity

```
1. Node (Python)
   ↓ Create DID locally

2. Identity Coordinator
   ↓ Call Holochain DHT via WebSocket

3. Holochain Identity DNA - did_registry zome
   ↓ create_did(did_document, factors, credentials)

4. DHT Validation
   ↓ Peers validate DID document structure
   ↓ Peers verify factor signatures
   ↓ Peers check credential proofs

5. DHT Storage
   ↓ DID document stored on DHT
   ↓ Identity factors stored on DHT
   ↓ Identity signals computed and stored

6. Identity Coordinator
   ↓ Receives DID resolution confirmation
   ↓ Caches DID → signals mapping locally
   ↓ Assigns initial reputation based on assurance level

7. Gradient Submission
   ↓ Reputation queries DHT for identity signals
   ↓ Byzantine resistance with identity enhancement
```

---

## Holochain DNA Design

### DNA Structure: `zerotrustml_identity`

```
zerotrustml_identity/
├── dna.yaml                 # DNA configuration
└── zomes/
    ├── did_registry/        # DID creation and resolution
    │   └── src/
    │       ├── lib.rs       # Main entry point
    │       ├── did.rs       # DID document operations
    │       └── resolution.rs # DID resolution logic
    ├── identity_store/      # Identity factors and signals
    │   └── src/
    │       ├── lib.rs       # Main entry point
    │       ├── factors.rs   # Multi-factor storage
    │       ├── credentials.rs # Verifiable credentials
    │       └── signals.rs   # Trust signal computation
    ├── reputation_sync/     # Cross-network reputation
    │   └── src/
    │       ├── lib.rs       # Main entry point
    │       └── aggregation.rs # Reputation aggregation
    └── guardian_graph/      # Guardian network validation
        └── src/
            ├── lib.rs       # Main entry point
            └── validation.rs # Cartel detection
```

---

## Implementation Phases

### Phase 1: DID Registry Zome (Days 1-2)

**Goal**: Store and resolve W3C DID Documents on Holochain DHT

#### Entry Types

```rust
// zomes/did_registry/src/did.rs

use hdk::prelude::*;
use serde::{Deserialize, Serialize};

/// W3C DID Document stored on DHT
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DIDDocument {
    pub id: String,                    // "did:mycelix:abc123"
    pub controller: AgentPubKey,       // Holochain agent
    pub verification_methods: Vec<VerificationMethod>,
    pub authentication: Vec<String>,   // Key IDs for authentication
    pub created: i64,                  // Timestamp
    pub updated: i64,                  // Last update timestamp
    pub proof: Option<DIDProof>,       // Self-signature
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct VerificationMethod {
    pub id: String,                    // "did:mycelix:abc123#key-1"
    pub method_type: String,           // "Ed25519VerificationKey2020"
    pub controller: String,            // DID that controls this key
    pub public_key_multibase: String,  // Multibase-encoded public key
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct DIDProof {
    pub signature: Vec<u8>,            // Ed25519 signature
    pub verification_method: String,    // Key used to sign
    pub created: i64,                  // When proof was created
}
```

#### Zome Functions

```rust
// zomes/did_registry/src/lib.rs

#[hdk_extern]
pub fn create_did(input: CreateDIDInput) -> ExternResult<ActionHash> {
    // 1. Validate DID document structure
    validate_did_document(&input.document)?;

    // 2. Verify self-signature (proof)
    verify_did_proof(&input.document)?;

    // 3. Create entry on DHT
    let action_hash = create_entry(EntryTypes::DIDDocument(input.document.clone()))?;

    // 4. Create path for DID resolution (enables lookup by DID string)
    let path = Path::from(format!("did.{}", input.document.id));
    path.ensure()?;
    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::DIDResolution,
        LinkTag::new("did_document")
    )?;

    info!("DID created: {}", input.document.id);
    Ok(action_hash)
}

#[hdk_extern]
pub fn resolve_did(did: String) -> ExternResult<Option<DIDDocument>> {
    // 1. Get path for DID
    let path = Path::from(format!("did.{}", did));

    // 2. Get links (should be one)
    let links = get_links(
        GetLinksInputBuilder::try_new(
            path.path_entry_hash()?,
            LinkTypes::DIDResolution
        )?.build()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // 3. Get DID document from action hash
    let action_hash = ActionHash::try_from(links[0].target.clone())?;
    let record = get(action_hash, GetOptions::default())?;

    if let Some(rec) = record {
        if let Some(entry) = rec.entry().as_option() {
            if let Entry::App(app_bytes) = entry {
                let did_doc = DIDDocument::try_from(
                    SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                )?;
                return Ok(Some(did_doc));
            }
        }
    }

    Ok(None)
}

#[hdk_extern]
pub fn update_did(input: UpdateDIDInput) -> ExternResult<ActionHash> {
    // 1. Resolve existing DID
    let existing = resolve_did(input.did.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("DID not found".into())))?;

    // 2. Verify caller is controller
    if existing.controller != agent_info()?.agent_latest_pubkey {
        return Err(wasm_error!(WasmErrorInner::Guest("Not DID controller".into())));
    }

    // 3. Validate new document
    validate_did_document(&input.new_document)?;
    verify_did_proof(&input.new_document)?;

    // 4. Update entry
    let action_hash = update_entry(
        input.original_action_hash,
        EntryTypes::DIDDocument(input.new_document.clone())
    )?;

    info!("DID updated: {}", input.did);
    Ok(action_hash)
}

// Helper functions

fn validate_did_document(doc: &DIDDocument) -> ExternResult<()> {
    // 1. Check DID format
    if !doc.id.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format: must start with 'did:mycelix:'".into()
        )));
    }

    // 2. Check verification methods exist
    if doc.verification_methods.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must have at least one verification method".into()
        )));
    }

    // 3. Check authentication references valid keys
    for auth_ref in &doc.authentication {
        if !doc.verification_methods.iter().any(|vm| vm.id == *auth_ref) {
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("Authentication references non-existent key: {}", auth_ref)
            )));
        }
    }

    Ok(())
}

fn verify_did_proof(doc: &DIDDocument) -> ExternResult<()> {
    let proof = doc.proof.as_ref()
        .ok_or(wasm_error!(WasmErrorInner::Guest("DID document missing proof".into())))?;

    // 1. Get verification method
    let vm = doc.verification_methods.iter()
        .find(|v| v.id == proof.verification_method)
        .ok_or(wasm_error!(WasmErrorInner::Guest("Proof verification method not found".into())))?;

    // 2. Decode public key (simplified - full implementation would use multibase)
    let public_key_bytes = base64::decode(&vm.public_key_multibase)
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid public key encoding".into())))?;

    // 3. Construct message (canonical JSON of document without proof)
    let mut doc_without_proof = doc.clone();
    doc_without_proof.proof = None;
    let message = serde_json::to_vec(&doc_without_proof)
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Failed to serialize document".into())))?;

    // 4. Verify Ed25519 signature
    // Note: Full implementation would use a proper Ed25519 verification function
    // This is simplified for illustration
    if vm.method_type != "Ed25519VerificationKey2020" {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Unsupported verification method type".into()
        )));
    }

    // TODO: Actual Ed25519 verification
    // verify_signature(public_key_bytes, message, proof.signature)?;

    Ok(())
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateDIDInput {
    pub document: DIDDocument,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateDIDInput {
    pub did: String,
    pub original_action_hash: ActionHash,
    pub new_document: DIDDocument,
}
```

#### Validation Rules

```rust
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(StoreEntry { entry, .. }) => {
            match entry {
                Entry::App(bytes) => {
                    // Try to deserialize as DIDDocument
                    if let Ok(did_doc) = DIDDocument::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        // Validate DID document
                        if let Err(e) = validate_did_document(&did_doc) {
                            return Ok(ValidateCallbackResult::Invalid(
                                format!("Invalid DID document: {:?}", e)
                            ));
                        }

                        // Verify proof
                        if let Err(e) = verify_did_proof(&did_doc) {
                            return Ok(ValidateCallbackResult::Invalid(
                                format!("Invalid DID proof: {:?}", e)
                            ));
                        }

                        return Ok(ValidateCallbackResult::Valid);
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
    Ok(ValidateCallbackResult::Valid)
}
```

---

### Phase 2: Identity Store Zome (Days 3-4)

**Goal**: Store identity factors, credentials, and trust signals on DHT

#### Entry Types

```rust
// zomes/identity_store/src/factors.rs

use hdk::prelude::*;

/// Identity factors for multi-factor authentication
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct IdentityFactors {
    pub did: String,                   // Associated DID
    pub factors: Vec<IdentityFactor>,  // All factors
    pub created: i64,
    pub updated: i64,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct IdentityFactor {
    pub factor_id: String,
    pub factor_type: String,           // "CryptoKey", "GitcoinPassport", etc.
    pub category: String,              // "PRIMARY", "REPUTATION", "SOCIAL", "BACKUP"
    pub status: String,                // "ACTIVE", "SUSPENDED", "REVOKED"
    pub metadata: String,              // JSON-encoded metadata
    pub added: i64,
    pub last_verified: i64,
}

/// Verifiable credentials
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerifiableCredential {
    pub id: String,                    // Credential ID
    pub issuer_did: String,            // Who issued
    pub subject_did: String,           // Who it's about
    pub vc_type: String,               // "VerifiedHuman", "DomainExpert", etc.
    pub claims: String,                // JSON-encoded claims
    pub proof: Vec<u8>,                // Ed25519 signature
    pub issued_at: i64,
    pub expires_at: Option<i64>,
}

/// Computed identity trust signals
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct IdentitySignals {
    pub did: String,
    pub assurance_level: String,       // "E0", "E1", "E2", "E3", "E4"
    pub sybil_resistance: f64,         // 0.0-1.0
    pub risk_level: String,            // "LOW", "MEDIUM", "HIGH"
    pub guardian_graph_diversity: f64, // 0.0-1.0
    pub verified_human: bool,
    pub credential_count: u32,
    pub computed_at: i64,
}
```

#### Zome Functions

```rust
// zomes/identity_store/src/lib.rs

#[hdk_extern]
pub fn store_identity_factors(input: StoreFactorsInput) -> ExternResult<ActionHash> {
    // 1. Verify caller controls the DID
    verify_did_controller(&input.did)?;

    // 2. Validate factors
    validate_factors(&input.factors)?;

    // 3. Create entry
    let identity_factors = IdentityFactors {
        did: input.did.clone(),
        factors: input.factors.clone(),
        created: sys_time()?.as_micros(),
        updated: sys_time()?.as_micros(),
    };

    let action_hash = create_entry(EntryTypes::IdentityFactors(identity_factors))?;

    // 4. Create link from DID
    let path = Path::from(format!("identity_factors.{}", input.did));
    path.ensure()?;
    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::IdentityFactorsLink,
        LinkTag::new("factors")
    )?;

    info!("Identity factors stored for DID: {}", input.did);
    Ok(action_hash)
}

#[hdk_extern]
pub fn get_identity_factors(did: String) -> ExternResult<Option<IdentityFactors>> {
    let path = Path::from(format!("identity_factors.{}", did));
    let links = get_links(
        GetLinksInputBuilder::try_new(
            path.path_entry_hash()?,
            LinkTypes::IdentityFactorsLink
        )?.build()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get most recent (last link)
    let action_hash = ActionHash::try_from(links.last().unwrap().target.clone())?;
    let record = get(action_hash, GetOptions::default())?;

    if let Some(rec) = record {
        if let Some(entry) = rec.entry().as_option() {
            if let Entry::App(app_bytes) = entry {
                let factors = IdentityFactors::try_from(
                    SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                )?;
                return Ok(Some(factors));
            }
        }
    }

    Ok(None)
}

#[hdk_extern]
pub fn store_verifiable_credential(input: StoreCredentialInput) -> ExternResult<ActionHash> {
    // 1. Verify issuer signature
    verify_credential_proof(&input.credential)?;

    // 2. Create entry
    let action_hash = create_entry(EntryTypes::VerifiableCredential(input.credential.clone()))?;

    // 3. Create link from subject DID
    let path = Path::from(format!("credentials.{}", input.credential.subject_did));
    path.ensure()?;
    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::CredentialLink,
        LinkTag::new(&input.credential.vc_type)
    )?;

    info!("Credential stored: {} for {}", input.credential.id, input.credential.subject_did);
    Ok(action_hash)
}

#[hdk_extern]
pub fn get_credentials(did: String) -> ExternResult<Vec<VerifiableCredential>> {
    let path = Path::from(format!("credentials.{}", did));
    let links = get_links(
        GetLinksInputBuilder::try_new(
            path.path_entry_hash()?,
            LinkTypes::CredentialLink
        )?.build()
    )?;

    let mut credentials = Vec::new();

    for link in links {
        let action_hash = ActionHash::try_from(link.target)?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(entry) = record.entry().as_option() {
                if let Entry::App(app_bytes) = entry {
                    if let Ok(cred) = VerifiableCredential::try_from(
                        SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                    ) {
                        // Check expiration
                        if let Some(expires_at) = cred.expires_at {
                            if sys_time()?.as_micros() > expires_at {
                                continue; // Skip expired
                            }
                        }
                        credentials.push(cred);
                    }
                }
            }
        }
    }

    Ok(credentials)
}

#[hdk_extern]
pub fn compute_identity_signals(did: String) -> ExternResult<IdentitySignals> {
    // 1. Get identity factors
    let factors = get_identity_factors(did.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No identity factors found".into())))?;

    // 2. Get credentials
    let credentials = get_credentials(did.clone())?;

    // 3. Compute assurance level
    let assurance_level = compute_assurance_level(&factors.factors)?;

    // 4. Compute Sybil resistance
    let sybil_resistance = compute_sybil_resistance(&factors.factors, &credentials)?;

    // 5. Check for VerifiedHuman credential
    let verified_human = credentials.iter().any(|c| c.vc_type == "VerifiedHuman");

    // 6. Create signals entry
    let signals = IdentitySignals {
        did: did.clone(),
        assurance_level,
        sybil_resistance,
        risk_level: compute_risk_level(sybil_resistance),
        guardian_graph_diversity: 0.0, // TODO: Implement from guardian_graph zome
        verified_human,
        credential_count: credentials.len() as u32,
        computed_at: sys_time()?.as_micros(),
    };

    // 7. Store signals
    let action_hash = create_entry(EntryTypes::IdentitySignals(signals.clone()))?;

    // 8. Create link from DID
    let path = Path::from(format!("identity_signals.{}", did));
    path.ensure()?;
    create_link(
        path.path_entry_hash()?,
        action_hash,
        LinkTypes::IdentitySignalsLink,
        LinkTag::new("signals")
    )?;

    Ok(signals)
}

// Helper functions

fn compute_assurance_level(factors: &[IdentityFactor]) -> ExternResult<String> {
    let active_factors: Vec<_> = factors.iter()
        .filter(|f| f.status == "ACTIVE")
        .collect();

    match active_factors.len() {
        0 => Ok("E0".to_string()), // Anonymous
        1 => Ok("E1".to_string()), // Testimonial
        2 => Ok("E2".to_string()), // Privately Verifiable
        3 => {
            // Check for Gitcoin ≥50
            let has_high_gitcoin = active_factors.iter().any(|f| {
                f.factor_type == "GitcoinPassport" &&
                // Parse score from metadata (simplified)
                true // TODO: Parse metadata JSON
            });
            if has_high_gitcoin {
                Ok("E3".to_string()) // Cryptographically Proven
            } else {
                Ok("E2".to_string())
            }
        }
        _ => {
            // E4 requires 4 categories
            let categories: std::collections::HashSet<_> = active_factors.iter()
                .map(|f| f.category.as_str())
                .collect();
            if categories.len() >= 4 {
                Ok("E4".to_string()) // Constitutionally Critical
            } else {
                Ok("E3".to_string())
            }
        }
    }
}

fn compute_sybil_resistance(factors: &[IdentityFactor], credentials: &[VerifiableCredential]) -> ExternResult<f64> {
    let active_count = factors.iter().filter(|f| f.status == "ACTIVE").count();
    let verified_human = credentials.iter().any(|c| c.vc_type == "VerifiedHuman");

    let base_score = match active_count {
        0 => 0.0,
        1 => 0.2,
        2 => 0.4,
        3 => 0.6,
        _ => 0.8,
    };

    let bonus = if verified_human { 0.2 } else { 0.0 };

    Ok((base_score + bonus).min(1.0))
}

fn compute_risk_level(sybil_resistance: f64) -> String {
    if sybil_resistance >= 0.7 {
        "LOW".to_string()
    } else if sybil_resistance >= 0.4 {
        "MEDIUM".to_string()
    } else {
        "HIGH".to_string()
    }
}

fn verify_did_controller(did: &str) -> ExternResult<()> {
    // TODO: Implement by querying did_registry zome
    Ok(())
}

fn validate_factors(factors: &[IdentityFactor]) -> ExternResult<()> {
    // Basic validation
    if factors.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest("At least one factor required".into())));
    }

    // Check for duplicate factor IDs
    let ids: std::collections::HashSet<_> = factors.iter().map(|f| &f.factor_id).collect();
    if ids.len() != factors.len() {
        return Err(wasm_error!(WasmErrorInner::Guest("Duplicate factor IDs".into())));
    }

    Ok(())
}

fn verify_credential_proof(credential: &VerifiableCredential) -> ExternResult<()> {
    // TODO: Verify Ed25519 signature using issuer's DID document
    Ok(())
}
```

---

### Phase 3: Reputation Sync Zome (Days 5-6)

**Goal**: Cross-network reputation tracking and aggregation

#### Entry Types

```rust
// zomes/reputation_sync/src/lib.rs

use hdk::prelude::*;

/// Reputation entry from a specific network
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ReputationEntry {
    pub did: String,
    pub network_id: String,            // "zerotrustml_mainnet", "compute_credits", etc.
    pub score: f64,                    // 0.0-1.0
    pub source: String,                // "gradient_accepted", "byzantine_detected", etc.
    pub assurance_level: String,       // Identity assurance at time of entry
    pub timestamp: i64,
    pub proof: Vec<u8>,                // Signature from network coordinator
}

/// Aggregated reputation across all networks
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AggregatedReputation {
    pub did: String,
    pub global_score: f64,             // Weighted average
    pub network_scores: String,        // JSON: { "network_id": score }
    pub total_contributions: u64,      // Count of reputation entries
    pub last_updated: i64,
}
```

#### Zome Functions

```rust
#[hdk_extern]
pub fn store_reputation_entry(input: StoreReputationInput) -> ExternResult<ActionHash> {
    // 1. Verify signature from network coordinator
    verify_reputation_proof(&input.entry)?;

    // 2. Create entry
    let action_hash = create_entry(EntryTypes::ReputationEntry(input.entry.clone()))?;

    // 3. Create link from DID
    let path = Path::from(format!("reputation.{}", input.entry.did));
    path.ensure()?;
    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::ReputationLink,
        LinkTag::new(&input.entry.network_id)
    )?;

    // 4. Trigger aggregation update
    update_aggregated_reputation(input.entry.did.clone())?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_reputation_for_network(did: String, network_id: String) -> ExternResult<Vec<ReputationEntry>> {
    let path = Path::from(format!("reputation.{}", did));
    let links = get_links(
        GetLinksInputBuilder::try_new(
            path.path_entry_hash()?,
            LinkTypes::ReputationLink
        )?.build()
    )?;

    let mut entries = Vec::new();

    for link in links {
        // Filter by network_id using link tag
        if link.tag.as_ref() != network_id.as_bytes() {
            continue;
        }

        let action_hash = ActionHash::try_from(link.target)?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(entry) = record.entry().as_option() {
                if let Entry::App(app_bytes) = entry {
                    if let Ok(rep_entry) = ReputationEntry::try_from(
                        SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                    ) {
                        entries.push(rep_entry);
                    }
                }
            }
        }
    }

    Ok(entries)
}

#[hdk_extern]
pub fn get_aggregated_reputation(did: String) -> ExternResult<Option<AggregatedReputation>> {
    let path = Path::from(format!("aggregated_reputation.{}", did));
    let links = get_links(
        GetLinksInputBuilder::try_new(
            path.path_entry_hash()?,
            LinkTypes::AggregatedReputationLink
        )?.build()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get most recent aggregation
    let action_hash = ActionHash::try_from(links.last().unwrap().target.clone())?;
    let record = get(action_hash, GetOptions::default())?;

    if let Some(rec) = record {
        if let Some(entry) = rec.entry().as_option() {
            if let Entry::App(app_bytes) = entry {
                let agg_rep = AggregatedReputation::try_from(
                    SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                )?;
                return Ok(Some(agg_rep));
            }
        }
    }

    Ok(None)
}

fn update_aggregated_reputation(did: String) -> ExternResult<ActionHash> {
    // 1. Get all reputation entries for this DID
    let path = Path::from(format!("reputation.{}", did));
    let links = get_links(
        GetLinksInputBuilder::try_new(
            path.path_entry_hash()?,
            LinkTypes::ReputationLink
        )?.build()
    )?;

    // 2. Collect all entries
    let mut network_scores = std::collections::HashMap::new();
    let mut total_contributions = 0u64;

    for link in links {
        let action_hash = ActionHash::try_from(link.target)?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(entry) = record.entry().as_option() {
                if let Entry::App(app_bytes) = entry {
                    if let Ok(rep_entry) = ReputationEntry::try_from(
                        SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                    ) {
                        network_scores.insert(rep_entry.network_id.clone(), rep_entry.score);
                        total_contributions += 1;
                    }
                }
            }
        }
    }

    // 3. Compute weighted average (simple mean for now)
    let global_score = if network_scores.is_empty() {
        0.5 // Default
    } else {
        network_scores.values().sum::<f64>() / network_scores.len() as f64
    };

    // 4. Create aggregated entry
    let aggregated = AggregatedReputation {
        did: did.clone(),
        global_score,
        network_scores: serde_json::to_string(&network_scores)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Failed to serialize scores".into())))?,
        total_contributions,
        last_updated: sys_time()?.as_micros(),
    };

    let action_hash = create_entry(EntryTypes::AggregatedReputation(aggregated))?;

    // 5. Create link from DID
    let agg_path = Path::from(format!("aggregated_reputation.{}", did));
    agg_path.ensure()?;
    create_link(
        agg_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AggregatedReputationLink,
        LinkTag::new("aggregated")
    )?;

    Ok(action_hash)
}

fn verify_reputation_proof(entry: &ReputationEntry) -> ExternResult<()> {
    // TODO: Verify signature from network coordinator
    Ok(())
}
```

---

### Phase 4: Guardian Graph Zome (Days 7-8)

**Goal**: Store and validate guardian networks for social recovery

#### Entry Types

```rust
// zomes/guardian_graph/src/lib.rs

use hdk::prelude::*;

/// Guardian network for a DID
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GuardianNetwork {
    pub did: String,                   // Subject DID
    pub guardians: Vec<String>,        // Guardian DIDs
    pub threshold: u32,                // Recovery threshold
    pub created: i64,
    pub updated: i64,
}

/// Guardian network validation metrics
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GuardianValidation {
    pub did: String,
    pub network_diversity: f64,        // 0.0-1.0 (low = cartel risk)
    pub connection_density: f64,       // How interconnected guardians are
    pub cartel_warning: bool,          // True if potential cartel detected
    pub validated_at: i64,
}
```

#### Zome Functions

```rust
#[hdk_extern]
pub fn store_guardian_network(input: StoreGuardianInput) -> ExternResult<ActionHash> {
    // 1. Verify caller controls DID
    verify_did_controller(&input.network.did)?;

    // 2. Validate guardian count
    if input.network.guardians.len() < 3 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Minimum 3 guardians required".into()
        )));
    }

    if input.network.threshold > input.network.guardians.len() as u32 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Threshold exceeds guardian count".into()
        )));
    }

    // 3. Create entry
    let action_hash = create_entry(EntryTypes::GuardianNetwork(input.network.clone()))?;

    // 4. Create link from DID
    let path = Path::from(format!("guardian_network.{}", input.network.did));
    path.ensure()?;
    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::GuardianNetworkLink,
        LinkTag::new("network")
    )?;

    // 5. Validate network (cartel detection)
    if let Some(graph) = input.guardian_graph {
        validate_guardian_network(input.network.did.clone(), graph)?;
    }

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_guardian_network(did: String) -> ExternResult<Option<GuardianNetwork>> {
    let path = Path::from(format!("guardian_network.{}", did));
    let links = get_links(
        GetLinksInputBuilder::try_new(
            path.path_entry_hash()?,
            LinkTypes::GuardianNetworkLink
        )?.build()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get most recent
    let action_hash = ActionHash::try_from(links.last().unwrap().target.clone())?;
    let record = get(action_hash, GetOptions::default())?;

    if let Some(rec) = record {
        if let Some(entry) = rec.entry().as_option() {
            if let Entry::App(app_bytes) = entry {
                let network = GuardianNetwork::try_from(
                    SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                )?;
                return Ok(Some(network));
            }
        }
    }

    Ok(None)
}

#[hdk_extern]
pub fn validate_guardian_network(did: String, graph: GuardianGraphInput) -> ExternResult<GuardianValidation> {
    // 1. Compute connection density
    let nodes = graph.nodes.len();
    let edges = graph.edges.len();
    let max_edges = (nodes * (nodes - 1)) / 2;

    let connection_density = if max_edges > 0 {
        edges as f64 / max_edges as f64
    } else {
        0.0
    };

    // 2. Check for cartel (high density = all guardians know each other)
    let cartel_warning = connection_density > 0.8;

    // 3. Compute network diversity (inverse of density)
    let network_diversity = 1.0 - connection_density;

    // 4. Create validation entry
    let validation = GuardianValidation {
        did: did.clone(),
        network_diversity,
        connection_density,
        cartel_warning,
        validated_at: sys_time()?.as_micros(),
    };

    let action_hash = create_entry(EntryTypes::GuardianValidation(validation.clone()))?;

    // 5. Create link from DID
    let path = Path::from(format!("guardian_validation.{}", did));
    path.ensure()?;
    create_link(
        path.path_entry_hash()?,
        action_hash,
        LinkTypes::GuardianValidationLink,
        LinkTag::new("validation")
    )?;

    Ok(validation)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GuardianGraphInput {
    pub nodes: Vec<String>,            // Guardian DIDs
    pub edges: Vec<(String, String)>,  // Connections between guardians
}

#[derive(Serialize, Deserialize, Debug)]
pub struct StoreGuardianInput {
    pub network: GuardianNetwork,
    pub guardian_graph: Option<GuardianGraphInput>,
}
```

---

## Python Client Integration

### Holochain WebSocket Client

```python
# src/zerotrustml/holochain/identity_dht_client.py

import asyncio
import json
from typing import Dict, List, Optional
import websockets
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DHTIdentityResult:
    """Result from DHT identity operations"""
    action_hash: str
    did: str
    assurance_level: str
    sybil_resistance: float


class HolochainIdentityDHTClient:
    """
    Python client for Holochain Identity DHT

    Connects to Holochain conductor via WebSocket and calls identity zome functions.
    """

    def __init__(self, conductor_url: str = "ws://localhost:8888"):
        self.conductor_url = conductor_url
        self.websocket = None
        self.request_id = 0

    async def connect(self):
        """Connect to Holochain conductor"""
        self.websocket = await websockets.connect(self.conductor_url)
        logger.info(f"Connected to Holochain conductor: {self.conductor_url}")

    async def disconnect(self):
        """Disconnect from conductor"""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from Holochain conductor")

    async def _call_zome(
        self,
        cell_id: List[bytes],
        zome_name: str,
        fn_name: str,
        payload: Dict
    ) -> Dict:
        """Call a zome function"""
        self.request_id += 1

        request = {
            "id": self.request_id,
            "type": "call_zome",
            "data": {
                "cell_id": cell_id,
                "zome_name": zome_name,
                "fn_name": fn_name,
                "payload": payload
            }
        }

        await self.websocket.send(json.dumps(request))
        response_str = await self.websocket.recv()
        response = json.loads(response_str)

        if response.get("type") == "error":
            raise Exception(f"Holochain error: {response.get('data')}")

        return response.get("data")

    # DID Registry Zome

    async def create_did(
        self,
        cell_id: List[bytes],
        did_document: Dict
    ) -> str:
        """Create DID on DHT"""
        result = await self._call_zome(
            cell_id=cell_id,
            zome_name="did_registry",
            fn_name="create_did",
            payload={"document": did_document}
        )

        action_hash = result["action_hash"]
        logger.info(f"DID created on DHT: {did_document['id']}, hash: {action_hash}")
        return action_hash

    async def resolve_did(
        self,
        cell_id: List[bytes],
        did: str
    ) -> Optional[Dict]:
        """Resolve DID from DHT"""
        result = await self._call_zome(
            cell_id=cell_id,
            zome_name="did_registry",
            fn_name="resolve_did",
            payload={"did": did}
        )

        return result.get("document")

    # Identity Store Zome

    async def store_identity_factors(
        self,
        cell_id: List[bytes],
        did: str,
        factors: List[Dict]
    ) -> str:
        """Store identity factors on DHT"""
        result = await self._call_zome(
            cell_id=cell_id,
            zome_name="identity_store",
            fn_name="store_identity_factors",
            payload={
                "did": did,
                "factors": factors
            }
        )

        return result["action_hash"]

    async def get_identity_factors(
        self,
        cell_id: List[bytes],
        did: str
    ) -> Optional[Dict]:
        """Get identity factors from DHT"""
        result = await self._call_zome(
            cell_id=cell_id,
            zome_name="identity_store",
            fn_name="get_identity_factors",
            payload={"did": did}
        )

        return result.get("factors")

    async def store_verifiable_credential(
        self,
        cell_id: List[bytes],
        credential: Dict
    ) -> str:
        """Store verifiable credential on DHT"""
        result = await self._call_zome(
            cell_id=cell_id,
            zome_name="identity_store",
            fn_name="store_verifiable_credential",
            payload={"credential": credential}
        )

        return result["action_hash"]

    async def compute_identity_signals(
        self,
        cell_id: List[bytes],
        did: str
    ) -> Dict:
        """Compute identity signals from factors and credentials"""
        result = await self._call_zome(
            cell_id=cell_id,
            zome_name="identity_store",
            fn_name="compute_identity_signals",
            payload={"did": did}
        )

        return result["signals"]

    # Reputation Sync Zome

    async def store_reputation_entry(
        self,
        cell_id: List[bytes],
        reputation_entry: Dict
    ) -> str:
        """Store reputation entry on DHT"""
        result = await self._call_zome(
            cell_id=cell_id,
            zome_name="reputation_sync",
            fn_name="store_reputation_entry",
            payload={"entry": reputation_entry}
        )

        return result["action_hash"]

    async def get_aggregated_reputation(
        self,
        cell_id: List[bytes],
        did: str
    ) -> Optional[Dict]:
        """Get aggregated reputation from DHT"""
        result = await self._call_zome(
            cell_id=cell_id,
            zome_name="reputation_sync",
            fn_name="get_aggregated_reputation",
            payload={"did": did}
        )

        return result.get("aggregated")

    # Guardian Graph Zome

    async def store_guardian_network(
        self,
        cell_id: List[bytes],
        guardian_network: Dict,
        guardian_graph: Optional[Dict] = None
    ) -> str:
        """Store guardian network on DHT"""
        result = await self._call_zome(
            cell_id=cell_id,
            zome_name="guardian_graph",
            fn_name="store_guardian_network",
            payload={
                "network": guardian_network,
                "guardian_graph": guardian_graph
            }
        )

        return result["action_hash"]

    async def validate_guardian_network(
        self,
        cell_id: List[bytes],
        did: str,
        graph: Dict
    ) -> Dict:
        """Validate guardian network for cartel detection"""
        result = await self._call_zome(
            cell_id=cell_id,
            zome_name="guardian_graph",
            fn_name="validate_guardian_network",
            payload={
                "did": did,
                "graph": graph
            }
        )

        return result["validation"]


# Example usage
async def example_usage():
    client = HolochainIdentityDHTClient("ws://localhost:8888")
    await client.connect()

    # Create DID
    did_document = {
        "id": "did:mycelix:abc123",
        "controller": "...",  # AgentPubKey
        "verification_methods": [...],
        "authentication": [...],
        "created": ...,
        "updated": ...,
        "proof": {...}
    }

    cell_id = [b"dna_hash", b"agent_pubkey"]
    action_hash = await client.create_did(cell_id, did_document)
    print(f"DID created: {action_hash}")

    # Resolve DID
    resolved = await client.resolve_did(cell_id, "did:mycelix:abc123")
    print(f"DID resolved: {resolved}")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(example_usage())
```

### Integration with Identity Coordinator

```python
# src/zerotrustml/identity/dht_integration.py

from typing import List, Optional, Dict
from zerotrustml.identity import DIDManager, IdentityFactor, VerifiableCredential
from zerotrustml.holochain.identity_dht_client import HolochainIdentityDHTClient
import logging

logger = logging.getLogger(__name__)


class IdentityDHTIntegration:
    """
    Integration layer between Identity Coordinator and Holochain DHT

    Responsibilities:
    1. Translate Python identity objects to Holochain format
    2. Store identity on DHT during node registration
    3. Resolve identity from DHT when needed
    4. Sync reputation to DHT
    """

    def __init__(self, dht_client: HolochainIdentityDHTClient, cell_id: List[bytes]):
        self.dht_client = dht_client
        self.cell_id = cell_id

    async def register_identity_on_dht(
        self,
        did: str,
        did_document: Dict,
        factors: List[IdentityFactor],
        credentials: List[VerifiableCredential],
        guardian_graph: Optional[Dict] = None
    ) -> Dict:
        """
        Register complete identity on DHT

        Returns:
            Dictionary with action hashes for all stored components
        """
        results = {}

        # 1. Create DID document
        results["did_action_hash"] = await self.dht_client.create_did(
            self.cell_id,
            did_document
        )

        # 2. Store identity factors
        factors_dict = [self._factor_to_dict(f) for f in factors]
        results["factors_action_hash"] = await self.dht_client.store_identity_factors(
            self.cell_id,
            did,
            factors_dict
        )

        # 3. Store credentials
        for credential in credentials:
            cred_dict = self._credential_to_dict(credential)
            cred_hash = await self.dht_client.store_verifiable_credential(
                self.cell_id,
                cred_dict
            )
            results[f"credential_{credential.id}"] = cred_hash

        # 4. Compute and store identity signals
        signals = await self.dht_client.compute_identity_signals(
            self.cell_id,
            did
        )
        results["signals"] = signals

        # 5. Store guardian network if provided
        if guardian_graph:
            guardian_network = {
                "did": did,
                "guardians": guardian_graph.get("nodes", []),
                "threshold": guardian_graph.get("threshold", 3),
                "created": ...,  # timestamp
                "updated": ...
            }
            results["guardian_action_hash"] = await self.dht_client.store_guardian_network(
                self.cell_id,
                guardian_network,
                guardian_graph
            )

        logger.info(f"Identity registered on DHT: {did}")
        return results

    async def resolve_identity_from_dht(self, did: str) -> Optional[Dict]:
        """
        Resolve complete identity from DHT

        Returns:
            Dictionary with all identity components
        """
        # 1. Resolve DID document
        did_document = await self.dht_client.resolve_did(self.cell_id, did)
        if not did_document:
            return None

        # 2. Get identity factors
        factors = await self.dht_client.get_identity_factors(self.cell_id, did)

        # 3. Get identity signals
        signals = await self.dht_client.compute_identity_signals(self.cell_id, did)

        # 4. Get aggregated reputation
        reputation = await self.dht_client.get_aggregated_reputation(self.cell_id, did)

        return {
            "did": did,
            "did_document": did_document,
            "factors": factors,
            "signals": signals,
            "reputation": reputation
        }

    async def sync_reputation_to_dht(
        self,
        did: str,
        network_id: str,
        score: float,
        source: str,
        assurance_level: str,
        proof: bytes
    ) -> str:
        """
        Sync reputation entry to DHT
        """
        reputation_entry = {
            "did": did,
            "network_id": network_id,
            "score": score,
            "source": source,
            "assurance_level": assurance_level,
            "timestamp": ...,  # current timestamp
            "proof": proof.hex()
        }

        action_hash = await self.dht_client.store_reputation_entry(
            self.cell_id,
            reputation_entry
        )

        logger.info(f"Reputation synced to DHT: {did} on {network_id}")
        return action_hash

    def _factor_to_dict(self, factor: IdentityFactor) -> Dict:
        """Convert IdentityFactor to DHT format"""
        return {
            "factor_id": factor.factor_id,
            "factor_type": factor.factor_type,
            "category": factor.category.value,
            "status": factor.status.value,
            "metadata": "",  # Serialize factor-specific data
            "added": ...,  # timestamp
            "last_verified": ...
        }

    def _credential_to_dict(self, credential: VerifiableCredential) -> Dict:
        """Convert VerifiableCredential to DHT format"""
        return {
            "id": credential.id,
            "issuer_did": credential.issuer_did,
            "subject_did": credential.subject_did,
            "vc_type": credential.vc_type.value,
            "claims": "",  # Serialize claims
            "proof": credential.proof.hex(),
            "issued_at": credential.issued_at,
            "expires_at": credential.expires_at
        }
```

---

## Success Criteria

### Week 5-6 Completion Checklist

#### Phase 1: DID Registry (Days 1-2)
- [ ] DIDDocument entry type defined
- [ ] `create_did()` zome function implemented
- [ ] `resolve_did()` zome function implemented
- [ ] `update_did()` zome function implemented
- [ ] DID validation rules working
- [ ] Path-based DID resolution working
- [ ] **Test**: Create 10 DIDs, resolve all successfully

#### Phase 2: Identity Store (Days 3-4)
- [ ] IdentityFactors entry type defined
- [ ] VerifiableCredential entry type defined
- [ ] IdentitySignals entry type defined
- [ ] `store_identity_factors()` implemented
- [ ] `get_identity_factors()` implemented
- [ ] `store_verifiable_credential()` implemented
- [ ] `compute_identity_signals()` implemented
- [ ] Assurance level computation working
- [ ] **Test**: Store factors for 10 nodes, retrieve all

#### Phase 3: Reputation Sync (Days 5-6)
- [ ] ReputationEntry entry type defined
- [ ] AggregatedReputation entry type defined
- [ ] `store_reputation_entry()` implemented
- [ ] `get_reputation_for_network()` implemented
- [ ] `get_aggregated_reputation()` implemented
- [ ] Cross-network aggregation working
- [ ] **Test**: Sync reputation from 3 networks, aggregate correctly

#### Phase 4: Guardian Graph (Days 7-8)
- [ ] GuardianNetwork entry type defined
- [ ] GuardianValidation entry type defined
- [ ] `store_guardian_network()` implemented
- [ ] `validate_guardian_network()` implemented
- [ ] Cartel detection working
- [ ] Network diversity computation working
- [ ] **Test**: Create 5 guardian networks, detect cartel in 1

#### Python Integration
- [ ] HolochainIdentityDHTClient implemented
- [ ] WebSocket connection working
- [ ] All zome functions callable from Python
- [ ] IdentityDHTIntegration layer working
- [ ] **Test**: Register node via Python → DHT → resolve back

#### Integration with Identity Coordinator
- [ ] IdentityCoordinator updated to use DHT
- [ ] Node registration stores on DHT
- [ ] Reputation queries use DHT
- [ ] Guardian graphs validated via DHT
- [ ] **Test**: Full FL workload with DHT identity

---

## Performance Targets

| Metric | Target | How to Measure |
|--------|--------|----------------|
| DID Resolution | <200ms | `resolve_did()` call time |
| Identity Signal Computation | <100ms | `compute_identity_signals()` time |
| Reputation Aggregation | <500ms | `get_aggregated_reputation()` time |
| DHT Gossip Latency | <5s | Time for entry to propagate to 90% of peers |
| Storage per Node | <50KB | Total entry sizes for one identity |

---

## Testing Strategy

### Unit Tests (Per Zome)
1. **Entry Creation**: Can create each entry type
2. **Entry Retrieval**: Can retrieve entries by ID/path
3. **Validation**: Invalid entries rejected
4. **Links**: Links created and queryable

### Integration Tests (Cross-Zome)
1. **Full Registration**: Create DID → store factors → compute signals
2. **Resolution**: Store identity → resolve from different node
3. **Reputation Flow**: Store entry → aggregate → query global score
4. **Guardian Validation**: Store network → validate → detect cartel

### End-to-End Tests (Python + Holochain)
1. **Node Registration**: Python coordinator → DHT → verify stored
2. **Identity Resolution**: Query DHT from Python → get full identity
3. **Reputation Sync**: Submit gradient → reputation → DHT sync
4. **Byzantine Attack**: Attack with DHT-verified identities → detect

---

## Deployment Plan

### Development (Week 5-6)
1. **Local Holochain Conductor**: Single node for development
2. **Python Integration**: Connect coordinator to local conductor
3. **Unit + Integration Tests**: All tests passing locally

### Staging (Week 7)
1. **Multi-Node DHT**: 5-10 nodes on local network
2. **Stress Testing**: 100+ identities, 1000+ reputation entries
3. **Byzantine Testing**: DHT-verified Byzantine attacks

### Production (Week 8+)
1. **Mainnet Launch**: Public Holochain DHT
2. **Monitoring**: DHT health, gossip latency, entry counts
3. **Upgrade Path**: DNA versioning for future updates

---

## Next Steps (After Design Approval)

1. **Create Rust Project Structure** for identity DNA
2. **Implement Phase 1**: DID Registry zome
3. **Test Phase 1**: Unit tests + local resolution
4. **Implement Phase 2**: Identity Store zome
5. **Test Phase 2**: Integration with DID registry
6. **Implement Phase 3**: Reputation Sync zome
7. **Implement Phase 4**: Guardian Graph zome
8. **Build Python Client**: WebSocket integration
9. **Integrate with Coordinator**: Update Identity Coordinator
10. **End-to-End Testing**: Full FL workload with DHT

---

**Status**: Ready for implementation 🚀

**Estimated Completion**: End of Week 6 (November 22, 2025)

**Documentation Created By**: Claude Code (Autonomous Development)
**Next Action**: Begin Phase 1 implementation (DID Registry Zome)

🍄 **Decentralized identity resolution via Holochain DHT - bridging reputation across the mycelial network** 🍄
