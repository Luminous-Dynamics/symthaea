# Mycelix Privacy Engineering Patterns

## Philosophy: Privacy as Foundation

> "Privacy is not about having something to hide. It is about having the power to selectively reveal oneself to the world."

This document provides concrete implementation patterns for privacy across the Mycelix ecosystem—not abstract principles, but actual code patterns and architectural decisions.

---

## Part 1: Privacy Architecture Overview

### 1.1 Privacy Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                         │
│  • Privacy controls visible and accessible                      │
│  • Clear indicators of what's shared                            │
│  • Consent flows integrated naturally                           │
├─────────────────────────────────────────────────────────────────┤
│                    APPLICATION LAYER                            │
│  • Privacy-by-default in all hApps                              │
│  • Minimal data collection                                      │
│  • Purpose limitation enforced                                  │
├─────────────────────────────────────────────────────────────────┤
│                    DATA LAYER                                   │
│  • Local-first storage                                          │
│  • Encrypted at rest                                            │
│  • Selective sharing via capabilities                           │
├─────────────────────────────────────────────────────────────────┤
│                    NETWORK LAYER                                │
│  • End-to-end encryption in transit                             │
│  • Minimal metadata exposure                                    │
│  • DHT with privacy protections                                 │
├─────────────────────────────────────────────────────────────────┤
│                    INFRASTRUCTURE LAYER                         │
│  • No central data stores                                       │
│  • Distributed validation                                       │
│  • No admin access to user data                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Privacy Principles in Code

```rust
// Privacy principles as code constraints
pub trait PrivacyCompliant {
    /// Data must have explicit purpose
    fn data_purpose(&self) -> DataPurpose;

    /// Minimum data collected for purpose
    fn is_minimal(&self) -> bool;

    /// User has consented to this use
    fn has_consent(&self, use: DataUse) -> bool;

    /// Data can be deleted
    fn is_deletable(&self) -> bool;

    /// Data is encrypted appropriately
    fn encryption_level(&self) -> EncryptionLevel;
}

// Every data structure implements privacy compliance
#[hdk_entry_helper]
pub struct PrivateEntry<T: PrivacyCompliant> {
    pub data: T,
    pub visibility: Visibility,
    pub consent_record: ConsentRecord,
    pub retention_policy: RetentionPolicy,
    pub encryption: EncryptionMetadata,
}
```

### 1.3 Data Classification System

```typescript
interface DataClassification {
  // Classification levels
  levels: {
    public: {
      description: "Visible to anyone";
      examples: ["community-name", "public-proposals", "published-content"];
      protection: "integrity-only";
    };

    community: {
      description: "Visible to community members";
      examples: ["member-list", "internal-discussions", "shared-resources"];
      protection: "community-encryption";
    };

    connections: {
      description: "Visible to direct connections";
      examples: ["profile-details", "activity-status", "location"];
      protection: "connection-encryption";
    };

    private: {
      description: "Visible only to self";
      examples: ["drafts", "personal-notes", "browsing-history"];
      protection: "personal-encryption";
    };

    sensitive: {
      description: "Special handling required";
      examples: ["health-data", "financial-details", "identity-documents"];
      protection: "enhanced-encryption-plus-access-controls";
    };
  };

  // Default classification
  defaultClassification: 'private'; // Privacy by default
}
```

---

## Part 2: Identity & Pseudonymity Patterns

### 2.1 Layered Identity Architecture

```typescript
interface LayeredIdentity {
  // Layer 0: Cryptographic root (never shared)
  cryptographicRoot: {
    type: "Ed25519 keypair";
    storage: "secure-enclave-or-encrypted-local";
    usage: "derive-all-other-keys";
    backup: "user-controlled-seed-phrase";
  };

  // Layer 1: Agent identity (Holochain level)
  agentIdentity: {
    type: "AgentPubKey";
    derivedFrom: "cryptographic-root";
    usage: "DHT-participation";
    linkability: "linkable-within-DNA";
  };

  // Layer 2: Community identity
  communityIdentity: {
    type: "CommunityMemberId";
    derivedFrom: "agent-identity-plus-community";
    usage: "community-participation";
    linkability: "unlinkable-across-communities-by-default";
  };

  // Layer 3: Contextual pseudonyms
  contextualPseudonyms: {
    type: "ContextualId";
    derivedFrom: "community-identity-plus-context";
    usage: "specific-interactions";
    linkability: "unlinkable-across-contexts";
    examples: ["discussion-pseudonym", "voting-id", "marketplace-handle"];
  };

  // Layer 4: Display identity
  displayIdentity: {
    type: "UserChosenName";
    chosenBy: "user";
    usage: "human-readable-identification";
    changeable: true;
  };
}
```

### 2.2 Pseudonymous Participation Patterns

```rust
// Pattern: Unlinkable community participation
pub fn create_unlinkable_community_id(
    agent_key: &AgentPubKey,
    community_id: &CommunityId,
    blinding_factor: &BlindingFactor,
) -> CommunityMemberId {
    // Derive deterministic but unlinkable ID
    let derived = derive_key(
        agent_key,
        &format!("community:{}", community_id),
        blinding_factor,
    );

    CommunityMemberId::from(derived)
}

// Pattern: Context-specific pseudonyms
pub fn create_contextual_pseudonym(
    community_id: &CommunityMemberId,
    context: &Context,
    nonce: &Nonce,
) -> ContextualPseudonym {
    // Different pseudonym per context
    // User can prove ownership without revealing link
    let pseudonym = derive_pseudonym(
        community_id,
        context,
        nonce,
    );

    ContextualPseudonym {
        id: pseudonym,
        context: context.clone(),
        proof_capability: generate_ownership_proof_capability(community_id, &pseudonym),
    }
}

// Pattern: Selective linkability
pub fn create_linkability_proof(
    pseudonym_a: &ContextualPseudonym,
    pseudonym_b: &ContextualPseudonym,
    community_id: &CommunityMemberId,
) -> LinkabilityProof {
    // User can prove two pseudonyms are same person
    // Without revealing their actual identity
    LinkabilityProof::generate(
        pseudonym_a,
        pseudonym_b,
        community_id,
    )
}
```

### 2.3 Anonymous Credentials

```rust
// Pattern: Zero-knowledge membership proof
pub struct MembershipCredential {
    pub community_id: CommunityId,
    pub issued_at: Timestamp,
    pub expiry: Timestamp,
    pub attributes: Vec<Attribute>,
    pub signature: BlindSignature,
}

impl MembershipCredential {
    // Prove membership without revealing identity
    pub fn prove_membership(
        &self,
        challenge: &Challenge,
    ) -> MembershipProof {
        // ZK proof that:
        // 1. Holder has valid credential for this community
        // 2. Credential is not expired
        // 3. Optionally: holder has certain attributes
        // Without revealing: which specific member
        ZKProof::membership_proof(self, challenge)
    }

    // Prove attribute without revealing identity
    pub fn prove_attribute(
        &self,
        attribute: &AttributeType,
        challenge: &Challenge,
    ) -> AttributeProof {
        // Prove e.g., "is over 18" without revealing age or identity
        ZKProof::attribute_proof(self, attribute, challenge)
    }
}

// Usage example: Anonymous voting
pub fn cast_anonymous_vote(
    credential: &MembershipCredential,
    proposal_id: &ProposalId,
    vote: &Vote,
) -> AnonymousVote {
    let eligibility_proof = credential.prove_membership(
        &Challenge::for_vote(proposal_id),
    );

    // Vote is valid but not linkable to voter
    AnonymousVote {
        proposal_id: proposal_id.clone(),
        vote: vote.clone(),
        eligibility_proof,
        nullifier: generate_vote_nullifier(credential, proposal_id), // Prevents double voting
    }
}
```

---

## Part 3: Data Minimization Patterns

### 3.1 Collection Minimization

```typescript
// Pattern: Collect only what's needed
interface MinimalDataCollection {
  // Anti-pattern: Over-collection
  badExample: {
    registration: {
      required: [
        "full-legal-name",    // Not needed
        "date-of-birth",      // Not needed
        "phone-number",       // Not needed
        "home-address",       // Not needed
        "government-id",      // Not needed
        "email"               // Only if needed for recovery
      ];
    };
  };

  // Pattern: Minimal collection
  goodExample: {
    registration: {
      required: [
        "display-name",       // User-chosen, changeable
        "public-key"          // For cryptographic identity
      ];
      optional: [
        "email",              // Only if user wants recovery/notifications
        "avatar"              // Only if user wants visual identity
      ];
      neverCollected: [
        "legal-name",         // Not needed for community participation
        "government-id",      // Not needed
        "location",           // Not needed for registration
        "phone"               // Not needed
      ];
    };
  };
}

// Pattern: Purpose-bound collection
interface PurposeBoundCollection {
  // Each data point has explicit purpose
  dataWithPurpose: {
    email: {
      purpose: "account-recovery-and-notifications";
      collectionMoment: "only-if-user-enables-feature";
      retention: "until-user-removes-or-deletes-account";
    };

    location: {
      purpose: "local-community-matching";
      collectionMoment: "only-when-searching-local";
      precision: "city-level-not-exact";
      retention: "transient-not-stored";
    };

    transactionHistory: {
      purpose: "economic-participation";
      collectionMoment: "when-transaction-occurs";
      retention: "per-community-policy-user-deletable";
    };
  };
}
```

### 3.2 Storage Minimization

```rust
// Pattern: Local-first with selective sync
pub struct LocalFirstStorage {
    // All user data stored locally first
    pub local_store: EncryptedLocalStore,

    // Only shared what user explicitly shares
    pub shared_entries: Vec<SharedEntryRef>,

    // Sync preferences
    pub sync_policy: SyncPolicy,
}

impl LocalFirstStorage {
    // Data stays local unless explicitly shared
    pub fn create_entry(&mut self, data: impl PrivacyCompliant) -> EntryHash {
        // Default: local only
        let entry = LocalEntry::new(data);
        self.local_store.insert(entry.clone());
        entry.hash()
    }

    // Explicit sharing action required
    pub fn share_entry(
        &mut self,
        entry_hash: &EntryHash,
        visibility: Visibility,
        consent: ConsentRecord,
    ) -> Result<SharedEntryRef, Error> {
        // Validate consent before sharing
        if !consent.is_valid_for(&visibility) {
            return Err(Error::InsufficientConsent);
        }

        let entry = self.local_store.get(entry_hash)?;
        let shared = SharedEntry::from_local(entry, visibility, consent);

        // Now sync to DHT
        self.shared_entries.push(shared.reference());
        Ok(shared.reference())
    }
}

// Pattern: Automatic data expiration
pub struct ExpiringEntry<T> {
    pub data: T,
    pub created: Timestamp,
    pub expires: Timestamp,
    pub auto_delete: bool,
}

impl<T> ExpiringEntry<T> {
    pub fn is_expired(&self) -> bool {
        Timestamp::now() > self.expires
    }

    // Automatic cleanup
    pub fn cleanup_expired(store: &mut Store) {
        store.entries
            .retain(|e| !e.is_expired() || !e.auto_delete);
    }
}
```

### 3.3 Aggregation Instead of Individual Data

```rust
// Pattern: Aggregate statistics without individual data
pub struct PrivacyPreservingAnalytics {
    // Never store individual data points
    aggregates: HashMap<MetricType, AggregateValue>,
}

impl PrivacyPreservingAnalytics {
    // Add to aggregate without storing individual value
    pub fn record_metric(&mut self, metric: MetricType, value: f64) {
        let agg = self.aggregates.entry(metric).or_default();

        // Only store count and sum (can derive mean)
        agg.count += 1;
        agg.sum += value;

        // For percentiles, use privacy-preserving sketches
        agg.sketch.add(value);

        // Individual value is NOT stored
    }

    // Get statistics without exposing individuals
    pub fn get_statistics(&self, metric: MetricType) -> Option<Statistics> {
        self.aggregates.get(&metric).map(|agg| {
            Statistics {
                count: agg.count,
                mean: agg.sum / agg.count as f64,
                percentiles: agg.sketch.get_percentiles(&[50, 90, 99]),
                // No individual values exposed
            }
        })
    }
}

// Pattern: Differential privacy for sensitive aggregates
pub fn add_differential_privacy(
    true_value: f64,
    epsilon: f64,  // Privacy budget
    sensitivity: f64,
) -> f64 {
    // Add Laplace noise to protect individual contributions
    let noise = laplace_noise(sensitivity / epsilon);
    true_value + noise
}

// Pattern: k-anonymity for shared data
pub fn ensure_k_anonymity(
    dataset: &mut Dataset,
    k: usize,
    quasi_identifiers: &[Column],
) -> Result<(), Error> {
    // Generalize quasi-identifiers until each combination
    // appears at least k times
    for qi in quasi_identifiers {
        dataset.generalize(qi)?;
    }

    // Suppress groups smaller than k
    dataset.suppress_small_groups(k)?;

    Ok(())
}
```

---

## Part 4: Encryption Patterns

### 4.1 Encryption at Rest

```rust
// Pattern: All local data encrypted
pub struct EncryptedLocalStore {
    master_key: MasterKey,  // Derived from user passphrase + device key
    entries: HashMap<EntryHash, EncryptedEntry>,
}

impl EncryptedLocalStore {
    // All entries encrypted before storage
    pub fn insert(&mut self, entry: impl Serialize) -> EntryHash {
        let plaintext = serialize(&entry);
        let nonce = generate_nonce();
        let ciphertext = encrypt(&self.master_key, &nonce, &plaintext);

        let encrypted = EncryptedEntry {
            ciphertext,
            nonce,
            hash: hash(&plaintext),
        };

        self.entries.insert(encrypted.hash.clone(), encrypted);
        encrypted.hash
    }

    // Decrypt on access
    pub fn get<T: Deserialize>(&self, hash: &EntryHash) -> Result<T, Error> {
        let encrypted = self.entries.get(hash).ok_or(Error::NotFound)?;
        let plaintext = decrypt(&self.master_key, &encrypted.nonce, &encrypted.ciphertext)?;
        deserialize(&plaintext)
    }
}

// Pattern: Key derivation
pub struct KeyHierarchy {
    // Root key from user credential
    root: RootKey,

    // Derived keys for different purposes
    derived: HashMap<KeyPurpose, DerivedKey>,
}

impl KeyHierarchy {
    pub fn from_passphrase(passphrase: &str, salt: &Salt) -> Self {
        let root = derive_root_key(passphrase, salt, KeyDerivationParams::secure());

        KeyHierarchy {
            root,
            derived: HashMap::new(),
        }
    }

    pub fn get_key(&mut self, purpose: KeyPurpose) -> &DerivedKey {
        self.derived.entry(purpose).or_insert_with(|| {
            derive_purpose_key(&self.root, &purpose)
        })
    }
}
```

### 4.2 End-to-End Encryption

```rust
// Pattern: E2E encrypted messaging
pub struct E2EMessage {
    pub sender: AgentPubKey,
    pub recipient: AgentPubKey,
    pub encrypted_content: EncryptedContent,
    pub signature: Signature,
}

impl E2EMessage {
    pub fn encrypt(
        sender_key: &SigningKey,
        recipient_pubkey: &AgentPubKey,
        content: &MessageContent,
    ) -> Self {
        // Generate ephemeral key for this message
        let ephemeral = generate_ephemeral_keypair();

        // Derive shared secret via X25519
        let shared_secret = derive_shared_secret(
            &ephemeral.secret,
            recipient_pubkey,
        );

        // Encrypt content with shared secret
        let encrypted = encrypt_authenticated(
            &shared_secret,
            &serialize(content),
        );

        // Sign the encrypted message
        let signature = sender_key.sign(&encrypted.as_bytes());

        E2EMessage {
            sender: sender_key.public_key(),
            recipient: recipient_pubkey.clone(),
            encrypted_content: EncryptedContent {
                ephemeral_pubkey: ephemeral.public,
                ciphertext: encrypted,
            },
            signature,
        }
    }

    pub fn decrypt(
        &self,
        recipient_key: &DecryptionKey,
    ) -> Result<MessageContent, Error> {
        // Verify signature first
        verify_signature(&self.sender, &self.encrypted_content.as_bytes(), &self.signature)?;

        // Derive shared secret
        let shared_secret = derive_shared_secret(
            recipient_key,
            &self.encrypted_content.ephemeral_pubkey,
        );

        // Decrypt
        let plaintext = decrypt_authenticated(&shared_secret, &self.encrypted_content.ciphertext)?;
        deserialize(&plaintext)
    }
}

// Pattern: Group encryption with key rotation
pub struct GroupEncryptionKey {
    pub group_id: GroupId,
    pub epoch: u64,
    pub key: SymmetricKey,
    pub member_keys: HashMap<AgentPubKey, EncryptedGroupKey>,
}

impl GroupEncryptionKey {
    // Each member has group key encrypted to their public key
    pub fn create_for_group(
        group_id: GroupId,
        members: &[AgentPubKey],
    ) -> Self {
        let group_key = generate_symmetric_key();

        let member_keys = members.iter().map(|member| {
            let encrypted = encrypt_to_pubkey(member, &group_key);
            (member.clone(), encrypted)
        }).collect();

        GroupEncryptionKey {
            group_id,
            epoch: 0,
            key: group_key,
            member_keys,
        }
    }

    // Rotate key when membership changes
    pub fn rotate(&mut self, new_members: &[AgentPubKey]) {
        let new_key = generate_symmetric_key();
        self.epoch += 1;

        self.member_keys = new_members.iter().map(|member| {
            let encrypted = encrypt_to_pubkey(member, &new_key);
            (member.clone(), encrypted)
        }).collect();

        self.key = new_key;
    }
}
```

### 4.3 Secure Multiparty Computation Patterns

```rust
// Pattern: Secret sharing for sensitive data
pub struct SecretSharing {
    threshold: usize,
    total_shares: usize,
}

impl SecretSharing {
    // Split secret into shares
    pub fn split(&self, secret: &[u8]) -> Vec<SecretShare> {
        shamir_split(secret, self.threshold, self.total_shares)
    }

    // Reconstruct from threshold shares
    pub fn reconstruct(&self, shares: &[SecretShare]) -> Result<Vec<u8>, Error> {
        if shares.len() < self.threshold {
            return Err(Error::InsufficientShares);
        }
        shamir_reconstruct(shares)
    }
}

// Pattern: Secure aggregation (e.g., for voting)
pub fn secure_sum(
    values: &[EncryptedValue],
    decryption_threshold: usize,
) -> Result<u64, Error> {
    // Each value encrypted with additive homomorphic encryption
    // Can sum without decrypting individual values

    let encrypted_sum = values.iter().fold(
        EncryptedValue::zero(),
        |acc, v| acc.add(v),
    );

    // Only sum can be decrypted (requires threshold)
    threshold_decrypt(encrypted_sum, decryption_threshold)
}
```

---

## Part 5: Access Control Patterns

### 5.1 Capability-Based Access

```rust
// Pattern: Capability tokens for access
#[hdk_entry_helper]
pub struct Capability {
    pub capability_id: CapabilityId,
    pub grantor: AgentPubKey,
    pub grantee: AgentPubKey,
    pub resource: ResourceId,
    pub permissions: Permissions,
    pub constraints: Constraints,
    pub expires: Option<Timestamp>,
    pub revocable: bool,
    pub delegatable: bool,
}

impl Capability {
    // Create new capability
    pub fn grant(
        grantor: &AgentPubKey,
        grantee: &AgentPubKey,
        resource: &ResourceId,
        permissions: Permissions,
    ) -> Self {
        Capability {
            capability_id: generate_capability_id(),
            grantor: grantor.clone(),
            grantee: grantee.clone(),
            resource: resource.clone(),
            permissions,
            constraints: Constraints::default(),
            expires: None,
            revocable: true,
            delegatable: false,
        }
    }

    // Check if capability authorizes action
    pub fn authorizes(&self, action: &Action) -> bool {
        self.permissions.allows(action) &&
        self.constraints.satisfied_by(action) &&
        !self.is_expired() &&
        !self.is_revoked()
    }

    // Delegate capability (if allowed)
    pub fn delegate(
        &self,
        new_grantee: &AgentPubKey,
        reduced_permissions: Permissions,
    ) -> Result<Capability, Error> {
        if !self.delegatable {
            return Err(Error::NotDelegatable);
        }

        if !self.permissions.includes(&reduced_permissions) {
            return Err(Error::CannotExpandPermissions);
        }

        Ok(Capability {
            capability_id: generate_capability_id(),
            grantor: self.grantee.clone(),  // Original grantee becomes grantor
            grantee: new_grantee.clone(),
            resource: self.resource.clone(),
            permissions: reduced_permissions,
            constraints: self.constraints.clone(),
            expires: self.expires,
            revocable: true,
            delegatable: false,  // Cannot further delegate by default
        })
    }
}

// Pattern: Fine-grained permissions
pub struct Permissions {
    pub read: bool,
    pub write: bool,
    pub delete: bool,
    pub share: bool,
    pub admin: bool,
    pub custom: HashMap<String, bool>,
}
```

### 5.2 Consent Management

```rust
// Pattern: Explicit consent records
#[hdk_entry_helper]
pub struct ConsentRecord {
    pub consent_id: ConsentId,
    pub data_subject: AgentPubKey,
    pub data_controller: AgentPubKey,
    pub purpose: Purpose,
    pub data_categories: Vec<DataCategory>,
    pub processing_types: Vec<ProcessingType>,
    pub granted_at: Timestamp,
    pub expires: Option<Timestamp>,
    pub withdrawn_at: Option<Timestamp>,
    pub evidence: ConsentEvidence,
}

impl ConsentRecord {
    // Check if consent is currently valid
    pub fn is_valid(&self) -> bool {
        self.withdrawn_at.is_none() &&
        self.expires.map_or(true, |exp| Timestamp::now() < exp)
    }

    // Withdraw consent
    pub fn withdraw(&mut self) -> Result<(), Error> {
        if self.withdrawn_at.is_some() {
            return Err(Error::AlreadyWithdrawn);
        }
        self.withdrawn_at = Some(Timestamp::now());
        Ok(())
    }

    // Check if consent covers specific use
    pub fn covers(&self, use_request: &UseRequest) -> bool {
        self.is_valid() &&
        self.purpose.includes(&use_request.purpose) &&
        use_request.data_categories.iter().all(|dc|
            self.data_categories.contains(dc)
        ) &&
        use_request.processing_types.iter().all(|pt|
            self.processing_types.contains(pt)
        )
    }
}

// Pattern: Consent-gated operations
pub fn with_consent<T>(
    operation: impl FnOnce() -> T,
    required_consent: &ConsentRequirement,
    available_consents: &[ConsentRecord],
) -> Result<T, Error> {
    // Check if any consent covers this operation
    let has_consent = available_consents.iter().any(|c|
        c.covers(&required_consent.to_use_request())
    );

    if !has_consent {
        return Err(Error::ConsentRequired(required_consent.clone()));
    }

    Ok(operation())
}
```

### 5.3 Privacy-Preserving Authorization

```rust
// Pattern: Prove authorization without revealing identity
pub struct BlindAuthorization {
    pub resource: ResourceId,
    pub action: Action,
    pub proof: AuthorizationProof,
}

impl BlindAuthorization {
    // Create authorization proof without revealing who you are
    pub fn create(
        capability: &Capability,
        resource: &ResourceId,
        action: &Action,
    ) -> Result<Self, Error> {
        if !capability.authorizes(action) {
            return Err(Error::NotAuthorized);
        }

        // ZK proof that:
        // 1. Prover holds a valid capability for this resource
        // 2. Capability includes the requested action
        // 3. Capability is not expired or revoked
        // Without revealing: which specific capability or who holds it

        let proof = ZKProof::authorization_proof(
            capability,
            resource,
            action,
        );

        Ok(BlindAuthorization {
            resource: resource.clone(),
            action: action.clone(),
            proof,
        })
    }

    // Verify authorization without learning identity
    pub fn verify(&self) -> bool {
        self.proof.verify(&self.resource, &self.action)
    }
}
```

---

## Part 6: Network Privacy Patterns

### 6.1 Metadata Protection

```rust
// Pattern: Minimize metadata in DHT
pub struct MetadataMinimalEntry {
    // Only essential metadata stored on DHT
    pub entry_hash: EntryHash,
    pub entry_type: EntryType,
    pub author: AgentPubKey,  // Required by Holochain
    pub timestamp: Timestamp,

    // These are NOT stored on DHT
    // - Access patterns
    // - Query history
    // - View counts
    // - IP addresses
}

// Pattern: Onion routing for sensitive operations
pub async fn route_through_mixnet<T>(
    request: T,
    hops: usize,
) -> Result<Response, Error> {
    let path = select_random_path(hops);

    // Encrypt in layers (like onion)
    let mut onion = serialize(&request);
    for hop in path.iter().rev() {
        onion = encrypt_layer(&hop.public_key, &onion);
    }

    // Send through path
    let mut current = onion;
    for hop in path.iter() {
        current = hop.relay(current).await?;
    }

    deserialize(&current)
}

// Pattern: Timing attack mitigation
pub fn add_random_delay() {
    let delay = random_delay(10, 100); // 10-100ms
    sleep(delay);
}

pub fn batch_operations(ops: Vec<Operation>) -> BatchedOperations {
    // Batch multiple operations to prevent timing analysis
    BatchedOperations {
        operations: ops,
        execute_at: next_batch_window(),
    }
}
```

### 6.2 Private Queries

```rust
// Pattern: Private information retrieval
pub async fn private_query(
    query: Query,
    servers: &[Server],
) -> Result<QueryResult, Error> {
    // Split query so no single server learns what you're looking for
    let query_shares = split_query(&query, servers.len());

    // Each server processes partial query
    let partial_results: Vec<_> = join_all(
        servers.iter().zip(query_shares.iter()).map(|(server, share)| {
            server.process_partial(share)
        })
    ).await;

    // Combine results locally
    combine_partial_results(&partial_results)
}

// Pattern: Encrypted search
pub fn searchable_encryption(
    plaintext: &str,
    search_key: &SearchKey,
) -> SearchableEncrypted {
    // Create encrypted tokens for each searchable term
    let tokens: Vec<_> = tokenize(plaintext)
        .into_iter()
        .map(|token| create_search_token(&token, search_key))
        .collect();

    // Encrypt the content
    let encrypted = encrypt(plaintext, search_key.encryption_key());

    SearchableEncrypted {
        encrypted_content: encrypted,
        search_tokens: tokens,
    }
}

pub fn search_encrypted(
    query: &str,
    search_key: &SearchKey,
    encrypted_docs: &[SearchableEncrypted],
) -> Vec<&SearchableEncrypted> {
    let query_token = create_search_token(query, search_key);

    encrypted_docs.iter()
        .filter(|doc| doc.search_tokens.contains(&query_token))
        .collect()
}
```

### 6.3 Decentralized Anonymity

```rust
// Pattern: Ring signatures for anonymous attestations
pub fn create_ring_signature(
    message: &[u8],
    signer_key: &SigningKey,
    ring: &[PublicKey],  // Group of possible signers
) -> RingSignature {
    // Anyone in ring could have signed
    // No way to determine which
    ring_sign(message, signer_key, ring)
}

pub fn verify_ring_signature(
    message: &[u8],
    signature: &RingSignature,
    ring: &[PublicKey],
) -> bool {
    // Verifies someone in ring signed
    // Without revealing who
    ring_verify(message, signature, ring)
}

// Pattern: Mixnets for anonymous messaging
pub struct MixNetwork {
    nodes: Vec<MixNode>,
    path_length: usize,
}

impl MixNetwork {
    pub async fn send_anonymous(
        &self,
        message: &Message,
        recipient: &AgentPubKey,
    ) -> Result<(), Error> {
        // Select random path through mix nodes
        let path = self.select_random_path();

        // Encrypt in layers
        let mut onion = encrypt_to_recipient(message, recipient);
        for node in path.iter().rev() {
            onion = node.wrap_layer(onion)?;
        }

        // Send through network
        // Each node removes one layer and forwards
        path[0].receive(onion).await
    }
}
```

---

## Part 7: User Control Patterns

### 7.1 Data Portability

```rust
// Pattern: Export all personal data
pub async fn export_all_data(
    agent: &AgentPubKey,
) -> Result<DataExport, Error> {
    let mut export = DataExport::new();

    // Collect from all hApps
    for happ in get_all_happs() {
        let happ_data = happ.get_all_user_data(agent).await?;
        export.add_happ_data(happ.id(), happ_data);
    }

    // Include metadata
    export.add_metadata(ExportMetadata {
        exported_at: Timestamp::now(),
        format_version: EXPORT_FORMAT_VERSION,
        agent: agent.clone(),
    });

    Ok(export)
}

// Pattern: Standard export format
pub struct DataExport {
    pub metadata: ExportMetadata,
    pub identity: IdentityExport,
    pub happs: HashMap<HappId, HappExport>,
    pub relationships: Vec<RelationshipExport>,
    pub credentials: Vec<CredentialExport>,
}

impl DataExport {
    // Export to standard format
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }

    // Export to encrypted archive
    pub fn to_encrypted_archive(
        &self,
        passphrase: &str,
    ) -> EncryptedArchive {
        let key = derive_key(passphrase);
        let plaintext = self.to_json();
        EncryptedArchive::create(&key, &plaintext)
    }
}
```

### 7.2 Right to Deletion

```rust
// Pattern: Comprehensive deletion
pub async fn delete_all_data(
    agent: &AgentPubKey,
    confirmation: &DeletionConfirmation,
) -> Result<DeletionReport, Error> {
    // Verify confirmation
    verify_deletion_confirmation(confirmation)?;

    let mut report = DeletionReport::new();

    // Delete from each hApp
    for happ in get_all_happs() {
        let result = happ.delete_user_data(agent).await;
        report.add_happ_result(happ.id(), result);
    }

    // Delete local data
    delete_local_storage(agent)?;
    report.add_local_deletion(true);

    // Request deletion from DHT (best effort)
    request_dht_deletion(agent).await;
    report.add_dht_request(true);

    Ok(report)
}

// Pattern: Deletion with grace period
pub struct DeletionRequest {
    pub agent: AgentPubKey,
    pub requested_at: Timestamp,
    pub execute_at: Timestamp,  // After grace period
    pub cancelled: bool,
}

impl DeletionRequest {
    pub fn new(agent: AgentPubKey) -> Self {
        DeletionRequest {
            agent,
            requested_at: Timestamp::now(),
            execute_at: Timestamp::now() + Duration::days(30),
            cancelled: false,
        }
    }

    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    pub fn should_execute(&self) -> bool {
        !self.cancelled && Timestamp::now() >= self.execute_at
    }
}

// Pattern: Cryptographic deletion
pub fn crypto_delete(
    entry: &EncryptedEntry,
    key: &EncryptionKey,
) {
    // Securely destroy the key
    secure_zero(key);

    // Entry still exists but is now undecryptable
    // Effectively deleted
}
```

### 7.3 Privacy Dashboard

```typescript
interface PrivacyDashboard {
  // What data exists
  dataInventory: {
    byHapp: Map<HappId, DataSummary>;
    byType: Map<DataType, DataSummary>;
    byVisibility: Map<Visibility, DataSummary>;
    totalSize: number;
  };

  // Who can see what
  accessOverview: {
    publicData: DataSummary;
    communityData: DataSummary;
    connectionData: DataSummary;
    privateData: DataSummary;
    activeCapabilities: Capability[];
  };

  // Consent status
  consents: {
    active: ConsentRecord[];
    withdrawn: ConsentRecord[];
    expired: ConsentRecord[];
  };

  // Activity
  recentActivity: {
    dataAccessed: AccessLog[];
    dataShared: ShareLog[];
    dataModified: ModificationLog[];
  };

  // Actions
  actions: {
    exportAllData: () => Promise<DataExport>;
    deleteAllData: () => Promise<DeletionRequest>;
    revokeCapability: (id: CapabilityId) => Promise<void>;
    withdrawConsent: (id: ConsentId) => Promise<void>;
    adjustVisibility: (entry: EntryId, visibility: Visibility) => Promise<void>;
  };
}
```

---

## Part 8: Privacy by Design Checklist

### For Every New Feature

```typescript
interface PrivacyDesignChecklist {
  // Data collection
  dataCollection: {
    "What data is collected?": string;
    "Is each piece necessary?": boolean;
    "What's the minimum needed?": string;
    "How long is it retained?": string;
    "How is it deleted?": string;
  };

  // Data storage
  dataStorage: {
    "Where is data stored?": "local" | "dht" | "both";
    "Is it encrypted at rest?": boolean;
    "Who has the keys?": string;
    "Can user export?": boolean;
    "Can user delete?": boolean;
  };

  // Data sharing
  dataSharing: {
    "Who can see this data?": Visibility;
    "Can user control sharing?": boolean;
    "Is consent required?": boolean;
    "Is sharing logged?": boolean;
  };

  // Access control
  accessControl: {
    "How is access controlled?": string;
    "Can access be revoked?": boolean;
    "Is access auditable?": boolean;
  };

  // Metadata
  metadata: {
    "What metadata is created?": string[];
    "Is metadata minimized?": boolean;
    "Is timing protected?": boolean;
  };

  // Threats
  threats: {
    "What could go wrong?": string[];
    "How are threats mitigated?": string[];
    "What's the worst case?": string;
  };
}
```

### Code Review Checklist

```rust
// Privacy code review checklist
pub fn privacy_review(code: &Code) -> PrivacyReview {
    PrivacyReview {
        // Data minimization
        collects_only_needed: check_minimal_collection(code),
        no_unnecessary_logging: check_no_excessive_logging(code),

        // Encryption
        sensitive_data_encrypted: check_sensitive_encrypted(code),
        keys_properly_managed: check_key_management(code),

        // Access control
        access_properly_checked: check_access_controls(code),
        capabilities_validated: check_capability_validation(code),

        // Consent
        consent_checked: check_consent_validation(code),
        consent_recorded: check_consent_logging(code),

        // Data lifecycle
        retention_enforced: check_retention_policies(code),
        deletion_supported: check_deletion_capabilities(code),

        // Metadata
        metadata_minimized: check_metadata_leakage(code),
        timing_protected: check_timing_attacks(code),
    }
}
```

---

## Appendix: Quick Reference

### Encryption Algorithms Used

| Purpose | Algorithm | Key Size |
|---------|-----------|----------|
| Symmetric encryption | XChaCha20-Poly1305 | 256-bit |
| Key derivation | Argon2id | N/A |
| Asymmetric encryption | X25519 | 256-bit |
| Signing | Ed25519 | 256-bit |
| Hashing | BLAKE3 | 256-bit |
| Secret sharing | Shamir's | Variable |

### Privacy Levels Quick Reference

| Level | Visibility | Encryption | Use Case |
|-------|------------|------------|----------|
| Public | Anyone | None | Published content |
| Community | Members | Community key | Internal discussions |
| Connections | Direct connections | Connection key | Profile details |
| Private | Self only | Personal key | Drafts, notes |
| Sensitive | Self + explicit share | Enhanced + access control | Health, financial |

### Data Lifecycle

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Collect │───►│  Store  │───►│  Use    │───►│ Archive │───►│ Delete  │
│ (min)   │    │ (encrypt)│   │(consent)│    │(if need)│    │(secure) │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼
  Purpose        Access         Audit         Retention      Crypto
  stated        controls       logging         policy       deletion
```

---

*"In a world of surveillance capitalism, privacy is resistance. In Mycelix, privacy is the foundation upon which genuine community can be built—not hiding, but the power to choose how we reveal ourselves to each other."*
