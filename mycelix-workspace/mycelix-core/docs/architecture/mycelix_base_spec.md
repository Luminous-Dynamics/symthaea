# MYCELIX BASE SPECIFICATION v1.0

**The Neutral Verification Kernel**

**Version**: 1.0.0  
**Status**: Draft for Community Review  
**License**: Apache 2.0 (or MIT, TBD by community)  
**Published**: January 2026  
**Compatibility**: Holochain RSM v0.2.x+, IPFS, Any DHT-compatible substrate

---

## Abstract

**Mycelix Base** is a governance-neutral, value-agnostic verification substrate for distributed systems. It provides:

1. **Decentralized Knowledge Graph (DKG)**: Immutable, append-only provenance for any claim
2. **Agent-Centric DHT**: Holochain-based distributed storage with cryptographic guarantees
3. **Proof Services**: Zero-knowledge proof generation and verification (zk-STARKs)
4. **Interoperability Standards**: Universal APIs for cross-system integration

**Mycelix Base does NOT include**:
- ❌ Economic models or token systems
- ❌ Governance frameworks or voting mechanisms
- ❌ Ethical constraints or value judgments
- ❌ Treasury management or oversight bodies

**What it DOES provide**:
- ✅ "Is this claim cryptographically valid?"
- ✅ "Can I verify the provenance of this data?"
- ✅ "Can I trust this agent's identity?"

**Use Mycelix Base when you need verifiable truth infrastructure without philosophical commitments.**

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Components](#2-core-components)
3. [Data Formats](#3-data-formats)
4. [API Specifications](#4-api-specifications)
5. [Cryptographic Primitives](#5-cryptographic-primitives)
6. [Interoperability Requirements](#6-interoperability-requirements)
7. [Implementation Guidelines](#7-implementation-guidelines)
8. [Security Considerations](#8-security-considerations)
9. [Versioning and Upgrades](#9-versioning-and-upgrades)
10. [Reference Implementations](#10-reference-implementations)

---

## 1. Architecture Overview

### 1.1 System Layers

```
┌─────────────────────────────────────────────┐
│  Application Layer (Your Code)              │
│  • Custom business logic                    │
│  • Domain-specific validation rules         │
└─────────────────────────────────────────────┘
                    ▲
                    │ Mycelix Base API
                    ▼
┌─────────────────────────────────────────────┐
│  Mycelix Base Kernel                        │
│  ┌─────────────────────────────────────┐   │
│  │ Decentralized Knowledge Graph (DKG) │   │
│  │ • Claim storage & retrieval         │   │
│  │ • Provenance tracking               │   │
│  │ • Temporal queries                  │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │ Agent-Centric DHT (Holochain)       │   │
│  │ • Source chains per agent           │   │
│  │ • Gossip protocol                   │   │
│  │ • Validation rules                  │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │ Cryptographic Proof Engine          │   │
│  │ • Signature verification            │   │
│  │ • ZK proof generation (optional)    │   │
│  │ • Hash chain validation             │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ▲
                    │ Network Protocol
                    ▼
┌─────────────────────────────────────────────┐
│  Transport Layer                            │
│  • WebRTC, QUIC, or custom P2P              │
└─────────────────────────────────────────────┘
```

### 1.2 Design Principles

**1. Neutrality**: No opinions on governance, economics, or ethics  
**2. Verifiability**: Every operation produces cryptographic proof  
**3. Modularity**: Components can be used independently  
**4. Interoperability**: Standard formats for data exchange  
**5. Minimalism**: Include only what's necessary for verification

---

## 2. Core Components

### 2.1 Decentralized Knowledge Graph (DKG)

**Purpose**: Store and query verifiable claims with provenance.

**Data Structure**:

```rust
pub struct Claim {
    pub claim_id: Uuid,               // Unique identifier
    pub claim_hash: Hash,             // SHA256 of content
    pub author_did: DID,              // W3C Decentralized Identifier
    pub timestamp: u64,               // Unix timestamp
    pub content: ClaimContent,        // The actual data
    pub signature: Signature,         // Ed25519 signature
    pub previous_claim: Option<Hash>, // For claim chains
}

pub struct ClaimContent {
    pub format: String,               // "json", "cbor", "protobuf"
    pub data: Vec<u8>,                // Serialized content
    pub schema_version: String,       // Semantic version
}
```

**Operations**:

```rust
// Create a claim
pub fn create_claim(
    did: &DID,
    content: ClaimContent,
    private_key: &PrivateKey,
) -> Result<Claim>;

// Retrieve a claim by ID
pub fn get_claim(claim_id: &Uuid) -> Result<Option<Claim>>;

// Verify claim signature
pub fn verify_claim(claim: &Claim, public_key: &PublicKey) -> Result<bool>;

// Query claims by author
pub fn query_by_author(author_did: &DID) -> Result<Vec<Claim>>;

// Query claims by time range
pub fn query_by_time(start: u64, end: u64) -> Result<Vec<Claim>>;
```

**Validation Rules** (enforced at DHT level):

1. Every claim must have valid signature
2. Timestamps must be monotonically increasing per author
3. Claim hash must match content
4. DIDs must be well-formed W3C DIDs

**Storage Backend**: Holochain DHT (default), IPFS (optional), Custom DHT (supported)

---

### 2.2 Agent-Centric DHT

**Purpose**: Distributed storage with agent sovereignty and cryptographic capability tokens.

**Based On**: Holochain RSM (Refactored State Model) v0.2.x+

**Key Concepts**:

```rust
pub struct Agent {
    pub did: DID,                     // Agent's identifier
    pub public_key: PublicKey,        // For signature verification
    pub source_chain: SourceChain,    // Agent's local immutable log
}

pub struct SourceChain {
    pub entries: Vec<Entry>,          // Ordered list of entries
    pub head: HeaderHash,             // Current chain head
}

pub struct Entry {
    pub entry_hash: EntryHash,
    pub entry_type: EntryType,
    pub author: DID,
    pub timestamp: u64,
    pub signature: Signature,
    pub previous_header: Option<HeaderHash>,
}
```

**DHT Operations**:

```rust
// Store entry (commits to agent's source chain + publishes to DHT)
pub fn commit_entry(entry: Entry) -> Result<EntryHash>;

// Retrieve entry from DHT
pub fn get_entry(hash: &EntryHash) -> Result<Option<Entry>>;

// Validate entry according to application rules
pub fn validate_entry(entry: &Entry) -> ValidationResult;

// Get entries by type
pub fn get_entries_by_type(entry_type: &EntryType) -> Result<Vec<Entry>>;
```

**Gossip Protocol**: Entries propagate via epidemic broadcast; redundancy factor configurable (default: 5 nodes per entry).

**Network Partition Resilience**: Agents continue operating locally; sync when network reunites.

---

### 2.3 Cryptographic Proof Engine

**Purpose**: Generate and verify cryptographic proofs for various claim types.

**Supported Proof Types**:

| Proof Type | Use Case | Library | Optional? |
|------------|----------|---------|-----------|
| **Ed25519 Signatures** | Identity, authorship | `ed25519-dalek` | ❌ Required |
| **SHA-256 Hashes** | Content integrity | `sha2` | ❌ Required |
| **Merkle Proofs** | Set membership | `rs-merkle` | ✅ Optional |
| **zk-STARKs** | Private computation verification | `risc0-zkvm` | ✅ Optional |
| **BLS Signatures** | Aggregatable signatures | `blst` | ✅ Optional |

**API**:

```rust
// Sign data with Ed25519
pub fn sign(data: &[u8], private_key: &PrivateKey) -> Signature;

// Verify Ed25519 signature
pub fn verify_signature(
    data: &[u8],
    signature: &Signature,
    public_key: &PublicKey,
) -> bool;

// Generate Merkle proof
pub fn generate_merkle_proof(
    tree: &MerkleTree,
    leaf_index: usize,
) -> MerkleProof;

// Verify Merkle proof
pub fn verify_merkle_proof(
    root: &Hash,
    proof: &MerkleProof,
    leaf: &Hash,
) -> bool;

// (Optional) Generate zk-STARK proof
#[cfg(feature = "zkp")]
pub fn generate_zk_proof(
    program: &Program,
    inputs: &[u8],
) -> Result<ZKProof>;

// (Optional) Verify zk-STARK proof
#[cfg(feature = "zkp")]
pub fn verify_zk_proof(
    proof: &ZKProof,
    public_inputs: &[u8],
) -> bool;
```

**Performance Requirements**:
- Ed25519 signature generation: <1ms
- Ed25519 verification: <100μs
- Merkle proof generation: <10ms for 1M leaves
- Merkle proof verification: <1ms

**ZK Proof Performance** (if enabled):
- Proof generation: <10s for typical computations
- Proof verification: <1s
- Proof size: <500KB

---

## 3. Data Formats

### 3.1 Decentralized Identifier (DID)

**Format**: W3C DID 1.0 Specification

```
did:mycelix:<identifier>

Examples:
did:mycelix:abc123def456
did:mycelix:member_alice
```

**DID Document** (stored in DKG):

```json
{
  "@context": "https://www.w3.org/ns/did/v1",
  "id": "did:mycelix:abc123def456",
  "verificationMethod": [
    {
      "id": "did:mycelix:abc123def456#keys-1",
      "type": "Ed25519VerificationKey2020",
      "controller": "did:mycelix:abc123def456",
      "publicKeyMultibase": "z6MkpTHR8VNsBxYAAWHut2Geadd9jSwuBV8xRoAnwWsdvktH"
    }
  ],
  "authentication": ["did:mycelix:abc123def456#keys-1"],
  "service": [
    {
      "id": "did:mycelix:abc123def456#endpoint",
      "type": "MessagingService",
      "serviceEndpoint": "https://node.example.com/alice"
    }
  ]
}
```

### 3.2 Claim Format

**JSON Schema**:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["claim_id", "claim_hash", "author_did", "timestamp", "content", "signature"],
  "properties": {
    "claim_id": {
      "type": "string",
      "format": "uuid"
    },
    "claim_hash": {
      "type": "string",
      "pattern": "^[0-9a-f]{64}$"
    },
    "author_did": {
      "type": "string",
      "pattern": "^did:mycelix:[a-zA-Z0-9_-]+$"
    },
    "timestamp": {
      "type": "integer",
      "minimum": 0
    },
    "content": {
      "type": "object",
      "properties": {
        "format": {
          "type": "string",
          "enum": ["json", "cbor", "protobuf", "plaintext"]
        },
        "data": {
          "type": "string",
          "contentEncoding": "base64"
        },
        "schema_version": {
          "type": "string",
          "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$"
        }
      }
    },
    "signature": {
      "type": "string",
      "contentEncoding": "base64"
    },
    "previous_claim": {
      "type": ["string", "null"],
      "pattern": "^[0-9a-f]{64}$"
    }
  }
}
```

**Example Claim**:

```json
{
  "claim_id": "550e8400-e29b-41d4-a716-446655440000",
  "claim_hash": "a3f5b8c2d1e4f7a9b0c3d6e8f1a4b7c0d3e6f9a2b5c8d1e4f7a0b3c6d9e2f5a8",
  "author_did": "did:mycelix:alice",
  "timestamp": 1704067200,
  "content": {
    "format": "json",
    "data": "eyAibWVzc2FnZSI6ICJIZWxsbyB3b3JsZCIgfQ==",
    "schema_version": "1.0.0"
  },
  "signature": "bXlzaWduYXR1cmVkYXRhZm9yZXhhbXBsZQ==",
  "previous_claim": null
}
```

### 3.3 Proof Format

**Merkle Proof**:

```json
{
  "type": "merkle_proof",
  "root": "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
  "leaf": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
  "proof": [
    "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
    "3e23e8160039594a33894f6564e1b1348bbd7a0088d42c4acb73eeaed59c009d"
  ],
  "path": [false, true]
}
```

**zk-STARK Proof** (optional):

```json
{
  "type": "zk_stark_proof",
  "proof": "<base64-encoded proof blob>",
  "public_inputs": "<base64-encoded public inputs>",
  "verification_key": "<base64-encoded verification key>",
  "proof_size_bytes": 487423
}
```

---

## 4. API Specifications

### 4.1 HTTP REST API

**Base URL**: `https://<node-address>/api/v1`

#### **Claims Endpoints**

**POST /claims**  
Create a new claim.

```http
POST /api/v1/claims
Content-Type: application/json

{
  "author_did": "did:mycelix:alice",
  "content": {
    "format": "json",
    "data": "eyAibWVzc2FnZSI6ICJIZWxsbyB3b3JsZCIgfQ=="
  },
  "signature": "bXlzaWduYXR1cmVkYXRhZm9yZXhhbXBsZQ=="
}

Response: 201 Created
{
  "claim_id": "550e8400-e29b-41d4-a716-446655440000",
  "claim_hash": "a3f5b8c2d1e4f7a9b0c3d6e8f1a4b7c0d3e6f9a2b5c8d1e4f7a0b3c6d9e2f5a8",
  "timestamp": 1704067200
}
```

**GET /claims/{claim_id}**  
Retrieve a claim by ID.

```http
GET /api/v1/claims/550e8400-e29b-41d4-a716-446655440000

Response: 200 OK
{
  "claim_id": "550e8400-e29b-41d4-a716-446655440000",
  "claim_hash": "...",
  "author_did": "did:mycelix:alice",
  "timestamp": 1704067200,
  "content": {...},
  "signature": "..."
}
```

**GET /claims?author={did}&start={timestamp}&end={timestamp}**  
Query claims by author and time range.

```http
GET /api/v1/claims?author=did:mycelix:alice&start=1704000000&end=1704100000

Response: 200 OK
{
  "claims": [...],
  "total": 42,
  "page": 1,
  "per_page": 20
}
```

#### **Verification Endpoints**

**POST /verify/signature**  
Verify a signature.

```http
POST /api/v1/verify/signature
Content-Type: application/json

{
  "data": "SGVsbG8gd29ybGQ=",
  "signature": "bXlzaWduYXR1cmU=",
  "public_key": "bXlwdWJsaWNrZXk="
}

Response: 200 OK
{
  "valid": true
}
```

**POST /verify/merkle**  
Verify a Merkle proof.

```http
POST /api/v1/verify/merkle
Content-Type: application/json

{
  "root": "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
  "leaf": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
  "proof": [...],
  "path": [false, true]
}

Response: 200 OK
{
  "valid": true
}
```

### 4.2 GraphQL API

**Endpoint**: `https://<node-address>/graphql`

**Schema**:

```graphql
type Query {
  claim(id: ID!): Claim
  claims(
    author: String
    startTime: Int
    endTime: Int
    limit: Int
    offset: Int
  ): [Claim!]!
  verifyClaim(claimId: ID!): Boolean!
}

type Mutation {
  createClaim(input: CreateClaimInput!): Claim!
}

type Claim {
  claimId: ID!
  claimHash: String!
  authorDid: String!
  timestamp: Int!
  content: ClaimContent!
  signature: String!
  previousClaim: String
}

type ClaimContent {
  format: String!
  data: String!
  schemaVersion: String!
}

input CreateClaimInput {
  authorDid: String!
  content: ClaimContentInput!
  signature: String!
}

input ClaimContentInput {
  format: String!
  data: String!
}
```

**Example Query**:

```graphql
query GetClaims {
  claims(author: "did:mycelix:alice", startTime: 1704000000, limit: 10) {
    claimId
    claimHash
    timestamp
    content {
      format
      data
    }
  }
}
```

### 4.3 WebSocket API (Real-Time Updates)

**Endpoint**: `wss://<node-address>/ws`

**Subscribe to new claims**:

```json
{
  "type": "subscribe",
  "channel": "claims",
  "filter": {
    "author": "did:mycelix:alice"
  }
}
```

**Receive updates**:

```json
{
  "type": "claim_created",
  "data": {
    "claim_id": "...",
    "author_did": "did:mycelix:alice",
    "timestamp": 1704067200
  }
}
```

---

## 5. Cryptographic Primitives

### 5.1 Required Primitives

| Primitive | Algorithm | Library | Purpose |
|-----------|-----------|---------|---------|
| **Signing** | Ed25519 | `ed25519-dalek` | Claim authorship |
| **Hashing** | SHA-256 | `sha2` | Content integrity |
| **Key Derivation** | HKDF-SHA256 | `hkdf` | Derive keys from seeds |

### 5.2 Optional Primitives

| Primitive | Algorithm | Library | Purpose |
|-----------|-----------|---------|---------|
| **ZK Proofs** | zk-STARK | `risc0-zkvm` | Private computation |
| **Aggregated Sigs** | BLS12-381 | `blst` | Batch verification |
| **Encrypted Storage** | ChaCha20-Poly1305 | `chacha20poly1305` | Private data |

### 5.3 Key Management

**Keys are managed by agents**, not by Mycelix Base. Implementations must:

1. Generate Ed25519 keypairs securely
2. Store private keys encrypted at rest
3. Never transmit private keys over the network
4. Support key rotation (via DID document updates)

**Reference Implementation**: Use platform-specific secure enclaves (iOS Keychain, Android Keystore, TPM on desktop).

---

## 6. Interoperability Requirements

### 6.1 Data Export

All Mycelix Base implementations MUST support exporting data in the following formats:

**JSON-LD Export**:

```json
{
  "@context": "https://mycelix.org/context/v1",
  "@type": "ClaimGraph",
  "claims": [
    {
      "@id": "urn:uuid:550e8400-e29b-41d4-a716-446655440000",
      "author": "did:mycelix:alice",
      "timestamp": "2024-01-01T00:00:00Z",
      "content": {...}
    }
  ]
}
```

**CSV Export** (for analytics):

```csv
claim_id,author_did,timestamp,content_format,content_size_bytes
550e8400-...,did:mycelix:alice,1704067200,json,247
```

### 6.2 Cross-Implementation Compatibility

**Implementations must be compatible at the wire protocol level.**

**Test Suite**: A canonical test suite (`mycelix-base-tests`) ensures compatibility:

```bash
$ cargo test --package mycelix-base-tests

Running compatibility tests...
✅ Claim creation/retrieval (Rust impl ↔ Go impl)
✅ Signature verification (Rust impl ↔ JavaScript impl)
✅ DHT gossip (Holochain impl ↔ Custom DHT impl)
✅ API compatibility (REST, GraphQL, WebSocket)
```

### 6.3 Bridge to External Systems

**Implementations SHOULD support bridging to**:

- IPFS (for content-addressed storage)
- Ethereum/EVM chains (for anchoring proofs)
- Traditional databases (PostgreSQL, MongoDB)

**Example Bridge**:

```rust
// Export claims to IPFS
pub fn export_to_ipfs(claims: Vec<Claim>) -> Result<IpfsHash>;

// Anchor claim hash to Ethereum
pub fn anchor_to_ethereum(
    claim_hash: Hash,
    contract_address: Address,
) -> Result<TxHash>;
```

---

## 7. Implementation Guidelines

### 7.1 Minimum Viable Implementation

**Core Requirements**:

1. ✅ DID generation and resolution
2. ✅ Claim creation with Ed25519 signatures
3. ✅ Claim storage in DHT (Holochain or equivalent)
4. ✅ Claim retrieval by ID
5. ✅ Signature verification
6. ✅ REST API (at minimum: POST /claims, GET /claims/{id})

**Optional Components** (can be added later):

- GraphQL API
- WebSocket subscriptions
- zk-STARK proofs
- Merkle tree support
- Cross-chain bridges

### 7.2 Reference Implementation (Rust + Holochain)

**Repository**: `github.com/mycelix/mycelix-base-rs`

**Dependencies**:

```toml
[dependencies]
holochain = "0.2.x"
ed25519-dalek = "2.0"
sha2 = "0.10"
uuid = { version = "1.0", features = ["v4"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

**Key Modules**:

```
mycelix-base/
├── src/
│   ├── lib.rs              # Public API
│   ├── dkg/
│   │   ├── mod.rs          # DKG operations
│   │   ├── claim.rs        # Claim struct & methods
│   │   └── query.rs        # Query operations
│   ├── crypto/
│   │   ├── mod.rs          # Crypto operations
│   │   ├── signing.rs      # Ed25519 signing/verification
│   │   └── hashing.rs      # SHA-256 hashing
│   ├── dht/
│   │   ├── mod.rs          # DHT interface
│   │   └── holochain.rs    # Holochain implementation
│   └── api/
│       ├── rest.rs         # REST API handlers
│       └── graphql.rs      # GraphQL schema
├── tests/
│   ├── integration/        # Integration tests
│   └── compatibility/      # Cross-impl tests
└── examples/
    ├── create_claim.rs
    ├── verify_signature.rs
    └── export_data.rs
```

### 7.3 Alternative Implementations

**Go Implementation**: `github.com/mycelix/mycelix-base-go`  
**JavaScript/TypeScript**: `github.com/mycelix/mycelix-base-js`  
**Python**: `github.com/mycelix/mycelix-base-py`

---

## 8. Security Considerations

### 8.1 Threat Model

**In Scope**:
- ✅ Signature forgery
- ✅ Content tampering
- ✅ Replay attacks
- ✅ Sybil attacks (at identity level)
- ✅ DHT pollution (malformed entries)

**Out of Scope** (handled by higher layers):
- ❌ Economic attacks (no tokens in Base)
- ❌ Governance capture (no governance in Base)
- ❌ Social engineering (application-specific)

### 8.2 Attack Mitigations

**Signature Forgery**:
- All claims require valid Ed25519 signatures
- Public keys verified against DID documents
- Mitigation: Cryptographic guarantee

**Content Tampering**:
- Claim hashes computed over canonical content representation
- Any modification invalidates hash
- Mitigation: Immutable append-only log

**Replay Attacks**:
- Timestamps must be monotonically increasing per author
- Duplicate claim IDs rejected by DHT
- Mitigation: Sequence number enforcement

**Sybil Attacks**:
- DIDs are cheap to create (not prevented at Base layer)
- Higher layers (Proofs, Ethical) add Sybil resistance
- Mitigation: None at Base layer (intentional)

**DHT Pollution**:
- Validation rules enforce well-formedness
- Invalid entries not propagated
- Mitigation: Entry validation at DHT level

### 8.3 Cryptographic Agility

**Mycelix Base supports algorithm upgrades via versioning:**

```rust
pub enum SignatureAlgorithm {
    Ed25519,        // Current default
    Ed448,          // Post-quantum candidate
    Dilithium3,     // NIST PQC standard
}

pub struct Claim {
    pub signature_algorithm: SignatureAlgorithm,
    pub signature: Vec<u8>,
    // ...
}
```

**Migration Plan**:
1. New algorithm added as optional
2. Implementations support both old and new
3. Grace period (12 months)
4. Old algorithm deprecated
5. New algorithm becomes mandatory

---

## 9. Versioning and Upgrades

### 9.1 Semantic Versioning

**Format**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes to wire protocol or API
- **MINOR**: Backward-compatible feature additions
- **PATCH**: Backward-compatible bug fixes

**Current Version**: 1.0.0

### 9.2 Deprecation Policy

**Features are deprecated via the following process**:

1. **Announce**: Publish deprecation notice 6 months in advance
2. **Warn**: Implementations log warnings when deprecated features used
3. **Remove**: Feature removed in next MAJOR version

**Example**:

```
Version 1.5.0: Announce deprecation of REST API v1
Version 1.6.0-2.0.0: Warning logs emitted
Version 2.0.0: REST API v1 removed, v2 required
```

### 9.3 Upgrade Paths

**Implementations must support**:

- Reading data written by previous MINOR versions
- Migrating data to new formats
- Running old and new versions concurrently (during transition)

**Example Migration**:

```bash
# Upgrade from 1.x to 2.x
$ mycelix-base migrate --from 1.5.0 --to 2.0.0

Analyzing data store...
Found 1,247,839 claims to migrate
Estimated time: 45 minutes

Proceed? [y/N] y

Migrating claims... [████████████████████] 100%
✅ Migration complete
```

---

## 10. Reference Implementations

### 10.1 Official Implementations

| Language | Repository | Status | Maintainer |
|----------|-----------|--------|------------|
| **Rust** | `mycelix/mycelix-base-rs` | ✅ Stable | Mycelix Foundation |
| **Go** | `mycelix/mycelix-base-go` | ⚠️ Beta | Community |
| **TypeScript** | `mycelix/mycelix-base-js` | ⚠️ Beta | Community |
| **Python** | `mycelix/mycelix-base-py` | 🔄 Alpha | Community |

### 10.2 Quick Start (Rust)

```bash
# Add to Cargo.toml
[dependencies]
mycelix-base = "1.0"

# Example usage
use mycelix_base::{Claim, DID, create_claim, verify_claim};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate DID and keypair
    let (did, keypair) = DID::generate()?;
    
    // Create a claim
    let content = ClaimContent {
        format: "json".to_string(),
        data: r#"{"message": "Hello, Mycelix!"}"#.as_bytes().to_vec(),
        schema_version: "1.0.0".to_string(),
    };
    
    let claim = create_claim(&did, content, &keypair.private)?;
    
    // Verify the claim
    let is_valid = verify_claim(&claim, &keypair.public)?;
    println!("Claim valid: {}", is_valid);
    
    Ok(())
}
```

### 10.3 Quick Start (JavaScript/TypeScript)

```bash
npm install @mycelix/base
```

```typescript
import { DID, createClaim, verifyClaim } from '@mycelix/base';

async function example() {
  // Generate DID and keypair
  const { did, keypair } = await DID.generate();
  
  // Create a claim
  const claim = await createClaim(
    did,
    {
      format: 'json',
      data: Buffer.from(JSON.stringify({ message: 'Hello, Mycelix!' })),
      schemaVersion: '1.0.0'
    },
    keypair.privateKey
  );
  
  // Verify the claim
  const isValid = await verifyClaim(claim, keypair.publicKey);
  console.log('Claim valid:', isValid);
}
```

### 10.4 Docker Deployment

```bash
# Pull official image
docker pull mycelix/base:1.0

# Run a node
docker run -p 8080:8080 -v /data:/mycelix/data mycelix/base:1.0

# Node is now available at http://localhost:8080
```

**Docker Compose Example**:

```yaml
version: '3.8'
services:
  mycelix-node:
    image: mycelix/base:1.0
    ports:
      - "8080:8080"
    volumes:
      - mycelix-data:/mycelix/data
    environment:
      - MYCELIX_NETWORK=mainnet
      - MYCELIX_LOG_LEVEL=info
    restart: unless-stopped

volumes:
  mycelix-data:
```

---

## 11. Performance Benchmarks

### 11.1 Expected Performance

**Hardware**: AWS c5.xlarge (4 vCPU, 8GB RAM)

| Operation | Throughput | Latency (p50) | Latency (p99) |
|-----------|------------|---------------|---------------|
| **Claim Creation** | 1,000/sec | 5ms | 15ms |
| **Claim Retrieval** | 10,000/sec | 2ms | 8ms |
| **Signature Verification** | 50,000/sec | 0.1ms | 0.5ms |
| **Query by Author (1000 claims)** | 500/sec | 20ms | 50ms |
| **DHT Gossip Propagation** | 90% network in 5s | 3s | 10s |

### 11.2 Scalability Limits

**Per Node**:
- Maximum claims/second: 1,000 (write-bound by DHT replication)
- Maximum storage: Limited by disk (recommend SSD for <1s retrieval)
- Maximum DHT connections: 500 peers

**Network-Wide**:
- No theoretical upper limit (horizontally scalable)
- Bottleneck is DHT redundancy (default 5x replication)

### 11.3 Optimization Techniques

**For High-Throughput Scenarios**:

1. **Batch Writes**: Combine multiple claims into a single DHT transaction
2. **Read Replicas**: Deploy read-only nodes for query-heavy workloads
3. **Sharding**: Partition DKG by author DID prefix
4. **Caching**: Use Redis/Memcached for frequently-accessed claims
5. **Indexing**: Build secondary indexes for complex queries

---

## 12. Testing and Validation

### 12.1 Test Coverage Requirements

**Implementations must maintain**:

- ✅ **Unit Tests**: >90% line coverage
- ✅ **Integration Tests**: All API endpoints
- ✅ **Compatibility Tests**: Cross-implementation (via `mycelix-base-tests`)
- ✅ **Performance Tests**: Meet benchmarks in Section 11.1
- ✅ **Security Tests**: Signature forgery, replay attacks, etc.

### 12.2 Continuous Integration

**CI Pipeline** (GitHub Actions example):

```yaml
name: Mycelix Base CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run tests
        run: cargo test --all-features
      - name: Run compatibility tests
        run: |
          git clone https://github.com/mycelix/mycelix-base-tests
          cd mycelix-base-tests
          cargo test --features rust-impl
      - name: Check coverage
        run: cargo tarpaulin --out Xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 12.3 Conformance Testing

**Implementations can be certified via**:

```bash
# Run official conformance suite
$ mycelix-conformance-test --impl ./my-implementation

Testing implementation: my-implementation v1.0.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ DID generation
✅ Claim creation
✅ Signature verification
✅ DHT operations
✅ REST API compliance
✅ Data export formats
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Result: ✅ CONFORMANT (100% tests passed)

Certificate ID: cert_20260115_abc123
Valid until: 2027-01-15
```

---

## 13. Governance of This Specification

### 13.1 Specification Updates

**This specification is governed by**:

- **Maintainer**: Mycelix Foundation (initial custodian)
- **Process**: Community RFC process (see below)
- **Versioning**: Semantic versioning (Section 9.1)

### 13.2 RFC Process

**To propose changes**:

1. **Draft RFC**: Write proposal in `mycelix/rfcs` repository
2. **Community Review**: 30-day minimum comment period
3. **Implementation**: Prototype in at least one reference implementation
4. **Vote**: Mycelix Foundation + 3 community maintainers approve
5. **Merge**: RFC accepted, specification updated

**RFC Template**:

```markdown
# RFC-XXXX: [Title]

**Author**: [Your Name]
**Status**: Draft
**Created**: YYYY-MM-DD

## Summary
One-paragraph explanation of the change.

## Motivation
Why is this change necessary?

## Detailed Design
Technical specification of the change.

## Backward Compatibility
How does this affect existing implementations?

## Alternatives Considered
What other approaches were evaluated?
```

### 13.3 Community Governance

**This specification remains neutral**. The Mycelix Foundation acts as **technical custodian only**, not philosophical authority.

**Anyone may**:
- Implement this specification
- Fork this specification
- Propose changes via RFC
- Certify implementations

**The Foundation may not**:
- Change the specification without community RFC process
- Require licensing fees for implementation
- Discriminate against implementations

---

## 14. Relationship to Higher Layers

**Mycelix Base is the foundation. Higher layers add value**:

```
┌────────────────────────────────────────────┐
│  Mycelix Ethical (Optional)                │
│  • Eight Harmonies                         │
│  • Constitutional governance               │
│  • Treasury management                     │
└────────────────────────────────────────────┘
                  ▲
                  │ Uses
                  ▼
┌────────────────────────────────────────────┐
│  Mycelix Proofs (Optional)                 │
│  • MATL (reputation scoring)               │
│  • LEM (epistemic tiers)                   │
│  • Audit trails                            │
└────────────────────────────────────────────┘
                  ▲
                  │ Uses
                  ▼
┌────────────────────────────────────────────┐
│  Mycelix Base (This Specification)         │
│  • DKG (verifiable claims)                 │
│  • DHT (distributed storage)               │
│  • Crypto (signatures, proofs)             │
└────────────────────────────────────────────┘
```

**Implementations may**:
- Use Base alone (verification infrastructure)
- Add Proofs layer (trust-as-a-service)
- Adopt Ethical layer (full governance)

**All layers interoperate at the Base protocol level.**

---

## 15. Frequently Asked Questions

### Q: Is Mycelix Base a blockchain?

**A**: No. It's a distributed hash table (DHT) with cryptographic proofs. Unlike blockchains:
- No global consensus (only local validation)
- No mining or proof-of-work
- No total ordering of transactions
- Scales horizontally without bottlenecks

### Q: Can I use Mycelix Base without tokens?

**A**: Yes. Mycelix Base has no native token. Economic models are added in higher layers (optional).

### Q: Is it compatible with Ethereum/Cosmos/Polkadot?

**A**: Mycelix Base can bridge to any blockchain via standard mechanisms (Merkle proofs, IBC, XCMP). Implementations should provide bridge adapters.

### Q: What about privacy?

**A**: Mycelix Base stores hashes and signatures publicly (for verification). Private data can be:
- Encrypted before storing
- Stored off-chain with only commitments on-chain
- Protected via zk-STARKs (optional feature)

### Q: How does this compare to IPFS?

**A**: IPFS provides content-addressed storage. Mycelix Base adds:
- Cryptographic authorship (DIDs + signatures)
- Structured queries (not just content hashes)
- Provenance tracking (claim chains)
- Integration with higher-layer governance (optional)

### Q: Can I monetize my implementation?

**A**: Yes. The specification is Apache 2.0 licensed. You may:
- Offer commercial support
- Provide managed hosting
- Build proprietary extensions
- Charge for value-added services

**You may not**:
- Patent the core specification
- Restrict others from implementing
- Require licensing fees for conformance

---

## 16. Roadmap

### Version 1.0 (Current)

✅ Core DKG with claims and DIDs  
✅ Ed25519 signatures  
✅ Holochain DHT backend  
✅ REST API  
✅ JSON-LD export

### Version 1.1 (Q2 2026)

🔄 GraphQL API  
🔄 WebSocket subscriptions  
🔄 Merkle proof support  
🔄 Performance optimizations (batch writes)

### Version 1.2 (Q3 2026)

🔄 zk-STARK integration (optional)  
🔄 BLS aggregated signatures  
🔄 IPFS bridge  
🔄 Ethereum anchor support

### Version 2.0 (2027+)

📋 Post-quantum cryptography (Dilithium)  
📋 Multi-party computation (MPC)  
📋 Advanced query language  
📋 Streaming data support

---

## 17. Contributing

**We welcome contributions\!**

**Ways to contribute**:
- 🐛 **Report bugs**: GitHub Issues
- 💡 **Propose features**: RFC process
- 🔧 **Submit code**: Pull requests
- 📖 **Improve docs**: Edit this specification
- 🧪 **Write tests**: Expand test suite

**Repository**: `https://github.com/mycelix/mycelix-base`

**Community**:
- Discord: `discord.gg/mycelix-base`
- Forum: `forum.mycelix.org`
- Twitter: `@mycelix_base`

---

## 18. License and Legal

**Specification License**: Creative Commons CC0 1.0 (Public Domain)

**Reference Implementation License**: Apache 2.0

**Patent Grant**: The Mycelix Foundation and contributors grant a worldwide, royalty-free, non-exclusive license to implement this specification. This specification is intended to be freely implementable by anyone without licensing restrictions.

**Trademark**: "Mycelix" is a trademark of the Mycelix Foundation. Use of the trademark requires compliance with this specification (via conformance testing).

---

## 19. Acknowledgments

**This specification builds upon**:
- Holochain RSM (distributed computing framework)
- W3C DID Specification (decentralized identifiers)
- IPFS (content-addressed storage)
- zk-STARK research (Risc0, StarkWare)

**Special thanks to**:
- The Holochain community for agent-centric computing
- The W3C DID Working Group for identity standards
- The zkML research community for verifiable computation
- All early adopters and contributors

---

## 20. Conclusion

**Mycelix Base provides the minimal, neutral verification substrate for distributed systems.**

**Use it when you need**:
- Verifiable claims with cryptographic provenance
- Distributed storage without central authority
- Interoperable identity (W3C DIDs)
- Foundation for trust layers (reputation, governance)

**It does not impose**:
- Economic models
- Governance structures
- Ethical frameworks

**Those are added in higher layers, by choice.**

**Start building**: `cargo install mycelix-base`

---

**END OF SPECIFICATION**

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Canonical URL**: `https://spec.mycelix.org/base/v1`  
**Contact**: `spec@mycelix.org`