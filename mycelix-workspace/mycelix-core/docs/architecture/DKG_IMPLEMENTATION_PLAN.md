# 🍄 DKG (Decentralized Knowledge Graph) Implementation Plan

**Version**: 1.0.0
**Date**: November 11, 2025
**Status**: Active Development
**Architecture**: Hybrid (Holochain + Optional PostgreSQL Accelerator)

---

## 🎯 Executive Summary

**The Challenge**: Build Layer 2 (DKG) to enable global searchability while maintaining decentralization principles.

**The Solution**: Hybrid 3-layer architecture:
1. **Layer 1 (DHT)** - Holochain source of truth (mandatory, peer-to-peer)
2. **Layer 2a (DKG Index Zome)** - Distributed search using Holochain links (mandatory)
3. **Layer 2b (PostgreSQL Accelerator)** - Optional server for complex queries (rebuilds from DHT)

**Why Hybrid?**
- ✅ Maintains pure P2P option (Layer 1 + 2a only)
- ✅ Enables fast search for those who want it (Layer 2b)
- ✅ PostgreSQL is **verifiable** (rebuilds from signed DHT data)
- ✅ Progressive enhancement (not a compromise)

---

## 🏗️ Architecture

### **Full System Diagram**

```
┌──────────────────────────────────────────────────────────────────┐
│  User Applications (Marketplace, Governance, etc.)               │
│  - Browse listings                                               │
│  - Search claims                                                 │
│  - Query reputation                                              │
└─────────────────┬────────────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ↓                   ↓
┌──────────────────┐  ┌─────────────────────────────────────────┐
│  Query Option 1  │  │  Query Option 2 (Power User)            │
│  Holochain Only  │  │  PostgreSQL Accelerator                 │
│  (Pure P2P)      │  │  (Faster, but requires trust in indexer)│
└────────┬─────────┘  └──────────┬──────────────────────────────┘
         │                       │
         ↓                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2a: DKG Index Zome (Holochain Native)                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Link-Based Indexing                                       │  │
│  │ - Token → Listing (for search)                            │  │
│  │ - Category → Listing (for filtering)                      │  │
│  │ - Price Range → Listing (for sorting)                     │  │
│  │ - Trust Score → Agent (for reputation)                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│  Performance: O(log n) lookups, scales horizontally            │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Gossip Protocol
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: DHT (Source of Truth)                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Epistemic Claims (v2.0 Schema)                            │  │
│  │ - claim_id: UUID                                          │  │
│  │ - epistemic_tier_e: E0-E4 (Empirical verifiability)      │  │
│  │ - epistemic_tier_n: N0-N3 (Normative authority)          │  │
│  │ - epistemic_tier_m: M0-M3 (Materiality/State mgmt)       │  │
│  │ - submitted_by_did: Agent's DID                           │  │
│  │ - content: { ... claim-specific data ... }                │  │
│  │ - signature: Ed25519 signature                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│  Validation: Agent-centric, consensus via DHT neighborhoods    │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Optional: Sync to Accelerator
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2b: PostgreSQL Accelerator (Optional)                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ RDF Triple Store (Apache Jena / Blazegraph)              │  │
│  │ - Subject-Predicate-Object triples                        │  │
│  │ - SPARQL query endpoint                                   │  │
│  │ - Full-text search (Elasticsearch-like)                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ PostgreSQL Schema                                         │  │
│  │ CREATE TABLE epistemic_claims (                           │  │
│  │   claim_id UUID PRIMARY KEY,                              │  │
│  │   epistemic_tier_e VARCHAR(2),  -- E0-E4                  │  │
│  │   epistemic_tier_n VARCHAR(2),  -- N0-N3                  │  │
│  │   epistemic_tier_m VARCHAR(2),  -- M0-M3                  │  │
│  │   submitted_by_did TEXT,                                  │  │
│  │   content JSONB,                                          │  │
│  │   claim_hash CHAR(64),                                    │  │
│  │   signature TEXT,                                         │  │
│  │   timestamp TIMESTAMPTZ,                                  │  │
│  │   -- Search indexes                                       │  │
│  │   content_tsv TSVECTOR                                    │  │
│  │ );                                                        │  │
│  │ CREATE INDEX ON epistemic_claims USING GIN(content_tsv); │  │
│  └───────────────────────────────────────────────────────────┘  │
│  Sync: Polls DHT every 30s, validates signatures, rebuilds     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Implementation Roadmap

### **Phase 1: Core Holochain DKG (Weeks 1-3)**

**Goal**: Pure P2P search working (no servers required)

#### Week 1: Epistemic Claim Entry Type
**File**: `mycelix-marketplace/zomes/dkg_index_integrity/src/lib.rs`

```rust
use hdk::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EpistemicClaim {
    pub claim_id: String,  // UUIDv4
    pub claim_hash: String,  // SHA-256 of content
    pub submitted_by_did: String,  // "did:mycelix:..."
    pub submitter_type: AgentType,  // HumanMember, InstrumentalActor, DAOCollective
    pub timestamp: Timestamp,

    // Epistemic Charter v2.0 - 3D Cube
    pub epistemic_tier_e: EpistemicTierE,  // E0-E4
    pub epistemic_tier_n: EpistemicTierN,  // N0-N3
    pub epistemic_tier_m: EpistemicTierM,  // M0-M3

    // Content (claim-specific)
    pub content: String,  // Serialized JSON (flexible schema)

    // Relationships
    pub related_claims: Vec<RelatedClaim>,  // SUPPORTS, REFUTES, SUPERCEDES

    // Trust tags (validation metadata)
    pub trust_tags: Vec<TrustTag>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub enum EpistemicTierE {
    E0,  // Null (unverifiable belief)
    E1,  // Testimonial (personal attestation)
    E2,  // Privately Verifiable (audit guild)
    E3,  // Cryptographically Proven (ZKP)
    E4,  // Publicly Reproducible (open data/code)
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub enum EpistemicTierN {
    N0,  // Personal (self only)
    N1,  // Communal (local DAO)
    N2,  // Network (global consensus)
    N3,  // Axiomatic (constitutional/mathematical)
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub enum EpistemicTierM {
    M0,  // Ephemeral (discard immediately)
    M1,  // Temporal (prune after state change)
    M2,  // Persistent (archive after time)
    M3,  // Foundational (preserve forever)
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct RelatedClaim {
    pub claim_id: String,
    pub relationship: ClaimRelationship,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub enum ClaimRelationship {
    SUPPORTS,
    REFUTES,
    SUPERCEDES,
    DUPLICATES,
}

// Validation rules
#[hdk_extern]
pub fn validate_create_entry_epistemic_claim(
    _action: EntryCreationAction,
    epistemic_claim: EpistemicClaim,
) -> ExternResult<ValidateCallbackResult> {
    // 1. Validate UUIDv4 format
    if !is_valid_uuid(&epistemic_claim.claim_id) {
        return Ok(ValidateCallbackResult::Invalid("Invalid UUID".into()));
    }

    // 2. Validate DID format
    if !epistemic_claim.submitted_by_did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid("Invalid DID".into()));
    }

    // 3. Validate claim_hash matches content
    let computed_hash = sha256(&epistemic_claim.content);
    if computed_hash != epistemic_claim.claim_hash {
        return Ok(ValidateCallbackResult::Invalid("Hash mismatch".into()));
    }

    // 4. Validate M3 (Foundational) claims require higher authority
    if epistemic_claim.epistemic_tier_m == EpistemicTierM::M3 {
        // M3 claims must be N2 (Network) or N3 (Axiomatic)
        match epistemic_claim.epistemic_tier_n {
            EpistemicTierN::N2 | EpistemicTierN::N3 => {},
            _ => return Ok(ValidateCallbackResult::Invalid(
                "M3 claims require N2/N3 authority".into()
            )),
        }
    }

    Ok(ValidateCallbackResult::Valid)
}
```

#### Week 2: Link-Based Indexing
**File**: `mycelix-marketplace/zomes/dkg_index/src/lib.rs`

```rust
use hdk::prelude::*;

#[hdk_extern]
pub fn create_claim(claim: EpistemicClaim) -> ExternResult<ActionHash> {
    // 1. Create the claim entry
    let claim_hash = create_entry(EntryTypes::EpistemicClaim(claim.clone()))?;

    // 2. Create searchable indexes (links)
    index_claim_for_search(&claim, &claim_hash)?;

    Ok(claim_hash)
}

fn index_claim_for_search(claim: &EpistemicClaim, claim_hash: &ActionHash) -> ExternResult<()> {
    // Parse content as JSON (marketplace-specific)
    let content: serde_json::Value = serde_json::from_str(&claim.content)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    // Index by title tokens (if it's a Listing)
    if let Some(title) = content.get("title").and_then(|t| t.as_str()) {
        let tokens = tokenize(title);
        for token in tokens {
            create_link(
                Path::from(format!("token.{}", token)).hash()?,
                claim_hash.clone(),
                LinkTypes::TokenToClaim,
                ()
            )?;
        }
    }

    // Index by category
    if let Some(category) = content.get("category").and_then(|c| c.as_str()) {
        create_link(
            Path::from(format!("category.{}", category)).hash()?,
            claim_hash.clone(),
            LinkTypes::CategoryToClaim,
            ()
        )?;
    }

    // Index by E-Tier (epistemic verifiability)
    create_link(
        Path::from(format!("e_tier.{:?}", claim.epistemic_tier_e)).hash()?,
        claim_hash.clone(),
        LinkTypes::ETierToClaim,
        ()
    )?;

    // Index by M-Tier (materiality - for pruning logic)
    create_link(
        Path::from(format!("m_tier.{:?}", claim.epistemic_tier_m)).hash()?,
        claim_hash.clone(),
        LinkTypes::MTierToClaim,
        ()
    )?;

    // Index by submitter (for "my claims" queries)
    create_link(
        agent_info()?.agent_latest_pubkey,
        claim_hash.clone(),
        LinkTypes::AgentToClaim,
        ()
    )?;

    Ok(())
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .filter(|token| token.len() >= 3)  // Ignore short words
        .take(10)  // Max 10 tokens per title (prevent spam)
        .map(String::from)
        .collect()
}

#[hdk_extern]
pub fn search_claims(query: SearchQuery) -> ExternResult<Vec<EpistemicClaim>> {
    let mut results = Vec::new();

    // Search by text token
    if let Some(search_text) = query.search_text {
        let tokens = tokenize(&search_text);
        for token in tokens {
            let links = get_links(
                Path::from(format!("token.{}", token)).hash()?,
                LinkTypes::TokenToClaim,
                None
            )?;

            for link in links {
                if let Some(claim) = get_claim_by_hash(&link.target.into())? {
                    results.push(claim);
                }
            }
        }
    }

    // Filter by category
    if let Some(category) = query.category {
        let links = get_links(
            Path::from(format!("category.{}", category)).hash()?,
            LinkTypes::CategoryToClaim,
            None
        )?;

        let category_claims: Vec<String> = links.iter()
            .map(|l| l.target.clone().into())
            .collect();

        results.retain(|c| category_claims.contains(&c.claim_id));
    }

    // Filter by E-Tier (e.g., "show only verified")
    if let Some(min_e_tier) = query.min_e_tier {
        results.retain(|c| c.epistemic_tier_e >= min_e_tier);
    }

    // Deduplicate
    results.sort_by(|a, b| a.claim_id.cmp(&b.claim_id));
    results.dedup_by(|a, b| a.claim_id == b.claim_id);

    Ok(results)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchQuery {
    pub search_text: Option<String>,
    pub category: Option<String>,
    pub min_e_tier: Option<EpistemicTierE>,
    pub max_results: Option<usize>,
}

fn get_claim_by_hash(hash: &ActionHash) -> ExternResult<Option<EpistemicClaim>> {
    let maybe_record = get(hash.clone(), GetOptions::default())?;
    match maybe_record {
        Some(record) => {
            let claim: EpistemicClaim = record.entry().to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest("Entry not found".into())))?;
            Ok(Some(claim))
        },
        None => Ok(None),
    }
}
```

#### Week 3: Materiality-Based Pruning
**File**: `mycelix-marketplace/zomes/dkg_index/src/pruning.rs`

```rust
use hdk::prelude::*;

/// Scheduled task: Prune ephemeral (M0) and temporal (M1) claims
#[hdk_extern]
pub fn prune_stale_claims(_: ()) -> ExternResult<()> {
    // 1. Query all M0 (Ephemeral) claims
    let m0_links = get_links(
        Path::from("m_tier.M0").hash()?,
        LinkTypes::MTierToClaim,
        None
    )?;

    for link in m0_links {
        // M0 claims are never stored on DKG, but may linger in DHT
        // Delete immediately (this is constitutional - see Charter Art I, Sec 8)
        delete_entry(link.target.into())?;
    }

    // 2. Query all M1 (Temporal) claims
    let m1_links = get_links(
        Path::from("m_tier.M1").hash()?,
        LinkTypes::MTierToClaim,
        None
    )?;

    for link in m1_links {
        let claim_hash: ActionHash = link.target.into();
        let claim = get_claim_by_hash(&claim_hash)?
            .ok_or(wasm_error!(WasmErrorInner::Guest("Claim not found".into())))?;

        // Check if state has changed (e.g., listing sold, message read)
        if is_state_obsolete(&claim)? {
            // Archive to IPFS (if important) or delete
            archive_or_delete_claim(&claim, &claim_hash)?;
        }
    }

    Ok(())
}

fn is_state_obsolete(claim: &EpistemicClaim) -> ExternResult<bool> {
    // Parse content to check status
    let content: serde_json::Value = serde_json::from_str(&claim.content)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    // Example: Marketplace listing
    if let Some(status) = content.get("status").and_then(|s| s.as_str()) {
        return Ok(status == "Sold" || status == "Removed");
    }

    // Example: Message
    if let Some(read) = content.get("read").and_then(|r| r.as_bool()) {
        return Ok(read);
    }

    // Default: not obsolete
    Ok(false)
}

fn archive_or_delete_claim(claim: &EpistemicClaim, hash: &ActionHash) -> ExternResult<()> {
    // For M1 claims, we archive metadata but remove from active DHT
    // (Full archival to IPFS/Filecoin is a Layer 2b concern)

    // Create archive record (stripped-down metadata only)
    let archive = ArchivedClaim {
        claim_id: claim.claim_id.clone(),
        submitted_by_did: claim.submitted_by_did.clone(),
        timestamp: claim.timestamp,
        claim_hash: claim.claim_hash.clone(),
        // NO CONTENT (just metadata for audit trail)
    };

    create_entry(EntryTypes::ArchivedClaim(archive))?;

    // Delete original from active DHT
    delete_entry(hash.clone())?;

    Ok(())
}

#[hdk_entry_helper]
pub struct ArchivedClaim {
    pub claim_id: String,
    pub submitted_by_did: String,
    pub timestamp: Timestamp,
    pub claim_hash: String,
    // No content - just metadata
}
```

**End of Week 3 Deliverable**:
- ✅ Pure P2P search working
- ✅ E/N/M tier filtering
- ✅ Automatic pruning per Constitutional mandate
- ✅ Can be used without any server (true decentralization)

---

### **Phase 2: PostgreSQL Accelerator (Weeks 4-6) - OPTIONAL**

**Goal**: Enable complex queries (SPARQL, full-text search, analytics) for power users

#### Week 4: DKG Sync Service
**File**: `mycelix-marketplace/dkg-sync-service/src/main.rs` (Python or Rust)

```rust
// This is a SEPARATE process (not part of Holochain conductor)
// Runs as a service, polls DHT, syncs to PostgreSQL

use holochain_client::*;
use sqlx::PgPool;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Connect to Holochain conductor
    let ws_url = "ws://localhost:8888";
    let client = AppWebsocket::connect(ws_url).await?;

    // 2. Connect to PostgreSQL
    let db_url = std::env::var("DATABASE_URL")?;
    let pool = PgPool::connect(&db_url).await?;

    // 3. Initialize schema
    initialize_schema(&pool).await?;

    // 4. Continuous sync loop
    loop {
        // Fetch new claims from DHT
        let new_claims = fetch_new_claims_from_dht(&client).await?;

        for claim in new_claims {
            // Validate signature (critical!)
            if !verify_claim_signature(&claim) {
                eprintln!("Invalid signature for claim {}", claim.claim_id);
                continue;
            }

            // Insert into PostgreSQL
            insert_claim_into_db(&pool, &claim).await?;

            // Index for full-text search
            update_search_index(&pool, &claim).await?;

            // Emit to RDF triple store (optional)
            emit_to_rdf_store(&claim).await?;
        }

        // Sleep 30 seconds before next poll
        tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
    }
}

async fn initialize_schema(pool: &PgPool) -> Result<(), sqlx::Error> {
    sqlx::query(r#"
        CREATE TABLE IF NOT EXISTS epistemic_claims (
            claim_id UUID PRIMARY KEY,
            claim_hash CHAR(64) NOT NULL,
            submitted_by_did TEXT NOT NULL,
            submitter_type TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,

            -- Epistemic tiers
            epistemic_tier_e VARCHAR(2) NOT NULL,  -- E0-E4
            epistemic_tier_n VARCHAR(2) NOT NULL,  -- N0-N3
            epistemic_tier_m VARCHAR(2) NOT NULL,  -- M0-M3

            -- Content
            content JSONB NOT NULL,

            -- Relationships
            related_claims JSONB,
            trust_tags JSONB,

            -- Signature (for verification)
            signature TEXT NOT NULL,

            -- Search index
            content_tsv TSVECTOR GENERATED ALWAYS AS (
                to_tsvector('english', content::text)
            ) STORED,

            -- Audit trail
            synced_at TIMESTAMPTZ DEFAULT NOW(),
            dht_address TEXT  -- Original DHT address for verification
        );

        CREATE INDEX IF NOT EXISTS idx_epistemic_e ON epistemic_claims(epistemic_tier_e);
        CREATE INDEX IF NOT EXISTS idx_epistemic_n ON epistemic_claims(epistemic_tier_n);
        CREATE INDEX IF NOT EXISTS idx_epistemic_m ON epistemic_claims(epistemic_tier_m);
        CREATE INDEX IF NOT EXISTS idx_submitted_by ON epistemic_claims(submitted_by_did);
        CREATE INDEX IF NOT EXISTS idx_content_search ON epistemic_claims USING GIN(content_tsv);
        CREATE INDEX IF NOT EXISTS idx_content_jsonb ON epistemic_claims USING GIN(content);
    "#).execute(pool).await?;

    Ok(())
}

async fn fetch_new_claims_from_dht(client: &AppWebsocket) -> Result<Vec<EpistemicClaim>, Box<dyn std::error::Error>> {
    // Call Holochain zome function to get all claims
    let response: Vec<EpistemicClaim> = client.call_zome(
        "dkg_index",
        "get_all_claims_since",
        GetClaimsSinceInput {
            since_timestamp: get_last_sync_timestamp()?,
        }
    ).await?;

    Ok(response)
}

fn verify_claim_signature(claim: &EpistemicClaim) -> bool {
    // Reconstruct the signed payload
    let canonical_json = serde_json::to_string(&claim.content).unwrap();
    let message = format!("{}{}", claim.claim_id, canonical_json);

    // Verify Ed25519 signature against submitted_by_did public key
    // (This is CRITICAL - prevents malicious indexers from injecting fake data)
    ed25519_dalek::verify(
        &claim.submitted_by_did_pubkey(),
        message.as_bytes(),
        &claim.signature
    ).is_ok()
}

async fn insert_claim_into_db(pool: &PgPool, claim: &EpistemicClaim) -> Result<(), sqlx::Error> {
    sqlx::query(r#"
        INSERT INTO epistemic_claims (
            claim_id, claim_hash, submitted_by_did, submitter_type,
            timestamp, epistemic_tier_e, epistemic_tier_n, epistemic_tier_m,
            content, related_claims, trust_tags, signature
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (claim_id) DO UPDATE SET
            content = EXCLUDED.content,
            synced_at = NOW()
    "#)
    .bind(claim.claim_id)
    .bind(&claim.claim_hash)
    .bind(&claim.submitted_by_did)
    .bind(format!("{:?}", claim.submitter_type))
    .bind(claim.timestamp)
    .bind(format!("{:?}", claim.epistemic_tier_e))
    .bind(format!("{:?}", claim.epistemic_tier_n))
    .bind(format!("{:?}", claim.epistemic_tier_m))
    .bind(serde_json::to_value(&claim.content)?)
    .bind(serde_json::to_value(&claim.related_claims)?)
    .bind(serde_json::to_value(&claim.trust_tags)?)
    .bind(&claim.signature)
    .execute(pool)
    .await?;

    Ok(())
}
```

#### Week 5: SPARQL Query Endpoint
**File**: `mycelix-marketplace/dkg-query-api/src/main.rs`

```rust
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use sqlx::PgPool;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let pool = PgPool::connect(&std::env::var("DATABASE_URL").unwrap())
        .await
        .unwrap();

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(pool.clone()))
            .route("/query/sparql", web::post().to(sparql_query))
            .route("/search/fulltext", web::get().to(fulltext_search))
            .route("/claims/{id}", web::get().to(get_claim_by_id))
            .route("/verify/{id}", web::get().to(verify_claim))
    })
    .bind(("0.0.0.0", 5000))?
    .run()
    .await
}

async fn sparql_query(
    pool: web::Data<PgPool>,
    query: web::Json<SparqlQuery>,
) -> impl Responder {
    // Example SPARQL query:
    // SELECT ?title ?price WHERE {
    //   ?claim a :Listing .
    //   ?claim :title ?title .
    //   ?claim :price ?price .
    //   FILTER(?price < 100)
    // }

    // Translate SPARQL to SQL (or use Apache Jena)
    let results = execute_sparql(&pool, &query.query).await;

    HttpResponse::Ok().json(results)
}

async fn fulltext_search(
    pool: web::Data<PgPool>,
    web::Query(params): web::Query<SearchParams>,
) -> impl Responder {
    let search_text = params.q;

    let results: Vec<EpistemicClaim> = sqlx::query_as(r#"
        SELECT * FROM epistemic_claims
        WHERE content_tsv @@ plainto_tsquery('english', $1)
        ORDER BY ts_rank(content_tsv, plainto_tsquery('english', $1)) DESC
        LIMIT 50
    "#)
    .bind(search_text)
    .fetch_all(pool.get_ref())
    .await
    .unwrap();

    HttpResponse::Ok().json(results)
}

async fn verify_claim(
    pool: web::Data<PgPool>,
    id: web::Path<String>,
) -> impl Responder {
    // 1. Fetch claim from PostgreSQL
    let claim: EpistemicClaim = sqlx::query_as(
        "SELECT * FROM epistemic_claims WHERE claim_id = $1"
    )
    .bind(id.as_str())
    .fetch_one(pool.get_ref())
    .await
    .unwrap();

    // 2. Verify signature
    let is_valid = verify_claim_signature(&claim);

    // 3. Query DHT to confirm claim exists there (ultimate source of truth)
    let dht_exists = query_dht_for_claim(&claim.claim_id).await.unwrap();

    HttpResponse::Ok().json(serde_json::json!({
        "claim_id": claim.claim_id,
        "signature_valid": is_valid,
        "exists_on_dht": dht_exists,
        "verified": is_valid && dht_exists
    }))
}
```

**End of Week 6 Deliverable**:
- ✅ PostgreSQL indexer syncing from DHT
- ✅ SPARQL query endpoint for complex queries
- ✅ Full-text search API
- ✅ Verification endpoint (proves data came from DHT)

---

## 🎯 Usage Patterns

### **Pattern 1: Pure P2P (No Server)**

```typescript
// Frontend directly calls Holochain conductor
import { AppWebsocket } from '@holochain/client';

const client = await AppWebsocket.connect('ws://localhost:8888');

// Search claims directly on DHT
const claims = await client.callZome({
  cap_secret: null,
  role_name: 'marketplace',
  zome_name: 'dkg_index',
  fn_name: 'search_claims',
  payload: {
    search_text: "vintage watch",
    min_e_tier: "E3",  // Only cryptographically proven items
  }
});
```

**Advantages**:
- ✅ Zero server cost
- ✅ Censorship-resistant
- ✅ True peer-to-peer
- ✅ Maximum privacy

**Tradeoffs**:
- ⚠️ Search limited to token matching (no fuzzy search)
- ⚠️ No analytics/reporting
- ⚠️ Depends on local DHT sync

### **Pattern 2: Hybrid (Optional Accelerator)**

```typescript
// Use PostgreSQL accelerator for complex queries
const response = await fetch('https://dkg-query.mycelix.net/search/fulltext?q=vintage+watch');
const claims = await response.json();

// Verify results came from DHT (trustless verification)
for (const claim of claims) {
  const verified = await fetch(`https://dkg-query.mycelix.net/verify/${claim.claim_id}`);
  const { exists_on_dht, signature_valid } = await verified.json();

  if (!exists_on_dht || !signature_valid) {
    console.warn(`Claim ${claim.claim_id} is NOT verified on DHT!`);
    // Don't display this result
  }
}
```

**Advantages**:
- ✅ Fast, fuzzy full-text search
- ✅ SPARQL for complex queries
- ✅ Analytics and reporting
- ✅ Verifiable (can check DHT)

**Tradeoffs**:
- ⚠️ Requires trusting indexer (or running your own)
- ⚠️ Server costs ($50-200/month)
- ⚠️ Potential centralization (if only one indexer exists)

---

## 🔐 Security & Trust Model

### **Why PostgreSQL is Still Decentralized-ish**

1. **DHT is Source of Truth**: PostgreSQL is just a cache
2. **Signatures are Verified**: Indexer can't inject fake data
3. **Anyone Can Run It**: Open-source sync service
4. **Verifiable**: Users can query DHT to confirm results
5. **Rebuildable**: Delete PostgreSQL, rebuild from DHT

### **Attack Scenarios**

| Attack | Mitigation |
|--------|------------|
| Malicious indexer injects fake claims | ❌ **Blocked**: Signatures verified |
| Indexer omits legitimate claims | ⚠️ **Detectable**: Users can query DHT directly |
| Indexer censors certain content | ⚠️ **Mitigated**: Run your own indexer, or use pure P2P |
| Indexer is down | ✅ **No problem**: Fall back to Holochain-only search |

---

## 📊 Performance Expectations

| Operation | Holochain Only | With PostgreSQL |
|-----------|----------------|-----------------|
| Search 10K listings | 2-5 seconds | 100-300ms |
| Filter by E-Tier | 1-2 seconds | 50ms |
| Complex SPARQL query | ❌ Not possible | 200-500ms |
| Verify claim authenticity | <1 second (query DHT) | <1 second (same) |
| Add new claim | <1 second (DHT write) | +30s (sync delay) |

---

## 🚀 Deployment Options

### **Option A: Pure P2P** (Recommended for MVP)
```bash
# Just run Holochain conductor
hc sandbox run
# That's it!
```

### **Option B: With PostgreSQL Accelerator** (For power users)
```bash
# 1. Run Holochain conductor
hc sandbox run -p 8888

# 2. Start PostgreSQL
docker run -d -p 5432:5432 postgres:15

# 3. Start DKG sync service
cd dkg-sync-service
cargo run --release

# 4. Start query API
cd dkg-query-api
cargo run --release

# 5. (Optional) Set up Nginx reverse proxy
nginx -c /etc/nginx/nginx.conf
```

---

## 📋 Next Steps

1. **This Week (Week 1)**: Implement Epistemic Claim entry type
2. **Week 2**: Build link-based indexing
3. **Week 3**: Add pruning logic
4. **Week 4-6**: (Optional) PostgreSQL accelerator

**Milestone**: After Week 3, you'll have a fully functional DKG with pure P2P search. The PostgreSQL accelerator is a nice-to-have for power users, not a blocker.

---

## 🤝 Contributing

See main [Mycelix CONTRIBUTING.md](../../CONTRIBUTING.md)

**Questions?**
- Holochain Forum: https://forum.holochain.org
- Email: tristan.stoltz@evolvingresonantcocreationism.com

---

**Status**: Ready to implement. Hybrid architecture validated. 🍄
