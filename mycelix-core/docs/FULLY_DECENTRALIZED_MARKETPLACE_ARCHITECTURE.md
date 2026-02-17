# 🍄 Mycelix Marketplace: Fully Decentralized Architecture v1.0

**Date**: November 11, 2025
**Status**: Implementation Design
**Commitment**: Zero centralized dependencies, pure peer-to-peer

---

## 🎯 Core Principle: True Decentralization

**No compromises**. This marketplace operates with:
- ✅ **Zero centralized databases** (no PostgreSQL fallback)
- ✅ **Pure Holochain DHT** for all data storage
- ✅ **IPFS** for all content (photos, documents)
- ✅ **Agent-centric trust** (calculated locally, verified peer-to-peer)
- ✅ **Distributed indexing** (DHT-based DKG, no central search service)
- ✅ **Peer-to-peer queries** (no REST APIs, direct agent communication)

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        User's Device                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │           Frontend (SvelteKit + Tauri)                    │ │
│  │  - Renders UI                                             │ │
│  │  - Calls Holochain Conductor locally                      │ │
│  │  - Fetches photos from IPFS gateway                       │ │
│  └──────────────────┬───────────────────────────────────────┘ │
│                     │                                           │
│  ┌──────────────────▼───────────────────────────────────────┐ │
│  │         Holochain Conductor (Local Node)                  │ │
│  │  - Runs Marketplace hApp                                  │ │
│  │  - Maintains source chain                                 │ │
│  │  - Gossips to DHT peers                                   │ │
│  └──────────────────┬───────────────────────────────────────┘ │
│                     │                                           │
└─────────────────────┼───────────────────────────────────────────┘
                      │
                      ▼ (Gossip Protocol)
        ┌─────────────────────────────────────┐
        │     Holochain DHT (Global P2P)      │
        │  ┌───────────────────────────────┐  │
        │  │  Validator Neighborhood 1     │  │
        │  │  - Validates entries          │  │
        │  │  - Stores sharded data        │  │
        │  └───────────────────────────────┘  │
        │  ┌───────────────────────────────┐  │
        │  │  Validator Neighborhood 2     │  │
        │  └───────────────────────────────┘  │
        │  ┌───────────────────────────────┐  │
        │  │  Validator Neighborhood N     │  │
        │  └───────────────────────────────┘  │
        └─────────────────────────────────────┘
                      │
                      ▼ (Content Addressing)
        ┌─────────────────────────────────────┐
        │         IPFS Network (Global)        │
        │  - Photos (image_xyz.jpg)            │
        │  - Product manuals (pdf_abc.pdf)     │
        │  - Authenticity certs (vc_123.json)  │
        └─────────────────────────────────────┘
```

### **Key Insight: No Servers**

Unlike the hybrid approach, there is **no backend server** to deploy. Every user runs:
1. A Holochain conductor (their personal node)
2. An IPFS client (for fetching photos)
3. The frontend (web app or desktop app via Tauri)

**Deployment model**: Distribute a **single executable** (via Nix standalone build) that bundles everything.

---

## 📦 Holochain DNA Structure

### **DNA: `marketplace.dna`**

```rust
// File: mycelix-marketplace/dnas/marketplace/dna.yaml
---
manifest_version: "1"
name: "mycelix_marketplace"
uid: "00000000-0000-0000-0000-000000000001"
properties: ~
zomes:
  - name: identity
    bundled: "./zomes/identity.wasm"
  - name: listings
    bundled: "./zomes/listings.wasm"
  - name: transactions
    bundled: "./zomes/transactions.wasm"
  - name: trust
    bundled: "./zomes/trust.wasm"
  - name: disputes
    bundled: "./zomes/disputes.wasm"
  - name: dkg_index
    bundled: "./zomes/dkg_index.wasm"
```

---

## 🧬 Zome-by-Zome Implementation

### **Zome 1: Identity** (`identity_zome`)

**Purpose**: DID creation, profile management, VC issuance

```rust
// File: zomes/identity/src/lib.rs
use hdk::prelude::*;

#[hdk_entry_helper]
pub struct AgentProfile {
    pub agent_pubkey: AgentPubKey,
    pub display_name: String,
    pub avatar_ipfs_cid: Option<String>,
    pub gitcoin_passport_score: u8,
    pub verified_human: bool,
    pub created_at: Timestamp,
}

#[hdk_entry_helper]
pub struct VerifiableCredential {
    pub vc_id: String,
    pub issuer_did: String,  // e.g., "did:mycelix:gucci_official"
    pub subject_did: String, // e.g., "did:mycelix:alice"
    pub credential_type: VCType,
    pub claims: String, // JSON string of claims
    pub signature: Signature,
    pub issued_at: Timestamp,
    pub expires_at: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum VCType {
    VerifiedHuman,           // From Gitcoin Passport
    ProductAuthenticity,     // From manufacturer
    TrustedSeller,           // From community DAO
    CertifiedAuthenticator,  // For authenticators
}

// Zome functions
#[hdk_extern]
pub fn create_profile(profile: AgentProfile) -> ExternResult<EntryHash> {
    create_entry(&EntryTypes::AgentProfile(profile))
}

#[hdk_extern]
pub fn get_my_profile() -> ExternResult<Option<AgentProfile>> {
    let agent = agent_info()?.agent_latest_pubkey;
    // Query DHT for profile by agent pubkey
    // ... implementation
    Ok(None) // Placeholder
}

#[hdk_extern]
pub fn issue_vc(vc: VerifiableCredential) -> ExternResult<EntryHash> {
    // Verify issuer has authority to issue this VC type
    let issuer = agent_info()?.agent_latest_pubkey;

    // TODO: Check if issuer is authorized (e.g., in allowlist)

    create_entry(&EntryTypes::VerifiableCredential(vc))
}

#[hdk_extern]
pub fn verify_vc(vc_hash: EntryHash) -> ExternResult<bool> {
    // Fetch VC from DHT
    let vc: VerifiableCredential = get(vc_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("VC not found".into())))?
        .entry()
        .to_app_option()?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid VC".into())))?;

    // Verify signature against issuer's pubkey
    // TODO: Implement crypto verification

    Ok(true) // Placeholder
}
```

---

### **Zome 2: Listings** (`listings_zome`)

**Purpose**: Create, read, update product listings as Epistemic Claims

```rust
// File: zomes/listings/src/lib.rs
use hdk::prelude::*;

#[hdk_entry_helper]
pub struct Listing {
    // Epistemic Claim metadata (per Charter)
    pub claim_id: String,
    pub submitted_by: AgentPubKey,
    pub timestamp: Timestamp,
    pub epistemic_tier: EpistemicTier,
    pub claim_materiality: ClaimMateriality,

    // Listing content
    pub title: String,
    pub description: String,
    pub category: String,  // e.g., "Electronics", "Fashion"
    pub photos_ipfs_cids: Vec<String>,
    pub condition: ProductCondition,

    // Pricing (multi-currency support)
    pub payment_options: Vec<PaymentOption>,

    // Authenticity
    pub authenticity_vc_hash: Option<EntryHash>,

    // Status
    pub status: ListingStatus,
    pub view_count: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum EpistemicTier {
    Null,                     // E0: Unverifiable
    Testimonial,              // E1: Seller's word
    PrivatelyVerifiable,      // E2: Community authenticated
    CryptographicallyProven,  // E3: Manufacturer VC
    PubliclyReproducible,     // E4: Open data (rare for products)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ClaimMateriality {
    Routine,        // < $100
    Significant,    // $100 - $1000
    Constitutional, // > $1000 (requires higher verification)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ProductCondition {
    New,
    LikeNew,
    Good,
    Fair,
    ForParts,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaymentOption {
    pub currency: Currency,
    pub amount: u64, // in smallest unit (cents, wei, etc.)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Currency {
    FLOW,
    TEND { credits: u64 },
    CGC,
    USD,
    // Add more as needed
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ListingStatus {
    Active,
    Sold,
    Removed,
    Disputed,
}

// Zome functions
#[hdk_extern]
pub fn create_listing(mut listing: Listing) -> ExternResult<EntryHash> {
    // Set metadata
    listing.claim_id = nanoid::nanoid!();
    listing.submitted_by = agent_info()?.agent_latest_pubkey;
    listing.timestamp = sys_time()?;

    // Determine epistemic tier based on VCs
    if listing.authenticity_vc_hash.is_some() {
        listing.epistemic_tier = EpistemicTier::CryptographicallyProven;
    } else {
        listing.epistemic_tier = EpistemicTier::Testimonial;
    }

    // Determine materiality based on price
    let max_price = listing.payment_options.iter()
        .filter_map(|opt| match opt.currency {
            Currency::USD => Some(opt.amount),
            _ => None,
        })
        .max()
        .unwrap_or(0);

    listing.claim_materiality = if max_price > 100_000 {
        ClaimMateriality::Constitutional
    } else if max_price > 10_000 {
        ClaimMateriality::Significant
    } else {
        ClaimMateriality::Routine
    };

    // Commit to source chain and gossip to DHT
    let listing_hash = create_entry(&EntryTypes::Listing(listing.clone()))?;

    // Create index entry for searchability (see DKG Index zome)
    call_zome(
        CallTargetCell::Local,
        ZomeName::from("dkg_index"),
        FunctionName::from("index_listing"),
        None,
        &listing
    )?;

    Ok(listing_hash)
}

#[hdk_extern]
pub fn get_listing(listing_hash: EntryHash) -> ExternResult<Option<Listing>> {
    let record = get(listing_hash, GetOptions::default())?;
    let listing = record
        .ok_or(wasm_error!(WasmErrorInner::Guest("Listing not found".into())))?
        .entry()
        .to_app_option()?;
    Ok(listing)
}

#[hdk_extern]
pub fn get_my_listings() -> ExternResult<Vec<Listing>> {
    let agent = agent_info()?.agent_latest_pubkey;

    // Query DHT for all listings by this agent
    // Note: This requires a link from agent -> listings
    let links = get_links(
        agent.into(),
        LinkTypes::AgentToListings,
        None
    )?;

    let mut listings = Vec::new();
    for link in links {
        if let Some(listing) = get_listing(link.target.into())? {
            listings.push(listing);
        }
    }

    Ok(listings)
}

#[hdk_extern]
pub fn update_listing_status(
    listing_hash: EntryHash,
    new_status: ListingStatus
) -> ExternResult<EntryHash> {
    // Get original listing
    let mut listing = get_listing(listing_hash.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Listing not found".into())))?;

    // Verify caller is the owner
    let agent = agent_info()?.agent_latest_pubkey;
    if listing.submitted_by != agent {
        return Err(wasm_error!(WasmErrorInner::Guest("Unauthorized".into())));
    }

    // Update status
    listing.status = new_status;

    // Create update entry
    let updated_hash = update_entry(listing_hash, &listing)?;

    Ok(updated_hash)
}
```

---

### **Zome 3: Transactions** (`transactions_zome`)

**Purpose**: Record purchases, reviews, and reputation events

```rust
// File: zomes/transactions/src/lib.rs
use hdk::prelude::*;

#[hdk_entry_helper]
pub struct Transaction {
    // Epistemic Claim metadata
    pub claim_id: String,
    pub submitted_by: AgentPubKey,  // Buyer
    pub timestamp: Timestamp,
    pub epistemic_tier: EpistemicTier, // Always Testimonial

    // Transaction details
    pub listing_hash: EntryHash,
    pub seller: AgentPubKey,
    pub payment_method: Currency,
    pub payment_amount: u64,
    pub payment_proof_hash: Option<String>, // e.g., blockchain tx hash

    // Delivery confirmation
    pub delivery_confirmed: bool,
    pub delivery_timestamp: Option<Timestamp>,

    // Review (PoGQ data!)
    pub review: Option<Review>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Review {
    pub rating: u8, // 1-5 stars
    pub comment: String,
    pub quality_as_described: bool,
    pub communication_rating: u8,
    pub shipping_speed_rating: u8,
}

#[hdk_extern]
pub fn record_transaction(tx: Transaction) -> ExternResult<EntryHash> {
    let tx_hash = create_entry(&EntryTypes::Transaction(tx.clone()))?;

    // Update seller's reputation (call trust zome)
    call_zome(
        CallTargetCell::Local,
        ZomeName::from("trust"),
        FunctionName::from("record_transaction_event"),
        None,
        &(tx.seller.clone(), tx.review.clone())
    )?;

    // Update listing status to "Sold"
    call_zome(
        CallTargetCell::Local,
        ZomeName::from("listings"),
        FunctionName::from("update_listing_status"),
        None,
        &(tx.listing_hash.clone(), ListingStatus::Sold)
    )?;

    Ok(tx_hash)
}

#[hdk_extern]
pub fn get_transactions_as_buyer() -> ExternResult<Vec<Transaction>> {
    let agent = agent_info()?.agent_latest_pubkey;

    let links = get_links(
        agent.into(),
        LinkTypes::BuyerToTransactions,
        None
    )?;

    let mut transactions = Vec::new();
    for link in links {
        if let Ok(Some(tx)) = get(link.target.clone(), GetOptions::default())? {
            if let Some(tx_data) = tx.entry().to_app_option()? {
                transactions.push(tx_data);
            }
        }
    }

    Ok(transactions)
}

#[hdk_extern]
pub fn get_transactions_as_seller() -> ExternResult<Vec<Transaction>> {
    let agent = agent_info()?.agent_latest_pubkey;

    let links = get_links(
        agent.into(),
        LinkTypes::SellerToTransactions,
        None
    )?;

    let mut transactions = Vec::new();
    for link in links {
        if let Ok(Some(tx)) = get(link.target.clone(), GetOptions::default())? {
            if let Some(tx_data) = tx.entry().to_app_option()? {
                transactions.push(tx_data);
            }
        }
    }

    Ok(transactions)
}
```

---

### **Zome 4: Trust** (`trust_zome`)

**Purpose**: Local trust calculation using 0TML algorithms

**KEY INSIGHT**: Trust scores are NOT stored globally. Each agent calculates them locally by querying transaction history from the DHT.

```rust
// File: zomes/trust/src/lib.rs
use hdk::prelude::*;

// No entry types here - trust is calculated, not stored

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrustScore {
    pub agent: AgentPubKey,
    pub composite_score: f64,  // 0.0 - 1.0
    pub pogq_score: f64,       // Proof of Quality
    pub tcdm_score: f64,       // Behavioral consistency
    pub entropy_score: f64,    // Account age/diversity
    pub civ_score: f64,        // Reputation stake
    pub confidence: f64,       // How much data we have
    pub calculated_at: Timestamp,
}

#[hdk_extern]
pub fn calculate_trust_score(agent: AgentPubKey) -> ExternResult<TrustScore> {
    // 1. Query all transactions for this agent as seller
    let transactions: Vec<Transaction> = call_zome(
        CallTargetCell::Local,
        ZomeName::from("transactions"),
        FunctionName::from("get_transactions_as_seller"),
        None,
        &agent
    )?.decode()?;

    // 2. Extract reviews for PoGQ calculation
    let reviews: Vec<Review> = transactions
        .iter()
        .filter_map(|tx| tx.review.clone())
        .collect();

    let pogq_score = calculate_pogq(&reviews);

    // 3. Calculate behavioral consistency (TCDM)
    let tcdm_score = calculate_tcdm(&transactions);

    // 4. Calculate entropy (account age, transaction diversity)
    let entropy_score = calculate_entropy(&agent, &transactions)?;

    // 5. Query CIV stake from governance (future: cross-zome call)
    let civ_score = 0.5; // Placeholder

    // 6. Composite score (weighted average)
    let composite_score =
        pogq_score * 0.4 +
        tcdm_score * 0.3 +
        entropy_score * 0.2 +
        civ_score * 0.1;

    Ok(TrustScore {
        agent,
        composite_score,
        pogq_score,
        tcdm_score,
        entropy_score,
        civ_score,
        confidence: calculate_confidence(reviews.len()),
        calculated_at: sys_time()?,
    })
}

fn calculate_pogq(reviews: &[Review]) -> f64 {
    if reviews.is_empty() {
        return 0.5; // Neutral for new sellers
    }

    let avg_rating = reviews.iter()
        .map(|r| r.rating as f64)
        .sum::<f64>() / reviews.len() as f64;

    // Normalize to 0-1 scale (5-star system)
    avg_rating / 5.0
}

fn calculate_tcdm(transactions: &[Transaction]) -> f64 {
    // TCDM = Temporal Consistency Detection Metric
    // Measures if behavior is stable over time

    if transactions.len() < 3 {
        return 0.5; // Insufficient data
    }

    // Calculate variance in ratings over time
    let ratings: Vec<f64> = transactions
        .iter()
        .filter_map(|tx| tx.review.as_ref().map(|r| r.rating as f64))
        .collect();

    if ratings.is_empty() {
        return 0.5;
    }

    let mean = ratings.iter().sum::<f64>() / ratings.len() as f64;
    let variance = ratings.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / ratings.len() as f64;

    // Low variance = high consistency
    // Map variance (0-4) to score (1-0)
    (4.0 - variance.min(4.0)) / 4.0
}

fn calculate_entropy(agent: &AgentPubKey, transactions: &[Transaction]) -> ExternResult<f64> {
    // Entropy = account age + transaction diversity

    // 1. Account age (query first activity)
    let creation_time = get_agent_creation_time(agent)?;
    let current_time = sys_time()?;
    let age_days = (current_time.as_millis() - creation_time.as_millis()) / (1000 * 60 * 60 * 24);

    // Age score: 0-1 over 365 days
    let age_score = (age_days as f64 / 365.0).min(1.0);

    // 2. Transaction diversity (different buyers)
    let unique_buyers: std::collections::HashSet<_> = transactions
        .iter()
        .map(|tx| tx.submitted_by.clone())
        .collect();

    // Diversity score: 0-1 for 10+ unique buyers
    let diversity_score = (unique_buyers.len() as f64 / 10.0).min(1.0);

    // Combined entropy
    Ok((age_score + diversity_score) / 2.0)
}

fn calculate_confidence(review_count: usize) -> f64 {
    // Confidence increases with more data
    // 0-1 scale, plateaus at 20 reviews
    (review_count as f64 / 20.0).min(1.0)
}

fn get_agent_creation_time(agent: &AgentPubKey) -> ExternResult<Timestamp> {
    // Query the agent's first activity on source chain
    // Placeholder: return current time
    sys_time()
}
```

---

### **Zome 5: Disputes** (`disputes_zome`)

**Purpose**: Epistemic dispute resolution (MRC implementation)

```rust
// File: zomes/disputes/src/lib.rs
use hdk::prelude::*;

#[hdk_entry_helper]
pub struct Dispute {
    pub claim_id: String,
    pub submitted_by: AgentPubKey, // Buyer
    pub timestamp: Timestamp,
    pub epistemic_tier: EpistemicTier, // Testimonial initially

    // Dispute details
    pub listing_hash: EntryHash,
    pub transaction_hash: EntryHash,
    pub dispute_type: DisputeType,
    pub evidence_ipfs_cids: Vec<String>, // Photos, lab reports
    pub description: String,

    // MRC adjudication
    pub mrc_panel: Option<Vec<AgentPubKey>>,
    pub ruling: Option<DisputeRuling>,
    pub ruling_timestamp: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DisputeType {
    Counterfeit,
    NotAsDescribed,
    NonDelivery,
    Defective,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DisputeRuling {
    pub verdict: Verdict,
    pub arbitrators: Vec<AgentPubKey>,
    pub evidence_review: String,
    pub enforcement_actions: Vec<EnforcementAction>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Verdict {
    BuyerWins,
    SellerWins,
    Split { buyer_refund_percent: u8 },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum EnforcementAction {
    InvalidateListing { listing_hash: EntryHash },
    SlashReputation { agent: AgentPubKey, amount: f64 },
    RefundBuyer { amount: u64 },
    BanSeller { duration_days: u64 },
}

#[hdk_extern]
pub fn file_dispute(mut dispute: Dispute) -> ExternResult<EntryHash> {
    dispute.claim_id = nanoid::nanoid!();
    dispute.submitted_by = agent_info()?.agent_latest_pubkey;
    dispute.timestamp = sys_time()?;
    dispute.epistemic_tier = EpistemicTier::Testimonial;

    let dispute_hash = create_entry(&EntryTypes::Dispute(dispute.clone()))?;

    // Select MRC panel (3 random arbitrators)
    let panel = select_mrc_panel()?;

    // Notify panel members (create links for their inbox)
    for arbitrator in &panel {
        create_link(
            arbitrator.clone().into(),
            dispute_hash.clone(),
            LinkTypes::ArbitratorToDisputes,
            ()
        )?;
    }

    Ok(dispute_hash)
}

#[hdk_extern]
pub fn submit_ruling(
    dispute_hash: EntryHash,
    ruling: DisputeRuling
) -> ExternResult<EntryHash> {
    // Get original dispute
    let mut dispute = get(dispute_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Dispute not found".into())))?
        .entry()
        .to_app_option::<Dispute>()?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid dispute".into())))?;

    // Verify caller is on MRC panel
    let caller = agent_info()?.agent_latest_pubkey;
    let panel = dispute.mrc_panel.clone()
        .ok_or(wasm_error!(WasmErrorInner::Guest("No panel assigned".into())))?;

    if !panel.contains(&caller) {
        return Err(wasm_error!(WasmErrorInner::Guest("Unauthorized".into())));
    }

    // Update dispute with ruling
    dispute.ruling = Some(ruling.clone());
    dispute.ruling_timestamp = Some(sys_time()?);
    dispute.epistemic_tier = EpistemicTier::PrivatelyVerifiable; // MRC verified

    let updated_hash = update_entry(dispute_hash, &dispute)?;

    // Execute enforcement actions
    for action in ruling.enforcement_actions {
        execute_enforcement_action(action)?;
    }

    Ok(updated_hash)
}

fn select_mrc_panel() -> ExternResult<Vec<AgentPubKey>> {
    // TODO: Query all eligible arbitrators (high CIV, staked FLOW)
    // For now, return placeholder
    Ok(vec![])
}

fn execute_enforcement_action(action: EnforcementAction) -> ExternResult<()> {
    match action {
        EnforcementAction::InvalidateListing { listing_hash } => {
            // Mark listing as disputed/invalid
            call_zome(
                CallTargetCell::Local,
                ZomeName::from("listings"),
                FunctionName::from("update_listing_status"),
                None,
                &(listing_hash, ListingStatus::Disputed)
            )?;
        },
        EnforcementAction::SlashReputation { agent, amount } => {
            // Record reputation slash event
            // Trust zome will see this when recalculating
            // TODO: Implement reputation event storage
        },
        EnforcementAction::RefundBuyer { amount: _ } => {
            // Trigger escrow contract release
            // TODO: Integrate with Layer 4 bridge
        },
        EnforcementAction::BanSeller { duration_days: _ } => {
            // Add to banlist (time-limited link)
            // TODO: Implement ban mechanism
        },
    }
    Ok(())
}
```

---

### **Zome 6: DKG Index** (`dkg_index_zome`)

**Purpose**: Distributed indexing for searchability (the hardest part!)

**Challenge**: Holochain's DHT doesn't have built-in full-text search. We need to build a distributed index.

**Solution**: Use **semantic hashing** and **DHT sharding** for search.

```rust
// File: zomes/dkg_index/src/lib.rs
use hdk::prelude::*;

// Index entries for search
#[hdk_entry_helper]
pub struct ListingIndex {
    pub listing_hash: EntryHash,
    pub seller: AgentPubKey,
    pub category: String,
    pub title_tokens: Vec<String>,  // Tokenized title for search
    pub price_range: PriceRange,
    pub epistemic_tier: EpistemicTier,
    pub timestamp: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum PriceRange {
    Under100,
    Between100And1000,
    Between1000And10000,
    Over10000,
}

#[hdk_extern]
pub fn index_listing(listing: Listing) -> ExternResult<()> {
    // Tokenize title for search
    let title_tokens = tokenize(&listing.title);

    // Determine price range
    let price_range = determine_price_range(&listing.payment_options);

    let index_entry = ListingIndex {
        listing_hash: hash_entry(&listing)?,
        seller: listing.submitted_by.clone(),
        category: listing.category.clone(),
        title_tokens: title_tokens.clone(),
        price_range,
        epistemic_tier: listing.epistemic_tier.clone(),
        timestamp: listing.timestamp,
    };

    let index_hash = create_entry(&EntryTypes::ListingIndex(index_entry))?;

    // Create searchable links
    // Link 1: Category -> Listing
    create_link(
        Path::from(format!("category.{}", listing.category)).hash()?,
        listing.claim_id.into(),
        LinkTypes::CategoryToListings,
        ()
    )?;

    // Link 2: Each token -> Listing (for text search)
    for token in title_tokens {
        create_link(
            Path::from(format!("token.{}", token)).hash()?,
            index_hash.clone(),
            LinkTypes::TokenToListings,
            ()
        )?;
    }

    // Link 3: Price range -> Listing
    create_link(
        Path::from(format!("price.{:?}", price_range)).hash()?,
        index_hash.clone(),
        LinkTypes::PriceToListings,
        ()
    )?;

    // Link 4: Epistemic tier -> Listing (for filtering)
    create_link(
        Path::from(format!("tier.{:?}", listing.epistemic_tier)).hash()?,
        index_hash.clone(),
        LinkTypes::TierToListings,
        ()
    )?;

    Ok(())
}

#[hdk_extern]
pub fn search_listings(query: SearchQuery) -> ExternResult<Vec<Listing>> {
    let mut result_hashes: Vec<EntryHash> = vec![];

    // 1. Search by category
    if let Some(category) = query.category {
        let links = get_links(
            Path::from(format!("category.{}", category)).hash()?,
            LinkTypes::CategoryToListings,
            None
        )?;
        result_hashes.extend(links.iter().map(|l| l.target.clone().into()));
    }

    // 2. Search by text tokens
    if let Some(search_text) = query.search_text {
        let tokens = tokenize(&search_text);
        for token in tokens {
            let links = get_links(
                Path::from(format!("token.{}", token)).hash()?,
                LinkTypes::TokenToListings,
                None
            )?;
            result_hashes.extend(links.iter().map(|l| l.target.clone().into()));
        }
    }

    // 3. Filter by price range
    if let Some(price_range) = query.price_range {
        let links = get_links(
            Path::from(format!("price.{:?}", price_range)).hash()?,
            LinkTypes::PriceToListings,
            None
        )?;
        result_hashes.extend(links.iter().map(|l| l.target.clone().into()));
    }

    // 4. Fetch actual listings
    let mut listings = Vec::new();
    for hash in result_hashes {
        if let Ok(Some(listing)) = call_zome(
            CallTargetCell::Local,
            ZomeName::from("listings"),
            FunctionName::from("get_listing"),
            None,
            &hash
        )?.decode() {
            listings.push(listing);
        }
    }

    // 5. Client-side filtering by trust score
    // (Frontend will call trust_zome for each seller and filter)

    Ok(listings)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchQuery {
    pub category: Option<String>,
    pub search_text: Option<String>,
    pub price_range: Option<PriceRange>,
    pub min_epistemic_tier: Option<EpistemicTier>,
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .filter(|s| s.len() > 2) // Skip short words
        .map(|s| s.to_string())
        .collect()
}

fn determine_price_range(payment_options: &[PaymentOption]) -> PriceRange {
    let max_usd = payment_options.iter()
        .filter_map(|opt| match opt.currency {
            Currency::USD => Some(opt.amount),
            _ => None,
        })
        .max()
        .unwrap_or(0);

    if max_usd < 10_000 {
        PriceRange::Under100
    } else if max_usd < 100_000 {
        PriceRange::Between100And1000
    } else if max_usd < 1_000_000 {
        PriceRange::Between1000And10000
    } else {
        PriceRange::Over10000
    }
}
```

---

## 🌐 Frontend Architecture (Fully Local)

### **Technology Stack**

```yaml
Framework: SvelteKit (SSR disabled, pure SPA)
Desktop App: Tauri (bundles Holochain conductor)
Styling: TailwindCSS + Shadcn/ui components
State Management: Svelte stores
Holochain Client: @holochain/client (JavaScript SDK)
IPFS Client: js-ipfs or ipfs-http-client
Wallet: WalletConnect + MetaMask for crypto
```

### **Key Frontend Files**

```
frontend/
├── src/
│   ├── lib/
│   │   ├── holochain/
│   │   │   ├── client.ts          # Holochain conductor connection
│   │   │   ├── listings.ts        # Listings zome calls
│   │   │   ├── trust.ts           # Trust zome calls
│   │   │   └── transactions.ts    # Transactions zome calls
│   │   ├── ipfs/
│   │   │   └── client.ts          # IPFS upload/fetch
│   │   └── stores/
│   │       ├── listings.ts        # Global listing state
│   │       ├── trust.ts           # Cached trust scores
│   │       └── user.ts            # Current user profile
│   ├── routes/
│   │   ├── +page.svelte           # Browse listings
│   │   ├── create/+page.svelte    # Create listing
│   │   ├── listing/[id]/+page.svelte  # Listing detail
│   │   ├── profile/+page.svelte   # User profile
│   │   └── disputes/+page.svelte  # File/track disputes
│   └── app.html
├── src-tauri/                     # Tauri desktop app config
│   ├── Cargo.toml
│   └── src/
│       └── main.rs                # Bundle Holochain conductor
└── package.json
```

---

## 🔗 IPFS Integration

### **Content Addressing Strategy**

All photos, documents, and large files are stored on IPFS, with CIDs referenced in Holochain entries.

```typescript
// File: frontend/src/lib/ipfs/client.ts
import { create } from 'ipfs-http-client';

// Connect to local IPFS node or public gateway
const ipfs = create({
  host: 'localhost',
  port: 5001,
  protocol: 'http'
});

export async function uploadFile(file: File): Promise<string> {
  const added = await ipfs.add(file);
  return added.cid.toString(); // Returns CID like "QmXxx..."
}

export function getIpfsUrl(cid: string): string {
  return `https://ipfs.io/ipfs/${cid}`;
  // Or use local gateway: http://localhost:8080/ipfs/${cid}
}
```

### **Photo Upload Flow**

```
1. User selects photos in frontend
2. Frontend uploads to local IPFS node
3. IPFS pins the content (returns CID)
4. Frontend creates listing with photos_ipfs_cids: ["QmXxx", "QmYyy"]
5. Holochain stores listing entry (with CIDs, not photos)
6. Other users fetch photos directly from IPFS by CID
```

---

## 🔐 Security & Trust Model

### **How Trust Filtering Works (Client-Side)**

```typescript
// File: frontend/src/routes/+page.svelte (Browse Listings)
<script lang="ts">
  import { onMount } from 'svelte';
  import { searchListings, type Listing } from '$lib/holochain/listings';
  import { calculateTrustScore, type TrustScore } from '$lib/holochain/trust';

  let listings: Listing[] = [];
  let trustScores: Map<string, TrustScore> = new Map();
  let minTrustFilter = 0.8; // User-adjustable

  onMount(async () => {
    // 1. Query DHT for listings
    listings = await searchListings({
      category: 'Electronics',
      search_text: 'laptop'
    });

    // 2. Calculate trust scores for all sellers (parallel)
    const sellers = [...new Set(listings.map(l => l.submitted_by))];
    const scores = await Promise.all(
      sellers.map(agent => calculateTrustScore(agent))
    );

    // 3. Build trust map
    scores.forEach((score, i) => {
      trustScores.set(sellers[i], score);
    });

    // 4. Filter by trust (client-side)
    listings = listings.filter(l => {
      const score = trustScores.get(l.submitted_by);
      return score && score.composite_score >= minTrustFilter;
    });

    // 5. Sort by trust (best first)
    listings.sort((a, b) => {
      const scoreA = trustScores.get(a.submitted_by)?.composite_score || 0;
      const scoreB = trustScores.get(b.submitted_by)?.composite_score || 0;
      return scoreB - scoreA;
    });
  });
</script>

<div class="trust-filter">
  <label>
    Minimum Seller Trust:
    <input
      type="range"
      min="0"
      max="1"
      step="0.1"
      bind:value={minTrustFilter}
      on:change={() => /* Re-filter listings */}
    />
    <span>{(minTrustFilter * 100).toFixed(0)}%</span>
  </label>

  <p class="text-sm text-gray-600">
    Showing {listings.length} trusted listings
    (Filtered out {/* original count - filtered count */} low-trust sellers)
  </p>
</div>

{#each listings as listing}
  <div class="listing-card">
    <img src={getIpfsUrl(listing.photos_ipfs_cids[0])} alt={listing.title} />
    <h3>{listing.title}</h3>

    <!-- Trust Badge -->
    <div class="trust-badge" class:high={trustScores.get(listing.submitted_by)?.composite_score > 0.9}>
      🛡️ Trust: {(trustScores.get(listing.submitted_by)?.composite_score * 100).toFixed(0)}%

      {#if listing.authenticity_vc_hash}
        <span class="verified">✓ Verified Authentic</span>
      {/if}
    </div>

    <p class="price">
      {listing.payment_options[0].currency}: ${listing.payment_options[0].amount / 100}
    </p>
  </div>
{/each}
```

---

## 🚀 Deployment Model

### **Standalone Executable (No Servers)**

Users download a **single executable** built with Nix that contains:

```
mycelix-marketplace-v1.0.0-linux-x86_64
├── holochain-conductor (bundled)
├── ipfs daemon (bundled)
└── tauri app (bundled)
    └── frontend (SvelteKit SPA)
```

**First run**:
```bash
./mycelix-marketplace
# Starts:
# 1. Holochain conductor on localhost:8888
# 2. IPFS daemon on localhost:5001
# 3. Tauri app (opens window)
```

**User sees**: A desktop app that looks like a normal marketplace website, but everything is running locally.

---

## 📊 Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Create listing | ~500ms | Local source chain write + DHT gossip |
| Search listings | ~2-5s | DHT queries across neighborhoods |
| Calculate trust score | ~100-500ms | Depends on transaction count |
| Fetch IPFS photo | ~1-3s | Depends on IPFS pinning |
| Submit dispute | ~300ms | Local write + MRC notification |

**Scalability**:
- Each DHT neighborhood stores ~5% of all listings
- Search performance degrades gracefully (not catastrophically)
- No central bottleneck—scales horizontally

---

## 🔄 Migration from 0TML

Your existing 0TML trust algorithms can be **directly ported** to the trust zome:

```rust
// File: zomes/trust/src/pogq.rs
// Import 0TML's PoGQ implementation
use zerotrustml::aggregation::pogq::calculate_pogq as ztm_pogq;

pub fn calculate_pogq_holochain(reviews: &[Review]) -> f64 {
    // Convert Review -> 0TML format
    let ztm_reviews: Vec<_> = reviews.iter().map(|r| {
        zerotrustml::types::Review {
            rating: r.rating,
            // ... map fields
        }
    }).collect();

    // Call 0TML algorithm
    ztm_pogq(&ztm_reviews)
}
```

**Benefit**: Reuse battle-tested code from 0TML without rewriting.

---

## 🎯 Next Steps

I'll now create:
1. ✅ Full architecture document (this file)
2. ⏳ Holochain DNA workspace setup
3. ⏳ Rust zome implementations
4. ⏳ Frontend SvelteKit scaffold
5. ⏳ NixOS build configuration for standalone executable

**Estimated completion**: 4-6 months with 2-3 developers

**First milestone (Week 4)**: "Hello World" listing created and queried via DHT

---

**This is the true decentralized vision. No compromises. Pure peer-to-peer.** 🍄
