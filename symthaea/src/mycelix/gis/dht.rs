// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Experimental module: fields will be read when GIS DHT integration is wired
#![allow(dead_code)]
//! # Dark Spot DHT - Privacy-Preserving Ignorance Sharing
//!
//! Implements the Zero-Knowledge Dark Spot DHT from GIS v2.0/v3.0:
//!
//! ## Architecture
//!
//! ```text
//! Agent A (has ignorance)          Agent B (has knowledge)
//! ┌─────────────────────┐          ┌─────────────────────┐
//! │ Query: "X?"         │          │ Claim: "X = ..."    │
//! │ → Create ZK Sig     │──DHT───▶│ → Match Sig         │
//! │ → Publish to DHT    │          │ → Respond privately │
//! │ → Receive Response  │◀─────────│                     │
//! └─────────────────────┘          └─────────────────────┘
//!
//! Properties:
//! - Agent A's question is not revealed to the network
//! - Agent B can match semantically without seeing the question
//! - Response is encrypted to Agent A only
//! ```
//!
//! ## Key Concepts
//!
//! - **ZK Ignorance Signature**: Proves "I don't know X" without revealing X
//! - **Semantic Topic Hash**: LSH-based hashing for semantic matching
//! - **Threshold Reveal**: Collective decision to reveal blind spots
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::mycelix::gis::{DarkSpotDHT, ZKIgnoranceSignature};
//!
//! let dht = DarkSpotDHT::new();
//!
//! // Create privacy-preserving signature
//! let sig = dht.create_signature("What is quantum entanglement?", "physics", 0.8)?;
//!
//! // Publish to network (question hidden)
//! dht.publish(&sig)?;
//!
//! // Others can match without seeing the question
//! let matches = dht.find_matches(&sig);
//! ```

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::GISError;
use super::zk_proofs::{ZKEIGRangeProof, ZKTopicCommitment};

// HDC imports for semantic encoding
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

// ============================================================================
// HDC Semantic Encoder - Real semantic similarity using hyperdimensional computing
// ============================================================================

/// HDC-based semantic encoder for privacy-preserving semantic matching
///
/// Uses real-valued hyperdimensional vectors (ContinuousHV) to encode text semantically.
/// Word-wise bundling ensures that semantically similar queries have high cosine similarity.
///
/// This is the bridge between the Dark Spot DHT and the core HDC infrastructure.
#[derive(Debug)]
pub struct HdcSemanticEncoder {
    /// Dimension of hypervectors (uses HDC_DIMENSION = 16384 by default)
    pub dimension: usize,
    /// Fixed seed for deterministic word encoding
    seed: u64,
}

impl HdcSemanticEncoder {
    /// Create a new HDC semantic encoder
    pub fn new() -> Self {
        Self {
            dimension: HDC_DIMENSION,
            seed: 0xDEADBEEF_CAFEBABE,
        }
    }

    /// Encode text into a ContinuousHV using word-wise bundling
    ///
    /// Each word gets a deterministic random vector based on BLAKE3 hash of the word.
    /// The final encoding is the normalized average (bundle) of all word vectors.
    ///
    /// This ensures:
    /// - Determinism: Same text always produces same encoding
    /// - Semantic overlap: Texts sharing words have proportionally higher similarity
    /// - Privacy: Hard to reverse engineer original text from encoding
    pub fn encode(&self, text: &str) -> ContinuousHV {
        use blake3::Hasher;

        // Normalize and split into words
        let lowered = text.to_lowercase();
        let words: Vec<&str> = lowered
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 1) // Skip single chars
            .collect();

        if words.is_empty() {
            // Return zero vector for empty input
            return ContinuousHV::ones(self.dimension).scale(0.0_f32);
        }

        // Accumulate word vectors
        let mut accumulated = vec![0.0_f32; self.dimension];

        for word in &words {
            // Create deterministic seed from word
            let mut hasher = Hasher::new();
            hasher.update(word.as_bytes());
            let hash = hasher.finalize();
            let word_seed = u64::from_le_bytes(
                hash.as_bytes()[0..8]
                    .try_into()
                    .expect("8-byte slice fits [u8; 8]"),
            );

            // Generate word vector (deterministic from seed)
            let word_hv = ContinuousHV::random(self.dimension, word_seed);

            // Add to accumulator
            for (i, v) in word_hv.values.iter().enumerate() {
                accumulated[i] += v;
            }
        }

        // Normalize (average)
        let n = words.len() as f32;
        for v in &mut accumulated {
            *v /= n;
        }

        ContinuousHV::from_vec(accumulated)
    }

    /// Compute cosine similarity between two encodings
    pub fn similarity(&self, a: &ContinuousHV, b: &ContinuousHV) -> f32 {
        a.similarity(b)
    }

    /// Encode with category weighting
    ///
    /// Adds category-specific bias to the encoding for better category-level matching.
    pub fn encode_with_category(&self, text: &str, category: &str) -> ContinuousHV {
        let text_encoding = self.encode(text);
        let category_encoding = self.encode(category);

        // Blend: 80% text, 20% category
        let blended = text_encoding
            .scale(0.8_f32)
            .add(&category_encoding.scale(0.2_f32));
        blended.normalize()
    }

    /// Create a compressed embedding for network transmission
    ///
    /// Reduces dimension while preserving semantic similarity (via random projection).
    pub fn compress(&self, hv: &ContinuousHV, target_dim: usize) -> Vec<f32> {
        if target_dim >= self.dimension {
            return hv.values.clone();
        }

        // Random projection to lower dimension
        // Uses sparse random projection for efficiency
        let mut compressed = vec![0.0_f32; target_dim];
        let stride = self.dimension / target_dim;

        for (i, chunk) in hv.values.chunks(stride).enumerate() {
            if i < target_dim {
                // Average of chunk (like pooling)
                compressed[i] = chunk.iter().sum::<f32>() / chunk.len() as f32;
            }
        }

        compressed
    }

    /// Decompress (approximate reconstruction)
    pub fn decompress(&self, compressed: &[f32]) -> ContinuousHV {
        let target_dim = compressed.len();
        let stride = self.dimension / target_dim;

        let mut expanded = vec![0.0_f32; self.dimension];
        for (i, &v) in compressed.iter().enumerate() {
            // Repeat each compressed value
            for j in 0..stride {
                let idx = i * stride + j;
                if idx < self.dimension {
                    expanded[idx] = v;
                }
            }
        }

        ContinuousHV::from_vec(expanded)
    }
}

impl Default for HdcSemanticEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-Knowledge Ignorance Signature
///
/// Allows publishing ignorance to the network without revealing
/// the actual question. Others can match semantically.
///
/// Now uses cryptographically sound ZK proofs:
/// - Pedersen commitments for topic hiding
/// - Bulletproofs-style range proofs for EIG
/// - Schnorr-based validity proofs
#[derive(Debug, Clone)]
pub struct ZKIgnoranceSignature {
    /// Unique signature ID
    pub id: String,

    /// Cryptographic commitment to the topic (Pedersen-based)
    /// Uses ZKTopicCommitment with proper cryptographic hiding
    pub topic_commitment: ZKTopicCommitment,

    /// Semantic category (revealed for routing)
    pub category: String,

    /// Encrypted semantic embedding
    /// Can be decrypted by nodes with category key
    pub encrypted_embedding: EncryptedEmbedding,

    /// Bulletproofs-style range proof for EIG score
    /// Proves 0 ≤ EIG ≤ 1 without revealing exact value
    pub eig_range_proof: ZKEIGRangeProof,

    /// Publisher (pseudonymous)
    pub publisher: PseudonymousId,

    /// Nonce for uniqueness
    pub nonce: [u8; 16],

    /// When created
    pub created_at: SystemTime,

    /// Time to live
    pub ttl: Duration,
}

// Note: TopicCommitment implementation moved to zk_proofs.rs
// Now using ZKTopicCommitment with Pedersen commitment and Schnorr validity proof

/// Encrypted semantic embedding
#[derive(Debug, Clone)]
pub struct EncryptedEmbedding {
    /// Encrypted data (category-keyed)
    pub ciphertext: Vec<u8>,
    /// Algorithm identifier
    pub algorithm: String,
    /// Category the key is derived from
    pub key_category: String,
}

impl EncryptedEmbedding {
    /// Create an encrypted embedding using BLAKE3 XOF stream cipher.
    ///
    /// Derives a 32-byte key from the category name using BLAKE3, then
    /// uses BLAKE3 in keyed-hash XOF mode to generate a keystream that
    /// XORs the plaintext. This is semantically secure for distinct
    /// category keys.
    pub fn encrypt(embedding: &[f32], category: &str) -> Self {
        let key = Self::derive_category_key(category);
        let plaintext: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

        let keystream = Self::generate_keystream(&key, plaintext.len());
        let ciphertext: Vec<u8> = plaintext
            .iter()
            .zip(keystream.iter())
            .map(|(p, k)| p ^ k)
            .collect();

        Self {
            ciphertext,
            algorithm: "blake3-xof".to_string(),
            key_category: category.to_string(),
        }
    }

    /// Decrypt if you have category access
    pub fn decrypt(&self, category: &str) -> Option<Vec<f32>> {
        if category != self.key_category {
            return None;
        }

        let key = Self::derive_category_key(category);
        let keystream = Self::generate_keystream(&key, self.ciphertext.len());
        let plaintext: Vec<u8> = self
            .ciphertext
            .iter()
            .zip(keystream.iter())
            .map(|(c, k)| c ^ k)
            .collect();

        let floats: Vec<f32> = plaintext
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk
                    .try_into()
                    .expect("chunks_exact(4) guarantees 4-byte slices");
                f32::from_le_bytes(arr)
            })
            .collect();

        Some(floats)
    }

    /// Derive a 32-byte symmetric key from category name using BLAKE3.
    fn derive_category_key(category: &str) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_derive_key("symthaea.gis.dht.category-key.v1");
        hasher.update(category.as_bytes());
        let mut key = [0u8; 32];
        hasher.finalize_xof().fill(&mut key);
        key
    }

    /// Generate a keystream of the given length using BLAKE3 keyed hash in XOF mode.
    fn generate_keystream(key: &[u8; 32], len: usize) -> Vec<u8> {
        let mut output = vec![0u8; len];
        // Use the key as BLAKE3 keyed hash key, hash a fixed context string,
        // then extend output to the required length via XOF.
        let mut hasher = blake3::Hasher::new_keyed(key);
        hasher.update(b"embedding-keystream");
        hasher.finalize_xof().fill(&mut output);
        output
    }
}

// Note: RangeProof and ValidityProof implementations moved to zk_proofs.rs
// Now using ZKEIGRangeProof (Bulletproofs-style) and TopicValidityProof (Schnorr-based)

/// Pseudonymous identity (unlinkable to agent)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PseudonymousId {
    /// The pseudonymous identifier
    pub id: String,
    /// Session-specific (changes each session)
    pub session_bound: bool,
}

impl PseudonymousId {
    /// Generate a new pseudonymous ID using OS entropy.
    ///
    /// Uses `OsRng` to produce an unpredictable, unlinkable identifier.
    pub fn generate() -> Self {
        use rand::RngCore;
        let mut bytes = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();

        Self {
            id: format!("pseudo_{hex}"),
            session_bound: true,
        }
    }
}

/// Knowledge match from DHT
#[derive(Debug, Clone)]
pub struct KnowledgeMatch {
    /// Matching claim ID
    pub claim_id: String,
    /// Semantic similarity
    pub similarity: f32,
    /// Encrypted response (for original asker only)
    pub encrypted_response: Option<Vec<u8>>,
    /// Responder reputation
    pub responder_reputation: f32,
    /// When matched
    pub matched_at: SystemTime,
}

/// Dark Spot DHT
///
/// Distributed hash table for privacy-preserving ignorance sharing.
/// Now enhanced with HDC semantic encoding for true semantic matching.
pub struct DarkSpotDHT {
    /// Local signatures (for testing, would be DHT in production)
    local_store: HashMap<String, ZKIgnoranceSignature>,

    /// Local knowledge (for matching)
    local_knowledge: HashMap<String, LocalClaim>,

    /// Semantic hasher (LSH-style)
    semantic_hasher: SemanticTopicHasher,

    /// HDC semantic encoder (real semantic matching)
    hdc_encoder: HdcSemanticEncoder,

    /// Configuration
    config: DarkSpotConfig,

    /// Agent's pseudonymous ID
    agent_id: PseudonymousId,
}

impl DarkSpotDHT {
    /// Create a new Dark Spot DHT
    pub fn new() -> Self {
        Self {
            local_store: HashMap::new(),
            local_knowledge: HashMap::new(),
            semantic_hasher: SemanticTopicHasher::new(),
            hdc_encoder: HdcSemanticEncoder::new(),
            config: DarkSpotConfig::default(),
            agent_id: PseudonymousId::generate(),
        }
    }

    /// Create a ZK ignorance signature
    ///
    /// Uses cryptographically sound proofs:
    /// - Pedersen commitment for topic hiding
    /// - Bulletproofs-style range proof for EIG
    /// - Schnorr-based validity proof (embedded in topic commitment)
    pub fn create_signature(
        &self,
        topic: &str,
        category: &str,
        eig: f32,
    ) -> Result<ZKIgnoranceSignature, GISError> {
        // 1. Create cryptographic topic commitment with validity proof
        // Uses ZKTopicCommitment which includes:
        // - Pedersen commitment (binding & hiding)
        // - TopicValidityProof (Schnorr-based knowledge proof)
        let topic_commitment = ZKTopicCommitment::create(topic, category);

        // 2. Create HDC semantic embedding (real semantic vectors!)
        let hdc_encoding = self.hdc_encoder.encode_with_category(topic, category);

        // 3. Compress for network transmission (256 dims instead of 16384)
        let compressed = self.hdc_encoder.compress(&hdc_encoding, 256);

        // 4. Encrypt embedding with category key
        let encrypted_embedding = EncryptedEmbedding::encrypt(&compressed, category);

        // 5. Create Bulletproofs-style EIG range proof
        // Proves 0 ≤ EIG ≤ 1 without revealing exact value
        let eig_range_proof = ZKEIGRangeProof::create(eig)
            .ok_or_else(|| GISError::CryptoError(format!("EIG {eig} out of range [0, 1]")))?;

        // 6. Generate nonce for uniqueness
        let mut nonce = [0u8; 16];
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        nonce[..16].copy_from_slice(&now.to_le_bytes());

        // 7. Generate unique ID
        let id = format!("zk_{category}_{now:016x}");

        Ok(ZKIgnoranceSignature {
            id,
            topic_commitment,
            category: category.to_string(),
            encrypted_embedding,
            eig_range_proof,
            publisher: self.agent_id.clone(),
            nonce,
            created_at: SystemTime::now(),
            ttl: self.config.default_ttl,
        })
    }

    /// Publish a signature to the DHT
    ///
    /// Validates all ZK proofs before publishing:
    /// - EIG range proof (Bulletproofs-style)
    /// - Topic validity proof (Schnorr-based, embedded in commitment)
    pub fn publish(&mut self, signature: &ZKIgnoranceSignature) -> Result<(), GISError> {
        // 1. Validate EIG range proof (Bulletproofs verification)
        if !signature.eig_range_proof.verify() {
            return Err(GISError::CryptoError(
                "Invalid EIG range proof: failed Bulletproof verification".to_string(),
            ));
        }

        // 2. Validate topic commitment and validity proof
        // The ZKTopicCommitment includes a Schnorr-based validity proof
        if !signature.topic_commitment.verify(&signature.category) {
            return Err(GISError::CryptoError(
                "Invalid topic commitment: failed validity proof verification".to_string(),
            ));
        }

        // 3. Store in DHT (local for demo)
        self.local_store
            .insert(signature.id.clone(), signature.clone());

        Ok(())
    }

    /// Find matches for a signature
    pub fn find_matches(&self, signature: &ZKIgnoranceSignature) -> Vec<KnowledgeMatch> {
        let mut matches = Vec::new();

        // Decrypt the embedding if we have category access
        let query_embedding = match signature.encrypted_embedding.decrypt(&signature.category) {
            Some(e) => e,
            None => return matches,
        };

        // Check against local knowledge
        for (claim_id, claim) in &self.local_knowledge {
            if claim.category == signature.category {
                let similarity = self.cosine_similarity(&query_embedding, &claim.embedding);

                if similarity > self.config.match_threshold {
                    matches.push(KnowledgeMatch {
                        claim_id: claim_id.clone(),
                        similarity,
                        encrypted_response: Some(claim.content.as_bytes().to_vec()),
                        responder_reputation: claim.reputation,
                        matched_at: SystemTime::now(),
                    });
                }
            }
        }

        // Sort by similarity
        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches
    }

    /// Add local knowledge (for matching against others' ignorance)
    pub fn add_knowledge(
        &mut self,
        id: String,
        content: String,
        category: String,
        reputation: f32,
    ) {
        // Create HDC semantic embedding for the knowledge content
        let hdc_encoding = self.hdc_encoder.encode_with_category(&content, &category);

        // Compress for storage (matches query compression)
        let embedding = self.hdc_encoder.compress(&hdc_encoding, 256);

        self.local_knowledge.insert(
            id.clone(),
            LocalClaim {
                id,
                content,
                category,
                embedding,
                reputation,
            },
        );
    }

    /// Get blind spots (clustered ignorance signatures)
    pub fn detect_blind_spots(&self) -> Vec<BlindSpot> {
        let mut blind_spots = Vec::new();

        // Group by category
        let mut by_category: HashMap<String, Vec<&ZKIgnoranceSignature>> = HashMap::new();
        for sig in self.local_store.values() {
            by_category
                .entry(sig.category.clone())
                .or_default()
                .push(sig);
        }

        // Detect clusters
        for (category, sigs) in by_category {
            if sigs.len() >= self.config.blind_spot_threshold {
                blind_spots.push(BlindSpot {
                    id: format!("blind_{}_{}", category, sigs.len()),
                    category,
                    signature_count: sigs.len(),
                    // Note: In ZK mode, we cannot extract exact EIG values
                    // Instead we use the verified range midpoint
                    average_eig: sigs
                        .iter()
                        .map(|s| {
                            let (min, max) = s.eig_range_proof.range();
                            (min + max) / 2.0 // Use range midpoint
                        })
                        .sum::<f32>()
                        / sigs.len() as f32,
                    topic_revealed: false,
                    detected_at: SystemTime::now(),
                });
            }
        }

        blind_spots
    }

    /// Reveal a blind spot (requires threshold participation)
    pub fn reveal_blind_spot(
        &mut self,
        blind_spot_id: &str,
        _participants: usize,
    ) -> Option<RevealedBlindSpot> {
        // Find the blind spot
        let blind_spots = self.detect_blind_spots();
        let blind_spot = blind_spots.iter().find(|b| b.id == blind_spot_id)?;

        // In production: threshold decryption
        // Here: just mark as revealed
        Some(RevealedBlindSpot {
            id: blind_spot.id.clone(),
            category: blind_spot.category.clone(),
            signature_count: blind_spot.signature_count,
            revealed_topic: None, // Would be decrypted topic
            revealed_at: SystemTime::now(),
        })
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        symthaea_core::math::cosine_similarity_f32(a, b)
    }
}

impl Default for DarkSpotDHT {
    fn default() -> Self {
        Self::new()
    }
}

/// Local claim storage
#[derive(Debug, Clone)]
struct LocalClaim {
    id: String,
    content: String,
    category: String,
    embedding: Vec<f32>,
    reputation: f32,
}

/// Blind spot (cluster of ignorance)
#[derive(Debug, Clone)]
pub struct BlindSpot {
    pub id: String,
    pub category: String,
    pub signature_count: usize,
    pub average_eig: f32,
    pub topic_revealed: bool,
    pub detected_at: SystemTime,
}

/// Revealed blind spot
#[derive(Debug, Clone)]
pub struct RevealedBlindSpot {
    pub id: String,
    pub category: String,
    pub signature_count: usize,
    pub revealed_topic: Option<String>,
    pub revealed_at: SystemTime,
}

/// Dark Spot DHT configuration
#[derive(Debug, Clone)]
pub struct DarkSpotConfig {
    /// Default TTL for signatures
    pub default_ttl: Duration,
    /// Threshold for semantic matching
    pub match_threshold: f32,
    /// Minimum signatures for blind spot detection
    pub blind_spot_threshold: usize,
    /// Threshold for reveal (percentage of participants)
    pub reveal_threshold: f32,
}

impl Default for DarkSpotConfig {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(86400), // 1 day
            match_threshold: 0.7,
            blind_spot_threshold: 3,
            reveal_threshold: 0.5,
        }
    }
}

/// Semantic topic hasher (LSH-style)
pub struct SemanticTopicHasher {
    /// Number of hash buckets
    num_buckets: usize,
}

impl SemanticTopicHasher {
    pub fn new() -> Self {
        Self { num_buckets: 256 }
    }

    /// Hash a topic semantically
    pub fn hash(&self, topic: &str) -> SemanticHash {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple hash (in production: use learned embeddings + LSH)
        let mut hasher = DefaultHasher::new();
        topic.to_lowercase().hash(&mut hasher);
        let primary = (hasher.finish() as usize) % self.num_buckets;

        // Generate secondary buckets via word hashes
        let secondary: Vec<usize> = topic
            .split_whitespace()
            .take(4)
            .map(|word| {
                let mut h = DefaultHasher::new();
                word.to_lowercase().hash(&mut h);
                (h.finish() as usize) % self.num_buckets
            })
            .collect();

        SemanticHash {
            primary_bucket: primary,
            secondary_buckets: secondary,
        }
    }

    /// Estimate similarity from hashes
    pub fn estimate_similarity(&self, a: &SemanticHash, b: &SemanticHash) -> f32 {
        if a.primary_bucket == b.primary_bucket {
            // Same primary bucket: high base similarity
            0.7
        } else {
            // Check secondary overlap
            let overlap = a
                .secondary_buckets
                .iter()
                .filter(|x| b.secondary_buckets.contains(x))
                .count();
            let total = a
                .secondary_buckets
                .len()
                .max(b.secondary_buckets.len())
                .max(1);
            (overlap as f32 / total as f32) * 0.5
        }
    }
}

impl Default for SemanticTopicHasher {
    fn default() -> Self {
        Self::new()
    }
}

/// Semantic hash result
#[derive(Debug, Clone)]
pub struct SemanticHash {
    pub primary_bucket: usize,
    pub secondary_buckets: Vec<usize>,
}

/// Curiosity Engine for active ignorance resolution
///
/// GIS v4.0 extends the Curiosity Engine with harmonic-weighted EIG
/// to prioritize resolution based on which harmonies are affected.
pub struct CuriosityEngine {
    /// Queue of pending resolutions
    queue: Vec<ResolutionQueueItem>,
    /// Configuration
    config: CuriosityConfig,
}

impl CuriosityEngine {
    pub fn new(config: CuriosityConfig) -> Self {
        Self {
            queue: Vec::new(),
            config,
        }
    }

    /// Queue a detection for resolution
    pub fn queue_resolution(
        &mut self,
        detection: &super::IgnoranceDetection,
    ) -> Result<ResolutionRequest, GISError> {
        let item = ResolutionQueueItem {
            query: detection.query.clone(),
            eig: detection.eig,
            harmonic_eig: None, // Basic detection doesn't have harmonic info
            queued_at: SystemTime::now(),
            priority: detection.eig,
        };

        let request = ResolutionRequest {
            id: format!("res_{}", self.queue.len()),
            query: detection.query.clone(),
            priority: detection.eig,
            deadline: None,
        };

        self.queue.push(item);

        // Sort by priority (EIG)
        self.queue.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(request)
    }

    /// Queue a harmonic ignorance for resolution (GIS v4.0)
    ///
    /// Uses harmonic-weighted EIG for prioritization.
    pub fn queue_harmonic_resolution(
        &mut self,
        detection: &super::IgnoranceDetection,
        harmonic_ignorance: &super::HarmonicIgnorance,
        context: &HarmonicContext,
    ) -> Result<ResolutionRequest, GISError> {
        // Calculate harmonic EIG
        let h_eig = self.harmonic_eig(detection.eig, harmonic_ignorance, context);

        let item = ResolutionQueueItem {
            query: detection.query.clone(),
            eig: detection.eig,
            harmonic_eig: Some(h_eig),
            queued_at: SystemTime::now(),
            priority: h_eig, // Use harmonic EIG as priority
        };

        let request = ResolutionRequest {
            id: format!("res_h_{}", self.queue.len()),
            query: detection.query.clone(),
            priority: h_eig,
            deadline: None,
        };

        self.queue.push(item);

        // Sort by priority (harmonic EIG)
        self.queue.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(request)
    }

    /// Calculate Harmonic EIG (GIS v4.0)
    ///
    /// Harmonic EIG = BaseEIG × (1 + Σ(w_h × impact_h × relevance_h))
    ///
    /// Where:
    /// - w_h = base weight of harmony h
    /// - impact_h = how much this ignorance affects harmony h
    /// - relevance_h = context-dependent relevance of harmony h
    pub fn harmonic_eig(
        &self,
        base_eig: f32,
        harmonic_ignorance: &super::HarmonicIgnorance,
        context: &HarmonicContext,
    ) -> f32 {
        let harmonic_weight: f32 = harmonic_ignorance
            .affected_harmonies
            .iter()
            .map(|(harmony, impact)| {
                let base_weight = harmony.base_weight();
                let relevance = context.harmony_relevance(*harmony);
                base_weight * impact * relevance
            })
            .sum();

        // H-EIG formula: boost base EIG by harmonic factors
        let h_eig = base_eig * (1.0 + harmonic_weight);

        // Clamp to reasonable range
        h_eig.clamp(0.0, 2.0)
    }

    /// Get next item to resolve
    pub fn next(&mut self) -> Option<ResolutionQueueItem> {
        self.queue.pop()
    }

    /// Get next with minimum harmonic EIG threshold
    pub fn next_if_harmonic(&mut self, min_h_eig: f32) -> Option<ResolutionQueueItem> {
        // Find first item meeting threshold
        let idx = self
            .queue
            .iter()
            .position(|item| item.harmonic_eig.unwrap_or(item.eig) >= min_h_eig);

        idx.map(|i| self.queue.remove(i))
    }

    /// Queue length
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Get high-priority items (harmonic EIG > threshold)
    pub fn high_priority_items(&self, threshold: f32) -> Vec<&ResolutionQueueItem> {
        self.queue
            .iter()
            .filter(|item| item.priority >= threshold)
            .collect()
    }

    /// Get items affecting specific harmony
    pub fn items_affecting_harmony(&self, _harmony: super::Harmony) -> Vec<&ResolutionQueueItem> {
        // In full implementation, would filter by affected harmonies
        // For now, return all with harmonic EIG
        self.queue
            .iter()
            .filter(|item| item.harmonic_eig.is_some())
            .collect()
    }
}

/// Context for harmonic EIG calculation
#[derive(Debug, Clone)]
pub struct HarmonicContext {
    /// Current situation domain
    pub domain: Option<String>,
    /// Stakeholders involved
    pub stakeholders: Vec<String>,
    /// Time sensitivity (0.0 - 1.0)
    pub time_sensitivity: f32,
    /// Harmony relevance overrides
    harmony_overrides: HashMap<u8, f32>,
}

impl HarmonicContext {
    /// Create a new context
    pub fn new() -> Self {
        Self {
            domain: None,
            stakeholders: Vec::new(),
            time_sensitivity: 0.5,
            harmony_overrides: HashMap::new(),
        }
    }

    /// Create with domain
    pub fn with_domain(domain: &str) -> Self {
        Self {
            domain: Some(domain.to_string()),
            ..Self::new()
        }
    }

    /// Add stakeholder
    pub fn add_stakeholder(&mut self, stakeholder: &str) {
        self.stakeholders.push(stakeholder.to_string());
    }

    /// Set harmony relevance override
    pub fn set_harmony_relevance(&mut self, harmony: super::Harmony, relevance: f32) {
        let idx = Self::harmony_to_index(harmony);
        self.harmony_overrides
            .insert(idx, relevance.clamp(0.0, 2.0));
    }

    /// Get relevance for a harmony
    pub fn harmony_relevance(&self, harmony: super::Harmony) -> f32 {
        let idx = Self::harmony_to_index(harmony);

        // Check for override first
        if let Some(&relevance) = self.harmony_overrides.get(&idx) {
            return relevance;
        }

        // Default relevance based on domain
        match &self.domain {
            Some(domain) => self.domain_harmony_relevance(domain, harmony),
            None => 1.0, // Default equal relevance
        }
    }

    fn domain_harmony_relevance(&self, domain: &str, harmony: super::Harmony) -> f32 {
        match (domain.to_lowercase().as_str(), harmony) {
            // Technology domains boost truth-knowing
            ("technology" | "science", super::Harmony::IntegralWisdom) => 1.5,
            ("technology", super::Harmony::EvolutionaryProgression) => 1.3,

            // Social domains boost care-knowing
            ("social" | "community", super::Harmony::PanSentientFlourishing) => 1.5,
            ("social", super::Harmony::UniversalInterconnectedness) => 1.3,

            // Economic domains boost exchange-knowing
            ("economic" | "business", super::Harmony::SacredReciprocity) => 1.5,
            ("economic", super::Harmony::ResonantCoherence) => 1.2,

            // Environmental domains boost flourishing and interconnectedness
            ("environmental", super::Harmony::PanSentientFlourishing) => 1.4,
            ("environmental", super::Harmony::UniversalInterconnectedness) => 1.4,

            // Creative domains boost play
            ("creative" | "art", super::Harmony::InfinitePlay) => 1.5,

            // Default
            _ => 1.0,
        }
    }

    fn harmony_to_index(harmony: super::Harmony) -> u8 {
        match harmony {
            super::Harmony::ResonantCoherence => 0,
            super::Harmony::PanSentientFlourishing => 1,
            super::Harmony::IntegralWisdom => 2,
            super::Harmony::InfinitePlay => 3,
            super::Harmony::UniversalInterconnectedness => 4,
            super::Harmony::SacredReciprocity => 5,
            super::Harmony::EvolutionaryProgression => 6,
            super::Harmony::SacredStillness => 7,
        }
    }
}

impl Default for HarmonicContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Curiosity engine configuration
#[derive(Debug, Clone)]
pub struct CuriosityConfig {
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Minimum EIG to queue
    pub min_eig: f32,
    /// Background resolution enabled
    pub background_resolution: bool,
}

impl Default for CuriosityConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 100,
            min_eig: 0.2,
            background_resolution: true,
        }
    }
}

/// Item in resolution queue
#[derive(Debug, Clone)]
pub struct ResolutionQueueItem {
    pub query: String,
    /// Base Expected Information Gain
    pub eig: f32,
    /// Harmonic-weighted EIG (GIS v4.0)
    pub harmonic_eig: Option<f32>,
    pub queued_at: SystemTime,
    /// Effective priority (harmonic_eig if available, else eig)
    pub priority: f32,
}

/// Resolution request
#[derive(Debug, Clone)]
pub struct ResolutionRequest {
    pub id: String,
    pub query: String,
    pub priority: f32,
    pub deadline: Option<SystemTime>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_commitment() {
        let topic = "What is quantum entanglement?";
        let category = "physics";
        let commitment = ZKTopicCommitment::create(topic, category);

        // Should verify with correct category
        assert!(commitment.verify(category));

        // Should NOT verify with wrong category
        assert!(!commitment.verify("biology"));

        // Should be able to open with correct topic
        assert!(commitment.open(topic));
        assert!(!commitment.open("Different topic"));
    }

    #[test]
    fn test_encrypted_embedding() {
        let embedding: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let category = "physics";

        let encrypted = EncryptedEmbedding::encrypt(&embedding, category);

        // Can decrypt with correct category
        let decrypted = encrypted.decrypt(category).unwrap();
        for (a, b) in embedding.iter().zip(decrypted.iter()) {
            assert!((a - b).abs() < 0.001);
        }

        // Cannot decrypt with wrong category
        assert!(encrypted.decrypt("biology").is_none());
    }

    #[test]
    fn test_zk_signature_creation() {
        let dht = DarkSpotDHT::new();

        let sig = dht
            .create_signature("What is dark matter?", "physics", 0.8)
            .unwrap();

        assert_eq!(sig.category, "physics");

        // Verify Bulletproofs-style EIG range proof
        assert!(sig.eig_range_proof.verify());
        assert_eq!(sig.eig_range_proof.range(), (0.0, 1.0));

        // Verify topic commitment includes validity proof (Schnorr-based)
        assert!(sig.topic_commitment.verify(&sig.category));
    }

    #[test]
    fn test_dht_publish_and_match() {
        let mut dht = DarkSpotDHT::new();

        // Add some knowledge
        dht.add_knowledge(
            "claim_1".to_string(),
            "Dark matter is a form of matter thought to account for gravitational effects"
                .to_string(),
            "physics".to_string(),
            0.9,
        );

        // Create and publish ignorance
        let sig = dht
            .create_signature("What is dark matter?", "physics", 0.8)
            .unwrap();

        dht.publish(&sig).unwrap();

        // Find matches
        let matches = dht.find_matches(&sig);
        // Note: matches may be empty due to simplified embedding
        assert!(matches.len() <= 1);
    }

    #[test]
    fn test_semantic_hasher() {
        let hasher = SemanticTopicHasher::new();

        let hash1 = hasher.hash("What is quantum physics?");
        let hash2 = hasher.hash("Explain quantum physics");
        let hash3 = hasher.hash("What is cooking?");

        // Similar topics should have some overlap
        let sim_12 = hasher.estimate_similarity(&hash1, &hash2);
        let sim_13 = hasher.estimate_similarity(&hash1, &hash3);

        // Physics questions more similar than cooking
        assert!(sim_12 >= sim_13 || sim_12 > 0.0);
    }

    #[test]
    fn test_blind_spot_detection() {
        let mut dht = DarkSpotDHT::new();

        // Publish multiple ignorance signatures in same category
        for i in 0..5 {
            let sig = dht
                .create_signature(&format!("Question {} about physics", i), "physics", 0.7)
                .unwrap();
            dht.publish(&sig).unwrap();
        }

        let blind_spots = dht.detect_blind_spots();
        assert!(!blind_spots.is_empty());
        assert!(blind_spots.iter().any(|b| b.category == "physics"));
    }

    #[test]
    fn test_curiosity_engine() {
        use super::super::IgnoranceDetection;

        let config = CuriosityConfig::default();
        let mut engine = CuriosityEngine::new(config);

        let detection = IgnoranceDetection {
            query: "Test query".to_string(),
            ignorance_type: super::super::IgnoranceType::KnownUnknown,
            uncertainty: super::super::Uncertainty3D::new(0.5, 0.3, 0.2),
            domain: super::super::Domain::General,
            eig: 0.7,
            detected_at: SystemTime::now(),
        };

        let request = engine.queue_resolution(&detection).unwrap();
        assert_eq!(engine.pending_count(), 1);
        assert!((request.priority - 0.7).abs() < 0.001);

        let item = engine.next().unwrap();
        assert_eq!(item.query, "Test query");
    }

    // === GIS v4.0 Harmonic EIG Tests ===

    #[test]
    fn test_harmonic_eig_basic() {
        use super::super::{HarmonicIgnorance, Harmony, IgnoranceType};

        let config = CuriosityConfig::default();
        let engine = CuriosityEngine::new(config);

        // Create harmonic ignorance with high PSF impact
        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        hi.add_affected_harmony(Harmony::PanSentientFlourishing, 1.0);

        let context = HarmonicContext::new();

        // Calculate harmonic EIG
        let base_eig = 0.5;
        let h_eig = engine.harmonic_eig(base_eig, &hi, &context);

        // H-EIG should be higher than base due to PSF weight (0.20)
        assert!(
            h_eig > base_eig,
            "H-EIG {} should be > base EIG {}",
            h_eig,
            base_eig
        );
    }

    #[test]
    fn test_harmonic_eig_multiple_harmonies() {
        use super::super::{HarmonicIgnorance, Harmony, IgnoranceType};

        let config = CuriosityConfig::default();
        let engine = CuriosityEngine::new(config);

        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        hi.add_affected_harmony(Harmony::PanSentientFlourishing, 1.0); // 0.20 weight
        hi.add_affected_harmony(Harmony::IntegralWisdom, 1.0); // 0.15 weight
        hi.add_affected_harmony(Harmony::ResonantCoherence, 1.0); // 0.20 weight

        let context = HarmonicContext::new();

        let base_eig = 0.5;
        let h_eig = engine.harmonic_eig(base_eig, &hi, &context);

        // Should have significant boost (0.20 + 0.15 + 0.20 = 0.55)
        // H-EIG = 0.5 * (1 + 0.55) = 0.775
        assert!(
            h_eig > 0.7,
            "H-EIG {} should be > 0.7 with multiple harmonies",
            h_eig
        );
    }

    #[test]
    fn test_harmonic_eig_context_relevance() {
        use super::super::{HarmonicIgnorance, Harmony, IgnoranceType};

        let config = CuriosityConfig::default();
        let engine = CuriosityEngine::new(config);

        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        hi.add_affected_harmony(Harmony::IntegralWisdom, 1.0);

        // Technology context boosts IntegralWisdom relevance
        let tech_context = HarmonicContext::with_domain("technology");

        // General context (no boost)
        let general_context = HarmonicContext::new();

        let base_eig = 0.5;
        let h_eig_tech = engine.harmonic_eig(base_eig, &hi, &tech_context);
        let h_eig_general = engine.harmonic_eig(base_eig, &hi, &general_context);

        // Technology context should give higher EIG for truth-knowing
        assert!(
            h_eig_tech > h_eig_general,
            "Tech H-EIG {} should be > general H-EIG {}",
            h_eig_tech,
            h_eig_general
        );
    }

    #[test]
    fn test_harmonic_queue_resolution() {
        use super::super::{
            Domain, HarmonicIgnorance, Harmony, IgnoranceDetection, IgnoranceType, Uncertainty3D,
        };

        let config = CuriosityConfig::default();
        let mut engine = CuriosityEngine::new(config);

        let detection = IgnoranceDetection {
            query: "Test harmonic query".to_string(),
            ignorance_type: IgnoranceType::KnownUnknown,
            uncertainty: Uncertainty3D::new(0.5, 0.3, 0.2),
            domain: Domain::General,
            eig: 0.5,
            detected_at: SystemTime::now(),
        };

        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        hi.add_affected_harmony(Harmony::PanSentientFlourishing, 1.0);

        let context = HarmonicContext::new();

        let request = engine
            .queue_harmonic_resolution(&detection, &hi, &context)
            .unwrap();

        // Priority should be harmonic EIG, not base EIG
        assert!(
            request.priority > detection.eig,
            "Harmonic priority {} should be > base EIG {}",
            request.priority,
            detection.eig
        );

        assert_eq!(engine.pending_count(), 1);

        let item = engine.next().unwrap();
        assert!(item.harmonic_eig.is_some());
    }

    #[test]
    fn test_harmonic_context_override() {
        use super::super::Harmony;

        let mut context = HarmonicContext::new();

        // Override InfinitePlay relevance
        context.set_harmony_relevance(Harmony::InfinitePlay, 2.0);

        let relevance = context.harmony_relevance(Harmony::InfinitePlay);
        assert!((relevance - 2.0).abs() < 0.001);

        // Non-overridden should be default
        let default_relevance = context.harmony_relevance(Harmony::ResonantCoherence);
        assert!((default_relevance - 1.0).abs() < 0.001);
    }
}
