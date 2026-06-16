// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sovereign Name Resolution — Decentralized DNS Alternative
//!
//! Resolves human-readable mesh names (e.g., `mycelix://joburg/water/tank-7`)
//! to network endpoints without centralized DNS. Resolution chain:
//! cache → mesh broadcast → DHT.
//!
//! # Name Format
//! `mycelix://{segment1}/{segment2}/.../{segmentN}`
//! - Lowercase alphanumeric + hyphens
//! - Max depth: 5 segments
//! - Max segment length: 63 characters

use std::collections::HashMap;

/// Maximum name depth (number of segments).
pub const MAX_NAME_DEPTH: usize = 5;

/// Maximum segment length in characters.
pub const MAX_SEGMENT_LEN: usize = 63;

/// Default name cache capacity.
pub const DEFAULT_CACHE_SIZE: usize = 256;

/// Name TTL in seconds (1 year default for DHT-registered names).
pub const DEFAULT_NAME_TTL_SECS: u64 = 365 * 24 * 3600;

/// Resolved endpoint types.
///
/// Represents the transport-level address that a mesh name resolves to.
/// New variants can be added as additional naming systems are integrated.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedEndpoint {
    /// Iroh node address (QUIC transport).
    IrohAddr(String),
    /// LoRa radio identifier.
    LoRaId([u8; 8]),
    /// Holochain agent public key.
    HolochainAgent(String),
    /// IP address (fallback for hybrid networks).
    IpAddr(String),
    /// ENS name (e.g., "mycelix.eth") — resolved via Ethereum Name Service.
    ///
    /// Integration point for decentralized naming. The string contains either
    /// the raw ENS name (for deferred resolution) or the resolved address.
    EnsName(String),
    /// Content-addressed resource (IPFS CID, IPLD link, etc.).
    ///
    /// Used for immutable content references resolved via decentralized storage.
    ContentAddress(String),
}

impl ResolvedEndpoint {
    /// String representation for logging.
    pub fn display(&self) -> String {
        match self {
            Self::IrohAddr(addr) => format!("iroh:{}", addr),
            Self::LoRaId(id) => {
                let hex_str: String = id.iter().map(|b| format!("{:02x}", b)).collect();
                format!("lora:{}", hex_str)
            }
            Self::HolochainAgent(key) => format!("hc:{}", &key[..8.min(key.len())]),
            Self::IpAddr(addr) => format!("ip:{}", addr),
            Self::EnsName(name) => format!("ens:{}", name),
            Self::ContentAddress(cid) => format!("cid:{}", &cid[..12.min(cid.len())]),
        }
    }
}

/// A parsed mesh name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MeshName {
    /// Name segments (e.g., ["joburg", "water", "tank-7"]).
    pub segments: Vec<String>,
}

impl MeshName {
    /// Parse a mesh name string.
    ///
    /// Accepts formats:
    /// - `mycelix://joburg/water/tank-7`
    /// - `joburg/water/tank-7` (shorthand)
    pub fn parse(name: &str) -> Result<Self, NameError> {
        let path = name
            .strip_prefix("mycelix://")
            .unwrap_or(name)
            .trim_matches('/');

        if path.is_empty() {
            return Err(NameError::Empty);
        }

        let segments: Vec<String> = path.split('/').map(String::from).collect();

        if segments.len() > MAX_NAME_DEPTH {
            return Err(NameError::TooDeep {
                depth: segments.len(),
                max: MAX_NAME_DEPTH,
            });
        }

        for seg in &segments {
            if seg.is_empty() {
                return Err(NameError::EmptySegment);
            }
            if seg.len() > MAX_SEGMENT_LEN {
                return Err(NameError::SegmentTooLong {
                    segment: seg.clone(),
                    len: seg.len(),
                    max: MAX_SEGMENT_LEN,
                });
            }
            if !seg
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
            {
                return Err(NameError::InvalidCharacters {
                    segment: seg.clone(),
                });
            }
            if seg.starts_with('-') || seg.ends_with('-') {
                return Err(NameError::InvalidCharacters {
                    segment: seg.clone(),
                });
            }
        }

        Ok(Self { segments })
    }

    /// Full canonical name string.
    pub fn to_canonical(&self) -> String {
        format!("mycelix://{}", self.segments.join("/"))
    }

    /// BLAKE3 hash of the canonical name (for mesh queries).
    pub fn hash(&self) -> [u8; 32] {
        *blake3::hash(self.to_canonical().as_bytes()).as_bytes()
    }

    /// Number of segments.
    pub fn depth(&self) -> usize {
        self.segments.len()
    }
}

impl std::fmt::Display for MeshName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_canonical())
    }
}

/// Name resolution errors.
#[derive(Debug, Clone)]
pub enum NameError {
    Empty,
    EmptySegment,
    TooDeep {
        depth: usize,
        max: usize,
    },
    SegmentTooLong {
        segment: String,
        len: usize,
        max: usize,
    },
    InvalidCharacters {
        segment: String,
    },
    NotFound {
        name: String,
    },
    Expired {
        name: String,
    },
}

impl std::fmt::Display for NameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "Empty name"),
            Self::EmptySegment => write!(f, "Empty segment in name"),
            Self::TooDeep { depth, max } => write!(f, "Name depth {} exceeds max {}", depth, max),
            Self::SegmentTooLong { segment, len, max } => {
                write!(
                    f,
                    "Segment '{}' length {} exceeds max {}",
                    segment, len, max
                )
            }
            Self::InvalidCharacters { segment } => {
                write!(f, "Invalid characters in segment '{}'", segment)
            }
            Self::NotFound { name } => write!(f, "Name '{}' not found", name),
            Self::Expired { name } => write!(f, "Name '{}' has expired", name),
        }
    }
}

/// A cached name resolution entry.
#[derive(Debug, Clone)]
struct CacheEntry {
    endpoint: ResolvedEndpoint,
    /// When this entry was resolved (Unix seconds).
    resolved_at: u64,
    /// TTL in seconds.
    ttl_secs: u64,
    /// Access count (for LRU).
    access_count: u64,
    /// Last access time (for LRU).
    last_access: u64,
}

impl CacheEntry {
    fn is_expired(&self, now_secs: u64) -> bool {
        now_secs.saturating_sub(self.resolved_at) > self.ttl_secs
    }
}

/// A registered name binding (for local registration).
#[derive(Debug, Clone)]
pub struct NameRegistration {
    /// The mesh name.
    pub name: MeshName,
    /// The resolved endpoint.
    pub endpoint: ResolvedEndpoint,
    /// Registration timestamp (Unix seconds).
    pub registered_at: u64,
    /// TTL in seconds.
    pub ttl_secs: u64,
    /// Owner's signer key (for update/transfer authorization).
    pub owner_key: String,
}

/// Sovereign name resolver with LRU cache.
pub struct NameResolver {
    /// LRU resolution cache.
    cache: HashMap<String, CacheEntry>,
    /// Maximum cache entries.
    cache_capacity: usize,
    /// Locally registered names (we are the authority).
    local_registrations: HashMap<String, NameRegistration>,
    /// Statistics.
    cache_hits: u64,
    cache_misses: u64,
}

impl NameResolver {
    /// Create a new NameResolver with the given cache capacity.
    pub fn new(cache_capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            cache_capacity: cache_capacity.max(16),
            local_registrations: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Resolve a name from cache.
    ///
    /// Returns `None` if not cached or expired.
    pub fn resolve_cached(&mut self, name: &MeshName, now_secs: u64) -> Option<ResolvedEndpoint> {
        let canonical = name.to_canonical();
        if let Some(entry) = self.cache.get_mut(&canonical) {
            if entry.is_expired(now_secs) {
                self.cache.remove(&canonical);
                self.cache_misses += 1;
                return None;
            }
            entry.access_count += 1;
            entry.last_access = now_secs;
            self.cache_hits += 1;
            return Some(entry.endpoint.clone());
        }
        self.cache_misses += 1;
        None
    }

    /// Insert a resolved name into the cache.
    pub fn cache_resolution(
        &mut self,
        name: &MeshName,
        endpoint: ResolvedEndpoint,
        ttl_secs: u64,
        now_secs: u64,
    ) {
        // Evict LRU if at capacity
        if self.cache.len() >= self.cache_capacity {
            if let Some(lru_key) = self
                .cache
                .iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(k, _)| k.clone())
            {
                self.cache.remove(&lru_key);
            }
        }

        self.cache.insert(
            name.to_canonical(),
            CacheEntry {
                endpoint,
                resolved_at: now_secs,
                ttl_secs,
                access_count: 1,
                last_access: now_secs,
            },
        );
    }

    /// Register a local name binding.
    pub fn register_local(
        &mut self,
        name: MeshName,
        endpoint: ResolvedEndpoint,
        owner_key: String,
        now_secs: u64,
    ) {
        let reg = NameRegistration {
            name: name.clone(),
            endpoint: endpoint.clone(),
            registered_at: now_secs,
            ttl_secs: DEFAULT_NAME_TTL_SECS,
            owner_key,
        };
        self.local_registrations.insert(name.to_canonical(), reg);
        // Also cache locally
        self.cache_resolution(&name, endpoint, DEFAULT_NAME_TTL_SECS, now_secs);
    }

    /// Check if we own a name locally.
    pub fn is_locally_registered(&self, name: &MeshName) -> bool {
        self.local_registrations.contains_key(&name.to_canonical())
    }

    /// Get a local registration.
    pub fn get_local_registration(&self, name: &MeshName) -> Option<&NameRegistration> {
        self.local_registrations.get(&name.to_canonical())
    }

    /// Cache hit ratio.
    pub fn hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Current cache size.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Local registration count.
    pub fn local_registration_count(&self) -> usize {
        self.local_registrations.len()
    }

    /// Expire stale cache entries.
    pub fn expire_cache(&mut self, now_secs: u64) {
        self.cache.retain(|_, entry| !entry.is_expired(now_secs));
    }

    /// Attempt external name resolution (ENS, Handshake, etc.).
    ///
    /// This is the **4th tier** in the resolution chain:
    /// 1. Cache lookup (instant)
    /// 2. Mesh broadcast (LAN peers)
    /// 3. DHT query (Holochain)
    /// 4. External decentralized naming (ENS, HNS) — **this method**
    ///
    /// Currently returns `None` — wiring to an Ethereum provider or HNS
    /// resolver is out of scope. This establishes the integration point
    /// for future ENS/Handshake resolution.
    ///
    /// When implemented, this should:
    /// - Check if the name ends in `.eth` → resolve via ENS
    /// - Check if the name ends in a Handshake TLD → resolve via HNS
    /// - Cache successful resolutions with appropriate TTL
    pub fn resolve_external(
        &mut self,
        _name: &MeshName,
        _now_secs: u64,
    ) -> Option<ResolvedEndpoint> {
        // Integration point for ENS/Handshake resolution.
        // Requires an Ethereum provider dependency (ethers-rs or alloy).
        None
    }
}

impl Default for NameResolver {
    fn default() -> Self {
        Self::new(DEFAULT_CACHE_SIZE)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_full_uri() {
        let name = MeshName::parse("mycelix://joburg/water/tank-7").unwrap();
        assert_eq!(name.segments, vec!["joburg", "water", "tank-7"]);
        assert_eq!(name.depth(), 3);
    }

    #[test]
    fn test_parse_shorthand() {
        let name = MeshName::parse("joburg/water").unwrap();
        assert_eq!(name.segments, vec!["joburg", "water"]);
    }

    #[test]
    fn test_parse_single_segment() {
        let name = MeshName::parse("node-alpha").unwrap();
        assert_eq!(name.segments, vec!["node-alpha"]);
    }

    #[test]
    fn test_canonical_roundtrip() {
        let name = MeshName::parse("a/b/c").unwrap();
        assert_eq!(name.to_canonical(), "mycelix://a/b/c");
    }

    #[test]
    fn test_reject_empty() {
        assert!(MeshName::parse("").is_err());
    }

    #[test]
    fn test_reject_too_deep() {
        assert!(MeshName::parse("a/b/c/d/e/f").is_err()); // 6 > MAX_NAME_DEPTH
    }

    #[test]
    fn test_reject_uppercase() {
        assert!(MeshName::parse("Hello/World").is_err());
    }

    #[test]
    fn test_reject_leading_hyphen() {
        assert!(MeshName::parse("-bad").is_err());
    }

    #[test]
    fn test_reject_trailing_hyphen() {
        assert!(MeshName::parse("bad-").is_err());
    }

    #[test]
    fn test_reject_long_segment() {
        let long = "a".repeat(64);
        assert!(MeshName::parse(&long).is_err());
    }

    #[test]
    fn test_hash_deterministic() {
        let a = MeshName::parse("test/name").unwrap();
        let b = MeshName::parse("test/name").unwrap();
        assert_eq!(a.hash(), b.hash());
    }

    #[test]
    fn test_cache_hit() {
        let mut resolver = NameResolver::new(16);
        let name = MeshName::parse("test/node").unwrap();
        let endpoint = ResolvedEndpoint::IrohAddr("abc123".to_string());
        resolver.cache_resolution(&name, endpoint.clone(), 3600, 1000);
        let resolved = resolver.resolve_cached(&name, 1000);
        assert_eq!(resolved, Some(endpoint));
    }

    #[test]
    fn test_cache_expiry() {
        let mut resolver = NameResolver::new(16);
        let name = MeshName::parse("test/node").unwrap();
        resolver.cache_resolution(
            &name,
            ResolvedEndpoint::IpAddr("1.2.3.4".to_string()),
            60,
            1000,
        );
        // Resolve after TTL
        assert!(resolver.resolve_cached(&name, 1061).is_none());
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut resolver = NameResolver::new(16); // minimum is 16
        // Fill cache to capacity
        for i in 0..16 {
            let n = MeshName::parse(&format!("node-{}", i)).unwrap();
            resolver.cache_resolution(
                &n,
                ResolvedEndpoint::IpAddr(format!("{}", i)),
                3600,
                100 + i as u64,
            );
        }
        // node-0 is LRU (last_access=100), should be evicted when overflow is added
        let overflow = MeshName::parse("overflow").unwrap();
        resolver.cache_resolution(
            &overflow,
            ResolvedEndpoint::IpAddr("new".to_string()),
            3600,
            200,
        );
        let n0 = MeshName::parse("node-0").unwrap();
        assert!(resolver.resolve_cached(&n0, 200).is_none());
        assert!(resolver.resolve_cached(&overflow, 200).is_some());
    }

    #[test]
    fn test_local_registration() {
        let mut resolver = NameResolver::new(16);
        let name = MeshName::parse("my-node").unwrap();
        resolver.register_local(
            name.clone(),
            ResolvedEndpoint::LoRaId([1; 8]),
            "key".to_string(),
            0,
        );
        assert!(resolver.is_locally_registered(&name));
        assert!(resolver.resolve_cached(&name, 0).is_some());
    }

    #[test]
    fn test_hit_ratio() {
        let mut resolver = NameResolver::new(16);
        let name = MeshName::parse("test").unwrap();
        resolver.cache_resolution(&name, ResolvedEndpoint::IpAddr("x".to_string()), 3600, 0);
        resolver.resolve_cached(&name, 0); // hit
        let miss_name = MeshName::parse("nope").unwrap();
        resolver.resolve_cached(&miss_name, 0); // miss
        assert!((resolver.hit_ratio() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_endpoint_display() {
        assert!(
            ResolvedEndpoint::IrohAddr("abc".to_string())
                .display()
                .starts_with("iroh:")
        );
        assert!(
            ResolvedEndpoint::LoRaId([0; 8])
                .display()
                .starts_with("lora:")
        );
        assert!(
            ResolvedEndpoint::HolochainAgent("xyz".to_string())
                .display()
                .starts_with("hc:")
        );
        assert!(
            ResolvedEndpoint::IpAddr("1.2.3.4".to_string())
                .display()
                .starts_with("ip:")
        );
        assert!(
            ResolvedEndpoint::EnsName("mycelix.eth".to_string())
                .display()
                .starts_with("ens:")
        );
        assert!(
            ResolvedEndpoint::ContentAddress(
                "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG".to_string()
            )
            .display()
            .starts_with("cid:")
        );
    }

    #[test]
    fn test_default_resolver() {
        let resolver = NameResolver::default();
        assert_eq!(resolver.cache_size(), 0);
    }

    #[test]
    fn test_expire_cache() {
        let mut resolver = NameResolver::new(16);
        let name = MeshName::parse("old").unwrap();
        resolver.cache_resolution(&name, ResolvedEndpoint::IpAddr("x".to_string()), 60, 1000);
        resolver.expire_cache(1100);
        assert_eq!(resolver.cache_size(), 0);
    }

    #[test]
    fn test_ens_endpoint_cache() {
        let mut resolver = NameResolver::new(16);
        let name = MeshName::parse("defi/bridge").unwrap();
        let endpoint = ResolvedEndpoint::EnsName("mycelix.eth".to_string());
        resolver.cache_resolution(&name, endpoint.clone(), 3600, 1000);
        let resolved = resolver.resolve_cached(&name, 1000);
        assert_eq!(resolved, Some(endpoint));
    }

    #[test]
    fn test_content_address_endpoint() {
        let cid = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG".to_string();
        let endpoint = ResolvedEndpoint::ContentAddress(cid.clone());
        let display = endpoint.display();
        assert!(display.starts_with("cid:"));
        assert_eq!(display.len(), 4 + 12); // "cid:" + 12 chars truncated
    }

    #[test]
    fn test_resolve_external_stub() {
        let mut resolver = NameResolver::new(16);
        let name = MeshName::parse("test/external").unwrap();
        // Stub returns None — integration point for future ENS/HNS
        assert!(resolver.resolve_external(&name, 1000).is_none());
    }
}
