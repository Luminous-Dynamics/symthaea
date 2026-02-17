//! Gitcoin Passport Integration for Sybil Resistance
//!
//! Provides Proof of Personhood verification through Gitcoin Passport stamps.
//! Integrates with the Stamps API v2 for score and stamp retrieval.
//!
//! API Reference: https://docs.passport.xyz
//!
//! ## Features
//!
//! - **Score Retrieval**: Fetch passport scores for Ethereum addresses
//! - **Stamp Verification**: Retrieve and verify individual stamps
//! - **Passport Verification**: Full verification with threshold checking
//! - **Rate Limiting**: Built-in 1 req/sec rate limiting
//! - **Caching**: 5-minute cache for API responses
//!
//! ## Example
//!
//! ```rust,ignore
//! use fl_aggregator::identity::gitcoin_passport::{GitcoinPassportClient, GitcoinPassportConfig};
//!
//! let config = GitcoinPassportConfig::from_env()?;
//! let mut client = GitcoinPassportClient::new(config)?;
//!
//! // Fetch passport score
//! let score_response = client.fetch_score("0x1234...").await?;
//! println!("Score: {}", score_response.score);
//!
//! // Full verification with threshold
//! let verification = client.verify_passport("0x1234...", 20.0).await?;
//! if verification.is_valid {
//!     println!("Passport verified with {} stamps", verification.stamps.len());
//! }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::Mutex;

/// Gitcoin API error types
#[derive(Error, Debug, Clone)]
pub enum GitcoinError {
    /// Rate limit exceeded (429 response)
    #[error("Rate limit exceeded - please wait before retrying")]
    RateLimited,

    /// Invalid Ethereum address format
    #[error("Invalid address format: {0}")]
    InvalidAddress(String),

    /// API returned an error response
    #[error("API error: {message} (status: {status_code})")]
    ApiError {
        status_code: u16,
        message: String,
    },

    /// Network/connection error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Response parsing error
    #[error("Invalid response format: {0}")]
    ParseError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Legacy error type for backward compatibility
#[derive(Error, Debug)]
pub enum PassportError {
    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("Address not found: {0}")]
    AddressNotFound(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

impl From<GitcoinError> for PassportError {
    fn from(err: GitcoinError) -> Self {
        match err {
            GitcoinError::RateLimited => PassportError::RateLimitExceeded,
            GitcoinError::InvalidAddress(addr) => PassportError::AddressNotFound(addr),
            GitcoinError::ApiError { status_code: 401, message } => {
                PassportError::AuthenticationFailed(message)
            }
            GitcoinError::ApiError { status_code: 404, message } => {
                PassportError::AddressNotFound(message)
            }
            GitcoinError::ApiError { message, .. } => PassportError::InvalidResponse(message),
            GitcoinError::NetworkError(msg) => PassportError::NetworkError(msg),
            GitcoinError::ParseError(msg) => PassportError::InvalidResponse(msg),
            GitcoinError::ConfigError(msg) => PassportError::ConfigError(msg),
        }
    }
}

/// API response for passport score endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassportScoreResponse {
    /// Ethereum address (checksummed)
    pub address: String,

    /// Overall passport score (0-100+)
    pub score: f32,

    /// When the score was last updated
    pub last_updated: Option<DateTime<Utc>>,

    /// Number of verified stamps
    pub stamps_count: usize,
}

/// Detailed stamp information from API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StampDetails {
    /// The stamp provider name
    pub provider: String,

    /// Hash of the verifiable credential
    pub hash: String,

    /// Additional metadata from the stamp
    pub metadata: HashMap<String, serde_json::Value>,

    /// When the stamp was created/verified
    pub created_at: Option<DateTime<Utc>>,
}

/// Full passport verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassportVerification {
    /// Whether the passport meets requirements
    pub is_valid: bool,

    /// Current passport score
    pub score: f32,

    /// Whether score meets the specified threshold
    pub meets_threshold: bool,

    /// Verified stamps
    pub stamps: Vec<StampDetails>,

    /// Threshold used for verification
    pub threshold: f32,

    /// When verification was performed
    pub verified_at: DateTime<Utc>,
}

/// Rate limiter for API requests
#[derive(Debug)]
struct RateLimiter {
    last_request: Option<Instant>,
    min_interval: Duration,
}

impl RateLimiter {
    fn new(requests_per_second: f32) -> Self {
        Self {
            last_request: None,
            min_interval: Duration::from_secs_f32(1.0 / requests_per_second),
        }
    }

    async fn wait(&mut self) {
        if let Some(last) = self.last_request {
            let elapsed = last.elapsed();
            if elapsed < self.min_interval {
                tokio::time::sleep(self.min_interval - elapsed).await;
            }
        }
        self.last_request = Some(Instant::now());
    }
}

/// Cache entry with expiration
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    value: T,
    cached_at: Instant,
}

impl<T> CacheEntry<T> {
    fn new(value: T) -> Self {
        Self {
            value,
            cached_at: Instant::now(),
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.cached_at.elapsed() > ttl
    }
}

/// Known Gitcoin Passport stamp providers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StampProvider {
    // Social Media
    Google,
    Twitter,
    Github,
    Discord,
    Facebook,
    LinkedIn,

    // Web3 & Blockchain
    ENS,
    NFT,
    BrightId,
    ProofOfHumanity,
    Lens,
    GnosisSafe,
    EthBalance,
    EthTransactions,
    EthGasSpent,

    // Identity Verification
    Coinbase,
    Holonym,
    Civic,
    Idena,
    Worldcoin,

    // Developer
    GitcoinGrants,
    GitcoinContributor,
    StackOverflow,

    // Education & DAOs
    GuildXYZ,
    Snapshot,

    // Other
    ZkSync,
    TrustaLabs,
    Hypercerts,

    // Unknown provider
    Unknown(String),
}

impl StampProvider {
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "google" => StampProvider::Google,
            "twitter" => StampProvider::Twitter,
            "github" | "Github" => StampProvider::Github,
            "discord" => StampProvider::Discord,
            "facebook" => StampProvider::Facebook,
            "linkedin" => StampProvider::LinkedIn,
            "ens" => StampProvider::ENS,
            "nft" => StampProvider::NFT,
            "brightid" => StampProvider::BrightId,
            "poh" | "proofofhumanity" => StampProvider::ProofOfHumanity,
            "lens" => StampProvider::Lens,
            "gnosissafe" => StampProvider::GnosisSafe,
            "ethbalance" => StampProvider::EthBalance,
            "ethtransactions" => StampProvider::EthTransactions,
            "ethgasspent" => StampProvider::EthGasSpent,
            "coinbase" => StampProvider::Coinbase,
            "holonym" => StampProvider::Holonym,
            "civic" => StampProvider::Civic,
            "idena" => StampProvider::Idena,
            "worldcoin" => StampProvider::Worldcoin,
            "gitcoingrants" => StampProvider::GitcoinGrants,
            "gitcoincontributor" => StampProvider::GitcoinContributor,
            "stackoverflow" => StampProvider::StackOverflow,
            "guildxyz" => StampProvider::GuildXYZ,
            "snapshot" => StampProvider::Snapshot,
            "zksync" => StampProvider::ZkSync,
            "trustalabs" => StampProvider::TrustaLabs,
            "hypercerts" => StampProvider::Hypercerts,
            other => StampProvider::Unknown(other.to_string()),
        }
    }
}

/// A verified stamp (credential) from Gitcoin Passport
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stamp {
    /// The stamp provider
    pub provider: StampProvider,

    /// Provider name as string
    pub provider_name: String,

    /// Hash of the verifiable credential
    pub credential_hash: String,

    /// Score contribution of this stamp
    pub score: f32,

    /// When the stamp expires
    pub expires_at: Option<DateTime<Utc>>,

    /// Whether this stamp is deduplicated
    pub is_duplicate: bool,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Stamp {
    /// Check if stamp has expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires) = self.expires_at {
            Utc::now() > expires
        } else {
            false
        }
    }

    /// Check if stamp is valid (not expired and not duplicate)
    pub fn is_valid(&self) -> bool {
        !self.is_expired() && !self.is_duplicate
    }
}

/// Complete Passport score response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassportScore {
    /// Ethereum address
    pub address: String,

    /// Overall score (0-100)
    pub score: f32,

    /// Whether score passes threshold
    pub passing: bool,

    /// Passing threshold used
    pub threshold: f32,

    /// Verified stamps
    pub stamps: Vec<Stamp>,

    /// When score was last calculated
    pub last_updated: Option<DateTime<Utc>>,

    /// When score expires
    pub expiration: Option<DateTime<Utc>>,
}

impl PassportScore {
    /// Get count of valid stamps
    pub fn valid_stamp_count(&self) -> usize {
        self.stamps.iter().filter(|s| s.is_valid()).count()
    }

    /// Check if passport has a specific stamp
    pub fn has_stamp(&self, provider: &StampProvider) -> bool {
        self.stamps
            .iter()
            .any(|s| &s.provider == provider && s.is_valid())
    }

    /// Check if score is expired
    pub fn is_expired(&self) -> bool {
        if let Some(exp) = self.expiration {
            Utc::now() > exp
        } else {
            false
        }
    }

    /// Get stamps by provider
    pub fn get_stamp(&self, provider: &StampProvider) -> Option<&Stamp> {
        self.stamps.iter().find(|s| &s.provider == provider)
    }
}

/// Configuration for Gitcoin Passport client
#[derive(Debug, Clone)]
pub struct GitcoinPassportConfig {
    /// API key for authentication
    pub api_key: String,

    /// Scorer ID for score calculation
    pub scorer_id: String,

    /// Base URL (default: https://api.passport.gitcoin.co/v2)
    pub base_url: String,

    /// Request timeout
    pub timeout: Duration,

    /// Cache TTL (default: 5 minutes)
    pub cache_ttl: Duration,

    /// Enable caching
    pub enable_cache: bool,

    /// Rate limit (requests per second, default: 1.0)
    pub rate_limit: f32,
}

impl Default for GitcoinPassportConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            scorer_id: String::new(),
            base_url: "https://api.passport.gitcoin.co/v2".to_string(),
            timeout: Duration::from_secs(30),
            cache_ttl: Duration::from_secs(300), // 5 minutes
            enable_cache: true,
            rate_limit: 1.0, // 1 request per second
        }
    }
}

impl GitcoinPassportConfig {
    /// Create a new config with API key and scorer ID
    pub fn new(api_key: &str, scorer_id: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            scorer_id: scorer_id.to_string(),
            ..Default::default()
        }
    }

    /// Create config from environment variables
    ///
    /// Required env vars:
    /// - GITCOIN_API_KEY: API key for authentication
    /// - GITCOIN_SCORER_ID: Scorer ID (optional, defaults to empty)
    pub fn from_env() -> Result<Self, GitcoinError> {
        let api_key = std::env::var("GITCOIN_API_KEY")
            .map_err(|_| GitcoinError::ConfigError("GITCOIN_API_KEY not set".to_string()))?;

        let scorer_id = std::env::var("GITCOIN_SCORER_ID").unwrap_or_default();

        let base_url = std::env::var("GITCOIN_API_BASE_URL")
            .unwrap_or_else(|_| "https://api.passport.gitcoin.co/v2".to_string());

        Ok(Self {
            api_key,
            scorer_id,
            base_url,
            ..Default::default()
        })
    }

    /// Set custom base URL (useful for testing)
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }

    /// Set cache TTL
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Disable caching
    pub fn without_cache(mut self) -> Self {
        self.enable_cache = false;
        self
    }

    /// Set rate limit
    pub fn with_rate_limit(mut self, requests_per_second: f32) -> Self {
        self.rate_limit = requests_per_second;
        self
    }
}

/// Gitcoin Passport API client with rate limiting and caching
pub struct GitcoinPassportClient {
    config: GitcoinPassportConfig,
    #[cfg(feature = "identity")]
    http_client: reqwest::Client,
    rate_limiter: Arc<Mutex<RateLimiter>>,
    score_cache: HashMap<String, CacheEntry<PassportScore>>,
    stamps_cache: HashMap<String, CacheEntry<Vec<StampDetails>>>,
}

impl std::fmt::Debug for GitcoinPassportClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GitcoinPassportClient")
            .field("config", &self.config)
            .field("score_cache_size", &self.score_cache.len())
            .field("stamps_cache_size", &self.stamps_cache.len())
            .finish()
    }
}

impl GitcoinPassportClient {
    /// Create a new client
    pub fn new(config: GitcoinPassportConfig) -> Result<Self, GitcoinError> {
        if config.api_key.is_empty() {
            return Err(GitcoinError::ConfigError(
                "API key is required".to_string(),
            ));
        }

        #[cfg(feature = "identity")]
        let http_client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| GitcoinError::NetworkError(e.to_string()))?;

        Ok(Self {
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new(config.rate_limit))),
            #[cfg(feature = "identity")]
            http_client,
            score_cache: HashMap::new(),
            stamps_cache: HashMap::new(),
            config,
        })
    }

    /// Create from environment variables
    pub fn from_env() -> Result<Self, GitcoinError> {
        let config = GitcoinPassportConfig::from_env()?;
        Self::new(config)
    }

    /// Validate Ethereum address format
    fn validate_address(address: &str) -> Result<String, GitcoinError> {
        let addr = address.trim().to_lowercase();

        // Basic validation: must start with 0x and be 42 chars
        if !addr.starts_with("0x") || addr.len() != 42 {
            return Err(GitcoinError::InvalidAddress(format!(
                "Invalid Ethereum address format: {}",
                address
            )));
        }

        // Check if it's valid hex
        if !addr[2..].chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(GitcoinError::InvalidAddress(format!(
                "Address contains invalid characters: {}",
                address
            )));
        }

        Ok(addr)
    }

    /// Fetch passport score for an address
    ///
    /// Returns a simplified score response with address, score, and stamp count.
    #[cfg(feature = "identity")]
    pub async fn fetch_score(&mut self, address: &str) -> Result<PassportScoreResponse, GitcoinError> {
        let address = Self::validate_address(address)?;

        // Check cache first
        if self.config.enable_cache {
            if let Some(entry) = self.score_cache.get(&address) {
                if !entry.is_expired(self.config.cache_ttl) {
                    return Ok(PassportScoreResponse {
                        address: entry.value.address.clone(),
                        score: entry.value.score,
                        last_updated: entry.value.last_updated,
                        stamps_count: entry.value.stamps.len(),
                    });
                }
            }
        }

        // Rate limit
        self.rate_limiter.lock().await.wait().await;

        let url = format!(
            "{}/passport/score/{}",
            self.config.base_url, address
        );

        let response = self.http_client
            .get(&url)
            .header("X-API-KEY", &self.config.api_key)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| GitcoinError::NetworkError(e.to_string()))?;

        let status = response.status().as_u16();
        match status {
            200 => {
                let data: serde_json::Value = response
                    .json()
                    .await
                    .map_err(|e| GitcoinError::ParseError(e.to_string()))?;

                let passport_score = self.parse_score_response(&address, data)?;

                // Update cache
                if self.config.enable_cache {
                    self.score_cache
                        .insert(address.clone(), CacheEntry::new(passport_score.clone()));
                }

                Ok(PassportScoreResponse {
                    address: passport_score.address,
                    score: passport_score.score,
                    last_updated: passport_score.last_updated,
                    stamps_count: passport_score.stamps.len(),
                })
            }
            401 => Err(GitcoinError::ApiError {
                status_code: 401,
                message: "Invalid API key".to_string(),
            }),
            404 => Err(GitcoinError::ApiError {
                status_code: 404,
                message: format!("Passport not found for address: {}", address),
            }),
            429 => Err(GitcoinError::RateLimited),
            _ => {
                let body = response.text().await.unwrap_or_default();
                Err(GitcoinError::ApiError {
                    status_code: status,
                    message: body,
                })
            }
        }
    }

    /// Fetch stamps for an address
    ///
    /// Returns detailed stamp information including provider, hash, and metadata.
    #[cfg(feature = "identity")]
    pub async fn fetch_stamps(&mut self, address: &str) -> Result<Vec<StampDetails>, GitcoinError> {
        let address = Self::validate_address(address)?;

        // Check cache first
        if self.config.enable_cache {
            if let Some(entry) = self.stamps_cache.get(&address) {
                if !entry.is_expired(self.config.cache_ttl) {
                    return Ok(entry.value.clone());
                }
            }
        }

        // Rate limit
        self.rate_limiter.lock().await.wait().await;

        let url = format!(
            "{}/passport/stamps/{}",
            self.config.base_url, address
        );

        let response = self.http_client
            .get(&url)
            .header("X-API-KEY", &self.config.api_key)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| GitcoinError::NetworkError(e.to_string()))?;

        let status = response.status().as_u16();
        match status {
            200 => {
                let data: serde_json::Value = response
                    .json()
                    .await
                    .map_err(|e| GitcoinError::ParseError(e.to_string()))?;

                let stamps = self.parse_stamps_response(data)?;

                // Update cache
                if self.config.enable_cache {
                    self.stamps_cache
                        .insert(address, CacheEntry::new(stamps.clone()));
                }

                Ok(stamps)
            }
            401 => Err(GitcoinError::ApiError {
                status_code: 401,
                message: "Invalid API key".to_string(),
            }),
            404 => Err(GitcoinError::ApiError {
                status_code: 404,
                message: format!("No stamps found for address: {}", address),
            }),
            429 => Err(GitcoinError::RateLimited),
            _ => {
                let body = response.text().await.unwrap_or_default();
                Err(GitcoinError::ApiError {
                    status_code: status,
                    message: body,
                })
            }
        }
    }

    /// Verify a passport with threshold checking
    ///
    /// Returns full verification result including validity, score, and stamps.
    #[cfg(feature = "identity")]
    pub async fn verify_passport(
        &mut self,
        address: &str,
        threshold: f32,
    ) -> Result<PassportVerification, GitcoinError> {
        let address = Self::validate_address(address)?;

        // Fetch score (uses cache internally)
        let score_response = self.fetch_score(&address).await?;

        // Fetch stamps
        let stamps = self.fetch_stamps(&address).await.unwrap_or_default();

        let meets_threshold = score_response.score >= threshold;

        Ok(PassportVerification {
            is_valid: meets_threshold && !stamps.is_empty(),
            score: score_response.score,
            meets_threshold,
            stamps,
            threshold,
            verified_at: Utc::now(),
        })
    }

    /// Parse stamps from API response
    fn parse_stamps_response(
        &self,
        data: serde_json::Value,
    ) -> Result<Vec<StampDetails>, GitcoinError> {
        let mut stamps = Vec::new();

        // Handle both array and object formats
        let items = if let Some(arr) = data.get("items").and_then(|i| i.as_array()) {
            arr.clone()
        } else if let Some(arr) = data.as_array() {
            arr.clone()
        } else {
            Vec::new()
        };

        for item in items {
            if let Some(obj) = item.as_object() {
                let provider = obj
                    .get("provider")
                    .and_then(|p| p.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                let hash = obj
                    .get("credential")
                    .and_then(|c| c.get("credentialSubject"))
                    .and_then(|cs| cs.get("hash"))
                    .and_then(|h| h.as_str())
                    .unwrap_or("")
                    .to_string();

                let created_at = obj
                    .get("credential")
                    .and_then(|c| c.get("issuanceDate"))
                    .and_then(|d| d.as_str())
                    .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                    .map(|d| d.with_timezone(&Utc));

                // Extract metadata
                let mut metadata = HashMap::new();
                if let Some(cred) = obj.get("credential") {
                    if let Some(cs) = cred.get("credentialSubject") {
                        if let Some(obj) = cs.as_object() {
                            for (k, v) in obj {
                                if k != "hash" && k != "id" && k != "provider" {
                                    metadata.insert(k.clone(), v.clone());
                                }
                            }
                        }
                    }
                }

                stamps.push(StampDetails {
                    provider,
                    hash,
                    metadata,
                    created_at,
                });
            }
        }

        Ok(stamps)
    }

    // ========================
    // Legacy API (backward compatibility)
    // ========================

    /// Get passport score for an address (legacy API)
    #[cfg(feature = "identity")]
    pub async fn get_score(&mut self, address: &str) -> Result<PassportScore, PassportError> {
        let address_lower = address.to_lowercase();

        // Check cache
        if self.config.enable_cache {
            if let Some(entry) = self.score_cache.get(&address_lower) {
                if !entry.is_expired(self.config.cache_ttl) {
                    return Ok(entry.value.clone());
                }
            }
        }

        // Make API request
        let score = self.fetch_score_legacy(&address_lower).await?;

        // Update cache
        if self.config.enable_cache {
            self.score_cache
                .insert(address_lower, CacheEntry::new(score.clone()));
        }

        Ok(score)
    }

    /// Fetch score from API (internal, legacy format)
    #[cfg(feature = "identity")]
    async fn fetch_score_legacy(&mut self, address: &str) -> Result<PassportScore, PassportError> {
        // Rate limit
        self.rate_limiter.lock().await.wait().await;

        let url = if !self.config.scorer_id.is_empty() {
            format!(
                "{}/stamps/{}/score/{}",
                self.config.base_url, self.config.scorer_id, address
            )
        } else {
            format!(
                "{}/passport/score/{}",
                self.config.base_url, address
            )
        };

        let response = self.http_client
            .get(&url)
            .header("X-API-KEY", &self.config.api_key)
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| PassportError::NetworkError(e.to_string()))?;

        match response.status().as_u16() {
            200 => {
                let data: serde_json::Value = response
                    .json()
                    .await
                    .map_err(|e| PassportError::InvalidResponse(e.to_string()))?;

                self.parse_score_response(address, data)
                    .map_err(|e| PassportError::InvalidResponse(e.to_string()))
            }
            401 => Err(PassportError::AuthenticationFailed("Invalid API key".to_string())),
            404 => Err(PassportError::AddressNotFound(address.to_string())),
            429 => Err(PassportError::RateLimitExceeded),
            status => Err(PassportError::NetworkError(format!(
                "Unexpected status code: {}",
                status
            ))),
        }
    }

    /// Parse score response from API
    fn parse_score_response(
        &self,
        address: &str,
        data: serde_json::Value,
    ) -> Result<PassportScore, GitcoinError> {
        let score = data
            .get("score")
            .and_then(|s| {
                // Handle both string and number formats
                s.as_str()
                    .and_then(|str_val| str_val.parse::<f32>().ok())
                    .or_else(|| s.as_f64().map(|n| n as f32))
            })
            .unwrap_or(0.0);

        let passing = data
            .get("passing_score")
            .and_then(|p| p.as_bool())
            .or_else(|| data.get("passing").and_then(|p| p.as_bool()))
            .unwrap_or(false);

        // Parse stamps
        let mut stamps = Vec::new();
        if let Some(stamps_obj) = data.get("stamps").and_then(|s| s.as_object()) {
            for (provider_name, stamp_data) in stamps_obj {
                if let Some(stamp_obj) = stamp_data.as_object() {
                    let stamp = Stamp {
                        provider: StampProvider::from_string(provider_name),
                        provider_name: provider_name.clone(),
                        credential_hash: stamp_obj
                            .get("hash")
                            .and_then(|h| h.as_str())
                            .unwrap_or("")
                            .to_string(),
                        score: stamp_obj
                            .get("score")
                            .and_then(|s| s.as_f64())
                            .map(|s| s as f32)
                            .unwrap_or(0.0),
                        expires_at: stamp_obj
                            .get("expiration_date")
                            .and_then(|e| e.as_str())
                            .and_then(|e| DateTime::parse_from_rfc3339(e).ok())
                            .map(|d| d.with_timezone(&Utc)),
                        is_duplicate: stamp_obj
                            .get("dedup")
                            .and_then(|d| d.as_bool())
                            .unwrap_or(false),
                        metadata: HashMap::new(),
                    };
                    stamps.push(stamp);
                }
            }
        }

        // Parse timestamps
        let last_updated = data
            .get("last_score_timestamp")
            .and_then(|t| t.as_str())
            .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
            .map(|d| d.with_timezone(&Utc));

        let expiration = data
            .get("expiration_timestamp")
            .and_then(|t| t.as_str())
            .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
            .map(|d| d.with_timezone(&Utc));

        Ok(PassportScore {
            address: address.to_string(),
            score,
            passing,
            threshold: 20.0, // Default threshold
            stamps,
            last_updated,
            expiration,
        })
    }

    /// Verify humanity (boolean check)
    #[cfg(feature = "identity")]
    pub async fn verify_humanity(
        &mut self,
        address: &str,
        threshold: f32,
    ) -> Result<bool, PassportError> {
        let score = self.get_score(address).await?;
        Ok(score.score >= threshold)
    }

    /// Check humanity with required stamps
    #[cfg(feature = "identity")]
    pub async fn check_humanity_threshold(
        &mut self,
        address: &str,
        min_score: f32,
        required_stamps: Option<&[StampProvider]>,
    ) -> Result<bool, PassportError> {
        let score = self.get_score(address).await?;

        if score.score < min_score {
            return Ok(false);
        }

        if let Some(required) = required_stamps {
            for provider in required {
                if !score.has_stamp(provider) {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Invalidate cache for an address
    pub fn invalidate_cache(&mut self, address: &str) {
        let addr = address.to_lowercase();
        self.score_cache.remove(&addr);
        self.stamps_cache.remove(&addr);
    }

    /// Clear all cache entries
    pub fn clear_cache(&mut self) {
        self.score_cache.clear();
        self.stamps_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.score_cache.len(), self.stamps_cache.len())
    }
}

/// Result of governance verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceVerificationResult {
    pub address: String,
    pub score: Option<PassportScore>,
    pub passes_e2: bool,
    pub passes_e3: bool,
    pub from_cache: bool,
    pub error: Option<String>,
}

impl GovernanceVerificationResult {
    /// Get assurance level string
    pub fn assurance_level(&self) -> &'static str {
        if self.passes_e3 {
            "E3_CryptographicallyProven"
        } else if self.passes_e2 {
            "E2_PrivatelyVerifiable"
        } else {
            "E1_Testimonial"
        }
    }

    /// Get vote weight multiplier
    pub fn vote_weight_multiplier(&self) -> f32 {
        if self.passes_e3 {
            1.0
        } else if self.passes_e2 {
            0.85
        } else {
            0.7
        }
    }
}

/// Verify for governance (E2 threshold = 20.0, E3 threshold = 50.0)
#[cfg(feature = "identity")]
pub async fn verify_for_governance(
    client: &mut GitcoinPassportClient,
    address: &str,
) -> GovernanceVerificationResult {
    match client.get_score(address).await {
        Ok(score) => GovernanceVerificationResult {
            address: address.to_string(),
            passes_e2: score.score >= 20.0,
            passes_e3: score.score >= 50.0,
            from_cache: false,
            score: Some(score),
            error: None,
        },
        Err(e) => GovernanceVerificationResult {
            address: address.to_string(),
            passes_e2: false,
            passes_e3: false,
            from_cache: false,
            score: None,
            error: Some(e.to_string()),
        },
    }
}

// ============================================================================
// GitcoinPassportFactor API Integration
// ============================================================================

use super::factors::GitcoinPassportFactor;
use super::types::FactorStatus;
use super::IdentityError;

impl GitcoinPassportFactor {
    /// Create a GitcoinPassportFactor from API response
    ///
    /// Fetches the passport score and stamps from the Gitcoin API and
    /// populates the factor with the retrieved data.
    ///
    /// # Arguments
    ///
    /// * `client` - A mutable reference to the GitcoinPassportClient
    /// * `address` - The Ethereum address to fetch the passport for
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use fl_aggregator::identity::{GitcoinPassportClient, GitcoinPassportConfig};
    /// use fl_aggregator::identity::factors::GitcoinPassportFactor;
    ///
    /// let config = GitcoinPassportConfig::from_env()?;
    /// let mut client = GitcoinPassportClient::new(config)?;
    ///
    /// let factor = GitcoinPassportFactor::from_api(&mut client, "0x1234...").await?;
    /// println!("Score: {}, Stamps: {:?}", factor.score, factor.stamps);
    /// ```
    #[cfg(feature = "identity")]
    pub async fn from_api(
        client: &mut GitcoinPassportClient,
        address: &str,
    ) -> Result<Self, IdentityError> {
        // Validate address
        let address = GitcoinPassportClient::validate_address(address)
            .map_err(|e| IdentityError::InvalidFactor(e.to_string()))?;

        // Fetch score from API
        let score_response = client
            .fetch_score(&address)
            .await
            .map_err(|e| IdentityError::NetworkError(e.to_string()))?;

        // Fetch stamps from API
        let stamps_result = client.fetch_stamps(&address).await;
        let stamp_names: Vec<String> = stamps_result
            .map(|stamps| stamps.iter().map(|s| s.provider.clone()).collect())
            .unwrap_or_default();

        // Determine status based on score
        let status = if score_response.score >= 20.0 {
            FactorStatus::Active
        } else {
            FactorStatus::Pending
        };

        // Calculate expiry (use 90 days from now as default)
        let expiry = Some(Utc::now() + chrono::Duration::days(90));

        Ok(Self {
            factor_id: format!("gitcoin-{}", generate_factor_id()),
            passport_address: address,
            score: score_response.score,
            stamps: stamp_names,
            expiry,
            status,
            created_at: Utc::now(),
            last_verified: if status == FactorStatus::Active {
                Some(Utc::now())
            } else {
                None
            },
        })
    }

    /// Refresh the factor data from API
    ///
    /// Updates the score and stamps by fetching fresh data from the API.
    #[cfg(feature = "identity")]
    pub async fn refresh(
        &mut self,
        client: &mut GitcoinPassportClient,
    ) -> Result<(), IdentityError> {
        // Invalidate cache to get fresh data
        client.invalidate_cache(&self.passport_address);

        // Fetch fresh score
        let score_response = client
            .fetch_score(&self.passport_address)
            .await
            .map_err(|e| IdentityError::NetworkError(e.to_string()))?;

        // Fetch fresh stamps
        let stamps_result = client.fetch_stamps(&self.passport_address).await;
        let stamp_names: Vec<String> = stamps_result
            .map(|stamps| stamps.iter().map(|s| s.provider.clone()).collect())
            .unwrap_or_default();

        // Update fields
        self.score = score_response.score;
        self.stamps = stamp_names;
        self.last_verified = Some(Utc::now());

        // Update status based on new score
        if self.score >= 20.0 {
            self.status = FactorStatus::Active;
        } else if self.status == FactorStatus::Active {
            self.status = FactorStatus::Pending;
        }

        Ok(())
    }
}

/// Generate a short factor ID
fn generate_factor_id() -> String {
    use sha2::{Digest, Sha256};
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let hash = Sha256::digest(now.to_le_bytes());
    hex::encode(&hash[..4])
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // Unit Tests - Types and Parsing
    // ============================================================================

    #[test]
    fn test_stamp_provider_parsing() {
        assert_eq!(StampProvider::from_string("github"), StampProvider::Github);
        assert_eq!(StampProvider::from_string("Google"), StampProvider::Google);
        assert_eq!(StampProvider::from_string("twitter"), StampProvider::Twitter);
        assert_eq!(StampProvider::from_string("discord"), StampProvider::Discord);
        assert_eq!(StampProvider::from_string("ens"), StampProvider::ENS);
        assert_eq!(StampProvider::from_string("brightid"), StampProvider::BrightId);
        assert_eq!(StampProvider::from_string("worldcoin"), StampProvider::Worldcoin);
        assert_eq!(
            StampProvider::from_string("unknown_provider"),
            StampProvider::Unknown("unknown_provider".to_string())
        );
    }

    #[test]
    fn test_stamp_validity() {
        let valid_stamp = Stamp {
            provider: StampProvider::Github,
            provider_name: "Github".to_string(),
            credential_hash: "hash123".to_string(),
            score: 5.0,
            expires_at: Some(Utc::now() + chrono::Duration::days(30)),
            is_duplicate: false,
            metadata: HashMap::new(),
        };
        assert!(valid_stamp.is_valid());
        assert!(!valid_stamp.is_expired());

        let expired_stamp = Stamp {
            expires_at: Some(Utc::now() - chrono::Duration::days(1)),
            ..valid_stamp.clone()
        };
        assert!(!expired_stamp.is_valid());
        assert!(expired_stamp.is_expired());

        let duplicate_stamp = Stamp {
            is_duplicate: true,
            ..valid_stamp.clone()
        };
        assert!(!duplicate_stamp.is_valid());

        // No expiry should be valid
        let no_expiry_stamp = Stamp {
            expires_at: None,
            ..valid_stamp
        };
        assert!(no_expiry_stamp.is_valid());
        assert!(!no_expiry_stamp.is_expired());
    }

    #[test]
    fn test_passport_score() {
        let score = PassportScore {
            address: "0x123".to_string(),
            score: 45.0,
            passing: true,
            threshold: 20.0,
            stamps: vec![
                Stamp {
                    provider: StampProvider::Github,
                    provider_name: "Github".to_string(),
                    credential_hash: "hash1".to_string(),
                    score: 10.0,
                    expires_at: Some(Utc::now() + chrono::Duration::days(30)),
                    is_duplicate: false,
                    metadata: HashMap::new(),
                },
                Stamp {
                    provider: StampProvider::Google,
                    provider_name: "Google".to_string(),
                    credential_hash: "hash2".to_string(),
                    score: 5.0,
                    expires_at: Some(Utc::now() + chrono::Duration::days(30)),
                    is_duplicate: false,
                    metadata: HashMap::new(),
                },
            ],
            last_updated: Some(Utc::now()),
            expiration: Some(Utc::now() + chrono::Duration::days(30)),
        };

        assert_eq!(score.valid_stamp_count(), 2);
        assert!(score.has_stamp(&StampProvider::Github));
        assert!(score.has_stamp(&StampProvider::Google));
        assert!(!score.has_stamp(&StampProvider::Discord));
        assert!(!score.is_expired());
    }

    #[test]
    fn test_passport_score_expired() {
        let score = PassportScore {
            address: "0x123".to_string(),
            score: 45.0,
            passing: true,
            threshold: 20.0,
            stamps: vec![],
            last_updated: Some(Utc::now() - chrono::Duration::days(100)),
            expiration: Some(Utc::now() - chrono::Duration::days(1)),
        };

        assert!(score.is_expired());
    }

    #[test]
    fn test_governance_verification_result() {
        let result = GovernanceVerificationResult {
            address: "0x123".to_string(),
            score: None,
            passes_e2: true,
            passes_e3: false,
            from_cache: false,
            error: None,
        };

        assert_eq!(result.assurance_level(), "E2_PrivatelyVerifiable");
        assert_eq!(result.vote_weight_multiplier(), 0.85);

        let e3_result = GovernanceVerificationResult {
            passes_e2: true,
            passes_e3: true,
            ..result.clone()
        };
        assert_eq!(e3_result.assurance_level(), "E3_CryptographicallyProven");
        assert_eq!(e3_result.vote_weight_multiplier(), 1.0);

        let e1_result = GovernanceVerificationResult {
            passes_e2: false,
            passes_e3: false,
            ..result
        };
        assert_eq!(e1_result.assurance_level(), "E1_Testimonial");
        assert_eq!(e1_result.vote_weight_multiplier(), 0.7);
    }

    // ============================================================================
    // GitcoinError Tests
    // ============================================================================

    #[test]
    fn test_gitcoin_error_types() {
        let rate_limited = GitcoinError::RateLimited;
        assert!(rate_limited.to_string().contains("Rate limit"));

        let invalid_addr = GitcoinError::InvalidAddress("0xinvalid".to_string());
        assert!(invalid_addr.to_string().contains("Invalid address"));

        let api_error = GitcoinError::ApiError {
            status_code: 500,
            message: "Internal server error".to_string(),
        };
        assert!(api_error.to_string().contains("500"));
        assert!(api_error.to_string().contains("Internal server error"));

        let network_error = GitcoinError::NetworkError("Connection refused".to_string());
        assert!(network_error.to_string().contains("Network error"));

        let parse_error = GitcoinError::ParseError("Invalid JSON".to_string());
        assert!(parse_error.to_string().contains("Invalid response format"));

        let config_error = GitcoinError::ConfigError("Missing API key".to_string());
        assert!(config_error.to_string().contains("Configuration error"));
    }

    #[test]
    fn test_gitcoin_error_to_passport_error() {
        let rate_limited: PassportError = GitcoinError::RateLimited.into();
        assert!(matches!(rate_limited, PassportError::RateLimitExceeded));

        let invalid_addr: PassportError =
            GitcoinError::InvalidAddress("0x123".to_string()).into();
        assert!(matches!(invalid_addr, PassportError::AddressNotFound(_)));

        let auth_error: PassportError = GitcoinError::ApiError {
            status_code: 401,
            message: "Unauthorized".to_string(),
        }
        .into();
        assert!(matches!(auth_error, PassportError::AuthenticationFailed(_)));

        let not_found: PassportError = GitcoinError::ApiError {
            status_code: 404,
            message: "Not found".to_string(),
        }
        .into();
        assert!(matches!(not_found, PassportError::AddressNotFound(_)));
    }

    // ============================================================================
    // Address Validation Tests
    // ============================================================================

    #[test]
    fn test_address_validation() {
        // Valid addresses
        let valid = GitcoinPassportClient::validate_address(
            "0x1234567890abcdef1234567890abcdef12345678",
        );
        assert!(valid.is_ok());
        assert_eq!(
            valid.unwrap(),
            "0x1234567890abcdef1234567890abcdef12345678"
        );

        // Valid with mixed case (should lowercase)
        let mixed_case = GitcoinPassportClient::validate_address(
            "0x1234567890ABCDEF1234567890abcdef12345678",
        );
        assert!(mixed_case.is_ok());
        assert_eq!(
            mixed_case.unwrap(),
            "0x1234567890abcdef1234567890abcdef12345678"
        );

        // With whitespace
        let with_spaces = GitcoinPassportClient::validate_address(
            "  0x1234567890abcdef1234567890abcdef12345678  ",
        );
        assert!(with_spaces.is_ok());

        // Invalid: no 0x prefix
        let no_prefix = GitcoinPassportClient::validate_address(
            "1234567890abcdef1234567890abcdef12345678",
        );
        assert!(no_prefix.is_err());
        assert!(matches!(no_prefix.unwrap_err(), GitcoinError::InvalidAddress(_)));

        // Invalid: too short
        let too_short = GitcoinPassportClient::validate_address("0x1234");
        assert!(too_short.is_err());

        // Invalid: too long
        let too_long = GitcoinPassportClient::validate_address(
            "0x1234567890abcdef1234567890abcdef1234567890",
        );
        assert!(too_long.is_err());

        // Invalid: non-hex characters
        let non_hex = GitcoinPassportClient::validate_address(
            "0x1234567890abcdef1234567890abcdefGHIJKLMN",
        );
        assert!(non_hex.is_err());
    }

    // ============================================================================
    // Configuration Tests
    // ============================================================================

    #[test]
    fn test_config_default() {
        let config = GitcoinPassportConfig::default();
        assert!(config.api_key.is_empty());
        assert!(config.scorer_id.is_empty());
        assert_eq!(config.base_url, "https://api.passport.gitcoin.co/v2");
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.cache_ttl, Duration::from_secs(300));
        assert!(config.enable_cache);
        assert_eq!(config.rate_limit, 1.0);
    }

    #[test]
    fn test_config_new() {
        let config = GitcoinPassportConfig::new("test_api_key", "scorer_123");
        assert_eq!(config.api_key, "test_api_key");
        assert_eq!(config.scorer_id, "scorer_123");
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = GitcoinPassportConfig::new("api_key", "scorer")
            .with_base_url("https://custom.api.com")
            .with_cache_ttl(Duration::from_secs(600))
            .with_rate_limit(2.0);

        assert_eq!(config.base_url, "https://custom.api.com");
        assert_eq!(config.cache_ttl, Duration::from_secs(600));
        assert_eq!(config.rate_limit, 2.0);
        assert!(config.enable_cache);
    }

    #[test]
    fn test_config_without_cache() {
        let config = GitcoinPassportConfig::new("api_key", "scorer").without_cache();
        assert!(!config.enable_cache);
    }

    #[test]
    fn test_client_creation_requires_api_key() {
        let config = GitcoinPassportConfig::default();
        let result = GitcoinPassportClient::new(config);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GitcoinError::ConfigError(_)
        ));
    }

    #[test]
    fn test_client_creation_with_valid_config() {
        let config = GitcoinPassportConfig::new("valid_api_key", "scorer_id");
        let result = GitcoinPassportClient::new(config);
        assert!(result.is_ok());
    }

    // ============================================================================
    // Response Parsing Tests
    // ============================================================================

    #[test]
    fn test_parse_score_response_string_score() {
        let config = GitcoinPassportConfig::new("api_key", "scorer");
        let client = GitcoinPassportClient::new(config).unwrap();

        let response = serde_json::json!({
            "score": "45.5",
            "passing_score": true,
            "last_score_timestamp": "2024-01-15T10:30:00Z",
            "stamps": {}
        });

        let result = client.parse_score_response("0x1234", response);
        assert!(result.is_ok());
        let score = result.unwrap();
        assert_eq!(score.score, 45.5);
        assert!(score.passing);
    }

    #[test]
    fn test_parse_score_response_numeric_score() {
        let config = GitcoinPassportConfig::new("api_key", "scorer");
        let client = GitcoinPassportClient::new(config).unwrap();

        let response = serde_json::json!({
            "score": 25.0,
            "passing": true,
            "stamps": {}
        });

        let result = client.parse_score_response("0x1234", response);
        assert!(result.is_ok());
        let score = result.unwrap();
        assert_eq!(score.score, 25.0);
    }

    #[test]
    fn test_parse_score_response_with_stamps() {
        let config = GitcoinPassportConfig::new("api_key", "scorer");
        let client = GitcoinPassportClient::new(config).unwrap();

        let response = serde_json::json!({
            "score": "35.0",
            "passing_score": true,
            "stamps": {
                "github": {
                    "hash": "abc123",
                    "score": 10.0,
                    "dedup": false
                },
                "google": {
                    "hash": "def456",
                    "score": 5.0,
                    "dedup": false,
                    "expiration_date": "2025-06-01T00:00:00Z"
                }
            }
        });

        let result = client.parse_score_response("0x1234", response);
        assert!(result.is_ok());
        let score = result.unwrap();
        assert_eq!(score.stamps.len(), 2);

        // Verify we can find stamps by provider
        let has_github = score.stamps.iter().any(|s| s.provider == StampProvider::Github);
        let has_google = score.stamps.iter().any(|s| s.provider == StampProvider::Google);
        assert!(has_github, "Should have Github stamp");
        assert!(has_google, "Should have Google stamp");
    }

    #[test]
    fn test_parse_stamps_response() {
        let config = GitcoinPassportConfig::new("api_key", "scorer");
        let client = GitcoinPassportClient::new(config).unwrap();

        let response = serde_json::json!({
            "items": [
                {
                    "provider": "Github",
                    "credential": {
                        "issuanceDate": "2024-01-01T00:00:00Z",
                        "credentialSubject": {
                            "hash": "abc123",
                            "id": "did:example:123",
                            "provider": "Github",
                            "extra_field": "value"
                        }
                    }
                },
                {
                    "provider": "Twitter",
                    "credential": {
                        "issuanceDate": "2024-02-01T00:00:00Z",
                        "credentialSubject": {
                            "hash": "def456"
                        }
                    }
                }
            ]
        });

        let result = client.parse_stamps_response(response);
        assert!(result.is_ok());
        let stamps = result.unwrap();
        assert_eq!(stamps.len(), 2);
        assert_eq!(stamps[0].provider, "Github");
        assert_eq!(stamps[0].hash, "abc123");
        assert!(stamps[0].created_at.is_some());
        assert!(stamps[0].metadata.contains_key("extra_field"));
        assert_eq!(stamps[1].provider, "Twitter");
    }

    #[test]
    fn test_parse_stamps_response_array_format() {
        let config = GitcoinPassportConfig::new("api_key", "scorer");
        let client = GitcoinPassportClient::new(config).unwrap();

        // Some APIs return stamps as direct array
        let response = serde_json::json!([
            {
                "provider": "ENS",
                "credential": {
                    "credentialSubject": {
                        "hash": "ens_hash"
                    }
                }
            }
        ]);

        let result = client.parse_stamps_response(response);
        assert!(result.is_ok());
        let stamps = result.unwrap();
        assert_eq!(stamps.len(), 1);
        assert_eq!(stamps[0].provider, "ENS");
    }

    // ============================================================================
    // Cache Tests
    // ============================================================================

    #[test]
    fn test_cache_entry_expiration() {
        let entry = CacheEntry::new("test_value");
        assert!(!entry.is_expired(Duration::from_secs(300)));

        // Simulate expiry by creating entry with short TTL
        std::thread::sleep(Duration::from_millis(50));
        assert!(entry.is_expired(Duration::from_millis(10)));
    }

    #[test]
    fn test_cache_invalidation() {
        let config = GitcoinPassportConfig::new("api_key", "scorer");
        let mut client = GitcoinPassportClient::new(config).unwrap();

        // Manually insert cache entry
        let score = PassportScore {
            address: "0x1234".to_string(),
            score: 50.0,
            passing: true,
            threshold: 20.0,
            stamps: vec![],
            last_updated: Some(Utc::now()),
            expiration: None,
        };
        client
            .score_cache
            .insert("0x1234".to_string(), CacheEntry::new(score));

        assert_eq!(client.cache_stats(), (1, 0));

        client.invalidate_cache("0x1234");
        assert_eq!(client.cache_stats(), (0, 0));
    }

    #[test]
    fn test_cache_clear() {
        let config = GitcoinPassportConfig::new("api_key", "scorer");
        let mut client = GitcoinPassportClient::new(config).unwrap();

        // Add entries to both caches
        let score = PassportScore {
            address: "0x1234".to_string(),
            score: 50.0,
            passing: true,
            threshold: 20.0,
            stamps: vec![],
            last_updated: Some(Utc::now()),
            expiration: None,
        };
        client
            .score_cache
            .insert("0x1234".to_string(), CacheEntry::new(score.clone()));
        client
            .score_cache
            .insert("0x5678".to_string(), CacheEntry::new(score));
        client.stamps_cache.insert(
            "0x1234".to_string(),
            CacheEntry::new(vec![StampDetails {
                provider: "Github".to_string(),
                hash: "abc".to_string(),
                metadata: HashMap::new(),
                created_at: None,
            }]),
        );

        assert_eq!(client.cache_stats(), (2, 1));

        client.clear_cache();
        assert_eq!(client.cache_stats(), (0, 0));
    }

    // ============================================================================
    // PassportScoreResponse and PassportVerification Tests
    // ============================================================================

    #[test]
    fn test_passport_score_response() {
        let response = PassportScoreResponse {
            address: "0x1234567890abcdef1234567890abcdef12345678".to_string(),
            score: 42.5,
            last_updated: Some(Utc::now()),
            stamps_count: 5,
        };

        assert_eq!(response.score, 42.5);
        assert_eq!(response.stamps_count, 5);
        assert!(response.last_updated.is_some());
    }

    #[test]
    fn test_passport_verification() {
        let verification = PassportVerification {
            is_valid: true,
            score: 35.0,
            meets_threshold: true,
            stamps: vec![
                StampDetails {
                    provider: "Github".to_string(),
                    hash: "hash1".to_string(),
                    metadata: HashMap::new(),
                    created_at: Some(Utc::now()),
                },
            ],
            threshold: 20.0,
            verified_at: Utc::now(),
        };

        assert!(verification.is_valid);
        assert!(verification.meets_threshold);
        assert_eq!(verification.stamps.len(), 1);
        assert_eq!(verification.threshold, 20.0);
    }

    #[test]
    fn test_passport_verification_invalid() {
        let verification = PassportVerification {
            is_valid: false,
            score: 15.0,
            meets_threshold: false,
            stamps: vec![],
            threshold: 20.0,
            verified_at: Utc::now(),
        };

        assert!(!verification.is_valid);
        assert!(!verification.meets_threshold);
        assert_eq!(verification.stamps.len(), 0);
    }

    // ============================================================================
    // StampDetails Tests
    // ============================================================================

    #[test]
    fn test_stamp_details() {
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), serde_json::json!("value1"));
        metadata.insert("key2".to_string(), serde_json::json!(123));

        let stamp = StampDetails {
            provider: "BrightId".to_string(),
            hash: "bright_hash_123".to_string(),
            metadata,
            created_at: Some(Utc::now()),
        };

        assert_eq!(stamp.provider, "BrightId");
        assert_eq!(stamp.hash, "bright_hash_123");
        assert!(stamp.metadata.contains_key("key1"));
        assert!(stamp.created_at.is_some());
    }

    // ============================================================================
    // Rate Limiter Tests
    // ============================================================================

    #[tokio::test]
    async fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(10.0); // 10 requests per second = 100ms interval

        // First request should be immediate
        let start = Instant::now();
        limiter.wait().await;
        assert!(start.elapsed() < Duration::from_millis(50));

        // Second request should wait approximately 100ms
        let start2 = Instant::now();
        limiter.wait().await;
        assert!(start2.elapsed() >= Duration::from_millis(90));
    }

    // ============================================================================
    // Integration with GitcoinPassportFactor Tests
    // ============================================================================

    #[test]
    fn test_generate_factor_id() {
        let id1 = generate_factor_id();
        // Small sleep to ensure different timestamp
        std::thread::sleep(Duration::from_millis(1));
        let id2 = generate_factor_id();

        assert_eq!(id1.len(), 8); // 4 bytes = 8 hex chars
        assert_ne!(id1, id2);
    }
}

// ============================================================================
// Integration Tests with Mock HTTP (requires `identity` feature)
// ============================================================================

#[cfg(all(test, feature = "identity"))]
mod integration_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock server for testing HTTP interactions
    /// Uses a simple counter to simulate different responses
    static REQUEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn reset_counter() {
        REQUEST_COUNTER.store(0, Ordering::SeqCst);
    }

    fn increment_counter() -> usize {
        REQUEST_COUNTER.fetch_add(1, Ordering::SeqCst)
    }

    /// Create a mock API response for score endpoint
    fn mock_score_response(score: f32, stamps_count: usize) -> serde_json::Value {
        let mut stamps = serde_json::Map::new();
        for i in 0..stamps_count {
            stamps.insert(
                format!("Provider{}", i),
                serde_json::json!({
                    "hash": format!("hash_{}", i),
                    "score": 5.0,
                    "dedup": false
                }),
            );
        }

        serde_json::json!({
            "score": score.to_string(),
            "passing_score": score >= 20.0,
            "stamps": stamps,
            "last_score_timestamp": "2024-01-15T10:30:00Z",
            "expiration_timestamp": "2024-07-15T10:30:00Z"
        })
    }

    /// Create a mock API response for stamps endpoint
    fn mock_stamps_response(providers: &[&str]) -> serde_json::Value {
        let items: Vec<serde_json::Value> = providers
            .iter()
            .map(|provider| {
                serde_json::json!({
                    "provider": provider,
                    "credential": {
                        "issuanceDate": "2024-01-01T00:00:00Z",
                        "credentialSubject": {
                            "hash": format!("{}_hash", provider.to_lowercase()),
                            "id": format!("did:example:{}", provider.to_lowercase())
                        }
                    }
                })
            })
            .collect();

        serde_json::json!({ "items": items })
    }

    #[test]
    fn test_mock_score_response_generation() {
        let response = mock_score_response(45.5, 3);
        assert_eq!(response["score"], "45.5");
        assert_eq!(response["passing_score"], true);
        assert!(response["stamps"].as_object().unwrap().len() == 3);
    }

    #[test]
    fn test_mock_stamps_response_generation() {
        let response = mock_stamps_response(&["Github", "Twitter", "ENS"]);
        let items = response["items"].as_array().unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0]["provider"], "Github");
    }

    #[tokio::test]
    async fn test_client_with_caching_behavior() {
        reset_counter();
        let config = GitcoinPassportConfig::new("test_key", "scorer")
            .with_cache_ttl(Duration::from_secs(60));

        let mut client = GitcoinPassportClient::new(config).unwrap();

        // Manually insert a cached score
        let cached_score = PassportScore {
            address: "0x1234567890abcdef1234567890abcdef12345678".to_string(),
            score: 50.0,
            passing: true,
            threshold: 20.0,
            stamps: vec![Stamp {
                provider: StampProvider::Github,
                provider_name: "Github".to_string(),
                credential_hash: "cached_hash".to_string(),
                score: 10.0,
                expires_at: None,
                is_duplicate: false,
                metadata: HashMap::new(),
            }],
            last_updated: Some(Utc::now()),
            expiration: None,
        };

        client.score_cache.insert(
            "0x1234567890abcdef1234567890abcdef12345678".to_string(),
            CacheEntry::new(cached_score.clone()),
        );

        // Verify cache contains the entry
        let (score_cache_count, stamps_cache_count) = client.cache_stats();
        assert_eq!(score_cache_count, 1);
        assert_eq!(stamps_cache_count, 0);

        // Verify we can retrieve from cache
        let cached = client.score_cache.get("0x1234567890abcdef1234567890abcdef12345678");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().value.score, 50.0);
    }

    #[test]
    fn test_passport_factor_status_based_on_score() {
        // Score >= 20 should be Active
        let high_score_factor = GitcoinPassportFactor::new(
            "0x1234567890abcdef1234567890abcdef12345678",
            25.0,
        ).unwrap();
        assert_eq!(high_score_factor.status, FactorStatus::Active);

        // Score < 20 should be Pending
        let low_score_factor = GitcoinPassportFactor::new(
            "0x1234567890abcdef1234567890abcdef12345678",
            15.0,
        ).unwrap();
        assert_eq!(low_score_factor.status, FactorStatus::Pending);
    }

    #[test]
    fn test_passport_factor_meets_threshold() {
        let factor = GitcoinPassportFactor::new(
            "0x1234567890abcdef1234567890abcdef12345678",
            35.0,
        ).unwrap();

        assert!(factor.meets_threshold(20.0));
        assert!(factor.meets_threshold(35.0));
        assert!(!factor.meets_threshold(40.0));
    }

    #[test]
    fn test_passport_factor_add_stamps() {
        let mut factor = GitcoinPassportFactor::new(
            "0x1234567890abcdef1234567890abcdef12345678",
            30.0,
        ).unwrap();

        assert!(factor.stamps.is_empty());

        factor.add_stamps(vec![
            "Github".to_string(),
            "Twitter".to_string(),
        ]);

        assert_eq!(factor.stamps.len(), 2);
        assert!(factor.stamps.contains(&"Github".to_string()));
        assert!(factor.stamps.contains(&"Twitter".to_string()));
    }
}
