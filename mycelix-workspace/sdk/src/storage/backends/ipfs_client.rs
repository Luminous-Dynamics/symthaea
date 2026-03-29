// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real IPFS HTTP Client
//!
//! Implements actual HTTP calls to IPFS daemon API.
//! Requires the `std` feature for `ureq` HTTP client.

#[cfg(feature = "std")]
use serde::Deserialize;

/// IPFS API response for /api/v0/add
#[cfg(feature = "std")]
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct AddResponse {
    /// Content identifier (CID) of the added content
    pub hash: String,
    /// Name of the file (empty for raw data)
    pub name: String,
    /// Size of the content in bytes
    pub size: String,
}

/// IPFS API response for /api/v0/pin/add
#[cfg(feature = "std")]
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PinAddResponse {
    /// List of pinned CIDs
    pub pins: Vec<String>,
}

/// IPFS API response for /api/v0/pin/rm
#[cfg(feature = "std")]
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PinRmResponse {
    /// List of unpinned CIDs
    pub pins: Vec<String>,
}

/// IPFS API error response
#[cfg(feature = "std")]
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct IpfsError {
    /// Error message
    pub message: String,
    /// Error code
    pub code: i32,
}

/// Result type for IPFS operations
#[cfg(feature = "std")]
pub type IpfsResult<T> = Result<T, IpfsClientError>;

/// IPFS client error types
#[cfg(feature = "std")]
#[derive(Debug)]
pub enum IpfsClientError {
    /// Network or connection error
    Network(String),
    /// IPFS daemon returned an error
    Api(IpfsError),
    /// Failed to parse response
    Parse(String),
    /// CID mismatch (content verification failed)
    CidMismatch {
        /// Expected CID based on local computation
        expected: String,
        /// Actual CID returned by IPFS daemon
        actual: String,
    },
    /// Timeout waiting for response
    Timeout,
}

#[cfg(feature = "std")]
impl std::fmt::Display for IpfsClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IpfsClientError::Network(msg) => write!(f, "IPFS network error: {}", msg),
            IpfsClientError::Api(err) => {
                write!(f, "IPFS API error: {} (code {})", err.message, err.code)
            }
            IpfsClientError::Parse(msg) => write!(f, "IPFS parse error: {}", msg),
            IpfsClientError::CidMismatch { expected, actual } => {
                write!(f, "CID mismatch: expected {}, got {}", expected, actual)
            }
            IpfsClientError::Timeout => write!(f, "IPFS request timeout"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for IpfsClientError {}

/// Real IPFS HTTP client
#[cfg(feature = "std")]
pub struct IpfsClient {
    /// Base URL for IPFS API (e.g., "http://localhost:5001")
    api_endpoint: String,
    /// Request timeout in seconds
    timeout_secs: u64,
}

#[cfg(feature = "std")]
impl IpfsClient {
    /// Create a new IPFS client
    pub fn new(api_endpoint: impl Into<String>, timeout_ms: u64) -> Self {
        Self {
            api_endpoint: api_endpoint.into().trim_end_matches('/').to_string(),
            timeout_secs: timeout_ms / 1000,
        }
    }

    /// Add content to IPFS
    ///
    /// Calls POST /api/v0/add with multipart form data
    pub fn add(&self, content: &[u8]) -> IpfsResult<AddResponse> {
        let url = format!("{}/api/v0/add?pin=false&quiet=false", self.api_endpoint);

        // Build multipart form
        let boundary = "----MycelixIpfsBoundary";
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"file\"; filename=\"data\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
        body.extend_from_slice(content);
        body.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

        let response = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .set(
                "Content-Type",
                &format!("multipart/form-data; boundary={}", boundary),
            )
            .send_bytes(&body)
            .map_err(|e| match e {
                ureq::Error::Status(_, resp) => {
                    let body = resp.into_string().unwrap_or_default();
                    if let Ok(err) = serde_json::from_str::<IpfsError>(&body) {
                        IpfsClientError::Api(err)
                    } else {
                        IpfsClientError::Network(body)
                    }
                }
                ureq::Error::Transport(t) => {
                    if t.kind() == ureq::ErrorKind::Io {
                        IpfsClientError::Timeout
                    } else {
                        IpfsClientError::Network(t.to_string())
                    }
                }
            })?;

        let body = response
            .into_string()
            .map_err(|e| IpfsClientError::Parse(e.to_string()))?;

        serde_json::from_str(&body)
            .map_err(|e| IpfsClientError::Parse(format!("Failed to parse add response: {}", e)))
    }

    /// Add and pin content to IPFS
    pub fn add_and_pin(&self, content: &[u8]) -> IpfsResult<AddResponse> {
        let url = format!("{}/api/v0/add?pin=true&quiet=false", self.api_endpoint);

        // Build multipart form
        let boundary = "----MycelixIpfsBoundary";
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"file\"; filename=\"data\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
        body.extend_from_slice(content);
        body.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());

        let response = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .set(
                "Content-Type",
                &format!("multipart/form-data; boundary={}", boundary),
            )
            .send_bytes(&body)
            .map_err(|e| match e {
                ureq::Error::Status(_, resp) => {
                    let body = resp.into_string().unwrap_or_default();
                    if let Ok(err) = serde_json::from_str::<IpfsError>(&body) {
                        IpfsClientError::Api(err)
                    } else {
                        IpfsClientError::Network(body)
                    }
                }
                ureq::Error::Transport(t) => IpfsClientError::Network(t.to_string()),
            })?;

        let body = response
            .into_string()
            .map_err(|e| IpfsClientError::Parse(e.to_string()))?;

        serde_json::from_str(&body)
            .map_err(|e| IpfsClientError::Parse(format!("Failed to parse add response: {}", e)))
    }

    /// Get content by CID
    ///
    /// Calls POST /api/v0/cat?arg={cid}
    pub fn cat(&self, cid: &str) -> IpfsResult<Vec<u8>> {
        let url = format!("{}/api/v0/cat?arg={}", self.api_endpoint, cid);

        let response = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .call()
            .map_err(|e| match e {
                ureq::Error::Status(_, resp) => {
                    let body = resp.into_string().unwrap_or_default();
                    if let Ok(err) = serde_json::from_str::<IpfsError>(&body) {
                        IpfsClientError::Api(err)
                    } else {
                        IpfsClientError::Network(body)
                    }
                }
                ureq::Error::Transport(t) => IpfsClientError::Network(t.to_string()),
            })?;

        let mut bytes = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut bytes)
            .map_err(|e| IpfsClientError::Parse(e.to_string()))?;

        Ok(bytes)
    }

    /// Pin content by CID
    ///
    /// Calls POST /api/v0/pin/add?arg={cid}
    pub fn pin_add(&self, cid: &str) -> IpfsResult<PinAddResponse> {
        let url = format!("{}/api/v0/pin/add?arg={}", self.api_endpoint, cid);

        let response = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .call()
            .map_err(|e| match e {
                ureq::Error::Status(_, resp) => {
                    let body = resp.into_string().unwrap_or_default();
                    if let Ok(err) = serde_json::from_str::<IpfsError>(&body) {
                        IpfsClientError::Api(err)
                    } else {
                        IpfsClientError::Network(body)
                    }
                }
                ureq::Error::Transport(t) => IpfsClientError::Network(t.to_string()),
            })?;

        let body = response
            .into_string()
            .map_err(|e| IpfsClientError::Parse(e.to_string()))?;

        serde_json::from_str(&body)
            .map_err(|e| IpfsClientError::Parse(format!("Failed to parse pin add response: {}", e)))
    }

    /// Unpin content by CID
    ///
    /// Calls POST /api/v0/pin/rm?arg={cid}
    pub fn pin_rm(&self, cid: &str) -> IpfsResult<PinRmResponse> {
        let url = format!("{}/api/v0/pin/rm?arg={}", self.api_endpoint, cid);

        let response = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .call()
            .map_err(|e| match e {
                ureq::Error::Status(_, resp) => {
                    let body = resp.into_string().unwrap_or_default();
                    if let Ok(err) = serde_json::from_str::<IpfsError>(&body) {
                        IpfsClientError::Api(err)
                    } else {
                        IpfsClientError::Network(body)
                    }
                }
                ureq::Error::Transport(t) => IpfsClientError::Network(t.to_string()),
            })?;

        let body = response
            .into_string()
            .map_err(|e| IpfsClientError::Parse(e.to_string()))?;

        serde_json::from_str(&body)
            .map_err(|e| IpfsClientError::Parse(format!("Failed to parse pin rm response: {}", e)))
    }

    /// Check if IPFS daemon is reachable
    pub fn is_available(&self) -> bool {
        let url = format!("{}/api/v0/id", self.api_endpoint);
        ureq::post(&url)
            .timeout(std::time::Duration::from_secs(5))
            .call()
            .is_ok()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_ipfs_client_creation() {
        let client = IpfsClient::new("http://localhost:5001", 30000);
        assert_eq!(client.api_endpoint, "http://localhost:5001");
        assert_eq!(client.timeout_secs, 30);
    }

    #[test]
    fn test_ipfs_client_trailing_slash() {
        let client = IpfsClient::new("http://localhost:5001/", 30000);
        assert_eq!(client.api_endpoint, "http://localhost:5001");
    }

    // Integration tests would require a running IPFS daemon
    // These are marked as ignored by default
    #[test]
    #[ignore]
    fn test_ipfs_add_and_cat() {
        let client = IpfsClient::new("http://localhost:5001", 30000);

        if !client.is_available() {
            eprintln!("IPFS daemon not available, skipping test");
            return;
        }

        let content = b"Hello, Mycelix IPFS!";
        let add_result = client.add(content).expect("Failed to add content");

        assert!(!add_result.hash.is_empty());

        let retrieved = client.cat(&add_result.hash).expect("Failed to get content");
        assert_eq!(retrieved, content);
    }

    #[test]
    #[ignore]
    fn test_ipfs_pin_operations() {
        let client = IpfsClient::new("http://localhost:5001", 30000);

        if !client.is_available() {
            eprintln!("IPFS daemon not available, skipping test");
            return;
        }

        let content = b"Pin test content";
        let add_result = client.add(content).expect("Failed to add content");

        // Pin
        let pin_result = client.pin_add(&add_result.hash).expect("Failed to pin");
        assert!(pin_result.pins.contains(&add_result.hash));

        // Unpin
        let unpin_result = client.pin_rm(&add_result.hash).expect("Failed to unpin");
        assert!(unpin_result.pins.contains(&add_result.hash));
    }
}
