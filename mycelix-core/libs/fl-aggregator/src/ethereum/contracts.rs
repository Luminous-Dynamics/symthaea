//! Contract ABIs and bindings
//!
//! This module provides type-safe contract bindings for Mycelix contracts.

#[cfg(feature = "ethereum")]
use ethers::utils::hex;

/// PaymentRouter ABI (simplified for key functions)
pub const PAYMENT_ROUTER_ABI: &str = r#"[
    {
        "inputs": [
            {"name": "recipients", "type": "address[]"},
            {"name": "shares", "type": "uint256[]"},
            {"name": "modelId", "type": "bytes32"}
        ],
        "name": "distributePayment",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "modelId", "type": "bytes32"}
        ],
        "name": "getModelPaymentInfo",
        "outputs": [
            {"name": "totalDistributed", "type": "uint256"},
            {"name": "lastDistribution", "type": "uint256"},
            {"name": "distributionCount", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": false,
        "inputs": [
            {"indexed": true, "name": "modelId", "type": "bytes32"},
            {"indexed": false, "name": "totalAmount", "type": "uint256"},
            {"indexed": false, "name": "recipientCount", "type": "uint256"}
        ],
        "name": "PaymentDistributed",
        "type": "event"
    }
]"#;

/// ReputationAnchor ABI (simplified for key functions)
pub const REPUTATION_ANCHOR_ABI: &str = r#"[
    {
        "inputs": [
            {"name": "agent", "type": "bytes32"},
            {"name": "score", "type": "uint256"},
            {"name": "evidenceHash", "type": "bytes32"}
        ],
        "name": "anchorReputation",
        "outputs": [
            {"name": "anchorId", "type": "uint256"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "agent", "type": "bytes32"}
        ],
        "name": "getReputation",
        "outputs": [
            {"name": "score", "type": "uint256"},
            {"name": "lastUpdate", "type": "uint256"},
            {"name": "evidenceHash", "type": "bytes32"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": false,
        "inputs": [
            {"indexed": true, "name": "agent", "type": "bytes32"},
            {"indexed": false, "name": "score", "type": "uint256"},
            {"indexed": false, "name": "anchorId", "type": "uint256"}
        ],
        "name": "ReputationAnchored",
        "type": "event"
    }
]"#;

/// ContributionRegistry ABI (simplified for key functions)
pub const CONTRIBUTION_REGISTRY_ABI: &str = r#"[
    {
        "inputs": [
            {"name": "modelId", "type": "bytes32"},
            {"name": "round", "type": "uint256"},
            {"name": "contributors", "type": "bytes32[]"},
            {"name": "contributions", "type": "uint256[]"},
            {"name": "proofHash", "type": "bytes32"}
        ],
        "name": "recordContributions",
        "outputs": [
            {"name": "recordId", "type": "uint256"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "modelId", "type": "bytes32"},
            {"name": "round", "type": "uint256"}
        ],
        "name": "getRoundContributions",
        "outputs": [
            {"name": "contributors", "type": "bytes32[]"},
            {"name": "contributions", "type": "uint256[]"},
            {"name": "proofHash", "type": "bytes32"},
            {"name": "timestamp", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": false,
        "inputs": [
            {"indexed": true, "name": "modelId", "type": "bytes32"},
            {"indexed": true, "name": "round", "type": "uint256"},
            {"indexed": false, "name": "recordId", "type": "uint256"}
        ],
        "name": "ContributionsRecorded",
        "type": "event"
    }
]"#;

/// Contract function selectors (first 4 bytes of keccak256 hash)
pub mod selectors {
    /// distributePayment(address[],uint256[],bytes32)
    pub const DISTRIBUTE_PAYMENT: [u8; 4] = [0x6d, 0x4c, 0xe6, 0x3c];
    /// anchorReputation(bytes32,uint256,bytes32)
    pub const ANCHOR_REPUTATION: [u8; 4] = [0x8a, 0x1b, 0x97, 0x4e];
    /// recordContributions(bytes32,uint256,bytes32[],uint256[],bytes32)
    pub const RECORD_CONTRIBUTIONS: [u8; 4] = [0x2f, 0x7a, 0x68, 0x91];
}

/// Model ID encoding helper
pub fn encode_model_id(model_id: &str) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(model_id.as_bytes());
    let result = hasher.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&result);
    bytes
}

/// Agent ID encoding helper
pub fn encode_agent_id(agent_id: &str) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(b"mycelix:agent:");
    hasher.update(agent_id.as_bytes());
    let result = hasher.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&result);
    bytes
}

/// Parse hex address to bytes
pub fn parse_address(address: &str) -> Result<[u8; 20], &'static str> {
    let address = address.strip_prefix("0x").unwrap_or(address);
    if address.len() != 40 {
        return Err("Invalid address length");
    }

    let mut bytes = [0u8; 20];
    for (i, chunk) in address.as_bytes().chunks(2).enumerate() {
        let hex_str = std::str::from_utf8(chunk).map_err(|_| "Invalid UTF-8")?;
        bytes[i] = u8::from_str_radix(hex_str, 16).map_err(|_| "Invalid hex")?;
    }
    Ok(bytes)
}

/// Format bytes as hex address
#[cfg(feature = "ethereum")]
pub fn format_address(bytes: &[u8; 20]) -> String {
    format!("0x{}", hex::encode(bytes))
}

/// Format bytes32 as hex
#[cfg(feature = "ethereum")]
pub fn format_bytes32(bytes: &[u8; 32]) -> String {
    format!("0x{}", hex::encode(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_model_id() {
        let model_id = "test-model-v1";
        let encoded = encode_model_id(model_id);
        assert_eq!(encoded.len(), 32);
        // Deterministic encoding
        assert_eq!(encode_model_id(model_id), encoded);
    }

    #[test]
    fn test_parse_address() {
        let address = "0x742d35Cc6634C0532925a3b844Bc9e7595f8fF2B";
        let bytes = parse_address(address).unwrap();
        assert_eq!(bytes.len(), 20);
    }

    #[test]
    fn test_format_address() {
        let bytes = [
            0x74, 0x2d, 0x35, 0xcc, 0x66, 0x34, 0xc0, 0x53, 0x29, 0x25,
            0xa3, 0xb8, 0x44, 0xbc, 0x9e, 0x75, 0x95, 0xf8, 0xff, 0x2b
        ];
        let address = format_address(&bytes);
        assert!(address.starts_with("0x"));
        assert_eq!(address.len(), 42);
    }
}
