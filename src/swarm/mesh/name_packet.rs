// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Name Query/Response Packets — Mesh Name Resolution Protocol
//!
//! Defines `PayloadType::NameQuery` (6) and `PayloadType::NameResponse` (7)
//! for resolving mesh names over the WisdomPacket transport.
//!
//! # Wire Format (in BinaryHV)
//!
//! ## NameQuery (PayloadType = 6)
//! ```text
//! Bytes 0-31:   name_hash  (BLAKE3 hash of canonical name)
//! Bytes 32-33:  query_id   (u16 LE, for response correlation)
//! ```
//!
//! ## NameResponse (PayloadType = 7)
//! ```text
//! Bytes 0-31:   name_hash     (echo back)
//! Bytes 32-33:  query_id      (echo back)
//! Bytes 34:     endpoint_type (0=Iroh, 1=LoRa, 2=Holochain, 3=IP)
//! Bytes 35-36:  endpoint_len  (u16 LE)
//! Bytes 37-292: endpoint_data (max 256 bytes)
//! Bytes 293-300: ttl_secs     (u64 LE)
//! ```

use symthaea_core::hdc::BinaryHV;

/// Parsed name query.
#[derive(Debug, Clone)]
pub struct NameQuery {
    /// BLAKE3 hash of the canonical mesh name.
    pub name_hash: [u8; 32],
    /// Query correlation ID.
    pub query_id: u16,
}

impl NameQuery {
    /// Encode into a BinaryHV for mesh transmission.
    pub fn encode(&self) -> BinaryHV {
        let mut bytes = [0u8; 2048];
        bytes[0..32].copy_from_slice(&self.name_hash);
        bytes[32..34].copy_from_slice(&self.query_id.to_le_bytes());
        BinaryHV(bytes)
    }

    /// Decode from a BinaryHV.
    pub fn decode(hv: &BinaryHV) -> Option<Self> {
        let bytes = &hv.0;
        if bytes.len() < 34 {
            return None;
        }
        let mut name_hash = [0u8; 32];
        name_hash.copy_from_slice(&bytes[0..32]);
        let query_id = u16::from_le_bytes(bytes[32..34].try_into().ok()?);
        Some(Self {
            name_hash,
            query_id,
        })
    }
}

/// Endpoint type discriminant for wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EndpointType {
    Iroh = 0,
    LoRa = 1,
    Holochain = 2,
    Ip = 3,
}

impl EndpointType {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Iroh),
            1 => Some(Self::LoRa),
            2 => Some(Self::Holochain),
            3 => Some(Self::Ip),
            _ => None,
        }
    }
}

/// Parsed name response.
#[derive(Debug, Clone)]
pub struct NameResponse {
    /// Echo of the query's name hash.
    pub name_hash: [u8; 32],
    /// Echo of the query's correlation ID.
    pub query_id: u16,
    /// Endpoint type.
    pub endpoint_type: EndpointType,
    /// Raw endpoint data.
    pub endpoint_data: Vec<u8>,
    /// TTL in seconds.
    pub ttl_secs: u64,
}

impl NameResponse {
    /// Encode into a BinaryHV.
    pub fn encode(&self) -> BinaryHV {
        let mut bytes = [0u8; 2048];
        bytes[0..32].copy_from_slice(&self.name_hash);
        bytes[32..34].copy_from_slice(&self.query_id.to_le_bytes());
        bytes[34] = self.endpoint_type as u8;
        let len = self.endpoint_data.len().min(256) as u16;
        bytes[35..37].copy_from_slice(&len.to_le_bytes());
        let data_len = len as usize;
        bytes[37..37 + data_len].copy_from_slice(&self.endpoint_data[..data_len]);
        bytes[293..301].copy_from_slice(&self.ttl_secs.to_le_bytes());
        BinaryHV(bytes)
    }

    /// Decode from a BinaryHV.
    pub fn decode(hv: &BinaryHV) -> Option<Self> {
        let bytes = &hv.0;
        if bytes.len() < 301 {
            return None;
        }
        let mut name_hash = [0u8; 32];
        name_hash.copy_from_slice(&bytes[0..32]);
        let query_id = u16::from_le_bytes(bytes[32..34].try_into().ok()?);
        let endpoint_type = EndpointType::from_byte(bytes[34])?;
        let len = u16::from_le_bytes(bytes[35..37].try_into().ok()?) as usize;
        if len > 256 {
            return None;
        }
        let endpoint_data = bytes[37..37 + len].to_vec();
        let ttl_secs = u64::from_le_bytes(bytes[293..301].try_into().ok()?);
        Some(Self {
            name_hash,
            query_id,
            endpoint_type,
            endpoint_data,
            ttl_secs,
        })
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_query_roundtrip() {
        let query = NameQuery {
            name_hash: *blake3::hash(b"mycelix://test/node").as_bytes(),
            query_id: 42,
        };
        let hv = query.encode();
        let decoded = NameQuery::decode(&hv).unwrap();
        assert_eq!(decoded.name_hash, query.name_hash);
        assert_eq!(decoded.query_id, 42);
    }

    #[test]
    fn test_name_response_roundtrip() {
        let response = NameResponse {
            name_hash: [0xAB; 32],
            query_id: 123,
            endpoint_type: EndpointType::Iroh,
            endpoint_data: b"some-iroh-addr".to_vec(),
            ttl_secs: 3600,
        };
        let hv = response.encode();
        let decoded = NameResponse::decode(&hv).unwrap();
        assert_eq!(decoded.name_hash, [0xAB; 32]);
        assert_eq!(decoded.query_id, 123);
        assert_eq!(decoded.endpoint_type, EndpointType::Iroh);
        assert_eq!(decoded.endpoint_data, b"some-iroh-addr");
        assert_eq!(decoded.ttl_secs, 3600);
    }

    #[test]
    fn test_endpoint_type_roundtrip() {
        for &et in &[
            EndpointType::Iroh,
            EndpointType::LoRa,
            EndpointType::Holochain,
            EndpointType::Ip,
        ] {
            assert_eq!(EndpointType::from_byte(et as u8), Some(et));
        }
        assert_eq!(EndpointType::from_byte(99), None);
    }

    #[test]
    fn test_response_max_endpoint_data() {
        let response = NameResponse {
            name_hash: [0; 32],
            query_id: 0,
            endpoint_type: EndpointType::Ip,
            endpoint_data: vec![0xAA; 256],
            ttl_secs: 60,
        };
        let hv = response.encode();
        let decoded = NameResponse::decode(&hv).unwrap();
        assert_eq!(decoded.endpoint_data.len(), 256);
    }

    #[test]
    fn test_response_empty_endpoint() {
        let response = NameResponse {
            name_hash: [0; 32],
            query_id: 0,
            endpoint_type: EndpointType::LoRa,
            endpoint_data: vec![],
            ttl_secs: 0,
        };
        let hv = response.encode();
        let decoded = NameResponse::decode(&hv).unwrap();
        assert!(decoded.endpoint_data.is_empty());
    }
}
