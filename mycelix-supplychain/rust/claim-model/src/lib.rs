// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claim and VC data models for Mycelix supply chain provenance.
//!
//! This crate defines the core types and validation logic for supply chain events,
//! Verifiable Credentials, and DKG claims.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Schema validation failed: {0}")]
    SchemaValidation(String),
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("Invalid value for field {field}: {reason}")]
    InvalidValue { field: String, reason: String },
}

/// Supply chain event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EventType {
    Produced,
    Transformed,
    Shipped,
    Received,
    Certified,
}

impl std::str::FromStr for EventType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "PRODUCED" => Ok(EventType::Produced),
            "TRANSFORMED" => Ok(EventType::Transformed),
            "SHIPPED" => Ok(EventType::Shipped),
            "RECEIVED" => Ok(EventType::Received),
            "CERTIFIED" => Ok(EventType::Certified),
            _ => Err(format!("Invalid event type: {}", s)),
        }
    }
}

/// Facility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Facility {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<Location>,
}

/// Geographic location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lat: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lon: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
}

/// Shipment details for SHIPPED/RECEIVED events
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Shipment {
    pub shipment_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub carrier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tracking_number: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub destination: Option<String>,
}

/// Certification details for CERTIFIED events
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Certification {
    pub cert_type: String,
    pub cert_body: String,
    pub cert_id: String,
    pub valid_from: DateTime<Utc>,
    pub valid_until: DateTime<Utc>,
}

/// Credential subject for supply chain events
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialSubject {
    pub event_type: EventType,
    pub product_id: String,
    pub batch_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prev_batch_ids: Option<Vec<String>>,
    pub quantity: f64,
    pub unit: String,
    pub facility: Facility,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shipment: Option<Shipment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub certification: Option<Certification>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Verifiable Credential for supply chain events
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SupplyEventVC {
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    #[serde(rename = "type")]
    pub vc_type: Vec<String>,
    pub issuer: String,
    pub issuance_date: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expiration_date: Option<DateTime<Utc>>,
    pub credential_subject: CredentialSubject,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof: Option<serde_json::Value>,
}

impl SupplyEventVC {
    /// Validate the VC structure
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Check required context
        if !self.context.contains(&"https://www.w3.org/2018/credentials/v1".to_string()) {
            return Err(ValidationError::SchemaValidation(
                "Missing required @context".to_string(),
            ));
        }

        // Check type includes VerifiableCredential
        if !self.vc_type.contains(&"VerifiableCredential".to_string()) {
            return Err(ValidationError::SchemaValidation(
                "type must include VerifiableCredential".to_string(),
            ));
        }

        // Check issuer is a DID
        if !self.issuer.starts_with("did:") {
            return Err(ValidationError::InvalidValue {
                field: "issuer".to_string(),
                reason: "Must be a DID".to_string(),
            });
        }

        // Check quantity is positive
        if self.credential_subject.quantity <= 0.0 {
            return Err(ValidationError::InvalidValue {
                field: "quantity".to_string(),
                reason: "Must be positive".to_string(),
            });
        }

        // Check TRANSFORMED events have prevBatchIds
        if self.credential_subject.event_type == EventType::Transformed {
            if self.credential_subject.prev_batch_ids.is_none()
                || self
                    .credential_subject
                    .prev_batch_ids
                    .as_ref()
                    .unwrap()
                    .is_empty()
            {
                return Err(ValidationError::MissingField(
                    "prevBatchIds required for TRANSFORMED events".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Lineage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lineage {
    pub hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_claims: Option<Vec<String>>,
}

/// DKG Claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DkgClaim {
    pub id: String,
    #[serde(rename = "type")]
    pub claim_type: String,
    pub issuer: String,
    pub subject: Subject,
    pub assertion: Assertion,
    pub evidence: Evidence,
    pub lineage: Lineage,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Subject {
    pub batch_id: String,
    pub product_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Assertion {
    pub event_type: EventType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub facility_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Evidence {
    pub vc_jwt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_documents: Option<Vec<Document>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    #[serde(rename = "type")]
    pub doc_type: String,
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

impl DkgClaim {
    /// Create a new claim from a VC
    pub fn from_vc(vc: &SupplyEventVC, vc_jwt: String, previous_claims: Option<Vec<String>>) -> Self {
        let id = Uuid::new_v4().to_string();
        let lineage_hash = Self::compute_lineage_hash(&vc_jwt, &previous_claims);

        Self {
            id: id.clone(),
            claim_type: "SupplyChainClaim".to_string(),
            issuer: vc.issuer.clone(),
            subject: Subject {
                batch_id: vc.credential_subject.batch_id.clone(),
                product_id: vc.credential_subject.product_id.clone(),
            },
            assertion: Assertion {
                event_type: vc.credential_subject.event_type.clone(),
                quantity: Some(vc.credential_subject.quantity),
                unit: Some(vc.credential_subject.unit.clone()),
                facility_id: Some(vc.credential_subject.facility.id.clone()),
            },
            evidence: Evidence {
                vc_jwt,
                additional_documents: None,
            },
            lineage: Lineage {
                hash: lineage_hash,
                previous_claims,
            },
            timestamp: Utc::now(),
            confidence: Some(1.0),
            metadata: None,
        }
    }

    fn compute_lineage_hash(vc_jwt: &str, previous_claims: &Option<Vec<String>>) -> String {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(vc_jwt.as_bytes());

        if let Some(prev) = previous_claims {
            for claim_id in prev {
                hasher.update(claim_id.as_bytes());
            }
        }

        let result = hasher.finalize();
        format!("{:x}", result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_valid_vc(event_type: EventType, prev_batch_ids: Option<Vec<String>>) -> SupplyEventVC {
        SupplyEventVC {
            context: vec!["https://www.w3.org/2018/credentials/v1".to_string()],
            vc_type: vec!["VerifiableCredential".to_string()],
            issuer: "did:mycelix:org:test".to_string(),
            issuance_date: Utc::now(),
            expiration_date: None,
            credential_subject: CredentialSubject {
                event_type,
                product_id: "SKU-001".to_string(),
                batch_id: "BATCH-001".to_string(),
                prev_batch_ids,
                quantity: 100.0,
                unit: "kg".to_string(),
                facility: Facility {
                    id: "FAC-001".to_string(),
                    name: "Test Facility".to_string(),
                    location: None,
                },
                timestamp: Utc::now(),
                shipment: None,
                certification: None,
                metadata: None,
            },
            proof: None,
        }
    }

    #[test]
    fn test_vc_validation_produced() {
        let vc = create_valid_vc(EventType::Produced, None);
        assert!(vc.validate().is_ok());
    }

    #[test]
    fn test_vc_validation_transformed_with_parents() {
        let vc = create_valid_vc(EventType::Transformed, Some(vec!["BATCH-000".to_string()]));
        assert!(vc.validate().is_ok());
    }

    #[test]
    fn test_vc_validation_transformed_without_parents_fails() {
        let vc = create_valid_vc(EventType::Transformed, None);
        let result = vc.validate();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ValidationError::MissingField(_)));
    }

    #[test]
    fn test_vc_validation_missing_context() {
        let mut vc = create_valid_vc(EventType::Produced, None);
        vc.context = vec!["https://example.com".to_string()];
        let result = vc.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_vc_validation_missing_vc_type() {
        let mut vc = create_valid_vc(EventType::Produced, None);
        vc.vc_type = vec!["SomeOtherType".to_string()];
        let result = vc.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_vc_validation_invalid_issuer() {
        let mut vc = create_valid_vc(EventType::Produced, None);
        vc.issuer = "not-a-did".to_string();
        let result = vc.validate();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ValidationError::InvalidValue { .. }));
    }

    #[test]
    fn test_vc_validation_negative_quantity() {
        let mut vc = create_valid_vc(EventType::Produced, None);
        vc.credential_subject.quantity = -10.0;
        let result = vc.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_vc_validation_zero_quantity() {
        let mut vc = create_valid_vc(EventType::Produced, None);
        vc.credential_subject.quantity = 0.0;
        let result = vc.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_event_type_serialization() {
        let produced = EventType::Produced;
        let json = serde_json::to_string(&produced).unwrap();
        assert_eq!(json, "\"PRODUCED\"");

        let transformed = EventType::Transformed;
        let json = serde_json::to_string(&transformed).unwrap();
        assert_eq!(json, "\"TRANSFORMED\"");
    }

    #[test]
    fn test_event_type_deserialization() {
        let produced: EventType = serde_json::from_str("\"PRODUCED\"").unwrap();
        assert_eq!(produced, EventType::Produced);

        let shipped: EventType = serde_json::from_str("\"SHIPPED\"").unwrap();
        assert_eq!(shipped, EventType::Shipped);
    }

    #[test]
    fn test_dkg_claim_from_vc() {
        let vc = create_valid_vc(EventType::Produced, None);
        let vc_jwt = "mock.jwt.token".to_string();
        let prev_claims = Some(vec!["claim-123".to_string()]);

        let claim = DkgClaim::from_vc(&vc, vc_jwt.clone(), prev_claims.clone());

        assert_eq!(claim.claim_type, "SupplyChainClaim");
        assert_eq!(claim.issuer, vc.issuer);
        assert_eq!(claim.subject.batch_id, vc.credential_subject.batch_id);
        assert_eq!(claim.subject.product_id, vc.credential_subject.product_id);
        assert_eq!(claim.assertion.event_type, EventType::Produced);
        assert_eq!(claim.evidence.vc_jwt, vc_jwt);
        assert_eq!(claim.lineage.previous_claims, prev_claims);
        assert_eq!(claim.confidence, Some(1.0));
        assert!(!claim.lineage.hash.is_empty());
    }

    #[test]
    fn test_dkg_claim_lineage_hash_deterministic() {
        let vc = create_valid_vc(EventType::Produced, None);
        let vc_jwt = "mock.jwt.token".to_string();
        let prev_claims = Some(vec!["claim-123".to_string()]);

        let claim1 = DkgClaim::from_vc(&vc, vc_jwt.clone(), prev_claims.clone());
        let claim2 = DkgClaim::from_vc(&vc, vc_jwt.clone(), prev_claims.clone());

        // Same inputs should produce same lineage hash
        assert_eq!(claim1.lineage.hash, claim2.lineage.hash);
    }

    #[test]
    fn test_dkg_claim_lineage_hash_different() {
        let vc = create_valid_vc(EventType::Produced, None);
        let vc_jwt1 = "mock.jwt.token.1".to_string();
        let vc_jwt2 = "mock.jwt.token.2".to_string();

        let claim1 = DkgClaim::from_vc(&vc, vc_jwt1, None);
        let claim2 = DkgClaim::from_vc(&vc, vc_jwt2, None);

        // Different JWTs should produce different hashes
        assert_ne!(claim1.lineage.hash, claim2.lineage.hash);
    }

    #[test]
    fn test_facility_with_location() {
        let facility = Facility {
            id: "FAC-001".to_string(),
            name: "Test Facility".to_string(),
            location: Some(Location {
                lat: Some(37.7749),
                lon: Some(-122.4194),
                address: Some("123 Main St".to_string()),
                country: Some("US".to_string()),
            }),
        };

        let json = serde_json::to_string(&facility).unwrap();
        let parsed: Facility = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, facility.id);
        assert_eq!(parsed.location.as_ref().unwrap().lat, facility.location.as_ref().unwrap().lat);
    }

    #[test]
    fn test_metadata_serialization() {
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), serde_json::json!("value1"));
        metadata.insert("key2".to_string(), serde_json::json!(42));

        let json = serde_json::to_string(&metadata).unwrap();
        let parsed: HashMap<String, serde_json::Value> = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.get("key1").unwrap(), "value1");
        assert_eq!(parsed.get("key2").unwrap(), 42);
    }

    #[test]
    fn test_shipment_serialization() {
        let shipment = Shipment {
            shipment_id: "SHIP-001".to_string(),
            carrier: Some("ACME Logistics".to_string()),
            tracking_number: Some("TRACK-123".to_string()),
            origin: Some("FAC-A".to_string()),
            destination: Some("WH-B".to_string()),
        };

        let json = serde_json::to_string(&shipment).unwrap();
        let parsed: Shipment = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.shipment_id, shipment.shipment_id);
        assert_eq!(parsed.carrier, shipment.carrier);
    }

    #[test]
    fn test_certification_serialization() {
        let cert = Certification {
            cert_type: "ISO-9001".to_string(),
            cert_body: "Test Org".to_string(),
            cert_id: "CERT-001".to_string(),
            valid_from: Utc::now(),
            valid_until: Utc::now() + chrono::Duration::days(365),
        };

        let json = serde_json::to_string(&cert).unwrap();
        let parsed: Certification = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.cert_type, cert.cert_type);
        assert_eq!(parsed.cert_id, cert.cert_id);
    }
}
