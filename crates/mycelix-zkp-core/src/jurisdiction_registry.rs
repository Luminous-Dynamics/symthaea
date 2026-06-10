// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Published jurisdiction bounding-box sets. Non-canonical: anyone can
//! publish a registry; verifiers pick which ones they trust. Seeds
//! this crate ships two starter registries (US tax residency and
//! South African SARS) as JSON-compatible static data.
//!
//! The registry has no governance authority — it is a shared
//! *convention* for naming the same physical region the same way, so
//! that proofs and verifiers reference the same `JurisdictionBox` ids.
//!
//! In practice, a community or a regulator will publish a registry,
//! users will import it, and verifiers will assert "I accept boxes
//! from registry X." The registry itself is content-addressed via a
//! BLAKE3 hash so mismatched versions fail loudly.

#[cfg(feature = "backend-winterfell")]
use crate::circuits::jurisdiction_proof::JurisdictionBox;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// A named collection of jurisdiction bounding boxes.
#[cfg(feature = "backend-winterfell")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JurisdictionRegistry {
    /// Human-readable registry name.
    pub name: String,
    /// Semver-style version string.
    pub version: String,
    /// ISO 8601 publication timestamp.
    pub published_at: String,
    /// Optional publisher DID or web URL.
    pub publisher: Option<String>,
    /// All boxes in this registry, indexed by `JurisdictionBox::id`.
    pub boxes: Vec<JurisdictionBox>,
}

#[cfg(feature = "backend-winterfell")]
impl JurisdictionRegistry {
    /// Content-addressed digest over the canonical serialization of
    /// the box list. Two registries with the same boxes have the same
    /// digest; verifiers should compare digests when reporting "I
    /// accept registry X version Y."
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"MYCELIX-JURISDICTION-REGISTRY:v1:");
        hasher.update(self.name.as_bytes());
        hasher.update(b":");
        hasher.update(self.version.as_bytes());
        hasher.update(b":");
        // Order-independent: hash the sorted ids to avoid accidental
        // reordering changing the digest.
        let mut ids: Vec<&str> = self.boxes.iter().map(|b| b.id.as_str()).collect();
        ids.sort_unstable();
        for id in ids {
            let b = self.boxes.iter().find(|b| b.id == id).expect("id present");
            hasher.update(b.id.as_bytes());
            hasher.update(b":");
            hasher.update(b.lat_min_biased.to_le_bytes());
            hasher.update(b.lat_max_biased.to_le_bytes());
            hasher.update(b.lng_min_biased.to_le_bytes());
            hasher.update(b.lng_max_biased.to_le_bytes());
            hasher.update(b";");
        }
        let digest = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    }

    /// Find a box by its id.
    pub fn box_by_id(&self, id: &str) -> Option<&JurisdictionBox> {
        self.boxes.iter().find(|b| b.id == id)
    }

    /// All boxes whose ids begin with a given prefix. Useful for
    /// "list all boxes in the US tax-residency set" queries.
    pub fn boxes_with_prefix(&self, prefix: &str) -> Vec<&JurisdictionBox> {
        self.boxes
            .iter()
            .filter(|b| b.id.starts_with(prefix))
            .collect()
    }

    /// All boxes that contain the given decimal-degree coordinate.
    /// A coordinate may be inside multiple overlapping boxes (common
    /// near borders or in nested registries).
    pub fn boxes_containing(&self, lat_degrees: f64, lng_degrees: f64) -> Vec<&JurisdictionBox> {
        self.boxes
            .iter()
            .filter(|b| b.contains_degrees(lat_degrees, lng_degrees))
            .collect()
    }

    /// Total number of boxes.
    pub fn len(&self) -> usize {
        self.boxes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }

    /// Build a lookup index by id. O(n) to construct, O(1) per lookup.
    pub fn index(&self) -> HashMap<&str, &JurisdictionBox> {
        self.boxes.iter().map(|b| (b.id.as_str(), b)).collect()
    }
}

/// Ship two starter registries as seeds. Real deployments SHOULD
/// replace these with community-governed published sets. These seeds
/// are deliberately *coarse* — single bounding boxes over the
/// mainland extent — so the demo flow works without pulling in a
/// geodata dependency. Fine-grained boxes are future work.
#[cfg(feature = "backend-winterfell")]
pub fn seed_registry_us_tax_residency() -> JurisdictionRegistry {
    JurisdictionRegistry {
        name: "US-tax-residency".to_string(),
        version: "v1-seed".to_string(),
        published_at: "2026-04-18T00:00:00Z".to_string(),
        publisher: Some("did:mycelix:luminous-dynamics".to_string()),
        boxes: vec![
            // CONUS — contiguous United States. Coarse.
            JurisdictionBox::from_degrees(
                "US-tax-residency-v1-conus",
                24.396308, // south Florida
                49.384358, // Canadian border
                -125.0,    // western coast
                -66.93457, // eastern seaboard
            )
            .expect("valid box"),
            // Alaska.
            JurisdictionBox::from_degrees(
                "US-tax-residency-v1-alaska",
                51.214183,
                71.538800,
                -179.148909,
                -129.979510,
            )
            .expect("valid box"),
            // Hawaii.
            JurisdictionBox::from_degrees(
                "US-tax-residency-v1-hawaii",
                18.776344,
                22.236,
                -160.247,
                -154.806,
            )
            .expect("valid box"),
        ],
    }
}

#[cfg(feature = "backend-winterfell")]
pub fn seed_registry_sa_sars() -> JurisdictionRegistry {
    JurisdictionRegistry {
        name: "ZA-SARS".to_string(),
        version: "v1-seed".to_string(),
        published_at: "2026-04-18T00:00:00Z".to_string(),
        publisher: Some("did:mycelix:luminous-dynamics".to_string()),
        boxes: vec![
            // South Africa mainland. Coarse.
            JurisdictionBox::from_degrees(
                "ZA-SARS-v1-mainland",
                -34.833333,
                -22.125,
                16.448056,
                32.891667,
            )
            .expect("valid box"),
        ],
    }
}

#[cfg(all(test, feature = "backend-winterfell"))]
mod tests {
    use super::*;

    #[test]
    fn us_seed_has_three_boxes() {
        let r = seed_registry_us_tax_residency();
        assert_eq!(r.len(), 3);
        assert!(r.box_by_id("US-tax-residency-v1-conus").is_some());
        assert!(r.box_by_id("US-tax-residency-v1-alaska").is_some());
        assert!(r.box_by_id("US-tax-residency-v1-hawaii").is_some());
    }

    #[test]
    fn sa_seed_has_one_box() {
        let r = seed_registry_sa_sars();
        assert_eq!(r.len(), 1);
        assert!(r.box_by_id("ZA-SARS-v1-mainland").is_some());
    }

    #[test]
    fn prefix_query_works() {
        let r = seed_registry_us_tax_residency();
        let matches = r.boxes_with_prefix("US-tax-residency-v1-");
        assert_eq!(matches.len(), 3);
        let nomatch = r.boxes_with_prefix("ZA-");
        assert!(nomatch.is_empty());
    }

    #[test]
    fn containment_query_for_roodepoort_za() {
        let r = seed_registry_sa_sars();
        let matches = r.boxes_containing(-26.1625, 27.8725);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, "ZA-SARS-v1-mainland");
    }

    #[test]
    fn containment_query_for_seattle_us() {
        let r = seed_registry_us_tax_residency();
        let matches = r.boxes_containing(47.6062, -122.3321);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, "US-tax-residency-v1-conus");
    }

    #[test]
    fn digest_deterministic_across_reorderings() {
        let mut a = seed_registry_us_tax_residency();
        let d_a = a.digest();
        a.boxes.reverse();
        let d_a_rev = a.digest();
        assert_eq!(d_a, d_a_rev);
    }

    #[test]
    fn digest_distinguishes_registries() {
        let us = seed_registry_us_tax_residency();
        let za = seed_registry_sa_sars();
        assert_ne!(us.digest(), za.digest());
    }

    #[test]
    fn empty_registry_has_len_zero() {
        let r = JurisdictionRegistry {
            name: "empty".to_string(),
            version: "0".to_string(),
            published_at: "2026-04-18T00:00:00Z".to_string(),
            publisher: None,
            boxes: vec![],
        };
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn index_lookup_round_trips() {
        let r = seed_registry_us_tax_residency();
        let idx = r.index();
        assert_eq!(idx.len(), 3);
        assert_eq!(
            idx["US-tax-residency-v1-conus"].id,
            "US-tax-residency-v1-conus"
        );
    }
}
