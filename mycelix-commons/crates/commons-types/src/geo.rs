//! Geographic types shared across Commons domains
//!
//! Used by property-registry, housing, water-steward, and other
//! location-aware zomes.

use serde::{Deserialize, Serialize};

/// Geographic coordinates
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude_m: Option<f64>,
}

/// Physical address
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Address {
    pub street: String,
    pub city: String,
    pub state_province: String,
    pub postal_code: String,
    pub country: String,
    pub location: Option<GeoLocation>,
}
