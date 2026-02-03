//! Conjunction Data Message (CDM) Generator
//!
//! Implements CCSDS 508.0-B-1 Conjunction Data Message format for
//! space situational awareness data exchange.
//!
//! CDM is the industry standard for sharing conjunction assessment
//! data between space operators, agencies, and tracking entities.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// CCSDS CDM Version
pub const CDM_VERSION: &str = "1.0";

/// CDM Message Type
pub const CDM_MSG_TYPE: &str = "CDM";

/// Reference Frame Types per CCSDS standards
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdmRefFrame {
    /// Earth Mean Equator and Equinox of J2000
    EME2000,
    /// Geocentric Celestial Reference Frame
    GCRF,
    /// International Terrestrial Reference Frame
    ITRF,
    /// True Equator Mean Equinox
    TEME,
    /// True of Date
    TOD,
}

impl std::fmt::Display for CdmRefFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CdmRefFrame::EME2000 => write!(f, "EME2000"),
            CdmRefFrame::GCRF => write!(f, "GCRF"),
            CdmRefFrame::ITRF => write!(f, "ITRF"),
            CdmRefFrame::TEME => write!(f, "TEME"),
            CdmRefFrame::TOD => write!(f, "TOD"),
        }
    }
}

/// CDM Maneuverable status
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Maneuverable {
    Yes,
    No,
    Unknown,
}

impl std::fmt::Display for Maneuverable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Maneuverable::Yes => write!(f, "YES"),
            Maneuverable::No => write!(f, "NO"),
            Maneuverable::Unknown => write!(f, "N/A"),
        }
    }
}

/// Object metadata section of CDM
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CdmObjectMetadata {
    /// Object designator (NORAD ID or other identifier)
    pub object_designator: String,
    /// Catalog name (e.g., "SATCAT")
    pub catalog_name: String,
    /// Object name
    pub object_name: String,
    /// International designator (YYYY-NNNPP format)
    pub international_designator: String,
    /// Object type (PAYLOAD, ROCKET BODY, DEBRIS, UNKNOWN)
    pub object_type: String,
    /// Operator contact organization
    pub operator_organization: Option<String>,
    /// Operator phone number
    pub operator_phone: Option<String>,
    /// Operator email
    pub operator_email: Option<String>,
    /// Ephemeris name/source
    pub ephemeris_name: String,
    /// Covariance method
    pub covariance_method: String,
    /// Maneuverable status
    pub maneuverable: Maneuverable,
    /// Reference frame
    pub ref_frame: CdmRefFrame,
}

/// State vector at TCA
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CdmStateVector {
    /// Position X (km)
    pub x: f64,
    /// Position Y (km)
    pub y: f64,
    /// Position Z (km)
    pub z: f64,
    /// Velocity X (km/s)
    pub x_dot: f64,
    /// Velocity Y (km/s)
    pub y_dot: f64,
    /// Velocity Z (km/s)
    pub z_dot: f64,
}

/// Covariance matrix (RTN - Radial, In-Track, Cross-Track)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CdmCovariance {
    /// Radial variance (km^2)
    pub cr_r: f64,
    /// In-track/Radial covariance (km^2)
    pub ct_r: f64,
    /// In-track variance (km^2)
    pub ct_t: f64,
    /// Cross-track/Radial covariance (km^2)
    pub cn_r: f64,
    /// Cross-track/In-track covariance (km^2)
    pub cn_t: f64,
    /// Cross-track variance (km^2)
    pub cn_n: f64,
    /// Radial dot/Radial covariance (km^2/s)
    pub crdot_r: f64,
    /// Radial dot/In-track covariance (km^2/s)
    pub crdot_t: f64,
    /// Radial dot/Cross-track covariance (km^2/s)
    pub crdot_n: f64,
    /// Radial dot variance ((km/s)^2)
    pub crdot_rdot: f64,
    /// In-track dot/Radial covariance (km^2/s)
    pub ctdot_r: f64,
    /// In-track dot/In-track covariance (km^2/s)
    pub ctdot_t: f64,
    /// In-track dot/Cross-track covariance (km^2/s)
    pub ctdot_n: f64,
    /// In-track dot/Radial dot covariance ((km/s)^2)
    pub ctdot_rdot: f64,
    /// In-track dot variance ((km/s)^2)
    pub ctdot_tdot: f64,
    /// Cross-track dot/Radial covariance (km^2/s)
    pub cndot_r: f64,
    /// Cross-track dot/In-track covariance (km^2/s)
    pub cndot_t: f64,
    /// Cross-track dot/Cross-track covariance (km^2/s)
    pub cndot_n: f64,
    /// Cross-track dot/Radial dot covariance ((km/s)^2)
    pub cndot_rdot: f64,
    /// Cross-track dot/In-track dot covariance ((km/s)^2)
    pub cndot_tdot: f64,
    /// Cross-track dot variance ((km/s)^2)
    pub cndot_ndot: f64,
}

impl Default for CdmCovariance {
    fn default() -> Self {
        Self {
            cr_r: 0.0, ct_r: 0.0, ct_t: 0.0,
            cn_r: 0.0, cn_t: 0.0, cn_n: 0.0,
            crdot_r: 0.0, crdot_t: 0.0, crdot_n: 0.0, crdot_rdot: 0.0,
            ctdot_r: 0.0, ctdot_t: 0.0, ctdot_n: 0.0, ctdot_rdot: 0.0, ctdot_tdot: 0.0,
            cndot_r: 0.0, cndot_t: 0.0, cndot_n: 0.0, cndot_rdot: 0.0, cndot_tdot: 0.0, cndot_ndot: 0.0,
        }
    }
}

/// Object data section of CDM
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CdmObjectData {
    /// Object metadata
    pub metadata: CdmObjectMetadata,
    /// State vector at TCA
    pub state_vector: CdmStateVector,
    /// Covariance matrix (optional)
    pub covariance: Option<CdmCovariance>,
}

/// Complete CCSDS Conjunction Data Message
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConjunctionDataMessage {
    // --- Header ---
    /// CDM version
    pub ccsds_cdm_vers: String,
    /// Creation date/time
    pub creation_date: DateTime<Utc>,
    /// Originator
    pub originator: String,
    /// Message ID
    pub message_id: String,

    // --- Relative Metadata ---
    /// Time of Closest Approach
    pub tca: DateTime<Utc>,
    /// Miss distance at TCA (km)
    pub miss_distance: f64,
    /// Relative speed at TCA (m/s)
    pub relative_speed: f64,
    /// Relative position R (Radial) (m)
    pub relative_position_r: f64,
    /// Relative position T (In-track) (m)
    pub relative_position_t: f64,
    /// Relative position N (Cross-track) (m)
    pub relative_position_n: f64,
    /// Relative velocity R (m/s)
    pub relative_velocity_r: f64,
    /// Relative velocity T (m/s)
    pub relative_velocity_t: f64,
    /// Relative velocity N (m/s)
    pub relative_velocity_n: f64,

    // --- Probability ---
    /// Collision probability
    pub collision_probability: f64,
    /// Collision probability method (e.g., "FOSTER-1992", "ALFANO-2005")
    pub collision_probability_method: String,

    // --- Object Data ---
    /// Object 1 (typically the "protected" or "primary" object)
    pub object1: CdmObjectData,
    /// Object 2 (typically the "debris" or "secondary" object)
    pub object2: CdmObjectData,

    // --- Optional screening data ---
    /// Screening entry time (when conjunction first entered screening)
    pub screening_entry_time: Option<DateTime<Utc>>,
    /// Screening data source
    pub screening_data_source: Option<String>,
    /// Hard body radius (m) used in Pc calculation
    pub hard_body_radius: Option<f64>,
}

impl ConjunctionDataMessage {
    /// Create a new CDM builder
    pub fn builder() -> CdmBuilder {
        CdmBuilder::new()
    }

    /// Format CDM as CCSDS KVN (Key-Value Notation) format
    pub fn to_kvn(&self) -> String {
        let mut kvn = String::new();

        // Header
        kvn.push_str(&format!("CCSDS_CDM_VERS = {}\n", self.ccsds_cdm_vers));
        kvn.push_str(&format!("CREATION_DATE = {}\n", self.creation_date.format("%Y-%m-%dT%H:%M:%S%.3f")));
        kvn.push_str(&format!("ORIGINATOR = {}\n", self.originator));
        kvn.push_str(&format!("MESSAGE_ID = {}\n", self.message_id));
        kvn.push_str("\n");

        // Relative Metadata
        kvn.push_str(&format!("TCA = {}\n", self.tca.format("%Y-%m-%dT%H:%M:%S%.3f")));
        kvn.push_str(&format!("MISS_DISTANCE = {:.6} [km]\n", self.miss_distance));
        kvn.push_str(&format!("RELATIVE_SPEED = {:.6} [m/s]\n", self.relative_speed));
        kvn.push_str(&format!("RELATIVE_POSITION_R = {:.6} [m]\n", self.relative_position_r));
        kvn.push_str(&format!("RELATIVE_POSITION_T = {:.6} [m]\n", self.relative_position_t));
        kvn.push_str(&format!("RELATIVE_POSITION_N = {:.6} [m]\n", self.relative_position_n));
        kvn.push_str(&format!("RELATIVE_VELOCITY_R = {:.6} [m/s]\n", self.relative_velocity_r));
        kvn.push_str(&format!("RELATIVE_VELOCITY_T = {:.6} [m/s]\n", self.relative_velocity_t));
        kvn.push_str(&format!("RELATIVE_VELOCITY_N = {:.6} [m/s]\n", self.relative_velocity_n));
        kvn.push_str("\n");

        // Collision Probability
        kvn.push_str(&format!("COLLISION_PROBABILITY = {:.6e}\n", self.collision_probability));
        kvn.push_str(&format!("COLLISION_PROBABILITY_METHOD = {}\n", self.collision_probability_method));
        kvn.push_str("\n");

        // Object 1
        kvn.push_str("COMMENT Object 1 (Primary)\n");
        kvn.push_str(&self.format_object_kvn(&self.object1, "OBJECT1"));
        kvn.push_str("\n");

        // Object 2
        kvn.push_str("COMMENT Object 2 (Secondary)\n");
        kvn.push_str(&self.format_object_kvn(&self.object2, "OBJECT2"));

        kvn
    }

    fn format_object_kvn(&self, obj: &CdmObjectData, prefix: &str) -> String {
        let mut kvn = String::new();

        // Metadata
        kvn.push_str(&format!("{}_OBJECT_DESIGNATOR = {}\n", prefix, obj.metadata.object_designator));
        kvn.push_str(&format!("{}_CATALOG_NAME = {}\n", prefix, obj.metadata.catalog_name));
        kvn.push_str(&format!("{}_OBJECT_NAME = {}\n", prefix, obj.metadata.object_name));
        kvn.push_str(&format!("{}_INTERNATIONAL_DESIGNATOR = {}\n", prefix, obj.metadata.international_designator));
        kvn.push_str(&format!("{}_OBJECT_TYPE = {}\n", prefix, obj.metadata.object_type));
        kvn.push_str(&format!("{}_EPHEMERIS_NAME = {}\n", prefix, obj.metadata.ephemeris_name));
        kvn.push_str(&format!("{}_COVARIANCE_METHOD = {}\n", prefix, obj.metadata.covariance_method));
        kvn.push_str(&format!("{}_MANEUVERABLE = {}\n", prefix, obj.metadata.maneuverable));
        kvn.push_str(&format!("{}_REF_FRAME = {}\n", prefix, obj.metadata.ref_frame));

        // State Vector
        kvn.push_str(&format!("{}_X = {:.6} [km]\n", prefix, obj.state_vector.x));
        kvn.push_str(&format!("{}_Y = {:.6} [km]\n", prefix, obj.state_vector.y));
        kvn.push_str(&format!("{}_Z = {:.6} [km]\n", prefix, obj.state_vector.z));
        kvn.push_str(&format!("{}_X_DOT = {:.9} [km/s]\n", prefix, obj.state_vector.x_dot));
        kvn.push_str(&format!("{}_Y_DOT = {:.9} [km/s]\n", prefix, obj.state_vector.y_dot));
        kvn.push_str(&format!("{}_Z_DOT = {:.9} [km/s]\n", prefix, obj.state_vector.z_dot));

        // Covariance (if present)
        if let Some(cov) = &obj.covariance {
            kvn.push_str(&format!("{}_CR_R = {:.10e} [km**2]\n", prefix, cov.cr_r));
            kvn.push_str(&format!("{}_CT_R = {:.10e} [km**2]\n", prefix, cov.ct_r));
            kvn.push_str(&format!("{}_CT_T = {:.10e} [km**2]\n", prefix, cov.ct_t));
            kvn.push_str(&format!("{}_CN_R = {:.10e} [km**2]\n", prefix, cov.cn_r));
            kvn.push_str(&format!("{}_CN_T = {:.10e} [km**2]\n", prefix, cov.cn_t));
            kvn.push_str(&format!("{}_CN_N = {:.10e} [km**2]\n", prefix, cov.cn_n));
            // ... additional covariance terms would follow
        }

        kvn
    }

    /// Format CDM as JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Generate a unique message ID
    pub fn generate_message_id(primary_id: u32, secondary_id: u32, tca: DateTime<Utc>) -> String {
        format!(
            "CDM-{}-{}-{}",
            primary_id,
            secondary_id,
            tca.format("%Y%m%dT%H%M%S")
        )
    }
}

/// Builder for constructing CDM messages
pub struct CdmBuilder {
    originator: String,
    message_id: Option<String>,
    tca: Option<DateTime<Utc>>,
    miss_distance_km: f64,
    relative_speed_ms: f64,
    relative_position_rtn: [f64; 3],
    relative_velocity_rtn: [f64; 3],
    collision_probability: f64,
    pc_method: String,
    object1: Option<CdmObjectData>,
    object2: Option<CdmObjectData>,
    hard_body_radius: Option<f64>,
}

impl CdmBuilder {
    pub fn new() -> Self {
        Self {
            originator: "MYCELIX-SPACE".to_string(),
            message_id: None,
            tca: None,
            miss_distance_km: 0.0,
            relative_speed_ms: 0.0,
            relative_position_rtn: [0.0, 0.0, 0.0],
            relative_velocity_rtn: [0.0, 0.0, 0.0],
            collision_probability: 0.0,
            pc_method: "ALFANO-2005".to_string(),
            object1: None,
            object2: None,
            hard_body_radius: None,
        }
    }

    pub fn originator(mut self, originator: impl Into<String>) -> Self {
        self.originator = originator.into();
        self
    }

    pub fn message_id(mut self, id: impl Into<String>) -> Self {
        self.message_id = Some(id.into());
        self
    }

    pub fn tca(mut self, tca: DateTime<Utc>) -> Self {
        self.tca = Some(tca);
        self
    }

    pub fn miss_distance_km(mut self, dist: f64) -> Self {
        self.miss_distance_km = dist;
        self
    }

    pub fn relative_speed_ms(mut self, speed: f64) -> Self {
        self.relative_speed_ms = speed;
        self
    }

    pub fn relative_position_rtn(mut self, r: f64, t: f64, n: f64) -> Self {
        self.relative_position_rtn = [r, t, n];
        self
    }

    pub fn relative_velocity_rtn(mut self, r: f64, t: f64, n: f64) -> Self {
        self.relative_velocity_rtn = [r, t, n];
        self
    }

    pub fn collision_probability(mut self, pc: f64) -> Self {
        self.collision_probability = pc;
        self
    }

    pub fn pc_method(mut self, method: impl Into<String>) -> Self {
        self.pc_method = method.into();
        self
    }

    pub fn object1(mut self, obj: CdmObjectData) -> Self {
        self.object1 = Some(obj);
        self
    }

    pub fn object2(mut self, obj: CdmObjectData) -> Self {
        self.object2 = Some(obj);
        self
    }

    pub fn hard_body_radius_m(mut self, hbr: f64) -> Self {
        self.hard_body_radius = Some(hbr);
        self
    }

    pub fn build(self) -> Result<ConjunctionDataMessage, &'static str> {
        let tca = self.tca.ok_or("TCA is required")?;
        let object1 = self.object1.ok_or("Object 1 is required")?;
        let object2 = self.object2.ok_or("Object 2 is required")?;

        let message_id = self.message_id.unwrap_or_else(|| {
            let primary_id: u32 = object1.metadata.object_designator.parse().unwrap_or(0);
            let secondary_id: u32 = object2.metadata.object_designator.parse().unwrap_or(0);
            ConjunctionDataMessage::generate_message_id(primary_id, secondary_id, tca)
        });

        Ok(ConjunctionDataMessage {
            ccsds_cdm_vers: CDM_VERSION.to_string(),
            creation_date: Utc::now(),
            originator: self.originator,
            message_id,
            tca,
            miss_distance: self.miss_distance_km,
            relative_speed: self.relative_speed_ms,
            relative_position_r: self.relative_position_rtn[0],
            relative_position_t: self.relative_position_rtn[1],
            relative_position_n: self.relative_position_rtn[2],
            relative_velocity_r: self.relative_velocity_rtn[0],
            relative_velocity_t: self.relative_velocity_rtn[1],
            relative_velocity_n: self.relative_velocity_rtn[2],
            collision_probability: self.collision_probability,
            collision_probability_method: self.pc_method,
            object1,
            object2,
            screening_entry_time: None,
            screening_data_source: None,
            hard_body_radius: self.hard_body_radius,
        })
    }
}

impl Default for CdmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to create CdmObjectData
pub fn create_object_data(
    norad_id: u32,
    name: &str,
    international_designator: &str,
    object_type: &str,
    position_km: [f64; 3],
    velocity_kms: [f64; 3],
    maneuverable: Maneuverable,
) -> CdmObjectData {
    CdmObjectData {
        metadata: CdmObjectMetadata {
            object_designator: norad_id.to_string(),
            catalog_name: "SATCAT".to_string(),
            object_name: name.to_string(),
            international_designator: international_designator.to_string(),
            object_type: object_type.to_string(),
            operator_organization: None,
            operator_phone: None,
            operator_email: None,
            ephemeris_name: "MYCELIX-SPACE".to_string(),
            covariance_method: "CALCULATED".to_string(),
            maneuverable,
            ref_frame: CdmRefFrame::TEME,
        },
        state_vector: CdmStateVector {
            x: position_km[0],
            y: position_km[1],
            z: position_km[2],
            x_dot: velocity_kms[0],
            y_dot: velocity_kms[1],
            z_dot: velocity_kms[2],
        },
        covariance: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdm_builder() {
        let obj1 = create_object_data(
            25544,
            "ISS (ZARYA)",
            "1998-067A",
            "PAYLOAD",
            [6800.0, 0.0, 0.0],
            [0.0, 7.66, 0.0],
            Maneuverable::Yes,
        );

        let obj2 = create_object_data(
            99999,
            "COSMOS 1408 DEB",
            "2021-101A",
            "DEBRIS",
            [6800.5, 0.0, 0.0],
            [0.0, 7.66, 0.0],
            Maneuverable::No,
        );

        let cdm = ConjunctionDataMessage::builder()
            .originator("MYCELIX-SPACE")
            .tca(Utc::now())
            .miss_distance_km(0.5)
            .relative_speed_ms(14500.0)
            .collision_probability(1e-5)
            .object1(obj1)
            .object2(obj2)
            .build()
            .expect("Failed to build CDM");

        assert_eq!(cdm.ccsds_cdm_vers, "1.0");
        assert_eq!(cdm.originator, "MYCELIX-SPACE");
        assert!((cdm.miss_distance - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_cdm_kvn_output() {
        let obj1 = create_object_data(
            25544,
            "ISS",
            "1998-067A",
            "PAYLOAD",
            [6800.0, 100.0, 50.0],
            [0.1, 7.66, 0.05],
            Maneuverable::Yes,
        );

        let obj2 = create_object_data(
            99999,
            "DEBRIS",
            "2021-101A",
            "DEBRIS",
            [6800.5, 100.0, 50.0],
            [0.1, 7.66, 0.05],
            Maneuverable::No,
        );

        let cdm = ConjunctionDataMessage::builder()
            .tca(Utc::now())
            .miss_distance_km(0.5)
            .relative_speed_ms(14500.0)
            .collision_probability(1e-5)
            .object1(obj1)
            .object2(obj2)
            .build()
            .expect("Failed to build CDM");

        let kvn = cdm.to_kvn();
        assert!(kvn.contains("CCSDS_CDM_VERS = 1.0"));
        assert!(kvn.contains("MISS_DISTANCE"));
        assert!(kvn.contains("OBJECT1_OBJECT_NAME = ISS"));
        assert!(kvn.contains("OBJECT2_OBJECT_NAME = DEBRIS"));
    }

    #[test]
    fn test_message_id_generation() {
        let tca = Utc::now();
        let msg_id = ConjunctionDataMessage::generate_message_id(25544, 99999, tca);
        assert!(msg_id.starts_with("CDM-25544-99999-"));
    }
}
