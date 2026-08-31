//! Offline-first Sentinel-1/2 bridge for Symthaea Planetary Perception.
//!
//! The initial bridge is deliberately transport-free: a frozen product catalog
//! can be replayed in CI and research lineages without credentials or a live
//! Copernicus service. A future network adapter can populate the same metadata
//! types without changing downstream Earth-observation semantics.

mod exact_window;
mod fixture_manifest;
pub use exact_window::*;
pub use fixture_manifest::*;

use symthaea_earth_observation::{
    BandDescriptor, ContentDigest, GeoFootprint, InstrumentId, MissionId, ObservationEvidence,
    ObservationId, ObservationSensitivity, ObservationUncertainty, ProcessingLineage, ProductId,
    Result as ObservationResult, SensorModality,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentinelMission {
    Sentinel1,
    Sentinel2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentinelProductKind {
    Sentinel1Grd,
    Sentinel1Slc,
    Sentinel2L1C,
    Sentinel2L2A,
}

impl SentinelProductKind {
    pub const fn mission(self) -> SentinelMission {
        match self {
            Self::Sentinel1Grd | Self::Sentinel1Slc => SentinelMission::Sentinel1,
            Self::Sentinel2L1C | Self::Sentinel2L2A => SentinelMission::Sentinel2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SentinelProductMetadata {
    pub observation_id: String,
    pub mission_id: String,
    pub instrument_id: String,
    pub product_id: String,
    pub product_kind: SentinelProductKind,
    pub acquired_at_unix_ms: i64,
    pub footprint: GeoFootprint,
    pub modality: SensorModality,
    pub bands: Vec<BandDescriptor>,
    pub uncertainty: ObservationUncertainty,
    pub source_digest: ContentDigest,
    pub lineage: ProcessingLineage,
}

impl SentinelProductMetadata {
    /// Convert product metadata into the provider-neutral evidence contract.
    ///
    /// The bridge intentionally assigns `SurfaceOnly`. Neither a Sentinel-1 nor
    /// Sentinel-2 product is upgraded to `DirectPenetrating` merely because it
    /// is radar/optical data. Indirect subsurface evidence (for example, a later
    /// InSAR displacement product) must be represented as a derived observation
    /// or feature with its own processing lineage.
    pub fn into_observation(self) -> ObservationResult<ObservationEvidence> {
        Ok(ObservationEvidence {
            id: ObservationId::new(self.observation_id)?,
            mission: MissionId::new(self.mission_id)?,
            instrument: InstrumentId::new(self.instrument_id)?,
            product: ProductId::new(self.product_id)?,
            acquired_at_unix_ms: self.acquired_at_unix_ms,
            footprint: self.footprint,
            modality: self.modality,
            sensitivity: ObservationSensitivity::SurfaceOnly,
            bands: self.bands,
            uncertainty: self.uncertainty,
            source_digest: self.source_digest,
            lineage: self.lineage,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CatalogQuery {
    pub mission: Option<SentinelMission>,
    pub product_kind: Option<SentinelProductKind>,
    pub acquired_not_before_unix_ms: Option<i64>,
    pub acquired_before_unix_ms: Option<i64>,
}

impl CatalogQuery {
    fn matches(self, product: &SentinelProductMetadata) -> bool {
        if let Some(mission) = self.mission {
            if product.product_kind.mission() != mission {
                return false;
            }
        }
        if let Some(kind) = self.product_kind {
            if product.product_kind != kind {
                return false;
            }
        }
        if let Some(start) = self.acquired_not_before_unix_ms {
            if product.acquired_at_unix_ms < start {
                return false;
            }
        }
        if let Some(end_exclusive) = self.acquired_before_unix_ms {
            if product.acquired_at_unix_ms >= end_exclusive {
                return false;
            }
        }
        true
    }
}

/// Provider abstraction intentionally small enough for deterministic fixture
/// implementations and future live catalog clients to share.
pub trait SentinelCatalog {
    type Error;

    fn search(
        &self,
        query: CatalogQuery,
    ) -> std::result::Result<Vec<SentinelProductMetadata>, Self::Error>;
    fn get(
        &self,
        product_id: &str,
    ) -> std::result::Result<Option<SentinelProductMetadata>, Self::Error>;
}

/// A deterministic, network-free catalog for tests, replay, and frozen evidence
/// lineages.
#[derive(Debug, Clone, Default)]
pub struct FrozenSentinelCatalog {
    products: Vec<SentinelProductMetadata>,
}

impl FrozenSentinelCatalog {
    pub fn new(mut products: Vec<SentinelProductMetadata>) -> Self {
        products.sort_by(|a, b| {
            a.acquired_at_unix_ms
                .cmp(&b.acquired_at_unix_ms)
                .then_with(|| a.product_id.cmp(&b.product_id))
        });
        Self { products }
    }

    pub fn len(&self) -> usize {
        self.products.len()
    }

    pub fn is_empty(&self) -> bool {
        self.products.is_empty()
    }
}

impl SentinelCatalog for FrozenSentinelCatalog {
    type Error = std::convert::Infallible;

    fn search(
        &self,
        query: CatalogQuery,
    ) -> std::result::Result<Vec<SentinelProductMetadata>, Self::Error> {
        Ok(self
            .products
            .iter()
            .filter(|product| query.matches(product))
            .cloned()
            .collect())
    }

    fn get(
        &self,
        product_id: &str,
    ) -> std::result::Result<Option<SentinelProductMetadata>, Self::Error> {
        Ok(self
            .products
            .iter()
            .find(|product| product.product_id == product_id)
            .cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_earth_observation::{Confidence, DigestAlgorithm, GeoPoint, Polarization, RadarBand};

    fn footprint() -> GeoFootprint {
        GeoFootprint::new(vec![
            GeoPoint::new(-25.0, 28.0).unwrap(),
            GeoPoint::new(-25.1, 28.0).unwrap(),
            GeoPoint::new(-25.0, 28.1).unwrap(),
        ])
        .unwrap()
    }

    fn product(
        id: &str,
        kind: SentinelProductKind,
        acquired_at_unix_ms: i64,
    ) -> SentinelProductMetadata {
        let (mission_id, instrument_id, modality) = match kind.mission() {
            SentinelMission::Sentinel1 => (
                "Sentinel-1",
                "C-SAR",
                SensorModality::SyntheticApertureRadar {
                    band: RadarBand::C,
                    polarization: Some(Polarization::Vv),
                },
            ),
            SentinelMission::Sentinel2 => (
                "Sentinel-2",
                "MSI",
                SensorModality::Multispectral,
            ),
        };

        SentinelProductMetadata {
            observation_id: format!("obs-{id}"),
            mission_id: mission_id.to_string(),
            instrument_id: instrument_id.to_string(),
            product_id: id.to_string(),
            product_kind: kind,
            acquired_at_unix_ms,
            footprint: footprint(),
            modality,
            bands: vec![],
            uncertainty: ObservationUncertainty::new(
                Some(Confidence::new(0.9).unwrap()),
                None,
                None,
            )
            .unwrap(),
            source_digest: ContentDigest::new(DigestAlgorithm::Sha256, "00").unwrap(),
            lineage: ProcessingLineage::default(),
        }
    }

    #[test]
    fn frozen_catalog_is_deterministically_sorted() {
        let catalog = FrozenSentinelCatalog::new(vec![
            product("later", SentinelProductKind::Sentinel2L2A, 200),
            product("early-b", SentinelProductKind::Sentinel1Grd, 100),
            product("early-a", SentinelProductKind::Sentinel2L2A, 100),
        ]);

        let results = catalog.search(CatalogQuery::default()).unwrap();
        let ids: Vec<_> = results.into_iter().map(|item| item.product_id).collect();
        assert_eq!(ids, vec!["early-a", "early-b", "later"]);
    }

    #[test]
    fn frozen_catalog_filters_by_mission_and_time() {
        let catalog = FrozenSentinelCatalog::new(vec![
            product("s1", SentinelProductKind::Sentinel1Grd, 100),
            product("s2-old", SentinelProductKind::Sentinel2L2A, 100),
            product("s2-new", SentinelProductKind::Sentinel2L2A, 200),
        ]);

        let results = catalog
            .search(CatalogQuery {
                mission: Some(SentinelMission::Sentinel2),
                acquired_not_before_unix_ms: Some(150),
                ..CatalogQuery::default()
            })
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].product_id, "s2-new");
    }

    #[test]
    fn sentinel_products_do_not_imply_direct_subsurface_sensitivity() {
        let observation = product("s1", SentinelProductKind::Sentinel1Grd, 100)
            .into_observation()
            .unwrap();
        assert_eq!(observation.sensitivity, ObservationSensitivity::SurfaceOnly);
    }

    #[test]
    fn exact_product_lookup_is_replayable() {
        let catalog = FrozenSentinelCatalog::new(vec![product(
            "fixture-product",
            SentinelProductKind::Sentinel2L2A,
            100,
        )]);
        assert!(catalog.get("missing").unwrap().is_none());
        assert_eq!(
            catalog.get("fixture-product").unwrap().unwrap().product_id,
            "fixture-product"
        );
    }
}
