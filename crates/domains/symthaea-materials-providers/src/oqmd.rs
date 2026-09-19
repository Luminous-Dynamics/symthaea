// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! OQMD formation-energy response normalization.
//!
//! This is intentionally an offline adapter profile. Live HTTP retrieval belongs to
//! a later transport layer; normalization consumes exact captured response bytes.

use crate::{DataLicenseRef, ProviderError, ProviderRawCapture, sha256_hex};
use serde::{Deserialize, Serialize};
use symthaea_epistemic_types::EpistemicCoordinate;
use symthaea_materials::{
    NormalizedThermodynamicEvidence, ThermodynamicDatabase, ThermodynamicEvidenceOrigin,
};

/// Stable identifier for this normalization implementation.
pub const OQMD_NORMALIZATION_CODE_VERSION: &str = "oqmd-formationenergy-v1";
/// Provider identifier used by this adapter.
pub const OQMD_PROVIDER_ID: &str = "oqmd";
/// License identifier currently documented by OQMD for its data.
pub const OQMD_LICENSE_ID: &str = "CC-BY-4.0";

/// Normalized output batch plus raw/normalization provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OqmdNormalizedBatch {
    /// Exact provider capture metadata.
    pub capture: ProviderRawCapture,
    /// Normalization implementation identity.
    pub normalization_code_version: String,
    /// SHA-256 of deterministic serialized normalized records.
    pub normalized_records_sha256: String,
    /// Normalized thermodynamic records.
    pub records: Vec<NormalizedThermodynamicEvidence>,
    /// Explicit transformations/missing-field notes.
    pub transformation_log: Vec<String>,
}

/// OQMD CC BY 4.0 provenance metadata.
pub fn oqmd_cc_by_4_license() -> DataLicenseRef {
    DataLicenseRef {
        identifier: OQMD_LICENSE_ID.to_string(),
        url: "https://creativecommons.org/licenses/by/4.0/".to_string(),
        attribution: "Open Quantum Materials Database (OQMD), Wolverton Group, Northwestern University".to_string(),
    }
}

/// Normalize a captured OQMD `/formationenergy` response profile.
///
/// External OQMD DFT values remain external-database evidence. This adapter never
/// constructs a `LocalDftCalculation` origin.
pub fn normalize_formation_energy_capture(
    raw: &[u8],
    capture: &ProviderRawCapture,
    epistemic: EpistemicCoordinate,
) -> Result<OqmdNormalizedBatch, ProviderError> {
    capture.verify_raw(raw)?;
    if capture.provider_id != OQMD_PROVIDER_ID {
        return Err(ProviderError::CaptureMetadataMismatch("provider_id"));
    }
    if capture.license.identifier != OQMD_LICENSE_ID {
        return Err(ProviderError::CaptureMetadataMismatch("license"));
    }

    let response: OqmdFormationEnergyResponse = serde_json::from_slice(raw)
        .map_err(|error| ProviderError::SchemaMismatch(error.to_string()))?;
    if response.meta.api_version != capture.provider_schema_version {
        return Err(ProviderError::CaptureMetadataMismatch(
            "provider_schema_version",
        ));
    }
    if response.meta.query.representation != capture.query.representation {
        return Err(ProviderError::CaptureMetadataMismatch("query representation"));
    }
    if response.response_message != "OK" {
        return Err(ProviderError::Normalization(format!(
            "OQMD response_message was {:?}",
            response.response_message
        )));
    }
    if response.meta.data_returned as usize != response.data.len() {
        return Err(ProviderError::Normalization(
            "OQMD data_returned did not match data length".to_string(),
        ));
    }

    let mut records = Vec::with_capacity(response.data.len());
    for entry in &response.data {
        let record = NormalizedThermodynamicEvidence {
            source_id: format!("oqmd:entry:{}", entry.entry_id),
            formula: entry.name.clone(),
            origin: ThermodynamicEvidenceOrigin::ExternalDatabase {
                database: ThermodynamicDatabase::Oqmd,
                dataset_version: Some(format!("api-{}", response.meta.api_version)),
            },
            formation_energy_ev_atom: Some(entry.delta_e),
            energy_above_hull_ev_atom: None,
            decomposition_energy_ev_atom: None,
            // The captured endpoint does not explicitly report a thermodynamic
            // temperature; preserving None is safer than inferring 0 K.
            temperature_k: None,
            method: format!(
                "OQMD external DFT REST record; spacegroup={}; ntypes={}; band_gap_eV={}; api={}",
                entry.spacegroup, entry.ntypes, entry.band_gap, response.meta.api_version
            ),
            epistemic,
            artifact_digest: Some(capture.raw_sha256.clone()),
        };
        record
            .validate()
            .map_err(|error| ProviderError::Normalization(format!("{error:?}")))?;
        records.push(record);
    }

    let normalized_bytes = serde_json::to_vec(&records)?;
    Ok(OqmdNormalizedBatch {
        capture: capture.clone(),
        normalization_code_version: OQMD_NORMALIZATION_CODE_VERSION.to_string(),
        normalized_records_sha256: sha256_hex(&normalized_bytes),
        records,
        transformation_log: vec![
            "OQMD `delta_e` mapped to formation_energy_ev_atom".to_string(),
            "energy_above_hull_ev_atom preserved as None because this capture profile did not request `stability`".to_string(),
            "temperature_k preserved as None because the response did not explicitly report temperature".to_string(),
            "provider DFT origin preserved as external database evidence".to_string(),
        ],
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OqmdFormationEnergyResponse {
    links: OqmdLinks,
    resource: serde_json::Value,
    data: Vec<OqmdFormationEnergyEntry>,
    meta: OqmdMeta,
    response_message: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OqmdLinks {
    next: Option<String>,
    previous: Option<String>,
    base_url: OqmdBaseUrl,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OqmdBaseUrl {
    href: String,
    meta: OqmdBaseMeta,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OqmdBaseMeta {
    #[serde(rename = "_oqmd_version")]
    oqmd_version: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OqmdFormationEnergyEntry {
    name: String,
    entry_id: u64,
    spacegroup: String,
    ntypes: u32,
    band_gap: f64,
    delta_e: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OqmdMeta {
    query: OqmdQuery,
    api_version: String,
    time_stamp: String,
    data_returned: u64,
    data_available: u64,
    comments: String,
    query_tree: String,
    more_data_available: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OqmdQuery {
    representation: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProviderRawCapture;
    use symthaea_epistemic_types::{
        EmpiricalLevel, EpistemicContext, MaterialityLevel, NormativeLevel,
    };
    use symthaea_materials::{MaterialsEvidenceKind, MaterialsEvidenceStage};

    const FIXTURE: &str = include_str!("../fixtures/oqmd_formationenergy_v1.json");

    fn coordinate() -> EpistemicCoordinate {
        EpistemicCoordinate {
            empirical: EmpiricalLevel::E0Null,
            normative: NormativeLevel::N0Personal,
            materiality: MaterialityLevel::M1Temporal,
            context: EpistemicContext::Scientific,
        }
    }

    fn capture(raw: &[u8]) -> ProviderRawCapture {
        ProviderRawCapture::new(
            OQMD_PROVIDER_ID.to_string(),
            "1.0".to_string(),
            "formationenergy-v1".to_string(),
            "/formationenergy?fields=name,entry_id,spacegroup,ntypes,band_gap,delta_e&icsd=True&limit=2&filter=element_set=(Al-Fe),O".to_string(),
            "2019-10-08T15:13:12Z".to_string(),
            "https://oqmd.org/oqmdapi/formationenergy?fields=name,entry_id,spacegroup,ntypes,band_gap,delta_e&icsd=True&limit=2&filter=element_set=(Al-Fe),O".to_string(),
            oqmd_cc_by_4_license(),
            raw,
        )
        .unwrap()
    }

    #[test]
    fn documented_fixture_normalizes_offline() {
        let raw = FIXTURE.as_bytes();
        let batch = normalize_formation_energy_capture(raw, &capture(raw), coordinate()).unwrap();
        assert_eq!(batch.records.len(), 2);
        assert_eq!(batch.records[0].source_id, "oqmd:entry:16974");
        assert_eq!(batch.records[0].formula, "NaAlH2CO5");
        assert_eq!(batch.records[0].formation_energy_ev_atom, Some(-2.05739610121138));
        assert_eq!(batch.records[0].energy_above_hull_ev_atom, None);
        assert_eq!(batch.normalized_records_sha256.len(), 64);
        assert_eq!(batch.capture.license.identifier, OQMD_LICENSE_ID);
    }

    #[test]
    fn external_oqmd_dft_stays_database_corroboration() {
        let raw = FIXTURE.as_bytes();
        let batch = normalize_formation_energy_capture(raw, &capture(raw), coordinate()).unwrap();
        let evidence = batch.records[0].as_database_evidence_record().unwrap();
        assert_eq!(evidence.stage, MaterialsEvidenceStage::DatabaseCorroborated);
        assert_eq!(evidence.kind, MaterialsEvidenceKind::ExternalDatabase);
        assert!(batch.records[0].as_dft_formation_evidence_record().is_err());
    }

    #[test]
    fn schema_drift_is_explicit_not_silently_dropped() {
        let mut value: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("unexpected_schema_field".to_string(), serde_json::json!(1));
        let changed = serde_json::to_vec(&value).unwrap();
        let result = normalize_formation_energy_capture(&changed, &capture(&changed), coordinate());
        assert!(matches!(result, Err(ProviderError::SchemaMismatch(_))));
    }

    #[test]
    fn capture_query_and_response_query_must_match() {
        let raw = FIXTURE.as_bytes();
        let mut capture = capture(raw);
        capture.query.representation = "/formationenergy?different=true".to_string();
        assert!(matches!(
            normalize_formation_energy_capture(raw, &capture, coordinate()),
            Err(ProviderError::CaptureMetadataMismatch("query representation"))
        ));
    }
}
