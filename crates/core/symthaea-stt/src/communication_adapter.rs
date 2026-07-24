//! Honest capability adapter for acoustic unit discovery.
//!
//! This adapter reports segmentation and clustering as unit discovery. Transition
//! statistics may support `Structure`; neither result establishes reference or intent.

use crate::discovery::{AcousticSegment, DiscoveredUnit};
use symthaea_communication::{
    CapabilityLevel, CommunicationEvidence, CommunicationResult, CommunicationUnit, Modality,
    Provenance, SignalObservation, TimeSpan, UnitRepresentation,
};

pub struct AcousticDiscoveryAdapter {
    provenance: Provenance,
    evidence: Vec<CommunicationEvidence>,
}

impl AcousticDiscoveryAdapter {
    pub fn new(provenance: Provenance, evidence: Vec<CommunicationEvidence>) -> Self {
        Self {
            provenance,
            evidence,
        }
    }

    pub fn observation(
        &self,
        id: impl Into<String>,
        samples: Vec<f32>,
        sample_rate_hz: u32,
    ) -> SignalObservation {
        let duration = if sample_rate_hz == 0 {
            0.0
        } else {
            samples.len() as f64 / sample_rate_hz as f64
        };
        let external_id = id.into();
        let mut observation = SignalObservation {
            id: String::new(),
            modality: Modality::Audio {
                sample_rate_hz,
                channels: 1,
            },
            samples,
            features: Default::default(),
            original_text: None,
            normalized_text: None,
            uncertain_spans: vec![],
            timing: TimeSpan {
                start_s: 0.0,
                end_s: duration,
            },
            location: None,
            calibration: Default::default(),
            source_identity: None,
            environment: [("external_id".into(), external_id)].into_iter().collect(),
        };
        observation
            .refresh_id()
            .expect("serializing an acoustic observation cannot fail");
        observation
    }

    pub fn units(
        &self,
        observation_id: &str,
        segments: &[AcousticSegment],
    ) -> CommunicationResult<Vec<CommunicationUnit>> {
        let units = segments
            .iter()
            .enumerate()
            .map(|(index, segment)| CommunicationUnit {
                id: segment
                    .unit_id
                    .clone()
                    .unwrap_or_else(|| format!("candidate_{index}")),
                observation_id: observation_id.to_owned(),
                boundary: TimeSpan {
                    start_s: segment.start_time as f64,
                    end_s: segment.end_time as f64,
                },
                boundary_confidence: segment.boundary_salience.clamp(0.0, 1.0),
                representation: UnitRepresentation::Symbol(
                    segment
                        .unit_id
                        .clone()
                        .unwrap_or_else(|| "unassigned".into()),
                ),
                recurrence_count: 1,
                parent_unit: None,
                alternatives: vec![],
            })
            .collect();
        CommunicationResult {
            value: units,
            capability: CapabilityLevel::Unit,
            confidence: mean_boundary_confidence(segments),
            alternatives: vec![],
            provenance: self.provenance.clone(),
            evidence: self.evidence.clone(),
        }
    }

    pub fn discovered_inventory(
        &self,
        units: &[DiscoveredUnit],
    ) -> CommunicationResult<Vec<String>> {
        let has_transitions = units.iter().any(|unit| !unit.following_units.is_empty());
        CommunicationResult {
            value: units.iter().map(|unit| unit.id.clone()).collect(),
            capability: if has_transitions {
                CapabilityLevel::Structure
            } else {
                CapabilityLevel::Unit
            },
            confidence: 0.0, // Must be calibrated by a versioned benchmark, not inferred from cluster counts.
            alternatives: vec![],
            provenance: self.provenance.clone(),
            evidence: self.evidence.clone(),
        }
    }
}

fn mean_boundary_confidence(segments: &[AcousticSegment]) -> f32 {
    if segments.is_empty() {
        return 0.0;
    }
    segments
        .iter()
        .map(|s| s.boundary_salience.clamp(0.0, 1.0))
        .sum::<f32>()
        / segments.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::HV16;

    #[test]
    fn clustering_never_claims_meaning() {
        let adapter = AcousticDiscoveryAdapter::new(
            Provenance {
                provider: "hdc-discovery".into(),
                provider_version: "test".into(),
                model_hash: "test".into(),
                feature_flags: vec![],
                transformations: vec![],
            },
            vec![],
        );
        let unit = DiscoveredUnit::new("UNIT_1", HV16::zero(), 10.0);
        assert!(adapter.discovered_inventory(&[unit]).capability <= CapabilityLevel::Structure);
    }

    #[test]
    fn observation_has_stable_content_addressed_id() {
        let adapter = AcousticDiscoveryAdapter::new(
            Provenance {
                provider: "test".into(),
                provider_version: "0".into(),
                model_hash: "mhash".into(),
                feature_flags: vec![],
                transformations: vec![],
            },
            vec![],
        );
        let obs = adapter.observation("ext-1", vec![0.0_f32; 64], 16000);
        // ID must be non-empty and stable across two constructions with same inputs.
        assert!(!obs.id.is_empty());
        let obs2 = adapter.observation("ext-1", vec![0.0_f32; 64], 16000);
        assert_eq!(obs.id, obs2.id);
    }

    #[test]
    fn units_from_segments_cap_at_unit_level() {
        use symthaea_communication::CapabilityLevel;
        let adapter = AcousticDiscoveryAdapter::new(
            Provenance {
                provider: "test".into(),
                provider_version: "0".into(),
                model_hash: "mhash".into(),
                feature_flags: vec![],
                transformations: vec![],
            },
            vec![],
        );
        // Construct AcousticSegment field-by-field (no Default derived).
        let seg = AcousticSegment {
            start_time: 0.0,
            end_time: 0.1,
            start_frame: 0,
            end_frame: 1,
            hv: HV16::zero(),
            unit_id: Some("A".into()),
            boundary_salience: 0.8,
            spectral_centroid: None,
        };
        let result = adapter.units("obs-1", &[seg]);
        assert!(
            result.capability <= CapabilityLevel::Unit,
            "segment-level result must not exceed Unit, got {:?}",
            result.capability
        );
        assert_eq!(result.value.len(), 1);
        assert_eq!(result.value[0].id, "A");
    }

    #[test]
    fn empty_segments_produce_zero_confidence() {
        let adapter = AcousticDiscoveryAdapter::new(
            Provenance {
                provider: "test".into(),
                provider_version: "0".into(),
                model_hash: "mhash".into(),
                feature_flags: vec![],
                transformations: vec![],
            },
            vec![],
        );
        let result = adapter.units("obs-empty", &[]);
        assert_eq!(result.confidence, 0.0);
        assert!(result.value.is_empty());
    }

    #[test]
    fn inventory_with_transitions_reaches_structure() {
        use symthaea_communication::CapabilityLevel;
        let adapter = AcousticDiscoveryAdapter::new(
            Provenance {
                provider: "test".into(),
                provider_version: "0".into(),
                model_hash: "mhash".into(),
                feature_flags: vec![],
                transformations: vec![],
            },
            vec![],
        );
        let mut unit = DiscoveredUnit::new("UNIT_A", HV16::zero(), 5.0);
        // following_units is HashMap<String, usize>; insert a transition count.
        unit.following_units.insert("UNIT_B".into(), 1);
        let result = adapter.discovered_inventory(&[unit]);
        assert_eq!(
            result.capability,
            CapabilityLevel::Structure,
            "unit with transitions should reach Structure"
        );
    }
}
