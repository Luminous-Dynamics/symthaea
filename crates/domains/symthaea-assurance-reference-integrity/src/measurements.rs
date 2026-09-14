use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::reference::MeasurementSelector;
use crate::util::{canonical_text, nonempty_refs, push_field, valid_digest};

const MEASUREMENT_SET_DIGEST_SCHEMA: &[u8] = b"symthaea-reference-integrity-measurement-set-v1\0";
const EXTRACTION_RECEIPT_DIGEST_SCHEMA: &[u8] =
    b"symthaea-reference-integrity-extraction-receipt-v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedIntegrityMeasurement {
    pub measurement_id: String,
    pub sequence: u64,
    pub selector: MeasurementSelector,
    pub digest: String,
    pub source_event_ref: String,
}

impl ObservedIntegrityMeasurement {
    pub fn validate(&self) -> bool {
        canonical_text(&self.measurement_id)
            && self.selector.validate()
            && valid_digest(&self.digest)
            && canonical_text(&self.source_event_ref)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedMeasurementSet {
    pub schema_version: String,
    pub measurement_set_id: String,
    pub replay_record_digest: String,
    pub log_bundle_digest: String,
    pub source_event_log_digest: String,
    pub source_final_events_digest: Option<String>,
    pub measurements: Vec<ObservedIntegrityMeasurement>,
    pub extracted_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl NormalizedMeasurementSet {
    pub fn validate(&self) -> bool {
        if !canonical_text(&self.schema_version)
            || !canonical_text(&self.measurement_set_id)
            || !valid_digest(&self.replay_record_digest)
            || !valid_digest(&self.log_bundle_digest)
            || !valid_digest(&self.source_event_log_digest)
            || !self
                .source_final_events_digest
                .as_ref()
                .is_none_or(|digest| valid_digest(digest))
            || self.measurements.is_empty()
            || self.extracted_at_ms == 0
            || !nonempty_refs(&self.evidence_refs)
        {
            return false;
        }

        let mut ids = BTreeSet::new();
        let mut last_sequence = None;
        for measurement in &self.measurements {
            if !measurement.validate()
                || !ids.insert(measurement.measurement_id.as_str())
                || last_sequence.is_some_and(|previous| measurement.sequence <= previous)
            {
                return false;
            }
            last_sequence = Some(measurement.sequence);
        }
        true
    }

    pub fn measurement_set_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();

        let mut hasher = blake3::Hasher::new();
        hasher.update(MEASUREMENT_SET_DIGEST_SCHEMA);
        for value in [
            self.schema_version.as_str(),
            self.measurement_set_id.as_str(),
            self.replay_record_digest.as_str(),
            self.log_bundle_digest.as_str(),
            self.source_event_log_digest.as_str(),
            self.source_final_events_digest.as_deref().unwrap_or(""),
        ] {
            push_field(&mut hasher, value);
        }
        for measurement in &self.measurements {
            push_field(&mut hasher, &measurement.measurement_id);
            push_field(&mut hasher, &measurement.sequence.to_string());
            push_field(&mut hasher, &measurement.selector.selector_digest());
            push_field(&mut hasher, &measurement.digest);
            push_field(&mut hasher, &measurement.source_event_ref);
        }
        push_field(&mut hasher, &self.extracted_at_ms.to_string());
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementExtractionVerificationReceipt {
    pub schema_version: String,
    pub receipt_id: String,
    pub verifier_ref: String,
    pub extraction_tool_ref: String,
    pub extraction_tool_digest: String,
    pub replay_record_digest: String,
    pub measurement_set_digest: String,
    pub log_bundle_digest: String,
    pub source_event_log_digest: String,
    pub source_final_events_digest: Option<String>,
    pub extracted_measurement_count: u64,
    pub all_events_extracted: bool,
    pub selected_pcr_events_covered: bool,
    pub verified_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl MeasurementExtractionVerificationReceipt {
    pub fn validate_for(&self, measurements: &NormalizedMeasurementSet) -> bool {
        measurements.validate()
            && canonical_text(&self.schema_version)
            && canonical_text(&self.receipt_id)
            && canonical_text(&self.verifier_ref)
            && canonical_text(&self.extraction_tool_ref)
            && valid_digest(&self.extraction_tool_digest)
            && self.replay_record_digest == measurements.replay_record_digest
            && self.measurement_set_digest == measurements.measurement_set_digest()
            && self.log_bundle_digest == measurements.log_bundle_digest
            && self.source_event_log_digest == measurements.source_event_log_digest
            && self.source_final_events_digest == measurements.source_final_events_digest
            && self.extracted_measurement_count == measurements.measurements.len() as u64
            && self.all_events_extracted
            && self.selected_pcr_events_covered
            && self.verified_at_ms >= measurements.extracted_at_ms
            && nonempty_refs(&self.evidence_refs)
    }

    pub fn receipt_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();

        let mut hasher = blake3::Hasher::new();
        hasher.update(EXTRACTION_RECEIPT_DIGEST_SCHEMA);
        for value in [
            self.schema_version.as_str(),
            self.receipt_id.as_str(),
            self.verifier_ref.as_str(),
            self.extraction_tool_ref.as_str(),
            self.extraction_tool_digest.as_str(),
            self.replay_record_digest.as_str(),
            self.measurement_set_digest.as_str(),
            self.log_bundle_digest.as_str(),
            self.source_event_log_digest.as_str(),
            self.source_final_events_digest.as_deref().unwrap_or(""),
        ] {
            push_field(&mut hasher, value);
        }
        push_field(&mut hasher, &self.extracted_measurement_count.to_string());
        push_field(&mut hasher, if self.all_events_extracted { "1" } else { "0" });
        push_field(
            &mut hasher,
            if self.selected_pcr_events_covered { "1" } else { "0" },
        );
        push_field(&mut hasher, &self.verified_at_ms.to_string());
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}
