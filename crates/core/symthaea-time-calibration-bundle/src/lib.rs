// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end binding between clock-calibration evidence and policy decisions.
//!
//! A decision receipt by itself is self-consistent with its frozen policy, but
//! it does not prove which exchanges produced the interval it names. This crate
//! binds that receipt to the exact `ClockCalibrationEvidence` set, recomputes
//! every interval and consensus, reruns the policy, and rejects any mismatch.

use std::fmt;

use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use symthaea_time_calibration::{
    CalibrationConsensus, CalibrationError, ClockCalibrationEvidence,
};
use symthaea_time_calibration_policy::{
    AcceptedOffsetEstimate, CalibrationDecisionReceipt, CalibrationPolicyError,
};

/// Structural cap for direct calibration records in one decision bundle.
/// Larger studies should content-address/chunk evidence rather than feed an
/// unbounded vector into a single wire object.
pub const MAX_CALIBRATION_EVIDENCE_RECORDS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationBundleError {
    EmptyEvidence,
    TooManyEvidence {
        actual: usize,
        max: usize,
    },
    DuplicateEvidence {
        first_index: usize,
        second_index: usize,
    },
    Calibration(CalibrationError),
    Policy(CalibrationPolicyError),
    DecisionReceiptMismatch,
}

impl fmt::Display for CalibrationBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEvidence => write!(f, "calibration decision bundle must contain evidence"),
            Self::TooManyEvidence { actual, max } => write!(
                f,
                "calibration decision bundle contains {actual} evidence records; maximum is {max}"
            ),
            Self::DuplicateEvidence {
                first_index,
                second_index,
            } => write!(
                f,
                "duplicate calibration evidence at indices {first_index} and {second_index}"
            ),
            Self::Calibration(error) => write!(f, "calibration evidence invalid: {error}"),
            Self::Policy(error) => write!(f, "calibration policy invalid: {error}"),
            Self::DecisionReceiptMismatch => write!(
                f,
                "decision receipt does not match the consensus and policy recomputed from the attached evidence"
            ),
        }
    }
}

impl std::error::Error for CalibrationBundleError {}

impl From<CalibrationError> for CalibrationBundleError {
    fn from(value: CalibrationError) -> Self {
        Self::Calibration(value)
    }
}

impl From<CalibrationPolicyError> for CalibrationBundleError {
    fn from(value: CalibrationPolicyError) -> Self {
        Self::Policy(value)
    }
}

fn validate_evidence_shape(
    evidence: &[ClockCalibrationEvidence],
) -> Result<(), CalibrationBundleError> {
    if evidence.is_empty() {
        return Err(CalibrationBundleError::EmptyEvidence);
    }
    if evidence.len() > MAX_CALIBRATION_EVIDENCE_RECORDS {
        return Err(CalibrationBundleError::TooManyEvidence {
            actual: evidence.len(),
            max: MAX_CALIBRATION_EVIDENCE_RECORDS,
        });
    }

    for first_index in 0..evidence.len() {
        for second_index in (first_index + 1)..evidence.len() {
            if evidence[first_index] == evidence[second_index] {
                return Err(CalibrationBundleError::DuplicateEvidence {
                    first_index,
                    second_index,
                });
            }
        }
    }
    Ok(())
}

fn deserialize_bounded_evidence<'de, D>(
    deserializer: D,
) -> Result<Vec<ClockCalibrationEvidence>, D::Error>
where
    D: Deserializer<'de>,
{
    struct EvidenceVisitor;

    impl<'de> Visitor<'de> for EvidenceVisitor {
        type Value = Vec<ClockCalibrationEvidence>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "at most {MAX_CALIBRATION_EVIDENCE_RECORDS} calibration evidence records"
            )
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let capacity = seq
                .size_hint()
                .unwrap_or(0)
                .min(MAX_CALIBRATION_EVIDENCE_RECORDS);
            let mut evidence = Vec::with_capacity(capacity);
            while let Some(value) = seq.next_element()? {
                if evidence.len() == MAX_CALIBRATION_EVIDENCE_RECORDS {
                    return Err(de::Error::custom(format_args!(
                        "calibration evidence count exceeds maximum {}",
                        MAX_CALIBRATION_EVIDENCE_RECORDS
                    )));
                }
                evidence.push(value);
            }
            Ok(evidence)
        }
    }

    deserializer.deserialize_seq(EvidenceVisitor)
}

/// Exact calibration evidence set plus the decision it produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CalibrationDecisionBundle {
    decision: CalibrationDecisionReceipt,
    evidence: Vec<ClockCalibrationEvidence>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationDecisionBundleWire {
    decision: CalibrationDecisionReceipt,
    #[serde(deserialize_with = "deserialize_bounded_evidence")]
    evidence: Vec<ClockCalibrationEvidence>,
}

impl<'de> Deserialize<'de> for CalibrationDecisionBundle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CalibrationDecisionBundleWire::deserialize(deserializer)?;
        Self::new(wire.decision, wire.evidence).map_err(de::Error::custom)
    }
}

impl CalibrationDecisionBundle {
    pub fn new(
        decision: CalibrationDecisionReceipt,
        evidence: Vec<ClockCalibrationEvidence>,
    ) -> Result<Self, CalibrationBundleError> {
        let bundle = Self { decision, evidence };
        bundle.verify_self()?;
        Ok(bundle)
    }

    pub fn decision(&self) -> &CalibrationDecisionReceipt {
        &self.decision
    }

    pub fn evidence(&self) -> &[ClockCalibrationEvidence] {
        &self.evidence
    }

    /// Recompute the consensus from the exact attached evidence set.
    pub fn consensus(&self) -> Result<CalibrationConsensus, CalibrationBundleError> {
        validate_evidence_shape(&self.evidence)?;
        for item in &self.evidence {
            item.verify_self()?;
        }
        Ok(CalibrationConsensus::from_evidence(&self.evidence)?)
    }

    /// Verify evidence -> consensus -> frozen policy -> decision end to end.
    pub fn verify_self(&self) -> Result<(), CalibrationBundleError> {
        self.decision.verify_self()?;
        let consensus = self.consensus()?;
        let expected = self.decision.policy().evaluate(&consensus);
        if expected != self.decision {
            return Err(CalibrationBundleError::DecisionReceiptMismatch);
        }
        Ok(())
    }

    pub fn accepted_estimate(&self) -> Result<AcceptedOffsetEstimate, CalibrationBundleError> {
        self.verify_self()?;
        Ok(self.decision.accepted_estimate()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_time_calibration::{FourTimestampExchange, TimestampEvidence};
    use symthaea_time_calibration_policy::{
        CalibrationDecision, CalibrationDecisionPolicy, CalibrationPolicyId,
    };
    use symthaea_time_integrity::{
        ClockDomainId, ClockEpochId, ContinuityStatus, TimeIntegrityReceipt, TimeUncertainty,
    };

    fn source_domain() -> ClockDomainId {
        ClockDomainId::new("sensor-a/monotonic").unwrap()
    }

    fn target_domain() -> ClockDomainId {
        ClockDomainId::new("capture-host/monotonic").unwrap()
    }

    fn source_epoch() -> ClockEpochId {
        ClockEpochId::new("sensor-a-boot-7").unwrap()
    }

    fn target_epoch() -> ClockEpochId {
        ClockEpochId::new("capture-host-boot-3").unwrap()
    }

    fn receipt(domain: ClockDomainId, epoch: ClockEpochId) -> TimeIntegrityReceipt {
        TimeIntegrityReceipt::declared(domain)
            .with_epoch(epoch)
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(0))
    }

    fn stamp(timestamp_us: u64, domain: ClockDomainId, epoch: ClockEpochId) -> TimestampEvidence {
        TimestampEvidence::new(timestamp_us, receipt(domain, epoch))
    }

    fn evidence(one_way_delay_us: u64, source_start_us: u64) -> ClockCalibrationEvidence {
        let target_offset = 500;
        let target_processing = 10;
        let t1 = source_start_us;
        let t2 = t1 + target_offset + one_way_delay_us;
        let t3 = t2 + target_processing;
        let t4 = t1 + one_way_delay_us + target_processing + one_way_delay_us;
        let exchange = FourTimestampExchange::new(
            stamp(t1, source_domain(), source_epoch()),
            stamp(t2, target_domain(), target_epoch()),
            stamp(t3, target_domain(), target_epoch()),
            stamp(t4, source_domain(), source_epoch()),
        )
        .unwrap();
        ClockCalibrationEvidence::derive(exchange).unwrap()
    }

    fn policy() -> CalibrationDecisionPolicy {
        CalibrationDecisionPolicy::new(
            CalibrationPolicyId::new("physical-fusion-v1").unwrap(),
            1,
            20,
            Some(100),
        )
        .unwrap()
    }

    #[test]
    fn exact_evidence_bundle_verifies_end_to_end() {
        let evidence = vec![evidence(10, 1_000), evidence(20, 2_000)];
        let consensus = CalibrationConsensus::from_evidence(&evidence).unwrap();
        let decision = policy().evaluate(&consensus);
        let bundle = CalibrationDecisionBundle::new(decision, evidence).unwrap();
        bundle.verify_self().unwrap();
        assert_eq!(bundle.decision().decision(), CalibrationDecision::Accepted);
        assert_eq!(bundle.accepted_estimate().unwrap().nominal_offset_us(), 500);
    }

    #[test]
    fn removing_decisive_evidence_breaks_binding() {
        // First exchange alone has radius 50 -> Inconclusive.
        // Second exchange has radius 20; intersection becomes radius 20 -> Accepted.
        let wide = evidence(50, 1_000);
        let narrow = evidence(20, 2_000);
        let full = vec![wide.clone(), narrow];
        let consensus = CalibrationConsensus::from_evidence(&full).unwrap();
        let decision = policy().evaluate(&consensus);
        assert_eq!(decision.decision(), CalibrationDecision::Accepted);

        let result = CalibrationDecisionBundle::new(decision, vec![wide]);
        assert_eq!(result.unwrap_err(), CalibrationBundleError::DecisionReceiptMismatch);
    }

    #[test]
    fn duplicate_evidence_is_rejected() {
        let item = evidence(10, 1_000);
        let consensus = CalibrationConsensus::from_evidence(std::slice::from_ref(&item)).unwrap();
        let decision = policy().evaluate(&consensus);
        let result = CalibrationDecisionBundle::new(decision, vec![item.clone(), item]);
        assert!(matches!(
            result,
            Err(CalibrationBundleError::DuplicateEvidence { .. })
        ));
    }

    #[test]
    fn empty_evidence_is_rejected() {
        let one = evidence(10, 1_000);
        let consensus = CalibrationConsensus::from_evidence(std::slice::from_ref(&one)).unwrap();
        let decision = policy().evaluate(&consensus);
        assert_eq!(
            CalibrationDecisionBundle::new(decision, vec![]).unwrap_err(),
            CalibrationBundleError::EmptyEvidence
        );
    }

    #[test]
    fn wire_roundtrip_recomputes_full_decision_chain() {
        let evidence = vec![evidence(10, 1_000), evidence(20, 2_000)];
        let consensus = CalibrationConsensus::from_evidence(&evidence).unwrap();
        let decision = policy().evaluate(&consensus);
        let bundle = CalibrationDecisionBundle::new(decision, evidence).unwrap();
        let json = serde_json::to_string(&bundle).unwrap();
        let decoded: CalibrationDecisionBundle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, bundle);
    }

    #[test]
    fn tampered_policy_decision_is_rejected_before_bundle_acceptance() {
        let evidence = vec![evidence(10, 1_000)];
        let consensus = CalibrationConsensus::from_evidence(&evidence).unwrap();
        let decision = policy().evaluate(&consensus);
        let bundle = CalibrationDecisionBundle::new(decision, evidence).unwrap();
        let mut json = serde_json::to_value(&bundle).unwrap();
        json["decision"]["decision"] = serde_json::json!("rejected");
        assert!(serde_json::from_value::<CalibrationDecisionBundle>(json).is_err());
    }

    #[test]
    fn unknown_bundle_wire_fields_fail_closed() {
        let evidence = vec![evidence(10, 1_000)];
        let consensus = CalibrationConsensus::from_evidence(&evidence).unwrap();
        let decision = policy().evaluate(&consensus);
        let bundle = CalibrationDecisionBundle::new(decision, evidence).unwrap();
        let mut json = serde_json::to_value(&bundle).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("unsupported".into(), serde_json::json!(true));
        assert!(serde_json::from_value::<CalibrationDecisionBundle>(json).is_err());
    }
}
