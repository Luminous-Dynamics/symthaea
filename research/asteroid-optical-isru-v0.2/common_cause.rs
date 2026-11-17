// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Common-cause uncertainty and bitemporal qualification reference.
// Run tests with:
//   rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/common_cause.rs -o /tmp/isru-common-cause
//   /tmp/isru-common-cause

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq)]
pub struct SharedCauseSpec {
    pub cause_id: String,
    pub reference_epoch_utc_s: i64,
    pub bias_at_reference: f64,
    pub drift_per_second: f64,
    pub latent_stddev: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InstrumentSpec {
    pub instrument_id: String,
    pub independent_noise_stddev: f64,
    /// True only when the dependency inventory is complete for the claim.
    /// False prevents the model from silently treating unlisted causes as zero.
    pub cause_inventory_complete: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CauseLoading {
    pub instrument_id: String,
    pub cause_id: String,
    pub loading: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CauseDisposition {
    Qualified,
    Unknown,
    Revoked,
    Superseded,
    ScopeMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualificationEvent {
    pub cause_id: String,
    pub disposition: CauseDisposition,
    /// Physical/evidence effective time; may be retroactive.
    pub effective_from_utc_s: i64,
    /// When this assessment was recorded by the system; must be monotonic.
    pub recorded_at_utc_s: i64,
    pub receipt_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeasurementClaim {
    pub claim_id: String,
    pub instrument_id: String,
    pub raw_value: f64,
    pub observed_at_utc_s: i64,
    pub recorded_at_utc_s: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommonCauseFinding {
    IncompleteCauseInventory(String),
    UnknownCause(String),
    UnqualifiedCause {
        cause_id: String,
        disposition: CauseDisposition,
    },
    MissingQualificationEvent(String),
    CauseEpochAfterMeasurement(String),
    MeasurementTimeInverted(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FactorContribution {
    pub cause_id: String,
    pub loading: f64,
    pub bias: f64,
    pub variance: f64,
    pub disposition: CauseDisposition,
    pub receipt_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FactorizedMeasurement {
    pub claim_id: String,
    pub instrument_id: String,
    pub raw_value: f64,
    pub adjusted_value: f64,
    pub observed_at_utc_s: i64,
    pub recorded_at_utc_s: i64,
    pub common_cause_bias: f64,
    pub common_cause_variance: f64,
    pub independent_variance: f64,
    pub total_variance: f64,
    pub contributions: Vec<FactorContribution>,
    pub findings: Vec<CommonCauseFinding>,
    pub eligible: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CommonCauseError {
    EmptyIdentity(&'static str),
    DuplicateCause(String),
    DuplicateInstrument(String),
    DuplicateLoading {
        instrument_id: String,
        cause_id: String,
    },
    UnknownInstrument(String),
    UnknownCause(String),
    InvalidNumeric(&'static str),
    DuplicateReceipt(String),
    RecordedTimeWentBackwards(String),
    UnknownMeasurement(String),
    IneligibleMeasurement(String),
    MeasurementTimeInverted(String),
    MeasurementNotYetRecorded(String),
    EmptyMeasurementSet,
}

#[derive(Clone, Debug)]
pub struct CommonCauseModel {
    causes: BTreeMap<String, SharedCauseSpec>,
    instruments: BTreeMap<String, InstrumentSpec>,
    loadings: BTreeMap<(String, String), f64>,
    qualification_events: BTreeMap<String, Vec<QualificationEvent>>,
    seen_receipts: BTreeSet<String>,
}

impl CommonCauseModel {
    pub fn new(
        causes: Vec<SharedCauseSpec>,
        instruments: Vec<InstrumentSpec>,
        loadings: Vec<CauseLoading>,
    ) -> Result<Self, CommonCauseError> {
        let mut cause_map = BTreeMap::new();
        for cause in causes {
            if cause.cause_id.trim().is_empty() {
                return Err(CommonCauseError::EmptyIdentity("cause_id"));
            }
            if !cause.bias_at_reference.is_finite()
                || !cause.drift_per_second.is_finite()
                || !cause.latent_stddev.is_finite()
                || cause.latent_stddev < 0.0
            {
                return Err(CommonCauseError::InvalidNumeric("shared_cause_spec"));
            }
            let id = cause.cause_id.clone();
            if cause_map.insert(id.clone(), cause).is_some() {
                return Err(CommonCauseError::DuplicateCause(id));
            }
        }

        let mut instrument_map = BTreeMap::new();
        for instrument in instruments {
            if instrument.instrument_id.trim().is_empty() {
                return Err(CommonCauseError::EmptyIdentity("instrument_id"));
            }
            if !instrument.independent_noise_stddev.is_finite()
                || instrument.independent_noise_stddev < 0.0
            {
                return Err(CommonCauseError::InvalidNumeric(
                    "independent_noise_stddev",
                ));
            }
            let id = instrument.instrument_id.clone();
            if instrument_map.insert(id.clone(), instrument).is_some() {
                return Err(CommonCauseError::DuplicateInstrument(id));
            }
        }

        let mut loading_map = BTreeMap::new();
        for link in loadings {
            if !instrument_map.contains_key(&link.instrument_id) {
                return Err(CommonCauseError::UnknownInstrument(
                    link.instrument_id,
                ));
            }
            if !cause_map.contains_key(&link.cause_id) {
                return Err(CommonCauseError::UnknownCause(link.cause_id));
            }
            if !link.loading.is_finite() {
                return Err(CommonCauseError::InvalidNumeric("cause_loading"));
            }
            let key = (link.instrument_id.clone(), link.cause_id.clone());
            if loading_map.insert(key, link.loading).is_some() {
                return Err(CommonCauseError::DuplicateLoading {
                    instrument_id: link.instrument_id,
                    cause_id: link.cause_id,
                });
            }
        }

        Ok(Self {
            causes: cause_map,
            instruments: instrument_map,
            loadings: loading_map,
            qualification_events: BTreeMap::new(),
            seen_receipts: BTreeSet::new(),
        })
    }

    /// Events are ordered by recorded time, not effective time. This permits
    /// later-recorded retroactive revocation without rewriting prior history.
    pub fn record_qualification(
        &mut self,
        event: QualificationEvent,
    ) -> Result<(), CommonCauseError> {
        if event.cause_id.trim().is_empty() {
            return Err(CommonCauseError::EmptyIdentity("cause_id"));
        }
        if event.receipt_id.trim().is_empty() {
            return Err(CommonCauseError::EmptyIdentity("receipt_id"));
        }
        if !self.causes.contains_key(&event.cause_id) {
            return Err(CommonCauseError::UnknownCause(event.cause_id));
        }
        if !self.seen_receipts.insert(event.receipt_id.clone()) {
            return Err(CommonCauseError::DuplicateReceipt(event.receipt_id));
        }
        let history = self
            .qualification_events
            .entry(event.cause_id.clone())
            .or_default();
        if history
            .last()
            .is_some_and(|previous| event.recorded_at_utc_s < previous.recorded_at_utc_s)
        {
            self.seen_receipts.remove(&event.receipt_id);
            return Err(CommonCauseError::RecordedTimeWentBackwards(
                event.cause_id,
            ));
        }
        history.push(event);
        Ok(())
    }

    fn qualification_at(
        &self,
        cause_id: &str,
        observed_at_utc_s: i64,
        as_of_recorded_at_utc_s: i64,
    ) -> Option<&QualificationEvent> {
        self.qualification_events.get(cause_id)?.iter().filter(|event| {
            event.effective_from_utc_s <= observed_at_utc_s
                && event.recorded_at_utc_s <= as_of_recorded_at_utc_s
        }).max_by_key(|event| (event.effective_from_utc_s, event.recorded_at_utc_s))
    }

    pub fn evaluate(
        &self,
        measurement: &MeasurementClaim,
        as_of_recorded_at_utc_s: i64,
    ) -> Result<FactorizedMeasurement, CommonCauseError> {
        if measurement.claim_id.trim().is_empty() {
            return Err(CommonCauseError::EmptyIdentity("claim_id"));
        }
        if !measurement.raw_value.is_finite() {
            return Err(CommonCauseError::InvalidNumeric("raw_value"));
        }
        if measurement.observed_at_utc_s > measurement.recorded_at_utc_s {
            return Err(CommonCauseError::MeasurementTimeInverted(
                measurement.claim_id.clone(),
            ));
        }
        if measurement.recorded_at_utc_s > as_of_recorded_at_utc_s {
            return Err(CommonCauseError::MeasurementNotYetRecorded(
                measurement.claim_id.clone(),
            ));
        }
        let instrument = self
            .instruments
            .get(&measurement.instrument_id)
            .ok_or_else(|| CommonCauseError::UnknownInstrument(
                measurement.instrument_id.clone(),
            ))?;

        let mut findings = Vec::new();
        if !instrument.cause_inventory_complete {
            findings.push(CommonCauseFinding::IncompleteCauseInventory(
                instrument.instrument_id.clone(),
            ));
        }

        let mut common_cause_bias = 0.0;
        let mut common_cause_variance = 0.0;
        let mut contributions = Vec::new();
        for ((instrument_id, cause_id), loading) in &self.loadings {
            if instrument_id != &measurement.instrument_id {
                continue;
            }
            let cause = self
                .causes
                .get(cause_id)
                .expect("loading references validated cause");
            let event = self.qualification_at(
                cause_id,
                measurement.observed_at_utc_s,
                as_of_recorded_at_utc_s,
            );
            let (disposition, receipt_id) = match event {
                Some(event) => (event.disposition, Some(event.receipt_id.clone())),
                None => {
                    findings.push(CommonCauseFinding::MissingQualificationEvent(
                        cause_id.clone(),
                    ));
                    (CauseDisposition::Unknown, None)
                }
            };
            if disposition != CauseDisposition::Qualified {
                findings.push(CommonCauseFinding::UnqualifiedCause {
                    cause_id: cause_id.clone(),
                    disposition,
                });
            }

            if measurement.observed_at_utc_s < cause.reference_epoch_utc_s {
                findings.push(CommonCauseFinding::CauseEpochAfterMeasurement(
                    cause_id.clone(),
                ));
            }
            let elapsed = if measurement.observed_at_utc_s < cause.reference_epoch_utc_s {
                0.0
            } else {
                measurement
                    .observed_at_utc_s
                    .saturating_sub(cause.reference_epoch_utc_s) as f64
            };
            let latent_bias = cause.bias_at_reference + cause.drift_per_second * elapsed;
            let loaded_bias = loading * latent_bias;
            let loaded_variance = (loading * cause.latent_stddev).powi(2);
            common_cause_bias += loaded_bias;
            common_cause_variance += loaded_variance;
            contributions.push(FactorContribution {
                cause_id: cause_id.clone(),
                loading: *loading,
                bias: loaded_bias,
                variance: loaded_variance,
                disposition,
                receipt_id,
            });
        }

        let independent_variance = instrument.independent_noise_stddev.powi(2);
        let total_variance = common_cause_variance + independent_variance;
        let eligible = findings.is_empty();
        Ok(FactorizedMeasurement {
            claim_id: measurement.claim_id.clone(),
            instrument_id: measurement.instrument_id.clone(),
            raw_value: measurement.raw_value,
            adjusted_value: measurement.raw_value - common_cause_bias,
            observed_at_utc_s: measurement.observed_at_utc_s,
            recorded_at_utc_s: measurement.recorded_at_utc_s,
            common_cause_bias,
            common_cause_variance,
            independent_variance,
            total_variance,
            contributions,
            findings,
            eligible,
        })
    }

    /// Covariance contributed only by shared latent causes. Instrument-specific
    /// random noise is treated as independent between measurements; any
    /// instrument drift must be modelled as an explicit cause node.
    pub fn common_cause_covariance(
        &self,
        left: &MeasurementClaim,
        right: &MeasurementClaim,
        as_of_recorded_at_utc_s: i64,
    ) -> Result<f64, CommonCauseError> {
        let left_eval = self.evaluate(left, as_of_recorded_at_utc_s)?;
        let right_eval = self.evaluate(right, as_of_recorded_at_utc_s)?;
        if !left_eval.eligible {
            return Err(CommonCauseError::IneligibleMeasurement(left.claim_id.clone()));
        }
        if !right_eval.eligible {
            return Err(CommonCauseError::IneligibleMeasurement(right.claim_id.clone()));
        }

        let mut covariance = 0.0;
        for (cause_id, left_loading) in self.loadings.iter().filter_map(
            |((instrument_id, cause_id), loading)| {
                (instrument_id == &left.instrument_id).then_some((cause_id, loading))
            },
        ) {
            if let Some(right_loading) = self
                .loadings
                .get(&(right.instrument_id.clone(), cause_id.clone()))
            {
                let cause = self
                    .causes
                    .get(cause_id)
                    .expect("loading references validated cause");
                covariance += left_loading * right_loading * cause.latent_stddev.powi(2);
            }
        }
        Ok(covariance)
    }

    pub fn correlation(
        &self,
        left: &MeasurementClaim,
        right: &MeasurementClaim,
        as_of_recorded_at_utc_s: i64,
    ) -> Result<f64, CommonCauseError> {
        let covariance =
            self.common_cause_covariance(left, right, as_of_recorded_at_utc_s)?;
        let left_eval = self.evaluate(left, as_of_recorded_at_utc_s)?;
        let right_eval = self.evaluate(right, as_of_recorded_at_utc_s)?;
        let denominator = (left_eval.total_variance * right_eval.total_variance).sqrt();
        if denominator <= f64::EPSILON {
            Ok(0.0)
        } else {
            Ok(covariance / denominator)
        }
    }

    /// Kish-style effective count for equally weighted measurements:
    /// n_eff = n^2 / sum_ij rho_ij. It is a redundancy diagnostic, not a
    /// replacement for the full covariance matrix in downstream inference.
    pub fn effective_independent_count(
        &self,
        measurements: &[MeasurementClaim],
        as_of_recorded_at_utc_s: i64,
    ) -> Result<f64, CommonCauseError> {
        if measurements.is_empty() {
            return Err(CommonCauseError::EmptyMeasurementSet);
        }
        let mut correlation_sum = 0.0;
        for (i, left) in measurements.iter().enumerate() {
            for (j, right) in measurements.iter().enumerate() {
                if i == j {
                    correlation_sum += 1.0;
                } else {
                    correlation_sum +=
                        self.correlation(left, right, as_of_recorded_at_utc_s)?;
                }
            }
        }
        let n = measurements.len() as f64;
        if correlation_sum <= f64::EPSILON {
            Ok(n)
        } else {
            Ok((n * n / correlation_sum).clamp(1.0, n))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CommonCauseModel {
        let causes = vec![
            SharedCauseSpec {
                cause_id: "lamp-reference".to_string(),
                reference_epoch_utc_s: 0,
                bias_at_reference: 0.05,
                drift_per_second: 0.0001,
                latent_stddev: 0.2,
            },
            SharedCauseSpec {
                cause_id: "environment".to_string(),
                reference_epoch_utc_s: 0,
                bias_at_reference: -0.02,
                drift_per_second: 0.0,
                latent_stddev: 0.1,
            },
        ];
        let instruments = vec![
            InstrumentSpec {
                instrument_id: "spectrometer-a".to_string(),
                independent_noise_stddev: 0.05,
                cause_inventory_complete: true,
            },
            InstrumentSpec {
                instrument_id: "spectrometer-b".to_string(),
                independent_noise_stddev: 0.08,
                cause_inventory_complete: true,
            },
            InstrumentSpec {
                instrument_id: "independent-qcm".to_string(),
                independent_noise_stddev: 0.03,
                cause_inventory_complete: true,
            },
        ];
        let loadings = vec![
            CauseLoading {
                instrument_id: "spectrometer-a".to_string(),
                cause_id: "lamp-reference".to_string(),
                loading: 1.0,
            },
            CauseLoading {
                instrument_id: "spectrometer-a".to_string(),
                cause_id: "environment".to_string(),
                loading: 0.5,
            },
            CauseLoading {
                instrument_id: "spectrometer-b".to_string(),
                cause_id: "lamp-reference".to_string(),
                loading: 0.8,
            },
            CauseLoading {
                instrument_id: "spectrometer-b".to_string(),
                cause_id: "environment".to_string(),
                loading: 1.0,
            },
        ];
        let mut model = CommonCauseModel::new(causes, instruments, loadings).unwrap();
        for cause_id in ["lamp-reference", "environment"] {
            model
                .record_qualification(QualificationEvent {
                    cause_id: cause_id.to_string(),
                    disposition: CauseDisposition::Qualified,
                    effective_from_utc_s: 0,
                    recorded_at_utc_s: 0,
                    receipt_id: format!("receipt-{cause_id}"),
                })
                .unwrap();
        }
        model
    }

    fn measurement(id: &str, instrument: &str, observed: i64, recorded: i64) -> MeasurementClaim {
        MeasurementClaim {
            claim_id: id.to_string(),
            instrument_id: instrument.to_string(),
            raw_value: 0.9,
            observed_at_utc_s: observed,
            recorded_at_utc_s: recorded,
        }
    }

    #[test]
    fn two_spectrometers_sharing_a_lamp_have_positive_covariance() {
        let value = model();
        let a = measurement("m-a", "spectrometer-a", 100, 101);
        let b = measurement("m-b", "spectrometer-b", 100, 101);
        let covariance = value.common_cause_covariance(&a, &b, 101).unwrap();
        assert!(covariance > 0.0);
        assert_eq!(covariance, 1.0 * 0.8 * 0.2_f64.powi(2)
            + 0.5 * 1.0 * 0.1_f64.powi(2));
    }

    #[test]
    fn instruments_with_explicitly_independent_complete_inventories_have_zero_shared_covariance() {
        let value = model();
        let a = measurement("m-a", "spectrometer-a", 100, 101);
        let b = measurement("m-qcm", "independent-qcm", 100, 101);
        assert_eq!(
            value.common_cause_covariance(&a, &b, 101),
            Ok(0.0)
        );
    }

    #[test]
    fn factorized_value_removes_shared_bias_and_retains_uncertainty() {
        let value = model();
        let reading = measurement("m-a", "spectrometer-a", 0.9, 100);
        let evaluated = value.evaluate(&reading, 100).unwrap();
        let expected_bias = (0.05 + 0.0001 * 100.0) + 0.5 * -0.02;
        assert!((evaluated.common_cause_bias - expected_bias).abs() < 1e-12);
        assert!((evaluated.adjusted_value - (0.9 - expected_bias)).abs() < 1e-12);
        assert!(evaluated.total_variance > evaluated.independent_variance);
        assert!(evaluated.eligible);
    }

    #[test]
    fn retroactive_revocation_changes_current_assessment_but_preserves_historical_view() {
        let mut value = model();
        let reading = measurement("m-a", "spectrometer-a", 100, 101);
        assert!(value.evaluate(&reading, 101).unwrap().eligible);

        value
            .record_qualification(QualificationEvent {
                cause_id: "lamp-reference".to_string(),
                disposition: CauseDisposition::Revoked,
                effective_from_utc_s: 50,
                recorded_at_utc_s: 200,
                receipt_id: "retroactive-revocation".to_string(),
            })
            .unwrap();

        assert!(value.evaluate(&reading, 199).unwrap().eligible);
        let current = value.evaluate(&reading, 200).unwrap();
        assert!(!current.eligible);
        assert!(current.findings.iter().any(|finding| matches!(
            finding,
            CommonCauseFinding::UnqualifiedCause {
                cause_id,
                disposition: CauseDisposition::Revoked
            } if cause_id == "lamp-reference"
        )));
    }

    #[test]
    fn unknown_cause_qualification_is_not_assumed_to_be_zero_or_qualified() {
        let causes = vec![SharedCauseSpec {
            cause_id: "unqualified-reference".to_string(),
            reference_epoch_utc_s: 0,
            bias_at_reference: 0.0,
            drift_per_second: 0.0,
            latent_stddev: 0.2,
        }];
        let instruments = vec![InstrumentSpec {
            instrument_id: "spectrometer".to_string(),
            independent_noise_stddev: 0.1,
            cause_inventory_complete: true,
        }];
        let links = vec![CauseLoading {
            instrument_id: "spectrometer".to_string(),
            cause_id: "unqualified-reference".to_string(),
            loading: 1.0,
        }];
        let value = CommonCauseModel::new(causes, instruments, links).unwrap();
        let evaluated = value
            .evaluate(&measurement("m", "spectrometer", 10, 11), 11)
            .unwrap();
        assert!(!evaluated.eligible);
        assert!(evaluated.findings.contains(
            &CommonCauseFinding::MissingQualificationEvent(
                "unqualified-reference".to_string()
            )
        ));
    }

    #[test]
    fn incomplete_dependency_inventory_blocks_independence_claims() {
        let value = CommonCauseModel::new(
            Vec::new(),
            vec![InstrumentSpec {
                instrument_id: "unknown-dependencies".to_string(),
                independent_noise_stddev: 0.1,
                cause_inventory_complete: false,
            }],
            Vec::new(),
        )
        .unwrap();
        let evaluated = value
            .evaluate(&measurement("m", "unknown-dependencies", 10, 11), 11)
            .unwrap();
        assert!(!evaluated.eligible);
        assert!(evaluated.findings.contains(
            &CommonCauseFinding::IncompleteCauseInventory(
                "unknown-dependencies".to_string()
            )
        ));
    }

    #[test]
    fn missing_cause_link_with_complete_inventory_is_explicitly_independent() {
        let value = CommonCauseModel::new(
            vec![SharedCauseSpec {
                cause_id: "lamp".to_string(),
                reference_epoch_utc_s: 0,
                bias_at_reference: 0.0,
                drift_per_second: 0.0,
                latent_stddev: 0.5,
            }],
            vec![InstrumentSpec {
                instrument_id: "independent".to_string(),
                independent_noise_stddev: 0.1,
                cause_inventory_complete: true,
            }],
            Vec::new(),
        )
        .unwrap();
        assert!(value
            .evaluate(&measurement("m", "independent", 10, 11), 11)
            .unwrap()
            .eligible);
    }

    #[test]
    fn shared_cause_effective_time_does_not_apply_to_earlier_measurements() {
        let mut value = model();
        value
            .record_qualification(QualificationEvent {
                cause_id: "environment".to_string(),
                disposition: CauseDisposition::ScopeMismatch,
                effective_from_utc_s: 200,
                recorded_at_utc_s: 200,
                receipt_id: "future-scope-change".to_string(),
            })
            .unwrap();
        let before = measurement("before", "spectrometer-a", 100, 201);
        let after = measurement("after", "spectrometer-a", 250, 251);
        assert!(value.evaluate(&before, 251).unwrap().eligible);
        assert!(!value.evaluate(&after, 251).unwrap().eligible);
    }

    #[test]
    fn recorded_time_order_is_monotonic_even_when_effective_time_is_retroactive() {
        let mut value = model();
        assert!(value
            .record_qualification(QualificationEvent {
                cause_id: "lamp-reference".to_string(),
                disposition: CauseDisposition::Unknown,
                effective_from_utc_s: 1,
                recorded_at_utc_s: -1,
                receipt_id: "too-early-record".to_string(),
            })
            .is_err());
        assert!(value
            .record_qualification(QualificationEvent {
                cause_id: "lamp-reference".to_string(),
                disposition: CauseDisposition::Revoked,
                effective_from_utc_s: -10,
                recorded_at_utc_s: 100,
                receipt_id: "retroactive-but-later-record".to_string(),
            })
            .is_ok());
    }

    #[test]
    fn effective_independent_count_is_lower_for_correlated_measurements() {
        let value = model();
        let shared = vec![
            measurement("a1", "spectrometer-a", 100, 101),
            measurement("a2", "spectrometer-b", 100, 101),
            measurement("a3", "spectrometer-a", 110, 111),
        ];
        let independent = vec![
            measurement("q1", "independent-qcm", 100, 101),
            measurement("q2", "independent-qcm", 110, 111),
            measurement("q3", "independent-qcm", 120, 121),
        ];
        let n_eff_shared = value.effective_independent_count(&shared, 111).unwrap();
        let n_eff_independent = value
            .effective_independent_count(&independent, 121)
            .unwrap();
        assert!(n_eff_shared < n_eff_independent);
        assert!(n_eff_independent <= 3.0);
    }

    #[test]
    fn qualification_receipt_must_be_unique_and_resolve_to_a_known_cause() {
        let mut value = model();
        assert_eq!(
            value.record_qualification(QualificationEvent {
                cause_id: "lamp-reference".to_string(),
                disposition: CauseDisposition::Qualified,
                effective_from_utc_s: 20,
                recorded_at_utc_s: 20,
                receipt_id: "receipt-lamp-reference".to_string(),
            }),
            Err(CommonCauseError::DuplicateReceipt(
                "receipt-lamp-reference".to_string()
            ))
        );
        assert!(matches!(
            value.record_qualification(QualificationEvent {
                cause_id: "unknown".to_string(),
                disposition: CauseDisposition::Qualified,
                effective_from_utc_s: 20,
                recorded_at_utc_s: 20,
                receipt_id: "receipt-unknown".to_string(),
            }),
            Err(CommonCauseError::UnknownCause(_))
        ));
    }

    #[test]
    fn covariance_for_unqualified_evidence_is_refused() {
        let mut value = model();
        value
            .record_qualification(QualificationEvent {
                cause_id: "lamp-reference".to_string(),
                disposition: CauseDisposition::Revoked,
                effective_from_utc_s: 0,
                recorded_at_utc_s: 200,
                receipt_id: "revocation".to_string(),
            })
            .unwrap();
        let left = measurement("left", "spectrometer-a", 100, 101);
        let right = measurement("right", "spectrometer-b", 100, 101);
        assert_eq!(
            value.common_cause_covariance(&left, &right, 200),
            Err(CommonCauseError::IneligibleMeasurement("left".to_string()))
        );
    }

    #[test]
    fn measurement_before_cause_reference_epoch_is_ineligible() {
        let value = model();
        let evaluated = value
            .evaluate(&measurement("pre-epoch", "spectrometer-a", -10, 0), 0)
            .unwrap();
        assert!(!evaluated.eligible);
        assert!(evaluated.findings.contains(
            &CommonCauseFinding::CauseEpochAfterMeasurement(
                "lamp-reference".to_string()
            )
        ));
    }

    #[test]
    fn measurement_is_not_visible_before_its_recorded_time() {
        let value = model();
        let reading = measurement("not-yet-recorded", "spectrometer-a", 100, 101);
        assert_eq!(
            value.evaluate(&reading, 100),
            Err(CommonCauseError::MeasurementNotYetRecorded(
                "not-yet-recorded".to_string()
            ))
        );
    }

    #[test]
    fn inverted_measurement_times_are_rejected() {
        let value = model();
        assert_eq!(
            value.evaluate(&measurement("bad", "spectrometer-a", 20, 10), 20),
            Err(CommonCauseError::MeasurementTimeInverted(
                "bad".to_string()
            ))
        );
    }
}
