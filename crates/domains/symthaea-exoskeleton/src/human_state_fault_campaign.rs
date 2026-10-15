// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic fault regressions for the SX-023 human-state evidence twin.
//!
//! These cases verify evidence containment only. They are not human physiology
//! validation and do not issue medical, actuator, oxygen, pressure, thermal, or
//! propulsion commands.

#[cfg(test)]
mod tests {
    use crate::human_state::{
        fuse_human_state, HumanMetric, HumanMetricStatus, HumanObservationSource,
        HumanStateConfig, HumanStateObservation,
    };

    fn obs(
        metric: HumanMetric,
        value: f64,
        uncertainty_abs: f64,
        at_s: f64,
        confidence: f64,
        source_id: &str,
        source: HumanObservationSource,
    ) -> HumanStateObservation {
        HumanStateObservation {
            metric,
            value,
            uncertainty_abs,
            observed_at_s: at_s,
            confidence,
            source_id: source_id.to_string(),
            source,
        }
    }

    fn fuse(items: &[HumanStateObservation]) -> crate::human_state::HumanStateFusionReport {
        fuse_human_state(items, 100.0, HumanStateConfig::simulation_reference()).unwrap()
    }

    #[test]
    fn stale_heart_rate_is_unknown_even_when_other_channels_are_fresh() {
        let inputs = [
            obs(
                HumanMetric::HeartRateBpm, 72.0, 2.0, 80.0, 0.99, "heart-primary",
                HumanObservationSource::Instrument,
            ),
            obs(
                HumanMetric::WholeBodyMetabolicPowerW, 400.0, 30.0, 99.0, 0.90, "metabolic-model",
                HumanObservationSource::ModelEstimate,
            ),
        ];
        let report = fuse(&inputs);
        assert_eq!(
            report.metric(HumanMetric::HeartRateBpm).unwrap().status,
            HumanMetricStatus::Stale
        );
        assert_eq!(
            report.metric(HumanMetric::HeartRateBpm).unwrap().value,
            None
        );
        assert_eq!(
            report.metric(HumanMetric::WholeBodyMetabolicPowerW).unwrap().status,
            HumanMetricStatus::Fresh
        );
    }

    #[test]
    fn contradictory_temperature_sensors_do_not_average_away_disagreement() {
        let inputs = [
            obs(
                HumanMetric::CoreTemperatureC, 37.2, 0.1, 99.5, 0.99, "core-temp-a",
                HumanObservationSource::Instrument,
            ),
            obs(
                HumanMetric::CoreTemperatureC, 39.4, 0.1, 99.5, 0.99, "core-temp-b",
                HumanObservationSource::Instrument,
            ),
        ];
        let report = fuse(&inputs);
        let temperature = report.metric(HumanMetric::CoreTemperatureC).unwrap();
        assert_eq!(temperature.status, HumanMetricStatus::Conflicted);
        assert_eq!(temperature.value, None);
        assert_eq!(temperature.source_ids.len(), 2);
    }

    #[test]
    fn uncertain_water_inventory_is_conflicted_not_optimistically_averaged() {
        let inputs = [
            obs(
                HumanMetric::PotableWaterRemainingMl, 250.0, 15.0, 99.0, 0.99, "tank-gauge",
                HumanObservationSource::Instrument,
            ),
            obs(
                HumanMetric::PotableWaterRemainingMl, 900.0, 25.0, 99.0, 0.98, "transfer-ledger",
                HumanObservationSource::Instrument,
            ),
        ];
        let report = fuse(&inputs);
        let water = report.metric(HumanMetric::PotableWaterRemainingMl).unwrap();
        assert_eq!(water.status, HumanMetricStatus::Conflicted);
        assert_eq!(water.value, None);
    }

    #[test]
    fn workload_and_local_fatigue_remain_distinct_metrics() {
        let inputs = [
            obs(
                HumanMetric::WholeBodyMetabolicPowerW, 650.0, 40.0, 99.0, 0.90, "workload",
                HumanObservationSource::ModelEstimate,
            ),
            obs(
                HumanMetric::LocalFatigueFraction, 0.85, 0.08, 99.0, 0.92, "glove-fatigue",
                HumanObservationSource::ModelEstimate,
            ),
            obs(
                HumanMetric::PerceivedExertion0To10, 8.0, 1.0, 99.0, 0.99, "wearer",
                HumanObservationSource::WearerReported,
            ),
        ];
        let report = fuse(&inputs);
        assert_eq!(
            report.metric(HumanMetric::WholeBodyMetabolicPowerW).unwrap().status,
            HumanMetricStatus::Fresh
        );
        assert_eq!(
            report.metric(HumanMetric::LocalFatigueFraction).unwrap().value,
            Some(0.85)
        );
        assert_eq!(
            report.metric(HumanMetric::PerceivedExertion0To10).unwrap().evidence_kind,
            crate::human_state::HumanStateEvidenceKind::WearerReported
        );
    }

    #[test]
    fn sensor_outlier_and_future_timestamp_never_create_a_valid_estimate() {
        let inputs = [
            obs(
                HumanMetric::RespiratoryRateBpm, 18.0, 1.0, 99.0, 0.99, "valid",
                HumanObservationSource::Instrument,
            ),
            obs(
                HumanMetric::RespiratoryRateBpm, 290.0, 1.0, 99.0, 0.99, "outlier",
                HumanObservationSource::Instrument,
            ),
            obs(
                HumanMetric::RespiratoryRateBpm, 20.0, 1.0, 101.0, 0.99, "future",
                HumanObservationSource::Instrument,
            ),
        ];
        let report = fuse(&inputs);
        let respiratory_rate = report.metric(HumanMetric::RespiratoryRateBpm).unwrap();
        assert_eq!(respiratory_rate.status, HumanMetricStatus::Conflicted);
        assert_eq!(respiratory_rate.value, None);
        assert_eq!(report.rejected_future_dated_observations, 1);
    }

    #[test]
    fn low_confidence_physiology_does_not_promote_itself_to_truth() {
        let inputs = [
            obs(
                HumanMetric::HeartRateBpm, 75.0, 10.0, 99.0, 0.30, "consumer-wearable",
                HumanObservationSource::Instrument,
            ),
        ];
        let report = fuse(&inputs);
        let heart_rate = report.metric(HumanMetric::HeartRateBpm).unwrap();
        assert_eq!(heart_rate.status, HumanMetricStatus::LowConfidence);
        assert_eq!(heart_rate.value, None);
        assert_eq!(report.excluded_low_confidence_observations, 1);
    }

    #[test]
    fn evidence_report_has_no_actuator_or_survival_hardware_command_surface() {
        // Compile-time architectural regression: the fusion result is only
        // measurements/evidence. It has no command, permission, or valve field.
        let report = fuse(&[]);
        assert_eq!(report.metrics.len(), crate::human_state::HUMAN_METRIC_COUNT);
        assert!(report.metrics.iter().all(|metric| metric.value.is_none()));
    }
}
