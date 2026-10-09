// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Frozen, deterministic qualification scenarios for energy-scheduling policies.
//!
//! The corpus compares policies on identical declared inputs. Metrics remain
//! separate: net cost, unserved energy, and battery equivalent-full-cycle
//! increment are never collapsed into one score. Scenario hashes establish
//! deterministic input identity, not source authenticity or field truth.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::battery::Battery;
use crate::scheduling::{
    try_run_scenario_with_receipt_profiles, ScenarioError, ScenarioTraceResult, TariffSchedule,
};

const MAX_CORPUS_STEPS: usize = 1_000_000;
const HASH_DOMAIN: &str = "symthaea-energy-scenario-v1";
const POLICY_SOURCE_REVISION_LENGTH: usize = 40;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BatterySpec {
    pub capacity_kwh: f64,
    pub power_rating_kw: f64,
    pub round_trip_efficiency: f64,
    pub initial_soc: f64,
    pub degradation_per_cycle: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenEnergyScenario {
    pub id: String,
    /// Duration represented by each profile sample, in hours.
    pub dt_hours: f64,
    /// Total simulated horizon, in hours. The last step may be shorter.
    pub total_hours: f64,
    /// Absolute clock hour used by profiles, tariffs, and policy decisions.
    pub start_hour: f64,
    pub battery: BatterySpec,
    pub tariff: TariffSchedule,
    /// One finite, non-negative kW sample at each step start.
    pub load_profile_kw: Vec<f64>,
    /// One finite, non-negative kW sample at each step start.
    pub generation_profile_kw: Vec<f64>,
    /// Grid state at each step start; supports outage and reconnection cases.
    pub grid_available_by_step: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyRunReceipt {
    pub scenario_id: String,
    pub scenario_input_digest: String,
    pub policy_id: String,
    /// Exact 40-character Git commit SHA supplied by the calling harness.
    pub policy_source_revision: String,
    pub trace: ScenarioTraceResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedPolicyReceipt {
    pub scenario_id: String,
    pub scenario_input_digest: String,
    pub policy_id: String,
    pub policy_source_revision: String,
    pub step_count: usize,
    pub recomputed_total_cost: f64,
    pub recomputed_unserved_energy_kwh: f64,
    pub recomputed_curtailed_energy_kwh: f64,
    pub recomputed_battery_cycles: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationError {
    InvalidScenario,
    ProfileLengthMismatch,
    InvalidBatteryConfiguration,
    InvalidTariff,
    InvalidProfile,
    InvalidPolicyIdentity,
    ScenarioExecution(ScenarioError),
    ScenarioIdentityMismatch,
    ReceiptStepCountMismatch,
    ReceiptTimeMismatch,
    ReceiptProfileMismatch,
    InvalidReceiptValue,
    PowerBalanceMismatch,
    BatteryTransitionMismatch,
    StepMetricMismatch,
    AggregateMetricMismatch,
    EvidenceSerializationFailed,
}

impl FrozenEnergyScenario {
    pub fn validate(&self) -> Result<(), QualificationError> {
        if !valid_label(&self.id)
            || !self.dt_hours.is_finite()
            || self.dt_hours <= 0.0
            || !self.total_hours.is_finite()
            || self.total_hours <= 0.0
            || !self.start_hour.is_finite()
        {
            return Err(QualificationError::InvalidScenario);
        }

        let estimated = (self.total_hours / self.dt_hours).ceil();
        if !estimated.is_finite() || estimated < 1.0 || estimated > MAX_CORPUS_STEPS as f64 {
            return Err(QualificationError::InvalidScenario);
        }
        let expected_steps = estimated as usize;
        if self.load_profile_kw.len() != expected_steps
            || self.generation_profile_kw.len() != expected_steps
            || self.grid_available_by_step.len() != expected_steps
        {
            return Err(QualificationError::ProfileLengthMismatch);
        }

        let battery = self.battery;
        let battery_values = [
            battery.capacity_kwh,
            battery.power_rating_kw,
            battery.round_trip_efficiency,
            battery.initial_soc,
            battery.degradation_per_cycle,
        ];
        if battery_values.iter().any(|value| !value.is_finite())
            || battery.capacity_kwh < 0.0
            || battery.power_rating_kw < 0.0
            || !(0.0..=1.0).contains(&battery.round_trip_efficiency)
            || !(0.0..=1.0).contains(&battery.initial_soc)
            || battery.degradation_per_cycle < 0.0
        {
            return Err(QualificationError::InvalidBatteryConfiguration);
        }

        let tariff_values = [
            self.tariff.off_peak_price_per_kwh,
            self.tariff.peak_price_per_kwh,
            self.tariff.peak_start_hour,
            self.tariff.peak_end_hour,
            self.tariff.export_price_per_kwh,
        ];
        if tariff_values.iter().any(|value| !value.is_finite())
            || !(0.0..24.0).contains(&self.tariff.peak_start_hour)
            || !(0.0..=24.0).contains(&self.tariff.peak_end_hour)
        {
            return Err(QualificationError::InvalidTariff);
        }

        if self
            .load_profile_kw
            .iter()
            .chain(self.generation_profile_kw.iter())
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(QualificationError::InvalidProfile);
        }
        Ok(())
    }

    pub fn step_count(&self) -> Result<usize, QualificationError> {
        self.validate()?;
        Ok((self.total_hours / self.dt_hours).ceil() as usize)
    }

    /// SHA-256 over a domain-separated canonical encoding of every scenario
    /// input. Profile order is significant because each sample is time-bound.
    pub fn input_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hash_field(&mut hasher, HASH_DOMAIN.as_bytes());
        hash_field(
            &mut hasher,
            b"power=kW;energy=kWh;duration=h;price=currency/kWh;soc=fraction",
        );
        hash_field(&mut hasher, self.id.as_bytes());
        hash_f64(&mut hasher, self.dt_hours);
        hash_f64(&mut hasher, self.total_hours);
        hash_f64(&mut hasher, self.start_hour);

        hash_f64(&mut hasher, self.battery.capacity_kwh);
        hash_f64(&mut hasher, self.battery.power_rating_kw);
        hash_f64(&mut hasher, self.battery.round_trip_efficiency);
        hash_f64(&mut hasher, self.battery.initial_soc);
        hash_f64(&mut hasher, self.battery.degradation_per_cycle);
        // Hash the model's actual constructor-established battery baseline,
        // not just the caller-facing battery specification.
        let initial_battery = Battery::new(
            self.battery.capacity_kwh,
            self.battery.power_rating_kw,
            self.battery.round_trip_efficiency,
        )
        .with_soc(self.battery.initial_soc)
        .with_degradation_per_cycle(self.battery.degradation_per_cycle);
        hash_f64(&mut hasher, initial_battery.state_of_health());
        hash_f64(&mut hasher, initial_battery.equivalent_full_cycles());

        hash_f64(&mut hasher, self.tariff.off_peak_price_per_kwh);
        hash_f64(&mut hasher, self.tariff.peak_price_per_kwh);
        hash_f64(&mut hasher, self.tariff.peak_start_hour);
        hash_f64(&mut hasher, self.tariff.peak_end_hour);
        hash_f64(&mut hasher, self.tariff.export_price_per_kwh);

        hash_count(&mut hasher, self.load_profile_kw.len());
        for value in &self.load_profile_kw {
            hash_f64(&mut hasher, *value);
        }
        hash_count(&mut hasher, self.generation_profile_kw.len());
        for value in &self.generation_profile_kw {
            hash_f64(&mut hasher, *value);
        }
        hash_count(&mut hasher, self.grid_available_by_step.len());
        for available in &self.grid_available_by_step {
            hasher.update(&[u8::from(*available)]);
        }

        format!("{:x}", hasher.finalize())
    }

    /// Run one explicitly identified policy against this scenario's frozen
    /// inputs and return raw receipts alongside the score. The policy commit
    /// SHA is supplied by the caller; this function does not authenticate it.
    pub fn run_policy(
        &self,
        policy_id: &str,
        policy_source_revision: &str,
        policy: impl FnMut(f64, f64, f64, &Battery) -> (f64, f64),
    ) -> Result<PolicyRunReceipt, QualificationError> {
        self.validate()?;
        if !valid_label(policy_id) || !valid_source_revision(policy_source_revision) {
            return Err(QualificationError::InvalidPolicyIdentity);
        }

        let mut battery = Battery::new(
            self.battery.capacity_kwh,
            self.battery.power_rating_kw,
            self.battery.round_trip_efficiency,
        )
        .with_soc(self.battery.initial_soc)
        .with_degradation_per_cycle(self.battery.degradation_per_cycle);

        let dt_hours = self.dt_hours;
        let start_hour = self.start_hour;
        let input_len = self.load_profile_kw.len();
        let load_profile = |time: f64| {
            let index = sample_index(time, start_hour, dt_hours, input_len);
            self.load_profile_kw[index]
        };
        let generation_profile = |time: f64| {
            let index = sample_index(time, start_hour, dt_hours, input_len);
            self.generation_profile_kw[index]
        };
        let grid_profile = |time: f64| {
            let index = sample_index(time, start_hour, dt_hours, input_len);
            self.grid_available_by_step[index]
        };

        let trace = try_run_scenario_with_receipt_profiles(
            &mut battery,
            &self.tariff,
            load_profile,
            generation_profile,
            self.dt_hours,
            self.total_hours,
            self.start_hour,
            grid_profile,
            policy,
        )
        .map_err(QualificationError::ScenarioExecution)?;

        Ok(PolicyRunReceipt {
            scenario_id: self.id.clone(),
            scenario_input_digest: self.input_digest(),
            policy_id: policy_id.to_string(),
            policy_source_revision: policy_source_revision.to_string(),
            trace,
        })
    }

    /// Independently recompute the reported score from raw per-step receipts.
    ///
    /// This verifies arithmetic and correspondence to the frozen scenario. It
    /// does not re-run or certify the policy, authenticate the supplied commit
    /// SHA, prove battery electrochemistry, or authenticate real-world sensors.
    pub fn verify_receipt(
        &self,
        receipt: &PolicyRunReceipt,
    ) -> Result<VerifiedPolicyReceipt, QualificationError> {
        self.validate()?;
        if receipt.scenario_id != self.id
            || receipt.scenario_input_digest != self.input_digest()
            || !valid_label(&receipt.policy_id)
            || !valid_source_revision(&receipt.policy_source_revision)
        {
            return Err(QualificationError::ScenarioIdentityMismatch);
        }

        let expected_steps = self.step_count()?;
        if receipt.trace.steps.len() != expected_steps {
            return Err(QualificationError::ReceiptStepCountMismatch);
        }

        let mut expected_elapsed = 0.0;
        let mut total_cost = 0.0;
        let mut unserved_kwh = 0.0;
        let mut curtailed_kwh = 0.0;
        let mut battery_cycles = 0.0;
        let mut previous_soc_after: Option<f64> = None;
        let mut previous_cycles_after: Option<f64> = None;

        for (index, step) in receipt.trace.steps.iter().enumerate() {
            let remaining = self.total_hours - expected_elapsed;
            let expected_duration = self.dt_hours.min(remaining);
            let expected_time = self.start_hour + expected_elapsed;
            if !close(step.elapsed_hours, expected_elapsed)
                || !close(step.absolute_time_hours, expected_time)
                || !close(step.step_duration_hours, expected_duration)
            {
                return Err(QualificationError::ReceiptTimeMismatch);
            }

            if !close(step.load_kw, self.load_profile_kw[index])
                || !close(step.generation_kw, self.generation_profile_kw[index])
                || !close(step.import_price_per_kwh, self.tariff.import_price(expected_time))
                || !close(step.export_price_per_kwh, self.tariff.export_price_per_kwh)
                || step.grid_available != self.grid_available_by_step[index]
            {
                return Err(QualificationError::ReceiptProfileMismatch);
            }

            let numeric_values = [
                step.elapsed_hours,
                step.absolute_time_hours,
                step.step_duration_hours,
                step.load_kw,
                step.generation_kw,
                step.battery_soc_before,
                step.battery_soc_after,
                step.battery_cycles_before,
                step.battery_cycles_after,
                step.charge_setpoint_kw,
                step.discharge_setpoint_kw,
                step.battery_charge_input_kw,
                step.battery_discharge_output_kw,
                step.net_kw,
                step.import_price_per_kwh,
                step.export_price_per_kwh,
                step.cost_delta,
                step.unserved_energy_delta_kwh,
                step.curtailed_energy_delta_kwh,
            ];
            if numeric_values.iter().any(|value| !value.is_finite())
                || step.step_duration_hours <= 0.0
                || step.load_kw < 0.0
                || step.generation_kw < 0.0
                || !(0.0..=1.0).contains(&step.battery_soc_before)
                || !(0.0..=1.0).contains(&step.battery_soc_after)
                || step.battery_cycles_before < 0.0
                || step.battery_cycles_after < step.battery_cycles_before
                || step.charge_setpoint_kw < 0.0
                || step.discharge_setpoint_kw < 0.0
                || step.charge_setpoint_kw
                    > self.battery.power_rating_kw + tolerance(self.battery.power_rating_kw)
                || step.discharge_setpoint_kw
                    > self.battery.power_rating_kw + tolerance(self.battery.power_rating_kw)
                || step.battery_charge_input_kw < 0.0
                || step.battery_charge_input_kw
                    > step.charge_setpoint_kw + tolerance(step.charge_setpoint_kw)
                || step.battery_discharge_output_kw < 0.0
                || step.battery_discharge_output_kw
                    > step.discharge_setpoint_kw + tolerance(step.discharge_setpoint_kw)
                || step.unserved_energy_delta_kwh < 0.0
                || step.curtailed_energy_delta_kwh < 0.0
                || (step.charge_setpoint_kw > 0.0 && step.discharge_setpoint_kw > 0.0)
            {
                return Err(QualificationError::InvalidReceiptValue);
            }

            if index == 0
                && (!close(step.battery_soc_before, self.battery.initial_soc)
                    || !close(step.battery_cycles_before, 0.0))
            {
                return Err(QualificationError::InvalidReceiptValue);
            }
            // Independently reconstruct the expected battery transition from
            // the frozen battery configuration and the previous state. This
            // catches a forged discharge/SoC receipt even if a producer also
            // edits net power and costs to keep aggregate sums consistent.
            let capacity_kwh = self.battery.capacity_kwh;
            let efficiency = self.battery.round_trip_efficiency.sqrt();
            let health_before = (1.0
                - step.battery_cycles_before * self.battery.degradation_per_cycle)
                .clamp(0.0, 1.0);
            let effective_capacity_kwh = capacity_kwh * health_before;
            let mut expected_soc_after = step.battery_soc_before;
            let mut expected_cycles_after = step.battery_cycles_before;
            let mut expected_charge_input_kw = 0.0;
            let mut expected_discharge_output_kw = 0.0;

            if step.charge_setpoint_kw > 0.0 {
                let requested_dc_kwh = step.charge_setpoint_kw
                    * step.step_duration_hours
                    * efficiency;
                let headroom_kwh =
                    (1.0 - step.battery_soc_before) * effective_capacity_kwh;
                let accepted_dc_kwh = requested_dc_kwh.min(headroom_kwh);
                if effective_capacity_kwh > 0.0 {
                    expected_soc_after = (step.battery_soc_before
                        + accepted_dc_kwh / effective_capacity_kwh)
                        .clamp(0.0, 1.0);
                }
                if capacity_kwh > 0.0 {
                    expected_cycles_after += accepted_dc_kwh / (2.0 * capacity_kwh);
                }
                expected_charge_input_kw = if efficiency > 0.0 {
                    accepted_dc_kwh / efficiency / step.step_duration_hours
                } else if effective_capacity_kwh > 0.0
                    && expected_soc_after < 1.0
                {
                    // With zero efficiency the battery stores nothing, but
                    // requested input is still consumed while headroom exists.
                    step.charge_setpoint_kw
                } else {
                    0.0
                };
            } else if step.discharge_setpoint_kw > 0.0 {
                let requested_dc_kwh = if efficiency > 0.0 {
                    step.discharge_setpoint_kw * step.step_duration_hours / efficiency
                } else {
                    0.0
                };
                let available_dc_kwh =
                    step.battery_soc_before * effective_capacity_kwh;
                let delivered_dc_kwh = requested_dc_kwh.min(available_dc_kwh);
                let delivered_ac_kwh = delivered_dc_kwh * efficiency;
                if effective_capacity_kwh > 0.0 {
                    expected_soc_after = (step.battery_soc_before
                        - delivered_dc_kwh / effective_capacity_kwh)
                        .clamp(0.0, 1.0);
                }
                if capacity_kwh > 0.0 {
                    expected_cycles_after += delivered_dc_kwh / (2.0 * capacity_kwh);
                }
                expected_discharge_output_kw =
                    delivered_ac_kwh / step.step_duration_hours;
            }

            if !close(step.battery_soc_after, expected_soc_after)
                || !close(step.battery_cycles_after, expected_cycles_after)
                || !close(step.battery_charge_input_kw, expected_charge_input_kw)
                || !close(
                    step.battery_discharge_output_kw,
                    expected_discharge_output_kw,
                )
            {
                return Err(QualificationError::BatteryTransitionMismatch);
            }

            if let Some(previous) = previous_soc_after {
                if !close(step.battery_soc_before, previous) {
                    return Err(QualificationError::InvalidReceiptValue);
                }
            }
            if let Some(previous) = previous_cycles_after {
                if !close(step.battery_cycles_before, previous) {
                    return Err(QualificationError::InvalidReceiptValue);
                }
            }
            previous_soc_after = Some(step.battery_soc_after);
            previous_cycles_after = Some(step.battery_cycles_after);

            let recomputed_net = step.load_kw + step.battery_charge_input_kw
                - step.generation_kw - step.battery_discharge_output_kw;
            if !close(step.net_kw, recomputed_net) {
                return Err(QualificationError::PowerBalanceMismatch);
            }

            let (expected_cost_delta, expected_unserved_delta, expected_curtailed_delta) =
                if step.net_kw > 0.0 {
                    if step.grid_available {
                        (
                            step.net_kw * step.step_duration_hours * step.import_price_per_kwh,
                            0.0,
                            0.0,
                        )
                    } else {
                        (0.0, step.net_kw * step.step_duration_hours, 0.0)
                    }
                } else if step.grid_available {
                    (
                        step.net_kw * step.step_duration_hours * step.export_price_per_kwh,
                        0.0,
                        0.0,
                    )
                } else if step.net_kw < 0.0 {
                    (0.0, 0.0, -step.net_kw * step.step_duration_hours)
                } else {
                    (0.0, 0.0, 0.0)
                };
            if !close(step.cost_delta, expected_cost_delta)
                || !close(step.unserved_energy_delta_kwh, expected_unserved_delta)
                || !close(step.curtailed_energy_delta_kwh, expected_curtailed_delta)
            {
                return Err(QualificationError::StepMetricMismatch);
            }

            total_cost += expected_cost_delta;
            unserved_kwh += expected_unserved_delta;
            curtailed_kwh += expected_curtailed_delta;
            battery_cycles += step.battery_cycles_after - step.battery_cycles_before;
            expected_elapsed += expected_duration;
        }

        if !close(expected_elapsed, self.total_hours)
            || !receipt.trace.result.total_cost.is_finite()
            || !receipt.trace.result.unserved_energy_kwh.is_finite()
            || !receipt.trace.result.battery_cycles.is_finite()
            || !receipt.trace.curtailed_energy_kwh.is_finite()
            || receipt.trace.result.unserved_energy_kwh < 0.0
            || receipt.trace.result.battery_cycles < 0.0
            || receipt.trace.curtailed_energy_kwh < 0.0
            || !close(receipt.trace.result.total_cost, total_cost)
            || !close(receipt.trace.result.unserved_energy_kwh, unserved_kwh)
            || !close(receipt.trace.curtailed_energy_kwh, curtailed_kwh)
            || !close(receipt.trace.result.battery_cycles, battery_cycles)
        {
            return Err(QualificationError::AggregateMetricMismatch);
        }

        Ok(VerifiedPolicyReceipt {
            scenario_id: self.id.clone(),
            scenario_input_digest: self.input_digest(),
            policy_id: receipt.policy_id.clone(),
            policy_source_revision: receipt.policy_source_revision.clone(),
            step_count: expected_steps,
            recomputed_total_cost: total_cost,
            recomputed_unserved_energy_kwh: unserved_kwh,
            recomputed_curtailed_energy_kwh: curtailed_kwh,
            recomputed_battery_cycles: battery_cycles,
        })
    }
}

/// A self-contained JSON evidence envelope with frozen inputs, policy
/// identity, raw per-step receipts, and independently recomputed metrics.
#[derive(Debug, Serialize)]
pub struct QualificationEvidencePacket<'a> {
    pub schema_version: &'static str,
    pub scenario: &'a FrozenEnergyScenario,
    pub policy_receipt: &'a PolicyRunReceipt,
    pub verified_metrics: VerifiedPolicyReceipt,
}

/// Verify first, then serialize the evidence packet. This prevents an invalid
/// report from being emitted as a verified packet through this helper.
pub fn serialize_evidence_packet(
    scenario: &FrozenEnergyScenario,
    receipt: &PolicyRunReceipt,
) -> Result<String, QualificationError> {
    let verified_metrics = scenario.verify_receipt(receipt)?;
    let packet = QualificationEvidencePacket {
        schema_version: "symthaea-energy-qualification-evidence-v1",
        scenario,
        policy_receipt: receipt,
        verified_metrics,
    };
    serde_json::to_string_pretty(&packet)
        .map_err(|_| QualificationError::EvidenceSerializationFailed)
}

fn valid_label(value: &str) -> bool {
    !value.trim().is_empty()
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn valid_source_revision(value: &str) -> bool {
    value.len() == POLICY_SOURCE_REVISION_LENGTH
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sample_index(time: f64, start_hour: f64, dt_hours: f64, len: usize) -> usize {
    // Rounded indexing prevents tiny floating-point subtraction error at
    // timestep boundaries from selecting the preceding profile sample.
    (((time - start_hour) / dt_hours).round().max(0.0) as usize).min(len - 1)
}

fn hash_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_count(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn hash_f64(hasher: &mut Sha256, value: f64) {
    hasher.update(value.to_bits().to_le_bytes());
}

fn tolerance(value: f64) -> f64 {
    1e-8 * value.abs().max(1.0)
}

fn close(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= tolerance(left.abs().max(right.abs()))
}

/// Fixed, small Rust-only corpus. The stored profiles are the scenario inputs;
/// no wall clock, network data, random seed, or live weather service is used.
pub fn deterministic_energy_corpus() -> Vec<FrozenEnergyScenario> {
    let tariff = TariffSchedule {
        off_peak_price_per_kwh: 0.10,
        peak_price_per_kwh: 0.35,
        peak_start_hour: 17.0,
        peak_end_hour: 21.0,
        export_price_per_kwh: 0.05,
    };
    let typical_load: Vec<f64> = (0..12)
        .map(|index| if index >= 9 { 40.0 } else { 10.0 })
        .collect();
    let typical_generation: Vec<f64> = (0..12)
        .map(|index| match index {
            2..=3 => 15.0,
            4..=7 => 25.0,
            8 => 10.0,
            _ => 0.0,
        })
        .collect();
    let typical_grid = vec![true; 12];
    let reserve_battery = BatterySpec {
        capacity_kwh: 80.0,
        power_rating_kw: 25.0,
        round_trip_efficiency: 0.90,
        initial_soc: 0.35,
        degradation_per_cycle: 0.0002,
    };

    let mut corpus = vec![
        FrozenEnergyScenario {
            id: "typical-solar-evening-peak-v1".into(),
            dt_hours: 1.0,
            total_hours: 12.0,
            start_hour: 8.0,
            battery: reserve_battery,
            tariff,
            load_profile_kw: typical_load.clone(),
            generation_profile_kw: typical_generation.clone(),
            grid_available_by_step: typical_grid.clone(),
        },
        FrozenEnergyScenario {
            id: "midday-cloud-transient-v1".into(),
            dt_hours: 1.0,
            total_hours: 12.0,
            start_hour: 8.0,
            battery: reserve_battery,
            tariff,
            load_profile_kw: typical_load,
            generation_profile_kw: vec![
                0.0, 0.0, 20.0, 25.0, 5.0, 0.0, 5.0, 20.0, 0.0, 0.0, 0.0, 0.0,
            ],
            grid_available_by_step: typical_grid,
        },
        FrozenEnergyScenario {
            id: "extended-overcast-day-v1".into(),
            dt_hours: 1.0,
            total_hours: 24.0,
            start_hour: 0.0,
            battery: BatterySpec { initial_soc: 0.55, ..reserve_battery },
            tariff,
            load_profile_kw: (0..24)
                .map(|hour| if (17..21).contains(&hour) { 35.0 } else { 10.0 })
                .collect(),
            generation_profile_kw: (0..24)
                .map(|hour| if (8..17).contains(&hour) { 2.0 } else { 0.0 })
                .collect(),
            grid_available_by_step: vec![true; 24],
        },
        FrozenEnergyScenario {
            id: "high-generation-battery-full-v1".into(),
            dt_hours: 1.0,
            total_hours: 8.0,
            start_hour: 10.0,
            battery: BatterySpec { initial_soc: 1.0, ..reserve_battery },
            tariff,
            load_profile_kw: vec![5.0; 8],
            generation_profile_kw: vec![35.0, 35.0, 35.0, 30.0, 20.0, 10.0, 5.0, 0.0],
            grid_available_by_step: vec![true; 8],
        },
        FrozenEnergyScenario {
            id: "grid-outage-during-evening-peak-v1".into(),
            dt_hours: 1.0,
            total_hours: 8.0,
            start_hour: 16.0,
            battery: BatterySpec {
                capacity_kwh: 30.0,
                power_rating_kw: 10.0,
                round_trip_efficiency: 0.90,
                initial_soc: 0.50,
                degradation_per_cycle: 0.0002,
            },
            tariff,
            load_profile_kw: vec![15.0, 20.0, 40.0, 40.0, 40.0, 20.0, 15.0, 10.0],
            generation_profile_kw: vec![0.0; 8],
            grid_available_by_step: vec![true, false, false, false, false, true, true, true],
        },
        FrozenEnergyScenario {
            id: "negative-import-tariff-and-export-v1".into(),
            dt_hours: 1.0,
            total_hours: 6.0,
            start_hour: 8.0,
            battery: BatterySpec { initial_soc: 0.60, ..reserve_battery },
            tariff: TariffSchedule {
                off_peak_price_per_kwh: -0.05,
                peak_price_per_kwh: 0.30,
                peak_start_hour: 10.0,
                peak_end_hour: 12.0,
                export_price_per_kwh: 0.02,
            },
            load_profile_kw: vec![5.0; 6],
            generation_profile_kw: vec![10.0, 10.0, 10.0, 10.0, 5.0, 0.0],
            grid_available_by_step: vec![true; 6],
        },
        FrozenEnergyScenario {
            id: "high-generation-battery-full-islanded-curtailment-v1".into(),
            dt_hours: 1.0,
            total_hours: 8.0,
            start_hour: 10.0,
            battery: BatterySpec { initial_soc: 1.0, ..reserve_battery },
            tariff,
            load_profile_kw: vec![5.0; 8],
            generation_profile_kw: vec![35.0, 35.0, 35.0, 30.0, 20.0, 10.0, 5.0, 0.0],
            grid_available_by_step: vec![false; 8],
        },
        FrozenEnergyScenario {
            id: "islanded-empty-battery-irregular-final-step-v1".into(),
            dt_hours: 0.6,
            total_hours: 1.0,
            start_hour: 0.0,
            battery: BatterySpec {
                capacity_kwh: 5.0,
                power_rating_kw: 3.0,
                round_trip_efficiency: 0.90,
                initial_soc: 0.0,
                degradation_per_cycle: 0.0002,
            },
            tariff,
            load_profile_kw: vec![4.0, 4.0],
            generation_profile_kw: vec![0.0, 0.0],
            grid_available_by_step: vec![false, false],
        },
    ];

    // Corpus construction is deliberately centralized so additions cannot
    // silently depend on the current clock, ambient environment, or RNG.
    for scenario in &corpus {
        debug_assert!(scenario.validate().is_ok(), "invalid frozen scenario {}", scenario.id);
    }
    corpus.shrink_to_fit();
    corpus
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::{naive_greedy_policy, ReserveAwarePolicy};

    const TEST_POLICY_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    fn run_paired(scenario: &FrozenEnergyScenario) -> (PolicyRunReceipt, PolicyRunReceipt) {
        let baseline = scenario
            .run_policy("naive-greedy", TEST_POLICY_REVISION, naive_greedy_policy)
            .unwrap();
        let reserve_policy = ReserveAwarePolicy {
            daytime_reserve_soc: 0.40,
            day_start_hour: 6.0,
            day_end_hour: 17.0,
        };
        let candidate = scenario
            .run_policy("reserve-aware-rule", TEST_POLICY_REVISION, |t, load, generation, battery| {
                reserve_policy.decide(t, load, generation, battery)
            })
            .unwrap();
        (baseline, candidate)
    }

    #[test]
    fn frozen_corpus_uses_stable_ids_and_paired_policy_inputs() {
        let corpus = deterministic_energy_corpus();
        assert!(corpus.len() >= 6);
        let mut ids = std::collections::BTreeSet::new();
        for scenario in &corpus {
            scenario.validate().unwrap();
            assert!(ids.insert(scenario.id.clone()), "duplicate scenario ID");
            assert_eq!(scenario.input_digest(), scenario.input_digest());
            let (baseline, candidate) = run_paired(scenario);
            assert_eq!(baseline.scenario_input_digest, candidate.scenario_input_digest);
            assert_eq!(baseline.scenario_id, candidate.scenario_id);

            let baseline_verified = scenario.verify_receipt(&baseline).unwrap();
            let candidate_verified = scenario.verify_receipt(&candidate).unwrap();
            assert_eq!(baseline_verified.step_count, scenario.step_count().unwrap());
            assert_eq!(candidate_verified.step_count, scenario.step_count().unwrap());
            // These remain independent metrics; no weighted composite can mask
            // critical unserved energy or a cost/cycle regression.
            assert!(baseline_verified.recomputed_unserved_energy_kwh >= 0.0);
            assert!(candidate_verified.recomputed_unserved_energy_kwh >= 0.0);
            assert!(baseline_verified.recomputed_battery_cycles >= 0.0);
            assert!(candidate_verified.recomputed_battery_cycles >= 0.0);
        }
    }

    #[test]
    fn changing_one_frozen_input_changes_the_scenario_digest() {
        let original = deterministic_energy_corpus().remove(0);
        let mut changed = original.clone();
        changed.load_profile_kw[0] += 0.125;
        assert_ne!(original.input_digest(), changed.input_digest());
        let receipt = original
            .run_policy("naive-greedy", TEST_POLICY_REVISION, naive_greedy_policy)
            .unwrap();
        assert_eq!(
            changed.verify_receipt(&receipt),
            Err(QualificationError::ScenarioIdentityMismatch)
        );
    }

    #[test]
    fn independent_checker_rejects_tampered_step_and_aggregate_metrics() {
        let scenario = deterministic_energy_corpus().remove(4);
        let (receipt, _) = run_paired(&scenario);
        let mut tampered_step = receipt.clone();
        tampered_step.trace.steps[0].cost_delta += 1.0;
        assert_eq!(
            scenario.verify_receipt(&tampered_step),
            Err(QualificationError::StepMetricMismatch)
        );

        let mut forged_battery = receipt.clone();
        let first = &mut forged_battery.trace.steps[0];
        first.battery_discharge_output_kw += 1.0;
        first.net_kw = first.load_kw + first.battery_charge_input_kw
            - first.generation_kw - first.battery_discharge_output_kw;
        assert_eq!(
            scenario.verify_receipt(&forged_battery),
            Err(QualificationError::BatteryTransitionMismatch)
        );

        let mut tampered_total = receipt;
        tampered_total.trace.result.total_cost += 1.0;
        assert_eq!(
            scenario.verify_receipt(&tampered_total),
            Err(QualificationError::AggregateMetricMismatch)
        );
    }

    #[test]
    fn outage_corpus_records_grid_state_per_step() {
        let scenario = deterministic_energy_corpus()
            .into_iter()
            .find(|scenario| scenario.id == "grid-outage-during-evening-peak-v1")
            .unwrap();
        let (baseline, _) = run_paired(&scenario);
        assert!(baseline.trace.steps.iter().any(|step| !step.grid_available));
        assert!(baseline.trace.result.unserved_energy_kwh > 0.0);
        scenario.verify_receipt(&baseline).unwrap();
    }

    #[test]
    fn islanded_surplus_is_reported_as_curtailment_not_export_credit() {
        let scenario = deterministic_energy_corpus()
            .into_iter()
            .find(|scenario| scenario.id == "high-generation-battery-full-islanded-curtailment-v1")
            .unwrap();
        let (baseline, _) = run_paired(&scenario);
        assert_eq!(baseline.trace.result.total_cost, 0.0);
        assert!(baseline.trace.curtailed_energy_kwh > 0.0);
        let verified = scenario.verify_receipt(&baseline).unwrap();
        assert!(
            (verified.recomputed_curtailed_energy_kwh - baseline.trace.curtailed_energy_kwh).abs()
                < 1e-8
        );
    }

    #[test]
    fn irregular_final_step_is_accounted_for_exactly() {
        let scenario = deterministic_energy_corpus().pop().unwrap();
        let (baseline, _) = run_paired(&scenario);
        assert_eq!(baseline.trace.steps.len(), 2);
        assert!((baseline.trace.steps[1].step_duration_hours - 0.4).abs() < 1e-9);
        scenario.verify_receipt(&baseline).unwrap();
    }

    #[test]
    fn evidence_packet_contains_inputs_raw_steps_and_verified_metrics() {
        let scenario = deterministic_energy_corpus().remove(0);
        let (receipt, _) = run_paired(&scenario);
        let packet = serialize_evidence_packet(&scenario, &receipt).unwrap();
        assert!(packet.contains("scenario_input_digest"));
        assert!(packet.contains("load_profile_kw"));
        assert!(packet.contains("curtailed_energy_delta_kwh"));
        assert!(packet.contains("recomputed_curtailed_energy_kwh"));
    }

    #[test]
    fn digest_is_frozen_to_a_known_canonical_scenario() {
        // Filled with a fixed expected value after independently reproducing
        // the canonical byte encoding; change only with an intentional schema
        // version change.
        let scenario = deterministic_energy_corpus().remove(0);
        assert_eq!(
            scenario.input_digest(),
            "4182a0bd42b1085875cd91b262b9142a3a44b7dd48476c4599900cf04427d090"
        );
    }
}
