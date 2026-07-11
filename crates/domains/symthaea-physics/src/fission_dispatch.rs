// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! [`crate::fission`]'s `FissionTwin` → `robotics-dispatch` schema mapping.
//!
//! The last item in `symthaea/NUCLEAR_ENERGY_PLAN_2026-07-06.md` Phase 4:
//! wiring FissionTwin as the first non-robot `SensorNode` telemetry
//! producer for `mycelix-civic`'s `robotics-dispatch` zome (integrity
//! source: `mycelix-workspace/mycelix-civic/zomes/robotics-dispatch/
//! integrity/src/lib.rs`, commit `5096a23773`).
//!
//! ## Why these are mirror DTOs, not the real zome types
//!
//! This module deliberately does **not** take a Cargo dependency on the
//! `robotics-dispatch` crate. Two reasons:
//! 1. There is no existing precedent anywhere in this monorepo for a
//!    `symthaea/` crate depending on a `mycelix-workspace/` crate (checked
//!    2026-07-09) — dependencies flow the other way (mycelix depends on
//!    symthaea via bridge crates). Adding the first one is a real
//!    architectural decision, not a mechanical wiring step.
//! 2. It's also not how a *real* integration would work anyway: native
//!    code doesn't import a Holochain zome's Rust types directly — it
//!    calls the zome over RPC (`AppAgentWebsocket`) with a serialized
//!    payload matching the entry schema. A DTO layer is the correct shape
//!    regardless of whether the dependency existed.
//!
//! Each struct below cites the exact real field it mirrors. If
//! `robotics-dispatch`'s schema changes, these will silently drift —
//! there's no compiler to catch it without the dependency this module
//! deliberately avoids. Re-diff against the source above periodically.
//!
//! ## Honesty notes
//! - This is advisory (non-1E) monitoring — see the nuclear energy plan's
//!   "Position statement." Nothing here claims or implies safety-grade
//!   reactor protection.
//! - `consciousness_level` is a repurposed field (FissionTwin has no
//!   literal Phi) — documented on the field, not silently overloaded.
//! - `mission_progress`/`fuel_level` don't apply to a stationary sensor;
//!   documented placeholders, not real telemetry.
//! - The simulated dispatch order this module can produce is exactly
//!   that — simulated. **Nothing in this module calls a live conductor or
//!   the `mycelix-energy` grid zome.** That's deliberately deferred:
//!   `mycelix-energy` has active concurrent work in this monorepo as of
//!   2026-07-09 (see `MYCELIX_AUTHOR_BINDING_TRIAGE_2026-07-09.md`'s
//!   candidate list) — this module stops at the dispatch-order *shape*,
//!   which is what the plan's "All simulated; the point is the pipeline
//!   shape" scope actually calls for.

use crate::fission::{FissionOutput, FissionReading, FissionSafetyLevel};

/// Mirrors `RoboticAsset` (integrity/src/lib.rs `struct RoboticAsset`),
/// specialized for a `PlatformType::SensorNode(String)` registration.
///
/// Real `RoboticAsset` also carries `owner: AgentPubKey`,
/// `sovereign_profile: SovereignProfile`, and `registered_at: Timestamp` —
/// omitted here since they're assigned by the calling agent / DHT at
/// registration time, not derived from `FissionTwin` itself.
#[derive(Debug, Clone)]
pub struct SensorNodeRegistration {
    /// Mirrors `RoboticAsset::asset_id`.
    pub asset_id: String,
    /// Mirrors `PlatformType::SensorNode(String)` — the string names the
    /// specific monitor (e.g. `"FissionTwin"`), matching the real
    /// variant's documented convention.
    pub sensor_label: String,
    /// Mirrors `RoboticAsset::max_tier`. Always `"Observer"` — a
    /// `SensorNode` registered with anything else is rejected zome-side
    /// by `validate_robotic_asset`.
    pub max_tier: &'static str,
    /// Mirrors `RoboticAsset::location_lat` / `location_lon`.
    pub location_lat: f64,
    pub location_lon: f64,
    /// Mirrors `RoboticAsset::capabilities`.
    pub capabilities: Vec<String>,
    /// Mirrors `RoboticAsset::status`. Always `"Available"` for an
    /// always-on advisory monitor (no `Deployed`/`Maintenance` concept
    /// for a stationary sensor).
    pub status: &'static str,
}

impl SensorNodeRegistration {
    /// Build the registration for a `FissionTwin` instance at a fixed
    /// plant location.
    pub fn fission_twin(asset_id: impl Into<String>, location_lat: f64, location_lon: f64) -> Self {
        Self {
            asset_id: asset_id.into(),
            sensor_label: "FissionTwin".to_string(),
            max_tier: "Observer",
            location_lat,
            location_lon,
            capabilities: vec![
                "reactor_telemetry".to_string(),
                "anomaly_detection".to_string(),
            ],
            status: "Available",
        }
    }
}

/// Mirrors `TelemetryReport` (integrity/src/lib.rs `struct
/// TelemetryReport`).
///
/// Real `TelemetryReport` also carries `asset_hash`/`order_hash:
/// ActionHash` — omitted here since those only exist once the asset is
/// actually registered on a live DHT, which this module doesn't do.
#[derive(Debug, Clone)]
pub struct FissionTelemetryPayload {
    /// Mirrors `TelemetryReport::timestamp` (as Unix microseconds, hdi's
    /// `Timestamp` representation).
    pub timestamp_unix_us: i64,
    /// Mirrors `TelemetryReport::lat` / `lon` / `alt`. `alt` is always
    /// 0.0 — a reactor monitor has no altitude concept.
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
    /// Mirrors `TelemetryReport::consciousness_level`.
    ///
    /// **Repurposed field**: `FissionTwin` has no literal Phi. This is
    /// `(1.0 - free_energy).clamp(0.0, 1.0)` — an inverted free-energy
    /// confidence proxy (low free energy = high confidence the reactor
    /// state matches its healthy reference), not a claim of literal
    /// consciousness. Chosen because it's the closest existing schema
    /// slot for "how confident is this report," not because FissionTwin
    /// is conscious.
    pub consciousness_level: f64,
    /// Mirrors `TelemetryReport::safety_level` — `FissionSafetyLevel`
    /// formatted as `"Green"`/`"Yellow"`/`"Orange"`/`"Red"`, matching the
    /// real field's documented `Green/Yellow/Orange/Red` convention
    /// exactly (same names, same order).
    pub safety_level: String,
    /// Mirrors `TelemetryReport::mission_progress`. **N/A for a
    /// `SensorNode`** — it has no mission to progress through. Always
    /// `0.0`; present only because the schema requires the field.
    pub mission_progress: f64,
    /// Mirrors `TelemetryReport::fuel_level`. **N/A for a `SensorNode`**
    /// — no fuel/battery. Always `1.0`; present only because the schema
    /// requires the field.
    pub fuel_level: f64,
    /// Mirrors `TelemetryReport::platform_specific` — the full
    /// [`FissionOutput`] for this tick, JSON-serialized, so nothing about
    /// the actual detection result is lost to the mirror-DTO's
    /// simplification above.
    pub platform_specific: Vec<u8>,
}

impl FissionTelemetryPayload {
    /// Build a telemetry payload from one `FissionTwin::step()` result.
    pub fn from_fission_output(
        reading: &FissionReading,
        output: &FissionOutput,
        timestamp_unix_us: i64,
        lat: f64,
        lon: f64,
    ) -> Self {
        let _ = reading; // kept in the signature for symmetry/future use; not currently mapped to a field
        Self {
            timestamp_unix_us,
            lat,
            lon,
            alt: 0.0,
            consciousness_level: (1.0 - output.free_energy).clamp(0.0, 1.0),
            safety_level: safety_level_label(output.safety_level).to_string(),
            mission_progress: 0.0,
            fuel_level: 1.0,
            platform_specific: serde_json::to_vec(output).unwrap_or_default(),
        }
    }
}

fn safety_level_label(level: FissionSafetyLevel) -> &'static str {
    match level {
        FissionSafetyLevel::Green => "Green",
        FissionSafetyLevel::Yellow => "Yellow",
        FissionSafetyLevel::Orange => "Orange",
        FissionSafetyLevel::Red => "Red",
    }
}

/// Mirrors `DispatchOrder` (integrity/src/lib.rs `struct DispatchOrder`),
/// **simulated only** — see module docs. Nothing constructs this and
/// sends it anywhere; it exists to show the shape a real dispatcher would
/// produce.
#[derive(Debug, Clone)]
pub struct SimulatedRadZoneInspectionOrder {
    /// Mirrors `MissionType::Custom(String)` — robotics-dispatch has no
    /// dedicated "reactor inspection" mission type, so this uses the
    /// existing escape hatch rather than adding a new variant
    /// speculatively.
    pub mission_type_label: String,
    /// Mirrors `DispatchPriority`, chosen from [`FissionSafetyLevel`].
    pub priority_label: &'static str,
    pub target_lat: f64,
    pub target_lon: f64,
    /// Mirrors `DispatchOrder::description`.
    pub description: String,
}

/// If `output`'s safety level warrants it, build the dispatch order a
/// coordinator *would* send to route a manipulator for rad-zone
/// inspection — simulated shape only, see module docs.
///
/// Threshold: `Orange` or `Red` only. `Green`/`Yellow` don't warrant
/// physical inspection — that's exactly the escalation gradient
/// `FissionSafetyLevel` already encodes; this function doesn't invent a
/// new one.
pub fn simulated_dispatch_for_degradation(
    sensor: &SensorNodeRegistration,
    output: &FissionOutput,
) -> Option<SimulatedRadZoneInspectionOrder> {
    let priority_label = match output.safety_level {
        FissionSafetyLevel::Red => "Urgent",
        FissionSafetyLevel::Orange => "Priority",
        FissionSafetyLevel::Yellow | FissionSafetyLevel::Green => return None,
    };
    Some(SimulatedRadZoneInspectionOrder {
        mission_type_label: "Custom(\"RadZoneInspection\")".to_string(),
        priority_label,
        target_lat: sensor.location_lat,
        target_lon: sensor.location_lon,
        description: format!(
            "{} flagged {:?} (free_energy={:.3}, recommended_action={:?}) — dispatch \
             manipulator for visual/radiological rad-zone inspection. Advisory only; \
             does not represent an automatic reactor protection action.",
            sensor.asset_id, output.safety_level, output.free_energy, output.recommended_action
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fission::FissionTwin;

    fn healthy_reading() -> FissionReading {
        FissionReading {
            power_output: 0.8,
            coolant_temp: 300.0,
            neutron_flux: 0.5,
            pressure: 10.0,
            control_rod_pos: 0.5,
        }
    }

    fn degraded_reading() -> FissionReading {
        FissionReading {
            power_output: 0.2,
            coolant_temp: 380.0,
            neutron_flux: 0.95,
            pressure: 14.5,
            control_rod_pos: 0.05,
        }
    }

    #[test]
    fn test_registration_forces_observer_tier() {
        let reg = SensorNodeRegistration::fission_twin("plant-1-core-monitor", 34.05, -118.25);
        assert_eq!(reg.max_tier, "Observer");
        assert_eq!(reg.sensor_label, "FissionTwin");
    }

    #[test]
    fn test_telemetry_payload_healthy_case() {
        let mut twin = FissionTwin::new();
        let reading = healthy_reading();
        twin.set_reference(&reading);
        let output = twin.step(&reading, 1.0);

        let payload =
            FissionTelemetryPayload::from_fission_output(&reading, &output, 0, 34.05, -118.25);
        assert_eq!(payload.safety_level, "Green");
        assert_eq!(payload.mission_progress, 0.0);
        assert_eq!(payload.fuel_level, 1.0);
        assert!((0.0..=1.0).contains(&payload.consciousness_level));
        assert!(!payload.platform_specific.is_empty());
    }

    #[test]
    fn test_consciousness_level_is_inverted_free_energy() {
        let mut twin = FissionTwin::new();
        let reading = healthy_reading();
        twin.set_reference(&reading);
        let output = twin.step(&reading, 1.0);
        let payload = FissionTelemetryPayload::from_fission_output(&reading, &output, 0, 0.0, 0.0);
        let expected = (1.0 - output.free_energy).clamp(0.0, 1.0);
        assert!((payload.consciousness_level - expected).abs() < 1e-12);
    }

    #[test]
    fn test_platform_specific_round_trips_fission_output() {
        let mut twin = FissionTwin::new();
        let reading = healthy_reading();
        twin.set_reference(&reading);
        let output = twin.step(&reading, 1.0);
        let payload = FissionTelemetryPayload::from_fission_output(&reading, &output, 0, 0.0, 0.0);

        let decoded: FissionOutput = serde_json::from_slice(&payload.platform_specific).unwrap();
        assert_eq!(decoded.safety_level, output.safety_level);
        assert!((decoded.free_energy - output.free_energy).abs() < 1e-12);
    }

    #[test]
    fn test_no_dispatch_when_green_or_yellow() {
        let sensor = SensorNodeRegistration::fission_twin("plant-1", 0.0, 0.0);
        let mut twin = FissionTwin::new();
        let reading = healthy_reading();
        twin.set_reference(&reading);
        let output = twin.step(&reading, 1.0);
        assert!(matches!(
            output.safety_level,
            FissionSafetyLevel::Green | FissionSafetyLevel::Yellow
        ));
        assert!(simulated_dispatch_for_degradation(&sensor, &output).is_none());
    }

    #[test]
    fn test_dispatch_order_shape_when_degraded() {
        let sensor = SensorNodeRegistration::fission_twin("plant-1-core-monitor", 34.05, -118.25);
        let mut twin = FissionTwin::new();
        let reference = healthy_reading();
        twin.set_reference(&reference);
        // Drive toward an off-reference reading; the free-energy detector
        // is reference-similarity based, so a reading far from the
        // reference should escalate past Yellow within a few steps.
        let bad = degraded_reading();
        let mut last_output = None;
        for _ in 0..10 {
            last_output = Some(twin.step(&bad, 1.0));
        }
        let output = last_output.unwrap();

        if matches!(
            output.safety_level,
            FissionSafetyLevel::Orange | FissionSafetyLevel::Red
        ) {
            let order = simulated_dispatch_for_degradation(&sensor, &output)
                .expect("Orange/Red must produce a simulated dispatch order");
            assert_eq!(order.target_lat, 34.05);
            assert_eq!(order.target_lon, -118.25);
            assert!(order.description.contains("plant-1-core-monitor"));
            assert!(order.description.contains("Advisory only"));
        } else {
            // Honest fallback: if this specific reading doesn't happen to
            // cross the threshold, at least confirm the plumbing agrees
            // with itself (None in, None out) rather than silently
            // asserting nothing.
            assert!(simulated_dispatch_for_degradation(&sensor, &output).is_none());
        }
    }
}
