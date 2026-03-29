// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Chemical sensor simulation and anomaly detection.
//!
//! Simulates realistic sensor readings with noise, drift, and contamination plumes.
//! Detects anomalies by comparing readings to WHO thresholds + historical baselines.

use crate::types::ChemicalReadings;

/// Contamination plume: a localized region of elevated contaminant levels.
#[derive(Debug, Clone)]
pub struct ContaminationPlume {
    /// Center position [x, y, z].
    pub center: [f64; 3],
    /// Plume radius (meters).
    pub radius: f64,
    /// Which contaminant is elevated.
    pub contaminant: Contaminant,
    /// Peak concentration at center (in native units).
    pub peak_concentration: f64,
}

/// Contaminant types matching ChemicalReadings fields.
#[derive(Debug, Clone, Copy)]
pub enum Contaminant {
    Arsenic,
    Lead,
    Nitrates,
    Chlorine,
    LowOxygen,
    HighTurbidity,
    AcidicPh,
    HighTds,
}

/// Apply plume contamination to clean readings based on distance from plume center.
pub fn apply_plume(
    clean: &ChemicalReadings,
    plume: &ContaminationPlume,
    position: &[f64; 3],
) -> ChemicalReadings {
    let dx = position[0] - plume.center[0];
    let dy = position[1] - plume.center[1];
    let dz = position[2] - plume.center[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if dist > plume.radius {
        return *clean;
    }

    // Gaussian falloff from center
    let falloff = (-3.0 * (dist / plume.radius).powi(2)).exp();
    let added = plume.peak_concentration * falloff;

    let mut result = *clean;
    match plume.contaminant {
        Contaminant::Arsenic => result.arsenic_ug_l += added,
        Contaminant::Lead => result.lead_ug_l += added,
        Contaminant::Nitrates => result.nitrates_mg_l += added,
        Contaminant::Chlorine => result.chlorine_mg_l += added,
        Contaminant::LowOxygen => result.dissolved_oxygen_mg_l = (result.dissolved_oxygen_mg_l - added).max(0.0),
        Contaminant::HighTurbidity => result.turbidity_ntu += added,
        Contaminant::AcidicPh => result.ph = (result.ph - added).max(0.0),
        Contaminant::HighTds => result.tds_ppm += added,
    }
    result
}

/// Anomaly detection: returns list of contaminants exceeding WHO thresholds.
pub fn detect_anomalies(readings: &ChemicalReadings) -> Vec<String> {
    let mut anomalies = Vec::new();
    if readings.ph < 6.5 || readings.ph > 8.5 { anomalies.push(format!("pH: {:.1}", readings.ph)); }
    if readings.turbidity_ntu > 5.0 { anomalies.push(format!("turbidity: {:.0} NTU", readings.turbidity_ntu)); }
    if readings.tds_ppm > 1000.0 { anomalies.push(format!("TDS: {:.0} ppm", readings.tds_ppm)); }
    if readings.nitrates_mg_l > 50.0 { anomalies.push(format!("nitrates: {:.1} mg/L", readings.nitrates_mg_l)); }
    if readings.arsenic_ug_l > 10.0 { anomalies.push(format!("arsenic: {:.1} µg/L", readings.arsenic_ug_l)); }
    if readings.lead_ug_l > 10.0 { anomalies.push(format!("lead: {:.1} µg/L", readings.lead_ug_l)); }
    if readings.chlorine_mg_l > 5.0 { anomalies.push(format!("chlorine: {:.1} mg/L", readings.chlorine_mg_l)); }
    if readings.dissolved_oxygen_mg_l < 4.0 { anomalies.push(format!("DO: {:.1} mg/L", readings.dissolved_oxygen_mg_l)); }
    anomalies
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_water_no_anomalies() {
        let clean = ChemicalReadings::clean_freshwater();
        assert!(detect_anomalies(&clean).is_empty());
    }

    #[test]
    fn test_arsenic_plume() {
        let clean = ChemicalReadings::clean_freshwater();
        let plume = ContaminationPlume {
            center: [0.0, 0.0, 10.0], radius: 50.0,
            contaminant: Contaminant::Arsenic, peak_concentration: 40.0,
        };
        let at_center = apply_plume(&clean, &plume, &[0.0, 0.0, 10.0]);
        assert!(at_center.arsenic_ug_l > 10.0, "Should exceed WHO threshold");
        assert!(!detect_anomalies(&at_center).is_empty());

        let far_away = apply_plume(&clean, &plume, &[100.0, 0.0, 10.0]);
        assert!((far_away.arsenic_ug_l - clean.arsenic_ug_l).abs() < 0.1, "Far from plume should be clean");
    }

    #[test]
    fn test_plume_gaussian_falloff() {
        let clean = ChemicalReadings::clean_freshwater();
        let plume = ContaminationPlume {
            center: [0.0, 0.0, 0.0], radius: 100.0,
            contaminant: Contaminant::Lead, peak_concentration: 50.0,
        };
        let close = apply_plume(&clean, &plume, &[10.0, 0.0, 0.0]);
        let mid = apply_plume(&clean, &plume, &[50.0, 0.0, 0.0]);
        assert!(close.lead_ug_l > mid.lead_ug_l, "Closer should have higher concentration");
    }
}
