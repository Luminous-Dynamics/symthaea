// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Bioelectric Data Ingest Parsers.
//!
//! Provides utilities to parse real-world bioelectric datasets (CSV, JSON)
//! from Dr. Michael Levin's publications and public MEA repositories.
//!
//! Supported Formats:
//! 1. Hansali et al. (2025) - Planarian Regenerative Signaling (OSF)
//! 2. McMillen & Levin (2024) - Optical Estimation Fluorescence (FLIM)
//! 3. General MEA (Time, ID, Voltage)

use crate::morpho_mesh::{MeaPacket, MorphoMeshAdapter};
use crate::morpho_topology::TissueSnapshot;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Ingest Result
pub type IngestResult<T> = Result<T, Box<dyn Error>>;

/// Parser for Hansali et al. (2025) Planarian OSF dataset.
#[derive(Debug, Deserialize)]
#[allow(non_snake_case)]
pub struct HansaliPlanarianRecord {
    pub Worm_ID: String,
    pub Treatment: String,
    pub Concentration: String,
    pub Outcome: String,
    pub Vmem_mV: f32,
    pub Variance: f32,
    pub Bistability: bool,
    pub Time_Point: String,
}

/// Parser for McMillen & Levin (2024) Fluorescence Tabular format.
#[derive(Debug, Deserialize)]
#[allow(non_snake_case)]
pub struct McMillenFluorescenceRecord {
    pub X: f32,
    pub Y: f32,
    pub Lifetime: f32,
}

/// Ingest utility for Bioelectric Data.
pub struct BioelectricIngest;

impl BioelectricIngest {
    /// Parse Hansali et al. (2025) CSV data.
    pub fn parse_hansali_csv(path: &str) -> IngestResult<Vec<HansaliPlanarianRecord>> {
        let file = File::open(path)?;
        let mut rdr = csv::Reader::from_reader(BufReader::new(file));
        let mut records = Vec::new();
        for result in rdr.deserialize() {
            let record: HansaliPlanarianRecord = result?;
            records.push(record);
        }
        Ok(records)
    }

    /// Parse McMillen & Levin (2024) Tabular CSV data.
    pub fn parse_mcmillen_csv(path: &str) -> IngestResult<Vec<McMillenFluorescenceRecord>> {
        let file = File::open(path)?;
        let mut rdr = csv::Reader::from_reader(BufReader::new(file));
        let mut records = Vec::new();
        for result in rdr.deserialize() {
            let record: McMillenFluorescenceRecord = result?;
            records.push(record);
        }
        Ok(records)
    }

    /// Convert McMillen records into a MorphoMeshAdapter and TissueSnapshot.
    pub fn mcmillen_to_mesh(
        dim: usize,
        records: &[McMillenFluorescenceRecord],
        seed: u64,
        hyper_proto: ContinuousHV,
        depol_proto: ContinuousHV,
    ) -> (MorphoMeshAdapter, TissueSnapshot) {
        let coords: Vec<(f32, f32)> = records.iter().map(|r| (r.X, r.Y)).collect();
        let adapter = MorphoMeshAdapter::new_from_mea(
            dim,
            &coords,
            5.0, // Adjacency threshold for microns
            seed,
            hyper_proto.clone(),
            depol_proto.clone(),
        );

        let voltages: Vec<f32> = records.iter().map(|r| r.Lifetime).collect();
        let packet = MeaPacket {
            timestamp_ms: 0,
            electrode_voltages: voltages,
        };

        let tissue_hv = adapter.ingest_mea_packet(&packet);
        let spatial_hvs = adapter.spatial_coordinates();

        // Reconstruct cell list for supervisor
        let cell_hvs: Vec<ContinuousHV> = records
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let state = if r.Lifetime >= 0.0 {
                    &hyper_proto
                } else {
                    &depol_proto
                };
                spatial_hvs[i].bind(state)
            })
            .collect();

        (
            adapter,
            TissueSnapshot {
                state_hv: tissue_hv,
                cell_hvs,
            },
        )
    }
}
