// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::evidence::EvidencePacket;
use serde::{Deserialize, Serialize};

/// Maps EvidencePackets to Morphos/Symthaea Bioregion Channels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioregionChannelUpdate {
    pub channel_name: String,
    pub value: f64,
    pub evidence_packet_id: uuid::Uuid,
}

impl BioregionChannelUpdate {
    pub fn from_evidence(packet: &EvidencePacket) -> Self {
        // Map feature names to BioregionSteward channels
        let channel_name = match packet.feature_name.as_str() {
            "VegetationHealth" => "VegetationHealth",
            "WaterExtent" => "WaterExtent",
            "FloodExtent" => "FloodExtentSAR",
            "Deforestation" => "DeforestationChange",
            _ => "GenericEcologicalChannel",
        }
        .to_string();

        Self {
            channel_name,
            value: packet.value,
            evidence_packet_id: packet.id,
        }
    }
}
