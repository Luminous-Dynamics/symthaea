// SPDX-License-Identifier: AGPL-3.0-or-later

// ChronicleEventEnvelope defines the structure for persistent events in the log.
// Used for tracking history, repair outcomes, and player precedents.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChronicleEventEnvelope {
    pub schema_version: String,
    pub event_id: String,
    pub worldline_id: String,
    pub site_id: Option<String>,
    pub logical_time: u64,
    pub event_type: String,
    pub actor_id: String,
    pub prev_hash: String,
    pub payload: serde_json::Value,
    pub hash: String,
    pub signature: String,
}
