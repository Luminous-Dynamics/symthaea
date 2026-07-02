// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::json;
use std::fs::OpenOptions;
use std::io::Write;

pub fn append_event(
    event: &crate::systems::chronicle::ChronicleEventEnvelope,
) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("chronicle/events.jsonl")?;

    let event_json = serde_json::to_string(event)?;
    writeln!(file, "{}", event_json)?;
    Ok(())
}

pub fn record_demo_event() -> std::io::Result<()> {
    let event = crate::systems::chronicle::ChronicleEventEnvelope {
        schema_version: "chronicle.event.v0".to_string(),
        event_id: "evt_00000001".to_string(),
        worldline_id: "seed_age_firstlight".to_string(),
        site_id: Some("old_waterworks".to_string()),
        logical_time: 1,
        event_type: "DeadAuthorityLockInspected".to_string(),
        actor_id: "player_local".to_string(),
        prev_hash: "GENESIS".to_string(),
        payload: json!({
            "pump_id": "PUMP_1",
            "tank_level_percent": 12,
            "authority": "DEAD_AUTHORITY_LOCK"
        }),
        hash: "PLACEHOLDER_HASH".to_string(),
        signature: "PLACEHOLDER_SIG".to_string(),
    };

    append_event(&event)
}

pub fn record_chronicle_event(event_type: &str, payload: serde_json::Value) {
    let _ = std::fs::create_dir_all("chronicle");
    let logical_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let event_id = format!("evt_{}", logical_time);

    let event = crate::systems::chronicle::ChronicleEventEnvelope {
        schema_version: "chronicle.event.v0".to_string(),
        event_id,
        worldline_id: "seed_age_firstlight".to_string(),
        site_id: Some("firstlight_basin".to_string()),
        logical_time,
        event_type: event_type.to_string(),
        actor_id: "player_local".to_string(),
        prev_hash: "GENESIS".to_string(),
        payload,
        hash: "PLACEHOLDER_HASH".to_string(),
        signature: "PLACEHOLDER_SIG".to_string(),
    };

    if let Err(e) = append_event(&event) {
        eprintln!("[chronicle] Failed to write event to log: {:?}", e);
    }
}
