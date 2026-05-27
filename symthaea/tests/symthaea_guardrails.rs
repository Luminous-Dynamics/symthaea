// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use symthaea::Symthaea;
use tempfile::NamedTempFile;

#[tokio::test]
async fn test_hdc_dim_zero_rejected() {
    let result = Symthaea::new(0, 4).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_resume_rejects_zero_dim_state() {
    let mut symthaea = Symthaea::new(512, 4).await.expect("symthaea init");
    let state_file = NamedTempFile::new().expect("temp state file");
    let path = state_file.path().to_string_lossy().to_string();

    symthaea.pause(&path).expect("pause state");

    let mut state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read state"))
            .expect("parse state json");
    state["hdc_dim"] = serde_json::Value::from(0);
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&state).expect("serialize"),
    )
    .expect("write state");

    let resumed = Symthaea::resume(&path);
    assert!(resumed.is_err());
}