//! Conductor submission module (feature-gated behind `submit`).
//!
//! Connects to a running Holochain conductor via WebSocket and submits
//! scanned dependencies + usage receipts using the bulk registration externs.

use anyhow::{Context, Result};
use holochain_client::{AppWebsocket, ZomeCallTarget};
use serde::Serialize;

/// Payload matching the zome's DependencyIdentity entry type.
/// Timestamps and booleans are set by the zome on creation.
#[derive(Serialize, Clone, Debug)]
pub struct SubmitDependency {
    pub id: String,
    pub name: String,
    pub ecosystem: String,
    pub maintainer_did: String,
    pub repository_url: Option<String>,
    pub license: Option<String>,
    pub description: String,
    pub version: Option<String>,
    pub registered_at: i64, // microseconds
    pub verified: bool,
}

/// Payload matching the zome's UsageReceipt entry type.
#[derive(Serialize, Clone, Debug)]
pub struct SubmitUsageReceipt {
    pub id: String,
    pub dependency_id: String,
    pub user_did: String,
    pub organization: Option<String>,
    pub usage_type: String,
    pub scale: Option<String>,
    pub version_range: Option<String>,
    pub context: Option<String>,
    pub attested_at: i64, // microseconds
}

pub async fn submit_to_conductor(
    ws_url: &str,
    app_id: &str,
    deps: Vec<SubmitDependency>,
    receipts: Vec<SubmitUsageReceipt>,
) -> Result<()> {
    eprintln!("Connecting to conductor at {}...", ws_url);

    let app_ws = AppWebsocket::connect(ws_url.to_string())
        .await
        .context("Failed to connect to Holochain conductor")?;

    // Bulk register dependencies
    let dep_count = deps.len();
    if !deps.is_empty() {
        eprintln!(
            "Submitting {} dependencies via bulk_register_dependencies...",
            dep_count
        );

        let payload = serde_json::to_value(&deps).context("Failed to serialize dependencies")?;

        let _result = app_ws
            .call_zome(
                ZomeCallTarget::RoleName {
                    role_name: "attribution".into(),
                    zome_name: "registry".into(),
                    fn_name: "bulk_register_dependencies".into(),
                },
                payload,
            )
            .await
            .context("bulk_register_dependencies call failed")?;

        eprintln!("  Registered {} dependencies", dep_count);
    }

    // Bulk record usage
    let receipt_count = receipts.len();
    if !receipts.is_empty() {
        eprintln!(
            "Submitting {} usage receipts via bulk_record_usage...",
            receipt_count
        );

        let payload =
            serde_json::to_value(&receipts).context("Failed to serialize usage receipts")?;

        let _result = app_ws
            .call_zome(
                ZomeCallTarget::RoleName {
                    role_name: "attribution".into(),
                    zome_name: "usage".into(),
                    fn_name: "bulk_record_usage".into(),
                },
                payload,
            )
            .await
            .context("bulk_record_usage call failed")?;

        eprintln!("  Recorded {} usage receipts", receipt_count);
    }

    eprintln!(
        "Done: {} deps + {} receipts submitted to {} (app: {})",
        dep_count, receipt_count, ws_url, app_id
    );

    Ok(())
}
