// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Policy-enforced browser journey against a local Praxis instance.
//!
//! This example uses the canonical `BrowserExecutor`; local-network access is
//! explicitly granted for development and every navigation produces an action
//! receipt.
//!
//! Prerequisites: Chrome running with `--remote-debugging-port=9222`
//!                Praxis served on `localhost:8107`

use anyhow::{Result, ensure};
use symthaea_browser::{
    BrowserAction, BrowserAgentConfig, BrowserExecutor, BrowserSafetyPolicy, CdpSession,
};

const PHI: f64 = 0.72;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Symthaea Browser Agent: Policy Journey ===\n");

    let mut safety = BrowserSafetyPolicy::interactive();
    safety.allow_private_networks = true;
    safety.url_allowlist = vec!["localhost".to_string()];

    let config = BrowserAgentConfig {
        cdp_url: Some("http://localhost:9222".to_string()),
        safety: safety.clone(),
        ..BrowserAgentConfig::default()
    };
    let session = CdpSession::connect_existing_with_config(&config).await?;
    let executor = BrowserExecutor::new(&session, &safety, PHI);

    let routes = [
        ("Home", "/"),
        ("Dashboard", "/dashboard"),
        ("Knowledge Garden", "/skill-map"),
        ("Review", "/review"),
        ("Exam Prep", "/exam-prep"),
        ("Mock Exam", "/mock-exam"),
        ("Pathways", "/pathways"),
        ("Governance", "/governance"),
    ];

    for (name, route) in routes {
        let url = format!("http://localhost:8107{route}");
        let receipt = executor
            .execute(BrowserAction::Navigate { url: url.clone() })
            .await;
        println!(
            "[→] {name}: {:?} ({} ms)",
            receipt.outcome, receipt.elapsed_ms
        );
        ensure!(
            receipt.succeeded(),
            "navigation to {url} failed: {:?}",
            receipt.outcome
        );

        let observation = session.observe().await?;
        println!(
            "    {} — {} elements, {} interactive",
            observation.title,
            observation.elements.len(),
            observation.interactive_count()
        );
        for element in observation
            .elements
            .iter()
            .filter(|element| {
                matches!(
                    element.role.as_str(),
                    "button" | "link" | "textbox" | "combobox" | "checkbox"
                )
            })
            .take(6)
        {
            println!("      {}", element.to_text_line());
        }
    }

    println!("\n=== Journey Complete ===");
    println!("Every navigation was capability-, Phi-, URL-, and origin-checked.");
    Ok(())
}
