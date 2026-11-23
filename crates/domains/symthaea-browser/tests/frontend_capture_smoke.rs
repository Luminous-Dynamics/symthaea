// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Manual smoke test for the frontend evidence capture path.
//!
//! Run explicitly on a host with Chromium installed:
//! `cargo test -p symthaea-browser --test frontend_capture_smoke -- --ignored --nocapture`
//!
//! The test launches a fresh headless browser and captures about:blank. It does
//! not navigate to the network or use a pre-existing user's browser profile.

use symthaea_browser::{CdpSession, BrowserAgentConfig, FrontendEvidenceContext};

#[tokio::test]
#[ignore = "requires a local Chrome/Chromium binary; no network navigation is used"]
async fn captures_and_verifies_blank_page_evidence() -> anyhow::Result<()> {
    let config = BrowserAgentConfig::default();
    let session = CdpSession::connect(&config).await?;

    let context = FrontendEvidenceContext {
        run_id: "manual-capture-smoke".into(),
        task_id: "SFEB-CAPTURE-SMOKE".into(),
        fixture_id: "about-blank".into(),
        source_commit: "0".repeat(40),
        environment_manifest_sha256: "0".repeat(64),
    };

    let bundle = session.capture_frontend_evidence(context).await?;
    assert!(bundle.verify_integrity());
    assert!(bundle.screenshot_png.starts_with(b"\x89PNG\r\n\x1a\n"));
    assert_eq!(bundle.receipt.viewport.width, config.viewport_width);
    assert_eq!(bundle.receipt.viewport.height, config.viewport_height);
    assert_eq!(bundle.receipt.artifacts.len(), 3);
    assert_eq!(
        bundle.receipt.artifacts[0].kind,
        "screenshot",
        "receipt artifact order is stable"
    );

    Ok(())
}
