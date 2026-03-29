// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Event processing pipeline
//!
//! Transforms events into VCs, signs them, and creates DKG claims.

use crate::{lineage, vc, AppState};
use anyhow::Result;
use claim_model::{DkgClaim, SupplyEventVC};
use std::sync::Arc;
use tracing::debug;

/// Result of processing an event
pub struct ProcessedEvent {
    pub vc_jwt: String,
    pub claim: DkgClaim,
}

/// Process a supply chain event through the full pipeline
pub async fn process_event(state: &Arc<AppState>, vc: SupplyEventVC) -> Result<ProcessedEvent> {
    debug!("Processing event for batch: {}", vc.credential_subject.batch_id);

    // Step 1: Sign the VC to create a JWT
    let vc_jwt = vc::sign_vc(&state.keypair, &vc)?;

    // Step 2: Resolve lineage (find previous claims for this batch)
    let previous_claims = lineage::resolve_previous_claims(state, &vc).await;

    // Step 3: Create DKG claim with lineage
    let claim = DkgClaim::from_vc(&vc, vc_jwt.clone(), previous_claims);

    // Step 4: (Future) Publish to DKG network
    // dkg_client::publish_claim(&claim).await?;

    Ok(ProcessedEvent { vc_jwt, claim })
}
