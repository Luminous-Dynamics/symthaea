#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// issuer-trust-tier coordinator — classify credential issuers into tiers.
//
// The tier is a hint for verifiers: "this issuer is sovereign" means
// a credential from this issuer carries more weight than a peer
// classification. The tier NEVER influences Mycelix governance weight
// or consciousness-tier computation. It lives here as metadata for
// the lawful-identity credential import + verification flow only.

use hdk::prelude::*;
use issuer_trust_tier_integrity::{EntryTypes, IssuerClassification, IssuerTier, LinkTypes};

// ============================================================================
// Helpers
// ============================================================================

fn now_iso_8601() -> ExternResult<String> {
    let ts = sys_time()?;
    let (secs, nanos) = ts.as_seconds_and_nanos();
    Ok(format!("{}.{:09}Z", secs, nanos))
}

/// Path anchoring classifications for a specific tier.
fn tier_path(tier: IssuerTier) -> Path {
    Path::from(format!("tier/{}", tier.as_str()))
}

/// Path anchoring classification history for a specific issuer.
fn issuer_path(issuer_did: &str) -> Path {
    Path::from(format!("issuer/{}", issuer_did))
}

// ============================================================================
// Classify
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClassifyIssuerInput {
    pub issuer_did: String,
    pub tier: IssuerTier,
    pub rationale: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClassifyIssuerOutput {
    pub action_hash: ActionHash,
}

#[hdk_extern]
pub fn classify_issuer(input: ClassifyIssuerInput) -> ExternResult<ClassifyIssuerOutput> {
    if input.issuer_did.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "issuer_did must not be empty".to_string()
        )));
    }

    let entry = IssuerClassification {
        issuer_did: input.issuer_did.clone(),
        tier: input.tier,
        classified_at: now_iso_8601()?,
        rationale: input.rationale,
    };

    let ah = create_entry(&EntryTypes::IssuerClassification(entry))?;

    // Link from the tier anchor for tier queries.
    let tier_anchor = tier_path(input.tier).path_entry_hash()?;
    create_link(
        tier_anchor,
        ah.clone(),
        LinkTypes::TierAnchor,
        LinkTag::new(input.issuer_did.as_bytes().to_vec()),
    )?;

    // Link from the issuer anchor for issuer-DID-keyed lookup.
    let issuer_anchor = issuer_path(&input.issuer_did).path_entry_hash()?;
    create_link(
        issuer_anchor,
        ah.clone(),
        LinkTypes::IssuerAnchor,
        LinkTag::new(input.tier.as_str().as_bytes().to_vec()),
    )?;

    Ok(ClassifyIssuerOutput { action_hash: ah })
}

// ============================================================================
// Lookup latest tier for an issuer
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LookupTierInput {
    pub issuer_did: String,
}

#[hdk_extern]
pub fn lookup_tier(input: LookupTierInput) -> ExternResult<Option<IssuerClassification>> {
    let anchor = issuer_path(&input.issuer_did).path_entry_hash()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::IssuerAnchor)?,
        GetStrategy::Local,
    )?;

    // Pick the most recent classification by timestamp.
    let mut best: Option<IssuerClassification> = None;
    for link in links {
        let ah: ActionHash = link
            .target
            .try_into()
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("bad link target".to_string())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Ok(Some(entry)) = record.entry().to_app_option::<IssuerClassification>() {
                match &best {
                    None => best = Some(entry),
                    Some(prev) if entry.classified_at > prev.classified_at => best = Some(entry),
                    _ => {}
                }
            }
        }
    }
    Ok(best)
}

// ============================================================================
// List issuers at a given tier
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ListByTierInput {
    pub tier: IssuerTier,
}

#[hdk_extern]
pub fn list_by_tier(input: ListByTierInput) -> ExternResult<Vec<IssuerClassification>> {
    let anchor = tier_path(input.tier).path_entry_hash()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::TierAnchor)?,
        GetStrategy::Local,
    )?;

    let mut out = Vec::with_capacity(links.len());
    for link in links {
        let ah: ActionHash = link
            .target
            .try_into()
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("bad link target".to_string())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            if let Ok(Some(entry)) = record.entry().to_app_option::<IssuerClassification>() {
                out.push(entry);
            }
        }
    }
    Ok(out)
}

// ============================================================================
// Ping
// ============================================================================

#[hdk_extern]
pub fn ping(_: ()) -> ExternResult<String> {
    Ok("issuer_trust_tier:pong".to_string())
}
