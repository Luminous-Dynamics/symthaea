#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;

use craft_graph_integrity::{
    Anchor, CompositeProfile, EntryTypes, LinkTypes, CraftProfile,
    PublishedCredential, RetentionCheck, SkillEndorsement,
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CraftProfileInput {
    pub display_name: String,
    pub headline: String,
    pub bio: String,
    pub location: Option<String>,
    pub website: Option<String>,
    pub avatar_url: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PublishedCredentialInput {
    pub credential_id: String,
    pub title: String,
    pub issuer: String,
    pub issued_on: String,
    pub expires_on: Option<String>,
    pub source_dna: Option<String>,
    pub entry_hash: Option<String>,
    pub action_hash: Option<String>,
    pub summary: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SkillEndorsementInput {
    pub endorsed_agent: String,
    pub skill: String,
    pub rationale: String,
    pub evidence: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SkillEndorsementView {
    pub endorsed_agent: String,
    pub skill: String,
    pub rationale: String,
    pub evidence: Option<String>,
    pub created_at: String,
}

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    hash_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    create_entry(&EntryTypes::Anchor(Anchor(anchor_str.to_string())))?;
    anchor_hash(anchor_str)
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let s = s.trim_end_matches('=');
    let mut result = Vec::with_capacity(s.len() * 3 / 4);
    let mut buffer: u32 = 0;
    let mut bits: u8 = 0;

    for c in s.bytes() {
        let val = match ALPHABET.iter().position(|&x| x == c) {
            Some(v) => v as u32,
            None => match c {
                b'-' => 62,
                b'_' => 63,
                b' ' | b'\n' | b'\r' | b'\t' => continue,
                _ => return None,
            },
        };
        buffer = (buffer << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            result.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }
    Some(result)
}

fn encode_base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    let mut i = 0;
    while i < bytes.len() {
        let b0 = bytes[i];
        let b1 = if i + 1 < bytes.len() { bytes[i + 1] } else { 0 };
        let b2 = if i + 2 < bytes.len() { bytes[i + 2] } else { 0 };
        let triple = ((b0 as u32) << 16) | ((b1 as u32) << 8) | b2 as u32;
        out.push(ALPHABET[((triple >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((triple >> 12) & 0x3f) as usize] as char);
        if i + 1 < bytes.len() {
            out.push(ALPHABET[((triple >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if i + 2 < bytes.len() {
            out.push(ALPHABET[(triple & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        i += 3;
    }
    out
}

fn decode_agent_pubkey(pubkey: &str) -> ExternResult<AgentPubKey> {
    let bytes = base64_decode(pubkey).ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest("Invalid agent pubkey encoding".into()))
    })?;
    if bytes.len() != 39 {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Agent pubkey wrong length: {} bytes (expected 39)",
            bytes.len()
        ))));
    }
    Ok(AgentPubKey::from_raw_39(bytes))
}

fn endorsement_to_view(endorsement: SkillEndorsement) -> SkillEndorsementView {
    SkillEndorsementView {
        endorsed_agent: encode_base64(endorsement.endorsed_agent.get_raw_39().as_ref()),
        skill: endorsement.skill,
        rationale: endorsement.rationale,
        evidence: endorsement.evidence,
        created_at: endorsement.created_at.to_string(),
    }
}

fn latest_action_from_links(links: Vec<Link>) -> Option<ActionHash> {
    links
        .into_iter()
        .max_by_key(|link| link.timestamp)
        .and_then(|link| link.target.into_action_hash())
}

fn load_entries<T: TryFrom<SerializedBytes>>(links: Vec<Link>) -> ExternResult<Vec<T>>
where
    <T as TryFrom<SerializedBytes>>::Error: std::fmt::Debug,
{
    let mut items = Vec::new();
    let mut links = links;
    links.sort_by_key(|link| link.timestamp);
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(entry) = T::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        items.push(entry);
                    }
                }
            }
        }
    }
    Ok(items)
}

#[hdk_extern]
pub fn set_profile(input: CraftProfileInput) -> ExternResult<()> {
    let profile = CraftProfile {
        display_name: input.display_name,
        headline: input.headline,
        bio: input.bio,
        location: input.location,
        website: input.website,
        avatar_url: input.avatar_url,
        updated_at: sys_time()?,
    };

    let action_hash = create_entry(&EntryTypes::CraftProfile(profile))?;
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(agent, action_hash.clone(), LinkTypes::AgentToProfile, ())?;

    Ok(())
}

#[hdk_extern]
pub fn get_my_profile() -> ExternResult<Option<CraftProfile>> {
    let agent = agent_info()?.agent_initial_pubkey;
    get_profile(agent)
}

#[hdk_extern]
pub fn get_profile(agent: AgentPubKey) -> ExternResult<Option<CraftProfile>> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToProfile)?,
        GetStrategy::default(),
    )?;
    if let Some(action_hash) = latest_action_from_links(links) {
        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(Entry::App(bytes)) = record.entry().as_option() {
                if let Ok(profile) = CraftProfile::try_from(
                    SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                ) {
                    return Ok(Some(profile));
                }
            }
        }
    }
    Ok(None)
}

#[hdk_extern]
pub fn publish_credential(input: PublishedCredentialInput) -> ExternResult<()> {
    let now_micros = sys_time()?.as_micros().to_string();
    let published = PublishedCredential {
        credential_id: input.credential_id,
        title: input.title,
        issuer: input.issuer,
        issued_on: input.issued_on,
        expires_on: input.expires_on,
        source_dna: input.source_dna,
        entry_hash: input.entry_hash,
        action_hash: input.action_hash,
        summary: input.summary,
        // Living Credentials: initialize vitality at full strength
        vitality_permille: Some(1000),
        last_retention_check: Some(now_micros),
        mastery_level_at_issue: None, // Set by caller if available
    };

    let action_hash = create_entry(&EntryTypes::PublishedCredential(published))?;
    let agent = agent_info()?.agent_initial_pubkey;
    create_link(agent.clone(), action_hash.clone(), LinkTypes::AgentToCredential, ())?;

    if let Some(profile_action) = latest_action_from_links(get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToProfile)?,
        GetStrategy::default(),
    )?) {
        create_link(profile_action, action_hash.clone(), LinkTypes::ProfileToCredential, ())?;
    }

    Ok(())
}

#[hdk_extern]
pub fn list_my_published_credentials() -> ExternResult<Vec<PublishedCredential>> {
    let agent = agent_info()?.agent_initial_pubkey;
    list_published_credentials(agent)
}

#[hdk_extern]
pub fn list_published_credentials(agent: AgentPubKey) -> ExternResult<Vec<PublishedCredential>> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToCredential)?,
        GetStrategy::default(),
    )?;
    load_entries(links)
}

#[hdk_extern]
pub fn endorse_skill(input: SkillEndorsementInput) -> ExternResult<()> {
    let endorsed_agent = decode_agent_pubkey(&input.endorsed_agent)?;
    let endorsement = SkillEndorsement {
        endorsed_agent: endorsed_agent.clone(),
        skill: input.skill.clone(),
        rationale: input.rationale,
        evidence: input.evidence,
        created_at: sys_time()?,
    };

    let action_hash = create_entry(&EntryTypes::SkillEndorsement(endorsement))?;
    let agent = agent_info()?.agent_initial_pubkey;

    create_link(agent, action_hash.clone(), LinkTypes::AgentToEndorsement, ())?;
    create_link(
        endorsed_agent,
        action_hash.clone(),
        LinkTypes::EndorsedAgentToEndorsement,
        (),
    )?;

    let anchor = ensure_anchor(&format!("skill/{}", input.skill.to_lowercase()))?;
    create_link(anchor, action_hash.clone(), LinkTypes::SkillToEndorsement, ())?;

    Ok(())
}

#[hdk_extern]
pub fn list_skill_endorsements(agent: AgentPubKey) -> ExternResult<Vec<SkillEndorsement>> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::EndorsedAgentToEndorsement)?,
        GetStrategy::default(),
    )?;
    load_entries(links)
}

#[hdk_extern]
pub fn get_my_endorsements_view() -> ExternResult<Vec<SkillEndorsementView>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let endorsements = list_skill_endorsements(agent)?;
    Ok(endorsements.into_iter().map(endorsement_to_view).collect())
}

#[hdk_extern]
pub fn list_skill_endorsements_view(agent_pubkey: String) -> ExternResult<Vec<SkillEndorsementView>> {
    let agent = decode_agent_pubkey(&agent_pubkey)?;
    let endorsements = list_skill_endorsements(agent)?;
    Ok(endorsements.into_iter().map(endorsement_to_view).collect())
}

#[hdk_extern]
pub fn get_my_agent_pubkey() -> ExternResult<String> {
    let agent = agent_info()?.agent_initial_pubkey;
    Ok(encode_base64(agent.get_raw_39().as_ref()))
}

// ============== Living Credentials ==============
//
// Ebbinghaus-based credential vitality: credentials decay over time unless
// the holder demonstrates retention via periodic challenges.
//
// GUARDRAIL: All time-decay math is in the coordinator, NEVER in integrity.
// Integrity only validates deterministic facts (did a RetentionCheck entry get created?).
// Nodes have slightly different clocks — continuous time functions in integrity
// would cause network forks.

/// Ebbinghaus stability: how many minutes before retention drops to 50%.
/// Higher mastery + more reviews = slower forgetting.
///
/// Formula: S = base_minutes × (0.5 + mastery × 2.0) × review_multiplier
/// - base = 1440 min (1 day) for zero mastery
/// - mastery 1.0 → S = 1440 × 2.5 = 3600 min (~2.5 days)
/// - Each successful review multiplies S by 1.5 (up to 10 reviews)
fn ebbinghaus_stability(mastery_permille: u16, successful_reviews: u32) -> f64 {
    let mastery = (mastery_permille as f64) / 1000.0;
    let base = 1440.0; // 1 day in minutes
    let mastery_factor = 0.5 + mastery * 2.0;
    let review_factor = 1.5_f64.powi(successful_reviews.min(10) as i32);
    base * mastery_factor * review_factor
}

/// Predict retention at a given elapsed time using Ebbinghaus forgetting curve.
/// R(t) = e^(-t/S) where S = stability, t = elapsed minutes.
/// Returns 0-1000 permille.
fn predict_vitality(stability_minutes: f64, elapsed_minutes: f64) -> u16 {
    if stability_minutes <= 0.0 || elapsed_minutes < 0.0 {
        return 1000; // Default: full vitality
    }
    let retention = (-elapsed_minutes / stability_minutes).exp();
    (retention * 1000.0).round().clamp(0.0, 1000.0) as u16
}

/// Credential vitality view returned to the frontend.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CredentialVitality {
    pub credential_id: String,
    pub title: String,
    pub vitality_permille: u16,
    pub stability_minutes: f64,
    pub minutes_since_last_check: f64,
    pub retention_checks_count: u32,
    pub next_review_recommended_minutes: f64,
    /// Relearning efficiency: ratio of latest relearn time to first learn time.
    /// 500 = relearning takes half the time (knowledge was latent, not lost).
    /// 1000 = no improvement. 0 = not enough data.
    pub relearning_efficiency_permille: u16,
}

/// Get the current vitality of a published credential.
///
/// Computes decay using Ebbinghaus: vitality = e^(-t/S) where S depends on
/// mastery at issuance and number of successful retention checks.
#[hdk_extern]
pub fn get_credential_vitality(credential_hash: ActionHash) -> ExternResult<CredentialVitality> {
    // Fetch credential
    let record = get(credential_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Credential not found".into())))?;

    let credential: PublishedCredential = match record.entry().as_option() {
        Some(Entry::App(bytes)) => PublishedCredential::try_from(
            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
        ).map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {:?}", e))))?,
        _ => return Err(wasm_error!(WasmErrorInner::Guest("Not an entry".into()))),
    };

    // Count retention checks (link-counting)
    let retention_links = get_links(
        LinkQuery::try_new(credential_hash, LinkTypes::CredentialToRetentionCheck)?,
        GetStrategy::Local,
    )?;
    let checks_count = retention_links.len() as u32;

    // Compute stability
    let mastery = credential.mastery_level_at_issue.unwrap_or(500); // Default: 50% mastery
    let stability = ebbinghaus_stability(mastery, checks_count);

    // Compute elapsed time since last retention check (or publish time).
    // Find the most recent RetentionCheck's checked_at timestamp from links,
    // since the credential's last_retention_check field is only set at publish time.
    let now = sys_time()?;
    let now_micros = now.as_micros();

    let most_recent_check_micros: Option<i64> = {
        let mut latest: Option<i64> = None;
        for link in &retention_links {
            if let Some(hash) = link.target.clone().into_action_hash() {
                if let Some(record) = get(hash, GetOptions::default())? {
                    if let Some(Entry::App(bytes)) = record.entry().as_option() {
                        if let Ok(check) = RetentionCheck::try_from(
                            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                        ) {
                            let ts = check.checked_at.as_micros();
                            latest = Some(latest.map_or(ts, |prev: i64| prev.max(ts)));
                        }
                    }
                }
            }
        }
        latest
    };

    let last_check_micros = most_recent_check_micros.unwrap_or_else(|| {
        // No retention checks yet — use publish timestamp or issued_on as baseline
        if let Some(ref check_ts) = credential.last_retention_check {
            check_ts.parse::<i64>().unwrap_or(now_micros)
        } else {
            credential.issued_on.parse::<i64>().unwrap_or(now_micros)
        }
    });

    let elapsed_minutes = ((now_micros - last_check_micros) as f64) / (60.0 * 1_000_000.0);
    let vitality = predict_vitality(stability, elapsed_minutes.max(0.0));

    // Recommended next review: when vitality would drop to 800 (80%)
    let target_retention = 0.8_f64;
    let next_review = if target_retention > 0.0 && target_retention < 1.0 {
        -stability * target_retention.ln()
    } else {
        stability
    };
    let remaining = (next_review - elapsed_minutes).max(0.0);

    // Compute relearning efficiency from retention check intervals.
    // If intervals between checks are shrinking, the learner relearns faster.
    let relearning_efficiency = if retention_links.len() >= 2 {
        // Fetch check timestamps to compute intervals
        let mut check_timestamps: Vec<i64> = Vec::new();
        for link in &retention_links {
            if let Some(hash) = link.target.clone().into_action_hash() {
                if let Some(record) = get(hash, GetOptions::default())? {
                    if let Some(Entry::App(bytes)) = record.entry().as_option() {
                        if let Ok(check) = RetentionCheck::try_from(
                            SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                        ) {
                            check_timestamps.push(check.checked_at.as_micros());
                        }
                    }
                }
            }
        }
        check_timestamps.sort();

        if check_timestamps.len() >= 2 {
            let first_interval = check_timestamps[1] - check_timestamps[0];
            let last_interval = check_timestamps[check_timestamps.len() - 1]
                - check_timestamps[check_timestamps.len() - 2];
            if first_interval > 0 {
                // Ratio: last/first. Lower = faster relearning.
                let ratio = last_interval as f64 / first_interval as f64;
                (ratio * 1000.0).clamp(0.0, 1000.0) as u16
            } else {
                0
            }
        } else {
            0
        }
    } else {
        0 // Not enough data
    };

    Ok(CredentialVitality {
        credential_id: credential.credential_id,
        title: credential.title,
        vitality_permille: vitality,
        stability_minutes: stability,
        minutes_since_last_check: elapsed_minutes,
        retention_checks_count: checks_count,
        next_review_recommended_minutes: remaining,
        relearning_efficiency_permille: relearning_efficiency,
    })
}

/// Input for recording a retention check.
#[derive(Serialize, Deserialize, Debug)]
pub struct RecordRetentionCheckInput {
    pub credential_hash: ActionHash,
    pub retention_score_permille: u16,
    pub questions_attempted: u16,
    pub questions_correct: u16,
}

/// Record a retention check for a credential (refreshes vitality).
///
/// Creates a RetentionCheck entry and links it to the credential.
/// The next call to `get_credential_vitality` will reflect the refreshed state.
#[hdk_extern]
pub fn record_retention_check(input: RecordRetentionCheckInput) -> ExternResult<ActionHash> {
    let now = sys_time()?;

    let check = RetentionCheck {
        credential_id: input.credential_hash.to_string(),
        retention_score_permille: input.retention_score_permille,
        questions_attempted: input.questions_attempted,
        questions_correct: input.questions_correct,
        checked_at: now,
    };

    let check_hash = create_entry(EntryTypes::RetentionCheck(check))?;

    // Link credential → retention check
    create_link(
        input.credential_hash,
        check_hash.clone(),
        LinkTypes::CredentialToRetentionCheck,
        vec![],
    )?;

    Ok(check_hash)
}

/// List all credentials needing review (vitality < 800 permille).
#[hdk_extern]
pub fn list_credentials_needing_review(_: ()) -> ExternResult<Vec<CredentialVitality>> {
    // Fetch credential action hashes via AgentToCredential links
    let agent_hash: AnyDhtHash = agent_info()?.agent_initial_pubkey.into();
    let links = get_links(
        LinkQuery::try_new(agent_hash, LinkTypes::AgentToCredential)?,
        GetStrategy::Local,
    )?;

    let mut needing_review = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Ok(vitality) = get_credential_vitality(hash) {
                if vitality.vitality_permille < 800 {
                    needing_review.push(vitality);
                }
            }
        }
    }

    Ok(needing_review)
}

// ============== Credential Composability ==============
//
// Detects when an agent's credentials collectively cover a career archetype.
// Composite profiles are more valuable than the sum of individual credentials.

/// Known career archetypes with required credential keywords.
/// The frontend/backend can extend this list dynamically.
const ARCHETYPES: &[(&str, &[&str])] = &[
    ("Full Stack Consciousness Developer", &["rust", "holochain", "wasm", "leptos"]),
    ("CAPS Mathematics Educator", &["mathematics", "teaching", "caps"]),
    ("Data Science Professional", &["python", "statistics", "machine learning", "data"]),
    ("Cybersecurity Specialist", &["cybersecurity", "network", "security", "encryption"]),
    ("Decentralized Systems Architect", &["holochain", "distributed", "p2p", "consensus"]),
];

/// Detect composite credential profiles for the calling agent.
///
/// Scans published credentials against known archetypes.
/// Creates CompositeProfile entries for any archetype with >70% coverage.
#[hdk_extern]
pub fn detect_composites(_: ()) -> ExternResult<Vec<CompositeProfile>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let credentials = list_published_credentials(agent.clone())?;
    let now = sys_time()?;

    // Extract all skill keywords from credential titles and summaries
    let agent_skills: std::collections::HashSet<String> = credentials
        .iter()
        .flat_map(|c| {
            let mut keywords = Vec::new();
            for word in c.title.to_lowercase().split_whitespace() {
                keywords.push(word.to_string());
            }
            if let Some(ref summary) = c.summary {
                for word in summary.to_lowercase().split_whitespace() {
                    keywords.push(word.to_string());
                }
            }
            keywords
        })
        .collect();

    let mut composites = Vec::new();

    for &(archetype_name, required) in ARCHETYPES {
        let matched: Vec<&str> = required
            .iter()
            .filter(|&&skill| agent_skills.contains(skill))
            .copied()
            .collect();

        let coverage = if required.is_empty() {
            0
        } else {
            (matched.len() * 1000 / required.len()) as u16
        };

        // Threshold: 70% coverage = composite detected
        if coverage >= 700 {
            let matching_creds: Vec<String> = credentials
                .iter()
                .filter(|c| {
                    let title_lower = c.title.to_lowercase();
                    matched.iter().any(|skill| title_lower.contains(skill))
                })
                .map(|c| c.title.clone())
                .collect();

            let composite = CompositeProfile {
                agent: agent.clone(),
                archetype_name: archetype_name.to_string(),
                credential_titles: matching_creds,
                coverage_permille: coverage,
                career_profile_match: Some(archetype_name.to_string()),
                detected_at: now,
            };

            let hash = create_entry(EntryTypes::CompositeProfile(composite.clone()))?;

            // Link: agent -> composite
            let agent_hash: AnyDhtHash = agent.clone().into();
            create_link(agent_hash, hash, LinkTypes::AgentToCompositeProfile, vec![])?;

            composites.push(composite);
        }
    }

    Ok(composites)
}

/// List existing composite profiles for the calling agent.
#[hdk_extern]
pub fn list_my_composites(_: ()) -> ExternResult<Vec<CompositeProfile>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_hash: AnyDhtHash = agent.into();

    let links = get_links(
        LinkQuery::try_new(agent_hash, LinkTypes::AgentToCompositeProfile)?,
        GetStrategy::Local,
    )?;

    let mut composites = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(c) = CompositeProfile::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec())),
                    ) {
                        composites.push(c);
                    }
                }
            }
        }
    }
    Ok(composites)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ebbinghaus_stability_zero_mastery() {
        let s = ebbinghaus_stability(0, 0);
        assert!((s - 720.0).abs() < 1.0); // 1440 * 0.5 = 720
    }

    #[test]
    fn test_ebbinghaus_stability_full_mastery() {
        let s = ebbinghaus_stability(1000, 0);
        assert!((s - 3600.0).abs() < 1.0); // 1440 * 2.5 = 3600
    }

    #[test]
    fn test_ebbinghaus_stability_with_reviews() {
        let s0 = ebbinghaus_stability(500, 0);
        let s3 = ebbinghaus_stability(500, 3);
        assert!(s3 > s0 * 3.0); // 1.5^3 = 3.375x multiplier
    }

    #[test]
    fn test_predict_vitality_at_zero_time() {
        let v = predict_vitality(1440.0, 0.0);
        assert_eq!(v, 1000); // e^0 = 1.0 → 1000 permille
    }

    #[test]
    fn test_predict_vitality_at_half_life() {
        // At t = S * ln(2), retention should be ~50%
        let stability = 1440.0;
        let half_life = stability * 2.0_f64.ln(); // ~998 minutes
        let v = predict_vitality(stability, half_life);
        assert!((v as i32 - 500).abs() < 10); // ~500 permille
    }

    #[test]
    fn test_predict_vitality_decays_over_time() {
        let s = 1440.0;
        let v1 = predict_vitality(s, 100.0);
        let v2 = predict_vitality(s, 1000.0);
        let v3 = predict_vitality(s, 10000.0);
        assert!(v1 > v2);
        assert!(v2 > v3);
    }

    #[test]
    fn test_predict_vitality_never_negative() {
        let v = predict_vitality(100.0, 1_000_000.0);
        assert!(v <= 1000);
    }

    #[test]
    fn test_stability_capped_at_10_reviews() {
        let s10 = ebbinghaus_stability(500, 10);
        let s20 = ebbinghaus_stability(500, 20);
        assert!((s10 - s20).abs() < 0.01); // Reviews beyond 10 don't increase stability
    }
}

#[hdk_extern]
pub fn list_skill_endorsements_for_skill(skill: String) -> ExternResult<Vec<SkillEndorsement>> {
    let anchor = anchor_hash(&format!("skill/{}", skill.to_lowercase()))?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::SkillToEndorsement)?,
        GetStrategy::default(),
    )?;
    load_entries(links)
}
