// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain summary cards, live-backed shell adapters, and launch URL resolution.

use domain_mail::types::{InboxSummary, ThreadSummary, TrustHealth, TrustTier};
use governance_leptos_types::{
    CharterView, ConsciousnessThresholds, CouncilView, ProposalStatus, ProposalType, ProposalView,
    TimelockView,
};
use leptos::prelude::*;
use mycelix_leptos_core::{
    AvailabilityState, AvailabilityStateKind, FreshnessBadge, FreshnessLevel, SummaryActionItem,
    SummaryAttentionItem as SharedSummaryAttentionItem, SummaryCard, SummaryMetricItem,
    SummaryStatusBadge,
};
use personal_leptos_types::{
    ActivityItemView, ConsentGrantView, HealthRecordView, ProfileView, StoredCredentialView,
};
use sensorium_domain_trait::{
    AttentionLevel, DomainAttentionItem, DomainAvailability, DomainLaunchTarget, DomainMetric,
    DomainSummaryCard, LaunchKind,
};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

use crate::identity::{ConductorStatus, SensoriumIdentity, VaultState};

#[component]
pub fn SensoriumDomainSummaryBlock(domain_id: String, summary: DomainSummaryCard) -> impl IntoView {
    match domain_id.as_str() {
        "personal" | "governance" | "commons" | "mail" => {
            view! { <LiveDomainSummary domain_id fallback=summary /> }.into_any()
        }
        _ => render_sensorium_domain_summary(domain_id, summary),
    }
}

#[component]
fn LiveDomainSummary(domain_id: String, fallback: DomainSummaryCard) -> impl IntoView {
    let identity = use_context::<SensoriumIdentity>().expect("SensoriumIdentity");
    let live_summary = RwSignal::new(fallback.clone());
    let loading = RwSignal::new(false);
    let refresh_message = live_refresh_message(&domain_id);
    let effect_domain_id = domain_id.clone();
    let render_domain_id = domain_id.clone();

    Effect::new(move |_| {
        let connected = identity.conductor_status.get() == ConductorStatus::Connected;
        let unlocked = identity.vault.get() == VaultState::Unlocked;
        let domain_key = effect_domain_id.clone();

        if domain_key == "personal" && !unlocked {
            let mut summary = fallback.clone();
            summary.availability = DomainAvailability::Locked;
            live_summary.set(summary);
            loading.set(false);
            return;
        }

        if !connected {
            let mut summary = fallback.clone();
            summary.availability = DomainAvailability::Mock;
            live_summary.set(summary);
            loading.set(false);
            return;
        }

        let identity = identity.clone();
        let domain_id = domain_key.clone();
        let fallback_summary = fallback.clone();
        let live_summary_signal = live_summary;
        let loading_signal = loading;
        loading_signal.set(true);
        wasm_bindgen_futures::spawn_local(async move {
            let summary = hydrate_domain_summary(&domain_id, identity, fallback_summary.clone())
                .await
                .unwrap_or(fallback_summary);
            live_summary_signal.set(summary);
            loading_signal.set(false);
        });
    });

    view! {
        <div>
            <Show when=move || loading.get()>
                <p style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem;">
                    {refresh_message}
                </p>
            </Show>
            {move || render_sensorium_domain_summary(render_domain_id.clone(), live_summary.get())}
        </div>
    }
}

fn live_refresh_message(domain_id: &str) -> &'static str {
    match domain_id {
        "personal" => "Refreshing vault posture from live conductor...",
        "governance" => "Refreshing governance posture from live conductor...",
        "commons" => "Refreshing commons coordination posture from live conductor...",
        "mail" => "Refreshing Pulse communication posture from live backend...",
        _ => "Refreshing domain posture from live conductor...",
    }
}

fn render_sensorium_domain_summary(domain_id: String, summary: DomainSummaryCard) -> AnyView {
    let actions = summary_actions(&domain_id, &summary);
    let freshness_view = summary.updated_at.map(|updated_at| {
        let detail = format!("Updated {}", format_unix_micros(updated_at));
        let level = freshness_level(updated_at);
        view! { <FreshnessBadge level detail /> }
    });

    if summary.availability != DomainAvailability::Live {
        let primary_action = summary.primary_launch.as_ref().map(|launch| {
            let label = if launch.requires_unlock {
                format!("{} (Unlock)", launch.label)
            } else {
                launch.label.to_string()
            };
            if let Some(href) = launch_href(&domain_id, launch) {
                view! {
                    <a class="btn btn-primary" href=href target="_blank" rel="noopener">
                        {label}
                    </a>
                }
                .into_any()
            } else {
                view! { <button class="btn btn-primary" disabled=true>{label}</button> }.into_any()
            }
        });

        return view! {
            <div class="sensorium-summary">
                <AvailabilityState
                    kind=map_availability_state(summary.availability)
                    title={summary.title}
                    description={summary.status_line}
                    action={primary_action}
                />
            </div>
        }
        .into_any();
    }

    view! {
        <div class="sensorium-summary">
            <div style="display: flex; justify-content: flex-end; margin-bottom: 0.5rem;">
                {freshness_view}
            </div>
            <SummaryCard
                title={summary.title}
                status_badge={SummaryStatusBadge {
                    label: summary.availability.label().into(),
                    class_name: Some(summary.availability.css_class().into()),
                }}
                status_line={summary.status_line}
                metrics={summary.metrics.into_iter().map(map_metric).collect::<Vec<_>>()}
                attention={summary.attention.into_iter().map(map_attention).collect::<Vec<_>>()}
                actions={actions}
                footer_note={None::<String>}
            />
        </div>
    }
    .into_any()
}

fn map_metric(metric: DomainMetric) -> SummaryMetricItem {
    SummaryMetricItem {
        id: metric.id.into(),
        label: metric.label,
        value: metric.value,
        hint: metric.hint,
        tone_class: metric.tone.map(|tone| format!("tone-{tone}")),
    }
}

fn map_attention(item: DomainAttentionItem) -> SharedSummaryAttentionItem {
    SharedSummaryAttentionItem {
        id: item.id,
        label: item.label,
        detail: item.detail,
        level_label: item.level.label().into(),
        level_class: Some(item.level.css_class().into()),
        accent_color: Some(attention_accent(item.level).into()),
    }
}

fn summary_actions(domain_id: &str, summary: &DomainSummaryCard) -> Vec<SummaryActionItem> {
    let mut actions = Vec::new();

    if let Some(launch) = &summary.primary_launch {
        actions.push(SummaryActionItem {
            id: launch.id.into(),
            label: if launch.requires_unlock {
                format!("{} (Unlock)", launch.label)
            } else {
                launch.label.into()
            },
            href: launch_href(domain_id, launch),
            primary: true,
            disabled: launch_href(domain_id, launch).is_none(),
        });
    }

    actions.extend(
        summary
            .secondary_launches
            .iter()
            .map(|launch| SummaryActionItem {
                id: launch.id.into(),
                label: launch.label.into(),
                href: launch_href(domain_id, launch),
                primary: false,
                disabled: launch_href(domain_id, launch).is_none(),
            }),
    );

    actions
}

fn attention_accent(level: AttentionLevel) -> &'static str {
    match level {
        AttentionLevel::Quiet => "#64748b",
        AttentionLevel::Notice => "#22c55e",
        AttentionLevel::ActionNeeded => "#f59e0b",
        AttentionLevel::Urgent => "#ef4444",
    }
}

fn format_unix_micros(timestamp_micros: i64) -> String {
    let millis = (timestamp_micros / 1000) as f64;
    let date = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(millis));
    date.to_locale_string("en-US", &wasm_bindgen::JsValue::UNDEFINED)
        .as_string()
        .unwrap_or_else(|| "recently".into())
}

fn freshness_level(timestamp_micros: i64) -> FreshnessLevel {
    let now_micros = (js_sys::Date::now() * 1000.0) as i64;
    let age_minutes = now_micros.saturating_sub(timestamp_micros) / 60_000_000;
    if age_minutes <= 5 {
        FreshnessLevel::Fresh
    } else if age_minutes <= 60 {
        FreshnessLevel::Aging
    } else {
        FreshnessLevel::Stale
    }
}

fn map_availability_state(availability: DomainAvailability) -> AvailabilityStateKind {
    match availability {
        DomainAvailability::Live => AvailabilityStateKind::Live,
        DomainAvailability::Mock => AvailabilityStateKind::Mock,
        DomainAvailability::Empty => AvailabilityStateKind::Empty,
        DomainAvailability::Locked => AvailabilityStateKind::Locked,
        DomainAvailability::Degraded => AvailabilityStateKind::Degraded,
        DomainAvailability::Unavailable => AvailabilityStateKind::Unavailable,
    }
}

const MAIL_API: &str = "http://localhost:3001";
const PULSE_SUMMARY_STORAGE_KEY: &str = "mycelix:pulse:inbox_summary:v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MailSummaryMode {
    Api,
    LocalBridge,
    Mock,
}

async fn fetch_mail_inbox_summary() -> Result<InboxSummary, String> {
    let window = web_sys::window().ok_or("no window")?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(
        window.fetch_with_str(&format!("{MAIL_API}/api/inbox/summary")),
    )
    .await
    .map_err(|e| format!("fetch: {:?}", e))?;
    let resp: web_sys::Response =
        wasm_bindgen::JsCast::dyn_into(resp_value).map_err(|_| "not a Response".to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let json = wasm_bindgen_futures::JsFuture::from(resp.json().map_err(|_| "json() failed")?)
        .await
        .map_err(|e| format!("json: {:?}", e))?;
    serde_wasm_bindgen::from_value(json).map_err(|e| format!("deserialize: {e}"))
}

fn load_local_mail_inbox_summary() -> Option<InboxSummary> {
    web_sys::window()
        .and_then(|window| window.local_storage().ok().flatten())
        .and_then(|storage| storage.get_item(PULSE_SUMMARY_STORAGE_KEY).ok().flatten())
        .and_then(|json| serde_json::from_str(&json).ok())
}

fn mock_mail_inbox() -> InboxSummary {
    InboxSummary {
        unread_count: 12,
        total_count: 247,
        urgent_count: 1,
        high_priority_count: 4,
        queued_actions: 0,
        recent_threads: vec![
            ThreadSummary {
                id: "t-001".into(),
                subject: "Re: Council meeting agenda for Thursday".into(),
                from_name: "Elena".into(),
                from_email: "elena@mycelix.local".into(),
                preview: "I've added the water allocation item to the agenda.".into(),
                trust_tier: TrustTier::High,
                timestamp: 1_711_954_920,
                is_read: false,
            },
            ThreadSummary {
                id: "t-002".into(),
                subject: "FL round 47 results - 94.2% accuracy".into(),
                from_name: "Coordinator".into(),
                from_email: "fl@mycelix.local".into(),
                preview: "Your gradient contribution improved the model by 0.3%.".into(),
                trust_tier: TrustTier::High,
                timestamp: 1_711_949_700,
                is_read: false,
            },
            ThreadSummary {
                id: "t-003".into(),
                subject: "Solar garden construction timeline".into(),
                from_name: "Block 7 Committee".into(),
                from_email: "block7@mycelix.local".into(),
                preview: "Foundation work starts next Monday.".into(),
                trust_tier: TrustTier::Medium,
                timestamp: 1_711_868_400,
                is_read: true,
            },
        ],
        trust_health: TrustHealth {
            trusted_contacts: 89,
            quarantined: 3,
            introductions_pending: 2,
            average_trust_score: 0.78,
        },
        source: domain_mail::types::InboxSummarySource::Mock,
        updated_at: Some(seconds_to_micros(1_711_954_920)),
    }
}

fn seconds_to_micros(timestamp_secs: i64) -> i64 {
    timestamp_secs.saturating_mul(1_000_000)
}

pub(crate) async fn hydrate_domain_summary(
    domain_id: &str,
    identity: SensoriumIdentity,
    fallback: DomainSummaryCard,
) -> Result<DomainSummaryCard, String> {
    match domain_id {
        "personal" => build_live_personal_summary(identity, fallback).await,
        "governance" => build_live_governance_summary(identity, fallback).await,
        "commons" => build_live_commons_summary(identity, fallback).await,
        "mail" => build_live_mail_summary(fallback).await,
        _ => Ok(fallback),
    }
}

async fn build_live_mail_summary(fallback: DomainSummaryCard) -> Result<DomainSummaryCard, String> {
    let (inbox, mode) = match fetch_mail_inbox_summary().await {
        Ok(inbox) => (inbox, MailSummaryMode::Api),
        Err(_) => match load_local_mail_inbox_summary() {
            Some(inbox) => (inbox, MailSummaryMode::LocalBridge),
            None => (mock_mail_inbox(), MailSummaryMode::Mock),
        },
    };

    let availability = if inbox.total_count == 0 && inbox.recent_threads.is_empty() {
        DomainAvailability::Empty
    } else if mode == MailSummaryMode::Mock {
        DomainAvailability::Mock
    } else if mode == MailSummaryMode::LocalBridge {
        DomainAvailability::Degraded
    } else {
        DomainAvailability::Live
    };

    let unread = inbox.unread_count;
    let urgent = inbox.urgent_count;
    let queued = inbox.queued_actions;
    let quarantined = inbox.trust_health.quarantined;
    let pending = inbox.trust_health.introductions_pending;
    let trust_pct = (inbox.trust_health.average_trust_score * 100.0).round() as u32;

    let mut attention = Vec::new();
    if unread > 0 {
        attention.push(DomainAttentionItem {
            id: "mail-unread".into(),
            label: "Unread triage active".into(),
            detail: format!(
                "{unread} unread thread{} currently need review.",
                if unread == 1 { "" } else { "s" }
            ),
            level: if unread >= 10 {
                AttentionLevel::ActionNeeded
            } else {
                AttentionLevel::Notice
            },
            path: Some("/mail/inbox".into()),
        });
    }
    if urgent > 0 {
        attention.push(DomainAttentionItem {
            id: "mail-urgent".into(),
            label: "Urgent Pulse threads".into(),
            detail: format!(
                "{urgent} urgent thread{} should be handled before routine triage.",
                if urgent == 1 { "" } else { "s" }
            ),
            level: AttentionLevel::ActionNeeded,
            path: Some("/mail/inbox".into()),
        });
    }
    if queued > 0 {
        attention.push(DomainAttentionItem {
            id: "mail-queued".into(),
            label: "Offline Pulse queue".into(),
            detail: format!(
                "{queued} queued action{} waiting for conductor/network recovery.",
                if queued == 1 { "" } else { "s" }
            ),
            level: AttentionLevel::Notice,
            path: Some("/mail/inbox".into()),
        });
    }
    if quarantined > 0 {
        attention.push(DomainAttentionItem {
            id: "mail-quarantine".into(),
            label: "Quarantine queue active".into(),
            detail: format!(
                "{quarantined} sender{} or thread{} are being held by trust filtering.",
                if quarantined == 1 { "" } else { "s" },
                if quarantined == 1 { " is" } else { "s are" }
            ),
            level: AttentionLevel::Notice,
            path: Some("/mail/trust".into()),
        });
    }
    if pending > 0 {
        attention.push(DomainAttentionItem {
            id: "mail-introductions".into(),
            label: "Introductions pending".into(),
            detail: format!(
                "{pending} trust introduction{} awaiting response.",
                if pending == 1 { "" } else { "s" }
            ),
            level: AttentionLevel::Notice,
            path: Some("/mail/trust".into()),
        });
    }

    let updated_at = inbox
        .updated_at
        .or_else(|| {
            inbox
                .recent_threads
                .iter()
                .map(|thread| thread.timestamp)
                .max()
                .map(seconds_to_micros)
        })
        .or(Some((js_sys::Date::now() as i64) * 1000));

    Ok(DomainSummaryCard {
        domain_id: "mail",
        title: "Signal Stream".into(),
        availability,
        status_line: if mode == MailSummaryMode::Mock {
            "Pulse summary is currently using mock inbox state because the live mail backend is unavailable."
                .into()
        } else if mode == MailSummaryMode::LocalBridge {
            format!(
                "Pulse REST is unavailable, but Sensorium is reading the local Pulse bridge: {unread} unread thread{} across {} total conversations.",
                if unread == 1 { "" } else { "s" },
                inbox.total_count
            )
        } else if unread > 0 {
            format!(
                "Live Pulse posture shows {unread} unread thread{} across {} total conversations.",
                if unread == 1 { "" } else { "s" },
                inbox.total_count
            )
        } else {
            format!(
                "Live Pulse posture is quiet right now across {} tracked conversation{}.",
                inbox.total_count,
                if inbox.total_count == 1 { "" } else { "s" }
            )
        },
        metrics: vec![
            DomainMetric {
                id: "unread",
                label: "Unread".into(),
                value: unread.to_string(),
                hint: Some(format!("{} total", inbox.total_count)),
                tone: if unread > 0 { Some("notice") } else { None },
            },
            DomainMetric {
                id: "urgent",
                label: "Urgent".into(),
                value: urgent.to_string(),
                hint: Some(format!("{} high priority", inbox.high_priority_count)),
                tone: if urgent > 0 { Some("warning") } else { None },
            },
            DomainMetric {
                id: "queued",
                label: "Queued".into(),
                value: queued.to_string(),
                hint: Some("offline sync".into()),
                tone: if queued > 0 { Some("notice") } else { None },
            },
            DomainMetric {
                id: "quarantine",
                label: "Quarantined".into(),
                value: quarantined.to_string(),
                hint: Some("trust-held".into()),
                tone: if quarantined > 0 {
                    Some("warning")
                } else {
                    None
                },
            },
            DomainMetric {
                id: "pending",
                label: "Introductions".into(),
                value: pending.to_string(),
                hint: Some("awaiting response".into()),
                tone: if pending > 0 { Some("notice") } else { None },
            },
            DomainMetric {
                id: "trust",
                label: "Trust Health".into(),
                value: format!("{trust_pct}%"),
                hint: Some(format!(
                    "{} trusted contacts",
                    inbox.trust_health.trusted_contacts
                )),
                tone: None,
            },
        ],
        attention: if attention.is_empty() {
            fallback.attention
        } else {
            attention
        },
        primary_launch: fallback.primary_launch,
        secondary_launches: fallback.secondary_launches,
        updated_at,
    })
}

async fn build_live_personal_summary(
    identity: SensoriumIdentity,
    fallback: DomainSummaryCard,
) -> Result<DomainSummaryCard, String> {
    let profile = identity
        .call_zome::<(), Option<ProfileView>>(
            "personal",
            "identity_vault",
            "get_my_profile_view",
            &(),
        )
        .await
        .unwrap_or(None);
    let credentials = identity
        .call_zome::<(), Vec<StoredCredentialView>>(
            "personal",
            "credential_wallet",
            "get_my_credentials_view",
            &(),
        )
        .await
        .unwrap_or_default();
    let consents = identity
        .call_zome::<(), Vec<ConsentGrantView>>(
            "personal",
            "health_vault",
            "get_my_consents_view",
            &(),
        )
        .await
        .unwrap_or_default();
    let records = identity
        .call_zome::<(), Vec<HealthRecordView>>(
            "personal",
            "health_vault",
            "get_my_records_view",
            &(),
        )
        .await
        .unwrap_or_default();
    let activity = identity
        .call_zome::<(), Vec<ActivityItemView>>(
            "personal",
            "personal_bridge",
            "get_recent_activity_view",
            &(),
        )
        .await
        .unwrap_or_default();

    let active_consents = consents.iter().filter(|c| c.active).count();
    let revoked_credentials = credentials.iter().filter(|c| c.revoked).count();
    let recent_disclosures = activity
        .iter()
        .filter(|item| item.kind.contains("query") || item.kind.contains("event"))
        .count();
    let display_name = profile
        .as_ref()
        .map(|p| p.display_name.clone())
        .filter(|name| !name.trim().is_empty());

    let availability = if credentials.is_empty()
        && consents.is_empty()
        && records.is_empty()
        && activity.is_empty()
        && profile.is_none()
    {
        DomainAvailability::Empty
    } else {
        DomainAvailability::Live
    };

    let mut attention = Vec::new();
    if display_name.is_none() {
        attention.push(DomainAttentionItem {
            id: "profile-incomplete".into(),
            label: "Profile incomplete".into(),
            detail:
                "Personal is live, but the profile still needs a display name or identity posture review."
                    .into(),
            level: AttentionLevel::Notice,
            path: Some("/identity".into()),
        });
    }
    if recent_disclosures > 0 {
        attention.push(DomainAttentionItem {
            id: "recent-disclosure".into(),
            label: "Recent disclosure activity".into(),
            detail: format!(
                "{recent_disclosures} recent bridge events or disclosures were observed."
            ),
            level: AttentionLevel::Notice,
            path: Some("/activity".into()),
        });
    }

    Ok(DomainSummaryCard {
        domain_id: "personal",
        title: "Sovereign Vault".into(),
        availability,
        status_line: if let Some(name) = display_name {
            format!(
                "Live vault posture for {name}. Identity, wallet, consent, and disclosure state are being read from the conductor."
            )
        } else {
            "Live vault posture is available from the conductor, but profile posture still looks incomplete."
                .into()
        },
        metrics: vec![
            DomainMetric {
                id: "credentials",
                label: "Credentials".into(),
                value: credentials.len().to_string(),
                hint: Some(format!("{revoked_credentials} revoked")),
                tone: None,
            },
            DomainMetric {
                id: "consents",
                label: "Active Consents".into(),
                value: active_consents.to_string(),
                hint: Some(format!("{} total", consents.len())),
                tone: if active_consents > 0 {
                    Some("notice")
                } else {
                    None
                },
            },
            DomainMetric {
                id: "health",
                label: "Health Records".into(),
                value: records.len().to_string(),
                hint: Some("private summary".into()),
                tone: None,
            },
            DomainMetric {
                id: "activity",
                label: "Recent Disclosures".into(),
                value: recent_disclosures.to_string(),
                hint: Some(format!("{} events shown", activity.len())),
                tone: if recent_disclosures > 0 {
                    Some("notice")
                } else {
                    None
                },
            },
        ],
        attention: if attention.is_empty() {
            fallback.attention
        } else {
            attention
        },
        primary_launch: fallback.primary_launch,
        secondary_launches: fallback.secondary_launches,
        updated_at: Some((js_sys::Date::now() as i64) * 1000),
    })
}

async fn build_live_governance_summary(
    identity: SensoriumIdentity,
    fallback: DomainSummaryCard,
) -> Result<DomainSummaryCard, String> {
    let proposals_raw = call_zome_optional::<(), Vec<Value>>(
        &identity,
        "governance",
        "proposals",
        "get_active_proposals",
        &(),
    )
    .await;
    let councils_raw = call_zome_optional::<(), Vec<Value>>(
        &identity,
        "governance",
        "councils",
        "get_all_councils",
        &(),
    )
    .await;
    let timelocks_raw = call_zome_optional::<(), Vec<Value>>(
        &identity,
        "governance",
        "execution",
        "get_pending_timelocks",
        &(),
    )
    .await;
    let charter_raw = call_zome_optional::<(), Option<Value>>(
        &identity,
        "governance",
        "constitution",
        "get_current_charter",
        &(),
    )
    .await;
    let thresholds = call_zome_optional::<(), ConsciousnessThresholds>(
        &identity,
        "governance",
        "bridge",
        "get_consciousness_thresholds",
        &(),
    )
    .await;

    let any_live = proposals_raw.is_some()
        || councils_raw.is_some()
        || timelocks_raw.is_some()
        || charter_raw.is_some()
        || thresholds.is_some();
    if !any_live {
        return Err("governance live summary unavailable".into());
    }

    let proposals = proposals_raw
        .map(decode_record_entries::<ProposalView>)
        .unwrap_or_default();
    let councils = councils_raw
        .map(decode_record_entries::<CouncilView>)
        .unwrap_or_default();
    let timelocks = timelocks_raw
        .map(decode_record_entries::<TimelockView>)
        .unwrap_or_default();
    let charter = charter_raw
        .and_then(|record| record.and_then(|value| decode_record_entry::<CharterView>(&value)));

    let emergency_proposals = proposals
        .iter()
        .filter(|proposal| proposal.proposal_type == ProposalType::Emergency)
        .count();
    let draft_proposals = proposals
        .iter()
        .filter(|proposal| proposal.status == ProposalStatus::Draft)
        .count();
    let active_proposals = proposals
        .iter()
        .filter(|proposal| proposal.status.is_active())
        .count();
    let pending_timelocks = timelocks.len();

    let mut attention = Vec::new();
    if active_proposals > 0 {
        attention.push(DomainAttentionItem {
            id: "governance-active".into(),
            label: "Participation window open".into(),
            detail: format!(
                "{active_proposals} active proposal{} currently need civic attention.",
                if active_proposals == 1 { "" } else { "s" }
            ),
            level: AttentionLevel::Notice,
            path: Some("/voting".into()),
        });
    }
    if pending_timelocks > 0 {
        attention.push(DomainAttentionItem {
            id: "governance-timelock".into(),
            label: "Execution queue pending".into(),
            detail: format!(
                "{pending_timelocks} proposal{} cleared voting and {} awaiting release.",
                if pending_timelocks == 1 { "" } else { "s" },
                if pending_timelocks == 1 { "is" } else { "are" }
            ),
            level: AttentionLevel::ActionNeeded,
            path: Some("/budgeting".into()),
        });
    }

    Ok(DomainSummaryCard {
        domain_id: "governance",
        title: "Consensus State".into(),
        availability: if proposals.is_empty()
            && councils.is_empty()
            && timelocks.is_empty()
            && charter.is_none()
            && thresholds.is_none()
        {
            DomainAvailability::Empty
        } else {
            DomainAvailability::Live
        },
        status_line: if let Some(charter) = &charter {
            format!(
                "Live governance posture is available from the conductor. Proposal flow, council topology, and charter version {} are reflected here.",
                charter.version
            )
        } else {
            "Live governance posture is available from the conductor, but charter state is not yet being returned.".into()
        },
        metrics: vec![
            DomainMetric {
                id: "proposals",
                label: "Active Proposals".into(),
                value: active_proposals.to_string(),
                hint: Some(format!("{draft_proposals} drafts")),
                tone: if active_proposals > 0 {
                    Some("notice")
                } else {
                    None
                },
            },
            DomainMetric {
                id: "councils",
                label: "Councils".into(),
                value: councils.len().to_string(),
                hint: Some("registered".into()),
                tone: None,
            },
            DomainMetric {
                id: "timelocks",
                label: "Pending Timelocks".into(),
                value: pending_timelocks.to_string(),
                hint: Some(format!("{emergency_proposals} emergency")),
                tone: if pending_timelocks > 0 {
                    Some("notice")
                } else {
                    None
                },
            },
            DomainMetric {
                id: "threshold",
                label: "Voting Threshold".into(),
                value: thresholds
                    .as_ref()
                    .map(|t| format!("{:.0}%", t.voting * 100.0))
                    .unwrap_or_else(|| "n/a".into()),
                hint: thresholds
                    .as_ref()
                    .map(|t| format!("constitutional {:.0}%", t.constitutional * 100.0)),
                tone: None,
            },
        ],
        attention: if attention.is_empty() {
            fallback.attention
        } else {
            attention
        },
        primary_launch: fallback.primary_launch,
        secondary_launches: fallback.secondary_launches,
        updated_at: Some((js_sys::Date::now() as i64) * 1000),
    })
}

#[derive(Clone, Serialize)]
struct CommonsSearchNeedsInput {
    category: Option<Value>,
    urgency: Option<Value>,
    emergency_only: bool,
    query: Option<String>,
    limit: Option<u32>,
}

#[derive(Clone, Serialize)]
struct CommonsSearchOffersInput {
    category: Option<Value>,
    query: Option<String>,
    limit: Option<u32>,
}

#[derive(Clone, Serialize)]
struct CommonsResourceStatusInput {
    resource_type: String,
    location: Option<String>,
    limit: u32,
}

async fn build_live_commons_summary(
    identity: SensoriumIdentity,
    fallback: DomainSummaryCard,
) -> Result<DomainSummaryCard, String> {
    let emergency_needs = call_zome_optional::<(), Vec<Value>>(
        &identity,
        "commons",
        "mutualaid-needs",
        "get_emergency_needs",
        &(),
    )
    .await;
    let open_needs: Option<Vec<Value>> = call_zome_optional(
        &identity,
        "commons",
        "mutualaid-needs",
        "search_needs",
        &CommonsSearchNeedsInput {
            category: None,
            urgency: None,
            emergency_only: false,
            query: None,
            limit: Some(50),
        },
    )
    .await;
    let offers: Option<Vec<Value>> = call_zome_optional(
        &identity,
        "commons",
        "mutualaid-needs",
        "search_offers",
        &CommonsSearchOffersInput {
            category: None,
            query: None,
            limit: Some(50),
        },
    )
    .await;
    let circles = call_zome_optional::<(), Vec<Value>>(
        &identity,
        "commons",
        "care-circles",
        "get_all_circles",
        &(),
    )
    .await;
    let events = call_zome_optional::<u32, Vec<Value>>(
        &identity,
        "commons",
        "community-calendar",
        "get_upcoming_events",
        &5,
    )
    .await;
    let water_readings: Option<Vec<Value>> = call_zome_optional(
        &identity,
        "commons",
        "resource-mesh",
        "get_resource_status",
        &CommonsResourceStatusInput {
            resource_type: "water".into(),
            location: None,
            limit: 10,
        },
    )
    .await;

    let any_live = emergency_needs.is_some()
        || open_needs.is_some()
        || offers.is_some()
        || circles.is_some()
        || events.is_some()
        || water_readings.is_some();
    if !any_live {
        return Err("commons live summary unavailable".into());
    }

    let emergency_need_count = emergency_needs.as_ref().map_or(0, Vec::len);
    let open_need_count = open_needs.as_ref().map_or(0, Vec::len);
    let offer_count = offers.as_ref().map_or(0, Vec::len);
    let circle_count = circles.as_ref().map_or(0, Vec::len);
    let event_count = events.as_ref().map_or(0, Vec::len);
    let water_signal_count = water_readings.as_ref().map_or(0, Vec::len);

    let mut attention = Vec::new();
    if emergency_need_count > 0 {
        attention.push(DomainAttentionItem {
            id: "commons-emergency-needs".into(),
            label: "Emergency mutual aid needs".into(),
            detail: format!(
                "{emergency_need_count} emergency need{} currently require stewardship response.",
                if emergency_need_count == 1 { "" } else { "s" }
            ),
            level: AttentionLevel::ActionNeeded,
            path: Some("/resources".into()),
        });
    }
    if water_signal_count == 0 {
        attention.push(DomainAttentionItem {
            id: "commons-water-telemetry".into(),
            label: "Water telemetry quiet".into(),
            detail: "No recent live resource-mesh water readings were returned for the current commons membrane.".into(),
            level: AttentionLevel::Notice,
            path: Some("/resources".into()),
        });
    }

    Ok(DomainSummaryCard {
        domain_id: "commons",
        title: "Commons Stewardship".into(),
        availability: if emergency_need_count == 0
            && open_need_count == 0
            && offer_count == 0
            && circle_count == 0
            && event_count == 0
            && water_signal_count == 0
        {
            DomainAvailability::Empty
        } else {
            DomainAvailability::Live
        },
        status_line: "Live commons posture is now partially backed by conductor state. Mutual-aid demand, standing offers, care circles, upcoming events, and water telemetry are reflected here while deeper housing/resource shells still need dedicated frontend contracts.".into(),
        metrics: vec![
            DomainMetric {
                id: "needs",
                label: "Open Needs".into(),
                value: open_need_count.to_string(),
                hint: Some(format!("{emergency_need_count} emergency")),
                tone: if emergency_need_count > 0 {
                    Some("notice")
                } else {
                    None
                },
            },
            DomainMetric {
                id: "offers",
                label: "Available Offers".into(),
                value: offer_count.to_string(),
                hint: Some("mutual aid".into()),
                tone: None,
            },
            DomainMetric {
                id: "circles",
                label: "Care Circles".into(),
                value: circle_count.to_string(),
                hint: Some("registered".into()),
                tone: None,
            },
            DomainMetric {
                id: "events",
                label: "Upcoming Events".into(),
                value: event_count.to_string(),
                hint: Some(format!("{water_signal_count} water signals")),
                tone: None,
            },
        ],
        attention: if attention.is_empty() {
            fallback.attention
        } else {
            attention
        },
        primary_launch: fallback.primary_launch,
        secondary_launches: fallback.secondary_launches,
        updated_at: Some((js_sys::Date::now() as i64) * 1000),
    })
}

async fn call_zome_optional<I, O>(
    identity: &SensoriumIdentity,
    role: &str,
    zome: &str,
    function: &str,
    payload: &I,
) -> Option<O>
where
    I: Serialize,
    O: DeserializeOwned,
{
    identity
        .call_zome::<I, O>(role, zome, function, payload)
        .await
        .ok()
}

fn decode_record_entries<T>(records: Vec<Value>) -> Vec<T>
where
    T: DeserializeOwned,
{
    records
        .iter()
        .filter_map(decode_record_entry::<T>)
        .collect()
}

fn decode_record_entry<T>(record: &Value) -> Option<T>
where
    T: DeserializeOwned,
{
    record
        .get("entry")
        .and_then(|entry| entry.get("Present"))
        .cloned()
        .or_else(|| Some(record.clone()))
        .and_then(|value| serde_json::from_value::<T>(value).ok())
}

fn launch_href(domain_id: &str, launch: &DomainLaunchTarget) -> Option<String> {
    match launch.kind {
        LaunchKind::Disabled => None,
        LaunchKind::ExternalApp => Some(launch.path.to_string()),
        LaunchKind::InternalRoute => {
            let base = domain_app_url(domain_id)?;
            if launch.path == "/" {
                Some(base)
            } else {
                Some(format!("{}{}", base.trim_end_matches('/'), launch.path))
            }
        }
    }
}

/// Map domain IDs to standalone app URLs.
/// Auto-detects public vs local deployment.
pub fn domain_app_url(domain_id: &str) -> Option<String> {
    let is_public = web_sys::window()
        .and_then(|w| w.location().hostname().ok())
        .map(|h| h.contains("mycelix.net") || h.contains("luminousdynamics.io"))
        .unwrap_or(false);

    if is_public {
        match domain_id {
            "governance" => Some("https://governance.mycelix.net".into()),
            "hearth" => Some("https://hearth.mycelix.net".into()),
            "praxis" => Some("https://praxis.mycelix.net".into()),
            "health" => Some("https://health.mycelix.net".into()),
            "music" => Some("https://music.mycelix.net".into()),
            "climate" => Some("https://climate.mycelix.net".into()),
            "knowledge" => Some("https://knowledge.mycelix.net".into()),
            "energy" => Some("https://energy.mycelix.net".into()),
            "finance" => Some("https://finance.mycelix.net".into()),
            "commons" => Some("https://commons.mycelix.net".into()),
            "attribution" => Some("https://attribution.mycelix.net".into()),
            "space" => Some("https://space.mycelix.net".into()),
            "supplychain" => Some("https://supplychain.mycelix.net".into()),
            "mail" => Some("https://mail.mycelix.net".into()),
            _ => None,
        }
    } else {
        match domain_id {
            "governance" => Some("http://localhost:8110".into()),
            "hearth" => Some("http://localhost:8112".into()),
            "praxis" => Some("http://localhost:8107".into()),
            "health" => Some("http://localhost:8111".into()),
            "music" => Some("http://localhost:8121".into()),
            "climate" => Some("http://localhost:8103".into()),
            "knowledge" => Some("http://localhost:8114".into()),
            "energy" => Some("http://localhost:8108".into()),
            "finance" => Some("http://localhost:8109".into()),
            "commons" => Some("http://localhost:8104".into()),
            "attribution" => Some("http://localhost:8101".into()),
            "space" => Some("http://localhost:8126".into()),
            "supplychain" => Some("http://localhost:8127".into()),
            "mail" => Some("http://localhost:8117".into()),
            _ => None,
        }
    }
}
