// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use crate::mail_context::use_mail;
use crate::holochain::{use_holochain, ConnectionStatus};
use crate::toasts::use_toasts;
use crate::components::ContactCard;

#[component]
pub fn ContactsPage() -> impl IntoView {
    let mail = use_mail();
    let hc = use_holochain();
    let toasts = use_toasts();
    let search = RwSignal::new(String::new());
    let show_favorites_only = RwSignal::new(false);
    let discover_query = RwSignal::new(String::new());
    let discover_result = RwSignal::new(Option::<DiscoverResult>::None);
    let discover_loading = RwSignal::new(false);
    let discover_adding = RwSignal::new(false);
    let is_mock = move || hc.status.get() == ConnectionStatus::Mock;

    let filtered = move || {
        let query = search.get().to_lowercase();
        let favs_only = show_favorites_only.get();
        mail.contacts.get().into_iter().filter(|c| {
            let matches_search = query.is_empty()
                || c.display_name.to_lowercase().contains(&query)
                || c.email.as_deref().unwrap_or("").to_lowercase().contains(&query)
                || c.organization.as_deref().unwrap_or("").to_lowercase().contains(&query);
            let matches_fav = !favs_only || c.is_favorite;
            matches_search && matches_fav
        }).collect::<Vec<_>>()
    };

    view! {
        <div class="page page-contacts">
            <div class="page-header">
                <h1>"Contacts"</h1>
                <div class="header-actions">
                    <span class="contact-count">{move || format!("{} contacts", mail.contacts.get().len())}</span>
                    <button class="btn btn-sm btn-secondary" on:click={
                        let mail_exp = mail.clone();
                        move |_| {
                            let contacts = mail_exp.contacts.get();
                            let mut vcf = String::new();
                            for c in &contacts {
                                vcf.push_str("BEGIN:VCARD\nVERSION:3.0\n");
                                vcf.push_str(&format!("FN:{}\n", c.display_name));
                                if let Some(ref e) = c.email { vcf.push_str(&format!("EMAIL:{}\n", e)); }
                                if let Some(ref o) = c.organization { vcf.push_str(&format!("ORG:{}\n", o)); }
                                vcf.push_str("END:VCARD\n\n");
                            }
                            let _ = js_sys::eval(&format!(
                                "const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([`{}`],{{type:'text/vcard'}}));a.download='contacts.vcf';a.click()",
                                vcf.replace('`', "\\`")
                            ));
                        }
                    }>"Export vCard"</button>
                </div>
            </div>

            <div class="contacts-toolbar">
                <input
                    type="text"
                    class="search-input"
                    placeholder="Search contacts..."
                    prop:value=move || search.get()
                    on:input=move |ev| {
                        use wasm_bindgen::JsCast;
                        let val = ev.target()
                            .and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok())
                            .map(|el| el.value())
                            .unwrap_or_default();
                        search.set(val);
                    }
                />
                <label class="filter-toggle">
                    <input
                        type="checkbox"
                        prop:checked=move || show_favorites_only.get()
                        on:change=move |_| show_favorites_only.update(|v| *v = !*v)
                    />
                    " Favorites only"
                </label>
            </div>

            // DSID Discovery Section
            <div class="setting-card" style="margin-bottom:16px">
                <div class="setting-card-label">"Discover on Network"</div>
                <div class="setting-card-body">
                    <form class="discover-search" on:submit={
                        let hc = hc.clone();
                        move |e: web_sys::SubmitEvent| {
                            e.prevent_default();
                            let query = discover_query.get_untracked();
                            if query.trim().is_empty() || hc.status.get_untracked() == ConnectionStatus::Mock { return; }
                            discover_loading.set(true);
                            discover_result.set(None);

                            let hc = hc.clone();
                            let wire_query = crate::dsid::wire_dsid(query.trim());
                            spawn_local(async move {
                                // Try resolve_did on identity DNA
                                match hc.call_zome_on_role::<serde_json::Value, serde_json::Value>(
                                    "identity", "did_registry", "resolve_did", &serde_json::json!(wire_query)
                                ).await {
                                    Ok(val) if !val.is_null() => {
                                        let agent_pub_key_value = hc.call_zome::<serde_json::Value, Option<serde_json::Value>>(
                                            "mail_bridge", "resolve_identity", &serde_json::json!(wire_query)
                                        ).await.ok().flatten();
                                        let agent_pub_key = agent_pub_key_value
                                            .as_ref()
                                            .map(crate::zome_adapter::json_hash_to_string);
                                        let name = val.get("controller")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("Unknown")
                                            .to_string();
                                        let dsid = val.get("id")
                                            .and_then(|v| v.as_str())
                                            .map(|s| crate::dsid::display_dsid(s))
                                            .unwrap_or_default();
                                        discover_result.set(Some(DiscoverResult {
                                            dsid,
                                            display_name: name,
                                            found: true,
                                            agent_pub_key,
                                            agent_pub_key_value,
                                        }));
                                    }
                                    Ok(_) => {
                                        discover_result.set(Some(DiscoverResult {
                                            dsid: crate::dsid::display_dsid(&wire_query),
                                            display_name: String::new(),
                                            found: false,
                                            agent_pub_key: None,
                                            agent_pub_key_value: None,
                                        }));
                                    }
                                    Err(e) => {
                                        web_sys::console::warn_1(&format!("[Mail] DSID lookup: {e}").into());
                                        discover_result.set(Some(DiscoverResult {
                                            dsid: query.clone(),
                                            display_name: format!("Error: {e}"),
                                            found: false,
                                            agent_pub_key: None,
                                            agent_pub_key_value: None,
                                        }));
                                    }
                                }
                                discover_loading.set(false);
                            });
                        }
                    }>
                        <input type="text"
                            placeholder="Enter DSID or agent key..."
                            prop:value=move || discover_query.get()
                            on:input=move |e| discover_query.set(event_target_value(&e))
                        />
                        <input type="submit" class="btn btn-primary" value="Search"
                            disabled=move || discover_loading.get() || discover_query.get().trim().is_empty()
                            style="padding:8px 16px;font-size:0.85rem"
                        />
                    </form>
                    <p style=move || if discover_loading.get() { "color:var(--teal);font-size:0.8rem" } else { "display:none" }>
                        "Searching the DHT..."
                    </p>
                    {move || discover_result.get().map(|r| {
                        if r.found {
                            let result = r.clone();
                            let dsid_display = result.dsid.clone();
                            let name = result.display_name.clone();
                            let already_added = mail.contacts.get().iter().any(|contact| contact_matches_discovery(contact, &result));
                            let toasts_add = toasts.clone();
                            let hc_add = hc.clone();
                            let mail_add = mail.clone();
                            let discover_result_set = discover_result;
                            let discover_query_set = discover_query;
                            view! {
                                <div class="discover-result">
                                    <div style="font-size:1.5rem">"\u{1F464}"</div>
                                    <div class="discover-result-info">
                                        <p style="font-weight:600;margin:0;font-size:0.9rem">{name}</p>
                                        <p class="discover-result-dsid">{dsid_display}</p>
                                    </div>
                                    <button class="btn btn-primary btn-sm"
                                        disabled=move || discover_adding.get() || already_added
                                        on:click=move |_| {
                                            if already_added {
                                                toasts_add.push("Contact already exists", "info");
                                                return;
                                            }

                                            discover_adding.set(true);
                                            let hc = hc_add.clone();
                                            let mail = mail_add.clone();
                                            let toasts = toasts_add.clone();
                                            let result = result.clone();

                                            spawn_local(async move {
                                                let now = (js_sys::Date::now() as u64).saturating_mul(1000);
                                                let payload = build_discovered_contact_payload(&result, now);

                                                match hc.call_zome::<serde_json::Value, serde_json::Value>(
                                                    "mail_contacts", "create_contact", &payload
                                                ).await {
                                                    Ok(_) => {
                                                        mail.versions.contacts.update(|v| *v += 1);
                                                        discover_result_set.set(None);
                                                        discover_query_set.set(String::new());
                                                        toasts.push("Contact added", "success");
                                                    }
                                                    Err(e) => {
                                                        toasts.push(format!("Add contact failed: {e}"), "error");
                                                    }
                                                }

                                                discover_adding.set(false);
                                            });
                                        }>
                                        {if already_added { "Added" } else { "Add" }}
                                    </button>
                                </div>
                            }.into_any()
                        } else {
                            view! {
                                <p style="color:var(--text-muted);font-size:0.8rem">
                                    "No DSID found for this identifier."
                                </p>
                            }.into_any()
                        }
                    })}
                    {move || if is_mock() {
                        Some(view! {
                            <p style="font-size:0.75rem;color:var(--text-muted);margin-top:8px">
                                "Discovery requires a live conductor connection."
                            </p>
                        })
                    } else { None }}
                </div>
            </div>

            <div class="contact-list" role="list">
                {move || {
                    let contacts = filtered();
                    if contacts.is_empty() {
                        view! {
                            <div class="empty-state">
                                <svg viewBox="0 0 100 80" width="100" height="80" class="empty-svg" xmlns="http://www.w3.org/2000/svg">
                                    <circle cx="35" cy="30" r="12" fill="none" stroke="#06D6C8" stroke-width="2" opacity="0.3" />
                                    <path d="M20 65 Q35 50 50 65" fill="none" stroke="#06D6C8" stroke-width="2" opacity="0.3" />
                                    <circle cx="65" cy="30" r="12" fill="none" stroke="#8b7ec8" stroke-width="2" opacity="0.3" />
                                    <path d="M50 65 Q65 50 80 65" fill="none" stroke="#8b7ec8" stroke-width="2" opacity="0.3" />
                                    <line x1="48" y1="30" x2="52" y2="30" stroke="#06D6C8" stroke-width="1" opacity="0.2" />
                                </svg>
                                <p class="empty-title">"No contacts found"</p>
                            </div>
                        }.into_any()
                    } else {
                        view! {
                            <div>
                                {contacts.into_iter().map(|c| {
                                    view! { <ContactCard contact=c /> }
                                }).collect::<Vec<_>>()}
                            </div>
                        }.into_any()
                    }
                }}
            </div>
        </div>
    }
}

#[derive(Clone, Debug)]
struct DiscoverResult {
    dsid: String,
    display_name: String,
    found: bool,
    agent_pub_key: Option<String>,
    agent_pub_key_value: Option<serde_json::Value>,
}

fn contact_id_from_discovery(result: &DiscoverResult) -> String {
    let seed = if !result.dsid.trim().is_empty() {
        result.dsid.trim()
    } else if !result.display_name.trim().is_empty() {
        result.display_name.trim()
    } else {
        "contact"
    };

    seed.chars()
        .map(|ch| match ch {
            'a'..='z' | '0'..='9' => ch,
            'A'..='Z' => ch.to_ascii_lowercase(),
            _ => '-',
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn contact_matches_discovery(contact: &mail_leptos_types::ContactView, result: &DiscoverResult) -> bool {
    if let Some(agent_pub_key) = result.agent_pub_key.as_deref() {
        if contact.agent_pub_key.as_deref() == Some(agent_pub_key) {
            return true;
        }
    }

    contact.id == contact_id_from_discovery(result)
}

fn build_discovered_contact_payload(result: &DiscoverResult, now_micros: u64) -> serde_json::Value {
    let display_name = if result.display_name.trim().is_empty() {
        result.dsid.clone()
    } else {
        result.display_name.clone()
    };

    serde_json::json!({
        "id": contact_id_from_discovery(result),
        "display_name": display_name,
        "nickname": serde_json::Value::Null,
        "emails": [],
        "phones": [],
        "addresses": [],
        "organization": serde_json::Value::Null,
        "title": serde_json::Value::Null,
        "notes": format!("Added from DSID discovery: {}", result.dsid),
        "avatar": serde_json::Value::Null,
        "groups": [],
        "labels": ["dsid-discovery"],
        "agent_pub_key": result.agent_pub_key_value.clone().unwrap_or(serde_json::Value::Null),
        "is_favorite": false,
        "is_blocked": false,
        "metadata": {
            "source": "dsid_discovery",
            "last_email_sent": serde_json::Value::Null,
            "last_email_received": serde_json::Value::Null,
            "email_count": 0,
            "response_rate": serde_json::Value::Null,
            "average_response_time": serde_json::Value::Null
        },
        "created_at": now_micros,
        "updated_at": now_micros,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result() -> DiscoverResult {
        DiscoverResult {
            dsid: "dsid:mycelix:alice".into(),
            display_name: "Alice Example".into(),
            found: true,
            agent_pub_key: Some("uhCAk_alice".into()),
            agent_pub_key_value: Some(serde_json::json!("uhCAk_alice")),
        }
    }

    #[test]
    fn contact_id_is_slugged_from_dsid() {
        let result = sample_result();
        assert_eq!(contact_id_from_discovery(&result), "dsid-mycelix-alice");
    }

    #[test]
    fn contact_match_prefers_agent_key() {
        let result = sample_result();
        let contact = mail_leptos_types::ContactView {
            hash: "hash".into(),
            id: "some-other-id".into(),
            display_name: "Someone Else".into(),
            nickname: None,
            email: None,
            agent_pub_key: Some("uhCAk_alice".into()),
            organization: None,
            avatar: None,
            groups: vec![],
            is_favorite: false,
            is_blocked: false,
            email_count: 0,
            trust_score: None,
        };
        assert!(contact_matches_discovery(&contact, &result));
    }

    #[test]
    fn build_discovery_payload_contains_expected_identity_fields() {
        let result = sample_result();
        let payload = build_discovered_contact_payload(&result, 1234);

        assert_eq!(payload.get("id").and_then(|v| v.as_str()), Some("dsid-mycelix-alice"));
        assert_eq!(payload.get("display_name").and_then(|v| v.as_str()), Some("Alice Example"));
        assert_eq!(payload.get("notes").and_then(|v| v.as_str()), Some("Added from DSID discovery: dsid:mycelix:alice"));
        assert_eq!(payload.get("agent_pub_key").and_then(|v| v.as_str()), Some("uhCAk_alice"));
        assert_eq!(payload.get("created_at").and_then(|v| v.as_u64()), Some(1234));
    }
}
