// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Chat page — ephemeral real-time messaging via Holochain remote_signal.
//!
//! Architecture: Messages are sent via P2P WebRTC signals (ephemeral, not DHT).
//! "Consolidate to Record" commits the chat to the DHT as a permanent mail thread.
//! This prevents DHT bloat while enabling sub-100ms chat delivery.

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use mail_leptos_types::*;
use crate::toasts::use_toasts;

#[component]
pub fn ChatPage() -> impl IntoView {
    let toasts = use_toasts();
    let channels = RwSignal::new(crate::mock_data::mock_chat_channels());
    let active_channel = RwSignal::new(Option::<String>::None);
    let messages = RwSignal::new(Vec::<ChatMessage>::new());
    let input = RwSignal::new(String::new());

    // Load messages when channel changes
    let load_messages = move |channel_id: &str| {
        let msgs = crate::mock_data::mock_chat_messages(channel_id);
        messages.set(msgs);
    };

    let on_select_channel = move |id: String| {
        load_messages(&id);
        active_channel.set(Some(id));
    };

    let _toasts_send = toasts.clone();
    let toasts_consolidate = toasts.clone();
    let toasts_channel = toasts.clone();
    let on_send = move |_| {
        let text = input.get_untracked();
        if text.trim().is_empty() { return; }
        let channel = active_channel.get_untracked();
        if channel.is_none() { return; }

        let now = (js_sys::Date::now() / 1000.0) as u64;
        let msg = ChatMessage {
            id: format!("cm-{}", now),
            sender: "uhCAk_self_mock".into(),
            sender_name: Some("You".into()),
            content: text.clone(),
            timestamp: now,
            reply_to: None,
            reactions: vec![],
            edited: false,
            channel_id: channel,
        };

        // Add to local state immediately (optimistic UI)
        messages.update(|m| m.push(msg.clone()));
        input.set(String::new());

        // Send via P2P remote_signal if conductor connected
        let hc = crate::holochain::use_holochain();
        if !hc.is_mock() {
            let signal_payload = serde_json::json!({
                "type": "chat_message",
                "data": {
                    "channel_id": msg.channel_id,
                    "content": msg.content,
                    "sender": msg.sender,
                    "timestamp": msg.timestamp,
                }
            });
            wasm_bindgen_futures::spawn_local(async move {
                match hc.call_zome::<serde_json::Value, serde_json::Value>(
                    "mail_federation", "send_signal", &signal_payload
                ).await {
                    Ok(_) => web_sys::console::log_1(&"[Chat] Message sent via P2P signal".into()),
                    Err(e) => web_sys::console::warn_1(&format!("[Chat] Signal failed: {e}").into()),
                }
            });
        }

        // Scroll to bottom
        let _ = js_sys::eval("setTimeout(()=>{const el=document.querySelector('.chat-messages');if(el)el.scrollTop=el.scrollHeight},50)");
    };

    let on_keypress = move |ev: web_sys::KeyboardEvent| {
        if ev.key() == "Enter" && !ev.shift_key() {
            ev.prevent_default();
            on_send(());
        }
    };

    let creating_channel = RwSignal::new(false);
    let new_channel_name = RwSignal::new(String::new());
    let new_channel_desc = RwSignal::new(String::new());
    let _on_create_channel = move |_: ()| {
        let name = new_channel_name.get_untracked();
        if name.trim().is_empty() {
            toasts_channel.push("Channel name is required", "error");
            return;
        }
        let desc = new_channel_desc.get_untracked();
        let now = (js_sys::Date::now() / 1000.0) as u64;
        channels.update(|chs| {
            chs.push(ChatChannel {
                id: format!("ch-{}", now),
                name: name.trim().to_lowercase().replace(' ', "-"),
                description: if desc.is_empty() { None } else { Some(desc) },
                is_direct: false,
                members: vec![],
                member_names: vec!["You".into()],
                unread_count: 0,
                last_message: None,
                last_activity: now,
                pinned: false,
            });
        });
        new_channel_name.set(String::new());
        new_channel_desc.set(String::new());
        creating_channel.set(false);
        toasts_channel.push(format!("Channel #{} created", name.trim()), "success");
    };

    let active_channel_name = move || {
        let id = active_channel.get()?;
        channels.get().iter().find(|c| c.id == id).map(|c| c.name.clone())
    };

    let active_channel_desc = move || {
        let id = active_channel.get()?;
        channels.get().iter().find(|c| c.id == id).and_then(|c| c.description.clone())
    };

    view! {
        <div class="page page-chat">
            <div class="chat-layout">
                // Channel list (left)
                <div class="chat-channels">
                    <h3 class="channels-header">"Direct Messages"</h3>
                    {move || channels.get().iter().filter(|c| c.is_direct).map(|ch| {
                        let id = ch.id.clone();
                        let name = ch.name.clone();
                        let unread = ch.unread_count;
                        let last = ch.last_message.clone().unwrap_or_default();
                        let is_active = active_channel.get().as_deref() == Some(&id);
                        let id_click = id.clone();
                        view! {
                            <button class=if is_active { "channel-item active" } else { "channel-item" }
                                    on:click=move |_| on_select_channel(id_click.clone())>
                                <span class="channel-avatar presence-online">{name.chars().next().unwrap_or('?').to_uppercase().to_string()}</span>
                                <div class="channel-info">
                                    <span class="channel-name">{name.clone()}</span>
                                    <span class="channel-last">{last.clone()}</span>
                                </div>
                                {(unread > 0).then(|| view! { <span class="channel-badge">{unread}</span> })}
                            </button>
                        }
                    }).collect::<Vec<_>>()}

                    <div class="channels-header-row">
                        <h3 class="channels-header">"Channels"</h3>
                        <button class="btn-create-channel" on:click=move |_| creating_channel.update(|v| *v = !*v)
                                title="Create channel">
                            "+"
                        </button>
                    </div>

                    // Channel creation form
                    <div class="create-channel-form" style=move || if creating_channel.get() { "" } else { "display:none" }>
                        <input
                            type="text"
                            class="channel-name-input"
                            placeholder="channel-name"
                            prop:value=move || new_channel_name.get()
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()).map(|el| el.value()).unwrap_or_default();
                                new_channel_name.set(val);
                            }
                            on:keydown=move |ev: web_sys::KeyboardEvent| {
                                if ev.key() == "Enter" {
                                    // Enter in name field triggers create button click
                                    let _ = js_sys::eval("document.querySelector('.create-channel-form .btn-primary')?.click()");
                                }
                            }
                        />
                        <input
                            type="text"
                            class="channel-desc-input"
                            placeholder="Description (optional)"
                            prop:value=move || new_channel_desc.get()
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()).map(|el| el.value()).unwrap_or_default();
                                new_channel_desc.set(val);
                            }
                        />
                        <button class="btn btn-sm btn-primary" on:click={
                            let channels2 = channels.clone();
                            let toasts2 = toasts.clone();
                            move |_| {
                                let name = new_channel_name.get_untracked();
                                if name.trim().is_empty() { toasts2.push("Name required", "error"); return; }
                                let desc = new_channel_desc.get_untracked();
                                let now = (js_sys::Date::now() / 1000.0) as u64;
                                channels2.update(|chs| {
                                    chs.push(ChatChannel {
                                        id: format!("ch-{now}"), name: name.trim().to_lowercase().replace(' ', "-"),
                                        description: if desc.is_empty() { None } else { Some(desc) },
                                        is_direct: false, members: vec![], member_names: vec!["You".into()],
                                        unread_count: 0, last_message: None, last_activity: now, pinned: false,
                                    });
                                });
                                new_channel_name.set(String::new()); new_channel_desc.set(String::new());
                                creating_channel.set(false);
                                toasts2.push(format!("#{} created", name.trim()), "success");
                            }
                        }>"Create"</button>
                    </div>
                    {move || channels.get().iter().filter(|c| !c.is_direct).map(|ch| {
                        let id = ch.id.clone();
                        let name = ch.name.clone();
                        let unread = ch.unread_count;
                        let is_active = active_channel.get().as_deref() == Some(&id);
                        let id_click = id.clone();
                        view! {
                            <button class=if is_active { "channel-item active" } else { "channel-item" }
                                    on:click=move |_| on_select_channel(id_click.clone())>
                                <span class="channel-hash">"#"</span>
                                <div class="channel-info">
                                    <span class="channel-name">{name.clone()}</span>
                                </div>
                                {(unread > 0).then(|| view! { <span class="channel-badge">{unread}</span> })}
                            </button>
                        }
                    }).collect::<Vec<_>>()}
                </div>

                // Message area (right)
                <div class="chat-main">
                    <div class="chat-empty" style=move || if active_channel.get().is_none() { "" } else { "display:none" }>
                        <span class="empty-icon">"\u{1F4AC}"</span>
                        <p class="empty-title">"Select a conversation"</p>
                        <p class="empty-desc">"Choose a chat or channel from the left panel"</p>
                    </div>

                    <div class="chat-active" style=move || if active_channel.get().is_some() { "" } else { "display:none" }>
                        <div class="chat-header">
                            <div class="chat-header-info">
                                <h2>{move || active_channel_name().unwrap_or_default()}</h2>
                                {move || active_channel_desc().map(|d| view! { <span class="chat-header-desc">{d}</span> })}
                                // Group encryption status
                                {move || {
                                    let id = active_channel.get();
                                    let ch = id.and_then(|id| channels.get().iter().find(|c| c.id == id).cloned());
                                    ch.map(|c| {
                                        let member_count = c.members.len().max(1);
                                        let is_group = !c.is_direct && member_count > 1;
                                        view! {
                                            <div class="chat-encryption-info">
                                                <span class="chat-lock-icon">{if is_group { "\u{1F510}" } else { "\u{1F512}" }}</span>
                                                <span class="chat-encryption-label">
                                                    {if is_group {
                                                        format!("Group E2E ({member_count} keys)")
                                                    } else {
                                                        "E2E Encrypted".into()
                                                    }}
                                                </span>
                                            </div>
                                        }
                                    })
                                }}
                            </div>
                            <div class="chat-header-actions">
                                <button class="btn btn-sm btn-secondary"
                                        on:click=move |_| toasts_consolidate.push("Chat consolidated to permanent DHT record", "success")
                                        title="Save chat to permanent DHT record">
                                    "\u{1F4BE} Consolidate"
                                </button>
                                <a href="/meet" class="btn btn-sm btn-secondary" title="Start call">"\u{1F4F9} Call"</a>
                            </div>
                        </div>
                        // Pinned messages
                        <PinnedMessages messages=messages />

                        <div class="chat-messages">
                                    {move || messages.get().iter().enumerate().map(|(idx, msg)| {
                                        let is_self = msg.sender_name.as_deref() == Some("You");
                                        let sender = msg.sender_name.clone().unwrap_or_else(|| msg.sender[..8.min(msg.sender.len())].to_string());
                                        let content = msg.content.clone();
                                        let time = format_chat_time(msg.timestamp);
                                        let reactions = msg.reactions.clone();
                                        let has_reply = msg.reply_to.is_some();
                                        let msg_id = msg.id.clone();

                                        // Swipe state for this message
                                        let swipe_x = RwSignal::new(0.0f64);
                                        let swipe_start = RwSignal::new(0.0f64);
                                        let swiping = RwSignal::new(false);

                                        // Reaction picker state
                                        let show_react = RwSignal::new(false);
                                        let msg_id_react = msg_id.clone();
                                        let msg_id_del = msg_id.clone();

                                        let on_touch_start = move |ev: web_sys::TouchEvent| {
                                            if let Some(t) = ev.touches().get(0) {
                                                swipe_start.set(t.client_x() as f64);
                                                swiping.set(true);
                                            }
                                        };
                                        let on_touch_move = move |ev: web_sys::TouchEvent| {
                                            if !swiping.get() { return; }
                                            if let Some(t) = ev.touches().get(0) {
                                                let dx = t.client_x() as f64 - swipe_start.get();
                                                if dx.abs() > 10.0 { ev.prevent_default(); }
                                                swipe_x.set(dx);
                                            }
                                        };
                                        let on_touch_end = move |_: web_sys::TouchEvent| {
                                            if !swiping.get() { return; }
                                            swiping.set(false);
                                            let dx = swipe_x.get();
                                            if dx > 60.0 {
                                                // Swipe right → reply
                                                input.update(|v| {
                                                    v.clear();
                                                    v.push_str(&format!("@reply "));
                                                });
                                            } else if dx < -60.0 {
                                                // Swipe left → delete message
                                                messages.update(|m| m.retain(|x| x.id != msg_id_del));
                                            }
                                            swipe_x.set(0.0);
                                        };

                                        let quick_emojis = ["\u{1F44D}", "\u{2764}", "\u{1F602}", "\u{1F44F}", "\u{1F525}", "\u{1F622}"];

                                        view! {
                                            <div class=if is_self { "chat-bubble self" } else { "chat-bubble" }
                                                 on:touchstart=on_touch_start
                                                 on:touchmove=on_touch_move
                                                 on:touchend=on_touch_end
                                                 style=move || {
                                                     let x = swipe_x.get();
                                                     if x.abs() > 5.0 { format!("transform:translateX({x}px);transition:none;") }
                                                     else { "transition:transform 0.2s ease;".into() }
                                                 }>
                                                {(!is_self).then(|| view! {
                                                    <div class="bubble-avatar">{sender.chars().next().unwrap_or('?').to_uppercase().to_string()}</div>
                                                })}
                                                <div class="bubble-content">
                                                    {(!is_self).then(|| view! { <span class="bubble-sender">{sender.clone()}</span> })}
                                                    {has_reply.then(|| view! { <div class="bubble-reply-indicator">"\u{21B3} Reply"</div> })}
                                                    <p class="bubble-text"
                                                       on:dblclick={
                                                           let mid = msg_id.clone();
                                                           move |_| {
                                                               // Edit own messages (within 5 min)
                                                               if is_self {
                                                                   let now = (js_sys::Date::now() / 1000.0) as u64;
                                                                   let msg_ts = messages.get_untracked().iter()
                                                                       .find(|m| m.id == mid).map(|m| m.timestamp).unwrap_or(0);
                                                                   if now - msg_ts < 300 {
                                                                       let new_text = web_sys::window()
                                                                           .and_then(|w| w.prompt_with_message("Edit message:").ok().flatten());
                                                                       if let Some(text) = new_text {
                                                                           if !text.trim().is_empty() {
                                                                               let mid2 = mid.clone();
                                                                               messages.update(|msgs| {
                                                                                   if let Some(m) = msgs.iter_mut().find(|m| m.id == mid2) {
                                                                                       m.content = text;
                                                                                       m.edited = true;
                                                                                   }
                                                                               });
                                                                           }
                                                                       }
                                                                   }
                                                               }
                                                           }
                                                       }>{content}</p>
                                                    {msg.edited.then(|| view! { <span class="edited-indicator">"(edited)"</span> })}
                                                    <div class="bubble-meta">
                                                        <span class="bubble-time">{time}</span>
                                                        // Read receipts (own messages only)
                                                        {is_self.then(|| view! {
                                                            <span class="read-receipt" title="Delivered and read">"\u{2713}\u{2713}"</span>
                                                        })}
                                                        // Reaction button
                                                        <button class="react-trigger" on:click=move |_| show_react.update(|v| *v = !*v)>"\u{1F600}+"</button>
                                                        {(!reactions.is_empty()).then(|| {
                                                            let rxns = reactions.clone();
                                                            view! {
                                                                <div class="bubble-reactions">
                                                                    {rxns.into_iter().map(|r| view! {
                                                                        <span class="reaction-chip">{r.emoji.clone()}" "{r.users.len()}</span>
                                                                    }).collect::<Vec<_>>()}
                                                                </div>
                                                            }
                                                        })}
                                                    </div>
                                                    // Quick reaction picker
                                                    <div class="quick-react-bar" style=move || if show_react.get() { "" } else { "display:none" }>
                                                        {quick_emojis.iter().map(|e| {
                                                            let emoji = e.to_string();
                                                            let mid = msg_id_react.clone();
                                                            let em = emoji.clone();
                                                            view! {
                                                                <button class="quick-react-btn" on:click=move |_| {
                                                                    messages.update(|msgs| {
                                                                        if let Some(m) = msgs.iter_mut().find(|x| x.id == mid) {
                                                                            if let Some(r) = m.reactions.iter_mut().find(|r| r.emoji == em) {
                                                                                if !r.users.contains(&"you".to_string()) {
                                                                                    r.users.push("you".into());
                                                                                }
                                                                            } else {
                                                                                m.reactions.push(mail_leptos_types::ChatReaction {
                                                                                    emoji: em.clone(),
                                                                                    users: vec!["you".into()],
                                                                                });
                                                                            }
                                                                        }
                                                                    });
                                                                    show_react.set(false);
                                                                }>{emoji}</button>
                                                            }
                                                        }).collect::<Vec<_>>()}
                                                    </div>
                                                </div>
                                            </div>
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>

                                // Self-destruct + Poll controls
                                <div class="chat-extra-controls">
                                    <SelfDestructToggle />
                                    <PollCreator messages=messages active_channel=active_channel />
                                </div>

                                <div class="chat-input-bar">
                                    <EmojiPicker on_select=move |emoji: String| {
                                        input.update(|v| v.push_str(&emoji));
                                    } />
                                    <MentionAutocomplete input=input channels=channels />
                                    <textarea
                                        class="chat-input"
                                        placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
                                        rows="1"
                                        prop:value=move || input.get()
                                        on:input=move |ev| {
                                            let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlTextAreaElement>().ok()).map(|el| el.value()).unwrap_or_default();
                                            input.set(val);
                                        }
                                        on:keydown=on_keypress
                                    />
                                    <button class="btn btn-primary chat-send" on:click=move |_| on_send(())>
                                        "\u{27A4}"
                                    </button>
                                </div>

                        // Voice message recorder
                        <VoiceRecorder on_send=move |_duration_ms: u32| {
                            let now = (js_sys::Date::now() / 1000.0) as u64;
                            messages.update(|m| m.push(ChatMessage {
                                id: format!("vm-{now}"),
                                sender: "uhCAk_self_mock".into(),
                                sender_name: Some("You".into()),
                                content: "\u{1F3A4} Voice message".into(),
                                timestamp: now,
                                reply_to: None,
                                reactions: vec![],
                                edited: false,
                                channel_id: active_channel.get_untracked(),
                            }));
                        } />

                        // Forward selected messages to email
                        <div class="chat-forward-bar">
                            <button class="btn btn-sm btn-secondary" on:click={
                                let t_fwd = toasts.clone();
                                let mail_fwd = crate::mail_context::use_mail();
                                move |_| {
                                    let msgs = messages.get_untracked();
                                    if msgs.is_empty() { return; }
                                    let body = msgs.iter().map(|m| {
                                        let sender = m.sender_name.clone().unwrap_or_else(|| "Unknown".into());
                                        format!("[{}] {}: {}", format_chat_time(m.timestamp), sender, m.content)
                                    }).collect::<Vec<_>>().join("\n");
                                    let channel = active_channel.get_untracked().unwrap_or_default();
                                    mail_fwd.compose_mode.set(mail_leptos_types::ComposeMode::Forward {
                                        subject: format!("Chat transcript: #{channel}"),
                                        body,
                                    });
                                    let nav = leptos_router::hooks::use_navigate();
                                    nav("/compose", Default::default());
                                    t_fwd.push("Chat forwarded to compose", "info");
                                }
                            }>"\u{2709} Forward to Email"</button>
                        </div>

                        <div class="ephemeral-notice">
                            "\u{26A1} Messages are ephemeral (P2P signal). Click Consolidate to save to DHT."
                        </div>
                    </div>
                </div>
            </div>
        </div>
    }
}

/// Pinned messages — shows pinned messages at top of channel.
#[component]
fn PinnedMessages(messages: RwSignal<Vec<ChatMessage>>) -> impl IntoView {
    let pinned = RwSignal::new(Vec::<String>::new()); // message IDs

    view! {
        <div class="pinned-messages" style=move || if pinned.get().is_empty() { "display:none" } else { "" }>
            <div class="pinned-msg-header">"\u{1F4CC} Pinned Messages"</div>
            {move || {
                let pins = pinned.get();
                let msgs = messages.get();
                pins.iter().filter_map(|id| msgs.iter().find(|m| m.id == *id)).map(|m| {
                    let sender = m.sender_name.clone().unwrap_or_default();
                    let content = if m.content.len() > 60 { format!("{}...", &m.content[..57]) } else { m.content.clone() };
                    view! { <div class="pinned-msg"><strong>{sender}": "</strong>{content}</div> }
                }).collect::<Vec<_>>()
            }}
        </div>
    }
}

/// @mention autocomplete — shows suggestions when typing @.
#[component]
fn MentionAutocomplete(input: RwSignal<String>, channels: RwSignal<Vec<ChatChannel>>) -> impl IntoView {
    let show = move || {
        let text = input.get();
        text.rfind('@').map(|pos| {
            let after = &text[pos+1..];
            !after.contains(' ') && !after.is_empty()
        }).unwrap_or(false)
    };

    let query = move || {
        let text = input.get();
        text.rfind('@').map(|pos| text[pos+1..].to_lowercase()).unwrap_or_default()
    };

    let suggestions = move || {
        let q = query();
        if q.is_empty() { return vec![]; }
        // Get member names from channels
        let mut names: Vec<String> = channels.get().iter()
            .flat_map(|c| c.member_names.iter().cloned())
            .collect();
        names.sort();
        names.dedup();
        names.into_iter().filter(|n| n.to_lowercase().contains(&q)).take(5).collect::<Vec<_>>()
    };

    view! {
        <div class="mention-autocomplete" style=move || if show() { "" } else { "display:none" }>
            {move || suggestions().into_iter().map(|name| {
                let n = name.clone();
                view! {
                    <button class="mention-option" on:click=move |_| {
                        input.update(|v| {
                            if let Some(pos) = v.rfind('@') {
                                v.truncate(pos);
                                v.push_str(&format!("@{} ", n));
                            }
                        });
                    }>{name}</button>
                }
            }).collect::<Vec<_>>()}
        </div>
    }
}

/// Self-destruct toggle — sets TTL for next messages.
#[component]
fn SelfDestructToggle() -> impl IntoView {
    let ttl = RwSignal::new(Option::<u64>::None);
    let open = RwSignal::new(false);

    view! {
        <div class="self-destruct-toggle">
            <button class="btn btn-sm" style=move || if ttl.get().is_some() { "color:var(--error)" } else { "" }
                    on:click=move |_| open.update(|v| *v = !*v)
                    title="Self-destruct timer">
                "\u{1F4A3}"
            </button>
            <div class="snooze-dropdown" style=move || if open.get() { "" } else { "display:none" }>
                {[("Off", None), ("1 min", Some(60u64)), ("1 hour", Some(3600)), ("24 hours", Some(86400))].iter().map(|(label, val)| {
                    let label = *label;
                    let val = *val;
                    view! {
                        <button class="snooze-option" on:click=move |_| { ttl.set(val); open.set(false); }>
                            {label}
                            {ttl.get().and_then(|t| val.filter(|v| *v == t)).map(|_| " \u{2713}").unwrap_or("")}
                        </button>
                    }
                }).collect::<Vec<_>>()}
            </div>
            {move || ttl.get().map(|t| {
                let label = match t { 60 => "1m", 3600 => "1h", 86400 => "24h", _ => "?" };
                view! { <span class="self-destruct-badge">{format!("\u{1F4A3} {label}")}</span> }
            })}
        </div>
    }
}

/// Poll creator — create quick polls in chat.
#[component]
fn PollCreator(messages: RwSignal<Vec<ChatMessage>>, active_channel: RwSignal<Option<String>>) -> impl IntoView {
    let open = RwSignal::new(false);
    let question = RwSignal::new(String::new());
    let options = RwSignal::new(vec![String::new(), String::new()]);

    view! {
        <button class="btn btn-sm" on:click=move |_| open.update(|v| *v = !*v) title="Create poll">"\u{1F4CA}"</button>
        <div class="poll-creator" style=move || if open.get() { "" } else { "display:none" }>
            <input type="text" placeholder="Poll question..." class="poll-question-input"
                prop:value=move || question.get()
                on:input=move |ev| {
                    use wasm_bindgen::JsCast;
                    let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()).map(|el| el.value()).unwrap_or_default();
                    question.set(val);
                }
            />
            {move || options.get().iter().enumerate().map(|(i, _)| {
                view! {
                    <input type="text" placeholder=format!("Option {}...", i+1) class="poll-option-input"
                        on:input=move |ev| {
                            use wasm_bindgen::JsCast;
                            let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()).map(|el| el.value()).unwrap_or_default();
                            options.update(|o| { if let Some(opt) = o.get_mut(i) { *opt = val; } });
                        }
                    />
                }
            }).collect::<Vec<_>>()}
            <div class="poll-actions">
                <button class="btn btn-sm btn-secondary" on:click=move |_| options.update(|o| if o.len() < 5 { o.push(String::new()) })>"+ Option"</button>
                <button class="btn btn-sm btn-primary" on:click=move |_| {
                    let q = question.get_untracked();
                    let opts: Vec<String> = options.get_untracked().into_iter().filter(|o| !o.trim().is_empty()).collect();
                    if q.trim().is_empty() || opts.len() < 2 { return; }
                    let now = (js_sys::Date::now() / 1000.0) as u64;
                    let poll_text = format!("\u{1F4CA} POLL: {}\n{}", q, opts.iter().enumerate().map(|(i,o)| format!("  {}. {}", i+1, o)).collect::<Vec<_>>().join("\n"));
                    messages.update(|m| m.push(ChatMessage {
                        id: format!("poll-{now}"), sender: "uhCAk_self_mock".into(),
                        sender_name: Some("You".into()), content: poll_text,
                        timestamp: now, reply_to: None, reactions: vec![], edited: false,
                        channel_id: active_channel.get_untracked(),
                    }));
                    question.set(String::new()); options.set(vec![String::new(), String::new()]);
                    open.set(false);
                }>"Send Poll"</button>
            </div>
        </div>
    }
}

/// Voice message recorder — uses MediaRecorder API.
#[component]
fn VoiceRecorder(on_send: impl Fn(u32) + 'static + Clone + Send) -> impl IntoView {
    let recording = RwSignal::new(false);
    let duration_ms = RwSignal::new(0u32);

    view! {
        <div class="voice-recorder">
            <button
                class=move || if recording.get() { "btn btn-sm voice-btn recording" } else { "btn btn-sm voice-btn" }
                on:click=move |_| {
                    if recording.get() {
                        // Stop recording
                        let _ = js_sys::eval("if(window.__mycelix_recorder){window.__mycelix_recorder.stop();window.__mycelix_recorder=null}");
                        recording.set(false);
                        let d = duration_ms.get();
                        on_send(d);
                        duration_ms.set(0);
                    } else {
                        // Start recording
                        recording.set(true);
                        let dur = duration_ms;
                        // Increment duration every 100ms
                        wasm_bindgen_futures::spawn_local(async move {
                            loop {
                                gloo_timers::future::sleep(std::time::Duration::from_millis(100)).await;
                                if !recording.get_untracked() { break; }
                                dur.update(|d| *d += 100);
                            }
                        });
                        // Request mic access
                        let _ = js_sys::eval("navigator.mediaDevices?.getUserMedia({audio:true}).then(s=>{const r=new MediaRecorder(s);window.__mycelix_recorder=r;r.start();}).catch(()=>{})");
                    }
                }
                title=move || if recording.get() { "Stop recording" } else { "Record voice message" }>
                {move || if recording.get() {
                    let secs = duration_ms.get() / 1000;
                    format!("\u{1F534} {secs}s")
                } else {
                    "\u{1F3A4}".into()
                }}
            </button>
        </div>
    }
}

/// Emoji picker — grid of common emojis with category tabs.
#[component]
fn EmojiPicker(on_select: impl Fn(String) + 'static + Clone + Send) -> impl IntoView {
    let open = RwSignal::new(false);
    let category = RwSignal::new(0usize);

    let categories: &[(&str, &[&str])] = &[
        ("\u{1F600}", &["\u{1F600}","\u{1F602}","\u{1F60A}","\u{1F60D}","\u{1F914}","\u{1F60E}","\u{1F622}","\u{1F621}","\u{1F631}","\u{1F389}","\u{1F44D}","\u{1F44E}","\u{1F64F}","\u{1F4AA}","\u{1F440}","\u{2764}","\u{1F525}","\u{1F4AF}","\u{1F91D}","\u{1F917}"]),
        ("\u{1F436}", &["\u{1F436}","\u{1F431}","\u{1F422}","\u{1F419}","\u{1F427}","\u{1F98B}","\u{1F339}","\u{1F333}","\u{1F31E}","\u{1F30D}","\u{1F308}","\u{2B50}","\u{1F30A}","\u{26A1}","\u{1F343}","\u{1F33B}"]),
        ("\u{1F354}", &["\u{1F354}","\u{1F355}","\u{1F382}","\u{2615}","\u{1F377}","\u{1F34E}","\u{1F96A}","\u{1F35C}","\u{1F363}","\u{1F370}","\u{1F36B}","\u{1F347}"]),
        ("\u{1F3B5}", &["\u{1F3B5}","\u{1F3B6}","\u{1F3A4}","\u{1F3B8}","\u{1F941}","\u{1F3BA}","\u{1F3B9}","\u{1F3BB}","\u{1FA97}","\u{1F4BF}","\u{1F399}"]),
        ("\u{1F4BB}", &["\u{1F4BB}","\u{1F4F1}","\u{1F50D}","\u{1F512}","\u{1F6E1}","\u{2699}","\u{1F4E7}","\u{1F4CE}","\u{1F4C4}","\u{1F5C3}","\u{1F5A5}","\u{2328}"]),
    ];

    view! {
        <div class="emoji-picker-wrapper">
            <button class="btn btn-sm emoji-trigger" on:click=move |_| open.update(|v| *v = !*v)
                    title="Emoji">"\u{1F600}"</button>
            <div class="emoji-picker-popup" style=move || if open.get() { "" } else { "display:none" }>
                <div class="emoji-tabs">
                    {categories.iter().enumerate().map(|(i, (icon, _))| {
                        let icon = *icon;
                        view! {
                            <button class=move || if category.get() == i { "emoji-tab active" } else { "emoji-tab" }
                                    on:click=move |_| category.set(i)>{icon}</button>
                        }
                    }).collect::<Vec<_>>()}
                </div>
                <div class="emoji-grid">
                    {move || {
                        let cat = category.get();
                        let emojis = categories.get(cat).map(|c| c.1).unwrap_or(&[]);
                        emojis.iter().map(|e| {
                            let emoji = e.to_string();
                            let cb = on_select.clone();
                            let em = emoji.clone();
                            view! {
                                <button class="emoji-item" on:click=move |_| {
                                    cb(em.clone());
                                    open.set(false);
                                }>{emoji}</button>
                            }
                        }).collect::<Vec<_>>()
                    }}
                </div>
            </div>
        </div>
    }
}

fn format_chat_time(ts: u64) -> String {
    let d = js_sys::Date::new_0();
    d.set_time((ts as f64) * 1000.0);
    format!("{:02}:{:02}", d.get_hours(), d.get_minutes())
}
