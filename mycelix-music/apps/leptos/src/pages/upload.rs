// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

const STRATEGIES: &[(&str, &str)] = &[
    ("pay_per_stream", "Pay Per Stream ($0.01/play)"),
    ("gift", "Gift Economy (free + tips)"),
    ("patronage", "Patronage (monthly support)"),
    ("freemium", "Freemium (free 128kbps, premium FLAC)"),
    ("premium", "Premium (2x rate)"),
];

#[component]
pub fn UploadPage() -> impl IntoView {
    let title = RwSignal::new(String::new());
    let ipfs_cid = RwSignal::new(String::new());
    let duration = RwSignal::new(180u32);
    let genre_input = RwSignal::new(String::new());
    let strategy = RwSignal::new("pay_per_stream".to_string());
    let status_msg = RwSignal::new(String::new());

    let on_submit = move |ev: web_sys::SubmitEvent| {
        ev.prevent_default();
        // TODO: use_zome_call for create_song
        // For now, show a status message
        if title.get().is_empty() || ipfs_cid.get().is_empty() {
            status_msg.set("Title and IPFS CID are required".into());
            return;
        }
        status_msg.set(format!(
            "Ready to upload: '{}' (CID: {}, strategy: {}). Connect Holochain conductor to submit.",
            title.get(),
            ipfs_cid.get(),
            strategy.get()
        ));
    };

    let strategy_options = STRATEGIES
        .iter()
        .map(|(val, label)| {
            view! { <option value=*val>{*label}</option> }
        })
        .collect_view();

    view! {
        <div class="page upload-page">
            <h1>"Upload Music"</h1>
            <p class="page-subtitle">"Share your music with the world. Choose your economic model."</p>

            <form class="upload-form" on:submit=on_submit>
                <div class="form-group">
                    <label for="title">"Song Title"</label>
                    <input
                        id="title"
                        type="text"
                        placeholder="Enter song title"
                        prop:value=move || title.get()
                        on:input=move |ev| title.set(event_target_value(&ev))
                        required
                    />
                </div>

                <div class="form-group">
                    <label for="ipfs_cid">"IPFS CID"</label>
                    <input
                        id="ipfs_cid"
                        type="text"
                        placeholder="QmYourAudioFileCID..."
                        prop:value=move || ipfs_cid.get()
                        on:input=move |ev| ipfs_cid.set(event_target_value(&ev))
                        required
                    />
                    <span class="help-text">"Upload your audio to IPFS first, then paste the CID here."</span>
                </div>

                <div class="form-group">
                    <label for="duration">"Duration (seconds): " {move || duration.get().to_string()}</label>
                    <input
                        id="duration"
                        type="range"
                        min="30"
                        max="600"
                        prop:value=move || duration.get().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                duration.set(v);
                            }
                        }
                    />
                </div>

                <div class="form-group">
                    <label for="genres">"Genres (comma-separated)"</label>
                    <input
                        id="genres"
                        type="text"
                        placeholder="Electronic, Ambient"
                        prop:value=move || genre_input.get()
                        on:input=move |ev| genre_input.set(event_target_value(&ev))
                    />
                </div>

                <div class="form-group">
                    <label for="strategy">"Economic Strategy"</label>
                    <select
                        id="strategy"
                        on:change=move |ev| strategy.set(event_target_value(&ev))
                    >
                        {strategy_options}
                    </select>
                </div>

                <button type="submit" class="btn btn-primary">"Upload Song"</button>
            </form>

            {move || {
                let msg = status_msg.get();
                if msg.is_empty() {
                    view! { <div></div> }.into_any()
                } else {
                    view! { <div class="status-message">{msg}</div> }.into_any()
                }
            }}
        </div>
    }
}
