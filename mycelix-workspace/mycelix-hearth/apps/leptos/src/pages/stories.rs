// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use crate::hearth_context::{use_hearth, member_name, mock_now};
use crate::components::TierGate;
use hearth_leptos_types::*;
use personal_leptos_types::TrustTier;

#[component]
pub fn StoriesPage() -> impl IntoView {
    let hearth = use_hearth();

    let (show_form, set_show_form) = signal(false);
    let (story_title, set_story_title) = signal(String::new());
    let (story_content, set_story_content) = signal(String::new());
    let (story_type, set_story_type) = signal("Memory".to_string());

    let submit_story = move |_| {
        let title = story_title.get();
        let content = story_content.get();
        if title.is_empty() || content.is_empty() { return; }
        let stype = match story_type.get().as_str() {
            "Tradition" => StoryType::Tradition,
            "Recipe" => StoryType::Recipe,
            "Wisdom" => StoryType::Wisdom,
            "Origin" => StoryType::Origin,
            _ => StoryType::Memory,
        };
        let from = hearth.my_agent.get();

        hearth.stories.update(|stories| {
            stories.insert(0, StoryView {
                hash: format!("story_user_{}", stories.len()),
                hearth_hash: "hearth_001".into(),
                title,
                content,
                storyteller: from,
                story_type: stype,
                tags: vec![],
                visibility: HearthVisibility::AllMembers,
                created_at: mock_now(),
            });
        });

        set_story_title.set(String::new());
        set_story_content.set(String::new());
        set_show_form.set(false);
    };

    view! {
        <div class="page stories-page">
            <h1 class="page-title">"stories & traditions"</h1>
            <p class="page-subtitle">"family memory, passed forward"</p>

            <TierGate min_tier=TrustTier::Basic action_label="share stories">
                <button
                    class="action-btn"
                    on:click=move |_| set_show_form.set(!show_form.get())
                    aria-label="share a story"
                >
                    {move || if show_form.get() { "cancel" } else { "+ share a story" }}
                </button>
            </TierGate>

            // Story form
            {move || show_form.get().then(|| view! {
                <div class="form-card" role="form" aria-label="share a story">
                    <div class="form-row">
                        <label for="story-title">"title"</label>
                        <input
                            id="story-title"
                            type="text"
                            placeholder="what do you want to remember?"
                            prop:value=move || story_title.get()
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                if let Some(input) = ev.target().and_then(|t| t.dyn_ref::<web_sys::HtmlInputElement>().cloned()) {
                                    set_story_title.set(input.value());
                                }
                            }
                        />
                    </div>
                    <div class="form-row">
                        <label for="story-type">"kind"</label>
                        <select id="story-type" on:change=move |ev| {
                            use wasm_bindgen::JsCast;
                            if let Some(sel) = ev.target().and_then(|t| t.dyn_ref::<web_sys::HtmlSelectElement>().cloned()) {
                                set_story_type.set(sel.value());
                            }
                        }>
                            <option value="Memory">"memory"</option>
                            <option value="Tradition">"tradition"</option>
                            <option value="Recipe">"recipe"</option>
                            <option value="Wisdom">"wisdom"</option>
                            <option value="Origin">"origin"</option>
                        </select>
                    </div>
                    <div class="form-row">
                        <label for="story-content">"the story"</label>
                        <textarea
                            id="story-content"
                            rows="5"
                            placeholder="tell it in your own words…"
                            prop:value=move || story_content.get()
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                if let Some(input) = ev.target().and_then(|t| t.dyn_ref::<web_sys::HtmlTextAreaElement>().cloned()) {
                                    set_story_content.set(input.value());
                                }
                            }
                        ></textarea>
                    </div>
                    <button class="action-btn" on:click=submit_story>"share"</button>
                </div>
            })}

            {move || {
                let members = hearth.members.get();
                let stories = hearth.stories.get();

                if stories.is_empty() {
                    view! { <div class="empty-state">"every family has an origin story. what’s yours?"</div> }.into_any()
                } else {
                    view! {
                        <div class="story-list" role="list" aria-label="family stories">
                            {stories.iter().map(|s| {
                                let title = s.title.clone();
                                let content = s.content.clone();
                                let teller = member_name(&members, &s.storyteller);
                                let stype = format!("{:?}", s.story_type);
                                let tags = s.tags.join(", ");
                                view! {
                                    <div class="story-card" role="listitem">
                                        <div class="story-header">
                                            <h3>{title}</h3>
                                            <span class="story-type">{stype}</span>
                                        </div>
                                        <p class="story-content">{content}</p>
                                        <div class="story-footer">
                                            <span class="story-teller">"told by " {teller}</span>
                                            {(!tags.is_empty()).then(|| view! {
                                                <span class="story-tags">{tags}</span>
                                            })}
                                        </div>
                                    </div>
                                }
                            }).collect_view()}
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
