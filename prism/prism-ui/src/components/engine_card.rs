// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Individual search engine result card for the comparison view.

use crate::external_search::{ExternalResult, SearchStatus};
use leptos::prelude::*;

#[component]
pub fn EngineCard(result: ExternalResult, is_prism: bool) -> impl IntoView {
    let engine = result.engine.clone();
    let card_class = if is_prism {
        "engine-card prism"
    } else {
        "engine-card"
    };

    let label_class = match engine.as_str() {
        "Prism" => "engine-label prism-label",
        "Wikipedia" => "engine-label wiki-label",
        "DuckDuckGo" => "engine-label ddg-label",
        "Ollama" => "engine-label ollama-label",
        "Brave" => "engine-label brave-label",
        "Perplexity" => "engine-label perplexity-label",
        _ => "engine-label",
    };

    view! {
        <div class=card_class>
            <div class="engine-header">
                <span class=label_class>{engine.clone()}</span>
                {if is_prism {
                    Some(view! { <span class="engine-badge">"Epistemic"</span> })
                } else {
                    None
                }}
            </div>

            {match &result.status {
                SearchStatus::Loading => view! {
                    <div class="engine-loading">"Searching..."</div>
                }.into_any(),

                SearchStatus::Error(msg) => view! {
                    <div class="engine-error">
                        <p class="engine-error-title">"Not available"</p>
                        <p class="engine-error-msg">{msg.clone()}</p>
                    </div>
                }.into_any(),

                SearchStatus::Done | SearchStatus::Idle => {
                    if result.hits.is_empty() {
                        view! {
                            <div class="engine-empty">"No results"</div>
                        }.into_any()
                    } else {
                        view! {
                            <div class="engine-results">
                                {result.hits.iter().map(|hit| {
                                    let source_class = match hit.source_type.as_str() {
                                        "ai-generated" => "hit-source ai",
                                        "instant-answer" => "hit-source instant",
                                        "article" => "hit-source article",
                                        _ => "hit-source link",
                                    };
                                    view! {
                                        <div class="engine-hit">
                                            <div class="hit-title">{hit.title.clone()}</div>
                                            <div class="hit-snippet">{hit.snippet.clone()}</div>
                                            {hit.url.as_ref().map(|url| view! {
                                                <a class="hit-url" href=url.clone() target="_blank">{
                                                    if url.len() > 50 { format!("{}...", &url[..47]) } else { url.clone() }
                                                }</a>
                                            })}
                                            <span class=source_class>{hit.source_type.clone()}</span>
                                        </div>
                                    }
                                }).collect_view()}
                            </div>
                        }.into_any()
                    }
                }
            }}
        </div>
    }
}
