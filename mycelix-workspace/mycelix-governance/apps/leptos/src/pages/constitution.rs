// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Constitution page: the charter that breathes.
//!
//! Immutable rights have visual bedrock weight. Amendable articles breathe.
//! AI-interactable: data-article-number, data-right, data-charter-version.

use leptos::prelude::*;
use crate::contexts::governance_context::use_governance;

#[component]
pub fn ConstitutionPage() -> impl IntoView {
    let gov = use_governance();

    // Known immutable rights (from ANTI_TYRANNY_DESIGN.md)
    let immutable_rights: Vec<&str> = vec![
        "consciousness gating", "term limits", "emergency power limits",
        "right to exit", "right to fork", "oversight funding",
        "sovereignty", "core principles", "veto override",
        "permission-less enforcement", "golden veto sunset",
    ];

    view! {
        <div class="constitution-page" data-page="constitution" role="main">
            <h1 class="page-title">"Constitution"</h1>

            {move || {
                gov.charter.get().map(|charter| {
                    let version = charter.version;
                    view! {
                        <article
                            class="charter-document"
                            data-charter-version=version.to_string()
                            data-section="charter"
                        >
                            // Preamble
                            <section class="charter-preamble-section" aria-label="preamble">
                                <h2 class="charter-section-title">"Preamble"</h2>
                                <blockquote class="charter-preamble-text">
                                    {charter.preamble}
                                </blockquote>
                            </section>

                            // Articles
                            <section class="charter-articles-section" aria-label="articles">
                                <h2 class="charter-section-title">"Articles"</h2>
                                <div class="articles-list">
                                    {charter.articles.into_iter().map(|article| {
                                        let num = article.number;
                                        view! {
                                            <div
                                                class="article-card"
                                                data-article-number=num.to_string()
                                                role="article"
                                            >
                                                <h3 class="article-title">
                                                    {format!("Article {num}: {}", article.title)}
                                                </h3>
                                                <p class="article-content">{article.content}</p>
                                            </div>
                                        }
                                    }).collect_view()}
                                </div>
                            </section>

                            // Immutable Rights
                            <section class="charter-rights-section" aria-label="immutable rights">
                                <h2 class="charter-section-title">"Immutable Rights"</h2>
                                <p class="rights-desc">
                                    "These rights cannot be removed even by unanimous vote. "
                                    "They are the bedrock on which the commons stands."
                                </p>
                                <div class="rights-grid">
                                    {charter.rights.into_iter().map(|right| {
                                        let is_immutable = immutable_rights.iter()
                                            .any(|ir| right.to_lowercase().contains(ir));
                                        let class = if is_immutable {
                                            "right-badge right-immutable"
                                        } else {
                                            "right-badge right-amendable"
                                        };
                                        let right_clone = right.clone();
                                        view! {
                                            <span
                                                class=class
                                                data-right=right
                                                data-immutable=is_immutable.to_string()
                                            >
                                                {right_clone}
                                            </span>
                                        }
                                    }).collect_view()}
                                </div>
                            </section>

                            <div class="charter-version" data-metric="version">
                                {format!("Charter v{version}")}
                            </div>
                        </article>
                    }
                })
            }}
        </div>
    }
}
