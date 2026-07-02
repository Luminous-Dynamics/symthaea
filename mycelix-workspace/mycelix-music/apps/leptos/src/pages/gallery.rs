// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

/// Gallery page for visual art — leverages the VisualArtMetadata, Gallery,
/// and Exhibition entry types added to the civic media zomes.
#[component]
pub fn GalleryPage() -> impl IntoView {
    // TODO: use_zome_call cross-cluster to civic media for galleries
    // For now, show placeholder UI

    view! {
        <div class="page gallery-page">
            <h1>"Art Gallery"</h1>
            <p class="page-subtitle">
                "Visual art, photography, and mixed media — curated by the community."
            </p>

            <section class="gallery-grid">
                <div class="gallery-card">
                    <div class="gallery-placeholder">"No galleries yet"</div>
                    <p>"Galleries and exhibitions are created by community curators (Participant+ tier)."</p>
                    <p>"Connect to a Holochain conductor to browse and create galleries."</p>
                </div>
            </section>

            <section class="gallery-info">
                <h2>"How Galleries Work"</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <h3>"Create"</h3>
                        <p>"Curators create galleries around themes — digital art, photography, sculpture."</p>
                    </div>
                    <div class="info-card">
                        <h3>"Curate"</h3>
                        <p>"Add publications with VisualArtMetadata — dimensions, medium, provenance chain."</p>
                    </div>
                    <div class="info-card">
                        <h3>"Exhibit"</h3>
                        <p>"Time-bounded exhibitions feature selected works from galleries."</p>
                    </div>
                </div>
            </section>
        </div>
    }
}
