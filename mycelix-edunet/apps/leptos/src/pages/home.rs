// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

#[component]
pub fn HomePage() -> impl IntoView {
    view! {
        <div class="home">
            <h1>"Mycelix EduNet"</h1>
            <p class="subtitle">"Privacy-preserving decentralized education"</p>
            <div class="feature-grid">
                <FeatureCard
                    title="Adaptive Learning"
                    description="BKT, ZPD, and VARK personalization"
                    href="/courses"
                />
                <FeatureCard
                    title="Spaced Repetition"
                    description="SM-2 algorithm for optimal retention"
                    href="/review"
                />
                <FeatureCard
                    title="Verifiable Credentials"
                    description="W3C standard achievement certificates"
                    href="/credentials"
                />
                <FeatureCard
                    title="Community Governance"
                    description="Quadratic voting on curriculum"
                    href="/governance"
                />
            </div>
        </div>
    }
}

#[component]
fn FeatureCard(
    title: &'static str,
    description: &'static str,
    href: &'static str,
) -> impl IntoView {
    view! {
        <a href=href class="feature-card">
            <h3>{title}</h3>
            <p>{description}</p>
        </a>
    }
}
