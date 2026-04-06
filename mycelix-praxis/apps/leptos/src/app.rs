// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use leptos_router::{
    components::{Route, Router, Routes, A},
    hooks::use_params_map,
    path,
};

use crate::adaptivity_provider::AdaptivityProvider;
use crate::consciousness::ConsciousnessProvider;
use crate::curriculum::provide_curriculum_context;
use crate::holochain::{HolochainProvider, ConnectionBadge};
use crate::learning_engine::LearningEngineProvider;
use crate::pages::*;
use crate::role::{provide_role_context, UserRole};
use crate::student_profile::provide_profile_context;
use crate::theme::{provide_theme_context, use_theme, use_set_theme};

#[component]
pub fn App() -> impl IntoView {
    view! {
        <HolochainProvider>
            <ConsciousnessProvider>
            <LearningEngineProvider>
            <AdaptivityProvider>
                <AppInner />
            </AdaptivityProvider>
            </LearningEngineProvider>
            </ConsciousnessProvider>
        </HolochainProvider>
    }
}

#[component]
fn AppInner() -> impl IntoView {
    let (role, _set_role) = provide_role_context();
    let (_theme, _set_theme) = provide_theme_context();
    let (_profile, _set_profile) = provide_profile_context();
    provide_curriculum_context();
    crate::study_tracker::provide_study_tracker();
    crate::i18n::provide_i18n();

    // Wire consciousness signals to CSS custom properties
    crate::consciousness_ui::init_consciousness_ui();

    view! {
        <Router>
            <nav class="navbar">
                <a href="/" class="logo">"Praxis"</a>
                <div class="nav-links">
                    <RoleNav role=role />
                </div>
                <crate::search::SearchBar />
                <div class="nav-actions">
                    <crate::i18n::LanguagePicker />
                    <ThemeToggle />
                    <ConnectionBadge />
                </div>
            </nav>
            <CelebrationOverlay />
            <main>
                <Routes fallback=|| view! { <p>"Page not found"</p> }>
                    <Route path=path!("/") view=HomePage />
                    <Route path=path!("/courses") view=CoursesPage />
                    <Route path=path!("/review") view=ReviewPage />
                    <Route path=path!("/dashboard") view=DashboardPage />
                    <Route path=path!("/skill-map") view=SkillMapPage />
                    <Route path=path!("/study/:id") view=StudyPageWrapper />
                    <Route path=path!("/exam-prep") view=ExamPrepPage />
                    <Route path=path!("/mock-exam") view=MockExamPage />
                    <Route path=path!("/pathways") view=PathwaysPage />
                    <Route path=path!("/teacher") view=TeacherDashboardPage />
                    <Route path=path!("/governance") view=GovernancePage />
                    <Route path=path!("/credentials") view=CredentialsPage />
                </Routes>
            </main>
            <MobileBottomNav />
        </Router>
    }
}

#[component]
fn ThemeToggle() -> impl IntoView {
    let theme = use_theme();
    let set_theme = use_set_theme();

    view! {
        <button
            class="theme-toggle"
            title=move || format!("Switch to {} theme", theme.get().next().label())
            on:click=move |_| set_theme.set(theme.get().next())
        >
            {move || theme.get().icon()}
        </button>
    }
}

#[component]
fn RoleNav(role: ReadSignal<Option<UserRole>>) -> impl IntoView {
    move || match role.get() {
        None => view! {
            // No role selected — home page handles onboarding
        }.into_any(),
        Some(UserRole::Teacher) => view! {
            <A href="/teacher">"Dashboard"</A>
            <A href="/courses">"Courses"</A>
            <A href="/skill-map">"Knowledge Garden"</A>
            <A href="/credentials">"Assessments"</A>
        }.into_any(),
        Some(UserRole::Student) => view! {
            <A href="/dashboard">"Dashboard"</A>
            <A href="/skill-map">"Knowledge Garden"</A>
            <A href="/review">"Review"</A>
            <A href="/exam-prep">"Exam Prep"</A>
        }.into_any(),
        Some(UserRole::Parent) => view! {
            <A href="/dashboard">"Progress"</A>
            <A href="/credentials">"Reports"</A>
        }.into_any(),
    }
}

/// Brief celebration overlay when a topic is mastered.
#[component]
fn CelebrationOverlay() -> impl IntoView {
    let celebrating = expect_context::<ReadSignal<bool>>();
    let topic = expect_context::<ReadSignal<String>>();

    view! {
        {move || {
            if celebrating.get() {
                let topic_name = topic.get();
                view! {
                    <div class="mastery-celebration">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem">"\u{1F331}"</div>
                        <div class="mastery-celebration-text">"Deeply rooted"</div>
                        {if !topic_name.is_empty() {
                            view! { <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.25rem">{topic_name}" has taken root in your garden"</div> }.into_any()
                        } else {
                            view! { <span></span> }.into_any()
                        }}
                    </div>
                }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }
        }}
    }
}

/// Mobile-only bottom navigation with icons — 5 main sections.
#[component]
fn MobileBottomNav() -> impl IntoView {
    view! {
        <nav class="mobile-bottom-nav" aria-label="Main navigation">
            <a href="/" class="bottom-tab">
                <span class="bottom-tab-icon">"\u{1F3E0}"</span>
                <span class="bottom-tab-label">"Home"</span>
            </a>
            <a href="/skill-map" class="bottom-tab">
                <span class="bottom-tab-icon">"\u{2728}"</span>
                <span class="bottom-tab-label">"Explore"</span>
            </a>
            <a href="/review" class="bottom-tab">
                <span class="bottom-tab-icon">"\u{1F4DD}"</span>
                <span class="bottom-tab-label">"Review"</span>
            </a>
            <a href="/dashboard" class="bottom-tab">
                <span class="bottom-tab-icon">"\u{1F4CA}"</span>
                <span class="bottom-tab-label">"Progress"</span>
            </a>
            <a href="/exam-prep" class="bottom-tab">
                <span class="bottom-tab-icon">"\u{1F3AF}"</span>
                <span class="bottom-tab-label">"Exam"</span>
            </a>
        </nav>
    }
}

/// Wrapper to extract the :id param and pass it to StudyPage.
#[component]
fn StudyPageWrapper() -> impl IntoView {
    let params = use_params_map();
    let node_id = move || {
        params.read().get("id").unwrap_or_default()
    };

    view! {
        {move || {
            let id = node_id();
            view! { <StudyPage node_id=id /> }
        }}
    }
}
