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
use crate::holochain::{ConnectionBadge, HolochainProvider};
use crate::learning_engine::LearningEngineProvider;
use crate::pages::*;
use crate::role::{provide_role_context, UserRole};
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
    provide_curriculum_context();

    view! {
        <Router>
            <nav class="navbar">
                <a href="/" class="logo">"EduNet"</a>
                <div class="nav-links">
                    <RoleNav role=role />
                </div>
                <div class="nav-actions">
                    <ThemeToggle />
                    <ConnectionBadge />
                </div>
            </nav>
            <main>
                <Routes fallback=|| view! { <p>"Page not found"</p> }>
                    <Route path=path!("/") view=HomePage />
                    <Route path=path!("/courses") view=CoursesPage />
                    <Route path=path!("/review") view=ReviewPage />
                    <Route path=path!("/dashboard") view=DashboardPage />
                    <Route path=path!("/skill-map") view=SkillMapPage />
                    <Route path=path!("/study/:id") view=StudyPageWrapper />
                    <Route path=path!("/teacher") view=TeacherDashboardPage />
                    <Route path=path!("/governance") view=GovernancePage />
                    <Route path=path!("/credentials") view=CredentialsPage />
                </Routes>
            </main>
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
            <A href="/skill-map">"Skill Map"</A>
            <A href="/credentials">"Assessments"</A>
        }.into_any(),
        Some(UserRole::Student) => view! {
            <A href="/dashboard">"Dashboard"</A>
            <A href="/review">"Review"</A>
            <A href="/skill-map">"Skill Map"</A>
            <A href="/credentials">"Achievements"</A>
        }.into_any(),
        Some(UserRole::Parent) => view! {
            <A href="/dashboard">"Progress"</A>
            <A href="/credentials">"Reports"</A>
        }.into_any(),
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
