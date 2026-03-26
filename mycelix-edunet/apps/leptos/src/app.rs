// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use leptos_router::{
    components::{Route, Router, Routes, A},
    path,
};

use crate::consciousness::ConsciousnessProvider;
use crate::holochain::{ConnectionBadge, HolochainProvider};
use crate::learning_engine::LearningEngineProvider;
use crate::pages::*;
use crate::role::{provide_role_context, UserRole};

#[component]
pub fn App() -> impl IntoView {
    view! {
        <HolochainProvider>
        <ConsciousnessProvider>
        <LearningEngineProvider>
            <AppInner />
        </LearningEngineProvider>
        </ConsciousnessProvider>
        </HolochainProvider>
    }
}

#[component]
fn AppInner() -> impl IntoView {
    let (role, _set_role) = provide_role_context();

    view! {
        <Router>
            <nav class="navbar">
                <a href="/" class="logo">"EduNet"</a>
                <div class="nav-links">
                    <RoleNav role=role />
                </div>
                <ConnectionBadge />
            </nav>
            <main>
                <Routes fallback=|| view! { <p>"Page not found"</p> }>
                    <Route path=path!("/") view=HomePage />
                    <Route path=path!("/courses") view=CoursesPage />
                    <Route path=path!("/review") view=ReviewPage />
                    <Route path=path!("/dashboard") view=DashboardPage />
                    <Route path=path!("/skill-map") view=SkillMapPage />
                    <Route path=path!("/teacher") view=TeacherDashboardPage />
                    <Route path=path!("/governance") view=GovernancePage />
                    <Route path=path!("/credentials") view=CredentialsPage />
                </Routes>
            </main>
        </Router>
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
            <A href="/courses">"My Class"</A>
            <A href="/skill-map">"Skill Map"</A>
            <A href="/credentials">"Assessments"</A>
        }.into_any(),
        Some(UserRole::Student) => view! {
            <A href="/dashboard">"My Learning"</A>
            <A href="/review">"Review"</A>
            <A href="/skill-map">"Skill Map"</A>
            <A href="/credentials">"Achievements"</A>
        }.into_any(),
        Some(UserRole::Parent) => view! {
            <A href="/dashboard">"My Child's Progress"</A>
            <A href="/credentials">"Reports"</A>
        }.into_any(),
    }
}
