// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Home page — the mentor's greeting.
//!
//! Instead of "select your role," the app asks: "What do you want to do today?"
//! The child is met as a whole person, not sorted into an institutional category.
//!
//! Teachers and parents can still access their tools via a small link at the
//! bottom — but the primary experience is the child's.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::adaptivity_provider::use_adaptivity;
use crate::cognitive_adaptivity::*;
use crate::role::{use_set_role, UserRole};
use crate::student_profile::use_profile;

fn event_target_value(ev: &leptos::ev::Event) -> String {
    ev.target()
        .and_then(|t| t.dyn_into::<web_sys::HtmlSelectElement>().ok())
        .map(|el| el.value())
        .or_else(|| {
            ev.target()
                .and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok())
                .map(|el| el.value())
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HomeView {
    /// The mentor's greeting — "What do you want to do today?"
    MentorGreeting,
    /// Teacher/parent setup (accessible from bottom link)
    TeacherSetup,
    ParentConnect,
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

#[component]
pub fn HomePage() -> impl IntoView {
    let (view_state, set_view_state) = signal(HomeView::MentorGreeting);
    let set_role = use_set_role();
    let adaptivity = use_adaptivity();

    let go_back = move |_| set_view_state.set(HomeView::MentorGreeting);

    view! {
        <div class="home-landing">
            {move || match view_state.get() {
                HomeView::MentorGreeting => view! {
                    <MentorGreeting set_role=set_role adaptivity=adaptivity.clone() set_view_state=set_view_state />
                }.into_any(),

                HomeView::TeacherSetup => view! {
                    <TeacherSetupForm set_role=set_role go_back=go_back />
                }.into_any(),

                HomeView::ParentConnect => view! {
                    <ParentConnectForm set_role=set_role go_back=go_back />
                }.into_any(),
            }}
        </div>
    }
}

// ---------------------------------------------------------------------------
// The mentor's greeting
// ---------------------------------------------------------------------------

#[component]
fn MentorGreeting(
    set_role: WriteSignal<Option<UserRole>>,
    adaptivity: crate::adaptivity_provider::AdaptivityCtx,
    set_view_state: WriteSignal<HomeView>,
) -> impl IntoView {
    let navigate = leptos_router::hooks::use_navigate();
    let nav_review = navigate.clone();
    let nav_explore = navigate.clone();
    let nav_create = navigate.clone();
    let nav_help = navigate.clone();
    let nav_play = navigate.clone();

    let set_role_review = set_role;
    let set_role_explore = set_role;
    let set_role_create = set_role;
    let set_role_help = set_role;
    let set_role_play = set_role;

    let adaptivity_sandbox = adaptivity.clone();

    let profile = use_profile();

    // Personalized greeting based on time of day + student name
    let greeting = {
        let hour = (js_sys::Date::new_0().get_hours()) as u8;
        let name = profile.get_untracked().name;
        let time_greeting = match hour {
            5..=11 => "Good morning",
            12..=16 => "Good afternoon",
            17..=20 => "Good evening",
            _ => "Hey",
        };
        if name.is_empty() {
            format!("{}!", time_greeting)
        } else {
            format!("{}, {}.", time_greeting, name)
        }
    };

    let on_practice = move |_| {
        set_role_review.set(Some(UserRole::Student));
        nav_review("/review", Default::default());
    };

    let on_explore = move |_| {
        set_role_explore.set(Some(UserRole::Student));
        nav_explore("/skill-map", Default::default());
    };

    let on_create = move |_| {
        set_role_create.set(Some(UserRole::Student));
        nav_create("/dashboard", Default::default());
    };

    let on_help = move |_| {
        set_role_help.set(Some(UserRole::Student));
        adaptivity.request_support();
        nav_help("/review", Default::default());
    };

    let on_play = move |_| {
        set_role_play.set(Some(UserRole::Student));
        adaptivity_sandbox.enter_sandbox();
        nav_play("/skill-map", Default::default());
    };

    view! {
        <div class="mentor-greeting">
            <div class="mentor-header">
                <h1 class="mentor-hello">{greeting}</h1>
                <p class="mentor-question">"What do you want to do today?"</p>
            </div>

            <div class="intention-cards">
                <button class="intention-card intention-practice" on:click=on_practice>
                    <span class="intention-icon">"\u{1f4dd}"</span>
                    <span class="intention-label">"Review flashcards"</span>
                    <span class="intention-hint">"Spaced repetition for exam prep"</span>
                </button>

                <button class="intention-card intention-explore" on:click=on_explore>
                    <span class="intention-icon">"\u{1f4d0}"</span>
                    <span class="intention-label">"Study a topic"</span>
                    <span class="intention-hint">"Browse the curriculum"</span>
                </button>

                <button class="intention-card intention-create" on:click=on_create>
                    <span class="intention-icon">"\u{1f4ca}"</span>
                    <span class="intention-label">"View my progress"</span>
                    <span class="intention-hint">"Track mastery and find gaps"</span>
                </button>

                <button class="intention-card intention-help" on:click=on_help>
                    <span class="intention-icon">"\u{1f91d}"</span>
                    <span class="intention-label">"I need help"</span>
                    <span class="intention-hint">"Get guided support"</span>
                </button>

                <button class="intention-card intention-play" on:click=on_play>
                    <span class="intention-icon">"\u{1f50d}"</span>
                    <span class="intention-label">"Explore freely"</span>
                    <span class="intention-hint">"Browse without tracking"</span>
                </button>
            </div>

            <div class="mentor-footer">
                <p class="privacy-note">
                    "Your learning data stays on your device. You own it."
                </p>
                <div class="adult-links">
                    <button
                        class="adult-link"
                        on:click=move |_| set_view_state.set(HomeView::TeacherSetup)
                    >
                        "I'm a teacher"
                    </button>
                    <span class="adult-link-divider">"\u{00b7}"</span>
                    <button
                        class="adult-link"
                        on:click=move |_| set_view_state.set(HomeView::ParentConnect)
                    >
                        "I'm a parent"
                    </button>
                </div>
            </div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Teacher setup (from small link at bottom)
// ---------------------------------------------------------------------------

#[component]
fn TeacherSetupForm(
    set_role: WriteSignal<Option<UserRole>>,
    go_back: impl Fn(leptos::ev::MouseEvent) + 'static,
) -> impl IntoView {
    let (grade, set_grade) = signal(String::new());
    let (subject, set_subject) = signal(String::new());
    let navigate = leptos_router::hooks::use_navigate();

    let on_create = move |_| {
        set_role.set(Some(UserRole::Teacher));
        navigate("/teacher", Default::default());
    };

    let can_create = move || !grade.get().is_empty() && !subject.get().is_empty();

    view! {
        <div class="setup-form">
            <button class="back-button" on:click=go_back>"\u{2190} Back"</button>
            <h2 class="setup-title">"Welcome, Teacher"</h2>
            <p class="setup-subtitle">"Let's set up your classroom."</p>

            <div class="form-group">
                <label for="grade">"What grade do you teach?"</label>
                <select id="grade"
                    on:change=move |ev| set_grade.set(event_target_value(&ev))
                >
                    <option value="" disabled selected>"Choose a grade..."</option>
                    <option value="prek">"Pre-K"</option>
                    <option value="k">"Kindergarten"</option>
                    {(1..=12).map(|g| {
                        let suffix = match g { 1 => "st", 2 => "nd", 3 => "rd", _ => "th" };
                        let val = g.to_string();
                        let label = format!("{}{} Grade", g, suffix);
                        view! { <option value=val>{label}</option> }
                    }).collect_view()}
                </select>
            </div>

            <div class="form-group">
                <label for="subject">"What subject?"</label>
                <select id="subject"
                    on:change=move |ev| set_subject.set(event_target_value(&ev))
                >
                    <option value="" disabled selected>"Choose a subject..."</option>
                    <option value="math">"Mathematics"</option>
                    <option value="ela">"English Language Arts"</option>
                    <option value="science">"Science"</option>
                    <option value="social-studies">"Social Studies"</option>
                    <option value="art">"Art"</option>
                    <option value="music">"Music"</option>
                    <option value="other">"Other"</option>
                </select>
            </div>

            <button
                class="primary-button"
                on:click=on_create
                disabled=move || !can_create()
            >
                "Create your classroom"
            </button>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Parent connect (from small link at bottom)
// ---------------------------------------------------------------------------

#[component]
fn ParentConnectForm(
    set_role: WriteSignal<Option<UserRole>>,
    go_back: impl Fn(leptos::ev::MouseEvent) + 'static,
) -> impl IntoView {
    let (code, set_code) = signal(String::new());
    let navigate = leptos_router::hooks::use_navigate();

    let on_connect = move |_| {
        set_role.set(Some(UserRole::Parent));
        navigate("/dashboard", Default::default());
    };

    let can_connect = move || code.get().len() == 6;

    view! {
        <div class="setup-form">
            <button class="back-button" on:click=go_back>"\u{2190} Back"</button>
            <h2 class="setup-title">"Welcome, Parent"</h2>
            <p class="setup-subtitle">"Stay connected with your child's learning."</p>

            <div class="form-group">
                <label for="child-code">"Enter your child's class code"</label>
                <input
                    id="child-code"
                    type="text"
                    maxlength="6"
                    placeholder="ABC123"
                    class="code-input"
                    on:input=move |ev| set_code.set(event_target_value(&ev).to_uppercase())
                    prop:value=code
                />
            </div>

            <button
                class="primary-button"
                on:click=on_connect
                disabled=move || !can_connect()
            >
                "Connect"
            </button>
        </div>
    }
}
