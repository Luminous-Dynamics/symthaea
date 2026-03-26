// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

use crate::role::{use_set_role, UserRole};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HomeState {
    RoleSelection,
    TeacherSetup,
    StudentJoin,
    ParentConnect,
}

#[component]
pub fn HomePage() -> impl IntoView {
    let (state, set_state) = signal(HomeState::RoleSelection);
    let set_role = use_set_role();

    let select_teacher = move |_| set_state.set(HomeState::TeacherSetup);
    let select_student = move |_| set_state.set(HomeState::StudentJoin);
    let select_parent = move |_| set_state.set(HomeState::ParentConnect);
    let go_back = move |_| set_state.set(HomeState::RoleSelection);

    view! {
        <div class="home-landing">
            <div class="landing-header">
                <h1 class="landing-title">"EduNet"</h1>
                <p class="landing-subtitle">"Learning that grows with you"</p>
            </div>

            {move || match state.get() {
                HomeState::RoleSelection => view! {
                    <div class="role-selection">
                        <div class="role-cards">
                            <button class="role-card role-teacher" on:click=select_teacher>
                                <span class="role-icon">"👩\u{200d}🏫"</span>
                                <span class="role-label">"I'm a Teacher"</span>
                                <span class="role-description">"Set up my classroom"</span>
                            </button>
                            <button class="role-card role-student" on:click=select_student>
                                <span class="role-icon">"👨\u{200d}🎓"</span>
                                <span class="role-label">"I'm a Student"</span>
                                <span class="role-description">"Join my class"</span>
                            </button>
                            <button class="role-card role-parent" on:click=select_parent>
                                <span class="role-icon">"👪"</span>
                                <span class="role-label">"I'm a Parent"</span>
                                <span class="role-description">"See my child's progress"</span>
                            </button>
                        </div>
                        <p class="privacy-note">
                            "Your learning data stays on your device. No cloud. No tracking. You own it."
                        </p>
                    </div>
                }.into_any(),

                HomeState::TeacherSetup => view! {
                    <TeacherSetupForm set_role=set_role go_back=go_back />
                }.into_any(),

                HomeState::StudentJoin => view! {
                    <StudentJoinForm set_role=set_role go_back=go_back />
                }.into_any(),

                HomeState::ParentConnect => view! {
                    <ParentConnectForm set_role=set_role go_back=go_back />
                }.into_any(),
            }}
        </div>
    }
}

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
            <button class="back-button" on:click=go_back>"< Back"</button>
            <h2 class="setup-title">"Welcome, Teacher!"</h2>
            <p class="setup-subtitle">"Let's set up your classroom."</p>

            <div class="form-group">
                <label for="grade">"What grade do you teach?"</label>
                <select id="grade"
                    on:change=move |ev| set_grade.set(event_target_value(&ev))
                >
                    <option value="" disabled selected>"Choose a grade..."</option>
                    <option value="prek">"Pre-K"</option>
                    <option value="k">"Kindergarten"</option>
                    <option value="1">"1st Grade"</option>
                    <option value="2">"2nd Grade"</option>
                    <option value="3">"3rd Grade"</option>
                    <option value="4">"4th Grade"</option>
                    <option value="5">"5th Grade"</option>
                    <option value="6">"6th Grade"</option>
                    <option value="7">"7th Grade"</option>
                    <option value="8">"8th Grade"</option>
                    <option value="9">"9th Grade"</option>
                    <option value="10">"10th Grade"</option>
                    <option value="11">"11th Grade"</option>
                    <option value="12">"12th Grade"</option>
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
                    <option value="world-language">"World Languages"</option>
                    <option value="pe">"Physical Education"</option>
                    <option value="cs">"Computer Science"</option>
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

#[component]
fn StudentJoinForm(
    set_role: WriteSignal<Option<UserRole>>,
    go_back: impl Fn(leptos::ev::MouseEvent) + 'static,
) -> impl IntoView {
    let (code, set_code) = signal(String::new());
    let navigate = leptos_router::hooks::use_navigate();
    let navigate_explore = navigate.clone();

    let on_join = move |_| {
        set_role.set(Some(UserRole::Student));
        navigate("/dashboard", Default::default());
    };

    let on_explore = move |_| {
        set_role.set(Some(UserRole::Student));
        navigate_explore("/skill-map", Default::default());
    };

    let can_join = move || code.get().len() == 6;

    view! {
        <div class="setup-form">
            <button class="back-button" on:click=go_back>"< Back"</button>
            <h2 class="setup-title">"Welcome, Student!"</h2>
            <p class="setup-subtitle">"Ready to learn something new?"</p>

            <div class="form-group">
                <label for="class-code">"Enter your class code"</label>
                <input
                    id="class-code"
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
                on:click=on_join
                disabled=move || !can_join()
            >
                "Join"
            </button>

            <div class="divider">
                <span>"or"</span>
            </div>

            <button class="secondary-button" on:click=on_explore>
                "Explore on my own"
            </button>
        </div>
    }
}

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
            <button class="back-button" on:click=go_back>"< Back"</button>
            <h2 class="setup-title">"Welcome, Parent!"</h2>
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
