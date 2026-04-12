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
use crate::curriculum::{use_progress, curriculum_graph};
use crate::role::{use_set_role, UserRole};
use crate::student_profile::{use_profile, use_set_profile};

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
    /// First-time onboarding — collect name and grade
    Onboarding,
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
    let profile = use_profile();
    let initial_view = if profile.get_untracked().onboarding_complete {
        HomeView::MentorGreeting
    } else {
        HomeView::Onboarding
    };
    let (view_state, set_view_state) = signal(initial_view);
    let set_role = use_set_role();
    let adaptivity = use_adaptivity();

    let go_back = move |_| set_view_state.set(HomeView::MentorGreeting);

    view! {
        <div class="home-landing">
            {move || match view_state.get() {
                HomeView::Onboarding => view! {
                    <OnboardingFlow set_view_state=set_view_state set_role=set_role />
                }.into_any(),
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
    let progress = use_progress();

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

    // Check if this is effectively a first visit (no progress yet)
    let is_first_visit = {
        let p = progress.get_untracked();
        p.mastered_count() == 0 && p.studying_count() == 0
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

            // First visit guidance — warm, specific, not overwhelming
            {if is_first_visit {
                // Find a good first topic based on the student's grade
                let grade = profile.get_untracked().grade;
                let graph = curriculum_graph();
                let grade_str = format!("Grade{}", grade);
                let first_topic = graph.nodes.iter()
                    .filter(|n| n.grade_levels.first().map(|g| g == &grade_str).unwrap_or(false))
                    .filter(|n| n.exam_weight.as_ref().map(|w| w.marks).unwrap_or(0) > 0)
                    .max_by_key(|n| n.exam_weight.as_ref().map(|w| w.marks).unwrap_or(0));

                if let Some(topic) = first_topic {
                    let href = format!("/study/{}", topic.id);
                    let title = topic.title.clone();
                    let marks = topic.exam_weight.as_ref().map(|w| w.marks).unwrap_or(0);
                    view! {
                        <div class="first-visit-guide" style="max-width: 700px; margin: 0 auto 1.5rem; padding: 1.25rem; background: var(--soil-rich); border-radius: 16px; text-align: left">
                            <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem">
                                "\u{1F331} Welcome to your knowledge garden"
                            </div>
                            <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.75rem; line-height: 1.6">
                                "Everything starts with one seed. Here's a good first topic for Grade "{grade}":"
                            </p>
                            <a href=href style="display: flex; align-items: center; justify-content: space-between; padding: 0.75rem; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; text-decoration: none; color: var(--text); transition: border-color 0.2s">
                                <span style="font-weight: 600; font-size: 0.9rem">{title}</span>
                                <span style="font-size: 0.75rem; color: var(--primary)">{marks}"m \u{2192}"</span>
                            </a>
                            <p style="font-size: 0.7rem; color: var(--text-tertiary); margin-top: 0.5rem">
                                "Or explore freely below \u{2014} there's no wrong way to start."
                            </p>
                        </div>
                    }.into_any()
                } else {
                    view! { <span></span> }.into_any()
                }
            } else {
                view! { <span></span> }.into_any()
            }}

            // Session orchestrator — smart start based on current state
            {
                let progress = use_progress();
                let navigate_session = navigate.clone();
                let set_role_session = set_role;
                let (session_dest, set_session_dest) = signal("/skill-map".to_string());
                let (session_hint, set_session_hint) = signal("Explore the Knowledge Garden and start growing".to_string());

                // Compute session recommendation reactively
                Effect::new(move |_| {
                    let p = progress.get();
                    let now = js_sys::Date::now();
                    let due_cards = p.srs_cards.values().filter(|c| now >= c.next_review_ms).count();
                    let weakest = p.weakest_topics(1);
                    let weakest_topic = weakest.first().map(|(id, pct)| {
                        let title = curriculum_graph().node(id).map(|n| n.title.clone()).unwrap_or_default();
                        (id.clone(), title, *pct)
                    });

                    if due_cards > 3 {
                        set_session_dest.set("/review".to_string());
                        set_session_hint.set(format!("{} cards due for review", due_cards));
                    } else if let Some((ref id, ref title, pct)) = weakest_topic {
                        set_session_dest.set(format!("/study/{}", id));
                        set_session_hint.set(format!("{} needs attention ({}% mastery)", title, (pct * 100.0) as u32));
                    } else {
                        set_session_dest.set("/skill-map".to_string());
                        set_session_hint.set("Explore the Knowledge Garden and start growing".to_string());
                    }
                });

                view! {
                    <button class="session-start-btn" on:click=move |_| {
                        set_role_session.set(Some(UserRole::Student));
                        let dest = session_dest.get();
                        navigate_session(&dest, Default::default());
                    }>
                        <span style="font-size: 1.2rem; font-weight: 700">"Begin your session"</span>
                        <span style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem">{move || session_hint.get()}</span>
                    </button>
                }
            }

            <div class="intention-cards">
                <button class="intention-card intention-practice" on:click=on_practice>
                    <span class="intention-icon">"\u{1f4dd}"</span>
                    <span class="intention-label">"Review flashcards"</span>
                    <span class="intention-hint">"Spaced repetition for exam prep"</span>
                </button>

                <button class="intention-card intention-explore" on:click=on_explore>
                    <span class="intention-icon">"\u{1f4d0}"</span>
                    <span class="intention-label">"Study a topic"</span>
                    <span class="intention-hint">"Browse the Knowledge Garden"</span>
                </button>

                <button class="intention-card intention-create" on:click=on_create>
                    <span class="intention-icon">"\u{1f4ca}"</span>
                    <span class="intention-label">"View my progress"</span>
                    <span class="intention-hint">"Track growth and find gaps"</span>
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

                <button class="intention-card intention-exam" on:click=move |_| {
                    set_role.set(Some(UserRole::Student));
                    navigate("/mock-exam", Default::default());
                }>
                    <span class="intention-icon">"\u{23F1}"</span>
                    <span class="intention-label">"Mock exam"</span>
                    <span class="intention-hint">"Practice under pressure"</span>
                </button>

                <a href="/pathways" class="intention-card intention-explore" style="text-decoration: none; color: var(--text)">
                    <span class="intention-icon">"\u{1F6E4}"</span>
                    <span class="intention-label">"Career pathways"</span>
                    <span class="intention-hint">"Beyond school \u{2014} explore careers"</span>
                </a>
            </div>

            // Today's personalized study plan
            <crate::study_tracker::TodaysPlan />

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

// ---------------------------------------------------------------------------
// Conversational onboarding — 30 seconds, no forms
// ---------------------------------------------------------------------------

#[component]
fn OnboardingFlow(
    set_view_state: WriteSignal<HomeView>,
    set_role: WriteSignal<Option<UserRole>>,
) -> impl IntoView {
    let set_profile = use_set_profile();
    let (step, set_step) = signal(0_u8);
    let (name, set_name) = signal(String::new());
    let (framework, set_framework) = signal("caps".to_string());
    let (grade, set_grade) = signal(12_u8);

    let finish_onboarding = move || {
        set_profile.update(|p| {
            p.name = name.get_untracked();
            p.framework = framework.get_untracked();
            p.grade = grade.get_untracked();
            p.onboarding_complete = true;
            if grade.get_untracked() == 12 {
                p.exam_date = "2026-10-26".to_string();
            }
        });
        set_role.set(Some(UserRole::Student));
        set_view_state.set(HomeView::MentorGreeting);
    };

    view! {
        <div class="onboarding" style="max-width: 500px; margin: 0 auto; padding: 3rem 1.5rem; text-align: center">
            // Step 0: Name
            {move || if step.get() == 0 {
                view! {
                    <div class="onboarding-step" style="animation: card-in 0.4s ease-out">
                        <h1 style="font-size: 2rem; margin-bottom: 0.5rem; color: var(--primary)">"Hey there."</h1>
                        <p style="color: var(--text-secondary); margin-bottom: 2rem">"Welcome to Praxis."</p>
                        <p style="font-size: 1.1rem; margin-bottom: 1rem">"What should I call you?"</p>
                        <input
                            type="text"
                            placeholder="Your name"
                            autofocus=true
                            style="width: 100%; max-width: 300px; padding: 0.75rem 1rem; background: var(--surface); border: 2px solid var(--border); border-radius: 12px; color: var(--text); font-size: 1.1rem; text-align: center; outline: none; font-family: inherit"
                            prop:value=move || name.get()
                            on:input=move |ev| set_name.set(leptos::prelude::event_target_value(&ev))
                            on:keydown=move |ev: leptos::ev::KeyboardEvent| {
                                if ev.key() == "Enter" && !name.get_untracked().is_empty() {
                                    set_step.set(1);
                                }
                            }
                        />
                        <br />
                        <button
                            style="margin-top: 1.5rem; padding: 0.6rem 2rem; background: var(--primary); color: var(--text-on-primary); border: none; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; font-family: inherit"
                            on:click=move |_| if !name.get_untracked().is_empty() { set_step.set(1); }
                        >"Continue"</button>
                    </div>
                }.into_any()
            // Step 1: Framework selection
            } else if step.get() == 1 {
                let name_val = name.get();
                let fw_btn_style = "padding: 0.75rem 1.25rem; background: var(--surface); border: 2px solid var(--border); border-radius: 12px; color: var(--text); font-size: 0.9rem; font-weight: 600; cursor: pointer; font-family: inherit; transition: border-color 0.15s";
                view! {
                    <div class="onboarding-step" style="animation: card-in 0.4s ease-out">
                        <h1 style="font-size: 1.5rem; margin-bottom: 0.5rem; color: var(--primary)">
                            "Nice to meet you, "{name_val}"."
                        </h1>
                        <p style="color: var(--text-secondary); margin-bottom: 1.5rem">"What curriculum are you following?"</p>

                        <div style="display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap; margin-bottom: 1rem">
                            <button
                                style=fw_btn_style
                                on:click=move |_| { set_framework.set("caps".to_string()); set_step.set(2); }
                            >"South African (CAPS)"</button>
                            <button
                                style=fw_btn_style
                                on:click=move |_| { set_framework.set("common_core".to_string()); set_step.set(2); }
                            >"US Common Core"</button>
                            <button
                                style=fw_btn_style
                                on:click=move |_| { set_framework.set("ib".to_string()); set_step.set(2); }
                            >"International Baccalaureate"</button>
                            <button
                                style=fw_btn_style
                                on:click=move |_| { set_framework.set("self_directed".to_string()); set_step.set(2); }
                            >"Self-directed"</button>
                        </div>
                    </div>
                }.into_any()

            // Step 2: Grade / level selection
            } else if step.get() == 2 {
                let fw = framework.get();
                let coming_soon = fw != "caps";
                view! {
                    <div class="onboarding-step" style="animation: card-in 0.4s ease-out">
                        <h1 style="font-size: 1.5rem; margin-bottom: 0.5rem; color: var(--primary)">
                            "What level are you at?"
                        </h1>

                        {if coming_soon {
                            view! {
                                <p style="color: var(--accent, var(--primary)); background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.85rem">
                                    "Coming soon \u{2014} curriculum data loading. You can still explore all content."
                                </p>
                            }.into_any()
                        } else {
                            view! { <span></span> }.into_any()
                        }}

                        // Quick path
                        <div style="display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap; margin-bottom: 1rem">
                            <button
                                style="padding: 0.75rem 1.25rem; background: var(--surface); border: 2px solid var(--primary); border-radius: 12px; color: var(--primary); font-size: 0.9rem; font-weight: 600; cursor: pointer; font-family: inherit"
                                on:click=move |_| { set_grade.set(16); finish_onboarding(); }
                            >"Learn a trade"</button>
                        </div>

                        // School level bands
                        <div style="display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap; margin-bottom: 1rem">
                            {[
                                ("Elementary (Gr 1-6)", 6_u8),
                                ("Middle (Gr 7-9)", 9_u8),
                                ("High School (Gr 10-12)", 12_u8),
                            ].into_iter().map(|(label, g)| {
                                view! {
                                    <button
                                        style="padding: 1.25rem 1.5rem; background: var(--surface); border: 2px solid var(--border); border-radius: 16px; color: var(--text); font-size: 1rem; font-weight: 600; cursor: pointer; min-width: 140px; font-family: inherit; transition: border-color 0.15s"
                                        on:click=move |_| {
                                            set_grade.set(g);
                                            finish_onboarding();
                                        }
                                    >
                                        {label}
                                    </button>
                                }
                            }).collect::<Vec<_>>()}
                        </div>

                        // Specific grade picker
                        <details style="margin-top: 0.5rem; text-align: center">
                            <summary style="font-size: 0.85rem; color: var(--text-tertiary); cursor: pointer; padding: 0.5rem">"Pick a specific grade"</summary>
                            <div style="display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; margin-top: 0.75rem">
                                {(1_u8..=12).map(|g| {
                                    view! {
                                        <button
                                            style="padding: 0.75rem 1rem; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; color: var(--text-secondary); font-size: 0.9rem; font-weight: 500; cursor: pointer; min-width: 60px; font-family: inherit"
                                            on:click=move |_| {
                                                set_grade.set(g);
                                                finish_onboarding();
                                            }
                                        >
                                            "Gr "{g}
                                        </button>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                            <p style="font-size: 0.75rem; color: var(--text-tertiary); margin-top: 0.75rem">
                                "No shame in going back to strengthen foundations. The strongest trees have the deepest roots."
                            </p>
                        </details>

                        // Beyond school
                        <details style="margin-top: 0.5rem; text-align: center">
                            <summary style="font-size: 0.85rem; color: var(--text-tertiary); cursor: pointer; padding: 0.5rem">"Beyond school?"</summary>
                            <div style="display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap; margin-top: 0.75rem">
                                <button
                                    style="padding: 0.75rem 1.25rem; background: var(--surface); border: 1px solid var(--primary); border-radius: 12px; color: var(--primary); font-size: 0.9rem; font-weight: 600; cursor: pointer; font-family: inherit"
                                    on:click=move |_| { set_grade.set(13); finish_onboarding(); }
                                >"University"</button>
                                <button
                                    style="padding: 0.75rem 1.25rem; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; color: var(--text-secondary); font-size: 0.9rem; font-weight: 500; cursor: pointer; font-family: inherit"
                                    on:click=move |_| { set_grade.set(16); finish_onboarding(); }
                                >"Working adult"</button>
                                <button
                                    style="padding: 0.75rem 1.25rem; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; color: var(--text-secondary); font-size: 0.9rem; font-weight: 500; cursor: pointer; font-family: inherit"
                                    on:click=move |_| { set_grade.set(16); finish_onboarding(); }
                                >"Lifelong learner"</button>
                            </div>
                            <p style="font-size: 0.75rem; color: var(--text-tertiary); margin-top: 0.75rem">
                                "Cybersecurity, Computer Science, Philosophy, Financial Literacy, and more."
                            </p>
                        </details>
                    </div>
                }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }}

            <p style="margin-top: 3rem; font-size: 0.8rem; color: var(--text-tertiary)">
                "Your data stays on your device. You own it."
            </p>
        </div>
    }
}
