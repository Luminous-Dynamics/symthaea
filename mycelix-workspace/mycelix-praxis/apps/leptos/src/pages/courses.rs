// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

use crate::holochain::use_holochain;

// ---------------------------------------------------------------------------
// Course data types
// ---------------------------------------------------------------------------

/// A course as returned by the learning_zome or mock fallback.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Course {
    pub title: String,
    pub description: String,
    pub domain: String,
}

/// Static mock courses used when no conductor is connected.
fn mock_courses() -> Vec<Course> {
    vec![
        Course {
            title: "Rust Fundamentals".into(),
            description: "Learn systems programming with Rust".into(),
            domain: "Programming".into(),
        },
        Course {
            title: "Distributed Systems".into(),
            description: "P2P architecture and consensus".into(),
            domain: "Computer Science".into(),
        },
        Course {
            title: "Regenerative Agriculture".into(),
            description: "Soil science and permaculture".into(),
            domain: "Agriculture".into(),
        },
        Course {
            title: "Cooperative Economics".into(),
            description: "Mutual aid, commons governance, and solidarity economy".into(),
            domain: "Economics".into(),
        },
    ]
}

// ---------------------------------------------------------------------------
// Courses page
// ---------------------------------------------------------------------------

#[component]
pub fn CoursesPage() -> impl IntoView {
    let hc = use_holochain();

    // Fetch courses from the conductor, falling back to mock data.
    let courses = LocalResource::new(move || {
        let hc = hc.clone();
        async move {
            match hc.call_zome_default::<(), Vec<Course>>("learning", "list_courses", &()).await {
                Ok(c) => c,
                Err(_) => mock_courses(),
            }
        }
    });

    view! {
        <div class="courses-page">
            <div class="page-header">
                <h2>"Courses"</h2>
                <ConnectionStatusTag />
            </div>
            <Suspense fallback=move || view! { <CoursesLoading /> }>
                {move || {
                    courses.get().map(|data| {
                        let data: Vec<Course> = data.clone();
                        view! {
                            <div class="course-grid">
                                {data.into_iter().map(|course| {
                                    view! {
                                        <div class="course-card">
                                            <h3>{course.title}</h3>
                                            <p>{course.description}</p>
                                            <span class="domain-tag">{course.domain}</span>
                                        </div>
                                    }
                                }).collect_view()}
                            </div>
                        }
                    })
                }}
            </Suspense>
        </div>
    }
}

/// Loading skeleton for the courses grid.
#[component]
fn CoursesLoading() -> impl IntoView {
    view! {
        <div class="course-grid">
            <div class="course-card skeleton"></div>
            <div class="course-card skeleton"></div>
            <div class="course-card skeleton"></div>
        </div>
    }
}

/// Small inline tag showing whether data is live or mocked.
#[component]
fn ConnectionStatusTag() -> impl IntoView {
    let hc = use_holochain();
    view! {
        <span class=move || {
            let status = hc.status.get();
            format!("connection-tag {}", status.css_class())
        }>
            {move || hc.status.get().label()}
        </span>
    }
}
