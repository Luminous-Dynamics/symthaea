// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

/// Mock course data until HolochainClient is wired.
struct MockCourse {
    title: &'static str,
    description: &'static str,
    domain: &'static str,
}

const MOCK_COURSES: &[MockCourse] = &[
    MockCourse {
        title: "Rust Fundamentals",
        description: "Learn systems programming with Rust",
        domain: "Programming",
    },
    MockCourse {
        title: "Distributed Systems",
        description: "P2P architecture and consensus",
        domain: "Computer Science",
    },
    MockCourse {
        title: "Regenerative Agriculture",
        description: "Soil science and permaculture",
        domain: "Agriculture",
    },
    MockCourse {
        title: "Cooperative Economics",
        description: "Mutual aid, commons governance, and solidarity economy",
        domain: "Economics",
    },
];

#[component]
pub fn CoursesPage() -> impl IntoView {
    // TODO: Wire to HolochainClient once mycelix-leptos-client is ready.
    // The React version uses useCourses() which calls client.learning.list_courses().
    // The Leptos equivalent will use a Resource + Signal pattern:
    //   let courses = Resource::new(|| (), |_| async { fetch_courses().await });

    view! {
        <div class="courses-page">
            <h2>"Courses"</h2>
            <div class="course-grid">
                {MOCK_COURSES.iter().map(|course| {
                    view! {
                        <div class="course-card">
                            <h3>{course.title}</h3>
                            <p>{course.description}</p>
                            <span class="domain-tag">{course.domain}</span>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
