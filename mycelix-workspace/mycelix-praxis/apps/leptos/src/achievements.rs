// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Achievement milestones — meaningful learning accomplishments.
//!
//! Not gamification for its own sake. Each achievement marks a genuine
//! learning milestone with educational significance.

use leptos::prelude::*;

use crate::curriculum::{curriculum_graph, use_progress, ProgressStatus};
use crate::study_tracker::use_tracker;

/// A learning milestone.
struct Achievement {
    id: &'static str,
    title: &'static str,
    description: &'static str,
    icon: &'static str,
    check: fn(usize, usize, u32, f32) -> bool, // (mastered, studying, streak, hours) -> earned
}

const ACHIEVEMENTS: &[Achievement] = &[
    Achievement {
        id: "first_seed",
        title: "First Seed",
        description: "Started studying your first topic",
        icon: "\u{1F331}",
        check: |_, studying, _, _| studying >= 1,
    },
    Achievement {
        id: "first_root",
        title: "First Root",
        description: "Mastered your first topic",
        icon: "\u{1F333}",
        check: |mastered, _, _, _| mastered >= 1,
    },
    Achievement {
        id: "five_roots",
        title: "Growing Garden",
        description: "Mastered 5 topics",
        icon: "\u{1F332}",
        check: |mastered, _, _, _| mastered >= 5,
    },
    Achievement {
        id: "ten_roots",
        title: "Dense Forest",
        description: "Mastered 10 topics",
        icon: "\u{1F3D5}",
        check: |mastered, _, _, _| mastered >= 10,
    },
    Achievement {
        id: "twenty_roots",
        title: "Ancient Grove",
        description: "Mastered 20 topics — deep knowledge",
        icon: "\u{1F30D}",
        check: |mastered, _, _, _| mastered >= 20,
    },
    Achievement {
        id: "streak_3",
        title: "Consistent",
        description: "3-day study streak",
        icon: "\u{1F525}",
        check: |_, _, streak, _| streak >= 3,
    },
    Achievement {
        id: "streak_7",
        title: "Weekly Rhythm",
        description: "7-day study streak — a habit is forming",
        icon: "\u{2B50}",
        check: |_, _, streak, _| streak >= 7,
    },
    Achievement {
        id: "streak_14",
        title: "Fortnight Strong",
        description: "14-day streak — discipline is a superpower",
        icon: "\u{1F31F}",
        check: |_, _, streak, _| streak >= 14,
    },
    Achievement {
        id: "streak_30",
        title: "Monthly Dedication",
        description: "30-day streak — you're unstoppable",
        icon: "\u{1F3C6}",
        check: |_, _, streak, _| streak >= 30,
    },
    Achievement {
        id: "hour_5",
        title: "Five Hours Deep",
        description: "5 hours of focused study time",
        icon: "\u{23F0}",
        check: |_, _, _, hours| hours >= 5.0,
    },
    Achievement {
        id: "hour_25",
        title: "Serious Scholar",
        description: "25 hours — that's real investment",
        icon: "\u{1F4DA}",
        check: |_, _, _, hours| hours >= 25.0,
    },
    Achievement {
        id: "hour_100",
        title: "Centurion",
        description: "100 hours of study — mastery takes time",
        icon: "\u{1F451}",
        check: |_, _, _, hours| hours >= 100.0,
    },
    Achievement {
        id: "explorer_10",
        title: "Curious Mind",
        description: "Explored 10 different topics",
        icon: "\u{1F50D}",
        check: |_, studying, _, _| studying >= 10,
    },
];

/// Achievement badges component for the dashboard.
#[component]
pub fn AchievementBadges() -> impl IntoView {
    let progress = use_progress();
    let tracker = use_tracker();

    view! {
        {move || {
            let p = progress.get();
            let t = tracker.get();
            let mastered = p.mastered_count();
            let studying = p.studying_count();
            let streak = t.current_streak();
            let hours = t.hours_studied();

            let earned: Vec<&Achievement> = ACHIEVEMENTS
                .iter()
                .filter(|a| (a.check)(mastered, studying, streak, hours))
                .collect();

            let total = ACHIEVEMENTS.len();
            let earned_count = earned.len();

            if earned.is_empty() {
                return view! {
                    <div class="achievements-empty">
                        <p style="font-size: 0.85rem; color: var(--text-secondary)">"Start studying to earn your first milestone"</p>
                    </div>
                }.into_any();
            }

            view! {
                <div class="achievements-section">
                    <h3 style="font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem">
                        "Milestones "{earned_count}"/"{total}
                    </h3>
                    <div class="achievements-grid">
                        {earned.iter().map(|a| {
                            view! {
                                <div class="achievement-badge" title=a.description>
                                    <span class="achievement-icon">{a.icon}</span>
                                    <span class="achievement-title">{a.title}</span>
                                </div>
                            }
                        }).collect::<Vec<_>>()}
                    </div>
                </div>
            }.into_any()
        }}
    }
}
