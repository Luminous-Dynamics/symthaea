// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Spaced Repetition Review page.
//!
//! Implements a flashcard review flow using the SM-2 quality scale (0-5).
//! Uses mock data until the Holochain transport is wired.

use leptos::prelude::*;

// ---------------------------------------------------------------------------
// Review state machine
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum ReviewState {
    Loading,
    NoDueCards,
    ShowingFront { card_index: usize },
    ShowingBack { card_index: usize },
    SessionComplete { reviewed: usize, correct: usize },
}

// ---------------------------------------------------------------------------
// Mock flashcard data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct MockFlashcard {
    front: &'static str,
    back: &'static str,
    tags: &'static str,
}

const MOCK_CARDS: &[MockFlashcard] = &[
    MockFlashcard {
        front: "What is the SM-2 algorithm?",
        back: "A spaced repetition algorithm by Piotr Wozniak that calculates \
               optimal review intervals using an ease factor (EF) adjusted by \
               recall quality ratings (0-5).",
        tags: "Learning Science",
    },
    MockFlashcard {
        front: "What is the spacing effect?",
        back: "The phenomenon where learning is more effective when study sessions \
               are distributed over time rather than massed together. Can improve \
               retention by 200-400%.",
        tags: "Cognitive Psychology",
    },
    MockFlashcard {
        front: "In Holochain, what is a zome?",
        back: "A zome (short for chromosome) is a module within a DNA that defines \
               entry types, link types, and validation rules. Each zome compiles to \
               WASM and runs in the Holochain conductor.",
        tags: "Holochain",
    },
    MockFlashcard {
        front: "What does BKT stand for in adaptive learning?",
        back: "Bayesian Knowledge Tracing - a probabilistic model that estimates a \
               learner's knowledge state using prior knowledge, learning rate, guess \
               probability, and slip probability.",
        tags: "Adaptive Learning",
    },
    MockFlashcard {
        front: "What is the Zone of Proximal Development (ZPD)?",
        back: "Vygotsky's concept describing the gap between what a learner can do \
               independently and what they can achieve with guidance. Optimal learning \
               occurs within this zone.",
        tags: "Education Theory",
    },
];

/// Quality rating labels for the SM-2 scale (0-5).
const QUALITY_LABELS: &[(&str, &str)] = &[
    ("0", "Blackout"),
    ("1", "Wrong"),
    ("2", "Wrong (easy)"),
    ("3", "Hard"),
    ("4", "Hesitant"),
    ("5", "Perfect"),
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

#[component]
pub fn ReviewPage() -> impl IntoView {
    // State signals
    let (state, set_state) = signal(ReviewState::Loading);
    let (ratings, set_ratings) = signal(Vec::<u8>::new());
    let (start_time, set_start_time) = signal(0.0_f64);
    let (card_start_time, set_card_start_time) = signal(0.0_f64);
    let (total_time_secs, set_total_time_secs) = signal(0.0_f64);

    // Simulate loading -> show first card (or NoDueCards if empty)
    // In a real implementation this would be a Resource fetching due cards.
    set_timeout(
        move || {
            if MOCK_CARDS.is_empty() {
                set_state.set(ReviewState::NoDueCards);
            } else {
                set_start_time.set(js_sys::Date::now());
                set_card_start_time.set(js_sys::Date::now());
                set_state.set(ReviewState::ShowingFront { card_index: 0 });
            }
        },
        std::time::Duration::from_millis(300),
    );

    // Reveal back of current card
    let reveal = move |_| {
        if let ReviewState::ShowingFront { card_index } = state.get() {
            set_state.set(ReviewState::ShowingBack { card_index });
        }
    };

    // Rate and advance to next card
    let rate = move |quality: u8| {
        if let ReviewState::ShowingBack { card_index } = state.get() {
            // Record the rating
            set_ratings.update(|r| r.push(quality));

            // Accumulate time for this card
            let elapsed = (js_sys::Date::now() - card_start_time.get()) / 1000.0;
            set_total_time_secs.update(|t| *t += elapsed);

            let next_index = card_index + 1;
            if next_index < MOCK_CARDS.len() {
                set_card_start_time.set(js_sys::Date::now());
                set_state.set(ReviewState::ShowingFront { card_index: next_index });
            } else {
                // Session complete
                let r = ratings.get();
                let correct = r.iter().filter(|&&q| q >= 3).count();
                // Include the rating we just pushed (it may not be in the snapshot yet)
                let correct = if quality >= 3 { correct.max(1) } else { correct };
                let reviewed = r.len();
                set_state.set(ReviewState::SessionComplete { reviewed, correct });
            }
        }
    };

    view! {
        <div class="review-page">
            {move || {
                let current = state.get();
                match current {
                    ReviewState::Loading => view! {
                        <div class="review-loading">
                            <div class="review-spinner"></div>
                            <p class="review-loading-text">"Loading due cards..."</p>
                        </div>
                    }.into_any(),

                    ReviewState::NoDueCards => view! {
                        <div class="review-empty">
                            <div class="review-empty-icon">"--"</div>
                            <h2>"All caught up!"</h2>
                            <p>"No cards are due for review right now."</p>
                            <div class="review-stats-mini">
                                <div class="stat-item">
                                    <span class="stat-value">"0"</span>
                                    <span class="stat-label">"Due today"</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-value">"5"</span>
                                    <span class="stat-label">"Total cards"</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-value">"3d"</span>
                                    <span class="stat-label">"Streak"</span>
                                </div>
                            </div>
                        </div>
                    }.into_any(),

                    ReviewState::ShowingFront { card_index } => {
                        let card = &MOCK_CARDS[card_index];
                        let total = MOCK_CARDS.len();
                        let progress_pct = ((card_index as f64 / total as f64) * 100.0) as u32;
                        view! {
                            <div class="review-session">
                                <div class="review-progress">
                                    <span class="progress-text">
                                        {format!("Card {} of {}", card_index + 1, total)}
                                    </span>
                                    <div class="progress-bar">
                                        <div class="progress-fill"
                                            style:width=format!("{}%", progress_pct)>
                                        </div>
                                    </div>
                                </div>
                                <div class="flashcard">
                                    <div class="flashcard-inner flashcard-front">
                                        <span class="card-tag">{card.tags}</span>
                                        <div class="card-content">
                                            <p>{card.front}</p>
                                        </div>
                                        <button class="reveal-btn" on:click=reveal>
                                            "Show Answer"
                                        </button>
                                    </div>
                                </div>
                            </div>
                        }.into_any()
                    }

                    ReviewState::ShowingBack { card_index } => {
                        let card = &MOCK_CARDS[card_index];
                        let total = MOCK_CARDS.len();
                        let progress_pct = (((card_index + 1) as f64 / total as f64) * 100.0) as u32;
                        view! {
                            <div class="review-session">
                                <div class="review-progress">
                                    <span class="progress-text">
                                        {format!("Card {} of {}", card_index + 1, total)}
                                    </span>
                                    <div class="progress-bar">
                                        <div class="progress-fill"
                                            style:width=format!("{}%", progress_pct)>
                                        </div>
                                    </div>
                                </div>
                                <div class="flashcard">
                                    <div class="flashcard-inner flashcard-back">
                                        <span class="card-tag">{card.tags}</span>
                                        <div class="card-content card-content-split">
                                            <div class="card-question">
                                                <span class="label">"Q: "</span>
                                                {card.front}
                                            </div>
                                            <hr class="card-divider" />
                                            <div class="card-answer">
                                                <span class="label">"A: "</span>
                                                {card.back}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="rating-buttons">
                                    <p class="rating-prompt">"How well did you recall?"</p>
                                    <div class="rating-grid">
                                        {QUALITY_LABELS.iter().enumerate().map(|(i, (num, label))| {
                                            let q = i as u8;
                                            let class = format!("rate-btn rate-{}", i);
                                            view! {
                                                <button
                                                    class=class
                                                    on:click=move |_| rate(q)
                                                >
                                                    <span class="rate-num">{*num}</span>
                                                    <span class="rate-label">{*label}</span>
                                                </button>
                                            }
                                        }).collect_view()}
                                    </div>
                                </div>
                            </div>
                        }.into_any()
                    }

                    ReviewState::SessionComplete { reviewed, correct } => {
                        let accuracy = if reviewed > 0 {
                            ((correct as f64 / reviewed as f64) * 100.0) as u32
                        } else {
                            0
                        };
                        let avg_time = if reviewed > 0 {
                            total_time_secs.get() / reviewed as f64
                        } else {
                            0.0
                        };
                        let xp = correct * 10 + (reviewed - correct) * 2;
                        view! {
                            <div class="review-complete">
                                <h2>"Session Complete"</h2>
                                <div class="session-summary">
                                    <div class="summary-stat">
                                        <span class="summary-value">{reviewed}</span>
                                        <span class="summary-label">"Cards Reviewed"</span>
                                    </div>
                                    <div class="summary-stat">
                                        <span class="summary-value">{format!("{}%", accuracy)}</span>
                                        <span class="summary-label">"Accuracy"</span>
                                    </div>
                                    <div class="summary-stat">
                                        <span class="summary-value">{format!("{:.1}s", avg_time)}</span>
                                        <span class="summary-label">"Avg Time"</span>
                                    </div>
                                    <div class="summary-stat summary-xp">
                                        <span class="summary-value">{format!("+{}", xp)}</span>
                                        <span class="summary-label">"XP Earned"</span>
                                    </div>
                                </div>
                                <div class="session-ratings">
                                    <h3>"Rating Breakdown"</h3>
                                    <div class="rating-breakdown">
                                        {ratings.get().iter().enumerate().map(|(i, &q)| {
                                            let class = format!("breakdown-dot rate-bg-{}", q);
                                            view! {
                                                <span class=class title=format!("Card {}: quality {}", i + 1, q)>
                                                    {q.to_string()}
                                                </span>
                                            }
                                        }).collect_view()}
                                    </div>
                                </div>
                            </div>
                        }.into_any()
                    }
                }
            }}
        </div>
    }
}
