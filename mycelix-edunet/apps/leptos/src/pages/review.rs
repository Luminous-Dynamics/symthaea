// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Spaced Repetition Review page.
//!
//! Implements a flashcard review flow using the SM-2 quality scale (0-5).
//! Integrated with the adaptivity engine: card content is adapted based on
//! cognitive state (text complexity, modality indicators), suggestions slide
//! in when the student struggles, and metacognitive prompts appear between
//! cards.

use leptos::prelude::*;

use crate::adaptivity_provider::use_adaptivity;
use crate::cognitive_adaptivity::*;
use crate::components::suggestion_overlay::SuggestionOverlay;

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
    mastery_permille: u16,
}

const MOCK_CARDS: &[MockFlashcard] = &[
    MockFlashcard {
        front: "4 \u{00d7} 7 = ?",
        back: "28",
        tags: "Multiplication",
        mastery_permille: 700,
    },
    MockFlashcard {
        front: "What is 1/4 of a pizza?",
        back: "One slice if cut into 4 equal pieces",
        tags: "Fractions",
        mastery_permille: 400,
    },
    MockFlashcard {
        front: "Round 67 to the nearest ten",
        back: "70",
        tags: "Rounding",
        mastery_permille: 550,
    },
    MockFlashcard {
        front: "Sara has 3 bags with 5 apples each. How many apples?",
        back: "15 apples (3 \u{00d7} 5 = 15)",
        tags: "Word Problems",
        mastery_permille: 350,
    },
    MockFlashcard {
        front: "What is the area of a rectangle that is 4 units wide and 3 units tall?",
        back: "12 square units (4 \u{00d7} 3 = 12)",
        tags: "Geometry",
        mastery_permille: 250,
    },
];

/// Kid-friendly rating options: emoji, label, and mapped SM-2 quality value.
struct KidRating {
    emoji: &'static str,
    label: &'static str,
    quality: u8,
    css_class: &'static str,
}

const KID_RATINGS: &[KidRating] = &[
    KidRating { emoji: "\u{1f61f}", label: "I don't know this yet", quality: 1, css_class: "kid-rate-red" },
    KidRating { emoji: "\u{1f914}", label: "I'm still learning", quality: 2, css_class: "kid-rate-orange" },
    KidRating { emoji: "\u{1f60a}", label: "I got it!", quality: 4, css_class: "kid-rate-green" },
    KidRating { emoji: "\u{1f31f}", label: "Too easy!", quality: 5, css_class: "kid-rate-gold" },
];

// ---------------------------------------------------------------------------
// Helper: apply text complexity to card content
// ---------------------------------------------------------------------------

fn adapt_card_text(text: &str, adaptation: &ContentAdaptation, sovereignty: &SovereigntyLevel) -> String {
    let rewrite = suggest_rewrite(sovereignty, text, &adaptation.text_complexity, 5);
    match rewrite {
        RewriteResult::Applied { rewritten, .. } => rewritten,
        RewriteResult::Offered { rewritten, .. } => {
            // In Guide mode, show the rewritten version (student can toggle back)
            rewritten
        }
        RewriteResult::Available { original } => original,
    }
}

/// Modality indicator text for the current adaptation.
fn modality_indicator(modality: &Modality) -> &'static str {
    match modality {
        Modality::Text => "Reading mode",
        Modality::Visual => "Visual mode",
        Modality::Auditory => "Listen mode",
        Modality::Kinesthetic => "Hands-on mode",
        Modality::MultiModal => "",
    }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

#[component]
pub fn ReviewPage() -> impl IntoView {
    let adaptivity = use_adaptivity();

    // Extract signals for closures (signals are Copy)
    let adaptation_sig = adaptivity.adaptation;
    let sovereignty_sig = adaptivity.sovereignty;
    let adaptivity_for_timeout = adaptivity.clone();
    let adaptivity_for_rate = adaptivity.clone();
    // State signals
    let (state, set_state) = signal(ReviewState::Loading);
    let (ratings, set_ratings) = signal(Vec::<u8>::new());
    let (_start_time, set_start_time) = signal(0.0_f64);
    let (card_start_time, set_card_start_time) = signal(0.0_f64);
    let (_total_time_secs, set_total_time_secs) = signal(0.0_f64);

    // Simulate loading -> show first card (or NoDueCards if empty)
    set_timeout(
        {
            let adaptivity = adaptivity_for_timeout;
            move || {
                if MOCK_CARDS.is_empty() {
                    set_state.set(ReviewState::NoDueCards);
                } else {
                    set_start_time.set(js_sys::Date::now());
                    set_card_start_time.set(js_sys::Date::now());
                    // Tell adaptivity about the first card
                    let card = &MOCK_CARDS[0];
                    adaptivity.set_skill(card.tags, card.mastery_permille);
                    set_state.set(ReviewState::ShowingFront { card_index: 0 });
                }
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

    // Store adaptivity ctx for rate closure (StoredValue is Copy)
    let rate_ctx = StoredValue::new(adaptivity_for_rate);

    // Rate and advance to next card
    let rate = move |quality: u8| {
        if let ReviewState::ShowingBack { card_index } = state.get() {
            // Record the rating
            set_ratings.update(|r| r.push(quality));

            // Tell adaptivity engine about the attempt
            let correct = quality >= 3;
            rate_ctx.with_value(|ctx| ctx.record_attempt(correct));

            // Accumulate time for this card
            let elapsed = (js_sys::Date::now() - card_start_time.get()) / 1000.0;
            set_total_time_secs.update(|t| *t += elapsed);

            let next_index = card_index + 1;
            if next_index < MOCK_CARDS.len() {
                set_card_start_time.set(js_sys::Date::now());
                // Tell adaptivity about the new card
                let card = &MOCK_CARDS[next_index];
                rate_ctx.with_value(|ctx| ctx.set_skill(card.tags, card.mastery_permille));
                set_state.set(ReviewState::ShowingFront { card_index: next_index });
            } else {
                // Session complete
                let r = ratings.get();
                let correct = r.iter().filter(|&&q| q >= 3).count();
                let correct = if quality >= 3 { correct.max(1) } else { correct };
                let reviewed = r.len();
                set_state.set(ReviewState::SessionComplete { reviewed, correct });
            }
        }
    };

    view! {
        <div class="review-page">
            // Suggestion overlay -- slides in from bottom, never blocks
            <SuggestionOverlay />

            // Difficulty/modality indicator bar
            {move || {
                let adapt = adaptation_sig.get();
                let mod_text = modality_indicator(&adapt.modality);
                let diff = adapt.difficulty_delta;
                let diff_label = if diff > 0.1 { "Harder" }
                    else if diff < -0.1 { "Easier" }
                    else { "" };

                if mod_text.is_empty() && diff_label.is_empty() {
                    view! { <div class="adaptation-indicator hidden"></div> }.into_any()
                } else {
                    view! {
                        <div class="adaptation-indicator visible">
                            {if !mod_text.is_empty() {
                                view! { <span class="modality-badge">{mod_text}</span> }.into_any()
                            } else {
                                view! { <span></span> }.into_any()
                            }}
                            {if !diff_label.is_empty() {
                                view! { <span class="difficulty-badge">{diff_label}</span> }.into_any()
                            } else {
                                view! { <span></span> }.into_any()
                            }}
                        </div>
                    }.into_any()
                }
            }}

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

                        // Apply text complexity adaptation to card content
                        let adapt = adaptation_sig.get();
                        let sov = sovereignty_sig.get();
                        let adapted_front = adapt_card_text(card.front, &adapt, &sov);

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
                                            <p>{adapted_front}</p>
                                        </div>
                                        // Show rewrite explanation in Guardian mode
                                        {move || {
                                            let adapt = adaptation_sig.get();
                                            if adapt.text_complexity != TextComplexity::Standard {
                                                let sov = sovereignty_sig.get();
                                                if sov.mode() == InteractionMode::Guardian {
                                                    return view! {
                                                        <p class="rewrite-note">
                                                            <small>"(Simplified to help you focus on the math)"</small>
                                                        </p>
                                                    }.into_any();
                                                }
                                            }
                                            view! { <span></span> }.into_any()
                                        }}
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
                                    <p class="rating-prompt">"How did you do?"</p>
                                    <div class="kid-rating-grid">
                                        {KID_RATINGS.iter().map(|r| {
                                            let q = r.quality;
                                            let class = format!("kid-rate-btn {}", r.css_class);
                                            let emoji = r.emoji;
                                            let label = r.label;
                                            view! {
                                                <button
                                                    class=class
                                                    on:click=move |_| rate(q)
                                                >
                                                    <span class="kid-rate-emoji">{emoji}</span>
                                                    <span class="kid-rate-label">{label}</span>
                                                </button>
                                            }
                                        }).collect_view()}
                                    </div>
                                </div>
                            </div>
                        }.into_any()
                    }

                    ReviewState::SessionComplete { reviewed, correct } => {
                        let xp = correct * 10 + (reviewed - correct) * 2;
                        let stars_filled = "\u{2b50}".repeat(correct.min(reviewed));
                        let stars_empty = "\u{2606}".repeat(reviewed.saturating_sub(correct));

                        // Show sovereignty growth summary
                        let sov = sovereignty_sig.get();
                        let mode_label = match sov.mode() {
                            InteractionMode::Guardian => "Helper Mode",
                            InteractionMode::Guide => "Guide Mode",
                            InteractionMode::Mirror => "Mirror Mode",
                            InteractionMode::Autonomous => "Independent Mode",
                        };

                        view! {
                            <div class="review-complete kid-complete">
                                <div class="kid-celebration">"\u{1f389}"</div>
                                <h2>"Great job!"</h2>
                                <p class="kid-summary-text">
                                    "You reviewed " {reviewed} " cards"
                                </p>
                                <div class="kid-stars">
                                    <span class="kid-stars-filled">{stars_filled}</span>
                                    <span class="kid-stars-empty">{stars_empty}</span>
                                </div>
                                <p class="kid-stars-label">
                                    {correct} " out of " {reviewed} " correct"
                                </p>
                                <div class="kid-xp-earned">
                                    <span class="kid-xp-badge">{format!("+{} XP earned!", xp)}</span>
                                </div>
                                // Sovereignty status
                                <div class="sovereignty-summary">
                                    <span class="sovereignty-badge">{mode_label}</span>
                                    <span class="sovereignty-level">
                                        {format!("Independence: {}/1000", sov.level)}
                                    </span>
                                </div>
                                <div class="kid-complete-actions">
                                    <a href="/review" class="btn-primary kid-btn">"Keep Going"</a>
                                    <a href="/" class="btn-secondary kid-btn">"Done for now"</a>
                                </div>
                            </div>
                        }.into_any()
                    }
                }
            }}
        </div>
    }
}
