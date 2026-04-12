// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Logical Fallacy Detector — interactive philosophy/critical thinking game.
//!
//! Presents arguments and asks the student to identify which logical fallacy
//! (if any) is present. Teaches informal logic and argumentation.

use leptos::prelude::*;

struct FallacyQuestion {
    argument: &'static str,
    fallacy: &'static str,
    explanation: &'static str,
    options: [&'static str; 4],
    correct: usize,
}

const QUESTIONS: &[FallacyQuestion] = &[
    FallacyQuestion {
        argument: "\"You can't trust Dr. Patel's research on climate change — she drives a petrol car.\"",
        fallacy: "Ad Hominem",
        explanation: "Attacking the person rather than their argument. The quality of research doesn't depend on the researcher's personal choices.",
        options: ["Ad Hominem", "Straw Man", "Appeal to Authority", "False Dilemma"],
        correct: 0,
    },
    FallacyQuestion {
        argument: "\"Either we ban all social media or we accept that children will be harmed.\"",
        fallacy: "False Dilemma",
        explanation: "Presenting only two extreme options when many moderate alternatives exist (age restrictions, time limits, education).",
        options: ["Slippery Slope", "False Dilemma", "Red Herring", "Bandwagon"],
        correct: 1,
    },
    FallacyQuestion {
        argument: "\"If we allow students to use calculators in class, they'll forget how to do arithmetic, then they won't be able to handle finances, and society will collapse.\"",
        fallacy: "Slippery Slope",
        explanation: "Assuming one event will inevitably lead to a chain of extreme consequences without evidence for each step.",
        options: ["Hasty Generalization", "Ad Hominem", "Slippery Slope", "Circular Reasoning"],
        correct: 2,
    },
    FallacyQuestion {
        argument: "\"Everyone I know uses TikTok, so it must be a good platform for learning.\"",
        fallacy: "Bandwagon / Appeal to Popularity",
        explanation: "Just because something is popular doesn't mean it's good or effective. Popularity and quality are separate claims.",
        options: ["Appeal to Authority", "Bandwagon", "Straw Man", "False Cause"],
        correct: 1,
    },
    FallacyQuestion {
        argument: "\"My grade improved after I wore my lucky socks to the exam. The socks caused my improvement.\"",
        fallacy: "False Cause (Post Hoc)",
        explanation: "Assuming that because B followed A, A must have caused B. Correlation does not imply causation — studying likely caused the improvement.",
        options: ["False Cause", "Circular Reasoning", "Red Herring", "Appeal to Emotion"],
        correct: 0,
    },
    FallacyQuestion {
        argument: "\"We should fund the arts programme because otherwise the children will have nothing creative in their lives and become depressed.\"",
        fallacy: "Appeal to Emotion",
        explanation: "Using emotional manipulation instead of evidence. A better argument would present data on arts education outcomes.",
        options: ["Ad Hominem", "False Dilemma", "Hasty Generalization", "Appeal to Emotion"],
        correct: 3,
    },
];

#[component]
pub fn FallacyDetector(node_id: String) -> impl IntoView {
    let (current_q, set_current_q) = signal(0_usize);
    let (selected, set_selected) = signal::<Option<usize>>(None);
    let (score, set_score) = signal(0_u32);
    let (answered, set_answered) = signal(0_u32);

    let is_complete = Memo::new(move |_| current_q.get() >= QUESTIONS.len());

    view! {
        <div class="game-container">
            <h3 style="font-size: 1rem; margin-bottom: 0.75rem">"Logical Fallacy Detector"</h3>

            {move || if is_complete.get() {
                let s = score.get();
                let a = answered.get();
                let pct = if a > 0 { s * 100 / a } else { 0 };
                view! {
                    <div style="text-align: center; padding: 2rem 0">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem">
                            {if pct >= 80 { "\u{1F331}" } else if pct >= 50 { "\u{1F33F}" } else { "\u{1F331}" }}
                        </div>
                        <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem">
                            {s}"/"{a}" correct ("{pct}"%)"
                        </div>
                        <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 1rem">
                            {if pct >= 80 { "Excellent critical thinking! You can spot manipulation." }
                             else if pct >= 50 { "Good start — keep practicing to sharpen your reasoning." }
                             else { "The ability to detect fallacies grows with practice. Try again!" }}
                        </p>
                        <button
                            class="praxis-filter-btn active"
                            on:click=move |_| {
                                set_current_q.set(0);
                                set_score.set(0);
                                set_answered.set(0);
                                set_selected.set(None);
                            }
                        >"Try Again"</button>
                    </div>
                }.into_any()
            } else {
                let idx = current_q.get() % QUESTIONS.len();
                let q = &QUESTIONS[idx];
                let argument = q.argument;
                let explanation = q.explanation;
                let correct_idx = q.correct;
                view! {
                    <div style="font-size: 0.75rem; color: var(--text-tertiary); margin-bottom: 0.75rem">
                        "Question "{current_q.get() + 1}"/" {QUESTIONS.len()}
                        " | Score: "{score.get()}
                    </div>

                    <div style="padding: 1rem; background: var(--soil-rich); border-radius: 8px; margin-bottom: 1rem; font-size: 0.95rem; line-height: 1.6; font-style: italic">
                        {argument}
                    </div>

                    <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.75rem">"Which fallacy is this?"</p>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1rem">
                        {q.options.iter().enumerate().map(|(i, opt)| {
                            let opt_text = *opt;
                            let is_correct = i == correct_idx;
                            view! {
                                <button
                                    style=move || {
                                        let base = "padding: 0.6rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 500; cursor: pointer; font-family: inherit; transition: all 0.2s; border: 1px solid var(--border);";
                                        match selected.get() {
                                            None => format!("{} background: var(--surface); color: var(--text);", base),
                                            Some(sel) if sel == i && is_correct => format!("{} background: var(--soil-rich); border-color: var(--mastery-green); color: var(--mastery-green);", base),
                                            Some(sel) if sel == i => format!("{} background: var(--weathering); border-color: var(--text-tertiary); color: var(--text-secondary);", base),
                                            Some(_) if is_correct => format!("{} background: var(--soil-rich); border-color: var(--mastery-green); color: var(--mastery-green);", base),
                                            _ => format!("{} background: var(--surface); color: var(--text-tertiary); opacity: 0.6;", base),
                                        }
                                    }
                                    disabled=move || selected.get().is_some()
                                    on:click=move |_| {
                                        set_selected.set(Some(i));
                                        set_answered.update(|a| *a += 1);
                                        if is_correct {
                                            set_score.update(|s| *s += 1);
                                        }
                                    }
                                >
                                    {opt_text}
                                </button>
                            }
                        }).collect::<Vec<_>>()}
                    </div>

                    // Explanation (shown after selecting)
                    {move || selected.get().map(|_| {
                        view! {
                            <div style="padding: 0.75rem; background: var(--soil-rich); border-left: 3px solid var(--info); border-radius: 0 8px 8px 0; font-size: 0.8rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 0.75rem">
                                {explanation}
                            </div>
                            <button
                                class="praxis-filter-btn active"
                                style="font-size: 0.8rem"
                                on:click=move |_| {
                                    set_current_q.update(|q| *q += 1);
                                    set_selected.set(None);
                                }
                            >"Next"</button>
                        }
                    })}
                }.into_any()
            }}
        </div>
    }
}
