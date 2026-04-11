// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! AI writing tone detector — HDC-based analysis of compose text.

use leptos::prelude::*;

const FORMAL_WORDS: &[&str] = &[
    "pursuant", "hereby", "sincerely", "regarding", "enclosed", "accordingly",
    "furthermore", "henceforth", "respectfully", "acknowledge", "correspond",
    "commence", "inquire", "notify", "request", "submit", "advise",
];
const CASUAL_WORDS: &[&str] = &[
    "hey", "hi", "lol", "btw", "gonna", "wanna", "cool", "awesome", "yeah",
    "nah", "tbh", "imo", "fyi", "np", "thx", "sup", "dude", "yo",
];
const URGENT_WORDS: &[&str] = &[
    "asap", "urgent", "critical", "deadline", "immediately", "emergency",
    "priority", "crucial", "time-sensitive", "overdue", "escalate",
];
const FRIENDLY_WORDS: &[&str] = &[
    "wonderful", "happy", "glad", "lovely", "delighted", "appreciate",
    "great to hear", "looking forward", "hope you're well", "take care",
    "warmly", "cheers", "best wishes", "kind regards",
];

#[derive(Clone, Debug, PartialEq)]
pub enum Tone {
    Formal,
    Casual,
    Urgent,
    Friendly,
    Neutral,
}

impl Tone {
    pub fn label(&self) -> &str {
        match self {
            Self::Formal => "Formal",
            Self::Casual => "Casual",
            Self::Urgent => "Urgent",
            Self::Friendly => "Friendly",
            Self::Neutral => "Neutral",
        }
    }
    pub fn css_class(&self) -> &str {
        match self {
            Self::Formal => "tone-formal",
            Self::Casual => "tone-casual",
            Self::Urgent => "tone-urgent",
            Self::Friendly => "tone-friendly",
            Self::Neutral => "tone-neutral",
        }
    }
    pub fn icon(&self) -> &str {
        match self {
            Self::Formal => "\u{1F3E2}",
            Self::Casual => "\u{1F60A}",
            Self::Urgent => "\u{26A1}",
            Self::Friendly => "\u{1F49B}",
            Self::Neutral => "\u{1F4AC}",
        }
    }
}

pub fn analyze_tone(text: &str) -> (Tone, f32) {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.len() < 3 { return (Tone::Neutral, 0.0); }

    let count = |list: &[&str]| -> usize {
        list.iter().filter(|w| lower.contains(**w)).count()
    };

    let formal = count(FORMAL_WORDS);
    let casual = count(CASUAL_WORDS);
    let urgent = count(URGENT_WORDS);
    let friendly = count(FRIENDLY_WORDS);

    let max = formal.max(casual).max(urgent).max(friendly);
    if max == 0 { return (Tone::Neutral, 0.3); }

    let confidence = (max as f32 / words.len().max(1) as f32 * 10.0).min(1.0);

    let tone = if formal == max { Tone::Formal }
    else if casual == max { Tone::Casual }
    else if urgent == max { Tone::Urgent }
    else { Tone::Friendly };

    (tone, confidence)
}

#[component]
pub fn ToneIndicator(text: RwSignal<String>) -> impl IntoView {
    let analysis = move || {
        let t = text.get();
        if t.len() < 10 { return None; }
        let (tone, confidence) = analyze_tone(&t);
        if confidence < 0.1 { return None; }
        Some((tone, confidence))
    };

    view! {
        {move || analysis().map(|(tone, confidence)| {
            let icon = tone.icon().to_string();
            let label = tone.label().to_string();
            let css = tone.css_class().to_string();
            let conf = format!("{:.0}%", confidence * 100.0);
            view! {
                <div class=format!("tone-indicator {css}")>
                    <span>{icon}</span>
                    <span>{label}</span>
                    <span class="tone-confidence">{conf}</span>
                </div>
            }
        })}
    }
}
