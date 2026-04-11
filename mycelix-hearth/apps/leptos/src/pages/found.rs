// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! The Founding Ceremony — creating a Hearth is a threshold crossing.
//!
//! Four passages:
//! 1. The Naming — give your hearth a name (ember-glow letter reveal)
//! 2. The Intention — what kind of family is this?
//! 3. The First Light — candle-lighting animation when created on-chain
//! 4. The Invitation — who will you bring to this fire?
//!
//! Campfire flames flicker in the background throughout.

use leptos::prelude::*;
use crate::components::{HearthFlame, FlameMode};

/// Which stage of the founding ceremony we're in.
#[derive(Clone, Copy, PartialEq)]
enum CeremonyStage {
    Naming,
    Intention,
    FirstLight,
    Invitation,
}

/// Hearth type — displayed as visual cards during The Intention.
#[derive(Clone, Copy, PartialEq)]
enum HearthKind {
    Chosen,
    Nuclear,
    Extended,
    Intentional,
    CoPod,
}

impl HearthKind {
    fn label(self) -> &'static str {
        match self {
            Self::Chosen => "Chosen Family",
            Self::Nuclear => "Nuclear Family",
            Self::Extended => "Extended Family",
            Self::Intentional => "Intentional Community",
            Self::CoPod => "Co-Living Pod",
        }
    }
    fn description(self) -> &'static str {
        match self {
            Self::Chosen => "The people you chose. Bonds of intention, not just blood.",
            Self::Nuclear => "Parents and children. The inner ring of care.",
            Self::Extended => "Grandparents, aunts, cousins. The full tree.",
            Self::Intentional => "A community united by shared purpose and values.",
            Self::CoPod => "Housemates, co-workers, creative collaborators.",
        }
    }
    fn icon(self) -> &'static str {
        match self {
            Self::Chosen => "\u{1F91D}",
            Self::Nuclear => "\u{1F3E0}",
            Self::Extended => "\u{1F333}",
            Self::Intentional => "\u{1F331}",
            Self::CoPod => "\u{2728}",
        }
    }
    fn zome_value(self) -> &'static str {
        match self {
            Self::Chosen => "Chosen",
            Self::Nuclear => "Nuclear",
            Self::Extended => "Extended",
            Self::Intentional => "Intentional",
            Self::CoPod => "CoPod",
        }
    }
}

const ALL_KINDS: [HearthKind; 5] = [
    HearthKind::Chosen,
    HearthKind::Nuclear,
    HearthKind::Extended,
    HearthKind::Intentional,
    HearthKind::CoPod,
];

#[component]
pub fn FoundingCeremony() -> impl IntoView {
    let (stage, set_stage) = signal(CeremonyStage::Naming);
    let (hearth_name, set_hearth_name) = signal(String::new());
    let (hearth_kind, set_hearth_kind) = signal(None::<HearthKind>);
    let (invite_did, set_invite_did) = signal(String::new());
    let (created, set_created) = signal(false);

    view! {
        <div class="founding-ceremony">
            // ── Campfire Flames Background ──
            <HearthFlame mode=FlameMode::Full />

            // ── Stage Content ──
            <div class="ceremony-content">
                // Stage 1: The Naming
                {move || {
                    if stage.get() != CeremonyStage::Naming { return view! { <div /> }.into_any(); }
                    view! {
                        <div class="ceremony-stage naming-stage">
                            <h1 class="ceremony-title">"The Naming"</h1>
                            <p class="ceremony-prompt">
                                "Every hearth begins with a name."
                                <br />
                                "What will you call this place of belonging?"
                            </p>
                            <div class="naming-input-wrap">
                                <input
                                    type="text"
                                    class="naming-input"
                                    placeholder="..."
                                    maxlength="64"
                                    autofocus=true
                                    on:input=move |ev| {
                                        set_hearth_name.set(event_target_value(&ev));
                                    }
                                    prop:value=hearth_name
                                />
                                <div class="naming-underline" />
                            </div>
                            {move || {
                                let name = hearth_name.get();
                                if name.len() >= 2 {
                                    view! {
                                        <button
                                            class="ceremony-btn"
                                            on:click=move |_| set_stage.set(CeremonyStage::Intention)
                                        >
                                            "continue"
                                        </button>
                                    }.into_any()
                                } else {
                                    view! { <div /> }.into_any()
                                }
                            }}
                        </div>
                    }.into_any()
                }}

                // Stage 2: The Intention
                {move || {
                    if stage.get() != CeremonyStage::Intention { return view! { <div /> }.into_any(); }
                    view! {
                        <div class="ceremony-stage intention-stage">
                            <h1 class="ceremony-title">"The Intention"</h1>
                            <p class="ceremony-prompt">
                                "What kind of family is "
                                <span class="hearth-name-echo">{hearth_name}</span>
                                "?"
                            </p>
                            <div class="intention-grid">
                                {ALL_KINDS.iter().map(|kind| {
                                    let k = *kind;
                                    let selected = move || hearth_kind.get() == Some(k);
                                    view! {
                                        <button
                                            class=move || if selected() { "intention-card selected" } else { "intention-card" }
                                            on:click=move |_| set_hearth_kind.set(Some(k))
                                        >
                                            <span class="intention-icon">{k.icon()}</span>
                                            <span class="intention-label">{k.label()}</span>
                                            <span class="intention-desc">{k.description()}</span>
                                        </button>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                            {move || {
                                if hearth_kind.get().is_some() {
                                    view! {
                                        <button
                                            class="ceremony-btn"
                                            on:click=move |_| {
                                                set_created.set(true);
                                                set_stage.set(CeremonyStage::FirstLight);
                                                // TODO: zome call to create_hearth
                                            }
                                        >
                                            "light the fire"
                                        </button>
                                    }.into_any()
                                } else {
                                    view! { <div /> }.into_any()
                                }
                            }}
                        </div>
                    }.into_any()
                }}

                // Stage 3: The First Light
                {move || {
                    if stage.get() != CeremonyStage::FirstLight { return view! { <div /> }.into_any(); }
                    view! {
                        <div class="ceremony-stage firstlight-stage">
                            <div class="candle-animation">
                                <div class="candle-wick" />
                                <div class="candle-flame-inner" />
                                <div class="candle-glow-expand" />
                            </div>
                            <h1 class="ceremony-title firstlight-title">
                                {hearth_name}
                            </h1>
                            <p class="ceremony-prompt firstlight-subtitle">
                                "has been lit"
                            </p>
                            <button
                                class="ceremony-btn"
                                on:click=move |_| set_stage.set(CeremonyStage::Invitation)
                            >
                                "enter"
                            </button>
                        </div>
                    }.into_any()
                }}

                // Stage 4: The Invitation
                {move || {
                    if stage.get() != CeremonyStage::Invitation { return view! { <div /> }.into_any(); }
                    view! {
                        <div class="ceremony-stage invitation-stage">
                            <h1 class="ceremony-title">"The Invitation"</h1>
                            <p class="ceremony-prompt">
                                "Who will you bring to this fire?"
                            </p>
                            <div class="naming-input-wrap">
                                <input
                                    type="text"
                                    class="naming-input"
                                    placeholder="their name or DID..."
                                    on:input=move |ev| {
                                        set_invite_did.set(event_target_value(&ev));
                                    }
                                    prop:value=invite_did
                                />
                                <div class="naming-underline" />
                            </div>
                            <div class="invitation-actions">
                                {move || {
                                    if invite_did.get().len() >= 2 {
                                        view! {
                                            <button class="ceremony-btn">
                                                "send invitation"
                                            </button>
                                        }.into_any()
                                    } else {
                                        view! { <div /> }.into_any()
                                    }
                                }}
                                <a href="/" class="ceremony-skip">
                                    "enter alone for now"
                                </a>
                            </div>
                        </div>
                    }.into_any()
                }}
            </div>
        </div>
    }
}
