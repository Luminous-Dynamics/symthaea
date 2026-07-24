// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Guided, blinded Foundry qualification. The archive stays behind the UI:
//! one audition, three quick decisions, then continue.

use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api;
use crate::state::MuseState;

const QUICK_ARTIFACT: &str = "excerpt_natural.wav";
const SESSION_SIZE: usize = 3;

fn artifact_label(name: &str) -> &'static str {
    match name {
        "motif_card.wav" => "Hear the core idea",
        "transformation_comparison.wav" => "Hear it change",
        "excerpt_neutral.wav" => "Neutral performance",
        "excerpt_natural.wav" => "30-second piece",
        "complete_natural.wav" => "Complete composition",
        "revision_excerpt.wav" => "Revised ending",
        "ablation_comparison.wav" => "Foundry vs baselines",
        _ => "Audition",
    }
}

fn friendly_cell(cell: &str) -> String {
    cell.replace(['_', '-'], " ")
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            chars
                .next()
                .map(|first| first.to_uppercase().collect::<String>() + chars.as_str())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[component]
pub fn FoundryQualificationPage() -> impl IntoView {
    let muse = use_context::<MuseState>().expect("MuseState provided by App");
    let entries = RwSignal::new(Vec::<api::FoundryQualificationEntry>::new());
    let current_index = RwSignal::new(0_usize);
    let active_artifact = RwSignal::new(None::<(String, String)>);
    let status = RwSignal::new("Preparing a short listening session…".to_string());
    let reveal = RwSignal::new(None::<api::FoundryQualificationReveal>);
    let committed = RwSignal::new(false);
    let saving = RwSignal::new(false);
    let session_complete = RwSignal::new(false);
    let all_complete = RwSignal::new(false);

    // Quick review: three decisions. Everything else is optional.
    let reaction = RwSignal::new(None::<u8>);
    let identity = RwSignal::new(None::<u8>);
    let mechanicalness = RwSignal::new(None::<u8>);
    let development = RwSignal::new(3_u8);
    let ending = RwSignal::new(3_u8);
    let distinctness = RwSignal::new(3_u8);
    let confidence = RwSignal::new(3_u8);
    let note = RwSignal::new(String::new());
    let exposure = RwSignal::new("first".to_string());

    Effect::new(move |_| {
        spawn_local(async move {
            match api::fetch_foundry_qualification(api::DEFAULT_BACKEND).await {
                Ok(manifest) if !manifest.entries.is_empty() => {
                    status.set("Ready. A session is only three pieces.".into());
                    entries.set(manifest.entries);
                }
                Ok(_) => status.set("No Foundry auditions are available yet.".into()),
                Err(error) => status.set(format!("Foundry Review is unavailable: {error}")),
            }
        });
    });

    let play_artifact = move |artifact: &'static str| {
        let Some(entry) = entries
            .get_untracked()
            .get(current_index.get_untracked())
            .cloned()
        else {
            status.set("The listening session is still loading.".into());
            return;
        };
        let Some(hash) = entry.artifacts.get(artifact).cloned() else {
            status.set("That audition is not available for this piece.".into());
            return;
        };
        active_artifact.set(Some((artifact.into(), hash)));
        reveal.set(None);
        committed.set(false);
        status.set(format!(
            "Playing {}…",
            artifact_label(artifact).to_lowercase()
        ));
        muse.play_review_audio(
            api::foundry_qualification_audio_url(
                api::DEFAULT_BACKEND,
                &entry.anonymous_id,
                artifact,
            ),
            format!(
                "Blinded Foundry piece {} · {}",
                current_index.get_untracked() + 1,
                artifact_label(artifact)
            ),
        );
    };

    let quick_choice = |label: &'static str, value: u8, signal: RwSignal<Option<u8>>| {
        view! {
            <button
                class:active=move || signal.get() == Some(value)
                on:click=move |_| signal.set(Some(value))
            >{label}</button>
        }
    };

    let advanced_rating = |label: &'static str, signal: RwSignal<u8>| {
        view! {
            <label class="qualification-rating">
                <span>{label}</span>
                <input type="range" min="1" max="5"
                    prop:value=move || signal.get().to_string()
                    on:input=move |event| signal.set(event_target_value(&event).parse().unwrap_or(3)) />
                <strong>{move || signal.get()}</strong>
            </label>
        }
    };

    view! {
        <main class="foundry-guided">
            <header class="foundry-guided-header">
                <div>
                    <span class="eyebrow">"Motif Foundry"</span>
                    <h2>"Help Muse choose what to remember"</h2>
                    <p>"Listen without labels. Trust your first response. Three pieces make one session."</p>
                </div>
                <div class="foundry-progress" aria-label="Review progress">
                    <strong>{move || {
                        let total = entries.get().len();
                        if total == 0 { "—".into() } else { format!("{} of {total}", current_index.get() + 1) }
                    }}</strong>
                    <span>{move || format!("piece {} of {} in this session", current_index.get() % SESSION_SIZE + 1, SESSION_SIZE)}</span>
                </div>
            </header>

            {move || if session_complete.get() {
                view! {
                    <section class="panel foundry-session-complete">
                        <span class="eyebrow">"Session complete"</span>
                        <h3>{move || if all_complete.get() { "The Foundry set is complete." } else { "That is enough listening for now." }}</h3>
                        <p>{move || if all_complete.get() {
                            "Your judgments are saved. Muse can use them without asking you to inspect the archive."
                        } else {
                            "Stop here, or begin another three-piece session when your ears feel fresh."
                        }}</p>
                        {move || if all_complete.get() {
                            view! { <a class="primary" href="/">"Return to Listen"</a> }.into_any()
                        } else {
                            view! { <button class="primary" on:click=move |_| {
                                session_complete.set(false);
                                status.set("Next session ready.".into());
                            }>"Begin next session"</button> }.into_any()
                        }}
                    </section>
                }.into_any()
            } else {
                view! {
                    <section class="panel foundry-current-audition">
                        <div class="foundry-audition-focus">
                            <span class="eyebrow">"Blinded piece"</span>
                            <h3>"Listen once, then answer three things"</h3>
                            <div class="foundry-listen-actions">
                                <button class="foundry-play primary" on:click=move |_| play_artifact(QUICK_ARTIFACT)>
                                    "Play 30-second piece"
                                </button>
                                <button class="foundry-play" on:click=move |_| play_artifact("complete_natural.wav")>
                                    "Play full composition"
                                </button>
                            </div>
                            <p class="muted">{move || status.get()}</p>
                        </div>

                        <div class="foundry-quick-questions">
                            <fieldset>
                                <legend>"Would you choose to hear it again?"</legend>
                                <div class="choice-row">
                                    {quick_choice("No", 2, reaction)}
                                    {quick_choice("Maybe", 3, reaction)}
                                    {quick_choice("Yes", 5, reaction)}
                                </div>
                            </fieldset>
                            <fieldset>
                                <legend>"Did it have an idea you could remember?"</legend>
                                <div class="choice-row">
                                    {quick_choice("Not yet", 2, identity)}
                                    {quick_choice("Somewhat", 3, identity)}
                                    {quick_choice("Clearly", 5, identity)}
                                </div>
                            </fieldset>
                            <fieldset>
                                <legend>"Did it feel musical or procedural?"</legend>
                                <div class="choice-row">
                                    {quick_choice("Musical", 1, mechanicalness)}
                                    {quick_choice("Mixed", 3, mechanicalness)}
                                    {quick_choice("Mechanical", 5, mechanicalness)}
                                </div>
                            </fieldset>
                        </div>

                        <details class="foundry-deep-review">
                            <summary>"Optional: compare transformations, revisions, or the complete piece"</summary>
                            <div class="qualification-artifacts">
                                {[
                                    "motif_card.wav",
                                    "transformation_comparison.wav",
                                    "excerpt_neutral.wav",
                                    "complete_natural.wav",
                                    "revision_excerpt.wav",
                                    "ablation_comparison.wav",
                                ].into_iter().map(|artifact| view! {
                                    <button on:click=move |_| play_artifact(artifact)>{artifact_label(artifact)}</button>
                                }).collect_view()}
                            </div>
                            <label class="qualification-rating">
                                <span>"Listening context"</span>
                                <select prop:value=move || exposure.get()
                                    on:change=move |event| exposure.set(event_target_value(&event))>
                                    <option value="first">"First listen"</option>
                                    <option value="next_day">"Next day"</option>
                                    <option value="one_week">"One week"</option>
                                    <option value="one_month">"One month"</option>
                                </select>
                            </label>
                            {advanced_rating("Development", development)}
                            {advanced_rating("Ending", ending)}
                            {advanced_rating("Distinctness", distinctness)}
                            {advanced_rating("Confidence", confidence)}
                            <textarea placeholder="Optional note" prop:value=move || note.get()
                                on:input=move |event| note.set(event_target_value(&event))></textarea>
                        </details>

                        <div class="foundry-commit-row">
                            <button class="primary" disabled=move || saving.get() || committed.get()
                                on:click=move |_| {
                                    let (Some(reaction_value), Some(identity_value), Some(mechanical_value)) =
                                        (reaction.get_untracked(), identity.get_untracked(), mechanicalness.get_untracked())
                                    else {
                                        status.set("Answer the three short questions first.".into());
                                        return;
                                    };
                                    let Some(entry) = entries.get_untracked().get(current_index.get_untracked()).cloned() else {
                                        status.set("The current blinded piece is unavailable.".into());
                                        return;
                                    };
                                    let (artifact_name, artifact_sha256) = active_artifact.get_untracked()
                                        .or_else(|| entry.artifacts.get(QUICK_ARTIFACT).cloned().map(|hash| (QUICK_ARTIFACT.into(), hash)))
                                        .expect("qualification entry has a quick artifact");
                                    saving.set(true);
                                    let judgment = api::FoundryQualificationJudgment {
                                        presentation_id: entry.anonymous_id,
                                        artifact_sha256,
                                        blind_session_id: format!("foundry-guided-session-{}", current_index.get_untracked() / SESSION_SIZE + 1),
                                        first_listen_or_repeat: exposure.get_untracked(),
                                        love: reaction_value,
                                        replay_desire: reaction_value,
                                        memorable_identity: identity_value,
                                        development: development.get_untracked(),
                                        ending: ending.get_untracked(),
                                        mechanicalness: mechanical_value,
                                        distinctness: distinctness.get_untracked().max(identity_value.min(5)),
                                        note: if note.get_untracked().trim().is_empty() {
                                            format!("guided review of {}", artifact_label(&artifact_name))
                                        } else { note.get_untracked() },
                                        confidence: confidence.get_untracked(),
                                    };
                                    spawn_local(async move {
                                        match api::record_foundry_qualification(api::DEFAULT_BACKEND, &judgment).await {
                                            Ok(committed_reveal) => {
                                                reveal.set(Some(committed_reveal));
                                                committed.set(true);
                                                status.set("Saved. Muse can now show what you heard.".into());
                                            }
                                            Err(error) => status.set(format!("Could not save this judgment: {error}")),
                                        }
                                        saving.set(false);
                                    });
                                }
                            >{move || if saving.get() { "Saving…" } else if committed.get() { "Saved" } else { "Save judgment" }}</button>
                        </div>

                        {move || reveal.get().map(|revealed| view! {
                            <aside class="qualification-reveal">
                                <span class="eyebrow">"Revealed after saving"</span>
                                <strong>{friendly_cell(&revealed.portfolio_cell)}</strong>
                                <details><summary>"Technical identity"</summary><code>{revealed.candidate_id}</code></details>
                            </aside>
                        })}

                        {move || committed.get().then(|| view! {
                            <button class="foundry-continue" on:click=move |_| {
                                let next = current_index.get_untracked() + 1;
                                let total = entries.get_untracked().len();
                                reaction.set(None);
                                identity.set(None);
                                mechanicalness.set(None);
                                active_artifact.set(None);
                                reveal.set(None);
                                committed.set(false);
                                note.set(String::new());
                                if next >= total {
                                    status.set("All available Foundry pieces have been reviewed.".into());
                                    all_complete.set(true);
                                    session_complete.set(true);
                                } else {
                                    current_index.set(next);
                                    session_complete.set(next % SESSION_SIZE == 0);
                                    status.set("Next blinded piece ready.".into());
                                }
                            }>"Continue"</button>
                        })}
                    </section>
                }.into_any()
            }}
        </main>
    }
}
