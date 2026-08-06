// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos_router::components::A;
use wasm_bindgen::JsCast;

use crate::api::{self, ImportAuthorization, ImportedWorkSummary, TeachingCorpusSummary};
use crate::state::MuseState;

#[component]
pub fn AddMusicPage() -> impl IntoView {
    let muse = expect_context::<MuseState>();
    let file_ref = NodeRef::<leptos::html::Input>::new();
    let title = RwSignal::new(String::new());
    let contributor = RwSignal::new(String::new());
    // `None` until the contributor makes an explicit choice -- there is no
    // safe default between "I made this" and "I'm just authorized to import
    // it", so unlike the old single checkbox this cannot default to checked.
    let authorization = RwSignal::new(None::<ImportAuthorization>);
    let working = RwSignal::new(false);
    let status = RwSignal::new(String::new());
    let imported = RwSignal::new(None::<ImportedWorkSummary>);
    let teaching = RwSignal::new(None::<TeachingCorpusSummary>);
    let teaching_status = RwSignal::new("Loading the authored etudes…".to_string());

    Effect::new(move |_| {
        spawn_local(async move {
            match api::fetch_teaching_corpus(api::DEFAULT_BACKEND).await {
                Ok(corpus) => {
                    teaching_status.set(format!(
                        "{} authored etudes available · none trusted automatically",
                        corpus.lessons.len()
                    ));
                    teaching.set(Some(corpus));
                }
                Err(error) => teaching_status.set(error),
            }
        });
    });

    let submit = move || {
        let Some(input) = file_ref.get_untracked() else {
            status.set("Choose a MIDI or MusicXML file.".into());
            return;
        };
        let input: web_sys::HtmlInputElement = input.unchecked_into();
        let Some(file) = input.files().and_then(|files| files.get(0)) else {
            status.set("Choose a MIDI or MusicXML file.".into());
            return;
        };
        if title.get_untracked().trim().is_empty() || contributor.get_untracked().trim().is_empty()
        {
            status.set("Add the work title and contributor name.".into());
            return;
        }
        let Some(basis) = authorization.get_untracked() else {
            status
                .set("Choose whether you created this work or are authorized to import it.".into());
            return;
        };
        working.set(true);
        status.set("Reconstructing score, form, and motif evidence…".into());
        let title_value = title.get_untracked();
        let contributor_value = contributor.get_untracked();
        spawn_local(async move {
            match api::import_music(
                api::DEFAULT_BACKEND,
                file,
                &title_value,
                &contributor_value,
                basis,
            )
            .await
            {
                Ok(work) => {
                    status.set("Imported privately. Nothing was contributed to Foundry or global learning.".into());
                    imported.set(Some(work));
                }
                Err(error) => status.set(error),
            }
            working.set(false);
        });
    };

    view! {
        <main class="add-music-page">
            <header class="add-music-intro">
                <span class="eyebrow">"Private symbolic import"</span>
                <h2>"Add your music to Muse"</h2>
                <p>
                    "Import a MIDI or MusicXML score for this private workspace. Muse may analyze and render it locally, but it cannot enter Foundry, shared listening, grammar research, or global learning."
                </p>
            </header>

            <section class="import-workbench">
                <label>
                    <span>"Symbolic score"</span>
                    <input node_ref=file_ref type="file" accept=".mid,.midi,.musicxml,.xml,.json" />
                    <small>"MIDI, MusicXML, or native Muse score JSON · maximum 12 MiB"</small>
                </label>
                <div class="import-fields">
                    <label><span>"Work title"</span><input type="text" maxlength="160"
                        prop:value=move || title.get()
                        on:input=move |event| title.set(event_target_value(&event)) /></label>
                    <label><span>"Contributor"</span><input type="text" maxlength="120"
                        prop:value=move || contributor.get()
                        on:input=move |event| contributor.set(event_target_value(&event)) /></label>
                </div>
                <fieldset class="authorship-choice">
                    <legend>"Whose work is this?"</legend>
                    <label class="authorship-confirmation">
                        <input type="radio" name="authorization-basis"
                            prop:checked=move || authorization.get() == Some(ImportAuthorization::OwnWork)
                            on:change=move |_| authorization.set(Some(ImportAuthorization::OwnWork)) />
                        <span>"I created this work."</span>
                    </label>
                    <label class="authorship-confirmation">
                        <input type="radio" name="authorization-basis"
                            prop:checked=move || authorization.get() == Some(ImportAuthorization::AuthorizedImport)
                            on:change=move |_| authorization.set(Some(ImportAuthorization::AuthorizedImport)) />
                        <span>"I'm authorized to import and privately analyze someone else's work."</span>
                    </label>
                </fieldset>
                <aside class="privacy-receipt">
                    <strong>"Permission receipt · version 1"</strong>
                    <span>"Private project only"</span>
                    <span>"All musical rights reserved"</span>
                    <span>"No Foundry contribution"</span>
                    <span>"No global Muse learning"</span>
                </aside>
                <button class="primary-action" type="button" disabled=move || working.get()
                    on:click=move |_| submit()>
                    {move || if working.get() { "Analyzing…" } else { "Import privately" }}
                </button>
                <p class="muted">{move || status.get()}</p>
            </section>

            {move || imported.get().map(|work| {
                let work_id = work.manifest.work_id.0.clone();
                let audio_id = work_id.clone();
                let title = work.manifest.title.clone();
                let playback_title = title.clone();
                let analysis = work.analysis.clone();
                let instrumentation_note = work.instrumentation_note.clone();
                view! {
                    <section class="import-analysis">
                        <header>
                            <div><span class="eyebrow">"Source-native interpretation"</span><h3>{title}</h3></div>
                            <span class="private-badge">"Private"</span>
                        </header>
                        <div class="import-analysis-facts">
                            <span><small>"Notes"</small><strong>{analysis.note_count}</strong></span>
                            <span><small>"Voices"</small><strong>{analysis.voice_count}</strong></span>
                            <span><small>"Tempo"</small><strong>{format!("{:.0} BPM", analysis.tempo_bpm)}</strong></span>
                            <span><small>"Meter"</small><strong>{format!("{}/4", analysis.meter)}</strong></span>
                            <span><small>"Territory"</small><strong>"Unclassified"</strong></span>
                        </div>
                        <p>"Muse preserved the work as source-native. Section and motif findings remain reconstructed suggestions until you confirm them."</p>
                        {analysis.motifs.first().cloned().map(|motif| view! {
                            <div class="import-motif-result">
                                <strong>"Possible opening motif"</strong>
                                <span>{format!("{} notes · {} occurrence{} · {:.0}% confidence",
                                    motif.midi_pitches.len(), motif.occurrence_count,
                                    if motif.occurrence_count == 1 { "" } else { "s" }, motif.confidence * 100.0)}</span>
                            </div>
                        })}
                        <div class="import-actions">
                            <button type="button" on:click=move |_| muse.play_review_audio(
                                api::imported_audio_url(api::DEFAULT_BACKEND, &audio_id), playback_title.clone())>
                                "Play reconstructed rendition"
                            </button>
                            <A href="/liked">"Open Library"</A>
                            <A href="/atlas">"Show in private Atlas"</A>
                        </div>
                        <p class="muted small">{instrumentation_note}</p>
                        <details>
                            <summary>"Unresolved interpretations"</summary>
                            <ul>{analysis.unresolved_interpretations.into_iter().map(|item| view! { <li>{item}</li> }).collect_view()}</ul>
                            <p class="muted small">"Contributor correction is the next layer; Muse's original inference will remain append-only when corrected."</p>
                        </details>
                        <small class="muted">{"Imported work identity: "}{work_id}</small>
                    </section>
                }
            })}

            <section class="teaching-shelf">
                <header class="add-music-intro">
                    <span class="eyebrow">"Teach Muse · shadow mode"</span>
                    <h2>"Sixteen ways a composition can think"</h2>
                    <p>
                        "These provenance-clean etudes teach abstract strategies, never source notes. The first four have typed shadow mappings; every lesson remains authored until evidence earns a stronger status."
                    </p>
                    <small class="muted">{move || teaching_status.get()}</small>
                </header>
                <div class="teaching-grid">
                    {move || teaching.get().map(|corpus| corpus.lessons.into_iter().map(|lesson| {
                        let lesson_id = lesson.lesson_id.clone();
                        let play_id = lesson_id.clone();
                        let title = lesson.title.clone();
                        let playback_title = title.clone();
                        view! {
                            <article class="teaching-card">
                                <div>
                                    <span class="eyebrow">{lesson.primary_dimension}</span>
                                    <h3>{title}</h3>
                                    <p>{lesson.subtitle}</p>
                                </div>
                                <p class="teaching-rule">{lesson.abstract_rule}</p>
                                <div class="teaching-meta">
                                    <span>{format!("{} sections", lesson.section_count)}</span>
                                    <span>{format!("{:.0}s", lesson.duration_seconds)}</span>
                                    <span>{if lesson.typed_shadow_mapping { "Shadow-ready" } else { "Authored" }}</span>
                                </div>
                                <button type="button" on:click=move |_| muse.play_review_audio(
                                    api::teaching_audio_url(api::DEFAULT_BACKEND, &play_id),
                                    format!("Teaching etude · {playback_title}"),
                                )>
                                    "Hear the etude"
                                </button>
                                <small class="muted">{lesson_id}</small>
                            </article>
                        }
                    }).collect_view())}
                </div>
            </section>
        </main>
    }
}
