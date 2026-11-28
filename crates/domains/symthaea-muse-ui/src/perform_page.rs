// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Perform workspace shell (MEL-DJ-001 P0).
//!
//! This is an honest product shell, not a playback implementation. The
//! existing Muse player remains the single-source listening surface. Deck
//! transport, mixing, audio routing, controller I/O, stem playback, and
//! session recording must arrive through a separately qualified native
//! runtime; this page must not simulate those capabilities.

use leptos::prelude::*;

#[component]
pub fn PerformPage() -> impl IntoView {
    view! {
        <main class="perform-page">
            <header class="perform-heading">
                <div>
                    <p class="perform-eyebrow">"MUSE / PERFORMANCE WORKSPACE"</p>
                    <h2>"Perform with intention."</h2>
                    <p class="perform-intro">
                        "Prepare material, shape a set, and make transitions accountable to the music."
                    </p>
                </div>
                <span class="perform-status-chip" role="status">
                    <span class="perform-status-dot" aria-hidden="true"></span>
                    "WORKSPACE PREVIEW"
                </span>
            </header>

            <aside class="perform-boundary" role="note" aria-label="Runtime capability boundary">
                <span class="perform-boundary-mark" aria-hidden="true">"i"</span>
                <div>
                    <strong>"No live DJ runtime is connected yet."</strong>
                    <p>
                        "This workspace does not control the Listen player. Multi-deck playback, audio routing, headphone cue, stems, and hardware controllers are not implemented here."
                    </p>
                </div>
            </aside>

            <nav class="perform-subnav" aria-label="Performance workspace sections">
                <a href="#prepare"><span>"01"</span>"Prepare"</a>
                <a href="#mix-plan"><span>"02"</span>"Mix plan"</a>
                <a href="#performance"><span>"03"</span>"Performance"</a>
                <a href="#history"><span>"04"</span>"History"</a>
            </nav>

            <section id="prepare" class="perform-section">
                <div class="perform-section-heading">
                    <div>
                        <p class="perform-kicker">"01 / SOURCE MATERIAL"</p>
                        <h3>"Prepare"</h3>
                    </div>
                    <span class="perform-section-state">"NOT CONNECTED"</span>
                </div>
                <div class="perform-prepare-grid">
                    <article class="perform-card perform-library-card">
                        <div class="perform-card-icon" aria-hidden="true">"♫"</div>
                        <p class="perform-card-kicker">"LOCAL LIBRARY"</p>
                        <h4>"Your music, under your control."</h4>
                        <p>"A future local-first intake flow will index source files without uploading your collection or rewriting originals."</p>
                        <div class="perform-empty-state">
                            <span class="perform-empty-orbit" aria-hidden="true"></span>
                            <strong>"No library connected"</strong>
                            <small>"No files have been read or modified by this workspace."</small>
                        </div>
                    </article>
                    <article class="perform-card">
                        <p class="perform-card-kicker">"ANALYSIS EVIDENCE"</p>
                        <h4>"Know what the system knows."</h4>
                        <ul class="perform-evidence-list">
                            <li><span>"Beat and tempo hypotheses"</span><b>"—"</b></li>
                            <li><span>"Phrase and structure map"</span><b>"—"</b></li>
                            <li><span>"Stem / role quality"</span><b>"—"</b></li>
                            <li><span>"Uncertainty and provenance"</span><b>"—"</b></li>
                        </ul>
                        <p class="perform-footnote">"Unknown is not the same as zero confidence, and neither is a verified analysis."</p>
                    </article>
                    <article class="perform-card">
                        <p class="perform-card-kicker">"PERFORMANCE MARKERS"</p>
                        <h4>"Cues that travel with the track."</h4>
                        <div class="perform-marker-row">
                            <span class="perform-marker hotcue">"CUE"</span>
                            <span class="perform-marker loop">"LOOP"</span>
                            <span class="perform-marker phrase">"PHRASE"</span>
                        </div>
                        <p>"Cue points, loops, beatgrid hypotheses, and corrections will be versioned sidecars. Original audio remains immutable."</p>
                        <div class="perform-card-rule"></div>
                        <span class="perform-inline-state">"AUTHORING NOT AVAILABLE"</span>
                    </article>
                </div>
            </section>

            <section id="mix-plan" class="perform-section">
                <div class="perform-section-heading">
                    <div>
                        <p class="perform-kicker">"02 / SET DESIGN"</p>
                        <h3>"Mix plan"</h3>
                    </div>
                    <span class="perform-section-state">"DESIGN TARGET"</span>
                </div>
                <div class="perform-plan-grid">
                    <article class="perform-card perform-plan-card">
                        <p class="perform-card-kicker">"SET TRAJECTORY"</p>
                        <h4>"Shape the room, not just the queue."</h4>
                        <p>"Explore intentional energy, density, tension, familiarity, and contrast across a whole set. The planner should return trade-offs, not one universal taste score."</p>
                        <div class="perform-trajectory" aria-label="Illustrative set trajectory, not computed from tracks">
                            <div class="perform-trajectory-line"></div>
                            <span class="perform-trajectory-point p1"></span>
                            <span class="perform-trajectory-point p2"></span>
                            <span class="perform-trajectory-point p3"></span>
                            <span class="perform-trajectory-point p4"></span>
                            <div class="perform-trajectory-labels"><small>"OPEN"</small><small>"BUILD"</small><small>"PEAK"</small><small>"LAND"</small></div>
                        </div>
                        <p class="perform-footnote">"Illustrative only — no tracks, preferences, or trajectory have been evaluated."</p>
                    </article>
                    <article class="perform-card">
                        <p class="perform-card-kicker">"TRANSITION CONTRACT"</p>
                        <h4>"A mix should explain itself."</h4>
                        <ol class="perform-contract-steps">
                            <li><b>"01"</b><span><strong>"Align"</strong><small>"Tempo map, phrase, and timing evidence"</small></span></li>
                            <li><b>"02"</b><span><strong>"Protect"</strong><small>"Vocal phrases, hooks, transients, and intent"</small></span></li>
                            <li><b>"03"</b><span><strong>"Exchange"</strong><small>"Role-aware gain and routing trajectories"</small></span></li>
                            <li><b>"04"</b><span><strong>"Verify"</strong><small>"Constraints, uncertainty, preview, and replay receipt"</small></span></li>
                        </ol>
                    </article>
                </div>
            </section>

            <section id="performance" class="perform-section">
                <div class="perform-section-heading">
                    <div>
                        <p class="perform-kicker">"03 / LIVE CONTROL"</p>
                        <h3>"Performance"</h3>
                    </div>
                    <span class="perform-section-state">"RUNTIME NOT IMPLEMENTED"</span>
                </div>
                <div class="perform-deck-grid">
                    <article class="perform-deck">
                        <div class="perform-deck-top"><span>"DECK A"</span><span>"— / — BPM"</span></div>
                        <div class="perform-deck-wave" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
                        <div class="perform-deck-empty"><strong>"No track loaded"</strong><small>"Transport unavailable"</small></div>
                        <div class="perform-deck-bottom"><span>"KEY —"</span><span>"GRID —"</span><span>"CUE —"</span></div>
                    </article>
                    <article class="perform-deck">
                        <div class="perform-deck-top"><span>"DECK B"</span><span>"— / — BPM"</span></div>
                        <div class="perform-deck-wave wave-b" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
                        <div class="perform-deck-empty"><strong>"No track loaded"</strong><small>"Transport unavailable"</small></div>
                        <div class="perform-deck-bottom"><span>"KEY —"</span><span>"GRID —"</span><span>"CUE —"</span></div>
                    </article>
                </div>
                <div class="perform-mixer-note">
                    <div><span class="perform-mixer-glyph" aria-hidden="true">"◫"</span><strong>"Mixer, monitoring, and controller runtime"</strong></div>
                    <p>"Not connected. No crossfader, gain, EQ, cue, routing, or MIDI/HID controls are presented as live or functional."</p>
                </div>
            </section>

            <section id="history" class="perform-section">
                <div class="perform-section-heading">
                    <div>
                        <p class="perform-kicker">"04 / REPLAY & ACCOUNTABILITY"</p>
                        <h3>"History"</h3>
                    </div>
                    <span class="perform-section-state">"NO SESSIONS"</span>
                </div>
                <div class="perform-history">
                    <div class="perform-history-mark" aria-hidden="true">"↺"</div>
                    <div>
                        <strong>"Every set should be replayable."</strong>
                        <p>"When session capture exists, this space will link the set timeline to source hashes, loaded analysis versions, controller events, transition receipts, recordings, and explicit export/loss reports."</p>
                        <small>"No session history exists in this preview."</small>
                    </div>
                </div>
            </section>
            <footer class="perform-footer">
                <span>"MEL-DJ-001 / P0"</span>
                <span>"Evidence before automation · performer authority first"</span>
            </footer>
        </main>
    }
}
