use leptos::prelude::*;

/// SVG glyph rendering for the Glyph Codex.
///
/// Displays the currently active glyph with its ID and echo text.
/// Will be wired to live glyph data from the SporeEngine worker.
#[component]
pub fn GlyphDisplay() -> impl IntoView {
    view! {
        <div class="glyph-display">
            <svg viewBox="0 0 80 80" width="80" height="80" xmlns="http://www.w3.org/2000/svg">
                // Placeholder glyph: concentric circles with a dot
                <circle cx="40" cy="40" r="35" fill="none" stroke="rgba(232,197,71,0.2)" stroke-width="0.5" />
                <circle cx="40" cy="40" r="22" fill="none" stroke="rgba(232,197,71,0.3)" stroke-width="0.5" />
                <circle cx="40" cy="40" r="10" fill="none" stroke="rgba(232,197,71,0.4)" stroke-width="0.5" />
                <circle cx="40" cy="40" r="2" fill="rgba(232,197,71,0.6)" />
            </svg>
            <div class="glyph-id">"omega_0"</div>
            <div class="glyph-echo">"Before the first word, there was listening."</div>
        </div>
    }
}
