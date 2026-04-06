// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-time spectral vector visualizer for the search bar.
//!
//! As the user types, the 16,384-bit BinaryHV is computed and visualized
//! as a spectral barcode — each byte maps to a color frequency, creating
//! a living geometric fingerprint of the user's thought.

use leptos::prelude::*;
use prism_search::encode_text;

/// Renders a spectral visualization of the BinaryHV encoded from the input text.
/// Shows as a thin colorful bar below the search input.
#[component]
pub fn VectorSpectrum(input: ReadSignal<String>) -> impl IntoView {
    // Compute a compact spectral representation of the HDC vector.
    // We take 128 samples from the 2048-byte BinaryHV and map them to
    // CSS colors — creating a spectral barcode.
    let spectrum = move || {
        let text = input.get();
        if text.trim().is_empty() {
            return "transparent".to_string();
        }

        let hv = encode_text(&text);
        let bytes = &hv.0;

        // Sample 64 evenly-spaced bytes from the 2048-byte vector
        let samples: Vec<String> = (0..64)
            .map(|i| {
                let idx = i * 32; // 64 samples across 2048 bytes
                let b = bytes[idx];
                // Map byte to a spectral color
                // Low bits → teal/cyan, mid bits → green/yellow, high bits → violet/pink
                let hue = (b as f32 / 255.0) * 300.0 + 160.0; // 160-460 (wrap around)
                let hue = hue % 360.0;
                let sat = 80.0 + (bytes[(idx + 1) % 2048] as f32 / 255.0) * 20.0;
                let light = 45.0 + (bytes[(idx + 7) % 2048] as f32 / 255.0) * 25.0;
                format!("hsl({:.0}, {:.0}%, {:.0}%)", hue, sat, light)
            })
            .collect();

        // Build CSS gradient
        let stops: Vec<String> = samples
            .iter()
            .enumerate()
            .map(|(i, color)| {
                let pct = (i as f32 / samples.len() as f32) * 100.0;
                format!("{} {:.1}%", color, pct)
            })
            .collect();

        format!("linear-gradient(90deg, {})", stops.join(", "))
    };

    let is_active = move || {
        let text = input.get();
        let trimmed = text.trim();
        // Only show spectrum for search queries, not URLs
        if trimmed.is_empty() || trimmed.starts_with("http") || trimmed.starts_with("prism://") {
            "vector-spectrum"
        } else {
            "vector-spectrum active"
        }
    };

    view! {
        <div class=is_active>
            <div class="vector-spectrum-bar" style:background=spectrum></div>
        </div>
    }
}
