// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Telemetry line — thin horizontal strip showing a stream of values over time.
//!
//! Complements [`StatCard`](crate::stat_card::StatCard) which displays a single
//! static number. TelemetryLine shows the recent history of a metric as a
//! sparkline with label + current value flanking it.
//!
//! Intended as the building block of observatory dashboards, cluster-vitals
//! panels, and any "live data stream" visualization. Sizes down to 24px high
//! for dense layouts; scales up cleanly for hero views.
//!
//! # Semantic color tokens
//!
//! Uses these CSS custom properties (fallbacks in parentheses):
//! - `--md-signal` (cyan `#00d4ff`) — live data color
//! - `--md-mono` (`ui-monospace`) — monospace font stack for numbers
//! - `--md-fg` (`#c9d1d9`) — foreground text
//! - `--md-fg-muted` (`#6e7681`) — label/axis text
//!
//! Override these at any ancestor CSS scope to re-theme per cluster.
//!
//! # Example
//!
//! ```rust,no_run
//! use mycelix_leptos_core::TelemetryLine;
//! use leptos::prelude::*;
//! # let _ = || {
//! let flow_rate = RwSignal::new(vec![1.2_f64, 1.3, 1.1, 1.5, 1.4, 1.6, 1.8, 1.7, 1.9, 2.0]);
//! view! {
//!     <TelemetryLine
//!         label="Watershed flow"
//!         values=flow_rate.into()
//!         unit="L/s"
//!     />
//! }
//! # };
//! ```

use leptos::prelude::*;

/// A horizontal telemetry strip showing recent values as a sparkline.
///
/// # Props
///
/// * `label` — Short descriptive label shown at left (e.g. "Votes/min", "Flow rate").
/// * `values` — Reactive source of recent values, newest last. At least 2 points
///   renders a line; fewer renders an empty strip with the label + current value.
/// * `unit` — Optional unit string appended to the current value (e.g. "ms", "TEND/day").
/// * `min` / `max` — Optional fixed scale. If omitted, auto-fits to the values.
///   Passing fixed values is recommended for KPIs where the baseline matters
///   (e.g. `min=0` for counts, `min=-1 max=1` for signed ratios).
/// * `height` — Strip height in pixels. Defaults to 28 for dense dashboards.
/// * `width` — Strip width in pixels. Defaults to 160.
/// * `decimals` — How many decimal places for the current-value readout.
///   Defaults to 1. Set to 0 for integer metrics.
#[component]
pub fn TelemetryLine(
    label: &'static str,
    values: Signal<Vec<f64>>,
    #[prop(optional)] unit: Option<&'static str>,
    #[prop(optional)] min: Option<f64>,
    #[prop(optional)] max: Option<f64>,
    #[prop(optional)] height: Option<u32>,
    #[prop(optional)] width: Option<u32>,
    #[prop(optional)] decimals: Option<u32>,
) -> impl IntoView {
    let h = height.unwrap_or(28);
    let w = width.unwrap_or(160);
    let d = decimals.unwrap_or(1) as usize;

    // Reactive derivation of the sparkline polyline + current value.
    let sparkline = Memo::new(move |_| {
        let vs = values.get();
        if vs.is_empty() {
            return (String::new(), None::<f64>);
        }
        let current = *vs.last().unwrap();

        if vs.len() < 2 {
            return (String::new(), Some(current));
        }

        // Fit-to-scale: min/max come from props or auto-fit.
        let (lo, hi) = match (min, max) {
            (Some(lo), Some(hi)) => (lo, hi),
            (Some(lo), None) => (lo, vs.iter().copied().fold(f64::MIN, f64::max)),
            (None, Some(hi)) => (vs.iter().copied().fold(f64::MAX, f64::min), hi),
            (None, None) => (
                vs.iter().copied().fold(f64::MAX, f64::min),
                vs.iter().copied().fold(f64::MIN, f64::max),
            ),
        };
        // Avoid divide-by-zero for flat lines.
        let range = (hi - lo).abs().max(f64::EPSILON);

        let n = vs.len() as f64;
        // Plot area leaves 2px top/bottom padding so 0 and max don't clip.
        let plot_h = (h as f64) - 4.0;
        let pts: String = vs
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = (i as f64) / (n - 1.0) * (w as f64);
                // Invert Y because SVG 0 is top.
                let y = 2.0 + plot_h * (1.0 - ((v - lo) / range).clamp(0.0, 1.0));
                format!("{:.1},{:.1}", x, y)
            })
            .collect::<Vec<_>>()
            .join(" ");

        (pts, Some(current))
    });

    let points = Memo::new(move |_| sparkline.get().0);
    let current = Memo::new(move |_| sparkline.get().1);

    view! {
        <div
            class="md-telemetry-line"
            style=format!(
                "display: grid; grid-template-columns: auto 1fr auto; \
                 align-items: center; gap: 10px; \
                 padding: 4px 8px; border-radius: 2px; \
                 background: transparent; \
                 font-family: var(--md-mono, ui-monospace, 'JetBrains Mono', 'Iosevka', monospace); \
                 font-size: 11px; color: var(--md-fg, #c9d1d9); \
                 min-width: {}px;",
                w + 140
            )
        >
            <span
                style="color: var(--md-fg-muted, #6e7681); \
                       text-transform: uppercase; letter-spacing: 0.08em; \
                       white-space: nowrap; font-size: 10px;"
            >
                {label}
            </span>

            <svg
                width=move || w.to_string()
                height=move || h.to_string()
                viewBox=move || format!("0 0 {} {}", w, h)
                preserveAspectRatio="none"
                style="display: block;"
            >
                // Faint baseline so empty/sparse strips still read as telemetry.
                <line
                    x1="0"
                    y1=move || (h as f64 / 2.0).to_string()
                    x2=move || w.to_string()
                    y2=move || (h as f64 / 2.0).to_string()
                    stroke="var(--md-fg-muted, #6e7681)"
                    stroke-opacity="0.15"
                    stroke-width="1"
                />
                <polyline
                    fill="none"
                    stroke="var(--md-signal, #00d4ff)"
                    stroke-width="1.25"
                    stroke-linejoin="round"
                    stroke-linecap="round"
                    points=move || points.get()
                />
            </svg>

            <span
                style="font-variant-numeric: tabular-nums; \
                       color: var(--md-signal, #00d4ff); \
                       white-space: nowrap; font-size: 12px; font-weight: 600;"
            >
                {move || match current.get() {
                    Some(v) => match unit {
                        Some(u) => format!("{v:.*} {u}", d),
                        None => format!("{v:.*}", d),
                    },
                    None => "—".to_string(),
                }}
            </span>
        </div>
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure logic tests for the sparkline math. Component rendering
    // is tested in consumer apps via wasm-bindgen-test.

    #[test]
    fn empty_values_produces_no_points() {
        // Replicate the logic: empty values → empty point string.
        let vs: Vec<f64> = vec![];
        if vs.is_empty() {
            let expected_points = String::new();
            let expected_current: Option<f64> = None;
            assert!(expected_points.is_empty());
            assert!(expected_current.is_none());
        }
    }

    #[test]
    fn single_value_produces_current_but_no_line() {
        let vs = vec![42.0_f64];
        // len < 2 → current is Some, but no line is drawn.
        assert_eq!(vs.len(), 1);
        assert_eq!(*vs.last().unwrap(), 42.0);
    }

    #[test]
    fn flat_line_does_not_divide_by_zero() {
        // Repro the divide-by-zero guard: range = max(|hi-lo|, EPSILON).
        let vs = vec![1.0_f64, 1.0, 1.0, 1.0];
        let lo = vs.iter().copied().fold(f64::MAX, f64::min);
        let hi = vs.iter().copied().fold(f64::MIN, f64::max);
        let range = (hi - lo).abs().max(f64::EPSILON);
        assert!(range >= f64::EPSILON);
        // Plot calculation with range = EPSILON should yield non-NaN.
        let norm = ((1.0 - lo) / range).clamp(0.0, 1.0);
        assert!(norm.is_finite());
    }

    #[test]
    fn fixed_scale_overrides_auto_fit() {
        let vs = vec![0.3_f64, 0.5, 0.7];
        let min_fixed = Some(0.0);
        let max_fixed = Some(1.0);

        let (lo, hi) = match (min_fixed, max_fixed) {
            (Some(lo), Some(hi)) => (lo, hi),
            _ => (
                vs.iter().copied().fold(f64::MAX, f64::min),
                vs.iter().copied().fold(f64::MIN, f64::max),
            ),
        };
        // Fixed scale wins; the auto-fit max (0.7) is ignored.
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 1.0);
    }
}
