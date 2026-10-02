// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Root application: Converse (chat with the facade) + Observe (live
//! telemetry from the experience bridge, when enabled on the daemon).
//!
//! Design note (standing project rule): this is meant to read as a living
//! organism's vitals, not an admin dashboard — the vital orb's pulse rate
//! and hue are driven directly by consciousness_level/valence/thermodynamic
//! load rather than laid out as a grid of numeric gauges. v0 keeps this
//! modest (one orb + a few readouts); it should grow toward that goal
//! rather than toward more grid cells.

use base64::Engine as _;
use leptos::prelude::*;
use leptos::task::spawn_local_scoped_with_cancellation;
use serde_json::Value;
use std::cell::Cell;
use std::rc::Rc;
use wasm_bindgen::JsCast;

use crate::api::{self};
use crate::presentation::{
    cognitive_spans, event_between, next_event_sequence, next_telemetry_session,
    push_cognitive_event, telemetry_session_is_current, CognitiveEvent, CognitiveEventKind,
    CognitiveState,
};

const DEFAULT_GATEWAY: &str = "http://127.0.0.1:8090";

/// Allocate an authority generation without ever reusing an identity.
///
/// Request/status generations gate asynchronous responses. Saturating at
/// u64::MAX would make the terminal generation repeat forever, which could
/// allow an arbitrarily old response to become authoritative again. Exhaustion
/// is therefore a deliberate fail-closed state.
fn next_authority_generation(current: &mut u64) -> Option<u64> {
    let next = current.checked_add(1)?;
    *current = next;
    Some(next)
}

#[derive(Clone, Debug)]
struct Turn {
    role: &'static str,
    text: String,
}

/// Latest telemetry snapshot extracted from a `CycleMetadata` payload.
/// Fields default to a neutral resting state until the first message
/// arrives — see `Vitals::from_json` for the exact wire paths, which
/// mirror `src/cognitive_loop/types/telemetry.rs`.
#[derive(Clone, Copy, Debug, Default)]
struct Vitals {
    consciousness_level: f64,
    valence: f32,
    arousal: f32,
    mood_temperature: f32,
    thermodynamic_load: f32,
    moral_score: f64,
    coherence: f64,
    gwt_broadcast: bool,
    dream_insights: u64,
    surprise_triggered: bool,
    reasoning_confidence: f32,
    prediction_error: f32,
}

impl Vitals {
    /// `CycleMetadata`'s sub-structs (`consciousness`, `embodied`, `temporal`,
    /// `attention`, `memory`, `harmonics`, `ethics`, `neuromod`, ...) are ALL
    /// `#[serde(flatten)]`, so despite the nested Rust field access used
    /// elsewhere in this codebase (e.g. `m.consciousness.consciousness_level`),
    /// the actual wire JSON is flat — every field is a top-level key. Confirmed
    /// live against a real daemon 2026-07-12 (an earlier nested-path version of
    /// this function silently read `None` for everything).
    fn from_json(v: &Value) -> Option<Self> {
        // These four fields are the minimum evidence required to construct a
        // semantic cognitive state. Missing telemetry must not silently become
        // zeros: doing so would turn malformed gateway data into apparently
        // valid observations and could generate false timeline events.
        let required_f64 = |key: &str| -> Option<f64> {
            let value = v.get(key)?.as_f64()?;
            value.is_finite().then_some(value)
        };
        let required_f32 = |key: &str| -> Option<f32> {
            let value = required_f64(key)? as f32;
            value.is_finite().then_some(value)
        };
        let consciousness_level = required_f64("consciousness_level")?;
        let valence = required_f32("affective_valence")?;
        let arousal = required_f32("affective_arousal")?;
        let mood_temperature = required_f32("mood_temperature")?;
        let thermodynamic_load = required_f32("thermodynamic_load")?;
        let moral_score = required_f64("value_evaluator_score")?;
        let coherence = required_f64("harmonic_field_coherence")?;
        let reasoning_confidence = required_f32("reasoning_confidence")?;
        let prediction_error = required_f32("prediction_error")?;
        Some(Self {
            consciousness_level,
            valence,
            arousal,
            mood_temperature,
            thermodynamic_load,
            moral_score,
            coherence,
            gwt_broadcast: v["gwt_broadcast"].as_bool().unwrap_or(false),
            dream_insights: v["dream_insights"].as_u64().unwrap_or(0),
            surprise_triggered: v["surprise_triggered"].as_bool().unwrap_or(false),
            reasoning_confidence,
            prediction_error,
        })
    }
}

/// A decoded mental movie: RGBA frames ready for canvas `putImageData`.
/// Wire form (`mental_movie` key on the ws-live payload): base64 raw frames
/// of `width*height*channels` bytes, channels 1 or 3 — expanded to RGBA here
/// once, at parse time.
#[derive(Clone)]
struct Movie {
    frames_rgba: Vec<Vec<u8>>,
    width: u32,
    height: u32,
    semantic_coherence: f32,
}

/// Generous upper bound on a single "mental movie" frame's pixel count
/// (far more than a telemetry visualization frame plausibly needs). The
/// gateway URL is a user-editable text field, so a malicious or
/// compromised gateway must not be able to drive an unbounded (or, on
/// 32-bit wasm, integer-overflowing) allocation via `width`/`height`.
const MAX_MOVIE_PIXELS: usize = 4096 * 4096;
/// Bound the total RGBA allocation for one remote projection, not merely each
/// individual frame. A 4096² frame is already far larger than the UI canvas;
/// without a total budget, a bounded 120-frame sequence could still request
/// multi-gigabyte browser memory before the canvas ever displays it.
const MAX_MOVIE_RGBA_BYTES: usize = 64 * 1024 * 1024;

impl Movie {
    fn from_json(v: &Value) -> Option<Movie> {
        use base64::Engine as _;
        let m = v.get("mental_movie")?;
        // Decode dimensions without truncating hostile u64 JSON values into
        // the wasm32 u32/usize domain. Truncation here could turn an enormous
        // remote dimension into a small allocation and then leave indexing
        // arithmetic inconsistent with the declared wire shape.
        let width = u32::try_from(m["width"].as_u64()?).ok()?;
        let height = u32::try_from(m["height"].as_u64()?).ok()?;
        let channels = usize::try_from(m["channels"].as_u64()?).ok()?;
        if channels != 1 && channels != 3 {
            return None;
        }
        let engine = base64::engine::general_purpose::STANDARD;
        let px = (width as usize).checked_mul(height as usize)?;
        if px == 0 || px > MAX_MOVIE_PIXELS {
            return None;
        }
        let bytes_per_frame = px.checked_mul(channels)?;
        let rgba_capacity = px.checked_mul(4)?;
        const MAX_MOVIE_FRAMES: usize = 120;
        let frames = m["frames_b64"].as_array()?;
        if frames.len() > MAX_MOVIE_FRAMES {
            return None;
        }
        let total_rgba_bytes = rgba_capacity.checked_mul(frames.len())?;
        if total_rgba_bytes > MAX_MOVIE_RGBA_BYTES {
            return None;
        }
        // Standard base64 expands binary data by at most 4/3. Bound each
        // encoded frame before decoding so a hostile string cannot force a
        // large temporary allocation merely to be rejected for its size.
        let max_encoded_frame_bytes = bytes_per_frame
            .checked_add(2)?
            .checked_div(3)?
            .checked_mul(4)?;
        // A malformed frame invalidates the whole projection. Silently
        // dropping bad frames would make a partially accepted remote movie
        // look authoritative while hiding transport corruption or hostile
        // payloads from the caller.
        let mut frames_rgba = Vec::with_capacity(frames.len());
        for frame in frames {
            let encoded = frame.as_str()?;
            if encoded.len() > max_encoded_frame_bytes {
                return None;
            }
            let raw = engine.decode(encoded).ok()?;
            if raw.len() != bytes_per_frame {
                return None;
            }

            let mut rgba = Vec::with_capacity(rgba_capacity);
            for i in 0..px {
                let (r, g, b) = if channels >= 3 {
                    (
                        raw[i * channels],
                        raw[i * channels + 1],
                        raw[i * channels + 2],
                    )
                } else {
                    (raw[i], raw[i], raw[i])
                };
                rgba.extend_from_slice(&[r, g, b, 255]);
            }
            frames_rgba.push(rgba);
        }
        if frames_rgba.is_empty() {
            return None;
        }
        Some(Movie {
            frames_rgba,
            width,
            height,
            semantic_coherence: m["semantic_coherence"]
                .as_f64()
                .filter(|value| value.is_finite())
                .and_then(|value| {
                    let value = value as f32;
                    value.is_finite().then_some(value)
                })
                .unwrap_or(0.0),
        })
    }
}

/// SVG numeric tokens are parsed as finite f64 values and kept within a
/// deliberately conservative magnitude bound. This prevents values such as
/// enormous exponents from reaching browser SVG/geometry machinery even when
/// their source string is otherwise character-safe.
const MAX_PORTRAIT_NUMBER_ABS: f64 = 1_000_000.0;
const MAX_PORTRAIT_NUMBER_LENGTH: usize = 32;
const MAX_PORTRAIT_NUMBER_TOKENS: usize = 256;
const MAX_PORTRAIT_TRANSFORMS: usize = 16;
const MAX_PORTRAIT_PATH_COMMANDS: usize = 256;

fn svg_number_is_bounded(token: &str) -> bool {
    token.len() <= MAX_PORTRAIT_NUMBER_LENGTH
        && token
            .parse::<f64>()
            .map(|value| value.is_finite() && value.abs() <= MAX_PORTRAIT_NUMBER_ABS)
            .unwrap_or(false)
}

fn svg_numeric_tokens(value: &str) -> Option<Vec<f64>> {
    let bytes = value.as_bytes();
    let mut numbers = Vec::new();
    let mut cursor = 0usize;

    while cursor < bytes.len() {
        while cursor < bytes.len()
            && (bytes[cursor] == b',' || bytes[cursor].is_ascii_whitespace())
        {
            cursor += 1;
        }
        if cursor == bytes.len() {
            break;
        }

        let start = cursor;
        if matches!(bytes[cursor], b'+' | b'-') {
            cursor += 1;
        }

        let integer_start = cursor;
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            cursor += 1;
        }
        let has_integer = cursor > integer_start;

        let mut has_fraction = false;
        if cursor < bytes.len() && bytes[cursor] == b'.' {
            cursor += 1;
            let fraction_start = cursor;
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                cursor += 1;
            }
            has_fraction = cursor > fraction_start;
        }

        if !has_integer && !has_fraction {
            return None;
        }

        if cursor < bytes.len() && matches!(bytes[cursor], b'e' | b'E') {
            cursor += 1;
            if cursor < bytes.len() && matches!(bytes[cursor], b'+' | b'-') {
                cursor += 1;
            }
            let exponent_start = cursor;
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                cursor += 1;
            }
            if cursor == exponent_start {
                return None;
            }
        }

        let token = &value[start..cursor];
        if !svg_number_is_bounded(token) {
            return None;
        }
        numbers.push(token.parse::<f64>().ok()?);
        if numbers.len() > MAX_PORTRAIT_NUMBER_TOKENS {
            return None;
        }

        if cursor < bytes.len()
            && !bytes[cursor].is_ascii_whitespace()
            && bytes[cursor] != b','
            && bytes[cursor] != b'+'
            && bytes[cursor] != b'-'
        {
            return None;
        }
    }

    Some(numbers)
}

fn svg_numeric_list_is_bounded(value: &str) -> bool {
    svg_numeric_tokens(value)
        .map(|numbers| !numbers.is_empty())
        .unwrap_or(false)
}

fn svg_single_number_is_bounded(value: &str) -> bool {
    svg_numeric_tokens(value)
        .map(|numbers| numbers.len() == 1)
        .unwrap_or(false)
}

fn svg_viewbox_is_bounded(value: &str) -> bool {
    svg_numeric_tokens(value)
        .map(|numbers| {
            numbers.len() == 4
                && numbers[2] > 0.0
                && numbers[3] > 0.0
        })
        .unwrap_or(false)
}

fn svg_single_nonnegative_number_is_bounded(value: &str) -> bool {
    svg_numeric_tokens(value)
        .map(|numbers| {
            numbers.len() == 1
                && numbers[0] >= 0.0
        })
        .unwrap_or(false)
}

fn svg_single_unit_number_is_bounded(value: &str) -> bool {
    svg_numeric_tokens(value)
        .map(|numbers| {
            numbers.len() == 1
                && (0.0..=1.0).contains(&numbers[0])
        })
        .unwrap_or(false)
}

fn svg_points_is_bounded(value: &str) -> bool {
    svg_numeric_tokens(value)
        .map(|numbers| numbers.len() >= 2 && numbers.len() % 2 == 0)
        .unwrap_or(false)
}

fn svg_path_data_is_bounded(value: &str) -> bool {
    // SVG path data is a real command/parameter grammar. Character-level
    // allowlisting alone would accept incomplete commands, wrong arities, or
    // numbers in places where a command is required. Parse only the supported
    // command vocabulary and require each parameter group to be complete.
    #[derive(Clone)]
    enum Token {
        Command(char),
        Number(String),
    }

    fn parameter_count(command: char) -> Option<usize> {
        match command {
            'M' | 'm' => Some(2),
            'L' | 'l' => Some(2),
            'H' | 'h' => Some(1),
            'V' | 'v' => Some(1),
            'C' | 'c' => Some(6),
            'S' | 's' => Some(4),
            'Q' | 'q' => Some(4),
            'T' | 't' => Some(2),
            'A' | 'a' => Some(7),
            'Z' | 'z' => Some(0),
            _ => None,
        }
    }

    let mut tokens = Vec::<Token>::new();
    let mut number = String::new();
    let mut previous = None;

    let flush_number = |number: &mut String, tokens: &mut Vec<Token>| -> bool {
        if number.is_empty() {
            return true;
        }
        if tokens
            .iter()
            .filter(|token| matches!(token, Token::Number(_)))
            .count()
            >= MAX_PORTRAIT_NUMBER_TOKENS
        {
            return false;
        }
        if !svg_number_is_bounded(number) {
            return false;
        }
        tokens.push(Token::Number(number.clone()));
        number.clear();
        true
    };

    for ch in value.chars() {
        if parameter_count(ch).is_some() {
            if !flush_number(&mut number, &mut tokens) {
                return false;
            }
            tokens.push(Token::Command(ch));
            previous = Some(ch);
            continue;
        }

        let separator = ch == ',' || ch.is_ascii_whitespace();
        let sign_starts_number = matches!(ch, '+' | '-')
            && (number.is_empty() || !matches!(previous, Some('e' | 'E')));

        if separator {
            if !flush_number(&mut number, &mut tokens) {
                return false;
            }
            previous = Some(ch);
            continue;
        }

        if sign_starts_number {
            if !number.is_empty() && !flush_number(&mut number, &mut tokens) {
                return false;
            }
            number.push(ch);
            previous = Some(ch);
            continue;
        }

        if matches!(ch, '+' | '-') && matches!(previous, Some('e' | 'E')) {
            number.push(ch);
            previous = Some(ch);
            continue;
        }

        if ch == 'e' || ch == 'E' || ch == '.' || ch.is_ascii_digit() {
            number.push(ch);
            previous = Some(ch);
            continue;
        }

        return false;
    }

    if !flush_number(&mut number, &mut tokens) {
        return false;
    }

    // SVG path data is organized as subpaths, each beginning with a
    // moveto command. Reject draw/close commands before the first M/m
    // rather than relying on browser recovery semantics.
    if !matches!(
        tokens.first(),
        Some(Token::Command('M' | 'm'))
    ) {
        return false;
    }

    let mut index = 0usize;
    let mut command_count = 0usize;
    let mut active_command = None::<char>;
    let mut first_multiplicity = true;
    let mut subpath_start_required = true;

    while index < tokens.len() {
        let command = match &tokens[index] {
            Token::Command(command) => {
                let command = *command;
                if subpath_start_required && !matches!(command, 'M' | 'm') {
                    return false;
                }
                subpath_start_required = false;
                index += 1;
                command
            }
            Token::Number(_) => {
                if active_command.is_none() {
                    return false;
                }
                active_command.unwrap()
            }
        };

        let arity = match parameter_count(command) {
            Some(arity) => arity,
            None => return false,
        };

        command_count = command_count.checked_add(1).unwrap_or(usize::MAX);
        if command_count > MAX_PORTRAIT_PATH_COMMANDS {
            return false;
        }

        if arity == 0 {
            // ClosePath never consumes parameters; another number before the
            // next command would therefore be malformed.
            if index < tokens.len() && matches!(&tokens[index], Token::Number(_)) {
                return false;
            }
            active_command = None;
            first_multiplicity = true;
            subpath_start_required = true;
            continue;
        }

        active_command = Some(command);
        first_multiplicity = true;
        let mut consumed = 0usize;
        let mut completed_group = false;

        while index < tokens.len() && matches!(&tokens[index], Token::Number(_)) {
            let raw_number = match &tokens[index] {
                Token::Number(raw) => raw.as_str(),
                Token::Command(_) => return false,
            };
            consumed += 1;

            // SVG's elliptical-arc grammar makes the fourth and fifth
            // parameters flags, not general reals: each must be the exact
            // lexical token "0" or "1". Do not normalize arbitrary bounded
            // numerics such as "0.0", "+0", or "2" into an accepted flag.
            if matches!(command, 'A' | 'a')
                && matches!(consumed, 4 | 5)
                && !matches!(raw_number, "0" | "1")
            {
                return false;
            }

            index += 1;

            if consumed == arity {
                // MoveTo's additional coordinate pairs are implicit LineTo
                // commands of the same relative/absolute case. All other
                // commands simply repeat the same command grammar.
                let repeated = if first_multiplicity && matches!(command, 'M' | 'm') {
                    if command == 'M' { 'L' } else { 'l' }
                } else {
                    command
                };
                active_command = Some(repeated);
                first_multiplicity = false;
                consumed = 0;
                completed_group = true;

                if index < tokens.len() && matches!(&tokens[index], Token::Command(_)) {
                    break;
                }
            }
        }

        // Every non-close command must consume at least one complete
        // parameter group. An explicit command with no parameters is
        // malformed and must fail closed.
        if consumed != 0 || !completed_group {
            return false;
        }
    }

    !tokens.is_empty()
}
fn svg_transform_is_bounded(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut cursor = 0usize;
    let mut transform_count = 0usize;

    while cursor < bytes.len() {
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor == bytes.len() {
            break;
        }

        let remaining = &value[cursor..];
        let (name, arity_min, arity_max) = if remaining.starts_with("translate") {
            ("translate", 1usize, 2usize)
        } else if remaining.starts_with("rotate") {
            ("rotate", 1usize, 3usize)
        } else if remaining.starts_with("scale") {
            ("scale", 1usize, 2usize)
        } else {
            return false;
        };

        cursor += name.len();
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor >= bytes.len() || bytes[cursor] != b'(' {
            return false;
        }
        cursor += 1;

        let start = cursor;
        while cursor < bytes.len() && bytes[cursor] != b')' {
            if bytes[cursor] == b'(' {
                return false;
            }
            cursor += 1;
        }
        if cursor >= bytes.len() {
            return false;
        }

        let args = &value[start..cursor];
        let numbers = svg_numeric_tokens(args)?;
        let arg_count = numbers.len();
        if arg_count < arity_min || arg_count > arity_max {
            return false;
        }
        if name == "scale" && numbers.iter().any(|value| value.abs() > 8.0) {
            return false;
        }

        cursor += 1;
        transform_count = transform_count.checked_add(1).unwrap_or(usize::MAX);
        if transform_count > MAX_PORTRAIT_TRANSFORMS {
            return false;
        }
    }

    transform_count > 0
}

fn svg_color_is_bounded(value: &str) -> bool {
    if value == "none" {
        return true;
    }

    if value.starts_with('#')
        && (value.len() == 4 || value.len() == 7)
        && value[1..].chars().all(|c| c.is_ascii_hexdigit())
    {
        return true;
    }

    let Some(args) = value.strip_prefix("rgba(").and_then(|rest| rest.strip_suffix(')')) else {
        return false;
    };
    let components = args.split(',').map(str::trim).collect::<Vec<_>>();
    if components.len() != 4 {
        return false;
    }

    let rgb_ok = components[..3].iter().all(|component| {
        !component.is_empty()
            && component.len() <= 3
            && component.chars().all(|c| c.is_ascii_digit())
            && component
                .parse::<u16>()
                .map(|value| value <= 255)
                .unwrap_or(false)
    });
    let alpha_ok = !components[3].is_empty()
        && components[3].len() <= 32
        && components[3]
            .parse::<f64>()
            .map(|value| value.is_finite() && (0.0..=1.0).contains(&value))
            .unwrap_or(false);

    rgb_ok && alpha_ok
}

/// Extract the live cognitive self-portrait SVG as an image data URL.
///
/// The gateway payload is remote data. Rendering it through an image element keeps
/// SVG markup out of the application DOM. The validator below is deliberately
/// stricter than a denylist: only inert geometric tags and explicitly safe
/// attributes are accepted, and the XML structure must be well formed.
fn portrait_from_json(v: &Value) -> Option<String> {
    let svg = v.get("canvas_svg")?.as_str()?.trim();
    if svg.len() > 512 * 1024
        || !svg.starts_with("<svg")
        || !svg.ends_with("</svg>")
    {
        return None;
    }

    // The byte-size cap bounds input, but a hostile document can still
    // consume disproportionate parser work through deeply nested elements
    // or huge attribute sets. Keep structural complexity bounded as well.
    const MAX_PORTRAIT_ELEMENTS: usize = 256;
    const MAX_PORTRAIT_NESTING: usize = 32;
    const MAX_PORTRAIT_ATTRIBUTES: usize = 1024;
    const MAX_PORTRAIT_ATTRIBUTE_BYTES: usize = 128 * 1024;

    let mut stack = Vec::<String>::new();
    let mut seen_root = false;
    let mut element_count = 0usize;
    let mut attribute_count = 0usize;
    let mut attribute_bytes = 0usize;
    let mut cursor = 0usize;

    while cursor < svg.len() {
        let open = svg[cursor..].find('<')? + cursor;

        // Text is permitted only inside the descriptive elements. Elsewhere
        // only whitespace is accepted, preventing hidden markup-like payloads
        // from being smuggled through the scanner.
        let text = &svg[cursor..open];
        if !text.trim().is_empty()
            && !matches!(stack.last().map(String::as_str), Some("title" | "desc"))
        {
            return None;
        }

        let remainder = &svg[open..];
        let mut quote = None;
        let mut end = None;
        for (offset, byte) in remainder.bytes().enumerate().skip(1) {
            match quote {
                Some(q) if byte == q => quote = None,
                None if byte == b'"' || byte == b'\'' => quote = Some(byte),
                None if byte == b'>' => {
                    end = Some(offset);
                    break;
                }
                _ => {}
            }
        }
        let end = end?;
        let token = &remainder[..=end];
        if token.starts_with("<!--")
            || token.starts_with("<![")
            || token.starts_with("<?")
            || token.starts_with("<!")
        {
            return None;
        }

        let mut body = token[1..token.len() - 1].trim();
        let closing = body.starts_with('/');
        if closing {
            body = body[1..].trim_start();
        }
        let self_closing = !closing && body.ends_with('/');
        if self_closing {
            body = body[..body.len() - 1].trim_end();
        }

        let name_end = body
            .find(|c: char| c.is_ascii_whitespace() || c == '/')
            .unwrap_or(body.len());
        let name = &body[..name_end];
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphabetic()) {
            return None;
        }
        element_count = element_count.checked_add(1)?;
        if element_count > MAX_PORTRAIT_ELEMENTS {
            return None;
        }

        const ALLOWED_TAGS: &[&str] = &[
            "svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline",
            "polygon", "title", "desc",
        ];
        if !ALLOWED_TAGS.contains(&name) {
            return None;
        }

        if closing {
            if self_closing || !body[name_end..].trim().is_empty() {
                return None;
            }
            if stack.pop().as_deref() != Some(name) {
                return None;
            }
            cursor = open + end + 1;
            continue;
        }

        if name == "svg" {
            if seen_root || !stack.is_empty() {
                return None;
            }
            seen_root = true;
        } else if stack.is_empty() {
            return None;
        } else if matches!(stack.last().map(String::as_str), Some("title" | "desc")) {
            // Title and desc are text-only in this projection contract.
            // Reject child elements rather than relying on browser recovery.
            return None;
        }

        if !self_closing {
            if stack.len() >= MAX_PORTRAIT_NESTING {
                return None;
            }
        }

        let attrs = &body[name_end..];
        let mut rest = attrs.trim();
        let mut seen_attrs = Vec::<String>::new();
        while !rest.is_empty() {
            attribute_count = attribute_count.checked_add(1)?;
            if attribute_count > MAX_PORTRAIT_ATTRIBUTES {
                return None;
            }

            let key_end = rest
                .find(|c: char| c.is_ascii_whitespace() || c == '=')
                .unwrap_or(rest.len());
            let key = &rest[..key_end];
            let lower_key = key.to_ascii_lowercase();
            if key.is_empty() || key.contains(':') || lower_key.starts_with("on") {
                return None;
            }
            if seen_attrs.iter().any(|existing| existing == key) {
                return None;
            }
            seen_attrs.push(key.to_string());
            rest = rest[key_end..].trim_start();
            if !rest.starts_with('=') {
                return None;
            }
            rest = rest[1..].trim_start();
            let quote = rest.as_bytes().first().copied()?;
            if quote != b'"' && quote != b'\'' {
                return None;
            }
            rest = &rest[1..];
            let value_end = rest.find(quote as char)?;
            let value = &rest[..value_end];
            rest = rest[value_end + 1..].trim_start();

            attribute_bytes = attribute_bytes
                .checked_add(key.len())?
                .checked_add(value.len())?;
            if attribute_bytes > MAX_PORTRAIT_ATTRIBUTE_BYTES {
                return None;
            }

            let allowed = match name {
                "svg" => matches!(key, "viewBox" | "width" | "height" | "xmlns"),
                "g" | "path" | "rect" | "circle" | "ellipse" | "line"
                | "polyline" | "polygon" => matches!(
                    key,
                    "id"
                        | "transform"
                        | "fill"
                        | "stroke"
                        | "stroke-width"
                        | "opacity"
                        | "d"
                        | "x"
                        | "y"
                        | "width"
                        | "height"
                        | "rx"
                        | "ry"
                        | "cx"
                        | "cy"
                        | "r"
                        | "x1"
                        | "y1"
                        | "x2"
                        | "y2"
                        | "points"
                ),
                "title" | "desc" => false,
                _ => false,
            };
            if !allowed || value.len() > 16 * 1024 {
                return None;
            }

            // No URL-valued attributes are part of the portrait contract.
            // This also rejects javascript:, data:, fragment indirection,
            // CSS url(), and namespace-based resource references.
            let lower_value = value.to_ascii_lowercase();
            if key != "xmlns"
                && (lower_value.contains("url(")
                    || lower_value.contains("javascript:")
                    || lower_value.contains("data:")
                    || lower_value.contains("http:")
                    || lower_value.contains("https:")
                    || lower_value.contains("xlink:"))
            {
                return None;
            }

            match key.as_str() {
                "id" => {
                    if value.is_empty()
                        || value.len() > 64
                        || !value
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
                    {
                        return None;
                    }
                }
                "xmlns" => {
                    if value != "http://www.w3.org/2000/svg" {
                        return None;
                    }
                }
                "fill" | "stroke" => {
                    if !svg_color_is_bounded(value) {
                        return None;
                    }
                }
                "transform" => {
                    if !svg_transform_is_bounded(value) {
                        return None;
                    }
                }
                "viewBox" => {
                    if !svg_viewbox_is_bounded(value) {
                        return None;
                    }
                }
                "width" | "height" | "rx" | "ry" | "r" | "stroke-width" => {
                    if !svg_single_nonnegative_number_is_bounded(value) {
                        return None;
                    }
                }
                "opacity" => {
                    if !svg_single_unit_number_is_bounded(value) {
                        return None;
                    }
                }
                "cx" | "cy" | "x" | "y" | "x1" | "y1" | "x2" | "y2" => {
                    if !svg_single_number_is_bounded(value) {
                        return None;
                    }
                }
                "points" => {
                    if !svg_points_is_bounded(value) {
                        return None;
                    }
                }
                "d" => {
                    if !svg_path_data_is_bounded(value) {
                        return None;
                    }
                }
                _ => {}
            }
        }

        if !seen_root {
            return None;
        }
        if !self_closing {
            stack.push(name);
        }
        cursor = open + end + 1;
    }

    if !seen_root || !stack.is_empty() {
        return None;
    }

    let encoded = base64::engine::general_purpose::STANDARD.encode(svg.as_bytes());
    Some(format!("data:image/svg+xml;base64,{encoded}"))
}

#[component]
pub fn App() -> impl IntoView {
    let gateway = RwSignal::new(DEFAULT_GATEWAY.to_string());
    let ws_connected = RwSignal::new(false);
    let vitals = RwSignal::new(Vitals::default());
    let telemetry_count = RwSignal::new(0_u64);
    let last_cycle = RwSignal::new(0_u64);

    let history = RwSignal::new(Vec::<Turn>::new());
    let draft = RwSignal::new(String::new());
    let sending = RwSignal::new(false);
    let last_error = RwSignal::new(Option::<String>::None);
    // Request generations prevent a response from an obsolete gateway/session
    // from mutating the current conversation or clearing a newer send state.
    let request_generation = Rc::new(Cell::new(0_u64));
    // Status responses cross an async boundary too. Keep a generation so a
    // response from the previous gateway cannot overwrite the new gateway's
    // liveness state if it resolves during a session transition.
    let status_generation = Rc::new(Cell::new(0_u64));
    let daemon_status = RwSignal::new(Option::<Value>::None);
    let events = RwSignal::new(Vec::<CognitiveEvent>::new());
    // Pauses the presentation of the event stream, not the daemon itself.
    // Incoming telemetry continues so resuming returns to current truth.
    let timeline_paused = RwSignal::new(false);

    // This is a derived semantic value, not another mutable state machine.
    // The memo only changes when presence/mode changes, so assistive
    // technology is not fed the continuous telemetry stream.
    let semantic_key = Memo::new(move |_| {
        let v = vitals.get();
        let state = CognitiveState::from_observation(
            ws_connected.get(),
            sending.get(),
            v.coherence,
            v.thermodynamic_load,
            v.reasoning_confidence as f64,
            v.prediction_error as f64,
        );
        format!("{}:{}", state.presence.label(), state.mode.label())
    });
    let semantic_announcement = Memo::new(move |_| {
        let key = semantic_key.get();
        let (presence, mode) = key.split_once(':').unwrap_or(("unknown", "unknown"));
        format!("Symthaea is {presence} and {mode}.")
    });

    // Projection exits (VISION_PROJECTION_REVIEW_2026-07-15.md P1.2): the
    // live cognitive self-portrait and the imagination decode, previously
    // produced every applicable cycle and displayed nowhere.
    let portrait = RwSignal::new(Option::<String>::None);
    let movie = RwSignal::new(Option::<Movie>::None);
    let movie_frame = RwSignal::new(0_usize);
    let movie_canvas = NodeRef::<leptos::html::Canvas>::new();

    // The gateway is a session boundary: changing it starts a new telemetry
    // generation. Owner-scoped cancellation handles the normal teardown;
    // the generation guard additionally makes stale callbacks harmless if
    // they race with a reconnect or gateway change.
    let telemetry_generation = Rc::new(Cell::new(0_u64));
    // Event identity spans telemetry sessions because the retained timeline
    // survives reconnects and gateway changes. Resetting this counter inside
    // each WebSocket task would allow old and new events to share a key.
    let event_sequence = Rc::new(Cell::new(0_u64));

    Effect::new(move |_| {
        let gw = gateway.get();
        // Gateway changes are authority changes, not merely reconnects.
        // Clear gateway-derived presentation state immediately so the UI
        // cannot display the previous service's cognition while the new
        // session is still negotiating or has not produced a fresh sample.
        ws_connected.set(false);
        vitals.set(Vitals::default());
        telemetry_count.set(0);
        last_cycle.set(0);
        daemon_status.set(None);
        portrait.set(None);
        movie.set(None);
        movie_frame.set(0);
        // Cycle identifiers are daemon-local; do not carry the previous
        // session's cycle into a new session that has not emitted telemetry.
        last_cycle.set(0);
        let session = {
            let mut generation = telemetry_generation.get();
            let Some(session) = next_telemetry_session(&mut generation) else {
                leptos::logging::error!("telemetry session generation exhausted; refusing to reuse identity");
                return;
            };
            telemetry_generation.set(generation);
            session
        };
        let telemetry_callback_generation = Rc::clone(&telemetry_generation);
        let connected_callback_generation = Rc::clone(&telemetry_generation);
        let disconnect_generation = Rc::clone(&telemetry_generation);
        let telemetry_gateway = gateway;
        let message_gateway = gw.clone();
        let connected_gateway = gw.clone();
        let disconnect_gateway = gw.clone();
        let telemetry_sequence = Rc::clone(&event_sequence);
        let connected_sequence = Rc::clone(&event_sequence);
        let disconnect_sequence = Rc::clone(&event_sequence);
        spawn_local_scoped_with_cancellation(async move {
            let mut previous_state: Option<CognitiveState> = None;
            let connected = api::stream_telemetry(
                &gw,
                move |payload| {
                    if !telemetry_session_is_current(
                        telemetry_callback_generation.get(),
                        session,
                    ) || telemetry_gateway.get_untracked() != message_gateway
                    {
                        return;
                    }
                    let Some(v) = Vitals::from_json(&payload) else {
                        leptos::logging::warn!(
                            "telemetry payload missing finite required cognitive measurements"
                        );
                        return;
                    };
                    let Some(cycle) = payload["cycle"].as_u64() else {
                        leptos::logging::warn!("telemetry payload missing cycle identifier");
                        return;
                    };
                    let current_state = CognitiveState::from_observation(
                        true,
                        sending.get_untracked(),
                        v.coherence,
                        v.thermodynamic_load,
                        v.reasoning_confidence as f64,
                        v.prediction_error as f64,
                    );
                    // Allocate identity only when a semantic event is actually
                    // emitted. A quiet telemetry sample must not consume event IDs:
                    // sequence numbers identify retained presentation events, not
                    // observation count.
                    let mut sequence_counter = telemetry_sequence.get();
                    if let Some(sequence) = next_event_sequence(&mut sequence_counter) {
                        if let Some(event) = event_between(
                            previous_state,
                            current_state,
                            sequence,
                            cycle,
                            v.surprise_triggered,
                            v.gwt_broadcast,
                        ) {
                            telemetry_sequence.set(sequence_counter);
                            events.update(|items| push_cognitive_event(items, event));
                        }
                    } else {
                        leptos::logging::error!("cognitive event sequence exhausted; suppressing event identity reuse");
                    }
                    previous_state = Some(current_state);
                    last_cycle.set(cycle);
                    vitals.set(v);
                    telemetry_count.update(|n| *n += 1);
                    if let Some(svg) = portrait_from_json(&payload) {
                        portrait.set(Some(svg));
                    }
                    if let Some(m) = Movie::from_json(&payload) {
                        movie.set(Some(m));
                        movie_frame.set(0);
                    }
                },
                move || {
                    if !telemetry_session_is_current(
                        connected_callback_generation.get(),
                        session,
                    ) || telemetry_gateway.get_untracked() != connected_gateway
                    {
                        return;
                    }
                    ws_connected.set(true);
                    let mut sequence_counter = connected_sequence.get();
                    if let Some(sequence) = next_event_sequence(&mut sequence_counter) {
                        connected_sequence.set(sequence_counter);
                        events.update(|items| {
                            push_cognitive_event(
                                items,
                                CognitiveEvent::lifecycle(sequence, CognitiveEventKind::Connected, 0),
                            );
                        });
                    } else {
                        leptos::logging::error!("cognitive event sequence exhausted; suppressing connected marker");
                    }
                },
            )
            .await;
            if connected
                && telemetry_session_is_current(
                    disconnect_generation.get(),
                    session,
                )
                && telemetry_gateway.get_untracked() == disconnect_gateway
            {
                ws_connected.set(false);
                let mut sequence_counter = disconnect_sequence.get();
                if let Some(sequence) = next_event_sequence(&mut sequence_counter) {
                    disconnect_sequence.set(sequence_counter);
                    let cycle = last_cycle.get_untracked();
                    events.update(|items| {
                        push_cognitive_event(
                            items,
                            CognitiveEvent::lifecycle(
                                sequence,
                                CognitiveEventKind::Disconnected,
                                cycle,
                            ),
                        );
                    });
                } else {
                    leptos::logging::error!("cognitive event sequence exhausted; suppressing disconnected marker");
                }
            }
        });
    });

    // Gateway changes also invalidate in-flight conversation requests.
    // The old future may still resolve, but its response is no longer
    // authoritative for the current gateway generation.
    {
        let request_generation = Rc::clone(&request_generation);
        Effect::new(move |_| {
            let _gateway_generation = gateway.get();
            let mut generation = request_generation.get();
            if next_authority_generation(&mut generation).is_some() {
                request_generation.set(generation);
                sending.set(false);
            } else {
                leptos::logging::error!(
                    "conversation request generation exhausted; refusing to reuse authority"
                );
                sending.set(false);
                last_error.set(Some(
                    "conversation request authority exhausted; reload required".to_string(),
                ));
            }
        });
    }

    // Poll GET-equivalent /v1/service status every 5s. This is baseline
    // liveness feedback independent of the telemetry WS above, which stays
    // silent whenever the daemon's experience bridge is off (the common
    // case in production) — without this, an idle daemon would look
    // indistinguishable from an unreachable one.
    Effect::new(move |_| {
        // Track the gateway so changing it starts a fresh status loop; the
        // previous owner-scoped loop is cancelled on effect cleanup.
        let _gateway_generation = gateway.get();
        let status_session = {
            let mut generation = status_generation.get();
            let Some(next) = next_authority_generation(&mut generation) else {
                leptos::logging::error!(
                    "status generation exhausted; refusing to reuse authority"
                );
                return;
            };
            status_generation.set(generation);
            next
        };
        let status_authority = Rc::clone(&status_generation);
        spawn_local_scoped_with_cancellation(async move {
            loop {
                let gw = gateway.get_untracked();
                let response = api::send_simple(&gw, "status").await;
                // Owner cancellation handles normal teardown; the generation
                // and direct gateway check together cover a response that
                // races the gateway change before the dependent effect reruns.
                if status_authority.get() != status_session
                    || gateway.get_untracked() != gw
                {
                    return;
                }
                match response {
                    Ok(resp) if resp["type"] != "error" => daemon_status.set(Some(resp)),
                    _ => daemon_status.set(None),
                }
                gloo_timers::future::TimeoutFuture::new(5_000).await;
            }
        });
    });

    // Advance the imagination loop at ~3fps whenever a movie is present.
    Effect::new(move |_| {
        spawn_local_scoped_with_cancellation(async move {
            loop {
                gloo_timers::future::TimeoutFuture::new(300).await;
                if movie.with_untracked(|m| m.as_ref().is_some_and(|m| m.frames_rgba.len() > 1)) {
                    movie_frame.update(|i| *i = i.wrapping_add(1));
                }
            }
        });
    });

    // Draw the current imagination frame whenever the movie or frame index
    // changes. putImageData wants RGBA at native size; CSS scales it up with
    // image-rendering: pixelated.
    Effect::new(move |_| {
        let idx = movie_frame.get();
        let Some(canvas) = movie_canvas.get() else {
            return;
        };
        movie.with(|m| {
            let Some(m) = m.as_ref() else { return };
            canvas.set_width(m.width);
            canvas.set_height(m.height);
            let Some(ctx) = canvas
                .get_context("2d")
                .ok()
                .flatten()
                .and_then(|c| c.dyn_into::<web_sys::CanvasRenderingContext2d>().ok())
            else {
                return;
            };
            let frame = &m.frames_rgba[idx % m.frames_rgba.len()];
            if let Ok(img) =
                web_sys::ImageData::new_with_u8_clamped_array(wasm_bindgen::Clamped(frame), m.width)
            {
                let _ = ctx.put_image_data(&img, 0.0, 0.0);
            }
        });
    });

    let send = move || {
        let text = draft.get_untracked();
        if text.trim().is_empty() || sending.get_untracked() {
            return;
        }
        let gw = gateway.get_untracked();
        let request_id = {
            let mut generation = request_generation.get();
            let Some(next) = next_authority_generation(&mut generation) else {
                leptos::logging::error!(
                    "conversation request generation exhausted; refusing to reuse authority"
                );
                last_error.set(Some(
                    "conversation request authority exhausted; reload required".to_string(),
                ));
                return;
            };
            request_generation.set(generation);
            next
        };
        draft.set(String::new());
        sending.set(true);
        last_error.set(None);
        history.update(|h| {
            h.push(Turn {
                role: "you",
                text: text.clone(),
            })
        });
        let request_generation = Rc::clone(&request_generation);
        spawn_local_scoped_with_cancellation(async move {
            let response = api::send_query(&gw, &text).await;
            if request_generation.get() != request_id
                || gateway.get_untracked() != gw
            {
                return;
            }
            match response {
                Ok(resp) => {
                    if resp["type"] == "error" {
                        let msg = resp["message"]
                            .as_str()
                            .unwrap_or("unknown error")
                            .to_string();
                        last_error.set(Some(msg));
                    } else {
                        let content = resp["content"].as_str().unwrap_or("").to_string();
                        history.update(|h| {
                            h.push(Turn {
                                role: "symthaea",
                                text: content,
                            })
                        });
                    }
                }
                Err(e) => last_error.set(Some(e)),
            }
            sending.set(false);
        });
    };

    let visible_spans = move || {
        let snapshot = if timeline_paused.get() {
            events.get_untracked()
        } else {
            events.get()
        };
        cognitive_spans(&snapshot)
            .into_iter()
            .rev()
            .take(4)
            .collect::<Vec<_>>()
    };

    view! {
        <div class="shell">
            <header class="shell-header">
                <h1>"Symthaea"</h1>
                <div class="gateway-row">
                    <label for="gateway">"gateway"</label>
                    <input id="gateway" type="text"
                        prop:value=move || gateway.get()
                        on:change=move |ev| gateway.set(event_target_value(&ev))
                    />
                    <span class=move || if ws_connected.get() { "dot dot-live" } else { "dot dot-dark" }
                        title=move || if ws_connected.get() { "telemetry connected" } else { "telemetry disconnected" }
                    ></span>
                </div>
                <div class="daemon-status">
                    {move || match daemon_status.get() {
                        Some(s) => format!(
                            "up {}s · {} requests · {} sleep cycles",
                            s["uptime_seconds"].as_u64().unwrap_or(0),
                            s["requests_processed"].as_u64().unwrap_or(0),
                            s["sleep_cycles"].as_u64().unwrap_or(0),
                        ),
                        None => "daemon unreachable".to_string(),
                    }}
                </div>
            </header>

            <section class="cognitive-state" aria-label="Cognitive state">
                {move || {
                    let v = vitals.get();
                    let state = CognitiveState::from_observation(
                        ws_connected.get(),
                        sending.get(),
                        v.coherence,
                        v.thermodynamic_load,
                        v.reasoning_confidence as f64,
                        v.prediction_error as f64,
                    );
                    view! {
                        <span class="sr-only" aria-live="polite">{move || semantic_announcement.get()}</span>
                        <div class="state-presence">
                            <span class="state-mode">{state.mode.label()}</span>
                            <span class="state-separator">" · "</span>
                            <span class="state-presence-label">{state.presence.label()}</span>
                        </div>
                        <p class="state-description">
                            {format!(
                                "coherence {:.2} · load {:.2} · confidence {:.2} · prediction error {:.2}",
                                state.coherence, state.thermodynamic_load, state.confidence, state.prediction_error
                            )}
                        </p>
                    }
                }}
            </section>

            <section class="vitals">
                <div class="orb-wrap">
                    <div class="orb"
                        style:animation-duration=move || {
                            let load = vitals.get().thermodynamic_load.clamp(0.0, 1.0);
                            format!("{:.2}s", 2.6 - (load as f64 * 1.8))
                        }
                        style:background=move || {
                            let v = vitals.get();
                            let warm = ((v.valence + 1.0) / 2.0).clamp(0.0, 1.0) as f64;
                            let hue = 260.0 - warm * 220.0;
                            format!("radial-gradient(circle at 35% 30%, hsl({hue:.0} 90% 70%), hsl({hue:.0} 70% 35%))")
                        }
                    ></div>
                </div>
                <dl class="readouts">
                    <div><dt>"integration measure"</dt><dd>{move || format!("{:.2}", vitals.get().consciousness_level)}</dd></div>
                    <div><dt>"valence"</dt><dd>{move || format!("{:+.2}", vitals.get().valence)}</dd></div>
                    <div><dt>"arousal"</dt><dd>{move || format!("{:.2}", vitals.get().arousal)}</dd></div>
                    <div><dt>"mood temp"</dt><dd>{move || format!("{:.2}", vitals.get().mood_temperature)}</dd></div>
                    <div><dt>"thermodynamic load"</dt><dd>{move || format!("{:.2}", vitals.get().thermodynamic_load)}</dd></div>
                    <div><dt>"moral score"</dt><dd>{move || format!("{:.2}", vitals.get().moral_score)}</dd></div>
                    <div><dt>"coherence"</dt><dd>{move || format!("{:.2}", vitals.get().coherence)}</dd></div>
                    <div><dt>"reasoning confidence"</dt><dd>{move || format!("{:.2}", vitals.get().reasoning_confidence)}</dd></div>
                    <div><dt>"prediction error"</dt><dd>{move || format!("{:.2}", vitals.get().prediction_error)}</dd></div>
                    <div><dt>"gwt broadcast"</dt><dd>{move || if vitals.get().gwt_broadcast { "yes" } else { "no" }}</dd></div>
                    <div><dt>"dream insights"</dt><dd>{move || vitals.get().dream_insights.to_string()}</dd></div>
                    <div><dt>"surprise"</dt><dd>{move || if vitals.get().surprise_triggered { "triggered" } else { "—" }}</dd></div>
                    <div><dt>"cycles observed"</dt><dd>{move || telemetry_count.get().to_string()}</dd></div>
                </dl>
                <p class="vitals-note">
                    "Telemetry only flows once the daemon's experience bridge is enabled "
                    "(--experience-bridge) and a query has been sent — it is turn-synchronous, "
                    "not an idle clock."
                </p>
            </section>

            <section class="cognitive-timeline" aria-label="Cognitive timeline">
                <div class="timeline-header">
                    <div>
                        <h2>"recent events"</h2>
                        <span class="timeline-state">
                            {move || if timeline_paused.get() { "paused view" } else { "live view" }}
                        </span>
                    </div>
                    <div class="timeline-controls">
                        <span class="timeline-count">{
                            move || {
                                if timeline_paused.get() {
                                    events.get_untracked().len().to_string()
                                } else {
                                    events.get().len().to_string()
                                }
                            }
                        }</span>
                        <button
                            type="button"
                            class="timeline-pause"
                            aria-pressed=move || timeline_paused.get().to_string()
                            on:click=move |_| timeline_paused.update(|paused| *paused = !*paused)
                        >
                            {move || if timeline_paused.get() { "resume" } else { "pause" }}
                        </button>
                    </div>
                </div>
                {move || {
                    let spans = visible_spans();
                    if spans.is_empty() {
                        None
                    } else {
                        Some(view! {
                            <div class="timeline-spans" aria-label="Observed cognitive spans">
                                <span class="timeline-spans-label">"cycle spans"</span>
                                <For
                                    each=move || spans.clone().into_iter()
                                    key=|span| span.start_sequence
                                    children=move |span| {
                                        let range = match span.end_cycle {
                                            Some(end) => format!("cycle {}–{}", span.start_cycle, end),
                                            None => format!("cycle {}–open", span.start_cycle),
                                        };
                                        view! {
                                            <span class="timeline-span">
                                                <span class="timeline-span-kind">{span.kind.label()}</span>
                                                <span class="timeline-span-range">{range}</span>
                                            </span>
                                        }
                                    }
                                />
                                <span class="timeline-spans-note">"open means open within the retained event window"</span>
                            </div>
                        })
                    }
                }}
                <div class="timeline-list" role="log" aria-live="off">
                    <For
                        each=move || if timeline_paused.get() {
                            events.get_untracked().into_iter().rev().enumerate().collect::<Vec<_>>()
                        } else {
                            events.get().into_iter().rev().enumerate().collect::<Vec<_>>()
                        }
                        key=|(_, event)| event.sequence
                        children=move |(_, event)| {
                            view! {
                                <div class="timeline-event">
                                    <span class="timeline-kind">{event.kind.label()}</span>
                                    <span class="timeline-cycle">{format!("cycle {}", event.cycle)}</span>
                                    <span class="timeline-basis">{event.evidence_basis.label()}</span>
                                    {move || match event.kind {
                                        CognitiveEventKind::SurpriseDetected => Some(view! {
                                            <span class="timeline-evidence">
                                                {format!("prediction error {:.2}", event.prediction_error)}
                                            </span>
                                        }),
                                        CognitiveEventKind::EnteredRest | CognitiveEventKind::ExitedRest => Some(view! {
                                            <span class="timeline-evidence">
                                                {format!("load {:.2}", event.thermodynamic_load)}
                                            </span>
                                        }),
                                        _ => None,
                                    }}
                                </div>
                            }
                        }
                    />
                    {move || if events.with(|items| items.is_empty()) {
                        Some(view! {
                            <p class="timeline-empty">"No semantic events yet."</p>
                        })
                    } else {
                        None
                    }}
                </div>
            </section>

            // Projections: what she renders of herself. Panes appear only
            // once the corresponding stream has actually delivered content.
            <section class="projections" style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">
                {move || portrait.get().map(|portrait_src| view! {
                    <div class="projection-pane">
                        <h2 style="font-size:0.9em;opacity:0.7;">"self-portrait"</h2>
                        <img class="portrait" style="max-width:220px;" src=portrait_src
                            alt="Live cognitive self-portrait" />
                    </div>
                })}
                <div class="projection-pane"
                    style:display=move || if movie.with(|m| m.is_some()) { "block" } else { "none" }
                >
                    <h2 style="font-size:0.9em;opacity:0.7;">"imagination"</h2>
                    <canvas node_ref=movie_canvas
                        style="width:192px;height:192px;image-rendering:pixelated;border-radius:8px;"
                    ></canvas>
                    {move || movie.with(|m| m.as_ref().map(|m| view! {
                        <p class="movie-note" style="font-size:0.8em;opacity:0.6;">
                            {format!("{} frames · coherence {:.2}", m.frames_rgba.len(), m.semantic_coherence)}
                        </p>
                    }))}
                </div>
            </section>

            <section class="converse">
                <div class="transcript">
                    <For
                        each=move || history.get().into_iter().enumerate()
                        key=|(i, _)| *i
                        children=move |(_, turn): (usize, Turn)| {
                            view! {
                                <div class=format!("turn turn-{}", turn.role)>
                                    <span class="turn-role">{turn.role}</span>
                                    <span class="turn-text">{turn.text}</span>
                                </div>
                            }
                        }
                    />
                </div>
                {move || last_error.get().map(|e| view! { <div class="error">{e}</div> })}
                <form class="composer" on:submit=move |ev| { ev.prevent_default(); send(); }>
                    <input type="text" placeholder="talk to symthaea..."
                        prop:value=move || draft.get()
                        on:input=move |ev| draft.set(event_target_value(&ev))
                    />
                    <button type="submit" disabled=move || sending.get()>
                        {move || if sending.get() { "…" } else { "send" }}
                    </button>
                </form>
            </section>
        </div>
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn valid_payload() -> Value {
        serde_json::json!({
            "cycle": 42,
            "consciousness_level": 0.7,
            "affective_valence": 0.2,
            "affective_arousal": 0.3,
            "mood_temperature": 0.4,
            "thermodynamic_load": 0.5,
            "value_evaluator_score": 0.6,
            "harmonic_field_coherence": 0.8,
            "reasoning_confidence": 0.9,
            "prediction_error": 0.1,
            "gwt_broadcast": false,
            "dream_insights": 0,
            "surprise_triggered": false
        })
    }

    #[test]
    fn telemetry_requires_finite_cognitive_measurements() {
        assert!(Vitals::from_json(&valid_payload()).is_some());
    }

    #[test]
    fn portrait_rejects_active_or_external_svg_content() {
        for svg in [
            r#"<svg><script>alert(1)</script></svg>"#,
            r#"<svg><foreignObject><div>x</div></foreignObject></svg>"#,
            r#"<svg><use href="https://attacker.example/icon.svg#x"/></svg>"#,
            r#"<svg><image href="https://attacker.example/pixel.png"/></svg>"#,
            r#"<svg><rect style="fill:url(https://attacker.example/pixel.svg#x)"/></svg>"#,
            r#"<svg><rect onclick="alert(1)"/></svg>"#,
            r#"<svg><style>.x{fill:red}</style><rect/></svg>"#,
        ] {
            let payload = serde_json::json!({ "canvas_svg": svg });
            assert!(portrait_from_json(&payload).is_none(), "accepted: {svg}");
        }
    }

    #[test]
    fn portrait_accepts_inert_geometric_svg() {
        let payload = serde_json::json!({
            "canvas_svg": r#"<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg"><g id="root" opacity="0.8" transform="translate(1,2) rotate(3) scale(1)"><circle cx="5" cy="5" r="4" fill="#fff"/></g></svg>"#
        });
        assert!(portrait_from_json(&payload).is_some());
    }

    #[test]
    fn portrait_rejects_case_folded_svg_markup() {
        for svg in [
            r#"<SVG><circle cx="1" cy="1" r="1"/></SVG>"#,
            r#"<svg VIEWBOX="0 0 2 2"><circle cx="1" cy="1" r="1"/></svg>"#,
        ] {
            assert!(portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none(), "accepted: {svg}");
        }
    }

    #[test]
    fn portrait_rejects_nested_descriptive_elements() {
        let payload = serde_json::json!({
            "canvas_svg": r#"<svg><title><g/></title></svg>"#
        });
        assert!(portrait_from_json(&payload).is_none());
    }

    #[test]
    fn portrait_rejects_unbounded_numeric_geometry() {
        for svg in [
            r#"<svg><circle cx="1e9999" cy="0" r="1"/></svg>"#,
            r#"<svg><circle cx="1000001" cy="0" r="1"/></svg>"#,
            r#"<svg><rect x="0" y="0" width="1.2.3" height="1"/></svg>"#,
            r#"<svg><path d="M0 0 L1e9999 2"/></svg>"#,
        ] {
            assert!(portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none(), "accepted: {svg}");
        }
    }

    #[test]
    fn portrait_rejects_empty_parameter_path_commands() {
        for svg in [
            r#"<svg><path d="M"/></svg>"#,
            r#"<svg><path d="M0 0 L"/></svg>"#,
            r#"<svg><path d="M0 0 C"/></svg>"#,
            r#"<svg><path d="M0 0 A"/></svg>"#,
        ] {
            assert!(portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none(), "accepted: {svg}");
        }
    }

    #[test]
    fn portrait_rejects_incomplete_or_wrong_arity_path_commands() {
        for svg in [
            r#"<svg><path d="M 0"/></svg>"#,
            r#"<svg><path d="C 0 0 1 1 2"/></svg>"#,
            r#"<svg><path d="A 10 10 0 0 1 20"/></svg>"#,
            r#"<svg><path d="H 10 20 30"/></svg>"#,
            r#"<svg><path d="Z 10"/></svg>"#,
            r#"<svg><path d="M 0 0 1"/></svg>"#,
        ] {
            assert!(portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none(), "accepted: {svg}");
        }
    }

    #[test]
    fn portrait_rejects_non_binary_arc_flags() {
        for svg in [
            r#"<svg><path d="M0 0 A5 5 0 2 0 10 10"/></svg>"#,
            r#"<svg><path d="M0 0 A5 5 0 0.0 1 10 10"/></svg>"#,
            r#"<svg><path d="M0 0 A5 5 0 +0 1 10 10"/></svg>"#,
            r#"<svg><path d="M0 0 A5 5 0 0 1.0 10 10"/></svg>"#,
        ] {
            assert!(
                portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none(),
                "accepted: {svg}"
            );
        }
    }

    #[test]
    fn portrait_accepts_canonical_binary_arc_flags() {
        for svg in [
            r#"<svg><path d="M0 0 A5 5 0 0 0 10 10"/></svg>"#,
            r#"<svg><path d="M0 0 A5 5 0 0 1 10 10"/></svg>"#,
            r#"<svg><path d="M0 0 A5 5 0 1 0 10 10"/></svg>"#,
            r#"<svg><path d="M0 0 A5 5 0 1 1 10 10"/></svg>"#,
        ] {
            assert!(
                portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_some(),
                "rejected: {svg}"
            );
        }
    }

    #[test]
    fn portrait_accepts_exponent_signs_inside_path_numbers() {
        let payload = serde_json::json!({
            "canvas_svg": r#"<svg><path d="M1e+3 0 L2e-2 3e1"/></svg>"#
        });
        assert!(portrait_from_json(&payload).is_some());
    }

    #[test]
    fn portrait_accepts_leading_negative_path_coordinates() {
        let payload = serde_json::json!({
            "canvas_svg": r#"<svg><path d="M-10-20 L-1.5,-2.5 z"/></svg>"#
        });
        assert!(portrait_from_json(&payload).is_some());
    }

    #[test]
    fn portrait_accepts_repeated_and_implicit_move_to_path_commands() {
        for svg in [
            r#"<svg><path d="M 0 0 10 10 20 0"/></svg>"#,
            r#"<svg><path d="m0 0 10-5 20,5"/></svg>"#,
            r#"<svg><path d="M0 0 H10 V20 L0 20 Z"/></svg>"#,
            r#"<svg><path d="M0 0 C1 2 3 4 5 6 S7 8 9 10 Q11 12 13 14 T15 16 A4 4 0 0 1 19 20"/></svg>"#,
        ] {
            assert!(portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_some(), "rejected: {svg}");
        }
    }

    #[test]
    fn portrait_rejects_excessive_scale_amplification() {
        for svg in [
            r#"<svg><g transform="scale(8.0001)"/></svg>"#,
            r#"<svg><g transform="scale(-8.0001)"/></svg>"#,
            r#"<svg><g transform="scale(9 1)"/></svg>"#,
        ] {
            assert!(
                portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none(),
                "accepted: {svg}"
            );
        }
    }

    #[test]
    fn portrait_rejects_malformed_or_excessive_transform_grammar() {
        for svg in [
            r#"<svg><g transform="translate(1 2 3)"/></svg>"#,
            r#"<svg><g transform="rotate(1 2 3 4)"/></svg>"#,
            r#"<svg><g transform="scale(1e9999)"/></svg>"#,
            r#"<svg><g transform="translate(1) rotate(2)"/></svg> trailing"#,
            r#"<svg><g transform="translate((1))"/></svg>"#,
        ] {
            assert!(portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none(), "accepted: {svg}");
        }
    }

    #[test]
    fn portrait_preserves_svg_adjacent_sign_number_syntax() {
        let payload = serde_json::json!({
            "canvas_svg": r#"<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg"><path d="M10-5L2.5-3.25z"/></svg>"#
        });
        assert!(portrait_from_json(&payload).is_some());
    }

    #[test]
    fn portrait_accepts_bounded_numeric_and_transform_grammar() {
        let payload = serde_json::json!({
            "canvas_svg": r#"<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg"><g transform="translate(1,-2) rotate(3) scale(1,0.5)"><path d="M 0,0 L10-5 z"/></g></svg>"#
        });
        assert!(portrait_from_json(&payload).is_some());
    }

    #[test]
    fn portrait_rejects_unsafe_or_malformed_attributes() {
        for svg in [
            r#"<svg><circle cx="1" onload="alert(1)" r="2"/></svg>"#,
            r#"<svg><circle cx="1" style="fill:red" r="2"/></svg>"#,
            r#"<svg><circle cx="1" fill="url(#evil)" r="2"/></svg>"#,
            r#"<svg><circle cx="1" fill="https://attacker.example/x.svg" r="2"/></svg>"#,
            r#"<svg><circle cx="1" cx="2" r="2"/></svg>"#,
            r#"<svg><circle cx="1" r="2"></svg>"#,
            r#"<svg><circle cx="1" r="2>"#,
        ] {
            let payload = serde_json::json!({ "canvas_svg": svg });
            assert!(portrait_from_json(&payload).is_none(), "accepted: {svg}");
        }
    }

    #[test]
    fn portrait_rejects_invalid_svg_path_grammar() {
        for svg in [
            r#"<svg><path d="M0"/></svg>"#,
            r#"<svg><path d="M0 0 L1"/></svg>"#,
            r#"<svg><path d="M0 0 A1 1 0 2 0 5 5"/></svg>"#,
            r#"<svg><path d="L0 0"/></svg>"#,
            r#"<svg><path d="C0 0 1 1 2 2"/></svg>"#,
            r#"<svg><path d="Z"/></svg>"#,
            r#"<svg><path d="M0 0 Z L10 10"/></svg>"#,
            r#"<svg><path d="M0 0 Z C1 1 2 2 3 3"/></svg>"#,
            r#"<svg><path d="M0 0 C1 2 3"/></svg>"#,
        ] {
            assert!(
                portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none(),
                "accepted: {svg}"
            );
        }
    }

    #[test]
    fn portrait_accepts_multiple_moveto_subpaths() {
        for svg in [
            r#"<svg><path d="M0 0 L10 0 Z M20 20 L30 30 Z"/></svg>"#,
            r#"<svg><path d="m0 0 10 0 z m20 20 10 10 z"/></svg>"#,
        ] {
            assert!(
                portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_some(),
                "rejected: {svg}"
            );
        }
    }

    #[test]
    fn portrait_accepts_implicit_moveto_line_repetition() {
        let payload = serde_json::json!({
            "canvas_svg": r#"<svg viewBox="0 0 10 10"><path d="M0 0 10 0 10 10 0 10z"/></svg>"#
        });
        assert!(portrait_from_json(&payload).is_some());
    }

    #[test]
    fn portrait_rejects_invalid_points_arity() {
        for svg in [
            r#"<svg><polyline points="0,0 1"/></svg>"#,
            r#"<svg><polygon points="0 0 1"/></svg>"#,
            r#"<svg><polyline points="0,0 1,1 2"/></svg>"#,
        ] {
            assert!(
                portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none(),
                "accepted: {svg}"
            );
        }
    }

    #[test]
    fn portrait_rejects_unbounded_viewbox_numeric_geometry() {
        for svg in [
            r#"<svg viewBox="0 0 1e9999 10"><circle r="1"/></svg>"#,
            r#"<svg viewBox="0 0 1000001 10"><circle r="1"/></svg>"#,
            r#"<svg viewBox="0 0 1.2.3 10"><circle r="1"/></svg>"#,
        ] {
            assert!(
                portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none(),
                "accepted: {svg}"
            );
        }
    }

    #[test]
    fn portrait_enforces_viewbox_and_scalar_numeric_arity() {
        for svg in [
            r#"<svg viewBox="0 0 10"><circle r="1"/></svg>"#,
            r#"<svg viewBox="0 0 10 10 20"><circle r="1"/></svg>"#,
            r#"<svg><rect width="1 2"/></svg>"#,
            r#"<svg><circle r="1 2"/></svg>"#,
        ] {
            assert!(
                portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none(),
                "accepted: {svg}"
            );
        }
    }

    #[test]
    fn portrait_enforces_canonical_color_grammar() {
        for svg in [
            r#"<svg><circle fill="rgba(256,0,0,0.5)" r="1"/></svg>"#,
            r#"<svg><circle fill="rgba(1,2,3,1.5)" r="1"/></svg>"#,
            r#"<svg><circle fill="rgba(1,2,3)" r="1"/></svg>"#,
            r#"<svg><circle fill="rgba(1 2 3 0.5)" r="1"/></svg>"#,
        ] {
            assert!(
                portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none(),
                "accepted: {svg}"
            );
        }

        for svg in [
            r#"<svg><circle fill="#abc" r="1"/></svg>"#,
            r#"<svg><circle fill="#aabbcc" stroke="rgba(12,34,56,0.25)" r="1"/></svg>"#,
            r#"<svg><circle fill="none" r="1"/></svg>"#,
        ] {
            assert!(
                portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_some(),
                "rejected: {svg}"
            );
        }
    }

    #[test]
    fn remote_canvas_renderer_output_is_accepted_by_hostile_boundary() {
        use symthaea_canvas::color::Color;
        use symthaea_canvas::scene_graph::{
            FilterType, GradientStop, NodeKind, SceneNode, Style, Transform,
        };
        use symthaea_canvas::svg_renderer::render_svg_for_remote_projection;

        let gradient = SceneNode {
            kind: NodeKind::RadialGradient {
                id: "grad".into(),
                stops: vec![GradientStop {
                    offset: 0.0,
                    color: Color::rgba(0.2, 0.4, 0.8, 0.75),
                }],
            },
            transform: Transform::identity(),
            style: Style::default(),
            children: Vec::new(),
        };
        let filter = SceneNode {
            kind: NodeKind::Filter {
                id: "blur".into(),
                filter_type: FilterType::Blur { std_dev: 12.0 },
            },
            transform: Transform::identity(),
            style: Style::default(),
            children: Vec::new(),
        };
        let use_filter = SceneNode {
            kind: NodeKind::UseFilter {
                filter_id: "blur".into(),
            },
            transform: Transform::identity(),
            style: Style::default(),
            children: Vec::new(),
        };

        let child_style = Style {
            fill: Some(Color::rgba(0.1, 0.2, 0.3, 0.5)),
            stroke: Some(Color::rgb(0.9, 0.8, 0.7)),
            stroke_width: Some(2.0),
            opacity: Some(0.8),
            filter: Some("blur".into()),
            ..Style::default()
        };

        let root = SceneNode::group(Some("root"))
            .with_child(gradient)
            .with_child(filter)
            .with_child(use_filter)
            .with_child(
                SceneNode::circle(50.0, 50.0, 10.0)
                    .with_style(child_style.clone())
                    .with_transform(Transform {
                        translate_x: 4.0,
                        translate_y: -2.0,
                        rotate_deg: 15.0,
                        scale: 1.5,
                    }),
            )
            .with_child(SceneNode::ellipse(100.0, 100.0, 20.0, 10.0))
            .with_child(SceneNode::line(0.0, 0.0, 20.0, 30.0))
            .with_child(SceneNode::polygon(
                vec![(0.0, 0.0), (20.0, 0.0), (10.0, 20.0)],
                true,
            ))
            .with_child(SceneNode::rect(5.0, 5.0, 15.0, 10.0))
            .with_child(SceneNode::path(
                "M 0 0 C 1 2 3 4 5 6 S 7 8 9 10 Q 11 12 13 14 T 15 16",
            ))
            .with_child(
                SceneNode::path("M 10 10 A 4 4 0 0 1 18 18").with_style(Style {
                    fill_url: Some("grad".into()),
                    ..Style::default()
                }),
            );

        let svg = render_svg_for_remote_projection(&root);
        assert!(!svg.contains("<radialGradient"));
        assert!(!svg.contains("<filter"));
        assert!(!svg.contains("<style"));
        assert!(!svg.contains("url("));
        assert!(svg.contains(r#"fill="rgba(51,102,204,0.75)""#));
        assert!(portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_some());
    }

    #[test]
    fn actual_canvas_scene_pipeline_output_is_accepted_by_hostile_boundary() {
        use symthaea_canvas::build_scene;
        use symthaea_canvas::color::Color;
        use symthaea_canvas::scene_graph::{SceneNode, Style, Transform};
        use symthaea_canvas::{AestheticEngine, CognitiveSnapshot};
        use symthaea_canvas::svg_renderer::render_svg_for_remote_projection;

        let mut snapshots = vec![CognitiveSnapshot::dormant()];
        let mut active = CognitiveSnapshot::dormant();
        active.consciousness_level = 0.95;
        active.prediction_error = 0.35;
        active.living_mind_vitality = 0.8;
        active.living_mind_coherence = 0.9;
        active.dopamine = 0.9;
        active.noradrenaline = 0.7;
        active.serotonin = 0.8;
        active.acetylcholine = 0.7;
        active.oxytocin = 0.8;
        active.gaba = 0.6;
        active.allostatic_load = 0.4;
        active.betti_0 = 6;
        active.betti_1 = 8;
        active.betti_2 = 2;
        active.persistence_components = vec![
            [0.05, 0.8],
            [0.1, 0.7],
            [0.2, 0.9],
        ];
        active.persistence_cycles = vec![[0.15, 0.85], [0.25, 0.95]];
        active.cantor_metacognitive_depth = 0.9;
        active.cantor_last_depth = 6;
        active.valence = 0.7;
        active.arousal = 0.9;
        active.harmony_activations = [0.2, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3];
        active.thought_vector = vec![0.8, -0.6];
        active.cycle_count = 999;
        snapshots.push(active);

        for snapshot in snapshots {
            let mut engine = AestheticEngine::new();
            let state = engine.process(&snapshot);
            let scene = build_scene(&state);
            let svg = render_svg_for_remote_projection(&scene);

            assert!(svg.len() <= 512 * 1024);
            assert!(portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_some());
        }

        // Keep one explicit styled node in this contract corpus too: the
        // producer must flatten/sanitize these fields to the same grammar the
        // hostile consumer accepts.
        let styled = SceneNode::circle(16.0, 16.0, 4.0)
            .with_style(Style {
                fill: Some(Color::rgba(0.25, 0.5, 0.75, 0.5)),
                stroke: Some(Color::rgb(0.9, 0.1, 0.2)),
                stroke_width: Some(1.5),
                opacity: Some(0.75),
                ..Style::default()
            })
            .with_transform(Transform {
                translate_x: 2.0,
                translate_y: -1.0,
                rotate_deg: 12.0,
                scale: 1.25,
            });
        let svg = render_svg_for_remote_projection(&styled);
        assert!(portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_some());
    }

    #[test]
    fn remote_canvas_renderer_drops_pathological_transforms_before_boundary() {
        use symthaea_canvas::scene_graph::{SceneNode, Transform};
        use symthaea_canvas::svg_renderer::render_svg_for_remote_projection;

        let root = SceneNode::group(None)
            .with_child(SceneNode::circle(1.0, 1.0, 1.0).with_transform(Transform {
                scale: 9.0,
                ..Transform::identity()
            }))
            .with_child(SceneNode::circle(2.0, 2.0, 1.0).with_transform(Transform {
                translate_x: 1_000_001.0,
                ..Transform::identity()
            }));

        let svg = render_svg_for_remote_projection(&root);
        assert!(!svg.contains(r#"scale(9.000)"#));
        assert!(!svg.contains(r#"translate(1000001.0"#));
        assert!(portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_some());
    }

    #[test]
    fn portrait_rejects_invalid_numeric_domains() {
        for svg in [
            r#"<svg><circle r="-1"/></svg>"#,
            r#"<svg><rect width="-1" height="1"/></svg>"#,
            r#"<svg><circle opacity="1.1" r="1"/></svg>"#,
            r#"<svg viewBox="0 0 0 10"><circle r="1"/></svg>"#,
            r#"<svg viewBox="0 0 10 -1"><circle r="1"/></svg>"#,
        ] {
            assert!(
                portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none(),
                "accepted: {svg}"
            );
        }
    }

    #[test]
    fn portrait_accepts_valid_polyline_points() {
        let payload = serde_json::json!({
            "canvas_svg": r#"<svg viewBox="0 0 10 10"><polyline points="0,0 10,-5 10,10"/></svg>"#
        });
        assert!(portrait_from_json(&payload).is_some());
    }

    #[test]
    fn portrait_rejects_unknown_markup_even_without_known_dangerous_markers() {
        let payload = serde_json::json!({
            "canvas_svg": r#"<svg><metadata><foo/></metadata></svg>"#
        });
        assert!(portrait_from_json(&payload).is_none());
    }

    #[test]
    fn portrait_rejects_excessive_structural_complexity() {
        let nested = format!(
            "<svg>{}</svg>",
            "<g>".repeat(32) + &"</g>".repeat(32)
        );
        assert!(portrait_from_json(&serde_json::json!({ "canvas_svg": nested })).is_none());

        let too_many_elements = format!(
            "<svg>{}</svg>",
            "<g/>".repeat(256)
        );
        assert!(portrait_from_json(&serde_json::json!({
            "canvas_svg": too_many_elements
        })).is_none());
    }

    #[test]
    fn portrait_rejects_excessive_attribute_budget() {
        let svg = format!(
            r#"<svg><g id="{}"/></svg>"#,
            "a".repeat(65)
        );
        assert!(portrait_from_json(&serde_json::json!({ "canvas_svg": svg })).is_none());

        let allowed_attrs = [
            ("transform", "translate(1)"),
            ("fill", "none"),
            ("stroke", "none"),
            ("stroke-width", "1"),
            ("opacity", "1"),
        ];
        let mut many_attrs = String::new();
        for i in 0..205 {
            many_attrs.push_str(&format!("<g id=\"g{i}\" "));
            for (key, value) in allowed_attrs {
                many_attrs.push_str(&format!("{key}=\"{value}\" "));
            }
            many_attrs.push_str("/>");
        }
        let many_attrs = "<svg>".to_string() + &many_attrs + "</svg>";
        assert!(portrait_from_json(&serde_json::json!({
            "canvas_svg": many_attrs
        })).is_none());
    }

    #[test]
    fn movie_rejects_oversized_base64_before_decode() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 1,
                "height": 1,
                "channels": 1,
                "frames_b64": ["A".repeat(1024 * 1024)]
            }
        });
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_rejects_any_malformed_frame_instead_of_silently_dropping_it() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 1,
                "height": 1,
                "channels": 1,
                "frames_b64": ["AQ==", "not-base64"]
            }
        });
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_dimensions_and_channels_fail_closed_without_integer_truncation() {
        let mut payload = serde_json::json!({
            "mental_movie": {
                "width": (u32::MAX as u64) + 1,
                "height": 1,
                "channels": 1,
                "frames_b64": ["AA=="]
            }
        });
        assert!(Movie::from_json(&payload).is_none());

        payload["mental_movie"]["width"] = 1;
        payload["mental_movie"]["height"] = 1;
        payload["mental_movie"]["channels"] = 2;
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_semantic_coherence_normalizes_non_finite_or_f32_overflow() {
        let mut payload = serde_json::json!({
            "mental_movie": {
                "width": 1,
                "height": 1,
                "channels": 1,
                "frames_b64": ["AA=="],
                "semantic_coherence": f64::MAX
            }
        });
        assert_eq!(Movie::from_json(&payload).unwrap().semantic_coherence, 0.0);

        payload["mental_movie"]["semantic_coherence"] = serde_json::json!(f64::NAN);
        assert_eq!(Movie::from_json(&payload).unwrap().semantic_coherence, 0.0);
    }

    #[test]
    fn movie_frame_byte_count_is_checked_before_filtering() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 2,
                "height": 2,
                "channels": u64::MAX,
                "frames_b64": ["AA=="]
            }
        });
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_rejects_unsupported_four_channel_frames() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 1,
                "height": 1,
                "channels": 4,
                "frames_b64": ["AAAA"]
            }
        });
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_rejects_trailing_frame_bytes_instead_of_silently_ignoring_them() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 1,
                "height": 1,
                "channels": 1,
                "frames_b64": ["AQID"]
            }
        });
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_rejects_more_than_the_bounded_frame_count() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 1,
                "height": 1,
                "channels": 1,
                "frames_b64": vec!["AQ=="; 121]
            }
        });
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_rejects_excessive_total_rgba_allocation() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 4096,
                "height": 4096,
                "channels": 1,
                "frames_b64": vec!["AA=="; 120]
            }
        });
        assert!(Movie::from_json(&payload).is_none());
    }

    #[test]
    fn movie_accepts_a_bounded_total_rgba_allocation() {
        let payload = serde_json::json!({
            "mental_movie": {
                "width": 64,
                "height": 64,
                "channels": 1,
                "frames_b64": vec!["AA=="; 120]
            }
        });
        assert!(Movie::from_json(&payload).is_some());
    }

    #[test]
    fn authority_generation_is_monotonic() {
        let mut generation = 0_u64;
        let first = next_authority_generation(&mut generation).unwrap();
        let second = next_authority_generation(&mut generation).unwrap();
        assert_eq!(generation, 2);
        assert_eq!(first, 1);
        assert_eq!(second, 2);
        assert_ne!(first, second);
    }

    #[test]
    fn authority_generation_fails_closed_at_exhaustion() {
        let mut generation = u64::MAX - 1;
        assert_eq!(next_authority_generation(&mut generation), Some(u64::MAX));
        assert_eq!(generation, u64::MAX);
        assert_eq!(next_authority_generation(&mut generation), None);
        assert_eq!(generation, u64::MAX);
        assert_eq!(next_authority_generation(&mut generation), None);
        assert_eq!(generation, u64::MAX);
    }

    #[test]
    fn missing_required_telemetry_is_rejected_instead_of_zero_filled() {
        let mut payload = valid_payload();
        payload.as_object_mut().unwrap().remove("harmonic_field_coherence");
        assert!(Vitals::from_json(&payload).is_none());
    }

    #[test]
    fn f32_overflow_after_decode_is_rejected() {
        let mut payload = valid_payload();
        payload["thermodynamic_load"] = serde_json::json!(f64::MAX);
        assert!(Vitals::from_json(&payload).is_none());
    }

    #[test]
    fn f32_representable_boundary_remains_accepted() {
        let mut payload = valid_payload();
        payload["thermodynamic_load"] = serde_json::json!(f32::MAX as f64);
        assert!(Vitals::from_json(&payload).is_some());
    }
}
