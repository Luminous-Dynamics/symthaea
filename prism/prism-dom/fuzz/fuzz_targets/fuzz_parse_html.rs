// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Only fuzz valid UTF-8 — HTML parser expects str
    if let Ok(html) = std::str::from_utf8(data) {
        // parse_html should never panic regardless of input
        let tree = prism_dom::parse_html(html);

        // Exercise text extraction (walks the tree)
        let _text = tree.extract_visible_text();

        // Exercise title extraction
        let _title = tree.title();
    }
});
