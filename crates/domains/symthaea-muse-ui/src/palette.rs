// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Style-reactive palettes: the room changes with the music.
//!
//! Ported verbatim from `symthaea-muse/studio/index.html`'s
//! `STYLE_PALETTES`/`LISTEN_STYLES` so the new Listen Mode reads as the same
//! product, not a re-tuned one. `a`/`b` are `"r,g,b"` strings (used directly
//! in `rgba(...)` canvas fill/stroke styles); `bg` is the two-stop radial
//! background gradient.

pub struct Palette {
    pub a: &'static str,
    pub b: &'static str,
    pub bg0: &'static str,
    pub bg1: &'static str,
}

macro_rules! palette {
    ($a:expr, $b:expr, $bg0:expr, $bg1:expr) => {
        Palette {
            a: $a,
            b: $b,
            bg0: $bg0,
            bg1: $bg1,
        }
    };
}

/// Every style Muse can compose in, in the order the Listen radio cycles
/// them. Kept as a plain list (rather than iterating the palette map) so
/// the "no style selected" default ordering matches the legacy page.
///
/// SECOND SOURCE OF THE SAME SET: Create Mode no longer reads this list — it
/// navigates `symthaea_muse_protocol::catalog::CATALOG` by constellation and
/// offers each entry's `composer_style`. Verified 2026-07-29 that the two
/// agree exactly (both are the 29 `symthaea_music_theory::Style` Debug names),
/// so the switch changed the picker without changing what gets composed.
///
/// The roster is intentionally ordered for the Listen experience, but it must
/// cover the same set of routes as the executable public catalog. The test
/// below compares the sets directly so a newly implemented engine style cannot
/// silently remain absent from Listen while appearing in Create.
pub const LISTEN_STYLES: &[&str] = &[
    "Classical",
    "Nocturne",
    "Cinematic",
    "Passacaglia",
    "Lullaby",
    "Waltz",
    "Folk",
    "Tango",
    "Fugue",
    "ModalFolk",
    "March",
    "Playful",
    "Celtic",
    "Blues",
    "Impressionism",
    "SacredChoral",
    "Minimalism",
    "JazzBallad",
    "BaroqueSuite",
    "ProgFolk",
    "Ambient",
    "Sonata",
    "RenaissancePolyphony",
    "AfroCuban",
    "Flamenco",
    "BossaNova",
    "Opera",
    "IrishTraditional",
    "HindustaniInspired",
    "Rondo",
    "ThemeAndVariations",
    "Erosion",
    "Lineage",
    "Sarabande",
    "MinuetAndTrio",
    "Mazurka",
    "Toccata",
];

pub fn palette_for(style: &str) -> Palette {
    match style {
        "Classical" => palette!("232,196,138", "217,160,91", "#241e16", "#16130f"),
        "Nocturne" => palette!("158,150,224", "104,96,190", "#191a2e", "#0f1018"),
        "Tango" => palette!("235,110,97", "178,49,58", "#2a1414", "#140d0d"),
        "Cinematic" => palette!("126,196,207", "62,130,148", "#12222a", "#0b1418"),
        "Passacaglia" => palette!("150,196,132", "84,132,84", "#16231a", "#0e150f"),
        "Lullaby" => palette!("238,178,189", "196,116,138", "#271a20", "#151013"),
        "Folk" => palette!("206,196,120", "148,132,66", "#22200f", "#14130a"),
        "Waltz" => palette!("240,214,170", "196,164,110", "#262016", "#15120d"),
        "Fugue" => palette!("168,180,198", "104,120,146", "#181c24", "#0f1115"),
        "ModalFolk" => palette!("186,160,210", "124,96,152", "#201a28", "#121017"),
        "March" => palette!("224,178,96", "166,116,44", "#241c10", "#14100a"),
        "Playful" => palette!("240,150,120", "196,96,80", "#281a14", "#15100c"),
        "Celtic" => palette!("120,186,158", "58,124,102", "#132420", "#0b1613"),
        "Blues" => palette!("98,140,210", "46,74,150", "#101a2e", "#0a1019"),
        "Impressionism" => palette!("196,186,224", "140,128,180", "#1c1826", "#100e16"),
        "SacredChoral" => palette!("236,222,182", "196,164,96", "#221c12", "#14100a"),
        "Minimalism" => palette!("120,224,210", "52,158,148", "#0e201c", "#081210"),
        "JazzBallad" => palette!("212,142,120", "150,78,66", "#241511", "#150c0a"),
        "BaroqueSuite" => palette!("196,120,132", "138,62,74", "#241318", "#140b0d"),
        "ProgFolk" => palette!("150,90,230", "92,48,168", "#180f28", "#0d0916"),
        "Ambient" => palette!("168,188,196", "104,124,134", "#12181c", "#0a0d0f"),
        "Sonata" => palette!("120,140,180", "70,86,124", "#11151f", "#090b12"),
        "RenaissancePolyphony" => palette!("158,148,96", "108,100,62", "#181610", "#0d0c08"),
        "AfroCuban" => palette!("232,142,64", "186,84,36", "#251507", "#150c05"),
        "Flamenco" => palette!("214,54,58", "138,26,32", "#210a0b", "#140606"),
        "BossaNova" => palette!("92,182,168", "48,124,114", "#0c1c19", "#07110f"),
        "Opera" => palette!("176,96,182", "112,52,120", "#180c1c", "#0e0710"),
        "IrishTraditional" => palette!("110,184,96", "62,128,52", "#0e1c0c", "#081007"),
        "HindustaniInspired" => palette!("224,168,72", "164,116,40", "#201708", "#130e05"),
        "Rondo" => palette!("222,184,95", "144,112,54", "#211a0c", "#110e08"),
        "ThemeAndVariations" => palette!("190,154,224", "130,94,168", "#201528", "#100c14"),
        "Erosion" => palette!("163,156,144", "103,96,88", "#1d1a17", "#0e0d0c"),
        "Lineage" => palette!("116,194,142", "54,132,88", "#102218", "#080f0b"),
        "Sarabande" => palette!("222,154,120", "148,68,58", "#25140f", "#120a08"),
        "MinuetAndTrio" => palette!("174,198,224", "112,142,174", "#18202c", "#0d1118"),
        "Mazurka" => palette!("236,122,94", "180,70,48", "#28130f", "#150a08"),
        "Toccata" => palette!("102,218,201", "218,176,92", "#0e2424", "#071211"),
        _ => palette!("232,196,138", "217,160,91", "#241e16", "#16130f"), // Classical fallback
    }
}

/// Pick a pseudo-random style using `Math.random`-equivalent entropy from
/// `js_sys`, matching the legacy page's `styles[Math.floor(Math.random() *
/// styles.length)]`.
pub fn random_style() -> &'static str {
    let idx = (js_sys::Math::random() * LISTEN_STYLES.len() as f64) as usize;
    LISTEN_STYLES[idx.min(LISTEN_STYLES.len() - 1)]
}

/// The roster "Today's Discoveries" rotates through — ported verbatim from
/// `DISCOVERY_STYLES` in `studio/index.html`. Deliberately a separate,
/// slightly shorter list than [`LISTEN_STYLES`] (missing `Lullaby`/`March`/
/// `Playful`) — the legacy page uses two different rosters for the two
/// features, so `today_seed() % DISCOVERY_STYLES.len()` reproduces the same
/// day→style mapping a user would see on the legacy page.
pub const DISCOVERY_STYLES: &[&str] = &[
    "Classical",
    "Nocturne",
    "Tango",
    "Cinematic",
    "Passacaglia",
    "Folk",
    "Waltz",
    "Fugue",
    "ModalFolk",
    "Celtic",
    "Blues",
    "Impressionism",
    "SacredChoral",
    "Minimalism",
    "JazzBallad",
    "BaroqueSuite",
    "ProgFolk",
    "Ambient",
    "Sonata",
    "RenaissancePolyphony",
    "AfroCuban",
    "Flamenco",
    "BossaNova",
    "Opera",
    "IrishTraditional",
    "HindustaniInspired",
];

/// A date-deterministic seed (`YYYYMMDD` as an integer, local time) — the
/// same all day, different tomorrow. Ported verbatim from `todaySeed()`.
pub fn today_seed() -> u64 {
    let date = js_sys::Date::new_0();
    date.get_full_year() as u64 * 10_000
        + (date.get_month() as u64 + 1) * 100
        + date.get_date() as u64
}

/// Today's discovery style, deterministic from [`today_seed`].
pub fn today_style() -> &'static str {
    let seed = today_seed();
    DISCOVERY_STYLES[(seed as usize) % DISCOVERY_STYLES.len()]
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn listen_roster_matches_composable_catalog_routes_exactly() {
        let listen: HashSet<&str> = LISTEN_STYLES.iter().copied().collect();
        assert_eq!(
            listen.len(),
            LISTEN_STYLES.len(),
            "LISTEN_STYLES must not contain duplicate routes"
        );

        let catalog: HashSet<&str> =
            symthaea_muse_protocol::catalog::implemented_catalog()
                .filter_map(|entry| entry.composer_style)
                .collect();
        assert_eq!(
            catalog.len(),
            LISTEN_STYLES.len(),
            "the Listen roster and composer catalog must expose the same number of routes"
        );
        assert_eq!(
            listen, catalog,
            "every executable catalog style must be reachable in Listen, and no unknown style may be listed"
        );
    }

    #[test]
    fn every_listen_style_has_an_explicit_unique_palette() {
        let fallback = palette_for("__unknown_style__");
        let accents: HashSet<&str> = LISTEN_STYLES
            .iter()
            .map(|style| {
                let palette = palette_for(style);
                assert_ne!(
                    palette.a, fallback.a,
                    "{style} falls through to the Classical palette"
                );
                palette.a
            })
            .collect();
        assert_eq!(
            accents.len(),
            LISTEN_STYLES.len(),
            "each selectable style needs a distinct accent color"
        );
    }
}
