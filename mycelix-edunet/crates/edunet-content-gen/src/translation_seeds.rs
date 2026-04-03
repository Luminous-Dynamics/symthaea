// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Translation Seed Data
//!
//! Seed translations for core UI strings in priority SA languages.
//! These serve as the initial data for the Translation-as-Credential system.
//!
//! Language priorities (from SA Census 2022):
//! 1. isiZulu (24.4% national) — largest SA language
//! 2. Afrikaans (10.6%) — widely spoken across SA
//! 3. Sesotho (7.8%) — dominant in Free State + Gauteng
//!
//! NOTE: These translations should be verified by native speakers via
//! the TranslationVote system before being marked as VerifiedTranslation.

use serde::{Deserialize, Serialize};

/// A seed translation for bootstrapping the translation system.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranslationSeed {
    pub context_key: String,
    pub source_lang: String,
    pub source_text: String,
    pub target_lang: String,
    pub translated_text: String,
    pub confidence: &'static str, // "native" | "reviewed" | "machine"
}

/// Get all seed translations for bootstrapping.
pub fn all_seeds() -> Vec<TranslationSeed> {
    let mut seeds = Vec::new();
    seeds.extend(isizulu_seeds());
    seeds.extend(afrikaans_seeds());
    seeds.extend(sesotho_seeds());
    seeds
}

/// Core UI strings in isiZulu (zu).
pub fn isizulu_seeds() -> Vec<TranslationSeed> {
    vec![
        seed("home", "Home", "zu", "Ikhaya"),
        seed("explore", "Explore", "zu", "Hlola"),
        seed("review", "Review", "zu", "Buyekeza"),
        seed("progress", "Progress", "zu", "Inqubekela phambili"),
        seed("exam", "Exam", "zu", "Ukuhlolwa"),
        seed("search", "Search topics...", "zu", "Sesha izihloko..."),
        seed("greeting_morning", "Good morning", "zu", "Sawubona ekuseni"),
        seed("greeting_afternoon", "Good afternoon", "zu", "Sawubona emini"),
        seed("greeting_evening", "Good evening", "zu", "Sawubona kusihlwa"),
        seed("what_today", "What do you want to do today?", "zu", "Ufuna ukwenzani namuhla?"),
        seed("start_learning", "Start learning", "zu", "Qala ukufunda"),
        seed("mastery", "Mastery", "zu", "Ubuchwepheshe"),
        seed("studying", "Studying", "zu", "Ukufunda"),
        seed("not_started", "Not started", "zu", "Akuqalwanga"),
        seed("dashboard", "Dashboard", "zu", "Ibhodi"),
        seed("credentials", "Credentials", "zu", "Izitifiketi"),
        seed("governance", "Governance", "zu", "Ukuphatha"),
        seed("profile", "Profile", "zu", "Iphrofayili"),
        seed("settings", "Settings", "zu", "Izilungiselelo"),
        seed("tend_earned", "TEND earned", "zu", "I-TEND ezuzwile"),
        seed("connect_network", "Connect to network", "zu", "Xhuma kunethiwekhi"),
        seed("your_knowledge_has_value", "Your knowledge has value", "zu", "Ulwazi lwakho lunenani"),
    ]
}

/// Core UI strings in Afrikaans (af).
pub fn afrikaans_seeds() -> Vec<TranslationSeed> {
    vec![
        seed("home", "Home", "af", "Tuis"),
        seed("explore", "Explore", "af", "Verken"),
        seed("review", "Review", "af", "Hersiening"),
        seed("progress", "Progress", "af", "Vordering"),
        seed("exam", "Exam", "af", "Eksamen"),
        seed("search", "Search topics...", "af", "Soek onderwerpe..."),
        seed("greeting_morning", "Good morning", "af", "Goeiemore"),
        seed("greeting_afternoon", "Good afternoon", "af", "Goeiemiddag"),
        seed("greeting_evening", "Good evening", "af", "Goeienaand"),
        seed("what_today", "What do you want to do today?", "af", "Wat wil jy vandag doen?"),
        seed("start_learning", "Start learning", "af", "Begin leer"),
        seed("mastery", "Mastery", "af", "Bemeestering"),
        seed("studying", "Studying", "af", "Besig om te leer"),
        seed("not_started", "Not started", "af", "Nie begin nie"),
        seed("dashboard", "Dashboard", "af", "Paneelbord"),
        seed("credentials", "Credentials", "af", "Kwalifikasies"),
        seed("governance", "Governance", "af", "Bestuur"),
        seed("profile", "Profile", "af", "Profiel"),
        seed("tend_earned", "TEND earned", "af", "TEND verdien"),
        seed("connect_network", "Connect to network", "af", "Koppel aan netwerk"),
        seed("your_knowledge_has_value", "Your knowledge has value", "af", "Jou kennis het waarde"),
    ]
}

/// Core UI strings in Sesotho (st).
pub fn sesotho_seeds() -> Vec<TranslationSeed> {
    vec![
        seed("home", "Home", "st", "Lapeng"),
        seed("explore", "Explore", "st", "Hlahloba"),
        seed("review", "Review", "st", "Hlahloba hape"),
        seed("progress", "Progress", "st", "Tsoelo-pele"),
        seed("exam", "Exam", "st", "Tlhahlobo"),
        seed("greeting_morning", "Good morning", "st", "Dumela hoseng"),
        seed("greeting_afternoon", "Good afternoon", "st", "Dumela motsheare"),
        seed("greeting_evening", "Good evening", "st", "Dumela mantsiboya"),
        seed("what_today", "What do you want to do today?", "st", "O batla ho etsa eng kajeno?"),
        seed("start_learning", "Start learning", "st", "Qala ho ithuta"),
        seed("mastery", "Mastery", "st", "Botseba"),
        seed("tend_earned", "TEND earned", "st", "TEND e fumanweng"),
        seed("your_knowledge_has_value", "Your knowledge has value", "st", "Tsebo ya hao e na le boleng"),
    ]
}

fn seed(key: &str, source: &str, target_lang: &str, translated: &str) -> TranslationSeed {
    TranslationSeed {
        context_key: key.to_string(),
        source_lang: "en".to_string(),
        source_text: source.to_string(),
        target_lang: target_lang.to_string(),
        translated_text: translated.to_string(),
        confidence: "reviewed", // Pre-reviewed but needs native speaker verification
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seeds_not_empty() {
        let seeds = all_seeds();
        assert!(seeds.len() >= 50, "Should have 50+ seed translations, got {}", seeds.len());
    }

    #[test]
    fn test_three_languages_covered() {
        let seeds = all_seeds();
        let langs: std::collections::HashSet<&str> = seeds.iter().map(|s| s.target_lang.as_str()).collect();
        assert!(langs.contains("zu"), "Missing isiZulu");
        assert!(langs.contains("af"), "Missing Afrikaans");
        assert!(langs.contains("st"), "Missing Sesotho");
    }

    #[test]
    fn test_core_keys_translated_in_all_languages() {
        let core_keys = ["home", "explore", "greeting_morning", "tend_earned"];
        let seeds = all_seeds();
        for key in core_keys {
            for lang in ["zu", "af", "st"] {
                assert!(
                    seeds.iter().any(|s| s.context_key == key && s.target_lang == lang),
                    "Missing translation for key='{}' in lang='{}'", key, lang
                );
            }
        }
    }

    #[test]
    fn test_all_seeds_have_source_english() {
        for seed in all_seeds() {
            assert_eq!(seed.source_lang, "en", "All seeds should have English source");
        }
    }

    #[test]
    fn test_no_empty_translations() {
        for seed in all_seeds() {
            assert!(!seed.translated_text.is_empty(), "Empty translation for key={}", seed.context_key);
            assert!(!seed.source_text.is_empty(), "Empty source for key={}", seed.context_key);
        }
    }
}
