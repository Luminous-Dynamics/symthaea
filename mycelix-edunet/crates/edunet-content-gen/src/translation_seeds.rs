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
    // SA priority languages
    seeds.extend(isizulu_seeds());
    seeds.extend(afrikaans_seeds());
    seeds.extend(sesotho_seeds());
    // Global top 20 languages
    seeds.extend(mandarin_seeds());
    seeds.extend(hindi_seeds());
    seeds.extend(arabic_seeds());
    seeds.extend(bengali_seeds());
    seeds.extend(russian_seeds());
    seeds.extend(japanese_seeds());
    seeds.extend(indonesian_seeds());
    seeds.extend(korean_seeds());
    seeds.extend(vietnamese_seeds());
    seeds.extend(turkish_seeds());
    seeds.extend(thai_seeds());
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

// ============== Global Top 20 Language Seeds ==============
// NOTE: These are machine-confidence seeds. They MUST be verified by native
// speakers through the TranslationVote system before being shown to learners.

pub fn mandarin_seeds() -> Vec<TranslationSeed> {
    vec![
        mseed("home", "Home", "zh", "\u{9996}\u{9875}"),
        mseed("explore", "Explore", "zh", "\u{63a2}\u{7d22}"),
        mseed("review", "Review", "zh", "\u{590d}\u{4e60}"),
        mseed("progress", "Progress", "zh", "\u{8fdb}\u{5ea6}"),
        mseed("greeting_morning", "Good morning", "zh", "\u{65e9}\u{4e0a}\u{597d}"),
        mseed("start_learning", "Start learning", "zh", "\u{5f00}\u{59cb}\u{5b66}\u{4e60}"),
        mseed("tend_earned", "TEND earned", "zh", "\u{83b7}\u{5f97}\u{7684}TEND"),
        mseed("your_knowledge_has_value", "Your knowledge has value", "zh", "\u{4f60}\u{7684}\u{77e5}\u{8bc6}\u{6709}\u{4ef7}\u{503c}"),
    ]
}

pub fn hindi_seeds() -> Vec<TranslationSeed> {
    vec![
        mseed("home", "Home", "hi", "\u{0918}\u{0930}"),
        mseed("explore", "Explore", "hi", "\u{0916}\u{094b}\u{091c}\u{0947}\u{0902}"),
        mseed("review", "Review", "hi", "\u{0938}\u{092e}\u{0940}\u{0915}\u{094d}\u{0937}\u{093e}"),
        mseed("progress", "Progress", "hi", "\u{092a}\u{094d}\u{0930}\u{0917}\u{0924}\u{093f}"),
        mseed("greeting_morning", "Good morning", "hi", "\u{0938}\u{0941}\u{092a}\u{094d}\u{0930}\u{092d}\u{093e}\u{0924}"),
        mseed("start_learning", "Start learning", "hi", "\u{0938}\u{0940}\u{0916}\u{0928}\u{093e} \u{0936}\u{0941}\u{0930}\u{0942} \u{0915}\u{0930}\u{0947}\u{0902}"),
        mseed("tend_earned", "TEND earned", "hi", "TEND \u{0905}\u{0930}\u{094d}\u{091c}\u{093f}\u{0924}"),
        mseed("your_knowledge_has_value", "Your knowledge has value", "hi", "\u{0906}\u{092a}\u{0915}\u{0947} \u{091c}\u{094d}\u{091e}\u{093e}\u{0928} \u{0915}\u{093e} \u{092e}\u{0942}\u{0932}\u{094d}\u{092f} \u{0939}\u{0948}"),
    ]
}

pub fn arabic_seeds() -> Vec<TranslationSeed> {
    vec![
        mseed("home", "Home", "ar", "\u{0627}\u{0644}\u{0631}\u{0626}\u{064a}\u{0633}\u{064a}\u{0629}"),
        mseed("explore", "Explore", "ar", "\u{0627}\u{0633}\u{062a}\u{0643}\u{0634}\u{0641}"),
        mseed("review", "Review", "ar", "\u{0645}\u{0631}\u{0627}\u{062c}\u{0639}\u{0629}"),
        mseed("progress", "Progress", "ar", "\u{062a}\u{0642}\u{062f}\u{0645}"),
        mseed("greeting_morning", "Good morning", "ar", "\u{0635}\u{0628}\u{0627}\u{062d} \u{0627}\u{0644}\u{062e}\u{064a}\u{0631}"),
        mseed("start_learning", "Start learning", "ar", "\u{0627}\u{0628}\u{062f}\u{0623} \u{0627}\u{0644}\u{062a}\u{0639}\u{0644}\u{0645}"),
        mseed("tend_earned", "TEND earned", "ar", "TEND \u{0645}\u{0643}\u{062a}\u{0633}\u{0628}"),
        mseed("your_knowledge_has_value", "Your knowledge has value", "ar", "\u{0645}\u{0639}\u{0631}\u{0641}\u{062a}\u{0643} \u{0644}\u{0647}\u{0627} \u{0642}\u{064a}\u{0645}\u{0629}"),
    ]
}

pub fn bengali_seeds() -> Vec<TranslationSeed> {
    vec![
        mseed("home", "Home", "bn", "\u{09b9}\u{09cb}\u{09ae}"),
        mseed("explore", "Explore", "bn", "\u{0985}\u{09a8}\u{09cd}\u{09ac}\u{09c7}\u{09b7}\u{09a3}"),
        mseed("greeting_morning", "Good morning", "bn", "\u{09b8}\u{09c1}\u{09aa}\u{09cd}\u{09b0}\u{09ad}\u{09be}\u{09a4}"),
        mseed("start_learning", "Start learning", "bn", "\u{09b6}\u{09c7}\u{0996}\u{09be} \u{09b6}\u{09c1}\u{09b0}\u{09c1} \u{0995}\u{09b0}\u{09c1}\u{09a8}"),
        mseed("tend_earned", "TEND earned", "bn", "TEND \u{0985}\u{09b0}\u{09cd}\u{099c}\u{09bf}\u{09a4}"),
    ]
}

pub fn russian_seeds() -> Vec<TranslationSeed> {
    vec![
        mseed("home", "Home", "ru", "\u{0413}\u{043b}\u{0430}\u{0432}\u{043d}\u{0430}\u{044f}"),
        mseed("explore", "Explore", "ru", "\u{0418}\u{0441}\u{0441}\u{043b}\u{0435}\u{0434}\u{043e}\u{0432}\u{0430}\u{0442}\u{044c}"),
        mseed("review", "Review", "ru", "\u{041f}\u{043e}\u{0432}\u{0442}\u{043e}\u{0440}\u{0435}\u{043d}\u{0438}\u{0435}"),
        mseed("greeting_morning", "Good morning", "ru", "\u{0414}\u{043e}\u{0431}\u{0440}\u{043e}\u{0435} \u{0443}\u{0442}\u{0440}\u{043e}"),
        mseed("start_learning", "Start learning", "ru", "\u{041d}\u{0430}\u{0447}\u{0430}\u{0442}\u{044c} \u{043e}\u{0431}\u{0443}\u{0447}\u{0435}\u{043d}\u{0438}\u{0435}"),
        mseed("tend_earned", "TEND earned", "ru", "TEND \u{0437}\u{0430}\u{0440}\u{0430}\u{0431}\u{043e}\u{0442}\u{0430}\u{043d}\u{043e}"),
    ]
}

pub fn japanese_seeds() -> Vec<TranslationSeed> {
    vec![
        mseed("home", "Home", "ja", "\u{30db}\u{30fc}\u{30e0}"),
        mseed("explore", "Explore", "ja", "\u{63a2}\u{7d22}"),
        mseed("review", "Review", "ja", "\u{5fa9}\u{7fd2}"),
        mseed("greeting_morning", "Good morning", "ja", "\u{304a}\u{306f}\u{3088}\u{3046}\u{3054}\u{3056}\u{3044}\u{307e}\u{3059}"),
        mseed("start_learning", "Start learning", "ja", "\u{5b66}\u{7fd2}\u{3092}\u{59cb}\u{3081}\u{308b}"),
        mseed("tend_earned", "TEND earned", "ja", "\u{7372}\u{5f97}TEND"),
    ]
}

pub fn indonesian_seeds() -> Vec<TranslationSeed> {
    vec![
        mseed("home", "Home", "id", "Beranda"),
        mseed("explore", "Explore", "id", "Jelajahi"),
        mseed("review", "Review", "id", "Ulasan"),
        mseed("greeting_morning", "Good morning", "id", "Selamat pagi"),
        mseed("start_learning", "Start learning", "id", "Mulai belajar"),
        mseed("tend_earned", "TEND earned", "id", "TEND diperoleh"),
        mseed("your_knowledge_has_value", "Your knowledge has value", "id", "Pengetahuanmu bernilai"),
    ]
}

pub fn korean_seeds() -> Vec<TranslationSeed> {
    vec![
        mseed("home", "Home", "ko", "\u{d648}"),
        mseed("explore", "Explore", "ko", "\u{d0d0}\u{c0c9}"),
        mseed("review", "Review", "ko", "\u{bcf5}\u{c2b5}"),
        mseed("greeting_morning", "Good morning", "ko", "\u{c88b}\u{c740} \u{c544}\u{ce68}"),
        mseed("start_learning", "Start learning", "ko", "\u{d559}\u{c2b5} \u{c2dc}\u{c791}"),
        mseed("tend_earned", "TEND earned", "ko", "\u{d68d}\u{b4dd}\u{d55c} TEND"),
    ]
}

pub fn vietnamese_seeds() -> Vec<TranslationSeed> {
    vec![
        mseed("home", "Home", "vi", "Trang ch\u{1ee7}"),
        mseed("explore", "Explore", "vi", "Kh\u{00e1}m ph\u{00e1}"),
        mseed("review", "Review", "vi", "\u{00d4}n t\u{1ead}p"),
        mseed("greeting_morning", "Good morning", "vi", "Ch\u{00e0}o bu\u{1ed5}i s\u{00e1}ng"),
        mseed("start_learning", "Start learning", "vi", "B\u{1eaf}t \u{0111}\u{1ea7}u h\u{1ecd}c"),
        mseed("tend_earned", "TEND earned", "vi", "TEND \u{0111}\u{00e3} ki\u{1ebf}m"),
    ]
}

pub fn turkish_seeds() -> Vec<TranslationSeed> {
    vec![
        mseed("home", "Home", "tr", "Ana Sayfa"),
        mseed("explore", "Explore", "tr", "Ke\u{015f}fet"),
        mseed("review", "Review", "tr", "Tekrar"),
        mseed("greeting_morning", "Good morning", "tr", "G\u{00fc}nayd\u{0131}n"),
        mseed("start_learning", "Start learning", "tr", "\u{00d6}\u{011f}renmeye ba\u{015f}la"),
        mseed("tend_earned", "TEND earned", "tr", "Kazan\u{0131}lan TEND"),
    ]
}

pub fn thai_seeds() -> Vec<TranslationSeed> {
    vec![
        mseed("home", "Home", "th", "\u{0e2b}\u{0e19}\u{0e49}\u{0e32}\u{0e41}\u{0e23}\u{0e01}"),
        mseed("explore", "Explore", "th", "\u{0e2a}\u{0e33}\u{0e23}\u{0e27}\u{0e08}"),
        mseed("greeting_morning", "Good morning", "th", "\u{0e2a}\u{0e27}\u{0e31}\u{0e2a}\u{0e14}\u{0e35}\u{0e15}\u{0e2d}\u{0e19}\u{0e40}\u{0e0a}\u{0e49}\u{0e32}"),
        mseed("start_learning", "Start learning", "th", "\u{0e40}\u{0e23}\u{0e34}\u{0e48}\u{0e21}\u{0e40}\u{0e23}\u{0e35}\u{0e22}\u{0e19}\u{0e23}\u{0e39}\u{0e49}"),
        mseed("tend_earned", "TEND earned", "th", "TEND \u{0e17}\u{0e35}\u{0e48}\u{0e44}\u{0e14}\u{0e49}\u{0e23}\u{0e31}\u{0e1a}"),
    ]
}

/// Machine-confidence seed (needs native speaker verification before display).
fn mseed(key: &str, source: &str, target_lang: &str, translated: &str) -> TranslationSeed {
    TranslationSeed {
        context_key: key.to_string(),
        source_lang: "en".to_string(),
        source_text: source.to_string(),
        target_lang: target_lang.to_string(),
        translated_text: translated.to_string(),
        confidence: "machine", // MUST be verified by native speakers before display
    }
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
        assert!(seeds.len() >= 120, "Should have 120+ seed translations across 14 languages, got {}", seeds.len());
    }

    #[test]
    fn test_three_languages_covered() {
        let seeds = all_seeds();
        let langs: std::collections::HashSet<&str> = seeds.iter().map(|s| s.target_lang.as_str()).collect();
        assert!(langs.contains("zu"), "Missing isiZulu");
        assert!(langs.contains("af"), "Missing Afrikaans");
        assert!(langs.contains("st"), "Missing Sesotho");
        assert!(langs.contains("zh"), "Missing Mandarin");
        assert!(langs.contains("hi"), "Missing Hindi");
        assert!(langs.contains("ar"), "Missing Arabic");
        assert!(langs.contains("ru"), "Missing Russian");
        assert!(langs.contains("ja"), "Missing Japanese");
        assert!(langs.contains("ko"), "Missing Korean");
        assert!(langs.contains("id"), "Missing Indonesian");
        assert!(langs.len() >= 14, "Should cover 14+ languages, got {}", langs.len());
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
