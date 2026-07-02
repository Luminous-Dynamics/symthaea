// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Internationalization — multi-language UI support.
//!
//! Supports all 11 SA official languages + global languages.
//! UI strings are translated; lesson content stays in English
//! (content translation is a community governance task via DAO).

use leptos::prelude::*;
use crate::persistence;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Lang {
    En,    // English
    Zu,    // isiZulu
    Af,    // Afrikaans
    St,    // Sesotho
    Xh,    // isiXhosa
    Tn,    // Setswana
    Ns,    // Sepedi (Northern Sotho)
    Ts,    // Xitsonga
    Ss,    // siSwati
    Ve,    // Tshivenda
    Nr,    // isiNdebele
    Fr,    // French
    Pt,    // Portuguese
    Es,    // Spanish
    Sw,    // Kiswahili
    Ar,    // Arabic
    Hi,    // Hindi
    Zh,    // Chinese (Simplified)
}

impl Lang {
    pub fn code(&self) -> &'static str {
        match self {
            Lang::En => "en", Lang::Zu => "zu", Lang::Af => "af", Lang::St => "st",
            Lang::Xh => "xh", Lang::Tn => "tn", Lang::Ns => "nso", Lang::Ts => "ts",
            Lang::Ss => "ss", Lang::Ve => "ve", Lang::Nr => "nr", Lang::Fr => "fr",
            Lang::Pt => "pt", Lang::Es => "es", Lang::Sw => "sw", Lang::Ar => "ar",
            Lang::Hi => "hi", Lang::Zh => "zh",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Lang::En => "English", Lang::Zu => "isiZulu", Lang::Af => "Afrikaans",
            Lang::St => "Sesotho", Lang::Xh => "isiXhosa", Lang::Tn => "Setswana",
            Lang::Ns => "Sepedi", Lang::Ts => "Xitsonga", Lang::Ss => "siSwati",
            Lang::Ve => "Tshivenda", Lang::Nr => "isiNdebele", Lang::Fr => "Fran\u{00E7}ais",
            Lang::Pt => "Portugu\u{00EA}s", Lang::Es => "Espa\u{00F1}ol",
            Lang::Sw => "Kiswahili", Lang::Ar => "\u{0627}\u{0644}\u{0639}\u{0631}\u{0628}\u{064A}\u{0629}",
            Lang::Hi => "\u{0939}\u{093F}\u{0928}\u{094D}\u{0926}\u{0940}",
            Lang::Zh => "\u{4E2D}\u{6587}",
        }
    }

    pub fn native_label(&self) -> &'static str {
        self.label() // Same for now — label is already in native script
    }

    /// SA official languages
    pub fn sa_languages() -> &'static [Lang] {
        &[Lang::En, Lang::Zu, Lang::Af, Lang::St, Lang::Xh, Lang::Tn,
          Lang::Ns, Lang::Ts, Lang::Ss, Lang::Ve, Lang::Nr]
    }

    /// Global languages
    pub fn global_languages() -> &'static [Lang] {
        &[Lang::Fr, Lang::Pt, Lang::Es, Lang::Sw, Lang::Ar, Lang::Hi, Lang::Zh]
    }
}

/// Translate a UI key to the current language.
pub fn t(key: &str, lang: Lang) -> &'static str {
    match lang {
        Lang::Zu => t_zu(key),
        Lang::Af => t_af(key),
        Lang::St => t_st(key),
        Lang::Xh => t_xh(key),
        Lang::Fr => t_fr(key),
        Lang::Es => t_es(key),
        Lang::Pt => t_pt(key),
        Lang::Sw => t_sw(key),
        _ => t_en(key), // Default to English for untranslated languages
    }
}

fn t_en(key: &str) -> &'static str {
    match key {
        "home" => "Home",
        "explore" => "Explore",
        "review" => "Review",
        "progress" => "Progress",
        "exam" => "Exam",
        "pathways" => "Pathways",
        "search" => "Search topics...",
        "greeting_morning" => "Good morning",
        "greeting_afternoon" => "Good afternoon",
        "greeting_evening" => "Good evening",
        "greeting_hey" => "Hey",
        "what_today" => "What do you want to do today?",
        "review_flashcards" => "Review flashcards",
        "study_topic" => "Study a topic",
        "view_progress" => "View my progress",
        "need_help" => "I need help",
        "explore_freely" => "Explore freely",
        "mock_exam" => "Mock exam",
        "career_pathways" => "Career pathways",
        "begin_session" => "Begin your session",
        "knowledge_garden" => "Knowledge Garden",
        "rooted" => "rooted",
        "growing" => "growing",
        "seeds" => "seeds",
        "mastered" => "Mastered",
        "studying" => "Studying",
        "not_started" => "Not Started",
        "connections" => "Connections",
        "notes" => "Notes",
        "practice" => "Practice",
        "learn" => "Learn",
        "examples" => "Examples",
        "pitfalls" => "Pitfalls",
        "still_growing" => "This lesson is still growing",
        "your_data_stays" => "Your learning data stays on your device. You own it.",
        "beyond_school" => "Beyond school?",
        "earlier_grades" => "Earlier grades or need to review basics?",
        "deepest_roots" => "No shame in going back to strengthen foundations. The strongest trees have the deepest roots.",
        "language" => "Language",
        _ => "?",
    }
}

fn t_zu(key: &str) -> &'static str {
    match key {
        "home" => "Ikhaya",
        "explore" => "Hlola",
        "review" => "Buyekeza",
        "progress" => "Inqubekela phambili",
        "exam" => "Ukuhlolwa",
        "pathways" => "Izindlela",
        "search" => "Sesha izihloko...",
        "greeting_morning" => "Sawubona ekuseni",
        "greeting_afternoon" => "Sawubona ntambama",
        "greeting_evening" => "Sawubona kusihlwa",
        "greeting_hey" => "Sawubona",
        "what_today" => "Ufuna ukwenzani namuhla?",
        "review_flashcards" => "Buyekeza amakhadi",
        "study_topic" => "Funda isihloko",
        "view_progress" => "Buka inqubekela phambili",
        "need_help" => "Ngidinga usizo",
        "explore_freely" => "Hlola ngokukhululeka",
        "mock_exam" => "Ukuhlolwa okuzamayo",
        "career_pathways" => "Izindlela zomsebenzi",
        "begin_session" => "Qala iseshini yakho",
        "knowledge_garden" => "Ingadi Yolwazi",
        "rooted" => "kunezimpande",
        "growing" => "iyakhula",
        "seeds" => "imbewu",
        "mastered" => "Kuphumelele",
        "studying" => "Iyafunda",
        "not_started" => "Ayikaqali",
        "connections" => "Ukuxhumana",
        "notes" => "Amanothi",
        "practice" => "Ukuzilolonga",
        "learn" => "Funda",
        "examples" => "Izibonelo",
        "pitfalls" => "Izingozi",
        "still_growing" => "Lesi sifundo sisakhula",
        "your_data_stays" => "Idatha yakho yokufunda ihlala kudivayisi yakho. Ngeyakho.",
        "beyond_school" => "Ngaphezulu kwesikole?",
        "earlier_grades" => "Amabanga angaphambili noma udinga ukubuyekeza izisekelo?",
        "deepest_roots" => "Akukho mahloni ukubuyela emuva ukuze uqinise izisekelo. Imithi eqine kakhulu inezimpande ezijule kakhulu.",
        "language" => "Ulimi",
        _ => t_en(key),
    }
}

fn t_af(key: &str) -> &'static str {
    match key {
        "home" => "Tuis",
        "explore" => "Verken",
        "review" => "Hersien",
        "progress" => "Vordering",
        "exam" => "Eksamen",
        "pathways" => "Loopbane",
        "search" => "Soek onderwerpe...",
        "greeting_morning" => "Goeiem\u{00F4}re",
        "greeting_afternoon" => "Goeiemiddag",
        "greeting_evening" => "Goeienaand",
        "greeting_hey" => "Hallo",
        "what_today" => "Wat wil jy vandag doen?",
        "review_flashcards" => "Hersien flitskaarte",
        "study_topic" => "Bestudeer 'n onderwerp",
        "view_progress" => "Bekyk my vordering",
        "need_help" => "Ek het hulp nodig",
        "explore_freely" => "Verken vrylik",
        "mock_exam" => "Proefeksamen",
        "career_pathways" => "Loopbaanpaaie",
        "begin_session" => "Begin jou sessie",
        "knowledge_garden" => "Kennistuin",
        "rooted" => "gewortel",
        "growing" => "groeiend",
        "seeds" => "sade",
        "mastered" => "Bemeester",
        "studying" => "Besig om te leer",
        "not_started" => "Nie begin nie",
        "connections" => "Verbindings",
        "notes" => "Notas",
        "practice" => "Oefening",
        "learn" => "Leer",
        "examples" => "Voorbeelde",
        "pitfalls" => "Slaggate",
        "still_growing" => "Hierdie les groei nog",
        "your_data_stays" => "Jou leerdata bly op jou toestel. Dit behoort aan jou.",
        "beyond_school" => "Verby skool?",
        "earlier_grades" => "Vroeër grade of moet jy basiese beginsels hersien?",
        "deepest_roots" => "Geen skaamte om terug te gaan om fondamente te versterk nie. Die sterkste bome het die diepste wortels.",
        "language" => "Taal",
        _ => t_en(key),
    }
}

fn t_st(key: &str) -> &'static str {
    match key {
        "home" => "Lapeng",
        "explore" => "Hlahloba",
        "review" => "Hlahloba hape",
        "progress" => "Tsoelo-pele",
        "exam" => "Tlhahlobo",
        "greeting_morning" => "Dumela hoseng",
        "greeting_afternoon" => "Dumela motsheare",
        "greeting_evening" => "Dumela mantsibuya",
        "greeting_hey" => "Dumela",
        "what_today" => "O batla ho etsa eng kajeno?",
        "begin_session" => "Qala seshene ya hao",
        "knowledge_garden" => "Serapa sa Tsebo",
        "language" => "Puo",
        _ => t_en(key),
    }
}

fn t_xh(key: &str) -> &'static str {
    match key {
        "home" => "Ikhaya",
        "explore" => "Phonononga",
        "review" => "Phinda ufunde",
        "progress" => "Inkqubela",
        "exam" => "Uviwo",
        "greeting_morning" => "Molo kusasa",
        "greeting_afternoon" => "Molo emini",
        "greeting_evening" => "Molo ngokuhlwa",
        "greeting_hey" => "Molo",
        "what_today" => "Ufuna ukwenza ntoni namhlanje?",
        "begin_session" => "Qala iseshoni yakho",
        "knowledge_garden" => "Igadi Yolwazi",
        "language" => "Ulwimi",
        _ => t_en(key),
    }
}

fn t_fr(key: &str) -> &'static str {
    match key {
        "home" => "Accueil",
        "explore" => "Explorer",
        "review" => "R\u{00E9}viser",
        "progress" => "Progr\u{00E8}s",
        "exam" => "Examen",
        "pathways" => "Parcours",
        "search" => "Rechercher...",
        "greeting_morning" => "Bonjour",
        "greeting_afternoon" => "Bon apr\u{00E8}s-midi",
        "greeting_evening" => "Bonsoir",
        "what_today" => "Que voulez-vous faire aujourd'hui ?",
        "begin_session" => "Commencer votre session",
        "knowledge_garden" => "Jardin du Savoir",
        "language" => "Langue",
        _ => t_en(key),
    }
}

fn t_es(key: &str) -> &'static str {
    match key {
        "home" => "Inicio",
        "explore" => "Explorar",
        "review" => "Repasar",
        "progress" => "Progreso",
        "exam" => "Examen",
        "search" => "Buscar temas...",
        "greeting_morning" => "Buenos d\u{00ED}as",
        "greeting_afternoon" => "Buenas tardes",
        "greeting_evening" => "Buenas noches",
        "what_today" => "\u{00BF}Qu\u{00E9} quieres hacer hoy?",
        "begin_session" => "Comenzar tu sesi\u{00F3}n",
        "knowledge_garden" => "Jard\u{00ED}n del Conocimiento",
        "language" => "Idioma",
        _ => t_en(key),
    }
}

fn t_pt(key: &str) -> &'static str {
    match key {
        "home" => "In\u{00ED}cio",
        "explore" => "Explorar",
        "review" => "Revisar",
        "progress" => "Progresso",
        "exam" => "Exame",
        "greeting_morning" => "Bom dia",
        "greeting_afternoon" => "Boa tarde",
        "greeting_evening" => "Boa noite",
        "what_today" => "O que voc\u{00EA} quer fazer hoje?",
        "begin_session" => "Iniciar sua sess\u{00E3}o",
        "language" => "Idioma",
        _ => t_en(key),
    }
}

fn t_sw(key: &str) -> &'static str {
    match key {
        "home" => "Nyumbani",
        "explore" => "Chunguza",
        "review" => "Kagua",
        "progress" => "Maendeleo",
        "exam" => "Mtihani",
        "greeting_morning" => "Habari za asubuhi",
        "greeting_afternoon" => "Habari za mchana",
        "greeting_evening" => "Habari za jioni",
        "what_today" => "Unataka kufanya nini leo?",
        "begin_session" => "Anza kipindi chako",
        "language" => "Lugha",
        _ => t_en(key),
    }
}

// ============================================================
// Language context provider
// ============================================================

const LANG_KEY: &str = "praxis_lang";

/// Detect browser language
fn detect_language() -> Lang {
    web_sys::window()
        .and_then(|w| w.navigator().language())
        .map(|lang| {
            let code = lang.split('-').next().unwrap_or("en");
            match code {
                "zu" => Lang::Zu,
                "af" => Lang::Af,
                "st" => Lang::St,
                "xh" => Lang::Xh,
                "tn" => Lang::Tn,
                "nso" => Lang::Ns,
                "ts" => Lang::Ts,
                "ss" => Lang::Ss,
                "ve" => Lang::Ve,
                "nr" => Lang::Nr,
                "fr" => Lang::Fr,
                "pt" => Lang::Pt,
                "es" => Lang::Es,
                "sw" => Lang::Sw,
                "ar" => Lang::Ar,
                "hi" => Lang::Hi,
                "zh" => Lang::Zh,
                _ => Lang::En,
            }
        })
        .unwrap_or(Lang::En)
}

pub fn provide_i18n() {
    let initial = persistence::load::<Lang>(LANG_KEY).unwrap_or_else(detect_language);
    let (lang, set_lang) = signal(initial);

    Effect::new(move |_| {
        let l = lang.get();
        persistence::save(LANG_KEY, &l);
        // Update html lang attribute
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(el) = doc.document_element() {
                let _ = el.set_attribute("lang", l.code());
            }
        }
    });

    provide_context(lang);
    provide_context(set_lang);
}

pub fn use_lang() -> ReadSignal<Lang> {
    expect_context::<ReadSignal<Lang>>()
}

pub fn use_set_lang() -> WriteSignal<Lang> {
    expect_context::<WriteSignal<Lang>>()
}

/// Language picker component
#[component]
pub fn LanguagePicker() -> impl IntoView {
    let lang = use_lang();
    let set_lang = use_set_lang();

    view! {
        <select
            class="lang-picker"
            prop:value=move || lang.get().code().to_string()
            on:change=move |ev| {
                let code = leptos::prelude::event_target_value(&ev);
                let new_lang = match code.as_str() {
                    "zu" => Lang::Zu, "af" => Lang::Af, "st" => Lang::St,
                    "xh" => Lang::Xh, "tn" => Lang::Tn, "nso" => Lang::Ns,
                    "ts" => Lang::Ts, "ss" => Lang::Ss, "ve" => Lang::Ve,
                    "nr" => Lang::Nr, "fr" => Lang::Fr, "pt" => Lang::Pt,
                    "es" => Lang::Es, "sw" => Lang::Sw, "ar" => Lang::Ar,
                    "hi" => Lang::Hi, "zh" => Lang::Zh, _ => Lang::En,
                };
                set_lang.set(new_lang);
            }
        >
            <optgroup label="South Africa">
                {Lang::sa_languages().iter().map(|l| {
                    let code = l.code();
                    let label = l.label();
                    view! { <option value=code>{label}</option> }
                }).collect::<Vec<_>>()}
            </optgroup>
            <optgroup label="Global">
                {Lang::global_languages().iter().map(|l| {
                    let code = l.code();
                    let label = l.label();
                    view! { <option value=code>{label}</option> }
                }).collect::<Vec<_>>()}
            </optgroup>
        </select>
    }
}
