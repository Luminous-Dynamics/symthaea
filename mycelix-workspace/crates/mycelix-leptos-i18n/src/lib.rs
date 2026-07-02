// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared internationalization for all Mycelix cluster frontends.
//!
//! Supports all 11 SA official languages + 7 global languages.
//! Provides shared UI keys (home, save, cancel, search, loading, etc.).
//! Domain-specific keys stay in each cluster frontend — use
//! [`t_or`] with a fallback for domain keys not in the shared set.

use leptos::prelude::*;

/// Supported languages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Lang {
    En, // English
    Zu, // isiZulu
    Af, // Afrikaans
    St, // Sesotho
    Xh, // isiXhosa
    Tn, // Setswana
    Ns, // Sepedi (Northern Sotho)
    Ts, // Xitsonga
    Ss, // siSwati
    Ve, // Tshivenda
    Nr, // isiNdebele
    Fr, // French
    Pt, // Portuguese
    Es, // Spanish
    Sw, // Kiswahili
    Ar, // Arabic
    Hi, // Hindi
    Zh, // Chinese (Simplified)
}

impl Lang {
    pub fn code(&self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Zu => "zu",
            Lang::Af => "af",
            Lang::St => "st",
            Lang::Xh => "xh",
            Lang::Tn => "tn",
            Lang::Ns => "nso",
            Lang::Ts => "ts",
            Lang::Ss => "ss",
            Lang::Ve => "ve",
            Lang::Nr => "nr",
            Lang::Fr => "fr",
            Lang::Pt => "pt",
            Lang::Es => "es",
            Lang::Sw => "sw",
            Lang::Ar => "ar",
            Lang::Hi => "hi",
            Lang::Zh => "zh",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::Zu => "isiZulu",
            Lang::Af => "Afrikaans",
            Lang::St => "Sesotho",
            Lang::Xh => "isiXhosa",
            Lang::Tn => "Setswana",
            Lang::Ns => "Sepedi",
            Lang::Ts => "Xitsonga",
            Lang::Ss => "siSwati",
            Lang::Ve => "Tshivenda",
            Lang::Nr => "isiNdebele",
            Lang::Fr => "Fran\u{00E7}ais",
            Lang::Pt => "Portugu\u{00EA}s",
            Lang::Es => "Espa\u{00F1}ol",
            Lang::Sw => "Kiswahili",
            Lang::Ar => "\u{0627}\u{0644}\u{0639}\u{0631}\u{0628}\u{064A}\u{0629}",
            Lang::Hi => "\u{0939}\u{093F}\u{0928}\u{094D}\u{0926}\u{0940}",
            Lang::Zh => "\u{4E2D}\u{6587}",
        }
    }

    pub fn sa_languages() -> &'static [Lang] {
        &[
            Lang::En,
            Lang::Zu,
            Lang::Af,
            Lang::St,
            Lang::Xh,
            Lang::Tn,
            Lang::Ns,
            Lang::Ts,
            Lang::Ss,
            Lang::Ve,
            Lang::Nr,
        ]
    }

    pub fn global_languages() -> &'static [Lang] {
        &[
            Lang::Fr,
            Lang::Pt,
            Lang::Es,
            Lang::Sw,
            Lang::Ar,
            Lang::Hi,
            Lang::Zh,
        ]
    }

    pub fn all() -> &'static [Lang] {
        &[
            Lang::En,
            Lang::Zu,
            Lang::Af,
            Lang::St,
            Lang::Xh,
            Lang::Tn,
            Lang::Ns,
            Lang::Ts,
            Lang::Ss,
            Lang::Ve,
            Lang::Nr,
            Lang::Fr,
            Lang::Pt,
            Lang::Es,
            Lang::Sw,
            Lang::Ar,
            Lang::Hi,
            Lang::Zh,
        ]
    }

    fn from_code(code: &str) -> Lang {
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
    }
}

// ---------------------------------------------------------------------------
// Shared translation keys (UI chrome only — domain keys stay in each cluster)
// ---------------------------------------------------------------------------

/// Translate a shared UI key. Returns "?" for unknown keys.
pub fn t(key: &str, lang: Lang) -> &'static str {
    match lang {
        Lang::Zu => t_zu(key),
        Lang::Af => t_af(key),
        Lang::Fr => t_fr(key),
        Lang::Es => t_es(key),
        Lang::Pt => t_pt(key),
        Lang::Sw => t_sw(key),
        Lang::St => t_st(key),
        Lang::Xh => t_xh(key),
        _ => t_en(key),
    }
}

/// Translate with a fallback for domain-specific keys.
pub fn t_or(key: &str, lang: Lang, fallback: &'static str) -> &'static str {
    let result = t(key, lang);
    if result == "?" { fallback } else { result }
}

fn t_en(key: &str) -> &'static str {
    match key {
        // Navigation
        "home" => "Home",
        "profile" => "Profile",
        "settings" => "Settings",
        "back" => "Back",
        // Actions
        "save" => "Save",
        "cancel" => "Cancel",
        "close" => "Close",
        "submit" => "Submit",
        "confirm" => "Confirm",
        "delete" => "Delete",
        "edit" => "Edit",
        "create" => "Create",
        "search" => "Search...",
        "filter" => "Filter",
        "refresh" => "Refresh",
        // Status
        "loading" => "Loading...",
        "error" => "Error",
        "success" => "Success",
        "warning" => "Warning",
        "no_data" => "No data",
        "not_found" => "Not found",
        "connected" => "Connected",
        "disconnected" => "Disconnected",
        "connecting" => "Connecting...",
        "mock_mode" => "Mock mode",
        // Consciousness
        "observer" => "Observer",
        "participant" => "Participant",
        "citizen" => "Citizen",
        "steward" => "Steward",
        "guardian" => "Guardian",
        // Time
        "greeting_morning" => "Good morning",
        "greeting_afternoon" => "Good afternoon",
        "greeting_evening" => "Good evening",
        // Common
        "language" => "Language",
        "theme" => "Theme",
        "dark" => "Dark",
        "light" => "Light",
        _ => "?",
    }
}

fn t_zu(key: &str) -> &'static str {
    match key {
        "home" => "Ikhaya",
        "profile" => "Iphrofayela",
        "settings" => "Izilungiselelo",
        "back" => "Emuva",
        "save" => "Gcina",
        "cancel" => "Khansela",
        "close" => "Vala",
        "submit" => "Thumela",
        "confirm" => "Qinisekisa",
        "delete" => "Susa",
        "edit" => "Hlela",
        "create" => "Dala",
        "search" => "Sesha...",
        "loading" => "Iyalayisha...",
        "error" => "Iphutha",
        "success" => "Kuphumelele",
        "no_data" => "Ayikho idatha",
        "connected" => "Ixhunyiwe",
        "disconnected" => "Inqanyuliwe",
        "greeting_morning" => "Sawubona ekuseni",
        "greeting_afternoon" => "Sawubona ntambama",
        "greeting_evening" => "Sawubona kusihlwa",
        "language" => "Ulimi",
        _ => t_en(key),
    }
}

fn t_af(key: &str) -> &'static str {
    match key {
        "home" => "Tuis",
        "profile" => "Profiel",
        "settings" => "Instellings",
        "back" => "Terug",
        "save" => "Stoor",
        "cancel" => "Kanselleer",
        "close" => "Sluit",
        "submit" => "Dien in",
        "confirm" => "Bevestig",
        "delete" => "Verwyder",
        "edit" => "Wysig",
        "create" => "Skep",
        "search" => "Soek...",
        "loading" => "Laai...",
        "error" => "Fout",
        "success" => "Sukses",
        "no_data" => "Geen data",
        "greeting_morning" => "Goeiem\u{00F4}re",
        "greeting_afternoon" => "Goeiemiddag",
        "greeting_evening" => "Goeienaand",
        "language" => "Taal",
        _ => t_en(key),
    }
}

fn t_st(key: &str) -> &'static str {
    match key {
        "home" => "Lapeng",
        "save" => "Boloka",
        "cancel" => "Hlakola",
        "search" => "Batla...",
        "loading" => "E ntse e kenya...",
        "greeting_morning" => "Dumela hoseng",
        "greeting_afternoon" => "Dumela motsheare",
        "greeting_evening" => "Dumela mantsibuya",
        "language" => "Puo",
        _ => t_en(key),
    }
}

fn t_xh(key: &str) -> &'static str {
    match key {
        "home" => "Ikhaya",
        "save" => "Gcina",
        "cancel" => "Rhoxisa",
        "search" => "Khangela...",
        "loading" => "Iyalayisha...",
        "greeting_morning" => "Molo kusasa",
        "greeting_afternoon" => "Molo emini",
        "greeting_evening" => "Molo ngokuhlwa",
        "language" => "Ulwimi",
        _ => t_en(key),
    }
}

fn t_fr(key: &str) -> &'static str {
    match key {
        "home" => "Accueil",
        "profile" => "Profil",
        "settings" => "Param\u{00E8}tres",
        "back" => "Retour",
        "save" => "Enregistrer",
        "cancel" => "Annuler",
        "close" => "Fermer",
        "submit" => "Soumettre",
        "confirm" => "Confirmer",
        "delete" => "Supprimer",
        "edit" => "Modifier",
        "create" => "Cr\u{00E9}er",
        "search" => "Rechercher...",
        "loading" => "Chargement...",
        "error" => "Erreur",
        "success" => "Succ\u{00E8}s",
        "no_data" => "Aucune donn\u{00E9}e",
        "greeting_morning" => "Bonjour",
        "greeting_afternoon" => "Bon apr\u{00E8}s-midi",
        "greeting_evening" => "Bonsoir",
        "language" => "Langue",
        _ => t_en(key),
    }
}

fn t_es(key: &str) -> &'static str {
    match key {
        "home" => "Inicio",
        "profile" => "Perfil",
        "settings" => "Ajustes",
        "save" => "Guardar",
        "cancel" => "Cancelar",
        "close" => "Cerrar",
        "submit" => "Enviar",
        "search" => "Buscar...",
        "loading" => "Cargando...",
        "error" => "Error",
        "success" => "\u{00C9}xito",
        "greeting_morning" => "Buenos d\u{00ED}as",
        "greeting_afternoon" => "Buenas tardes",
        "greeting_evening" => "Buenas noches",
        "language" => "Idioma",
        _ => t_en(key),
    }
}

fn t_pt(key: &str) -> &'static str {
    match key {
        "home" => "In\u{00ED}cio",
        "profile" => "Perfil",
        "save" => "Salvar",
        "cancel" => "Cancelar",
        "search" => "Pesquisar...",
        "loading" => "Carregando...",
        "greeting_morning" => "Bom dia",
        "greeting_afternoon" => "Boa tarde",
        "greeting_evening" => "Boa noite",
        "language" => "Idioma",
        _ => t_en(key),
    }
}

fn t_sw(key: &str) -> &'static str {
    match key {
        "home" => "Nyumbani",
        "save" => "Hifadhi",
        "cancel" => "Ghairi",
        "search" => "Tafuta...",
        "loading" => "Inapakia...",
        "greeting_morning" => "Habari za asubuhi",
        "greeting_afternoon" => "Habari za mchana",
        "greeting_evening" => "Habari za jioni",
        "language" => "Lugha",
        _ => t_en(key),
    }
}

// ---------------------------------------------------------------------------
// Context provider
// ---------------------------------------------------------------------------

/// Initialize i18n context. Call once at app root.
///
/// Detects browser language, restores from localStorage, persists changes,
/// and sets the `<html lang>` attribute.
pub fn provide_i18n(storage_key: &'static str) {
    let initial = load_lang(storage_key).unwrap_or_else(detect_language);
    let (lang, set_lang) = signal(initial);

    Effect::new(move |_| {
        let l = lang.get();
        save_lang(storage_key, l);
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(el) = doc.document_element() {
                let _ = el.set_attribute("lang", l.code());
            }
        }
    });

    provide_context(lang);
    provide_context(set_lang);
}

/// Retrieve current language signal.
pub fn use_lang() -> ReadSignal<Lang> {
    expect_context::<ReadSignal<Lang>>()
}

/// Retrieve language setter.
pub fn use_set_lang() -> WriteSignal<Lang> {
    expect_context::<WriteSignal<Lang>>()
}

/// Language picker component with SA + Global optgroups.
#[component]
pub fn LanguagePicker() -> impl IntoView {
    let lang = use_lang();
    let set_lang = use_set_lang();

    view! {
        <select
            class="lang-picker form-select"
            prop:value=move || lang.get().code().to_string()
            on:change=move |ev| {
                let code = event_target_value(&ev);
                set_lang.set(Lang::from_code(&code));
            }
        >
            <optgroup label="South Africa">
                {Lang::sa_languages().iter().map(|l| {
                    let code = l.code();
                    let label = l.label();
                    view! { <option value=code>{label}</option> }
                }).collect_view()}
            </optgroup>
            <optgroup label="Global">
                {Lang::global_languages().iter().map(|l| {
                    let code = l.code();
                    let label = l.label();
                    view! { <option value=code>{label}</option> }
                }).collect_view()}
            </optgroup>
        </select>
    }
}

// ---------------------------------------------------------------------------
// localStorage helpers
// ---------------------------------------------------------------------------

fn detect_language() -> Lang {
    web_sys::window()
        .and_then(|w| w.navigator().language())
        .map(|lang| {
            let code = lang.split('-').next().unwrap_or("en");
            Lang::from_code(code)
        })
        .unwrap_or(Lang::En)
}

fn load_lang(key: &str) -> Option<Lang> {
    let storage = web_sys::window()?.local_storage().ok()??;
    let val = storage.get_item(key).ok()??;
    serde_json::from_str(&format!("\"{val}\""))
        .ok()
        .map(|code: String| Lang::from_code(&code))
}

fn save_lang(key: &str, lang: Lang) {
    if let Some(Ok(Some(storage))) = web_sys::window().map(|w| w.local_storage()) {
        let _ = storage.set_item(key, lang.code());
    }
}
