// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightweight i18n for NixForHumanity installer.
//! No external dependencies — just static string maps.

use std::collections::HashMap;
use std::sync::LazyLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lang {
    En,
    De,
    Fr,
    Es,
    Pt,
    Ja,
    Zh,
    Ko,
    Ru,
    Ar,
}

impl Lang {
    pub fn code(&self) -> &'static str {
        match self {
            Self::En => "en",
            Self::De => "de",
            Self::Fr => "fr",
            Self::Es => "es",
            Self::Pt => "pt",
            Self::Ja => "ja",
            Self::Zh => "zh",
            Self::Ko => "ko",
            Self::Ru => "ru",
            Self::Ar => "ar",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::En => "English",
            Self::De => "Deutsch",
            Self::Fr => "Fran\u{00e7}ais",
            Self::Es => "Espa\u{00f1}ol",
            Self::Pt => "Portugu\u{00ea}s",
            Self::Ja => "\u{65e5}\u{672c}\u{8a9e}",
            Self::Zh => "\u{4e2d}\u{6587}",
            Self::Ko => "\u{d55c}\u{ad6d}\u{c5b4}",
            Self::Ru => "\u{0420}\u{0443}\u{0441}\u{0441}\u{043a}\u{0438}\u{0439}",
            Self::Ar => "\u{0627}\u{0644}\u{0639}\u{0631}\u{0628}\u{064a}\u{0629}",
        }
    }

    pub fn is_rtl(&self) -> bool {
        matches!(self, Self::Ar)
    }

    pub fn from_code(code: &str) -> Self {
        match code {
            "de" => Self::De,
            "fr" => Self::Fr,
            "es" => Self::Es,
            "pt" => Self::Pt,
            "ja" => Self::Ja,
            "zh" => Self::Zh,
            "ko" => Self::Ko,
            "ru" => Self::Ru,
            "ar" => Self::Ar,
            _ => Self::En,
        }
    }

    pub fn all() -> &'static [Lang] {
        &[
            Self::En,
            Self::De,
            Self::Fr,
            Self::Es,
            Self::Pt,
            Self::Ja,
            Self::Zh,
            Self::Ko,
            Self::Ru,
            Self::Ar,
        ]
    }
}

/// Detect language from browser navigator.language
pub fn detect_browser_lang() -> Lang {
    web_sys::window()
        .and_then(|w| w.navigator().language())
        .map(|l| {
            let code = l.split('-').next().unwrap_or("en");
            Lang::from_code(code)
        })
        .unwrap_or(Lang::En)
}

/// Translation lookup — returns English fallback if key not found in target lang.
/// If key is missing from all translations, returns "???" (should never happen in practice).
pub fn t(lang: Lang, key: &str) -> &'static str {
    let map = translations(lang);
    map.get(key).copied().unwrap_or_else(move || {
        // Fallback to English
        translations(Lang::En).get(key).copied().unwrap_or("???")
    })
}

fn translations(lang: Lang) -> &'static HashMap<&'static str, &'static str> {
    match lang {
        Lang::En => &EN,
        Lang::De => &DE,
        Lang::Fr => &FR,
        Lang::Es => &ES,
        Lang::Pt => &PT,
        Lang::Ja => &JA,
        Lang::Zh => &ZH,
        Lang::Ko => &KO,
        Lang::Ru => &RU,
        Lang::Ar => &AR,
    }
}

macro_rules! translations {
    ($($key:expr => $val:expr),* $(,)?) => {{
        let mut m = HashMap::new();
        $(m.insert($key, $val);)*
        m
    }};
}

// ═══════════════════════════════════════════════════════
// English (complete — canonical keys)
// ═══════════════════════════════════════════════════════

static EN: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    translations! {
        // Hero
        "install_title" => "Install NixOS",
        "install_subtitle" => "From your browser. Encrypted. Reproducible. Yours.",
        "nixos_intro_toggle" => "New to NixOS? Learn what makes it different.",
        "nixos_intro_text" => "NixOS is a Linux distribution where your entire system is defined in one configuration file. Every change is reproducible \u{2014} you can rebuild your exact setup on any machine. Every update is atomic and reversible \u{2014} if something breaks, roll back in seconds. NixOS has 100,000+ packages, the largest collection of any Linux distribution.",

        // Wizard steps
        "step_system" => "Your System",
        "step_desktop" => "Desktop & Security",
        "step_apps" => "Your Apps",
        "step_config" => "Review Config",
        "step_install" => "Install",

        // Step 1: System
        "hostname" => "Hostname",
        "hostname_hint" => "Your machine\u{2019}s network name",
        "username" => "Username",
        "username_hint" => "Your login name (set password during install)",
        "timezone" => "Timezone",
        "timezone_hint" => "Auto-detected from your browser",
        "keyboard" => "Keyboard",
        "keyboard_hint" => "Console keymap (us, gb, de, fr, ...)",
        "locale" => "Locale",
        "platform" => "Platform",
        "cpu" => "CPU",
        "memory" => "Memory",
        "gpu" => "GPU",
        "display" => "Display",

        // Step 2: Desktop
        "desktop_title" => "Desktop Environment",
        "desktop_desc" => "Choose how your system looks and feels.",
        "security_title" => "Security",
        "security_desc" => "Protect your system from day one.",
        "encryption_label" => "Full disk encryption (LUKS2)",
        "encryption_hint" => "You will set a passphrase during install. Required to boot.",
        "secureboot_label" => "Secure Boot (lanzaboote)",
        "secureboot_hint" => "Keys are enrolled in firmware on first boot.",
        "tpm2_label" => "TPM2 auto-unlock",
        "tpm2_hint" => "Auto-unlocks disk using TPM chip. Falls back to passphrase.",
        "fido2_label" => "FIDO2/YubiKey unlock",
        "fido2_hint" => "Unlock disk with a hardware security key. Touch key at boot.",
        "filesystem_label" => "Filesystem",
        "btrfs_label" => "btrfs",
        "btrfs_desc" => "Recommended. Snapshots, compression, self-healing.",
        "zfs_label" => "ZFS",
        "zfs_desc" => "Advanced. Native encryption, RAID-Z, enterprise features.",
        "disk_wipe" => "Wipe entire disk",
        "disk_wipe_desc" => "Erases everything on the selected disk and installs NixOS. All existing data will be permanently deleted.",
        "disk_alongside" => "Install alongside existing OS",
        "disk_alongside_desc" => "Installs NixOS next to your current OS. Your existing OS and files are preserved. Requires at least 20GB free.",
        "optional" => "Optional",

        // Quick presets
        "presets_title" => "Quick Presets",
        "presets_desc" => "Choose a preset to auto-fill desktop, apps, and security settings.",
        "preset_dev" => "Developer Workstation",
        "preset_dev_desc" => "GNOME + VS Code + Docker + Git + terminal",
        "preset_server" => "Server",
        "preset_server_desc" => "No desktop, encrypted, SSH enabled",
        "preset_home" => "Home / Office",
        "preset_home_desc" => "GNOME + browser + office + media",
        "preset_gaming" => "Gaming",
        "preset_gaming_desc" => "Hyprland + Steam + Lutris + dev tools",

        // Step 3: Apps
        "apps_title" => "What do you use?",
        "apps_desc" => "Check your apps. We show what\u{2019}s available on NixOS.",
        "apps_search" => "Search apps...",
        "apps_scan" => "Scan My System",
        "apps_selected" => "apps selected",
        "apps_custom" => "Additional nixpkgs (comma-separated)",
        "apps_custom_hint" => "Type any nixpkgs attribute names. These are added to your config as-is.",

        // Step 4: Config
        "config_title" => "Your Configuration",
        "config_generate" => "Generate & Download",
        "config_preview" => "Preview",
        "profiles_title" => "Configuration Profiles",
        "profiles_desc" => "Save this config as a profile to reuse on other machines.",
        "profiles_save" => "Save Profile",
        "profiles_load" => "Load",
        "profiles_delete" => "Delete",
        "profiles_saved" => "Saved Profiles",

        // Step 5: Install
        "install_auto_title" => "Automated Install",
        "install_auto_desc" => "Boot the target machine from the NixForHumanity USB. Enter the IP address shown on its screen.",
        "target_addr" => "Target Address",
        "password" => "Password",
        "connect" => "Connect",

        // Buttons
        "next" => "Next",
        "back" => "Back",
        "start_over" => "Start Over",
        "what_means" => "What does this mean?",

        // Footer
        "footer_source" => "Source Code",
        "footer_no_tracking" => "No tracking \u{00b7} No data collection",

        // Post-install
        "install_complete" => "NixOS Installed Successfully!",
        "remove_usb" => "Remove the USB drive and reboot your machine.",
    }
});

// ═══════════════════════════════════════════════════════
// German
// ═══════════════════════════════════════════════════════

static DE: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    translations! {
        "install_title" => "NixOS installieren",
        "install_subtitle" => "Aus dem Browser. Verschl\u{00fc}sselt. Reproduzierbar. Deins.",
        "nixos_intro_toggle" => "Neu bei NixOS? Erfahre, was es besonders macht.",
        "step_system" => "Dein System",
        "step_desktop" => "Desktop & Sicherheit",
        "step_apps" => "Deine Apps",
        "step_config" => "Konfiguration pr\u{00fc}fen",
        "step_install" => "Installieren",
        "hostname" => "Hostname",
        "hostname_hint" => "Der Netzwerkname deines Rechners",
        "username" => "Benutzername",
        "username_hint" => "Dein Anmeldename (Passwort wird bei Installation gesetzt)",
        "timezone" => "Zeitzone",
        "timezone_hint" => "Automatisch erkannt",
        "keyboard" => "Tastatur",
        "keyboard_hint" => "Tastaturbelegung (us, gb, de, fr, ...)",
        "locale" => "Sprache",
        "platform" => "Plattform",
        "cpu" => "CPU",
        "memory" => "Arbeitsspeicher",
        "gpu" => "GPU",
        "display" => "Bildschirm",
        "desktop_title" => "Desktop-Umgebung",
        "desktop_desc" => "W\u{00e4}hle das Aussehen deines Systems.",
        "security_title" => "Sicherheit",
        "security_desc" => "Sch\u{00fc}tze dein System von Anfang an.",
        "encryption_label" => "Vollst\u{00e4}ndige Festplattenverschl\u{00fc}sselung (LUKS2)",
        "encryption_hint" => "Du setzt eine Passphrase bei der Installation.",
        "secureboot_label" => "Secure Boot (lanzaboote)",
        "tpm2_label" => "TPM2 Auto-Entsperrung",
        "fido2_label" => "FIDO2/YubiKey Entsperrung",
        "disk_wipe" => "Gesamte Festplatte l\u{00f6}schen",
        "disk_alongside" => "Neben bestehendem OS installieren",
        "presets_title" => "Schnellvorlagen",
        "preset_dev" => "Entwickler-Arbeitsplatz",
        "preset_dev_desc" => "GNOME + VS Code + Docker + Git + Terminal",
        "preset_server" => "Server",
        "preset_server_desc" => "Kein Desktop, verschl\u{00fc}sselt, SSH aktiviert",
        "preset_home" => "Zuhause / B\u{00fc}ro",
        "preset_home_desc" => "GNOME + Browser + Office + Medien",
        "preset_gaming" => "Gaming",
        "preset_gaming_desc" => "Hyprland + Steam + Lutris + Entwicklertools",
        "apps_title" => "Was verwendest du?",
        "apps_desc" => "W\u{00e4}hle deine Apps. Wir zeigen, was auf NixOS verf\u{00fc}gbar ist.",
        "apps_search" => "Apps suchen...",
        "apps_scan" => "System scannen",
        "apps_selected" => "Apps ausgew\u{00e4}hlt",
        "apps_custom" => "Zus\u{00e4}tzliche nixpkgs (kommagetrennt)",
        "config_title" => "Deine Konfiguration",
        "config_generate" => "Generieren & Herunterladen",
        "config_preview" => "Vorschau",
        "profiles_title" => "Konfigurationsprofile",
        "profiles_save" => "Profil speichern",
        "profiles_load" => "Laden",
        "profiles_delete" => "L\u{00f6}schen",
        "profiles_saved" => "Gespeicherte Profile",
        "next" => "Weiter",
        "back" => "Zur\u{00fc}ck",
        "start_over" => "Neu beginnen",
        "connect" => "Verbinden",
        "password" => "Passwort",
        "target_addr" => "Zieladresse",
        "what_means" => "Was bedeutet das?",
        "optional" => "Optional",
        "footer_no_tracking" => "Kein Tracking \u{00b7} Keine Datenerfassung",
        "install_complete" => "NixOS erfolgreich installiert!",
    }
});

// ═══════════════════════════════════════════════════════
// French
// ═══════════════════════════════════════════════════════

static FR: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    translations! {
        "install_title" => "Installer NixOS",
        "install_subtitle" => "Depuis votre navigateur. Chiffr\u{00e9}. Reproductible. Le v\u{00f4}tre.",
        "nixos_intro_toggle" => "Nouveau sur NixOS ? D\u{00e9}couvrez ce qui le rend diff\u{00e9}rent.",
        "step_system" => "Votre Syst\u{00e8}me",
        "step_desktop" => "Bureau & S\u{00e9}curit\u{00e9}",
        "step_apps" => "Vos Applications",
        "step_config" => "V\u{00e9}rifier la Config",
        "step_install" => "Installer",
        "hostname" => "Nom d\u{2019}h\u{00f4}te",
        "username" => "Nom d\u{2019}utilisateur",
        "timezone" => "Fuseau horaire",
        "keyboard" => "Clavier",
        "locale" => "Langue",
        "platform" => "Plateforme",
        "desktop_title" => "Environnement de Bureau",
        "desktop_desc" => "Choisissez l\u{2019}apparence de votre syst\u{00e8}me.",
        "security_title" => "S\u{00e9}curit\u{00e9}",
        "encryption_label" => "Chiffrement complet du disque (LUKS2)",
        "presets_title" => "Pr\u{00e9}r\u{00e9}glages rapides",
        "preset_dev" => "Station de d\u{00e9}veloppement",
        "preset_server" => "Serveur",
        "preset_home" => "Maison / Bureau",
        "preset_gaming" => "Jeux",
        "apps_title" => "Qu\u{2019}utilisez-vous ?",
        "apps_search" => "Rechercher des apps...",
        "apps_scan" => "Scanner mon syst\u{00e8}me",
        "config_title" => "Votre Configuration",
        "config_generate" => "G\u{00e9}n\u{00e9}rer & T\u{00e9}l\u{00e9}charger",
        "config_preview" => "Aper\u{00e7}u",
        "profiles_title" => "Profils de configuration",
        "profiles_save" => "Enregistrer le profil",
        "profiles_load" => "Charger",
        "profiles_delete" => "Supprimer",
        "next" => "Suivant",
        "back" => "Retour",
        "start_over" => "Recommencer",
        "connect" => "Connecter",
        "password" => "Mot de passe",
        "what_means" => "Qu\u{2019}est-ce que cela signifie ?",
        "optional" => "Optionnel",
        "install_complete" => "NixOS install\u{00e9} avec succ\u{00e8}s !",
    }
});

// ═══════════════════════════════════════════════════════
// Spanish
// ═══════════════════════════════════════════════════════

static ES: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    translations! {
        "install_title" => "Instalar NixOS",
        "install_subtitle" => "Desde tu navegador. Cifrado. Reproducible. Tuyo.",
        "nixos_intro_toggle" => "\u{00bf}Nuevo en NixOS? Descubre qu\u{00e9} lo hace diferente.",
        "step_system" => "Tu Sistema",
        "step_desktop" => "Escritorio y Seguridad",
        "step_apps" => "Tus Apps",
        "step_config" => "Revisar Config",
        "step_install" => "Instalar",
        "hostname" => "Nombre de host",
        "username" => "Nombre de usuario",
        "timezone" => "Zona horaria",
        "keyboard" => "Teclado",
        "locale" => "Idioma",
        "platform" => "Plataforma",
        "desktop_title" => "Entorno de escritorio",
        "desktop_desc" => "Elige c\u{00f3}mo se ve y se siente tu sistema.",
        "security_title" => "Seguridad",
        "encryption_label" => "Cifrado completo del disco (LUKS2)",
        "presets_title" => "Preajustes r\u{00e1}pidos",
        "preset_dev" => "Estaci\u{00f3}n de desarrollo",
        "preset_server" => "Servidor",
        "preset_home" => "Hogar / Oficina",
        "preset_gaming" => "Juegos",
        "apps_title" => "\u{00bf}Qu\u{00e9} usas?",
        "apps_search" => "Buscar apps...",
        "apps_scan" => "Escanear mi sistema",
        "config_title" => "Tu Configuraci\u{00f3}n",
        "config_generate" => "Generar y Descargar",
        "config_preview" => "Vista previa",
        "next" => "Siguiente",
        "back" => "Atr\u{00e1}s",
        "start_over" => "Empezar de nuevo",
        "connect" => "Conectar",
        "password" => "Contrase\u{00f1}a",
        "what_means" => "\u{00bf}Qu\u{00e9} significa esto?",
        "optional" => "Opcional",
        "install_complete" => "\u{00a1}NixOS instalado con \u{00e9}xito!",
    }
});

// ═══════════════════════════════════════════════════════
// Portuguese
// ═══════════════════════════════════════════════════════

static PT: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    translations! {
        "install_title" => "Instalar NixOS",
        "install_subtitle" => "Do seu navegador. Criptografado. Reprodut\u{00ed}vel. Seu.",
        "step_system" => "Seu Sistema",
        "step_desktop" => "Desktop e Seguran\u{00e7}a",
        "step_apps" => "Seus Apps",
        "step_config" => "Revisar Config",
        "step_install" => "Instalar",
        "hostname" => "Nome do host",
        "username" => "Nome de usu\u{00e1}rio",
        "next" => "Pr\u{00f3}ximo",
        "back" => "Voltar",
        "start_over" => "Recome\u{00e7}ar",
        "connect" => "Conectar",
        "password" => "Senha",
        "apps_search" => "Buscar apps...",
        "config_generate" => "Gerar e Baixar",
        "install_complete" => "NixOS instalado com sucesso!",
    }
});

// ═══════════════════════════════════════════════════════
// Japanese
// ═══════════════════════════════════════════════════════

static JA: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    translations! {
        "install_title" => "NixOS\u{3092}\u{30a4}\u{30f3}\u{30b9}\u{30c8}\u{30fc}\u{30eb}",
        "install_subtitle" => "\u{30d6}\u{30e9}\u{30a6}\u{30b6}\u{304b}\u{3089}\u{3002}\u{6697}\u{53f7}\u{5316}\u{3002}\u{518d}\u{73fe}\u{53ef}\u{80fd}\u{3002}\u{3042}\u{306a}\u{305f}\u{306e}\u{3082}\u{306e}\u{3002}",
        "step_system" => "\u{30b7}\u{30b9}\u{30c6}\u{30e0}",
        "step_desktop" => "\u{30c7}\u{30b9}\u{30af}\u{30c8}\u{30c3}\u{30d7}\u{3068}\u{30bb}\u{30ad}\u{30e5}\u{30ea}\u{30c6}\u{30a3}",
        "step_apps" => "\u{30a2}\u{30d7}\u{30ea}",
        "step_config" => "\u{8a2d}\u{5b9a}\u{78ba}\u{8a8d}",
        "step_install" => "\u{30a4}\u{30f3}\u{30b9}\u{30c8}\u{30fc}\u{30eb}",
        "next" => "\u{6b21}\u{3078}",
        "back" => "\u{623b}\u{308b}",
        "start_over" => "\u{6700}\u{521d}\u{304b}\u{3089}\u{3084}\u{308a}\u{76f4}\u{3059}",
        "connect" => "\u{63a5}\u{7d9a}",
        "apps_search" => "\u{30a2}\u{30d7}\u{30ea}\u{3092}\u{691c}\u{7d22}...",
        "config_generate" => "\u{751f}\u{6210}\u{3057}\u{3066}\u{30c0}\u{30a6}\u{30f3}\u{30ed}\u{30fc}\u{30c9}",
        "install_complete" => "NixOS\u{306e}\u{30a4}\u{30f3}\u{30b9}\u{30c8}\u{30fc}\u{30eb}\u{304c}\u{5b8c}\u{4e86}\u{3057}\u{307e}\u{3057}\u{305f}\u{ff01}",
    }
});

// ═══════════════════════════════════════════════════════
// Chinese (Simplified)
// ═══════════════════════════════════════════════════════

static ZH: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    translations! {
        "install_title" => "\u{5b89}\u{88c5} NixOS",
        "install_subtitle" => "\u{901a}\u{8fc7}\u{6d4f}\u{89c8}\u{5668}\u{5b89}\u{88c5}\u{3002}\u{52a0}\u{5bc6}\u{3002}\u{53ef}\u{590d}\u{73b0}\u{3002}\u{5c5e}\u{4e8e}\u{4f60}\u{3002}",
        "step_system" => "\u{7cfb}\u{7edf}",
        "step_desktop" => "\u{684c}\u{9762}\u{4e0e}\u{5b89}\u{5168}",
        "step_apps" => "\u{5e94}\u{7528}",
        "step_config" => "\u{68c0}\u{67e5}\u{914d}\u{7f6e}",
        "step_install" => "\u{5b89}\u{88c5}",
        "next" => "\u{4e0b}\u{4e00}\u{6b65}",
        "back" => "\u{8fd4}\u{56de}",
        "start_over" => "\u{91cd}\u{65b0}\u{5f00}\u{59cb}",
        "connect" => "\u{8fde}\u{63a5}",
        "apps_search" => "\u{641c}\u{7d22}\u{5e94}\u{7528}...",
        "config_generate" => "\u{751f}\u{6210}\u{5e76}\u{4e0b}\u{8f7d}",
        "install_complete" => "NixOS \u{5b89}\u{88c5}\u{6210}\u{529f}\u{ff01}",
    }
});

// ═══════════════════════════════════════════════════════
// Korean
// ═══════════════════════════════════════════════════════

static KO: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    translations! {
        "install_title" => "NixOS \u{c124}\u{ce58}",
        "install_subtitle" => "\u{be0c}\u{b77c}\u{c6b0}\u{c800}\u{c5d0}\u{c11c}. \u{c554}\u{d638}\u{d654}. \u{c7ac}\u{d604} \u{ac00}\u{b2a5}. \u{b2f9}\u{c2e0}\u{c758} \u{ac83}.",
        "step_system" => "\u{c2dc}\u{c2a4}\u{d15c}",
        "step_desktop" => "\u{b370}\u{c2a4}\u{d06c}\u{d1b1} & \u{bcf4}\u{c548}",
        "step_apps" => "\u{c571}",
        "step_config" => "\u{c124}\u{c815} \u{d655}\u{c778}",
        "step_install" => "\u{c124}\u{ce58}",
        "next" => "\u{b2e4}\u{c74c}",
        "back" => "\u{b4a4}\u{b85c}",
        "start_over" => "\u{cc98}\u{c74c}\u{bd80}\u{d130}",
        "connect" => "\u{c5f0}\u{acb0}",
        "apps_search" => "\u{c571} \u{ac80}\u{c0c9}...",
        "config_generate" => "\u{c0dd}\u{c131} \u{bc0f} \u{b2e4}\u{c6b4}\u{b85c}\u{b4dc}",
        "install_complete" => "NixOS \u{c124}\u{ce58} \u{c644}\u{b8cc}!",
    }
});

// ═══════════════════════════════════════════════════════
// Russian
// ═══════════════════════════════════════════════════════

static RU: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    translations! {
        "install_title" => "\u{0423}\u{0441}\u{0442}\u{0430}\u{043d}\u{043e}\u{0432}\u{0438}\u{0442}\u{044c} NixOS",
        "install_subtitle" => "\u{0418}\u{0437} \u{0431}\u{0440}\u{0430}\u{0443}\u{0437}\u{0435}\u{0440}\u{0430}. \u{0417}\u{0430}\u{0448}\u{0438}\u{0444}\u{0440}\u{043e}\u{0432}\u{0430}\u{043d}\u{043e}. \u{0412}\u{043e}\u{0441}\u{043f}\u{0440}\u{043e}\u{0438}\u{0437}\u{0432}\u{043e}\u{0434}\u{0438}\u{043c}\u{043e}. \u{0412}\u{0430}\u{0448}\u{0435}.",
        "step_system" => "\u{0412}\u{0430}\u{0448}\u{0430} \u{0441}\u{0438}\u{0441}\u{0442}\u{0435}\u{043c}\u{0430}",
        "step_desktop" => "\u{0420}\u{0430}\u{0431}\u{043e}\u{0447}\u{0438}\u{0439} \u{0441}\u{0442}\u{043e}\u{043b} \u{0438} \u{0431}\u{0435}\u{0437}\u{043e}\u{043f}\u{0430}\u{0441}\u{043d}\u{043e}\u{0441}\u{0442}\u{044c}",
        "step_apps" => "\u{041f}\u{0440}\u{0438}\u{043b}\u{043e}\u{0436}\u{0435}\u{043d}\u{0438}\u{044f}",
        "step_config" => "\u{041f}\u{0440}\u{043e}\u{0432}\u{0435}\u{0440}\u{043a}\u{0430} \u{043a}\u{043e}\u{043d}\u{0444}\u{0438}\u{0433}\u{0443}\u{0440}\u{0430}\u{0446}\u{0438}\u{0438}",
        "step_install" => "\u{0423}\u{0441}\u{0442}\u{0430}\u{043d}\u{043e}\u{0432}\u{043a}\u{0430}",
        "next" => "\u{0414}\u{0430}\u{043b}\u{0435}\u{0435}",
        "back" => "\u{041d}\u{0430}\u{0437}\u{0430}\u{0434}",
        "start_over" => "\u{041d}\u{0430}\u{0447}\u{0430}\u{0442}\u{044c} \u{0441}\u{043d}\u{0430}\u{0447}\u{0430}\u{043b}\u{0430}",
        "connect" => "\u{041f}\u{043e}\u{0434}\u{043a}\u{043b}\u{044e}\u{0447}\u{0438}\u{0442}\u{044c}",
        "password" => "\u{041f}\u{0430}\u{0440}\u{043e}\u{043b}\u{044c}",
        "apps_search" => "\u{041f}\u{043e}\u{0438}\u{0441}\u{043a} \u{043f}\u{0440}\u{0438}\u{043b}\u{043e}\u{0436}\u{0435}\u{043d}\u{0438}\u{0439}...",
        "config_generate" => "\u{0421}\u{0433}\u{0435}\u{043d}\u{0435}\u{0440}\u{0438}\u{0440}\u{043e}\u{0432}\u{0430}\u{0442}\u{044c} \u{0438} \u{0441}\u{043a}\u{0430}\u{0447}\u{0430}\u{0442}\u{044c}",
        "install_complete" => "NixOS \u{0443}\u{0441}\u{043f}\u{0435}\u{0448}\u{043d}\u{043e} \u{0443}\u{0441}\u{0442}\u{0430}\u{043d}\u{043e}\u{0432}\u{043b}\u{0435}\u{043d}\u{0430}!",
    }
});

// ═══════════════════════════════════════════════════════
// Arabic
// ═══════════════════════════════════════════════════════

static AR: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    translations! {
        "install_title" => "\u{062a}\u{062b}\u{0628}\u{064a}\u{062a} NixOS",
        "install_subtitle" => "\u{0645}\u{0646} \u{0645}\u{062a}\u{0635}\u{0641}\u{062d}\u{0643}. \u{0645}\u{0634}\u{0641}\u{0631}. \u{0642}\u{0627}\u{0628}\u{0644} \u{0644}\u{0644}\u{062a}\u{0643}\u{0631}\u{0627}\u{0631}. \u{0645}\u{0644}\u{0643}\u{0643}.",
        "step_system" => "\u{0646}\u{0638}\u{0627}\u{0645}\u{0643}",
        "step_desktop" => "\u{0633}\u{0637}\u{062d} \u{0627}\u{0644}\u{0645}\u{0643}\u{062a}\u{0628} \u{0648}\u{0627}\u{0644}\u{0623}\u{0645}\u{0627}\u{0646}",
        "step_apps" => "\u{062a}\u{0637}\u{0628}\u{064a}\u{0642}\u{0627}\u{062a}\u{0643}",
        "step_config" => "\u{0645}\u{0631}\u{0627}\u{062c}\u{0639}\u{0629} \u{0627}\u{0644}\u{0625}\u{0639}\u{062f}\u{0627}\u{062f}\u{0627}\u{062a}",
        "step_install" => "\u{062a}\u{062b}\u{0628}\u{064a}\u{062a}",
        "next" => "\u{0627}\u{0644}\u{062a}\u{0627}\u{0644}\u{064a}",
        "back" => "\u{0631}\u{062c}\u{0648}\u{0639}",
        "start_over" => "\u{0627}\u{0644}\u{0628}\u{062f}\u{0621} \u{0645}\u{0646} \u{062c}\u{062f}\u{064a}\u{062f}",
        "connect" => "\u{0627}\u{062a}\u{0635}\u{0627}\u{0644}",
        "password" => "\u{0643}\u{0644}\u{0645}\u{0629} \u{0627}\u{0644}\u{0645}\u{0631}\u{0648}\u{0631}",
        "apps_search" => "\u{0628}\u{062d}\u{062b} \u{0627}\u{0644}\u{062a}\u{0637}\u{0628}\u{064a}\u{0642}\u{0627}\u{062a}...",
        "install_complete" => "\u{062a}\u{0645} \u{062a}\u{062b}\u{0628}\u{064a}\u{062a} NixOS \u{0628}\u{0646}\u{062c}\u{0627}\u{062d}!",
    }
});
