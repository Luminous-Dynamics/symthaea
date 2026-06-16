// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// app_migration.rs — Pre-install app compatibility scanner
//
// Scans existing Windows/macOS installations (mounted read-only) for installed
// applications and maps them to nixpkgs equivalents. Shows users what will and
// won't work before they commit to installing NixOS.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Match quality between a detected app and a nixpkgs package
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MatchKind {
    /// Exact same app available in nixpkgs (e.g., Firefox → firefox)
    Exact,
    /// Open-source alternative with similar functionality
    Alternative,
    /// Runs via compatibility layer (Wine, Bottles, Proton)
    Compatibility,
    /// Web-based alternative available
    WebAlternative,
    /// No known replacement
    NoMatch,
}

/// A detected application from the existing OS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedApp {
    /// Display name (e.g., "Adobe Photoshop")
    pub name: String,
    /// Source OS (windows, macos, linux)
    pub source_os: String,
    /// Where it was found (e.g., "Program Files/Adobe/Photoshop")
    pub install_path: String,
    /// Category for grouping in the UI
    pub category: String,
}

/// A nixpkgs package that can replace the detected app
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixAlternative {
    /// nixpkgs attribute name (e.g., "gimp")
    pub nix_pkg: String,
    /// Human-readable name
    pub display_name: String,
    /// How well it replaces the original
    pub match_kind: MatchKind,
    /// Brief note about limitations or differences
    pub note: String,
}

/// Full compatibility report for one detected app
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppMapping {
    pub detected: DetectedApp,
    pub alternatives: Vec<NixAlternative>,
}

/// Overall migration report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationReport {
    pub source_os: String,
    pub total_apps: usize,
    pub exact_matches: usize,
    pub has_alternatives: usize,
    pub no_match: usize,
    pub apps: Vec<AppMapping>,
}

/// Build the app mapping database.
/// Maps (lowercase app identifier) → Vec<NixAlternative>
pub fn build_mapping_db() -> HashMap<String, Vec<NixAlternative>> {
    let mut db: HashMap<String, Vec<NixAlternative>> = HashMap::new();

    // ── Browsers ──
    exact(&mut db, "firefox", "firefox", "Firefox");
    exact(&mut db, "mozilla firefox", "firefox", "Firefox");
    exact(&mut db, "google chrome", "google-chrome", "Google Chrome");
    exact(&mut db, "chrome", "google-chrome", "Google Chrome");
    exact(&mut db, "chromium", "chromium", "Chromium");
    exact(
        &mut db,
        "microsoft edge",
        "microsoft-edge",
        "Microsoft Edge",
    );
    exact(&mut db, "brave", "brave", "Brave");
    exact(&mut db, "vivaldi", "vivaldi", "Vivaldi");
    exact(&mut db, "opera", "opera", "Opera");
    alt(
        &mut db,
        "safari",
        "firefox",
        "Firefox",
        "Safari is macOS-only; Firefox is the closest cross-platform equivalent",
    );

    // ── Office / Productivity ──
    alt(
        &mut db,
        "microsoft office",
        "libreoffice",
        "LibreOffice",
        "Full office suite; good .docx/.xlsx compatibility",
    );
    alt(
        &mut db,
        "microsoft word",
        "libreoffice-still",
        "LibreOffice Writer",
        "Good .docx compatibility",
    );
    alt(
        &mut db,
        "microsoft excel",
        "libreoffice-still",
        "LibreOffice Calc",
        "Good .xlsx compatibility; some macro limitations",
    );
    alt(
        &mut db,
        "microsoft powerpoint",
        "libreoffice-still",
        "LibreOffice Impress",
        "Good .pptx compatibility",
    );
    alt(
        &mut db,
        "microsoft outlook",
        "thunderbird",
        "Thunderbird",
        "Email client; Exchange via TbSync add-on",
    );
    exact(&mut db, "thunderbird", "thunderbird", "Thunderbird");
    alt(
        &mut db,
        "microsoft onenote",
        "joplin-desktop",
        "Joplin",
        "Note-taking with sync; also: obsidian",
    );
    alt(
        &mut db,
        "notion",
        "obsidian",
        "Obsidian",
        "Local-first knowledge base; or use Notion web app",
    );
    exact(&mut db, "obsidian", "obsidian", "Obsidian");
    web_alt(
        &mut db,
        "google docs",
        "Web app works natively in any browser",
    );
    web_alt(
        &mut db,
        "google sheets",
        "Web app works natively in any browser",
    );
    web_alt(
        &mut db,
        "google slides",
        "Web app works natively in any browser",
    );

    // ── Development ──
    exact(&mut db, "visual studio code", "vscode", "VS Code");
    exact(&mut db, "vscode", "vscode", "VS Code");
    exact(&mut db, "code", "vscode", "VS Code");
    alt(
        &mut db,
        "visual studio",
        "vscode",
        "VS Code",
        "VS Code covers most use cases; .NET via dotnet-sdk",
    );
    exact(&mut db, "sublime text", "sublime4", "Sublime Text");
    exact(&mut db, "neovim", "neovim", "Neovim");
    exact(&mut db, "vim", "vim", "Vim");
    exact(&mut db, "emacs", "emacs", "Emacs");
    exact(
        &mut db,
        "jetbrains intellij idea",
        "jetbrains.idea-ultimate",
        "IntelliJ IDEA",
    );
    exact(
        &mut db,
        "intellij idea",
        "jetbrains.idea-ultimate",
        "IntelliJ IDEA",
    );
    exact(
        &mut db,
        "jetbrains pycharm",
        "jetbrains.pycharm-professional",
        "PyCharm",
    );
    exact(
        &mut db,
        "pycharm",
        "jetbrains.pycharm-professional",
        "PyCharm",
    );
    exact(
        &mut db,
        "jetbrains webstorm",
        "jetbrains.webstorm",
        "WebStorm",
    );
    exact(&mut db, "jetbrains clion", "jetbrains.clion", "CLion");
    exact(&mut db, "jetbrains rider", "jetbrains.rider", "Rider");
    exact(&mut db, "jetbrains goland", "jetbrains.goland", "GoLand");
    exact(
        &mut db,
        "jetbrains rustrover",
        "jetbrains.rust-rover",
        "RustRover",
    );
    exact(
        &mut db,
        "github desktop",
        "github-desktop",
        "GitHub Desktop",
    );
    exact(&mut db, "git", "git", "Git");
    exact(&mut db, "docker desktop", "docker", "Docker");
    exact(&mut db, "docker", "docker", "Docker");
    exact(&mut db, "postman", "postman", "Postman");
    exact(&mut db, "insomnia", "insomnia", "Insomnia");
    exact(&mut db, "alacritty", "alacritty", "Alacritty");
    exact(&mut db, "iterm2", "alacritty", "Alacritty");
    alt(
        &mut db,
        "iterm",
        "alacritty",
        "Alacritty",
        "GPU-accelerated terminal; also: kitty, wezterm",
    );
    exact(&mut db, "warp", "warp-terminal", "Warp");
    exact(&mut db, "terminal", "alacritty", "Alacritty");
    exact(&mut db, "windows terminal", "alacritty", "Alacritty");
    exact(&mut db, "hyper", "hyper", "Hyper");
    exact(&mut db, "wezterm", "wezterm", "WezTerm");
    exact(&mut db, "kitty", "kitty", "Kitty");

    // ── Creative / Design ──
    alt(
        &mut db,
        "adobe photoshop",
        "gimp",
        "GIMP",
        "Full image editor; also: krita for painting",
    );
    alt(
        &mut db,
        "photoshop",
        "gimp",
        "GIMP",
        "Full image editor; also: krita for painting",
    );
    alt(
        &mut db,
        "adobe illustrator",
        "inkscape",
        "Inkscape",
        "Vector graphics editor",
    );
    alt(
        &mut db,
        "illustrator",
        "inkscape",
        "Inkscape",
        "Vector graphics editor",
    );
    alt(
        &mut db,
        "adobe premiere pro",
        "kdenlive",
        "Kdenlive",
        "Video editor; also: shotcut, davinci-resolve",
    );
    alt(
        &mut db,
        "premiere pro",
        "kdenlive",
        "Kdenlive",
        "Video editor; also: shotcut, davinci-resolve",
    );
    alt(
        &mut db,
        "adobe after effects",
        "natron",
        "Natron",
        "Compositing; limited compared to AE",
    );
    alt(
        &mut db,
        "adobe lightroom",
        "darktable",
        "darktable",
        "RAW photo processor and manager",
    );
    alt(
        &mut db,
        "lightroom",
        "darktable",
        "darktable",
        "RAW photo processor and manager",
    );
    alt(
        &mut db,
        "adobe indesign",
        "scribus",
        "Scribus",
        "Desktop publishing",
    );
    alt(
        &mut db,
        "adobe acrobat",
        "okular",
        "Okular",
        "PDF viewer; editing via LibreOffice Draw",
    );
    alt(
        &mut db,
        "adobe xd",
        "figma-linux",
        "Figma (Linux)",
        "UI/UX design; also: penpot (open-source)",
    );
    alt(
        &mut db,
        "sketch",
        "figma-linux",
        "Figma (Linux)",
        "UI/UX design; also: penpot (open-source)",
    );
    web_alt(
        &mut db,
        "figma",
        "Figma web app works natively; also figma-linux package",
    );
    web_alt(&mut db, "canva", "Web app works natively in any browser");
    exact(&mut db, "gimp", "gimp", "GIMP");
    exact(&mut db, "inkscape", "inkscape", "Inkscape");
    exact(&mut db, "krita", "krita", "Krita");
    exact(&mut db, "blender", "blender", "Blender");
    exact(&mut db, "obs studio", "obs-studio", "OBS Studio");
    exact(&mut db, "obs", "obs-studio", "OBS Studio");
    exact(&mut db, "audacity", "audacity", "Audacity");
    exact(
        &mut db,
        "davinci resolve",
        "davinci-resolve",
        "DaVinci Resolve",
    );
    exact(&mut db, "kdenlive", "kdenlive", "Kdenlive");
    exact(&mut db, "handbrake", "handbrake", "HandBrake");

    // ── Music / Audio ──
    exact(&mut db, "spotify", "spotify", "Spotify");
    alt(
        &mut db,
        "apple music",
        "spotify",
        "Spotify",
        "Or use Apple Music web player",
    );
    alt(
        &mut db,
        "itunes",
        "rhythmbox",
        "Rhythmbox",
        "Music library manager; Spotify for streaming",
    );
    exact(&mut db, "vlc", "vlc", "VLC");
    exact(&mut db, "vlc media player", "vlc", "VLC");
    alt(
        &mut db,
        "foobar2000",
        "deadbeef",
        "DeaDBeeF",
        "Lightweight audio player",
    );
    exact(&mut db, "ardour", "ardour", "Ardour");
    alt(
        &mut db,
        "ableton live",
        "ardour",
        "Ardour",
        "DAW; also: lmms, reaper (via wine)",
    );
    alt(
        &mut db,
        "fl studio",
        "lmms",
        "LMMS",
        "Music production; limited compared to FL Studio",
    );
    alt(
        &mut db,
        "logic pro",
        "ardour",
        "Ardour",
        "DAW; Logic Pro is macOS-only",
    );
    exact(&mut db, "musescore", "musescore", "MuseScore");

    // ── Communication ──
    exact(&mut db, "discord", "discord", "Discord");
    exact(&mut db, "slack", "slack", "Slack");
    exact(&mut db, "zoom", "zoom-us", "Zoom");
    exact(&mut db, "zoom meetings", "zoom-us", "Zoom");
    exact(&mut db, "telegram", "telegram-desktop", "Telegram");
    exact(&mut db, "telegram desktop", "telegram-desktop", "Telegram");
    exact(&mut db, "signal", "signal-desktop", "Signal");
    exact(&mut db, "signal desktop", "signal-desktop", "Signal");
    exact(&mut db, "element", "element-desktop", "Element");
    exact(&mut db, "teams", "teams-for-linux", "Teams for Linux");
    exact(
        &mut db,
        "microsoft teams",
        "teams-for-linux",
        "Teams for Linux",
    );
    no_match(
        &mut db,
        "facetime",
        "No Linux equivalent; use web-based video calls",
    );
    no_match(
        &mut db,
        "imessage",
        "No Linux equivalent; use Signal or other cross-platform messengers",
    );

    // ── File Management / Utilities ──
    exact(&mut db, "7-zip", "p7zip", "7-Zip");
    exact(&mut db, "7zip", "p7zip", "7-Zip");
    alt(
        &mut db,
        "winrar",
        "p7zip",
        "7-Zip",
        "Handles RAR and most archive formats",
    );
    exact(&mut db, "filezilla", "filezilla", "FileZilla");
    alt(
        &mut db,
        "total commander",
        "krusader",
        "Krusader",
        "Twin-panel file manager; also: midnight commander",
    );
    exact(&mut db, "bitwarden", "bitwarden-desktop", "Bitwarden");
    exact(&mut db, "keepassxc", "keepassxc", "KeePassXC");
    alt(
        &mut db,
        "1password",
        "bitwarden-desktop",
        "Bitwarden",
        "Open-source password manager; 1Password has a Linux client too",
    );
    alt(
        &mut db,
        "lastpass",
        "bitwarden-desktop",
        "Bitwarden",
        "Open-source password manager",
    );
    exact(&mut db, "syncthing", "syncthing", "Syncthing");
    alt(
        &mut db,
        "dropbox",
        "syncthing",
        "Syncthing",
        "P2P file sync; or use rclone for cloud storage",
    );
    alt(
        &mut db,
        "google drive",
        "rclone",
        "rclone",
        "Mount Google Drive and 40+ cloud backends",
    );
    alt(
        &mut db,
        "onedrive",
        "rclone",
        "rclone",
        "Mount OneDrive; also: onedrive (native client)",
    );
    exact(&mut db, "qbittorrent", "qbittorrent", "qBittorrent");
    alt(
        &mut db,
        "utorrent",
        "qbittorrent",
        "qBittorrent",
        "Open-source torrent client",
    );

    // ── System / Security ──
    alt(
        &mut db,
        "norton",
        "clamav",
        "ClamAV",
        "Linux has minimal virus risk; ClamAV for scanning",
    );
    alt(
        &mut db,
        "mcafee",
        "clamav",
        "ClamAV",
        "Linux has minimal virus risk; ClamAV for scanning",
    );
    alt(
        &mut db,
        "windows defender",
        "clamav",
        "ClamAV",
        "Linux has minimal virus risk",
    );
    alt(
        &mut db,
        "malwarebytes",
        "clamav",
        "ClamAV",
        "Linux has minimal virus risk",
    );
    alt(
        &mut db,
        "ccleaner",
        "bleachbit",
        "BleachBit",
        "System cleaner; NixOS garbage collection handles most of this",
    );
    exact(&mut db, "wireshark", "wireshark", "Wireshark");
    exact(&mut db, "virtualbox", "virtualbox", "VirtualBox");
    exact(
        &mut db,
        "vmware",
        "vmware-workstation",
        "VMware Workstation",
    );

    // ── Gaming ──
    exact(&mut db, "steam", "steam", "Steam");
    compat(
        &mut db,
        "epic games launcher",
        "heroic",
        "Heroic Games Launcher",
        "Open-source Epic/GOG launcher; games via Proton",
    );
    compat(
        &mut db,
        "epic games",
        "heroic",
        "Heroic Games Launcher",
        "Open-source Epic/GOG launcher; games via Proton",
    );
    exact(&mut db, "lutris", "lutris", "Lutris");
    compat(
        &mut db,
        "battle.net",
        "lutris",
        "Lutris",
        "Run via Wine/Proton through Lutris",
    );
    exact(&mut db, "minecraft", "prismlauncher", "Prism Launcher");
    exact(&mut db, "retroarch", "retroarch", "RetroArch");

    // ── Virtualization / Containers ──
    exact(&mut db, "vagrant", "vagrant", "Vagrant");
    exact(&mut db, "podman", "podman", "Podman");
    alt(
        &mut db,
        "parallels",
        "virt-manager",
        "virt-manager",
        "KVM/QEMU frontend; also: gnome-boxes",
    );
    alt(
        &mut db,
        "vmware fusion",
        "virt-manager",
        "virt-manager",
        "KVM/QEMU frontend",
    );

    // ── macOS-specific ──
    alt(
        &mut db,
        "finder",
        "nautilus",
        "GNOME Files",
        "Default file manager; also: dolphin (KDE), thunar (XFCE)",
    );
    alt(
        &mut db,
        "preview",
        "okular",
        "Okular",
        "Document/image viewer",
    );
    alt(
        &mut db,
        "keynote",
        "libreoffice-still",
        "LibreOffice Impress",
        "Presentation software",
    );
    alt(
        &mut db,
        "numbers",
        "libreoffice-still",
        "LibreOffice Calc",
        "Spreadsheet",
    );
    alt(
        &mut db,
        "pages",
        "libreoffice-still",
        "LibreOffice Writer",
        "Word processor",
    );
    alt(
        &mut db,
        "garageband",
        "lmms",
        "LMMS",
        "Music production; also: ardour",
    );
    alt(
        &mut db,
        "final cut pro",
        "kdenlive",
        "Kdenlive",
        "Video editor; also: davinci-resolve",
    );
    alt(
        &mut db,
        "xcode",
        "vscode",
        "VS Code",
        "IDE; platform SDKs are macOS-only",
    );
    alt(
        &mut db,
        "homebrew",
        "nix",
        "Nix",
        "NixOS IS the package manager",
    );

    db
}

/// Categorize an app by its name
pub fn categorize_app(name: &str) -> &'static str {
    let lower = name.to_lowercase();
    if lower.contains("chrome")
        || lower.contains("firefox")
        || lower.contains("brave")
        || lower.contains("edge")
        || lower.contains("safari")
        || lower.contains("opera")
        || lower.contains("vivaldi")
        || lower.contains("browser")
    {
        "Browsers"
    } else if lower.contains("office")
        || lower.contains("word")
        || lower.contains("excel")
        || lower.contains("powerpoint")
        || lower.contains("outlook")
        || lower.contains("libre")
        || lower.contains("notion")
        || lower.contains("obsidian")
        || lower.contains("onenote")
    {
        "Office & Productivity"
    } else if lower.contains("visual studio")
        || lower.contains("vscode")
        || lower.contains("code")
        || lower.contains("jetbrains")
        || lower.contains("intellij")
        || lower.contains("pycharm")
        || lower.contains("docker")
        || lower.contains("git")
        || lower.contains("terminal")
        || lower.contains("postman")
        || lower.contains("xcode")
    {
        "Development"
    } else if lower.contains("photoshop")
        || lower.contains("illustrator")
        || lower.contains("gimp")
        || lower.contains("inkscape")
        || lower.contains("blender")
        || lower.contains("premiere")
        || lower.contains("after effects")
        || lower.contains("lightroom")
        || lower.contains("krita")
        || lower.contains("figma")
        || lower.contains("sketch")
        || lower.contains("obs")
        || lower.contains("kdenlive")
        || lower.contains("davinci")
    {
        "Creative & Design"
    } else if lower.contains("spotify")
        || lower.contains("music")
        || lower.contains("itunes")
        || lower.contains("audacity")
        || lower.contains("ardour")
        || lower.contains("ableton")
        || lower.contains("fl studio")
        || lower.contains("logic pro")
        || lower.contains("vlc")
    {
        "Music & Media"
    } else if lower.contains("discord")
        || lower.contains("slack")
        || lower.contains("zoom")
        || lower.contains("teams")
        || lower.contains("telegram")
        || lower.contains("signal")
        || lower.contains("facetime")
        || lower.contains("imessage")
        || lower.contains("element")
    {
        "Communication"
    } else if lower.contains("steam")
        || lower.contains("epic")
        || lower.contains("minecraft")
        || lower.contains("battle.net")
        || lower.contains("lutris")
        || lower.contains("retroarch")
    {
        "Gaming"
    } else if lower.contains("norton")
        || lower.contains("mcafee")
        || lower.contains("defender")
        || lower.contains("wireshark")
        || lower.contains("virtualbox")
        || lower.contains("vmware")
        || lower.contains("ccleaner")
        || lower.contains("7-zip")
        || lower.contains("winrar")
    {
        "System & Utilities"
    } else {
        "Other"
    }
}

/// Generate a shell script that scans for installed apps on the mounted partition.
/// Returns a script that outputs JSON to stdout.
pub fn generate_scan_script(mount_point: &str, os_type: &str) -> String {
    match os_type {
        "windows" => format!(
            r#"#!/bin/bash
# Scan Windows installation for installed programs
MOUNT="{mount}"
echo '['
FIRST=true

# Scan Program Files directories
for dir in "$MOUNT/Program Files" "$MOUNT/Program Files (x86)"; do
  if [ -d "$dir" ]; then
    for app in "$dir"/*/; do
      name=$(basename "$app")
      # Skip system/framework dirs
      case "$name" in
        "Common Files"|"Windows*"|"Microsoft*Update*"|"Reference*"|"dotnet"|"MSBuild") continue ;;
      esac
      if [ "$FIRST" = true ]; then FIRST=false; else echo ','; fi
      printf '{{"name":"%s","path":"%s","source":"windows"}}' \
        "$(echo "$name" | sed 's/"/\\"/g')" \
        "$(echo "$app" | sed 's/"/\\"/g')"
    done
  fi
done

# Scan Start Menu for .lnk files (more complete than Program Files)
START_MENU="$MOUNT/ProgramData/Microsoft/Windows/Start Menu/Programs"
if [ -d "$START_MENU" ]; then
  find "$START_MENU" -name "*.lnk" -maxdepth 2 2>/dev/null | while read lnk; do
    name=$(basename "$lnk" .lnk)
    case "$name" in
      "Uninstall*"|"*Help*"|"*README*"|"*License*") continue ;;
    esac
    if [ "$FIRST" = true ]; then FIRST=false; else echo ','; fi
    printf '{{"name":"%s","path":"%s","source":"windows-startmenu"}}' \
      "$(echo "$name" | sed 's/"/\\"/g')" \
      "$(echo "$lnk" | sed 's/"/\\"/g')"
  done
fi

echo ']'
"#,
            mount = mount_point
        ),

        "macos" => format!(
            r#"#!/bin/bash
# Scan macOS installation for installed applications
MOUNT="{mount}"
echo '['
FIRST=true

# Scan /Applications
if [ -d "$MOUNT/Applications" ]; then
  for app in "$MOUNT/Applications"/*.app; do
    [ -d "$app" ] || continue
    name=$(basename "$app" .app)
    if [ "$FIRST" = true ]; then FIRST=false; else echo ','; fi
    printf '{{"name":"%s","path":"%s","source":"macos"}}' \
      "$(echo "$name" | sed 's/"/\\"/g')" \
      "$(echo "$app" | sed 's/"/\\"/g')"
  done
fi

# Scan Homebrew (Intel)
if [ -d "$MOUNT/usr/local/Cellar" ]; then
  for pkg in "$MOUNT/usr/local/Cellar"/*/; do
    name=$(basename "$pkg")
    if [ "$FIRST" = true ]; then FIRST=false; else echo ','; fi
    printf '{{"name":"%s","path":"%s","source":"homebrew"}}' \
      "$(echo "$name" | sed 's/"/\\"/g')" \
      "$(echo "$pkg" | sed 's/"/\\"/g')"
  done
fi

# Scan Homebrew (Apple Silicon)
if [ -d "$MOUNT/opt/homebrew/Cellar" ]; then
  for pkg in "$MOUNT/opt/homebrew/Cellar"/*/; do
    name=$(basename "$pkg")
    if [ "$FIRST" = true ]; then FIRST=false; else echo ','; fi
    printf '{{"name":"%s","path":"%s","source":"homebrew"}}' \
      "$(echo "$name" | sed 's/"/\\"/g')" \
      "$(echo "$pkg" | sed 's/"/\\"/g')"
  done
fi

echo ']'
"#,
            mount = mount_point
        ),

        _ => "echo '[]'".to_string(),
    }
}

/// Match detected apps against the mapping database and produce a report
pub fn generate_report(detected_apps: &[DetectedApp]) -> MigrationReport {
    let db = build_mapping_db();
    let mut apps = Vec::new();
    let mut exact_count = 0;
    let mut alt_count = 0;
    let mut no_match_count = 0;

    for app in detected_apps {
        let key = app.name.to_lowercase();
        let alternatives = db.get(&key).cloned().unwrap_or_default();

        if alternatives.is_empty() {
            no_match_count += 1;
        } else if alternatives
            .iter()
            .any(|a| a.match_kind == MatchKind::Exact)
        {
            exact_count += 1;
        } else {
            alt_count += 1;
        }

        apps.push(AppMapping {
            detected: app.clone(),
            alternatives,
        });
    }

    let source_os = detected_apps
        .first()
        .map(|a| a.source_os.clone())
        .unwrap_or_default();

    MigrationReport {
        source_os,
        total_apps: detected_apps.len(),
        exact_matches: exact_count,
        has_alternatives: alt_count,
        no_match: no_match_count,
        apps,
    }
}

// ── Helper functions for building the database ──

fn exact(db: &mut HashMap<String, Vec<NixAlternative>>, key: &str, pkg: &str, name: &str) {
    db.entry(key.to_string()).or_default().push(NixAlternative {
        nix_pkg: pkg.to_string(),
        display_name: name.to_string(),
        match_kind: MatchKind::Exact,
        note: String::new(),
    });
}

fn alt(
    db: &mut HashMap<String, Vec<NixAlternative>>,
    key: &str,
    pkg: &str,
    name: &str,
    note: &str,
) {
    db.entry(key.to_string()).or_default().push(NixAlternative {
        nix_pkg: pkg.to_string(),
        display_name: name.to_string(),
        match_kind: MatchKind::Alternative,
        note: note.to_string(),
    });
}

fn compat(
    db: &mut HashMap<String, Vec<NixAlternative>>,
    key: &str,
    pkg: &str,
    name: &str,
    note: &str,
) {
    db.entry(key.to_string()).or_default().push(NixAlternative {
        nix_pkg: pkg.to_string(),
        display_name: name.to_string(),
        match_kind: MatchKind::Compatibility,
        note: note.to_string(),
    });
}

fn web_alt(db: &mut HashMap<String, Vec<NixAlternative>>, key: &str, note: &str) {
    db.entry(key.to_string()).or_default().push(NixAlternative {
        nix_pkg: String::new(),
        display_name: format!("{} (web)", key),
        match_kind: MatchKind::WebAlternative,
        note: note.to_string(),
    });
}

fn no_match(db: &mut HashMap<String, Vec<NixAlternative>>, key: &str, note: &str) {
    db.entry(key.to_string()).or_default().push(NixAlternative {
        nix_pkg: String::new(),
        display_name: String::new(),
        match_kind: MatchKind::NoMatch,
        note: note.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapping_db_has_common_apps() {
        let db = build_mapping_db();
        assert!(db.contains_key("firefox"));
        assert!(db.contains_key("google chrome"));
        assert!(db.contains_key("adobe photoshop"));
        assert!(db.contains_key("microsoft office"));
        assert!(db.contains_key("discord"));
        assert!(db.contains_key("steam"));
        assert!(db.contains_key("visual studio code"));
    }

    #[test]
    fn test_exact_match_detection() {
        let db = build_mapping_db();
        let firefox = &db["firefox"];
        assert_eq!(firefox[0].match_kind, MatchKind::Exact);
        assert_eq!(firefox[0].nix_pkg, "firefox");
    }

    #[test]
    fn test_alternative_detection() {
        let db = build_mapping_db();
        let photoshop = &db["adobe photoshop"];
        assert_eq!(photoshop[0].match_kind, MatchKind::Alternative);
        assert_eq!(photoshop[0].nix_pkg, "gimp");
    }

    #[test]
    fn test_no_match() {
        let db = build_mapping_db();
        let imessage = &db["imessage"];
        assert_eq!(imessage[0].match_kind, MatchKind::NoMatch);
    }

    #[test]
    fn test_generate_report() {
        let apps = vec![
            DetectedApp {
                name: "Firefox".into(),
                source_os: "windows".into(),
                install_path: "C:\\Program Files\\Mozilla Firefox".into(),
                category: "Browsers".into(),
            },
            DetectedApp {
                name: "Adobe Photoshop".into(),
                source_os: "windows".into(),
                install_path: "C:\\Program Files\\Adobe\\Photoshop".into(),
                category: "Creative & Design".into(),
            },
            DetectedApp {
                name: "SomeObscureApp".into(),
                source_os: "windows".into(),
                install_path: "C:\\Program Files\\SomeObscure".into(),
                category: "Other".into(),
            },
        ];

        let report = generate_report(&apps);
        assert_eq!(report.total_apps, 3);
        assert_eq!(report.exact_matches, 1); // Firefox
        assert_eq!(report.has_alternatives, 1); // Photoshop → GIMP
        assert_eq!(report.no_match, 1); // SomeObscureApp
    }

    #[test]
    fn test_categorize_app() {
        assert_eq!(categorize_app("Google Chrome"), "Browsers");
        assert_eq!(categorize_app("Microsoft Word"), "Office & Productivity");
        assert_eq!(categorize_app("Visual Studio Code"), "Development");
        assert_eq!(categorize_app("Adobe Photoshop"), "Creative & Design");
        assert_eq!(categorize_app("Spotify"), "Music & Media");
        assert_eq!(categorize_app("Discord"), "Communication");
        assert_eq!(categorize_app("Steam"), "Gaming");
        assert_eq!(categorize_app("RandomApp"), "Other");
    }

    #[test]
    fn test_windows_scan_script_generation() {
        let script = generate_scan_script("/mnt/windows", "windows");
        assert!(script.contains("Program Files"));
        assert!(script.contains("Start Menu"));
        assert!(script.contains("/mnt/windows"));
    }

    #[test]
    fn test_macos_scan_script_generation() {
        let script = generate_scan_script("/mnt/macos", "macos");
        assert!(script.contains("/Applications"));
        assert!(script.contains("homebrew"));
        assert!(script.contains("/mnt/macos"));
    }

    #[test]
    fn test_mapping_db_size() {
        let db = build_mapping_db();
        // Should have at least 100 app mappings
        assert!(
            db.len() >= 80,
            "DB has {} entries, expected >= 80",
            db.len()
        );
    }
}
