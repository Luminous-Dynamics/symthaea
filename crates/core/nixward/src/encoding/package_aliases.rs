// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Package alias registry for natural language package discovery.
//!
//! Maps common software names, abbreviations, and proprietary equivalents
//! to their NixOS package attribute paths. Ported from luminous-nix Python
//! (extended_aliases_500.py + package_aliases.py), deduplicated and merged.
//!
//! Usage:
//! ```
//! use nixward::encoding::package_aliases::{resolve_alias, search_aliases, Category};
//!
//! assert_eq!(resolve_alias("chrome"), Some("google-chrome"));
//! assert_eq!(resolve_alias("vscode"), Some("vscode"));
//! assert!(search_aliases("fire").len() >= 2); // firefox, firebird, firecracker...
//! ```

use std::collections::HashMap;
use std::sync::LazyLock;

/// Resolve a user-provided name to its canonical NixOS package attribute.
///
/// Case-insensitive lookup. Returns `None` if no alias matches.
pub fn resolve_alias(name: &str) -> Option<&'static str> {
    ALIASES.get(name.to_lowercase().as_str()).copied()
}

/// Search aliases by substring match (case-insensitive).
///
/// Matches against both alias names and package attribute paths.
pub fn search_aliases(query: &str) -> Vec<(&'static str, &'static str)> {
    let q = query.to_lowercase();
    ALIASES
        .iter()
        .filter(|(alias, pkg)| alias.contains(q.as_str()) || pkg.contains(q.as_str()))
        .map(|(a, p)| (*a, *p))
        .collect()
}

/// Minimum Jaro-Winkler similarity to consider a suggestion relevant.
/// Aliases scoring below this are omitted from results.
const SUGGEST_MIN_SIMILARITY: f64 = 0.6;

/// Suggest similar aliases using edit distance (for typo correction).
///
/// Skips aliases whose length differs too much from the query (heuristic
/// early-exit) and filters out entries below `SUGGEST_MIN_SIMILARITY`.
pub fn suggest_similar(name: &str, max: usize) -> Vec<(&'static str, &'static str)> {
    let q = name.to_lowercase();
    let q_len = q.len();
    // Aliases whose length differs by more than 50% are unlikely matches
    let min_len = q_len / 2;
    let max_len = q_len.saturating_mul(2);

    let mut scored: Vec<_> = ALIASES
        .iter()
        .filter(|(alias, _)| {
            let a_len = alias.len();
            a_len >= min_len && a_len <= max_len
        })
        .map(|(alias, pkg)| {
            let dist = strsim::jaro_winkler(alias, &q);
            (dist, *alias, *pkg)
        })
        .filter(|(dist, _, _)| *dist >= SUGGEST_MIN_SIMILARITY)
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(max)
        .map(|(_, a, p)| (a, p))
        .collect()
}

/// Software categories for browsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Browsers,
    Editors,
    Terminals,
    Development,
    Languages,
    Databases,
    Cloud,
    Media,
    Graphics,
    Office,
    Communication,
    System,
    Security,
    Gaming,
    Network,
    FileManagers,
    Virtualization,
    CredentialManagers,
    Backup,
    Scientific,
}

/// Get representative packages for a category.
pub fn category_packages(cat: Category) -> &'static [&'static str] {
    match cat {
        Category::Browsers => &[
            "chrome",
            "firefox",
            "brave",
            "opera",
            "vivaldi",
            "edge",
            "tor",
            "librewolf",
        ],
        Category::Editors => &[
            "vscode", "vim", "neovim", "emacs", "sublime", "helix", "zed",
        ],
        Category::Terminals => &["alacritty", "kitty", "wezterm", "foot", "zellij"],
        Category::Development => &["git", "docker", "kubectl", "terraform", "ansible"],
        Category::Languages => &["python", "node", "rust", "go", "java", "ruby", "php"],
        Category::Databases => &["postgres", "mysql", "mongodb", "redis", "sqlite", "duckdb"],
        Category::Cloud => &[
            "kubernetes",
            "terraform",
            "ansible",
            "aws",
            "gcloud",
            "azure",
        ],
        Category::Media => &[
            "vlc",
            "mpv",
            "spotify",
            "obs",
            "ffmpeg",
            "handbrake",
            "kdenlive",
        ],
        Category::Graphics => &[
            "gimp",
            "inkscape",
            "krita",
            "blender",
            "darktable",
            "freecad",
        ],
        Category::Office => &[
            "libreoffice",
            "latex",
            "pandoc",
            "obsidian",
            "calibre",
            "zathura",
        ],
        Category::Communication => &[
            "slack", "discord", "teams", "zoom", "telegram", "signal", "element",
        ],
        Category::System => &["htop", "btop", "tmux", "zsh", "fzf", "ripgrep", "bat", "fd"],
        Category::Security => &[
            "nmap",
            "wireshark",
            "metasploit",
            "burpsuite",
            "clamav",
            "gpg",
        ],
        Category::Gaming => &[
            "steam",
            "lutris",
            "wine",
            "retroarch",
            "minecraft",
            "heroic",
        ],
        Category::Network => &["curl", "wget", "httpie", "rsync", "mosh", "filezilla"],
        Category::FileManagers => &["nautilus", "dolphin", "thunar", "ranger", "nnn", "lf", "mc"],
        Category::Virtualization => &["virtualbox", "qemu", "virt-manager", "podman", "lxc"],
        Category::CredentialManagers => &["bitwarden", "keepassxc", "pass", "gopass", "1password"],
        Category::Backup => &["syncthing", "restic", "borg", "rclone", "timeshift"],
        Category::Scientific => &["jupyter", "octave", "gnuplot", "rstudio", "maxima"],
    }
}

// ─── Master Alias Table ────────────────────────────────────────────────────
// Merged from luminous-nix extended_aliases_500.py (550 entries) and
// package_aliases.py (257 entries), deduplicated. ~600 unique mappings.

static ALIASES: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    HashMap::from([
        // ── Web Browsers ──
        ("chrome", "google-chrome"),
        ("chromium", "chromium"),
        ("firefox", "firefox"),
        ("brave", "brave"),
        ("opera", "opera"),
        ("vivaldi", "vivaldi"),
        ("edge", "microsoft-edge"),
        ("safari", "epiphany"),
        ("tor", "tor-browser-bundle-bin"),
        ("tor-browser", "tor-browser-bundle-bin"),
        ("lynx", "lynx"),
        ("w3m", "w3m"),
        ("qutebrowser", "qutebrowser"),
        ("midori", "midori"),
        ("falkon", "falkon"),
        ("pale moon", "palemoon-bin"),
        ("seamonkey", "seamonkey"),
        ("waterfox", "waterfox"),
        ("icecat", "icecat"),
        ("librewolf", "librewolf"),
        ("ungoogled-chromium", "ungoogled-chromium"),
        // ── Text Editors & IDEs ──
        ("vscode", "vscode"),
        ("code", "vscode"),
        ("visual studio code", "vscode"),
        ("vscodium", "vscodium"),
        ("vim", "vim"),
        ("neovim", "neovim"),
        ("nvim", "neovim"),
        ("emacs", "emacs"),
        ("spacemacs", "emacs"),
        ("doom emacs", "emacs"),
        ("sublime", "sublime4"),
        ("sublime text", "sublime4"),
        ("atom", "atom"),
        ("brackets", "brackets"),
        ("gedit", "gedit"),
        ("kate", "kate"),
        ("nano", "nano"),
        ("micro", "micro"),
        ("helix", "helix"),
        ("kakoune", "kakoune"),
        ("xi", "xi-editor"),
        ("amp", "amp"),
        ("ne", "ne"),
        ("joe", "joe"),
        ("jed", "jed"),
        ("leafpad", "leafpad"),
        ("mousepad", "mousepad"),
        ("pluma", "pluma"),
        ("geany", "geany"),
        ("bluefish", "bluefish"),
        ("eclipse", "eclipse"),
        ("intellij", "jetbrains.idea-community"),
        ("idea", "jetbrains.idea-community"),
        ("pycharm", "jetbrains.pycharm-community"),
        ("webstorm", "jetbrains.webstorm"),
        ("phpstorm", "jetbrains.phpstorm"),
        ("rubymine", "jetbrains.ruby-mine"),
        ("goland", "jetbrains.goland"),
        ("clion", "jetbrains.clion"),
        ("rider", "jetbrains.rider"),
        ("android studio", "android-studio"),
        ("notepad++", "notepadqq"),
        ("zed", "zed-editor"),
        // ── Terminal Emulators ──
        ("terminal", "gnome-terminal"),
        ("gnome-terminal", "gnome-terminal"),
        ("konsole", "konsole"),
        ("xterm", "xterm"),
        ("urxvt", "rxvt-unicode"),
        ("rxvt", "rxvt-unicode"),
        ("alacritty", "alacritty"),
        ("kitty", "kitty"),
        ("terminator", "terminator"),
        ("tilix", "tilix"),
        ("hyper", "hyper"),
        ("termite", "termite"),
        ("st", "st"),
        ("cool-retro-term", "cool-retro-term"),
        ("guake", "guake"),
        ("tilda", "tilda"),
        ("yakuake", "yakuake"),
        ("terminology", "terminology"),
        ("sakura", "sakura"),
        ("roxterm", "roxterm"),
        ("lxterminal", "lxterminal"),
        ("xfce4-terminal", "xfce4-terminal"),
        ("mate-terminal", "mate-terminal"),
        ("foot", "foot"),
        ("wezterm", "wezterm"),
        ("warp", "warp-terminal"),
        // ── Development Tools ──
        ("git", "git"),
        ("github", "gh"),
        ("gh", "gh"),
        ("gitlab", "gitlab"),
        ("svn", "subversion"),
        ("subversion", "subversion"),
        ("mercurial", "mercurial"),
        ("hg", "mercurial"),
        ("fossil", "fossil"),
        ("cvs", "cvs"),
        ("make", "gnumake"),
        ("cmake", "cmake"),
        ("ninja", "ninja"),
        ("meson", "meson"),
        ("autotools", "autoconf"),
        ("automake", "automake"),
        ("gcc", "gcc"),
        ("g++", "gcc"),
        ("clang", "clang"),
        ("llvm", "llvm"),
        ("rust", "rustc"),
        ("rustc", "rustc"),
        ("cargo", "cargo"),
        ("rustup", "rustup"),
        ("go", "go"),
        ("golang", "go"),
        ("python", "python3"),
        ("python3", "python3"),
        ("python2", "python2"),
        ("pip", "python3Packages.pip"),
        ("node", "nodejs"),
        ("nodejs", "nodejs"),
        ("npm", "nodejs"),
        ("yarn", "yarn"),
        ("pnpm", "pnpm"),
        ("deno", "deno"),
        ("ruby", "ruby"),
        ("rails", "rubyPackages.rails"),
        ("java", "jdk"),
        ("jdk", "jdk"),
        ("jre", "jre"),
        ("maven", "maven"),
        ("gradle", "gradle"),
        ("ant", "ant"),
        ("scala", "scala"),
        ("sbt", "sbt"),
        ("kotlin", "kotlin"),
        ("groovy", "groovy"),
        ("perl", "perl"),
        ("php", "php"),
        ("composer", "phpPackages.composer"),
        ("lua", "lua"),
        ("julia", "julia"),
        ("r", "R"),
        ("octave", "octave"),
        ("dotnet", "dotnet-sdk"),
        (".net", "dotnet-sdk"),
        ("bazel", "bazel"),
        // ── Databases ──
        ("mysql", "mysql"),
        ("mariadb", "mariadb"),
        ("postgres", "postgresql"),
        ("postgresql", "postgresql"),
        ("sqlite", "sqlite"),
        ("sqlite3", "sqlite"),
        ("mongodb", "mongodb"),
        ("mongo", "mongodb"),
        ("redis", "redis"),
        ("cassandra", "cassandra"),
        ("couchdb", "couchdb"),
        ("elasticsearch", "elasticsearch"),
        ("neo4j", "neo4j"),
        ("influxdb", "influxdb"),
        ("prometheus", "prometheus"),
        ("grafana", "grafana"),
        ("memcached", "memcached"),
        ("rabbitmq", "rabbitmq-server"),
        ("kafka", "apacheKafka"),
        ("dynamodb", "dynamodb-local"),
        ("firebird", "firebird"),
        ("orientdb", "orientdb"),
        ("arangodb", "arangodb"),
        ("rethinkdb", "rethinkdb"),
        ("cockroachdb", "cockroachdb"),
        ("timescaledb", "timescaledb"),
        ("clickhouse", "clickhouse"),
        ("questdb", "questdb"),
        ("duckdb", "duckdb"),
        ("rocksdb", "rocksdb"),
        // ── Web Servers ──
        ("nginx", "nginx"),
        ("apache", "apacheHttpd"),
        ("httpd", "apacheHttpd"),
        ("caddy", "caddy"),
        ("traefik", "traefik"),
        // ── Cloud & DevOps ──
        ("docker", "docker"),
        ("docker-compose", "docker-compose"),
        ("podman", "podman"),
        ("kubernetes", "kubernetes"),
        ("k8s", "kubernetes"),
        ("kubectl", "kubectl"),
        ("minikube", "minikube"),
        ("k3s", "k3s"),
        ("k9s", "k9s"),
        ("helm", "kubernetes-helm"),
        ("terraform", "terraform"),
        ("ansible", "ansible"),
        ("puppet", "puppet"),
        ("chef", "chef"),
        ("salt", "salt"),
        ("vagrant", "vagrant"),
        ("packer", "packer"),
        ("consul", "consul"),
        ("vault", "vault"),
        ("nomad", "nomad"),
        ("jenkins", "jenkins"),
        ("gitlab-runner", "gitlab-runner"),
        ("github-actions", "act"),
        ("circleci", "circleci-cli"),
        ("travis", "travis"),
        ("aws", "awscli2"),
        ("aws-cli", "awscli2"),
        ("gcloud", "google-cloud-sdk"),
        ("azure", "azure-cli"),
        ("az", "azure-cli"),
        ("digitalocean", "doctl"),
        ("doctl", "doctl"),
        ("linode", "linode-cli"),
        ("heroku", "heroku"),
        ("fly", "flyctl"),
        ("serverless", "serverless"),
        ("localstack", "localstack"),
        ("minio", "minio"),
        ("s3cmd", "s3cmd"),
        ("rclone", "rclone"),
        ("restic", "restic"),
        ("borg", "borgbackup"),
        // ── Media Players ──
        ("vlc", "vlc"),
        ("mpv", "mpv"),
        ("mplayer", "mplayer"),
        ("smplayer", "smplayer"),
        ("kodi", "kodi"),
        ("plex", "plex"),
        ("jellyfin", "jellyfin"),
        ("emby", "emby"),
        ("spotify", "spotify"),
        ("rhythmbox", "rhythmbox"),
        ("amarok", "amarok"),
        ("clementine", "clementine"),
        ("strawberry", "strawberry"),
        ("audacious", "audacious"),
        ("banshee", "banshee"),
        ("deadbeef", "deadbeef"),
        ("cmus", "cmus"),
        ("mpd", "mpd"),
        ("ncmpcpp", "ncmpcpp"),
        ("moc", "moc"),
        ("sox", "sox"),
        ("ffmpeg", "ffmpeg"),
        ("handbrake", "handbrake"),
        ("obs", "obs-studio"),
        ("obs-studio", "obs-studio"),
        ("kdenlive", "kdenlive"),
        ("audacity", "audacity"),
        // ── Graphics & Design ──
        ("gimp", "gimp"),
        ("photoshop", "gimp"),
        ("inkscape", "inkscape"),
        ("illustrator", "inkscape"),
        ("krita", "krita"),
        ("blender", "blender"),
        ("freecad", "freecad"),
        ("openscad", "openscad"),
        ("librecad", "librecad"),
        ("darktable", "darktable"),
        ("rawtherapee", "rawtherapee"),
        ("digikam", "digikam"),
        ("shotwell", "shotwell"),
        ("gthumb", "gthumb"),
        ("imagemagick", "imagemagick"),
        ("graphicsmagick", "graphicsmagick"),
        ("scribus", "scribus"),
        ("mypaint", "mypaint"),
        ("pinta", "pinta"),
        ("xfig", "xfig"),
        ("dia", "dia"),
        ("draw.io", "drawio"),
        ("pencil", "pencil"),
        ("gravit", "gravit-designer"),
        ("figma", "figma-linux"),
        ("aseprite", "aseprite"),
        ("godot", "godot"),
        ("unity", "unityhub"),
        ("unreal", "unrealengine"),
        ("love2d", "love"),
        // ── Office & Productivity ──
        ("libreoffice", "libreoffice"),
        ("office", "libreoffice"),
        ("word", "libreoffice"),
        ("excel", "libreoffice"),
        ("powerpoint", "libreoffice"),
        ("openoffice", "openoffice"),
        ("onlyoffice", "onlyoffice-bin"),
        ("calligra", "calligra"),
        ("abiword", "abiword"),
        ("gnumeric", "gnumeric"),
        ("latex", "texlive.combined.scheme-full"),
        ("tex", "texlive.combined.scheme-full"),
        ("pandoc", "pandoc"),
        ("markdown", "pandoc"),
        ("zotero", "zotero"),
        ("mendeley", "mendeley"),
        ("calibre", "calibre"),
        ("evince", "evince"),
        ("okular", "okular"),
        ("zathura", "zathura"),
        ("mupdf", "mupdf"),
        ("xournal", "xournalpp"),
        ("remarkable", "remarkable"),
        ("obsidian", "obsidian"),
        ("notion", "notion-app-enhanced"),
        // ── Communication ──
        ("slack", "slack"),
        ("discord", "discord"),
        ("teams", "teams"),
        ("zoom", "zoom-us"),
        ("skype", "skypeforlinux"),
        ("telegram", "telegram-desktop"),
        ("signal", "signal-desktop"),
        ("whatsapp", "whatsapp-for-linux"),
        ("element", "element-desktop"),
        ("riot", "element-desktop"),
        ("matrix", "element-desktop"),
        ("irc", "hexchat"),
        ("hexchat", "hexchat"),
        ("irssi", "irssi"),
        ("weechat", "weechat"),
        ("pidgin", "pidgin"),
        ("empathy", "empathy"),
        ("thunderbird", "thunderbird"),
        ("evolution", "evolution"),
        ("geary", "geary"),
        ("kmail", "kmail"),
        ("mutt", "mutt"),
        ("neomutt", "neomutt"),
        ("aerc", "aerc"),
        ("alpine", "alpine"),
        ("mailspring", "mailspring"),
        ("birdtray", "birdtray"),
        ("rambox", "rambox"),
        ("franz", "franz"),
        ("ferdi", "ferdi"),
        // ── System Tools ──
        ("htop", "htop"),
        ("btop", "btop"),
        ("gotop", "gotop"),
        ("glances", "glances"),
        ("iotop", "iotop"),
        ("iftop", "iftop"),
        ("nethogs", "nethogs"),
        ("ncdu", "ncdu"),
        ("dust", "dust"),
        ("duf", "duf"),
        ("tree", "tree"),
        ("exa", "eza"),
        ("eza", "eza"),
        ("lsd", "lsd"),
        ("bat", "bat"),
        ("fd", "fd"),
        ("find", "findutils"),
        ("fzf", "fzf"),
        ("ripgrep", "ripgrep"),
        ("rg", "ripgrep"),
        ("ag", "silver-searcher"),
        ("ack", "ack"),
        ("tmux", "tmux"),
        ("screen", "screen"),
        ("zsh", "zsh"),
        ("fish", "fish"),
        ("bash", "bash"),
        ("oh-my-zsh", "oh-my-zsh"),
        ("starship", "starship"),
        ("powerline", "powerline-go"),
        ("neofetch", "neofetch"),
        ("fastfetch", "fastfetch"),
        ("screenfetch", "screenfetch"),
        ("pfetch", "pfetch"),
        ("ranger", "ranger"),
        ("nnn", "nnn"),
        ("mc", "mc"),
        ("midnight commander", "mc"),
        ("midnight-commander", "mc"),
        ("vifm", "vifm"),
        ("lf", "lf"),
        ("broot", "broot"),
        ("jq", "jq"),
        ("yq", "yq"),
        ("sed", "gnused"),
        ("awk", "gawk"),
        ("grep", "gnugrep"),
        ("zip", "zip"),
        ("unzip", "unzip"),
        ("tar", "gnutar"),
        ("7zip", "p7zip"),
        ("lm-sensors", "lm_sensors"),
        ("sensors", "lm_sensors"),
        // ── Security Tools ──
        ("nmap", "nmap"),
        ("wireshark", "wireshark"),
        ("tcpdump", "tcpdump"),
        ("metasploit", "metasploit"),
        ("burpsuite", "burpsuite"),
        ("burp", "burpsuite"),
        ("john", "john"),
        ("hashcat", "hashcat"),
        ("aircrack", "aircrack-ng"),
        ("aircrack-ng", "aircrack-ng"),
        ("hydra", "hydra"),
        ("sqlmap", "sqlmap"),
        ("nikto", "nikto"),
        ("dirb", "dirb"),
        ("gobuster", "gobuster"),
        ("ffuf", "ffuf"),
        ("masscan", "masscan"),
        ("zap", "zap"),
        ("openvas", "openvas-scanner"),
        ("lynis", "lynis"),
        ("chkrootkit", "chkrootkit"),
        ("rkhunter", "rkhunter"),
        ("clamav", "clamav"),
        ("fail2ban", "fail2ban"),
        ("ufw", "ufw"),
        ("iptables", "iptables"),
        ("gpg", "gnupg"),
        ("openssl", "openssl"),
        ("pass", "pass"),
        ("keepass", "keepassxc"),
        ("keepassxc", "keepassxc"),
        ("bitwarden", "bitwarden"),
        ("1password", "_1password-gui"),
        ("gopass", "gopass"),
        ("lastpass", "lastpass-cli"),
        ("dashlane", "dashlane"),
        ("enpass", "enpass"),
        ("buttercup", "buttercup-desktop"),
        // ── Gaming ──
        ("steam", "steam"),
        ("lutris", "lutris"),
        ("wine", "wine"),
        ("proton", "proton-caller"),
        ("playonlinux", "playonlinux"),
        ("retroarch", "retroarch"),
        ("dolphin-emu", "dolphin-emu"),
        ("pcsx2", "pcsx2"),
        ("rpcs3", "rpcs3"),
        ("yuzu", "yuzu-mainline"),
        ("cemu", "cemu"),
        ("mame", "mame"),
        ("dosbox", "dosbox"),
        ("scummvm", "scummvm"),
        ("minecraft", "minecraft"),
        ("multimc", "multimc"),
        ("heroic", "heroic"),
        ("bottles", "bottles"),
        ("gamehub", "gamehub"),
        ("gamemode", "gamemode"),
        // ── Network Tools ──
        ("curl", "curl"),
        ("wget", "wget"),
        ("aria2", "aria2"),
        ("httpie", "httpie"),
        ("postman", "postman"),
        ("insomnia", "insomnia"),
        ("netcat", "netcat"),
        ("nc", "netcat"),
        ("socat", "socat"),
        ("telnet", "telnet"),
        ("ssh", "openssh"),
        ("mosh", "mosh"),
        ("rsync", "rsync"),
        ("scp", "openssh"),
        ("sftp", "openssh"),
        ("filezilla", "filezilla"),
        ("transmission", "transmission"),
        ("qbittorrent", "qbittorrent"),
        ("deluge", "deluge"),
        ("rtorrent", "rtorrent"),
        // ── File Managers ──
        ("nautilus", "gnome.nautilus"),
        ("dolphin", "dolphin"),
        ("thunar", "xfce.thunar"),
        ("pcmanfm", "pcmanfm"),
        ("nemo", "cinnamon.nemo"),
        ("caja", "mate.caja"),
        ("spacefm", "spacefm"),
        ("doublecmd", "doublecmd"),
        ("krusader", "krusader"),
        ("xfe", "xfe"),
        ("rox", "rox-filer"),
        // ── Scientific & Math ──
        ("matlab", "octave"),
        ("mathematica", "sage"),
        ("scipy", "python3Packages.scipy"),
        ("numpy", "python3Packages.numpy"),
        ("pandas", "python3Packages.pandas"),
        ("jupyter", "jupyter"),
        ("jupyterlab", "jupyterlab"),
        ("rstudio", "rstudio"),
        ("spyder", "spyder"),
        ("gnuplot", "gnuplot"),
        ("maxima", "maxima"),
        ("scilab", "scilab"),
        ("geogebra", "geogebra"),
        ("stellarium", "stellarium"),
        ("celestia", "celestia"),
        // ── Backup & Sync ──
        ("dropbox", "dropbox"),
        ("nextcloud", "nextcloud-client"),
        ("owncloud", "owncloud-client"),
        ("syncthing", "syncthing"),
        ("seafile", "seafile-client"),
        ("duplicati", "duplicati"),
        ("duplicity", "duplicity"),
        ("deja-dup", "deja-dup"),
        ("backintime", "backintime"),
        ("timeshift", "timeshift"),
        ("rsnapshot", "rsnapshot"),
        ("rdiff-backup", "rdiff-backup"),
        ("bacula", "bacula"),
        ("amanda", "amanda"),
        ("tarsnap", "tarsnap"),
        // ── Virtualization ──
        ("virtualbox", "virtualbox"),
        ("vbox", "virtualbox"),
        ("vmware", "vmware-workstation"),
        ("qemu", "qemu"),
        ("kvm", "qemu_kvm"),
        ("virt-manager", "virt-manager"),
        ("gnome-boxes", "gnome.gnome-boxes"),
        ("xen", "xen"),
        ("lxc", "lxc"),
        ("lxd", "lxd"),
        ("multipass", "multipass"),
        ("lima", "lima"),
        ("firecracker", "firecracker"),
        // ── Shells & Terminal ──
        ("zellij", "zellij"),
        ("asciinema", "asciinema"),
        ("ttyd", "ttyd"),
        ("tmate", "tmate"),
        ("byobu", "byobu"),
        ("dvtm", "dvtm"),
        ("abduco", "abduco"),
        ("dtach", "dtach"),
        // ── Package Managers & Build ──
        ("flatpak", "flatpak"),
        ("snap", "snapd"),
        ("appimage", "appimage-run"),
        ("nix", "nix"),
        ("home-manager", "home-manager"),
        ("poetry", "poetry"),
        ("pipenv", "pipenv"),
        ("conda", "conda"),
        // ── Monitoring ──
        ("loki", "grafana-loki"),
        ("datadog", "datadog-agent"),
        ("nagios", "nagios"),
        ("zabbix", "zabbix"),
        // ── AI/ML ──
        ("ollama", "ollama"),
        ("stable-diffusion", "stable-diffusion"),
        ("tensorflow", "python3Packages.tensorflow"),
        ("pytorch", "python3Packages.pytorch"),
        ("scikit-learn", "python3Packages.scikit-learn"),
        // ── Misc System ──
        ("gparted", "gparted"),
        ("gnome-disks", "gnome.gnome-disk-utility"),
        ("baobab", "gnome.baobab"),
        ("filelight", "filelight"),
        ("cat", "coreutils"),
        ("ls", "coreutils"),
        ("top", "htop"),
    ])
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_exact() {
        assert_eq!(resolve_alias("firefox"), Some("firefox"));
        assert_eq!(resolve_alias("chrome"), Some("google-chrome"));
        assert_eq!(resolve_alias("vscode"), Some("vscode"));
    }

    #[test]
    fn test_resolve_case_insensitive() {
        assert_eq!(resolve_alias("Firefox"), Some("firefox"));
        assert_eq!(resolve_alias("CHROME"), Some("google-chrome"));
    }

    #[test]
    fn test_resolve_unknown() {
        assert_eq!(resolve_alias("nonexistent-package-xyz"), None);
    }

    #[test]
    fn test_search() {
        let results = search_aliases("fire");
        assert!(results.iter().any(|(a, _)| *a == "firefox"));
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_suggest_typo() {
        let suggestions = suggest_similar("fierfox", 3);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].0, "firefox");
    }

    #[test]
    fn test_category_packages() {
        let browsers = category_packages(Category::Browsers);
        assert!(browsers.contains(&"firefox"));
        assert!(browsers.contains(&"chrome"));
    }

    #[test]
    fn test_alias_count() {
        assert!(
            ALIASES.len() >= 500,
            "Expected 500+ aliases, got {}",
            ALIASES.len()
        );
    }

    #[test]
    fn test_jetbrains_mapping() {
        assert_eq!(resolve_alias("intellij"), Some("jetbrains.idea-community"));
        assert_eq!(
            resolve_alias("pycharm"),
            Some("jetbrains.pycharm-community")
        );
        assert_eq!(resolve_alias("goland"), Some("jetbrains.goland"));
    }

    #[test]
    fn test_proprietary_alternatives() {
        assert_eq!(resolve_alias("photoshop"), Some("gimp"));
        assert_eq!(resolve_alias("illustrator"), Some("inkscape"));
        assert_eq!(resolve_alias("word"), Some("libreoffice"));
        assert_eq!(resolve_alias("matlab"), Some("octave"));
    }

    #[test]
    fn test_suggest_similar_filters_low_similarity() {
        // "zzzzzzzzz" matches nothing well — should get empty or very few results
        let results = suggest_similar("zzzzzzzzz", 10);
        // All returned results should have similarity >= SUGGEST_MIN_SIMILARITY
        // (we can't directly check score, but absurdly different strings should be filtered)
        assert!(
            results.len() <= 3,
            "Garbage input should produce few suggestions, got {}",
            results.len()
        );
    }

    #[test]
    fn test_suggest_similar_length_filter() {
        // Very short query — should skip very long aliases
        let results = suggest_similar("ab", 5);
        // Results should exist (some short aliases match) but not include
        // aliases like "midnight commander" which is way longer
        for (alias, _) in &results {
            assert!(
                alias.len() <= 4,
                "Short query 'ab' should not match long alias '{alias}'"
            );
        }
    }

    #[test]
    fn test_suggest_similar_still_finds_typos() {
        // "fierfox" is a classic typo for "firefox" — should still be found
        let results = suggest_similar("fierfox", 3);
        assert!(!results.is_empty(), "Should find suggestions for typo");
        assert_eq!(results[0].0, "firefox", "Top suggestion should be firefox");
    }
}
