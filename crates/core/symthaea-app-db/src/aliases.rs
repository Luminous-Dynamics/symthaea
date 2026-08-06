// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Package alias registry for nixpkgs attribute path resolution.
//!
//! Maps common application names, abbreviations, and proprietary equivalents
//! to their canonical nixpkgs attribute paths. Ported from
//! `nixward::encoding::package_aliases` (~600 entries), trimmed to the
//! most useful subset for the Sovereign Inoculation installer.
//!
//! WASM-compatible: no LazyLock, no HashMap, no strsim — pure static data
//! with linear search.

/// Sorted static alias table: (common_name, nixpkgs_attr).
///
/// Kept sorted by alias for binary search in `lookup_alias`.
static ALIASES: &[(&str, &str)] = &[
    // ── AI / ML ──
    ("ollama", "ollama"),
    ("pytorch", "python3Packages.pytorch"),
    ("scikit-learn", "python3Packages.scikit-learn"),
    ("stable-diffusion", "stable-diffusion"),
    ("tensorflow", "python3Packages.tensorflow"),
    // ── Backup & Sync ──
    ("backintime", "backintime"),
    ("borg", "borgbackup"),
    ("borgbackup", "borgbackup"),
    ("deja-dup", "deja-dup"),
    ("dropbox", "dropbox"),
    ("duplicati", "duplicati"),
    ("duplicity", "duplicity"),
    ("nextcloud", "nextcloud-client"),
    ("rclone", "rclone"),
    ("rdiff-backup", "rdiff-backup"),
    ("restic", "restic"),
    ("rsnapshot", "rsnapshot"),
    ("syncthing", "syncthing"),
    ("tarsnap", "tarsnap"),
    ("timeshift", "timeshift"),
    // ── Cloud & DevOps ──
    ("ansible", "ansible"),
    ("aws", "awscli2"),
    ("aws-cli", "awscli2"),
    ("az", "azure-cli"),
    ("azure", "azure-cli"),
    ("chef", "chef"),
    ("consul", "consul"),
    ("digitalocean", "doctl"),
    ("doctl", "doctl"),
    ("docker", "docker"),
    ("docker-compose", "docker-compose"),
    ("fly", "flyctl"),
    ("gcloud", "google-cloud-sdk"),
    ("github-actions", "act"),
    ("helm", "kubernetes-helm"),
    ("heroku", "heroku"),
    ("jenkins", "jenkins"),
    ("k3s", "k3s"),
    ("k8s", "kubernetes"),
    ("k9s", "k9s"),
    ("kubectl", "kubectl"),
    ("kubernetes", "kubernetes"),
    ("localstack", "localstack"),
    ("minikube", "minikube"),
    ("minio", "minio"),
    ("nomad", "nomad"),
    ("packer", "packer"),
    ("podman", "podman"),
    ("puppet", "puppet"),
    ("s3cmd", "s3cmd"),
    ("salt", "salt"),
    ("serverless", "serverless"),
    ("terraform", "terraform"),
    ("vagrant", "vagrant"),
    ("vault", "vault"),
    // ── Communication ──
    ("aerc", "aerc"),
    ("alpine", "alpine"),
    ("discord", "discord"),
    ("element", "element-desktop"),
    ("ferdi", "ferdi"),
    ("franz", "franz"),
    ("hexchat", "hexchat"),
    ("irc", "hexchat"),
    ("irssi", "irssi"),
    ("kmail", "kmail"),
    ("mailspring", "mailspring"),
    ("matrix", "element-desktop"),
    ("mutt", "mutt"),
    ("neomutt", "neomutt"),
    ("pidgin", "pidgin"),
    ("rambox", "rambox"),
    ("riot", "element-desktop"),
    ("signal", "signal-desktop"),
    ("slack", "slack"),
    ("skype", "skypeforlinux"),
    ("teams", "teams"),
    ("telegram", "telegram-desktop"),
    ("thunderbird", "thunderbird"),
    ("weechat", "weechat"),
    ("whatsapp", "whatsapp-for-linux"),
    ("zoom", "zoom-us"),
    // ── Databases ──
    ("cassandra", "cassandra"),
    ("clickhouse", "clickhouse"),
    ("cockroachdb", "cockroachdb"),
    ("couchdb", "couchdb"),
    ("duckdb", "duckdb"),
    ("elasticsearch", "elasticsearch"),
    ("influxdb", "influxdb"),
    ("mariadb", "mariadb"),
    ("memcached", "memcached"),
    ("mongo", "mongodb"),
    ("mongodb", "mongodb"),
    ("mysql", "mysql"),
    ("neo4j", "neo4j"),
    ("postgres", "postgresql"),
    ("postgresql", "postgresql"),
    ("prometheus", "prometheus"),
    ("questdb", "questdb"),
    ("rabbitmq", "rabbitmq-server"),
    ("redis", "redis"),
    ("rethinkdb", "rethinkdb"),
    ("rocksdb", "rocksdb"),
    ("sqlite", "sqlite"),
    ("sqlite3", "sqlite"),
    // ── Development Tools ──
    (".net", "dotnet-sdk"),
    ("ant", "ant"),
    ("autotools", "autoconf"),
    ("automake", "automake"),
    ("bazel", "bazel"),
    ("clang", "clang"),
    ("cmake", "cmake"),
    ("cvs", "cvs"),
    ("dotnet", "dotnet-sdk"),
    ("fossil", "fossil"),
    ("g++", "gcc"),
    ("gcc", "gcc"),
    ("gh", "gh"),
    ("git", "git"),
    ("github", "gh"),
    ("gradle", "gradle"),
    ("hg", "mercurial"),
    ("llvm", "llvm"),
    ("make", "gnumake"),
    ("maven", "maven"),
    ("mercurial", "mercurial"),
    ("meson", "meson"),
    ("ninja", "ninja"),
    ("sbt", "sbt"),
    ("subversion", "subversion"),
    ("svn", "subversion"),
    // ── Editors & IDEs ──
    ("amp", "amp"),
    ("android studio", "android-studio"),
    ("atom", "atom"),
    ("bluefish", "bluefish"),
    ("brackets", "brackets"),
    ("clion", "jetbrains.clion"),
    ("code", "vscode"),
    ("doom emacs", "emacs"),
    ("eclipse", "eclipse"),
    ("emacs", "emacs"),
    ("geany", "geany"),
    ("gedit", "gedit"),
    ("goland", "jetbrains.goland"),
    ("helix", "helix"),
    ("idea", "jetbrains.idea-community"),
    ("intellij", "jetbrains.idea-community"),
    ("jed", "jed"),
    ("joe", "joe"),
    ("kakoune", "kakoune"),
    ("kate", "kate"),
    ("leafpad", "leafpad"),
    ("micro", "micro"),
    ("mousepad", "mousepad"),
    ("nano", "nano"),
    ("ne", "ne"),
    ("neovim", "neovim"),
    ("notepad++", "notepadqq"),
    ("nvim", "neovim"),
    ("phpstorm", "jetbrains.phpstorm"),
    ("pluma", "pluma"),
    ("pycharm", "jetbrains.pycharm-community"),
    ("rider", "jetbrains.rider"),
    ("rubymine", "jetbrains.ruby-mine"),
    ("spacemacs", "emacs"),
    ("sublime", "sublime4"),
    ("sublime text", "sublime4"),
    ("vim", "vim"),
    ("visual studio code", "vscode"),
    ("vscode", "vscode"),
    ("vscodium", "vscodium"),
    ("webstorm", "jetbrains.webstorm"),
    ("xi", "xi-editor"),
    ("zed", "zed-editor"),
    // ── File Managers ──
    ("caja", "mate.caja"),
    ("dolphin", "dolphin"),
    ("doublecmd", "doublecmd"),
    ("krusader", "krusader"),
    ("nautilus", "gnome.nautilus"),
    ("nemo", "cinnamon.nemo"),
    ("pcmanfm", "pcmanfm"),
    ("spacefm", "spacefm"),
    ("thunar", "xfce.thunar"),
    // ── Fonts ──
    ("fira-code", "fira-code"),
    ("jetbrains-mono", "jetbrains-mono"),
    ("nerd-fonts", "nerdfonts"),
    ("nerdfonts", "nerdfonts"),
    // ── Gaming ──
    ("bottles", "bottles"),
    ("cemu", "cemu"),
    ("dolphin-emu", "dolphin-emu"),
    ("dosbox", "dosbox"),
    ("gamemode", "gamemode"),
    ("heroic", "heroic"),
    ("lutris", "lutris"),
    ("mame", "mame"),
    ("minecraft", "minecraft"),
    ("multimc", "multimc"),
    ("proton", "proton-caller"),
    ("retroarch", "retroarch"),
    ("scummvm", "scummvm"),
    ("steam", "steam"),
    ("wine", "wine"),
    // ── Graphics & Design ──
    ("aseprite", "aseprite"),
    ("blender", "blender"),
    ("darktable", "darktable"),
    ("dia", "dia"),
    ("digikam", "digikam"),
    ("draw.io", "drawio"),
    ("figma", "figma-linux"),
    ("freecad", "freecad"),
    ("gimp", "gimp"),
    ("godot", "godot"),
    ("graphicsmagick", "graphicsmagick"),
    ("illustrator", "inkscape"),
    ("imagemagick", "imagemagick"),
    ("inkscape", "inkscape"),
    ("krita", "krita"),
    ("librecad", "librecad"),
    ("mypaint", "mypaint"),
    ("openscad", "openscad"),
    ("photoshop", "gimp"),
    ("pinta", "pinta"),
    ("rawtherapee", "rawtherapee"),
    ("scribus", "scribus"),
    // ── Languages ──
    ("cargo", "cargo"),
    ("composer", "phpPackages.composer"),
    ("conda", "conda"),
    ("deno", "deno"),
    ("go", "go"),
    ("golang", "go"),
    ("groovy", "groovy"),
    ("java", "jdk"),
    ("jdk", "jdk"),
    ("jre", "jre"),
    ("julia", "julia"),
    ("kotlin", "kotlin"),
    ("lua", "lua"),
    ("node", "nodejs"),
    ("nodejs", "nodejs"),
    ("npm", "nodejs"),
    ("perl", "perl"),
    ("php", "php"),
    ("pip", "python3Packages.pip"),
    ("pipenv", "pipenv"),
    ("pnpm", "pnpm"),
    ("poetry", "poetry"),
    ("python", "python3"),
    ("python2", "python2"),
    ("python3", "python3"),
    ("r", "R"),
    ("ruby", "ruby"),
    ("rust", "rustc"),
    ("rustc", "rustc"),
    ("rustup", "rustup"),
    ("scala", "scala"),
    ("yarn", "yarn"),
    // ── Media Players & Tools ──
    ("audacious", "audacious"),
    ("audacity", "audacity"),
    ("clementine", "clementine"),
    ("cmus", "cmus"),
    ("deadbeef", "deadbeef"),
    ("ffmpeg", "ffmpeg"),
    ("handbrake", "handbrake"),
    ("jellyfin", "jellyfin"),
    ("kdenlive", "kdenlive"),
    ("kodi", "kodi"),
    ("moc", "moc"),
    ("mpd", "mpd"),
    ("mplayer", "mplayer"),
    ("mpv", "mpv"),
    ("ncmpcpp", "ncmpcpp"),
    ("obs", "obs-studio"),
    ("obs-studio", "obs-studio"),
    ("plex", "plex"),
    ("rhythmbox", "rhythmbox"),
    ("smplayer", "smplayer"),
    ("sox", "sox"),
    ("spotify", "spotify"),
    ("strawberry", "strawberry"),
    ("vlc", "vlc"),
    ("yt-dlp", "yt-dlp"),
    // ── Monitoring ──
    ("datadog", "datadog-agent"),
    ("grafana", "grafana"),
    ("loki", "grafana-loki"),
    ("nagios", "nagios"),
    ("zabbix", "zabbix"),
    // ── Network Tools ──
    ("aria2", "aria2"),
    ("curl", "curl"),
    ("filezilla", "filezilla"),
    ("httpie", "httpie"),
    ("insomnia", "insomnia"),
    ("mosh", "mosh"),
    ("nc", "netcat"),
    ("netcat", "netcat"),
    ("nmap", "nmap"),
    ("postman", "postman"),
    ("rsync", "rsync"),
    ("socat", "socat"),
    ("ssh", "openssh"),
    ("tailscale", "tailscale"),
    ("tcpdump", "tcpdump"),
    ("telnet", "telnet"),
    ("transmission", "transmission"),
    ("wget", "wget"),
    ("wireshark", "wireshark"),
    // ── NixOS / System ──
    ("appimage", "appimage-run"),
    ("flatpak", "flatpak"),
    ("home-manager", "home-manager"),
    ("nh", "nh"),
    ("nix", "nix"),
    ("nix-direnv", "nix-direnv"),
    ("nix-output-monitor", "nix-output-monitor"),
    ("nom", "nix-output-monitor"),
    ("snap", "snapd"),
    // ── Office & Productivity ──
    ("abiword", "abiword"),
    ("calibre", "calibre"),
    ("calligra", "calligra"),
    ("evince", "evince"),
    ("excel", "libreoffice"),
    ("gnumeric", "gnumeric"),
    ("latex", "texlive.combined.scheme-full"),
    ("libreoffice", "libreoffice"),
    ("mendeley", "mendeley"),
    ("mupdf", "mupdf"),
    ("notion", "notion-app-enhanced"),
    ("obsidian", "obsidian"),
    ("office", "libreoffice"),
    ("okular", "okular"),
    ("onlyoffice", "onlyoffice-bin"),
    ("pandoc", "pandoc"),
    ("powerpoint", "libreoffice"),
    ("tex", "texlive.combined.scheme-full"),
    ("word", "libreoffice"),
    ("xournal", "xournalpp"),
    ("zathura", "zathura"),
    ("zotero", "zotero"),
    // ── Scientific & Math ──
    ("geogebra", "geogebra"),
    ("gnuplot", "gnuplot"),
    ("jupyter", "jupyter"),
    ("jupyterlab", "jupyterlab"),
    ("maxima", "maxima"),
    ("numpy", "python3Packages.numpy"),
    ("octave", "octave"),
    ("pandas", "python3Packages.pandas"),
    ("rstudio", "rstudio"),
    ("scilab", "scilab"),
    ("scipy", "python3Packages.scipy"),
    ("spyder", "spyder"),
    ("stellarium", "stellarium"),
    // ── Security ──
    ("1password", "_1password-gui"),
    ("aircrack", "aircrack-ng"),
    ("aircrack-ng", "aircrack-ng"),
    ("bitwarden", "bitwarden"),
    ("burp", "burpsuite"),
    ("burpsuite", "burpsuite"),
    ("clamav", "clamav"),
    ("fail2ban", "fail2ban"),
    ("gopass", "gopass"),
    ("gpg", "gnupg"),
    ("hashcat", "hashcat"),
    ("hydra", "hydra"),
    ("iptables", "iptables"),
    ("john", "john"),
    ("keepass", "keepassxc"),
    ("keepassxc", "keepassxc"),
    ("lynis", "lynis"),
    ("metasploit", "metasploit"),
    ("nikto", "nikto"),
    ("openssl", "openssl"),
    ("openvas", "openvas-scanner"),
    ("pass", "pass"),
    ("rkhunter", "rkhunter"),
    ("sqlmap", "sqlmap"),
    ("ufw", "ufw"),
    // ── Shells & Terminal Multiplexers ──
    ("asciinema", "asciinema"),
    ("bash", "bash"),
    ("byobu", "byobu"),
    ("fish", "fish"),
    ("oh-my-zsh", "oh-my-zsh"),
    ("powerline", "powerline-go"),
    ("screen", "screen"),
    ("starship", "starship"),
    ("tmate", "tmate"),
    ("tmux", "tmux"),
    ("zellij", "zellij"),
    ("zsh", "zsh"),
    // ── System Tools ──
    ("7zip", "p7zip"),
    ("ack", "ack"),
    ("ag", "silver-searcher"),
    ("bat", "bat"),
    ("broot", "broot"),
    ("btop", "btop"),
    ("duf", "duf"),
    ("dust", "dust"),
    ("exa", "eza"),
    ("eza", "eza"),
    ("fastfetch", "fastfetch"),
    ("fd", "fd"),
    ("fzf", "fzf"),
    ("glances", "glances"),
    ("gotop", "gotop"),
    ("gparted", "gparted"),
    ("htop", "htop"),
    ("iftop", "iftop"),
    ("iotop", "iotop"),
    ("jq", "jq"),
    ("lf", "lf"),
    ("lm-sensors", "lm_sensors"),
    ("lsd", "lsd"),
    ("mc", "mc"),
    ("ncdu", "ncdu"),
    ("neofetch", "neofetch"),
    ("nethogs", "nethogs"),
    ("nnn", "nnn"),
    ("pfetch", "pfetch"),
    ("ranger", "ranger"),
    ("ripgrep", "ripgrep"),
    ("rg", "ripgrep"),
    ("screenfetch", "screenfetch"),
    ("sensors", "lm_sensors"),
    ("tree", "tree"),
    ("vifm", "vifm"),
    ("yq", "yq"),
    ("zoxide", "zoxide"),
    // ── Torrent / P2P ──
    ("deluge", "deluge"),
    ("qbittorrent", "qbittorrent"),
    ("rtorrent", "rtorrent"),
    // ── Virtualization ──
    ("firecracker", "firecracker"),
    ("gnome-boxes", "gnome.gnome-boxes"),
    ("kvm", "qemu_kvm"),
    ("lima", "lima"),
    ("lxc", "lxc"),
    ("lxd", "lxd"),
    ("multipass", "multipass"),
    ("qemu", "qemu"),
    ("vbox", "virtualbox"),
    ("virt-manager", "virt-manager"),
    ("virtualbox", "virtualbox"),
    // ── Web Browsers ──
    ("brave", "brave"),
    ("chrome", "google-chrome"),
    ("chromium", "chromium"),
    ("edge", "microsoft-edge"),
    ("falkon", "falkon"),
    ("firefox", "firefox"),
    ("icecat", "icecat"),
    ("librewolf", "librewolf"),
    ("lynx", "lynx"),
    ("midori", "midori"),
    ("qutebrowser", "qutebrowser"),
    ("safari", "epiphany"),
    ("seamonkey", "seamonkey"),
    ("tor", "tor-browser-bundle-bin"),
    ("tor-browser", "tor-browser-bundle-bin"),
    ("ungoogled-chromium", "ungoogled-chromium"),
    ("vivaldi", "vivaldi"),
    ("w3m", "w3m"),
    ("waterfox", "waterfox"),
    // ── Web Servers ──
    ("apache", "apacheHttpd"),
    ("caddy", "caddy"),
    ("httpd", "apacheHttpd"),
    ("nginx", "nginx"),
    ("traefik", "traefik"),
    // ── Terminal Emulators ──
    ("alacritty", "alacritty"),
    ("cool-retro-term", "cool-retro-term"),
    ("foot", "foot"),
    ("gnome-terminal", "gnome-terminal"),
    ("guake", "guake"),
    ("hyper", "hyper"),
    ("kitty", "kitty"),
    ("konsole", "konsole"),
    ("lxterminal", "lxterminal"),
    ("mate-terminal", "mate-terminal"),
    ("roxterm", "roxterm"),
    ("sakura", "sakura"),
    ("st", "st"),
    ("terminal", "gnome-terminal"),
    ("terminator", "terminator"),
    ("terminology", "terminology"),
    ("tilda", "tilda"),
    ("tilix", "tilix"),
    ("urxvt", "rxvt-unicode"),
    ("warp", "warp-terminal"),
    ("wezterm", "wezterm"),
    ("xfce4-terminal", "xfce4-terminal"),
    ("xterm", "xterm"),
    ("yakuake", "yakuake"),
];

/// Look up a common app name and return its canonical nixpkgs attribute path.
///
/// Case-insensitive. Returns `None` if no alias matches.
pub fn lookup_alias(name: &str) -> Option<&'static str> {
    let lower = name.to_lowercase();
    let key = lower.as_str();
    for &(alias, pkg) in ALIASES {
        if alias == key {
            return Some(pkg);
        }
    }
    None
}

/// Return up to 5 fuzzy matches using substring and prefix matching.
///
/// No external dependencies (no Levenshtein / strsim) — WASM-safe.
pub fn suggest_similar(name: &str) -> Vec<(&'static str, &'static str)> {
    let lower = name.to_lowercase();
    let key = lower.as_str();

    // Collect candidates scored by match quality:
    // 3 = exact prefix match on alias
    // 2 = alias contains query
    // 1 = nixpkgs attr contains query
    let mut scored: Vec<(u8, &str, &str)> = Vec::new();

    for &(alias, pkg) in ALIASES {
        if alias.starts_with(key) {
            scored.push((3, alias, pkg));
        } else if alias.contains(key) {
            scored.push((2, alias, pkg));
        } else if pkg.to_lowercase().contains(key) {
            scored.push((1, alias, pkg));
        }
    }

    // Sort by score descending, then alphabetically
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    scored.truncate(5);
    scored.into_iter().map(|(_, a, p)| (a, p)).collect()
}

/// Return the full alias table for autocomplete / enumeration.
pub fn all_aliases() -> &'static [(&'static str, &'static str)] {
    ALIASES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_exact() {
        assert_eq!(lookup_alias("firefox"), Some("firefox"));
        assert_eq!(lookup_alias("chrome"), Some("google-chrome"));
        assert_eq!(lookup_alias("vscode"), Some("vscode"));
        assert_eq!(lookup_alias("htop"), Some("htop"));
        assert_eq!(lookup_alias("docker"), Some("docker"));
    }

    #[test]
    fn test_lookup_case_insensitive() {
        assert_eq!(lookup_alias("Firefox"), Some("firefox"));
        assert_eq!(lookup_alias("CHROME"), Some("google-chrome"));
        assert_eq!(lookup_alias("Docker"), Some("docker"));
    }

    #[test]
    fn test_lookup_unknown() {
        assert_eq!(lookup_alias("nonexistent-package-xyz"), None);
    }

    #[test]
    fn test_lookup_nix_tools() {
        assert_eq!(lookup_alias("home-manager"), Some("home-manager"));
        assert_eq!(lookup_alias("nix-direnv"), Some("nix-direnv"));
        assert_eq!(lookup_alias("nh"), Some("nh"));
        assert_eq!(
            lookup_alias("nix-output-monitor"),
            Some("nix-output-monitor")
        );
        assert_eq!(lookup_alias("nom"), Some("nix-output-monitor"));
    }

    #[test]
    fn test_lookup_languages() {
        assert_eq!(lookup_alias("python3"), Some("python3"));
        assert_eq!(lookup_alias("nodejs"), Some("nodejs"));
        assert_eq!(lookup_alias("rustc"), Some("rustc"));
        assert_eq!(lookup_alias("cargo"), Some("cargo"));
        assert_eq!(lookup_alias("go"), Some("go"));
        assert_eq!(lookup_alias("ruby"), Some("ruby"));
        assert_eq!(lookup_alias("php"), Some("php"));
        assert_eq!(lookup_alias("lua"), Some("lua"));
        assert_eq!(lookup_alias("perl"), Some("perl"));
    }

    #[test]
    fn test_lookup_devops() {
        assert_eq!(lookup_alias("kubectl"), Some("kubectl"));
        assert_eq!(lookup_alias("helm"), Some("kubernetes-helm"));
        assert_eq!(lookup_alias("terraform"), Some("terraform"));
        assert_eq!(lookup_alias("ansible"), Some("ansible"));
        assert_eq!(lookup_alias("vagrant"), Some("vagrant"));
        assert_eq!(lookup_alias("podman"), Some("podman"));
    }

    #[test]
    fn test_lookup_editors() {
        assert_eq!(lookup_alias("neovim"), Some("neovim"));
        assert_eq!(lookup_alias("nvim"), Some("neovim"));
        assert_eq!(lookup_alias("vim"), Some("vim"));
        assert_eq!(lookup_alias("emacs"), Some("emacs"));
        assert_eq!(lookup_alias("helix"), Some("helix"));
        assert_eq!(lookup_alias("kakoune"), Some("kakoune"));
        assert_eq!(lookup_alias("nano"), Some("nano"));
        assert_eq!(lookup_alias("micro"), Some("micro"));
    }

    #[test]
    fn test_lookup_media() {
        assert_eq!(lookup_alias("mpv"), Some("mpv"));
        assert_eq!(lookup_alias("vlc"), Some("vlc"));
        assert_eq!(lookup_alias("ffmpeg"), Some("ffmpeg"));
        assert_eq!(lookup_alias("imagemagick"), Some("imagemagick"));
        assert_eq!(lookup_alias("yt-dlp"), Some("yt-dlp"));
    }

    #[test]
    fn test_lookup_network() {
        assert_eq!(lookup_alias("curl"), Some("curl"));
        assert_eq!(lookup_alias("wget"), Some("wget"));
        assert_eq!(lookup_alias("aria2"), Some("aria2"));
        assert_eq!(lookup_alias("nmap"), Some("nmap"));
        assert_eq!(lookup_alias("wireshark"), Some("wireshark"));
        assert_eq!(lookup_alias("tailscale"), Some("tailscale"));
    }

    #[test]
    fn test_lookup_fonts() {
        assert_eq!(lookup_alias("nerd-fonts"), Some("nerdfonts"));
        assert_eq!(lookup_alias("nerdfonts"), Some("nerdfonts"));
        assert_eq!(lookup_alias("fira-code"), Some("fira-code"));
        assert_eq!(lookup_alias("jetbrains-mono"), Some("jetbrains-mono"));
    }

    #[test]
    fn test_lookup_cli_tools() {
        assert_eq!(lookup_alias("btop"), Some("btop"));
        assert_eq!(lookup_alias("ripgrep"), Some("ripgrep"));
        assert_eq!(lookup_alias("rg"), Some("ripgrep"));
        assert_eq!(lookup_alias("fd"), Some("fd"));
        assert_eq!(lookup_alias("bat"), Some("bat"));
        assert_eq!(lookup_alias("eza"), Some("eza"));
        assert_eq!(lookup_alias("zoxide"), Some("zoxide"));
        assert_eq!(lookup_alias("fzf"), Some("fzf"));
        assert_eq!(lookup_alias("starship"), Some("starship"));
        assert_eq!(lookup_alias("fish"), Some("fish"));
        assert_eq!(lookup_alias("zsh"), Some("zsh"));
        assert_eq!(lookup_alias("tmux"), Some("tmux"));
        assert_eq!(lookup_alias("zellij"), Some("zellij"));
        assert_eq!(lookup_alias("neofetch"), Some("neofetch"));
        assert_eq!(lookup_alias("fastfetch"), Some("fastfetch"));
    }

    #[test]
    fn test_suggest_prefix_match() {
        let results = suggest_similar("fire");
        assert!(!results.is_empty());
        // "firefox" and "firecracker" both start with "fire"
        assert!(results.iter().any(|(a, _)| *a == "firefox"));
        assert!(results.iter().any(|(a, _)| *a == "firecracker"));
    }

    #[test]
    fn test_suggest_substring_match() {
        let results = suggest_similar("top");
        assert!(!results.is_empty());
        // btop, htop, gotop, iotop, iftop all contain "top"
        let aliases: Vec<_> = results.iter().map(|(a, _)| *a).collect();
        assert!(aliases.contains(&"btop") || aliases.contains(&"htop"));
    }

    #[test]
    fn test_suggest_returns_max_5() {
        let results = suggest_similar("a");
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_suggest_no_match() {
        let results = suggest_similar("xyzzyplugh12345");
        assert!(results.is_empty());
    }

    #[test]
    fn test_suggest_pkg_attr_match() {
        // "gnupg" is in the pkg attr for gpg
        let results = suggest_similar("gnupg");
        assert!(results.iter().any(|(a, _)| *a == "gpg"));
    }

    #[test]
    fn test_all_aliases_nonempty() {
        assert!(all_aliases().len() >= 200);
    }

    #[test]
    fn test_alias_count() {
        let count = all_aliases().len();
        assert!(count >= 200, "Expected at least 200 aliases, got {}", count);
    }

    #[test]
    fn test_proprietary_alternatives() {
        assert_eq!(lookup_alias("photoshop"), Some("gimp"));
        assert_eq!(lookup_alias("illustrator"), Some("inkscape"));
        assert_eq!(lookup_alias("word"), Some("libreoffice"));
    }

    // ── Additional alias tests ──

    #[test]
    fn lookup_common_aliases() {
        assert_eq!(lookup_alias("firefox"), Some("firefox"));
        assert_eq!(lookup_alias("chrome"), Some("google-chrome"));
        assert_eq!(lookup_alias("code"), Some("vscode"));
        assert_eq!(lookup_alias("vim"), Some("vim"));
    }

    #[test]
    fn lookup_case_insensitive_extra() {
        assert_eq!(lookup_alias("Firefox"), Some("firefox"));
        assert_eq!(lookup_alias("CHROME"), Some("google-chrome"));
    }

    #[test]
    fn suggest_returns_results() {
        let suggestions = suggest_similar("firef");
        assert!(
            !suggestions.is_empty(),
            "Should suggest packages for 'firef'"
        );
    }

    #[test]
    fn alias_count_minimum() {
        assert!(
            all_aliases().len() >= 200,
            "Should have at least 200 aliases, got {}",
            all_aliases().len()
        );
    }
}
