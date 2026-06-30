//! HDC Semantic Package Search — find packages by meaning, not just name.
//!
//! Uses hyperdimensional computing to encode package names and descriptions
//! as vectors. Search queries are encoded the same way, and nearest neighbors
//! are found by cosine similarity.
//!
//! Example: "web browser" → firefox, chromium, brave, vivaldi
//!          "text editor" → vim, neovim, helix, emacs, nano

use std::sync::LazyLock;

const DIM: usize = 256; // Lightweight vectors for WASM (not full 16,384)

/// A lightweight binary hypervector for package search.
#[derive(Clone)]
struct HV {
    bits: Vec<u8>, // Packed bits: DIM/8 bytes
}

impl HV {
    fn random_from_seed(seed: u64) -> Self {
        // Deterministic pseudo-random HV from seed
        let mut bits = vec![0u8; DIM / 8];
        let mut state = seed;
        for byte in bits.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *byte = (state >> 33) as u8;
        }
        HV { bits }
    }

    fn zeros() -> Self {
        HV {
            bits: vec![0u8; DIM / 8],
        }
    }

    /// XOR bind (associative memory)
    #[allow(dead_code)]
    fn bind(&self, other: &HV) -> HV {
        let bits: Vec<u8> = self
            .bits
            .iter()
            .zip(&other.bits)
            .map(|(a, b)| a ^ b)
            .collect();
        HV { bits }
    }

    /// Cyclic permutation (for sequence encoding)
    fn permute(&self, shift: usize) -> HV {
        let total_bits = DIM;
        let shift = shift % total_bits;
        if shift == 0 {
            return self.clone();
        }

        let mut new_bits = vec![0u8; DIM / 8];
        for i in 0..total_bits {
            let src = (i + total_bits - shift) % total_bits;
            let src_byte = src / 8;
            let src_bit = src % 8;
            if (self.bits[src_byte] >> src_bit) & 1 == 1 {
                let dst_byte = i / 8;
                let dst_bit = i % 8;
                new_bits[dst_byte] |= 1 << dst_bit;
            }
        }
        HV { bits: new_bits }
    }

    /// Hamming similarity (0.0 to 1.0, where 1.0 = identical)
    fn similarity(&self, other: &HV) -> f32 {
        let matching: u32 = self
            .bits
            .iter()
            .zip(&other.bits)
            .map(|(a, b)| (!(a ^ b)).count_ones())
            .sum();
        matching as f32 / DIM as f32
    }
}

/// Threshold bundle: majority vote across multiple vectors
fn bundle(vectors: &[HV]) -> HV {
    if vectors.is_empty() {
        return HV::zeros();
    }
    if vectors.len() == 1 {
        return vectors[0].clone();
    }

    let mut counts = vec![0i32; DIM];
    for v in vectors {
        for i in 0..DIM {
            let byte = i / 8;
            let bit = i % 8;
            if (v.bits[byte] >> bit) & 1 == 1 {
                counts[i] += 1;
            }
        }
    }

    let threshold = vectors.len() as i32 / 2;
    let mut result = vec![0u8; DIM / 8];
    for i in 0..DIM {
        if counts[i] > threshold {
            result[i / 8] |= 1 << (i % 8);
        }
    }
    HV { bits: result }
}

/// Encode a string into an HV using character n-gram binding
fn encode_text(text: &str) -> HV {
    let text = text.to_lowercase();
    let chars: Vec<char> = text
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ')
        .collect();

    if chars.is_empty() {
        return HV::zeros();
    }

    // Character-level encoding with position binding
    let ngram_vectors: Vec<HV> = chars
        .windows(3)
        .enumerate()
        .map(|(pos, window)| {
            let trigram: String = window.iter().collect();
            let char_hv = HV::random_from_seed(hash_str(&trigram));
            char_hv.permute(pos % 7) // Position sensitivity
        })
        .collect();

    if ngram_vectors.is_empty() {
        // Fallback for very short strings
        return HV::random_from_seed(hash_str(&text));
    }

    bundle(&ngram_vectors)
}

fn hash_str(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for b in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64);
    }
    hash
}

/// A searchable package entry with pre-computed vector
struct PackageEntry {
    name: &'static str,
    nixpkg: &'static str,
    description: &'static str,
    vector: HV,
}

/// Pre-computed package database with HDC vectors
static PACKAGE_DB: LazyLock<Vec<PackageEntry>> = LazyLock::new(|| {
    let packages: Vec<(&str, &str, &str)> = vec![
        // Browsers
        (
            "firefox",
            "firefox",
            "web browser privacy open source mozilla",
        ),
        (
            "chromium",
            "chromium",
            "web browser google chrome open source",
        ),
        ("google chrome", "google-chrome", "web browser google fast"),
        ("brave", "brave", "web browser privacy crypto ad-blocking"),
        ("vivaldi", "vivaldi", "web browser customizable power user"),
        (
            "tor browser",
            "tor-browser",
            "web browser anonymous privacy tor network",
        ),
        (
            "microsoft edge",
            "microsoft-edge",
            "web browser microsoft windows",
        ),
        // Text Editors
        ("vim", "vim", "text editor terminal modal vi"),
        (
            "neovim",
            "neovim",
            "text editor terminal modal lua modern vim",
        ),
        ("emacs", "emacs", "text editor extensible lisp org-mode"),
        ("helix", "helix", "text editor terminal modal rust modern"),
        ("nano", "nano", "text editor terminal simple beginner"),
        ("micro", "micro", "text editor terminal simple modern"),
        ("kakoune", "kakoune", "text editor terminal modal selection"),
        (
            "visual studio code",
            "vscode",
            "code editor ide microsoft typescript extensions",
        ),
        (
            "vscodium",
            "vscodium",
            "code editor ide open source telemetry-free",
        ),
        ("zed", "zed", "code editor fast rust gpu collaborative"),
        // Terminals
        ("alacritty", "alacritty", "terminal emulator gpu rust fast"),
        ("kitty", "kitty", "terminal emulator gpu python images"),
        (
            "wezterm",
            "wezterm",
            "terminal emulator gpu rust lua multiplexer",
        ),
        ("foot", "foot", "terminal emulator wayland fast minimal"),
        // Shells
        ("bash", "bash", "shell scripting default posix"),
        ("zsh", "zsh", "shell interactive plugins oh-my-zsh"),
        ("fish", "fish", "shell friendly interactive auto-complete"),
        ("nushell", "nushell", "shell structured data modern rust"),
        // Development
        ("git", "git", "version control distributed source code"),
        (
            "docker",
            "docker",
            "container runtime virtualization devops",
        ),
        (
            "podman",
            "podman",
            "container runtime rootless docker compatible",
        ),
        ("nodejs", "nodejs", "javascript runtime server npm"),
        (
            "python",
            "python3",
            "programming language scripting data science",
        ),
        (
            "rust",
            "rustc",
            "programming language systems memory safety",
        ),
        (
            "go",
            "go",
            "programming language google concurrent fast compile",
        ),
        ("java", "jdk", "programming language jvm enterprise"),
        ("gcc", "gcc", "compiler c c++ fortran gnu"),
        ("clang", "clang", "compiler c c++ llvm"),
        ("cmake", "cmake", "build system generator c c++"),
        ("make", "gnumake", "build automation tool"),
        ("direnv", "direnv", "environment variable shell directory"),
        (
            "nix-direnv",
            "nix-direnv",
            "nix development environment shell integration",
        ),
        // Media
        ("vlc", "vlc", "media player video audio versatile"),
        ("mpv", "mpv", "media player video minimal command line"),
        (
            "ffmpeg",
            "ffmpeg",
            "video audio conversion encoding streaming",
        ),
        (
            "obs studio",
            "obs-studio",
            "screen recording streaming broadcast",
        ),
        ("kdenlive", "kdenlive", "video editor non-linear kde"),
        ("audacity", "audacity", "audio editor recording sound"),
        ("gimp", "gimp", "image editor photo manipulation graphics"),
        (
            "inkscape",
            "inkscape",
            "vector graphics editor svg illustration",
        ),
        ("blender", "blender", "3d modeling animation rendering"),
        ("krita", "krita", "digital painting drawing art"),
        (
            "imagemagick",
            "imagemagick",
            "image conversion processing command line",
        ),
        ("yt-dlp", "yt-dlp", "video downloader youtube command line"),
        ("spotify", "spotify", "music streaming service player"),
        // Office
        (
            "libreoffice",
            "libreoffice",
            "office suite word processor spreadsheet presentation",
        ),
        (
            "thunderbird",
            "thunderbird",
            "email client mozilla calendar",
        ),
        (
            "calibre",
            "calibre",
            "ebook manager reader converter library",
        ),
        ("zathura", "zathura", "pdf viewer minimal keyboard vim"),
        ("okular", "okular", "pdf document viewer kde"),
        ("evince", "evince", "pdf document viewer gnome"),
        // Communication
        ("discord", "discord", "chat voice gaming community"),
        ("slack", "slack", "team communication workplace messaging"),
        (
            "signal",
            "signal-desktop",
            "messaging encrypted private secure",
        ),
        (
            "telegram",
            "telegram-desktop",
            "messaging cloud fast groups",
        ),
        (
            "element",
            "element-desktop",
            "matrix chat decentralized encrypted",
        ),
        ("zoom", "zoom-us", "video conferencing meeting call"),
        (
            "teams",
            "teams-for-linux",
            "microsoft teams video chat work",
        ),
        // System Tools
        ("htop", "htop", "system monitor process viewer terminal"),
        (
            "btop",
            "btop",
            "system monitor resource usage terminal modern",
        ),
        (
            "neofetch",
            "neofetch",
            "system information display terminal",
        ),
        (
            "fastfetch",
            "fastfetch",
            "system information display fast modern",
        ),
        ("tmux", "tmux", "terminal multiplexer session manager"),
        ("zellij", "zellij", "terminal multiplexer rust modern"),
        // CLI Tools
        ("ripgrep", "ripgrep", "search grep fast recursive rust"),
        ("fd", "fd", "find files fast alternative rust"),
        ("bat", "bat", "cat alternative syntax highlighting"),
        ("eza", "eza", "ls alternative modern colors git"),
        ("fzf", "fzf", "fuzzy finder search filter"),
        ("zoxide", "zoxide", "cd alternative smart directory jump"),
        (
            "starship",
            "starship",
            "shell prompt customizable rust cross-shell",
        ),
        ("jq", "jq", "json processor command line query"),
        ("curl", "curl", "http client download transfer"),
        ("wget", "wget", "download manager http ftp"),
        ("aria2", "aria2", "download manager multi-protocol parallel"),
        // Network
        ("nmap", "nmap", "network scanner security port"),
        (
            "wireshark",
            "wireshark",
            "network protocol analyzer packet capture",
        ),
        ("tailscale", "tailscale", "vpn mesh network wireguard"),
        (
            "wireguard",
            "wireguard-tools",
            "vpn tunnel fast modern encryption",
        ),
        ("nginx", "nginx", "web server reverse proxy load balancer"),
        // Gaming
        ("steam", "steam", "gaming platform store valve linux proton"),
        ("lutris", "lutris", "gaming platform wine runner manager"),
        (
            "wine",
            "wineWowPackages.stable",
            "windows compatibility layer gaming",
        ),
        (
            "mangohud",
            "mangohud",
            "gaming overlay fps monitoring vulkan",
        ),
        ("gamemode", "gamemode", "gaming performance optimizer linux"),
        (
            "retroarch",
            "retroarch",
            "emulator retro gaming multi-system",
        ),
        (
            "minecraft",
            "prismlauncher",
            "game sandbox building adventure",
        ),
        // Virtualization
        ("qemu", "qemu", "virtual machine emulator kvm"),
        (
            "virt-manager",
            "virt-manager",
            "virtual machine manager gui libvirt",
        ),
        ("virtualbox", "virtualbox", "virtual machine oracle desktop"),
        // Security
        (
            "keepassxc",
            "keepassxc",
            "password manager encrypted offline",
        ),
        (
            "bitwarden",
            "bitwarden-desktop",
            "password manager cloud encrypted",
        ),
        // Fonts
        (
            "nerd fonts",
            "nerd-fonts.fira-code",
            "programming font icons ligatures",
        ),
        (
            "jetbrains mono",
            "nerd-fonts.jetbrains-mono",
            "programming font monospace",
        ),
        ("noto fonts", "noto-fonts", "unicode font international cjk"),
    ];

    packages
        .iter()
        .map(|(name, nixpkg, desc)| {
            // Encode name + description together
            let name_hv = encode_text(name);
            let desc_hv = encode_text(desc);
            let vector = bundle(&[name_hv, desc_hv]);

            PackageEntry {
                name,
                nixpkg,
                description: desc,
                vector,
            }
        })
        .collect()
});

/// Search result from semantic package search
pub struct SearchResult {
    pub name: &'static str,
    pub nixpkg: &'static str,
    pub description: &'static str,
    pub similarity: f32,
}

/// Semantic search: find packages by meaning.
/// Returns up to `limit` results sorted by similarity.
pub fn semantic_search(query: &str, limit: usize) -> Vec<SearchResult> {
    let query_hv = encode_text(query);

    let mut results: Vec<SearchResult> = PACKAGE_DB
        .iter()
        .map(|entry| SearchResult {
            name: entry.name,
            nixpkg: entry.nixpkg,
            description: entry.description,
            similarity: entry.vector.similarity(&query_hv),
        })
        .collect();

    // Sort by similarity descending
    results.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);

    // Filter out low-similarity results (below 0.55 = basically random)
    results.retain(|r| r.similarity > 0.55);

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_web_browser() {
        let results = semantic_search("web browser", 5);
        assert!(!results.is_empty(), "Should find browsers");
        let names: Vec<&str> = results.iter().map(|r| r.name).collect();
        // At least one browser should be in top 5
        assert!(
            names
                .iter()
                .any(|n| ["firefox", "chromium", "brave", "google chrome", "vivaldi"].contains(n)),
            "Should find a browser in results: {:?}",
            names
        );
    }

    #[test]
    fn search_text_editor() {
        let results = semantic_search("text editor", 5);
        assert!(!results.is_empty(), "Should find editors");
        let names: Vec<&str> = results.iter().map(|r| r.name).collect();
        assert!(
            names
                .iter()
                .any(|n| ["vim", "neovim", "emacs", "helix", "nano", "micro"].contains(n)),
            "Should find an editor in results: {:?}",
            names
        );
    }

    #[test]
    fn search_exact_name() {
        let results = semantic_search("firefox", 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].name, "firefox", "Exact name should rank first");
    }

    #[test]
    fn search_description_words() {
        let results = semantic_search("password manager", 5);
        assert!(!results.is_empty());
        let names: Vec<&str> = results.iter().map(|r| r.name).collect();
        assert!(
            names.iter().any(|n| ["keepassxc", "bitwarden"].contains(n)),
            "Should find password managers: {:?}",
            names
        );
    }

    #[test]
    fn search_gaming() {
        let results = semantic_search("gaming", 5);
        assert!(!results.is_empty());
        let names: Vec<&str> = results.iter().map(|r| r.name).collect();
        assert!(
            names
                .iter()
                .any(|n| ["steam", "lutris", "wine", "gamemode", "retroarch"].contains(n)),
            "Should find gaming tools: {:?}",
            names
        );
    }

    #[test]
    fn search_returns_nixpkg_names() {
        let results = semantic_search("vim", 1);
        assert!(!results.is_empty());
        assert!(!results[0].nixpkg.is_empty(), "Should have nixpkg name");
    }

    #[test]
    fn search_video_editing() {
        let results = semantic_search("video editing", 5);
        assert!(!results.is_empty());
        let names: Vec<&str> = results.iter().map(|r| r.name).collect();
        assert!(
            names
                .iter()
                .any(|n| ["kdenlive", "obs studio", "ffmpeg", "blender"].contains(n)),
            "Should find video tools: {:?}",
            names
        );
    }

    #[test]
    fn search_terminal() {
        let results = semantic_search("terminal emulator", 5);
        assert!(!results.is_empty());
        let names: Vec<&str> = results.iter().map(|r| r.name).collect();
        assert!(
            names
                .iter()
                .any(|n| ["alacritty", "kitty", "wezterm", "foot"].contains(n)),
            "Should find terminals: {:?}",
            names
        );
    }

    #[test]
    fn search_empty_query() {
        let results = semantic_search("", 5);
        // Empty query should return nothing or low-similarity results
        // Results may exist but should all be low similarity
        assert!(results.len() <= 20); // bounded
    }

    #[test]
    fn search_similarity_scores_bounded() {
        let results = semantic_search("firefox", 100);
        for r in &results {
            assert!(
                r.similarity >= 0.0 && r.similarity <= 1.0,
                "Similarity {} out of bounds for {}",
                r.similarity,
                r.name
            );
        }
    }

    #[test]
    fn search_nixpkg_names_valid() {
        let results = semantic_search("docker", 5);
        for r in &results {
            assert!(!r.nixpkg.is_empty(), "Package {} has empty nixpkg", r.name);
            assert!(
                !r.nixpkg.contains(' '),
                "Package {} nixpkg has spaces: {}",
                r.name,
                r.nixpkg
            );
        }
    }
}
