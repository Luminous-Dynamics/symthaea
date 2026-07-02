# 🍄 Mycelix Desktop

**Native P2P Consciousness Network Application**

![Status](https://img.shields.io/badge/status-week%201-blue)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey)
![License](https://img.shields.io/badge/license-SRL%20v4.0-green)

---

## 🎯 Vision

A privacy-first, P2P desktop application for human collaboration and consciousness networking. Built on Tauri + Holochain, featuring:

- **No Central Servers**: True peer-to-peer architecture
- **Privacy First**: All data stays local or encrypted P2P
- **Memory Safe**: Rust backend, modern web frontend
- **Beautiful UX**: 3D network visualization, fluid interactions
- **AI Augmented**: Optional local AI assistants (Ollama)
- **Cross-Platform**: Linux, macOS, Windows support

---

## 🚀 Quick Start

### Prerequisites

- NixOS or Nix package manager
- 4GB RAM (8GB recommended)
- 2GB disk space for development

### Development

```bash
# Clone repository
cd /srv/luminous-dynamics/Mycelix-Core/mycelix-desktop

# Enter development environment (downloads dependencies first time)
nix develop

# You'll see:
# 🍄 Mycelix Desktop Development Environment
# Rust: 1.88.0
# Node: v20.x.x
# Cargo: 1.88.0

# Create the Tauri project (first time only)
npm create tauri-app@latest .
# Choose: SolidJS, TypeScript, pnpm

# Install frontend dependencies
pnpm install

# Run development server
cargo tauri dev

# Build for production
cargo tauri build
```

---

## 📁 Project Structure

```
mycelix-desktop/
├── flake.nix                    # Nix development environment
├── src/                         # Frontend (SolidJS + TypeScript)
│   ├── App.tsx                 # Main application
│   ├── components/             # UI components
│   ├── services/               # API clients
│   └── types/                  # TypeScript types
├── src-tauri/                  # Backend (Rust)
│   ├── src/
│   │   ├── main.rs            # Tauri entry point
│   │   ├── commands.rs        # Tauri commands (API)
│   │   ├── holochain.rs       # Holochain integration
│   │   └── ai/                # AI agent system
│   ├── Cargo.toml
│   └── tauri.conf.json        # Tauri configuration
├── docs/                       # Documentation
│   ├── ADR-001-technology-stack.md
│   └── architecture/
└── README.md                   # This file
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│  Frontend: SolidJS + Three.js          │
│  State: Solid Store                    │
│  Styling: TailwindCSS                  │
└──────────────┬──────────────────────────┘
               │ Tauri Commands (IPC)
┌──────────────▼──────────────────────────┐
│  Backend: Rust                          │
│  • Holochain Conductor                  │
│  • AI Agents (Optional)                 │
│  • Local Storage (SQLite)               │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  P2P Network: Holochain DHT             │
│  • Presence Protocol                    │
│  • Messaging                            │
│  • Connection Management                │
└─────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Frontend** | SolidJS + TypeScript | Reactive, performant, 7KB |
| **Backend** | Rust + Tauri | Memory-safe, fast, small binaries |
| **P2P** | Holochain | Proven DHT, agent-centric architecture |
| **Rendering** | WebKitGTK 4.1 | Production-ready, cross-platform |
| **Styling** | TailwindCSS | Utility-first, rapid development |
| **3D Viz** | Three.js | Industry standard, well-documented |
| **AI** | Local LLM (Ollama) | Privacy-first, runs on-device |
| **State** | SQLite | Embedded, reliable, SQL |

See [ADR-001](docs/ADR-001-technology-stack.md) for detailed rationale.

---

## 📖 Documentation

- **[Development Roadmap](../MYCELIX_DESKTOP_ROADMAP.md)** - 6-month plan
- **[Architecture Decisions](docs/)** - ADRs for key decisions
- **[Week 1 Progress](WEEK_1_PROGRESS.md)** - Current status
- **[API Reference](docs/api/)** - Backend API docs (coming soon)

---

## 🎯 Current Status

**Week 1: Foundation & Architecture** (In Progress)

- ✅ Technology stack selected (Tauri + WebKitGTK + Holochain)
- ✅ Development environment configured (flake.nix)
- ✅ Architecture documented (ADR-001)
- 🚧 Proof-of-concept Tauri app (in progress)
- ⏳ Holochain integration test (upcoming)

**Next Milestone**: Week 2 - Basic UI and P2P discovery

---

## 🤝 Contributing

We're in early development! If you want to help:

1. **Try it out**: Follow Quick Start above
2. **Report bugs**: Open GitHub issues
3. **Suggest features**: Discussions welcome
4. **Code contributions**: PRs accepted (check roadmap first)

### Development Workflow

```bash
# Make changes
vim src-tauri/src/main.rs

# Test
cargo tauri dev

# Format
cargo fmt
pnpm prettier --write .

# Lint
cargo clippy
pnpm eslint .

# Commit
git add .
git commit -m "feat: add awesome feature"
```

---

## 📊 Roadmap

### Phase 1: Core MVP (Weeks 1-9)
- [x] Week 1: Technology validation
- [ ] Week 2-3: Basic UI shell
- [ ] Week 4-5: Holochain integration
- [ ] Week 6-7: P2P messaging
- [ ] Week 8-9: Network visualization

### Phase 2: P2P Features (Weeks 10-17)
- [ ] 3D consciousness field visualization
- [ ] Connection topology optimization
- [ ] Presence & status system
- [ ] File sharing

### Phase 3: AI Agents (Weeks 18-23)
- [ ] Network optimizer (invisible)
- [ ] ConnectorBot (discovery)
- [ ] ArchivistBot (memory)
- [ ] RubberDuck (debugging)

### Phase 4: Launch (Weeks 24-29)
- [ ] UI/UX polish
- [ ] Performance optimization
- [ ] Security audit
- [ ] Documentation complete
- [ ] Beta testing
- [ ] Public release

Full roadmap: [MYCELIX_DESKTOP_ROADMAP.md](../MYCELIX_DESKTOP_ROADMAP.md)

---

## 🔒 Privacy & Security

**Core Principles:**
- **Local-First**: Data stays on your device
- **End-to-End Encrypted**: All P2P communication encrypted
- **No Telemetry**: Zero tracking or analytics
- **User Sovereignty**: You control all data
- **Optional AI**: All AI features opt-in, run locally

See [Privacy Policy](docs/PRIVACY.md) for details.

---

## 📜 License

**Sacred Reciprocity License v4.0** with dual MIT licensing

- **Free for individuals**: Personal use, education, research
- **Free for communities**: Non-profits, cooperatives
- **Pay-what-you-can for organizations**: Support sustainability
- **MIT for core libraries**: Reusable components

See [LICENSE](../LICENSE) for full terms.

---

## 🙏 Acknowledgments

Built with:
- [Tauri](https://tauri.app/) - Desktop app framework
- [Holochain](https://holochain.org/) - P2P networking
- [SolidJS](https://solidjs.com/) - Reactive UI
- [Three.js](https://threejs.org/) - 3D visualization
- [NixOS](https://nixos.org/) - Reproducible builds

Inspired by consciousness-first computing principles.

---

## 💬 Contact

- **GitHub**: [Luminous-Dynamics/mycelix-desktop](https://github.com/Luminous-Dynamics/mycelix-desktop)
- **Email**: tristan.stoltz@evolvingresonantcocreationism.com
- **Website**: [mycelix.net](https://mycelix.net)
- **Matrix**: #mycelix:matrix.org (coming soon)

---

## 🌟 Star History

Help us grow! ⭐ Star the repo if you believe in privacy-first P2P collaboration.

---

*"Technology that amplifies human consciousness, not exploits it."*

**Status**: Active Development | **Target Release**: Q2 2026 | **Current Version**: 0.1.0-alpha