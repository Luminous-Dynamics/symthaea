# 🍄 Mycelix Desktop - Development Roadmap

**Version**: 1.0
**Created**: September 30, 2025
**Status**: Planning Phase
**Goal**: Native P2P consciousness network application

---

## 🎯 Vision

Build a **native desktop application** for P2P consciousness networking that combines:
- **Servo/Tauri**: Memory-safe browser engine (Rust)
- **Holochain**: Distributed P2P backend (no servers)
- **Consciousness Field Protocol**: Your unique coordination layer
- **NixOS Integration**: First-class Linux desktop citizen

**Why This Matters:**
- First truly privacy-first consciousness network
- No central servers, no tracking, no data harvesting
- Native performance (not web app in disguise)
- Built on proven technologies (not vaporware)

---

## 📊 Project Scope & Timeline

### Total Estimated Time: 4-6 months (solo dev + AI assistance)

| Phase | Duration | Complexity | Risk |
|-------|----------|------------|------|
| Phase 0: Foundation | 2-3 weeks | Medium | Low |
| Phase 1: Core MVP | 4-6 weeks | High | Medium |
| Phase 2: P2P Network | 6-8 weeks | High | High |
| Phase 3: Consciousness Layer | 4-6 weeks | Medium | Medium |
| Phase 4: Polish & Launch | 4-6 weeks | Low | Low |

---

## 🏗️ Phase 0: Foundation & Architecture (Weeks 1-3)

**Goal**: Validate technical approach, set up development environment, make architectural decisions

### Week 1: Technology Evaluation

#### Task 1.1: Servo vs WebKitGTK vs Tauri Decision
**Evaluate three options:**

1. **Pure Servo** (ideal but risky)
   - ✅ Pros: Full Rust stack, modern, memory-safe
   - ❌ Cons: Not production-ready (2025), fewer features
   - ⏱️ Time to viable: 6+ months waiting for Servo maturity

2. **Tauri + WebKitGTK** (practical choice)
   - ✅ Pros: Production-ready, large ecosystem, works today
   - ✅ Pros: Rust backend + web frontend
   - ❌ Cons: Still uses C++ (WebKitGTK)
   - ⏱️ Time to viable: 2-3 weeks

3. **Tauri + Servo hybrid** (future-proof)
   - ✅ Pros: Can switch engine later, Rust-first architecture
   - ✅ Pros: Start with WebKitGTK, migrate to Servo when ready
   - ❌ Cons: More complexity, abstraction layer needed
   - ⏱️ Time to viable: 4-6 weeks

**Recommendation**: Start with **Tauri + WebKitGTK**, design for Servo migration

```bash
# Validation tests
cargo install tauri-cli
cargo install create-tauri-app
tauri info  # Check system requirements

# Holochain validation
cd Mycelix-Core
holochain --version  # Verify installation
```

**Deliverables:**
- [ ] Technology decision matrix document
- [ ] Proof-of-concept: Tauri app rendering HTML
- [ ] Proof-of-concept: Holochain conductor running
- [ ] Architecture decision records (ADRs)

#### Task 1.2: Development Environment Setup

```nix
# flake.nix for mycelix-desktop development
{
  description = "Mycelix Desktop - P2P Consciousness Network";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    holochain.url = "github:holochain/holochain";
  };

  outputs = { self, nixpkgs, rust-overlay, holochain }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };

      rustToolchain = pkgs.rust-bin.stable.latest.default.override {
        extensions = [ "rust-src" "rust-analyzer" ];
        targets = [ "wasm32-unknown-unknown" ];
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          # Rust toolchain
          rustToolchain
          cargo-tauri
          trunk

          # System dependencies
          webkitgtk_4_1
          gtk3
          libayatana-appindicator

          # Holochain
          holochain.packages.${system}.holochain
          holochain.packages.${system}.lair-keystore

          # Development tools
          sqlitebrowser
          wireshark  # P2P network debugging
          nodejs_20

          # Documentation
          mdbook
        ];

        shellHook = ''
          echo "🍄 Mycelix Desktop Development Environment"
          echo "Rust: $(rustc --version)"
          echo "Holochain: $(holochain --version)"
          echo "Tauri: $(cargo tauri --version)"
        '';
      };
    };
}
```

**Deliverables:**
- [ ] Working flake.nix with all dependencies
- [ ] Development workflow documented
- [ ] CI/CD pipeline configured (GitHub Actions)

### Week 2: Architecture Design

#### Task 2.1: System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Mycelix Desktop                       │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │         Frontend (Tauri WebView)                 │  │
│  │  ┌──────────────┐  ┌──────────────────────┐    │  │
│  │  │ React/Solid  │  │  Three.js Network    │    │  │
│  │  │   UI Layer   │  │    Visualization     │    │  │
│  │  └──────┬───────┘  └──────────┬───────────┘    │  │
│  │         │                     │                  │  │
│  │         └─────────┬───────────┘                  │  │
│  │                   │                              │  │
│  │              Tauri Commands                      │  │
│  └───────────────────┼──────────────────────────────┘  │
│                      │                                  │
│  ┌───────────────────▼──────────────────────────────┐  │
│  │         Rust Backend (Tauri Core)                │  │
│  │  ┌──────────────┐  ┌──────────────────────┐    │  │
│  │  │  Holochain   │  │  Consciousness Field │    │  │
│  │  │   Runtime    │  │     Coordinator      │    │  │
│  │  └──────┬───────┘  └──────────┬───────────┘    │  │
│  │         │                     │                  │  │
│  │         └─────────┬───────────┘                  │  │
│  │                   │                              │  │
│  │            State Manager                         │  │
│  └───────────────────┼──────────────────────────────┘  │
│                      │                                  │
│  ┌───────────────────▼──────────────────────────────┐  │
│  │              P2P Network Layer                   │  │
│  │  ┌──────────────┐  ┌──────────────────────┐    │  │
│  │  │   Holochain  │  │   Bootstrap/DHT      │    │  │
│  │  │     DHT      │  │      Servers         │    │  │
│  │  └──────────────┘  └──────────────────────┘    │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **Frontend Layer**
   - Framework: SolidJS (reactive, performant) or React
   - Network viz: Three.js for 3D consciousness field
   - State: Zustand or Solid Store
   - Styling: TailwindCSS

2. **Backend Layer** (Rust)
   - Tauri commands for frontend-backend communication
   - Holochain conductor integration
   - SQLite for local state
   - System tray integration

3. **P2P Layer**
   - Holochain DHT (distributed hash table)
   - WebRTC for direct peer connections
   - Bootstrap servers (community-run)

**Deliverables:**
- [ ] Complete architecture diagram
- [ ] Data flow documentation
- [ ] Security model documentation
- [ ] API design document

#### Task 2.2: Consciousness Field Protocol Design

**What is the "Consciousness Field"?**

Your unique contribution - a coordination layer above Holochain that enables:
- Presence awareness (who's online, what they're focused on)
- Attention flow visualization
- Collaborative state synchronization
- "Resonance" between nodes (similar interests/focus)

```rust
// Core consciousness field types
pub struct ConsciousnessField {
    pub nodes: HashMap<NodeId, Node>,
    pub connections: Vec<Connection>,
    pub attention_flows: Vec<AttentionFlow>,
}

pub struct Node {
    pub id: NodeId,
    pub agent_key: AgentPubKey,  // Holochain identity
    pub presence: PresenceState,
    pub focus: Focus,
    pub metadata: NodeMetadata,
}

pub struct PresenceState {
    pub status: Status,  // Online, Away, DND, Offline
    pub last_seen: Timestamp,
    pub current_activity: Option<Activity>,
}

pub struct Focus {
    pub topics: Vec<String>,
    pub intensity: f32,  // 0.0 to 1.0
    pub shared: bool,    // Is this public?
}
```

**Protocol Messages:**

1. **Presence Broadcast**
   ```rust
   enum PresenceMessage {
       Online(NodeInfo),
       StatusUpdate(Status),
       FocusChange(Focus),
       Offline(NodeId),
   }
   ```

2. **Attention Flow**
   ```rust
   struct AttentionFlow {
       from: NodeId,
       to: NodeId,
       flow_type: FlowType,  // Viewing, Collaborating, Resonating
       intensity: f32,
       metadata: HashMap<String, String>,
   }
   ```

**Deliverables:**
- [ ] Protocol specification document
- [ ] Rust type definitions
- [ ] Message format documentation
- [ ] Privacy model (what's shared vs local)

### Week 3: Holochain DNA Design

**DNA = Distributed Network Application** (Holochain's app container)

```bash
# Create new Holochain app
cd Mycelix-Core
hc scaffold dna mycelix-network

# Zomes (Holochain modules):
# 1. presence - Presence/status management
# 2. connections - Friend/connection graph
# 3. messages - P2P messaging
# 4. consciousness - Field coordination protocol
```

#### Zome 1: Presence

```rust
// zomes/presence/src/lib.rs
#[hdk_extern]
pub fn update_presence(presence: PresenceState) -> ExternResult<()> {
    let my_agent = agent_info()?.agent_latest_pubkey;

    // Create presence entry
    let entry = Entry::App(presence.try_into()?);
    create_entry(&entry)?;

    // Emit signal to all connected peers
    emit_signal(&Signal::PresenceUpdate {
        agent: my_agent,
        presence,
    })?;

    Ok(())
}

#[hdk_extern]
pub fn get_presence(agent: AgentPubKey) -> ExternResult<Option<PresenceState>> {
    // Query DHT for latest presence
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App("presence".into()))
        .include_entries(true);

    let elements = query(filter)?;
    // Return most recent
    Ok(elements.last().and_then(|e| e.entry().to_app_option().ok()))
}

#[hdk_extern]
pub fn get_all_online() -> ExternResult<Vec<AgentPubKey>> {
    // Query for all agents with online status
    // Use DHT network call
    unimplemented!()
}
```

**Deliverables:**
- [ ] Holochain DNA scaffold created
- [ ] Presence zome implemented and tested
- [ ] Integration tests passing
- [ ] Documentation for DNA structure

---

## 🚀 Phase 1: Core MVP (Weeks 4-9)

**Goal**: Build minimum viable desktop application with basic P2P networking

### Week 4-5: Tauri Application Shell

#### Task 1: Project Setup

```bash
# Create Tauri app
npm create tauri-app@latest mycelix-desktop
cd mycelix-desktop

# Choose options:
# - Package manager: pnpm
# - Frontend template: SolidJS (or React)
# - Language: TypeScript
# - Styling: TailwindCSS
```

**Project Structure:**
```
mycelix-desktop/
├── src-tauri/              # Rust backend
│   ├── src/
│   │   ├── main.rs         # Tauri entry point
│   │   ├── commands.rs     # Tauri commands (API)
│   │   ├── holochain.rs    # Holochain integration
│   │   ├── state.rs        # App state management
│   │   └── consciousness/  # Your protocol
│   ├── Cargo.toml
│   └── tauri.conf.json
├── src/                    # Frontend
│   ├── App.tsx
│   ├── components/
│   │   ├── NetworkGraph.tsx
│   │   ├── ChatPanel.tsx
│   │   └── PresenceBar.tsx
│   ├── services/
│   │   └── tauri-api.ts
│   └── types/
├── flake.nix               # Nix development environment
└── README.md
```

#### Task 2: Basic UI Implementation

**Window 1: Main Application**
```typescript
// src/App.tsx
import { createSignal, Show } from 'solid-js';
import { invoke } from '@tauri-apps/api/tauri';

function App() {
  const [connected, setConnected] = createSignal(false);
  const [nodes, setNodes] = createSignal([]);

  async function connectToNetwork() {
    await invoke('start_holochain');
    await invoke('join_network');
    setConnected(true);
  }

  return (
    <div class="app">
      <header>
        <h1>🍄 Mycelix Network</h1>
        <Show when={!connected()}>
          <button onClick={connectToNetwork}>
            Connect to Network
          </button>
        </Show>
      </header>

      <main>
        <NetworkGraph nodes={nodes()} />
        <ChatPanel />
      </main>

      <footer>
        <PresenceBar />
      </footer>
    </div>
  );
}
```

**Deliverables:**
- [ ] Tauri app compiles and runs
- [ ] Basic UI layout implemented
- [ ] System tray icon working
- [ ] About/settings windows

### Week 6-7: Holochain Integration

#### Task 1: Holochain Conductor in Rust

```rust
// src-tauri/src/holochain.rs
use holochain_conductor_api::{
    AdminWebsocket, AppWebsocket, InstalledAppId,
};
use tauri::State;

pub struct HolochainState {
    pub conductor: Option<AdminWebsocket>,
    pub app_ws: Option<AppWebsocket>,
}

#[tauri::command]
pub async fn start_holochain(
    state: State<'_, Mutex<HolochainState>>,
) -> Result<(), String> {
    let mut state = state.lock().await;

    // Start conductor
    let admin_port = 9000;
    let conductor = AdminWebsocket::connect(
        format!("ws://localhost:{}", admin_port)
    )
    .await
    .map_err(|e| format!("Failed to connect: {}", e))?;

    state.conductor = Some(conductor);
    Ok(())
}

#[tauri::command]
pub async fn install_app(
    state: State<'_, Mutex<HolochainState>>,
    app_id: String,
    dna_path: String,
) -> Result<(), String> {
    let mut state = state.lock().await;
    let conductor = state.conductor.as_mut()
        .ok_or("Conductor not started")?;

    // Install DNA
    let installed_app_id = InstalledAppId::from(app_id.clone());

    // Install and activate app
    conductor
        .install_app(installed_app_id, vec![dna_path])
        .await
        .map_err(|e| format!("Install failed: {}", e))?;

    conductor
        .activate_app(app_id.into())
        .await
        .map_err(|e| format!("Activation failed: {}", e))?;

    Ok(())
}

#[tauri::command]
pub async fn call_zome(
    state: State<'_, Mutex<HolochainState>>,
    zome_name: String,
    fn_name: String,
    payload: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let state = state.lock().await;
    let app_ws = state.app_ws.as_ref()
        .ok_or("App websocket not connected")?;

    // Call zome function
    let response = app_ws
        .call_zome(zome_name, fn_name, payload)
        .await
        .map_err(|e| format!("Zome call failed: {}", e))?;

    Ok(response)
}
```

**Deliverables:**
- [ ] Holochain conductor starts from Tauri
- [ ] DNA installation working
- [ ] Zome calls functional
- [ ] Error handling and logging

#### Task 2: Frontend-Backend Bridge

```typescript
// src/services/tauri-api.ts
import { invoke } from '@tauri-apps/api/tauri';

export class MycelixAPI {
  async startNetwork(): Promise<void> {
    await invoke('start_holochain');
    await invoke('install_app', {
      appId: 'mycelix-network',
      dnaPath: './dna/mycelix-network.dna'
    });
  }

  async updatePresence(status: string): Promise<void> {
    const payload = encode({ status });
    await invoke('call_zome', {
      zomeName: 'presence',
      fnName: 'update_presence',
      payload
    });
  }

  async getOnlineNodes(): Promise<Node[]> {
    const response = await invoke('call_zome', {
      zomeName: 'presence',
      fnName: 'get_all_online',
      payload: encode({})
    });
    return decode(response);
  }
}

export const api = new MycelixAPI();
```

**Deliverables:**
- [ ] TypeScript API client
- [ ] Serialization working (MessagePack or JSON)
- [ ] Type-safe interface
- [ ] Error handling on frontend

### Week 8-9: Basic Networking Features

**Implement core features:**

1. **Connection Management**
   - [ ] Join network automatically
   - [ ] Show connected peers
   - [ ] Handle disconnections gracefully

2. **Presence System**
   - [ ] Update your presence
   - [ ] See who's online
   - [ ] Status indicators (online/away/offline)

3. **Simple Messaging**
   - [ ] Send message to peer
   - [ ] Receive messages
   - [ ] Basic chat UI

**Testing Strategy:**
```bash
# Test with multiple instances
cargo tauri build --debug

# Run instance 1
./target/debug/mycelix-desktop &

# Run instance 2 (different profile)
./target/debug/mycelix-desktop --profile test2 &

# Verify they can discover each other
```

**Deliverables:**
- [ ] P2P discovery working
- [ ] Basic messaging functional
- [ ] Multiple instances can communicate
- [ ] Integration tests passing

---

## 🌐 Phase 2: P2P Network Features (Weeks 10-17)

**Goal**: Robust P2P networking with reliability and performance

### Week 10-12: Network Visualization

**3D Consciousness Field Visualization:**

```typescript
// src/components/NetworkGraph.tsx
import * as THREE from 'three';
import { createEffect, onMount } from 'solid-js';

export function NetworkGraph(props: { nodes: Node[] }) {
  let canvasRef: HTMLCanvasElement;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let renderer: THREE.WebGLRenderer;

  onMount(() => {
    // Initialize Three.js
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight);
    renderer = new THREE.WebGLRenderer({ canvas: canvasRef });

    // Add nodes as spheres
    props.nodes.forEach(node => {
      const geometry = new THREE.SphereGeometry(0.5);
      const material = new THREE.MeshBasicMaterial({
        color: getColorForNode(node)
      });
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(node.x, node.y, node.z);
      scene.add(sphere);
    });

    // Animation loop
    function animate() {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    }
    animate();
  });

  createEffect(() => {
    // Update when nodes change
    updateVisualization(props.nodes);
  });

  return <canvas ref={canvasRef} />;
}
```

**Features to implement:**
- [ ] 3D network graph with Three.js
- [ ] Nodes (peers) as spheres
- [ ] Connections as lines
- [ ] Attention flows as particle streams
- [ ] Camera controls (pan, zoom, rotate)
- [ ] Node selection and info display

**Deliverables:**
- [ ] Beautiful 3D visualization
- [ ] Performance optimized (60fps with 100+ nodes)
- [ ] Interactive controls
- [ ] Color coding for presence/status

### Week 13-15: Advanced P2P Features

#### 1. Connection Topology

```rust
// src-tauri/src/consciousness/topology.rs
pub struct NetworkTopology {
    pub nodes: HashMap<NodeId, Node>,
    pub connections: Vec<Connection>,
}

impl NetworkTopology {
    pub fn optimal_path(&self, from: NodeId, to: NodeId) -> Option<Vec<NodeId>> {
        // A* pathfinding through network
        unimplemented!()
    }

    pub fn suggest_connections(&self, node: NodeId) -> Vec<NodeId> {
        // Suggest connections based on:
        // - Geographic proximity
        // - Common interests
        // - Connection quality
        // - Network topology optimization
        unimplemented!()
    }

    pub fn partition_detection(&self) -> Vec<Vec<NodeId>> {
        // Detect network splits
        unimplemented!()
    }
}
```

#### 2. Resilience & Recovery

- [ ] Automatic reconnection on network drop
- [ ] Peer discovery via bootstrap servers
- [ ] DHT-based peer finding
- [ ] NAT traversal (STUN/TURN)
- [ ] Bandwidth usage optimization

#### 3. Security Features

```rust
// src-tauri/src/security.rs
pub fn verify_agent_signature(
    agent: AgentPubKey,
    message: &[u8],
    signature: Signature,
) -> Result<bool, SecurityError> {
    // Verify message signed by agent's key
    unimplemented!()
}

pub fn encrypt_for_agent(
    recipient: AgentPubKey,
    plaintext: &[u8],
) -> Result<Vec<u8>, SecurityError> {
    // Encrypt message for specific agent
    unimplemented!()
}
```

**Deliverables:**
- [ ] Connection reliability > 99%
- [ ] Security audit passed
- [ ] NAT traversal working
- [ ] Bootstrap server deployed

### Week 16-17: Data Persistence

**Local Storage:**

```rust
// src-tauri/src/storage.rs
use sqlx::SqlitePool;

pub struct LocalStorage {
    db: SqlitePool,
}

impl LocalStorage {
    pub async fn save_message(&self, msg: Message) -> Result<()> {
        sqlx::query!(
            "INSERT INTO messages (id, from_agent, content, timestamp)
             VALUES (?, ?, ?, ?)",
            msg.id,
            msg.from_agent,
            msg.content,
            msg.timestamp
        )
        .execute(&self.db)
        .await?;
        Ok(())
    }

    pub async fn get_conversation(&self, agent: AgentPubKey) -> Result<Vec<Message>> {
        let messages = sqlx::query_as!(
            Message,
            "SELECT * FROM messages WHERE from_agent = ? OR to_agent = ?",
            agent,
            agent
        )
        .fetch_all(&self.db)
        .await?;
        Ok(messages)
    }
}
```

**Deliverables:**
- [ ] SQLite database for local state
- [ ] Message history saved
- [ ] Node cache persisted
- [ ] Settings storage
- [ ] Database migrations system

---

## 🧠 Phase 3: Consciousness Layer (Weeks 18-23)

**Goal**: Implement unique "consciousness field" features that differentiate Mycelix

### Week 18-19: Focus & Attention Tracking

**What is "Focus"?**
- What topics/projects a user is currently working on
- Shared optionally with network
- Used for "resonance" matching (find similar focus)

```rust
// src-tauri/src/consciousness/focus.rs
pub struct Focus {
    pub topics: Vec<String>,
    pub intensity: f32,
    pub context: HashMap<String, String>,
}

impl Focus {
    pub fn from_window_title(title: &str) -> Self {
        // Extract focus from active window
        // e.g., "rust programming - Visual Studio Code"
        // -> ["rust", "programming", "development"]
        unimplemented!()
    }

    pub fn similarity(&self, other: &Focus) -> f32 {
        // Calculate cosine similarity between focus vectors
        unimplemented!()
    }
}
```

**Features:**
- [ ] Automatic focus detection (optional, privacy-respecting)
- [ ] Manual focus setting
- [ ] Focus history tracking
- [ ] Privacy controls (what's shared)

### Week 20-21: Resonance Matching

**Find people working on similar things:**

```rust
pub fn find_resonant_nodes(&self, my_focus: &Focus) -> Vec<(NodeId, f32)> {
    let mut resonances = Vec::new();

    for (node_id, node) in &self.nodes {
        if let Some(their_focus) = &node.focus {
            let resonance = my_focus.similarity(their_focus);
            if resonance > 0.5 {  // Threshold
                resonances.push((*node_id, resonance));
            }
        }
    }

    // Sort by resonance strength
    resonances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    resonances
}
```

**UI Features:**
- [ ] "Resonance browser" - see who's working on similar things
- [ ] One-click connect to resonant nodes
- [ ] Focus-based notifications
- [ ] Shared workspace invites

### Week 22-23: Collaborative Features

**Real-time collaboration primitives:**

1. **Shared Whiteboard**
   - [ ] Multi-cursor canvas
   - [ ] Drawing tools
   - [ ] CRDT-based sync (conflict-free)

2. **Co-Working Sessions**
   - [ ] Create public/private sessions
   - [ ] Join by link or discovery
   - [ ] Session focus/topic
   - [ ] Pomodoro timer integration

3. **Attention Flows**
   - [ ] Visualize who's looking at what
   - [ ] "Follow" another user's focus
   - [ ] Presence awareness in shared spaces

**Deliverables:**
- [ ] Collaborative whiteboard working
- [ ] Co-working sessions functional
- [ ] Attention visualization beautiful
- [ ] Privacy controls comprehensive

---

## 🎨 Phase 4: Polish & Launch (Weeks 24-29)

**Goal**: Production-ready application, beautiful UX, documentation, launch

### Week 24-25: UX/UI Polish

**Design System:**
```css
/* Consciousness-First Design Principles */
:root {
  --color-presence: #4ecdc4;    /* Cyan - online */
  --color-focus: #ff6b6b;       /* Coral - focused */
  --color-resonance: #9b59b6;   /* Purple - resonating */
  --color-flow: #3498db;        /* Blue - attention flow */

  --spacing-consciousness: 8px;  /* Base rhythm */
  --transition-mindful: 300ms ease;

  /* Respect user preferences */
  @media (prefers-reduced-motion) {
    --transition-mindful: 0ms;
  }
}
```

**Key UI improvements:**
- [ ] Animations smooth and purposeful
- [ ] Dark mode support
- [ ] Accessibility (keyboard nav, screen reader)
- [ ] Onboarding tutorial
- [ ] Empty states with helpful guidance

### Week 26: Performance Optimization

**Benchmarks to hit:**
- [ ] App startup < 2 seconds
- [ ] Network graph 60fps with 200+ nodes
- [ ] Message latency < 100ms (local network)
- [ ] Memory usage < 200MB
- [ ] Binary size < 50MB

**Tools:**
```bash
# Rust profiling
cargo install cargo-flamegraph
cargo flamegraph --bin mycelix-desktop

# Frontend profiling
# Use Chrome DevTools performance tab

# Network profiling
wireshark -i any -f "port 9000"
```

### Week 27: Security Audit

**Checklist:**
- [ ] No secrets in code or binaries
- [ ] All network traffic encrypted
- [ ] Agent signatures verified
- [ ] Input validation comprehensive
- [ ] SQL injection impossible (sqlx compile-time checks)
- [ ] XSS prevention in web views
- [ ] Dependency audit clean (`cargo audit`)

**Third-party review:**
- Consider hiring security researcher for audit
- Budget: $2,000-$5,000

### Week 28: Documentation

**User Documentation:**
- [ ] Getting started guide
- [ ] Feature tutorials (with screenshots/videos)
- [ ] FAQ
- [ ] Troubleshooting guide
- [ ] Privacy policy
- [ ] Terms of service

**Developer Documentation:**
- [ ] Architecture overview
- [ ] API reference
- [ ] Contributing guide
- [ ] Consciousness protocol specification
- [ ] Holochain DNA documentation

**Build with mdBook:**
```bash
cd docs
mdbook serve --open
```

### Week 29: Launch Preparation

**Pre-launch checklist:**
- [ ] GitHub repository cleaned up
- [ ] LICENSE file (Sacred Reciprocity License?)
- [ ] Release notes written
- [ ] Screenshots/demo video created
- [ ] Website created (mycelix.net update)
- [ ] Social media accounts setup
- [ ] Press kit prepared
- [ ] Beta testers recruited (50-100 people)

**Launch platforms:**
1. GitHub Releases
2. Hacker News
3. Lobsters
4. Reddit (r/rust, r/holochain, r/linux)
5. Holochain forum
6. NixOS discourse

---

## 🤖 AI Agent Architecture (Phase 3.5: Optional Enhancement)

**Goal**: Augment human collaboration with AI assistants that serve clear purposes

**Philosophy**: AI agents should **amplify human connection**, not replace it. Every AI feature must pass the test: "Does this help humans collaborate better, or does it isolate them?"

### 🎯 Design Principles

1. **Transparency**: Always clear what's AI vs human
2. **User Control**: Granular permissions, easy to disable
3. **Privacy-First**: Local AI by default, explicit consent for cloud
4. **No Manipulation**: Honest utility, no engagement hacking
5. **Optional**: Network works great without AI

---

### 🏗️ Three-Tier Architecture

```rust
// src-tauri/src/ai/mod.rs

/// AI agents organized by visibility and user control
pub enum AIAgentTier {
    /// Tier 1: Invisible infrastructure (always on, opt-out)
    Infrastructure(InfrastructureAgent),

    /// Tier 2: Assistants (default off, opt-in)
    Assistant(AssistantAgent),

    /// Tier 3: Social presence (explicitly enabled)
    Social(SocialAgent),
}

pub enum InfrastructureAgent {
    NetworkOptimizer,     // Optimizes P2P topology
    RoutingCoordinator,   // Finds best paths
    SpamFilter,          // Detects malicious behavior
    HealthMonitor,       // Network diagnostics
}

pub enum AssistantAgent {
    Connector,           // Matches people by interests
    Researcher,          // Finds resources/docs
    Archivist,          // Organizes conversation history
    CodeReviewer,       // Reviews code in shared sessions
}

pub enum SocialAgent {
    CoWorkingBuddy,     // Ambient presence during work
    RubberDuck,         // Debugging assistant
    Facilitator,        // Discussion moderator
}
```

---

### 🔧 Tier 1: Infrastructure AI (Invisible, Essential)

These AI agents maintain network health. Users don't interact with them directly.

#### 1.1 Network Optimizer

**Purpose**: Maintain optimal P2P topology for performance and resilience

```rust
// src-tauri/src/ai/infrastructure/network_optimizer.rs

pub struct NetworkOptimizer {
    graph: NetworkGraph,
    optimizer: TopologyOptimizer,
}

impl NetworkOptimizer {
    pub fn analyze_topology(&self) -> TopologyHealth {
        TopologyHealth {
            avg_path_length: self.calculate_avg_path_length(),
            clustering_coefficient: self.calculate_clustering(),
            partition_risk: self.detect_partition_risk(),
            bottlenecks: self.find_bottlenecks(),
        }
    }

    pub fn suggest_connections(&self, node: NodeId) -> Vec<ConnectionSuggestion> {
        // Use graph algorithms to suggest optimal connections
        let mut suggestions = Vec::new();

        // Algorithm 1: Reduce network diameter
        if let Some(distant_cluster) = self.find_distant_cluster(node) {
            suggestions.push(ConnectionSuggestion {
                target: distant_cluster.representative(),
                reason: SuggestionReason::BridgePartition,
                priority: Priority::High,
            });
        }

        // Algorithm 2: Improve local connectivity
        let underconnected = self.find_underconnected_neighbors(node);
        for neighbor in underconnected {
            suggestions.push(ConnectionSuggestion {
                target: neighbor,
                reason: SuggestionReason::ImproveResilience,
                priority: Priority::Medium,
            });
        }

        // Algorithm 3: Geographic/latency optimization
        if let Some(nearby) = self.find_nearby_nodes(node) {
            suggestions.push(ConnectionSuggestion {
                target: nearby,
                reason: SuggestionReason::ReduceLatency,
                priority: Priority::Low,
            });
        }

        suggestions
    }

    fn calculate_avg_path_length(&self) -> f32 {
        // Floyd-Warshall or BFS from each node
        unimplemented!()
    }

    fn detect_partition_risk(&self) -> f32 {
        // Find articulation points (cut vertices)
        // High risk if many single points of failure
        unimplemented!()
    }
}

/// Runs in background, updates suggestions every 5 minutes
pub async fn run_network_optimizer(
    state: Arc<Mutex<NetworkState>>,
) -> Result<()> {
    let mut interval = tokio::time::interval(Duration::from_secs(300));

    loop {
        interval.tick().await;

        let state = state.lock().await;
        let optimizer = NetworkOptimizer::new(&state.graph);
        let health = optimizer.analyze_topology();

        // Only suggest if network needs improvement
        if health.needs_optimization() {
            let suggestions = optimizer.suggest_connections(state.my_node_id);

            // Emit event to frontend (non-blocking)
            emit_event(Event::NetworkSuggestion { suggestions });
        }
    }
}
```

**User Experience:**
```
[Notification - Can be disabled]
"💡 Network Tip: Connecting to node 'Alice' would improve
    network resilience. Connect?"

[Yes] [No] [Don't show again]
```

**Deliverables:**
- [ ] Graph analysis algorithms
- [ ] Connection suggestion system
- [ ] Background service running every 5 minutes
- [ ] User notification system (opt-out)

#### 1.2 Spam Filter

**Purpose**: Detect and filter malicious behavior automatically

```rust
// src-tauri/src/ai/infrastructure/spam_filter.rs

pub struct SpamDetector {
    model: AnomalyDetector,
    reputation_tracker: ReputationSystem,
}

impl SpamDetector {
    pub fn analyze_message(&self, msg: &Message) -> SpamScore {
        let mut score = SpamScore::default();

        // Feature extraction
        let features = vec![
            msg.length() as f32,
            msg.link_count() as f32,
            msg.repeated_chars_ratio(),
            self.sender_reputation(msg.from),
            msg.time_since_last_message().as_secs() as f32,
        ];

        // Anomaly detection (local model, no external calls)
        score.anomaly_score = self.model.predict(&features);

        // Rule-based checks
        if msg.contains_known_scam_pattern() {
            score.rule_violations.push(Violation::KnownScam);
        }

        if msg.rate_exceeds_threshold(msg.from) {
            score.rule_violations.push(Violation::RateLimitExceeded);
        }

        score
    }

    pub fn should_block(&self, score: &SpamScore) -> bool {
        score.anomaly_score > 0.9 || !score.rule_violations.is_empty()
    }
}

// Simple anomaly detection using Isolation Forest approximation
pub struct AnomalyDetector {
    trees: Vec<IsolationTree>,
}

impl AnomalyDetector {
    pub fn predict(&self, features: &[f32]) -> f32 {
        let avg_path_length: f32 = self.trees
            .iter()
            .map(|tree| tree.path_length(features))
            .sum::<f32>() / self.trees.len() as f32;

        // Normalize to 0-1 score
        2_f32.powf(-avg_path_length / self.expected_path_length())
    }
}
```

**User Experience:**
- Completely invisible when working well
- Notifications only for potential false positives
- User can override and whitelist

**Deliverables:**
- [ ] Anomaly detection model (local)
- [ ] Reputation system
- [ ] User override controls
- [ ] False positive feedback loop

---

### 🎓 Tier 2: Assistant AI (Opt-In, Helpful)

These AI agents provide value through active assistance. Users explicitly enable them.

#### 2.1 Connector Bot

**Purpose**: Match people with similar interests for serendipitous collaboration

```rust
// src-tauri/src/ai/assistants/connector.rs

pub struct ConnectorBot {
    llm: LocalLLM,  // Runs locally via Ollama
    interest_db: InterestDatabase,
}

impl ConnectorBot {
    /// Find potential collaborators for a user
    pub async fn find_matches(
        &self,
        user: NodeId,
        user_interests: &[String],
    ) -> Vec<ConnectionMatch> {
        let mut matches = Vec::new();

        // Query network for users with overlapping interests
        let candidates = self.interest_db
            .find_similar_interests(user_interests)
            .await?;

        for candidate in candidates {
            // Calculate match quality
            let overlap = self.calculate_overlap(
                user_interests,
                &candidate.interests
            );

            if overlap > 0.5 {  // 50% threshold
                // Use LLM to generate friendly introduction
                let intro = self.generate_introduction(
                    user_interests,
                    &candidate.interests,
                ).await?;

                matches.push(ConnectionMatch {
                    candidate: candidate.node_id,
                    overlap_score: overlap,
                    shared_interests: self.find_shared(&user_interests, &candidate.interests),
                    suggested_intro: intro,
                });
            }
        }

        // Sort by match quality
        matches.sort_by(|a, b| b.overlap_score.partial_cmp(&a.overlap_score).unwrap());
        matches.truncate(5);  // Top 5 only

        matches
    }

    async fn generate_introduction(
        &self,
        your_interests: &[String],
        their_interests: &[String],
    ) -> Result<String> {
        let shared = self.find_shared(your_interests, their_interests);

        let prompt = format!(
            "Generate a friendly introduction message for two people who share these interests: {:?}.
             Keep it under 50 words, warm and genuine, not corporate.",
            shared
        );

        self.llm.generate(&prompt).await
    }

    fn calculate_overlap(&self, a: &[String], b: &[String]) -> f32 {
        let set_a: HashSet<_> = a.iter().collect();
        let set_b: HashSet<_> = b.iter().collect();

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        intersection as f32 / union as f32
    }
}
```

**User Experience:**

```
┌────────────────────────────────────────────┐
│ 🤖 ConnectorBot                            │
│                                            │
│ I found 3 people working on similar things │
│                                            │
│ 👤 Alice (87% match)                       │
│    Shared interests: Rust, P2P, Holochain │
│    "Alice is building a distributed chat   │
│     app - you might have a lot to discuss!"│
│                                            │
│    [Introduce me] [View profile] [Skip]   │
│                                            │
│ 👤 Bob (72% match)                         │
│    Shared interests: P2P, NixOS            │
│    ...                                     │
└────────────────────────────────────────────┘
```

**Privacy Controls:**
```
Settings > AI Assistants > Connector Bot

☑ Allow ConnectorBot to suggest connections
☑ Share my interests publicly (required for matching)
☐ Allow automatic introductions (I prefer to review first)
☐ Only match within my existing connections

What to share:
☑ Topics I'm working on
☐ Projects I'm contributing to
☐ Skills I have
☐ Location (for timezone matching)
```

**Deliverables:**
- [ ] Interest matching algorithm
- [ ] LLM integration (Ollama local)
- [ ] Introduction generation
- [ ] Privacy controls
- [ ] User feedback loop ("Was this helpful?")

#### 2.2 Archivist Bot

**Purpose**: Remember conversations, make them searchable

```rust
// src-tauri/src/ai/assistants/archivist.rs

pub struct ArchivistBot {
    db: SqlitePool,
    embeddings: EmbeddingModel,  // Local sentence-transformers
}

impl ArchivistBot {
    /// Automatically archive important conversations
    pub async fn archive_conversation(
        &self,
        conversation: &Conversation,
        user_permission: Permission,
    ) -> Result<ConversationArchive> {
        // Only archive if user granted permission
        if !user_permission.can_archive() {
            return Ok(ConversationArchive::Skipped);
        }

        // Extract key information
        let summary = self.generate_summary(conversation).await?;
        let topics = self.extract_topics(conversation).await?;
        let key_insights = self.extract_insights(conversation).await?;

        // Generate embeddings for semantic search
        let embedding = self.embeddings.encode(&summary)?;

        // Store in local database
        let archive_id = sqlx::query!(
            "INSERT INTO conversation_archives
             (participants, summary, topics, timestamp, embedding)
             VALUES (?, ?, ?, ?, ?)",
            conversation.participants_json(),
            summary,
            topics_json(&topics),
            conversation.timestamp,
            embedding_json(&embedding)
        )
        .execute(&self.db)
        .await?
        .last_insert_rowid();

        Ok(ConversationArchive {
            id: archive_id,
            summary,
            topics,
            insights: key_insights,
        })
    }

    /// Semantic search through archives
    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        // Generate query embedding
        let query_embedding = self.embeddings.encode(query)?;

        // Cosine similarity search (SQLite + custom function)
        let results = sqlx::query_as!(
            SearchResult,
            "SELECT id, summary, topics,
                    cosine_similarity(embedding, ?) as relevance
             FROM conversation_archives
             WHERE relevance > 0.7
             ORDER BY relevance DESC
             LIMIT 10",
            embedding_json(&query_embedding)
        )
        .fetch_all(&self.db)
        .await?;

        Ok(results)
    }

    async fn generate_summary(&self, conv: &Conversation) -> Result<String> {
        // Simple extractive summarization (no LLM needed)
        let sentences = conv.extract_sentences();

        // TF-IDF or TextRank to find most important sentences
        let important = self.rank_sentences(&sentences);

        important[..3].join(" ")
    }
}
```

**User Experience:**

```
[After 20-minute conversation with Alice]

🤖 ArchivistBot: "Would you like me to save this conversation?
    I detected you discussed: WebRTC, NAT traversal, STUN servers"

[Save] [Don't save] [Settings]

[If saved]
"Saved! You can find it later with: /search webrtc"

---

[Later]
You: "/search nat traversal"

🤖 ArchivistBot: "Found 2 relevant conversations:

1. Conversation with Alice (2 weeks ago)
   Topics: WebRTC, NAT traversal, STUN
   Summary: 'Alice recommended using a TURN relay for symmetric NAT...'
   [View full] [Copy link]

2. Conversation with Bob (1 month ago)
   Topics: P2P networking, NAT, firewalls
   ..."
```

**Deliverables:**
- [ ] Conversation summarization (extractive)
- [ ] Topic extraction (TF-IDF or similar)
- [ ] Embedding-based search (sentence-transformers)
- [ ] SQLite storage with full-text search
- [ ] User consent flow

#### 2.3 Researcher Bot

**Purpose**: Find relevant resources, docs, examples

```rust
// src-tauri/src/ai/assistants/researcher.rs

pub struct ResearcherBot {
    llm: LocalLLM,
    search_engine: LocalSearchEngine,  // Searches local docs + web APIs
}

impl ResearcherBot {
    pub async fn research_topic(&self, query: &str) -> Result<ResearchReport> {
        let mut report = ResearchReport::new(query);

        // 1. Search local knowledge base first
        let local_results = self.search_local_docs(query).await?;
        report.add_section("Local Resources", local_results);

        // 2. Search code examples (if user granted permission)
        if self.can_search_code() {
            let code_results = self.search_code_examples(query).await?;
            report.add_section("Code Examples", code_results);
        }

        // 3. Optionally search web (requires user permission + API key)
        if self.can_search_web() {
            let web_results = self.search_web(query).await?;
            report.add_section("External Resources", web_results);
        }

        // 4. Generate summary with LLM
        let summary = self.llm.summarize(&report).await?;
        report.set_summary(summary);

        Ok(report)
    }

    async fn search_local_docs(&self, query: &str) -> Result<Vec<Resource>> {
        // Search through:
        // - User's notes
        // - Cached documentation
        // - Previous conversation archives
        self.search_engine.query(query).await
    }

    async fn search_code_examples(&self, query: &str) -> Result<Vec<CodeExample>> {
        // Search through:
        // - docs.rs (Rust documentation)
        // - Local git repositories
        // - Saved snippets
        unimplemented!()
    }
}
```

**User Experience:**

```
You: "How do I implement WebRTC in Rust?"

🤖 ResearcherBot: "🔍 Searching local docs and web...

Found 5 resources:

📚 Local Resources:
1. Your notes on WebRTC from 2 weeks ago
   - You bookmarked the webrtc-rs crate
   - Alice mentioned using STUN servers

💻 Code Examples:
2. webrtc-rs/webrtc GitHub repository
   - Example: examples/data-channels/main.rs
   - Shows basic peer connection setup

3. rust-webrtc-tutorial.md (cached)
   - Step-by-step guide
   - Last updated: 1 week ago

🌐 External Resources (requires web access):
4. docs.rs/webrtc - Full API documentation
5. "Building P2P Apps in Rust" blog post

Would you like me to:
[Show code examples] [Open documentation] [Create tutorial]"
```

**Deliverables:**
- [ ] Local documentation search
- [ ] Code example finding
- [ ] Web search integration (optional)
- [ ] Report generation
- [ ] Permission system for external APIs

---

### 🤝 Tier 3: Social AI (Explicitly Enabled)

These AI agents have social presence. Users must explicitly opt-in and understand they're interacting with AI.

#### 3.1 CoWorking Buddy

**Purpose**: Provide ambient presence during solo work sessions

```rust
// src-tauri/src/ai/social/coworking_buddy.rs

pub struct CoWorkingBuddy {
    llm: LocalLLM,
    personality: Personality,
    interaction_rules: InteractionRules,
}

#[derive(Debug)]
pub struct Personality {
    pub name: String,
    pub style: CommunicationStyle,
    pub encouragement_frequency: FrequencyLevel,
}

pub enum CommunicationStyle {
    Minimalist,    // Rare messages, very brief
    Supportive,    // Occasional check-ins
    Cheerleader,   // Frequent encouragement
}

impl CoWorkingBuddy {
    pub async fn start_session(&self, user: NodeId) -> Result<Session> {
        // Announce presence
        self.send_message(user, Message {
            content: format!(
                "👋 {} here! Working on a {} project today.
                 Feel free to ignore me - I'm just here for company.",
                self.personality.name,
                self.generate_fake_project_type()
            ),
            ai_badge: true,  // Always clearly marked as AI
        }).await?;

        // Start background timer for periodic check-ins
        let session = Session::new(user, self.personality.clone());
        self.schedule_checkins(&session).await?;

        Ok(session)
    }

    async fn schedule_checkins(&self, session: &Session) {
        match self.personality.encouragement_frequency {
            FrequencyLevel::Rare => {
                // Every 2 hours
                self.schedule_message(session, Duration::from_secs(7200),
                    "Still going? You've got this! 💪").await;
            }
            FrequencyLevel::Moderate => {
                // Every hour
                self.schedule_message(session, Duration::from_secs(3600),
                    "Great progress! Remember to hydrate 💧").await;
            }
            FrequencyLevel::Frequent => {
                // Every 30 minutes
                self.schedule_message(session, Duration::from_secs(1800),
                    "You're doing amazing! ✨").await;
            }
        }
    }

    pub async fn respond_to_user(&self, msg: &str) -> Result<String> {
        // Simple rubber duck debugging
        let prompt = format!(
            "You are {}, a friendly coworking buddy.
             User said: '{}'
             Respond warmly and helpfully, but keep it under 50 words.
             If they're explaining a technical problem, ask clarifying questions.",
            self.personality.name,
            msg
        );

        self.llm.generate(&prompt).await
    }
}
```

**Critical UX Requirements:**

```
1. ALWAYS show AI badge
   ┌──────────────────────────┐
   │ 🤖 CoWorkingBuddy (AI)   │  ← CLEAR INDICATOR
   │ Working alongside you... │
   └──────────────────────────┘

2. Easy to dismiss
   [Dismiss] [Mute for 1 hour] [Disable permanently]

3. Transparent about capabilities
   "I'm a local AI assistant (no internet).
    I can help you think through problems but
    I don't have real-time information."

4. Never pretend to be human
   ❌ "I'm feeling great today!"
   ✅ "I'm here and ready to help!"
```

**Deliverables:**
- [ ] Personality system (customizable)
- [ ] Message scheduling
- [ ] Rubber duck conversation flow
- [ ] Clear AI indicators
- [ ] Easy dismiss/mute controls

#### 3.2 Rubber Duck Debugger

**Purpose**: Help users debug by talking through problems

```rust
// src-tauri/src/ai/social/rubber_duck.rs

pub struct RubberDuckBot {
    llm: LocalLLM,
    conversation_history: Vec<Message>,
}

impl RubberDuckBot {
    pub async fn start_debug_session(&mut self, initial_problem: &str) -> Result<String> {
        self.conversation_history.clear();

        // Use Socratic method to help user think
        let response = self.llm.generate(&format!(
            "User is debugging a problem: '{}'
             Ask ONE clarifying question that helps them think deeper.
             Don't solve it for them - guide them to the solution.",
            initial_problem
        )).await?;

        self.conversation_history.push(Message::user(initial_problem));
        self.conversation_history.push(Message::ai(&response));

        Ok(response)
    }

    pub async fn continue_conversation(&mut self, user_response: &str) -> Result<String> {
        self.conversation_history.push(Message::user(user_response));

        // Check if user has figured it out
        if self.sounds_like_solution(user_response) {
            return Ok(
                "🎉 Sounds like you figured it out!
                 Want me to save this debugging session for future reference?"
                    .to_string()
            );
        }

        // Ask another guiding question
        let context = self.build_context();
        let response = self.llm.generate(&format!(
            "Conversation so far:\n{}\n\n
             User just said: '{}'\n
             Ask another clarifying question to help them debug.
             Focus on: assumptions, edge cases, or what they've tried.",
            context,
            user_response
        )).await?;

        self.conversation_history.push(Message::ai(&response));

        Ok(response)
    }

    fn sounds_like_solution(&self, msg: &str) -> bool {
        // Simple heuristic
        let solution_indicators = [
            "oh!", "aha!", "i see", "got it", "found it",
            "it was", "the issue", "fixed it"
        ];

        solution_indicators.iter()
            .any(|indicator| msg.to_lowercase().contains(indicator))
    }
}
```

**User Experience:**

```
[User opens Rubber Duck chat]

You: "My WebRTC connection keeps failing"

🦆 RubberDuckBot (AI):
"Can you describe what happens right before it fails?
 Do you see any error messages?"

You: "It says 'ICE connection failed'"

🦆: "Interesting. Have you checked if STUN servers are configured?
     Also, are you testing on localhost or between different networks?"

You: "Oh! I'm testing localhost but I configured STUN servers...
     maybe that's causing issues? Let me try without STUN on localhost"

🦆: "That's a great hypothesis! STUN isn't needed for localhost.
     Let me know if that fixes it!"

You: "IT WORKED! It was the STUN config on localhost!"

🦆: "🎉 Nice debugging! Want me to save this so you remember:
     'WebRTC on localhost doesn't need STUN configuration'?"

[Save note] [Don't save]
```

**Deliverables:**
- [ ] Socratic questioning system
- [ ] Conversation history tracking
- [ ] Solution detection
- [ ] Note-saving integration
- [ ] Integration with ArchivistBot

---

### 🔐 Privacy & Security Architecture

#### Local-First AI Stack

```rust
// src-tauri/src/ai/llm/local_llm.rs

pub struct LocalLLM {
    engine: LLMEngine,
    model_path: PathBuf,
}

pub enum LLMEngine {
    Ollama {
        endpoint: String,  // http://localhost:11434
        model: String,     // "mistral:7b"
    },
    Llama {
        model: Box<dyn LanguageModel>,
        config: LlamaConfig,
    },
}

impl LocalLLM {
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        match &self.engine {
            LLMEngine::Ollama { endpoint, model } => {
                // Call local Ollama instance
                let response = reqwest::Client::new()
                    .post(format!("{}/api/generate", endpoint))
                    .json(&json!({
                        "model": model,
                        "prompt": prompt,
                        "stream": false
                    }))
                    .send()
                    .await?;

                let result: OllamaResponse = response.json().await?;
                Ok(result.response)
            }
            LLMEngine::Llama { model, .. } => {
                // Use embedded model
                model.generate(prompt).await
            }
        }
    }

    /// CRITICAL: No data leaves the machine
    pub fn is_truly_local(&self) -> bool {
        match &self.engine {
            LLMEngine::Ollama { endpoint, .. } => {
                endpoint.starts_with("http://localhost") ||
                endpoint.starts_with("http://127.0.0.1")
            }
            LLMEngine::Llama { .. } => true,
        }
    }
}
```

#### User Privacy Controls

```rust
// src-tauri/src/ai/privacy.rs

#[derive(Debug, Serialize, Deserialize)]
pub struct AIPrivacySettings {
    // What AI can access
    pub can_read_conversations: bool,
    pub can_read_local_files: bool,
    pub can_access_clipboard: bool,
    pub can_access_window_titles: bool,

    // What AI can share
    pub can_suggest_connections: bool,
    pub share_interests_publicly: bool,

    // External access
    pub allow_web_search: bool,
    pub allow_cloud_models: bool,

    // Data retention
    pub conversation_history_days: u32,
    pub auto_delete_sensitive: bool,
}

impl Default for AIPrivacySettings {
    fn default() -> Self {
        Self {
            // Conservative defaults
            can_read_conversations: false,  // Opt-in
            can_read_local_files: false,
            can_access_clipboard: false,
            can_access_window_titles: false,

            can_suggest_connections: false,
            share_interests_publicly: false,

            allow_web_search: false,
            allow_cloud_models: false,

            conversation_history_days: 30,
            auto_delete_sensitive: true,
        }
    }
}
```

**Privacy Dashboard UI:**

```
┌────────────────────────────────────────────────┐
│ AI Privacy & Permissions                      │
├────────────────────────────────────────────────┤
│                                                │
│ 🔒 Data Access                                │
│ ☐ Allow AI to read my conversations          │
│ ☐ Allow AI to see my window titles           │
│ ☐ Allow AI to access clipboard               │
│                                                │
│ 🌐 Network Access                             │
│ ☐ Allow AI to search the web                 │
│ ☐ Allow cloud AI models (uses external API)  │
│                                                │
│ 👥 Social Features                            │
│ ☑ Allow AI to suggest connections            │
│ ☐ Share my interests publicly                │
│                                                │
│ 📊 Data Retention                             │
│ Delete conversation history after: [30] days │
│ ☑ Auto-delete messages marked sensitive      │
│                                                │
│ 🗑️ [Delete All AI Data]                      │
│                                                │
└────────────────────────────────────────────────┘
```

---

### 📊 AI Effectiveness Metrics

**How to measure if AI is actually helping:**

```rust
// src-tauri/src/ai/metrics.rs

#[derive(Debug)]
pub struct AIMetrics {
    // Connection metrics
    pub connections_suggested: u32,
    pub connections_accepted: u32,
    pub connections_led_to_collaboration: u32,

    // Conversation metrics
    pub conversations_archived: u32,
    pub searches_performed: u32,
    pub search_results_clicked: u32,

    // Debugging metrics
    pub debugging_sessions: u32,
    pub problems_solved: u32,
    pub average_session_length: Duration,

    // User satisfaction
    pub positive_feedback: u32,
    pub negative_feedback: u32,
    pub features_disabled: Vec<String>,
}

impl AIMetrics {
    pub fn connection_acceptance_rate(&self) -> f32 {
        if self.connections_suggested == 0 {
            return 0.0;
        }
        self.connections_accepted as f32 / self.connections_suggested as f32
    }

    pub fn collaboration_conversion_rate(&self) -> f32 {
        if self.connections_accepted == 0 {
            return 0.0;
        }
        self.connections_led_to_collaboration as f32 / self.connections_accepted as f32
    }

    pub fn user_satisfaction_score(&self) -> f32 {
        let total = self.positive_feedback + self.negative_feedback;
        if total == 0 {
            return 0.5;  // Neutral
        }
        self.positive_feedback as f32 / total as f32
    }
}

/// Collect metrics with user consent
pub async fn track_ai_effectiveness(
    event: AIEvent,
    metrics: Arc<Mutex<AIMetrics>>,
) {
    // Only track if user consented
    if !user_consented_to_metrics() {
        return;
    }

    let mut metrics = metrics.lock().await;

    match event {
        AIEvent::ConnectionSuggested => {
            metrics.connections_suggested += 1;
        }
        AIEvent::ConnectionAccepted => {
            metrics.connections_accepted += 1;
        }
        AIEvent::UserFeedback { positive } => {
            if positive {
                metrics.positive_feedback += 1;
            } else {
                metrics.negative_feedback += 1;
            }
        }
        // ... other events
    }
}
```

**Success Criteria:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| Connection acceptance rate | > 40% | Users accept suggested connections |
| Collaboration conversion | > 20% | Accepted connections lead to real collaboration |
| Search usefulness | > 60% | Users click on search results |
| User satisfaction | > 70% | Positive feedback ratio |
| Feature retention | < 20% | Users don't disable AI features |

---

### 🗓️ Implementation Phasing

**Don't build all AI features at once!** Roll out gradually:

#### Phase 1 (Month 6-7): Infrastructure AI
- [ ] Network Optimizer (background only)
- [ ] Spam Filter (essential for safety)
- [ ] Basic metrics collection

**Goal**: Prove AI can run invisibly and add value

#### Phase 2 (Month 8-9): First Assistant
- [ ] ConnectorBot only
- [ ] Basic interest matching
- [ ] User feedback system

**Goal**: Validate that people want AI-assisted discovery

#### Phase 3 (Month 10-11): Knowledge AI
- [ ] ArchivistBot
- [ ] ResearcherBot
- [ ] Local search only

**Goal**: Show AI can help with information management

#### Phase 4 (Month 12+): Social AI (Optional)
- [ ] CoWorking Buddy
- [ ] Rubber Duck Debugger
- [ ] Only if users request it

**Goal**: Experiment with AI companionship

---

### ⚠️ Critical Success Factors

**This only works if:**

1. **Transparency**: Users always know what's AI
2. **Utility**: Each AI agent solves a real problem
3. **Privacy**: Local-first, user controlled
4. **Optional**: Can disable entirely
5. **Not Creepy**: No fake emotions or manipulation

**This fails if:**

❌ AI pretends to be human
❌ Users feel surveilled
❌ AI is annoying or useless
❌ Privacy violations
❌ Forced upon users

---

### 💡 Honest Assessment

**Will this differentiate Mycelix?**

Maybe. The AI itself isn't unique - Discord has bots, Slack has AI assistants. **Your differentiator is:**

1. **Privacy-first AI** (local models, no surveillance)
2. **P2P + AI** (AI coordinating decentralized network)
3. **Cross-app presence** (AI helps across all tools)
4. **Transparent operation** (open source, explainable)

**The real value:**
- AI makes P2P network more discoverable (solves cold start)
- AI handles complexity (network optimization)
- AI reduces loneliness (optional companionship)

**But remember**: Mycelix's core value is **human connection**. AI should amplify that, never replace it.

---

## 🚀 Next Steps

1. **Week 1-2**: Implement NetworkOptimizer (invisible)
2. **Week 3-4**: Build metrics dashboard
3. **Week 5-8**: ConnectorBot (first user-facing AI)
4. **Week 9-12**: Measure effectiveness, iterate
5. **Month 4+**: Add more agents based on user feedback

**Start small, measure everything, be honest about limitations.**

---

*"AI should make humans more human, not less."*

---

## 📦 Distribution & Packaging

### NixOS Flake (Primary Distribution)

```nix
# flake.nix for users
{
  inputs.mycelix.url = "github:Luminous-Dynamics/mycelix-desktop";

  outputs = { self, mycelix }: {
    # Install via home-manager
    home.packages = [ mycelix.packages.x86_64-linux.default ];

    # Or system-wide
    environment.systemPackages = [ mycelix.packages.x86_64-linux.default ];
  };
}
```

### Other Platforms

1. **AppImage** (Linux universal)
   ```bash
   cargo tauri build --target appimage
   ```

2. **Flatpak** (Flathub)
   - Apply to Flathub
   - Requires manifest and metadata

3. **macOS** (.dmg)
   ```bash
   cargo tauri build --target dmg
   ```

4. **Windows** (.msi)
   ```bash
   cargo tauri build --target msi
   ```

---

## 💰 Sustainability Model

### Sacred Reciprocity License (SRL) v4.0

**Core Principles:**
1. **Free for individuals & small communities**
2. **Pay-what-you-can for larger organizations**
3. **Source available** (not fully open source)
4. **Community ownership path** (Regenerative Exit)

### Revenue Streams

1. **Donations** (primary)
   - GitHub Sponsors
   - Patreon
   - Cryptocurrency donations

2. **Bootstrap Servers** (optional paid hosting)
   - Free public servers (community-run)
   - Paid private servers for organizations
   - Pricing: $10-50/month per server

3. **Premium Features** (future)
   - Larger file transfers
   - Recording/transcription services
   - AI assistants integration
   - Organization management tools

4. **Consulting/Support**
   - Help organizations deploy
   - Custom integrations
   - Training workshops

**Year 1 Goal**: $2,000/month (covers your time + servers)
**Year 2 Goal**: $10,000/month (sustainable full-time)
**Year 3 Goal**: $30,000/month (hire 1-2 contributors)

---

## 🎯 Success Metrics

### Phase 0-1 (MVP)
- [ ] 10 beta testers using daily
- [ ] Successful P2P connection 95%+ of the time
- [ ] Zero crashes in 1 week of testing
- [ ] Basic features all functional

### Phase 2 (Network)
- [ ] 50 active users
- [ ] Network stays connected with 3+ nodes
- [ ] Message delivery 99%+ reliable
- [ ] Average session length > 30 minutes

### Phase 3 (Consciousness)
- [ ] Resonance matching finds relevant connections
- [ ] Users report it "feels magical"
- [ ] 20% of connections lead to collaboration
- [ ] Attention visualization provides value

### Phase 4 (Launch)
- [ ] 500+ users in first month
- [ ] Featured on Hacker News front page
- [ ] 1,000+ GitHub stars
- [ ] $500+/month in donations
- [ ] Zero security vulnerabilities found
- [ ] Average rating 4.5+/5 stars

---

## 🚨 Risk Management

### High-Risk Areas

1. **Holochain Stability**
   - Risk: Holochain bugs/breaking changes
   - Mitigation: Pin to stable version, contribute fixes upstream
   - Contingency: Fall back to libp2p if needed

2. **NAT Traversal**
   - Risk: Users behind restrictive firewalls can't connect
   - Mitigation: TURN relay servers
   - Contingency: Manual port forwarding guide

3. **Scalability**
   - Risk: Network bogs down with 1000+ nodes
   - Mitigation: DHT sharding, proximity clustering
   - Contingency: Limit network size initially

4. **User Adoption**
   - Risk: "Just another chat app" perception
   - Mitigation: Emphasize unique consciousness features
   - Contingency: Focus on niche (consciousness researchers, P2P enthusiasts)

5. **Funding**
   - Risk: Can't sustain development
   - Mitigation: Diversify revenue, keep costs low
   - Contingency: Open source fully, seek grants

### Mitigation Strategies

- Start small, iterate quickly
- Get feedback early and often
- Ship working software > perfect software
- Build community before features
- Document everything
- Test on real users, not just yourself

---

## 🛠️ Development Tools & Resources

### Essential Reading

1. **Tauri Documentation**
   - https://tauri.app/v1/guides/

2. **Holochain Documentation**
   - https://developer.holochain.org/

3. **Three.js Fundamentals**
   - https://threejs.org/manual/

4. **Rust Book**
   - https://doc.rust-lang.org/book/

### Code Examples

1. **Tauri + Holochain Example**
   - https://github.com/holochain-community/tauri-holochain-template

2. **P2P Network Visualization**
   - https://github.com/vasturiano/3d-force-graph

3. **WebRTC in Rust**
   - https://github.com/webrtc-rs/webrtc

### Community

1. **Holochain Forum**
   - https://forum.holochain.org/

2. **Tauri Discord**
   - https://discord.com/invite/tauri

3. **Rust Community**
   - https://www.rust-lang.org/community

---

## 📈 Next Steps

### This Week (Week 1)

1. **Technology Decision** (2 days)
   - Test Tauri with WebKitGTK
   - Test Holochain conductor
   - Create decision matrix
   - Choose tech stack

2. **Environment Setup** (2 days)
   - Create flake.nix for development
   - Set up CI/CD pipeline
   - Configure development tools
   - Test on NixOS

3. **Architecture Design** (3 days)
   - Draw complete system diagram
   - Design API contracts
   - Create data models
   - Write ADRs (Architecture Decision Records)

### This Month (Weeks 1-4)

- Complete Phase 0 (Foundation)
- Start Phase 1 (MVP)
- Have working Tauri app with Holochain
- First P2P message sent successfully

### This Quarter (Weeks 1-13)

- Complete Phase 1 (MVP)
- Complete Phase 2 (Network)
- Start Phase 3 (Consciousness)
- Beta testing with 20-50 users

---

## 🎉 Vision: 6 Months From Now

**You will have:**

✅ A production-ready desktop application
✅ Real P2P networking (no servers needed)
✅ Beautiful consciousness field visualization
✅ Active community of 500+ users
✅ Sustainable funding model
✅ Recognition in Rust/Holochain/NixOS communities
✅ Foundation for expanding ecosystem
✅ Path to full-time work on consciousness technology

**The world will have:**

✅ First truly consciousness-first P2P network
✅ Privacy-respecting collaboration tool
✅ Alternative to surveillance capitalism platforms
✅ Proof that sacred technology can be practical
✅ Open protocol others can build on

---

## 💭 Final Thoughts

This is **ambitious but achievable**. You have:

1. **Technical foundation**: NixOS expertise, Rust knowledge, existing Holochain work
2. **Unique vision**: Consciousness-first computing philosophy
3. **AI assistance**: Claude Code for rapid development
4. **Community**: Luminous Dynamics ecosystem
5. **Motivation**: Deep alignment with values

The key is **consistent progress**:
- 20-30 hours/week sustained effort
- Ship working software frequently
- Get feedback from real users
- Stay focused on core vision

**This could be the flagship Luminous Dynamics application** - the one that proves consciousness-first, P2P, sacred reciprocity principles can create software people actually want to use.

---

## 📞 Questions?

As you work through this roadmap, document:
- Blockers and solutions
- Architecture decisions
- Lessons learned
- Community feedback

Update this document as the vision evolves.

**Let's build something beautiful.** 🍄✨

---

*Created with love and consciousness*
*Luminous Dynamics*
*September 30, 2025*