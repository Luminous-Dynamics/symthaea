# Mycelix Workspace — Unified Communication Platform Plan

## Vision

**Mycelix Workspace** is not "decentralized Teams." It's the first communication platform where messages, payments, governance, knowledge, and organization share a single encrypted substrate. Organizations don't use it — they emerge from it.

---

## What Already Exists (Infrastructure Audit)

### Production-Ready (Reuse Directly)

| Component | Location | What It Does |
|-----------|----------|-------------|
| **Mail messages zome** | `mycelix-mail/holochain/zomes/messages/` | 7 real-time signal types, typing indicators, delivery/read receipts, threading, PQC encryption |
| **CRDT sync zome** | `mycelix-mail/holochain/zomes/sync/` | Vector clocks, offline-first queue, conflict resolution, 9 sync signal types, peer coordination |
| **WebRTC P2P** | `mycelix-mail/holochain/client/webrtc/SignalingService.ts` | RTCPeerConnection, data channels, ICE handling, Holochain-based signaling (no central server), 50 peers max |
| **Emergency channels** | `mycelix-civic/zomes/emergency-comms/` | Group channels, priority levels, TTL/expiration, offline-first, mesh broadcasting, location targeting |
| **Peer management** | `mycelix-workspace/sdk-ts/rtc/peer.ts` | Full audio/video stream handling, data channels, connection stats, 48kHz audio worklets |
| **Resonance feed** | `mycelix-civic/zomes/resonance-feed/` | Content publishing, voting/reactions, trending algorithm, domain filtering |
| **All 134 Mycelix zomes** | 16 clusters via bridge-common | Finance, governance, commons, health, identity, knowledge — all accessible via `CallTargetCell::OtherRole` |

### Partial (Needs Adaptation)

| Component | Location | Adaptation Needed |
|-----------|----------|-------------------|
| **WebSocket gateway** | `mycelix-music/realtime/realtime-infrastructure.ts` | Generalize from music-specific to universal workspace gateway. Room types exist (chat, jam, listen, broadcast). |
| **Presence system** | `mycelix-music/realtime/` | Extract from music context. Already has: activity monitoring, last-seen, typing, awareness broadcast. |
| **Push notifications** | `mycelix-music/realtime/` | Already supports FCM, APNs, Web Push with dedup. Needs workspace-specific notification rules. |
| **Collaborative editing** | `mycelix-music/realtime/` | Has OT/CRDT-based change propagation. Needs document schema for workspace notes/agendas. |

### Missing (Must Build)

| Component | Priority | Complexity |
|-----------|----------|-----------|
| **Chat UI (Leptos)** | P1 | Medium — new pages in existing frontend |
| **Channel management zome** | P1 | Medium — adapt emergency-comms |
| **Message editing/deletion** | P1 | Low — tombstone entries in DHT |
| **@mentions** | P1 | Low — link types + notification routing |
| **Reactions/emoji** | P1 | Low — adapt resonance votes |
| **Video call UI** | P2 | Medium — wire existing WebRTC peer management |
| **Screen sharing** | P2 | Medium — browser `getDisplayMedia()` API |
| **SFU for group calls** | P3 | High — mesh topology for 3+ participants |
| **Bot/integration framework** | P3 | Medium — agent-based with DID identity |

---

## Architecture

### Unified Communication Model

```
┌──────────────────────────────────────────────────────┐
│                  MYCELIX WORKSPACE                    │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │   MAIL   │  │   CHAT   │  │   MEET   │           │
│  │ (async)  │  │ (real-   │  │ (video/  │           │
│  │          │  │  time)   │  │  audio)  │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │                 │
│  ┌────▼──────────────▼──────────────▼─────┐          │
│  │         UNIFIED MESSAGE LAYER           │          │
│  │                                         │          │
│  │  Holochain DHT ← E2E/PQC Encrypted     │          │
│  │  CRDT Sync ← Offline-First             │          │
│  │  WebRTC ← P2P Data/Media Channels      │          │
│  │  Trust Gate ← MATL + TEND Staking       │          │
│  └────┬──────────────┬──────────────┬─────┘          │
│       │              │              │                 │
│  ┌────▼────┐  ┌──────▼─────┐  ┌────▼────┐           │
│  │ Finance │  │ Governance │  │Knowledge│           │
│  │ Payments│  │ Voting     │  │ Facts   │           │
│  │ Bounties│  │ Proposals  │  │ Claims  │           │
│  └─────────┘  └────────────┘  └─────────┘           │
└──────────────────────────────────────────────────────┘
```

### Message Type Unification

Every communication is a single DHT entry type with a `mode` field:

```rust
enum MessageMode {
    /// Traditional email — async, signed, encrypted
    Email {
        subject: String,
        priority: Priority,
        read_receipt_requested: bool,
    },
    /// Instant message — real-time, lightweight
    Chat {
        reply_to: Option<ActionHash>,
        reactions: Vec<Reaction>,
        edited: bool,
    },
    /// Channel post — public or group, feed-style
    Post {
        channel_id: String,
        pinned: bool,
        mentions: Vec<AgentPubKey>,
    },
    /// Call event — metadata for voice/video
    Call {
        call_type: CallType, // Audio, Video, Screen
        participants: Vec<AgentPubKey>,
        duration_secs: Option<u32>,
        recording_hash: Option<EntryHash>,
    },
    /// System event — governance, finance, health notifications
    System {
        source_cluster: String,
        action_required: bool,
    },
}
```

This means: **the same zome, the same encryption, the same trust graph, the same search index** handles email, chat, calls, and system notifications. No data silos.

---

## Frontend Extension Plan

### New Pages (within existing Leptos frontend)

```
mycelix-mail/apps/leptos/src/pages/
  inbox.rs          ← existing (email + unified inbox view)
  compose.rs        ← existing (email compose + quick chat)
  read.rs           ← existing (email thread + chat thread)
  chat.rs           ← NEW: 1:1 and group chat
  channels.rs       ← NEW: persistent topic channels
  meet.rs           ← NEW: video/audio calls
  workspace.rs      ← NEW: unified view (sidebar: spaces)
```

### Sidebar Evolution

Current:
```
📥 Inbox (3)
📤 Sent
📝 Drafts (1)
⭐ Starred
📦 Archive
⚠ Spam
🗑 Trash
```

Workspace:
```
── MAIL ──
📥 Inbox (3)
📤 Sent
📝 Drafts (1)

── CHATS ──
💬 Alice Nakamura (2)
💬 Bob Mthembu
👥 Water Council (5)
👥 Engineering

── CHANNELS ──
# governance
# engineering
# education
# announcements

── MEET ──
🎥 Start Call
📅 Scheduled: Council Thu 2pm

── SPACES ──
🏗 Tool Library (DAO)
🌊 Water Stewardship
```

### Chat UI Components

```rust
// New components needed
ChatView         — split-pane: contact list + message stream
MessageBubble    — sender-aligned bubble with avatar, timestamp, reactions
ReactionPicker   — emoji grid triggered by hover/long-press
MentionPopup     — @-mention autocomplete from contacts
ChannelHeader    — channel name, member count, pinned messages, call button
CallControls     — mute, camera, screen share, hang up, participants
PresenceBar      — who's online in this channel/chat
ThreadPanel      — right-side panel for threaded replies within chat
```

---

## Implementation Phases

### Phase 1: Chat Foundation (v0.3.0)

**Goal**: 1:1 and group chat alongside email in the same app.

**Backend:**
- [ ] Add `ChatMessage` entry type to messages zome (or adapt existing EncryptedEmail with `mode: Chat`)
- [ ] Add `Channel` entry type (adapt from emergency-comms channels)
- [ ] Add reactions (compact DHT entries linked to messages)
- [ ] Add message editing (new entry linked to original, tombstone pattern)
- [ ] Add @mention links (AgentPubKey → mentioned message)
- [ ] Wire WebRTC data channel for sub-second message delivery

**Frontend:**
- [ ] `chat.rs` — Split-pane chat view: contact/channel list + message stream
- [ ] `MessageBubble` component — alignment (self=right, other=left), avatar, timestamp
- [ ] Real-time message rendering via WebSocket signal subscription
- [ ] `ReactionPicker` — emoji reactions on messages
- [ ] `PresenceBar` — online/offline/typing indicators for chat contacts
- [ ] Sidebar: add "Chats" section with unread badges
- [ ] Unified notification: new chat messages + new emails in one toast system

**Integration:**
- [ ] Wire `music/realtime-infrastructure` presence system into mail frontend
- [ ] Connect mail zome signals → WebSocket gateway → Leptos reactive signals
- [ ] Chat messages searchable via same search zome (BM25)

### Phase 2: Channels & Threads (v0.4.0)

**Goal**: Persistent topic channels (like Slack channels) with threaded replies.

**Backend:**
- [ ] Channel management: create, archive, invite, permissions
- [ ] Channel membership with consciousness-gated access tiers
- [ ] Pinned messages per channel
- [ ] Channel-level notification preferences
- [ ] Thread replies within channels (reply → opens side thread)
- [ ] Channel discovery (public channels browsable by topic)

**Frontend:**
- [ ] `channels.rs` — Channel list + message feed + thread panel
- [ ] `ChannelHeader` — name, description, member count, settings, call button
- [ ] `ThreadPanel` — right-side sliding panel for threaded replies
- [ ] Channel creation wizard (name, description, members, access tier)
- [ ] Channel settings (notifications, pinned messages, integrations)
- [ ] Sidebar: add "Channels" section with # prefix

**Cross-Cluster:**
- [ ] Governance channel: inline proposal voting within channel messages
- [ ] Finance channel: inline payment/bounty widgets
- [ ] Knowledge channel: auto fact-checking of claims in messages

### Phase 3: Video & Audio Calls (v0.5.0)

**Goal**: P2P encrypted video/audio calls, screen sharing.

**Backend:**
- [ ] Call initiation/acceptance signaling via mail zome
- [ ] Call metadata entries (participants, duration, type)
- [ ] Call recording option (encrypted, stored on DHT)
- [ ] Meeting scheduling (bridge to commons calendar zome)

**Frontend:**
- [ ] `meet.rs` — Call interface: video grid, controls, participant list
- [ ] `CallControls` — mute, camera toggle, screen share, end call, raise hand
- [ ] 1:1 calls: direct WebRTC peer connection (existing infrastructure)
- [ ] Group calls (3-6 participants): mesh WebRTC topology
- [ ] Screen sharing via `getDisplayMedia()` + WebRTC video track
- [ ] Call from chat: "Start Call" button in chat/channel header
- [ ] Incoming call notification: ring + accept/decline
- [ ] Scheduled meetings: calendar integration via commons

**Reuse:**
- [ ] `mycelix-workspace/sdk-ts/rtc/peer.ts` — audio/video stream management
- [ ] `mycelix-music/apps/web/lib/collaboration/webrtc.ts` — peer-to-peer connection
- [ ] Holochain signaling (`wss://dev-test-bootstrap2.holochain.org/`) — no TURN server needed for WebRTC

### Phase 4: Workspace View & Emergence (v0.6.0)

**Goal**: Unified workspace where mail, chat, calls, governance, and finance coexist.

- [ ] `workspace.rs` — Split layout: spaces sidebar + content area + detail panel
- [ ] Spaces = working groups that emerged from threads (thread-to-DAO)
- [ ] Each space has: chat, files, governance, treasury, member list
- [ ] Cross-cluster activity feed: governance votes, payments, health alerts, all in one stream
- [ ] Semantic clustering via Symthaea Spore: messages auto-organized by meaning
- [ ] Attention budgets: per-space notification throttling
- [ ] Bot agents: non-human participants with DIDs (IoT sensors, AI assistants)

### Phase 5: Enterprise & Scale (v0.7.0+)

- [ ] SFU (Selective Forwarding Unit) for large group calls (7+ participants)
- [ ] Meeting transcription via local Whisper model (privacy-preserving)
- [ ] Compliance dashboard: audit trail, data retention policies, export
- [ ] SSO bridge: SAML/OIDC federation for enterprise identity
- [ ] Admin panel: user management, policy enforcement, analytics
- [ ] Mobile native (Tauri + Soma bridge)
- [ ] Desktop native (Tauri with system tray, native notifications)

---

## Competitive Positioning

### What Teams/Slack Cannot Do

| Capability | Teams/Slack | Mycelix Workspace |
|-----------|------------|-------------------|
| E2E encryption by default | No (server reads all) | Yes (PQC) |
| User owns their data | No (Microsoft/Salesforce owns it) | Yes (Holochain DHT) |
| Admin can read DMs | Yes | No (cryptographically impossible) |
| Works without internet | Limited | Full offline with CRDT sync |
| Inline payments | No | Yes (TEND tokens) |
| Inline governance | No | Yes (proposals, voting) |
| Trust-based spam prevention | No | Yes (MATL) |
| Thread → organization | No | Yes (emergent DAOs) |
| Semantic AI (local) | No (cloud AI) | Yes (Symthaea in WASM) |
| Interoperability | Walled garden | Federation (SMTP, Matrix, ActivityPub) |
| Censorship resistant | No | Yes (no central server) |
| Data sovereignty | Vendor lock-in | Self-sovereign |

### Target Markets

1. **Privacy-conscious organizations** — NGOs, journalism, activism, legal, healthcare
2. **Cooperatives & DAOs** — organizations that need governance + communication unified
3. **Developing world** — offline-first, mesh-capable, no cloud dependency
4. **Regulated industries** — GDPR/HIPAA compliance through zero-knowledge architecture
5. **Post-Web2 communities** — Holochain/crypto ecosystem, decentralization advocates

### The "Gmail Moment"

Gmail didn't win on features. It won on **one paradigm shift**: search instead of folders. Everything else (threaded conversations, labels, 1GB storage) reinforced that single insight.

Mycelix Workspace's "Gmail moment" is: **your communication IS your organization**. You don't chat about work and then go do the work somewhere else. The chat becomes the decision, the decision becomes the treasury allocation, the treasury allocation becomes the payment — all in one encrypted, trust-verified, sovereign flow.

---

## Naming

**Mycelix Workspace** is the working title. Alternatives:

- **Mycelix Pulse** — communication as the heartbeat of the organism
- **Mycelix Weave** — threads that weave into organizational fabric  
- **Mycelix Commons** — shared communication space (but conflicts with existing cluster)
- **Mycelix Hub** — central nervous system

Recommendation: **Mycelix Pulse** — it captures the real-time, living nature of the platform. Email is a message. Chat is a pulse. Organizations emerge from the rhythm.

---

## Technical Notes

### Why Not a Separate App?

A separate frontend would mean:
- Duplicate authentication flow
- Separate conductor connection
- No shared context (reading an email, then switching to chat about it = context loss)
- Two WASM bundles to download

By extending the existing mail frontend:
- Same conductor WebSocket (ws://localhost:8888)
- Same trust graph, same encryption keys
- Seamless transition: read an email → reply in chat → start a call
- One WASM bundle (~600KB gzipped)
- One PWA install

### WebRTC Topology

- **1:1 calls**: Direct peer connection (existing infrastructure)
- **3-6 participants**: Full mesh (each peer connects to all others)
- **7+ participants**: Need SFU (Selective Forwarding Unit) — this is Phase 5
- **Signaling**: Via Holochain's bootstrap/signal server — no TURN needed for most NAT configurations

### Performance Targets

- Chat message delivery: <200ms (WebRTC data channel, not DHT)
- DHT persistence: <2s (eventual consistency, not blocking UI)
- Video call setup: <3s (WebRTC offer/answer via Holochain signal)
- Presence update: <1s (WebSocket heartbeat)
- Offline queue flush: <5s after reconnect (CRDT merge)

---

*"Communication is not a tool you use. It's the substrate you live in."*
