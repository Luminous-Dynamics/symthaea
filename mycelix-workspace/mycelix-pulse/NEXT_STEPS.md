# Mycelix Pulse — Next Steps Plan

## Current State (April 5, 2026)

| Metric | Value |
|--------|-------|
| LOC | 10,400+ (7,400 Rust + 2,600 CSS + 400 HTML/JS) |
| WASM | **506KB gzipped** (was 669KB — 24% reduction via wasm-opt + strip + panic=abort) |
| Pages | 12 + landing page |
| Compile errors | 0 |
| TODOs | 0 |
| Panics | 0 |
| Live URL | https://mail.mycelix.net |
| Conductor | 6 hApps installed (mail, finance, governance, identity, commons, hearth) |
| Services | NixOS persistent (mail-conductor + mail-spa) |
| Performance | 453ms DOM ready, 3.1MB heap, 2,559 nodes |

---

## Phase 1: Symthaea Integration (Highest Value)

### The Triple Encoder Stack

**Why**: No other email client has on-device semantic understanding. This is the differentiator that makes Mycelix Pulse genuinely unique — not just "decentralized Gmail" but "email with a brain."

**What to build** (~300KB WASM addition):

#### 1a. Semantic Email Search
- Import `TextEncoder` from symthaea-core
- On email load: encode each email body → 16,384D hypervector
- Store HVs in IndexedDB (2KB per email)
- Search: encode query → Hamming similarity against all stored HVs
- Show results ranked by semantic similarity, not just keyword match
- **Impact**: "Find emails about the water proposal" works even if the word "water" isn't in the subject

#### 1b. Auto-Thread Clustering
- Compute pairwise similarity between email HVs
- Group emails with similarity > 0.65 into semantic clusters
- Display as "Semantic Clusters" view alongside Threaded/Flat toggles
- Clusters get auto-generated labels via dominant term extraction
- **Impact**: Emails organize themselves by meaning, not chronology

#### 1c. Smart Reply Suggestions
- Use `generate_text_with_input(email_body, 50)` via Broca
- Generate 3 short reply options (like Gmail's Smart Reply)
- Display below email in read view
- One-click to insert into compose
- **Impact**: Saves 30 seconds per reply, entirely on-device

#### 1d. Ethical Content Flags
- Run `MoralParser.parse(email_body)` on incoming messages
- Detect consent violations, harm language, manipulation patterns
- Show subtle flag icon on email cards for flagged content
- Expandable detail: "This message requests action without clear consent"
- **Impact**: Trust layer that goes beyond spam — detects social engineering

### Build Approach
```toml
# Add to mail frontend Cargo.toml
symthaea-core = { path = "../../../../symthaea/symthaea-core", default-features = false, features = ["hdc"] }
```

Only pull in the HDC module — not the full consciousness engine, dream engine, or robotics. The TextEncoder, BinaryHV, and MoralParser are self-contained.

---

## Phase 2: Onboarding Flow

### First-Visit Experience

Currently: User hits mail.mycelix.net → mock data → amber banner → confusion.

**Build a welcome modal** that appears on first visit (check localStorage):

```
┌─────────────────────────────────────────┐
│  Welcome to Mycelix Pulse               │
│                                         │
│  Your communication, encrypted by you.  │
│                                         │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │  Explore     │  │  Connect        │   │
│  │  Demo Mode   │  │  Conductor      │   │
│  │              │  │                 │   │
│  │  Try all     │  │  Run your own   │   │
│  │  features    │  │  Holochain      │   │
│  │  with sample │  │  node for real  │   │
│  │  data        │  │  encrypted      │   │
│  │              │  │  messaging      │   │
│  └─────────────┘  └─────────────────┘   │
│                                         │
│  [Learn more about Holochain →]         │
└─────────────────────────────────────────┘
```

**Demo mode enhancements**:
- Make compose actually "send" mock emails (add to local inbox)
- Make star/archive/delete persist in localStorage
- Make contact creation persist in localStorage
- Show "This is demo mode" watermark subtly, not a blocking banner

**Connect mode**:
- Guide user to install Holochain
- Detect local conductor automatically
- Show step-by-step: install → start conductor → install hApp → connected

---

## Phase 3: Quality & Polish

### Performance Optimizations (already applied)
- [x] wasm-opt -Oz (46% raw reduction, 24% gzip reduction)
- [x] strip = true (debug symbols removed)
- [x] panic = "abort" (unwinding machinery removed)
- [ ] Audit unused web-sys features
- [ ] Cloudflare Brotli verification (should be automatic)

### Reliability
- [ ] Retry loop for conductor connection (currently fires once after 300ms)
- [ ] Connection recovery (auto-reconnect on WebSocket drop)
- [ ] Error boundary components (catch WASM panics gracefully)
- [ ] Offline persistence (star/archive state in localStorage, sync on reconnect)

### Accessibility
- [ ] Full keyboard navigation for all interactive elements
- [ ] Screen reader testing (NVDA/VoiceOver)
- [ ] Focus management on page transitions
- [ ] Color contrast audit (WCAG 2.1 AA)

---

## Phase 4: Adoption Strategy

### Who uses it first?

1. **You (Tristan)** — Daily driver on local conductor. Prove it works for one person.
2. **Luminous Dynamics team** — 2-5 people on the same DHT network. Prove multi-agent messaging.
3. **Mycelix community** — Cooperatives, DAOs, privacy-conscious orgs.
4. **Public** — mail.mycelix.net as the demo, Holochain installer for real use.

### What drives switching?

Not features. Not ideology. **One workflow that's better than Gmail.**

The most promising candidate: **"Find that email about X"** with semantic search.

Gmail search is keyword-based. If you search "budget proposal" but the email said "financial plan for Q3", Gmail misses it. Symthaea's HDC semantic search finds it because the meaning is similar even though the words aren't.

If semantic search works well, it's the single feature that makes someone say "I'll check Mycelix first."

---

## Decision Framework

When choosing what to build next, ask:

1. **Does it require the conductor?** If yes, it only works for users who've installed Holochain. Prioritize features that work in demo mode too.
2. **Does it differentiate from Gmail?** If it's something Gmail already does, we're competing on their turf. Build things they structurally cannot.
3. **Can one person test it?** Features that need multiple users (chat, calls) are harder to validate than single-user features (search, calendar, compose).
4. **Does it make the demo impressive?** First impressions matter. Features visible in the first 30 seconds of demo mode are worth more than features behind 3 clicks.

By this framework: **Semantic search is #1** (works in demo, differentiates from Gmail, single-user testable, visible immediately in the inbox).

---

*The goal is not to build everything. It's to build the one thing that makes someone tell a friend.*
