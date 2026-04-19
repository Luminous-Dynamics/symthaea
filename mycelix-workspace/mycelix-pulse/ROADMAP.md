# Mycelix Mail — Paradigm Roadmap

## Vision

Mycelix Mail is not an email client. It's a **semantic trust network with message delivery** — where communication, economics, governance, health, and knowledge flow through a single encrypted, decentralized substrate.

---

## Current State (v0.1.0 — April 2026)

- **35 files, 7,000+ LOC** (Rust + CSS)
- **502KB gzipped WASM**, trunk-verified, SPA routing working
- **41 commodity features** at Gmail/Outlook parity (81%)
- **14 unique features** no competitor has
- **2 paradigm features** implemented (Trust-Gated Inbox, Revocable Email)
- **12 backend zomes**, 172 extern functions, 18MB hApp packed
- **Deployment ready**: mail.mycelix.net route configured, NixOS service created

---

## Phase 1: Convenience & Polish (v0.2.0)

Ship the live deployment and add the features that make daily use pleasant.

### Convenience Features
- [ ] **Recent Emails view** — Quick-access list of last 20 opened/interacted emails
- [ ] **Quick Reply inline** — Reply box at bottom of read view without navigating to compose
- [ ] **Email preview on hover** — Tooltip preview of email body on inbox hover
- [ ] **Pin conversations** — Pin important threads to top of inbox
- [ ] **Bulk label assignment** — Apply labels to multiple selected emails at once
- [ ] **Contact quick-actions** — One-click call/message from contact card
- [ ] **Unread filter toggle** — One-click "Show unread only" in inbox header
- [ ] **Mark all as read** — Folder-level action
- [ ] **Infinite scroll** — Load more emails on scroll (paginated zome calls)
- [ ] **Date separators** — "Today", "Yesterday", "This Week" headers in email list
- [ ] **Avatar colors** — Consistent color per sender (hash-based) instead of single teal

### Deployment
- [ ] Add mail-services.nix to NixOS imports, `sudo nixos-rebuild switch`
- [ ] Run deploy.sh — verify mail.mycelix.net loads
- [ ] Add Cloudflare DNS route if needed
- [ ] Test service worker offline caching in browser
- [ ] Test PWA install on Pixel 8 Pro

---

## Phase 2: Paradigm Features (v0.3.0 — v0.5.0)

Features that structurally cannot exist on centralized email.

### 2a: Economic Email (Priority 1)
Mycelix-finance integration. Every email can carry payments, invoices, bounties.

- [ ] **Inline payments** — "Send 50 TEND" button in compose, settled via finance cluster
- [ ] **One-tap invoices** — Sender creates invoice, recipient taps "Pay" → atomic settlement
- [ ] **Bounty emails** — "Offering 100 TEND for the first person who fixes the irrigation pump" — paid on claim verification
- [ ] **Dividend distribution** — Treasurer sends one email, 200 members each receive their share atomically
- [ ] **Payment receipts** — Automatic encrypted receipt as reply when payment settles
- **Cluster integration**: `CallTargetCell::OtherRole("finance")` → `payments`, `tend`, `staking` zomes
- **Complexity**: Medium — finance infrastructure exists

### 2b: Immune Memory (Priority 2)
Distributed threat learning via HDC semantic fingerprints.

- [ ] **Threat signature extraction** — When user flags spam, local Symthaea computes 16,384D binary fingerprint
- [ ] **Antibody DHT entries** — Anonymized threat patterns shared via DHT gossip
- [ ] **Preemptive matching** — New emails checked against antibody database (Hamming distance)
- [ ] **Trust-weighted confidence** — Higher-tier flaggers produce stronger antibodies
- [ ] **Tolerance mechanism** — Auto-decay antibodies that generate false positives
- **Cluster integration**: Symthaea-spore HDC encoder + bridge-common reputation
- **Complexity**: Medium — HDC infrastructure exists

### 2c: Verifiable Claims (Priority 3)
Machine-checkable assertions linked to the knowledge graph.

- [ ] **Claim tagging in compose** — Tag a sentence as a verifiable claim
- [ ] **Proof chain attachment** — Link claim to evidence in knowledge cluster
- [ ] **Recipient-side verification** — Local agent checks proof chain, renders checkmark/warning
- [ ] **Auto-detection** — Symthaea identifies claimable assertions in email body
- [ ] **Epistemic badges** — E-level (empirical confidence) rendered on each claim
- **Cluster integration**: `mycelix-knowledge` → `factcheck`, `claims`, `inference` zomes
- **Complexity**: Medium — knowledge infrastructure exists

### 2d: Temporal Sovereignty (Priority 4)
Messages as state machines with time-based transitions.

- [ ] **Dead-man's switch** — "If I don't send a heartbeat in 7 days, reveal this content"
- [ ] **Graduated reveal** — Content sections unlock over time
- [ ] **Auto-escalation** — Priority increases as response deadline approaches
- [ ] **Conditional redaction** — Sections redact based on external events (governance vote, etc.)
- **Cluster integration**: Existing revocable-mail pattern + linked entry chains
- **Complexity**: Medium — natural DHT state machine

### 2e: Ecological Awareness (Priority 5)
Attention budgets and communication cost visibility.

- [ ] **Complexity scoring** — Local Symthaea estimates reading time, decision burden, emotional weight
- [ ] **Attention budget** — Configurable daily limit (complexity-minutes), enforced locally
- [ ] **Sender visibility** — See recipient's remaining budget before sending
- [ ] **Budget-aware sorting** — Inbox sorted by signal-to-cost ratio
- [ ] **Reduced-footprint mode** — Compose hints to keep message under recipient's threshold
- **Cluster integration**: Extends MATL staking + local Symthaea analysis
- **Complexity**: Medium

---

## Phase 3: Collective Intelligence (v0.6.0)

Features that require Symthaea-spore WASM kernel running in-browser.

### 3a: Epistemic Inbox (Semantic Clustering)
- [ ] **HDC encoding** — Incoming emails mapped to 16,384D semantic space in-browser
- [ ] **Semantic clusters** — Emails grouped by meaning instead of chronology
- [ ] **Epistemic classification** — E/N/M/H axis scoring on incoming claims
- [ ] **Cognitive load indicators** — Per-email complexity assessment before opening
- **Dependency**: Symthaea-spore WASM kernel integration

### 3b: Group Sense-Making
- [ ] **Structured reply types** — Assertion, question, proposal, evidence tags on replies
- [ ] **Thread map visualization** — Graph view of positions and agreements
- [ ] **Convergence scoring** — Measure when a thread reaches consensus
- [ ] **Formalize-to-governance** — One-click push thread consensus to governance proposal
- **Dependency**: Symthaea-spore + mycelix-governance integration

### 3c: Symthaea Secretary
- [ ] **Pre-analysis** — Cognitive load score and "Action Required" tag before opening
- [ ] **Smart scheduling** — Cross-reference calendar, auto-negotiate meeting times
- [ ] **Consciousness-coupled replies** — Broca generates drafts matching sender's tone
- [ ] **Harmony scoring** — Eight Harmonies ethical alignment on incoming messages
- **Dependency**: Full Symthaea-spore with Broca language model

---

## Phase 4: Infrastructure Paradigms (v0.7.0+)

### 4a: Programmable Email (Mail-lets)
- [ ] **Sandboxed WASM contracts** — Emails carrying executable logic
- [ ] **Conditional delivery** — "Deliver when recipient signs with DID"
- [ ] **Cross-cluster triggers** — Mail-let invokes governance/finance/commons atomically
- [ ] **Contract DSL** — Restricted Rust subset for mail-let authoring
- **Complexity**: High — requires WASM sandbox within WASM sandbox

### 4b: Anti-Forward CRDTs
- [ ] **Shared thread objects** — Email thread as a live CRDT, not duplicated text
- [ ] **Editable action items** — Collaborative action-item block at thread top
- [ ] **Real-time presence** — See who's viewing the thread
- **Dependency**: Sync zome CRDT + WASM operational transform

### 4c: Emergent Organization (Thread-to-DAO)
- [ ] **Structure detection** — Identify when thread has working-group characteristics
- [ ] **Formalization wizard** — One-click create governance + treasury + resources
- [ ] **Auto-routing** — Future thread emails routed to working group inbox
- **Complexity**: High — multi-cluster atomic orchestration

### 4d: Interspecies Communication
- [ ] **IoT agent identities** — Sensors get DIDs and trust tiers
- [ ] **Machine message rendering** — Distinct visual treatment for non-human senders
- [ ] **Sensor-to-thread bridging** — IoT data feeds participate in sense-making threads
- [ ] **Attention-budget aware** — Machine messages don't drain human budgets
- **Dependency**: IoT bridge infrastructure, lightweight Holochain agents

### 4e: Spatial/Contextual Email
- [ ] **Location-bound delivery** — "Deliver to whoever is at the community garden Saturday"
- [ ] **Condition predicates** — "Deliver when soil moisture < 20%"
- [ ] **Event triggers** — "Activate during next council meeting"
- **Dependency**: mycelix-position cluster maturity, IoT bridges

---

## Cross-Cluster Integration Map

```
                    ┌─────────────┐
                    │  MYCELIX    │
                    │    MAIL     │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │  FINANCE    │ │ GOVERNANCE  │ │  IDENTITY   │
    │             │ │             │ │             │
    │ Payments    │ │ Proposals   │ │ DIDs        │
    │ TEND stake  │ │ Voting      │ │ Credentials │
    │ Invoices    │ │ Councils    │ │ Trust/WoT   │
    │ Treasury    │ │ Budgets     │ │ MFA         │
    └─────────────┘ └─────────────┘ └─────────────┘
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │  COMMONS    │ │  KNOWLEDGE  │ │   HEALTH    │
    │             │ │             │ │             │
    │ Calendar    │ │ Fact-check  │ │ FHIR data   │
    │ Mutual aid  │ │ Claims      │ │ Consent     │
    │ Resources   │ │ Inference   │ │ Rx refills  │
    │ Care plans  │ │ Graph       │ │ Telehealth  │
    └─────────────┘ └─────────────┘ └─────────────┘
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │   CIVIC     │ │   MUSIC     │ │  SYMTHAEA   │
    │             │ │             │ │   SPORE     │
    │ Emergency   │ │ Royalties   │ │             │
    │ Justice     │ │ Attribution │ │ HDC encode  │
    │ Media       │ │ Streaming   │ │ Sentiment   │
    │ Resonance   │ │ Discovery   │ │ Broca gen   │
    └─────────────┘ └─────────────┘ └─────────────┘
```

**Every arrow is a `CallTargetCell::OtherRole()` dispatch through bridge-common's routing_registry.**

---

## Unique Feature Inventory (Current + Planned)

### Implemented (14)
1. Post-quantum encryption (Kyber1024 + Dilithium3)
2. Zero-knowledge DHT storage
3. Trust-gated inbox (TEND staking)
4. Revocable email (key destruction)
5. MATL trust scoring with Byzantine detection
6. CRDT offline sync (vector clocks)
7. Federation bridges (SMTP, Matrix, ActivityPub)
8. Immutable audit trail
9. Data sovereignty (no server)
10. Sender verification (Ed25519/Dilithium signatures)
11. Trust-based spam prevention
12. Content-addressed attachments (SHA-256 chunks)
13. Key rotation with automatic refresh
14. Consciousness-gated access tiers

### Planned (16)
15. Economic email (inline payments, invoices, bounties)
16. Immune memory (distributed HDC threat learning)
17. Verifiable claims (machine-checkable assertions)
18. Temporal sovereignty (self-modifying state machines)
19. Ecological awareness (attention budgets)
20. Epistemic inbox (semantic clustering)
21. Group sense-making (convergence scoring)
22. Symthaea secretary (pre-analysis, smart scheduling)
23. Programmable email (WASM mail-lets)
24. Anti-forward CRDTs (shared thread objects)
25. Emergent organization (thread-to-DAO)
26. Interspecies communication (IoT agents)
27. Spatial/contextual delivery (location-bound)
28. FHIR-compliant health notifications
29. Governance inline voting
30. Cross-cluster knowledge enrichment

**Total: 30 unique features. Gmail has 0. Outlook has 0.**

---

## On-Device Intelligence: "Symthaea Edge"

**Principle:** no Big Tech model weights ship with Mycelix. The on-device
intelligence layer — Athena — runs on Symthaea's own cognitive
architecture (HDC binding + CfC dynamics + Broca decoder), not a
quantized Gemma/Llama/Mistral drop-in. "Sovereign suite" means
sovereign all the way down, including the weights.

**Surface area for Pulse v0.x:** calendar NER (Quick Add extraction
from natural-language utterances), recipient auto-complete, intent
classification for inbox routing. None of this needs a generalized
LLM — it needs a fast, rigid entity extractor with a closed
output grammar.

**Bundle target:**
- ~50MB total on-disk footprint (checkpoint + kg + code)
- ~80-120MB RAM resident (mmap'd checkpoint, lazy CfC init)
- <50ms cold start (measured, not claimed)
- Inference on `tokio::task::spawn_blocking`, UI stays 60fps

**Architecture sketch** (`crates/symthaea-edge-calendar/`, not yet built):

```
symthaea-edge-calendar/
├── src/
│   ├── lib.rs              # pub: extract_intent(text) -> CalendarEvent
│   ├── kg/
│   │   ├── intents.bin     # ~200 retrieval blocks, rkyv-serialized
│   │   └── contacts.rs     # user-local contacts index, loaded at boot
│   ├── broca_lite.rs       # 4-channel encoder + 4,096D HDC binding
│   └── decode.rs           # autoregressive head → structured output
├── checkpoints/
│   └── broca-calendar-v1.safetensors  # ~20MB Q8 quantized
└── tests/
    ├── in_distribution.rs
    └── holdout.rs          # honest generalization eval — ship-gate
```

**Parameter reduction from research-build Broca:**
- 43-channel ThoughtEncoder → 4-channel (time, intent, attendee, title).
  Epistemic Cube (E/N/M/H axes) dropped — irrelevant for calendar NER.
- 16,384D HDC → 4,096D. Calendar-intent grammar is small enough.
- Vocabulary = calendar lexicon + personal contacts only.

**Ship-gate (from Nix-codegen lesson, `symthaea_coding_ai_phase1_p2_m5`):**
Holdout evaluation ≥90% PASS on 100+ prompts the model never saw during
training. The Apr 19 Nix work demonstrated that in-distribution wins
(12/13 with retrieval) can collapse to 0/9 on true holdout. Apply the
same FNV-bucket split discipline here. Do not ship the edge bundle
without a published holdout number.

**Risk to validate before committing to the architecture:**
The Nix codegen worked in part because `nix-instantiate` gave the
scorer compiler ground-truth. Calendar NER has no equivalent oracle
(intent extraction is judgment, not compilation). Prototype the
evaluation harness before locking the architecture — if a retrieval-
only approach without compiler ground-truth doesn't generalize, the
answer may be a different shape (e.g., small distilled transformer
trained on Mycelix-owned NER corpus, still 100% internal weights).

**Cryptographic model provenance** (the marketing flex):
Larger optional models — if we ever add them for heavier inference —
distribute via IPFS CIDs, pinned by the attribution cluster. Users
can audit the cryptographic hash of the exact weights loaded on
their device. Provenance is a DHT fact, not a "trust us" claim.
Competitors cannot tell this story because their models live behind
closed serving infrastructure.

**Not a replacement for the research-build Symthaea.** The full
Broca + Epistemic Cube + full HDC stack continues to live in the
main Symthaea workspace as the sovereignty-research platform.
Symthaea Edge is a *ship target* — a stripped, calendar-specific
artifact derived from that platform. The research build improves;
the edge build re-distills periodically.

---

## Principles

1. **Local-first** — All computation happens on the user's device. No cloud AI. No Big Tech model weights.
2. **Economic alignment** — Communication has cost. Attention is valued.
3. **Trust over filters** — Humans decide trust; machines enforce it.
4. **Sovereignty** — Your keys, your data, your rules. Revocable forever.
5. **Emergence** — Let organization arise from communication, not the reverse.
6. **Ecological** — Make the true cost of communication visible.
7. **Provenance is a fact, not a claim** — Model weights, message signatures, identity — all cryptographically verifiable on-device.

---

*"Email is not a feed. It's a living network."*
