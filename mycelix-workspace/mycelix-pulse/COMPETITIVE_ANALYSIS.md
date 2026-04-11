# Mycelix Mail — Competitive Feature Analysis

**Last updated**: 2026-04-04

## Feature Parity Matrix

Legend: **Y** = Yes | **P** = Partial | **N** = No | **U** = Unique (no competitor has it)

### Core Email

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Rich text compose (bold, italic, lists) | Y | Y | **N** | BUILD |
| Inline images in compose | Y | Y | **N** | BUILD |
| Formatting toolbar | Y | Y | **N** | BUILD |
| Email templates | Y (Labs) | Y | **N** | BUILD |
| Multiple signatures | Y | Y | **N** | BUILD |
| Conversation/thread view | Y | Y | **Y** | - |
| Read receipts | Y (paid) | Y | **Y** | - |
| Scheduled send | Y | Y | **Y** (zome ready) | WIRE |
| Undo send (delay) | Y (30s) | Y (10s) | **N** | BUILD |
| Email recall | N | Y (Exchange) | **N** | N/A (DHT immutable) |
| Priority inbox / smart sorting | Y | Y (Focused) | **N** | BUILD |
| Snooze | Y | Y | **N** | BUILD |
| Nudges ("haven't replied") | Y | Y (MyAnalytics) | **N** | BUILD |
| Vacation auto-responder | Y | Y | **N** | BUILD |
| Email aliases | Y | Y | **N** | PLAN |
| Inline reply | Y | Y | **N** | BUILD |

### Organization

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Labels/tags (multiple per email) | Y | Y (categories) | **N** | BUILD |
| Folders (hierarchical) | N (labels only) | Y | **Y** | - |
| Auto-filters/rules | Y | Y | **N** | BUILD |
| Stars/flags (multiple types) | Y (12 types) | Y (categories+flags) | **Y** (1 type) | EXPAND |
| Tabs (Primary/Social/Promo) | Y | Y (Focused/Other) | **N** | BUILD |
| Archive vs delete | Y | Y | **Y** | - |
| Batch select + actions | Y | Y | **Y** | - |
| Drag-and-drop to folders | Y | Y | **Y** | - |
| Mute conversation | Y | Y | **N** | BUILD |
| Importance markers | Y (auto) | Y (manual) | **P** (priority field) | - |

### Search

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Full-text body search | Y | Y | **Y** (BM25) | - |
| Operators (from:, to:, has:) | Y (20+) | Y (15+) | **N** | BUILD |
| Visual filter chips | Y | Y | **P** (checkboxes) | BUILD |
| Natural language search | Y | N | **N** | PLAN |
| Search within attachments | Y | Y (Office files) | **N** | PLAN |
| Saved searches | N | Y | **N** | BUILD |
| Fuzzy matching | N | N | **Y** (BM25) | ADVANTAGE |
| Keyboard shortcut to search | Y (/) | Y (Alt+Q) | **Y** (/) | - |

### Contacts & Address Book

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Contact cards with details | Y | Y | **Y** | - |
| Contact groups/lists | Y | Y (distribution) | **Y** (groups) | - |
| Auto-suggest from history | Y | Y | **Y** | - |
| Directory integration (LDAP/AD) | Y (Workspace) | Y (Azure AD) | **N** (DHT-native) | N/A |
| Contact merge/dedup | Y | Y | **N** | BUILD |
| Import/export (vCard, CSV) | Y | Y | **N** | BUILD |
| Quick-compose from contact | Y | Y | **Y** | - |
| Trust scoring per contact | N | N | **U** (MATL) | ADVANTAGE |
| Byzantine detection | N | N | **U** | ADVANTAGE |

### Attachments

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Drag-and-drop upload | Y | Y | **N** | BUILD |
| Inline preview (PDF, images) | Y | Y | **N** | BUILD |
| Cloud storage integration | Y (Drive) | Y (OneDrive) | **N** (IPFS planned) | PLAN |
| Large file handling | Y (Drive link >25MB) | Y (OneDrive) | **Y** (10GB chunked) | ADVANTAGE |
| Download all as zip | Y | Y | **N** | BUILD |
| Content-addressed storage | N | N | **U** (SHA-256 chunks) | ADVANTAGE |

### Security & Privacy

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Spam filtering | Y (ML-based) | Y (ML-based) | **Y** (MATL trust) | DIFFERENT |
| Phishing detection | Y | Y | **P** (trust scores) | BUILD |
| Two-factor auth | Y | Y | **Y** (MFA zome) | - |
| End-to-end encryption | N (in-transit only) | N (S/MIME optional) | **Y** (default E2E) | ADVANTAGE |
| Post-quantum cryptography | N | N | **U** (Kyber+Dilithium) | ADVANTAGE |
| Confidential mode (expiring) | Y | Y (IRM) | **Y** (expires_at) | - |
| S/MIME support | Y (paid) | Y | N/A (own crypto) | - |
| Zero-knowledge storage | N | N | **U** (DHT encrypted) | ADVANTAGE |
| Data sovereignty (no server) | N | N | **U** (Holochain) | ADVANTAGE |
| Key rotation | N/A | N/A | **Y** (automatic) | ADVANTAGE |
| Sender verification (sig) | DKIM/SPF | DKIM/SPF | **Y** (Ed25519/Dilithium) | ADVANTAGE |
| Trust-based delivery | N | N | **U** (MATL algorithm) | ADVANTAGE |
| Audit trail | N | Y (compliance) | **Y** (immutable DHT) | ADVANTAGE |
| Federation (cross-network) | N | N | **U** (SMTP/Matrix/AP bridge) | ADVANTAGE |

### Notifications

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Desktop notifications | Y | Y | **N** (sound only) | BUILD |
| Sound alerts | Y | Y | **Y** | - |
| Tab badge/title count | Y | Y | **Y** | - |
| Per-label notification rules | Y | Y | **N** | BUILD |
| Mobile push | Y | Y | **N** (PWA limitation) | TAURI/SOMA |

### Offline

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Offline read | Y | Y | **Y** (SW cache) | - |
| Queued sending | Y | Y | **Y** (offline queue) | - |
| CRDT sync on reconnect | N | N | **U** (sync zome) | ADVANTAGE |
| Conflict resolution | N/A | N/A | **U** (vector clocks) | ADVANTAGE |

### Calendar & Tasks

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Event creation from email | Y | Y | **N** | PLAN |
| Task creation from email | Y | Y | **N** | PLAN |
| RSVP inline | Y | Y | **N** | PLAN |
| Calendar sidebar | Y | Y | **N** | PLAN |

### AI Features

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Smart compose / autocomplete | Y | Y (Copilot) | **N** | BUILD (local LLM) |
| Smart reply suggestions | Y (3 options) | Y (Copilot) | **N** | BUILD (local LLM) |
| Email summarization | Y | Y (Copilot) | **N** | BUILD (Symthaea) |
| Action item extraction | Y | Y | **N** | BUILD |
| Writing assistance/rewrite | Y | Y (Copilot) | **N** | BUILD (local LLM) |
| Privacy-preserving AI | N (cloud) | N (cloud) | **PLANNED** (on-device) | ADVANTAGE |

### Accessibility

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Keyboard shortcuts | Y (70+) | Y (50+) | **Y** (10) | EXPAND |
| Screen reader (ARIA) | Y | Y | **P** | IMPROVE |
| High contrast mode | Y | Y | **N** | BUILD |
| Font size controls | Y | Y (zoom) | **N** | BUILD |
| Reduced motion support | N | P | **Y** | ADVANTAGE |

### Mobile

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Responsive web | Y | Y | **Y** | - |
| Native apps | Y (iOS/Android) | Y (iOS/Android) | **N** (PWA only) | TAURI/SOMA |
| Bottom navigation | Y (app) | Y (app) | **Y** | - |
| Swipe gestures | Y | Y | **N** | BUILD |
| Installable (PWA) | N | N | **Y** | ADVANTAGE |

### Collaboration

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Shared mailboxes | Y (Workspace) | Y | **Y** (capabilities zome) | - |
| Delegation (send-as) | Y | Y | **Y** (capabilities zome) | - |
| Shared labels | Y (Workspace) | Y (shared folders) | **N** | BUILD |

### Settings & Customization

| Feature | Gmail | Outlook | Mycelix Mail | Gap |
|---------|-------|---------|-------------|-----|
| Dark/light theme | Y | Y | **Y** | - |
| Custom theme colors | N | Y (limited) | **N** | BUILD |
| Density (compact/comfortable) | Y (3 levels) | Y (3 levels) | **N** | BUILD |
| Reading pane position | N | Y (right/bottom/off) | **N** | BUILD |
| Custom swipe actions | Y | Y | **N** | BUILD |
| Multiple signatures mgmt | Y | Y | **N** | BUILD |

---

## Scoring Summary

| Category | Gmail | Outlook | Mycelix Mail |
|----------|-------|---------|-------------|
| Core Email | 15/16 | 16/16 | **6/16** |
| Organization | 9/10 | 10/10 | **6/10** |
| Search | 5/7 | 6/7 | **4/7** |
| Contacts | 6/8 | 7/8 | **6/8** (+2 unique) |
| Attachments | 5/6 | 5/6 | **2/6** (+2 unique) |
| Security | 5/13 | 6/13 | **12/13** (+6 unique) |
| Notifications | 4/5 | 5/5 | **2/5** |
| Offline | 2/4 | 2/4 | **4/4** (+2 unique) |
| Calendar/Tasks | 4/4 | 4/4 | **0/4** |
| AI | 5/6 | 6/6 | **0/6** |
| Accessibility | 3/5 | 4/5 | **3/5** |
| Mobile | 4/5 | 4/5 | **3/5** |
| Collaboration | 3/3 | 3/3 | **2/3** |
| Customization | 3/6 | 5/6 | **1/6** |

### Overall: Gmail 73/96 (76%) | Outlook 81/96 (84%) | Mycelix 51/96 (53%) + 12 unique features

---

## Strategic Assessment

### Where Mycelix Mail Already Wins (KEEP)
1. **Security** — E2E encryption by default, PQC, zero-knowledge storage, audit trail
2. **Privacy** — No server, no data mining, MATL trust > ML spam
3. **Decentralization** — CRDT sync, federation, DHT immutability
4. **Offline** — Full CRDT sync with conflict resolution
5. **Trust** — Byzantine detection, reputation scoring, trust-based delivery

### Critical Gaps (BUILD FIRST — users won't switch without these)
1. **Rich text compose** — Toolbar with bold/italic/lists/links/inline images
2. **Auto-filters/rules** — "If from X, move to Y, mark read"
3. **Labels/tags** — Multiple labels per email (not just folders)
4. **Attachment preview** — Inline PDF/image rendering
5. **Drag-and-drop file upload** in compose
6. **Undo send** (3-10 second delay)
7. **Snooze** (reappear in inbox at set time)
8. **Search operators** (from:, to:, has:attachment, before:, after:)
9. **Desktop notifications** (Notification API)
10. **Density/reading pane** options

### Differentiators to Amplify (MARKET)
1. Post-quantum encryption (nobody else has this)
2. Federation (SMTP + Matrix + ActivityPub bridges)
3. Trust-based spam prevention (no ML, no cloud, no false positives)
4. Data sovereignty (your keys, your data, no server)
5. CRDT offline sync (works in mesh networks, disaster scenarios)

### Strategic Skips (DON'T BUILD)
1. Calendar/Tasks integration — Different cluster (mycelix-commons has calendar)
2. AI email features — Plan for Phase 2 with local Symthaea/Broca, not cloud
3. Email recall — Impossible on immutable DHT (feature of centralization)
4. Enterprise directory (LDAP/AD) — Not our market

---

## Phase 1 Priority: Close the Gap (Target: 75/96 = 78%)

### P1a: Rich Compose (biggest user-facing gap)
- Formatting toolbar: bold, italic, underline, strikethrough
- Bullet/numbered lists
- Links (insert URL dialog)
- Inline images (paste or button)
- Code blocks (monospace)
- Block quotes

### P1b: Organization Power
- Labels/tags system (multiple per email, colored)
- Auto-filter rules (from, subject, contains → action)
- Snooze (scheduler zome already supports this)
- Undo send (client-side delay before calling send_email)

### P1c: Search Operators
- `from:alice`, `to:bob`, `subject:proposal`
- `has:attachment`, `is:unread`, `is:starred`
- `before:2026-04-01`, `after:2026-03-01`
- `label:governance`
- Visual chips for each operator

### P1d: Attachment Experience
- Drag-and-drop upload to compose
- Inline image/PDF preview in read view
- File type icons
- Download button per attachment

### P1e: Desktop Notifications
- Notification API permission request
- Per-folder notification rules
- Sound + visual + badge combined
