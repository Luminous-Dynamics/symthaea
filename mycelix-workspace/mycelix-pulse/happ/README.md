# 📧 Mycelix Mail - Trust-Based Decentralized Email

**Revolutionary spam filtering using MATL reputation scores at the protocol level.**

## 🌟 What Is This?

Mycelix Mail is a decentralized email system built on Holochain that replaces content-based spam filtering with **sender reputation filtering**. Instead of scanning your emails (surveillance), we score the sender's trust and reject low-trust senders before the message even downloads.

### Key Innovations

1. **MATL Integration**: Uses your existing Mycelix Adaptive Trust Layer (45% Byzantine tolerance) for spam detection
2. **E2E Encryption**: All message bodies encrypted with recipient's public key
3. **Agent-Centric**: Your inbox lives on YOUR device, not a corporate server
4. **Legacy Compatible**: SMTP bridge for Gmail/Outlook interoperability (planned Phase 3)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Integrity Zome (mycelix_mail_integrity)                │
│  • MailMessage (core message type)                      │
│  • TrustScore (MATL scores cached on DHT)              │
│  • EpistemicTier (from Epistemic Charter v2.0)         │
│  • Validation rules                                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Coordinator Zomes                                       │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  mail_messages                                    │   │
│  │  • send_message()                                 │   │
│  │  • get_inbox()                                    │   │
│  │  • get_outbox()                                   │   │
│  │  • get_thread()                                   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  trust_filter (THE SPAM KILLER)                   │   │
│  │  • check_sender_trust()                           │   │
│  │  • filter_inbox(min_trust)                        │   │
│  │  • update_trust_score()                           │   │
│  │  • report_spam()                                  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Holochain DHT                                           │
│  • Source chains (your personal mailbox)                │
│  • DHT storage (message metadata + trust scores)        │
│  • IPFS (encrypted message bodies)                      │
└─────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
mycelix-mail/
├── dna/                          # Holochain DNA
│   ├── integrity/                # Core data types
│   │   └── src/
│   │       └── lib.rs           # MailMessage, TrustScore, validation
│   ├── zomes/
│   │   ├── mail_messages/       # Send/receive logic
│   │   │   └── src/lib.rs
│   │   └── trust_filter/        # MATL integration & spam filter
│   │       └── src/lib.rs
│   ├── Cargo.toml               # Rust workspace
│   └── dna.yaml                 # DNA configuration
├── ui/                          # User interfaces (Phase 4)
│   ├── tauri-app/               # Desktop client (Rust + React)
│   └── web/                     # Web client (Holochain + Vue)
├── smtp-bridge/                 # Legacy email bridge (Phase 3)
│   └── validator.py             # SMTP ↔ Holochain translator
├── scripts/                     # Build and deployment scripts
└── docs/                        # Documentation
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install Holochain (via Nix)
nix-shell -p holochain

# Verify installation
hc --version  # Should be 0.3.0-beta or later
```

### Build the DNA

```bash
cd mycelix-mail/dna

# Build all zomes
cargo build --release --target wasm32-unknown-unknown

# Copy WASM files
cp target/wasm32-unknown-unknown/release/mycelix_mail_integrity.wasm .
cp target/wasm32-unknown-unknown/release/mail_messages.wasm .
cp target/wasm32-unknown-unknown/release/trust_filter.wasm .

# Pack the DNA
hc dna pack .
```

### Run in Sandbox

```bash
# Start a Holochain sandbox
hc sandbox create -d mycelix-mail.dna

# Call functions
hc sandbox call -- send_message '{"from_did": "did:mycelix:alice", "to_did": "did:mycelix:bob", "subject_encrypted": [72,101,108,108,111], "body_cid": "QmTest123", "timestamp": 1234567890, "thread_id": null, "epistemic_tier": "Tier1Testimonial"}'

# Get inbox
hc sandbox call -- get_inbox

# Filter by trust
hc sandbox call -- filter_inbox 0.7
```

### Register Your DID Inside the DNA

Before other agents can route mail to you, publish your DID → AgentPubKey binding inside the DNA:

```bash
hc sandbox call -- register_my_did '{"did": "did:mycelix:alice"}'
```

You can verify the binding later by calling `resolve_did` (CLI support is coming next) or by inspecting the `did_index` path via `hc dump`.

## 🎯 Implementation Status

### ✅ Phase 1: Core DNA (COMPLETE)

- [x] Integrity types (MailMessage, TrustScore, Contact)
- [x] Validation rules
- [x] send_message() / get_inbox() / get_outbox()
- [x] Trust score lookup
- [x] filter_inbox() spam filtering
- [x] Thread support
- [x] Epistemic Charter v2.0 integration

### 🚧 Phase 2: MATL Integration (NEXT)

- [ ] Python bridge to connect Holochain ↔ 0TML/MATL
- [ ] Periodic trust score sync from MATL to DHT
- [ ] Spam report feedback loop
- [ ] Real-time trust score updates

### ⏳ Phase 3: SMTP Bridge (FUTURE)

- [ ] SMTP validator nodes
- [ ] DAO governance for validators
- [ ] Staking mechanism (FLOW tokens)
- [ ] Email → Mycelix translator
- [ ] Mycelix → Email translator
- [ ] DNS configuration for mycelix.net

### ⏳ Phase 4: UI (FUTURE)

- [ ] Tauri desktop app
- [ ] Web interface (via Holo)
- [ ] Compose UI
- [ ] Contact management
- [ ] Settings panel

## 🔧 Development

### Run Tests

```bash
cd dna
cargo test
```

### Debug Mode

```bash
# Set RUST_LOG for detailed logging
RUST_LOG=debug hc sandbox call -- get_inbox
```

### Integration with Existing MATL

The `trust_filter` zome is designed to integrate with your existing MATL system in `Mycelix-Core/0TML/`. To connect them:

```python
# Create a bridge service
from matl import MATLClient
from holochain_client import HolochainClient

matl = MATLClient(mode="mode1", oracle_endpoint="http://localhost:8080")
hc = HolochainClient("ws://localhost:8888")

# Sync trust scores every 5 minutes
async def sync_trust_scores():
    while True:
        dids = await get_active_dids()  # Query Holochain for active users
        for did in dids:
            score = await matl.get_composite_trust_score(did)
            await hc.call_zome(
                "trust_filter",
                "update_trust_score",
                {
                    "did": did,
                    "score": score.final_score,
                    "last_updated": int(time.time()),
                    "matl_source": "mode1_oracle"
                }
            )
        await asyncio.sleep(300)
```

## 🎓 How Spam Filtering Works

Traditional email filters your mail by **reading it** (surveillance). Mycelix filters by **scoring the sender** (privacy-preserving).

### Example Flow

1. **Spammer sends 10,000 messages**
   - MATL's entropy score detects bot-like behavior (low entropy)
   - MATL's TCDM detects cartel coordination (high clustering)
   - Spammer gets trust score: **0.05**

2. **You check your inbox**
   - You call `filter_inbox(0.3)` (minimum trust threshold)
   - Holochain queries trust scores for all senders
   - Messages from score < 0.3 are hidden
   - **Spam never downloads, never decrypts, never seen**

3. **Trusted colleague sends message**
   - They have high PoGQ (validation accuracy)
   - Diverse network connections
   - Human-like behavior patterns
   - Trust score: **0.85**
   - Message appears in inbox instantly

### MATL Components Used

- **PoGQ (40%)**: Historical accuracy of their interactions
- **TCDM (30%)**: Are they part of a spam cartel?
- **Entropy (30%)**: Do they behave like a bot or human?

## 🔐 Security Model

### Threat Model

| Attack | Defense |
|--------|---------|
| Spam (volume) | Trust score < threshold → reject |
| Phishing | DIDs + VCs verify sender identity |
| Sybil | MATL cartel detection |
| Content scanning | E2E encryption, never decrypt suspicious messages |
| Backdoor | Open source, auditable |

### Privacy Guarantees

- **Your inbox is yours**: Stored on your device (Holochain source chain)
- **No one reads your mail**: E2E encrypted
- **Metadata minimized**: Only sender DID + timestamp visible to DHT
- **No surveillance**: Spam filtering happens at protocol layer, not content

## 📊 Performance Targets

- **Send latency**: <2s (P2P)
- **Inbox query**: <500ms (local source chain)
- **Trust lookup**: <100ms (DHT query)
- **Spam filter accuracy**: >99%

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Areas

- **Testing**: Write integration tests for zome functions
- **MATL Bridge**: Implement Python bridge in `smtp-bridge/`
- **UI**: Build Tauri desktop app in `ui/tauri-app/`
- **SMTP Bridge**: Validator implementation

## 📚 References

- [Mycelix Protocol Architecture v5.2](https://github.com/Luminous-Dynamics/Mycelix-Core/blob/main/docs/architecture/Mycelix%20Protocol_%20Integrated%20System%20Architecture%20v5.2.md)
- [MATL Architecture](https://github.com/Luminous-Dynamics/Mycelix-Core/blob/main/0TML/docs/06-architecture/matl_architecture.md)
- [Epistemic Charter v2.0](https://github.com/Luminous-Dynamics/Mycelix-Core/blob/main/docs/architecture/THE%20EPISTEMIC%20CHARTER%20(v2.0).md)
- [Holochain Documentation](https://developer.holochain.org/)

## 🎯 Next Actions

1. **Build and test the DNA**: `cargo build && hc dna pack`
2. **Run in sandbox**: Verify send_message and filter_inbox work
3. **Create MATL bridge**: Connect to existing 0TML system
4. **Build simple UI**: Tauri desktop app to test real usage

---

**Status**: Phase 1 DNA complete ✅ | Phase 2 MATL integration next 🚧

**Questions?** Open an issue or contact: tristan.stoltz@evolvingresonantcocreationism.com

🍄 **"The best spam filter is one that never sees your mail."** 🍄
