# Mycelix Music Holochain DNA

Zero-cost music streaming through agent-centric architecture.

## The Innovation

Traditional streaming platforms charge fees for every micro-transaction. Mycelix Music uses Holochain's agent-centric model to eliminate these costs:

1. **Plays are FREE** - Each play is just an entry on the listener's local source chain
2. **Settlement is batched** - Only periodic cashouts touch the blockchain
3. **Trust is distributed** - Web-of-trust verification, no central authority

## Architecture

```
mycelix-music DNA
├── catalog/          # Song metadata, albums, artist profiles
│   ├── integrity     # Entry definitions & validation
│   └── coordinator   # CRUD operations, search
│
├── plays/            # Zero-cost play recording
│   ├── integrity     # PlayRecord, SettlementBatch validation
│   └── coordinator   # Recording, batching, settlement
│
├── balances/         # Credit/debit tracking
│   ├── integrity     # Account, Deposit, Cashout validation
│   └── coordinator   # Balance management, transfers
│
└── trust/            # MATL integration
    ├── integrity     # TrustClaim, CDN reputation validation
    └── coordinator   # Verification, CDN routing, Byzantine detection
```

## Zome Overview

### Catalog Zome
Manages the music catalog - songs, albums, and artist profiles.

**Key Features:**
- Song metadata with IPFS CIDs for audio
- Album collections with ordered tracks
- Artist profiles with payment addresses
- Genre-based discovery

### Plays Zome
The heart of zero-cost streaming.

**How it works:**
1. Listener plays a song → `PlayRecord` created on their source chain (FREE!)
2. Plays accumulate with calculated `amount_owed`
3. Periodically, plays batch into `SettlementBatch`
4. Only the batch settlement touches the blockchain (amortized cost)

**Play Economics:**
- Base rate: 0.001 USD per full play
- Minimum: 30 seconds OR 50% completion
- Strategy multipliers: premium (2x), patronage (1.5x), gift (free)

### Balances Zome
Tracks all credits and debits without touching the blockchain.

**Features:**
- Listener accounts (pre-funded balance)
- Artist accounts (pending earnings)
- Deposit verification (oracle-based)
- Cashout requests (batch settlement)

### Trust Zome
Implements Multi-Agent Trust Logic (MATL) for decentralized verification.

**Components:**
- **Web-of-trust**: Artists verified through community vouching
- **CDN reputation**: PoGQ scoring for content delivery nodes
- **Byzantine detection**: Report and penalize bad actors

## Building

```bash
# Enter nix shell with Holochain tools
nix develop

# Build all zomes
cargo build --release --target wasm32-unknown-unknown

# Package DNA
hc dna pack .
```

## Integration with Rust API

The Rust API (Axum) serves as the bridge between:
- Web clients (REST/WebSocket)
- Holochain DNA (via conductor)
- On-chain contracts (Gnosis Chain)

```
[Client] → [Rust API] → [Holochain Conductor] → [DNA Zomes]
                    ↘ [Gnosis Chain] ← (cashouts only)
```

## Economic Flow

```
1. Artist uploads song → Catalog zome stores metadata
2. Listener deposits xDAI → Balances records deposit
3. Listener plays song → Plays records FREE play on source chain
4. Plays accumulate → Amount owed calculated per strategy
5. Periodic batching → Settlement batch created
6. Artist requests cashout → On-chain tx (single tx for many plays!)
```

## Why This Matters

| Metric | Traditional Streaming | Mycelix Music |
|--------|----------------------|---------------|
| Cost per play | $0.0001-0.001 (gas) | $0 (local) |
| Settlement | Per-play | Batched |
| Trust model | Platform authority | Web-of-trust |
| CDN | Centralized | Community-owned |

## License

MIT
