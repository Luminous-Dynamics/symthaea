# Civic hApp

**Decentralized civic knowledge and AI agent reputation for Symthaea**

This hApp provides two core zomes that enable Symthaea AI agents to:
1. Store and retrieve civic knowledge on the Holochain DHT
2. Track reputation using MATL-based Byzantine-resistant trust scoring

## Architecture

```
civic-happ/
├── dna/
│   ├── dna.yaml                    # DNA manifest
│   └── zomes/
│       ├── civic_knowledge/        # Civic knowledge storage
│       │   ├── integrity/          # Entry types and validation
│       │   └── coordinator/        # CRUD and search operations
│       └── agent_reputation/       # Agent trust scoring
│           ├── integrity/          # MATL score types
│           └── coordinator/        # Reputation computation
├── client/                         # TypeScript client
│   └── src/
│       ├── types.ts               # Type definitions
│       ├── knowledge.ts           # Knowledge client
│       ├── reputation.ts          # Reputation client
│       └── index.ts               # Combined client
├── happ.yaml                      # hApp manifest
├── Cargo.toml                     # Rust workspace
└── README.md
```

## Civic Knowledge Zome

Stores civic information (FAQs, eligibility rules, contact info, etc.) organized by:
- **Domain**: benefits, permits, tax, voting, justice, housing, employment, education, health, emergency, general
- **Geographic scope**: State, city, zip code
- **Keywords**: For search

### Entry Types

```rust
CivicKnowledge {
    domain: CivicDomain,
    knowledge_type: KnowledgeType,
    title: String,
    content: String,
    geographic_scope: Option<String>,
    keywords: Vec<String>,
    source: Option<String>,
    last_verified: Option<u64>,
    expires_at: Option<u64>,
    links: Vec<String>,
    contact_phone: Option<String>,
    address: Option<String>,
}
```

### Functions

- `create_knowledge` - Add new civic knowledge
- `get_knowledge` - Get by action hash
- `search_by_domain` - Find all knowledge for a domain
- `search_by_location` - Find knowledge for a geographic scope
- `search_by_keyword` - Full-text keyword search
- `search_knowledge` - Combined search
- `validate_knowledge` - Submit authority validation
- `update_knowledge` - Submit update proposal

## Agent Reputation Zome

Implements MATL (Multi-Agent Trust Layer) scoring for AI agents:

```
Composite = 0.4·Quality + 0.3·Consistency + 0.3·Reputation
```

### Entry Types

```rust
AgentProfile {
    agent_type: AgentType,
    name: String,
    specializations: Vec<AgentSpecialization>,
    description: String,
    is_active: bool,
    created_at: u64,
    last_active: u64,
}

TrustScore {
    agent_pubkey: AgentPubKey,
    quality: f32,        // Response accuracy
    consistency: f32,    // Behavior predictability
    reputation: f32,     // Citizen feedback
    composite: f32,      // MATL score (0.0 - 1.0)
    confidence: f32,     // Based on event count
}
```

### Functions

- `register_agent` - Create agent profile
- `record_event` - Record feedback (helpful, accurate, etc.)
- `compute_trust_score` - Calculate MATL score
- `get_trustworthy_agents` - Find agents above threshold
- `get_agents_by_specialization` - Find domain experts

### Trustworthiness

An agent is considered trustworthy when:
- `composite >= 0.55` (45% Byzantine tolerance)
- `confidence >= 0.3` (at least 30 events)

## TypeScript Client

```typescript
import { AppWebsocket } from '@holochain/client';
import { CivicClient, MATL } from '@mycelix/civic-client';

// Connect to conductor
const client = await AppWebsocket.connect('ws://localhost:8888');
const civic = new CivicClient(client);

// Search for benefits knowledge
const results = await civic.knowledge.searchByDomain('benefits');

// Record feedback for an agent
await civic.reputation.recordHelpful(agentPubkey, 'conv-123');

// Check if agent is trustworthy
const score = await civic.reputation.getTrustScore(agentPubkey);
if (MATL.isTrustworthy(score)) {
  console.log('Agent is trustworthy:', score.composite);
}
```

## Building

```bash
# Build WASM zomes
cargo build --release --target wasm32-unknown-unknown

# Package DNA
hc dna pack dna/

# Package hApp
hc app pack .

# Build TypeScript client
cd client && npm install && npm run build
```

## Integration with SMS Gateway

The SMS gateway uses this hApp to:
1. Retrieve domain-specific knowledge to augment AI responses
2. Track agent reputation based on citizen feedback
3. Route to trustworthy agents based on MATL scores

See `services/sms-gateway/src/holochain-adapter.ts` for integration.

## License

Apache-2.0
