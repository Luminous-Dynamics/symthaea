# Mycelix Mail Architecture

## Overview

Mycelix Mail is a decentralized, privacy-first email system built on Holochain. It provides end-to-end encrypted messaging with a sophisticated trust network (MATL algorithm) for spam prevention and sender verification.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Interface                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │  React UI   │  │   Vue UI    │  │  Mobile App │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│         └────────────────┼────────────────┘                         │
│                          │                                          │
│  ┌───────────────────────▼───────────────────────────────────────┐  │
│  │              TypeScript Client (@mycelix/holochain-client)     │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │  │
│  │  │ Services│  │  Hooks  │  │ Stores  │  │ Signals │          │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────────────┘
                           │ WebSocket
┌──────────────────────────▼──────────────────────────────────────────┐
│                      Holochain Conductor                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Mycelix Mail DNA                            │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │                  Coordinator Zomes                       │  │  │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │  │  │
│  │  │  │Messages │ │  Trust  │ │  Sync   │ │ Federat.│  ...  │  │  │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │                   Integrity Zomes                        │  │  │
│  │  │  (Entry types, link types, validation rules)             │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ DHT
┌──────────────────────────▼──────────────────────────────────────────┐
│                         DHT Network                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │  Node 1 │──│  Node 2 │──│  Node 3 │──│  Node N │                │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Holochain Zomes

#### Messages Zome
- Email composition and storage
- Threading and conversation management
- Inbox/Sent/Draft/Archive organization
- Attachment handling

#### Trust Zome (MATL Algorithm)
- Multi-dimensional trust attestations
- Direct, network, and temporal trust factors
- Stake-weighted scoring
- Spam prevention through reputation

#### Sync Zome (CRDT)
- Offline-first operation
- Conflict-free replicated data types
- Automatic conflict resolution
- Background synchronization

#### Federation Zome
- Cross-cell communication
- Network discovery
- Envelope routing
- Protocol bridging

#### Additional Zomes
- **Contacts**: Address book management
- **Capabilities**: Access control and delegation
- **Search**: Local search indexing
- **Backup**: Data export/import
- **Scheduler**: Delayed/recurring sends
- **Audit**: Compliance logging

### 2. TypeScript Client

```typescript
// Core client usage
import { MycelixMailClient, MycelixClient } from '@mycelix/holochain-client';

// Create client
const client = await MycelixClient.create({
  appId: 'mycelix-mail',
  websocketUrl: 'ws://localhost:8888',
});

// Send email with trust check
const result = await client.messages.sendEmailWithTrustCheck(
  recipientPubKey,
  'Subject',
  'Body',
  { requireTrustLevel: 0.3 }
);
```

### 3. React Integration

```tsx
import { HolochainProvider, useInbox, useTrustNetwork } from '@mycelix/holochain-client/react';

function App() {
  return (
    <HolochainProvider autoConnect>
      <Inbox />
    </HolochainProvider>
  );
}

function Inbox() {
  const { emails, isLoading, refresh } = useInbox();
  // ...
}
```

## Data Flow

### Sending an Email

```
1. User composes email in UI
2. Client validates recipient trust level
3. Email encrypted with recipient's public key
4. Signed with sender's private key
5. Entry committed to local source chain
6. Published to DHT
7. Signal sent to recipient (if online)
8. Recipient's conductor receives via gossip/signal
9. Entry validated and stored
10. UI updated via signal subscription
```

### Trust Attestation Flow

```
1. User creates attestation for another agent
2. Attestation includes trust level (0.0-1.0), context, category
3. Stake may be allocated to back the attestation
4. Entry committed and published to DHT
5. MATL algorithm recalculates network trust scores
6. Trust updates propagate through the network
```

## Security Architecture

### Encryption
- All emails are end-to-end encrypted using X3DH key exchange
- Double Ratchet algorithm for forward secrecy
- Attachments encrypted separately with streaming

### Authentication
- Agent identity via Holochain's Ed25519 keypairs
- Capability tokens for delegated access
- No passwords or central authentication

### Validation
- Entry validation in integrity zomes
- Source chain integrity verification
- Network-level consensus on valid entries

## Deployment Options

### Development
```bash
# Local sandbox
hc sandbox generate --run 8888

# With Docker
docker-compose up -d
```

### Production
- Multi-node conductor clusters
- Load balancing via Caddy/nginx
- Persistent volume storage
- Monitoring via Prometheus/Grafana

## Performance Considerations

### Caching
- LRU cache for frequently accessed emails
- Trust score memoization
- Contact information caching

### Optimization
- Batch operations for bulk actions
- Pagination for large mailboxes
- Lazy loading of email bodies
- Background sync scheduling

## Extensibility

### Plugin System
```typescript
const plugin: Plugin = {
  manifest: {
    id: 'my-plugin',
    name: 'My Plugin',
    permissions: ['emails:read', 'ui:toolbar'],
    hooks: ['email:beforeSend'],
  },
  async activate(context) {
    // Plugin initialization
  },
  async onHook(hook, data) {
    // Process hook
    return data;
  },
};
```

### Theme System
```typescript
const customTheme: Theme = {
  id: 'custom',
  name: 'Custom Theme',
  mode: 'dark',
  colors: {
    primary: { /* color palette */ },
    // ...
  },
};
```
