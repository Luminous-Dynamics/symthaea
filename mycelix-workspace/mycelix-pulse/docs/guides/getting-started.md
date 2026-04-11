# Getting Started with Mycelix Mail

This guide will help you set up and start developing with Mycelix Mail, a decentralized, trust-based email system built on Holochain.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Rust** (1.70+) with the `wasm32-unknown-unknown` target
- **Node.js** (18+) and npm
- **Holochain** development tools (`hc` CLI, `holochain` conductor)
- **Nix** (recommended for reproducible builds)

### Installing Holochain Tools

If you have Nix installed, enter the development environment:

```bash
nix develop
```

Otherwise, install Holochain tools manually:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup target add wasm32-unknown-unknown

# Install Holochain
cargo install holochain --locked
cargo install hc --locked
```

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/luminous-dynamics/mycelix-mail
cd mycelix-mail

# Install dependencies
make install
```

### 2. Build the Project

```bash
# Build all components
make build

# Or build specific parts:
make build-zomes     # Build Holochain zomes (Rust/WASM)
make build-client    # Build TypeScript client
make build-ui        # Build UI frontend
```

### 3. Run Tests

```bash
# Run all tests
make test

# Or run specific test suites:
make test-zomes      # Rust unit tests
make test-client     # TypeScript tests
make test-e2e        # End-to-end tests
```

### 4. Start Development Server

```bash
# Start full development environment
make dev

# Or start components individually:
make dev-holochain   # Start Holochain sandbox
make dev-ui          # Start UI development server
```

## Project Structure

```
mycelix-mail/
├── holochain/           # Holochain backend
│   ├── zomes/           # Coordinator and integrity zomes
│   │   ├── messages/    # Email messaging
│   │   ├── contacts/    # Contact management
│   │   ├── trust/       # Web of trust (MATL algorithm)
│   │   ├── keys/        # Post-quantum key management
│   │   ├── federation/  # Cross-network federation
│   │   └── ...
│   └── client/          # TypeScript client for zome calls
├── sdk/                 # Official SDKs
│   ├── typescript/      # TypeScript/JavaScript SDK
│   └── python/          # Python SDK
├── ui/                  # Web UI frontend
├── backend/             # REST API gateway
└── docs/                # Documentation
```

## Using the TypeScript SDK

### REST API Client

```typescript
import { MycelixClient } from '@mycelix/sdk';

const client = new MycelixClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.mycelix.example.com',
});

// List inbox emails
const emails = await client.emails.list({ folder: 'inbox' });

// Send an email
const sent = await client.emails.send({
  to: ['recipient@example.com'],
  subject: 'Hello from Mycelix',
  body: 'This is a secure, trust-verified email.',
});

// Check trust score
const trustScore = await client.trust.getScore('contact@example.com');
console.log(`Trust level: ${trustScore.level}`);
```

### Direct Holochain Client

For direct P2P communication without a gateway server:

```typescript
import { createHolochainClient } from '@mycelix/sdk/holochain';

const client = createHolochainClient({
  appId: 'mycelix-mail',
  url: 'ws://localhost:8888',
});

await client.connect();

// Send encrypted email via P2P
const result = await client.messages.sendEmail({
  recipients: [recipientAgentPubKey],
  encrypted_subject: encryptedSubject,
  encrypted_body: encryptedBody,
  priority: 'Normal',
  read_receipt_requested: true,
});

// Create trust attestation
await client.trust.createAttestation({
  trustee: contactAgentPubKey,
  trust_level: 0.8,
  category: 'Identity',
  evidence: 'In-person verification at conference',
});
```

## Core Concepts

### Web of Trust

Mycelix Mail uses a decentralized web of trust instead of centralized certificate authorities. Trust is:

- **Direct**: You personally attest trust in contacts you verify
- **Transitive**: Trust propagates through your trusted network with decay
- **Categorical**: Different trust levels for identity, communication, etc.
- **Byzantine-resistant**: Detection of Sybil attacks and collusion

### End-to-End Encryption

All emails are encrypted using:

- **X25519** or **Kyber1024** (post-quantum) for key exchange
- **ChaCha20-Poly1305** or **AES-256-GCM** for symmetric encryption
- **Ed25519** or **Dilithium** (post-quantum) for signatures

### Decentralized Storage

Emails are stored on the Holochain DHT (Distributed Hash Table):

- No central server owns your data
- Content-addressed and tamper-proof
- Real-time P2P delivery via signals
- Offline-first with CRDT sync

## Next Steps

- [Architecture Overview](./architecture.md) - Deep dive into system design
- [Deployment Guide](./deployment.md) - Production deployment instructions
- [API Reference](/docs/api/) - Complete API documentation

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/luminous-dynamics/mycelix-mail/issues)
- **Discussions**: [Community forum](https://github.com/luminous-dynamics/mycelix-mail/discussions)
- **Contributing**: See [CONTRIBUTING.md](/CONTRIBUTING.md)
