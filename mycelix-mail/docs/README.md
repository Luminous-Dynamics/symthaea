# Mycelix Mail Documentation

Welcome to the Mycelix Mail documentation. This documentation covers everything you need to integrate with or extend the Mycelix Mail decentralized email system.

## Documentation Structure

```
docs/
├── api/                    # Generated API documentation
│   ├── rust/               # Rust/Holochain zome documentation (cargo doc)
│   ├── typescript/         # TypeScript SDK documentation (TypeDoc)
│   └── index.html          # Unified documentation index
├── guides/                 # Developer guides
│   ├── getting-started.md  # Quick start guide
│   ├── architecture.md     # System architecture overview
│   └── deployment.md       # Production deployment guide
└── README.md               # This file
```

## Quick Links

### For Developers

- **[Getting Started](guides/getting-started.md)** - Set up your development environment and build your first integration
- **[Architecture Overview](guides/architecture.md)** - Deep dive into the system design, data models, and algorithms
- **[Deployment Guide](guides/deployment.md)** - Production deployment with Docker, Kubernetes, and monitoring

### API Reference

- **[Rust API Documentation](api/rust/index.html)** - Holochain zome function reference
- **[TypeScript SDK Documentation](api/typescript/index.html)** - TypeScript/JavaScript SDK reference
- **[Unified API Index](api/index.html)** - Start here for API browsing

## Generating Documentation

Documentation is generated using `cargo doc` for Rust and TypeDoc for TypeScript.

### Generate All Documentation

```bash
# Using make
make docs

# Using the script directly
./scripts/generate-docs.sh
```

### Generate Specific Documentation

```bash
# Rust documentation only
make docs-rust

# TypeScript documentation only
make docs-typescript

# Clean and regenerate
make docs-clean

# Open in browser
make docs-open
```

### Script Options

```bash
./scripts/generate-docs.sh --help

Options:
  --rust-only    Generate only Rust documentation
  --ts-only      Generate only TypeScript documentation
  --clean        Clean existing docs before generating
  --open         Open documentation in browser after generation
```

## Holochain Zomes

The Mycelix Mail hApp consists of the following zomes:

| Zome | Description |
|------|-------------|
| `messages` | Email storage, delivery, folders, and threading |
| `contacts` | Contact management, groups, and blocking |
| `trust` | Web of trust (MATL algorithm), attestations |
| `keys` | Post-quantum key management and rotation |
| `capabilities` | Mailbox sharing and delegation |
| `federation` | Cross-network routing and bridges |
| `search` | Full-text search indexing |
| `sync` | CRDT-based offline sync |
| `backup` | Encrypted backup and restore |
| `scheduler` | Delayed and recurring sends |
| `audit` | Compliance logging |

Each zome has:
- **Integrity zome**: Entry/link types and validation
- **Coordinator zome**: Business logic and queries

## TypeScript SDK

The SDK provides two clients:

### REST API Client

For applications using the gateway server:

```typescript
import { MycelixClient } from '@mycelix/sdk';

const client = new MycelixClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.mycelix.example.com',
});

const emails = await client.emails.list({ folder: 'inbox' });
```

### Holochain Direct Client

For P2P applications without a gateway:

```typescript
import { createHolochainClient } from '@mycelix/sdk/holochain';

const client = createHolochainClient({
  appId: 'mycelix-mail',
  url: 'ws://localhost:8888',
});

await client.connect();
const emails = await client.messages.getInbox();
```

## Key Concepts

### Web of Trust (MATL)

Mycelix Mail uses the **Mycelix Advanced Trust Logic** algorithm:

- **Direct Trust**: Personal attestations with evidence
- **Transitive Trust**: Network propagation with decay
- **Byzantine Detection**: Sybil and collusion resistance
- **Category Scoring**: Different trust for identity, communication, etc.

### End-to-End Encryption

All emails use strong encryption:

- **Key Exchange**: X25519 or Kyber1024 (post-quantum)
- **Symmetric**: ChaCha20-Poly1305 or AES-256-GCM
- **Signatures**: Ed25519 or Dilithium (post-quantum)

### Decentralized Storage

Data is stored on the Holochain DHT:

- No central server
- Content-addressed entries
- Real-time P2P signals
- Offline-first with CRDT sync

## Contributing to Documentation

We welcome documentation improvements! To contribute:

1. Fork the repository
2. Edit documentation files
3. Run `make docs` to verify generation
4. Submit a pull request

### Documentation Standards

- Use clear, concise language
- Include code examples where helpful
- Keep examples up-to-date with API changes
- Test all code snippets

## Support

- **Issues**: [GitHub Issues](https://github.com/luminous-dynamics/mycelix-mail/issues)
- **Discussions**: [GitHub Discussions](https://github.com/luminous-dynamics/mycelix-mail/discussions)
- **Contributing**: [CONTRIBUTING.md](/CONTRIBUTING.md)

## License

Mycelix Mail is licensed under the MIT License. See [LICENSE](/LICENSE) for details.
