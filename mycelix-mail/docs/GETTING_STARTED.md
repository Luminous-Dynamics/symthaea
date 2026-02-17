# Getting Started with Mycelix Mail

## Prerequisites

- **Rust** 1.75+ with `wasm32-unknown-unknown` target
- **Node.js** 20+
- **Holochain CLI** (`hc`) v0.3+
- **Lair Keystore** v0.4+
- **Docker** (optional, for containerized deployment)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/luminous-dynamics/mycelix-mail.git
cd mycelix-mail
```

### 2. Install Dependencies

```bash
# Install all dependencies
make install

# Or manually:
cd holochain && cargo fetch
cd ../holochain/client && npm install
cd ../../ui/frontend && npm install
```

### 3. Build the Project

```bash
# Build everything
make build

# Or step by step:
make build-zomes      # Build Holochain zomes
make build-client     # Build TypeScript client
make build-ui         # Build UI frontend
```

### 4. Run Development Environment

```bash
# Start everything
make dev

# Or manually:
# Terminal 1: Start Holochain
cd holochain && hc sandbox generate --run 8888

# Terminal 2: Start UI
cd ui/frontend && npm run dev
```

### 5. Open the Application

Visit `http://localhost:5173` in your browser.

## Project Structure

```
mycelix-mail/
├── holochain/                 # Holochain backend
│   ├── Cargo.toml            # Workspace configuration
│   ├── zomes/                # Zome implementations
│   │   ├── messages/         # Email messages
│   │   │   ├── integrity/    # Entry/link types
│   │   │   └── coordinator/  # Business logic
│   │   ├── trust/            # MATL trust system
│   │   ├── sync/             # CRDT synchronization
│   │   ├── federation/       # Cross-cell communication
│   │   └── ...
│   ├── client/               # TypeScript client
│   │   ├── src/
│   │   │   ├── zomes/        # Zome client wrappers
│   │   │   ├── services/     # Application services
│   │   │   ├── components/   # React components
│   │   │   ├── vue/          # Vue composables
│   │   │   └── ...
│   │   └── tests/
│   └── workdir/              # DNA/hApp packaging
├── ui/
│   └── frontend/             # React frontend
├── docker/                   # Docker configuration
├── docs/                     # Documentation
└── Makefile                  # Build automation
```

## Development Workflow

### Running Tests

```bash
# All tests
make test

# Rust zome tests
make test-zomes

# TypeScript tests
make test-client

# E2E tests
make test-e2e

# With coverage
cd holochain/client && npm run test:coverage
```

### Code Quality

```bash
# Lint all code
make lint

# Format all code
make format

# Full check (lint + typecheck)
make check
```

### Storybook

```bash
# Run Storybook
cd holochain/client && npm run storybook

# Build static Storybook
npm run storybook:build
```

## Configuration

### Environment Variables

```bash
# Holochain connection
VITE_HOLOCHAIN_URL=ws://localhost:8888
VITE_APP_ID=mycelix-mail

# Development
RUST_LOG=info
```

### Client Configuration

```typescript
import { connectToHolochain } from '@mycelix/holochain-client';

await connectToHolochain({
  appWebsocketUrl: 'ws://localhost:8888',
  adminWebsocketUrl: 'ws://localhost:8889',
  appId: 'mycelix-mail',
  roleName: 'mycelix_mail',
  timeout: 30000,
});
```

## Using the Client Library

### Installation

```bash
npm install @mycelix/holochain-client
```

### Basic Usage

```typescript
import { MycelixMailClient, createMycelixMailClient } from '@mycelix/holochain-client';
import { AppWebsocket } from '@holochain/client';

// Connect to Holochain
const appWebsocket = await AppWebsocket.connect('ws://localhost:8888');

// Create client
const client = createMycelixMailClient(appWebsocket);
await client.initialize();

// Get inbox
const emails = await client.getInboxWithTrust(50);

// Send email
await client.sendEmailWithTrustCheck(
  recipientPubKey,
  'Hello!',
  'This is a test email.',
  { requireTrustLevel: 0.3 }
);

// Create trust attestation
await client.trust.createAttestation({
  target: agentPubKey,
  trust_level: 0.8,
  context: 'verified colleague',
  category: 'Communication',
});
```

### React Hooks

```tsx
import {
  HolochainProvider,
  useInbox,
  useSendEmail,
  useTrustNetwork,
  useContacts,
} from '@mycelix/holochain-client/react';

function App() {
  return (
    <HolochainProvider
      autoConnect
      loadingComponent={<Loading />}
      errorComponent={(error, retry) => <Error error={error} retry={retry} />}
    >
      <MailApp />
    </HolochainProvider>
  );
}

function MailApp() {
  const { emails, isLoading, refresh } = useInbox();
  const { sendEmail, isSending } = useSendEmail();
  const { nodes, edges } = useTrustNetwork();

  // ...
}
```

### Vue Composables

```vue
<script setup>
import { useMessages, useTrust, useContacts } from '@mycelix/holochain-client/vue';

const { messages, loading, refresh } = useMessages();
const { trustNetwork, createAttestation } = useTrust();
const { contacts, addContact } = useContacts();
</script>
```

## Docker Deployment

### Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production

```bash
# Build production images
docker-compose -f docker/docker-compose.yml build

# Start with environment
POSTGRES_PASSWORD=secret JWT_SECRET=long-secret docker-compose up -d
```

## Troubleshooting

### Common Issues

**1. Holochain connection failed**
```bash
# Check if conductor is running
hc sandbox list

# Restart sandbox
hc sandbox clean && hc sandbox generate --run 8888
```

**2. WASM build errors**
```bash
# Ensure wasm target is installed
rustup target add wasm32-unknown-unknown

# Clear cargo cache
cargo clean && cargo build --target wasm32-unknown-unknown
```

**3. Node module issues**
```bash
# Clear node_modules
rm -rf node_modules package-lock.json
npm install
```

### Getting Help

- **GitHub Issues**: https://github.com/luminous-dynamics/mycelix-mail/issues
- **Documentation**: https://docs.mycelix.mail
- **Discord**: https://discord.gg/mycelix

## Next Steps

1. Read the [Architecture Guide](./ARCHITECTURE.md)
2. Explore the [API Reference](./API.md)
3. Review the [Security Documentation](./SECURITY.md)
4. Check out [Contributing Guidelines](../CONTRIBUTING.md)
