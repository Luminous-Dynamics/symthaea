# Mycelix-Mail Development Guide

This guide helps you set up a development environment for Mycelix-Mail.

## Quick Start

### Option 1: Nix (Recommended)

If you have Nix with flakes enabled:

```bash
# Enter development shell with all tools
nix develop

# Run all services
just dev
```

### Option 2: Manual Setup

```bash
# Run the setup script
./scripts/setup-dev.sh

# Start backend
cd happ/backend-rs && cargo run

# In another terminal, start frontend
cd ui/frontend && npm run dev
```

## Project Structure

```
Mycelix-Mail/
├── happ/
│   ├── backend-rs/      # Rust/Axum REST API backend
│   │   ├── src/
│   │   │   ├── main.rs           # Entry point
│   │   │   ├── config.rs         # Configuration
│   │   │   ├── routes/           # API endpoints
│   │   │   ├── services/         # Business logic
│   │   │   │   ├── crypto.rs     # E2E encryption
│   │   │   │   ├── storage.rs    # IPFS/local storage
│   │   │   │   ├── holochain.rs  # Holochain client
│   │   │   │   └── trust_cache.rs
│   │   │   ├── middleware/       # Auth, etc.
│   │   │   ├── types.rs          # API types
│   │   │   └── validation.rs     # Input validation
│   │   └── Cargo.toml
│   │
│   ├── dna/             # Holochain DNA
│   │   ├── integrity/   # Entry definitions, validation
│   │   └── zomes/       # Coordinator zomes
│   │       ├── mail_messages/
│   │       └── trust_filter/
│   │
│   ├── cli/             # Command-line interface
│   └── _deprecated/     # Old Python services (reference only)
│
├── ui/
│   ├── frontend/        # React/TypeScript frontend
│   │   ├── src/
│   │   │   ├── components/
│   │   │   ├── services/
│   │   │   ├── store/
│   │   │   └── types/
│   │   └── package.json
│   └── backend/         # (Old Express backend - deprecated)
│
├── flake.nix           # Nix development environment
├── justfile            # Development commands
└── DEVELOPMENT.md      # This file
```

## Development Commands

Using `just` (install with `cargo install just`):

| Command | Description |
|---------|-------------|
| `just dev` | Start all services |
| `just build` | Build all components |
| `just test` | Run all tests |
| `just fmt` | Format all code |
| `just lint` | Check code style |
| `just clean` | Remove build artifacts |

### Component-specific commands:

```bash
# Backend
just backend        # Run backend
just backend-watch  # Run with auto-reload
just test-backend   # Run backend tests

# Frontend
just frontend       # Run dev server
just test-frontend  # Run frontend tests

# DNA
just build-dna      # Build Holochain zomes
just test-dna       # Run DNA tests
```

## Configuration

### Backend (happ/backend-rs/.env)

```env
# Server
HOST=0.0.0.0
PORT=3001

# Holochain
HOLOCHAIN_URL=ws://localhost:4444

# JWT (REQUIRED - generate securely for production)
JWT_SECRET=your-secret-here

# CORS
CORS_ORIGINS=http://localhost:5173

# IPFS (optional)
IPFS_API_URL=http://localhost:5001
```

### Frontend (ui/frontend/.env)

```env
VITE_API_URL=http://localhost:3001
VITE_WS_URL=ws://localhost:3001
```

## Architecture

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/login` | POST | Authenticate user |
| `/api/auth/register` | POST | Create account |
| `/api/emails/` | POST | Send email |
| `/api/emails/inbox` | GET | Get inbox with filtering |
| `/api/emails/outbox` | GET | Get sent emails |
| `/api/emails/:id` | GET | Get single email |
| `/api/emails/:id/spam` | POST | Report as spam |
| `/api/trust/:did` | GET | Get trust score |
| `/api/did/register` | POST | Register DID on DHT |
| `/api/did/resolve/:did` | GET | Resolve DID to pubkey |
| `/ws/events` | WebSocket | Real-time updates |
| `/health` | GET | Health check |

### Security Features

1. **E2E Encryption**: ChaCha20-Poly1305 with X25519 key exchange
2. **Input Validation**: DID format, trust score bounds, content limits
3. **JWT Authentication**: Token-based auth with configurable expiration
4. **Trust Filtering**: MATL-based spam prevention

### Storage

- **Message bodies**: IPFS (with local fallback)
- **Message metadata**: Holochain DHT
- **Trust scores**: Cached with TTL, synced from MATL

## Testing

```bash
# Run all tests
just test

# Run specific test suites
cargo test -p mycelix-mail-backend         # Backend
cargo test -p mycelix_mail_integrity       # DNA integrity
cargo test -p mail_messages                # Mail zome
cargo test -p trust_filter                 # Trust zome

# With coverage
just test-coverage
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `just test`
5. Format code: `just fmt`
6. Lint code: `just lint`
7. Commit: `git commit -m "Add my feature"`
8. Push: `git push origin feature/my-feature`
9. Create a Pull Request

### Code Style

- **Rust**: Use `cargo fmt` and `cargo clippy`
- **TypeScript**: Use ESLint and Prettier
- **Commits**: Use conventional commit messages

## Troubleshooting

### "IPFS not available"

The backend will use local storage as fallback. To enable IPFS:

```bash
# Install and start IPFS
ipfs init
ipfs daemon
```

### "Holochain connection failed"

The backend uses stub mode when Holochain isn't running. For full functionality:

```bash
# Using Nix
nix develop
holochain sandbox generate --run=4444 workdir
```

### "JWT_SECRET not set"

Create a `.env` file:

```bash
cd happ/backend-rs
cp .env.example .env
# Edit .env to set JWT_SECRET
```

## Resources

- [Holochain Documentation](https://developer.holochain.org/)
- [Axum Framework](https://docs.rs/axum/)
- [MATL Trust Layer](../Mycelix-Core/docs/)
