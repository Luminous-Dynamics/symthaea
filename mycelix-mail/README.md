# Mycelix-Mail

**Decentralized email on Holochain with MATL trust-based spam filtering**

[![CI](https://github.com/Luminous-Dynamics/Mycelix-Mail/actions/workflows/ci.yml/badge.svg)](https://github.com/Luminous-Dynamics/Mycelix-Mail/actions/workflows/ci.yml)
[![Status](https://img.shields.io/badge/status-beta-blue)](https://github.com/Luminous-Dynamics/Mycelix-Mail)
[![Holochain](https://img.shields.io/badge/holochain-0.5.x-purple)](https://holochain.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

Mycelix-Mail is a fully decentralized email system built on Holochain. Unlike traditional email:

- **No corporate servers** - Your data stays on your device and the DHT
- **Trust-based spam filtering** - MATL reputation system instead of keyword blacklists
- **End-to-end encryption** - X25519 key exchange + ChaCha20-Poly1305
- **Agent-centric** - You own your identity and data via DIDs

## Architecture

```
                    React Frontend (Port 5173)
                           │
                           │ HTTP/WebSocket
                           ▼
┌──────────────────────────────────────────────────────────┐
│                 Axum Backend (Port 3001)                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │  Auth/JWT   │ │ Trust Cache │ │  IPFS Storage       │ │
│  │  Middleware │ │ (Moka LRU)  │ │  (Message Bodies)   │ │
│  └─────────────┘ └─────────────┘ └─────────────────────┘ │
└──────────────────────────────────────────────────────────┘
                           │
                           │ AppAgentWebsocket
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  Holochain Conductor                     │
│  ┌───────────────────────────────────────────────────┐   │
│  │              Mycelix-Mail DNA                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│  │  │ Integrity   │  │ mail_       │  │ trust_    │  │   │
│  │  │ Zome        │  │ messages    │  │ filter    │  │   │
│  │  │             │  │ Zome        │  │ Zome      │  │   │
│  │  └─────────────┘  └─────────────┘  └───────────┘  │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### Project Structure

```
Mycelix-Mail/
├── happ/
│   ├── dna/                    # Holochain DNA
│   │   ├── integrity/          # Entry validation
│   │   └── zomes/
│   │       ├── mail_messages/  # Email operations
│   │       └── trust_filter/   # MATL spam filtering
│   └── backend-rs/             # Rust/Axum API server
│       ├── src/
│       │   ├── routes/         # REST + WebSocket endpoints
│       │   ├── services/       # Holochain, storage, crypto
│       │   └── middleware/     # JWT auth, rate limiting
│       └── Cargo.toml
├── ui/
│   └── frontend/               # React + TypeScript + Vite
│       └── src/
│           ├── components/     # 30+ UI components
│           ├── services/       # API + WebSocket clients
│           └── store/          # Zustand state
├── .github/workflows/ci.yml    # CI/CD pipeline
├── docker-compose.yml          # Full stack deployment
└── flake.nix                   # Nix development environment
```

## Features

### Core Email
- Send/receive emails via Holochain DHT
- Email threading with conversation view
- Labels, folders, and smart filtering
- Draft autosave and templates
- Advanced search (from:, to:, subject:, has:, is:, label:)
- Keyboard shortcuts (Gmail-style j/k navigation)

### Trust & Security
- **MATL trust scoring** - Reputation-weighted spam detection
- **E2E encryption** - X25519 + ChaCha20-Poly1305
- **Epistemic tiers** - Classify message verifiability (0-4)
- **Byzantine detection** - Identify malicious actors
- **Rate limiting** - 60 req/min per IP

### Real-time Updates
- WebSocket events for new mail
- Trust score change notifications
- Connection status monitoring

### API Documentation
- Swagger UI at `/api-docs`
- OpenAPI JSON at `/api-docs/openapi.json`

## Quick Start

### Prerequisites
- [Nix](https://nixos.org/download.html) with flakes enabled
- Or: Docker + Docker Compose

### Development (Nix)

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/Mycelix-Mail.git
cd Mycelix-Mail

# Enter Nix environment
nix develop

# Setup environment files
just setup

# Start all services (Holochain, Backend, Frontend)
just dev
```

### Development (Docker)

```bash
# Clone and enter directory
git clone https://github.com/Luminous-Dynamics/Mycelix-Mail.git
cd Mycelix-Mail

# Start full stack
docker compose up -d

# With Holochain (requires conductor setup)
docker compose --profile with-holochain up -d
```

### Access Points

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:3001/api |
| API Documentation | http://localhost:3001/api-docs |
| Health Check | http://localhost:3001/health |

## API Overview

### Authentication
```bash
# Register
curl -X POST http://localhost:3001/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"did": "did:key:z6Mk...", "password": "secret"}'

# Login
curl -X POST http://localhost:3001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"did": "did:key:z6Mk...", "password": "secret"}'
```

### Emails
```bash
# Send email
curl -X POST http://localhost:3001/api/emails \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "to_did": "did:key:z6Mk...",
    "subject": "Hello from Mycelix!",
    "body": "This message is end-to-end encrypted.",
    "epistemic_tier": "Testimonial"
  }'

# Get inbox
curl http://localhost:3001/api/emails/inbox \
  -H "Authorization: Bearer <token>"
```

### Trust Scores
```bash
# Get trust score
curl http://localhost:3001/api/trust/did:key:z6Mk... \
  -H "Authorization: Bearer <token>"

# Check Byzantine status
curl http://localhost:3001/api/trust/did:key:z6Mk.../byzantine \
  -H "Authorization: Bearer <token>"
```

## Configuration

Environment variables for the backend (`happ/backend-rs/.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `3001` | Server port |
| `HOLOCHAIN_URL` | `ws://localhost:4444` | Conductor WebSocket |
| `JWT_SECRET` | (required) | Secret for JWT signing |
| `JWT_EXPIRATION_HOURS` | `24` | Token lifetime |
| `TRUST_CACHE_TTL_SECS` | `300` | Trust cache TTL |
| `DEFAULT_MIN_TRUST` | `0.3` | Minimum trust threshold |
| `BYZANTINE_THRESHOLD` | `0.2` | Byzantine detection threshold |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed CORS origins |
| `RATE_LIMIT_RPM` | `100` | Requests per minute |

## Testing

```bash
# Run all tests
just test

# DNA tests only
cd happ/dna && cargo test

# Backend tests only
cd happ/backend-rs && cargo test

# Frontend tests only
cd ui/frontend && npm test
```

**Test Coverage:**
- 14 backend unit tests (crypto, storage, validation, rate limiting)
- Frontend component tests with Vitest + Testing Library

## Status

**Beta** - Core features complete, ready for testing

### Complete
- [x] Holochain DNA with mail + trust zomes
- [x] Rust/Axum backend with full API
- [x] E2E encryption (X25519 + ChaCha20)
- [x] MATL trust scoring integration
- [x] React frontend with full UI
- [x] WebSocket real-time updates
- [x] Docker deployment
- [x] OpenAPI documentation
- [x] CI/CD pipeline

### Planned
- [ ] Mobile app (Tauri)
- [ ] SMTP bridge for legacy email
- [ ] Federation with other Mycelix apps
- [ ] Governance integration

## Related Projects

- [Mycelix-Core](https://github.com/Luminous-Dynamics/Mycelix-Core) - MATL trust layer
- [mycelix.net](https://mycelix.net) - Protocol documentation

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `just test`
4. Run lints: `just lint`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## License

MIT - See [LICENSE](LICENSE)

---

*Part of the [Mycelix Protocol](https://mycelix.net) ecosystem*
