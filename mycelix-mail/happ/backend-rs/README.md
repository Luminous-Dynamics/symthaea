# Mycelix-Mail Rust Backend

Axum-based REST API backend that connects the React frontend to the Holochain DNA.

## Architecture

```
React Frontend
     │
     │ HTTP/WebSocket
     ▼
┌─────────────────┐
│  Axum Backend   │  ← This service
│  (Rust)         │
└────────┬────────┘
         │ AppAgentWebsocket
         ▼
┌─────────────────┐
│   Holochain     │
│   Conductor     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ mycelix_mail    │
│ DNA             │
└─────────────────┘
```

## Key Design Decisions

### DID Resolution via DHT Only

**All DID operations go through the Holochain DHT.** The Python DID Registry is deprecated and should be removed. This ensures:

- True decentralization (no single point of failure)
- Consistency with Holochain's agent-centric model
- No external database dependencies

### Trust Score Caching

Trust scores are cached locally using `moka` (async LRU cache) to reduce DHT query load:

- Configurable TTL (default: 5 minutes)
- Automatic cache invalidation on spam reports
- Batch fetching for efficiency

### JWT Authentication

- Stateless authentication via JWT
- Tokens contain DID and AgentPubKey
- Argon2 password hashing (for local storage)

## API Endpoints

### Authentication (`/api/auth`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/register` | Register new user (creates DID on DHT) |
| POST | `/login` | Login and get JWT token |
| GET | `/me` | Get current user info |
| POST | `/refresh` | Refresh JWT token |

### Emails (`/api/emails`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/` | Send an email |
| GET | `/inbox` | Get inbox (with trust filtering) |
| GET | `/outbox` | Get sent emails |
| GET | `/:id` | Get single email |
| DELETE | `/:id` | Delete email |
| GET | `/:id/thread` | Get thread for email |
| POST | `/:id/spam` | Report as spam |
| POST | `/:id/not-spam` | Mark as not spam |

### Trust (`/api/trust`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/score/:did` | Get trust score |
| POST | `/scores` | Batch get trust scores |
| GET | `/byzantine/:did` | Check Byzantine status |
| GET | `/cross-happ/:did` | Get cross-hApp reputation |
| GET | `/cache/stats` | Get cache statistics |
| POST | `/cache/invalidate/:did` | Invalidate cache entry |

### DID (`/api/did`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/register` | Register DID on DHT |
| GET | `/resolve/:did` | Resolve DID to AgentPubKey |
| GET | `/whoami` | Get current user's DID |

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your values
```

Required variables:
- `JWT_SECRET`: Secret key for JWT signing (generate a secure random string)
- `HOLOCHAIN_URL`: WebSocket URL to Holochain conductor

## Development

### Prerequisites

- Rust 1.75+
- Holochain conductor running
- DNA installed in conductor

### Run

```bash
# Development
cargo run

# Production
cargo build --release
./target/release/mycelix-mail-backend
```

### Test

```bash
cargo test
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `axum` | Web framework |
| `tokio` | Async runtime |
| `holochain_client` | Holochain conductor client |
| `jsonwebtoken` | JWT authentication |
| `moka` | Async cache for trust scores |
| `tower-http` | HTTP middleware (CORS, compression) |
| `tracing` | Logging |

## Replaces

This Rust backend replaces:
- `ui/backend/` (Express.js)
- `happ/did-registry/` (Python SQLite)
- `happ/matl-bridge/` (Python)

All functionality is now handled by:
1. Direct Holochain client calls
2. Trust cache service
3. DHT-based DID resolution
