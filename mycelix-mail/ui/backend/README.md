# Express Backend (DEPRECATED)

> **DEPRECATED**: This Express.js backend has been replaced by the Rust/Axum backend at `happ/backend-rs/`.
>
> This code is kept for reference only. Do not use for new development.

## Replacement

The new backend provides:

- **Better performance** - Rust/Axum vs Node.js/Express
- **Type safety** - Native Rust types vs TypeScript
- **Direct Holochain integration** - AppAgentWebsocket client
- **IPFS storage** - Message body storage on content-addressed network
- **Moka caching** - High-performance async trust cache
- **OpenAPI documentation** - Swagger UI at `/api-docs`

## Migration Guide

| Old (Express) | New (Axum) |
|---------------|------------|
| `POST /api/auth/login` | `POST /api/auth/login` |
| `POST /api/auth/register` | `POST /api/auth/register` |
| `GET /api/emails` | `GET /api/emails/inbox` |
| `POST /api/emails` | `POST /api/emails` |
| `GET /api/folders` | (removed - folders via labels) |
| `GET /api/trust/:did` | `GET /api/trust/:did` |
| WebSocket `/ws` | WebSocket `/ws/events` |

## New Backend Location

```
happ/backend-rs/
├── src/
│   ├── routes/          # REST endpoints
│   ├── services/        # Holochain, storage, crypto
│   ├── middleware/      # JWT, rate limiting
│   └── main.rs          # Entry point
├── Cargo.toml
└── README.md            # Full documentation
```

## Running the New Backend

```bash
cd happ/backend-rs
cp .env.example .env
# Edit .env with your JWT_SECRET
cargo run --release
```

API docs: http://localhost:3001/api-docs

---

**This directory will be removed in a future release.**
