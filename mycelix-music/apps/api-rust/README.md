# Mycelix Music API (Rust)

High-performance Rust backend for Mycelix Music platform.

## Why Rust?

1. **Ecosystem Consistency** - Matches Mycelix-Core and other Mycelix projects
2. **Performance** - Sub-millisecond latencies for music streaming
3. **Safety** - Memory safety and strong typing
4. **Holochain Ready** - Direct integration path to Holochain zomes

## Stack

- **Web Framework**: Axum 0.7
- **Async Runtime**: Tokio
- **Database**: PostgreSQL via SQLx
- **Caching**: Redis
- **Storage**: IPFS
- **Blockchain**: Ethers-rs for Gnosis Chain

## Getting Started

```bash
# Enter nix shell (recommended)
nix develop

# Or install dependencies manually
cargo build

# Set environment variables
cp ../../.env.example .env

# Run migrations
sqlx migrate run

# Start the server
cargo run
```

## API Endpoints

### Songs
- `GET /api/songs` - List songs
- `POST /api/songs` - Create song
- `GET /api/songs/:id` - Get song
- `POST /api/songs/:id/play` - Record play

### Artists
- `GET /api/artists/:address` - Get artist profile
- `GET /api/artists/:address/songs` - Get artist's songs

### Analytics
- `GET /api/analytics/artist/:address` - Artist earnings
- `GET /api/analytics/song/:id` - Song performance
- `GET /api/analytics/top-songs` - Leaderboard

### Strategies
- `GET /api/strategies` - List economic strategies
- `POST /api/strategies/:id/preview` - Preview splits

### Uploads
- `POST /api/upload` - Upload file to IPFS

## Architecture

```
src/
├── main.rs           # Server setup
├── routes/           # HTTP handlers
│   ├── songs.rs
│   ├── artists.rs
│   ├── analytics.rs
│   ├── uploads.rs
│   └── strategies.rs
├── services/         # Business logic
│   ├── ipfs.rs       # IPFS integration
│   ├── blockchain.rs # Contract calls
│   └── cache.rs      # Redis caching
└── models/           # Data structures
    └── mod.rs
```

## Integration with Mycelix-Core

```toml
# Coming soon: shared crates
[dependencies]
zerotrustml = { path = "../../../Mycelix-Core/0TML" }
```

This enables:
- PoGQ for CDN node quality scoring
- Byzantine detection for distributed operations
- Shared cryptographic primitives

## Migration from Express

The Rust API is designed to be a drop-in replacement for the Express API:
- Same endpoints
- Same request/response formats
- Same database schema

Migrate gradually by routing specific endpoints to Rust.

## Performance Targets

| Metric | Express | Rust | Improvement |
|--------|---------|------|-------------|
| Latency (p99) | 50ms | 5ms | 10x |
| Throughput | 1K rps | 50K rps | 50x |
| Memory | 200MB | 20MB | 10x |

## License

MIT
