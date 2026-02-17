# Quick Start Guide

Get up and running with Mycelix-DeSci in **5 minutes**! 🚀

## What is Mycelix-DeSci?

Mycelix-DeSci is a decentralized science platform that provides:
- **Cryptographic verification** of scientific claims
- **Tiered epistemic trust** (E0-E4) based on peer review
- **Provenance tracking** for research data
- **Trust networks** for researcher reputation

## Prerequisites

- **Docker** and **Docker Compose** (recommended), OR
- **Rust** 1.75+ (for building from source)
- **Git**

## Option 1: Docker (Recommended)

### 1. Clone the repository

```bash
git clone https://github.com/Luminous-Dynamics/mycelix-desci
cd mycelix-desci
```

### 2. Start the API server

```bash
docker-compose up -d
```

That's it! The API is now running at `http://localhost:8080`

### 3. Verify it's working

```bash
curl http://localhost:8080/health
```

You should see:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  ...
}
```

### 4. View API documentation

Open your browser to: `http://localhost:8080/docs`

You'll see interactive Swagger UI documentation where you can try all API endpoints!

## Option 2: Build from Source

### 1. Clone and build

```bash
git clone https://github.com/Luminous-Dynamics/mycelix-desci
cd mycelix-desci
cargo build --release
```

### 2. Run the API server

```bash
cargo run --release --package mycelix-desci-api
```

### 3. Run the CLI tool

```bash
cargo run --release --package mycelix-cli -- --help
```

## Your First API Call

Let's create your first scientific claim!

### 1. Create a claim

```bash
curl -X POST http://localhost:8080/api/v1/claims \
  -H "Content-Type: application/json" \
  -d '{
    "tier": "E0",
    "content": {
      "dataset_hash": "blake3:example123",
      "description": "My first scientific claim on Mycelix-DeSci!",
      "category": "test",
      "keywords": ["demo", "first-claim"]
    },
    "creator": "you@example.com"
  }'
```

You'll get a response like:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "tier": "E0",
  "content": {
    "dataset_hash": "blake3:example123",
    "description": "My first scientific claim on Mycelix-DeSci!",
    ...
  },
  ...
}
```

### 2. Retrieve your claim

Copy the `id` from the response and use it here:

```bash
curl http://localhost:8080/api/v1/claims/550e8400-e29b-41d4-a716-446655440000
```

### 3. Search for claims

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "category": "test",
    "page": 1,
    "page_size": 10
  }'
```

## Using the CLI Tool

The CLI provides a more user-friendly interface!

### 1. Build the CLI

```bash
cargo build --release --package mycelix-cli
```

### 2. Create a claim from a file

Create `my-claim.json`:
```json
{
  "tier": "E0",
  "content": {
    "dataset_hash": "blake3:mydata123",
    "description": "Groundbreaking longevity research",
    "category": "longevity",
    "keywords": ["aging", "NAD+", "clinical-trial"]
  },
  "creator": "researcher@uni.edu"
}
```

Then create the claim:
```bash
./target/release/mycelix claims create my-claim.json
```

### 3. Search claims

```bash
./target/release/mycelix query search --category longevity
```

### 4. Check system health

```bash
./target/release/mycelix system health
```

### 5. Get help

```bash
./target/release/mycelix --help
./target/release/mycelix claims --help
```

## Try the Examples

We've included comprehensive examples showing real-world usage:

### 1. Complete research publication workflow

```bash
cargo run --example research_publication_workflow
```

This demonstrates:
- Creating a claim from research data
- Adding provenance
- Collecting peer reviews
- Upgrading epistemic tier (E0 → E4)

### 2. Data integrity verification

```bash
cargo run --example data_integrity_pipeline
```

Shows how to verify dataset integrity using cryptographic hashes.

### 3. Simple API usage

```bash
cargo run --example simple_api_usage
```

Basic API operations for getting started.

## Understanding Epistemic Tiers

Mycelix-DeSci uses a tiered verification system:

| Tier | Verifications | Trust Level | Description |
|------|--------------|-------------|-------------|
| **E0** | 0 | Unverified | Initial claim, not yet reviewed |
| **E1** | 1-2 | Low | Some peer review |
| **E2** | 3 | Medium | Multiple reviewers agree |
| **E3** | 4 | High | Strong consensus |
| **E4** | 5+ | Highest | Highly verified and trusted |

Claims automatically upgrade tiers as they receive verifications!

## API Endpoints Overview

### Claims
- `POST /api/v1/claims` - Create a new claim
- `GET /api/v1/claims/{id}` - Get claim by ID
- `PUT /api/v1/claims/{id}/verify` - Add verification
- `PUT /api/v1/claims/{id}/provenance` - Add provenance

### Query
- `POST /api/v1/query` - Search claims
- `GET /api/v1/query/categories` - List categories
- `GET /api/v1/query/stats` - Get statistics

### Trust
- `GET /api/v1/trust/{participant}` - Get trust score
- `PUT /api/v1/trust/{participant}` - Update trust score
- `GET /api/v1/trust/stats` - Network statistics

### System
- `GET /api/v1/system/health` - Health check
- `GET /api/v1/system/metrics` - System metrics
- `GET /api/v1/system/version` - Version info

## Configuration

### API Server

Environment variables:
```bash
PORT=8080                      # API port
RUST_LOG=mycelix_api=info      # Log level
CORS_ORIGINS=*                 # CORS settings
```

### CLI Tool

Config file at `~/.mycelix/config.toml`:
```toml
api_url = "http://localhost:8080"
output_format = "table"
verbose = false
```

Or use environment variables:
```bash
export MYCELIX_API_URL=http://localhost:8080
```

## Next Steps

Now that you're up and running:

1. 📖 **Read the [API Reference](API_REFERENCE.md)** for detailed endpoint documentation
2. 🔍 **Explore the [examples/](../examples/)** directory for more use cases
3. 🚀 **Check out the [Deployment Guide](DEPLOYMENT.md)** for production setup
4. 👥 **Join the community** and contribute!

## Troubleshooting

### "Connection refused" errors

Make sure the API server is running:
```bash
docker-compose ps
# OR
curl http://localhost:8080/health
```

### CLI can't find API

Check your API URL:
```bash
./target/release/mycelix --api-url http://localhost:8080 system health
```

### Port already in use

Change the port in `docker-compose.yml` or set the `PORT` environment variable.

## Getting Help

- **Documentation**: Check `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: [GitHub Issues](https://github.com/Luminous-Dynamics/mycelix-desci/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Luminous-Dynamics/mycelix-desci/discussions)

## What's Next?

You now know how to:
- ✅ Run the Mycelix-DeSci API server
- ✅ Create and retrieve scientific claims
- ✅ Search for claims
- ✅ Use the CLI tool
- ✅ Run examples

Ready to dive deeper? Check out the full documentation in the `docs/` directory!

---

**Happy decentralized science! 🔬✨**
