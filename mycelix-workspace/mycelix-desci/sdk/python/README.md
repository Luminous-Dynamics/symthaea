# Mycelix-Py: Python SDK for Mycelix-DeSci

Official Python SDK for the Mycelix-DeSci decentralized science platform.

## Features

- 🐍 **Pythonic API** - Intuitive, idiomatic Python interface
- ⚡ **Async Support** - Built-in async/await support with httpx
- 📦 **Type Safe** - Full type hints with Pydantic models
- 🔍 **Auto-complete** - Rich IDE support
- 🧪 **Well Tested** - Comprehensive test suite
- 📚 **Documented** - Detailed docstrings and examples

## Installation

```bash
pip install mycelix-py
```

For async support:
```bash
pip install mycelix-py[async]
```

## Quick Start

### Synchronous Usage

```python
from mycelix import MycelixClient, EpistemicTier

# Create client
client = MycelixClient(base_url="http://localhost:8080")

# Create a claim
claim = client.claims.create(
    tier=EpistemicTier.E0,
    content={
        "dataset_hash": "blake3:a1b2c3...",
        "description": "Novel NAD+ supplementation study",
        "category": "longevity",
        "keywords": ["NAD+", "aging", "clinical-trial"],
    },
    creator="dr.alice@university.edu"
)
print(f"Created claim: {claim.id}")

# Get a claim
claim = client.claims.get(claim_id)
print(f"Claim tier: {claim.tier}")

# Query claims
results = client.query.search(
    category="longevity",
    tier=EpistemicTier.E3,
    keywords=["NAD+"]
)
print(f"Found {len(results.claims)} claims")

# Manage trust scores
score = client.trust.get_score("dr.alice@university.edu")
print(f"Trust score: {score.score:.2f}")
```

### Async Usage

```python
import asyncio
from mycelix import AsyncMycelixClient, EpistemicTier

async def main():
    async with AsyncMycelixClient(base_url="http://localhost:8080") as client:
        # Create claim
        claim = await client.claims.create(
            tier=EpistemicTier.E0,
            content={
                "dataset_hash": "blake3:abc123",
                "description": "Research finding",
                "category": "longevity",
                "keywords": ["NAD+"]
            },
            creator="researcher@uni.edu"
        )

        # Add verification
        await client.claims.add_verification(
            claim_id=claim.id,
            verifier="peer@institution.edu",
            signature="hex_signature_here"
        )

        # Query
        results = await client.query.search(category="longevity")
        print(f"Found {results.total_count} claims")

asyncio.run(main())
```

## API Reference

### Client

```python
from mycelix import MycelixClient

client = MycelixClient(
    base_url="http://localhost:8080",
    timeout=30.0,
    api_key=None  # Optional API key for authentication
)
```

### Claims API

```python
# Create claim
claim = client.claims.create(tier, content, creator)

# Get claim
claim = client.claims.get(claim_id)

# List claims
claims = client.claims.list(limit=10, offset=0)

# Add verification
client.claims.add_verification(claim_id, verifier, signature)

# Add provenance
client.claims.add_provenance(claim_id, source, relationship)

# Delete claim
client.claims.delete(claim_id)
```

### Query API

```python
# Search claims
results = client.query.search(
    category="longevity",
    tier=EpistemicTier.E3,
    keywords=["NAD+", "aging"],
    creator="researcher@uni.edu",
    limit=100
)

# Filter claims
results = client.query.filter(
    min_tier=EpistemicTier.E2,
    categories=["longevity", "neuroscience"]
)
```

### Trust API

```python
# Get trust score
score = client.trust.get_score(participant_id)

# Update trust score
client.trust.update_score(participant_id, new_score, confidence)

# Get trust statistics
stats = client.trust.get_stats()
```

## Models

All API models are Pydantic models with full type safety:

```python
from mycelix.models import (
    Claim,
    EpistemicTier,
    ClaimContent,
    Verification,
    Provenance,
    TrustScore,
    QueryResult,
)

# Access claim fields
claim: Claim
print(claim.id)
print(claim.tier)
print(claim.content.description)
print(claim.created_at)
```

## Error Handling

```python
from mycelix import MycelixClient, MycelixError, ClaimNotFoundError

client = MycelixClient()

try:
    claim = client.claims.get("invalid-id")
except ClaimNotFoundError:
    print("Claim not found")
except MycelixError as e:
    print(f"API error: {e}")
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=mycelix --cov-report=html

# Format code
black src/ tests/

# Type check
mypy src/

# Lint
ruff check src/
```

## Examples

See the [examples/](examples/) directory for more usage examples:

- `basic_usage.py` - Simple claim creation and retrieval
- `async_example.py` - Async client usage
- `research_workflow.py` - Complete research publication workflow
- `batch_operations.py` - Bulk operations
- `trust_management.py` - Trust score management

## License

MIT License - see [LICENSE](../../LICENSE) for details

## Contributing

Contributions welcome! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## Support

- GitHub Issues: https://github.com/Luminous-Dynamics/mycelix-desci/issues
- Documentation: https://github.com/Luminous-Dynamics/mycelix-desci/tree/main/docs
