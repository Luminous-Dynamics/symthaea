# 🗄️ Zero-TrustML Backend Selection Guide

## Overview

Zero-TrustML Phase 10 Coordinator now supports **modular storage backends**, allowing you to choose the storage system that best fits your deployment requirements.

## Available Backends

### 1. PostgreSQL Backend ⭐ **Recommended for Most Deployments**

**Best For:**
- Enterprise deployments
- Production systems requiring ACID guarantees
- Complex queries and analytics
- HIPAA/GDPR compliance needs
- High-performance scenarios

**Pros:**
- ✅ Battle-tested technology (30+ years)
- ✅ ACID transactions
- ✅ Complex SQL queries for analytics
- ✅ Excellent tooling and monitoring
- ✅ Built-in replication and backup
- ✅ Strong consistency guarantees

**Cons:**
- ❌ Requires database server setup
- ❌ Single point of failure (without replication)
- ❌ Centralized architecture

**Configuration:**
```python
from zerotrustml.core.phase10_coordinator import Phase10Config

config = Phase10Config(
    postgres_enabled=True,
    postgres_host="localhost",
    postgres_port=5432,
    postgres_db="zerotrustml",
    postgres_user="zerotrustml",
    postgres_password="your_password",
    storage_strategy="primary"
)
```

---

### 2. Holochain Backend 🌐 **For Decentralized Deployments**

**Best For:**
- Fully decentralized FL networks
- Web3/crypto FL applications
- Censorship-resistant systems
- When trustlessness is critical
- Eliminating single points of failure

**Pros:**
- ✅ Fully decentralized (DHT)
- ✅ Immutable audit trail
- ✅ Agent-centric architecture
- ✅ No central server needed
- ✅ Intrinsic data integrity
- ✅ Censorship-resistant

**Cons:**
- ❌ More complex setup
- ❌ Requires Holochain conductor
- ❌ Limited query capabilities
- ❌ Eventual consistency
- ❌ Experimental for FL (cutting edge)

**Configuration:**
```python
config = Phase10Config(
    holochain_enabled=True,
    holochain_admin_url="ws://localhost:8888",
    holochain_app_url="ws://localhost:8889",
    holochain_app_id="zerotrustml",
    storage_strategy="primary"
)
```

---

### 3. LocalFile Backend 🧪 **For Testing & Development**

**Best For:**
- Unit testing
- Development
- Quick prototypes
- Demo mode
- CI/CD pipelines (isolated tests)

**Pros:**
- ✅ No database required
- ✅ Human-readable JSON files
- ✅ Fast iteration
- ✅ Easy debugging
- ✅ Git-friendly (can commit test data)
- ✅ Zero setup time

**Cons:**
- ❌ Not suitable for production
- ❌ No concurrent access safety
- ❌ Limited performance
- ❌ No transactions
- ❌ No integrity guarantees

**Configuration:**
```python
config = Phase10Config(
    localfile_enabled=True,
    localfile_data_dir="/tmp/zerotrustml_data",
    storage_strategy="primary"
)
```

---

### 4. Blockchain Backend 🔗 **Coming Soon**

**Best For:**
- Public blockchain FL networks
- Token-based incentive systems
- Smart contract integration
- Maximum transparency

**Status:** Planned for future release

**Supported Chains (Planned):**
- Ethereum
- Polkadot
- Cosmos
- Polygon

---

## Storage Strategies

Choose how data is written across multiple backends:

### Strategy: `"primary"` (Default)
**Use When:** Single backend or performance is critical

- Writes to **first backend only**
- Fastest performance
- Lowest redundancy
- Good for single-backend deployments

```python
config = Phase10Config(
    postgres_enabled=True,
    storage_strategy="primary"
)
```

### Strategy: `"all"`
**Use When:** Maximum redundancy is needed

- Writes to **all backends in parallel**
- Highest redundancy
- Survives any single backend failure
- Good for critical data

```python
config = Phase10Config(
    postgres_enabled=True,
    holochain_enabled=True,
    localfile_enabled=True,  # For local backup
    storage_strategy="all"
)
```

### Strategy: `"quorum"`
**Use When:** Balanced redundancy + performance

- Writes to **majority of backends**
- Medium redundancy
- Good fault tolerance
- Balanced performance

```python
config = Phase10Config(
    postgres_enabled=True,
    holochain_enabled=True,
    localfile_enabled=True,
    storage_strategy="quorum"  # 2/3 must succeed
)
```

---

## Common Deployment Scenarios

### Scenario 1: Enterprise Production

**Requirements:**
- High reliability
- ACID transactions
- Compliance (HIPAA/GDPR)
- Analytics support

**Recommended Configuration:**
```python
config = Phase10Config(
    # PostgreSQL only
    postgres_enabled=True,
    postgres_host="db.company.internal",
    postgres_port=5432,
    postgres_db="zerotrustml_prod",

    # ZK-PoC for privacy
    zkpoc_enabled=True,
    zkpoc_hipaa_mode=True,
    zkpoc_gdpr_mode=True,

    # Single backend
    storage_strategy="primary"
)
```

### Scenario 2: Decentralized FL Network

**Requirements:**
- No central authority
- Censorship resistance
- Global participant distribution
- Immutable audit trail

**Recommended Configuration:**
```python
config = Phase10Config(
    # Holochain only
    holochain_enabled=True,
    holochain_admin_url="ws://localhost:8888",
    holochain_app_url="ws://localhost:8889",

    # ZK-PoC for quality
    zkpoc_enabled=True,

    # Single backend
    storage_strategy="primary"
)
```

### Scenario 3: Hybrid (Best of Both Worlds)

**Requirements:**
- Fast queries (PostgreSQL)
- Immutable audit (Holochain)
- Maximum reliability

**Recommended Configuration:**
```python
config = Phase10Config(
    # Both backends
    postgres_enabled=True,
    postgres_host="localhost",
    postgres_port=5432,
    postgres_db="zerotrustml",

    holochain_enabled=True,
    holochain_admin_url="ws://localhost:8888",
    holochain_app_url="ws://localhost:8889",

    # ZK-PoC for privacy
    zkpoc_enabled=True,

    # Write to both
    storage_strategy="all"
)
```

### Scenario 4: Testing & Development

**Requirements:**
- Fast iteration
- No infrastructure
- Easy debugging

**Recommended Configuration:**
```python
config = Phase10Config(
    # LocalFile only
    localfile_enabled=True,
    localfile_data_dir="./test_data",

    # ZK-PoC optional
    zkpoc_enabled=True,

    # Single backend
    storage_strategy="primary"
)
```

---

## Performance Comparison

| Backend | Write Speed | Read Speed | Query Capability | Setup Complexity |
|---------|------------|-----------|-----------------|------------------|
| **PostgreSQL** | Fast (10ms) | Very Fast (1ms) | Excellent (SQL) | Medium |
| **Holochain** | Medium (50ms) | Medium (30ms) | Limited | High |
| **LocalFile** | Very Fast (1ms) | Very Fast (0.5ms) | Basic (scan) | Zero |
| **Blockchain** | Slow (1-10s) | Fast (100ms) | Limited | High |

---

## Migration Guide

### From Single Backend to Multi-Backend

**Before:**
```python
# Old code (hardcoded PostgreSQL)
coordinator = Phase10Coordinator(config)
await coordinator.initialize()
```

**After:**
```python
# New code (modular backends)
config = Phase10Config(
    postgres_enabled=True,
    holochain_enabled=True,  # Added!
    storage_strategy="all"    # Write to both
)

coordinator = Phase10Coordinator(config)
await coordinator.initialize()
# Works identically, but now writes to both backends!
```

### Adding Holochain to Existing PostgreSQL Deployment

1. **Install Holochain conductor:**
   ```bash
   nix-shell -p holochain
   holochain -c conductor.yaml
   ```

2. **Update configuration:**
   ```python
   config.holochain_enabled = True
   config.storage_strategy = "all"  # Write to both
   ```

3. **Restart coordinator:**
   ```python
   await coordinator.initialize()
   ```

Done! New data writes to both backends. Old data remains in PostgreSQL.

---

## Troubleshooting

### PostgreSQL Connection Fails

**Error:** `asyncpg.exceptions.InvalidCatalogNameError`

**Solution:**
```bash
# Create database
createdb zerotrustml

# Run schema migration
psql zerotrustml < schema.sql
```

### Holochain Connection Timeout

**Error:** `TimeoutError: Holochain conductor not responding`

**Solution:**
```bash
# Check conductor is running
ps aux | grep holochain

# Restart conductor
holochain -c conductor.yaml

# Verify admin port
curl http://localhost:8888
```

### LocalFile Permission Denied

**Error:** `PermissionError: [Errno 13] Permission denied: '/tmp/zerotrustml_data'`

**Solution:**
```bash
# Create directory with permissions
mkdir -p /tmp/zerotrustml_data
chmod 755 /tmp/zerotrustml_data
```

---

## Best Practices

1. **Use PostgreSQL for production** unless you have specific decentralization needs
2. **Enable multiple backends for critical data** using `strategy="all"`
3. **Use LocalFile for testing** to avoid database setup in CI/CD
4. **Monitor backend health** using `coordinator.get_stats()`
5. **Verify integrity** periodically with `coordinator.verify_integrity(gradient_id)`
6. **Start with `strategy="primary"`** and upgrade to `"all"` or `"quorum"` when needed

---

## Future Roadmap

### Q1 2025
- ✅ PostgreSQL backend
- ✅ Holochain backend
- ✅ LocalFile backend
- ✅ Storage strategies

### Q2 2025
- 🚧 Blockchain backend (Ethereum)
- 🚧 IPFS backend (distributed storage)
- 🚧 Hybrid consensus protocols

### Q3 2025
- 🔮 Multi-chain support (Polkadot, Cosmos)
- 🔮 Automatic failover
- 🔮 Backend health monitoring dashboard

---

## Questions?

For support or questions:
- GitHub Issues: https://github.com/your-org/zerotrustml/issues
- Documentation: https://docs.zerotrustml.org/backends

---

**Last Updated:** October 2025
**Version:** Phase 10 Modular Architecture v1.0
