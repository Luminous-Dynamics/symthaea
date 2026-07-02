# Mycelix Federated Learning -- Deployment Guide

## Quick Start

```bash
# Build and run the coordinator
docker compose up coordinator

# In production, FL nodes connect externally:
# Each hospital/organization runs their own node client
```

## Architecture

```
+-------------+     +-------------+     +-------------+
|  Hospital A  |     |  Hospital B  |     |  Hospital C  |
|  (FL Node)   |     |  (FL Node)   |     |  (FL Node)   |
+------+-------+     +------+-------+     +------+-------+
       |                    |                    |
       |     gRPC (TLS)     |                    |
       +--------------------+--------------------+
                            |
                   +--------+--------+
                   |  Coordinator    |
                   |  (PoGQ v4.1)   |
                   |                |
                   |  * Byzantine   |
                   |    Detection   |
                   |  * Aggregation |
                   |  * STARK Proofs|
                   +----------------+
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MYCELIX_FL_MIN_NODES` | 3 | Minimum nodes to start a round |
| `MYCELIX_FL_MAX_NODES` | 100 | Maximum registered nodes |
| `MYCELIX_FL_ROUND_TIMEOUT` | 300 | Seconds to wait for gradients |
| `MYCELIX_FL_DEFENSE` | `pogq_v41` | Defense algorithm |
| `MYCELIX_FL_LOG_LEVEL` | `info` | Logging verbosity |

## Defense Algorithms

| Algorithm | Flag | BFT Tolerance | Use Case |
|-----------|------|---------------|----------|
| `pogq_v41` | Default | **45%** | Production (Byzantine-resilient) |
| `fedavg` | Baseline | 0% | Testing only |
| `krum` | Conservative | 33% | Small networks |
| `trimmed_mean` | Moderate | Variable | Known attack models |
| `coordinate_median` | Strong | 50% | Coordinate-wise attacks |

## Security

### Node Authentication

Nodes register with an Ed25519 public key and/or DID. The coordinator
verifies identity before accepting gradient submissions.

### Privacy

- Differential privacy applied per gradient (configurable epsilon, delta)
- HyperFeel V2 compression reduces information leakage
- STARK proofs provide verifiable detection without revealing raw gradients

### Rate Limiting

- Per-node: 10 submissions/minute, 1 per round
- Automatic blacklisting on repeated violations

## Monitoring

The coordinator exposes:

- Round completion metrics (duration, node count, excluded count)
- PoGQ detection events (quarantine/release)
- Node health (registration, authentication failures)

## Production Checklist

- [ ] TLS certificates configured for gRPC
- [ ] Node authentication enabled (`require_authentication: true`)
- [ ] Privacy budget configured (epsilon=1.0, delta=1e-5 for healthcare)
- [ ] Rate limiting tuned for expected node count
- [ ] STARK proof verification enabled for compliance
- [ ] Backup and disaster recovery configured
- [ ] Monitoring and alerting set up

## Scaling

The coordinator is single-process by design -- FL rounds are sequential.
For higher throughput, run multiple coordinators partitioned by model or
organization. The `fl-data` volume persists round history and node state
across container restarts.

Resource recommendations by deployment size:

| Nodes | Memory | CPUs | Notes |
|-------|--------|------|-------|
| 3-10 | 512 MB | 1 | Development/testing |
| 10-50 | 2 GB | 2 | Small production |
| 50-100 | 4 GB | 4 | Full production |

## License

AGPL-3.0-or-later. Commercial licensing available from
[Luminous Dynamics](https://luminousdynamics.org).

Contact: tristan.stoltz@evolvingresonantcocreationism.com
