# Mycelix-Core Docker Deployment

One-command deployment for the complete Mycelix Federated Learning system with Byzantine Fault Tolerance.

## Quick Start

```bash
# From the Mycelix-Core root directory
./launch.sh
```

That's it! The complete system will be running in under 5 minutes.

## What Gets Deployed

| Service | Description | Port |
|---------|-------------|------|
| **FL Coordinator** | Python + Rust federated learning coordinator | 8080 |
| **FL Nodes (x5)** | Simulated federated learning clients | - |
| **PostgreSQL** | State storage and persistence | 5432 |
| **Redis** | Caching and rate limiting | 6379 |
| **Prometheus** | Metrics collection | 9090 |
| **Grafana** | Dashboards and visualization | 3000 |
| **Holochain** | (Optional) Decentralized coordination | 8888, 8889 |

## Access Points

### FL Coordinator API

- **Base URL**: http://localhost:8080
- **Health Check**: http://localhost:8080/health
- **Detailed Status**: http://localhost:8080/status
- **Prometheus Metrics**: http://localhost:8080/metrics

#### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Simple health check |
| `/status` | GET | Detailed system status |
| `/metrics` | GET | Prometheus metrics |
| `/nodes/register` | POST | Register a new FL node |
| `/nodes` | GET | List all registered nodes |
| `/rounds/new` | POST | Start a new training round |
| `/rounds/{id}/submit` | POST | Submit model update |
| `/analytics/convergence` | GET | Model convergence metrics |
| `/analytics/byzantine` | GET | Byzantine detection stats |

### Grafana Dashboards

- **URL**: http://localhost:3000
- **Default Login**: admin / admin

Pre-configured dashboards:

1. **Byzantine Detection** - Monitor trust scores and malicious node detection
2. **Performance Metrics** - Request rates, latencies, throughput
3. **Node Health** - Individual node status and training times
4. **Model Convergence** - Track model training progress

### Prometheus

- **URL**: http://localhost:9090
- Pre-configured to scrape coordinator and all nodes

## Launch Script Options

```bash
# Start all services
./launch.sh

# Rebuild containers before starting
./launch.sh --build

# Include Holochain for decentralization
./launch.sh --holochain

# Include CLI tools container
./launch.sh --tools

# Stop all services
./launch.sh --down

# Check service status
./launch.sh --status

# Follow all logs
./launch.sh --logs

# Follow specific service logs
./launch.sh --logs coordinator

# Stop and remove all data
./launch.sh --clean

# Show help
./launch.sh --help
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp docker/.env.example docker/.env
```

Key configuration options:

| Variable | Default | Description |
|----------|---------|-------------|
| `BFT_THRESHOLD` | 0.33 | Byzantine fault tolerance threshold |
| `MIN_NODES_FOR_AGGREGATION` | 3 | Minimum nodes for aggregation |
| `DIFFERENTIAL_PRIVACY_EPSILON` | 1.0 | DP epsilon parameter |
| `NODE_COUNT` | 5 | Number of FL nodes |
| `GRAFANA_ADMIN_PASSWORD` | admin | Grafana admin password |

### Byzantine Testing

To test Byzantine detection, enable Byzantine mode on node-5:

```bash
BYZANTINE_NODE_5=true ./launch.sh
```

## CLI Tools

Run CLI commands using Docker:

```bash
# Show system status
docker exec -it mycelix-cli mycelix status

# List nodes
docker exec -it mycelix-cli mycelix nodes list

# Watch status in real-time
docker exec -it mycelix-cli mycelix watch

# Start a new training round
docker exec -it mycelix-cli mycelix rounds new

# View Byzantine statistics
docker exec -it mycelix-cli mycelix analytics byzantine
```

Or start an interactive CLI session:

```bash
./launch.sh --tools
docker exec -it mycelix-cli bash
mycelix --help
```

## Architecture

```
                                    ┌─────────────────┐
                                    │    Grafana      │
                                    │   :3000         │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │   Prometheus    │
                                    │   :9090         │
                                    └────────┬────────┘
                                             │
┌─────────┐  ┌─────────┐  ┌─────────┐       │
│ Node 1  │  │ Node 2  │  │ Node 3  │       │
└────┬────┘  └────┬────┘  └────┬────┘       │
     │            │            │            │
     └────────────┼────────────┘            │
                  │                         │
           ┌──────▼──────┐                  │
           │ Coordinator │◄─────────────────┘
           │   :8080     │
           └──────┬──────┘
                  │
     ┌────────────┴────────────┐
     │                         │
┌────▼────┐              ┌─────▼────┐
│ PostgreSQL             │  Redis   │
│  :5432  │              │  :6379   │
└─────────┘              └──────────┘
```

## Troubleshooting

### Services Not Starting

1. Check Docker is running:
   ```bash
   docker info
   ```

2. Check for port conflicts:
   ```bash
   lsof -i :8080  # Coordinator
   lsof -i :3000  # Grafana
   lsof -i :9090  # Prometheus
   ```

3. View container logs:
   ```bash
   ./launch.sh --logs
   ```

### Coordinator Health Check Failing

1. Wait for startup (can take 30-60 seconds)
2. Check coordinator logs:
   ```bash
   ./launch.sh --logs coordinator
   ```

3. Verify dependencies are healthy:
   ```bash
   docker exec mycelix-postgres pg_isready
   docker exec mycelix-redis redis-cli ping
   ```

### Nodes Not Connecting

1. Ensure coordinator is healthy first
2. Check node logs:
   ```bash
   docker logs mycelix-node-1
   ```

3. Verify network connectivity:
   ```bash
   docker network inspect mycelix-core_mycelix-network
   ```

### Grafana Dashboards Not Loading

1. Verify Prometheus is running:
   ```bash
   curl http://localhost:9090/-/healthy
   ```

2. Check Grafana datasource:
   - Go to http://localhost:3000/datasources
   - Test the Prometheus connection

### Out of Memory

Reduce resource limits in `.env`:
```bash
NODE_MEMORY_LIMIT=256m
COORDINATOR_MEMORY_LIMIT=1g
```

### Reset Everything

```bash
./launch.sh --clean
./launch.sh
```

## Production Deployment

For production use:

1. **Change all default passwords** in `.env`
2. Enable TLS:
   ```bash
   TLS_ENABLED=true
   TLS_CERT_PATH=/certs/server.crt
   TLS_KEY_PATH=/certs/server.key
   ```
3. Increase Byzantine threshold if needed
4. Configure external PostgreSQL/Redis if scaling
5. Set up proper monitoring alerts in Grafana
6. Use Docker Swarm or Kubernetes for orchestration

## Security Considerations

- Default passwords are for development only
- The coordinator exposes metrics that may contain sensitive info
- Redis and PostgreSQL are accessible on localhost by default
- Consider network isolation in production
- Enable JWT authentication for the API
- Use secure aggregation with encryption for sensitive data

## Support

For issues and questions:
- Check the troubleshooting section above
- Review container logs
- Open an issue on GitHub

## License

Part of the Mycelix-Core project.
