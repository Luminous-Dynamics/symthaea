# Mycelix Deployment Guide

Step-by-step instructions for deploying a production Mycelix network.

## Prerequisites

- Docker 24+ and Docker Compose v2
- TLS certificates for your domain (e.g., via Let's Encrypt / certbot)
- At least 2 GB RAM and 10 GB disk for initial deployment
- Ports 80, 443 open on the host firewall

## 1. Configure Environment

```bash
cd mycelix-workspace/deploy/
cp .env.example .env
```

Edit `.env` and fill in:

- **LAIR_PASSPHRASE**: generate with `openssl rand -base64 32`. This encrypts agent keys at rest. Back it up securely.
- **TLS_CERT_PATH / TLS_KEY_PATH**: paths to your TLS certificate and private key on the host filesystem.
- **BOOTSTRAP_URL / SIGNAL_URL**: keep the defaults (holo.host) for public Holochain networks, or point to your own bootstrap/signal servers for private deployments.

## 2. Build Images

```bash
docker-compose -f docker-compose.prod.yml build
```

This runs a multi-stage build:
1. Compiles Rust zomes to WASM (cached across rebuilds)
2. Packs DNA bundles and the unified hApp
3. Creates a minimal runtime image with the Holochain conductor

Expect the first build to take 10-20 minutes (WASM compilation). Subsequent builds use Docker layer caching.

## 3. Start the Stack

```bash
docker-compose -f docker-compose.prod.yml up -d
```

Services start in dependency order: conductor -> observatory -> nginx.

## 4. Install the hApp

After the conductor starts, install the Mycelix unified hApp:

```bash
# Connect to the admin interface
docker exec -it mycelix-conductor hc sandbox call \
  --admin-port 4444 \
  install-app /opt/mycelix/happs/mycelix-unified.happ
```

If the unified hApp bundle was not packed during build (missing cluster DNAs), install individual hApps:

```bash
docker exec -it mycelix-conductor hc sandbox call \
  --admin-port 4444 \
  install-app /opt/mycelix/happs/<happ-name>.happ
```

## 5. Verify

### Health Checks

```bash
# Check all containers are running
docker-compose -f docker-compose.prod.yml ps

# Conductor health
curl -sf http://localhost:8888 || echo "Conductor running (WS only, no HTTP response expected)"

# Observatory
curl -sf https://localhost/ -k && echo "Observatory: OK"

# Logs
docker-compose -f docker-compose.prod.yml logs -f
```

### From the justfile (run from mycelix-workspace/)

```bash
just deploy-health
```

## 6. Backup

### Holochain Data

The conductor stores all data in the `holochain-data` Docker volume. Back it up regularly:

```bash
# Create a backup
docker run --rm \
  -v holochain-data:/data \
  -v $(pwd)/backups:/backup \
  debian:bookworm-slim \
  tar czf /backup/holochain-data-$(date +%Y%m%d).tar.gz -C /data .

# Restore from backup
docker-compose -f docker-compose.prod.yml down
docker run --rm \
  -v holochain-data:/data \
  -v $(pwd)/backups:/backup \
  debian:bookworm-slim \
  bash -c "rm -rf /data/* && tar xzf /backup/holochain-data-YYYYMMDD.tar.gz -C /data"
docker-compose -f docker-compose.prod.yml up -d
```

### Lair Passphrase

Store the LAIR_PASSPHRASE in a secure password manager. Without it, agent keys in the volume are unrecoverable.

## 7. Troubleshooting

### Conductor fails to start

**Symptom**: `mycelix-conductor` exits immediately.

- Check logs: `docker logs mycelix-conductor`
- Verify `LAIR_PASSPHRASE` is set in `.env`
- If the Lair socket already exists from a previous run, the conductor may fail. Clear the lair directory:
  ```bash
  docker-compose -f docker-compose.prod.yml down
  docker volume rm deploy_holochain-data  # WARNING: destroys all data
  docker-compose -f docker-compose.prod.yml up -d
  ```

### Observatory shows "Simulation Mode"

The Observatory falls back to simulation when it cannot reach the conductor WebSocket.

- Verify the conductor is running: `docker-compose ps`
- Check that `VITE_CONDUCTOR_URL` resolves correctly inside the Docker network
- Check nginx logs: `docker logs mycelix-nginx`

### WebSocket connection refused

- Ensure nginx.conf has the WebSocket upgrade headers (it does by default)
- Verify the conductor's app interface is on port 8888: `docker logs mycelix-conductor | grep 8888`
- Test direct WS connection bypassing nginx: `wscat -c ws://localhost:8888`

### TLS certificate errors

- Verify cert paths in `.env` point to readable files
- For Let's Encrypt, ensure the cert volume mount includes the full chain:
  ```yaml
  volumes:
    - /etc/letsencrypt:/etc/letsencrypt:ro
  ```
- Check cert expiry: `openssl x509 -in /path/to/cert.crt -noout -dates`

### Slow peer discovery

Holochain uses gossip for peer discovery, which can take 30-60 seconds on a new network.

- Check bootstrap connectivity: `curl -sf https://bootstrap.holo.host` from inside the conductor container
- For private networks, ensure all nodes point to the same bootstrap and signal servers

## Architecture Overview

```
                    Internet
                       |
                   [nginx:443]
                   /         \
            HTTPS /           \ WSS (/ws)
                 /             \
    [observatory:3000]    [conductor:8888]
                                |
                         [conductor:4444]
                          (admin, internal)
                                |
                         [holochain-data]
                          (Docker volume)
```

- **nginx** terminates TLS and routes traffic to internal services
- **observatory** serves the SvelteKit dashboard over HTTP (nginx adds HTTPS)
- **conductor** runs the Holochain runtime with all Mycelix DNAs
- **holochain-data** persists agent keys, source chains, and DHT databases
