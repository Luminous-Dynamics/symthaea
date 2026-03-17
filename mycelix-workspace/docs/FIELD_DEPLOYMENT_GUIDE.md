# Mycelix Resilience Kit -- Field Deployment Guide

A step-by-step guide for community operators deploying the Mycelix Resilience Kit.
This document assumes you can follow terminal commands but are not a software developer.


## Prerequisites

### Hardware

Any machine that meets these minimums will work:

| Requirement | Minimum |
|-------------|---------|
| Architecture | x86_64 or ARM64 (Raspberry Pi 4/5, etc.) |
| RAM | 4 GB |
| Disk | 10 GB free |
| Network | Stable internet for initial setup (not needed afterward if using mesh) |

A Raspberry Pi 4 with 4 GB RAM and a 32 GB SD card is sufficient.

### Software (choose one)

**Option A -- Docker (recommended for most operators)**

- Docker 24 or newer: https://docs.docker.com/get-docker/
- Docker Compose v2: https://docs.docker.com/compose/install/

Verify both are installed:

```
docker --version
docker compose version
```

**Option B -- NixOS**

If your machine runs NixOS, all dependencies are handled by the project flake.
No additional software installation is needed.


## Step 1: Get the Code

Clone the repository and enter the development environment:

```
git clone https://github.com/luminous-dynamics/mycelix-workspace.git
cd mycelix-workspace
```

If using NixOS or if you have Nix installed:

```
nix develop
```

This drops you into a shell with all required tools (Holochain, hc, lair-keystore, just, pnpm).

If using Docker, you do not need `nix develop` -- Docker handles the build environment.

### Validate your setup

Run the dry-run check to confirm everything is in order before proceeding:

```
./scripts/resilience-bootstrap.sh --dry-run
```

This checks for required tools, port availability, disk space, and key files. Fix any issues it reports before continuing.


## Step 2: Configure Your Community

Edit the community configuration file to match your community:

```
observatory/src/lib/community-config.json
```

Open it in any text editor. Here is what each field means:

| Field | What to enter | Example |
|-------|---------------|---------|
| `community_name` | Your community or cooperative name | `"Sunnyvale Food Co-op"` |
| `basket_name` | Name for your local value basket | `"Sunnyvale Essentials Basket"` |
| `dao_did` | A short identifier (lowercase, hyphens, no spaces) | `"sunnyvale-food-coop"` |
| `currency_code` | Your local currency ISO code | `"USD"`, `"ZAR"`, `"KES"`, `"EUR"` |
| `currency_symbol` | Currency symbol for display | `"$"`, `"R"`, `"KSh"` |
| `labor_hour_value` | Local minimum hourly wage in TEND | `15.00` |
| `labor_hour_source` | Citation for the wage figure | `"US Federal Minimum Wage 2026"` |
| `tax_year_start_month` | Month your tax year begins (1=January) | `1` |
| `tax_form_name` | Local tax form name (for export feature) | `"1099-MISC"` |
| `tax_authority` | Tax authority name | `"IRS"` |

### Configure the value basket

The `basket_items` array defines the goods your community tracks to anchor TEND purchasing power. Replace the example items with goods that matter locally. Each item needs:

| Field | Meaning |
|-------|---------|
| `key` | Machine-readable identifier (lowercase, underscores) |
| `name` | Human-readable name with quantity |
| `unit` | Unit of measure (loaf, litre, kg, kWh, trip) |
| `default_price` | Starting price in TEND (operators update this later) |
| `weight` | Importance weight (all weights should sum to approximately 1.0) |

Example for a US community:

```json
{
  "basket_items": [
    { "key": "bread_loaf", "name": "Bread (1 loaf)", "unit": "loaf", "default_price": 0.20, "weight": 0.12 },
    { "key": "eggs_dozen", "name": "Eggs (dozen)", "unit": "dozen", "default_price": 0.35, "weight": 0.10 },
    { "key": "gas_gallon", "name": "Gasoline (1 gal)", "unit": "gallon", "default_price": 0.50, "weight": 0.12 },
    { "key": "milk_gallon", "name": "Milk (1 gal)", "unit": "gallon", "default_price": 0.25, "weight": 0.08 },
    { "key": "rice_5lb", "name": "Rice (5 lb)", "unit": "bag", "default_price": 0.30, "weight": 0.15 },
    { "key": "electricity_kwh", "name": "Electricity (1 kWh)", "unit": "kWh", "default_price": 0.10, "weight": 0.10 },
    { "key": "bus_fare", "name": "Bus fare (local)", "unit": "trip", "default_price": 0.15, "weight": 0.08 },
    { "key": "chicken_whole", "name": "Whole chicken", "unit": "chicken", "default_price": 0.60, "weight": 0.10 },
    { "key": "cooking_oil_48oz", "name": "Cooking oil (48 oz)", "unit": "bottle", "default_price": 0.25, "weight": 0.08 },
    { "key": "sugar_5lb", "name": "Sugar (5 lb)", "unit": "bag", "default_price": 0.15, "weight": 0.07 }
  ]
}
```

Save the file after editing.


## Step 3: Build and Test

### Run the test suite

From the `mycelix-workspace` directory:

```
just resilience-test
```

This runs approximately 80 Rust tests (mesh-bridge and price oracle) and 75 frontend tests (Observatory). All tests should pass. If any fail, do not proceed to deployment -- check the error messages and ask for help.

### Build the resilience hApp

```
just resilience-build
```

This compiles all 8 DNA bundles (identity, finance, commons, civic, governance, hearth, knowledge, supplychain) and packs them into a single hApp file at `happs/mycelix-resilience.happ`.

The first build takes 10-20 minutes (Rust compilation to WASM). Subsequent builds are much faster.


## Step 4: Deploy

### Docker deployment (recommended)

From the `mycelix-workspace` directory, run the bootstrap script:

```
./scripts/resilience-bootstrap.sh --docker
```

This script will:
1. Build the resilience hApp if not already built
2. Generate a LAIR_PASSPHRASE (save this -- you need it for restarts)
3. Create the `deploy/.env` configuration
4. Start all containers (conductor, observatory, nginx)

**Save your LAIR_PASSPHRASE.** The script prints it to the terminal. Write it down or store it in a password manager. Without it, your agent keys are unrecoverable.

### Verify the deployment

Check that all three containers are running:

```
cd deploy
docker compose -f docker-compose.prod.yml -f docker-compose.resilience.yml ps
```

You should see three containers with status "Up":
- `mycelix-conductor`
- `mycelix-observatory`
- `mycelix-nginx`

The Observatory should now be accessible at **http://localhost** (or your server's IP address).

### Local deployment (NixOS alternative)

If you are using NixOS with `nix develop`, you can start everything locally instead:

```
just resilience-up
```

The Observatory will be available at **http://localhost:5173**.


## Step 5: Smoke Test

Run the smoke test to verify all 20 routes are reachable:

```
./scripts/smoke-test.sh
```

This builds the Observatory, starts a preview server, and checks that each route returns a successful response. The routes tested are:

```
/                /resilience      /tend            /food
/mutual-aid      /emergency       /value-anchor    /water
/household       /knowledge       /care-circles    /shelter
/supplies        /admin           /governance      /network
/analytics       /attribution     /welcome         /print
```

All 20 routes should show "OK". If any fail, see Troubleshooting below.

You can also check overall system status at any time:

```
./scripts/resilience-bootstrap.sh --status
```


## Step 6: Operator Setup

### Access the admin panel

Open your browser and go to **/admin** (e.g., `http://localhost/admin` for Docker or `http://localhost:5173/admin` for local).

On the admin panel:
- Enter your name and contact information as the community operator.
- Review the community name and settings pulled from `community-config.json`.
- Confirm the 8 DNAs are connected (the admin panel shows connection status for each).

### Configure the value basket

Go to **/value-anchor** in your browser. This is where you set current local prices for each basket item.

For each item in your basket:
1. Enter the current price in your local currency.
2. The system calculates how much TEND that item costs based on the labor-hour anchor.
3. The overall TEND purchasing power index updates automatically.

This is the core of the resilience economy: 1 TEND = 1 hour of labor, and the basket shows what that hour can buy in your community.


## Step 7: Onboard Members

### Share access

Give community members the URL to your Observatory:
- If deployed on a local network: `http://<your-machine-ip>`
- If deployed with a domain and TLS: `https://your-domain.org`

The **/welcome** page is designed for first-time members. Share the URL or print a QR code pointing to it.

### Walk through onboarding

New members should:

1. Visit **/welcome** and follow the setup steps to create their Holochain agent (identity).
2. Once registered, visit **/tend** to see their TEND balance and make a test exchange.
3. Visit **/emergency** to verify they can send and receive emergency messages.

### Test a TEND exchange

Have two members complete a small exchange:
1. Member A offers 1 hour of work (e.g., gardening, tutoring).
2. Member B accepts and transfers 1 TEND.
3. Both balances should update. Check on the **/tend** page.

### Test emergency messaging

1. One member sends a test emergency message from **/emergency**.
2. Confirm all online members receive the alert.
3. If mesh is enabled (Step 8), test with one node offline then reconnecting.


## Step 8: Enable Mesh (Optional)

The mesh bridge allows nodes to communicate over LoRa radio or WiFi-direct when internet connectivity is unavailable. This step is optional and requires additional hardware. See `docs/MESH_HARDWARE_GUIDE.md` for the full bill of materials (approximately $80-120 per node).

### Docker deployment

Add the mesh profile to your compose command. First, set the transport type in `deploy/.env`:

```
# For testing without hardware:
MESH_TRANSPORT=loopback

# For LoRa hardware (SX1276 HAT):
MESH_TRANSPORT=lora
LORA_FREQ_MHZ=868.0
LORA_SPI_PATH=/dev/spidev0.0
```

Then start with the mesh profile enabled:

```
cd deploy
docker compose -f docker-compose.prod.yml -f docker-compose.resilience.yml --profile mesh up -d
```

If using LoRa hardware, uncomment the `devices` section in `docker-compose.resilience.yml` to pass through the SPI device:

```yaml
    devices:
      - /dev/spidev0.0:/dev/spidev0.0
```

### Verify mesh bridge health

```
curl http://localhost:9100/health
```

You should see a JSON response with `messages_sent`, `messages_received`, `peers_seen`, and `poll_cycles` fields.

### Important: frequency selection

Use the correct ISM frequency for your region:

| Region | Frequency |
|--------|-----------|
| Africa / Europe | 868 MHz |
| Americas | 915 MHz |
| Asia-Pacific | 923 MHz |

All nodes in your mesh must use the same frequency.


## Step 9: Ongoing Operations

### Daily

- Check **/admin** for system health. Confirm the conductor is connected and no DNAs show errors.
- Review **/emergency** for any unresolved alerts.

### Weekly

- Update basket prices at **/value-anchor** to reflect current local market prices. This keeps the TEND purchasing power index accurate.
- Check **/analytics** for network activity trends.

### Monthly

- Visit **/print** to generate a printable summary of community activity (TEND circulation, mutual aid hours, food production, emergency events).
- Share this summary at community meetings.
- Review **/governance** for any open proposals or votes.

### Backups

Back up your Holochain data regularly. For Docker deployments:

```
docker run --rm \
  -v deploy_holochain-data:/data \
  -v $(pwd)/backups:/backup \
  debian:bookworm-slim \
  tar czf /backup/holochain-data-$(date +%Y%m%d).tar.gz -C /data .
```

Store backups on a separate drive or off-site. You also need your LAIR_PASSPHRASE to restore -- keep it safe.

### Stopping and restarting

Docker:

```
cd deploy
docker compose -f docker-compose.prod.yml -f docker-compose.resilience.yml down
docker compose -f docker-compose.prod.yml -f docker-compose.resilience.yml up -d
```

Local (NixOS):

```
just stop
just resilience-up
```


## Troubleshooting

### Conductor will not start

**Symptom**: The `mycelix-conductor` container exits immediately or restarts repeatedly.

**Check logs**:
```
docker logs mycelix-conductor
```

**Common causes**:
- `LAIR_PASSPHRASE` is not set or is wrong. Check `deploy/.env` and make sure the passphrase matches what was generated during initial setup.
- The Lair socket from a previous run is stale. Stop all containers, then:
  ```
  docker compose -f docker-compose.prod.yml -f docker-compose.resilience.yml down
  docker volume rm deploy_holochain-data
  docker compose -f docker-compose.prod.yml -f docker-compose.resilience.yml up -d
  ```
  WARNING: This destroys all stored data. Only do this if you have no data to preserve or have a backup.
- Port 8888 is already in use by another process. Check with `ss -tlnp | grep 8888` or `lsof -i :8888`.

### Routes return 404

**Symptom**: Smoke test shows FAIL for one or more routes.

**Check**:
1. Make sure you ran `just resilience-build` before deploying. The Observatory needs the built frontend assets.
2. For Docker: rebuild the observatory image:
   ```
   docker compose -f docker-compose.prod.yml -f docker-compose.resilience.yml build observatory
   docker compose -f docker-compose.prod.yml -f docker-compose.resilience.yml up -d observatory
   ```
3. For local: make sure the Observatory dev server is running (`cd observatory && pnpm dev`).

### Observatory shows "Simulation Mode"

This means the Observatory cannot reach the Holochain conductor WebSocket.

- Verify the conductor is running: `docker compose ps` should show `mycelix-conductor` as "Up".
- Check that `VITE_CONDUCTOR_URL` in `deploy/.env` is set to `ws://conductor:8888` (for Docker) or `ws://localhost:8888` (for local).
- Check nginx logs: `docker logs mycelix-nginx`.

### Offline queue stuck (mesh bridge)

**Symptom**: Messages are queued but not delivering even when connectivity is restored.

**Check**:
1. Verify both nodes are on the same frequency: `LORA_FREQ_MHZ` must match.
2. Check mesh bridge health on both nodes: `curl http://localhost:9100/health`.
3. If `poll_cycles` is incrementing but `peers_seen` is 0, the radio link is not established. Check antenna connections and ensure nodes are within range.
4. Restart the mesh bridge:
   ```
   docker compose -f docker-compose.prod.yml -f docker-compose.resilience.yml restart mesh-bridge
   ```
5. Clear the dedup cache if messages are being incorrectly deduplicated:
   ```
   rm -f /tmp/mycelix-mesh-bridge/*.cache
   ```

### WebSocket connection refused

- Ensure nginx is running and has the WebSocket upgrade headers configured (the default `nginx.conf` includes these).
- Test a direct connection bypassing nginx: `wscat -c ws://localhost:8888` (install wscat with `npm install -g wscat`).
- Check that port 8888 is not blocked by a firewall.

### Build fails on ARM64

The mesh-bridge Docker image currently targets x86_64. On ARM64 (Raspberry Pi), the conductor and Observatory will build, but the mesh bridge may fail. For ARM64 mesh bridge deployment, use the NixOS native build path instead of Docker:

```
cd mesh-bridge
cargo build --release --features lora
```

See `docs/MESH_HARDWARE_GUIDE.md` for NixOS service configuration.

### Tests fail during Step 3

- Make sure you are inside the `nix develop` shell (for local builds).
- Run `pnpm install` in the `observatory/` directory if frontend tests fail with missing modules.
- If Rust tests fail with compilation errors, ensure you are on the correct branch (`main`) and have pulled the latest code.


## Getting Help

- Consult the existing deployment docs at `deploy/DEPLOYMENT.md` for general production deployment details.
- For mesh hardware questions, see `docs/MESH_HARDWARE_GUIDE.md`.
- For governance and cooperative structure, see `docs/COOPERATIVE_CONSTITUTION_TEMPLATE.md`.
