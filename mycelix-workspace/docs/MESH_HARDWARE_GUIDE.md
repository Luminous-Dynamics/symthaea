# Mesh Hardware Guide -- Mycelix LoRa Deployment

Practical guide for community operators deploying the Mycelix mesh bridge.
The `mesh-bridge` binary runs alongside a Holochain conductor and relays
TEND exchanges, food harvests, emergency messages, water alerts, shelter
updates, mutual aid offers, and other Mycelix DHT entries over LoRa
(SX1276 SPI) or B.A.T.M.A.N. WiFi-direct when internet is unavailable.

**Source**: `mesh-bridge/` (Rust, `mycelix-mesh-bridge` crate)
**NixOS module**: `deploy/mesh-bridge.nix`

---

## Architecture

```
  Node A (home)                         Node B (community center)
  +------------------+                  +------------------+
  |  Holochain       |                  |  Holochain       |
  |  Conductor :8888 |                  |  Conductor :8888 |
  |                  |                  |                  |
  |  mesh-bridge     |                  |  mesh-bridge     |
  |   poller (out)   |                  |   relay (in)     |
  |   relay  (in)    |                  |   poller (out)   |
  |   health :9100   |                  |   health :9100   |
  +--------+---------+                  +--------+---------+
           | SPI /dev/spidev0.0                  | SPI
  +--------+---------+    LoRa 868 MHz  +--------+---------+
  | SX1276 HAT       |<--------------->| SX1276 HAT       |
  | + antenna        |   2-15 km       | + antenna        |
  +------------------+                  +------------------+
```

Each node runs two async tasks:

- **Poller** -- polls the local conductor for new entries, serializes them
  with bincode, fragments to 255-byte LoRa frames, and transmits.
- **Relay** -- listens for incoming frames, reassembles, deduplicates by
  BLAKE3 content hash, and replays as zome calls on the local conductor.

A health endpoint on port 9100 exposes JSON metrics (messages sent/received,
fragments dropped, peers seen, connection failures) and sends systemd
watchdog pings.

---

## 1. Bill of Materials

### Per Node (approximately $80-120 USD)

| Component | Example Model | Est. Price | Notes |
|-----------|---------------|-----------|-------|
| Single-board computer | Raspberry Pi 4 Model B (2 GB+) or Pi 5 | $35-60 | Pi 4 is sufficient; Pi 5 gives headroom |
| LoRa HAT | Dragino LoRa/GPS HAT, Waveshare SX1276 LoRa HAT | $15-25 | Must use SX1276 chipset; match frequency to your region |
| Antenna | 868 MHz or 915 MHz SMA whip antenna | $5-15 | Whip for omni 2-5 km; Yagi for directional 10-15 km |
| MicroSD card | 32 GB+ Class 10 / A1 | $8-12 | NixOS image + Holochain + mesh-bridge + dedup cache |
| Power supply (mains) | Official RPi USB-C 5V 3A PSU | $10 | For indoor/grid-powered nodes |
| Weatherproof enclosure | IP65 ABS junction box, ~200x150x75 mm | $10-20 | Required for any outdoor deployment |
| Ethernet cable or WiFi | Cat5e or onboard WiFi | $0-5 | For conductor-to-conductor LAN sync when internet returns |

### Solar Power Kit (approximately $50-80 additional)

| Component | Specification | Est. Price | Notes |
|-----------|--------------|-----------|-------|
| Solar panel | 20 W 12 V monocrystalline (minimum) | $25-30 | 30 W recommended for cloudy regions |
| Charge controller | PWM or MPPT, 10 A minimum | $10-15 | MPPT preferred for efficiency |
| Battery | 12 V 7 Ah SLA (sealed lead-acid) or 18650 Li-ion pack | $15-25 | Provides ~24 h runtime without sun (see sizing below) |
| Buck converter | 12 V to 5 V 3 A, USB-C output | $5-8 | Powers the Pi from battery; ensure 3 A sustained |
| Fuse | 5 A blade fuse on battery lead | $1 | Prevents fire in case of short circuit |

---

## 2. Assembly

### Step 1: Prepare the Raspberry Pi

1. Flash NixOS aarch64 to the MicroSD card (or your preferred distro).
2. Insert the MicroSD card and boot the Pi to verify it works.
3. Shut down the Pi before attaching the HAT.

### Step 2: Attach the LoRa HAT

1. **Power off the Pi.** Never connect or disconnect the HAT while powered.
2. Align the HAT's 40-pin female header with the Pi's GPIO header.
3. Press down firmly and evenly -- the HAT should sit flush.
4. The SX1276 communicates over **SPI0, CE0** (`/dev/spidev0.0`).
   GPIO pins used: MOSI (GPIO 10), MISO (GPIO 9), SCLK (GPIO 11),
   CE0 (GPIO 8), plus RST and DIO0 (varies by HAT -- check your
   vendor documentation).

### Step 3: Connect the Antenna

1. Screw the antenna onto the SMA connector on the HAT.
2. **Never power the LoRa module without an antenna connected.**
   Transmitting without a load can damage the SX1276 RF front-end.
3. For outdoor deployments, route the antenna cable through a
   weatherproof gland in the enclosure.

### Step 4: Enable SPI

**NixOS** (recommended):

```nix
hardware.raspberry-pi."4".spi.enable = true;
# Or for config.txt-based overlays:
hardware.raspberry-pi.config.all.base-dt-params.spi = {
  enable = true;
};
```

**Raspbian / other distros**:

```bash
sudo raspi-config
# Interface Options -> SPI -> Enable
# Reboot
```

Verify SPI is active:

```bash
ls -l /dev/spidev0.0
# Should exist and be readable by your user/group
```

### Step 5: Solar Wiring (if applicable)

```
Solar Panel (+) ---> Charge Controller (Solar +)
Solar Panel (-) ---> Charge Controller (Solar -)
Battery (+)    ---> Fuse ---> Charge Controller (Batt +)
Battery (-)    ---> Charge Controller (Batt -)
Charge Controller (Load +) ---> Buck Converter (IN +)
Charge Controller (Load -) ---> Buck Converter (IN -)
Buck Converter (USB-C out) ---> Raspberry Pi
```

Mount the solar panel at an angle facing the equator (north in the
Southern Hemisphere, south in the Northern Hemisphere). Tilt angle
should roughly match your latitude for year-round performance.

---

## 3. NixOS Configuration

The project ships a NixOS module at `deploy/mesh-bridge.nix`. Import it
in your node's `configuration.nix`:

```nix
{ config, pkgs, ... }:

let
  mesh-bridge-module = import ../deploy/mesh-bridge.nix;
in {
  imports = [ mesh-bridge-module ];

  # Enable SPI hardware
  hardware.raspberry-pi."4".spi.enable = true;

  services.mycelix-mesh-bridge = {
    enable = true;

    # Path to the built mesh-bridge package
    package = pkgs.callPackage ../mesh-bridge { };

    # Holochain conductor AppWebsocket URL
    conductorUrl = "ws://127.0.0.1:8888";

    # Poll interval (seconds between conductor queries)
    pollIntervalSecs = 10;

    # Transport backend: "lora", "batman", or "loopback"
    meshTransport = "lora";

    # SPI device path for SX1276 HAT
    loraDevice = "/dev/spidev0.0";

    # Frequency in MHz -- MUST match your region (see Section 7)
    loraFrequencyMhz = 868;  # EU/ZA: 868, US/AU: 915

    # Optional: conductor auth token file
    # appTokenFile = /run/secrets/mesh-app-token;
  };

  # Ensure the mycelix user can access SPI and GPIO
  users.users.mycelix = {
    isSystemUser = true;
    group = "mycelix";
    extraGroups = [ "spi" "gpio" "dialout" ];
  };
  users.groups.mycelix = {};
}
```

The NixOS module automatically:

- Creates the `mycelix` system user with SPI/GPIO group membership.
- Adds a udev rule granting the `spi` group access to `/dev/spidev*`.
- Runs the service as `Type=notify` with `WatchdogSec=90`.
- Applies systemd hardening (NoNewPrivileges, ProtectSystem=strict,
  MemoryDenyWriteExecute, etc.).
- Restarts on failure with a 10-second delay.

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `CONDUCTOR_URL` | `ws://localhost:8888` | Holochain conductor AppWebsocket URL |
| `POLL_INTERVAL_SECS` | `30` | Seconds between conductor poll cycles |
| `MESH_TRANSPORT` | `loopback` | Transport: `lora`, `batman`, or `loopback` |
| `LORA_SPI_PATH` | `/dev/spidev0.0` | SPI device path for SX1276 |
| `LORA_FREQ_MHZ` | `868.0` | LoRa frequency in MHz |
| `MESH_APP_TOKEN` | (empty) | Holochain conductor app auth token |
| `HEALTH_PORT` | `9100` | HTTP health/metrics endpoint port |
| `MESH_CACHE_DIR` | `/tmp/mycelix-mesh-bridge` | Persistent dedup cache directory |
| `RUST_LOG` | `info` | Log level (`debug`, `info`, `warn`, `error`) |

---

## 4. Docker Alternative

For non-NixOS deployments, use Docker with SPI device passthrough:

```yaml
# docker-compose.yml
version: "3.8"

services:
  mesh-bridge:
    image: ghcr.io/luminous-dynamics/mycelix-mesh-bridge:latest
    # Or build locally:
    # build:
    #   context: ./mesh-bridge
    #   args:
    #     FEATURES: "lora"
    restart: always
    devices:
      - "/dev/spidev0.0:/dev/spidev0.0"
    environment:
      CONDUCTOR_URL: "ws://host.docker.internal:8888"
      MESH_TRANSPORT: "lora"
      LORA_SPI_PATH: "/dev/spidev0.0"
      LORA_FREQ_MHZ: "868.0"
      POLL_INTERVAL_SECS: "10"
      MESH_APP_TOKEN: "${MESH_APP_TOKEN}"
      RUST_LOG: "info"
      MESH_CACHE_DIR: "/data/dedup"
    ports:
      - "9100:9100"
    volumes:
      - mesh-data:/data/dedup
    # Required for GPIO/SPI access
    privileged: false
    group_add:
      - "spi"

volumes:
  mesh-data:
```

**Important notes for Docker**:

- The `devices` directive passes the SPI device into the container.
  The host kernel must have SPI enabled (`dtparam=spi=on`).
- If the conductor runs on the host, use `host.docker.internal` (Docker
  Desktop) or `--network=host` (Linux) to reach `ws://localhost:8888`.
- Mount a volume for `MESH_CACHE_DIR` so dedup caches survive container
  restarts. The bridge persists BLAKE3 hashes of relayed entries on
  shutdown and reloads them on startup.

Build from source:

```bash
docker build \
  --build-arg FEATURES="lora" \
  -t mycelix-mesh-bridge:local \
  mesh-bridge/
```

---

## 5. Network Topology

### Star Topology (Single Hub)

```
   Node A ----+
              |
   Node B ----+----- Hub Node (elevated, good antenna)
              |
   Node C ----+
```

- One central hub with a high-gain antenna (rooftop or tower).
- Spoke nodes use standard whip antennas.
- Simplest to deploy and troubleshoot.
- Single point of failure at the hub.
- Best for: small communities (5-15 nodes), all within range of the hub.

### Mesh Topology (Multi-Hop)

```
   Node A ---- Node B ---- Node C
      \                      |
       \                     |
        Node D ----------- Node E
```

- Every node relays frames it receives (store-and-forward).
- The mesh bridge broadcasts all frames; any node within radio range
  picks them up, deduplicates by BLAKE3 hash, and relays to its
  local conductor.
- No single point of failure.
- Latency increases with each hop.
- Best for: larger communities, areas with obstacles, redundancy needs.

### Range Estimates

| Environment | Whip Antenna | Yagi (Directional) |
|-------------|-------------|-------------------|
| Dense urban (buildings, trees) | 1-2 km | 3-5 km |
| Suburban / township | 2-5 km | 5-10 km |
| Open terrain / line of sight | 5-10 km | 10-15 km |
| Elevated (rooftop to rooftop) | 5-8 km | 10-20 km |

Range depends heavily on antenna height, obstacles, and interference.
Elevating the antenna by even 2-3 meters (e.g., mounting on a pole above
a corrugated roof) can double effective range in township environments.

### B.A.T.M.A.N. WiFi-Direct (Short Range, High Bandwidth)

For nodes within WiFi range (~50-100 m), B.A.T.M.A.N. provides much
higher throughput than LoRa:

```bash
# On each node:
modprobe batman-adv
ip link set wlan0 down
iwconfig wlan0 mode ad-hoc essid "mycelix-mesh" channel 6
ip link set wlan0 up
batctl if add wlan0
ip link set bat0 up
ip addr add 10.0.0.X/24 dev bat0   # unique X per node

# Start mesh-bridge with batman transport
MESH_TRANSPORT=batman cargo run --features batman
```

The batman transport uses UDP broadcast on port 9735 over the `bat0`
interface. Useful for dense deployments (market, school campus) where
nodes are close together.

---

## 6. Solar Power Sizing

### Power Budget

| Component | Typical Draw | Peak Draw |
|-----------|-------------|-----------|
| Raspberry Pi 4 (idle + mesh-bridge) | 3-4 W | 6 W |
| Raspberry Pi 5 (idle + mesh-bridge) | 4-5 W | 8 W |
| SX1276 LoRa HAT (receive mode) | ~0.03 W | -- |
| SX1276 LoRa HAT (transmit) | ~0.4 W | ~0.6 W |
| **Total (Pi 4 + HAT)** | **~4 W average** | **~6.6 W** |
| **Total (Pi 5 + HAT)** | **~5 W average** | **~8.6 W** |

The LoRa HAT spends most of its time in receive mode (~10 mA at 3.3 V).
Transmit bursts are brief (a 255-byte frame at SF7 takes ~70 ms).

### Sizing for 24-Hour Operation

Target: run continuously with 6 hours of effective sun per day.

**Battery capacity (for 18 hours without sun)**:

```
Pi 4:  4 W x 18 h = 72 Wh
       72 Wh / 12 V = 6.0 Ah  -->  use 7 Ah battery (minimum)

Pi 5:  5 W x 18 h = 90 Wh
       90 Wh / 12 V = 7.5 Ah  -->  use 9 Ah or 12 Ah battery
```

A 12 V 7 Ah SLA battery provides 84 Wh. With a buck converter at
~90% efficiency, usable energy is ~76 Wh -- enough for a Pi 4 node
through one night.

**Solar panel sizing (to recharge in 6 sun-hours)**:

```
Pi 4:  Daytime load:   4 W x 6 h  = 24 Wh
       Recharge battery:            = 72 Wh
       Total needed:                = 96 Wh
       Panel:  96 Wh / 6 h         = 16 W minimum  -->  use 20 W panel

Pi 5:  Total needed:                = 120 Wh
       Panel:  120 Wh / 6 h        = 20 W minimum  -->  use 30 W panel
```

**Recommendations by region**:

| Sun Hours | Pi 4 Panel | Pi 5 Panel | Battery |
|-----------|-----------|-----------|---------|
| 6+ h/day (summer, tropics) | 20 W | 30 W | 12 V 7 Ah |
| 4-5 h/day (winter, temperate) | 30 W | 40 W | 12 V 12 Ah |
| 2-3 h/day (overcast, high latitude) | 50 W | 60 W | 12 V 20 Ah |

For multi-day autonomy (cloud cover), double the battery capacity.

---

## 7. Frequency Regulations

All mesh bridge LoRa communication uses license-free ISM (Industrial,
Scientific, Medical) bands. You must comply with your region's rules.

| Region | Frequency | Standard | Key Rules |
|--------|-----------|----------|-----------|
| EU | 868.0-868.6 MHz | ETSI EN 300 220 | 1% duty cycle (max 36 s/hour TX), 25 mW (14 dBm) ERP |
| South Africa | 868 MHz | ICASA TR 70-001 | Follows ETSI rules; 25 mW ERP, 1% duty cycle |
| USA | 902-928 MHz (915 MHz center) | FCC Part 15.247 | 1 W (30 dBm) with frequency hopping or digital modulation |
| Australia | 915-928 MHz | ACMA LIPD Class License | 1 W EIRP, no duty cycle limit |
| India | 865-867 MHz | WPC | 1 W, indoor/outdoor |
| Japan | 920-928 MHz | ARIB STD-T108 | 20 mW (13 dBm), LBT (listen-before-talk) required |
| Brazil | 902-907.5 / 915-928 MHz | ANATEL | 1 W, frequency hopping |
| Some regions | 433 MHz | ISM | 10 mW in EU, varies elsewhere; shorter range, more interference |

### Duty Cycle Compliance (EU/ZA)

The 1% duty cycle on 868 MHz sub-band g1 means a device may transmit
for at most 36 seconds in any 1-hour window. At SF7/125kHz, a single
255-byte frame takes ~70 ms. This allows roughly 500 frames per hour --
far more than the mesh bridge needs (typical: 10-50 frames/hour
depending on community activity).

The mesh bridge's heartbeat interval (every 4 poll cycles) and
priority scheduling (safety-critical entries every cycle, lower-priority
every other cycle) are designed to stay well within duty cycle limits
at the default 10-second poll interval.

### Practical Guidance

- **Always verify** local regulations before deployment. ISM band rules
  change and may have additional restrictions.
- Set `loraFrequencyMhz` in the NixOS config (or `LORA_FREQ_MHZ` env
  var) to match your regional frequency.
- All nodes in a mesh **must** use the same frequency.
- The SX1276 supports 137-1020 MHz, but your HAT and antenna must be
  rated for the chosen band.

---

## 8. Troubleshooting

### SPI Not Detected

| Symptom | Cause | Fix |
|---------|-------|-----|
| `/dev/spidev0.0` does not exist | SPI not enabled in device tree | NixOS: set `hardware.raspberry-pi."4".spi.enable = true;` and rebuild. Raspbian: `sudo raspi-config` -> Interface Options -> SPI -> Enable. Reboot. |
| `Permission denied` on `/dev/spidev0.0` | User not in `spi` group | `sudo usermod -aG spi mycelix` and restart the service. The NixOS module handles this automatically via udev rules. |
| `LoRa SX1276 initialized` log but no TX/RX | HAT not seated properly | Power off, reseat the HAT on the GPIO header, check all 40 pins are aligned. |
| SPI open succeeds but no LoRa traffic | Wrong SPI bus/CE | Some HATs use `spidev0.1` (CE1). Check your HAT documentation and set `LORA_SPI_PATH` accordingly. |

### No Peers Discovered

| Symptom | Cause | Fix |
|---------|-------|-----|
| `peers_seen: 0` on `/health` | Nodes on different frequencies | Verify all nodes use identical `LORA_FREQ_MHZ`. |
| Heartbeats sent but none received | Out of range | Move nodes closer or elevate antennas. Start testing at <500 m line-of-sight. |
| Nodes within range but no traffic | Antenna not connected | Check SMA connector. Transmitting without an antenna can also damage the module. |
| Intermittent connectivity | Fresnel zone obstruction | LoRa needs a clear path between antennas. At 868 MHz over 5 km, the first Fresnel zone is ~20 m wide at the midpoint. Trees, buildings, and terrain can block it. |

### High Fragment Drop Rate

| Symptom | Cause | Fix |
|---------|-------|-----|
| `fragments_dropped` climbing | Interference or noise | Move away from other 868/915 MHz devices. Check for nearby LoRaWAN gateways. |
| Reassembly timeouts (>30 s) | Fragments from large payloads lost in transit | Reduce payload size. The bridge fragments at 255-byte LoRa frames with an 8-byte header (247 usable bytes per frame). Multi-frame payloads are more susceptible to loss. |
| Duplicate fragments counted as drops | Hash prefix collision | Extremely rare (4-byte prefix). Check logs for `Failed to deserialize relay payload` -- this indicates a real corruption issue. |
| Drops only at certain times | Duty cycle saturation or competing traffic | Monitor with `RUST_LOG=debug` to see TX/RX timing. Increase `POLL_INTERVAL_SECS` to reduce airtime. |

### Conductor Connection Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Poll failed` in logs, `connection_failures` increasing | Conductor not running or wrong URL | Verify Holochain conductor is running: `curl -s http://localhost:8888` or check `systemctl status holochain-conductor`. |
| `MESH_APP_TOKEN not set` warning | Missing auth token | Set the `MESH_APP_TOKEN` env var to the token returned when the hApp was installed. In NixOS, use `appTokenFile`. |
| Exponential backoff (long pauses between polls) | Conductor consistently unreachable | The bridge backs off from 5 s to 300 s max. Fix the conductor connection and the bridge will automatically recover on the next successful poll. |

### General Diagnostics

```bash
# Check service status
systemctl status mycelix-mesh-bridge

# View live logs
journalctl -u mycelix-mesh-bridge -f

# Check health endpoint (JSON metrics)
curl -s http://localhost:9100/health | python3 -m json.tool

# Verify SPI device
ls -la /dev/spidev*

# Check SPI kernel module
lsmod | grep spi

# Test LoRa at debug level
RUST_LOG=debug MESH_TRANSPORT=lora mesh-bridge
```

### Testing Without Hardware

```bash
# Default transport is loopback (in-memory, no radio needed)
cargo run

# Run the test suite
cargo test

# Run property tests (fragment/reassemble roundtrips)
cargo test -- prop_
```

---

## Relayed Message Types

The mesh bridge relays 11 entry types plus heartbeats. Safety-critical
types are polled every cycle; lower-priority types every other cycle.

| Type | Priority | Cluster Role | Zome |
|------|----------|-------------|------|
| TEND Exchange | Safety-critical | `finance` | `tend` |
| Emergency Message | Safety-critical | `civic` | `emergency_comms` |
| Water Alert | Safety-critical | `commons_care` | `water_purity` |
| Hearth Alert | Safety-critical | `hearth` | `hearth_emergency` |
| Food Harvest | Safety-critical | `commons_care` | `food_production` |
| Knowledge Claim | Lower-priority | `knowledge` | `claims` |
| Care Circle Update | Lower-priority | `commons_care` | `care_circles` |
| Shelter Update | Lower-priority | `commons_care` | `housing_units` |
| Supply Update | Lower-priority | `supplychain` | `inventory_coordinator` |
| Mutual Aid Offer | Lower-priority | `commons_care` | `mutualaid_timebank` |
| Price Report | Lower-priority | `finance` | `price_oracle` |
| Heartbeat | Periodic (every 4 polls) | -- | -- |

All payloads are serialized with bincode, BLAKE3-hashed for
deduplication, and fragmented to fit within the 255-byte LoRa frame
limit. A typical TEND exchange or emergency message fits in a single
frame. Larger payloads (e.g., knowledge claims with many tags) may span
2-3 frames and are reassembled on the receiving end.
