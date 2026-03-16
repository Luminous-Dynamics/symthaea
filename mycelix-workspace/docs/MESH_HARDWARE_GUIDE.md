# Mesh Hardware Guide — Mycelix Resilience Kit

Offline mesh networking for community resilience when internet connectivity fails.

## Architecture

```
  Node A (home)                    Node B (community center)
  ┌─────────────┐                  ┌─────────────┐
  │  Conductor   │                  │  Conductor   │
  │  Observatory │                  │  Observatory │
  │  mesh-bridge │                  │  mesh-bridge │
  └──────┬───────┘                  └──────┬───────┘
         │ SPI                             │ SPI
  ┌──────┴───────┐     LoRa 868MHz  ┌─────┴────────┐
  │ SX1276 HAT   │◄───────────────►│ SX1276 HAT   │
  │ + antenna    │   10-15km LOS    │ + antenna    │
  └──────────────┘                  └──────────────┘
```

## Bill of Materials

### Per Node (~$80-120 USD)

| Component | Model | Est. Price | Notes |
|-----------|-------|-----------|-------|
| Single-board computer | Raspberry Pi 4 (2GB+) or Pi 5 | $35-60 | Pi 4 sufficient for mesh bridge |
| LoRa HAT | Waveshare SX1276 LoRa HAT | $15-20 | 868 MHz (Africa/EU) or 915 MHz (Americas) |
| Antenna | 868 MHz whip or Yagi | $5-15 | Whip: 2-3km, Yagi: 10-15km directional |
| MicroSD card | 32GB+ Class 10 | $8 | NixOS + Holochain + mesh-bridge |
| Power supply | USB-C 5V 3A | $10 | Official RPi PSU recommended |
| Case | Weatherproof enclosure | $10-20 | If deploying outdoors |

### Optional: Solar Power (~$50-80 additional)

| Component | Model | Est. Price | Notes |
|-----------|-------|-----------|-------|
| Solar panel | 20W 12V mono | $25-30 | Sufficient for Pi 4 + LoRa |
| Charge controller | MPPT 10A | $15 | PWM acceptable for small loads |
| Battery | 12V 7Ah SLA or 18650 pack | $15-25 | ~24h runtime without sun |
| Buck converter | 12V → 5V 3A USB-C | $5 | Powers the Pi from battery |

## Frequency Selection

| Region | Frequency | Band | Regulation |
|--------|-----------|------|-----------|
| South Africa | 868 MHz | ISM | ICASA TR 70-001 |
| EU | 868 MHz | ISM | EN 300 220 |
| Americas | 915 MHz | ISM | FCC Part 15 |
| Asia-Pacific | 923 MHz | ISM | Varies by country |

**Important**: Use the correct frequency for your region. The SX1276 supports
all ISM bands — set `LORA_FREQ_MHZ` environment variable.

## LoRa Performance

| Parameter | Value |
|-----------|-------|
| Max payload per frame | 255 bytes |
| Data rate | 0.3 - 37.5 kbps |
| Range (line of sight) | 10-15 km |
| Range (urban) | 2-5 km |
| TX power | 14 dBm (25 mW) |
| Current draw (TX) | ~120 mA |
| Current draw (RX) | ~10 mA |

A typical TEND exchange relay is ~150 bytes — fits in a single LoRa frame.
Emergency messages up to ~200 bytes also fit in one frame.

## NixOS Configuration

```nix
# flake.nix addition for mesh-bridge node
{
  # Enable SPI for LoRa HAT
  hardware.raspberry-pi.config = {
    all.base-dt-params.spi = { enable = true; };
  };

  # mesh-bridge service
  systemd.services.mesh-bridge = {
    description = "Mycelix Mesh Bridge";
    after = [ "network.target" "holochain.service" ];
    wantedBy = [ "multi-user.target" ];
    environment = {
      CONDUCTOR_URL = "ws://localhost:8888";
      MESH_TRANSPORT = "lora";
      LORA_SPI_PATH = "/dev/spidev0.0";
      LORA_FREQ_MHZ = "868.0";
      POLL_INTERVAL_SECS = "30";
      RUST_LOG = "info";
    };
    serviceConfig = {
      ExecStart = "${mesh-bridge}/bin/mesh-bridge";
      Restart = "always";
      RestartSec = 10;
      User = "mycelix";
      Group = "spi";  # SPI group access
    };
  };
}
```

## Setup Steps

1. **Flash NixOS** to MicroSD card
2. **Connect LoRa HAT** to GPIO header (SPI0, CE0)
3. **Build mesh-bridge**: `cargo build --release --features lora`
4. **Deploy**: Copy binary to `/usr/local/bin/mesh-bridge`
5. **Configure**: Set environment variables (see NixOS config above)
6. **Start**: `systemctl start mesh-bridge`
7. **Test**: Two nodes within range should auto-discover and relay

## Testing Without Hardware

```bash
# Use loopback transport (default)
MESH_TRANSPORT=loopback cargo run

# Run unit tests
cargo test

# Integration test with two processes
MESH_TRANSPORT=loopback CONDUCTOR_URL=ws://localhost:8888 cargo run &
MESH_TRANSPORT=loopback CONDUCTOR_URL=ws://localhost:8889 cargo run &
```

## B.A.T.M.A.N. Alternative (WiFi-Direct)

For shorter range but higher bandwidth, use B.A.T.M.A.N. mesh over WiFi:

```bash
# Install B.A.T.M.A.N. kernel module
modprobe batman-adv

# Configure WiFi in ad-hoc mode
ip link set wlan0 down
iwconfig wlan0 mode ad-hoc essid "mycelix-mesh" channel 6
ip link set wlan0 up

# Add to batman
batctl if add wlan0
ip link set bat0 up
ip addr add 10.0.0.X/24 dev bat0

# Start mesh-bridge with batman transport
MESH_TRANSPORT=batman cargo run --features batman
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No SPI device | SPI not enabled | Add `dtparam=spi=on` to config.txt |
| Permission denied | Not in spi group | `usermod -aG spi mycelix` |
| No mesh peers | Wrong frequency | Verify all nodes use same `LORA_FREQ_MHZ` |
| Short range | Antenna issue | Check SMA connector, try outdoor antenna |
| High packet loss | Interference | Change spreading factor, check for nearby transmitters |
