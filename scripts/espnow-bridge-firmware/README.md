# ESP-NOW Bridge Firmware

Minimal Arduino/ESP-IDF firmware for an ESP32 that bridges serial SLIP ↔ ESP-NOW.

## Purpose

Gives Linux hosts (RPi, laptop) access to ESP-NOW's peer-to-peer WiFi radio
without native support. The ESP32 is a "dumb radio" — all intelligence lives
in the Spore daemon and mesh-bridge on the host.

## Protocol

Host sends SLIP-encoded frames over serial. ESP32 broadcasts them via ESP-NOW.
ESP32 receives ESP-NOW frames from air, sends them back via SLIP to serial.

### SLIP Framing (RFC 1055)
```
[0xC0] [payload with ESC sequences] [0xC0]
0xDB 0xDC → literal 0xC0
0xDB 0xDD → literal 0xDB
```

### Wiring
```
RPi GPIO14 (TX) → ESP32 GPIO16 (RX)
RPi GPIO15 (RX) → ESP32 GPIO17 (TX)
RPi GND         → ESP32 GND
RPi 3.3V        → ESP32 3.3V (or power via USB)
```

Or simply connect via USB — the ESP32's USB-UART bridge handles it.

## Hardware

Any ESP32 board works: ESP32-DevKitC (~$5), ESP32-S3 (~$8), etc.
The board just needs a serial port and WiFi radio (all ESP32s have both).

## Build

```bash
# Arduino CLI
arduino-cli compile --fqbn esp32:esp32:esp32 espnow_bridge.ino
arduino-cli upload --fqbn esp32:esp32:esp32 -p /dev/ttyUSB0 espnow_bridge.ino

# ESP-IDF (alternative)
idf.py build flash monitor
```

## Configuration

- `WIFI_CHANNEL`: 1-13 (default: 1, must match all peers)
- `BAUD_RATE`: 921600 (matches ESPNOW_BAUD in mesh-bridge)
- `BROADCAST_MAC`: FF:FF:FF:FF:FF:FF (all peers)
