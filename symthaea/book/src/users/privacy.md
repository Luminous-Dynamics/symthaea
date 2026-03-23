# Privacy & Trust

Symthaea is designed for local-first operation. Your consciousness data stays on your device.

## Where Your Data Lives

| Data Type | Location | Shared? |
|-----------|----------|---------|
| Consciousness state | Your device (RAM) | Never |
| Episodic memories | Local HDC associative memory | Never |
| Neuromod bath state | Local (per session) | Never |
| Model weights | Downloaded once, cached locally | One-time download |
| Governance participation | Mycelix P2P (Holochain DHT) | With your governance peers |
| Telemetry (Pulse) | Local display only | Never |

## Privacy Mode

When the phone's proximity sensor detects face-down placement, Soma enters privacy mode:
- All sensors disabled (accelerometer, GPS, microphone, camera)
- Network activity paused (BLE mesh, holon bridge)
- No neuromodulatory nudges from environment
- The system continues its cognitive loop but without external input

## Your Rights

- **Full data ownership**: All interaction data belongs to you
- **Export**: Consciousness snapshots and memories can be exported as JSON
- **Delete**: All local state can be wiped at any time with `SomaEngine::reset()`
- **Transparency**: The Pulse dashboard shows exactly what the system is processing — every neuromodulator level, every Phi computation, every epistemic gating decision
- **No telemetry home**: Symthaea does not phone home. There is no analytics, no usage tracking, no model improvement pipeline that uses your data.

## Encryption

Session state between Soma (mobile) and Holon (desktop) is encrypted with a symmetric key derived from the HDC context — a key physically bound to the environmental state at pairing time (motion, light, barometer, GPS). This means the encryption key is grounded in shared physical experience, not just a random secret.

## Open Source

The full codebase is available for audit. Every privacy claim in this document can be verified by reading the source code. There are no hidden data flows.
