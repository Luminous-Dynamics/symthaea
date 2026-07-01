# Vendored scrcpy-server

## What's here

| File | Purpose |
|---|---|
| `scrcpy-server-v2.4.jar` | scrcpy v2.4 server component — pushed to `/data/local/tmp/scrcpy-server.jar` on the device, executed via `app_process` over ADB shell |
| `scrcpy-server-v2.4.jar.sha256` | Pinned SHA256 — verified against upstream `SHA256SUMS.txt` at fetch time |

## Provenance

- **Upstream project**: https://github.com/Genymobile/scrcpy
- **Release**: v2.4 (2024)
- **Asset URL**: https://github.com/Genymobile/scrcpy/releases/download/v2.4/scrcpy-server-v2.4
- **Upstream SHA256SUMS**: https://github.com/Genymobile/scrcpy/releases/download/v2.4/SHA256SUMS.txt
- **License**: Apache 2.0 (see scrcpy LICENSE at https://github.com/Genymobile/scrcpy/blob/v2.4/LICENSE)
- **Fetched at**: 2026-04-14 by Phase I.B session

## Why v2.4

- First scrcpy release with stable AV1 encoding support (`--video-codec=av1`)
- Compatible with Android 14 (Pixel 8 Pro ships Android 14+)
- Stable binary framing on the reverse-tunnel wire

## Why vendored (not installed via nix)

- `scrcpy` the host package is not required — Symthaea only consumes the
  `.jar` which it pushes to the device itself via ADB
- Pinning the exact JAR + SHA guarantees reproducible device-side behavior
  across host NixOS rebuilds
- Offline-capable: the binary framing parser in `src/scrcpy.rs` is pinned
  to this server version; an upstream scrcpy-server update could change
  the wire format, so we pin

## Rebuild instructions

To upgrade to a newer scrcpy release:

```bash
cd crates/symthaea-phone-embodiment/vendor
VERSION=2.5  # or whatever
curl -sSL -O "https://github.com/Genymobile/scrcpy/releases/download/v${VERSION}/scrcpy-server-v${VERSION}"
mv "scrcpy-server-v${VERSION}" "scrcpy-server-v${VERSION}.jar"
sha256sum "scrcpy-server-v${VERSION}.jar" > "scrcpy-server-v${VERSION}.jar.sha256"

# Verify against upstream
curl -sSL "https://github.com/Genymobile/scrcpy/releases/download/v${VERSION}/SHA256SUMS.txt" | grep "scrcpy-server-v${VERSION}$"
# Must match the line in the .sha256 file above (modulo the .jar suffix)

# Then update `SCRCPY_SERVER_JAR` const in `src/scrcpy.rs` to point at
# the new file name, and re-run the codec probe to confirm binary
# framing compatibility.
```

## Runtime verification

At startup, `src/scrcpy.rs::push_scrcpy_server()` verifies the on-disk
JAR SHA matches `scrcpy-server-v2.4.jar.sha256` before pushing to the
device. This catches:

- Supply-chain tampering (file modified in-tree after commit)
- Accidental corruption (partial git checkout, storage bit-flip)
- Version drift (someone replaced the JAR without updating the SHA file)
