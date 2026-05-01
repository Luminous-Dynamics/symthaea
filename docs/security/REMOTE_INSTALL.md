# Sovereign Inoculation: Hardened Remote Install Flow

This repo is security-hardened to avoid the historical pattern of "installer runs as a LAN-wide root backdoor".
The intended remote install flow is now:

- Target machine: runs the installer ISO, exposes **SSH only**.
- Operator machine: runs the **SSH relay locally** (localhost-only + token auth) and drives install from the Sensorium UI.

## What Changed (Security Posture)

- Installer ISO:
  - No static root password.
  - Generates a **one-time root password on boot** and prints it on `tty1`.
  - Firewall enabled; only TCP/22 allowed.
  - No Avahi/mDNS broadcast.
- SSH relay:
  - Binds to `127.0.0.1` only.
  - Requires a randomly generated **WebSocket auth token**.
  - Enforces SSH host-key verification (known_hosts).

## Remote Install Steps

1) Boot the installer ISO on the target machine.

2) Connect the target to the network (Ethernet or Wi-Fi).

3) On the target console (`tty1`), note:
   - Target IP address
   - One-time root password

   The password is also stored on the ISO at:
   - `/run/sovereign-inoculation/root-password`

4) From the operator machine, verify you can SSH in:

```bash
ssh root@<TARGET_IP>
```

5) On the operator machine, start the localhost-only relay and capture the printed token:

```bash
cd symthaea
RUSTC_WRAPPER= cargo run -p symthaea-spore --features server --bin ssh-relay
```

6) Open the Sensorium UI on the operator machine and paste the relay token when prompted.

Notes:
- The relay listens on `127.0.0.1`, so the Sensorium must run on the same machine as the relay, or you must port-forward.
- If you need to access the Sensorium/relay from a second machine, prefer SSH port-forwarding rather than exposing services:

```bash
# Example: forward relay from operator to your laptop
ssh -L 8091:127.0.0.1:8091 <operator_user>@<operator_ip>
```

## Safety Notes

- Avoid untrusted/shared Wi-Fi during install.
- Treat the one-time password as a secret; it regenerates on every reboot.
- Do not bind the relay to `0.0.0.0` (it is intentionally not supported).

## Security Verification (VM Test)

From `symthaea/`, you can run a NixOS VM test that verifies the installer posture:

```bash
nix build .#checks.x86_64-linux.installer-iso-security
```
