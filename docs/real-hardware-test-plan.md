# Real Hardware Test Plan

## Your Machine

| Component | Detail |
|-----------|--------|
| **GPU 0 (NVIDIA)** | RTX 2070 Max-Q [10de:1f10] — card0, nvidia driver loaded, nvidia-smi working |
| **GPU 1 (Intel)** | UHD 630 [8086:3e9b] — card1, i915 driver loaded |
| **Display** | Wayland on NVIDIA (nvidia_drm) |
| **Network** | USB tethering via Pixel 8 Pro (`enp0s20f0u3i1`, IP 10.124.141.121) |
| **WiFi** | Dead (no wireless interface) |
| **Storage** | Samsung 970 EVO 1TB + Intel 1TB NVMe |
| **TPM** | 2.0 available |
| **EFI** | Yes, Secure Boot disabled (setup mode) |
| **Tailscale** | Installed but interface DOWN |

## Key Insight: Both GPUs Are Active

- NVIDIA is card0 (primary display, 82 i915 users means Intel is also active)
- Intel is card1 (i915 loaded with 82 references — likely handling some outputs)
- The display is Wayland running through nvidia_drm

This means **GPU passthrough to a VM is NOT safe** right now — NVIDIA is driving the display.

## What We CAN Test

### 1. Probe-Only Mode Against Real Hardware (Safe)
Run the hardware probe via SSH to localhost — detects real GPUs, real NVMe, real TPM.
No destructive operations. This validates the probe output format on real hardware.

**Risk: ZERO** — probe only reads, never writes.

### 2. USB Tethering Detection
The probe should detect `enp0s20f0u3i1` as a network interface.
Currently the probe checks for WiFi but not USB tethering.
Should add: "Internet via USB tethering (Pixel 8 Pro)" detection.

### 3. Hybrid GPU Detection Validation
The probe should detect BOTH GPUs and flag hybrid graphics.
Should generate: NVIDIA PRIME offload config + Intel as iGPU.
This is the most common laptop config — testing it on real hardware is high value.

### 4. Dead WiFi Card Detection
If the WiFi hardware exists but has no driver, the probe should report:
"WiFi hardware detected but not functional. Check firmware/driver."
Instead of just "WiFi: not available."

## What We CANNOT Test (yet)

### GPU Passthrough
- NVIDIA is driving the display — can't passthrough without losing display
- Would need to: switch to Intel display → unbind NVIDIA → pass to VM
- This requires: `vfio-pci` kernel module, IOMMU enabled in BIOS, reboot
- Not worth the risk during a dev session

### Full Install on Real Hardware
- Would wipe one of the NVMe drives
- Not recommended until we've tested everything else

## Plan

### Phase 1: Safe Probe (do now)
1. Add SSH key to localhost so relay can connect
2. Run `probe_hardware` against localhost
3. Verify: both GPUs detected, hybrid flagged, TPM detected, NVMe detected
4. Compare output to what we know is real

### Phase 2: USB Tethering Detection (do now)
1. Add USB network detection to hardware probe
2. Check for `enp*u*` interfaces (USB ethernet naming convention)
3. Report: "Internet via USB" with device name

### Phase 3: Dead WiFi Detection (do now)
1. Check `lspci` for WiFi hardware even if no interface exists
2. Check for `rfkill` blocked status
3. Report: "WiFi hardware found but not functional"

### Phase 4: NVIDIA PRIME Config Generation (do now)
1. Detect hybrid GPU (count > 1 VGA devices)
2. Parse bus IDs: Intel PCI:0:2:0, NVIDIA PCI:1:0:0
3. Generate `hardware.nvidia.prime.offload` config with correct bus IDs
4. Test through SovereignConfigGenerator
