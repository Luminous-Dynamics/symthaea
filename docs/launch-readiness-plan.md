# Sovereign Inoculation — Launch Readiness Plan

## Philosophy

**No error messages. Solutions with permission.**

Instead of "Error: hostname invalid", say "Your hostname has characters NixOS can't use. Can I change it to [suggested-fix]?"

Instead of "Error: ESP too small", say "Your boot partition is 100MB. NixOS needs ~200MB. I can resize it safely — this won't affect Windows. Shall I proceed?"

Instead of "Error: WebSocket disconnected", show "The install is still running on the target. I lost the live view — reconnecting..." and poll until reconnected.

The user should never need to understand what went wrong. They just need to say yes or no to a proposed fix.

---

## Phase 1: Safety (prevent data loss or security issues)

### 1.1 Input Sanitization (all fields)

| Field | Validation | Auto-fix |
|-------|-----------|----------|
| Hostname | `^[a-z][a-z0-9-]{0,62}$` | Lowercase, replace spaces/special chars with `-`, truncate |
| Disk path | Must start with `/dev/`, no `;&#\|` chars | Reject with "That doesn't look like a disk path" |
| Username | POSIX: `^[a-z_][a-z0-9_-]{0,31}$` | Lowercase, replace spaces |
| Timezone | Must exist in IANA tz database | Fuzzy match: "America/Chciago" → "Did you mean America/Chicago?" |
| Keyboard | Must be in known layout list | Dropdown only (can't type free text) |
| Passphrase | Min 8 chars | Show strength meter, suggest longer |
| Relay URL | Must be `ws://` or `wss://` + valid host | Auto-prepend `ws://` if missing |

**Implementation**: Validate in the portal JS (immediate feedback) AND in the relay Rust code (defense in depth). Shell scripts must never receive unsanitized input.

```rust
// In ssh_relay.rs, before generate_install_script():
fn sanitize_hostname(h: &str) -> String {
    let clean: String = h.to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' })
        .collect();
    let clean = clean.trim_matches('-').to_string();
    if clean.is_empty() { "nixos".into() }
    else if clean.len() > 63 { clean[..63].to_string() }
    else { clean }
}

fn validate_disk_path(d: &str) -> Result<(), String> {
    if !d.starts_with("/dev/") { return Err("Disk must start with /dev/".into()); }
    if d.contains(';') || d.contains('&') || d.contains('|') || d.contains('`')
       || d.contains('$') || d.contains('(') || d.contains(')') {
        return Err("Disk path contains unsafe characters".into());
    }
    Ok(())
}
```

### 1.2 Passphrase Safety

- Never log the passphrase (not even to stderr)
- Pipe directly to `cryptsetup` stdin, never write to a file
- Zero memory after use (use `zeroize` crate)
- Portal: mask input, no autocomplete, clear on disconnect

### 1.3 ESP Size Check (prevent bootloader corruption)

Before alongside install:
```bash
# Check ESP size
ESP_SIZE=$(lsblk -bnro SIZE "$EFI_PART")
ESP_MB=$((ESP_SIZE / 1048576))
if [ "$ESP_MB" -lt 150 ]; then
  echo "PROMPT: Your boot partition is ${ESP_MB}MB. NixOS needs about 200MB."
  echo "PROMPT: I can create a second small boot partition for NixOS instead of sharing."
  echo "PROMPT: Your Windows bootloader stays untouched. Approve?"
  # Wait for user approval via WebSocket
fi
```

### 1.4 Pre-Install Validation

Before any destructive operation:
```bash
# Verify disk exists and is a block device
[ -b "$DISK" ] || { echo "ERROR: $DISK is not a block device"; exit 1; }

# Verify disk is not mounted
if mount | grep -q "$DISK"; then
  echo "PROMPT: $DISK has mounted partitions. I need to unmount them to proceed. Allow?"
fi

# Verify disk is big enough (minimum 20GB)
DISK_SIZE=$(($(blockdev --getsize64 "$DISK") / 1073741824))
if [ "$DISK_SIZE" -lt 20 ]; then
  echo "PROMPT: This disk is only ${DISK_SIZE}GB. NixOS needs at least 20GB. Choose a different disk?"
fi
```

### 1.5 Atomic Install with Rollback Point

```bash
# Take partition table snapshot BEFORE any destructive operation
sfdisk -d "$DISK" > /tmp/pre-install-partition-table.dump
echo "ROLLBACK: If anything goes wrong, restore with: sfdisk $DISK < /tmp/pre-install-partition-table.dump"

# If nixos-install fails:
if ! nixos-install --no-root-passwd; then
  echo "PROMPT: Installation failed. Your disk has been partitioned but NixOS isn't fully installed."
  echo "PROMPT: Options:"
  echo "PROMPT: 1. Retry the install (keeps current partitions)"
  echo "PROMPT: 2. Restore original partition table (undo everything)"
  echo "PROMPT: 3. Open a terminal to debug manually"
fi
```

---

## Phase 2: Resilience (handle failures gracefully)

### 2.1 WebSocket Reconnect

When the WebSocket drops during install:

```javascript
// In tab-inoculate.js
var reconnectAttempts = 0;
var installSessionId = null;

ws.onclose = function() {
  if (installInProgress) {
    sshStatus.textContent = 'Lost connection — the install is still running. Reconnecting...';
    sshStatus.style.color = 'var(--solar-gold)';
    attemptReconnect();
  }
};

function attemptReconnect() {
  reconnectAttempts++;
  var delay = Math.min(reconnectAttempts * 2000, 10000); // 2s, 4s, 6s, 8s, 10s
  setTimeout(async function() {
    try {
      ws = new WebSocket(relayUrl);
      ws.onopen = function() {
        // Reconnected — check install status
        ws.send(JSON.stringify({
          action: 'check_install_status',
          host: host, port: port, username: user, password: pass
        }));
        sshStatus.textContent = 'Reconnected! Checking install status...';
        reconnectAttempts = 0;
      };
      ws.onmessage = originalHandler;
      ws.onclose = function() { attemptReconnect(); };
    } catch(e) {
      sshStatus.textContent = 'Reconnecting... (attempt ' + reconnectAttempts + ')';
      attemptReconnect();
    }
  }, delay);
}
```

New relay action: `check_install_status`
```bash
# SSH into target, check if install is still running
if pgrep -f symthaea-install > /dev/null; then
  echo '{"status":"running","log_tail":"..last 10 lines..."}'
elif [ -f /tmp/symthaea-install.log ] && grep -q "COMPLETE" /tmp/symthaea-install.log; then
  echo '{"status":"complete"}'
else
  echo '{"status":"failed","log_tail":"..last 10 lines..."}'
fi
```

### 2.2 Interactive Prompts via WebSocket

Instead of `echo "ERROR"` → `exit 1`, the install script can pause and ask the user:

New message type: `prompt`
```json
{
  "type": "prompt",
  "id": "esp-too-small",
  "message": "Your boot partition is 100MB. I can create a separate boot partition for NixOS.",
  "options": [
    {"id": "create-separate", "label": "Create separate boot partition", "recommended": true},
    {"id": "share-anyway", "label": "Share existing boot partition (risky)", "recommended": false},
    {"id": "cancel", "label": "Cancel installation"}
  ]
}
```

Portal shows a modal with the options. User taps one. Response sent back:
```json
{"type": "prompt_response", "id": "esp-too-small", "choice": "create-separate"}
```

The install script reads the response from a named pipe:
```bash
# In install script:
ask_user() {
  local prompt_id="$1"
  local message="$2"
  # Write prompt to a file the relay watches
  echo "$message" > /tmp/symthaea-prompt
  echo "$prompt_id" > /tmp/symthaea-prompt-id
  # Wait for response (relay writes it)
  while [ ! -f /tmp/symthaea-response ]; do sleep 1; done
  RESPONSE=$(cat /tmp/symthaea-response)
  rm /tmp/symthaea-response /tmp/symthaea-prompt /tmp/symthaea-prompt-id
  echo "$RESPONSE"
}

# Usage:
ESP_MB=$(get_esp_size)
if [ "$ESP_MB" -lt 150 ]; then
  CHOICE=$(ask_user "esp-small" "Boot partition is ${ESP_MB}MB. Create separate?")
  case "$CHOICE" in
    "create-separate") create_separate_boot_partition ;;
    "share-anyway") echo "Proceeding with shared ESP..." ;;
    "cancel") exit 0 ;;
  esac
fi
```

### 2.3 Alongside: Handle "No Free Space"

Instead of failing, offer to help:
```
"Your disk has no free space for NixOS. Here's what I can do:

1. Shrink [Windows partition] by 100GB (safe — I'll check for data first)
2. Use a different disk [show disk selector again]
3. Replace the existing OS entirely (full wipe)"
```

### 2.4 Alongside: Handle "No Existing OS"

```
"You chose 'Alongside' but this disk is empty. Did you mean:

1. Full disk install (recommended for empty disks)
2. Alongside — I'll leave the rest of the disk empty for another OS later"
```

---

## Phase 3: Custom ISO + Auto-Discovery

### 3.1 Custom NixOS ISO with Relay Built In

```nix
# nix/installer-iso.nix
{ config, pkgs, modulesPath, ... }:
{
  imports = [
    (modulesPath + "/installer/cd-dvd/installation-cd-minimal.nix")
  ];

  # Pre-install the relay
  environment.systemPackages = with pkgs; [
    sovereign-inoculation-relay  # our relay binary as a Nix package
    btrfs-progs parted cryptsetup
    ntfs3g  # for Windows partition scanning
  ];

  # Auto-start relay on boot
  systemd.services.sovereign-relay = {
    description = "Sovereign Inoculation SSH Relay";
    after = [ "network.target" "sshd.service" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      ExecStart = "${pkgs.sovereign-inoculation-relay}/bin/ssh-relay --port 8094";
      Restart = "always";
    };
  };

  # Auto-start SSH
  services.openssh = {
    enable = true;
    settings.PermitRootLogin = "yes";
  };

  # Set root password to "sovereign"
  users.users.root.initialPassword = "sovereign";

  # mDNS for auto-discovery
  services.avahi = {
    enable = true;
    publish = {
      enable = true;
      addresses = true;
      workstation = true;
    };
    extraServiceFiles.sovereign = ''
      <?xml version="1.0" standalone='no'?>
      <!DOCTYPE service-group SYSTEM "avahi-service.dtd">
      <service-group>
        <name>Sovereign Inoculation</name>
        <service>
          <type>_sovereign._tcp</type>
          <port>8094</port>
          <txt-record>version=1.0</txt-record>
        </service>
      </service-group>
    '';
  };

  # Show clear instructions on console
  services.getty.helpLine = ''

    ╔══════════════════════════════════════════════════════════════╗
    ║  SOVEREIGN INOCULATION — NixOS Installer                   ║
    ║                                                            ║
    ║  Open in your browser (any device on this network):        ║
    ║    https://install.nixforhumanity.org                      ║
    ║                                                            ║
    ║  SSH Relay: ws://<this-ip>:8094                            ║
    ║  SSH Password: sovereign                                   ║
    ║                                                            ║
    ║  Or install manually:                                      ║
    ║    Login as root / sovereign                               ║
    ╚══════════════════════════════════════════════════════════════╝
  '';

  # Display IP address on login screen
  services.getty.greetingLine = ''
    NixOS Sovereign Inoculation ISO
    Network: \4
  '';
}
```

Build: `nix build .#nixosConfigurations.installer-iso.config.system.build.isoImage`

### 3.2 mDNS Auto-Discovery in Portal

```javascript
// In tab-inoculate.js — auto-discover relay on LAN
async function discoverRelay() {
  // Method 1: Try common LAN patterns
  var candidates = [];

  // Method 2: If we have the hostname, try .local
  candidates.push('ws://sovereign-inoculation.local:8094');
  candidates.push('ws://nixos.local:8094');

  // Method 3: Scan common private IPs (fast parallel check)
  // Get our own IP's subnet
  try {
    var rtc = new RTCPeerConnection({iceServers: []});
    rtc.createDataChannel('');
    var offer = await rtc.createOffer();
    await rtc.setLocalDescription(offer);
    await new Promise(r => setTimeout(r, 500));
    var match = rtc.localDescription.sdp.match(/c=IN IP4 (\d+\.\d+\.\d+)\./);
    if (match) {
      var subnet = match[1];
      // Try .1 through .10 and .100-.110 (common DHCP ranges)
      for (var i of [1,2,3,4,5,100,101,102,103,104,105,110]) {
        candidates.push('ws://' + subnet + '.' + i + ':8094');
      }
    }
    rtc.close();
  } catch(e) {}

  // Try each candidate with 2s timeout
  for (var url of candidates) {
    try {
      var ws = await new Promise(function(resolve, reject) {
        var s = new WebSocket(url);
        s.onopen = function() { resolve(s); };
        s.onerror = function() { reject(); };
        setTimeout(function() { reject(); }, 2000);
      });
      ws.close();
      return url; // Found it!
    } catch(e) {}
  }
  return null; // Not found — user must enter manually
}

// On page load, try auto-discovery
discoverRelay().then(function(url) {
  if (url) {
    var relayInput = document.getElementById('ssh-relay-url');
    if (relayInput && !relayInput.value) {
      relayInput.value = url;
      relayInput.style.borderColor = 'var(--leaf-green)';
      window.addNarration('Found relay at ' + url);
    }
  }
});
```

### 3.3 Every Failure Becomes a Conversation

Replace all `echo "ERROR:..."` / `exit 1` patterns with interactive prompts:

| Situation | Old Behavior | New Behavior |
|-----------|-------------|-------------|
| Disk too small | "ERROR: Less than 20GB" exit | "This disk is 15GB. NixOS needs 20GB minimum. [Pick another disk] [Try minimal install (experimental)]" |
| nixos-install fails | Terminal shows error | "The install hit a problem: [error]. [Retry] [Show details] [Restore disk and cancel]" |
| Can't mount Windows | "ERROR: ntfs mount failed" | "I can't read the Windows partition. The ntfs driver might be missing from this ISO. [Skip app scanning] [Try different driver]" |
| Network timeout during download | Install hangs silently | "Package download stalled. [Retry] [Switch to a different cache mirror] [Continue offline with cached packages]" |
| Partition table already exists | Overwrites silently | "This disk already has partitions: [list]. [Wipe everything] [Use free space only] [Cancel]" |
| Hostname conflict (dual-boot) | Ignores | "There's already a system called 'guardian' on this network. [Keep name] [Change to guardian-2] [Choose custom name]" |

### 3.4 Progress That Doesn't Lie

```javascript
// Replace percentage-based progress with stage-based
var INSTALL_STAGES = [
  { id: 'snapshot', label: 'Backing up disk state', weight: 1 },
  { id: 'partition', label: 'Creating partitions', weight: 2 },
  { id: 'format', label: 'Formatting with btrfs', weight: 2 },
  { id: 'mount', label: 'Mounting filesystems', weight: 1 },
  { id: 'config', label: 'Generating configuration', weight: 2 },
  { id: 'sysconfig', label: 'Applying your choices', weight: 1 },
  { id: 'swap', label: 'Setting up swap', weight: 1 },
  { id: 'download', label: 'Downloading packages', weight: 40 },  // biggest
  { id: 'install', label: 'Installing NixOS', weight: 15 },
  { id: 'boot', label: 'Installing bootloader', weight: 2 },
  { id: 'verify', label: 'Verifying installation', weight: 1 },
  { id: 'gitinit', label: 'Version-controlling your config', weight: 1 },
  { id: 'firstbreath', label: 'First breath', weight: 1 },
];

// Show: "Downloading packages (847 of ~4,500)..." not "70%"
// The "downloading" stage uses byte-counting for smooth progress
// Other stages use elapsed/estimated time
```

---

## Implementation Priority

### Session 1 (Critical Safety)
1. Input sanitization (hostname, disk path, timezone) — Rust + JS
2. Passphrase safety (never log, zeroize)
3. ESP size check before alongside
4. Disk validation (exists, big enough, not mounted)
5. Pre-install partition table backup (already partially done)

### Session 2 (Resilience)
6. WebSocket reconnect flow
7. `check_install_status` relay action
8. Interactive prompts via WebSocket (prompt/response protocol)
9. Alongside: handle no free space, no existing OS
10. nixos-install failure → retry/rollback offer

### Session 3 (Custom ISO + Discovery)
11. Package the relay as a Nix derivation
12. Build custom ISO with relay + SSH + avahi auto-start
13. mDNS auto-discovery in portal
14. Console greeting with instructions + IP address

### Session 4 (Polish)
15. Stage-based progress (not fake percentages)
16. Every error → conversation with options
17. Test all 13 personas against the new prompts
18. Mobile test on Pixel 8 Pro with custom ISO
