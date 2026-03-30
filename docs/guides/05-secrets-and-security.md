# Secrets & Security

*Your NixOS config is in git. Your passwords should NOT be. Here's how to handle both.*

---

## The Problem

NixOS configs are declarative and version-controlled. But some things are secrets:
- WiFi passwords
- API keys
- Database credentials
- SSH private keys
- TLS certificates

If you put these in `configuration.nix`, they end up in the Nix store (world-readable) and your git history.

---

## Option 1: sops-nix (Recommended)

**sops-nix** encrypts secrets with your age key. They're decrypted at activation time and placed where services expect them.

### Setup

```bash
# Generate an age key (if you don't have one)
mkdir -p ~/.config/sops/age
age-keygen -o ~/.config/sops/age/keys.txt

# Get your public key
age-keygen -y ~/.config/sops/age/keys.txt
# Output: age1abc123...
```

Add to your flake:
```nix
{
  inputs.sops-nix.url = "github:Mic92/sops-nix";

  outputs = { nixpkgs, sops-nix, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      modules = [
        sops-nix.nixosModules.sops
        ./configuration.nix
      ];
    };
  };
}
```

### Create a secrets file

```bash
# Create .sops.yaml in your config root
cat > .sops.yaml << 'EOF'
keys:
  - &admin age1abc123...  # your public key
creation_rules:
  - path_regex: secrets/.*\.yaml$
    key_groups:
      - age:
          - *admin
EOF

# Create an encrypted secrets file
mkdir secrets
sops secrets/passwords.yaml
# This opens your editor. Add secrets as YAML:
# wifi_password: "your-wifi-password"
# db_password: "postgres-password"
# api_key: "sk-abc123"
```

### Use secrets in your config

```nix
# configuration.nix
{
  sops.defaultSopsFile = ./secrets/passwords.yaml;
  sops.age.keyFile = "/home/youruser/.config/sops/age/keys.txt";

  sops.secrets.wifi_password = { };
  sops.secrets.db_password = {
    owner = "postgres";
    group = "postgres";
  };

  # Use the secret (it's a file path at runtime)
  services.postgresql.initialScript = pkgs.writeText "init.sql" ''
    ALTER USER postgres PASSWORD '$(cat ${config.sops.secrets.db_password.path})';
  '';

  # WiFi with secret password
  networking.networkmanager.ensureProfiles.profiles.home = {
    connection = { id = "HomeWiFi"; type = "wifi"; };
    wifi.ssid = "HomeWiFi";
    wifi-security = {
      key-mgmt = "wpa-psk";
      psk = "$(__FILE{${config.sops.secrets.wifi_password.path}})";
    };
  };
}
```

The secrets file is encrypted in git. Only your age key can decrypt it. The Nix store never sees the plaintext.

---

## Option 2: agenix (Simpler)

If you prefer a lighter approach:

```nix
{
  inputs.agenix.url = "github:ryantm/agenix";

  outputs = { nixpkgs, agenix, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      modules = [
        agenix.nixosModules.default
        ./configuration.nix
      ];
    };
  };
}
```

```nix
# secrets.nix (lists who can decrypt what)
let
  myKey = "ssh-ed25519 AAAA...";  # your SSH public key
in {
  "secrets/wifi.age".publicKeys = [ myKey ];
  "secrets/db.age".publicKeys = [ myKey ];
}
```

```bash
# Encrypt
agenix -e secrets/wifi.age
# Decrypts with your SSH key at activation time
```

---

## Security Hardening Checklist

```nix
{
  # Firewall (already enabled by Sovereign Inoculation)
  networking.firewall = {
    enable = true;
    allowedTCPPorts = [ 22 ];  # only SSH
    # allowedTCPPorts = [ 80 443 ];  # add if running a web server
  };

  # SSH hardening
  services.openssh = {
    enable = true;
    settings = {
      PasswordAuthentication = false;  # key-only
      PermitRootLogin = "no";
      X11Forwarding = false;
    };
  };

  # Automatic security updates
  system.autoUpgrade = {
    enable = true;
    dates = "04:00";
    allowReboot = false;  # true if you want unattended reboots
  };

  # Fail2ban (brute force protection)
  services.fail2ban = {
    enable = true;
    maxretry = 3;
    bantime = "1h";
  };

  # Audit logging
  security.auditd.enable = true;

  # Kernel hardening
  boot.kernel.sysctl = {
    "kernel.unprivileged_bpf_disabled" = 1;
    "net.core.bpf_jit_harden" = 2;
    "kernel.kptr_restrict" = 2;
  };

  # AppArmor (optional, significant security improvement)
  security.apparmor.enable = true;
}
```

---

## Disk Encryption Maintenance

If you installed with LUKS (Sovereign Inoculation's encrypted layout):

```bash
# Check encryption status
sudo cryptsetup luksDump /dev/nvme0n1p2

# Add a backup passphrase (in case you forget the primary)
sudo cryptsetup luksAddKey /dev/nvme0n1p2

# Change your passphrase
sudo cryptsetup luksChangeKey /dev/nvme0n1p2

# If you have TPM2 auto-unlock and it fails after firmware update:
sudo systemd-cryptenroll /dev/nvme0n1p2 --wipe-slot=tpm2
sudo systemd-cryptenroll /dev/nvme0n1p2 --tpm2-device=auto --tpm2-pcrs=0+7
```

---

## Secure Boot Maintenance

If Sovereign Inoculation enabled Secure Boot (lanzaboote):

```bash
# Check Secure Boot status
bootctl status

# Verify all boot entries are signed
sbctl verify

# After NixOS rebuild, keys are automatically applied
# If you need to re-enroll keys:
sudo sbctl enroll-keys --microsoft

# List enrolled keys
sbctl list-keys
```

---

*Your system is as secure as you make it. NixOS gives you the tools — you choose how tight to lock the door.*
