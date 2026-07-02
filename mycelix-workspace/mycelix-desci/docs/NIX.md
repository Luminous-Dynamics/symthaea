# NixOS & Nix Flakes Guide for Mycelix-DeSci

## Why NixOS for Science? 🔬

Mycelix-DeSci provides **first-class NixOS support** because **reproducibility is fundamental to science**.

### Benefits of Nix for Scientific Computing

1. **🔒 Bit-for-Bit Reproducibility**
   - Same inputs → Same outputs, always
   - No "works on my machine" problems
   - Essential for scientific validity

2. **📦 Declarative Dependencies**
   - All dependencies explicitly declared
   - No hidden system dependencies
   - Complete dependency tree captured

3. **⏪ Atomic Rollbacks**
   - Upgrade safely, rollback instantly
   - No broken system states
   - Production deployments with confidence

4. **🔐 Isolation & Security**
   - Each build in isolated environment
   - No dependency conflicts
   - Minimal attack surface

5. **📚 Self-Documenting**
   - Build process IS the documentation
   - Reproducible across time and machines
   - Long-term archival of research environments

---

## Quick Start

### Prerequisites

Install Nix with flakes enabled:

```bash
# Install Nix (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install

# Or use the official installer
sh <(curl -L https://nixos.org/nix/install) --daemon

# Enable flakes (add to ~/.config/nix/nix.conf or /etc/nix/nix.conf)
experimental-features = nix-command flakes
```

### Build and Run

```bash
# Clone repository
git clone https://github.com/Luminous-Dynamics/mycelix-desci
cd mycelix-desci

# Build all packages
nix build

# Run API server
nix run .#api

# Run CLI tool
nix run .#cli -- --help

# Enter development shell
nix develop
```

That's it! Nix handles all dependencies automatically.

---

## Building Packages

### Build Individual Components

```bash
# Build core library
nix build .#mycelix-core

# Build API server
nix build .#mycelix-api

# Build CLI tool
nix build .#mycelix-cli

# Build all at once
nix build .#all

# Build Docker image
nix build .#dockerImage
docker load < result
```

### Check Build Reproducibility

```bash
# Build twice and compare hashes
nix build .#mycelix-api
nix build .#mycelix-api --rebuild

# Both builds should produce identical output
nix path-info ./result
```

---

## Running Applications

### Run API Server

```bash
# Default configuration
nix run .#api

# With custom port
PORT=3000 nix run .#api

# With custom log level
RUST_LOG=debug nix run .#api
```

### Run CLI Tool

```bash
# Show help
nix run .#cli -- --help

# Check system health
nix run .#cli -- system health

# Create a claim
nix run .#cli -- claims create claim.json
```

---

## Development Environment

### Automatic with direnv (Recommended)

```bash
# Install direnv and nix-direnv
nix-env -iA nixpkgs.direnv nixpkgs.nix-direnv

# Add to your ~/.bashrc or ~/.zshrc
eval "$(direnv hook bash)"   # or zsh, fish, etc.

# Allow direnv in project directory
cd mycelix-desci
direnv allow

# Environment automatically loads when you cd into the directory!
```

### Manual Entry

```bash
# Enter development shell
nix develop

# You now have all development tools available:
cargo build --release
cargo test --all
cargo bench
```

### Development Tools Included

The Nix development shell includes:

- ✅ Rust 1.75.0 (pinned for reproducibility)
- ✅ rust-analyzer (LSP)
- ✅ cargo-watch (auto-rebuild)
- ✅ cargo-edit (dependency management)
- ✅ cargo-audit (security audits)
- ✅ cargo-tarpaulin (code coverage)
- ✅ jq, curl, httpie (utilities)
- ✅ Docker & docker-compose

---

## NixOS System Integration

### Enable as a System Service

Add to your `/etc/nixos/configuration.nix`:

```nix
{
  # Import the Mycelix-DeSci flake
  inputs.mycelix-desci.url = "github:Luminous-Dynamics/mycelix-desci";

  # In your configuration:
  imports = [
    inputs.mycelix-desci.nixosModules.default
  ];

  # Enable the service
  services.mycelix-desci = {
    enable = true;
    port = 8080;
    logLevel = "info";
    openFirewall = true;
  };
}
```

Then rebuild your system:

```bash
sudo nixos-rebuild switch
```

### Service Configuration Options

```nix
services.mycelix-desci = {
  enable = true;              # Enable the service
  port = 8080;                # API port (default: 8080)
  host = "0.0.0.0";           # Bind address (default: 0.0.0.0)
  logLevel = "info";          # Log level: trace|debug|info|warn|error
  corsOrigins = "*";          # CORS origins
  user = "mycelix";           # Service user (default: mycelix)
  group = "mycelix";          # Service group (default: mycelix)
  dataDir = "/var/lib/mycelix";  # Data directory
  openFirewall = true;        # Open firewall port

  # Additional environment variables
  extraEnvironment = {
    CUSTOM_VAR = "value";
  };
};
```

### Service Management

```bash
# Start service
sudo systemctl start mycelix-api

# Stop service
sudo systemctl stop mycelix-api

# Restart service
sudo systemctl restart mycelix-api

# Check status
sudo systemctl status mycelix-api

# View logs
sudo journalctl -u mycelix-api -f

# Enable at boot
sudo systemctl enable mycelix-api
```

---

## Docker with Nix

### Build Reproducible Docker Image

```bash
# Build Docker image with Nix
nix build .#dockerImage

# Load into Docker
docker load < result

# Run the image
docker run -p 8080:8080 mycelix-api:latest

# Check system health
curl http://localhost:8080/health
```

### Why Nix-built Docker Images?

- ✅ **Reproducible** - Same image every time
- ✅ **Minimal** - Only necessary dependencies included
- ✅ **Layered** - Efficient storage and transfer
- ✅ **Secure** - No unnecessary tools in runtime
- ✅ **Auditable** - Complete build process captured

---

## Continuous Integration

### GitHub Actions with Nix

Example `.github/workflows/nix-build.yml`:

```yaml
name: Nix Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Nix
        uses: DeterminateSystems/nix-installer-action@main

      - name: Build all packages
        run: nix build .#all

      - name: Run checks
        run: nix flake check

      - name: Build Docker image
        run: nix build .#dockerImage
```

---

## Reproducibility Best Practices

### 1. Pin Your Inputs

The flake.lock file ensures reproducibility:

```bash
# Update all inputs
nix flake update

# Update specific input
nix flake update nixpkgs

# Commit flake.lock to git
git add flake.lock
git commit -m "Update flake dependencies"
```

### 2. Verify Reproducibility

```bash
# Build and check hash
nix build .#mycelix-api
nix path-info ./result --json | jq -r '.[].narHash'

# Rebuild and verify same hash
rm -rf result
nix build .#mycelix-api --rebuild
nix path-info ./result --json | jq -r '.[].narHash'
# Should be identical!
```

### 3. Archive Complete Build Environment

```bash
# Export closure (all dependencies)
nix-store --export $(nix-store -qR result) > mycelix-api.closure

# Import on another machine
nix-store --import < mycelix-api.closure

# Run imported binary
./result/bin/mycelix-api
```

---

## Troubleshooting

### Common Issues

#### "experimental-features" error

**Problem:** `error: experimental Nix feature 'flakes' is disabled`

**Solution:** Enable flakes in your Nix configuration:

```bash
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
```

#### Build fails with Cargo.lock issues

**Problem:** `error: hash mismatch in Cargo.lock`

**Solution:** Update the Cargo.lock hash:

```bash
# Remove old lock
rm Cargo.lock

# Regenerate
cargo generate-lockfile

# Rebuild with Nix
nix build
```

#### direnv not loading

**Problem:** Environment doesn't load automatically

**Solution:**

```bash
# Allow direnv
direnv allow

# Check status
direnv status

# Reload manually
direnv reload
```

### Getting Help

- **Nix Manual**: https://nixos.org/manual/nix/stable/
- **Nix Flakes**: https://nixos.wiki/wiki/Flakes
- **NixOS Options**: https://search.nixos.org/options
- **Discourse**: https://discourse.nixos.org/

---

## Advanced Usage

### Custom Package Overlays

Add Mycelix-DeSci packages to your system:

```nix
# In your configuration.nix
nixpkgs.overlays = [
  inputs.mycelix-desci.overlays.default
];

# Now available as:
# pkgs.mycelix-desci-core
# pkgs.mycelix-api
# pkgs.mycelix-cli
```

### Development with Different Rust Versions

```nix
# Override Rust version in flake.nix
rustToolchain = pkgs.rust-bin.stable."1.76.0".default;

# Or use nightly
rustToolchain = pkgs.rust-bin.nightly."2024-01-01".default;
```

### Cross-Compilation

```bash
# Build for different architectures
nix build .#mycelix-api --system aarch64-linux

# Or use cross-compilation
nix build .#mycelix-api.override {
  stdenv = pkgs.pkgsCross.aarch64-multiplatform.stdenv;
}
```

---

## Why This Matters for Science

### Research Reproducibility Crisis

Traditional software deployment has a **reproducibility problem**:
- "Works on my machine" syndrome
- Dependency drift over time
- Missing system dependencies
- Non-deterministic builds

### Nix Solves This

With Nix, **your research environment is**:

1. **Completely Specified** - Every dependency declared
2. **Bit-for-Bit Reproducible** - Same inputs → same outputs
3. **Archivable** - Export complete environment
4. **Verifiable** - Anyone can verify the build
5. **Long-term Stable** - Works in 10 years

### Real-World Impact

```bash
# Researcher in 2025 publishes results
nix build github:researcher/study-2025
./result/bin/run-analysis

# Reviewer in 2035 verifies results
nix build github:researcher/study-2025
./result/bin/run-analysis
# Exact same environment, exact same results!
```

**This is what science needs.** 🔬

---

## Comparison: Docker vs Nix

| Feature | Docker | Nix | Winner |
|---------|--------|-----|--------|
| Reproducibility | Partial | Complete | Nix |
| Build speed | Slow | Fast (cached) | Nix |
| Disk usage | High | Shared (deduplicated) | Nix |
| Security | Containers | Isolation | Nix |
| Complexity | Medium | Medium | Tie |
| Ecosystem | Huge | Growing | Docker |
| Long-term stability | Poor | Excellent | Nix |

**Best of both worlds:** Use Nix to build Docker images! 🎯

---

## Next Steps

1. ✅ **Try it out**: `nix run github:Luminous-Dynamics/mycelix-desci#cli`
2. ✅ **Enter dev shell**: `nix develop`
3. ✅ **Deploy on NixOS**: Add to `configuration.nix`
4. ✅ **Build Docker image**: `nix build .#dockerImage`
5. ✅ **Verify reproducibility**: Build twice, compare hashes

---

## Resources

- **Nix Website**: https://nixos.org
- **Nix Manual**: https://nixos.org/manual/nix/stable/
- **NixOS Manual**: https://nixos.org/manual/nixos/stable/
- **Nix Pills**: https://nixos.org/guides/nix-pills/
- **Zero to Nix**: https://zero-to-nix.com/

---

**Reproducible science starts here.** 🚀✨
