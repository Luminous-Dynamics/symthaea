# Nix

NixOS packaging and development environment configuration.

## Files

- `flake.nix` (in root) - Main flake configuration
- `shell.nix` - Legacy shell configuration
- Additional Nix expressions

## Usage

```bash
# Enter development shell
nix develop

# Build package
nix build

# Run directly
nix run
```

## Features

The flake provides:

- Rust toolchain with required components
- Python with scientific packages
- System dependencies (OpenSSL, etc.)
- Development tools (clippy, rustfmt)

## NixOS Module

For system-wide installation on NixOS, see the module in this directory.
