# NixOS RISC Zero Toolchain Setup

## Problem
RISC Zero's `rzup` tool installs pre-compiled binaries for generic Linux that won't run on NixOS without patching.

## Solution (Complete)

### Step 1: Install rzup
```bash
curl -L https://risczero.com/install | bash
```

### Step 2: Patch rzup binary
```bash
nix-shell -p patchelf --run "patchelf --set-interpreter \$(cat \$NIX_CC/nix-support/dynamic-linker) ~/.risc0/bin/rzup"
```

### Step 3: Install toolchain
```bash
~/.risc0/bin/rzup install
```

### Step 4: Patch all toolchain binaries
```bash
nix-shell -p patchelf --run '
cd ~/.risc0/toolchains/v1.88.0-rust-x86_64-unknown-linux-gnu/bin
for bin in *; do
  if file $bin | grep -q "ELF.*dynamically linked"; then
    echo Patching $bin...
    patchelf --set-interpreter $(cat $NIX_CC/nix-support/dynamic-linker) $bin 2>/dev/null || true
  fi
done
'
```

### Step 5: Set RPATH for shared libraries (REQUIRED)
```bash
nix-shell -p patchelf zlib --run '
cd ~/.risc0/toolchains/v1.88.0-rust-x86_64-unknown-linux-gnu/bin
for bin in *; do
  if file $bin | grep -q "ELF.*dynamically linked"; then
    echo Setting RPATH for $bin...
    patchelf --set-rpath "$(patchelf --print-rpath $bin 2>/dev/null):${zlib}/lib" $bin 2>/dev/null || true
  fi
done
'
```

## Alternative: Use NixOS Flake (Recommended)

Create `flake.nix` in project root:
```nix
{
  description = "VSV-STARK development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    risc0-nix.url = "github:risc0/risc0-nix";
  };

  outputs = { self, nixpkgs, risc0-nix }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          rustc
          cargo
          risc0-nix.packages.${system}.default
        ];
      };
    };
}
```

Then: `nix develop`

## Status - Manual Patching Findings
- ✅ rzup v0.5.0 installed and patched
- ✅ Toolchain installed (rust 1.88.0, cpp 2024.1.5, r0vm 3.0.3)
- ✅ Binaries patched with correct interpreter
- ✅ `rustc` works with `LD_LIBRARY_PATH` set (zlib + stdenv.cc.cc.lib)
- ⚠️ **Cascading Dependencies**: Guest builds also need `ld.lld`, `cc`, and other linker tools patched
- 🎯 **Conclusion**: Manual patching is feasible but complex; FHS devShell is recommended

## Complete Working Solution (Tested)

### LD_LIBRARY_PATH Approach (Quick Test)
```bash
export LD_LIBRARY_PATH="$(nix eval --raw nixpkgs#zlib)/lib:$(nix eval --raw nixpkgs#stdenv.cc.cc.lib)/lib:$LD_LIBRARY_PATH"
~/.risc0/toolchains/v1.88.0-rust-x86_64-unknown-linux-gnu/bin/rustc --version  # ✅ Works
```

However, guest builds need `ld.lld` patched too, creating cascading complexity.

## Recommended Solutions

### Short-Term (Day 2-3): Standard Rust Implementation
Implement CanaryCNN in standard Rust (`src/lib.rs`) without zkVM, port to guest on Day 3.

### Medium-Term (Day 3-4): FHS DevShell (Recommended)
Add to `0TML/vsv-stark/flake.nix`:

```nix
devShells.default = pkgs.buildFHSUserEnv {
  name = "vsv-stark-env";
  targetPkgs = pkgs: with pkgs; [
    rustup
    zlib
    stdenv.cc.cc.lib
  ];
  runScript = "bash";
};
```

Then: `nix develop && ~/.risc0/bin/rzup install && cargo build`

### Long-Term (Q1 2026): risc0-nix
Adopt official `risc0-nix` flake when mature for fully declarative builds.
