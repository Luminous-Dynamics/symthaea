# OSWorld NixOS VM

This flake builds a deterministic NixOS VM image for OSWorld-Verified runs.

## Build

```
cd benchmarks/external/osworld-verified/nixos
nix build .#vm
```

The VM is produced under `result/`.

## Run

```
./result/bin/run-osworld-vm
```

## Notes

- Default user: `symthaea` (set password via `initialHashedPassword` in configuration.nix)
- XFCE desktop is enabled for GUI tasks.
- Adjust `configuration.nix` to match OSWorld task requirements.
