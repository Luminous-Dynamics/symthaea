# OSWorld-Verified (NixOS)

OSWorld-Verified measures computer autonomy on a real OS. We standardize on
NixOS for deterministic environments.

## Data Layout

Place OSWorld assets and task files here:

```
benchmarks/external/osworld-verified/data/
```

## Fetch

```
./fetch.sh
```

This is a manual download/build step. The NixOS derivation should produce a
reproducible VM image.

## Run

```
./run.sh
```

Results should be written to:

```
benchmarks/external/results/osworld-verified.json
```

## Notes

- Use a NixOS VM/derivation for deterministic task execution.
- Record ActionIR traces and environment hash in the results.
- Set `SYMTHAEA_OSWORLD_RUNNER` to point to an external harness command, or
  `SYMTHAEA_OSWORLD_RESULT_JSON` to wrap precomputed results.
 - Provide `SYMTHAEA_OSWORLD_ENV_HASH` to embed the VM/environment hash in results.

## NixOS VM

A starter flake is provided at:

```
benchmarks/external/osworld-verified/nixos/flake.nix
```

Build the VM with:

```
nix build .#vm
```
