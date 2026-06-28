# SVMP Prototype Packet Format

`SVMP` means **Symthaea Visual Memory Packet**.

The current format is intentionally plain text so developers can inspect it in a terminal, diff it in Git, and reason about what the crate is actually storing.

## Version

Current magic:

```text
SVMP 0.1
```

The alpha.2 code still reads/writes `SVMP 0.1` for compatibility with alpha.1 packets. A future alpha should introduce `SVMP 0.2` with explicit checksums and metadata.

## Sections

```text
SVMP 0.1
dims <width> <height>
block_size <n>
keep_coeffs <k>
topology <count>
t <threshold> <beta0> <beta1>
...
blocks <count>
b <block_x> <block_y> <coeff_count> <index:value> ... hdc <word0> ... <word15>
...
```

## Meaning

- `dims`: original grayscale image dimensions
- `block_size`: square block side length used for DCT-style analysis
- `keep_coeffs`: target maximum coefficients per block
- `topology`: thresholded structural fingerprint samples
- `beta0`: approximate foreground connected components
- `beta1`: approximate foreground holes inferred from enclosed background components
- `blocks`: sparse spectral block packets
- `hdc`: deterministic 1024-bit binary hypervector signature per block

## What the format is not

This is not a compact production codec. Text packets may be larger than raw grayscale images. That is acceptable at this stage.

The point of the alpha is to test whether the packet can support:

- reconstruction experiments
- query-without-decode
- structural anomaly detection
- before/after infrastructure comparison
- Chronicle evidence artifacts

## Compatibility rule

Readers should be stricter than writers during early alpha. Bad packets should fail loudly rather than silently invent visual evidence.
