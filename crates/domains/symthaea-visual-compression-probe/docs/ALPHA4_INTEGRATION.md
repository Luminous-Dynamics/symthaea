# Alpha.4 Integration Notes

Alpha.4 turns the probe from a CLI-only experiment into a small integration surface for Symthaea perception and Symtropy Field Deck experiments.

## New public API

- `EncodingParams`
- `VisualMemoryPacket::encode_with_params()`
- `CognitiveScanSummary`
- `visual_summary()`
- `topology_complexity()`
- `RankedPacket`
- `rank_packets()`
- `packet_manifest_header()`
- `packet_manifest_row()`

## Intended use

Use this crate to create visual-memory packets from grayscale diagnostic scans, not to replace ordinary image codecs. The packet is useful when downstream systems need to compare, retrieve, classify, or remember visual structure without immediately reconstructing pixels.

## Suggested Symthaea hook

```rust
use symthaea_visual_compression_probe::{
    EncodingParams, GrayImage, VisualMemoryPacket, visual_summary,
};

let image = GrayImage::read_pgm("scan.pgm")?;
let params = EncodingParams::new(8, 10, 16)?;
let packet = VisualMemoryPacket::encode_with_params(&image, params)?;
let summary = visual_summary(&image, params)?;

println!("{}", summary.to_pretty_text());
packet.write_text("scan.svmp")?;
```

## Suggested Symtropy Field Deck hook

1. Convert a diagnostic Field Deck view into a grayscale evidence plane.
2. Encode it into `.svmp`.
3. Store the packet hash and summary in the local evidence bundle.
4. Use `query` or `rank_packets()` to find prior similar failures.
5. Keep the original image only when human-review fidelity is required.

## CLI additions

```bash
svcp summary fixtures/tiny_pump_scan.pgm --json
svcp batch-encode fixtures /tmp/svcp-corpus --manifest /tmp/svcp-manifest.tsv
svcp matrix /tmp/svcp-corpus /tmp/svcp-similarity.csv
```

## Claim boundary

Alpha.4 adds useful integration surfaces. It still does not prove superior compression. Treat all metrics as experiment probes until validated on a larger corpus with baselines.
