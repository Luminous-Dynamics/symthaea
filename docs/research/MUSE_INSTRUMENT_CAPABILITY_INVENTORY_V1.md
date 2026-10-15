# Muse Instrument Capability Inventory v1

Status: implementation slice for [Melothaea instrument-coverage issue #7416](https://github.com/Luminous-Dynamics/symthaea/issues/7416).

## Purpose

Muse has multiple kinds of instrument support that must not be conflated:

1. symbolic instrument identity in the Rust Instrument enum;
2. a declared procedural/fallback rendering path;
3. a declared mapping to a sample-library directory;
4. sample files on disk whose names the current indexer can interpret;
5. a real SoundFont file identified by its content hash;
6. preset names/bank/program values parsed from its RIFF SoundFont preset-header table;
7. the ability of a particular preset/articulation/range to render correctly;
8. evidence that the resulting sound is musically convincing.

The inventory CLI records source and filesystem evidence plus structurally parsed SoundFont preset headers. It does **not** certify full SoundFont validity, FluidSynth loadability, sample-zone presence, note-range coverage, or perceptual realism. Those remain separate gates.

## Running the inventory

From the repository root:

~~~sh
cargo run -p symthaea-muse --bin muse_instrument_inventory -- \
  --vcsl-root data/samples/vcsl \
  --vsco2-root data/samples/vsco2-ce
~~~

To include a specific SoundFont file in the receipt:

~~~sh
cargo run -p symthaea-muse --bin muse_instrument_inventory -- \
  --vcsl-root data/samples/vcsl \
  --vsco2-root data/samples/vsco2-ce \
  --soundfont /path/to/soundfont.sf2
~~~

The roots default to SYMTHAEA_VCSL_DIR and SYMTHAEA_VSCO2_DIR when set; otherwise the CLI uses the repository's conventional data/samples/vcsl and sibling data/samples/vsco2-ce paths. Explicit command-line roots override the defaults. JSON is emitted to stdout.

Missing optional asset roots are a valid observation and are reported as unavailable; they are not counted as sample coverage. An unreadable present root, or an explicitly requested SoundFont that cannot be read, causes a nonzero exit rather than a misleading partial receipt. In an automated build, set SYMTHAEA_SOURCE_REV to the exact inspected source commit; GitHub Actions' GITHUB_SHA is used as a fallback. Local builds without either variable emit a null source revision.

## Evidence fields

- instrument_count and instruments enumerate the current 42 source-declared instrument variants.
- procedural_renderer_path is derived from the instrument's source dispatch classifiers: karplus_strong, frequency_modulation, or additive. This is not an audio-quality score.
- gm_program_0_based is Muse's output mapping, not an asserted original source-patch identity. Some instruments use documented nearest-family approximations because General MIDI does not have exact names for all instruments.
- soundfont_bank0_program_matches is null when no SoundFont was supplied. With a SoundFont, it lists declared preset names matching bank 0 and that instrument's selected program; an empty list is evidence of no exact header match, not proof that no suitable sound is available in another bank or via a fallback.
- sustained_samples and short_articulation_samples report the mapped library, relative mapping directory, whether the directory exists, and how many WAV filenames are accepted by the current sample indexer.
- vcsl_root and vsco2_root record root presence, number of recursively discovered lowercase .wav files, and a deterministic path_size_manifest_sha256.
- soundfont, when requested, records the file name, byte size, and SHA-256 of the SoundFont bytes, plus parsed preset-header names, bank/program values, bank count, and duplicate bank/program slots.

path_size_manifest_sha256 hashes sorted relative WAV paths and file sizes. It is a compact layout/inventory digest, **not** a content digest; changing audio bytes while preserving each file path and size will not change it. indexed_sample_count establishes filename/indexer compatibility only; it does not prove WAV decoding, sample quality, pitch coverage, velocity-layer coverage, or successful audible rendering. The preset table is parsed from the RIFF/sfbk pdta/phdr chunk, including the terminal EOP-record check; invalid containers, chunk boundaries, empty preset names and out-of-range program values fail closed. The parsed table reports what the file declares. It does not prove that the SoundFont's sample chunks or instrument zones are complete, that libfluidsynth will load it, or that any program can render audible output. The SoundFont SHA-256 proves the byte identity of the file that was indexed.

The inventory intentionally contains no absolute root paths and no wall-clock timestamp. With the same source revision, roots' relative path/size manifests, optional SoundFont bytes, and environment-free values, its JSON is deterministic.

## Interpretation policy

Keep the following status dimensions separate when consuming the inventory:

| Question | Evidence required |
| --- | --- |
| Is the symbolic identity represented? | Source-declared Instrument variant |
| Which procedural renderer family does source dispatch select? | Source-derived Karplus-Strong / FM / additive classifier, separate from fidelity evidence |
| Is a sample mapping declared? | Explicit sample_subdir or staccato_subdir match |
| Does the mapped directory exist locally? | Filesystem observation in this receipt |
| Can the current indexer index files? | indexed_sample_count > 0; decode still unproven |
| Which SoundFont file was inspected? | Optional content SHA-256 |
| Does its preset-header table declare a bank/program/name? | Parsed RIFF SoundFont preset headers |
| Does bank 0 contain Muse's selected GM program? | Exact bank-0/program header match, or an explicit empty match list |
| Does the SoundFont actually render that preset? | Not answered by header parsing; requires FluidSynth load/render evidence |
| Is the audible rendering convincing? | Independent audio/render and listening evaluation |

Never infer an instrument mapping from register, VoiceRole, cultural label, or GM number alone. A mapped sustained articulation does not imply a mapped staccato articulation, and a known sample directory does not imply complete pitch or dynamic coverage. Timpani remains unmapped in the pitched-sample indexer because the repository documents its source filenames as drum-number-based rather than scientific-pitch-based.

## Sample-library provenance

- [VCSL README](https://github.com/sgossner/VCSL/blob/master/README.md) describes VCSL as CC0 and notes that most instruments have only a small number of round robins and velocity layers, with exceptions. CC0 permits reuse; the inventory must still record which files were actually inspected.
- [VSCO-2-CE repository](https://github.com/sgossner/VSCO-2-CE) includes a CC0 license. Its [Readme.txt](https://github.com/sgossner/VSCO-2-CE/blob/master/Readme.txt) also requests courtesy credit and asks users not to sell samples directly. Preserve that courtesy attribution in downstream asset notices.
- The [MIDI Association's General MIDI Level 1 overview](https://midi.org/general-midi-level-1) explains that GM assigns program names for compatibility, but the specification does not define each sound's acoustic characteristics. A GM name is not proof of a particular sound, source instrument, or preset quality.

## Next tranche

After this path-derived inventory is qualified:

1. Cross-check parsed header entries against FluidSynth's runtime preset listing and a deterministic render probe; keep the parser and renderer evidence separate.
2. Add note-range, dynamic-layer and articulation coverage from real, inspected files rather than assuming one sample equals full support.
3. Add a drum-number-aware index and dedicated percussion semantics instead of forcing percussion into pitched-note matching.
4. Add import → instrumentation sidecar → MIDI export → re-import fixtures for patch, bank, channel, part identity and instrument-change preservation.
5. Add rendered-output and listener evidence as independent layers; do not promote inventory counts into claims of musical quality.

This is a coverage-observability tool, not a claim that all 42 instruments have equally complete or perceptually equivalent implementations.
