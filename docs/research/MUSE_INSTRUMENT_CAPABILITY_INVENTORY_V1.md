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

The inventory CLI records source and filesystem evidence, fully consumes the sample data of WAV files accepted by its current filename indexer, and structurally parses SoundFont preset headers. When explicitly requested, it can additionally invoke a selected FluidSynth executable to render a small generated MIDI probe for each distinct bank-0 GM program declared in the 42 instrument mappings. It does **not** certify full SoundFont validity, complete SoundFont sample-zone coverage, complete note-range/articulation coverage, or perceptual realism. Runtime results are local, executable/SoundFont-specific observations—not release qualification.

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

To opt into runtime render probes, pass FluidSynth's executable explicitly:

~~~sh
cargo run -p symthaea-muse --bin muse_instrument_inventory -- \
  --vcsl-root data/samples/vcsl \
  --vsco2-root data/samples/vsco2-ce \
  --soundfont /path/to/soundfont.sf2 \
  --fluidsynth /path/to/fluidsynth
~~~

The probe flag is deliberately opt-in. It requires `--soundfont`; the tool resolves the executable, records its file SHA-256 and first version line when available, and invokes offline MIDI-to-WAV rendering without a live MIDI or audio device. Each distinct GM program referenced by the inventory is tested once, but a render is attempted only when the SoundFont header table declares the exact bank-0/program slot. The probe uses a fixed 3×3 MIDI grid: notes 48, 60 and 72, each at velocities 48, 96 and 120; sample rate 22050 Hz; gain 0.8; reverb and chorus disabled via FluidSynth's documented `-R0` and `-C0` switches; and one half-second note followed by a half-second tail. Each program/note/velocity cell gets its own MIDI/WAV artifact and status. It records the executable hash/version, command template, output WAV hash, and decoded signal metrics. Running probes takes longer than a filesystem-only inventory and results may differ between FluidSynth builds or SoundFonts.

To bind the receipt to the bytes in each indexed WAV file, add `--hash-indexed-samples` to any invocation. This opt-in performs a second streaming read of every WAV accepted by the sample filename indexer and emits a per-mapped-directory content-manifest SHA-256. It can add substantial I/O for large libraries; the default inventory remains lightweight and leaves these hashes null. If a file cannot be read consistently during hashing, the command fails instead of emitting a partial content fingerprint.

The roots default to SYMTHAEA_VCSL_DIR and SYMTHAEA_VSCO2_DIR when set; otherwise the CLI uses the repository's conventional data/samples/vcsl and sibling data/samples/vsco2-ce paths. Explicit command-line roots override the defaults. JSON is emitted to stdout.

Missing optional asset roots are a valid observation and are reported as unavailable; they are not counted as sample coverage. An unreadable present root, or an explicitly requested SoundFont that cannot be read, causes a nonzero exit rather than a misleading partial receipt. In an automated build, set SYMTHAEA_SOURCE_REV to the exact inspected source commit; GitHub Actions' GITHUB_SHA is used as a fallback. Local builds without either variable emit a null source revision.

## Evidence fields

- instrument_count and instruments enumerate the current 42 source-declared instrument variants.
- procedural_renderer_path is derived from the instrument's source dispatch classifiers: karplus_strong, frequency_modulation, or additive. This is not an audio-quality score.
- gm_program_0_based is Muse's output mapping, not an asserted original source-patch identity. Some instruments use documented nearest-family approximations because General MIDI does not have exact names for all instruments.
- soundfont_bank0_program_matches is null when no SoundFont was supplied. With a SoundFont, it lists declared preset names matching bank 0 and that instrument's selected program; an empty list is evidence of no exact header match, not proof that no suitable sound is available in another bank or via a fallback.
- sustained_samples and short_articulation_samples report the mapped library, relative mapping directory, whether the directory exists, and how many WAV filenames are accepted by the current sample indexer. They also report filename-derived minimum/maximum MIDI note and distinct pitch/dynamic-value counts, plus an audio-decode audit: decode status, fully decoded file count, failed full-decode count, near-silent file count, and diversity of sample rates/channel counts among successfully decoded files. For failed indexed WAVs, the inventory retains up to 20 path-sorted filename/reason diagnostics per mapped directory; each reason is capped at 200 characters, filenames are emitted without absolute root paths, and omitted failures are counted explicitly. These diagnostics help locate malformed/truncated assets without allowing one damaged library to create an unbounded report. With `--hash-indexed-samples`, each mapped directory with indexable files additionally reports `indexed_sample_content_manifest_sha256`: a deterministic digest over sorted filename, size, and per-file SHA-256 records for the exact indexed WAV bytes.
- vcsl_root and vsco2_root record root presence, number of recursively discovered lowercase .wav files, and a deterministic path_size_manifest_sha256.
- soundfont, when requested, records the file name, byte size, and SHA-256 of the SoundFont bytes, plus parsed preset-header names, bank/program values, bank count, and duplicate bank/program slots.
- fluidsynth_probe is null unless `--fluidsynth` is explicitly supplied. When present, it binds the render results to the resolved executable's file hash, reported version (if available), and exact SoundFont SHA-256. Each program/note/velocity probe records matching bank-0 header names, note, velocity, command exit code, decode status, output WAV SHA-256, peak/RMS, sample rate, channel count, and decoded frame count. The outer receipt records distinct program count, total probe count, non-silent probe count, the fixed note/velocity grid, sample rate, effect settings, executable hash/version, and a stable argument template. Programs with no exact bank-0 header match are skipped instead of silently probing a renderer fallback.

path_size_manifest_sha256 hashes sorted relative WAV paths and file sizes. It is a compact layout/inventory digest, **not** a content digest; changing audio bytes while preserving each file path and size will not change it. The optional indexed_sample_content_manifest_sha256 includes file contents and therefore changes when indexed audio bytes change even if a WAV's path and size remain the same. It covers files accepted by the current indexer, not every WAV in the root directory, and is not a perceptual or sample-validity score. indexed_sample_count establishes filename/indexer compatibility only. The WAV audit opens each indexed file and consumes its complete hound sample iterator, rejecting empty/incomplete frames, decoder errors, invalid sample-rate/channel/bit-depth metadata and non-finite float samples. The JSON report includes up to 20 deterministic filename-only failure diagnostics per mapped directory (with bounded reason strings) and separately counts any failures omitted by that limit. A successful decode only proves the file can be fully decoded by this audit; it does not prove the samples are musically useful, that their named pitch is correct, or that the renderer can play them in context. The silent-file threshold is peak absolute normalized sample value <= 1e-5, a mechanical silence check rather than a perceptual test. The observed min/max MIDI values and distinct pitch/dynamic values describe filename metadata, not calibrated acoustic velocity layers or rendered range. Two files can name the same pitch at different dynamics; filename coverage must not be treated as one-sample-per-note full support. The preset table is parsed from the RIFF/sfbk pdta/phdr chunk, including the terminal EOP-record check; invalid containers, chunk boundaries, empty preset names and out-of-range program values fail closed. The parsed table reports what the file declares. It does not prove that the SoundFont's sample chunks or instrument zones are complete, that libfluidsynth will load it, or that any program can render audible output. The SoundFont SHA-256 proves the byte identity of the file that was indexed. Runtime probes use a minimal generated Standard MIDI File that explicitly sends MIDI Bank Select MSB and LSB as zero, then one program-change event and one note. The current runtime matrix spans MIDI notes 48/60/72 and velocities 48/96/120; this is a smoke grid, not a complete instrument range or calibrated velocity-layer test. A `rendered_non_silent` status proves only that the specified executable exited successfully and produced a decodable WAV with peak amplitude above the mechanical threshold. The output WAV SHA-256 preserves the exact bytes of that render for later comparison; it does not establish that independent runs must be byte-identical or that the sound is musically useful. It does not assess timbre, prove complete preset zones, or establish quality. Duplicate bank/program header records remain name-ambiguous.

The inventory intentionally contains no absolute sample-root paths and no wall-clock timestamp. Without `--fluidsynth`, its JSON is deterministic for the same source revision, identical relative sample paths and WAV contents, the same optional SoundFont bytes and filename, and the same environment-derived inputs. Runtime-probe receipts include audio metrics, executable identity and an output WAV content hash; compare them only when source revision, SoundFont hash, executable hash/version and probe settings are held constant.

## Interpretation policy

Keep the following status dimensions separate when consuming the inventory:

| Question | Evidence required |
| --- | --- |
| Is the symbolic identity represented? | Source-declared Instrument variant |
| Which procedural renderer family does source dispatch select? | Source-derived Karplus-Strong / FM / additive classifier, separate from fidelity evidence |
| Is a sample mapping declared? | Explicit sample_subdir or staccato_subdir match |
| Does the mapped directory exist locally? | Filesystem observation in this receipt |
| Can the current indexer index files? | indexed_sample_count > 0; audio still unproven |
| Do indexed WAVs fully decode, or are some silent? | audio_decode_status, decoded/failed counts, and silent_sample_file_count |
| Which indexed WAV files failed decoding? | reported_sample_decode_failures (bounded filename/reason entries) and omitted_sample_decode_failure_count |
| Did indexed WAV bytes change while paths and file sizes stayed the same? | Compare indexed_sample_content_manifest_sha256 when `--hash-indexed-samples` was enabled on both inventories |
| What WAV formats occur among decoded files? | distinct sample-rate and channel-count fields |
| What pitch/dynamic filename coverage is visible? | min/max MIDI, distinct pitch count, and distinct normalized dynamic-value count |
| Which SoundFont file was inspected? | Optional content SHA-256 |
| Does its preset-header table declare a bank/program/name? | Parsed RIFF SoundFont preset headers |
| Does bank 0 contain Muse's selected GM program? | Exact bank-0/program header match, or an explicit empty match list |
| Does the SoundFont actually render that preset? | An opt-in `fluidsynth_probe` result with exact header match, successful process exit, fully decoded output WAV and non-silent samples |
| Is the render repeatable or musically faithful? | Not established by this probe; compare executable/SoundFont identity and add signal-based/listening evaluation |
| Is the audible rendering convincing? | Independent audio/render and listening evaluation |

Never infer an instrument mapping from register, VoiceRole, cultural label, or GM number alone. A mapped sustained articulation does not imply a mapped staccato articulation, and a known sample directory does not imply complete pitch or dynamic coverage. Timpani remains unmapped in the pitched-sample indexer because the repository documents its source filenames as drum-number-based rather than scientific-pitch-based.

## Sample-library provenance

- [VCSL README](https://github.com/sgossner/VCSL/blob/master/README.md) describes VCSL as CC0 and notes that most instruments have only a small number of round robins and velocity layers, with exceptions. CC0 permits reuse; the inventory must still record which files were actually inspected.
- [VSCO-2-CE repository](https://github.com/sgossner/VSCO-2-CE) includes a CC0 license. Its [Readme.txt](https://github.com/sgossner/VSCO-2-CE/blob/master/Readme.txt) also requests courtesy credit and asks users not to sell samples directly. Preserve that courtesy attribution in downstream asset notices.
- The [MIDI Association's General MIDI Level 1 overview](https://midi.org/general-midi-level-1) explains that GM assigns program names for compatibility, but the specification does not define each sound's acoustic characteristics. A GM name is not proof of a particular sound, source instrument, or preset quality.

## Next tranche

After this path-derived inventory is qualified:

1. Capture runtime preset enumeration via FluidSynth's preset API or a carefully parsed diagnostic, so bank/program selection is tied to the engine's runtime view rather than just the static SoundFont header.
2. Compare repeated renders for byte equality and tolerant signal similarity under pinned executable/SoundFont identity; retain hashes as receipts rather than assuming hashes alone prove audible determinism.
3. Add note-range, dynamic-layer and articulation coverage from real, decoded files rather than assuming one sample equals full support.
4. Add a drum-number-aware index and dedicated percussion semantics instead of forcing percussion into pitched-note matching.
5. Add import → instrumentation sidecar → MIDI export → re-import fixtures for patch, bank, channel, part identity and instrument-change preservation, then add independent rendered-output and listening evidence.

This is a coverage-observability tool, not a claim that all 42 instruments have equally complete or perceptually equivalent implementations.
