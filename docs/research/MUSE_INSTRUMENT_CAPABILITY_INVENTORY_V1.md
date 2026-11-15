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

To bind the receipt to the bytes in each indexed WAV file, add `--hash-indexed-samples` to any invocation. This opt-in performs a second streaming read of every WAV accepted by the sample filename indexer and emits a per-mapped-directory content-manifest SHA-256. It can add substantial I/O for large libraries; the default inventory remains lightweight and leaves these hashes null. If an indexed file cannot be opened or fully read during hashing, or its size changes while being hashed, the command fails instead of emitting a partial content fingerprint. The top-level `indexed_sample_content_hashing_enabled` flag makes it explicit whether this optional pass was requested, even when no mapped directory has indexable files.

The roots default to SYMTHAEA_VCSL_DIR and SYMTHAEA_VSCO2_DIR when set; otherwise the CLI uses the repository's conventional data/samples/vcsl and sibling data/samples/vsco2-ce paths. Explicit command-line roots override the defaults. JSON is emitted to stdout.

Missing optional asset roots are a valid observation and are reported as unavailable; they are not counted as sample coverage. An unreadable present root, or an explicitly requested SoundFont that cannot be read, causes a nonzero exit rather than a misleading partial receipt. In an automated build, set SYMTHAEA_SOURCE_REV to the exact inspected source commit; GitHub Actions' GITHUB_SHA is used as a fallback. Local builds without either variable emit a null source revision.

## Evidence fields

- instrument_count and instruments enumerate the current 42 source-declared instrument variants. The top-level `indexed_sample_content_hashing_enabled` boolean records whether the optional sample-byte hashing pass was requested; a null per-directory digest with this flag true means that mapping had no indexable WAV files to fingerprint.
- procedural_renderer_path is derived from the instrument's source dispatch classifiers: karplus_strong, frequency_modulation, or additive. This is not an audio-quality score.
- gm_program_0_based is Muse's output mapping, not an asserted original source-patch identity. Some instruments use documented nearest-family approximations because General MIDI does not have exact names for all instruments.
- soundfont_bank0_program_matches is null when no SoundFont was supplied. With a SoundFont, it lists declared preset names matching bank 0 and that instrument's selected program; an empty list is evidence of no exact header match, not proof that no suitable sound is available in another bank or via a fallback.
- sustained_samples and short_articulation_samples report the mapped library, relative mapping directory, whether the directory exists, and how many WAV filenames are accepted by the current sample indexer. `additional_sample_styles` catalogs verified alternate sample folders for selected instruments, each with a `style_name`, a descriptive `style_kind`, and a full `SampleDirectoryEvidenceV1` record. `VcslLibrary` now also loads the non-release variants into named style banks and exposes `has_sample_style` and `can_play_style` checks. `synth::render_arrangement_with_sample_styles` accepts named overrides keyed by voice index and returns audio plus deterministic per-note selection/fallback receipts; the original `render_arrangement` signature and default policy remain unchanged. Release-only folders remain catalogued but are not exposed as ordinary note-on banks. Missing local folders remain explicit evidence, not proof that the upstream style does not exist. They also report a per-pitch evidence topology in `pitch_coverage`: for each observed MIDI label, indexed file count, fully decoded/failed/silent file counts, and per-dynamic-strength bucket counts with the same audio outcomes. Both pitch and dynamic-bucket rows include a derived `audio_evidence_status` chosen from an explicit conservative taxonomy: `inconsistent_audio_evidence_counts`, `no_indexed_samples`, `all_indexed_samples_failed_decode`, `decoded_silent_only`, `decoded_silent_and_other_samples_failed`, `mixed_silent_and_non_silent_decoded_samples`, `mixed_silent_non_silent_and_decode_failures`, `non_silent_decoded_samples_and_decode_failures`, or `non_silent_decoded_sample_present`. Before classification, the helper verifies that indexed count equals decoded plus failed count and that silent count cannot exceed decoded count; impossible combinations fail closed as `inconsistent_audio_evidence_counts`. These labels depend only on complete-decode counts and the mechanical silence threshold; they are not perceptual or musical-quality classifications. This lets downstream tooling see whether a pitch is well represented, represented only by corrupt files, or only by silent data rather than compressing the entire mapped directory into a single coverage count. Dynamic strengths are normalized filename-derived values—not calibrated MIDI velocity layers—and counts include every file accepted by the filename indexer even if decode fails. The inventory also reports filename-derived minimum/maximum MIDI note and distinct pitch/dynamic-value counts, plus `filename_pitch_gaps_midi_within_observed_range`: MIDI pitch labels absent strictly between the observed minimum and maximum labels. This makes internal filename-coverage holes visible rather than letting min/max imply continuity, but it is not a claim about the renderer's achievable range or which pitches the instrument ought to support. The audio-decode audit separately reports decode status, fully decoded file count, failed full-decode count, near-silent file count, and diversity of sample rates/channel counts among successfully decoded files. Pitch-level counts are a breakdown of this same audit; they are not an additional independent validation pass. For failed indexed WAVs, the inventory retains up to 20 path-sorted filename/reason diagnostics per mapped directory; each record also contains the filename-parsed MIDI note and normalized dynamic strength so it can be joined to the pitch/dynamic coverage topology. These are filename metadata, not acoustic measurements or calibrated velocity layers. Each reason is capped at 200 characters, filenames are emitted without absolute root paths, and omitted failures are counted explicitly. These diagnostics help locate malformed/truncated assets without allowing one damaged library to create an unbounded report. With `--hash-indexed-samples`, each mapped directory with indexable files additionally reports `indexed_sample_content_manifest_sha256`: a deterministic digest over sorted filename, size, and per-file SHA-256 records for the exact indexed WAV bytes.
- vcsl_root and vsco2_root record root presence, number of recursively discovered lowercase .wav files, and a deterministic path_size_manifest_sha256.
- soundfont, when requested, records the file name, byte size, and SHA-256 of the SoundFont bytes, plus parsed preset-header names, bank/program values, bank count, and duplicate bank/program slots.
- fluidsynth_probe is null unless `--fluidsynth` is explicitly supplied. When present, it binds the render results to the resolved executable's file hash, reported version (if available), and exact SoundFont SHA-256. Each program/note/velocity probe records matching bank-0 header names, note, velocity, command exit code, decode status, output WAV SHA-256, peak/RMS, sample rate, channel count, and decoded frame count. The outer receipt records distinct program count, total probe count, non-silent probe count, the fixed note/velocity grid, sample rate, effect settings, executable hash/version, and a stable argument template. Programs with no exact bank-0 header match are skipped instead of silently probing a renderer fallback.

path_size_manifest_sha256 hashes sorted relative WAV paths and file sizes. It is a compact layout/inventory digest, **not** a content digest; changing audio bytes while preserving each file path and size will not change it. The optional indexed_sample_content_manifest_sha256 includes file contents and therefore changes when indexed audio bytes change even if a WAV's path and size remain the same. It covers files accepted by the current indexer, not every WAV in the root directory, and is not a perceptual or sample-validity score. indexed_sample_count establishes filename/indexer compatibility only. The WAV audit opens each indexed file and consumes its complete hound sample iterator, rejecting empty/incomplete frames, decoder errors, invalid sample-rate/channel/bit-depth metadata and non-finite float samples. The JSON report includes up to 20 deterministic filename-only failure diagnostics per mapped directory (with bounded reason strings) and separately counts any failures omitted by that limit. A successful decode only proves the file can be fully decoded by this audit; it does not prove the samples are musically useful, that their named pitch is correct, or that the renderer can play them in context. The silent-file threshold is peak absolute normalized sample value <= 1e-5, a mechanical silence check rather than a perceptual test. The per-pitch topology relates each indexed filename pitch to the decode and silence outcomes of its files and the normalized dynamic values parsed from those filenames. Its `audio_evidence_status` is a deterministic summary of those counts, not an additional validation pass. The classifier checks count invariants before assigning a status and has explicit regression cases for impossible combinations. It is intended as a machine-readable triage aid, not a numeric quality score; keep the component counts available so consumers can audit the label. The observed min/max MIDI values, distinct pitch/dynamic values and filename pitch gaps describe filename metadata, not calibrated acoustic velocity layers or rendered range. Filename gaps are limited to missing integer MIDI labels between the lowest and highest indexed pitch; they do not account for pitch-shifting reach, layers, articulations, or musical expectations. Two files can name the same pitch at different dynamics; filename coverage must not be treated as one-sample-per-note full support. The preset table is parsed from the RIFF/sfbk pdta/phdr chunk, including the terminal EOP-record check; invalid containers, chunk boundaries, empty preset names and out-of-range program values fail closed. The parsed table reports what the file declares. It does not prove that the SoundFont's sample chunks or instrument zones are complete, that libfluidsynth will load it, or that any program can render audible output. The SoundFont SHA-256 proves the byte identity of the file that was indexed. Runtime probes use a minimal generated Standard MIDI File that explicitly sends MIDI Bank Select MSB and LSB as zero, then one program-change event and one note. The current runtime matrix spans MIDI notes 48/60/72 and velocities 48/96/120; this is a smoke grid, not a complete instrument range or calibrated velocity-layer test. A `rendered_non_silent` status proves only that the specified executable exited successfully and produced a decodable WAV with peak amplitude above the mechanical threshold. The output WAV SHA-256 preserves the exact bytes of that render for later comparison; it does not establish that independent runs must be byte-identical or that the sound is musically useful. It does not assess timbre, prove complete preset zones, or establish quality. Duplicate bank/program header records remain name-ambiguous.

## Alternate style inventory

The `additional_sample_styles` field catalogs **42 source-verified alternate folders** separately from each instrument's default sustain and short-articulation paths. `VcslLibrary` loads 39 non-release variants into named style banks; `has_sample_style` reports whether a local pitch-triggered bank is loaded, and `can_play_style` checks whether the requested pitch/velocity can select and decode a sample. Three release-articulation variants stay catalogue-only because piano note-release data must be consumed at note-off, not triggered as an ordinary note. This bank-level API is not yet wired to per-voice requests in `synth::render_arrangement`, so it does not silently alter existing composition output. A present upstream path is not proof that a machine has downloaded it or that its recordings are usable.

| Instrument | Style label | Style kind | Relative directory |
| --- | --- | --- | --- |
| Piano / PianoPP | Kawai grand piano | alternate instrument model | `Chordophones/Zithers/Grand Piano, Kawai/Sustains` |
| Piano / PianoPP | Knight upright piano | alternate instrument model | `Chordophones/Zithers/Upright Piano, Knight/Sustains` |
| Piano / PianoPP | Steinway no-sustain samples | alternate articulation | `Chordophones/Zithers/Grand Piano, Steinway B/NoSus` |
| Piano / PianoPP | Steinway release samples | release articulation | `Chordophones/Zithers/Grand Piano, Steinway B/Rel` |
| Piano / PianoPP | Kawai release samples | release articulation | `Chordophones/Zithers/Grand Piano, Kawai/Releases` |
| Piano / PianoPP | Knight upright piano release | release articulation | `Chordophones/Zithers/Upright Piano, Knight/Releases` |
| Harp | Folk harp | alternate instrument model | `Chordophones/Composite Chordophones/Folk Harp` |
| Harpsichord | English lute disposition | alternate stringing | `Chordophones/Zithers/Harpsichord, English/Sustains/Lute` |
| Harpsichord | French harpsichord | alternate instrument model | `Chordophones/Zithers/Harpsichord, French/Sustains` |
| Harpsichord | Italian stop 1 | alternate stop | `Chordophones/Zithers/Harpsichord, Italian/Sustains/stop1` |
| Harpsichord | Flemish low register | register variant | `Chordophones/Zithers/Harpsichord, Flemish/Sustains/Low` |
| Harpsichord | Flemish high register | register variant | `Chordophones/Zithers/Harpsichord, Flemish/Sustains/High` |
| Harpsichord | Unclassified harpsichord | alternate instrument model | `Chordophones/Zithers/Harpsichord, Unk/Sustains` |
| Xylophone | Medium mallets | mallet hardness | `Idiophones/Struck Idiophones/Xylophone/Medium Mallets` |
| Xylophone | Soft mallets | mallet hardness | `Idiophones/Struck Idiophones/Xylophone/Soft Mallets` |
| Violin | Solo violin pizzicato | pizzicato | `Strings/Solo Violin/Pizz` |
| Violin | Solo violin tremolo | tremolo | `Strings/Solo Violin/Trem` |
| Viola | Viola section pizzicato | pizzicato | `Strings/Viola Section/pizz` |
| Viola | Viola section tremolo | tremolo | `Strings/Viola Section/trem` |
| Cello | Cello section pizzicato | pizzicato | `Strings/Cello Section/pizzT` |
| Cello | Cello section tremolo | tremolo | `Strings/Cello Section/trem` |
| Flute | Flute expressive vibrato | expressive vibrato | `Woodwinds/Flute/expvib` |
| Flute | Flute non-vibrato sustain | sustain without vibrato | `Woodwinds/Flute/susNV` |
| Oboe | Oboe vibrato | vibrato | `Woodwinds/Oboe/Vib` |
| Bassoon | Bassoon vibrato | vibrato | `Woodwinds/Bassoon/vib` |
| Trumpet | Trumpet harmon mute | muted | `Brass/Trumpet/harmonM-sus` |
| Trumpet | Trumpet straight mute | muted | `Brass/Trumpet/straightM-sus` |
| Trumpet | Trumpet non-vibrato sustain | sustain without vibrato | `Brass/Trumpet/sus` |
| FrenchHorn | French horn muted | muted | `Brass/F Horn/mute` |
| Saxophone | Tenor saxophone non-vibrato | sustain without vibrato | `Aerophones/Reed Aerophones/Tenor Saxophone/Non-Vibrato` |
| Saxophone | Tenor saxophone staccato | staccato | `Aerophones/Reed Aerophones/Tenor Saxophone/Staccato` |
| Organ | Pipe organ quiet | registration variant | `Aerophones/Edge-blown Aerophones/Pipe Organ/Quiet` |
| Organ | Pipe organ loud pedal | pedal registration | `Aerophones/Edge-blown Aerophones/Pipe Organ/Loud Pedal` |
| Organ | Pipe organ quiet pedal | pedal registration | `Aerophones/Edge-blown Aerophones/Pipe Organ/Quiet Pedal` |
| Vibraphone | Vibraphone soft mallets | mallet hardness | `Idiophones/Struck Idiophones/Vibraphone/Soft Mallets` |
| Vibraphone | Bowed vibraphone | bowed | `Idiophones/Struck Idiophones/Vibraphone/Bowed` |
| Marimba | Balafon hard mallet | mallet hardness | `Idiophones/Struck Idiophones/Balafon/Hard Mallet` |
| Marimba | Balafon soft mallet | mallet hardness | `Idiophones/Struck Idiophones/Balafon/Soft Mallet` |
| UprightBass | Vibrato sustain | sustain with vibrato | `Strings/Solo Contrabass/SusVib` |
| UprightBass | Pizzicato | pizzicato | `Strings/Solo Contrabass/Pizz` |
| UprightBass | Tremolo | tremolo | `Strings/Solo Contrabass/Trem` |
| Trombone | Tenor trombone vibrato | vibrato | `Brass/Tenor Trombone/vib` |

The folder paths were checked against [VCSL tree `c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e`](https://github.com/sgossner/VCSL/tree/c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e) and [VSCO-2-CE tree `440300901dfe9275fd84e0b7763af1f8443ae62e`](https://github.com/sgossner/VSCO-2-CE/tree/440300901dfe9275fd84e0b7763af1f8443ae62e). The deterministic fixture now checks representative filenames from all 42 folders, verifies each variant appears in the serialized inventory with one decoded WAV, and exercises all 39 note-on style banks through named style lookup/pitch selection. It confirms the three release-only styles remain excluded from ordinary note-on selection. It tests path/indexer compatibility; it does **not** qualify actual downloaded recordings, prove full musical range, or assess timbre.

The new entry point accepts a style override for each voice index without mutating the serialized `Voice` schema. Successful note-on selection reports `style_selected` / `named_sample_style`. Unsupported or unavailable requests still use the established renderer fallback and report a reason such as `unknown_style_for_instrument_fell_back`, `release_style_requires_note_off_path`, `style_bank_not_loaded_fell_back`, or `style_sample_unavailable_fell_back`, together with the actual rendered path. Out-of-range voice requests and empty voices are represented explicitly instead of disappearing. The legacy renderer delegates with an empty request map, preserving its public signature and default policy.

A key next step remains real performance validation. Initial targets: (1) 100% receipt coverage for every requested note and every invalid/empty voice request; (2) deterministic receipt ordering and source selection for a fixed asset manifest, arrangement, and seed; (3) bit-identical legacy audio before/after this change on an identical build; (4) no silent substitution—every fallback has a reason and actual path; (5) a 39-style × representative-pitch × velocity-layer coverage matrix, reporting unsupported cells rather than counting folder presence as playability; and (6) separate listening evaluation for articulation identity and musical quality. Suggested performance target, not a current result: under 2% CPU overhead for the legacy no-style path and receipt allocation proportional to requested note count. Pizzicato/tremolo, vibrato/non-vibrato, muted/unmuted, mallet hardness, and organ registrations remain distinct controls, not interchangeable fallbacks.

### Perceptual evaluation and measurement references

- Use a randomized, level-matched MUSHRA-style comparison for broad style/naturalness questions, following the test-design principles of [ITU-R BS.1534-3](https://www.itu.int/rec/R-REC-BS.1534/en). Use [ITU-R BS.1116-3](https://www.itu.int/rec/R-REC-BS.1116/) when the goal is to detect small impairments between otherwise similar renders. These are complementary evaluation designs, not a single universal score.
- Record loudness and true-peak separately using the current [ITU-R BS.1770-5](https://www.itu.int/rec/R-REC-BS.1770) measurement algorithm. Muse's existing RMS normalization and sample-peak limiter are not substitutes for standards-based integrated loudness and true-peak measurement.
- Freeze and hash the corpus, source sample manifest, renderer revision, arrangement, sample rate, and style request. Render a balanced grid over style × register (low/mid/high) × velocity (soft/medium/loud), then add articulation-appropriate contexts (isolated, repeated, and legato where meaningful). Keep coverage and perceptual quality as separate gates: a decoded sample can pass coverage while failing to communicate its intended articulation.
- Ask blinded listeners to score style identity, naturalness, musical fit, artifacts, and preference separately. Randomize identifiers and presentation order, level-match clips without revealing fallback status, preserve per-cell receipts for analysis, and report sample size plus uncertainty rather than a single unqualified mean. The first engineering target remains 100% traceable receipts for explicit requests; proposed real-time target remains <2% overhead on the legacy no-style path, not a measured result.

## Completing verified sample styles without inventing coverage

A source-tree audit against the current public library revisions expanded the mapper for five previously unmapped Muse instruments:

| Muse instrument | Sample library | Sustained/primary path | Short articulation |
| --- | --- | --- | --- |
| Harpsichord | VCSL | `Chordophones/Zithers/Harpsichord, English/Sustains/Normal` | Not mapped |
| Xylophone | VCSL | `Idiophones/Struck Idiophones/Xylophone/Hard Mallets` | Not mapped |
| UprightBass | VSCO-2-CE | `Strings/Solo Contrabass/SusNV` | `Strings/Solo Contrabass/Spic` |
| Tuba | VSCO-2-CE | `Brass/Tuba/sus` | `Brass/Tuba/stac` |
| Trombone | VSCO-2-CE | `Brass/Tenor Trombone/sus` | `Brass/Tenor Trombone/stac` |

The exact upstream trees inspected were [VCSL at `c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e`](https://github.com/sgossner/VCSL/tree/c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e) and [VSCO-2-CE at `440300901dfe9275fd84e0b7763af1f8443ae62e`](https://github.com/sgossner/VSCO-2-CE/tree/440300901dfe9275fd84e0b7763af1f8443ae62e). Representative paths: [VCSL Harpsichord](https://github.com/sgossner/VCSL/tree/c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e/Chordophones/Zithers/Harpsichord%2C%20English/Sustains/Normal), [VCSL Xylophone](https://github.com/sgossner/VCSL/tree/c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e/Idiophones/Struck%20Idiophones/Xylophone/Hard%20Mallets), [VSCO-2-CE Tuba](https://github.com/sgossner/VSCO-2-CE/tree/440300901dfe9275fd84e0b7763af1f8443ae62e/Brass/Tuba), [VSCO-2-CE Tenor Trombone](https://github.com/sgossner/VSCO-2-CE/tree/440300901dfe9275fd84e0b7763af1f8443ae62e/Brass/Tenor%20Trombone), and [VSCO-2-CE Solo Contrabass](https://github.com/sgossner/VSCO-2-CE/tree/440300901dfe9275fd84e0b7763af1f8443ae62e/Strings/Solo%20Contrabass).

A deterministic synthetic-fixture test exercises the production library loader, note selection, and inventory output for these paths; it also verifies that UprightBass loads its non-vibrato sustain and real spiccato bank, and that Tuba and Trombone load their real staccato banks. The fixture proves that the mapping and indexer agree on the declared paths and filename tokens. It does **not** replace auditing the actual downloaded recordings or listening to rendered examples.

Duduk and ChoirOoh remain synthesis-only because neither inspected upstream tree contained a matching sample set. Timpani also remains out of this pitched-note mapping because the available hit filenames encode drum identities, not scientific pitch. Completing the catalogue honestly means adding a dedicated drum-number-aware schema for Timpani and acquiring/locating a genuinely matching sample set for Duduk and ChoirOoh—not assigning the nearest available patch and calling it coverage.

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
| Which indexed WAV files failed decoding, and which coverage bucket owns each one? | reported_sample_decode_failures with filename, MIDI label, normalized dynamic strength and reason; omitted_sample_decode_failure_count |
| Did indexed WAV bytes change while paths and file sizes stayed the same? | Confirm `indexed_sample_content_hashing_enabled` is true, then compare indexed_sample_content_manifest_sha256 for mappings with indexable files |
| What WAV formats occur among decoded files? | distinct sample-rate and channel-count fields |
| What pitch/dynamic filename coverage is visible? | min/max MIDI, distinct pitch count, per-pitch `pitch_coverage`, per-row `audio_evidence_status`, and filename_pitch_gaps_midi_within_observed_range |
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

1. Capture runtime preset enumeration from the loaded FluidSynth instance, not only the static SoundFont header. The official API exposes preset iteration (`fluid_sfont_iteration_start` / `fluid_sfont_iteration_next`) and preset identity getters for name, bank, and program; keep this behind an optional engine capability so a machine without libfluidsynth can still generate the static inventory. Compare runtime-enumerated slots with the RIFF `phdr` header and preserve ambiguous duplicate-slot evidence. Reference: [FluidSynth SoundFont Loader API](https://www.fluidsynth.org/api/group__soundfont__loader.html).
2. Compare repeated renders for byte equality and tolerant signal similarity under pinned executable/SoundFont identity; retain hashes as receipts rather than assuming hashes alone prove audible determinism.
3. Add note-range, dynamic-layer and articulation coverage from real, decoded files rather than assuming one sample equals full support.
4. Add a drum-number-aware index and dedicated percussion semantics instead of forcing percussion into pitched-note matching.
5. Add import → instrumentation sidecar → MIDI export → re-import fixtures for patch, bank, channel, part identity and instrument-change preservation, then add independent rendered-output and listening evidence.

This is a coverage-observability tool, not a claim that all 42 instruments have equally complete or perceptually equivalent implementations.
