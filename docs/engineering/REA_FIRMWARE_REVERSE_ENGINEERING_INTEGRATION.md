# ENG-REA-001: REA firmware reverse-engineering evidence integration

**Research snapshot:** 2026-10-18  
**Status:** design spike only; no upstream provider was executed from this Symthaea workspace.  
**Tracking issue:** [#8022](https://github.com/Luminous-Dynamics/symthaea/issues/8022)  
**Related engineering work:** [#5670 ENG-DEVICE-000](https://github.com/Luminous-Dynamics/symthaea/issues/5670), [#5633 ENG-SPICE-001](https://github.com/Luminous-Dynamics/symthaea/issues/5633), [#5661 ENG-SPICE-002](https://github.com/Luminous-Dynamics/symthaea/issues/5661)

## Decision

**Use REA as an optional, evidence-producing reverse-engineering toolchain for a bounded pilot. Do not make REA a core dependency, and do not treat REA output as proof of physical or engineering correctness.**

Canonical upstream: [morluto/rea](https://github.com/morluto/rea), [firmware analysis guide](https://github.com/morluto/rea/blob/main/docs/firmware-analysis.md), [MCP runtime contracts](https://github.com/morluto/rea/blob/main/docs/mcp-contracts.md), and [upstream roadmap](https://github.com/morluto/rea/blob/main/docs/roadmap.md).

REA is MIT-licensed for REA-authored code. Third-party tools and material retain their own licenses and terms. The upstream install flow requires Node.js 22.19+, 24.11+, or 26+ and npm; that runtime must remain outside Symthaea's default Rust build. Use the canonical upstream/release source, not lookalike forks.

## What the upstream can and cannot currently contribute

The following is based on upstream documentation, not an execution result from this repository.

| Capability | REA currently provides | It does not establish |
|---|---|---|
| Firmware triage | Binwalk-based region inspection and Unblob-based bounded extraction; source hashes, producer reports, region/range data, extraction hashes and explicit gaps/limitations | Correct partition semantics, decryption, completeness of arbitrary firmware reconstruction, or runtime behavior |
| Native code hand-off | A separately selected extracted ELF can be sent to a supported native-analysis provider, such as configured Ghidra | That arbitrary firmware bytes are executable, that every architecture is supported, or that pseudocode is source-equivalent |
| Recorded logic signals | A recorded-signal analysis path using caller-supplied sigrok-cli, returning annotations and sample ranges | Live acquisition, electrical validity, correct probe wiring, or hardware-in-the-loop results |
| Broader reverse engineering | Native binary inspection, structured evidence, and selected runtime/application observations | A circuit solver, PCB layout engine, hardware emulator, automatic firmware vulnerability scanner, or physical qualification |

### Current upstream firmware boundary

The documented first firmware milestone is verified on Linux x64 with Binwalk 3.1.x and Unblob 26.6.x. The report parsers were verified against Binwalk 3.1.0 and Unblob 26.6.4; accepted builds on those version lines may still carry a limitation that that exact build was not verified. These dependencies are caller-supplied, not automatically installed by REA.

The upstream guide explicitly excludes bootloader/device-tree semantics, raw flash load profiles, vulnerability scanning, emulation and hardware runtime capture from this milestone. Unknown or compressed child bytes cannot always be mapped back to original offsets. A reported extraction status of complete means only that the configured extraction finished without its reported gaps; it is not proof that all firmware formats, content or meaning were recovered.

## Integration architecture

Keep each stage separately identifiable and separately reviewable:

```text
authorized firmware / saved logic capture
  -> immutable input snapshot + SHA-256
  -> optional REA provider (Binwalk / Unblob / native provider / sigrok)
  -> raw report + tool identity + limitations
  -> narrow Rust-side normalization/import
  -> Symthaea engineering evidence and hypothesis graph
  -> new test or solver request
  -> independent verification / physical measurement
```

REA should provide observations and analysis candidates. Symthaea may use its cognitive/search machinery to correlate them, generate hypotheses and select the next informative experiment. Deterministic tools and independent measurements must decide whether an engineering property is established.

### Evidence to preserve

The integration should retain, at minimum:

- Original input SHA-256 and byte length; immutable snapshot identity; selected source range, where applicable.
- Producer/operation identity, reported version, actual executable path and executable SHA-256 when the provider supplies them, plus explicit unknowns for fields the provider cannot establish.
- Exact normalized operation parameters and an identity for relevant configuration; do not hash ambient environment variables as though they were fully controlled.
- Raw producer report bytes or a stable content-addressed artifact reference, its digest, and the normalization/schema version.
- Observed region offsets, extraction hashes, source-range lineage, sample ranges, coverage declarations, diagnostics and extraction gaps.
- Explicit limitations and unsupported semantics; distinguish observed, inferred, unknown, partial, contradicted and rejected.
- Any mapping between firmware regions, extracted binaries, functions, recorded signals, circuit assumptions and downstream simulation requests.

A digest establishes byte identity relative to the bytes that were hashed; it does not authenticate who produced the bytes or prove the claims are true. Producer claims and model-written explanations must not be promoted into verifier decisions without independent checking.

### Execution and safety boundary

- Initial milestone is static/file-based and read-only. Do not flash, program, probe, actuate or execute extracted firmware.
- Require explicit user-selected input and output locations. Refuse in-place modification and existing extraction destinations.
- Keep the REA, Binwalk, Unblob, Ghidra and sigrok toolchains optional. Do not add them to default workspace dependencies or silently auto-install them.
- Use exact configured executables and a documented, pinned profile. Do not accept arbitrary agent-generated shell strings as tool authority.
- Preserve upstream resource, depth, timeout, output and sandbox errors; add a Symthaea-level bound if an adapter introduces a weaker execution boundary.
- Missing, unsupported, partial, unparseable or contradictory evidence stays non-authorizing. An exit code, detected signature, parsed report, matching version string or agent confidence cannot qualify hardware.
- Run against authorized or synthetic/source-derived artifacts only. Public availability or possession alone does not establish permission for every analysis operation.

## Smallest credible pilot

**Reuse REA's existing firmware integration lane instead of inventing a competing fixture suite.** The upstream repository already includes:

- `scripts/fixtures/firmware/generate.mjs` and `generate.py` for a source-derived host ELF inside a deterministic gzip/USTAR firmware-like image.
- An independent `oracle.json` containing expected image hash/size, gzip offset/length, and fixture-file hashes/sizes.
- `scripts/verify/firmware/analysis.mjs`, which asserts real-provider behavior across CLI and MCP, expected region offsets and file hashes, selected-range extraction, explicit unknown-chunk gaps, depth limits, optional ext4 extraction/missing-extractor behavior, and a separately selected Ghidra handoff.
- Documented entry points `npm run fixtures:firmware` and `npm run verify:firmware` in the [upstream firmware guide](https://github.com/morluto/rea/blob/main/docs/firmware-analysis.md).

That is a stronger starting point than a second locally designed fixture because its verifier already defines an independent oracle and actual-provider checks. Do not label the lane passed merely because the code exists: it requires execution against an exact, recorded REA revision and the listed toolchain.

1. Pin the canonical REA commit/release and record hashes for the checkout, generated fixture manifest, source image, configured Binwalk/Unblob executables, and relevant runtime/configuration. REA's recorded executable digest/version should be preserved; a version banner alone is not full source-build provenance.
2. Run upstream fixture generation and its real-provider verifier in a documented Linux x64 environment. Capture stdout/stderr, exit codes, tool availability, oracle, and raw Evidence records. The optional ext4 and Ghidra lanes require their own prerequisites and must be reported independently.
3. Compare normalized CLI and MCP results using the existing upstream invariants. Verify input bytes unchanged, all expected file digests, selected ranges, and explicit partial/gap semantics. Report every unavailable lane rather than silently counting it as a pass.
4. Import a saved Evidence record through a deliberately narrow Rust parser/schema boundary. If no stable fixture contract exists, stop at artifact archival and report the mismatch instead of inventing a schema.
5. Add Symthaea-specific negative controls beyond the upstream lane where not already covered: corrupt/truncated Evidence envelope, mismatched source digest, changed selected range, changed tool executable identity/configuration, unsupported child architecture, missing/unknown producer schema, and attempted promotion of partial evidence. Keep existing upstream unknown-chunk, missing-extractor, and depth-limit tests as controls rather than duplicating them without a demonstrated gap.
6. Only after file-evidence interoperability passes, run a separate known-answer recorded-signal lane using sigrok-cli. Keep recorded-file analysis distinct from live device acquisition.
7. Build a held-out diagnosis evaluation only after the import boundary is stable. Compare a baseline engineer/agent using the task description alone against the same baseline receiving REA evidence, using blinded case labels and independently adjudicated answers.

## Benchmark protocol

| Metric | Initial acceptance target | Evidence required |
|---|---|---|
| Known-region localization | 100% of expected fixture regions at the manifest byte ranges; unexpected detections reported rather than silently ignored | Independent fixture manifest + raw provider report |
| Extraction integrity | 100% of required extracted fixture files match expected SHA-256; source bytes unchanged | Independent hash manifest before/after |
| Negative-control safety | 100% of malformed, partial, unknown, unsupported or missing-tool cases remain non-authorizing | Named negative fixtures and machine-checked decision table |
| Provenance sensitivity | Changing input bytes, tool executable identity, selected range or relevant configuration changes the appropriate evidence identity | Paired positive/negative identity tests |
| Coverage honesty | 100% of known injected extraction gaps remain visible as partial/unknown; never infer absence from an unobserved region | Gap injection and report round-trip tests |
| Reproducibility | Same pinned toolchain, input bytes and parameters reproduce normalized semantic evidence; volatile timing/path fields are excluded or modeled explicitly | Repeat runs plus documented canonicalization |
| Diagnosis benefit | On held-out labeled cases, improve correct fault localization or time-to-first-correct hypothesis without increasing unsupported assertions | Blind baseline comparison, adjudicated labels and evidence citations |
| Incremental value of REA | On the same held-out cases, REA evidence improves top-1 root-cause localization and/or reduces median diagnosis effort over the no-REA baseline; report both, do not collapse into a single score | Paired cases, randomized run order, frozen prompts/tools, case-level outcomes and adjudication |
| Abstention calibration | At least 95% correct abstention on cases whose available artifacts do not distinguish the injected causes; report coverage-versus-error curve | Deliberately ambiguous cases, known answerability labels, explicit unknowns |
| Evidence traceability | 100% of accepted causal claims cite a specific observed record or test; unsupported causal leaps separately counted | Claim-to-evidence ledger reviewed against each ground-truth case |

These targets apply only to the defined synthetic fixture set. They do not generalize to unknown vendor firmware. Before reporting model capability improvements, publish the number and diversity of held-out cases, baseline definition, abstention/error rates and the full confusion matrix. A benchmark pass would demonstrate fixture-level compatibility, not firmware safety or board qualification.

For a later logic-signal lane, use known-answer captures with explicit sample rates, decoder settings, expected sample ranges and protocol annotations. Do not conflate decoded protocol events with voltage thresholds, signal integrity or proof that the firmware generated the signal.

## Where this fits in Symthaea

- **ENG-DEVICE-000 (#5670):** owns reusable physical-device/component/evidence contracts. REA findings should enter that common model, not create a second device authority hierarchy.
- **ENG-SPICE-001 (#5633):** owns truthful real ngspice output parsing, convergence and provenance. REA evidence complements this work; it is not a substitute for circuit simulation.
- **ENG-SPICE-002 (#5661):** owns transitive solver-input closure. Future circuit results must bind their actual solver inputs and configuration, independently of reverse-engineering observations.
- **Engineering Trust Kernel / formal-safety boundary:** keeps candidate observations separate from admitted evidence, qualification and execution authority.

## Research conclusion

The promising paradigm shift is **reverse engineering as evidence recovery and uncertainty reduction**. The long-term chain is:

firmware bytes -> region/extraction lineage -> native control-flow hypothesis -> recorded signal evidence -> testable circuit/firmware model -> independently checked result

Each arrow should be a typed, auditable mapping with explicit unknowns, not a language-model inference presented as fact. Success means faster, more accurate engineering diagnosis with fewer unsupported claims—not automatic cloning or a claim that a reconstructed implementation is correct.

## Verification status

This file records public-document research and a reuse plan grounded in REA's existing generated fixture/oracle and real-provider verifier. No REA provider, Binwalk, Unblob, Ghidra or sigrok executable was run as part of this documentation change. No hardware was accessed. The existence of upstream verification code is not a test result; provider support in the Symthaea workflow, compatibility and all benchmark results remain unverified until the exact pinned pilot is executed.
