# REA behavior observation sidecar profile v0 — 9 November 2026

**Status:** original synthetic development fixture and capture-integrity checker. No original executable was run to produce these examples; they are synthetic bytes used to validate the profile shape and capture digest logic.

**Owner:** [#11227 — observation-to-spec fidelity for black-box reconstruction](https://github.com/Luminous-Dynamics/symthaea/issues/11227).

## Purpose

Black-box reconstruction begins with an observation problem. Before an agent can produce a behavioral rule, it must bind the exact invocation and environment to the bytes and effects actually observed.

This profile records a *sidecar* that points to an existing REA Evidence record. It does not define another Evidence identity, digest projection, signing system or benchmark receipt.

## Trust boundary

REA Evidence v1's semantic identity includes subject digest, provider/profile, operation, parameters, raw and normalized results, confidence/authority, environment, limitations and locations. Its raw importer separately computes the raw-file SHA-256 and rejects duplicate keys/oversized files. The compatibility core expressly does not authenticate the producer or prove that the observation happened.

This profile therefore requires:
- the existing Evidence ID;
- the exact Evidence raw-file SHA-256;
- the same subject digest;
- explicit `admission_state = unverified_reference` in v0.

The checker verifies the *sidecar's captured byte strings* against their declared hashes. It does not admit or verify the referenced REA Evidence record. Until the established `verify_evidence_json` API is invoked on the exact referenced Evidence bytes, the binding stays unverified. A syntactically valid `ev_<sha256>` is not proof that the referenced bytes were supplied.

## Captured fields

Each observation binds:
- target subject and executable SHA-256;
- runner OS, architecture, locale, timezone, isolation and network policy;
- argv as exact per-argument bytes (base64), stdin capture and working directory;
- environment variable capture state: captured, redacted or absent;
- timeout budget;
- start/end times and termination kind (exit, signal, timeout, launch error);
- stdout/stderr capture states with exact captured bytes, digest and length;
- filesystem capture state and per-path create/modify/delete digests;
- assertions scoped to the capture they reference.

### Scope and portability

v0 represents argv as arbitrary bytes, not UTF-8-only strings, so it can preserve non-UTF-8 POSIX arguments. Working-directory and filesystem paths are normalized relative paths inside the declared sandbox; absolute paths, parent traversal and ambiguous separators are rejected by the checker. Environment values may be redacted, but redaction means the invocation cannot be fully replayed from this record alone.

A `complete` stream means the collector claims it captured the whole stream for that exact execution. It is not proof that the program's possible behaviors were exhaustively explored. A `partial`, `unavailable` or `not_applicable` stream cannot support an exact-output assertion.

A complete filesystem diff means complete coverage only within the declared sandbox/root and tracer contract. It does not imply that there were no external network, kernel, device or out-of-sandbox effects.

## Capture states

| State | Meaning | Permitted inference |
|---|---|---|
| complete | Full stream/diff captured under declared collector | Exact observation for this invocation/scope |
| partial | Some bytes/items captured; coverage gap known | Only statements about the captured subset |
| unavailable | Collector absent/failed | No claim about the unobserved channel |
| not_applicable | Channel intentionally not used by the invocation contract | No behavior claim about that channel |

Never collapse timeout, launch failure, signal termination, missing output, empty output and exit code zero into one success/failure boolean.

## Assertions

An assertion references one complete capture:
- `stdout_sha256`, `stderr_sha256`;
- exact `exit_code` or `signal`;
- `filesystem_diff_empty`;
- `filesystem_path_present`.

A hash or assertion describes an observed output, not its semantic meaning. Higher-level behavioral clauses must carry scope, evidence references, tested input class, falsification conditions and an explicit support/unknown state in a separate layer.

## Qualification command

```bash
cargo test --manifest-path tools/rea-behavior-observation-check/Cargo.toml --locked
cargo clippy --manifest-path tools/rea-behavior-observation-check/Cargo.toml --all-targets --locked -- -D warnings
```

The checker:
1. meta-validates the embedded Draft 2020-12 schema;
2. validates the synthetic fixture;
3. recomputes SHA-256 and length for argv/stdin/stdout/stderr/environment captures;
4. rejects path traversal and duplicate environment/filesystem identities;
5. rejects assertions against incomplete/unavailable captures;
6. preserves the explicit unverified Evidence binding state.

## Benchmark implication

Behavior2Code's September 2026 design independently validates original-program oracle consistency, discoverability of graded behavior, and deterministic repeatability. Its 60 programs, 6,872 hidden differential assertions and six-model/1,080-run study show how much headroom remains in black-box reconstruction. The task suite checks stdout, stderr, exit status and filesystem changes; the original program must pass all its own tests. It reports GPT-6 Astra at pass@3 19/60 (31.7%) and 35 programs unsolved by all evaluated models. These are the benchmark authors' results, not Symthaea measurements.

REA's extension should bind every discovered behavioral clause to exact sidecar observations and target/environment scope, then separately run hidden differential checks against the original executable. Do not compare pass@3 directly with SRE-Bench or AgentRE-Bench.

Reference: https://www.turing.com/blog/behavior2code-benchmark-reverse-engineer-software

## Claim ceiling

This v0 validates sidecar capture structure and consistency only. It does not authenticate the referenced Evidence record, prove producer identity, verify that a runner really executed the named binary, establish absence of unrecorded side effects, prove behavioral completeness, or establish equivalence for untested inputs. Synthetic fixtures are not original-program observations.
