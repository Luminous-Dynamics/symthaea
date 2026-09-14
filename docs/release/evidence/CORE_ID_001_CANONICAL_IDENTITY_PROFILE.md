# CORE-ID-001 Explicit Cross-Language Canonical Lexical Identity Profile

Status: independent reference semantics; cross-language Python/Rust candidate.

## Purpose

The repository currently has multiple local identifier validators and PIE-ID-001's qualified reference semantics use Python's runtime `str.strip()` table. CORE-ID-001 freezes a domain-neutral lexical waist whose semantics are explicit enough to implement identically in Rust and Python without making either runtime's evolving Unicode tables protocol authority.

Profile ID:

`symthaea.core.canonical-lexical-id.v1`

## Core admission algorithm

Given raw UTF-8 bytes and a positive domain-supplied maximum byte length:

1. input bytes must be nonempty;
2. byte length must be `<= max_utf8_bytes`;
3. bytes must decode as strict UTF-8;
4. first and last Unicode scalar values must not belong to the frozen edge-whitespace set below;
5. no scalar value may be ASCII C0 `U+0000..U+001F` or DEL `U+007F`;
6. accepted bytes are preserved exactly;
7. case is preserved;
8. no Unicode normalization or case folding is performed.

The maximum byte length is **profile/domain policy**, not a global constant. PIE may use 4,096 bytes while existing fabrication namespaces may use 256 without changing the lexical algorithm.

## Frozen edge-whitespace set

V1 contains exactly 25 code points:

- `U+0009..U+000D`;
- `U+0020`;
- `U+0085`;
- `U+00A0`;
- `U+1680`;
- `U+2000..U+200A`;
- `U+2028`;
- `U+2029`;
- `U+202F`;
- `U+205F`;
- `U+3000`.

Implementations test membership in this explicit set/range list. They do not call host-language whitespace predicates or trimming functions for admission.

This makes the profile stable even if Rust, Python, libc, ICU, or Unicode-property tables change later.

## Compatibility with qualified PIE-ID-001

PIE-ID-001 qualified the Python 3.13.5 relation:

```text
valid UTF-8
+ 1..4096 bytes
+ value == value.strip()
+ no ASCII C0/DEL
```

The CORE-ID-001 Python oracle includes an exhaustive compatibility proof over every Unicode code point at both identifier edges:

```text
legacy PIE-ID-001 Python 3.13.5 acceptance
    ==
CORE-ID-001 explicit-set acceptance
```

for the edge-whitespace/control dimension.

This matters because Python additionally treats some C0 separators as whitespace; those are already rejected by PIE-ID-001's independent C0-control rule, so the explicit 25-code-point edge set preserves the final admission relation rather than copying an implementation quirk unnecessarily.

The qualified PIE-ID-001 subject remains immutable historical evidence. CORE-ID-001 is a successor profile; it does not relabel or mutate PIE-ID-001.

## Cross-language oracle pair

Two independent implementations are checked in:

- `scripts/core-canonical-identity-oracle.py`;
- `scripts/core-canonical-identity-oracle.rs`.

Both consume the same language-neutral TSV corpus:

- `docs/release/evidence/CORE_ID_001_GOLDEN_VECTORS.tsv`.

The qualification workflow compiles/runs them under pinned Python 3.13.5 and Rust 1.96.0, captures their vector outputs, and requires byte-for-byte identical results.

The Python implementation uses `.strip()` exactly once, only inside the exhaustive **legacy-equivalence audit**. The new admission algorithm itself uses only the explicit code-point table. The Rust implementation uses no `trim()` or `char::is_whitespace()` authority.

## Golden-vector boundaries

The shared corpus freezes, among other cases:

- every V1 edge-whitespace code point on both leading and trailing edges -> reject;
- interior NBSP, EM SPACE and NEL -> accepted because V1 forbids them only at edges and they are not ASCII C0/DEL;
- embedded NUL, unit separator and DEL -> reject;
- invalid UTF-8 -> reject;
- case variants -> independently accepted and byte-distinct;
- NFC and NFD forms -> independently accepted and byte-distinct;
- Latin/Cyrillic lookalikes -> independently accepted and byte-distinct;
- `U+200B` ZERO WIDTH SPACE -> accepted at an edge under V1;
- `U+FEFF` BOM/ZWNBSP -> accepted at an edge under V1;
- `U+180E` MONGOLIAN VOWEL SEPARATOR -> accepted at an edge under V1;
- `U+00AD` SOFT HYPHEN -> accepted at an edge under V1.

Those last cases are intentional compatibility facts, not claims that the profile solves visual spoofing or security-confusable problems.

## Byte-limit profile tests

Both implementations separately test exact ASCII and multibyte 4,096-byte boundaries and a 256-byte profile boundary. This demonstrates that lexical semantics are shared while domain length budgets remain explicit policy inputs.

## Deserialization/newtype direction

This oracle freezes only the lexical predicate. A later Rust production tranche should use it behind a validated newtype with private state and validation-preserving deserialization, for example conceptually:

```text
CanonicalIdentifier<P>
```

where `P` supplies the maximum byte budget and profile identity.

Production migration should be incremental. Existing serialized historical identifiers must not be silently reinterpreted.

## Non-goals

CORE-ID-001 is not:

- a Unicode normalization profile;
- a confusable/spoof-defense system;
- a namespace registry;
- a global uniqueness theorem;
- content addressing;
- identity proof or PKI;
- provenance/authenticity;
- currentness;
- evidence applicability;
- authorization.

Human-readable labels, notes, reasons and prose are not automatically canonical identifiers.

## Intended adoption

1. independently qualify the explicit cross-language profile;
2. add a small shared Rust lexical primitive/newtype;
3. make PIE-ID a domain profile/alias over that primitive;
4. migrate new stronger boundaries first;
5. migrate duplicated fabrication/aesthetic/legal validators only in focused compatibility tranches.

Tracks #2990, #2870, #2867, #2785, #2935, #3114, #2826, and master #1604.
