# PIE-ID-001 Canonical Lexical Identity Oracle

Status: independent reference semantics; synthetic fixtures only.

Profile: `symthaea.pie.canonical-id.v1`

## Purpose

Freeze one narrow V1 lexical contract for durable PIE identifiers and references so stronger layers do not independently reinvent incompatible trimming, case-folding, Unicode-normalization, control-character, or length rules.

This theorem is intentionally smaller than graph uniqueness, provenance, authenticity, content identity, currentness, applicability, feasibility, or authority.

## Core theorem

```text
non-empty text != canonical durable identifier

canonical durable identifier V1 =
    valid UTF-8 text
    + 1..4096 encoded bytes
    + exact equality with Unicode-whitespace trim
    + no ASCII C0 controls or DEL
    + case preserved
    + no silent Unicode normalization
```

Admission rejects rather than mutates. A caller that wants a normalized identifier must construct that identifier explicitly before admission.

## Exact V1 rules

1. Input must be a string that is validly UTF-8 encodable.
2. UTF-8 encoding must contain at least one byte and at most 4,096 bytes.
3. The original string must equal its Unicode-whitespace-trimmed representation. Leading/trailing whitespace is rejected, not removed.
4. Embedded ASCII C0 controls U+0000..U+001F and DEL U+007F are rejected.
5. Case is identity-bearing in V1.
6. Unicode normalization is not applied. Canonically equivalent NFC/NFD spellings remain distinct byte identities unless a later namespace-specific profile freezes normalization semantics.
7. Reference binding requires exact admitted UTF-8 equality. Similar-looking, differently cased, differently normalized, or otherwise byte-distinct references do not resolve.

## Compatibility rationale

The qualified lower PIE ontology uses `require_text`, which rejects only empty/all-whitespace strings and therefore still accepts surrounding whitespace in otherwise non-empty identifiers. PIE-ID-001 does not rewrite those qualified lower subjects.

Instead, this contract extracts the stricter identity boundary already used independently by higher layers such as PIE-002H-A and PIE-002F, so future production migrations can reuse one explicit semantic authority.

Human-readable names, source prose, notes, and descriptive grade labels are not automatically durable identifiers and are outside this rule unless a specific schema field explicitly adopts the profile.

## Adversarial fixtures

The self-test proves:

- `p1` is admitted and round-trips exactly;
- ` p1`, `p1 `, tab/newline-wrapped IDs, NBSP-prefixed IDs, and EM-SPACE-suffixed IDs are rejected;
- empty/all-whitespace IDs are rejected;
- NUL, embedded newline, other C0 controls, and DEL are rejected;
- an unpaired Unicode surrogate is rejected through the canonical-ID validation boundary as non-UTF-8-encodable text;
- the limit is 4,096 UTF-8 bytes, including multibyte Unicode cases;
- case variants remain distinct;
- NFC `é` and NFD `e + combining acute` remain distinct even though Unicode normalization can equate them;
- visually confusable Latin/Cyrillic forms remain distinct;
- references require exact admitted identity;
- non-string values are not coerced.

## Deliberate non-claims

PIE-ID-001 does not prove:

- global or namespace uniqueness;
- semantic equivalence of visually similar identifiers;
- Unicode spoof/confusable resistance;
- content identity;
- source provenance or authenticity;
- authority-currentness;
- evidence applicability;
- feasibility or execution authority.

A future production tranche may introduce validated newtypes/constructors and migrate selected PIE join-key fields with compatibility evidence. It must not silently reinterpret already-qualified historical records.

Tracks #2870, #2867, #2785, #2935, #2782, #1610, #1647, and master #1604.
