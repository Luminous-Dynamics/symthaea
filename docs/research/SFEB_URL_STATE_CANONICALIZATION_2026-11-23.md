# SFEB URL-state canonicalization and cross-language conformance

**Date:** 2026-11-23  
**Status:** JavaScript model logic executed in a shimmed isolated V8 harness; Rust, WASM and browser execution pending  
**Task:** SFEB-L2-005  
**Related:** [#13329](https://github.com/Luminous-Dynamics/symthaea/issues/13329), [PR #13162](https://github.com/Luminous-Dynamics/symthaea/pull/13162)

## Why this matters

The Rust reference model, independent JavaScript oracle and Leptos URL-restoration path must agree on the same accepted input language. Otherwise, the benchmark can certify behavior that the actual application interprets differently, or its independent oracle can reject a correct implementation for a behavior not defined by the task.

Static inspection found four cross-language risks:
- Rust's URL decoder accepted raw UTF-8 bytes while the JavaScript decoder required percent encoding.
- Rust's Unicode control-character check includes C1 controls, while the JavaScript check only covered C0 and DEL.
- Rust performed ASCII-only lowercasing as required by the task, while JavaScript used Unicode lowercasing.
- The Leptos startup path converted every invalid deep link into the default query, and an initial effect could overwrite the supplied invalid URL.

A further issue was incomplete URL-state objects: Rust deserialization and JavaScript default merging both supplied omitted fields, even though a shareable state URL is required to restore the complete state.

These are source-level semantic gaps, not observed runtime failures.

## Canonical state contract

The shared policy and test vectors are in:
- `fixtures/search-filter-explorer/url-state-conformance-v1.json`
- `url-state-conformance-v1.schema.json`

The accepted URL form is a single `?state=` query parameter containing percent-encoded UTF-8 JSON.

Rules:
1. The URL must contain exactly one state parameter; unknown additional query parameters are rejected.
2. Literal bytes in the encoded component must be ASCII RFC 3986 unreserved characters only: letters, digits, hyphen, period, underscore and tilde.
3. Reserved and non-ASCII content must be percent-encoded as UTF-8 bytes. A plus sign is data, not form-encoding whitespace, and must be encoded as `%2B`.
4. Percent escapes are case-insensitive; the decoded byte sequence must be valid UTF-8.
5. The JSON object must contain exactly eight fields: `q`, `category`, `available_only`, `min_price_cents`, `max_price_cents`, `sort`, `page` and `page_size`. Missing and unknown fields are rejected.
6. Query text rejects Unicode C0 and C1 control characters (U+0000–U+001F and U+007F–U+009F).
7. Search and name sorting use ASCII A–Z case folding only. Non-ASCII code points are not Unicode case-folded; this preserves the contract's explicit ASCII-case-insensitive semantics.
8. Monetary values and synthetic dataset numeric fields must be non-negative safe integers no greater than 9,007,199,254,740,991, so Rust `u64` and JavaScript `Number` cannot silently diverge.
9. Invalid percent escapes, invalid UTF-8, malformed JSON, missing/unknown state fields and invalid query values fail closed.

The writer emits uppercase percent hex and leaves only unreserved bytes literal. The reader accepts either hex case to avoid rejecting otherwise valid shared URLs.

## Implementation changes

### Rust reference evaluator

- Rejects raw reserved and raw non-ASCII bytes in URL components.
- Requires the exact eight state fields before deserialization.
- Rejects price values above the JavaScript safe-integer limit.
- Keeps C0/C1 control rejection through Rust's Unicode control predicate.
- Adds shared-vector unit tests for valid, invalid-state and invalid-component cases.

### JavaScript independent oracle

- Uses an explicit ASCII-only lowercase helper in search and sorting.
- Rejects C0 and C1 control characters.
- Requires safe, non-negative integer-cent bounds.
- Uses the same unreserved-only raw URL component grammar and exact field set.
- Executes the same shared vectors and reports their denominator separately from the existing semantic mutation count.

### Leptos candidate

- Initial URL parsing now returns a Result rather than silently defaulting.
- Invalid URL state is surfaced as an accessible alert.
- The initial invalid URL is not rewritten until the user deliberately changes a filter or clears the state.
- User interactions clear the initial URL error and then persist a valid complete state URL.

## Reproduction commands (pending a Rust-capable environment)

```sh
node benchmarks/frontend/independent-oracle/search-filter-reference.mjs
cargo test --manifest-path benchmarks/frontend/contract-check/Cargo.toml
cargo test --manifest-path benchmarks/frontend/solutions/sfeb-l2-search-filter-reference/Cargo.toml
```

The independent JavaScript oracle includes shared-vector assertions. Rust tests load the exact same JSON file. Browser validation is still required for the invalid deep-link alert and fresh-session URL restoration.

### 2026-11-23 V8 preflight (JavaScript logic only)

The committed JavaScript oracle body was executed in an isolated V8 harness after replacing only its Node file-loading/assertion/UTF-8 bootstrap with local equivalents and injecting the exact repository dataset, golden cases and shared URL vectors. The model logic produced:

- 11/11 golden cases;
- 5,000/5,000 deterministic generated states;
- 9/9 pre-existing semantic fault models;
- 8/8 URL-field omission controls;
- 14/14 new URL-state conformance vectors (3 valid, 6 invalid state objects, 5 invalid URL components).

The first run exposed an obsolete omission-control assumption: a strict decoder now rejects a missing state field rather than defaulting it, so the mutation test itself needed to count a thrown rejection as a detected omission. That test harness was corrected and the full V8 logic run then passed.

This is an executed JavaScript-model preflight in V8 with a compatibility shim, not a stock Node.js CLI run and not a Rust, Leptos/WASM, Chromium, trusted-executor or independent-verifier receipt. The standard reproduction commands remain the intended repository commands and should be run in the pinned host environment.


## Current evidence boundary

This increment authors implementation and test source. It has not been compiled or executed in Rust, built to WASM, or tested in Chromium. The shared vectors and schemas can be structurally inspected as repository artifacts, but they are not yet a test receipt. SFEB-L2-005 remains fixture_ready, the task-family plan remains at zero runnable tasks, and no benchmark qualification or performance result is claimed.

The next runner-independent step is to add differential/fuzz-generated URL strings to the same shared corpus, and to keep parser acceptance/rejection and user-visible invalid-state handling as separate assertions.
