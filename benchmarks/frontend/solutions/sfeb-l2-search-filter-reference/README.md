# SFEB-L2-005 reference candidate

This is a Leptos 0.8 CSR candidate for public-pilot task SFEB-L2-005. It
uses the same pure Rust search/filter/sort/pagination module as the
standalone checker at `benchmarks/frontend/contract-check/src/search_filter.rs`.

It implements search over name/SKU/description, conjunctive category and
availability filters, inclusive integer-cent price bounds, deterministic
sorting, one-based pagination and complete query-state deep links. The page
uses labelled controls, a skip link, visible focus, semantic table headers,
status announcements and responsive table containment.

The synthetic dataset is compiled into the WASM bundle. No remote API,
fonts, images, scripts, CDN or third-party runtime are used. Price parsing
and display use integer cents, not floating-point arithmetic.

## Run locally

Install Rust with the `wasm32-unknown-unknown` target and Trunk, then run:

```sh
cargo test
trunk serve --address 127.0.0.1
```

For a release bundle:

```sh
cargo build --release --target wasm32-unknown-unknown
trunk build --release
```

A candidate-local Cargo.lock has not yet been generated. Browser
qualification is still required for keyboard behavior, responsive geometry,
URL restoration, accessibility output, request egress and the independent
mutation controls.

## Evidence status

This package is authored source, not a verified build. It has **not** been
compiled, tested, formatted, linted or opened in Chromium in the authoring
environment. Its existence does not make SFEB-L2-005 runnable and does not
increase the runnable task count.


## Canonical URL state and invalid links

Shared state URLs use exactly one `?state=` parameter with exactly eight fields. Reserved and non-ASCII UTF-8 bytes are percent-encoded; raw URL-component bytes are limited to RFC 3986 unreserved characters. Search/name sorting use ASCII-only case folding and C0/C1 query controls are rejected. Price values remain within the JavaScript safe-integer range.

A malformed or partial initial URL is shown as an accessible error. The candidate does not replace the user's URL with default state until the user deliberately changes a filter or clears the state. These behaviors are source-authored and require browser qualification; they are not yet browser-verified.

Shared conformance vectors: `benchmarks/frontend/fixtures/search-filter-explorer/url-state-conformance-v1.json`.
