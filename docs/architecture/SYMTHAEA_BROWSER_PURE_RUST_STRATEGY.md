# Symthaea Browser: Pure-Rust Strategy and Engine Boundary

**Status:** Proposed architecture decision for review  
**Date:** 2026-10-12  
**Scope:** `crates/domains/symthaea-browser`, Luminous Platform's no-JS application policy, and any future browser-shell / browser-engine research  
**Decision sought:** Remove handwritten JavaScript from Symthaea's browser-control layer where native Chrome DevTools Protocol (CDP) commands can provide the behavior; explicitly separate the existing Rust browser agent from any future browser engine project.

## 1. Executive decision

**Yes, the code we own can become Rust-first with no authored JavaScript snippets in the browser agent. That does not mean an arbitrary modern website can be rendered and executed by a browser with no JavaScript engine.**

The current `symthaea-browser` crate is a Rust browser **agent/controller**, not an independently implemented HTML/CSS/browser engine. Its Cargo manifest depends on `chromiumoxide`; `CdpSession` launches or connects to Chrome/Chromium through CDP. The agent observes pages, encodes observations into HDC vectors, proposes actions, checks capabilities and safety policy, and records action receipts.

This is a useful foundation. Do not throw it away or immediately begin a browser-engine rewrite.

Adopt three separate language/runtime commitments:

1. **First-party control plane:** Rust only. No handwritten JS/TS for policies, case logic, browser actions, evidence processing, permissions, source connectors, or application services.
2. **Initial Luminous civic UI:** Rust/Leptos server-rendered HTML and standard HTTP forms. No client-side JS/WASM loader for core workflows.
3. **Arbitrary-web compatibility:** Until a separately qualified alternative exists, Chrome/Chromium remains an external, sandboxed page-rendering and JavaScript-execution dependency. The engine may execute page-authored JavaScript as web content; our agent should not inject its own helper programs into pages just to perform ordinary browser operations.

The full browser-engine effort is a separate research programme, not a prerequisite for the civic data demonstrator.

## 2. Current repository evidence

Reviewed against the current Symthaea `main` tree on 2026-10-12:

- [`Cargo.toml`](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-browser/Cargo.toml) declares `chromiumoxide` and Tokio; it does not provide a page renderer itself.
- [`src/lib.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-browser/src/lib.rs) describes the agent as a bridge to headless Chrome/Chromium via CDP.
- [`src/cdp.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-browser/src/cdp.rs) uses `Page::evaluate` for multiple controller functions: accessible-element matching, CSS scrolling, history back/forward, text extraction, JavaScript-based fallback observation, setting a focused field's value, focused-element detection, and an internal evaluation helper.
- [`src/actions.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-browser/src/actions.rs) defines typed actions and capabilities.
- [`src/executor.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-browser/src/executor.rs) is intended to be the only normal route from a cognitive action proposal to a browser mutation.
- [`src/safety.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-browser/src/safety.rs) validates action policy, URL structure, origin scope and local-network restrictions before CDP dispatch.
- [`examples/praxis_active.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-browser/examples/praxis_active.rs) and [`examples/praxis_conscious.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-browser/examples/praxis_conscious.rs) contain direct CDP/JS exploration helpers and explicitly warn that they bypass `BrowserExecutor`; they are not production safety evidence, but should also be retired or rewritten if the no-authored-JS rule is applied to maintained examples.
- [Patch Sets 79–84 verification report](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-browser/symthaea-browser-hardening-bundle-79-84/symthaea-browser-hardening-verification-79-84.md) explicitly says Cargo compilation, unit/integration tests and live Chromium tests were not run in its environment because Cargo was unavailable.

This distinction is important: the existing crate has meaningful Rust safety/control logic, but it currently still contains authored JavaScript **strings sent into the page runtime**. The proposed policy must not claim those strings have already been removed.

### Scope beyond production code

The same inventory found direct `Page::evaluate` helper functions in the two active Praxis exploration examples. Their source comments already label them raw CDP harnesses that bypass `BrowserExecutor`; this is an important scope distinction, not an excuse to treat them as production. The implementation issue should either rewrite them to use the typed executor or move them to an explicitly archived / opt-in research area. Maintained examples should not teach a shortcut that the product safety architecture prohibits.

## 3. What “pure Rust” should mean

The phrase conflates several different goals. Track them separately.

| Goal | Achievable? | Recommendation |
|---|---|---|
| All first-party browser-agent control and policy code is Rust | Yes | Required |
| No handwritten JS snippets injected into page contexts for routine observation/actions | Yes in principle via native CDP commands, subject to protocol/API coverage | First implementation target |
| Civic admin UI works without JavaScript | Yes | Required in the first release |
| The browser application shell is written in Rust | Yes | Possible separate product surface |
| Render static HTML/CSS without any JavaScript engine | Yes, for a defined compatibility profile | Useful as a controlled internal viewer |
| Execute arbitrary modern web JavaScript without a JS engine | No | Not a realistic requirement |
| Full modern web browser engine with rendering, DOM/Web APIs, JavaScript and broad compatibility, all Rust | Possible as a research ambition, but a major long-term project | Do not make it a near-term dependency |

A browser running third-party JavaScript is not the same as Luminous authoring JavaScript. If the strict goal is **no JavaScript execution at all**, the user experience must be an explicit static/limited-web profile and must not claim compatibility with websites whose essential behavior requires scripts.

## 4. Immediate engineering target: native CDP operations, no page-evaluation helpers

Create a typed backend layer in which each action maps to a documented, bounded browser protocol operation. Use `chromiumoxide`'s native command objects / CDP domain APIs where available, confirmed against the version actually resolved by the workspace lockfile.

Candidate migration map:

| Current behavior in `cdp.rs` | Preferred replacement | Required evidence |
|---|---|---|
| Read accessibility tree | CDP Accessibility domain | Existing AX path is primary; bound node counts/text and handle unavailable tree explicitly |
| Click CSS/accessibility target | Resolve target through DOM/Accessibility, obtain an unambiguous node/box model, then use DOM focus or coordinate/input commands | Tests for 0, 1, and multiple matches; frame/shadow-root limits explicit; no guessed target |
| Scroll an element | Native DOM scroll-into-view CDP command, where supported | Missing/stale node and detached-frame tests |
| History back/forward | Page navigation history commands | Empty history and navigation-race tests |
| Extract visible text | Bounded accessibility tree first; when more is needed, native DOM protocol plus a Rust HTML/text extraction path | Limits, visibility caveats, dynamic content and truncation are explicit |
| Focused-element detection | Native focus/DOM protocol where available; otherwise report unknown rather than execute an injected script | Tests for focus loss, stale node and ambiguous controls |
| Set active input value by assigning `el.value` and dispatching events | Use real input/key/text-insertion commands and appropriate focus operations | Controlled React/Vue-style inputs, contenteditable, password fields and failure behavior tested; do not fake successful typing |
| Fallback `get_page_state_js` | Prefer a bounded AX snapshot; if unavailable, return an explicit `ObservationUnavailable` or limited DOM-derived observation | Must not silently fall back to injecting JavaScript |
| Arbitrary `eval_js` helper | Remove from production surface or confine to an explicitly test-only diagnostic feature that is off by default | Build/release test proves routine paths do not call page Runtime.evaluate |

**Do not perform mechanical replacements without checking semantics.** Some CDP operations can replace a snippet cleanly; others cannot exactly reproduce page-visible text or framework-specific input events. Where a protocol-level replacement cannot meet the semantic contract, expose a typed unsupported/partial result and keep the test honest rather than quietly keeping JavaScript.

First inspect the exact locked `chromiumoxide` version and generated protocol types. Do not code against latest docs if the workspace uses a different version. Add tests and execute them before calling the no-injected-JS property qualified.

## 5. Make the backend replaceable before changing engines

Introduce a narrow internal abstraction only if it reduces coupling and can be validated without creating a second policy system. The interface should describe observations and typed actions, not expose arbitrary evaluation:

- `BrowserBackend::observe`
- `BrowserBackend::navigate`
- `BrowserBackend::activate`
- `BrowserBackend::enter_text`
- `BrowserBackend::scroll_to`
- `BrowserBackend::history`
- `BrowserBackend::screenshot`

The existing `BrowserExecutor` and safety policy remain above the backend. All engines must use the same explicit capability checks, URL policy, timeout/resource budgets, receipt semantics and human-consent rules. No backend is permitted to bypass the executor by exposing raw mutation primitives to the cognitive loop.

Backend conformance fixtures should run against the mock/deterministic backend first and against real engines as separately qualified integration lanes. Do not promise identical semantics where one engine lacks a capability.

## 6. Security boundary requirements

Native CDP instead of injected JavaScript reduces hidden in-page execution, but **does not make browser automation inherently safe**. The renderer still processes hostile HTML, CSS, images, downloads and possibly site JavaScript.

Before this becomes a supported automation surface:

- Launch Chromium with a dedicated ephemeral profile per isolated task or tenant; do not reuse the user's logged-in profile by default.
- Keep the DevTools endpoint on an isolated loopback channel or inherited pipe, not a network-listening debug port exposed to other tenants or LAN clients.
- Enforce URL policy at navigation and redirect boundaries, including resolved destination addresses; account for DNS rebinding, alternate IP notation, IPv4-mapped IPv6 and local-network transitions. URL parsing alone is not enough.
- Apply network egress restrictions outside the page process; do not rely on a UI blocklist as the only SSRF defense.
- Restrict downloads, file access, clipboard access, permissions, popups, external protocols and cross-origin workflows unless a specific capability and approval permit them.
- Bound navigation duration, redirects, frame depth, tabs, DOM/AX nodes, response bytes, screenshots, memory and CPU.
- Treat page text, DOM attributes, alt text, labels, QR-decoded data, links and embedded instructions as **untrusted input**, never as policy or authority.
- Bind each action receipt to the target origin, normalized target description, initiating case/task, capability, pre-action observation digest, policy version and post-action outcome. Do not put secrets or unrestricted page contents into receipts.
- Use read-only observation by default. For forms and external side effects, require a distinct capability and explicit approval. A model's confidence or Phi value cannot mint authority.
- Add a no-browser / browser-crash fallback. The rest of Symthaea must not assume that a browser process is present or trustworthy.

## 7. Research options for a Rust browser engine

### Option A — Keep Chromium as the compatibility backend, Rust owns the controller

**Best near-term choice.** This preserves the most useful web compatibility while reducing the first-party language surface and making agent policy independent of the renderer. Keep Chromium as a replaceable, sandboxed dependency and do not describe this configuration as a pure-Rust browser engine.

### Option B — Evaluate Servo as a mostly-Rust web engine

Servo is a Rust browser-engine project, but its documented JavaScript execution currently relies on Mozilla's SpiderMonkey, which is written in C++. It is therefore not a fully-Rust end-to-end JS-capable browser today. Embedding it would require evaluating its embedder API, supported web platform, sandbox/process model, codecs, fonts, platform integrations and actual workload compatibility. It is an experiment candidate—not an automatic drop-in replacement.

Sources:
- https://github.com/servo/servo
- https://github.com/servo/servo/wiki/Servo-and-SpiderMonkey-Report

### Option C — Evaluate Boa for a constrained Rust-only web runtime

Boa is an embeddable JavaScript engine written in Rust, but it is an ECMAScript engine, not a complete modern browser engine. It does not by itself supply a full DOM, CSS layout/rendering, HTML document lifecycle, browser security model, media stack, storage and web APIs. Its published conformance figures and feature roadmap must be pinned to an exact release before adoption.

Boa could be evaluated for a **limited, controlled browser profile** or internal synthetic pages. Replacing Chromium's execution engine with Boa is not a crate swap: it requires implementing and qualifying the enormous bridge among the DOM, Web APIs, event loop, promises/microtasks, fetch, storage, security origin model and JS objects.

Sources:
- https://github.com/boa-dev/boa
- https://docs.rs/boa_engine/latest/boa_engine/

### Option D — Build a fully native Luminous browser engine

This is theoretically possible but requires a sustained browser-engine programme: HTML parsing, DOM, CSS parsing/cascade/layout, painting/compositing, fonts, Unicode, images, video/audio, networking/TLS/certificates, cookies/storage, permissions, origin isolation, JavaScript, Web APIs, accessibility, downloads, extensions, sandboxing and continual standards conformance. A minimal HTML/CSS viewer can be built much earlier, but must explicitly limit compatibility.

Do not initiate a from-scratch engine simply to eliminate a handful of injected JS snippets. Prove the controller's language boundary first; let measured compatibility needs justify any engine research.

## 8. Recommended staged plan

### P0 — Make claims accurate

- Explicitly describe `symthaea-browser` as a Rust browser agent backed by Chromium/CDP.
- Inventory every page `evaluate` and classify it as observation, mutation, navigation, typing or diagnostic.
- Preserve an exact list of all in-page JavaScript snippets until each has a tested replacement or an explicit approved exception.

### P1 — Remove first-party page-evaluation helpers

- Pin and inspect the resolved `chromiumoxide`/CDP types.
- Migrate one operation at a time, each with a regression test.
- Remove unrestricted `eval_js` from default production builds.
- Prefer the accessibility tree; never silently degrade into executing an injected script.
- Add a testable invariant that the normal action path cannot dispatch arbitrary source text to `Runtime.evaluate`.

### P2 — Qualify controller and hostile-page behavior

- Run `cargo fmt --check -p symthaea-browser`, strict Clippy, crate tests and the actual live-Chromium integration suite on an exact commit.
- Test prompt-injection content in page text, deceptive links, origin redirects, DNS changes, private-network targets, stale elements, ambiguous controls, form side effects, downloads, crashes and timeouts.
- Preserve the failure record if any gate is unavailable. Tests written or queued are not passes.

### P3 — Introduce the backend contract

- Keep safety policy and action receipts above the backend boundary.
- Exercise deterministic mock behavior, then the existing Chromium implementation.
- Only after the abstraction is stable, prototype a second backend or a native restricted renderer.

### P4 — Engine feasibility study

Compare a concrete workload corpus across Chromium, Servo and any constrained Boa-backed prototype. Measure standards compatibility, accessibility output, isolation, memory, startup, rendering correctness and failure modes. No replacement is approved by code-language purity alone.

## 9. Acceptance criteria for “Rust-controlled, no injected JS”

The P1 property is accepted only when:

- No routine observation, navigation, click, typing or scroll path calls CDP `Runtime.evaluate`.
- The arbitrary-evaluation helper is absent from production builds, or is separately gated, non-default and not reachable through the agent action protocol.
- Typed CDP requests are bound to the actual pinned dependency version.
- Tests prove missing/ambiguous targets fail safely; zero, one and multiple matches do not collapse into the same result.
- Text extraction and input changes retain documented semantics; unsupported cases are explicit, not silently misreported as success.
- No page-provided string can alter policy, capabilities, permitted origins or authority receipts.
- Cargo, formatting, Clippy, unit/integration and live-browser tests have visible passing results bound to the exact source revision.
- The documentation says **Rust controller / Chromium engine** until an alternative engine itself passes a separate qualification.

## 10. Implementation update — 2026-10-13

The native-CDP implementation is under review in [PR #7429](https://github.com/Luminous-Dynamics/symthaea/pull/7429), on branch `feat/symthaea-browser-native-cdp-actions-001`. The current implementation head at the time of this update is `a562162e04ddec8b6bd67c4b14972df54e519e3c`.

The maintained `cdp.rs` controller path no longer uses authored page-evaluation helpers for accessibility-target resolution, scroll, clicks, history, focus detection, title retrieval or text extraction. Operations use typed CDP requests and Rust `scraper 0.25.0`; CSS selectors for actions/extraction must match exactly one node, and click/type targets require live Accessibility-tree evidence. Disabled click targets and disabled/read-only text inputs fail closed. The old JavaScript fallback observation and arbitrary `eval_js` helper were removed, and the dead JS-named compatibility shim was removed as well.

Raw session reads and screenshot methods are crate-private. Maintained `WebAgent` and `praxis_journey.rs` now route perception through `BrowserExecutor::observe()`, which checks the Observe capability/Phi and current URL before returning the accessibility tree. URLs in returned observations redact credentials and common secret-bearing query values. The executor also checks the current URL before click/type/scroll/extract/screenshot, checks history destination URLs before history navigation, checks the landed URL after top-level navigation before returning an `Executed` receipt, and serializes complete executor transactions per session.

Text extraction excludes script/style/noscript/template and explicit hidden-state subtrees, bounds parsed markup at 2 MiB, and caps output at 32 Ki characters. The size check occurs after CDP returns the markup, so it is not a transport-level response-size limit; the result is DOM text, not exact pixel-rendered `innerText`. Computed CSS visibility, shadow-root flattening, and iframe text are not synthesized.

The raw Praxis exploration examples are preserved under `examples/archive/` with archive banners; the maintained journey uses the policy-checked executor. A structural unit test scans all maintained browser modules and the active journey for direct evaluation and JS-backed Chromiumoxide convenience APIs, checks that raw session observation APIs are not public, and prevents the maintained WebAgent/BrowserBridge/Praxis callers from invoking low-level CdpSession action or observation methods directly.

**Verification remains unqualified.** At the last check, broad repository CI, formatting, Clippy and regression jobs for the implementation head were queued/pending, with the benchmark suite skipped. No Cargo formatting, compilation, unit/integration or live-Chromium hostile-page pass is confirmed for the exact head; this environment does not have Cargo/Rust installed. Queued/skipped checks are not passes.

This patch is **not a network sandbox or redirect interceptor**. A disallowed redirect can issue a request before the controller observes the landing. Pre-request/redirect/subresource enforcement with external egress constraints is tracked in [issue #7435](https://github.com/Luminous-Dynamics/symthaea/issues/7435). Audit receipts also still embed the full action and may expose typed secrets or page output; receipt-safe summaries and output separation are tracked in [issue #7436](https://github.com/Luminous-Dynamics/symthaea/issues/7436).

This does not replace Chromium's renderer or website JavaScript engine. It narrows the first-party authored-JavaScript surface in the Rust controller; it does not claim an end-to-end pure-Rust browser engine.

## 11. References

- Symthaea Browser manifest: https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-browser/Cargo.toml
- Symthaea Browser CDP wrapper: https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-browser/src/cdp.rs
- Symthaea Browser executor: https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-browser/src/executor.rs
- Symthaea Browser safety policy: https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-browser/src/safety.rs
- Chromiumoxide CDP client: https://docs.rs/chromiumoxide/latest/chromiumoxide/
- Servo engine: https://github.com/servo/servo
- Boa Rust JavaScript engine: https://github.com/boa-dev/boa
- Luminous Platform zero-JS first-release policy: https://github.com/Luminous-Dynamics/luminous-platform/blob/docs/public-service-operations-profile-rfc-001/docs/LANGUAGE_AND_RUNTIME_POLICY.md

---

**Qualification note:** This strategy document itself changes no browser behavior. The separate behavior changes are under PR #7429 and remain unqualified until exact-head Cargo, formatting, Clippy, unit/integration and live-browser evidence is recorded. No document here certifies Chromium/Servo/Boa as a pure-Rust browser engine.
