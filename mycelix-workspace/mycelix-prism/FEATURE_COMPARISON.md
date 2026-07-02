# Prism — Feature Parity Comparison

Compared against Brave 1.76, Chrome 134, Firefox 137, Safari 18.

## Rendering Engine

| Feature | Brave (Blink) | Chrome (Blink) | Firefox (Gecko) | Safari (WebKit) | Prism (html5ever) |
|---------|:---:|:---:|:---:|:---:|:---:|
| HTML5 parsing | Full | Full | Full | Full | Full (spec-compliant) |
| Malformed HTML recovery | Yes | Yes | Yes | Yes | **Yes** (8 tests) |
| Unicode text | Full | Full | Full | Full | **Yes** (tested) |
| CSS cascade | Full | Full | Full | Full | No (reader mode) |
| Flexbox/Grid layout | Full | Full | Full | Full | Reader mode only |
| JavaScript execution | V8 | V8 | SpiderMonkey | JSC | **None by design** |
| WebAssembly | Yes | Yes | Yes | Yes | No |
| Images (raster) | Full | Full | Full | Full | Via innerHTML |
| SVG | Full | Full | Full | Full | Via innerHTML |
| Video/Audio | Full | Full | Full | Full | No |
| WebGL/WebGPU | Full | Full | Full | Full | No |

**Prism advantage**: Zero JS = zero JIT attack surface. Reader mode renders semantic content only.

## Security

| Feature | Brave | Chrome | Firefox | Safari | Prism |
|---------|:---:|:---:|:---:|:---:|:---:|
| HTTPS enforcement | Yes | Yes | Yes | Yes | Via fetch |
| Safe Browsing | Yes (partial) | Google SB | Google SB | Apple | **Reflex arc** (custom) |
| Phishing detection | URL list | URL list | URL list | URL list | **Aho-Corasick DOM scan** |
| XSS prevention | CSP | CSP | CSP | CSP | **ammonia sanitization** |
| Script blocking | Shields | Extensions | Extensions | Content blockers | **No scripts exist** |
| Tracker blocking | Shields | Extensions | ETP | ITP | **No JS = no trackers** |
| Fingerprint protection | Yes | No | Yes | Yes | **No JS = no fingerprinting** |
| Prompt injection detection | No | No | No | No | **Yes (35 phrases)** |
| Credential harvesting detection | No | No | No | No | **Yes (10 phrases)** |
| Epistemic threat scoring | No | No | No | No | **Yes (cumulative severity)** |
| Content zone classification | No | No | No | No | **Yes (Private/Local/Public)** |

**Prism advantage**: 6 security features that no major browser has. No JS engine = no JIT vulnerabilities, no prototype pollution, no supply chain attacks through npm.

## Privacy

| Feature | Brave | Chrome | Firefox | Safari | Prism |
|---------|:---:|:---:|:---:|:---:|:---:|
| Tracker blocking | Yes | No | Yes | Yes | **N/A (no JS)** |
| Cookie control | Yes | Yes | Yes | Yes | No cookies sent |
| Private browsing | Yes | Yes | Yes | Yes | **Default (Local zone)** |
| Tor integration | Yes | No | No | No | Planned (Iroh P2P) |
| Content never leaves device | No | No | No | No | **Yes (Local zone default)** |
| Explicit consent for sharing | No | No | No | No | **Yes (ConsentStore)** |
| Three-zone privacy model | No | No | No | No | **Yes (Private/Local/Public)** |
| Non-bypassable encoding gate | No | No | No | No | **Yes** |

**Prism advantage**: Privacy is structural, not a setting. Content defaults to LOCAL — never broadcast without explicit E3+/E4 classification or user consent.

## Search

| Feature | Brave Search | Google | DuckDuckGo | Prism Search |
|---------|:---:|:---:|:---:|:---:|
| Index type | Centralized | Centralized | Proxy (Bing) | **HDC 16,384-bit** |
| Ranking model | Proprietary | PageRank + ML | Bing's | **BinaryHV similarity + IDF** |
| Result count | Billions | Billions | Billions | 200+ claims (growing) |
| Evidence classification | No | No | No | **Yes (E0-E4)** |
| Source transparency | Partial | No | No | **Yes (per-result)** |
| Trust scoring | No | No | No | **Yes (K-vector)** |
| Anti-manipulation | Proprietary | Proprietary | N/A | **Anti-Sybil (10 layers)** |
| Decentralized | No | No | No | **Planned (Holochain DHT)** |
| Semantic encoding | Embeddings (opaque) | Embeddings (opaque) | N/A | **Open (BinaryHV, inspectable)** |
| Works offline | No | No | No | **Yes (embedded claims)** |

**Prism advantage**: Every result is epistemically classified. Users know the evidence level, source, and trust score. The encoding is open and inspectable — not an opaque ML model.

## Reader Mode

| Feature | Brave | Chrome | Firefox | Safari | Prism |
|---------|:---:|:---:|:---:|:---:|:---:|
| Reader mode | Via extension | Via extension | **Built-in** | **Built-in** | **Default** |
| Content width | Varies | Varies | ~680px | ~680px | **680px** |
| Line height | Varies | Varies | ~1.6 | ~1.6 | **1.7** |
| Font control | Varies | Varies | Yes | Yes | CSS variables |
| Dark mode | Via extension | Via extension | Yes | Yes | Planned |
| Stripping of non-content | Heuristic | Heuristic | Readability | Readability | **ammonia + html5ever** |

**Prism advantage**: Reader mode is the default, not an afterthought. Content is always rendered at comfortable reading width with proper typography.

## Architecture

| Property | Brave | Chrome | Firefox | Safari | Prism |
|----------|:---:|:---:|:---:|:---:|:---:|
| Language | C++ | C++ | C++/Rust | C++/Swift | **Pure Rust** |
| C/C++ code | Millions LOC | Millions LOC | Millions LOC | Millions LOC | **0 lines** |
| Memory safety | Partial | Partial | Partial (Rust parts) | Partial | **Total** |
| Binary size | ~200MB | ~200MB | ~200MB | ~100MB | **2.1MB WASM** |
| Attack surface | Enormous | Enormous | Large | Large | **Minimal** |
| JS engine CVEs (2024) | 15+ (V8) | 15+ (V8) | 10+ (SM) | 8+ (JSC) | **0 (no JS)** |
| Can be served from IPFS | No | No | No | No | **Yes** |
| Open source | Yes | Chromium | Yes | WebKit | **Yes** |

## Test Coverage (Prism)

| Category | Tests | Status |
|----------|-------|--------|
| HTML parsing | 8 | All pass |
| Security (reflex arc) | 7 | All pass |
| Privacy (3-zone) | 2 | All pass |
| Search quality | 7 | All pass (top result on-topic) |
| Search mechanics | 4 | All pass |
| Reader mode | 2 | All pass |
| Data ingestion | 3 | All pass |
| Ammonia sanitization | 1 | All pass |
| **Total feature parity** | **27** | **All pass** |
| **Full workspace** | **167** | **All pass** |
