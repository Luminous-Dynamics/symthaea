# symthaea-browser hardening campaign

These commits are intentionally sequential and independently reviewable:

1. **Policy firewall** — canonical URL parsing, local-network denial, finite
   policy validation, and typed decisions.
2. **Sensory boundary** — bounded observations, secret redaction, untrusted web
   content markers, navigation timeouts, and enforced element limits.
3. **Capability executor** — explicit authority independent of Phi, canonical
   action dispatch, action receipts, ambiguity failures, and safe JS argument
   serialization.
4. **Evidence truth** — search snippets are discovery-only, actual visited URLs
   are recorded, local sufficiency is quality-based, and only corroborated
   evidence may be promoted to Prism.
5. **HDC and telemetry truth** — bounded text projection, reduced allocation,
   redacted URL encoding, and separation of perceptual change from observation
   confidence.
6. **Example continuity** — the canonical local journey uses the executor and
   raw exploratory demos disclose that they bypass production enforcement.

## Verification status

`git diff --check` passes for every commit. This standalone archive does not
include the `symthaea-core` path dependency and the execution environment used
for this campaign has no Rust toolchain, so compilation and browser integration
must be run in the parent Symthaea workspace.

Recommended workspace gates:

```bash
cargo fmt --check -p symthaea-browser
cargo clippy -p symthaea-browser --all-targets --all-features -- -D warnings
cargo test -p symthaea-browser --all-targets
```
