# symthaea-evidence-quarantine

Cross-cutting quarantine for safety evidence defects that are broader than one receipt or one deployed configuration.

Examples include:

- a verifier/process later found compromised,
- a corrupted evidence artifact reused across deployments,
- a methodology flaw affecting every receipt for one obligation,
- a specific receipt discovered to be suspect.

A quarantine directive can match evidence by:

- receipt id,
- evidence content digest,
- verifier reference,
- evidence object reference,
- stable obligation key.

Quarantine directives are reviewed evidence objects with an effective time and reason reference. A resolution can either:

- **lift after review** — existing matching receipts may become usable again, or
- **require replacement** — receipts verified on or before the resolution remain quarantined; only newly verified replacement evidence can be used afterward.

This is deliberately distinct from a lifecycle contradiction. A contradiction invalidates that receipt permanently and requires replacement even after the contradiction is resolved. Quarantine is an administrative/evidence-integrity hold that can have a reviewed disposition.

Future-dated, malformed, duplicated, or inconsistent directives/resolutions fail closed.

The quarantine layer never grants physical authority.

```bash
cargo test -p symthaea-evidence-quarantine
```
