# symthaea-evidence-verifier-diversity

Common-cause diversity assurance for verified safety evidence.

A set of receipts is not independent merely because the receipt ids or verifier references differ. This layer lets reviewed policy require diversity across explicit verifier fault domains such as:

- verifier identity
- organization
- review process
- verification toolchain
- underlying evidence source

The assessor is monotonic: it can retain or reduce ordinary strict-readiness status, but it cannot upgrade a blocked or invalid safety case.

A verifier profile is external reviewed evidence. Multiple receipt ids from one verifier do not create additional verifier diversity, and different verifier ids in one organization/process/toolchain/source do not manufacture diversity in those domains.

This crate grants no physical authority.

```bash
cargo test -p symthaea-evidence-verifier-diversity
```
