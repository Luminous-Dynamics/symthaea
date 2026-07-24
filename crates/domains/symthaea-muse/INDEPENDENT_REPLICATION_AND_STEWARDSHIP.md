# Independent Replication and Long-Term Stewardship

## Purpose

V13 governs what happens after a V12 confirmatory result is publicly released. It does not assume the original result is positive, and it does not treat one successful study as permanent evidence. It provides a prospective, independently executable path for direct replication, cross-site synthesis, archival recovery, revision control, and research-release stewardship.

## Claim boundary

V13 can establish that:

- independent sites received the same frozen public authority without original participant data, codebooks, or randomization secrets;
- each site preserved blinding and withheld the source outcomes until its own collection closed;
- each site executed the frozen analysis and an independent crosscheck;
- site conclusions and the cross-site conclusion were derived from the frozen practical margin;
- the public evidence was archived with file-level fixity and restored successfully;
- a stable **research release** received distributed stewardship.

V13 cannot establish that Symthaea improves music until real replication records satisfy the protocol. “Stable research release” does not mean a safety-certified product, autonomous artistic authority, or universal musical superiority.

## Required sequence

1. Publish and externally timestamp the immutable V12 final release.
2. Freeze `FrozenReplicationProtocol` before recruitment.
3. Register at least two independent organizations and their local governance evidence.
4. Issue one least-privilege `ReplicationSitePackage` per active site.
5. Advance the replication lifecycle to `CollectionOpen` only after all packages are receipted.
6. Keep source outcomes inaccessible until every prospective collection is irreversibly closed.
7. Seal one `ReplicationSiteExecutionRecord` per site.
8. Run the independent analysis and crosscheck before publishing the site result.
9. Synthesize all registered sites; do not omit inconvenient sites.
10. Publish null, mixed, descriptive-only, and failed-replication outcomes with the same release machinery as positive outcomes.
11. Deposit the complete public evidence at two or more independent archive providers.
12. Perform a restoration drill against the file-root commitment.
13. Freeze the stewardship charter and revision-governance policy.
14. Promote only evidence-supported research stages.
15. Build the V13 `StewardshipReleaseBundle` only after the lifecycle reaches `StewardshipReleased`.

## Direct replication constraints

A direct replication may not change the primary endpoint, policy arms, analysis plan, or blinding design. Local consent language or equivalent governance-required wording may be permitted only when declared in the frozen protocol. Any material post-freeze deviation demotes that site to `DescriptiveOnly`.

## Site independence

Each site must:

- belong to a distinct organization;
- declare conflicts of interest;
- be independent of the source authors;
- separate principal-investigator, data-custodian, and analyst identities;
- obtain local human-study governance approval;
- bind its execution environment and local protocol by digest.

The source authors may answer public technical questions, but they may not receive outcome-bearing data before site closure or privately repair a site analysis after seeing its result.

## Cross-site conclusion

V13 uses a random-effects inverse-variance synthesis and reports:

- fixed and random pooled estimates;
- confidence interval;
- Cochran’s Q;
- between-site variance;
- I²;
- attenuation relative to the published source estimate;
- direction concordance.

`IndependentlyReplicated` requires the pooled confidence interval to clear the frozen practical margin and at least two sites to independently support replication. A material site deviation makes the synthesis descriptive-only. Heterogeneous or threshold-crossing evidence is `MixedEvidence`, not a success.

## Revision governance

Published evidence is immutable. Documentation, defect, model, protocol, and outcome-definition changes create a new lineage node. Claim-relevant revisions require role-specific independent review and evidence. Protocol or endpoint changes require a new preregistration and confirmation; they cannot inherit the old claim automatically.

## Archival and continuity requirements

The public archive must contain source, environment locks, final release, replication authorities, site executions, synthesis, analysis code, independent verifier, documentation, and license. It must exclude randomization secrets and raw personal data. At least two independent providers must hold the same file-root object, and an independent recovery drill must reconstruct that root.

## Stewardship requirements

At least three people across two organizations must collectively cover release maintenance, reproducibility, archiving, security, participant protection, and independent methods review. No person may hold more than two critical operational roles. Succession, vulnerability disclosure, evidence correction, end-of-life, and funding-conflict policies are mandatory.

## Operational commands

The `cognitive_study` binary exposes V13 commands for protocol sealing, site registration, package issue, site execution, synthesis, lifecycle transitions, revision governance, archive sealing, promotion, and root release construction. Run the independent verifier against the final directory:

```sh
python3 scripts/verify_cognition_study_v13.py /path/to/v13-release
```

The verifier recomputes JSON commitments, site conclusions, the cross-site meta-analysis, archive file root, package/execution roots, lifecycle chain, and final release bindings without calling Rust.

## Negative and mixed outcomes

A V13 release remains valid when replication fails. The correct response is to publish `DidNotReplicate`, `MixedEvidence`, `DescriptiveOnly`, or `InsufficientEvidence`, preserve every site, and revise claims accordingly. The machinery must never be used to search repeatedly for a favorable subset of sites.
