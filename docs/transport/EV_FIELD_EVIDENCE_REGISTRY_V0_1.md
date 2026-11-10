# Rights-Aware Battery Field Evidence Registry v0.1

**Status:** Metadata/provenance and rights audit prototype. It does not download, host, harmonize or redistribute third-party raw battery data.
**Owner:** [EV-FIELD-EVIDENCE-001 #11421](https://github.com/Luminous-Dynamics/symthaea/issues/11421).

## Research basis

Masse, *Public Battery Field Data*, arXiv:2610.05811, first posted 5 October 2026: https://arxiv.org/abs/2610.05811. The paper's initial snapshot described 22 releases from 16 research groups. The living companion registry has since advanced to v1.2.0 (16 October 2026): 35 releases, 3,187 battery systems, 149,188 Li-ion cells, 6,465 faults (5,680 fault-labelled charging sessions), 115.6 MWh (35.84 stated + 79.76 computed), 3,040 unit-years and 331,704 observation-hours; 31,449 charging sessions are held separately from battery counts. The dashboard reports 256 GB of release data and does not host publisher raw files. These are registry snapshot totals, not a harmonized label set or a single independently comparable benchmark. Sources: https://github.com/Astrolabe-Analytics/public-battery-field-data (v1.2.0, 2026-10-16), https://astrolabe-analytics.github.io/public-battery-field-data/dashboard/.

Battery Data Commons is a living metadata registry, not a central raw-data host: https://batterydatacommons.org/ and Hassini et al., *Joule* (online 23 October 2026), https://doi.org/10.1016/j.joule.2026.102712.

The sources are not a ready-made harmonized SOH benchmark. Some releases have restrictive reuse terms, and missing/unlabelled time is not negative evidence.

## Paradigm: federated metadata, task-specific eligibility

This contract keeps source data with the publisher. Each registry entry binds:
- registry snapshot and exact dataset release/source checksum;
- publisher, citation/source URLs, retrieval timestamp and licence/terms reference + digest;
- independently evidenced permissions for access, local processing, derived-data publication, model training, weight release, commercial use and raw redistribution;
- intended uses, checked one-by-one against those permissions;
- application/chemistry, physical hierarchy, system/unit/fault counts and observation period;
- measurement vs receive-time availability, timezone semantics, unit/sentinel/missingness mappings;
- adapter/source-schema/mapping/test receipts;
- six task-specific dispositions: sensor integrity, fault detection, SOH, remaining life, intervention outcomes and pack capability.

Public download does not imply permission to train, publish derivatives, release model weights, use commercially or redistribute raw data. Conditional/unknown permissions do not satisfy an intended operation. Metadata-only entries can exist with no dataset processing claims.

## Hard semantic and rights gates

- Unlabelled periods must not be converted to negative fault labels.
- Sentinel codes must not be collapsed to valid numeric zero.
- Full SOH support requires cell hierarchy and an explicit reference target/label source.
- Fault-detection support requires labelled faults and event/receive timestamps.
- Remaining-life support requires a censoring-aware observation contract.
- Pack-capability support requires cell/module/pack hierarchy plus topology descriptor and qualification evidence.
- Partial or unvalidated adapters cannot claim fully supported tasks.
- Raw data are never hosted or copied into the registry; derived artifacts must retain their source rights lineage.

## Frozen synthetic corpus

Eighteen fixtures: six accepted metadata/rights/task profiles and twelve rejected controls. Accepted fixtures include permissive open data, controlled non-commercial research use, metadata-only catalogue entries and a partially validated adapter. Negative controls exercise unknown rights, commercial use prohibited by terms, raw redistribution prohibited, conditional rights without clearance, unlabelled periods treated as negative, sentinels collapsed to zero, SOH claimed without reference labels, fault detection claimed without timestamps, pack capability claimed without qualified topology, intervention outcome claims without completion/follow-up/matched comparison, unmapped sentinels with supported tasks, and unknown timezones with fault-detection claims.

These fixtures verify the schema/auditor only. No real field dataset has been imported or evaluated by this PR.

## Limits

The auditor checks internal consistency of producer-declared rights evidence and metadata. It does not provide legal advice, authenticate a publisher's licence, prove that data checksums match a source download, resolve legal conflicts among terms, or infer that an available label is valid. Rights decisions remain dataset- and jurisdiction-specific.