# Human Agency Kernel Audit v1 — Scope

**Status:** documentation-only architecture tranche.

**Base:** `main@2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`

## In scope

HAK-001 audits the common and non-common semantics required for human-centered augmentation across Symthaea and Mycelix.

Specifically in scope:

- baseline human standing versus scoped role qualification;
- recommendation, consent, delegation, execution, outcome, and human-benefit separation;
- bounded human-to-machine delegation semantics;
- contestability, withdrawal, appeal, and reciprocal accountability;
- immediate assisted performance versus delayed independent capability;
- model calibration versus human reliance calibration;
- independent elicitation, dissent preservation, and collective-intelligence diversity;
- multidimensional agency/capability measurement without a universal scalar;
- preregistration of confirmatory human-benefit claims;
- advisory institutional linting;
- identifying current Symthaea/Mycelix surfaces that should inform later HAK work;
- identifying governance-profile semantics that require independent review before wider authority use;
- sequencing later HAK/Mycelix implementation tranches.

## Explicitly out of scope

HAK-001 does **not**:

- create a new Rust crate;
- change `Cargo.toml`, `Cargo.lock`, workflows, feature flags, or runtime source;
- change Mycelix voting, governance tiers, constitutional rules, or Holochain DNA;
- change or deprecate the sovereign-profile dependency;
- change SubPassport serialization/signature semantics;
- create a new consent protocol;
- modify Hearth youth/guardian autonomy;
- implement an Institutional Linter;
- implement an Intent Graph;
- implement a human-user model or psychological profile;
- collect, infer, or persist personal behavioral data;
- define a universal flourishing score;
- claim legal, medical, psychological, or regulatory compliance;
- establish that a specific governance system is morally or democratically legitimate;
- establish AI personhood, sentience, consciousness, or moral-patient status;
- transfer scientific evidence into governance or action authority;
- conduct real-participant research;
- claim that any HAK intervention improves human outcomes.

## Repositories and surfaces referenced

HAK-001 may refer to current semantics in:

### Symthaea

- psych-bench metacognitive calibration;
- subterranean rescue-consent continuity;
- epistemic/evidence authority separation;
- scientific-method audit patterns;
- Sovereignty Papers as normative/governance design material.

### Mycelix

- reciprocal-accountability draft work;
- bridge-common SubPassport delegation;
- sovereign-profile civic gating;
- governance and anti-tyranny design;
- Hearth graduated autonomy.

Reference does not imply qualification or endorsement.

## Cross-PR relationship

HAK-001 is intentionally independent of SCI-001.

SCI-001 asks how scientific artifacts, observations, claims, execution evidence, replication, and dispositions should be separated.

HAK-001 asks how assistance, assessment, consent, delegation, legitimacy, action, and human outcomes should be separated.

They share architectural discipline but not authority or qualification.

```text
SCI-001 PASS != HAK-001 PASS
HAK-001 PASS != SCI-001 PASS
```

Likewise, no referenced Mycelix draft PR transfers qualification into HAK-001.

## Change boundary

HAK-001 should remain a small closed documentation set under:

```text
docs/architecture/HUMAN_AGENCY_KERNEL_AUDIT_V1*.md
```

Any product-code, manifest, workflow, governance-policy, cryptographic-transcript, or runtime change belongs in a separately reviewed tranche.

## Review boundary

Do not review this tranche as a referendum on one-person-one-vote, epistocracy, liquid democracy, sortition, expert panels, or the current sovereign-profile model.

Review whether the audit preserves enough semantic separation that later governance experiments can compare those mechanisms **without first assuming that competence, contribution, consensus similarity, reputation, or machine inference is equivalent to legitimate human standing**.
