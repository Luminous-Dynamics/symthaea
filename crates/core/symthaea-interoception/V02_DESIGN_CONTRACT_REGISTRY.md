# Affective Emergence v0.2 — Design Contract Registry

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract defines one canonical machine-readable registry for the normative v0.2 design documents. Its purpose is to prevent the `DesignFreezeManifest` and later evidence roots from silently omitting a newly added scientific contract.

## 1. Principle

A design freeze should not depend on a hand-maintained scattering of optional digest fields.

Instead, create one validated `DesignContractRegistryManifest` containing the complete ordered set of normative design-contract identities. The `DesignFreezeManifest` binds the registry digest as an authoritative closure over the design surface.

If a normative contract is added, removed, replaced, or reclassified, the registry digest changes and therefore the design-freeze identity changes.

## 2. Proposed registry schema

A future `DesignContractRegistryManifest` should contain at minimum:

- registry schema/version;
- exact v0.2 design source commit;
- ordered `contracts: Vec<DesignContractEntry>`;
- canonical registry SHA-256.

Each `DesignContractEntry` should bind:

- stable `contract_role` enum;
- repository-relative path;
- content SHA-256;
- contract schema/version or prose-contract version where applicable;
- normative status;
- optional supersedes/superseded-by identity;
- optional architecture-blocking flag.

Paths are useful for auditability but are not sufficient identity by themselves; the content digest is authoritative.

## 3. Contract roles

Initial normative roles should include at least:

- `ObservationalAffectPlan`;
- `InformationFirewall`;
- `TemporalAlignment`;
- `CandidateDefinition`;
- `CandidateFactorSpace`;
- `WeightingDecomposition`;
- `ChannelAggregation`;
- `AllostaticExposureDecomposition`;
- `IdentifiabilityAndDiscrimination`;
- `ExecutionMode`;
- `ScenarioManifest`;
- `BlindingCustody`;
- `CapabilityTypedApi`;
- `AdversarialValidation`;
- `EvidenceRoot`;
- `DesignFreeze`.

The registry-contract specification itself is bound separately by the design freeze as `design_contract_registry_contract_digest`, avoiding a self-hash cycle.

## 4. Normative status

Suggested status enum:

- `Normative` — implementation/evidence must conform;
- `Supporting` — explanatory material, not independently architecture-defining;
- `Superseded` — retained for history but cannot authorize new evidence;
- `Invalidated` — known unusable contract.

Only `Normative` entries satisfy required contract roles for a freeze.

A required role with zero or multiple active normative entries is a validation failure unless the registry schema explicitly declares that role multi-valued.

## 5. Closure rule

Before `FrozenBlockedOnV01` or `FrozenImplementationAuthorized`, registry validation must prove:

1. every required contract role exists exactly once as active normative content;
2. every listed path resolves to the exact recorded digest in the frozen source tree;
3. no active normative `V02_*` contract file is omitted from the registry;
4. no registry entry points outside the exact v0.2 design source commit;
5. supersession references are acyclic and resolvable;
6. no required normative contract is `Superseded` or `Invalidated`;
7. architecture-blocking contracts are explicitly marked and included;
8. canonical ordering is deterministic.

Rule 3 is important: merely checking that required known roles are present would still allow a newly added normative contract to be forgotten. The registry builder should audit the declared normative design directory/pattern and require explicit classification of every eligible contract.

## 6. Canonical ordering

Prefer deterministic ordering by stable `contract_role`, then path when a role is legitimately multi-valued.

Do not make filesystem enumeration order part of scientific identity.

The canonical registry must avoid unordered maps unless their serialization order is formally fixed.

## 7. Self-reference avoidance

Do not include the registry manifest's own final SHA-256 as an entry inside itself.

Use this dependency direction:

`V02_DESIGN_CONTRACT_REGISTRY.md content digest`
→ defines registry schema/validation
→ `DesignContractRegistryManifest` over all other active normative contracts
→ registry manifest SHA-256
→ `DesignFreezeManifest`.

The design freeze therefore binds both:

- registry-contract specification digest;
- realized registry-manifest digest.

## 8. Design freeze binding

`DesignFreezeManifest` should treat the contract registry as authoritative.

It may retain selected individual contract digests for human/audit convenience, but those fields must equal the corresponding entries in the registry. They cannot substitute for registry closure.

A freeze is invalid if:

- an individual digest disagrees with its registry entry;
- the registry omits a normative contract;
- the registry belongs to a different design source commit;
- the registry contract/version is not the one declared by the freeze.

The channel-aggregation and identifiability/discrimination roles are normative even if an older descriptive list inside the prose freeze has not enumerated them individually; authoritative membership is the validated registry, not a stale prose enumeration.

## 9. Evidence root binding

The prospective `ObservationalEvidenceRootManifest` should also bind the same exact design-contract-registry digest, directly or through a validated design freeze plus an equality assertion.

Recommended explicit redundancy:

- evidence root records `design_freeze_sha256`;
- evidence root records `design_contract_registry_sha256`;
- validator requires that the registry digest equals the one embedded in the referenced design freeze.

This lets a reproducer inspect the scientific design surface without reconstructing it indirectly.

## 10. Identifiability closure

A complete design registry is not enough if the locked scenarios cannot distinguish the locked candidates.

Before confirmatory freeze, require a valid `CandidateDiscriminationManifest` under `V02_IDENTIFIABILITY_AND_DISCRIMINATION.md` that binds the finite candidate set, scenario/cut-point set, required primary-vs-baseline pairwise obligations, equivalence tolerances, and discriminator coverage.

A candidate may not be promoted as superior to a baseline when the locked design lacks a discriminator capable of separating them.

## 11. Change severity

Registry changes inherit the severity of the contract change they represent.

- adding a purely supporting explanation may be Class I if it is explicitly non-normative and changes no normative registry entry;
- changing a candidate, weighting, channel aggregation, temporal aggregation, identifiability, scenario, analysis, or evidence contract is Class II;
- changing future-information authority, execution mode, feedback authority, or causal-output boundaries is Class III.

Adding a new active normative contract after freeze always changes the registry and supersedes the old freeze, even if the prose author considers the change small.

## 12. Review and CI gates

Future implementation should mechanically test:

- stable round-trip registry digest;
- every entry digest changes when content changes;
- missing required role rejected;
- duplicate active role rejected where single-valued;
- omitted eligible normative file rejected;
- path/digest mismatch rejected;
- source-commit mismatch rejected;
- supersession cycle rejected;
- freeze/registry digest mismatch rejected;
- evidence-root/freeze/registry mismatch rejected;
- missing aggregation contract rejected;
- missing identifiability/discrimination contract rejected;
- confirmatory freeze rejected when a required primary-vs-baseline pair lacks a registered discriminator.

A malicious fixture should add an unregistered normative-looking contract and prove the closure audit catches it.

## 13. Initial registry membership

At the current v0.2 design stage, the active normative set is intended to cover the roles represented by:

- `V02_OBSERVATIONAL_AFFECT_PLAN.md`
- `V02_INFORMATION_FIREWALL.md`
- `V02_TEMPORAL_ALIGNMENT.md`
- `V02_CANDIDATE_DEFINITION_CONTRACT.md`
- `V02_CANDIDATE_FACTOR_SPACE.md`
- `V02_WEIGHTING_DECOMPOSITION.md`
- `V02_CHANNEL_AGGREGATION_CONTRACT.md`
- `V02_ALLOSTATIC_EXPOSURE_DECOMPOSITION.md`
- `V02_IDENTIFIABILITY_AND_DISCRIMINATION.md`
- `V02_EXECUTION_MODE_CONTRACT.md`
- `V02_SCENARIO_MANIFEST.md`
- `V02_BLINDING_CUSTODY.md`
- `V02_CAPABILITY_TYPED_API.md`
- `V02_ADVERSARIAL_VALIDATION.md`
- `V02_EVIDENCE_ROOT.md`
- `V02_DESIGN_FREEZE.md`

This list is descriptive of the current design stage, not a permanently hard-coded schema. Future additions require explicit registry classification and a new registry/freeze identity.

## 14. Claim boundary

A closed contract registry proves only that the frozen evidence lineage names a complete, explicit set of design contracts under the declared closure rule. An identifiability manifest additionally establishes that the locked design contains declared discriminators for the comparisons it intends to make.

Neither proves those contracts are scientifically correct or that any regulatory candidate is affect, emotion, subjective valence, mood, sentience, or consciousness.
