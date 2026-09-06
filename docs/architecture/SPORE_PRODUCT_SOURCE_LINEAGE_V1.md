# Spore Product Source Lineage v1

Status: **pre-extraction source ownership audit**

This document identifies the exact implementation that the reviewed `nixos-config` host lineage actually consumes for Spore recovery. It supplements the host-integration provenance in #504 and the behavioral parity corpus in #512.

It is provenance and architecture evidence only. It does not create a destination repository, move product code, or transfer historical qualification.

## 1. The actual host-consumed recovery source

At host commit:

`Tristan-Stoltz-ERC/nixos-config@5d80360768ee329c50756e71fbce4692ac3a8e45`

`modules/system/symthaea-boot.nix` imports:

`inputs.symthaea.outPath + "/nix/packages/spore-boot-tools.nix"`

The corresponding `flake.lock` input named `symthaea` is not the current `Luminous-Dynamics/symthaea` repository. It is pinned with `flake = false` to:

`Luminous-Dynamics/luminous-dynamics@4fe8b1e2ca5fb60463de16c0b9ec649e1fc059a2`

with NAR hash:

`sha256-G/SffFTFiEyLmQ4Y9O6vEhrEM0kLLi/7lwQ6b22/1JA=`

That exact commit is also the head of historical branch:

`spore/recovery-v0.3.4f-qualified`

and has tree:

`d53df873dd204738fe669066ca9aa22db88b22c1`

Therefore the extraction source model has three separate roles:

```text
nixos-config
    = host integration + host qualification experiments

luminous-dynamics@4fe8b1e2...
    = actual host-consumed recovery + mixed boot package source

symthaea@937b2247...
    = current presentation / clean-host rehydration reference lineage
```

These roles MUST NOT be collapsed into one implied source repository.

## 2. Mixed package boundary

The pinned `nix/packages/spore-boot-tools.nix` builds three binaries from one package:

- `quicken-fb` from `symthaea-quicken-fb` — presentation;
- `spore-boot-state` from `symthaea-boot-state` — state/lifecycle tooling;
- `spore-recovery-linux` from `symthaea-boot-state` — Linux recovery coordinator.

This package is therefore a **mixed ownership artifact**. It MUST NOT be copied wholesale into canonical Spore as though all three binaries have recovery authority.

The destination steady state should have independently buildable packages:

```text
Spore recovery package
    -> recovery protocol/state/planner/executor/Linux adapter/CLI

Symthaea presentation package
    -> quicken-fb / Boot Ecology rendering
```

The host may compose both, but recovery must not require presentation to build or execute.

## 3. Mixed crate boundary

The current `symthaea-boot-state` crate is recovery-adjacent, but it directly depends on `symthaea-boot-ecology`.

`boot-ecology` itself combines two categories:

1. factual/transportable boot types such as `BootStateReceipt`, `PreviousTermination`, `GenerationTransition`, `GenerationHealth`, and `StorageState`;
2. presentation semantics such as `MorphologyLineage`, morphology families/parameters, `BootGenome`, and `BootEcologyComposer`.

Extraction therefore requires a semantic split, not a crate rename.

A useful target shape is conceptually:

```text
spore-protocol
    factual boot/recovery evidence types only

spore-recovery
    persistent state + planner + executor + Linux adapter

symthaea boot presentation
    morphology lineage + composer + renderer
```

Names are not frozen here. The ownership rule is.

## 4. Authority finding: semantic LKG vs morphology mirror

The current `PrepareInput` contract says:

- host semantic Last Known Good is recovery authority;
- `MorphologyLineage.last_known_good_generation` is a presentation mirror only.

`LinuxRecoveryCoordinator::prepare_boot()` correctly reads the semantic recovery roots and writes `roots.last_known_good` into `PrepareInput.authoritative_last_known_good_generation`.

But `BootStateStore::prepare()` currently derives both `generation_transition` and `generation_health` from `lineage.last_known_good_generation`.

This means a stale/divergent presentation mirror can change the factual receipt's rollback/known-good classification.

Tracking issue:

`Luminous-Dynamics/luminous-dynamics#51`

The current qualification planner independently uses boot identity, semantic roots, and observed health and does not directly use those receipt fields to decide LKG promotion, so this audit does **not** relabel the finding as a demonstrated false-promotion exploit. It is still an evidence-truth and authority-boundary defect that must not be copied into the destination recovery core.

## 5. Authority finding: presentation can veto preparation and LKG promotion

The presentation dependency begins before qualification.

`BootStateStore::prepare()` currently loads persistent `MorphologyLineage`, loads or creates a machine visual seed, constructs a mixed factual/presentation `BootStateReceipt`, and journals runtime receipt + runtime lineage + persistent boot state together. Corrupt morphology state or failure of presentation-seed/state handling can therefore prevent factual preparation from completing and leave no recovery qualification artifact.

The host integration remains fail-open for ordinary boot, but qualification availability is still presentation-dependent.

The current qualification artifact is also weakly subject-bound: `LinuxRecoveryCoordinator::qualify_last_known_good()` reduces the runtime receipt and lineage to `Missing | Present | Invalid`, while `BootObservation.boot_id` is not bound to the parsed receipt and `BootStateReceipt` contains no exact boot ID / physical-generation subject. The reviewed host shell mitigates stale runtime evidence by scrubbing `/run`, but generic recovery correctness should not rely on shell cleanup as the primary subject-binding mechanism.

The dependency then continues through promotion. `recovery::plan_qualification()` requires `MorphologyLineage` to be present and parseable, and the promotion sequence is:

```text
StageLastKnownGood
    -> BlessState
    -> CommitLastKnownGood
```

`LinuxRecoveryPersistence::apply(BlessState)` calls `BootStateStore::bless()`. That method currently:

1. requires the pre-bless runtime morphology lineage;
2. runs `BootEcologyComposer::compose()`;
3. uses visual `genome.seed` plus morphology state as replay/idempotence identity;
4. can reject divergent persistent/runtime morphology history;
5. persists a morphology outcome before authoritative bless completion.

Therefore missing, corrupt, divergent, or unwritable presentation/history state can veto semantic recovery preparation or LKG commitment even when the physical boot identity and local health evidence are otherwise valid.

There is also a separate per-boot truth defect inside the same authority area. Every new preparation resets `last_boot_blessed = false`, but a healthy boot whose generation is already the semantic LKG returns `Preserve(AlreadyKnownGood)` with no bless operation. The exact current boot therefore remains unblessed even after passing the qualification health gate, and a later update can incorrectly describe that prior healthy boot as incomplete. Semantic generation identity and exact current-boot qualification are different facts and must remain different facts after extraction.

Tracking issue:

`Luminous-Dynamics/luminous-dynamics#56`

This audit does **not** claim presentation can falsely promote an arbitrary generation. The demonstrated defect is a **presentation -> recovery veto authority edge plus incomplete exact-subject and per-boot qualification binding**.

The repaired direction must be:

```text
physical boot identity
    -> recovery-native factual preparation subject
    + semantic recovery roots
    + local health
        -> exact-subject qualification
        -> exact-subject bless of this healthy prepared boot
        -> semantic LKG commit when generation promotion is needed

semantic recovery outcome
        -> downstream presentation/history mirror
```

Never the reverse.

The existing preparation journal already provides a suitable recovery-native identity basis: exact kernel `boot_id`, exact generation, and the durably paired boot counter. Wall clock and visual genome identity are not needed as recovery authority.

A healthy `AlreadyKnownGood` path still needs the exact-subject bless even though it does not need to rewrite the semantic LKG root. That preserves per-boot truth without conflating boot qualification with generation promotion.

## 6. Pre-extraction repair gates

Before destination parity can be called architecture-preserving rather than merely bug-preserving:

1. **REPAIR-001 — Semantic LKG wins.** Receipt rollback/known-good classification derives from the semantic recovery root input, never the morphology mirror.
2. **REPAIR-002 — Presentation cannot veto recovery.** Remove morphology/presentation state from factual recovery preparation, qualification, blessing, and LKG-commit authority. Preparation and replay use a recovery-native exact prepared-boot identity independent of `BootEcologyComposer`, `BootGenome`, `genome.seed`, visual seed availability, and morphology history. Every healthy prepared boot, including `AlreadyKnownGood`, receives an exact-subject bless so per-boot qualification truth remains correct. Presentation mirroring occurs only downstream of committed recovery truth.
3. **REPAIR-003 — Protocol/presentation split.** Factual recovery protocol types can build without morphology/composer code.
4. **REPAIR-004 — Package split.** Recovery binaries can build/qualify without `quicken-fb`.

These are targeted authority repairs, not permission for broad redesign during migration.

## 7. Evidence rule

The historical `spore/recovery-v0.3.4f-qualified` branch remains frozen evidence.

Do not repair it in place.

Any correction should create a child lineage with:

```text
exact historical source
        ↓
explicit authority repair
        ↓
exact committed repaired bytes
        ↓
fresh tests / parity evidence
        ↓
fresh qualification
```

A successful source-side repair still does not qualify the future `Luminous-Dynamics/spore` destination. Destination parity and destination product qualification remain separate evidence tiers.

## 8. Extraction ownership theorem

The destination boundary should satisfy all of the following simultaneously:

```text
Host policy != recovery implementation
Presentation state != recovery truth
Presentation availability != factual preparation availability
Presentation validity != qualification eligibility
Presentation persistence != LKG commit availability
Semantic LKG generation != exact current-boot qualification
Visual replay identity != recovery transaction identity
Parseable evidence != current exact-subject evidence
Source provenance != destination parity
Destination parity != product qualification
```

If any migration step collapses one of those distinctions, extraction has made Spore less trustworthy even if the resulting code still boots.
