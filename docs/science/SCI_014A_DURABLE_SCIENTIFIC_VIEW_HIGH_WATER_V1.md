# SCI-014A — Durable Scientific-View High-Water and External Witness Reuse

Status: architecture draft

Related: #808, #886, #908, #1946, #2092, #3143, #3148, #3178, #3182, #3189, #3191, #3201, #3205, #3238, #3279

## Purpose

SCI-014 requires present scientific currentness to survive process restart, local-store rollback, source reprovisioning, and fresh-machine recovery without creating a parallel science-specific persistence or transparency system.

This tranche therefore defines the science-owned semantics that sit above the repository's existing anti-rollback/high-water work.

The core rule is:

```text
scientific domain semantics
    own which authority coordinates are material

shared anti-rollback substrate
    owns durable high-water / external-witness mechanics
```

SCI-014 must not introduce a separate Rekor client, independent transparency protocol, or second crash-durability model.

## Governing non-equivalences

```text
valid historical scientific state
    != durable local current state

crash-consistent local state
    != fresh-machine rollback resistance

external inclusion
    != split-view-resistant consistency

split-view resistance
    != point-of-use currentness

point-of-use scientific currentness
    != scientific support
    != proposition truth
    != recommendation
    != action authority
```

## Scientific authority high-water is a vector

A scientific view may depend on several independently advancing mutable authorities. Do not reduce them to one global scalar generation.

Conceptually:

```text
ScientificAuthorityHighWaterV1
    namespace
    scientific_view_profile
    source_roster_commitment
    canonical authority-coordinate vector
    selected scientific-view head
    effective selection decision
    predecessor high-water checkpoint
    checkpoint commitment
```

The exact authority roles come from the committed `ScientificViewProfileV1`, not from caller-supplied runtime lists.

Possible roles include, where scientifically material for that profile:

- research-semantic lineage head;
- evidence/admission lifecycle head;
- subject-selection head;
- verifier-policy head;
- scientific-view selection-decision lifecycle head;
- discovery/completeness head.

Each coordinate must bind:

```text
role
source identity
source provisioning epoch / occurrence identity
lineage position
state commitment
```

The source occurrence is authority-bearing. Equal state bytes under a new source occurrence do not preserve currentness automatically.

## Profile and source-roster anti-rollback

The high-water checkpoint must commit both:

```text
ScientificViewProfileId
source_roster_commitment
```

Otherwise a rollback could restore an older profile that simply omits a newly material authority source.

Therefore:

```text
rollback authority values
    != rollback definition of which authority values matter
```

Both must fail closed.

Adding a scientifically material authority role or discovery surface requires a versioned view-profile transition. Historical views under the prior profile remain historically valid; they do not satisfy a current-use policy requiring the newer profile.

## Partial-order freshness classification

Generalize the provider-neutral high-water comparison semantics already developed by EUREKA.

For one exact view profile and source roster, classify a local scientific authority vector relative to the accepted external high-water as:

```text
ExactWitnessed
RollbackDetected
UnwitnessedAhead
EquivocationDetected
Incomparable
```

### ExactWitnessed

Every coordinate position and bound commitment exactly matches the accepted witness state.

### RollbackDetected

Every comparable coordinate is less than or equal to the witnessed coordinate and at least one is lower, with no coordinate ahead.

An authentic historical state can therefore be a rollback.

### UnwitnessedAhead

Every comparable coordinate is greater than or equal to the witnessed coordinate and at least one is ahead, and the local state supplies the exact valid continuation proof required by the high-water profile.

This is not rollback. It means the local state may have legitimately advanced but the configured external-freshness theorem has not caught up.

### EquivocationDetected

The same coordinate position or sequence vector binds a different authority/state commitment.

No timestamp, majority, or "latest file" heuristic selects a winner.

### Incomparable

Examples include:

- one authority coordinate ahead while another is behind;
- wrong genesis/provisioning anchor;
- different scientific-view profile or source roster;
- broken predecessor checkpoint;
- unsupported skipped/multi-step transition;
- crossed source epochs without a qualified migration theorem.

No automatic branch preference is allowed.

## Currentness is not one scalar strength

External witnessing introduces orthogonal assurance dimensions. Do not collapse them into `CurrentnessLevel::Strong`.

A currentness/use policy should state requirements across at least:

```text
lineage strength
local durability
external retention
split-view resistance
point-of-use revalidation
```

Examples:

```text
HistoricalInspection
    immutable qualified historical view

HistoricalReanalysis
    immutable qualified historical view

ExploratoryCurrentAssessment
    may permit locally durable currentness

ConfirmatoryResultClosure
    requires configured durable/external freshness
    + fresh point-of-use currentness

HighAssurancePublicCurrentAssessment
    may additionally require a configured
    split-view-resistant external consistency theorem
```

The requested-use policy itself is semantically committed. Changing the required assurance dimensions changes the scientific-use theorem rather than silently strengthening an old result.

## External publication is authority metadata, not science payload

The external witness layer should retain enough information for independent high-water replay without publishing underlying experimental artifacts.

Prefer commitments to:

- view namespace/profile;
- source-roster identity;
- canonical authority vector;
- selected scientific-view head;
- effective selection decision;
- predecessor high-water checkpoint;
- scientific high-water checkpoint.

Do not require external publication of experimental data, unreleased results, private evidence, or other underlying scientific payloads merely for anti-rollback protection.

## Reuse boundary with EUREKA

The repository already contains a staged anti-rollback spine:

```text
#3143  canonical high-water mechanics
#3148  crash-conscious durable checkpoint persistence
#3178  governance-bound transition mechanics
#3182  durable authorized-transition binding
#3189  ambiguity-safe retained-lock reconciliation
#3201  external high-water witness mechanics
#3205  external publication / inclusion architecture
#3238  provider-trust admission and anti-rollback architecture
```

SCI-014 should consume or generalize those semantics, not duplicate them.

Extraction rule:

```text
first qualified consumer: EUREKA
second real consumer: SCI-014
then extract the exact shared waist if both consumers prove it
```

The candidate shared waist is limited to domain-neutral mechanics such as:

```text
canonical high-water statement
crash-conscious durable checkpoint
external publication envelope
external inclusion verification
provider-trust qualification
optional consistency/witness qualification
```

Science retains ownership of the authority vector, source roster, view semantics, and requested-use policy.

## External-provider trust is another authority lineage

A cryptographically valid transparency-log checkpoint does not prove that the provider identity is currently trusted for scientific freshness.

SCI-014 inherits the provider-trust boundary:

```text
valid provider signature
    != admitted provider

historically admitted provider
    != currently admitted provider

provider URL equality
    != provider identity
```

Provider-trust currentness terminates at an explicit deployment/provisioning root. It must not be recursively derived from scientific evidence.

Provider rotation/sharding changes provider occurrence identity. Historical provider identities remain useful for historical verification without automatically remaining current for new publications.

## Source replacement and migration

Source replacement is occurrence-bearing even when content is identical.

```text
same state commitment
+ new source provisioning epoch
    != same authority occurrence
```

A source migration may preserve semantics only through an explicit predecessor-bound migration theorem.

A migrated source should establish a transition such as:

```text
old source epoch final occurrence
    -> qualified migration
    -> new source epoch genesis occurrence
```

Old source handles and runtime capabilities cannot cross the migration boundary merely because identifiers or state bytes match.

## Fresh-machine recovery

A restored or fresh Symthaea process must not deserialize present scientific authority.

Conceptually:

```text
load local durable scientific high-water
    ↓
resolve newest accepted external high-water
under qualified provider trust
    ↓
classify local vector
    ↓
ExactWitnessed
| RollbackDetected
| UnwitnessedAhead
| EquivocationDetected
| Incomparable
    ↓
apply exact requested-use policy
    ↓
fresh coherent scientific-authority capture
    ↓
cross-object closure
    ↓
PRE current-use qualification
    ↓
scientific operation
    ↓
POST exact authority-envelope recheck
```

No serialized checkpoint, witness, baseline, or report recreates a live `CurrentScientificAuthorityEnvelope`.

## Availability rule

Do not synchronously require external publication before every valid local transition can become durably recorded.

Preferred progression:

```text
valid scientific-authority transition
    ↓
durable local commit
    ↓
UnwitnessedAhead
    ↓
external publication / acceptance
    ↓
ExactWitnessed
```

This prevents third-party transparency outages from freezing ordinary local science.

Stronger requested-use policies may still fail closed while the state remains `UnwitnessedAhead`.

## Point-of-use still requires fresh currentness

Even `ExactWitnessed` is historical/bounded evidence about a high-water state. It is not a bearer token for present scientific use.

The live path remains:

```text
accepted high-water state
    + fresh qualified authority-source observation
    + coherent capture
    + cross-object closure
    + requested-use policy
    -> ephemeral CurrentScientificAuthorityEnvelope<'a>
```

The envelope remains owner-local, private-construction, non-Serde, and operation-scoped.

For pure current-use computation:

```text
PRE exact authority envelope
-> compute
-> POST exact authority envelope
-> release owned result only if unchanged
```

Do not automatically retry callbacks after `PostUseAuthorityChanged`.

## Effectful operations remain separate

High-water/current-use qualification does not make persistence or publication effects transactionally safe.

Effectful operations continue to require:

```text
fresh authority envelope
-> candidate effect
-> final fresh authority recheck
-> deterministic operation ID
-> durable CAS/effect
-> CommittedExact | ProvenNotCommitted | OutcomeUnknown
-> reconcile uncertain outcomes before any retry
```

## Adversarial qualification corpus

A future implementation should prove at least:

1. an older authentic local scientific checkpoint is `RollbackDetected` against a newer accepted external high-water;
2. one exact valid successor ahead of the external witness is `UnwitnessedAhead` rather than rollback;
3. one coordinate ahead and another behind is `Incomparable`;
4. same coordinate position with a different state commitment is explicit equivocation;
5. identical state bytes under a new source provisioning epoch are a distinct authority occurrence;
6. rollback to an older view profile/source roster cannot omit a now-material authority role for a policy requiring the newer profile;
7. external inclusion without the required consistency theorem cannot satisfy a split-view-resistant policy;
8. historical provider keys/configurations remain historical evidence but cannot silently satisfy current provider admission after provider-trust rotation;
9. process restart cannot deserialize/reconstruct a live current-use capability;
10. external-witness outage may permit configured weaker local use while blocking configured high-assurance public/confirmatory use;
11. `ExactWitnessed` still requires fresh point-of-use coherent capture and PRE/POST revalidation;
12. no checkpoint, witness, publication receipt, or currentness capability grants proposition truth or action authority.

## Integration order

Recommended SCI-014 path:

```text
#1946 research-semantic event lineage
-> historical evidence admission + exact execution occurrence
-> ScientificViewProfile + exact mutable-role roster
-> sealed ScientificAuthorityOwner + admitted source occurrences
-> coherent scientific-view capture + cross-root closure
-> exact-predecessor view DAG / fork handling / selection lifecycle
-> local scientific high-water checkpoint + durable anti-rollback
-> external high-water reuse + assurance-dimension policy
-> CurrentScientificAuthorityEnvelope<'a>
-> EvidenceEligibleWithinView / ASSURE
-> two-phase ResultClosing -> ResultClosed
-> LL-009AJ / Site01 first end-to-end consumer
```

The historical/coherent-view implementation must not be blocked on eventual production qualification of every external-witness provider. Stronger currentness profiles may depend on those later tranches.

## Non-claims

This architecture does not claim:

- current EUREKA draft provider integrations are executable-qualified;
- global scientific truth or consensus;
- global evidence completeness;
- automatic resolution of equivocation/forks;
- scientific support merely from currentness;
- deployment, governance, medical, resource, or physical-effect authority.
