# Physical Storage Bootstrap Survivability — v0.1

**Programme:** PHYSICAL-STORAGE-BOOTSTRAP-SURVIVABILITY-001  
**Tracking:** [Symthaea #13774](https://github.com/Luminous-Dynamics/symthaea/issues/13774)  
**Spatial layout:** [Symthaea #13662](https://github.com/Luminous-Dynamics/symthaea/issues/13662)  
**Parameterized layout:** [Symthaea #13726](https://github.com/Luminous-Dynamics/symthaea/issues/13726)  
**Recovery capsule:** [Symtropy #7571](https://github.com/Luminous-Dynamics/symtropy/issues/7571)  
**Optical observations:** [Symtropy #7534](https://github.com/Luminous-Dynamics/symtropy/issues/7534)

## 1. Problem statement

The existing physical layout benchmark reserves control/spare tiles, but its objective evaluates payload-shard erasures only. This leaves a serious gap: all payload symbols may survive while the format descriptor, geometry, calibration, code parameters, index or decoder bootstrap is unavailable or stale.

An archive has two distinct but coupled information planes:

- **Payload plane:** user bytes, payload ECC, interleaving, reconstruction and exact digest.
- **Bootstrap plane:** format/version, code definition, physical map, coordinate registration, calibration references, index/address, known-answer vectors and portable reader/decoder contract.

A result is not a full archive recovery unless both planes qualify. Report their outcomes separately. Do not let payload success hide bootstrap failure or vice versa.

This v0.1 is a synthetic geometry/contract reference. It does not establish physical control-tile failure rates, calibration drift, optical performance or media lifetime.

## 2. Current layout anchor

The existing layout-search source uses:
- grid: 4 × 4 × 3;
- 48 physical tile positions;
- six reserved positions: [0, 15, 16, 31, 32, 47];
- 42 active payload tile positions;
- seven stripes × six shards;
- known-erasure budget: two shards per stripe.

The source code treats the six reserved positions as reserved but does not assign their semantic role. For this new synthetic fixture only, we declare positions [0, 15, 16, 31] as bootstrap controls and [32, 47] as spare positions. This is a fixture convention, not a claim about measured or qualified placement.

The existing payload layout's deterministic cuboid denominators are [27, 12, 18, 8, 3]. The row-major contiguous versus shard-major interleaved comparison reports 50/68 versus 8/68 unrecoverable masks across the combined synthetic corpus. Those are payload-shard geometry results only.

## 3. Bootstrap record model

A bootstrap object records:
- stable object ID and semantic type;
- exact format/version and canonical content digest;
- declared byte length;
- source/reference and provenance identity;
- physical location set and coordinate frame;
- encoding, redundancy and recovery threshold;
- calibration scope, reader identity, validity and applicability where relevant;
- external Recovery Capsule reference, when present;
- failure-domain references and unresolved common-cause dependencies.

Required semantic classes:
1. Format identity and schema version.
2. Code definition: family, generator matrix, field polynomial/basis, code rate, shard order, shortening/padding and parity placement.
3. Physical layout: grid dimensions, coordinate frame, logical-to-physical map and orientation.
4. Calibration: specimen/material/batch, reader/instrument, calibration reference, validity/applicability and raw-observation contract.
5. Index/address and object/stripe/shard directory.
6. Known-answer recovery vectors.
7. Portable reader bootstrap: source, build, dependencies and reader input contract.
8. Recovery Capsule identifier, digest and version.

The scorer's expected original bytes/digest remain outside candidate access and are not made authoritative merely by storing an in-media copy. Hash/fixity, authenticity, physical calibration validity and recovery correctness remain separate claims.

## 4. Qualification states

Each bootstrap object has one of:
- available_current;
- recovered_current;
- missing;
- corrupt;
- stale;
- wrong_but_valid;
- unknown;
- not_applicable.

The candidate cannot treat unknown as available. Stale or wrong_but_valid is not a valid recovery, even if the descriptor parses and its internal checksum is correct.

Overall bootstrap disposition:
- bootstrap_passed: all required objects are current and content digests/relations verify;
- bootstrap_recovered: required objects were reconstructed within declared bounds and verified against the external manifest;
- bootstrap_failed: invalid, contradictory, wrong-version or unrecoverable required objects;
- bootstrap_incomplete: required evidence or reference is unavailable;
- bootstrap_not_evaluated: no test was run.

Payload disposition remains separate. Full archive recovery requires exact payload plus bootstrap_passed or bootstrap_recovered.

## 5. Control-plane placement tournament

Compare on the same 4×4×3 geometry, same total reserved-volume budget, same payload code, same fault masks and the same external capsule availability:

1. centralized control block;
2. per-layer local control replicas;
3. spatially interleaved controls;
4. independently coded bootstrap stripes;
5. controls in separate physical planes/regions, only when the grid models them;
6. external Recovery Capsule with minimal in-media bootstrap stub;
7. hybrid external capsule and in-media control copies.

For every layout publish payload/control tile overhead separately, net payload tiles/bytes, bootstrap success by fault class, exact payload recovery conditional on bootstrap, full archive recovery, wrong-but-valid metadata false acceptance and common-cause exposure.

Do not compare a candidate with extra control volume against one with more payload volume. Do not collapse outcomes into one weighted score. Report a Pareto frontier across net payload density, payload recovery, bootstrap availability and false-acceptance constraints.

## 6. Required fault families

The frozen fixture includes:
- pristine bootstrap and payload;
- one control-tile erasure at each declared control position;
- all pairs of control tiles erased;
- whole control plane/layer dropout;
- calibration reference corruption;
- stale calibration for the same reader;
- calibration copied from a different reader/firmware;
- valid but wrong format descriptor;
- wrong GF polynomial or generator matrix;
- wrong shard order or layout mapping;
- index/address corruption and alias collision;
- known-answer vector loss;
- reader bootstrap unavailable;
- external Recovery Capsule unavailable in offline mode;
- payload retained but bootstrap unreadable;
- bootstrap readable but payload exceeds ECC budget;
- both planes readable but provenance/authenticity unresolved;
- common-cause site/power/reader failure across nominally distinct copies;
- unknown failure domain;
- wrong metadata digest and corrupted signature.

Fault schedules are deterministic synthetic test inputs. They are not estimates of physical probabilities.

## 7. Independent scoring and negative controls

The candidate receives public observations and permitted format information only. The scorer retains original bytes/digests and expected control relations separately.

Required negative controls:
- descriptor is syntactically valid and self-consistent but refers to a different layout;
- calibration record is cryptographically intact but stale or for another instrument;
- generator matrix uses another valid polynomial/format;
- coordinate transform shifts layers while preserving some relative geometry;
- two control replicas share a declared power/site dependency;
- unresolved dependency is silently interpreted as independent;
- capsule reference is missing or its digest differs;
- checksum-valid record has untrusted or revoked provenance.

Hash success alone does not establish authenticity or calibration truth. A signature alone does not prove physical correctness. Mycelix's existing dependency/cut-set authority remains canonical for operational independence.

## 8. Metrics

Report separate outcomes for:
- payload recovered exactly;
- bootstrap current/recovered/failed/incomplete;
- full archive recovery;
- false bootstrap acceptance;
- wrong-format/parameter/coordinate admissions;
- control erasure and corruption coverage;
- payload/control common-cause exposure;
- payload tile, control tile and spare overhead;
- recovery latency and abstention;
- missing external dependencies and unassessed failure domains.

Report every finite control case individually. Do not infer physical failure probability from a deterministic fixture.

## 9. Acceptance goals

Prospective finite-corpus targets:
- 100% bootstrap reconstruction for positive known-answer cases within declared redundancy bounds;
- zero false acceptance for wrong-format, stale calibration, descriptor substitution, address collision and unresolved provenance controls;
- zero full-archive pass when required bootstrap state is missing, stale, corrupt, wrong-but-valid or unknown;
- every payload receipt carries a separate bootstrap disposition, exact format/version and bootstrap digest;
- no payload recovery regression on the original payload-only corpus when no bootstrap faults are injected;
- equal total storage budget across control-layout comparisons;
- full disclosure of payload/control/spare overhead and unsupported fault classes.

These are software contract goals, not physical reliability thresholds.

## 10. Qualification and ownership

- Symthaea #13726/#13662 own payload layout search and synthetic geometry.
- Symthaea #13774 owns bootstrap placement/fault corpus and synthetic bootstrap scoring.
- Symtropy #7534 owns instrument-bound optical observations and calibration provenance.
- Symtropy #7571 owns Recovery Capsule package/representation information.
- Symthaea #13734 owns independent decoder/reader scoring.
- Mycelix #5762 remains canonical for operational/common-cause dependency closure.

The optimizer proposes; the independent scorer verifies bytes and bootstrap; the physical owner qualifies instrument/calibration inputs; Mycelix assesses common-cause claims.

## 11. Evidence ceiling

This protocol and synthetic fixtures demonstrate contract shape and finite fault semantics only. They do not qualify physical optical reference placement, reader calibration, media, retention, authenticity, manufacturing yield or independent readers. Real qualification requires registered specimen/readout maps, instrument calibration, independent reader trials and a preregistered physical campaign.
