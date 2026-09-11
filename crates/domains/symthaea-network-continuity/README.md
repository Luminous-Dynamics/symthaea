# symthaea-network-continuity

Provider-neutral evidence semantics for the network-device/fabric continuity program in #1100.

## Boundary

This crate does **not** configure devices, run Batfish/KNE/containerlab, qualify a fabric, or mint execution authority. It consumes normalized observations from `symthaea-support` and turns them into typed witness envelopes that downstream continuity verifiers can evaluate.

```text
support observations / packet evidence / telemetry
                    ↓
       NetworkContinuityWitnessV1
                    ↓
 evidence/currentness/identity admission
                    ↓
  downstream static/twin/hardware verifier
                    ↓
    future continuity qualification layer
                    ↓
 independent bounded execution authority
```

Core non-equivalences:

```text
observation != network truth
configuration accepted != behavior verified
witness admissible != requirement proven
requirement proven != fabric qualified
fabric qualified != execution authority
```

## V1 witness identity

A witness binds:

- canonical subject identity;
- requirement identity and class;
- support graph revision;
- exact observation IDs;
- evidence class;
- provider/version;
- verifier-profile digest where applicable;
- topology/config digest where applicable;
- distributed member-set digest where applicable;
- support/challenge/indeterminate disposition.

Default admission requires fresh evidence, exact live graph revision, an exact topology digest, and at least one observation whose canonical primary subject is the witness subject. Redundancy witnesses additionally require a member-set digest.

This prevents a packet/config/verifier artifact for one device or fabric from silently satisfying another merely because hostnames, addresses, or labels look similar.

## Evidence classes

V1 distinguishes normalized observation, static verifier, network twin, hardware lab, and production observation evidence. Those classes are not interchangeable. A future continuity policy may require several classes simultaneously.

## Bundles

`NetworkContinuityEvidenceBundleV1` is deterministic audit grouping for already-admissible witness digests. It is **not** a qualification token. A downstream qualification record must bind its own exact verifier profile, topology/currentness roots, participant set, contract/policy identity, and qualification generation.

## Planned adapters

Per #1100, likely future adapters include:

- Batfish static/differential verification;
- KNE/containerlab/Linux-namespace + FRR network twins;
- OpenConfig/gNMI observers;
- NETCONF/RESTCONF transaction adapters;
- gNOI lifecycle adapters;
- Redfish/BMC out-of-band evidence;
- vendor-native adapters only where standards do not expose required semantics.

Adapters should emit ordinary evidence. None may mint continuity or execution authority merely because an API request succeeded.
