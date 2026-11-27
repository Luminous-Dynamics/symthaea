# CMAQ-028 — Recovery-Option Value and Resilient Portfolios

**Status:** dependency-free Rust evaluator and deterministic tests authored; exact-head formatting, tests and strict Clippy pending.  
**Relationship:** follows CMAQ-027; adds redundancy and option value across multiple recovery modes.  
**Claim ceiling:** finite failure-domain portfolio accounting. No proof of physical independence, authenticated certificate validity, or fault-tolerant safety guarantee.

## Research idea

CMAQ-027 establishes whether one trusted recovery mode can recover from a predicted commitment prefix. But a single recovery controller, sensor path, actuator dependency, compute resource or proof authority may itself fail.

CMAQ-028 treats recoverability as a **portfolio of options**. At every candidate execution prefix, the system must retain at least (k) independently scoped recovery modes, where independence is based on externally attested failure domains rather than different profile IDs.

For single-domain-failure tolerance, the initial deterministic witness uses (k=2). Real deployments must set (k) from a fault model and evidence about common-cause dependencies.

## Portfolio contract

Each recovery option binds:
- profile, plant, context, controller and external certificate-receipt digests;
- a failure-domain digest;
- an entry radius;
- maximum high-performance dwell;
- expected recovery steps and energy cost.

Each candidate prefix provides a context digest and reachable-error upper bound. An option covers a prefix only if plant/context scope matches, its entry radius contains the full bound, and the prefix is before its dwell deadline.

Profiles sharing a failure-domain digest count as one option. Within a failure domain, the evaluator retains the fastest/lowest-energy profile as the representative. Across independent domains, it reports the portfolio frontier ordered by recovery steps, energy, and profile identity.

The evaluator requires at least (k) distinct covered failure domains at every committed prefix. It stops before the first prefix where that minimum is lost and recommends switching while the earlier options still exist.

## Deterministic witnesses

1. One profile is admissible for (k=1), but cannot satisfy (k=2).
2. Two independent domains satisfy a single-domain-failure target.
3. Two profile IDs in one common-cause domain count once.
4. Context mismatch removes that option rather than averaging its evidence.
5. A later reachable prefix can lose one option; commitment stops before the redundancy floor is crossed.
6. A slower independent recovery option remains valuable beside a fast profile with a shared failure domain.
7. Losing every recovery option at the next prefix forces an immediate switch.
8. Unknown prior execution status blocks new portfolio decisions.
9. Malformed receipts, duplicate profile IDs, and non-contiguous prefixes are rejected.

## Metrics

- longest high-performance commitment prefix retaining at least (k) distinct domains;
- minimum option count over all committed prefixes;
- option count remaining after one declared domain failure;
- recovery time/energy frontier among distinct domains;
- option-retention and option-loss events;
- earliest prefix requiring handoff;
- worst-context coverage and independent scenario-cluster statistics.

The evaluator does not infer true statistical independence from distinct strings. Failure-domain identity must be independently justified by dependency analysis, hardware topology, controller diversity, supply-chain provenance, and proof-authority separation. A duplicated controller image with two profile IDs is not two independent options.

## Matched baselines

- single-profile CMAQ-027;
- first-admissible recovery profile;
- fastest profile only;
- largest entry radius only;
- naive profile count without domain deduplication (negative control);
- portfolio selection with independent-domain threshold;
- action selection without recovery-option value.

Compare at identical action segments, reachable-prefix evidence, context availability, and recovery profiles. Report utility and recovery option coverage separately.

## Integration boundary

CMAQ-028 assumes recovery profiles have already passed the CMAQ-027 profile checks and an independent evidence verifier. This crate validates the profile bindings and failure-domain accounting but does not execute the recovery policy, authenticate receipts, or prove physical independence.

The intended stack is:
- CMAQ-025: is the response identifiable?
- CMAQ-026: is a new probe worth its cost?
- CMAQ-027: is a switch to one recovery mode qualified?
- CMAQ-028: does the proposed action preserve enough independent recovery options?
