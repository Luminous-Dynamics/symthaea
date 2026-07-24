# Helicopter qualification protocol

Qualification artifacts use `symthaea-helicopter-qualification-v1` and are
machine-evaluated from explicit scenario definitions and observations.

A scenario declares:

- a stable scenario identifier;
- the minimum number of distinct deterministic seeds;
- faults that must actually be exercised;
- metric direction and threshold for every acceptance gate.

Every observation must be complete, contain each required metric exactly once,
name the exercised faults, reference a flight-log digest, and report successful
replay-chain verification. Missing seeds, metrics, fault coverage, completion,
or verified evidence produce `Incomplete`, never a synthetic pass. Any observed
threshold violation produces `Fail` even when other campaign evidence is still
incomplete.

The report preserves scenario order and sorts observations by seed before
aggregation. Canonical JSON and an FNV-1a artifact digest support deterministic
replay equality. The digest is not a digital signature.

## Independent contingency assurance

Qualification runs must audit every phase-changing or safety-limiting mission
decision with `MissionAssuranceKernel`. The audit independently reconstructs
the documented hazard precedence and records whether the observed directive and
reason are verified, rejected, or incomplete. A qualification result cannot
pass when a rejected decision audit exists, or when navigation-loss duration
needed to evaluate the grace interval is absent.

## Real-time qualification gates

Physical/HIL campaigns should gate maximum absolute control-loop jitter, missed
control deadlines, and maximum sensor-to-actuator latency. Repeated deadline
misses that reach `RealtimeHealth::Unsafe` are a known failure, not incomplete
evidence. Missing timing observations remain incomplete.


## Parameter-uncertainty campaign design

Nominal scenario seeds are supplemented by a versioned `CampaignPlan` bound to
the base scenario digest. Each uncertain parameter declares units, minimum,
nominal, maximum, and coverage bins. The generator includes nominal, per-axis
boundaries, global corners, and deterministic stratified samples. Release
evidence reports missing boundaries and under-covered pairwise bins. Coverage
is an adequacy signal only; it does not establish that untested continuous
regions are safe.
