# VART-WORLD-CREATIVE-001 Confirmatory Execution v1

## Purpose

This contract governs execution after a prospective freeze has passed the confirmatory launch gate. It adds no scientific hypotheses, thresholds, metrics, policies, fixtures, seeds, or stopping criteria.

The execution instrument may only instantiate the already-frozen trial inventory.

## Immutable inputs

Before the first confirmatory trial, execution must bind:

- exact raw `confirmatory_freeze_v3.json` bytes and externally recorded SHA-256;
- exact frozen trial inventory bytes and SHA-256;
- exact clean subject HEAD/TREE;
- exact clean instrument HEAD/TREE;
- exact qualified launch-gate implementation;
- one fresh confirmatory evidence root;
- the pre-bound deterministic pseudo-random `run_order` stored in the inventory.

`run_order` is part of the prospective design. The runner MUST NOT generate, shuffle, optimize, or change order at execution time.

## Zero-peeking

Until all attempted trials are accounted for and the campaign is sealed:

- console output may expose trial identity, ordinal position, process/integrity status, and crash/abort status only;
- outcome magnitudes, policy rankings, human preference values, calibration values, and comparative summaries MUST NOT be emitted to the operator console;
- captured runtime stdout/stderr are evidence and may be retained privately, but the runner MUST report only their cryptographic digests during execution;
- no primary or secondary scientific analysis may run;
- no policy-specific retry is permitted.

## Trial execution

Each inventory row represents one revision-trial. The runner executes rows strictly by ascending frozen `run_order`.

Each trial receives a private staging directory and, at minimum, the frozen fields:

- `trial_id`
- `subcampaign`
- `policy`
- `fixture`
- `seed` when present
- `revision_index`
- `world_cluster_sha256`
- `world_lineage_sha256`
- `run_order`
- experiment/campaign identifiers
- evidence output root for that trial

The runtime command is supplied by an execution adapter, but its resolved argv is sealed into the campaign evidence before execution begins.

## Failure semantics

A nonzero runtime process exit is an operational/integrity event, not a scientific negative result.

The runner:

1. preserves the attempted trial staging evidence and stdout/stderr digests;
2. records the process return code;
3. performs no automatic retry;
4. stops the campaign fail-closed.

A scientifically bad but structurally valid result must be emitted by the runtime as a normal valid trial with exit code 0 and retained in the evidence package.

Any restart after a process/integrity abort must follow the frozen abort/missingness policy. The execution wrapper itself does not decide whether the existing campaign lineage can continue.

## Campaign closure

After all frozen trials have been attempted successfully:

1. verify complete trial-ID accounting against the frozen inventory;
2. compute a deterministic tree closure over the evidence root excluding only the final campaign receipt;
3. write exactly one `CONFIRMATORY_CAMPAIGN_RECEIPT.json` containing input hashes, source identities, ordered trial execution receipts, and the evidence closure;
4. run the frozen qualified verifier against the sealed root and externally anchored freeze hash;
5. expose only verifier acceptance/rejection status;
6. keep `claim_authorized=false`.

Scientific analysis begins only after sealed evidence verification succeeds.

## Authority boundary

A successful execution receipt may state only that the preregistered campaign executed and sealed under the frozen instrument.

It does not establish H1, H2, H3, creativity, general intelligence, consciousness, or a scientific claim.
