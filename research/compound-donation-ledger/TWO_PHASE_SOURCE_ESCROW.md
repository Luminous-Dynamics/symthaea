# Two-phase source escrow: exact accounting and protocol refinement

## Why this architecture exists

The fixed-destination reference writer makes a reserve and chooses its recipient in one operation. That is simple and projective, but it couples the donor's evidence clock to the recipient's utility clock.

The two-phase architecture separates those clocks:

1. **Reserve:** donor evidence creates source-owned, unassigned escrow. The donor is charged immediately.
2. **Assign:** at a later pre-observation checkpoint, a policy transfers part of that unassigned balance into a destination-bound subtoken.
3. **Credit:** after the recipient is active, the assigned subtoken is credited to its incoming balance.

The reserve is not re-created during assignment. Assignment is not another donation, and incoming credit is never donor budget.

## 1. Pathwise accounting lemma

For donor i let X_i,t be a nonnegative, adapted, pathwise nondecreasing base process, and let the cumulative source reserve D_i,t satisfy

\[
0\le D_{i,t}\le\gamma_i\min(X_{i,t},1),\qquad
0\le\Delta D_{i,t}\le\gamma_i\Delta X_{i,t}.
\]

Define O_i,t as the cumulative amount reserved from donor i. It includes every state of that capital:
- unassigned source escrow;
- assigned but not yet credited destination escrow;
- already credited capital.

Let I_j,t be capital actually credited to recipient j. Each assignment binds source i to one immutable destination j>i, and each credit discharges only a prior assignment.

For any finite prefix [m], every credited amount entering j<=m has source i<j<=m. The donor outflow is already included in O_i,t. Unassigned amounts and amounts assigned to destinations j>m remain unmatched outflows at this cut. Therefore, pathwise,

\[
\sum_{j\le m}(I_{j,t}-O_{j,t})\le0.
\]

The weighted compound score

\[
Z_{i,t}=\gamma_iX_{i,t}+I_{i,t}-O_{i,t}
\]

is nonnegative because \(O_{i,t}\le\gamma_i\min(X_{i,t},1)\). It is pathwise nondecreasing because

\[
\Delta Z_{i,t}
=\gamma_i\Delta X_{i,t}+\Delta I_{i,t}-\Delta O_{i,t}\ge0.
\]

Assignments do not change I or O. They only move a balance between two escrow subaccounts. Credits increase I and reduce assigned escrow by the same amount. The original compound null-mass argument therefore remains unchanged under the assumptions of #9894: globally valid coordinate e-processes, a common global filtration, fixed globally summable weights, and the same time-monotone adjustment. This lemma is an accounting result, not a new general SupFDR theorem.

## 2. Token state machine

For each source token:

    original_amount
      = unassigned_remaining
      + sum(assigned_subtoken_remaining)
      + sum(credited_amount).

The allowed transitions are:

    UnassignedReserved(source, token_id, amount, creation_round)
        -> Assigned(source, token_id, assignment_id, destination, amount, round)
        -> Credited(source, token_id, assignment_id, destination, amount, round)

A token may be split over several assignment events. Each assignment subtoken has its own immutable destination and source-scoped assignment ordinal. The sum of all assignment amounts cannot exceed the source token's face amount.

The verifier must reject:
- destination <= source;
- assignment on the reservation round or earlier;
- assignment amount above the unassigned balance;
- duplicate assignment ID;
- destination mutation or retargeting;
- credit before assignment or recipient activation;
- credit greater than assigned remaining balance;
- repeated credit or replayed assignment;
- incoming re-donation;
- evidence, round, or policy-digest rollback.

## 3. Predictability and projectivity

An assignment for round t must be selected before requesting round-t observations, from the permitted global past \(\mathcal F_{t-1}\). The source token has already been reserved at an earlier round. The assignment is therefore predictable, even when the allocator uses lagged recipient evidence.

The two modes from #9923 remain distinct:

- **Projective:** the allocator reads only horizon-invariant metadata and the donor's allowed prefix information cone. Cut-projected horizon equality can be maintained.
- **Adaptive:** the allocator reads past histories of currently visible higher-ID recipients. The prefix budget and compound accounting can still hold for a fixed online stream, but strong initial-segment projectivity is not claimed.

A token's source identity and reserved amount are stable across horizons. Its later assignment may not be if the allocator sees newly appended recipients.

## 4. Liquidity-release schedule: next policy hypothesis

The first boundary-aware experiment assigned too early when a bank was smaller than the full predicted threshold gap, and a strict fill-only rule retained too much capital. The next design should test a **precommitted maturity ladder** rather than tune one greedy rule.

Split each reserved token into time tranches using globally fixed fractions, for example:

    immediate tranche: 1/2
    medium tranche:   1/4, releasable after L1 rounds
    long tranche:     1/4, releasable after L2 rounds

The release lags L1 and L2 are fixed in the policy digest before evidence collection. Each tranche remains unassigned until its release time. At release, the allocator uses only the pre-round history and eligible future recipients. Unused capital stays in the bank; it is never forcibly spent.

This is a time-diversified capital policy. It trades early utilization for the possibility of serving late arrivals or late-borderline recipients. It must be benchmarked against immediate assignment and a no-release diagnostic; no power gain is assumed.

## 5. Data-structure interaction

The prefix-budget index from #9995 remains valid:
- reservation at source i applies a negative suffix update to prefix balances;
- assignment changes no prefix balance;
- credit at destination j applies a positive suffix update.

The token ledger adds two maps:
- source token map keyed by (source_id, source_local_ordinal);
- assignment map keyed by (source_id, token_ordinal, assignment_ordinal).

The writer can target O(log n + log T) reserve/assign/credit transitions once the prefix-max tree is implemented. The independent verifier remains a separate linear replay and must recompute source and assignment conservation from the event stream.

## 6. Exact-rational structural screen

The companion runner models source reservations, partial assignments, partial credits and token-local conservation with fractions.Fraction. A separate replay implementation consumes only the immutable event list and candidate weights; it does not call the writer's transition functions.

Recorded run:
- 250 deterministic traces;
- 16 candidate IDs;
- 40 rounds per trace;
- 132,838 immutable events independently replayed;
- five tamper controls rejected;
- donor caps, own-increment caps, prefix budgets, score monotonicity/nonnegativity, per-token conservation and global escrow conservation checked after every transition.

The result is an exact Python accounting screen only. It does not validate a Rust implementation, statistical assumptions, source authenticity, or FDR/SupFDR.

## 7. Next statistical experiment

The current process-valued generator in #10033 is the matched evidence source. Add:
- immediate fixed-next;
- two-phase next-recipient at the next pre-observation checkpoint;
- two-phase boundary allocator;
- two-phase maturity ladder with fixed release lags;
- projective static-prior baseline;
- no-donation baseline.

Include a late predictable-arrival scenario in which the donor reserves early and the useful recipient activates later. Also include no-useful-recipient and far-recipient negative cases.

Report SupFDR, fixed-time FDR, power, time-to-discovery, reserve volume, assigned/unassigned/credited balances, destination concentration, and exact invariant failures. Start with 5,000 exploratory paired streams; use #9714's 50,000+ frozen-stream protocol for confirmatory cells. The two-phase mechanism is not a way around the e-process or filtration assumptions.

## Claim boundary

This is a protocol and accounting design. The recorded exact replay does not prove statistical validity, power improvement, projectivity of adaptive assignment, or superiority over donation e-LOND / dynamic e-closure.