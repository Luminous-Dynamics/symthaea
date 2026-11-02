# Cut-projectivity and adaptive routing: refinement to Issue #9923

## Executive result

The fixed-destination ledger in PR #9919 remains the implementation baseline. This note sharpens the adaptive-routing theorem target in #9923 in three ways:

1. Projectivity must compare **cut-projected state**, not literal balances of tokens whose destinations lie outside the old horizon.
2. Token identity must be source-scoped, not allocated by one global sequence counter.
3. Pre-round predictability, current-round adaptedness, and horizon projectivity are distinct properties and should be benchmarked separately.

The accompanying exact-Fraction harness is a bookkeeping regression, not an e-process validity test or an FDR/SupFDR experiment.

## 1. The accounting theorem is routing-policy agnostic

Fix a deterministic global candidate universe with IDs 1,...,N and fixed weights gamma_i >= 0, sum_i gamma_i <= 1. For a finite prefix [m], let X_i,t be nonnegative, adapted, pathwise nondecreasing base evidence processes. Under a true null, assume E[X_i,t] <= 1 at each fixed t.

Let O_i,t be cumulative source-only withdrawals and I_i,t cumulative credits received. Require, pathwise:

- O_i,t and I_i,t are nonnegative and nondecreasing;
- O_i,t <= gamma_i min(X_i,t, 1);
- Delta O_i,t <= gamma_i Delta X_i,t;
- every token has one source i, one immutable destination j>i, one face amount, and a nonincreasing remaining balance;
- credits never exceed remaining balance, tokens cannot be duplicated, and incoming credit cannot finance a donor withdrawal;
- every credited amount is the discharge of a previously reserved token.

The destination-selection rule does not appear in these inequalities. It may be a static rule, a pre-round predictable rule, or a current-round adapted rule, provided the actual route is fixed before the resulting credit is applied and no token is rewritten after creation.

For every prefix [m], forward-only edges imply the pathwise cut budget:

    sum_{i <= m} (I_i,t - O_i,t) <= 0.

For any true-null set N and alternatives A=[m]\\N, define the weighted compound score

    Z_i,t = gamma_i X_i,t + I_i,t - O_i,t.

Then

    sum_{i in N} Z_i,t
      = sum_{i in N} gamma_i X_i,t + sum_{i in N}(I_i,t - O_i,t)
      <= sum_{i in N} gamma_i X_i,t + sum_{i in A} O_i,t
      <= sum_{i in N} gamma_i X_i,t + sum_{i in A} gamma_i.

Taking expectations gives

    E[sum_{i in N} Z_i,t] <= sum_{i in N} gamma_i + sum_{i in A} gamma_i
                           <= 1.

Also,

    Delta Z_i,t = gamma_i Delta X_i,t + Delta I_i,t - Delta O_i,t >= 0,

and Z_i,t >= 0 follows from the donor cap and I_i,t >= 0.

This is the compound null-mass accounting ingredient already used by #9894. It is not, by itself, a new general SupFDR theorem. The existing e-BH/compound-e-value argument and its assumptions must still be invoked. For a stopped or supremum claim, require the score processes to be adapted to the operational filtration and use the established monotone-process argument; the accounting inequality alone does not authenticate the filtration or arrival process.

**Important distinction:** current-evidence-aware routing can preserve the accounting theorem. Predictability before the current observation is not needed for the displayed pathwise budget algebra. Online execution still requires decisions and score states to be available at the time they are used (adaptedness). A pre-round route commitment is a stricter anti-lookahead and governance policy, not a consequence of the prefix-budget proof.

## 2. Three routing contracts

Keep these modes separate in both API and benchmarks:

| Mode | Route may read | What it establishes |
|---|---|---|
| Static | Immutable candidate metadata | Baseline; simplest audit |
| Pre-round predictable | Prefix history through round t-1 | Strong prequential commitment |
| Prefix-current adapted | Current-round evidence for IDs <= donor i, plus prior prefix history and fixed-universe metadata | Uses the donor's just-observed increment while retaining initial-segment projectivity |
| Global-current adapted | Current-round evidence from any ID in the active horizon | May retain accounting validity, but is not projective when higher IDs affect an older route |

For the first relaxed projective mode define

    G_i,t = sigma(fixed universe metadata, events/evidence for IDs <= i through round t).

The route for donor i in round t must be a deterministic or auditable function of G_i,t. Process candidates in stable ID order within each round: donor i's current observation is available before its route is selected; observations for IDs greater than i in that round are not. The route is then frozen into the token before any later recipient credit. This makes route selection adapted, but not predictable with respect to G_i,t-1.

A stricter pre-round mode uses G_i,t-1 and remains a valid optional policy. It should not be imposed as a mathematical necessity without a separate result showing why same-round adapted routing fails.

## 3. Correct projectivity: compare the cut projection

Consider two initial-segment horizons m<n, the same fixed global universe metadata, the same base evidence and activation events for IDs <=m, and the same prefix-local routing policy. Then, for every donor i<=m:

- its route inputs are identical in both replays;
- its route decisions and source withdrawals are identical;
- its token destination, creation round and face amount are identical;
- all candidate evidence, incoming/outgoing balances and weighted scores for IDs <=m are identical;
- any token whose destination is also <=m has identical remaining balance and internal credit history.

A token with source i<=m and destination j>m is different. In the m-horizon replay, j is absent and the token may remain escrowed. In the n-horizon replay, j may activate and receive some or all of it. Requiring literal remaining-balance equality for this token is therefore impossible in general, even though no old candidate state changes.

Define its cut-projected outside balance as

    outside_balance(token, m)
      = remaining_escrow(token) + credits_already_delivered_to_destinations > m.

By token conservation this is always the immutable face amount. The cut projection records:

- complete token state for edges i<j<=m;
- immutable token identity/source/destination/creation round/face amount for edges i<=m<j;
- one collapsed outside balance for each crossing token;
- candidate states and credits only for recipients <=m.

Credits to destinations >m are erased from the prefix event projection or represented as transfers to a single cut sink. They are not compared as ordinary in-prefix credits.

Under the stated prefix-coherence conditions, these projected records are identical across m and n. Do not claim literal equality of all global token remaining balances or complete unprojected receipts.

## 4. Token identity is part of projectivity

The current reference writer allocates token IDs from a single monotonically increasing global counter. This counter is not horizon-stable: a donor with ID >m can create a token in an earlier round and shift the ID assigned to a later token from donor i<=m.

Use a source-scoped identity, for example

    TokenId { source_id: u64, source_ordinal: u64 }

where source_ordinal increases only when that source creates a token. The identity is stable under appending unrelated higher-ID donors. A digest may be layered on top for content authentication, but a bare truncated hash should not replace collision-free identity unless collision handling is explicit.

Canonical receipt projection should preserve source-scoped IDs. Full global sequential event numbers may remain as transport offsets, but must not be treated as semantic token identity.

## 5. Negative controls the implementation must reject

1. **Higher-ID lookahead:** route donor 1 to candidate 2 in the short horizon, but to candidate 3 when candidate 3's evidence is visible in the extended horizon. This can alter candidate 2's old-prefix score.
2. **Global token counter:** donor 3 creates an earlier token in the extended replay, shifting donor 1's later token ID.
3. **Literal cross-cut token equality:** a token to candidate 3 stays in escrow in horizon 2 but is credited in horizon 4. Its remaining balance differs; its cut-projected outside balance does not.
4. **Retargeting:** changing a token destination after creation must fail replay.
5. **Incoming re-donation:** a recipient's incoming credit must not enlarge that recipient's source withdrawal cap.
6. **Same-round higher-ID evidence:** in prefix-current adapted mode, donor i cannot inspect observations from IDs >i in the same round.

## 6. Executed structural screen

The standard-library Python harness uses exact fractions, a fixed global universe of 16 candidates, 40 rounds, 250 paired traces, and prefixes cycled from 1 through 15. Evidence streams are independently seeded per (candidate, round), so extending the horizon cannot shift random draws for old candidates. Its route rule uses current evidence for IDs <= donor i and fixed global-universe metadata.

Recorded result:

- 250/250 short/extended pairs match exactly under cut projection;
- all candidate states through the old prefix match;
- all internal token states and prefix credit receipts match;
- token identities are source-scoped and stable;
- every prefix budget, donor cap, own-increment cap, monotone/nonnegative score and escrow-conservation assertion holds;
- literal remaining balances differ for crossing tokens in 250/250 pairs.

This is intentionally a structural stress test. It does not generate valid null e-processes, estimate statistical error rates, prove filtration measurability in a real service, or run Rust. The committed runner is independently reproducible with:

    python3 routing_cut_projectivity_check.py --traces 250 --rounds 40

## 7. Implementation and benchmark plan

1. Keep PR #9919's fixed-destination writer and verifier unchanged as the baseline.
2. Introduce a semantic source-scoped TokenId before claiming token-identity projectivity.
3. Add a canonical cut-projection function to both writer-side receipts and the independent verifier.
4. Add explicit round/ID event phases so prefix-current adapted decisions cannot inspect a later ID's same-round evidence.
5. Keep a separate pre-round predictable mode for stricter governance deployments.
6. Benchmark static, pre-round predictable, prefix-current adapted and global-current adapted routing on identical evidence streams.
7. Report candidate-state projectivity and token cut-equivalence separately from statistical validity.

The confirmatory programme remains 50,000+ frozen streams per statistical cell, with low-tail error-rate cells requiring substantially more. Compare donation e-LOND / online e-closure, fixed-destination ledger, SCORE variants under their exact assumptions, e-GAI/LORD/SAFFRON in matching dependence lanes, and dynamic e-closure / persistent adjusters. Report FDR, SupFDR, stopped-FDR, FDX, power, time-to-discovery, reserve volume, utilization, stranded escrow, recipient concentration, worst-regime regret, p50/p95 latency, memory and independent-verifier time.

**Claim ceiling:** this note strengthens the accounting and projectivity specification. It does not claim a theorem about real-world arrival authentication, a complete adaptive router implementation, improved power, or superiority over SOTA.
