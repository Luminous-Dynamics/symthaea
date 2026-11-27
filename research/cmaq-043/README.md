# CMAQ-043 — Fleet-wide hierarchical anytime risk budget

## Why this matters

A valid anytime monitor at alpha=0.05 controls one stream's probability of ever falsely alarming. Starting 100 separate monitors at alpha=0.05 does not give the fleet a 5% false-alarm budget. In the worst case, a union bound only gives 100 × 5% (capped at one).

CMAQ-043 allocates one global false-alarm budget over a countable hierarchy of assets, evidence channels and deployments/restarts. It does not assume independence among those monitors.

## Allocation rule

For one-based indices:
- i: frozen asset priority;
- j: frozen evidence-channel priority within the inventory;
- k: frozen deployment/restart ordinal;

allocate

    alpha[i,j,k] =
        alpha_global / (i(i+1) * j(j+1) * k(k+1)).

Each level uses the telescoping weights

    w_i = 1/(i(i+1)) = 1/i - 1/(i+1)
    sum_{i=1}^infinity w_i = 1.

Therefore

    sum_{i,j,k >= 1} alpha[i,j,k] = alpha_global.

Each local anytime-valid monitor must enforce its own alpha[i,j,k] (equivalently Ville threshold 1/alpha[i,j,k]). By the union bound, the probability of at least one false alarm across the countable fleet is at most alpha_global. No independence assumption is needed.

The ordering is a policy decision: earlier priority indices receive larger risk budgets. The asset/channel/deployment ordering and their mapping to real IDs must be precommitted, versioned and bound by the plan and inventory digests. Budgets cannot be reallocated after observing a signal or outcome.

## Exact implementation

The crate uses rational arithmetic with checked u128 products and gcd reduction. No floating-point value participates in allocation or threshold policy. Each receipt contains:
- allocation-rule version;
- global plan and inventory digests;
- stream, asset, channel and deployment IDs;
- three one-based hierarchy indices;
- exact global alpha and exact local alpha as numerator/denominator pairs.

The reciprocal Ville threshold is also exposed as an exact rational pair. A logarithmic threshold is available only as a display/interop value.

AllocationLedger is constructed from one immutable BudgetPlan and refuses a different plan/inventory/global-alpha binding. It enforces one-to-one identity/index bindings for assets, channels within assets, and deployments within asset/channel pairs. It consumes each stream ID and each hierarchy-index triple at most once in memory. It computes the receipt before mutating its ledger, so overflow or validation errors leave the ledger unchanged.

## Durability and trust boundary

The current ledger is in-memory only. A deployment needs a durable, authenticated allocation ledger shared across supervisors/hosts and protected across process restarts. It must reject replayed receipts, enforce asset/channel/deployment assignment authenticity and preserve the global plan version. Structural SHA-256 shape checks do not authenticate signatures or establish who is authorized to assign an index.

Every monitor must independently satisfy its own anytime-valid assumptions and correctly enforce its allocated local alpha. The allocator cannot repair an invalid monitor, bad score, unqualified reference, omitted common cause, or untrusted event stream.

## Tests

The 12 deterministic tests cover exact fraction reduction, the telescoping partial sum identity, bounded three-level budget sums, exact local alpha and Ville threshold, monotonic priority allocation, one-use ledger behavior, plan binding, duplicate IDs/scope, invalid indices, malformed plans and fail-closed arithmetic overflow.

Run:

    cargo +1.100.0 test --manifest-path research/cmaq-043/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-043/Cargo.toml --locked --all-targets -- -D warnings

## Connection to the CMAQ programme

- CMAQ-031 / 033 / 038 / 040 provide per-stream anytime-valid rank monitors.
- CMAQ-034 provides fail-closed quarantine and recovery state.
- CMAQ-036 / 039 provide common-cause closure and effective calibration-block lineage.
- CMAQ-041 evaluates warning, quarantine and fallback outcomes.
- CMAQ-042 optionally calibrates a fixed finite mission-window threshold under its separate exchangeability conditions.
- CMAQ-043 allocates a single anytime error budget across assets, channels and mission restarts without assuming independence.

This is an assurance-plane primitive. It is not a safety certification or a claim that the fleet's physical failure modes have all been identified.
