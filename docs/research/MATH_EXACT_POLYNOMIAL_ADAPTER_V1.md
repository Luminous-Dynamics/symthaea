# MATH-REP-001B — Exact polynomial adapter v1

Status: **draft qualification harness**

Normalizer identity:

`symthaea-exact-polynomial-term-v1`

Predecessor contract: MATH-REP-001A / `math-normalization-receipt-v1`.

## Purpose

Qualify a narrow conversion boundary from `FolFormulaExt::Term` into the repository's existing exact rational/multivariate-polynomial representation without changing runtime retrieval, theorem authority, mathematical memory, or the frozen structural-HDC experiment.

The adapter is intentionally example-local first. Passing this harness is a prerequisite for promotion into the public `symthaea-core` module surface.

## Authority boundary

This adapter performs canonicalization for retrieval/measurement only.

```text
same canonical polynomial
        !=
formal proof of an arbitrary theorem

normalization failure
        !=
non-equivalence
        !=
falsehood
```

A later receipt layer must bind successful results to `math-normalization-receipt-v1` with `authority = RetrievalOnly` and `source_preserved = true`.

## Frozen v1 fragment

Accepted:

- variables;
- integer literals;
- exact rational literals in the `Real` domain;
- rational literals with denominator 1 in `Int`/`Nat`;
- addition;
- subtraction;
- multiplication;
- unary negation;
- non-negative integer powers (the only exponent form representable by `FolFormulaExt::Term::Pow`).

Rejected/unsupported:

- every `RealLit(f64)` — v1 never guesses an exact rational from an inexact floating representation;
- every `Div` node — even constant division is deferred rather than broadening the fragment after the contract was frozen;
- malformed zero-denominator rational ASTs;
- non-integral rational literals under `Int`/`Nat`;
- expressions exceeding resource/representation limits;
- arithmetic that cannot remain exact inside the existing `i64` coefficient envelope.

## Reuse of existing algebra

The final representation is the existing:

- `hdc::polynomial_algebra::Rat`;
- `hdc::polynomial_algebra::Monomial`;
- `hdc::polynomial_algebra::Poly`.

`Poly::from_terms` remains the repository's canonical monomial-order/storage boundary.

The adapter does **not** call the polynomial-division implementation and does not introduce Gröbner, SMT, or solver authority into normalization.

## Checked arithmetic shell

The existing `Rat` operator implementations use `i64` arithmetic. MATH-REP-001B therefore does not feed unchecked intermediate arithmetic through those operators.

Instead it:

1. maintains a unique sparse monomial map;
2. performs coefficient arithmetic through checked `i128` intermediates;
3. cross-reduces rational products before multiplication;
4. checks exponent addition;
5. reduces every coefficient before converting back to `Rat`;
6. rejects an operation when the reduced result cannot be represented exactly in the existing `i64` coefficient envelope;
7. gives `Poly::from_terms` already-unique monomials so its combining path does not need to perform coefficient addition.

An overflow is therefore a `ResourceLimit`-class non-conclusive outcome, never wrapped arithmetic.

## Frozen resource envelope

```text
MAX_VARIABLES  = 64
MAX_MONOMIALS  = 4096
MAX_AST_DEPTH  = 256
```

These are engineering limits, not mathematical statements.

Crossing one means only:

`this v1 adapter declined the normalization request`.

It does not mean the term is non-polynomial, non-equivalent, false, hard, or unimportant.

## Variable identity correction

MATH-REP-001B deliberately **does not alpha-normalize free variables**.

A standalone `Term` has no binder context. Renaming/permuting free variables at this layer could accidentally make distinct objects appear identical. Therefore v1 canonical serialization uses:

```text
variable_identity = PreserveFreeVariableNames
```

and variables are deterministically ordered by the existing `Term::free_vars()` stable ordering.

Binder-aware alpha normalization belongs in a later `FolFormulaExt` formula adapter, where quantified-variable identity can be changed safely while free-variable identity remains explicit.

This means the v1 receipt transformation list must **not** claim `AlphaNormalizeVariables`.

## Domain separation

The canonical serialization contains the explicit numeric domain:

- `Int`
- `Nat`
- `Real`

Thus syntactically identical polynomial terms in different numeric domains do not share a canonical identity.

The current function is named `normalize_term_uniform_domain` intentionally. It requires its caller to provide one domain for the standalone term. Mixed-domain formulas require a later binder/type-environment adapter and must not be routed through this API under a fabricated uniform domain.

## Canonical serialization

The byte-level canonical serialization is JSON over a closed ordered struct containing:

- format identity;
- normalizer identity;
- numeric domain;
- free-variable identity policy;
- stable variable list;
- monomial-order identity;
- canonical polynomial terms with exact numerator/denominator and exponent vectors.

The normalizer identity is part of the serialization, so an implementation-rule change necessarily creates a different evidence/retrieval lineage.

The adapter does not yet emit the SHA-256 receipt digest. Receipt binding is a later step and must use the predecessor contract's exact digest semantics rather than introducing a second digest convention inside this example harness.

## Mechanical canaries

The focused example tests require:

### Positive exact-normal-form pairs

```text
x * x                 <-> x^2
x + x                 <-> 2*x
(x + 1)^2             <-> x^2 + 2*x + 1
x + y                 <-> y + x
(2/4)*x               <-> (1/2)*x    [Real]
```

### Required distinctions

```text
x - y                 !=NF y - x
free x identity       !=NF free y identity
Int(x + 1)            !=NF Nat(x + 1)
Int(x + 1)            !=NF Real(x + 1)
Nat(x + 1)            !=NF Real(x + 1)
```

### Non-conclusive rejection canaries

```text
RealLit(0.5)           -> Unsupported / OtherUnsupported
x / 2                  -> Unsupported / NonPolynomialDivision
RatLit(1, 0)           -> Rejected / OtherUnsupported
RatLit(1, 2) under Int -> Unsupported / DomainAmbiguity
coefficient overflow   -> Unsupported / ResourceLimit
exponent overflow      -> Unsupported / ResourceLimit
>64 variables          -> Unsupported / ResourceLimit
>4096 monomials        -> Unsupported / ResourceLimit
AST depth >256         -> Unsupported / ResourceLimit
```

No rejection may be interpreted as a negative mathematical result.

## Intended focused execution

```bash
cargo test -p symthaea-core --example math_exact_polynomial_q0
cargo run -p symthaea-core --example math_exact_polynomial_q0
```

The executable prints only normalizer identity and equivalence-canary booleans. It does not publish theorem, proof, novelty, or HDC-performance claims.

## Deliberate nonclaims

MATH-REP-001B does not establish that:

- this harness has compiled or executed in CI yet;
- canonical-polynomial equality is sufficient for every mathematical equivalence;
- the accepted fragment covers divisions, transcendental functions, inequalities, side-condition-sensitive cancellation, mixed-domain formulas, or arbitrary algebra;
- normalization improves retrieval;
- HDC improves normalized-form retrieval;
- any theorem is true or novel;
- a retrieval normal form is formal authority.

## Next tranche: MATH-REP-001C

After focused qualification, freeze an equivalence-retrieval challenge set **before changing this normalizer**.

At minimum compare:

```text
S   syntax-only retrieval
N   normal-form-only retrieval
S+N fused syntax + normal-form retrieval
```

under equal retrieved-item/byte/compute budgets.

Include adversarial families where normalization must refuse rather than guess, plus form-transfer families such as factored/expanded polynomials.

Any change to accepted syntax, arithmetic rules, variable-identity policy, resource envelope, or canonical serialization requires a new normalizer identity and a fresh holdout lineage.
