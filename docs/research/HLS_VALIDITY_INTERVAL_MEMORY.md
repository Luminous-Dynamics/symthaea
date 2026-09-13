# HLS causal-time validity interval memory

This note defines the first historical-memory substrate built on top of the temporal phasor algebra from #2745.

## Scope

This tranche does **not** change HLS recurrence and does not alter the current-state preregistration in #2662.

Its only question is whether vector-symbolic temporal binding can represent exact historical `as_of_event` semantics with measurable cleanup margins under finite-dimensional superposition.

## Two time domains

Symthaea now has two deliberately distinct notions of time:

1. **physical/dynamical time** — irregular real-valued `dt`, used by liquid state evolution;
2. **causal/version time** — the ordered state checkpoint after event `n`, used by historical validity.

Conflating these would be a category error. Wall-clock proximity does not determine which value is authoritative after a mutation. The benchmark's historical oracle is defined by event order.

Checkpoint `n` is therefore represented at the temporal coordinate

`n + 1/2`.

A relation value valid after checkpoints `start .. end-1` is encoded as

`S[start,end) = sum_{n=start}^{end-1} T(n + 1/2)`.

This gives the required half-open predecessor semantics:

- the old value occupies checkpoints before the mutation;
- the new value begins exactly at the mutation checkpoint;
- no measure-zero continuous-time boundary convention is needed.

## Analytic O(D) span encoding

For one Fourier frequency `omega`, the finite series is

`sum exp(i * omega * (n + 1/2))`.

Using the geometric/Dirichlet identity, the complete interval is encoded per dimension with constant work, independent of the number of checkpoints covered.

Therefore a closed validity span costs `O(D)` to archive rather than `O(length * D)`.

## Orthogonality hypothesis

The temporal axis samples frequencies uniformly from `[-pi, pi)`.

For distinct integer checkpoints `n != m`, the expected temporal inner product is

`E[cos(omega * (n-m))] = 0`.

The active checkpoint contributes expected signal 1; other checkpoints contribute zero-mean finite-dimensional crosstalk.

This does **not** imply perfect finite-D retrieval. Cleanup accuracy and margin are empirical capacity questions.

## Key/value association

A closed relation history is stored as

`key * value * S[start,end)`

where `key` and `value` are bipolar unitary HDC roles.

At checkpoint `q`, a candidate value is scored by correlating the memory with

`key * candidate * T(q + 1/2)`.

For the matching key/value/checkpoint, the unitary roles cancel and the active temporal component supplies the signal. Other keys, values, and checkpoints appear as crosstalk.

## Benchmark bridge

`StateTrackingValidityArchive` converts the benchmark event stream into closed validity spans:

- each `MoveEntity(e -> l)` starts a new entity-location span;
- each `TransferObject(o -> e)` starts a new object-owner span;
- the next mutation of that same relation closes the previous span;
- remaining open spans close at `events.len()`.

Every observed mutation therefore creates exactly one eventual closed span, so a complete archive contains exactly `benchmark.events.len()` spans.

Historical queries use `as_of_event` directly as the causal checkpoint.

Historical object-location remains compositional:

1. clean up `object -> owner` at the requested checkpoint;
2. use the recovered entity identity as the subject of `entity -> location` at the same checkpoint.

No learned decoder is involved.

## Prior-art boundary

Fractional binding, FHRR/SSP continuous coordinates, phasor-domain translations, and trajectory/region representations are prior art. This tranche does not claim to invent them.

The research contribution being tested is narrower: whether those algebraic tools, combined with Symthaea's self-keying relation codebook and explicit causal checkpoint semantics, form a practical historical state archive that composes with the HLS research line.

## Mechanical qualification

The current tests require:

- exact half-open predecessor semantics on small histories;
- independent keys to retain distinct histories;
- a moderate shared-memory superposition to retain positive cleanup margins;
- malformed intervals/candidate sets to fail closed;
- one closed archive span per benchmark event;
- a seeded historical benchmark to match the oracle, including two-hop object-location queries, with positive cleanup margin.

These are mechanical research gates, not a capacity result.

## Not yet claimed

This tranche does not yet establish:

- a scaling law for maximum facts per dimension;
- robust retrieval at million-event horizons;
- online storage of the currently open interval;
- end-to-end learned HLS + historical archive performance;
- superiority to attention, SSM, differentiable memory, database, or explicit event-log baselines;
- a confirmatory historical-memory result.

## Next experiment

The next tranche should freeze a deterministic capacity sweep across:

- HDC dimension;
- number of relation keys;
- number of candidate values;
- checkpoint horizon;
- mutation density.

Report accuracy, minimum/mean cleanup margin, memory bytes, and write/query cost. Null and failure regimes are part of the result.

Only after the capacity boundary is measured should historical queries be reintroduced into an integrated HLS experiment.
