# Consolidation-failure audit: why do `symthaea-core` extractions become orphans?

**Scope: audit only. No architectural code changes made.** 12 read-only evidence agents (9
suspected duplicate pairs + 3 controls), then synthesis. Full per-agent records in the workflow
transcript at `subagents/workflows/wf_284cffe7-3b7/journal.jsonl`.

The audit was commissioned to explain why `symthaea-hdc-ltc` and `symthaea-hdc-crypto` — both
deliberately extracted from `symthaea-core` — ended up with 1 and 0 dependents while the in-core
originals remained canonical. It answered that, and overturned the hypothesis it was given.

---

## The headline: the facade hypothesis is unsupported

The prior plan (and this author's recommendation) held that the failure mechanism was
*"re-export from `symthaea-core` as a facade so no dependent changes on day one"* — that zero
day-one change removes all pressure to migrate.

**The evidence points the other way.** Facades appear in **both** successful controls and in
**none** of the failures:

| case | facade? | origin definitions? | outcome |
|---|---|---|---|
| `CONTROL-fep` | yes — `pub use symthaea_fep as fep_active_inference;`, 35 legacy call sites | **deleted** (origin dir absent from disk) | **success** |
| `CONTROL-cognitive-types` | yes — pure re-export, 39 legacy call sites | **deleted** (25 defs on `-` lines, 0 on `+`) | **success** |
| `hdc_ltc_unified` | **no** — core keeps `pub mod` + re-exports its *own* types | kept | failure |
| `hdc_crypto` | **no** | kept | failure |
| `broca-tools/geodesic_bridge` | **no** (structurally impossible — would cycle) | re-duplicated back | failure |

A facade preserves **paths**. Only surviving **definitions** cause drift. Removing facades would
have targeted the wrong thing.

**Stated honestly by the audit itself:** the hypothesis was not merely refuted, it was
*untestable on this data* — it requires a case where a facade was built *and* origin definitions
were left in place, and that configuration appears nowhere in the sample.

---

## The real discriminator: definition-site cardinality

For every public symbol in the extracted abstraction, how many definition sites exist
workspace-wide?

- `CONTROL-cognitive-types`: 27 types, each grepped individually — **exactly one site each**.
- `CONTROL-fep`: one directory; the origin path does not resolve on disk.
- `hdc_ltc_unified`, `hdc_crypto`, `geodesic_bridge`: **two sites each.**

This separates all four successes from all three failures. It is cheap, mechanical, and
independent of usage.

**Adoption count is NOT the discriminator** — the controls falsify it in both directions:

- `CONTROL-broca-tools`: **0** dependents, 24 of 26 modules cleanly extracted — healthy.
- `CONTROL-cognitive-types`: **1** dependent, zero direct namers — complete success.
- `hdc_ltc_unified`: **1** dependent, which imports only `ContinuousHV` and never touches the
  neuron that was the entire point — failure.

The mechanical tell for the failures: `git show --stat 9baa58af21` reports **18 files changed,
4,647 insertions, 0 deletions**, and `git ls-tree` confirms the core files were present and
untouched in that same tree. **It was a copy that was called an extraction.**

---

## Seven of nine suspects were never migrations

The name-matching heuristic that generated the suspect list ran at roughly **22% precision**.

- **`projection` is a homonym.** Core: trainable linear maps between HDC dimensionalities
  (`LearnedProjection` with Adam, `RandomProjection` with Johnson–Lindenstrauss). Crate: a
  serde/chrono DTO for 2.5D telemetry rendering. Zero shared symbols, zero shared dependencies,
  and the crate does not depend on `symthaea-core` at all.
- **`consciousness_topology` is a false pair.** The crate *is* a successful extraction — of a
  *different* file (`src/consciousness/consciousness_topology.rs`, 97% rename in `f53ccac34e`).
  The audited core module is a sibling that has coexisted since 2026-02-05. The two never shared
  a blob.

### The larger finding: this is a discovery failure, not a migration failure

Five crates (`statistics`, `graph_theory`, `combinatorics`, `complex`, `game_theory`) were
created inside a **48-hour window on 2026-07-09/10**, each asserting in its commit message or
`lib.rs` docs that the workspace lacked the capability. **Every one of those claims is false**
against a `symthaea-core/src/hdc/` module predating it by 2–5 months.

That produced **more duplicate pairs (5) than migration failure did (2)**.

`DOMAIN_CRATES_INDEX.md` exists to prevent exactly this and warns about it in its first three
lines — but it indexes only `crates/domains/`, so in-core modules are **invisible** to the check
it was built to perform.

---

## A live build break, confirmed by compilation

`symthaea-broca` does not compile under `--features mamba-cpu,code-sheaf-eval`:

```
Checking symthaea-broca v0.1.0
error[E0061]: this method takes 2 arguments but 3 arguments were supplied
exit_code   : 101
```

**Mechanism — duplicate divergence, concretely:** the `signature` parameter was added to the
*extracted* copy (`symthaea-broca-tools/src/geodesic_bridge.rs:39`, 3 params) and the caller in
`broca/src/liquid_mamba.rs:1057` was updated to match — but **`broca`'s own copy
(`geodesic_bridge.rs:55`) was never updated** (2 params), and the caller resolves to *that* one
(`liquid_mamba.rs:749` types the field as `crate::geodesic_bridge::GeodesicBridge`).

Undetected because no CI leg compiles that feature pair, and **neither copy has any tests**.

> **Verification note, recorded because it matters more than the finding.** This was first
> reported as "confirmed" on the strength of reading three files, then "verified" by a compile
> that returned **exit 0** — because that run used only `code-sheaf-eval`, which gates
> `geodesic_bridge` but *not* `liquid_mamba` (gated on `mamba-cpu`). The call site never
> compiled, so no error could appear. A green result from a run that never exercised the thing
> under test. The provenance wrapper faithfully logged `exit_code: 0` — true, and useless, since
> provenance proves *which tree* compiled but not *whether the path was reached*. Only reading
> the log body caught it.

---

## Recommendations

### Retirement candidate: `symthaea-hdc-crypto`

Zero dependents in `Cargo.lock`; zero code references anywhere outside its own doc comments.
Content is a near-verbatim copy of the core module — the code-only diff reduces to an
import-path change, two constructor renames, and rustfmt wrapping. Compiles on every workspace
build via the `crates/core/*` glob. In-repo precedent exists (`symthaea-perception`, excluded so
"the workspace stops compiling a dead island").

**Confirm before deleting:** (1) author intent — the README frames *both* copies as quarantined
self-declared-broken research crypto, and "minimal-dependency copy for adversarial study" is a
defensible reading the code cannot settle; (2) no out-of-tree consumer (check the standalone
sync scripts, and query crates.io **with a User-Agent** — a bare `curl` 403s and reports every
crate unpublished); (3) no publication plan; (4) resolver-level confirmation via
`cargo tree -i` — every count here is grep-derived.

**Not a security fix.** The broken primitives remain live in `symthaea-core` behind an
unconditional `pub mod` with two real consumers. The *crate* has a feature gate; core does not.

### Low-risk migration: `symthaea-graph-theory` → `symthaea-core::hdc::graph_theory` (core wins)

Core has 3 live consumers, weighted edges, Dijkstra, the Laplacian/spectral suite, and
`Graph::encode() -> BinaryHV` — HDC integration the crate structurally cannot host. 25 tests vs
8. The crate has zero consumers, so **zero call sites change**. Work is a purely additive port
of four genuinely absent capabilities (PageRank, `degree_centrality`, Edmonds–Karp `max_flow`,
`connected_components`), ~200 lines of pure `std`.

**Required precaution:** `Graph::bfs(&self, usize) -> Vec<usize>` exists on both sides with an
**identical signature and different meaning** — core returns per-node distances (length `n`,
`usize::MAX` sentinel), the crate returns traversal order (length = reachable count). Rename at
port time; a mechanical import-rewrite would compile clean and be silently wrong.

**Explicitly rejected:** `hdc_ltc_unified` (19 importing crates; the crate has no
`backward`/gradients/SIMD/genesis-seeding, and swapping scalar for per-dimension gating would
silently alter the ~31 Hz loop the Keystone work just calibrated); `statistics` (core's
`variance`/`std_dev` are *population* estimators, the crate's are *sample* — a mechanical swap
flips ÷n to ÷n−1 with no compile error).

---

## The canonical-migration contract

Each rule cites the evidence that produced it. **Crate creation is not the deliverable;
canonical migration is.**

1. **An extraction commit must show deletions in the origin.** 0 deletions ⇒ it is a copy;
   reject. — `9baa58af21`: 4,647 insertions, 0 deletions.
2. **Definition-site cardinality must return to 1 for every public symbol**, verified by
   workspace-wide grep, before the extraction is called done. — The only check separating all
   four successes from all three failures.
3. **The origin may retain a namespace, never an implementation.** Only permitted residue is a
   `pub use` with a provenance comment. — Both successes ship exactly this. *Facades are not the
   failure mechanism and this contract must not discourage them.*
4. **Tests move with the code, in the same commit.** — `geodesic_bridge` has zero tests on either
   side, which is why a 2-vs-3-arg divergence went unnoticed.
5. **Every feature-flag path the extraction touches must be compiled by ≥1 CI leg, and the
   extracted crate must appear by name in CI.** — `code-sheaf-eval` is compiled by no job;
   `symthaea-broca-tools` appears in zero CI jobs. The same trap has bitten this crate twice.
6. **A cross-crate change is staged as one commit spanning both path prefixes.** — `df4d6c7f56`
   committed one half and left the other unstaged. (Since resolved by another session.)
7. **A dependency cycle is resolved by hoisting to a shared lower crate, never by re-duplicating
   into the origin.** — `geodesic_bridge` was re-added to `broca` as a documented "deliberate
   duplicate" whose stated justification is factually inverted; the copies diverged within three
   weeks into the E0061 above.
8. **Before creating a crate for a "missing" capability, search in-crate modules, not just the
   domain-crate index.** — Five crates, 48 hours, five false claims. `DOMAIN_CRATES_INDEX.md`
   indexes only `crates/domains/`.
9. **Name duplicates by abstraction, not identifier — the error runs both ways.** — `projection`
   shares zero abstraction yet was flagged; `game_theory` shares one identifier yet is the same
   abstraction. Known false friends in-tree: `GenerativeModel` (2 incompatible shapes),
   `RandomProjection` (**3**, two inside `symthaea-core`), `LearnedProjection` (2),
   `StructuralVerdict` (2), `bfs` (distances vs order), `partitions` (enumerate vs count).

---

## Limits — read before acting

**Nothing was executed by the audit agents.** All 12 were read-only; no `cargo build`, `test`,
`metadata`, or `tree`. Dependent counts are grep/manifest-derived, not resolver-derived. Test
counts are `#[test]` marker counts, not observed passes — `symthaea-statistics`' 70 tests import
~40 symbols the crate does not export, so its headline count is largely illusory.

**All twelve records self-report `confidence: high`. None is low.** That uniformity deserves
suspicion rather than reassurance. At least two contain material self-corrections showing the
first read was wrong. Read "high" as "the record's author was convinced," not as independent
verification. Residual doubt is highest on: `combinatorics` provenance (core and crate share a
*verbatim* comment line and a structurally identical loop — a reviewer could reasonably call it a
fork); `hdc_crypto` disposition ("incomplete migration" and "deliberately maintained quarantine
artifact" fit the same evidence, and both copies are patched in lockstep by the same commits).

**The primary failure mechanism rests on n=1.** Both confirmed incomplete migrations descend from
a single commit — one event, two artifacts, not two independent observations. This audit cannot
establish a base rate; the suspect list was name-matched and not drawn for prevalence.

**The true duplication count is higher than twelve pairs and this audit does not bound it.**
Unaudited leads: `symthaea-spore`'s self-declared WASM fork of `symthaea-fep` (811 lines, its own
`ActiveInferenceAgent`); `symthaea-hodge`'s **third** `consciousness_topology`, which is on the
*production* cognitive path via `cognitive_loop/consciousness_engine/topological_measure.rs`;
three `RandomProjection` structs; a third and fourth `ContinuousHV`; a reported second Dijkstra
in `symthaea-operations-research`.

**A reader must not conclude:**

- that nine duplicate pairs exist — **two** do;
- that `symthaea-core` is riddled with failed extractions — one commit produced both confirmed
  cases, and seven of nine suspects were never extractions;
- that the independent-same-domain crates are safe to delete — five contain real capability core
  lacks (graph-theory's PageRank/max-flow, game-theory's general-sum `mixed_nash_2x2`,
  statistics' exact Student's-t and χ² CDFs plus the layer the live `StatisticsDomainPlugin`
  depends on, combinatorics' exact `u128` where core silently saturates in `u64`, complex's
  `roots_of_unity`);
- that low adoption means dead — zero adoption is a *documented, intended* staging state for the
  `crates/domains/` family;
- that these findings are current — this tree moves fast under heavy concurrency. Re-verify.
