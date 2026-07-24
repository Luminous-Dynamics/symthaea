# Muse Provenance & Federation: Reality and Boundaries

**Status:** Reality-check / terminology reference, not a design spec.
**Purpose:** Keep "what Muse's genealogy/provenance work actually does today" and
"what a future federated publication layer might look like" from blurring
together in conversation or docs. Written after a review session
(2026-07-23/24) that initially described a Holochain → DHT → DKG → Mycelix →
RHN publication architecture as though parts of it already existed. They
don't. This file is the correction, kept short on purpose — see
"Non-goals" below for why it isn't a bigger spec.

---

## 1. Implemented now

| Capability | Where |
|---|---|
| Ephemeral in-session candidates | `muse_studio`'s `Studio.candidates: Mutex<HashMap<u64, Candidate>>` — ids reset every restart, evicted under memory pressure |
| Durable keepers | `data/taste/keepers.jsonl` (append-only, atomic replace) + `data/taste/audio/<key>/...` bundle directories |
| `PieceRecipe` | `symthaea-muse/src/piece_recipe.rs` — deterministic `(intent, resolved_spec)` replay |
| Artifact/provenance identities | `PieceProvenanceBundle`, `ReproducibilityClaim` (`symthaea-muse-protocol`), served at `GET /api/piece/{id}/provenance` |
| Local analyst verification | `symthaea_muse::analyst::analyze_piece`/`analyze_audio_integrity`, served at `GET /api/piece/{id}/analyst` |
| **SQLite genealogy ledger** | `symthaea_muse::genealogy::GenealogyStore` — allocates a `Root`-relation manifest per kept piece, at keep time, in `data/genealogy/ledger.sqlite3`. Served at `GET /api/genealogy/{id}`, `/children`, `/ancestry`. `keeper()` now returns the manifest in its response body instead of bare `204`. |

What the genealogy ledger deliberately does NOT do yet, and why:
- No non-`Root` relation is ever constructed — nothing in the compose/keep
  flow tracks "this kept piece was derived from that kept piece."
- No multi-parent/DAG edges — no feature in this codebase produces
  multi-source derivation (sampling, mashup) to need one.
- No branch-ordinal allocation per `(parent, namespace)` — moot while every
  manifest is a root.
- No content-derived family key or decimal human-facing address — no UI
  surface renders one yet.
- `data/taste/audio/<key>/` is **not** content-addressed (`keeper_artifact_key`
  is `(unix_nanos, pid, candidate_id, sequence)`, not a hash of content) —
  the genealogy ledger stores and verifies the real sha256 hashes without
  assuming the underlying storage is CAS.
- `keepers.jsonl` remains the operational keeper index; SQLite is *not* a
  drop-in replacement for it yet (that's a distinct future refactor, not
  bundled into this work — see §4).

## 2. Grounded future design (real documents, marked proposed)

- `MUSE_ARTIST_CAPABILITY_AND_PUBLISHING_INTEGRITY_DESIGN_SPEC.md` — **Status: Proposed** (stated at the top of the file). §10 "DKG provenance model" describes
  a claim graph of works/versions/arrangements/performances/rights, correctly
  scoped as "proves who asserted what and with what evidence," not "proves
  legal ownership." Real design thinking; zero implementation.
- `IMPROVEMENT_PLAN.md` (lines ~1303-1332) references the same idea, and
  explicitly assumes `mycelix-attribution` and `mycelix-knowledge` as
  already-built clusters to bridge into.
- A Holochain-backed federated artist-network design (agent-centric source
  chains, public/private data boundaries, catalog/settlement on Holochain)
  exists in separate proposed Mycelix Music documents. It is real prior
  design work — but it is a Mycelix Music proposal, not a Muse Studio
  dependency, and nothing in `symthaea-muse` calls into it.

## 3. What's actually missing (verified against the filesystem, not doc claims)

- **`mycelix-knowledge/` does not exist** in the current checkout — only a
  stale CI build-cache remnant. CLAUDE.md's "Built" status for this cluster
  is stale.
- **`mycelix-attribution/` does not exist** either, despite being cited as a
  built cluster in Muse's own improvement plan.
- **`symthaea-muse` has zero Holochain dependency** — no `hdk`/`hdi`/
  holochain reference anywhere in its `Cargo.toml` or source tree.
- No network publication outbox, no signed export manifest, no Mycelix
  identity/consent/rights wiring exists in Muse today.

**Rule going forward: the current filesystem, `cargo metadata`, and passing
tests are authoritative over narrative "Built" labels in CLAUDE.md or plan
docs.** If a doc says a cluster is built, verify it exists before relying on
that claim (this file's own §3 is an example of that check actually being
done).

## 4. Acronym ownership (the collision that made this file necessary)

| Term | Real meaning in THIS monorepo | Where |
|---|---|---|
| `DKG` | **Distributed Key Generation** — Feldman threshold signing, a real, shipped governance feature | `mycelix-governance`, `symtropy/src/systems/dkg_ceremony.rs` |
| `RHN` | **Resonant Hypergraph Network** — a Symthaea-core HDC research concept about knowledge topology/routing inside the cognitive loop | `symthaea-core/src/hdc/cantor_pyramid.rs`, `symthaea-broca/src/highway_projection.rs` |

Both `DKG` and `RHN` are **already spoken for** by real, unrelated, built (or
partially built) subsystems. A future Muse/Mycelix provenance-claim layer
must NOT reuse either acronym. Until code exists, refer to it in plain
language: **"Mycelix claim graph"** (not "knowledge graph" — it stores
assertions, evidence, contradiction, and supersession, not settled facts).
Candidate crate names for later, not claimed or reserved by anything today:
`mycelix-provenance` (Muse-facing) / `mycelix-claims` (general substrate).

RHN's real, legitimate future role here — if and when this is ever built —
is advisory retrieval only (motif-neighborhood search, ranking evidence for
human review), never the authority over which claim is true. Claims and
evidence live in the claim graph; RHN, if used, only helps navigate them.

## 5. Sequencing (each step needs a real consumer before the next is built)

```
local durable genealogy (done)
→ explicit derivation workflows (not started — needs a UI/API action that
  records "derived from")
→ imported-source provenance (needs an actual sample-import feature first)
→ deterministic export bundles
→ signed manifests
→ local publication queue / outbox
→ actual Mycelix claim-graph substrate (needs mycelix-provenance built)
→ federation adapter (Holochain or otherwise — undecided, unbuilt)
→ RHN-assisted retrieval over retained claims
```

Imported samples are the most likely first real trigger for multi-parent
lineage, source rights, and contested derivation — that's a stronger reason
to build a DAG/edges table than doing it speculatively now.

## 6. Non-goals of this document

This file is not a design spec for the claim-graph/federation layer — that
already exists, marked Proposed, in §10 of
`MUSE_ARTIST_CAPABILITY_AND_PUBLISHING_INTEGRITY_DESIGN_SPEC.md`. This file
exists only to keep "implemented" and "proposed" from blurring together in
conversation, and to record the acronym collision so it isn't repeated. If
work actually starts on the claim-graph layer, that's a new plan document,
not an expansion of this one.
