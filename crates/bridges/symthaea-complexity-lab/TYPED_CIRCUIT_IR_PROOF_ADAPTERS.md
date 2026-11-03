# ETH-026: typed tractable-circuit IR and proof-format adapters

**Status:** Versioned research IR and exact Python reference checker committed. Eight finite semantic/schema fixtures executed locally against the pinned Git blob. No CPOG, VeriPB, EVDD or tensor-network adapter is implemented yet.

## 1. Why this changes the architecture

The objective is not to invent another isolated solver or universal proof format. It is to compile exact finite factor systems into a shared typed circuit representation and connect that representation to established proof and tractable-circuit ecosystems where their guarantees match.

The IR binds the requested output semantics, coefficient domain, variable/quantifier scope, circuit structure, and source provenance. An adapter must refuse translation when it would silently lose a guarantee.

## 2. Prior art and architectural implications

### #SAT proof systems

The 2026 *Relative Strength of #SAT Proof Systems* paper establishes that CPOG p-simulates both MICE and kcps, while MICE and kcps are exponentially incomparable. CPOG is therefore the strongest of these three under the established proof-simulation ordering. This is a theoretical relation, not a measured checker-runtime result:
- https://doi.org/10.1007/s10817-026-09757-w
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13385077/

The AAAI-26 work connects MICE and annotated Decision-DNNF:
- https://doi.org/10.1609/aaai.v40i17.38431

### Projected enumeration

The CP2026 VeriPB work extends proof logging to projected enumeration and uses the formally verified CakePB checker/elaborator for verified enumerations:
- https://doi.org/10.4230/LIPIcs.CP.2026.43
- https://eprints.gla.ac.uk/386496/

This is directly relevant to enumerating visible solutions. It does not establish that a generic compact PMC count or signed/weighted PWMC value already has a VeriPB proof format. Do not overstate the adapter.

### Weighted circuits and tensor networks

The KR2026 work gives constructive correspondences: tensor trains to nondeterministic edge-valued decision diagrams, and tree tensor networks to edge-valued structured-decomposable circuits. Its framework includes real, negative and complex weights:
- https://doi.org/10.24963/kr.2026/52
- https://proceedings.kr.org/2026/52/kr2026-0052-quist-et-al.pdf

This gives ETH-019/020 a path to weighted tractable-circuit representations. It is prior art, not an invention claim.

## 3. IR v1 reference subset

Files:
- \`reference/typed_circuit_ir.py\`
- \`reference/typed_circuit_ir_schema.json\`
- \`reference/typed_circuit_ir_results.json\`

Header:
- schema ID;
- output contract;
- declared variable universe;
- visible-variable set;
- coefficient-domain identifier;
- topologically ordered node list;
- root node.

Node kinds:
- indicator literal;
- exact constant;
- decomposable product;
- weighted sum, optionally carrying a determinism assertion.

Checked invariants:
- unique node IDs and backward-only edges;
- exact coefficient-domain match;
- canonical reduced rational encodings;
- pairwise disjoint child scopes for product gates;
- extensional disjointness for sums marked deterministic;
- projected contracts require a visible-variable subset;
- PWMC weights apply to distinct supported visible assignments once, after hidden witness collapse.

The root can have a scope smaller than the full variable universe. Omitted variables remain free and are enumerated by the reference evaluator.

## 4. Executed local experiment

The committed file was fetched from GitHub and its exact text was executed locally under Python 3.13.5. The Git blob SHA matches the executed source: \`1d22ecdc1a9d8ed5b174471ee204d9a8d6ac9350\`.

8/8 fixtures passed:
1. #SAT with one hidden free variable: 2.
2. PMC over the same circuit: 1 distinct projection.
3. PWMC on that projection: exactly 2/3.
4. Signed rational WMC cancellation: exactly zero.
5. Complex Q(i) AMC cancellation: exactly zero.
6. Reject overlapping product scopes.
7. Reject a false determinism claim.
8. Reject a Boolean support circuit mislabeled as complex AMC.

The deterministic-sum assertion is checked extensionally on every assignment of its finite scope. This is a tiny reference validation, not a proof for all inputs.

## 5. Adapter boundaries

| Target | Intended adapter | Current boundary |
|---|---|---|
| CPOG | Export/import proof-friendly #SAT circuits | Not implemented; no proof log emitted |
| MICE / annotated Decision-DNNF | Interchange #SAT circuit structures | Not implemented |
| VeriPB / CakePB | Export projected enumerations with the supported VeriPB proof contract | Not implemented; no claim for compact weighted counts |
| EVDD / EV-SDNNF | Weighted exact function circuits over Q or Q(i) | Not implemented; exact evaluator only |
| Tensor train / tree tensor network | Import/export structured decompositions with scope and vtree witnesses | Not implemented |
| ETH-022 replay trace | Convert verified elimination steps into a circuit plus gate-evaluation trace | Not implemented |

## 6. Next research and benchmark gates

1. Add strict schema validation and canonical serialization with input/root digests.
2. Add a direct truth-table oracle for every small circuit and property-based circuit generation.
3. Implement one adapter at a time: first #SAT structure export/import; then VeriPB projected enumeration; then exact EVDD/EV-SDNNF weighted Q(i) evaluation; then tensor-network metadata.
4. Every adapter must preserve output contract, coefficient domain, visible/hidden partition and exactness.
5. Compare circuit node count, translation overhead, evaluation cost, proof bytes, checker time/RSS, coefficient bit-height, width and memory.
6. Compare theoretical proof-simulation statements separately from practical runtimes and proof sizes.
7. Keep exact modes separate from probabilistic/approximate modes.

Acceptance:
- 100% small truth-table equivalence for supported translations;
- zero nondecomposable product gates accepted;
- no determinism assertion accepted without a checkable disjointness witness;
- zero coefficient-domain reinterpretations;
- zero hidden-witness multiplicity in PMC/PWMC;
- exact Q(i) cancellation to zero;
- all raw instances, circuits, proof logs, checker outputs and version identifiers pinned.

## 7. Claim ceiling

ETH-026 currently demonstrates an exact typed-circuit reference subset over Boolean, Q and Q(i), with eight finite fixtures. It is not an implemented CPOG or VeriPB adapter, a general weighted proof system, a formally verified translation, a scalable solver, or a SOTA result.
