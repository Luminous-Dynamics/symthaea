# TEXTILE-MATERIAL-FLOW-001 — Lot lineage, split/merge and material conservation

**Status:** executable contract and synthetic regression suite added; hosted exact-head qualification pending. No physical lot-flow trial is claimed.

Related work:
- [TEXTILE-DESIGN-TO-RECOVERY-BENCHMARK-001](TEXTILE_DESIGN_TO_RECOVERY_BENCHMARK_001.md)
- [TEXTILE-SOUTH-AFRICA-CIRCULARITY-BASELINE-001](TEXTILE_SOUTH_AFRICA_CIRCULARITY_BASELINE_001.md)
- [TEXTILE-ROUTE-QUALIFICATION-001 #11898](https://github.com/Luminous-Dynamics/symthaea/issues/11898)
- [TEXTILE-MATERIAL-FLOW-001 #11998](https://github.com/Luminous-Dynamics/symthaea/issues/11998)
- [Symthaea sensing, design and recovery PR #11744](https://github.com/Luminous-Dynamics/symthaea/pull/11744)
- [Symtropy multi-stage factory verifier PR #4927](https://github.com/Luminous-Dynamics/symtropy/pull/4927)
- [Symtropy attachment-removal evidence PR #4945](https://github.com/Luminous-Dynamics/symtropy/pull/4945)

## 1. Paradigm shift: make material lineage the unit of truth

A single garment or process-trial record is not enough to qualify a circular supply chain. Physical material can be sorted into several streams, merged with other lots, split into batches, opened into fibre, rejected by a downstream receiver or left as an unresolved residual. If these transformations are not represented explicitly, a system can double-count an intermediate lot as if it were final output, or lose a portion of the incoming mass without an accountable disposition.

The benchmark should therefore treat an immutable **material lot and its transformation lineage** as first-class objects:

`incoming lot → sorting → split streams → repair / disassembly / opening → output lots → receiver decision`

A model prediction, process schedule and lot ledger answer different questions:
- sensing: what material identity is supported by the observations?
- design: what components and joining topology were declared?
- recovery: is one candidate route supported by physical evidence and receiver criteria?
- material flow: where did each measured mass fraction go across all transformations?
- scheduling: are only route-qualified jobs assigned to available machines, operators and utilities?

No layer should self-authorize the next.

## 2. Executable Rust contract

The new `symthaea_materials::textile_material_flow` module adds:
- immutable lot identities and integer-milligram masses;
- incoming lots bound to `PhysicalInventory` evidence;
- output lots bound to exactly one producing event;
- typed physical process events for sorting, repair, component disassembly, mechanical opening, chemical recycling and downcycling;
- explicit complete-lot inputs and newly-created output lots;
- single-consumption semantics: a lot cannot be used by two events; partial use requires an explicit split event and remainder lot;
- directed acyclic lineage and event chronology;
- per-event mass accounting:
  `sum(input lot mass) = sum(output lot mass) + explicit unaccounted residual ± measurement uncertainty`;
- separate uncertainty and maximum residual fractions in parts per million;
- terminal dispositions for awaiting receiver, accepted, rejected, quarantined and waste;
- exact lot-specific independent receiver acceptance before output is counted as accepted;
- independent receiving organization and verifier, distinct from the input/process issuer;
- receiver evidence that is later than and byte-distinct from the process/inventory evidence it judges;
- deterministic findings and terminal-only totals.

The `SortingTrial` evidence kind is explicit; a composition spectrum is not silently treated as proof that a physical sorting event took place.

## 3. Conservation and counting rules

Each event consumes complete lots. If only part of a lot is routed onward, the process must create two new lots (the routed portion and the remainder), with both masses accounted for. A lot ID is never reused after transformation.

For event (e):

[
M_{in,e} = sum_{i in Inputs(e)} M_i
]

[
M_{out,e} = sum_{j in Outputs(e)} M_j
]

[
|M_{in,e} - M_{out,e} - M_{unaccounted,e}|
le U_e
]

where (U_e) is the declared expanded mass uncertainty. A separate policy bounds both the uncertainty fraction and the unaccounted residual fraction relative to the event input.

Unaccounted residual is never accepted output. A process loss cannot be hidden by relabelling it as fibre, trim, component or recovered material. A receiving decision can qualify only the exact terminal output lot it inspected.

The report distinguishes:
- incoming source mass;
- terminal independently accepted mass;
- terminal receiver-rejected mass;
- terminal waste mass;
- terminal lots awaiting receiver decision;
- unaccounted residual across events.

Intermediate lots are not added to terminal output totals. This prevents the same material from being counted once after sorting, again after mechanical opening and a third time after spinning.

## 4. Synthetic known-answer regression

The multi-stage fixture declares:
- 10,000 mg incoming textile;
- sorting outputs: 6,000 mg cotton stream + 3,800 mg polyester stream;
- sorting residual: 200 mg;
- separate downstream routes: 5,900 mg mechanically opened cotton + 100 mg residual, and 3,730 mg downcycled polyester + 70 mg residual.

Expected terminal queue mass is **9,630 mg**, not 19,430 mg or 25,430 mg. The 6,000 mg and 3,800 mg sorting outputs are intermediate lots, not additional final products. Total event-unaccounted residual is **370 mg**. These are synthetic test values, not a measured physical result.

Negative controls cover:
- the same incoming lot consumed by two events;
- mismatched event/output producer identity;
- unexplained mass beyond declared uncertainty;
- unaccounted mass above policy;
- a lot that disappears without a consumer or terminal disposition;
- event chronology reversal;
- cyclic event lineage;
- synthetic/simulation evidence promoted as a physical process event;
- conditional receiver disposition promoted as accepted;
- receiver issuer/verifier not independent from the process issuer;
- receiver decision before the producing trial;
- receiver evidence reusing the process/inventory bytes.

## 5. Benchmark targets — proposed, not achieved

After real-data and measurement-repeatability baselines are available:
- 100% of sampled incoming lots have a unique ID, custody record and measured mass;
- 100% of physical transformation events bind to exact input/output lot IDs and process evidence;
- zero double-consumed lot IDs;
- zero event-graph cycles or producer ambiguity;
- 100% explicit split/merge accounting;
- zero accepted output without independent, lot-specific receiver acceptance;
- 100% of event mass residuals and uncertainty are represented separately;
- no intermediate lot is counted as terminal accepted output;
- report terminal accepted mass, rejected mass, waste, awaiting-receiver mass and unaccounted residual by route and material class;
- 100% of physical output-lot claims link to the exact upstream trial, mass balance, receiver specification and independent decision.

Do not optimise a single recovery percentage that trades away mass closure, product grade, uncertainty, labour conditions or receiver acceptance.

## 6. South African pilot relevance

South Africa's circular-textile opportunity requires connecting collection, reuse/repair, sorting, recovery and local downstream buyers—not simply importing a visual classifier. GreenCape's May 2026 work describes a collaborative, industry-led voluntary circular-textile initiative and frames value retention across clothing, components and fibres. UNEP's InTex Phase II includes South Africa through 2028, with CARES and CSIR partners and work on a national textile life-cycle inventory and practical approaches to textile EPR.

These programmes are potential data-alignment and collaboration contexts, not evidence of a Luminous partnership or a currently available dataset. South Africa's 2026 proposed consolidated EPR regulations were published for comment with a 5 December 2026 deadline; the draft must not be treated as a final textile-specific obligation. Keep jurisdiction, legal source notice, status, effective date and version in a separate legal provenance layer.

Sources:
- GreenCape, *Developing a voluntary circular textile initiative for South Africa* (4 May 2026): https://greencape.co.za/library/developing-a-voluntary-circular-textile-initiative-for-south-africa-potential-pathways-and-value-proposition/
- UNEP InTex Phase II — Africa (2025–2028): https://www.unep.org/en/intex
- South African proposed EPR Regulations 2026, Government Notice 8145 / Gazette 55679 (5 November 2026): https://www.gov.za/sites/default/files/gcis_document/202611/55679gon8145.pdf
- GreenCape, *Circular Economy for South Africa's Textiles Industry* (May 2026): https://greencape.co.za/wp-content/uploads/2026/05/CE-for-South-Africas-textiles-industry.pdf

## 7. Claim ceiling

The contract validates declared lot lineage, event mass accounting, receiver identity and terminal output semantics. It does not authenticate the evidence bytes, certify fibre composition, establish chemical purity, prove worker safety, determine legal compliance, establish environmental benefit or prove that any physical process was performed. Those claims require the source artifacts, calibrated instruments, independent inspectors, physical process trials and actual receiving-process acceptance.
