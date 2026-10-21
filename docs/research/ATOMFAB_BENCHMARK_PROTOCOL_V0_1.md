# AtomFab Benchmark Protocol v0.1

**Status:** pre-registered experimental plan, not a claim of industrial performance.

## Objective

Measure whether the Symthaea/Symtropy capability planner reduces the time, cost, waste, or dependence required to deliver a conforming electronic assembly **without relaxing the product requirements**. Report the result as a set of trade-offs and confidence intervals, not a single unqualified "better than SOTA" score.

## Comparison contract: compare like with like

Freeze a benchmark generation before running it:

- Product identity, revision, native design/schematic, frozen netlist, BOM/approved substitutions, firmware/test profile, quantity and acceptance criteria.
- Board material, stackup, required copper geometry, holes/plating, soldermask, component package constraints, and any impedance or environmental requirements.
- Process route, machine/tool profile, operator skill level, equipment calibration, material/lot IDs, tool and command versions.
- Time/cost boundaries, labour rate, currency/date, lot size, shipping and queue-time measurement.
- Required inspection coverage, acceptance class, rework policy, and definition of a unique accepted assembly.

A route is not eligible for the economic comparison until it meets the same frozen requirements. An isolation-milled single/double-layer prototype that cannot meet a plated-through-via requirement is *infeasible for that benchmark*, not a cheap winner.

## Benchmark ladder and gates

### A. Software capability/evidence contract

1. Grow the frozen known-answer corpus toward at least **100 cases**, deliberately stratified across feasible, infeasible, unresolved, incomplete-evidence and rejected-input states.
2. Require **100% expected-disposition agreement** on the frozen corpus; every intentional malformed-input, stale-revision, invalid-digest, unsafe-path and false-authority mutation must be rejected or explicitly incomplete.
3. Require **zero false accepts** in an adversarial suite of at least 50 independently generated mutations. Preserve every mutation as a regression case after discovery.
4. Receipts must bind exact Git commit, corpus/manifest digest, tool versions and disposition. Deterministic fields must replay byte-for-byte; nondeterministic output must be segregated and explicitly normalized, not silently ignored.
5. A mutated expected label must not make an implementation pass. Expected labels cannot be consumed as the evaluator's decision input.

### B. Native EDA handoff

- A clean native KiCad board must produce a parseable DRC report with exit code 0 under a pinned toolchain.
- A deliberately invalid shorted-net control must produce a parseable report and the documented DRC violation exit code.
- Expected Gerber layers and Excellon drill files must exist, be non-empty and checksum-bound.
- Repeated DRC and exports from identical inputs/toolchain must be byte-identical or have a documented source of nondeterminism.
- This establishes an EDA-tool contract only. DRC alone does not prove the intended schematic matches the board, that the board can be fabricated to tolerances, or that a physical board is conforming.

### C. Discrete-event / scheduling experiments

- Compare FIFO or the actual current workflow against the intervention on exactly the same frozen jobs, stations, buffer capacities, service distributions and release conditions.
- Use paired seeds/entity-keyed random streams to reduce comparison noise; run at least **100 replications** for a first robust simulation report.
- Report throughput, flow time, WIP, starvation/blocking, changeover time, missed due dates, utilization, and cost per accepted assembly where the model has a justified cost model.
- Report baseline and treatment estimates, paired differences, 95% confidence intervals, seed count, rejected/error runs and sensitivity to key parameters.
- Synthetic event parameters demonstrate logic only. Before forecasting a real cell, calibrate against timestamped physical/MES observations and hold out separate runs for validation. Report residuals and prediction intervals; do not tune and claim validation on the same trace.

### D. Physical pilot and confirmation

**Pilot:** begin with 10 attempted assemblies per eligible route to expose handling, setup, data-capture and acceptance-test problems. Pilot results are for procedure development, not qualification.

**First confirmation:** target at least 30 attempted assemblies per eligible route, spread across at least three separate run/batch opportunities where practical. Freeze design/process/acceptance criteria first; log all starts, failures, scrap, rework and final dispositions. Blind the independent inspector to route identity where practical. Report first-pass yield, final yield, the raw numerator/denominator and a binomial confidence interval. Treat this as an early estimate, not a high-reliability demonstration.

The sample size must follow the risk claim. For zero observed failures in independent trials, the one-sided 95% upper bound on failure probability is approximately the rule of three, (3/n). Thus zero failures in 30 trials still permits an upper bound near 9.5%; to put that upper bound below 1% requires **299** independent failure-free trials. Independence, representative lots, test coverage and the applicable product/industry sampling standard still matter.

### E. Economics and lifecycle impact

Primary economic metric:

`cost per unique accepted assembly = total in-scope route cost / number of unique assemblies accepted at final test`

Include setup/NRE (reported separately and amortized under declared volume scenarios), components/material, bare board, labour, machine time, tools/consumables, inspection/test, depreciation, utilities, packaging/shipping, queue delay if costed, scrap and rework. A route with zero accepted assemblies has an undefined/infinite cost per accepted assembly and fails the quality gate; it must not be assigned a zero cost or silently removed from the comparison.

Keep supplier quote, invoice, public list price, measured observation, estimate, synthetic parameter and unknown as distinct evidence classes. Unknown is not zero. Compare quotes/invoices at the same board revision, quantity, acceptance standard and delivery terms, timestamp the quote, and report one-off/NRE separately from recurring cost. Report local work, outsourced work, imports, unavailable capabilities and qualified capabilities by subsystem; a locally assembled board is not locally fabricated semiconductor silicon.

## Industry / state-of-the-art comparisons

Use a hierarchy of baselines, each on the same frozen product requirements:

1. Current manual or local bench process (measured, with all starts and labour captured).
2. Current scripted CAD/export/test pipeline.
3. Eligible local prototype route and eligible outsourced bare-board/assembly route.
4. Supplier or contract-manufacturer quote/invoice and quality data at comparable volume/class.
5. Symthaea-guided route selection and Symtropy scheduling, compared against a simple human-designed route, FIFO dispatch, and a declared generic search baseline.

For electronics assembly, collect the same dimensions used in industrial quality work: first-pass yield, defects/defect opportunities, final yield, scrap/rework, returns, and on-time delivery. Do not claim a percentile or superiority from an industry study unless the full benchmark definitions, numerical results, acceptance classes and denominator are actually available. IPC's public quality-study announcement notes 59 assembly companies; its detailed benchmark distributions are not a substitute for the user's own comparable run data.

References: [KiCad 9.0 CLI manual](https://docs.kicad.org/9.0/en/cli/cli.html); [official pinned KiCad 9.0.9 Linux/amd64 image](https://hub.docker.com/layers/kicad/kicad/9.0.9/images/sha256-cef4a572df1465b98fa63cdac88780e0f8d39f03c5493434802bcd92f62123fb); [NISTIR 8620 — Digital Twins Workshops Summary Report (2026)](https://doi.org/10.6028/NIST.IR.8620), which identifies interoperability and verification, validation and uncertainty quantification as manufacturing digital-twin priorities; [IPC quality benchmarking study announcement](https://www.ipc.org/news-release/ipc-releases-quality-benchmark-study-electronics-assembly); [IPC-6012F rigid-board qualification](https://www.ipc.org/news-release/ipc-releases-ipc-6012f-qualification-and-performance-specification-rigid-printed); [IPC J-STD-001J / IPC-A-610J assembly standards update](https://www.ipc.org/news-release/ipc-releases-j-revisions-two-leading-standards-electronics-assembly).

## Promotion gates

- **Software pass**: all frozen and adversarial expected behaviors pass on exact head.
- **EDA pass**: native clean/violated controls, expected exports and replay pass on exact head.
- **Simulation pass**: paired comparison is repeatable and reports uncertainty; real-world forecasting is prohibited until calibration and held-out validation exist.
- **Physical pilot pass**: the declared board profile, inspection and functional tests pass with complete denominators; still not an industry qualification.
- **Qualification / production release**: separate authorized engineering review against applicable product, manufacturing and safety standards, supported by physical test evidence. No preceding gate grants machinery authority.

A successful benchmark may still conclude that outsourcing is preferable. The target is reliable productive closure and a defensible decision, not forcing every process to become local.


## F. Copper-trace routing is a separate benchmark domain

Do not conflate **manufacturing-process route selection** (local vs outsourced fabrication/assembly and cost per accepted assembly) with **PCB copper-trace routing** (connecting pads/nets while satisfying DRC). The former is covered by PCB-CELL-001 / PCB-COST-001; the latter has its own new Symtropy program: [PCB-ROUTE-BENCH-001 #2473](https://github.com/Luminous-Dynamics/symtropy/issues/2473).

### State-of-the-art reference

[PCBWorld](https://github.com/LGAI-Research/PCBWorld) and its [PCBWorld-Bench paper](https://arxiv.org/abs/2607.05915) (July 2026; KDD 2026 Workshop) provide an open, engine-grounded benchmark. The paper documents grid-based and gridless synthetic families plus **679 real open-source boards**, scored by the same native KiCad routing engine and DRC. Its comparison includes rule-based routing, reinforcement-learning policies and tool-using LLMs; it reports that large-board routing remains difficult and learned methods can trail strong rule-based routers on larger boards. A separate 2026 DreamerV3 + FreeRouting paper reports 96% completion in its own setup. These published numbers are **not directly comparable** until we reproduce the same dataset split, engine build, DRC counting policy, timeout and evaluation protocol.

### Experiment contract

Start with six known-answer synthetic controls, then a declared, stratified pilot of 30 board-licensed real examples, and expand only after the pipeline is reliable. Compare available pinned rule-based baselines and Symthaea-guided closed-loop decisions on the *same starting boards*. Keep native KiCad PNS/DRC as geometric authority; do not create a second grid abstraction or train against DRC-free rewards alone.

An instance counts as valid only when every required net is connected and the routing-relevant DRC policy reports zero violations, including crossings/shorts, clearance errors, dangling segments/vias and net conflicts. Record upstream non-routing violations separately. Report invalid/timed-out boards, not just successful runs. For each result retain completion/reason, unrouted nets, violation types, wirelength, via count, elapsed/CPU time, memory, engine calls and input/output digests.

Optimize lexicographically: (1) zero routing violations and all required nets connected; (2) lower runtime at equal validity; (3) Pareto improvement in copper wirelength and via count. Compare per-board outcomes, completion percentage, median/p90 runtime and paired 95% confidence intervals. Stratify by source board, routing family and board complexity; use held-out board-source families to limit duplicate leakage.

**Proposed stretch target, not a result:** on a frozen held-out split, seek either a statistically supported absolute completion improvement of at least 5 percentage points or a Pareto improvement at equal completion. Among boards completed by both methods, seek at least a 10% lower median runtime without worsening validity; report wirelength/via changes and 95% paired intervals. Do not claim SOTA superiority unless the complete pinned comparison supports it.

### Integration and licensing boundary

Use PCBWorld and its engine as explicitly pinned external processes. The published environment repository is BSD-3-Clause while the separate engine/submodule is GPLv3; individual board fixtures retain their source licenses. Do not vendor/link engine internals into production Symthaea/Symtropy or build a combined image. A Rust-first orchestrator can launch declared external commands and verify outputs without making Python a production dependency. Every run receipt binds exact source commit, environment/engine/KiCad versions, dataset manifest digest, command/configuration and output hashes.

A passing routing benchmark still does not certify impedance, crosstalk, SI/PI, thermal behavior, manufacturing yield or physical board quality. Those require separate design analyses and physical validation.
