# FactoryBench SOTA comparison and co-optimization plan v0.2

**Owner:** ATELIER-FACTORYBENCH-001 #12289  
**Status:** literature-informed benchmark plan; no Atelier performance result  
**Machine-readable matrix:** `sota-comparison.v0.2.json`

## 1. What current research changes

The first FactoryBench corpus compared four factory architectures. That is necessary but not sufficient. Current apparel literature studies several optimization layers separately:

1. Pattern dependencies and seam-level work content.
2. Fabric cut-order planning and marker yield.
3. Machine allocation and multi-line scheduling.
4. Facility layout and material flow.
5. Supplier/outsourcer allocation and dynamic order assignment.
6. Discrete-event behavior under variability and order insertion.

A stronger benchmark must connect these layers while preserving the scope of each method and its hard feasibility constraints.

## 2. Reference papers and fair comparison

| Reference | Method and task | Reported result | Correct use in FactoryBench |
|---|---|---|---|
| Textile SLP, 5 Mar 2026 | SLP + AHP/TOPSIS, textile factory layout | 45% lower average travel time and 34% lower travel distance in that case | Reconstruct its From-To and relationship method; no transfer claim |
| FDSM apparel GA, 9 Feb 2026 | Fuzzy design-structure matrix + GA for shirt assembly layout | 42% objective reduction; 15.37% workload-balance improvement; four fewer machines; 11 extra simulated products/day | Reproduce exact objective, manual baseline and FlexSim task before comparison |
| Pattern-driven GA, 1 Sep 2026 | CAD seam workloads + machine-specific sewing speed + GA + event simulation | Directional gains in time/workload; no portable numeric effect extracted here | Compare exact pattern revision, machine assignments and urgent-order insertion |
| Cut-order PSO/GA, Mar 2026 | Fabric/marker/cutting/shortage/overproduction cost model | 32 scenarios; competitive cost reduction reported | Add marker and cut-yield layer; compare both algorithms and incumbent |
| Multi-line scheduling, Feb 2026 | MILP benchmark + enhanced GA, lot streaming and heterogeneous skills | MILP supplies exact small-instance reference | Compare heuristic/GA gaps, workforce feasibility and makespan |
| Green textile supplier allocation, 5 Oct 2026 | Dynamic order allocation over multiple outsourcing entities; green criteria lexicographically applied | Method-level comparator | Add make/buy/partner allocation while keeping green preference behind technical feasibility |
| Integrated layout, 21 Jul 2026 (transformer factory) | SLP → discrete-event simulation → GA → simulation | 35% lower material handling cost; 2.46% higher monthly throughput in that non-textile case | Cross-industry method comparator only, not a textile target |

Full source identifiers, caveats and reproduction gates are in the JSON matrix.

## 3. Paradigm shift: product-to-factory co-optimization

Do not ask one monolithic optimizer to solve all of apparel manufacturing. Use a **feasibility-first hierarchy** with explicit interfaces and iterative feedback:

```
Pattern revision / seam graph
        ↓
Seam workload and operation dependencies
        ↓
Cut order / marker / roll and material-lot plan
        ↓
Machine + tooling + worker-skill allocation
        ↓
Facility layout + typed travel network
        ↓
Supplier / CMT / partner allocation
        ↓
Multi-line schedule / lot streaming / inspection
        ↓
Discrete-event variability / urgent-order insertion
        ↓
Measured production and quality feedback
```

A later stage may send a constraint back upstream. For example, a machine bottleneck may motivate a different seam construction; an unavailable fabric width may change the marker and pattern; an urgent order may move an operation to a qualified partner; a repair-return surge may change the reverse-flow layout.

These feedback changes create new design/process revisions. Historical approvals are not silently carried forward.

## 4. Baselines and algorithms

For every subproblem:
- establish a transparent manual or SLP baseline;
- implement a deterministic greedy heuristic;
- use exact MILP/brute-force methods on small cases as a correctness/optimality oracle;
- reproduce the paper-specific GA/PSO method when the task definition matches;
- test larger instances and report runtime, optimality gap, variability and constraint violations.

Genetic algorithms are not inherently better than exact methods or simple heuristics. The benchmark should reward solution quality and feasibility, not algorithm novelty.

## 5. Hard feasibility gates before optimization

These are non-compensatory:
- safety, egress, accessibility and worker protections;
- machine/process capability and calibration;
- worker skill/calendar and non-overlapping assignments;
- operation precedence and complete operation coverage;
- material lot, recipe and fabric-width compatibility;
- required in-process and final inspection authority;
- qualified supplier/partner routes and transport/lead-time assumptions;
- repair/returns segregation;
- privacy and ownership constraints.

A weighted score cannot compensate for an infeasible layout, an unqualified machine, a missing inspector or a prohibited material route.

## 6. Report a Pareto frontier, not one factory score

Report separately:
- lead time and makespan (P50/P95 where stochastic);
- material and person travel;
- WIP, queues, bottleneck utilization and workload balance;
- machine count, setup/changeover and reconfiguration time;
- first-pass yield, defects, rework and scrap;
- marker/fabric yield, shortage and overproduction;
- accepted output per machine-hour and labour-hour;
- fully loaded cost per accepted garment;
- energy/water/material per accepted unit with declared boundaries;
- recovery time and backlog under disruptions.

Keep simulation estimates, paper-reported values, vendor targets and physical measurements in separate evidence classes.

## 7. FactoryBench v0.2 work packages

1. **Pattern workload bridge:** consume the exact garment revision and typed operation graph; preserve seam lengths, stitch/operation class and machine assumptions from Sewing Plan Compiler #12134.
2. **Cut-order benchmark:** add sizes/colours, order demand, fabric-roll widths/lots, marker plans, shortages, overproduction and offcut accounting.
3. **Resource allocation:** compare FDSM/manual, pattern-driven GA, heuristics and exact small-instance oracles.
4. **Layout and movement:** use the new integer-distance layout evaluator; keep material, WIP, return and people movement distinct.
5. **Partner allocation:** add internal vs CMT/knit/finish partner capabilities, lead times, transport, cost and lot traceability.
6. **Multi-line scheduling:** compose FACTORY-OPS #7677 and Symtropy #4875; compare MILP small cases and heuristics/GA larger cases.
7. **Event simulation:** urgent order insertion, supplier rejection, machine outage, worker absence, inspection hold, rework and returns surge.
8. **Physical calibration:** replace synthetic inputs only when time studies, geometry, machine and lot evidence are available; preserve synthetic benchmark versions.

## 8. South African programme correction

The current dtic programme page is the **Clothing, Textiles, Footwear and Leather Growth Programme (CTFLGP)**, not the legacy CTCP. Its revised guidelines (effective 9 July 2024) explicitly state that CTFLGP replaced CTCP, and list competitiveness improvement, expansionary working capital, start-up funding (including early-stage businesses and fashion designers), and cluster funding. The guidelines also say resources are identified annually and may be amended. Treat the programme as a potential enquiry route only; confirm current intake, eligibility, qualifying costs and funding terms with the CTFLGP Desk/IDC.

Official page: https://www.thedtic.gov.za/financial-and-non-financial-support/incentives/clothing-textiles-footwear-and-leather-growth-programme-ctflgp/

Guidelines: https://www.thedtic.gov.za/wp-content/uploads/Guidelines.pdf

## 9. Claim ceiling

This is a benchmark design and literature reproduction plan. No reported paper result is an Atelier result. Synthetic inputs are not observed demand or manufacturing times. Simulation is not physical qualification. A factory layout score does not establish safe occupancy, worker wellbeing, cost competitiveness, environmental benefit or production capability.
