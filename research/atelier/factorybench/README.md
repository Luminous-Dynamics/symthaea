# Atelier FactoryBench v0.1 synthetic corpus

This directory is the initial, synthetic-only case corpus for ATELIER-FACTORYBENCH-001 #12289.

- factory-bench-corpus.v0.1.json — four product operation graphs, synthetic quantities/work ticks, capability/skill pools, four layout alternatives and twelve disruption cases.
- factory-from-to-matrix.v0.1.json — deterministic aggregation of product routes into synthetic zone-to-zone move counts.
- README.md — evidence and interpretation boundaries.

Important:
- Synthetic work ticks are not minutes or standard allowed minutes (SMV).
- Synthetic batch quantities are not customer demand.
- No floor geometry, distances, costs, utility data, equipment inventory or measured capacity is supplied.
- A layout comparison must use identical product/operation/demand/skill/quality constraints for all four alternatives.
- Physical factory claims require measured data and professional site/facility/safety qualification.
