# SAFRB shared-pump decision-aware evidence-acquisition case

This synthetic fixture is an evidence-acquisition consumer for Symthaea #13696, paired with Symtropy's frozen scenario https://github.com/Luminous-Dynamics/symtropy/pull/7543.

The specific decision gap is pump throughput: the admitted conservative lower bound is 8 t, the optimistic upper bound is 14 t, and the beneficiary floor is 12 t. Stock capacity is not the binding unknown in this fixture. Title/custody/access remains a separate Mycelix-owned obligation.

The fixture declares an unweighted finite model family. It deliberately does not invent priors or average model hypotheses. Candidate actions are scored against decision-relevant proof obligations, dependency roots, cost, delay, safety and permission. A duplicated dashboard derived from the same sensor is not an independent measurement.

## Benchmark alignment

Align the experimental discipline with NIST's 2026 **TIDAL** research initiative: human-machine teaming, contextualized industrial data, digital-twin VVUQ, shared test benches and context-ablation benchmarks. NIST's stated benchmark dimensions include accuracy, latency, cost and grounding. This SAFRB case adds decision regret, false-safe outcomes, common-cause-aware evidence independence, proof-obligation coverage, authority compliance and whether evidence actually changes the service decision. This is not a claim to supersede or duplicate TIDAL.

Reference: https://www.nist.gov/programs-projects/super-intelligence-si-manufacturing

This is a policy test fixture, not a production diagnosis, inspection booking, maintenance action or authority system. Symthaea may recommend a permissioned evidence request; only the asset owner may authorize it, and only the relevant verifier may admit the resulting evidence.

## Fixture-level independent checks

The decision fixture now references both the single-window and two-window Symtropy cases. It carries separate service obligations per window, an unweighted admissible model family, and distinct candidate investigations for physical throughput, same-source telemetry, stock quantity and authority/access.

Run the independent standard-library fixture verifier:

```sh
python3 research/safrb-shared-pump-v0.1/verify_evidence_acquisition.py
```

It rejects unjustified model weights, correlated-dashboard independence promotion, non-binding measurements misrepresented as decision-relevant, post-deadline tests, horizon aggregation that hides a window breach, and authority escalation.

**Claim ceiling:** this validates fixture semantics only. It does not implement a VoI/EVSI policy, calibrate likelihoods, estimate real costs, prove actual model coverage, book an inspection, or authorize operational action. The dedicated workflow is committed as a future conformance hook; no CI pass is claimed for this draft head.
