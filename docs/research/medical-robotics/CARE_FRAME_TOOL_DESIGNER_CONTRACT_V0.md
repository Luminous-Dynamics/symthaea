# Careframe Tool Designer Contract v0

**Status:** research interface specification. It describes how Symthaea may propose simulation-only tool concepts for the Symtropy Careframe workstream. It is not a CAD compiler, verified mechanism, medical-device design freeze, safety case, or permission to manufacture, power or clinically use a tool.

## Responsibility split

- **Symthaea proposes.** It converts a reviewed task brief into multiple explicit design hypotheses, candidate manifest(s), uncertainties, constraints and a comparison plan. It must expose missing requirements rather than invent clinical constraints.
- **The Symtropy manifest gate rejects.** It independently validates schema, permitted tool/task/contact pairing, conservative design-envelope fields, authority flags and unknown-field rejection. It does not infer correctness from Symthaea's confidence or explanation.
- **Symtropy simulates.** It evaluates accepted proposals only inside the stated synthetic scenario and simulator capabilities. Simulation does not validate hardware or clinical claims.
- **Independent engineering and domain reviewers decide next stages.** Mechanical, electrical, human-factors, infection-control, clinical, cybersecurity, ethics and regulatory review remain separate from generated output.
- **The proposal cannot grant its own authority.** Every v0 candidate must set physical actuation, patient contact and clinical use authorization to false. The independent gate must emit its own receipt.

## Design loop

1. Parse a versioned, non-patient-identifying task brief into typed constraints and hard exclusions.
2. Generate a small bounded set of alternatives, not a single unchallenged answer.
3. For each candidate, output a candidate tool manifest plus requirement-to-design traceability, assumptions, uncertainty and falsifiable hypotheses.
4. Validate each manifest with the Symtropy medical-robot-tool-gate using a pinned code revision and dependency lock.
5. Reject blocked candidates; preserve the rejection and findings in the run report instead of silently repairing them.
6. Simulate eligible candidates against identical scenarios, task goals, payloads, tool interfaces, environment and resource budgets. Run weak baselines and adversarial mutations.
7. Rank only on predeclared non-clinical, non-compensatory gates and task-specific utility metrics. A weighted average cannot compensate for violation of a hard safety/authority constraint.
8. Return the candidate set, exact digests, negative results, uncertainty, metric denominators and the claim ceiling. Do not state that a concept is “safe,” “sterile,” “clinically validated,” or “regulatory approved.”
9. Any relevant change to task, geometry, manifest, simulation config, source, model or constraints invalidates the affected receipt and triggers re-evaluation.

## Machine-readable interface

The companion schema, schemas/careframe-tool-design-proposal-v0.schema.json, defines a strict proposal envelope with additionalProperties: false at every object boundary. Each proposed candidate carries a strict tool manifest aligned with Symtropy's v0 manifest contract.

Proposal envelopes may carry:
- a privacy-safe task objective and permitted work zone;
- requirement IDs, thresholds, rationale and source/evidence references;
- candidate manifests (at most eight) for tool concepts supported by the v0 manifest contract;
- design hypotheses and explicit falsifiers;
- known uncertainties and coverage limitations;
- matched simulator baselines, scenario digests and metrics;
- the generating model/configuration and exact software/catalog digests.

The interface does **not** accept patient records, clinical diagnoses, patient-specific images, free-form motor commands, medication instructions or a field that can elevate clinical/physical authority. If a future design requires those, it is a new intended-use and governance review rather than a schema extension silently accepted by v0.

A synthetic example is provided at examples/careframe-tool-design-proposal-v0.example.json with its scenario fixture at examples/careframe-example-scenario-set-v0.json. Digests for the sample model/configuration/task are hashes of explicitly named placeholder strings, not hashes of real model artifacts. They demonstrate shape only and must never be used as provenance receipts.

## Articulated-body proposal contract

The body/morphology is a separate typed artifact, not a field silently smuggled into the tool manifest. The companion schema `schemas/careframe-morphology-v0.schema.json` and synthetic fixture `examples/careframe-wheeled-upper-body-morphology-v0.example.json` specify a rooted link/joint graph with fixed/revolute joint metadata, bounding primitives, units, limits, and hard-disabled command/authority flags. The exact fixture is shared byte-for-byte with the current Symtropy draft's morphology file, and its Rust validator separately checks unique IDs, link references, root/parent count, graph reachability, joint-axis/limit validity and fail-closed command state.

JSON Schema can check field shape and local constraints but not graph reachability, cycle absence, normalized 3D axes, or engine-import correctness. Those remain separate graph/runtime conformance gates. Values in this fixture are synthetic placeholders, not engineering specifications.

## Contract alignment with Symtropy

Current supported pairs:

| Tool kind | Intended task | Contact class |
|---|---|---|
| sealed_supply_gripper | sealed_supply_transfer | object_only |
| tray_carrier | tray_transfer | object_only |
| telepresence_head | telepresence_support | none |
| trainer_probe | training_mannequin_measurement | mannequin_only |

These pairs are simulation proposal templates only. Payload, force, torque, mass and geometry values are illustrative modeling envelopes rather than physically validated specifications. Patient transfer, medication dispensing, invasive tools and patient contact are outside v0.

## Success criteria

- 100% of schema-invalid or authority-escalating proposals rejected by the independent gate on the frozen suite.
- Each accepted proposal is linked to a manifest digest and exact validator revision.
- Every design hypothesis includes a metric and a falsifier before simulation begins.
- Alternatives are compared under the same task, geometry envelope, scenario set, resource budget and baseline definitions.
- All failures, out-of-distribution cases and missing measurements are preserved; no post-hoc deletion to improve results.
- Simulation scorecards do not imply physical or clinical qualification.

## References

- Symtropy Careframe draft: https://github.com/Luminous-Dynamics/symtropy/pull/2723
- Symtropy tool-manifest contract: https://github.com/Luminous-Dynamics/symtropy/blob/research/medical-humanoid-platform-v0/docs/medical-robotics/TOOL_MANIFEST_CONTRACT_V0.md
- Careframe benchmark-validity audit: https://github.com/Luminous-Dynamics/symtropy/blob/research/medical-humanoid-platform-v0/docs/medical-robotics/CARE_FRAME_BENCHMARK_VALIDITY_AUDIT_V0.md
- ISO/TS 24971-2:2026 identifies lifecycle risk questions for ML-enabled medical devices, but explicitly excludes devices employing LLMs or generative AI. It is therefore a useful adjacent reference, not a direct validation framework for the proposed generator: https://www.iso.org/standard/24971-2
