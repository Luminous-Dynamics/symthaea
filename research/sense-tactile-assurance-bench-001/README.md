# SENSE-TACTILE-ASSURANCE-BENCH-001

Versioned manifest and scorecard contract for tactile measurement, transfer, prediction and decision-value experiments.

## Why this exists

Tactile systems are evaluated with many incompatible metrics: raw sensor force error, image/latent reconstruction, task success, manipulation policy performance, transfer, inference speed and robot-level safety. These scores are not interchangeable.

This contract keeps three result planes separate:

1. **Physical measurement** — calibration, reference integrity, observability, native-data identifiability, sensor-object/sensorized-body transfer, lifecycle validity and physical uncertainty.
2. **Representation and prediction** — contact-field prediction, transfer, contact/slip prediction, action alignment and predictive uncertainty.
3. **Decision value** — paired task improvement, critical physical violations, recovery, warning lead time and complete sensor-to-action latency.

Resource measurements and evidence-integrity checks are recorded separately. An overall weighted score is intentionally not part of the schema.

## Artifact contract

Every run binds:
- the exact source repository and commit;
- workflow/run identifier, build recipe and toolchain;
- comparator paper/release, code commit, checkpoint and dataset version where applicable;
- reproduction status and transfer regime;
- exact sensor article, modality, electronics/readout, firmware, skin/material, host and mounting context;
- sensor-object or sensorized-body stage;
- experiment protocol, task, measurand, stimuli, action and environment;
- complete-episode and physical-article/lot/host/skin split digests;
- raw observation and evaluator artifact digests;
- primary and independent check-reference receipts, ancestry and common-cause analysis;
- calibration, observability, model-adequacy and lifecycle applicability dispositions.

A URI or receipt ID is only a reference. The consuming verifier must retrieve and validate the actual evidence artifact. The schema does not make reference independence true merely because two IDs differ.

## Dataset, code and checkpoint rights

Treat access, training, redistribution and commercial-use permission as separate fields for every dataset, codebase and checkpoint. A project being publicly visible or described as "open" is not a sufficient rights determination.

The current OpenNeoData release card declares **CC BY-NC-SA 4.0** for its data and code. The official card also requires gated access and declarations that the user will use it only for non-commercial research and will not redistribute the dataset. The Creative Commons deed prohibits commercial use and imposes attribution and ShareAlike conditions. Sources:
- [OpenNeoData official dataset card](https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData)
- [OpenNeoData official README / access gate](https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData/blob/main/README.md)
- [CC BY-NC-SA 4.0 deed](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Therefore:
- Do not use OpenNeoData for commercial Luminous model training or product development under the currently declared license.
- Do not redistribute copies, mirrors, transformed copies or dataset-derived artefacts without a separate rights review and applicable permission.
- Treat non-commercial research use as subject to the publisher's gated access terms and attribution / ShareAlike obligations.
- Record code and checkpoint rights independently; the dataset license must not be assumed to cover every model weight or project component.
- Keep the OpenNeoData comparator as paper-/project-reported or non-commercial-only until the exact dataset revision, access agreement, intended use, code and checkpoint licenses are reviewed.

The schema requires a separate rights disposition for dataset, comparator code and checkpoint. Unknown or pending is an acceptable state; silently assuming permission is not.

## Transfer taxonomy

Use exact categories and disclose target adaptation:
- within-unit held-out episodes;
- cross-unit zero-shot;
- frozen-encoder few-shot adaptation;
- target-specific encoder trained on target demonstrations;
- cross-modality;
- cross-lot / manufacturing revision;
- host / skin / mount transfer;
- out-of-envelope challenge.

Do not call target-specific encoder training “zero-shot”.

## Claim classes

The manifest distinguishes:
- synthetic benchmark output;
- software conformance only;
- physical measurement result;
- paper-reported result;
- protocol-compatible reproduction.

A synthetic kernel, simulated reference, generated field or model-predicted force cannot be described as physical metrology. A paper-reported result cannot enter a reproduced leaderboard until its compatible artifacts and splits are verified.

## Split discipline

Primary splits are grouped by complete contact episode/trajectory, physical article, manufacturing lot, skin/material revision, host body, mounting and campaign. Adjacent frames from one contact episode must not be divided across train/test.

Keep selector-visible stimuli, primary labels, independent check-reference observations and held-out evaluator truth on separate paths.

## Scorecard rules

- Report the worst taxel/cell and spatial-tail metrics alongside pooled means.
- Report sensitivity support separately from unique native-data identifiability.
- Report prior-selected/underidentified estimates separately from measured/identifiable quantities.
- Report calibration burden in independent reference-contact operations, fixture operations, active taxel-site excitations, operator time and elapsed setup time. Do not equate pattern rows with physical reference contacts.
- Report p50/p95/p99 full-path latency including sensing, transfer, inference, projection/RTA, communication and actuator response when applicable.
- Never let task success compensate for false-qualified measurements, critical force violations or invalid reference chains.
- Never promote a learned embedding or predicted field into an independent physical witness.

## Validation of this schema

The JSON Schema validates structure and required identity fields. It does not verify cryptographic signatures, physical references, units, calibration validity, common-cause independence, or the truth of reported metrics. Those require separate evidence verifiers and human/physical review.

Files:
- schema/manifest.schema.json — structural contract.
- examples/synthetic-field-campaign.json — illustrative synthetic-only record, not a physical result.
