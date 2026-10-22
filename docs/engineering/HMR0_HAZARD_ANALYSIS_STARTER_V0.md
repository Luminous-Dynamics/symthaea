# HMR-0 Hazard Analysis Starter v0.1

**Status:** hazard identification starter only—not a formal ISO 14971 risk-management file, quantitative risk assessment, safety case or authorization to operate.

This work makes design hazards explicit before the humanoid platform or its tools move beyond concept/simulation. Risk severity, likelihood, acceptability and residual risk are **not scored here**. Those decisions depend on intended use, users, operating environment, actual hardware, and qualified review.

## Governing principle

Symthaea may propose designs and candidate controls; it must not be the sole judge that those controls are sufficient. Each control needs a requirement, owner, exact hardware/software revision, verification evidence and independent review. A successful synthetic test means only that the declared software contract behaved as expected on that test.

The machine-readable inventory is docs/engineering/hmr0-hazard-analysis-starter-v0.1.json. It begins with twelve hazard families:

| ID | Hazard family | Initial verification emphasis |
|---|---|---|
| HMR0-HZ-001 | Unintended motion or fall | Stop/watchdog, power loss, state-estimation faults |
| HMR0-HZ-002 | Pinch, crush or entrapment | Swept volume, force/torque bounds, jams |
| HMR0-HZ-003 | Tool detachment/wrong tool | Attachment identity, detach detection, stale tool revision |
| HMR0-HZ-004 | Dropped/slipping load | Payload/centre-of-mass, friction, grasp-state uncertainty |
| HMR0-HZ-005 | Stale plan/world/permission | Revision binding, revocation, replay, revalidation |
| HMR0-HZ-006 | Inventory identity error | Ambiguous labels, duplicates, wrong bin, unreadable state |
| HMR0-HZ-007 | Network/supervision loss | Expired commands, sequence order, reconnect/recovery |
| HMR0-HZ-008 | Energy reserve/power failure | Uncertain energy model, reserve, controlled stop |
| HMR0-HZ-009 | Contamination/false clean claim | Unassessed cleaning status, material/geometry evidence |
| HMR0-HZ-010 | Privacy or data exposure | Non-patient scope, local processing, retention controls |
| HMR0-HZ-011 | Cyber compromise/task injection | Signed provenance, capability allowlist, least privilege |
| HMR0-HZ-012 | Unintended clinical-function expansion | Strict task/function schema and separate authorization |

Each JSON entry includes initiating conditions, potential harm, candidate controls, verification cases and unresolved engineering questions. These are candidate controls and test ideas—not proof that a risk is controlled.

## Why the initial scope is deliberately narrow

HMR-0 begins with sealed mock tote handling, synthetic inventory labels, passive tray docking and simulation-only fixture interactions. It excludes diagnosis, medication delivery, invasive/surgical action, patient lifting or transfer, patient contact, sterile-field access and physical actuation by the current benchmark. Any later scope expansion requires a separately reviewed intended-use and risk package.

## Standards watchlist

ISO 14971:2019 specifies a medical-device risk-management process and is listed by ISO as current after review/confirmation in 2025. IEC 80601-2-78 has a narrower medical-robot scope for certain patient-interacting movement-function applications; the applicable edition and relevance depend on intended use. No standard name in this document is a finding of conformity.

- ISO 14971:2019: https://www.iso.org/standard/72704.html
- IEC 80601-2-78:2019: https://www.iso.org/standard/68474.html
- IEC/DIS 80601-2-78 Edition 2, under development in the ISO catalogue as of 2026-10-22: https://www.iso.org/standard/90453.html

## Required continuation

Before any physical prototype is considered, define intended use and environment, perform a task-specific risk assessment, review actual hardware and control architectures, set measurable limits with engineering evidence, and have qualified independent reviewers examine the control/evidence chain. Risk scores should not be guessed to make a prototype look complete.
