# First marine serviceability mock-up experiment — V0.1

**Status:** proposed safe mechanical usability experiment; not yet performed.  
**Parent:** [MARINE-SERVICEABILITY-001 #11049](https://github.com/Luminous-Dynamics/symthaea/issues/11049) / draft [PR #11051](https://github.com/Luminous-Dynamics/symthaea/pull/11051)

## 1. Question

Can a declared modular access/carrier arrangement reduce the time and effort to remove and replace a representative equipment package, without creating new access obstructions, part loss, awkward handling or configuration errors?

This first experiment tests only the mechanical access/service sequence. It does not test an operating watermaker, electrical equipment, pressurized fluid system, battery, inverter, propulsion, seawater intake, vessel stability or return-to-service safety.

## 2. Article and safety boundary

Use an inert, non-powered, non-pressurized dummy module with:
- measured mass, centre-of-gravity location and external envelope;
- blunt/guarded edges and safe handling points;
- inert dummy fasteners/connectors that cannot energize, pressurize, leak or release stored energy;
- unique removable module and carrier identifiers;
- a documented stable support and lifting/manual-handling plan approved by the responsible workshop safety lead.

Do not use a live battery, inverter, propulsion part, running pump, refrigerant circuit, pressure vessel, sea chest, fuel/oil system or actual pressurized watermaker housing in this pilot.

The mock-up should represent a candidate LOS service space only after a controlled geometry model or measured mock-up has been defined. Do not claim it represents the final vessel until that exact layout revision exists.

## 3. Alternatives to compare

Compare the same dummy module and task in three mechanically distinct mock-ups:
1. **Fixed installation:** equipment mounted directly to the support structure with a declared access panel.
2. **Hinged / removable service access:** the access surface opens or removes, but the module remains mounted during initial access.
3. **Replaceable carrier / slide-out module:** module is secured to a carrier that can be extracted through the declared envelope.

These are hypotheses, not preferred designs. Each alternative must preserve the same access boundary and task completion definition; disclose any extra access volume, mass, fasteners, connector count and service hardware.

## 4. Controlled task

The task is to exchange an inert module for a uniquely identified matching spare:
1. find and positively identify the target module;
2. open the declared service access;
3. remove the required inert fasteners / release mechanisms;
4. disconnect the non-functional dummy connector;
5. extract the module through the declared route;
6. place it in the designated safe holding location;
7. retrieve and verify the matching spare identity/revision;
8. insert, secure and reconnect the spare;
9. complete the declared visual / mechanical completion check;
10. update the mock configuration record and close the access.

The sequence is a test script for inert mock-up hardware only. It is not a real maintenance instruction.

## 5. Instrumentation and observations

Record:
- wall-clock and phase durations, including pauses and recovery;
- exact task start/finish events using one monotonic clock;
- operator count and declared training/familiarity level;
- tools used and tool changes;
- fastener count removed/reinstalled and dropped/missing parts;
- module/fixture contacts, collisions and blocked movements;
- measured access aperture, module envelope and minimum clearance using a calibrated method;
- module mass/CoG and handling aid used;
- lifting/handling force only if suitable calibrated instrumentation and a competent ergonomics/safety reviewer approve the measurement;
- task completion, identity/revision error, rework, assistance and operator override;
- operator-reported difficulty using a predeclared, non-diagnostic rating scale, without collecting unnecessary personal/health information;
- raw video/photography only with explicit consent and a documented retention/access policy; otherwise retain event logs and geometry measurements.

Record calibration identity, measurement uncertainty, mock-up revision, part identity, tool identity, test date, environmental conditions and raw artifact digest. Measured values must be tagged as measured; the design envelope remains the mock-up's exact configuration.

## 6. Experimental design

- Use a repeated-measures pilot: each participant performs all alternatives, but randomize/counterbalance the order to reduce learning effects.
- Provide the same training, task definition, module mass, spare identification, tools and completion criterion for each alternative.
- Include a familiarization run that is excluded from the primary timing summary; retain it in the raw record.
- Repeat each alternative enough times to expose procedural variation; preregister the number of participants/repetitions and analysis before testing.
- Keep operator-level and trial-level results separate. Do not treat repeated trials from the same participant as independent people.
- Include an intentional negative-control variant with an obstructed extraction route or wrong-revision spare; the task should be blocked or the mismatch detected, not completed by force or silent substitution.
- Include a second negative control with a missing fastener/part receipt; the mock configuration must remain incomplete until reconciled.

A small pilot estimates usability and reveals design defects; it does not establish population-level ergonomics, fleet MTTR or long-term reliability.

## 7. Metrics

Report for every alternative and task phase:
- median, range and p90 duration with participant/trial counts;
- task completion and assistance rate;
- phase where obstruction or rework occurs;
- tool changes, fastener operations, part drops and identity errors;
- minimum clearance and access-path deviations;
- handling demand and any safety stop;
- configuration record completeness and exact replay;
- additional access envelope, hardware mass/volume and number of service interfaces.

Do not collapse results into one score. A design that is quicker but introduces a wrong-part error or a handling/safety stop is not superior.

## 8. Predeclared pilot interpretation

- An alternative is **not eligible for a positive serviceability result** if it requires unplanned removal of an unrelated critical mock-up system, exceeds its declared handling envelope, permits a wrong-revision install, or fails to preserve the part/configuration receipt.
- A safety stop, dropped load, damage or unsafe movement stops the trial and requires review; do not repeat until the setup and risk review are reconciled.
- If clearance or mass/CoG measurements are missing or exceed uncertainty, disposition is Indeterminate.
- Report no numeric improvement target before the baseline and measurement method are frozen.
- Any selected concept proceeds to real vessel spatial coordination and qualified structural/human-factors review before it can enter a construction design.

## 9. What this experiment can and cannot establish

It can establish task timing, access obstruction and part-identity behavior on the exact inert mock-up revision under the tested participants, tools and environment.

It cannot establish actual at-sea MTTR, operation in waves, human performance for all crews, stability during removal, structural adequacy, electrical safety, pressure/flood control, watertight/fire integrity, high-voltage safety, seaworthiness, class/flag acceptance or permission to perform maintenance.
