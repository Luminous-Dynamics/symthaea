# GEOM-003B1 — Reversible Global-Workspace Broadcast Lesion

## Status

Architecture-specific adapter under #2975 / #2979, stacked on GEOM-003A exact head `7b62690c8b44840b8392235d7381e524905125f4`.

This experiment tests the role of global-workspace broadcasting in Symthaea's measured dynamics. It does **not** define broadcasting as consciousness.

## Existing implementation boundary

`WorkspaceConfig` already exposes:

`enable_broadcasting: bool`

and `GlobalWorkspace::process()` performs competition/workspace entry before conditionally broadcasting. That gives us a narrow intervention surface without changing competition, thresholding, decay, capacity, or duration.

## Quartet construction

Given one baseline with broadcasting enabled:

- **intact**: exact baseline configuration, no field write;
- **lesion**: one explicit write `enable_broadcasting = false`;
- **matched sham**: one explicit write `enable_broadcasting = true`;
- **rescue**: one explicit restoration write `enable_broadcasting = true`.

The sham and lesion therefore use the same configuration-field operation count. Rescue is configuration-equivalent to intact.

## Invariants

Across all four conditions, these fields remain byte-identical:

- `max_capacity`;
- `entry_threshold`;
- `decay_rate`;
- `winner_takes_all`;
- `max_duration`.

GEOM-003A metadata is cloned identically across the quartet.

## Behavioral control

A matched one-cycle probe submits the same high-activation content to four fresh workspaces.

Expected result:

- workspace-entry count is identical across intact, lesion, sham, and rescue;
- intact emits at least one broadcast;
- lesion emits zero broadcasts;
- sham matches intact broadcast count;
- rescue matches intact broadcast count.

This establishes that the adapter changes the broadcast path while leaving workspace entry intact for the control probe.

## Fail-closed rule

A baseline with broadcasting already disabled is rejected. Turning an already-disabled mechanism "off" is not a lesion and cannot support a rescue claim.

## Claim boundary

Allowed:

> Disabling the existing broadcast path, while preserving the declared workspace configuration and matched probe entry, changed the measured GEOM observables by X; restoring broadcasting changed them by Y.

Not allowed:

> Broadcasting is consciousness.

Not allowed:

> Loss of broadcast proves loss of consciousness.

Not allowed:

> A broadcast lesion establishes any gravity/consciousness physical link.

## Next adapter gate

Only after this adapter and GEOM-003A qualify should the same pattern be generalized to other mechanisms such as recurrent temporal coupling, memory access, active-inference action selection, or HDC relational binding.
