# WorldTest / MARA reference-runner audit v0.1

**Status:** source-level audit only; no MARA process, Autumn interpreter or benchmark dataset was executed.  
**Pinned repository:** BasisResearch/MARAProtocol at d0d4e9251151778415cc5195c5dfb8296a5ded24 (unsigned).

## Findings

The generic MARA agent FeedbackRequest contains a reward field. The pinned evaluation controller forwards StepResponse.reward to that field on every feedback call, including interactive exploration. Therefore it is inaccurate to say the reward field is omitted.

The pinned interactive exploration implementation returns exactly zero for ordinary actions, explicit reset and the go-to-test action. The reward-free property of the reference exploration phase is therefore **constant-zero reward semantics**, not the absence of a transport field. The production adapter must record the reward at the environment boundary, verify it is zero on every exploration step and fail closed on any nonzero value before the candidate receives feedback.

The go-to-test action is candidate-issued and returns zero. It is a charged exploration interaction. The trusted adapter appends the terminal BeginTest marker only after the environment accepts the transition; that marker is metadata, not a candidate action.

## Pinned source identities

- mara_agent.proto: 70ae75dad22be364af45417fb24007ddbfbe61a3
- mara_environment_service.proto: 58069aa74cb1bf084155c115111dc2a2aa6b58c6
- evaluation_controller.py: 8650b314b3f0e5f9301340dce89aed5aa4de0ae8
- environment_interfaces.py: 7048358744b74f042db4c43f734d9271468f727b
- concrete_envs.py: 961858e7f47ad1e572bc20936d27ae0b4fee000b

All identities refer to the pinned MARAProtocol commit above. The lock and schema are worldtest-runner-audit.lock.json and worldtest-runner-audit.schema.json.

## Required adapter checks

1. Verify every reward value during exploration is exactly zero; do not infer this from a task name or generic protocol.
2. Include explicit reset and go-to-test in the interaction budget and event trace.
3. Require exactly one go-to-test action immediately before the trusted terminal BeginTest marker.
4. Exclude answer keys, true change times and evaluator labels from candidate-visible state.
5. Keep test-phase reward/outcome handling in the separate scoring phase.

This audit does not establish runtime parity, dataset integrity, benchmark replication or model performance.
