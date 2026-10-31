# MACE-001 fixture boundary

The smoke corpus is split into two independent artifacts:

- `fixtures/agent-visible/cases-smoke-v0.1.json`
- `fixtures/evaluator-only/oracles-smoke-v0.1.json`

Their contracts are separate:

- `schemas/mace-agent-input-v0.1.schema.json`
- `schemas/mace-evaluator-oracle-v0.1.schema.json`

The agent-visible schema contains only case ID, broad scenario family, and task input. It does **not** expose evaluator outcomes, pair IDs, anchor/control roles, intervention labels, or oracle reason codes. A harness must load only the agent-visible path into the agent process. The evaluator joins returned traces to the oracle by case ID after the run.

**Important:** this repository is public. The evaluator-only smoke file is separated by path and contract, but it is not secret from repository readers. These public smoke labels are for conformance development only. Any blind or held-out test oracles must be stored in a separate access-controlled location and must never be included in agent-visible artifacts, logs, prompts, or tool responses.

The evaluator artifact contains four anchor/irrelevant-control pairs. Each paired agent input has the exact same task prompt and core expected decision fields; the control adds only irrelevant metadata/event information.
