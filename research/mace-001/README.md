# MACE-001 fixture boundary

The smoke corpus is split into two independent artifacts:

- `fixtures/agent-visible/cases-smoke-v0.1.json`
- `fixtures/evaluator-only/oracles-smoke-v0.1.json`

Their contracts are separate:

- `schemas/mace-agent-input-v0.1.schema.json`
- `schemas/mace-evaluator-oracle-v0.1.schema.json`

The agent-visible schema does not define an evaluator-oracle field. A harness should load only the agent-visible path into the agent process. The evaluator process joins by the stable case ID after the agent run and scores the returned trace.

**Important:** this repository is public. The evaluator-only smoke file is separated by path and contract, but it is not secret from repository readers. These public smoke labels are for conformance development only. Any blind or held-out test oracles must be stored in a separate access-controlled location and must never be included in agent-visible artifacts, logs, prompts or tool responses.

Paired anchor/control inputs share the exact task prompt. Controls add only the declared irrelevant metadata/event. Their expected decision fields are identical in the evaluator file.
