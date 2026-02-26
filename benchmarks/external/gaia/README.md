# GAIA (Public Dev)

GAIA evaluates general assistant tool-use. We use the public dev split only.

## Data Layout

Place the dev split here:

```
benchmarks/external/gaia/data/
  dev.json
```

## Fetch

```
./fetch.sh
```

This is a manual download step (see script output for instructions).

## Run

```
./run.sh
```

The GAIA runner should:

- Route tasks through ActionIR tool bindings
- Use the local LLM as translation-only
- Record tool traces and action budgets

Results should be written to:

```
benchmarks/external/results/gaia-dev.json
```

## Agent Integration

Set `SYMTHAEA_AGENT_CMD` to use a custom agent. The protocol is documented in:

```
benchmarks/external/agent_protocol.md
```

### Cyborg Mind Agent (Local)

Build the GAIA agent once:

```
cargo build --bin symthaea-gaia-agent --release
```

Then set:

```
export SYMTHAEA_AGENT_CMD="benchmarks/external/gaia/agent.sh"
```

Budget overrides (optional):

```
export SYMTHAEA_AGENT_SHELL_BUDGET=50
export SYMTHAEA_AGENT_WRITE_BUDGET=50
export SYMTHAEA_AGENT_BYTES_BUDGET=10485760
```

HDC+CfC/LTC brain controls:

```
export SYMTHAEA_AGENT_PROFILE=standard
export SYMTHAEA_AGENT_TEMPORAL_BACKEND=cfc
export SYMTHAEA_AGENT_CYCLES=3
export SYMTHAEA_AGENT_CONTEXT_LIMIT=4000
# export SYMTHAEA_AGENT_GENESIS=cyborg-mind-v1
```

Optional web fetch (curl) for GAIA tasks that include URLs:

```
export SYMTHAEA_AGENT_ALLOW_WEB=1
```
