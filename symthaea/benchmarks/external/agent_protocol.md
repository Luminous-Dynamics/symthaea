# External Benchmark Agent Protocol

The external benchmark runners can call an agent command for per-task inference.
This keeps datasets and harnesses reproducible while allowing different agent
implementations.

## Invocation

Set the environment variable:

```
export SYMTHAEA_AGENT_CMD="/path/to/agent"
```

The command is invoked once per task. A JSON task payload is sent on stdin.

## Input (stdin)

```
{
  "task_id": "gaia-dev-0001",
  "task": "...user question...",
  "files": [],
  "metadata": {"source": "gaia"}
}
```

## Output (stdout)

The agent should print a JSON object:

```
{
  "answer": "...final answer...",
  "status": "ok",
  "actions": [
    {"action": "ReadFile", "path": "...", "outcome": "Success"}
  ],
  "notes": "optional"
}
```

If the output is not valid JSON, the runner treats stdout as the answer text.
