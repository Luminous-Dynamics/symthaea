# Experiment Artifacts

Logs and machine-generated outputs that were previously scattered in the repo
root live here.

- `logs/` — runtime logs (pytest, conductor, deployment runs)
- `results/` — JSON summaries and benchmark results
- `tex/` — LaTeX tables/figures generated for reports

When capturing new runs, redirect output here to keep the workspace clean, e.g.:

```bash
poetry run python experiment.py > artifacts/logs/experiment-$(date +%Y%m%d).log
```
