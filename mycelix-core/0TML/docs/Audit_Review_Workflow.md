# Audit Guild Review Workflow

## 1. Prepare Review Packet

```bash
python scripts/export_audit_review.py \
  --queue 0TML/tests/results/audit_queue.jsonl \
  --output results/audit_review_queue.csv \
  --limit 250
```

- The CSV contains the key signals (round, node, consensus score, ML prediction/confidence).
- Additional columns `human_label` and `notes` are intentionally blank for reviewers to fill in (`human_label` ∈ {`honest`, `byzantine`}).

## 2. Distribute to the Audit Guild

1. Upload `results/audit_review_queue.csv` to the shared Audit Guild workspace.
2. Reviewers confirm each entry:
   - Compare `consensus_score` with honest baselines.
   - Inspect the raw logs if needed (trace JSON files live in `results/label_skew_trace_*.jsonl`).
3. Reviewers fill in the `human_label` and optional `notes` columns.

## 3. Capture Feedback

After review, save the annotated CSV as `results/audit_review_completed.csv` (or similar). Convert back to JSON for retraining:

```bash
python - <<'PY'
import csv, json
from pathlib import Path

input_csv = Path('results/audit_review_completed.csv')
output_json = Path('results/audit_feedback_human.jsonl')

with input_csv.open() as csv_file, output_json.open('w') as json_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        row['human_label'] = row.get('human_label', '').strip().lower()
        if row['human_label'] not in {'honest', 'byzantine'}:
            continue  # skip undecided entries
        json.dump(row, json_file)
        json_file.write('\n')
print('✅ wrote', output_json)
PY
```

## 4. Integrate into Retraining Datasets

- Merge the human-labelled JSON with the baseline feature log (example shown earlier in the engineering notes).
- When retraining, treat audit-labelled samples as a separate cohort so weighting/sampling can be controlled.
- Always validate via the attack matrix before promoting a new detector snapshot.

## 5. Governance Notes

- Keep the original human-reviewed CSVs archived for auditability.
- Record who reviewed each batch (add a `reviewer` column if needed).
- Re-run `export_audit_review.py` after each nightly regression to capture fresh borderline cases.
