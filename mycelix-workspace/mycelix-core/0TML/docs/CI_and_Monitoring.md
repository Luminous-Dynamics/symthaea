# CI & Monitoring Playbook

## Nightly Regression Workflow (`matl-regression.yml`)

- Runs every night at 02:30 UTC (and on manual dispatch).
- Executes `run_attack_matrix.sh` with:
  - `ATTACK_DISTRIBUTIONS=iid,label_skew`
  - `LABEL_SKEW_ALPHA=0.2`
  - `LABEL_SKEW_COMMITTEE_PERCENTILE=8`
  - `ML_DETECTOR_PATH=models/byzantine_detector_logged_v3`
- Produces `0TML/tests/results/bft_attack_matrix.json` plus per-attack artefacts.
- Fails if any IID scenario (excluding experimental attacks) drops below 95 % detection or exceeds 5 % FP.
- Generates a Markdown summary of the label-skew runs in the Actions “Summary” tab for quick triage.

## Prometheus Metrics

1. `scripts/export_bft_metrics.py` writes `artifacts/matl_metrics.prom` containing:
   - `matl_detection_rate_percent{attack,distribution,ratio[,alpha]}`
   - `matl_false_positive_rate_percent{…}`
   - `matl_regression_success{…}` (0/1)
2. The workflow uploads both the matrix JSON and the Prometheus file as artefacts.
3. To scrape metrics:
   ```bash
   curl -sL https://github.com/<org>/<repo>/actions/runs/<run-id>/artifacts/<artifact-id>/matl_metrics.prom
   ```
   (or mirror the artefact to an internal storage bucket).

## Grafana Integration Checklist

1. Configure Prometheus (or a sidecar exporter) to ingest `matl_metrics.prom` after each workflow.
2. Add Grafana panels with the PromQL equivalents:
   - `matl_detection_rate_percent{distribution="iid"}`
   - `matl_false_positive_rate_percent{distribution="label_skew", alpha="0.2"}`
3. Set alert thresholds that match today’s baseline (IID success, label-skew FP ≈ 7 %).
4. The workflow summary URL can be linked directly from Grafana annotations for quick drill-down.

## Manual Verification

```bash
ATTACK_DISTRIBUTIONS=iid,label_skew \
LABEL_SKEW_ALPHA=0.2 \
LABEL_SKEW_COMMITTEE_PERCENTILE=8 \
ML_DETECTOR_PATH=models/byzantine_detector_logged_v3 \
USE_ML_DETECTOR=1 \
./run_attack_matrix.sh

python scripts/export_bft_metrics.py \
  --matrix 0TML/tests/results/bft_attack_matrix.json \
  --output artifacts/matl_metrics.prom
```

Review `artifacts/matl_metrics.prom` locally before pushing to ensure values look sane.
