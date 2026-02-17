# CIFAR-10 Diagnostics
- Timestamp: 2025-11-07T04:57:09.200771+00:00
- Seed: 42
- Mode1 Detection: 0.00 %
- Mode1 FPR: 0.00 %
- Mode0 FPR: 100.00 %
- Adaptive Threshold: 0.4042
- AUROC: 0.132
- AUPRC: 0.243
- Cohen's d: 1.422

## Quality Score Stats
- Honest mean ± std: 0.5188 ± 0.0355
- Byzantine mean ± std: 0.4676 ± 0.0316

## Validation Class Counts
- Class 0: 19
- Class 1: 20
- Class 2: 21
- Class 3: 24
- Class 4: 24
- Class 5: 18
- Class 6: 12
- Class 7: 21
- Class 8: 26
- Class 9: 15

## Detector Settings
- EMA alpha: 0.7
- Winsorize: 0.05
- Dispersion: mad
- Hybrid λ: 0.7
- Relative improvement: True
- Grad clip: 1.0
- Freeze BN: True
## Notes
- Review quality_scores.csv for per-client insights.