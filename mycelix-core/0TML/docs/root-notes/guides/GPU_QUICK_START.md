# GPU Quick Start Guide

## ✅ What's Done

- [x] GPU detected: NVIDIA GeForce RTX 2070 (8GB)
- [x] Modified flake.nix for CUDA support
- [x] Building nix environment with torch-bin (in progress)

## 🚀 Once Build Completes (Next Steps)

### 1. Validate GPU (5 seconds)
```bash
nix develop --command python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 2070 with Max-Q Design
Memory: 8.0GB
```

### 2. Quick GPU Test (5 seconds)
```bash
nix develop --command python test_minimal.py
```

Expected: Completes in <5 seconds (vs 30s on CPU)

### 3. Run Full MNIST IID Experiment (30-60 minutes)
```bash
chmod +x run_mnist_iid_gpu.sh
nix develop --command bash run_mnist_iid_gpu.sh
```

This will:
- Check GPU is working
- Run 100 rounds with 7 baselines
- Complete in ~30-60 minutes
- Save results to `results/iid/`

### 4. Monitor Progress
Watch for output like:
```
Round   0: Train Loss=2.3000, Train Acc=0.1000, Test Loss=2.2800, Test Acc=0.1200
Round  10: Train Loss=0.4500, Train Acc=0.8700, Test Loss=0.4200, Test Acc=0.8800
Round  20: Train Loss=0.2100, Train Acc=0.9350, Test Loss=0.1900, Test Acc=0.9400
...
Round 100: Train Loss=0.0450, Train Acc=0.9850, Test Loss=0.0400, Test Acc=0.9870
```

### 5. Analyze Results (1 minute)
```bash
nix develop --command python experiments/utils/analyze_results.py \
  --results results/iid/mnist_iid_*.json \
  --output results/analysis/mnist_iid/
```

Generates:
- `convergence_accuracy.png` - Training curves
- `convergence_loss.png` - Loss curves
- `comparison_accuracy.png` - Final accuracy bar chart
- `comparison_loss.png` - Final loss bar chart
- `comparison_table.csv` - Performance metrics

## 🎯 Expected Results (MNIST IID)

All baselines should converge to **~98-99% accuracy**:

| Baseline | Final Accuracy | Convergence (rounds to 95%) |
|----------|----------------|------------------------------|
| FedAvg | ~98.5% | ~30 |
| FedProx | ~98.5% | ~30 |
| SCAFFOLD | ~98.5% | ~30 |
| Krum | ~98.0% | ~40 |
| Multi-Krum | ~98.3% | ~35 |
| Bulyan | ~98.2% | ~35 |
| Median | ~98.0% | ~40 |

**Key Findings** (expected):
- All methods work well on IID data
- Byzantine-robust methods (Krum, Bulyan, Median) slightly slower convergence
- No significant performance gap on IID (as expected from literature)

## 📊 What Comes Next

After MNIST IID completes (~1 hour):

### Option A: Continue with CIFAR-10 IID
```bash
nix develop --command python -u experiments/runner.py \
  --config experiments/configs/cifar10_iid.yaml
```
Runtime: ~2-4 hours

### Option B: Non-IID Experiments
```bash
# Dirichlet α=0.1 (highly non-IID)
nix develop --command python -u experiments/runner.py \
  --config experiments/configs/mnist_non_iid.yaml
```
Runtime: ~1-2 hours

### Option C: Byzantine Attack Testing
```bash
# 2 malicious clients
nix develop --command python -u experiments/runner.py \
  --config experiments/configs/mnist_byzantine.yaml
```
Runtime: ~30-45 minutes

## 💡 Pro Tips

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

Look for:
- GPU Utilization: Should be 90-100% during training
- Memory Usage: ~2-4GB for MNIST (well under 8GB limit)

### Speed Up Even More
Edit config files to use larger batches:
```yaml
federated:
  batch_size: 64  # vs 32 (2x faster)
```

### Run Multiple Experiments
Since experiments are fast now, you can run them sequentially:
```bash
#!/bin/bash
experiments=(
  "experiments/configs/mnist_iid.yaml"
  "experiments/configs/mnist_non_iid.yaml"
  "experiments/configs/mnist_byzantine.yaml"
  "experiments/configs/cifar10_iid.yaml"
)

for config in "${experiments[@]}"; do
  echo "Running $config..."
  nix develop --command python -u experiments/runner.py --config "$config"
done
```

Total runtime: ~5-6 hours for ALL experiments!

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check driver
nvidia-smi

# Check CUDA in PyTorch
python -c "import torch; print(torch.__config__.show())" | grep -i cuda
```

### Out of Memory
Reduce batch size in configs:
```yaml
federated:
  batch_size: 16  # vs 32
```

### Slow Performance
Check GPU isn't throttling:
```bash
nvidia-smi --query-gpu=temperature.gpu,clocks.gr,clocks.mem --format=csv
```

Ensure temps < 85°C and clocks at max.

---

**Status**: ⏳ Waiting for nix build to complete

**Once ready**: Run step 1 above to validate GPU, then launch full experiment!
