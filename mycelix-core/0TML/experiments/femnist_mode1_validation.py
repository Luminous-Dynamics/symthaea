#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FEMNIST Mode 1 validation harness (proxy implementation).

This script approximates FEMNIST using torchvision's EMNIST dataset and
partitions samples into pseudo-writers to create naturally heterogeneous
clients. It reuses the GroundTruthDetector to measure Mode 1 performance
across multiple Byzantine ratios and seeds.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ground_truth_detector import GroundTruthDetector
from meta_detector import MetaDetector
sys.path.insert(0, str(REPO_ROOT))
from experiments.datasets.femnist_leaf import FEMNISTUserDataset, load_femnist_datasets


class GroupNormCNN(nn.Module):
    """Simple CNN with GroupNorm layers for FEMNIST-style inputs."""

    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def sample_writer_subset(dataset: FEMNISTUserDataset, samples: int, seed: int) -> Subset:
    rng = torch.Generator().manual_seed(seed)
    total = len(dataset)
    if samples >= total:
        return dataset
    indices = torch.randperm(total, generator=rng)[:samples]
    return Subset(dataset, indices.tolist())


def build_validation_loader(test_users: Dict[str, FEMNISTUserDataset], count: int, batch_size: int, seed: int) -> DataLoader:
    user_ids = list(test_users.keys())
    rng = np.random.default_rng(seed)
    if count > len(user_ids):
        count = len(user_ids)
    selected = rng.choice(user_ids, size=count, replace=False)
    datasets = [test_users[user_id] for user_id in selected]
    concat = ConcatDataset(datasets)
    return DataLoader(concat, batch_size=batch_size, shuffle=True)


def generate_gradient(model, device, loader):
    criterion = nn.CrossEntropyLoss()
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)

    temp_model = GroupNormCNN(num_classes=model.fc2.out_features).to(device)
    temp_model.load_state_dict(model.state_dict())
    temp_model.train()
    temp_model.zero_grad()
    output = temp_model(data)
    loss = criterion(output, target)
    loss.backward()

    gradients = {}
    for name, param in temp_model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().cpu().numpy()
    return gradients


def run_scenario(seed: int, bft_ratio: float, args, train_users, test_users) -> Dict:
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    num_clients = args.num_clients
    num_byzantine = max(1, int(num_clients * bft_ratio))
    num_honest = num_clients - num_byzantine

    writer_ids = np.array(sorted(train_users.keys()))
    rng = np.random.default_rng(seed)
    if num_clients > len(writer_ids):
        raise ValueError(f"Requested {num_clients} clients, but only {len(writer_ids)} writers available.")
    selected_writers = rng.choice(writer_ids, size=num_clients, replace=False)

    val_loader = build_validation_loader(test_users, args.val_users, args.val_batch_size, seed)
    model = GroupNormCNN(num_classes=args.num_classes).to(device)
    detector = GroundTruthDetector(
        global_model=model,
        validation_loader=val_loader,
        learning_rate=args.ref_lr,
        adaptive_threshold=True,
        mad_multiplier=args.mad_multiplier,
        max_validation_batches=args.val_batches,
        loss_mode='ce_logits',
        winsorize_p=args.winsorize,
        dispersion=args.dispersion,
        threshold_mode='robust',
        ema_alpha=args.ema_alpha,
        freeze_batchnorm=True,
        relative_improvement=True,
        gradient_clip=args.grad_clip,
        hybrid_lambda=args.hybrid_lambda,
    )

    gradients = {}
    ground_truth = {}
    for idx, writer_id in enumerate(selected_writers):
        dataset = train_users[writer_id]
        subset = sample_writer_subset(dataset, args.samples_per_client, seed + idx)
        loader = DataLoader(subset, batch_size=args.client_batch_size, shuffle=True)
        grad = generate_gradient(model, device, loader)
        if idx >= num_honest:
            grad = {k: -args.attack_scale * v for k, v in grad.items()}
            ground_truth[idx] = True
        else:
            ground_truth[idx] = False
        gradients[idx] = grad

    detections = {}
    meta_summary = None
    if args.detector == "meta":
        meta = MetaDetector(
            detector,
            conformal_alpha=args.meta_alpha,
            conformal_buffer=args.meta_buffer,
            temporal_alpha=args.meta_temporal_alpha,
            dirsim_floor=args.meta_dir_floor,
            dirsim_warning=args.meta_dir_warning,
            z_max=args.meta_z_max,
            density_min=args.meta_density_min,
            density_warning=args.meta_density_warning,
            density_neighbors=args.meta_density_k,
        )
        decisions, _ = meta.detect(gradients)
        detections = {nid: (decisions.get(nid) != "accept") for nid in gradients}
        meta_summary = meta.summary()
    else:
        detections = detector.detect_byzantine(gradients)

    num_byz = num_byzantine
    num_hon = num_honest
    tp = sum(1 for cid in range(num_hon, num_clients) if detections[cid])
    fp = sum(1 for cid in range(num_hon) if detections[cid])
    fn = num_byz - tp
    tn = num_hon - fp

    result = {
        "seed": seed,
        "bft_ratio": bft_ratio,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tp / num_byz if num_byz else 0.0,
        "fpr": fp / num_hon if num_hon else 0.0,
        "threshold": detector.computed_threshold,
        "detector": args.detector,
        "meta_summary": meta_summary,
    }
    if args.detector == "meta":
        result["method"] = "meta"
        result["meta"] = {
            "tpr": result["tpr"],
            "fpr": result["fpr"],
            "summary": meta_summary,
        }
    else:
        result["method"] = "pogq"
    return result


def main():
    parser = argparse.ArgumentParser(description="FEMNIST Mode 1 validation (proxy)")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--bft-ratios", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--samples-per-client", type=int, default=128)
    parser.add_argument("--client-batch-size", type=int, default=32)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--val-users", type=int, default=25)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--ref-lr", type=float, default=5e-4)
    parser.add_argument("--mad-multiplier", type=float, default=3.5)
    parser.add_argument("--winsorize", type=float, default=0.05)
    parser.add_argument("--dispersion", choices=["mad", "biweight"], default="mad")
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hybrid-lambda", type=float, default=0.7)
    parser.add_argument("--attack-scale", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=62)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/femnist"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/femnist_validation.json"))
    parser.add_argument("--detector", choices=["pogq", "meta"], default="pogq")
    parser.add_argument("--meta-alpha", type=float, default=0.10)
    parser.add_argument("--meta-buffer", type=int, default=256)
    parser.add_argument("--meta-temporal-alpha", type=float, default=0.6)
    parser.add_argument("--meta-dir-floor", type=float, default=0.0)
    parser.add_argument("--meta-dir-warning", type=float, default=0.05)
    parser.add_argument("--meta-z-max", type=float, default=3.0)
    parser.add_argument("--meta-density-min", type=float, default=0.15)
    parser.add_argument("--meta-density-warning", type=float, default=0.25)
    parser.add_argument("--meta-density-k", type=int, default=3)
    args = parser.parse_args()

    train_users, test_users = load_femnist_datasets(Path(args.data_dir))

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    ratios = [float(r.strip()) for r in args.bft_ratios.split(",") if r.strip()]

    results = []
    for seed in seeds:
        for ratio in ratios:
            stats = run_scenario(seed, ratio, args, train_users, test_users)
            results.append(stats)
            print(f"Seed {seed} | BFT {ratio:.2f} -> TPR={stats['tpr']*100:.1f}% FPR={stats['fpr']*100:.1f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved FEMNIST proxy results to {args.output}")


if __name__ == "__main__":
    main()
