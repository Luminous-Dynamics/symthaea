#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FEMNIST PoGQ vs FLTrust head-to-head detection experiment.

Uses LEAF FEMNIST writer partitions to generate client gradients and compares:
  - Mode 1 PoGQ detector (GroundTruthDetector)
  - FLTrust cosine-trust scores vs. server validation gradient

Outputs JSON rows with detection metrics (TPR/FPR) for both methods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from ground_truth_detector import GroundTruthDetector
from experiments.datasets.femnist_leaf import load_femnist_datasets, FEMNISTUserDataset
from experiments.femnist_mode1_validation import GroupNormCNN, sample_writer_subset, build_validation_loader, generate_gradient


def flatten_gradient(grad: Dict[str, np.ndarray]) -> torch.Tensor:
    return torch.tensor(
        np.concatenate([g.reshape(-1) for g in grad.values()]),
        dtype=torch.float32,
    )


def compute_fltrust_scores(
    gradients: Dict[int, Dict[str, np.ndarray]],
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[int, float]:
    model.eval()
    model.zero_grad(set_to_none=True)
    loss_fn = nn.CrossEntropyLoss()

    for idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
        if idx + 1 >= 1:
            break

    ref_vec = torch.cat(
        [param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1)
         for param in model.parameters()]
    ).to(device)
    ref_norm = ref_vec.norm() + 1e-12

    trust_scores = {}
    for node_id, grad in gradients.items():
        vec = flatten_gradient(grad).to(device)
        cos = torch.dot(vec, ref_vec) / (vec.norm() * ref_norm + 1e-12)
        trust_scores[node_id] = max(0.0, cos.item())
    return trust_scores


def run_head_to_head(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    train_users, test_users = load_femnist_datasets(Path(args.data_dir))
    writer_ids = np.array(sorted(train_users.keys()))

    val_loader = build_validation_loader(test_users, args.val_users, args.val_batch_size, args.seed)
    model = GroupNormCNN(num_classes=args.num_classes).to(device)
    detector = GroundTruthDetector(
        global_model=model,
        validation_loader=val_loader,
        learning_rate=args.ref_lr,
        adaptive_threshold=True,
        mad_multiplier=args.mad_multiplier,
        max_validation_batches=args.val_batches,
        loss_mode="ce_logits",
        winsorize_p=args.winsorize,
        dispersion=args.dispersion,
        threshold_mode="robust",
        ema_alpha=args.ema_alpha,
        freeze_batchnorm=True,
        relative_improvement=True,
        gradient_clip=args.grad_clip,
        hybrid_lambda=args.hybrid_lambda,
    )

    rng = np.random.default_rng(args.seed)
    num_clients = args.num_clients
    selected = rng.choice(writer_ids, size=num_clients, replace=False)
    num_byz = max(1, int(num_clients * args.bft_ratio))
    num_hon = num_clients - num_byz

    gradients = {}
    ground_truth = {}
    for idx, writer_id in enumerate(selected):
        dataset = train_users[writer_id]
        subset = sample_writer_subset(dataset, args.samples_per_client, args.seed + idx)
        loader = DataLoader(subset, batch_size=args.client_batch_size, shuffle=True)
        grad = generate_gradient(model, device, loader)
        if idx >= num_hon:
            grad = {k: -args.attack_scale * v for k, v in grad.items()}
            ground_truth[idx] = True
        else:
            ground_truth[idx] = False
        gradients[idx] = grad

    detections = detector.detect_byzantine(gradients)
    trust_scores = compute_fltrust_scores(gradients, model, val_loader, device)

    def metrics(decisions: Dict[int, bool]):
        tp = sum(1 for nid in range(num_hon, num_clients) if decisions[nid])
        fp = sum(1 for nid in range(num_hon) if decisions[nid])
        fn = (num_clients - num_hon) - tp
        tn = num_hon - fp
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return tp, fp, tn, fn, tpr, fpr

    pogq_decisions = {nid: detections[nid] for nid in range(num_clients)}
    fltrust_decisions = {
        nid: trust_scores[nid] <= args.fltrust_threshold for nid in range(num_clients)
    }
    union_decisions = {
        nid: pogq_decisions[nid] or fltrust_decisions[nid] for nid in range(num_clients)
    }

    pogq_metrics = metrics(pogq_decisions)
    fl_metrics = metrics(fltrust_decisions)
    union_metrics = metrics(union_decisions)

    quality_map = {score.node_id: score.quality for score in detector.quality_scores}
    details = []
    for nid in range(num_clients):
        details.append({
            "seed": args.seed,
            "bft_ratio": args.bft_ratio,
            "node_id": nid,
            "is_byzantine": ground_truth[nid],
            "pogq_quality": float(quality_map.get(nid, 0.0)),
            "fltrust_trust": float(trust_scores.get(nid, 0.0)),
            "pogq_flag": bool(pogq_decisions.get(nid, False)),
            "fltrust_flag": bool(fltrust_decisions.get(nid, False)),
            "union_flag": bool(union_decisions.get(nid, False)),
        })

    return {
        "seed": args.seed,
        "dataset": "femnist",
        "num_clients": num_clients,
        "bft_ratio": args.bft_ratio,
        "pogq": {
            "tp": pogq_metrics[0],
            "fp": pogq_metrics[1],
            "tn": pogq_metrics[2],
            "fn": pogq_metrics[3],
            "tpr": pogq_metrics[4],
            "fpr": pogq_metrics[5],
        },
        "fltrust": {
            "tp": fl_metrics[0],
            "fp": fl_metrics[1],
            "tn": fl_metrics[2],
            "fn": fl_metrics[3],
            "tpr": fl_metrics[4],
            "fpr": fl_metrics[5],
        },
        "union": {
            "tp": union_metrics[0],
            "fp": union_metrics[1],
            "tn": union_metrics[2],
            "fn": union_metrics[3],
            "tpr": union_metrics[4],
            "fpr": union_metrics[5],
        },
        "fltrust_threshold": args.fltrust_threshold,
        "trust_scores": trust_scores,
        "details": details,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="FEMNIST PoGQ vs FLTrust head-to-head")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bft-ratio", type=float, default=0.35)
    ap.add_argument("--num-clients", type=int, default=20)
    ap.add_argument("--samples-per-client", type=int, default=128)
    ap.add_argument("--client-batch-size", type=int, default=32)
    ap.add_argument("--val-batches", type=int, default=4)
    ap.add_argument("--val-users", type=int, default=200)
    ap.add_argument("--val-batch-size", type=int, default=256)
    ap.add_argument("--ref-lr", type=float, default=5e-4)
    ap.add_argument("--mad-multiplier", type=float, default=3.5)
    ap.add_argument("--winsorize", type=float, default=0.05)
    ap.add_argument("--dispersion", choices=["mad", "biweight"], default="mad")
    ap.add_argument("--ema-alpha", type=float, default=0.3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--hybrid-lambda", type=float, default=0.7)
    ap.add_argument("--attack-scale", type=float, default=1.0)
    ap.add_argument("--fltrust-threshold", type=float, default=1e-6, help="Trust <= threshold flagged as Byzantine")
    ap.add_argument("--num-classes", type=int, default=62)
    ap.add_argument("--data-dir", type=Path, default=Path("datasets/femnist"))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--output", type=Path, default=Path("results/femnist_fltrust_vs_pogq.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    row = run_head_to_head(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        data = json.loads(args.output.read_text())
        if isinstance(data, list):
            data.append(row)
        else:
            data = [data, row]
    else:
        data = [row]
    args.output.write_text(json.dumps(data, indent=2))
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
