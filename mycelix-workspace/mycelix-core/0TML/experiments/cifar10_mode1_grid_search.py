#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
CIFAR-10 Mode 1 (PoGQ) hyperparameter grid search.

Explores validation batch counts, reference learning rates, MAD multipliers,
loss functions, and validation set sizes to recover ≥90% detection with ≤10% FPR
at 35% BFT.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score, average_precision_score
import math
import yaml
import time

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ground_truth_detector import GroundTruthDetector, QualityScore
from meta_detector import MetaDetector
from byzantine_attacks.basic_attacks import SignFlipAttack


class SimpleCNN(nn.Module):
    """Matches architecture used throughout CIFAR-10 experiments."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, image_size: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        flat_size = 64 * (image_size // 4) * (image_size // 4)
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def build_dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(num_clients)}

    for k in range(num_classes):
        class_indices = np.where(labels == k)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, cuts)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    for client_id in client_indices:
        rng.shuffle(client_indices[client_id])
    return client_indices


def create_subset_loader(dataset, indices: List[int], batch_size: int = 64):
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


def generate_honest_gradient(model: nn.Module, device: torch.device, train_loader: DataLoader) -> Dict[str, np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    temp_model = copy.deepcopy(model)
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


def pretrain_model(
    model: nn.Module,
    seed: int,
    device: torch.device,
    dataset: Optional[datasets.CIFAR10] = None,
    epochs: int = 5,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if dataset is None:
        dataset = datasets.CIFAR10(
            root=REPO_ROOT / "datasets" / "cifar10",
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )

    train_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model


class GridSearcher:
    def __init__(
        self,
        seeds: List[int],
        bft_ratio: float,
        num_clients: int,
        alpha: float,
        output_csv: Path,
        best_config_path: Path,
        max_configs: Optional[int] = None,
        auroc_gate: float = 0.8,
        model_name: str = "simple_cnn",
        logs_dir: Optional[Path] = None,
        prefilter_fltrust: bool = False,
        conformal_fpr: Optional[float] = None,
        expected_bft_ratio: Optional[float] = None,
        detector: str = "pogq",
        meta_params: Optional[Dict] = None,
    ):
        self.seeds = seeds
        self.bft_ratio = bft_ratio
        self.num_clients = num_clients
        self.alpha = alpha
        self.output_csv = output_csv
        self.best_config_path = best_config_path
        self.max_configs = max_configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.auroc_gate = auroc_gate
        self.model_name = model_name.lower()
        self.logs_dir = Path(logs_dir) if logs_dir else None
        self.prefilter_fltrust = prefilter_fltrust
        self.conformal_fpr = conformal_fpr
        self.expected_bft_ratio = expected_bft_ratio if expected_bft_ratio is not None else bft_ratio
        self.detector = detector
        self.meta_params = meta_params or {}

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.data_root = REPO_ROOT / "datasets" / "cifar10"

        self.train_dataset = datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )
        self.val_dataset = datasets.CIFAR10(
            root=self.data_root,
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )

        self.prepared_seeds: Dict[int, Dict] = {}

    def prepare_seed(self, seed: int):
        if seed in self.prepared_seeds:
            return

        model = self._build_model()
        augment_dataset = datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )
        pretrain_model(model, seed, self.device, dataset=augment_dataset)
        state_dict = copy.deepcopy(model.state_dict())

        num_byzantine = int(self.num_clients * self.bft_ratio)
        attack = SignFlipAttack(flip_intensity=1.0)

        labels = np.array(self.train_dataset.targets)
        partition = build_dirichlet_partition(labels, self.num_clients, self.alpha, seed)

        gradients = {}
        ground_truth = {}
        for client_id in range(self.num_clients):
            loader = create_subset_loader(self.train_dataset, partition[client_id])
            grad = generate_honest_gradient(model, self.device, loader)
            if client_id >= self.num_clients - num_byzantine:
                grad = {k: attack.generate(v, round_num=0) for k, v in grad.items()}
                ground_truth[client_id] = True
            else:
                ground_truth[client_id] = False
            gradients[client_id] = grad

        self.prepared_seeds[seed] = {
            "state_dict": state_dict,
            "gradients": gradients,
            "ground_truth": ground_truth,
        }

    def build_validation_loader(self, seed: int, val_size: int) -> Tuple[DataLoader, Dict[int, int]]:
        rng = np.random.default_rng(seed)
        total = len(self.val_dataset)
        requested = min(val_size, total)
        targets = np.array(self.val_dataset.targets)
        num_classes = len(np.unique(targets))

        per_class = requested // num_classes
        remainder = requested % num_classes
        indices: List[int] = []

        for cls in range(num_classes):
            cls_indices = np.where(targets == cls)[0]
            take = min(per_class + (1 if remainder > 0 else 0), len(cls_indices))
            remainder = max(0, remainder - 1)
            if take == 0:
                continue
            chosen = rng.choice(cls_indices, size=take, replace=False)
            indices.extend(chosen.tolist())

        if len(indices) < requested:
            remaining = np.setdiff1d(np.arange(total), np.array(indices, dtype=int), assume_unique=False)
            extra_take = min(requested - len(indices), len(remaining))
            if extra_take > 0:
                extra = rng.choice(remaining, size=extra_take, replace=False)
                indices.extend(extra.tolist())

        subset = Subset(self.val_dataset, indices)
        loader = DataLoader(subset, batch_size=64, shuffle=False)
        class_counts = self._compute_class_counts(subset)
        return loader, class_counts

    @staticmethod
    def _compute_class_counts(subset: Subset) -> Dict[int, int]:
        base_dataset = subset.dataset
        if hasattr(base_dataset, "targets"):
            all_targets = np.array(base_dataset.targets)
            subset_indices = subset.indices
            if hasattr(subset_indices, "tolist"):
                subset_indices = subset_indices.tolist()
            subset_indices = list(subset_indices)
            labels = all_targets[subset_indices]
        else:
            labels = []
            for idx in subset.indices:
                _, target = base_dataset[idx]
                labels.append(int(target))
            labels = np.array(labels)

        counts: Dict[int, int] = {}
        unique, freqs = np.unique(labels, return_counts=True)
        for cls, freq in zip(unique, freqs):
            counts[int(cls)] = int(freq)
        return counts

    @staticmethod
    def _compute_discriminative_metrics(quality_scores: List[QualityScore], ground_truth: Dict[int, bool]):
        labels = []
        raw_scores = []
        honest_scores = []
        byzantine_scores = []

        for score in quality_scores:
            node_id = score.node_id
            is_b = ground_truth.get(node_id)
            if is_b is None:
                continue
            raw = score.raw_quality if score.raw_quality is not None else score.quality
            labels.append(1 if is_b else 0)
            raw_scores.append(raw)
            if is_b:
                byzantine_scores.append(raw)
            else:
                honest_scores.append(raw)

        try:
            auroc = roc_auc_score(labels, raw_scores)
        except ValueError:
            auroc = 0.5
        try:
            auprc = average_precision_score(labels, raw_scores)
        except ValueError:
            auprc = 0.5

        if honest_scores and byzantine_scores:
            mean_diff = float(np.mean(honest_scores) - np.mean(byzantine_scores))
            var_h = float(np.var(honest_scores, ddof=1)) if len(honest_scores) > 1 else 0.0
            var_b = float(np.var(byzantine_scores, ddof=1)) if len(byzantine_scores) > 1 else 0.0
            pooled = math.sqrt(((len(honest_scores) - 1) * var_h + (len(byzantine_scores) - 1) * var_b) /
                               max(len(honest_scores) + len(byzantine_scores) - 2, 1))
            cohen_d = mean_diff / (pooled + 1e-12)
        else:
            cohen_d = 0.0

        return {"auroc": float(auroc), "auprc": float(auprc), "cohen_d": float(cohen_d)}

    def evaluate_config(self, config: Dict) -> Dict:
        config = dict(config)  # copy so we can inject derived fields
        seed_metrics = []
        aborted = False

        for seed in self.seeds:
            self.prepare_seed(seed)
            cached = self.prepared_seeds[seed]

            model = self._build_model().to(self.device)
            model.load_state_dict(cached["state_dict"])

            val_loader, class_counts = self.build_validation_loader(seed, config["val_size"])

            winsorize = config.get("winsorize")
            winsorize = winsorize if winsorize and winsorize > 0 else None
            ema_alpha = config.get("ema_alpha")
            if ema_alpha is not None and ema_alpha < 0:
                ema_alpha = None
            detector_mode = config.get("detector", self.detector)
            use_meta = detector_mode == "meta"
            config["detector"] = detector_mode

            prefilter_flag = config.get("prefilter_fltrust", self.prefilter_fltrust)
            conformal_fpr = config.get("conformal_fpr", self.conformal_fpr)
            expected_bft = config.get("expected_bft_ratio", self.expected_bft_ratio)
            if expected_bft is None:
                expected_bft = self.bft_ratio
            if use_meta:
                # Meta handles direction + conformal internally.
                prefilter_flag = False
                conformal_fpr = None
            config["prefilter_fltrust"] = prefilter_flag
            config["conformal_fpr"] = conformal_fpr
            config["expected_bft_ratio"] = expected_bft

            detector = GroundTruthDetector(
                global_model=model,
                validation_loader=val_loader,
                learning_rate=config["ref_lr"],
                adaptive_threshold=True,
                mad_multiplier=config["mad_multiplier"],
                max_validation_batches=config["val_batches"],
                loss_mode=config["loss_fn"],
                winsorize_p=winsorize,
                dispersion=config["dispersion"],
                threshold_mode=config["threshold_mode"],
                ema_alpha=ema_alpha,
                freeze_batchnorm=config["freeze_bn"],
                relative_improvement=config["relative_improvement"],
                gradient_clip=config["grad_clip"],
                hybrid_lambda=config["hybrid_lambda"],
                prefilter_fltrust=prefilter_flag,
                conformal_fpr=conformal_fpr,
                expected_bft_ratio=expected_bft,
                device=str(self.device),
            )
            detector.reset_statistics()

            meta_summary = None
            if use_meta:
                meta = MetaDetector(
                    detector,
                    conformal_alpha=self.meta_params.get("alpha", 0.10),
                    conformal_buffer=self.meta_params.get("buffer", 256),
                    temporal_alpha=self.meta_params.get("temporal_alpha", 0.6),
                    dirsim_floor=self.meta_params.get("dir_floor", 0.0),
                    dirsim_warning=self.meta_params.get("dir_warning", 0.05),
                    z_max=self.meta_params.get("z_max", 3.0),
                    density_min=self.meta_params.get("density_min", 0.15),
                    density_warning=self.meta_params.get("density_warning", 0.25),
                    density_neighbors=self.meta_params.get("density_k", 3),
                    abstain_on_conflict=self.meta_params.get("abstain", True),
                    teacher_ema=self.meta_params.get("teacher_ema", 0.7),
                )
                decisions, _ = meta.detect(cached["gradients"])
                detections = {nid: (decisions.get(nid) != "accept") for nid in cached["gradients"]}
                meta_summary = meta.summary()
            else:
                detections = detector.detect_byzantine(cached["gradients"])

            num_byz = sum(1 for v in cached["ground_truth"].values() if v)
            num_honest = len(cached["ground_truth"]) - num_byz

            tp = sum(1 for nid, flag in detections.items() if cached["ground_truth"][nid] and flag)
            fp = sum(1 for nid, flag in detections.items() if not cached["ground_truth"][nid] and flag)

            detection_rate = tp / num_byz if num_byz else 0.0
            fpr = fp / num_honest if num_honest else 0.0

            metrics = self._compute_discriminative_metrics(detector.quality_scores, cached["ground_truth"])

            stage_stats = None
            if not use_meta:
                stage1 = sum(1 for score in detector.quality_scores if score.stage1_flag)
                stage2 = sum(
                    1 for score in detector.quality_scores
                    if (score.stage1_flag is False) and score.stage2_flag
                )
                union = sum(1 for score in detector.quality_scores if score.is_byzantine)
                accepted = sum(1 for score in detector.quality_scores if not score.is_byzantine)
                stage_stats = {
                    "stage1": stage1,
                    "stage2": stage2,
                    "union": union,
                    "accepted": accepted,
                }

            seed_entry = {
                "seed": seed,
                "detection_rate": detection_rate,
                "fpr": fpr,
                "threshold": meta_summary["threshold"] if meta_summary else detector.computed_threshold,
                "mean_quality": float(np.mean([qs.quality for qs in detector.quality_scores])) if detector.quality_scores else 0.0,
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "cohen_d": metrics["cohen_d"],
                "class_counts": class_counts,
                "method": detector_mode,
            }
            if stage_stats:
                seed_entry["stage_stats"] = stage_stats
            if meta_summary:
                seed_entry["meta_summary"] = meta_summary
            seed_metrics.append(seed_entry)

            if metrics["auroc"] < self.auroc_gate:
                aborted = True
                break

        if not seed_metrics:
            return {
                "config": config,
                "seed_metrics": seed_metrics,
                "mean_detection": 0.0,
                "std_detection": 0.0,
                "mean_fpr": 0.0,
                "std_fpr": 0.0,
                "aborted_by_rule": True,
            }

        det_rates = [m["detection_rate"] for m in seed_metrics]
        fprs = [m["fpr"] for m in seed_metrics]

        return {
            "config": config,
            "seed_metrics": seed_metrics,
            "mean_detection": float(np.mean(det_rates)),
            "std_detection": float(np.std(det_rates)),
            "mean_fpr": float(np.mean(fprs)),
            "std_fpr": float(np.std(fprs)),
            "aborted_by_rule": aborted,
        }

    def run(self, search_space: Dict[str, List], target_detection: float, target_fpr: float):
        keys = list(search_space.keys())
        combos = list(itertools.product(*(search_space[k] for k in keys)))
        if self.max_configs:
            combos = combos[: self.max_configs]

        results = []
        best_config = None
        best_score = float("-inf")

        for idx, values in enumerate(combos, start=1):
            config = dict(zip(keys, values))
            print(f"\n=== Config {idx}/{len(combos)} :: {config}")
            evaluation = self.evaluate_config(config)
            results.append(evaluation)

            score = evaluation["mean_detection"] - max(0.0, evaluation["mean_fpr"] - target_fpr)
            if evaluation["mean_detection"] >= target_detection and evaluation["mean_fpr"] <= target_fpr:
                score += 1.0  # reward feasible configs

            if score > best_score:
                best_score = score
                best_config = evaluation

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.output_csv.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "val_batches",
                "ref_lr",
                "mad_multiplier",
                "loss_fn",
                "val_size",
                "dispersion",
                "winsorize",
                "threshold_mode",
                "ema_alpha",
                "freeze_bn",
                "relative_improvement",
                "grad_clip",
                "hybrid_lambda",
                "detector",
                "stage1_flags",
                "stage2_flags",
                "stage_union",
                "stage_accept",
                "meta_accept",
                "meta_reject",
                "meta_abstain",
                "prefilter_fltrust",
                "conformal_fpr",
                "expected_bft_ratio",
                "seed",
                "detection_rate",
                "fpr",
                "threshold",
                "mean_detection",
                "std_detection",
                "mean_fpr",
                "std_fpr",
                "auroc",
                "auprc",
                "cohen_d",
                "class_counts",
                "aborted_by_rule",
            ])
            for evaluation in results:
                for metric in evaluation["seed_metrics"]:
                    writer.writerow([
                        evaluation["config"]["val_batches"],
                        evaluation["config"]["ref_lr"],
                        evaluation["config"]["mad_multiplier"],
                        evaluation["config"]["loss_fn"],
                        evaluation["config"]["val_size"],
                        evaluation["config"]["dispersion"],
                        evaluation["config"]["winsorize"],
                        evaluation["config"]["threshold_mode"],
                        evaluation["config"]["ema_alpha"],
                        evaluation["config"]["freeze_bn"],
                        evaluation["config"]["relative_improvement"],
                        evaluation["config"]["grad_clip"],
                        evaluation["config"]["hybrid_lambda"],
                        evaluation["config"].get("detector", self.detector),
                        metric.get("stage_stats", {}).get("stage1", ""),
                        metric.get("stage_stats", {}).get("stage2", ""),
                        metric.get("stage_stats", {}).get("union", ""),
                        metric.get("stage_stats", {}).get("accepted", ""),
                        metric.get("meta_summary", {}).get("accept", "") if isinstance(metric.get("meta_summary"), dict) else "",
                        metric.get("meta_summary", {}).get("reject", "") if isinstance(metric.get("meta_summary"), dict) else "",
                        metric.get("meta_summary", {}).get("abstain", "") if isinstance(metric.get("meta_summary"), dict) else "",
                        evaluation["config"].get("prefilter_fltrust"),
                        evaluation["config"].get("conformal_fpr"),
                        evaluation["config"].get("expected_bft_ratio"),
                        metric["seed"],
                        metric["detection_rate"],
                        metric["fpr"],
                        metric["threshold"],
                        evaluation["mean_detection"],
                        evaluation["std_detection"],
                        evaluation["mean_fpr"],
                        evaluation["std_fpr"],
                        metric["auroc"],
                        metric["auprc"],
                        metric["cohen_d"],
                        json.dumps(metric["class_counts"]),
                        evaluation.get("aborted_by_rule", False),
                    ])

        if best_config:
            self.best_config_path.parent.mkdir(parents=True, exist_ok=True)
            self.best_config_path.write_text(json.dumps(best_config, indent=2))
            print(f"\nBest config written to {self.best_config_path}")
        if self.logs_dir:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.logs_dir / f"grid_run_{int(time.time())}.json"
            log_path.write_text(json.dumps(results, indent=2))

    def _build_model(self) -> nn.Module:
        if self.model_name == "resnet18":
            model = models.resnet18(num_classes=10)
        else:
            model = SimpleCNN()
        return model.to(self.device)


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 Mode 1 grid search")
    parser.add_argument("--config", type=Path, help="YAML config file describing the grid search")
    parser.add_argument("--output", type=Path, default=Path("results/cifar10_grid_search.csv"))
    parser.add_argument("--best-config", type=Path, default=Path("results/cifar10_grid_search_best.json"))
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--bft-ratio", type=float, default=0.35)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--auroc-gate", type=float, default=0.8, help="Minimum AUROC required to keep evaluating a config")
    parser.add_argument("--model", type=str, default="simple_cnn", help="Model architecture: simple_cnn or resnet18")
    parser.add_argument("--logs-dir", type=Path, default=None, help="Directory for per-config logs (optional)")
    parser.add_argument("--prefilter-fltrust", action="store_true", help="Enable FLTrust prefilter before PoGQ")
    parser.add_argument("--conformal-fpr", type=float, default=None, help="Conformal FPR cap (e.g., 0.10)")
    parser.add_argument("--expected-bft-ratio", type=float, default=None, help="Expected Byzantine ratio for FLTrust fallback logic")
    parser.add_argument("--detector", choices=["pogq", "meta"], default="pogq", help="Detector variant to evaluate")
    parser.add_argument("--meta-alpha", type=float, default=0.10, help="Conformal alpha for meta detector")
    parser.add_argument("--meta-buffer", type=int, default=256, help="Rolling buffer size for conformal calibrator")
    parser.add_argument("--meta-temporal-alpha", type=float, default=0.6, help="Temporal EMA for PoGQ quality")
    parser.add_argument("--meta-dir-floor", type=float, default=0.0, help="Minimum FLTrust cosine to accept")
    parser.add_argument("--meta-dir-warning", type=float, default=0.05, help="Warning threshold for cosine conflict")
    parser.add_argument("--meta-z-max", type=float, default=3.0, help="Outlier Z-score cutoff")
    parser.add_argument("--meta-density-min", type=float, default=0.15, help="Minimum normalized density")
    parser.add_argument("--meta-density-warning", type=float, default=0.25, help="Density warning for abstain logic")
    parser.add_argument("--meta-density-k", type=int, default=3, help="Neighbors to use for density estimation")
    parser.add_argument("--meta-no-abstain", action="store_true", help="Disable abstain-on-conflict behavior")
    parser.add_argument("--meta-teacher-ema", type=float, default=0.7, help="EMA for teacher gradient (dir-sim)")
    return parser.parse_args()


def _parse_seeds(value) -> List[int]:
    if isinstance(value, list):
        return [int(s) for s in value]
    return [int(s.strip()) for s in str(value).split(",") if s.strip()]


def _default_search_space() -> Dict[str, List]:
    return {
        "val_batches": [4, 8],
        "ref_lr": [5e-4, 1e-4],
        "mad_multiplier": [3.0, 3.5, 4.0],
        "loss_fn": ["mse_logits", "ce_logits"],
        "val_size": [1000, 2000],
        "dispersion": ["mad", "biweight"],
        "winsorize": [0.05],
        "threshold_mode": ["robust"],
        "ema_alpha": [0.7],
        "freeze_bn": [True],
        "relative_improvement": [True],
        "grad_clip": [None, 1.0],
        "hybrid_lambda": [None, 0.7],
    }


def main():
    args = parse_args()

    config_data = {}
    if args.config:
        with open(args.config, "r") as fh:
            config_data = yaml.safe_load(fh) or {}

    seeds = _parse_seeds(config_data.get("seeds", args.seeds))
    bft_ratio = config_data.get("bft_ratio", args.bft_ratio)
    num_clients = config_data.get("num_clients", args.num_clients)
    alpha = config_data.get("dirichlet_alpha", args.alpha)
    output_cfg = config_data.get("output", {})
    output_csv = Path(output_cfg.get("csv", args.output))
    best_json = Path(output_cfg.get("best_json", args.best_config))
    logs_dir = Path(output_cfg["logs_dir"]) if output_cfg.get("logs_dir") else args.logs_dir
    early_gate = config_data.get("early_gate", {})
    auroc_gate = early_gate.get("threshold", args.auroc_gate)
    model_name = config_data.get("model", args.model)
    search_space = config_data.get("search_space", _default_search_space())
    prefilter_fltrust = config_data.get("prefilter_fltrust", args.prefilter_fltrust)
    conformal_fpr = config_data.get("conformal_fpr", args.conformal_fpr)
    expected_bft_ratio = config_data.get("expected_bft_ratio", args.expected_bft_ratio)
    detector = config_data.get("detector", args.detector)
    meta_cfg = config_data.get("meta", {})
    meta_params = {
        "alpha": meta_cfg.get("alpha", args.meta_alpha),
        "buffer": meta_cfg.get("buffer", args.meta_buffer),
        "temporal_alpha": meta_cfg.get("temporal_alpha", args.meta_temporal_alpha),
        "dir_floor": meta_cfg.get("dir_floor", args.meta_dir_floor),
        "dir_warning": meta_cfg.get("dir_warning", args.meta_dir_warning),
        "z_max": meta_cfg.get("z_max", args.meta_z_max),
        "density_min": meta_cfg.get("density_min", args.meta_density_min),
        "density_warning": meta_cfg.get("density_warning", args.meta_density_warning),
        "density_k": meta_cfg.get("density_k", args.meta_density_k),
        "abstain": meta_cfg.get("abstain", not args.meta_no_abstain),
        "teacher_ema": meta_cfg.get("teacher_ema", args.meta_teacher_ema),
    }

    searcher = GridSearcher(
        seeds=seeds,
        bft_ratio=bft_ratio,
        num_clients=num_clients,
        alpha=alpha,
        output_csv=output_csv,
        best_config_path=best_json,
        max_configs=args.max_configs,
        auroc_gate=auroc_gate,
        model_name=model_name,
        logs_dir=logs_dir,
        prefilter_fltrust=prefilter_fltrust,
        conformal_fpr=conformal_fpr,
        expected_bft_ratio=expected_bft_ratio,
        detector=detector,
        meta_params=meta_params,
    )
    searcher.run(search_space, target_detection=0.90, target_fpr=0.10)


if __name__ == "__main__":
    main()
