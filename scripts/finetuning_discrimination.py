#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Fine-tuning Experiment for Phenomenal Discrimination

Tests whether targeted fine-tuning can increase phenomenal discrimination
in models that show inverse scaling.

Approach:
1. Take a large model with low discrimination (BERT-large)
2. Fine-tune on contrastive phenomenal/functional pairs
3. Measure if discrimination increases

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
        --run "python3 scripts/finetuning_discrimination.py"
"""

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


def load_corpus():
    """Load expanded concept corpus."""
    corpus_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/expanded_concept_corpus.json"
    with open(corpus_path) as f:
        corpus = json.load(f)
    return corpus["phenomenal_concepts"], corpus["functional_concepts"]


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning on phenomenal/functional pairs."""

    def __init__(self, phen_concepts: List[str], func_concepts: List[str], tokenizer, max_length=128):
        self.pairs = []
        self.labels = []

        # Create positive pairs (same class)
        for i, p1 in enumerate(phen_concepts):
            for p2 in phen_concepts[i+1:]:
                self.pairs.append((p1, p2))
                self.labels.append(1)  # Same class

        for i, f1 in enumerate(func_concepts):
            for f2 in func_concepts[i+1:]:
                self.pairs.append((f1, f2))
                self.labels.append(1)  # Same class

        # Create negative pairs (different class)
        for p in phen_concepts:
            for f in func_concepts:
                self.pairs.append((p, f))
                self.labels.append(0)  # Different class

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text1, text2 = self.pairs[idx]
        label = self.labels[idx]

        enc1 = self.tokenizer(text1, padding='max_length', truncation=True,
                              max_length=self.max_length, return_tensors='pt')
        enc2 = self.tokenizer(text2, padding='max_length', truncation=True,
                              max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids_1': enc1['input_ids'].squeeze(0),
            'attention_mask_1': enc1['attention_mask'].squeeze(0),
            'input_ids_2': enc2['input_ids'].squeeze(0),
            'attention_mask_2': enc2['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }


class ContrastiveHead(nn.Module):
    """Simple contrastive head for fine-tuning."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256)
        )

    def forward(self, x):
        return self.projection(x)


def get_pooled_output(model, input_ids, attention_mask):
    """Get mean-pooled output from model."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled


def compute_fisher_criterion(phen_reps: np.ndarray, func_reps: np.ndarray) -> float:
    """Compute Fisher's criterion."""
    phen_centroid = np.mean(phen_reps, axis=0)
    func_centroid = np.mean(func_reps, axis=0)
    centroid_distance = np.linalg.norm(phen_centroid - func_centroid)
    phen_within = np.mean([np.linalg.norm(r - phen_centroid) for r in phen_reps])
    func_within = np.mean([np.linalg.norm(r - func_centroid) for r in func_reps])
    avg_within = (phen_within + func_within) / 2
    return centroid_distance / avg_within if avg_within > 0 else 0


def evaluate_discrimination(model, tokenizer, phen_concepts, func_concepts, device):
    """Evaluate phenomenal discrimination."""
    model.eval()
    phen_reps = []
    func_reps = []

    with torch.no_grad():
        for text in phen_concepts:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pooled = get_pooled_output(model, inputs['input_ids'], inputs['attention_mask'])
            phen_reps.append(pooled.cpu().numpy().squeeze())

        for text in func_concepts:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pooled = get_pooled_output(model, inputs['input_ids'], inputs['attention_mask'])
            func_reps.append(pooled.cpu().numpy().squeeze())

    phen_reps = np.array(phen_reps)
    func_reps = np.array(func_reps)

    return compute_fisher_criterion(phen_reps, func_reps)


def contrastive_loss(proj1, proj2, label, temperature=0.5):
    """Compute contrastive loss."""
    # Normalize projections
    proj1 = nn.functional.normalize(proj1, dim=1)
    proj2 = nn.functional.normalize(proj2, dim=1)

    # Cosine similarity
    sim = (proj1 * proj2).sum(dim=1) / temperature

    # Binary cross entropy: same class -> high sim, different class -> low sim
    loss = nn.functional.binary_cross_entropy_with_logits(sim, label)
    return loss


def train_epoch(model, head, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    head.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        # Get pooled representations
        pooled1 = get_pooled_output(
            model,
            batch['input_ids_1'].to(device),
            batch['attention_mask_1'].to(device)
        )
        pooled2 = get_pooled_output(
            model,
            batch['input_ids_2'].to(device),
            batch['attention_mask_2'].to(device)
        )

        # Project
        proj1 = head(pooled1)
        proj2 = head(pooled2)

        # Compute loss
        labels = batch['label'].to(device)
        loss = contrastive_loss(proj1, proj2, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    print("\n" + "=" * 70)
    print("   FINE-TUNING EXPERIMENT FOR PHENOMENAL DISCRIMINATION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load corpus
    phen_concepts, func_concepts = load_corpus()

    # Use subset for training, rest for eval
    train_phen = phen_concepts[:30]
    train_func = func_concepts[:30]
    eval_phen = phen_concepts[30:50]
    eval_func = func_concepts[30:50]

    print(f"Training: {len(train_phen)} phenomenal, {len(train_func)} functional")
    print(f"Evaluation: {len(eval_phen)} phenomenal, {len(eval_func)} functional")

    # Test on BERT-large (known to have lower discrimination)
    model_name = "bert-large-uncased"
    print(f"\nModel: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)

    hidden_size = model.config.hidden_size
    head = ContrastiveHead(hidden_size).to(device)

    # Measure baseline discrimination
    print("\nMeasuring baseline discrimination...")
    baseline_train = evaluate_discrimination(model, tokenizer, train_phen, train_func, device)
    baseline_eval = evaluate_discrimination(model, tokenizer, eval_phen, eval_func, device)
    print(f"  Baseline (train set): F = {baseline_train:.4f}")
    print(f"  Baseline (eval set):  F = {baseline_eval:.4f}")

    # Create dataset and dataloader
    dataset = ContrastiveDataset(train_phen, train_func, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Fine-tune (only last few layers to preserve general knowledge)
    # Freeze all but last 2 encoder layers
    for name, param in model.named_parameters():
        if 'encoder.layer.22' not in name and 'encoder.layer.23' not in name and 'pooler' not in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable_params / 1e6:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=2e-5
    )

    # Training loop
    n_epochs = 3
    print(f"\nFine-tuning for {n_epochs} epochs...")

    results = {
        "baseline_train": baseline_train,
        "baseline_eval": baseline_eval,
        "epochs": []
    }

    for epoch in range(n_epochs):
        loss = train_epoch(model, head, dataloader, optimizer, device)

        # Evaluate after each epoch
        train_fisher = evaluate_discrimination(model, tokenizer, train_phen, train_func, device)
        eval_fisher = evaluate_discrimination(model, tokenizer, eval_phen, eval_func, device)

        epoch_result = {
            "epoch": epoch + 1,
            "loss": loss,
            "train_fisher": train_fisher,
            "eval_fisher": eval_fisher,
            "train_improvement": (train_fisher - baseline_train) / baseline_train * 100,
            "eval_improvement": (eval_fisher - baseline_eval) / baseline_eval * 100,
        }
        results["epochs"].append(epoch_result)

        print(f"  Epoch {epoch+1}: loss={loss:.4f}, train_F={train_fisher:.4f} ({epoch_result['train_improvement']:+.1f}%), "
              f"eval_F={eval_fisher:.4f} ({epoch_result['eval_improvement']:+.1f}%)")

    # Final results
    print("\n" + "=" * 70)
    print("   FINE-TUNING RESULTS")
    print("=" * 70 + "\n")

    final_train = results["epochs"][-1]["train_fisher"]
    final_eval = results["epochs"][-1]["eval_fisher"]
    train_change = (final_train - baseline_train) / baseline_train * 100
    eval_change = (final_eval - baseline_eval) / baseline_eval * 100

    print(f"  Train set: {baseline_train:.4f} → {final_train:.4f} ({train_change:+.1f}%)")
    print(f"  Eval set:  {baseline_eval:.4f} → {final_eval:.4f} ({eval_change:+.1f}%)")

    if eval_change > 5:
        print("\n  ✓ FINE-TUNING SUCCESSFULLY INCREASED DISCRIMINATION")
        finding = "success"
    elif eval_change > 0:
        print("\n  ~ MARGINAL IMPROVEMENT")
        finding = "marginal"
    else:
        print("\n  ✗ FINE-TUNING DID NOT IMPROVE DISCRIMINATION")
        finding = "failure"

    results["finding"] = finding
    results["final_train"] = final_train
    results["final_eval"] = final_eval
    results["train_improvement_pct"] = train_change
    results["eval_improvement_pct"] = eval_change

    # Save results
    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/finetuning_discrimination.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
