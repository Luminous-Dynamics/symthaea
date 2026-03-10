#!/usr/bin/env python3
"""
Token-Level Attention Analysis

Identifies which specific tokens drive the phenomenal attention pattern.
Tests whether words like "experience", "feeling", "subjective" receive
concentrated attention in phenomenal concepts.

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
        --run "python3 scripts/token_attention_analysis.py"
"""

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PHENOMENAL_CONCEPTS = [
    "The vivid experience of seeing red",
    "What it is like to taste chocolate",
    "The subjective feeling of pain",
    "The unified field of conscious awareness",
    "The felt quality of sadness",
    "Qualia of the color blue",
    "First-person subjective experience",
    "The redness of red as experienced",
    "Phenomenal consciousness itself",
    "The hard problem of consciousness",
]

FUNCTIONAL_CONCEPTS = [
    "The recursive algorithm terminates",
    "Matrix multiplication computes dot products",
    "Memory is allocated on the heap",
    "The compiler optimizes bytecode",
    "Hash tables provide O(1) lookup",
    "TCP ensures reliable delivery",
    "The function returns an integer",
    "Binary search has logarithmic complexity",
    "The database indexes the primary key",
    "Garbage collection frees unused memory",
]

# Hypothesized phenomenal keywords
PHENOMENAL_KEYWORDS = [
    "experience",
    "experiencing",
    "experienced",
    "feeling",
    "felt",
    "feel",
    "subjective",
    "subjectivity",
    "conscious",
    "consciousness",
    "awareness",
    "aware",
    "quale",
    "qualia",
    "sensation",
    "sensory",
    "perception",
    "perceive",
    "vivid",
    "raw",
    "phenomenal",
]


def extract_token_attention(
    model, tokenizer, text: str, layer_idx: int, head_idx: int, device: str = "cpu"
) -> Dict:
    """Extract token-level attention for a specific head."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get attention for specific layer and head
    attn = outputs.attentions[layer_idx][0, head_idx].cpu().numpy()  # (seq, seq)

    # Compute attention received by each token (sum over attending tokens)
    attention_received = attn.sum(axis=0)  # How much attention each token receives
    attention_given = attn.sum(axis=1)  # How much attention each token gives

    return {
        "tokens": tokens,
        "attention_received": attention_received.tolist(),
        "attention_given": attention_given.tolist(),
        "attention_matrix": attn.tolist(),
    }


def analyze_token_patterns(
    model,
    tokenizer,
    concepts: List[str],
    layer_idx: int,
    head_idx: int,
    device: str = "cpu",
) -> Dict:
    """Analyze token attention patterns across concepts."""
    token_attention_scores = defaultdict(list)

    for text in concepts:
        result = extract_token_attention(
            model, tokenizer, text, layer_idx, head_idx, device
        )
        tokens = result["tokens"]
        attn_received = result["attention_received"]

        for tok, attn in zip(tokens, attn_received):
            # Clean token (remove ## prefix for subwords)
            clean_tok = tok.lower().replace("##", "")
            if clean_tok not in ["[cls]", "[sep]", "[pad]"]:
                token_attention_scores[clean_tok].append(attn)

    # Compute mean attention per token
    token_means = {
        tok: np.mean(scores)
        for tok, scores in token_attention_scores.items()
        if len(scores) >= 2
    }

    return token_means


def main():
    print("\n" + "=" * 70)
    print("   TOKEN-LEVEL ATTENTION ANALYSIS")
    print("   Identifying phenomenal keyword attention patterns")
    print("=" * 70 + "\n")

    model_name = "bert-base-uncased"
    layer_idx = 10  # Layer 11 (0-indexed)
    head_idx = 3  # Head 4 (0-indexed) - the strongest phenomenal head

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")
    print(
        f"Target: Layer {layer_idx + 1}, Head {head_idx + 1} (strongest phenomenal head)\n"
    )

    # Analyze phenomenal concepts
    print("Analyzing phenomenal concept tokens...")
    phen_token_attn = analyze_token_patterns(
        model, tokenizer, PHENOMENAL_CONCEPTS, layer_idx, head_idx, device
    )

    # Analyze functional concepts
    print("Analyzing functional concept tokens...")
    func_token_attn = analyze_token_patterns(
        model, tokenizer, FUNCTIONAL_CONCEPTS, layer_idx, head_idx, device
    )

    # Top tokens by attention in phenomenal concepts
    print("\n" + "=" * 70)
    print("   TOP ATTENTION TOKENS IN PHENOMENAL CONCEPTS")
    print("=" * 70 + "\n")

    sorted_phen = sorted(phen_token_attn.items(), key=lambda x: x[1], reverse=True)[:20]
    print(f"{'Token':<20} {'Mean Attn':<12} {'Keyword?':<10}")
    print("-" * 42)
    for tok, attn in sorted_phen:
        is_keyword = "YES" if tok in PHENOMENAL_KEYWORDS else ""
        print(f"{tok:<20} {attn:<12.4f} {is_keyword:<10}")

    # Top tokens in functional concepts
    print("\n" + "=" * 70)
    print("   TOP ATTENTION TOKENS IN FUNCTIONAL CONCEPTS")
    print("=" * 70 + "\n")

    sorted_func = sorted(func_token_attn.items(), key=lambda x: x[1], reverse=True)[:20]
    print(f"{'Token':<20} {'Mean Attn':<12}")
    print("-" * 32)
    for tok, attn in sorted_func:
        print(f"{tok:<20} {attn:<12.4f}")

    # Compare keyword attention
    print("\n" + "=" * 70)
    print("   PHENOMENAL KEYWORD ATTENTION COMPARISON")
    print("=" * 70 + "\n")

    print(f"{'Keyword':<20} {'Phen Attn':<12} {'Func Attn':<12} {'Diff':<10}")
    print("-" * 54)

    keyword_diffs = []
    for keyword in PHENOMENAL_KEYWORDS:
        phen_attn = phen_token_attn.get(keyword, None)
        func_attn = func_token_attn.get(keyword, None)

        if phen_attn is not None:
            func_val = func_attn if func_attn is not None else 0
            diff = phen_attn - func_val
            keyword_diffs.append((keyword, diff))
            func_str = f"{func_val:.4f}" if func_attn else "N/A"
            print(f"{keyword:<20} {phen_attn:<12.4f} {func_str:<12} {diff:+.4f}")

    # Detailed example
    print("\n" + "=" * 70)
    print("   DETAILED EXAMPLE: Token-by-token attention")
    print("=" * 70 + "\n")

    example_phen = "The subjective feeling of pain"
    example_func = "The recursive algorithm terminates"

    print(f"Phenomenal: '{example_phen}'")
    phen_result = extract_token_attention(
        model, tokenizer, example_phen, layer_idx, head_idx, device
    )
    for tok, attn in zip(phen_result["tokens"], phen_result["attention_received"]):
        if tok not in ["[CLS]", "[SEP]"]:
            bar = "█" * int(attn * 20)
            print(f"  {tok:15s} {attn:.3f} {bar}")

    print(f"\nFunctional: '{example_func}'")
    func_result = extract_token_attention(
        model, tokenizer, example_func, layer_idx, head_idx, device
    )
    for tok, attn in zip(func_result["tokens"], func_result["attention_received"]):
        if tok not in ["[CLS]", "[SEP]"]:
            bar = "█" * int(attn * 20)
            print(f"  {tok:15s} {attn:.3f} {bar}")

    # Key finding
    print("\n" + "=" * 70)
    print("   KEY FINDING")
    print("=" * 70 + "\n")

    # Check if phenomenal keywords get more attention
    if keyword_diffs:
        mean_keyword_diff = np.mean([d for _, d in keyword_diffs])
        if mean_keyword_diff > 0.01:
            print("  PHENOMENAL KEYWORDS RECEIVE MORE ATTENTION")
            print(f"  Mean attention difference: {mean_keyword_diff:+.4f}")
            print("\n  Top phenomenal-specific tokens:")
            for kw, diff in sorted(keyword_diffs, key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {kw}: +{diff:.4f}")
        else:
            print("  No clear keyword-specific attention pattern")

    # Check concentration
    phen_top_attn = sorted_phen[0][1] if sorted_phen else 0
    func_top_attn = sorted_func[0][1] if sorted_func else 0

    print(f"\n  Max token attention:")
    print(
        f"    Phenomenal: {phen_top_attn:.4f} ('{sorted_phen[0][0]}')"
        if sorted_phen
        else ""
    )
    print(
        f"    Functional: {func_top_attn:.4f} ('{sorted_func[0][0]}')"
        if sorted_func
        else ""
    )

    # Save results
    output = {
        "model": model_name,
        "layer": layer_idx + 1,
        "head": head_idx + 1,
        "phenomenal_top_tokens": sorted_phen[:30],
        "functional_top_tokens": sorted_func[:30],
        "keyword_analysis": [
            {"keyword": k, "diff": float(d)} for k, d in keyword_diffs
        ],
    }

    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/token_attention_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
