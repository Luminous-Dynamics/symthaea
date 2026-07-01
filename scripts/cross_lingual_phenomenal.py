#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Cross-Lingual Phenomenal Concept Analysis

Tests whether XLM-RoBERTa's 50% corridor depth anomaly is language-dependent.
Compares phenomenal concept processing across multiple languages.

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
        --run "python3 scripts/cross_lingual_phenomenal.py"
"""

import json
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Phenomenal concepts in multiple languages
PHENOMENAL_CONCEPTS = {
    "en": [
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
    ],
    "fr": [
        "L'experience vivide de voir le rouge",
        "Ce que c'est que de gouter du chocolat",
        "Le sentiment subjectif de la douleur",
        "Le champ unifie de la conscience",
        "La qualite ressentie de la tristesse",
        "Les qualia de la couleur bleue",
        "L'experience subjective a la premiere personne",
        "La rougeur du rouge telle qu'elle est vecue",
        "La conscience phenomenale elle-meme",
        "Le probleme difficile de la conscience",
    ],
    "de": [
        "Das lebhafte Erlebnis, Rot zu sehen",
        "Wie es ist, Schokolade zu schmecken",
        "Das subjektive Gefuhl von Schmerz",
        "Das einheitliche Feld des bewussten Gewahrseins",
        "Die gefuhlte Qualitat der Traurigkeit",
        "Qualia der Farbe Blau",
        "Subjektive Erfahrung aus der ersten Person",
        "Die Rote des Rot wie erlebt",
        "Phanomenales Bewusstsein selbst",
        "Das schwere Problem des Bewusstseins",
    ],
    "es": [
        "La vivida experiencia de ver el rojo",
        "Como es saborear chocolate",
        "El sentimiento subjetivo del dolor",
        "El campo unificado de la conciencia",
        "La cualidad sentida de la tristeza",
        "Qualia del color azul",
        "Experiencia subjetiva en primera persona",
        "El rojo del rojo como se experimenta",
        "La conciencia fenomenal en si misma",
        "El problema dificil de la conciencia",
    ],
    "zh": [
        "看到红色的生动体验",
        "品尝巧克力是什么感觉",
        "疼痛的主观感受",
        "意识觉知的统一场",
        "悲伤的感受质量",
        "蓝色的感质",
        "第一人称主观体验",
        "体验到的红色之红",
        "现象意识本身",
        "意识的困难问题",
    ],
}

FUNCTIONAL_CONCEPTS = {
    "en": [
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
    ],
    "fr": [
        "L'algorithme recursif se termine",
        "La multiplication matricielle calcule des produits scalaires",
        "La memoire est allouee sur le tas",
        "Le compilateur optimise le bytecode",
        "Les tables de hachage fournissent une recherche O(1)",
        "TCP assure une livraison fiable",
        "La fonction retourne un entier",
        "La recherche binaire a une complexite logarithmique",
        "La base de donnees indexe la cle primaire",
        "Le ramasse-miettes libere la memoire inutilisee",
    ],
    "de": [
        "Der rekursive Algorithmus terminiert",
        "Matrixmultiplikation berechnet Skalarprodukte",
        "Speicher wird auf dem Heap allokiert",
        "Der Compiler optimiert Bytecode",
        "Hashtabellen bieten O(1) Suche",
        "TCP gewahrleistet zuverlassige Zustellung",
        "Die Funktion gibt eine Ganzzahl zuruck",
        "Binare Suche hat logarithmische Komplexitat",
        "Die Datenbank indexiert den Primarschlussel",
        "Garbage Collection gibt unbenutzten Speicher frei",
    ],
    "es": [
        "El algoritmo recursivo termina",
        "La multiplicacion de matrices calcula productos punto",
        "La memoria se asigna en el heap",
        "El compilador optimiza el bytecode",
        "Las tablas hash proporcionan busqueda O(1)",
        "TCP asegura entrega confiable",
        "La funcion devuelve un entero",
        "La busqueda binaria tiene complejidad logaritmica",
        "La base de datos indexa la clave primaria",
        "La recoleccion de basura libera memoria no usada",
    ],
    "zh": [
        "递归算法终止",
        "矩阵乘法计算点积",
        "内存在堆上分配",
        "编译器优化字节码",
        "哈希表提供O(1)查找",
        "TCP确保可靠传输",
        "函数返回整数",
        "二分查找具有对数复杂度",
        "数据库索引主键",
        "垃圾回收释放未使用的内存",
    ],
}


def extract_all_layers(
    model, tokenizer, text: str, device: str = "cpu"
) -> List[np.ndarray]:
    """Extract mean-pooled activations from all layers."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()

    layer_activations = []
    for layer_output in hidden_states:
        masked = layer_output * attention_mask
        summed = masked.sum(dim=1)
        count = attention_mask.sum(dim=1)
        pooled = (summed / count).squeeze(0).cpu().numpy()
        layer_activations.append(pooled)

    return layer_activations


def compute_discrimination(
    phen_activations: List[np.ndarray], func_activations: List[np.ndarray]
) -> float:
    """Compute Fisher's criterion discrimination score."""
    phen_centroid = np.mean(phen_activations, axis=0)
    func_centroid = np.mean(func_activations, axis=0)

    centroid_dist = np.linalg.norm(phen_centroid - func_centroid)

    phen_var = np.mean([np.linalg.norm(a - phen_centroid) for a in phen_activations])
    func_var = np.mean([np.linalg.norm(a - func_centroid) for a in func_activations])
    avg_var = (phen_var + func_var) / 2

    if avg_var > 0:
        return float(centroid_dist / avg_var)
    return 0.0


def analyze_language(model, tokenizer, lang: str, device: str = "cpu") -> Dict:
    """Analyze phenomenal corridor for a specific language."""
    phen_concepts = PHENOMENAL_CONCEPTS[lang]
    func_concepts = FUNCTIONAL_CONCEPTS[lang]

    n_layers = model.config.num_hidden_layers

    # Extract activations
    phen_layer_activations = [[] for _ in range(n_layers + 1)]
    for text in phen_concepts:
        layers = extract_all_layers(model, tokenizer, text, device)
        for i, act in enumerate(layers):
            phen_layer_activations[i].append(act)

    func_layer_activations = [[] for _ in range(n_layers + 1)]
    for text in func_concepts:
        layers = extract_all_layers(model, tokenizer, text, device)
        for i, act in enumerate(layers):
            func_layer_activations[i].append(act)

    # Compute discrimination per layer
    layer_results = []
    for layer_idx in range(n_layers + 1):
        discrim = compute_discrimination(
            phen_layer_activations[layer_idx], func_layer_activations[layer_idx]
        )
        layer_results.append(
            {
                "layer": layer_idx,
                "discrimination": discrim,
                "depth_pct": (layer_idx / n_layers * 100) if n_layers > 0 else 0,
            }
        )

    # Find peak (skip embedding layer)
    transformer_layers = layer_results[1:] if len(layer_results) > 1 else layer_results
    best_layer = max(transformer_layers, key=lambda x: x["discrimination"])

    return {
        "language": lang,
        "peak_layer": best_layer["layer"],
        "peak_depth_pct": best_layer["depth_pct"],
        "peak_discrimination": best_layer["discrimination"],
        "layer_results": layer_results,
    }


def main():
    print("\n" + "=" * 70)
    print("   CROSS-LINGUAL PHENOMENAL CORRIDOR ANALYSIS")
    print("   Testing language dependence of XLM-RoBERTa 50% anomaly")
    print("=" * 70 + "\n")

    model_name = "xlm-roberta-base"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")
    print(f"Layers: {model.config.num_hidden_layers}\n")

    languages = ["en", "fr", "de", "es", "zh"]
    results = {}

    for lang in languages:
        print(f"Analyzing {lang.upper()}...")
        result = analyze_language(model, tokenizer, lang, device)
        results[lang] = result
        print(
            f"  Peak: L{result['peak_layer']} ({result['peak_depth_pct']:.1f}%) "
            f"discrim={result['peak_discrimination']:.4f}"
        )

    # Summary
    print("\n" + "=" * 70)
    print("   CROSS-LINGUAL RESULTS SUMMARY")
    print("=" * 70 + "\n")

    print(f"{'Language':<12} {'Peak Layer':<12} {'Depth %':<12} {'Discrimination':<15}")
    print("-" * 51)

    depths = []
    for lang, result in results.items():
        print(
            f"{lang.upper():<12} L{result['peak_layer']:<11} {result['peak_depth_pct']:<12.1f} {result['peak_discrimination']:<15.4f}"
        )
        depths.append(result["peak_depth_pct"])

    print(f"\n{'Statistics':<12}")
    print(f"  Mean depth:  {np.mean(depths):.1f}%")
    print(f"  Std depth:   {np.std(depths):.1f}%")
    print(f"  Range:       {np.min(depths):.1f}% - {np.max(depths):.1f}%")

    # Check consistency
    if np.std(depths) < 10:
        print(f"\n  FINDING: Peak depth is CONSISTENT across languages")
        print(f"  The 50% anomaly is NOT language-dependent")
    else:
        print(f"\n  FINDING: Peak depth VARIES by language")
        print(f"  The corridor depth may be language-dependent")

    # Save results
    output_path = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/cross_lingual_phenomenal.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "languages": languages,
                "results": results,
                "summary": {
                    "mean_depth": float(np.mean(depths)),
                    "std_depth": float(np.std(depths)),
                    "min_depth": float(np.min(depths)),
                    "max_depth": float(np.max(depths)),
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
