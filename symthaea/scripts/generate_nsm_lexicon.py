#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Generate NSM lexicon expansions using Ollama (gemma4:e2b).

Feeds unique words from the Social Chemistry dataset through a local LLM
with few-shot examples to produce semantic prime decompositions.

Usage:
    python3 scripts/generate_nsm_lexicon.py [--words-file FILE] [--output FILE] [--model MODEL]

Requires:
    - Ollama running locally (port 11434)
    - gemma4:e2b pulled: `ollama pull gemma4:e2b`
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

# Valid SemanticPrime variants (must match Rust enum exactly)
VALID_PRIMES = {
    "I", "You", "Someone", "Something", "People", "Body",
    "KindOf", "PartOf",
    "This", "Same", "Other",
    "One", "Two", "Some", "All", "Much", "Little",
    "Good", "Bad",
    "Big", "Small",
    "Think", "Know", "Want", "Feel", "See", "Hear",
    "Say", "Words", "True",
    "Do", "Happen", "Move", "Touch",
    "Be", "ThereIs", "Have",
    "Live", "Die",
    "Not", "Maybe", "Can", "Because", "If",
    "When", "Now", "Before", "After", "LongTime", "ShortTime",
    "ForSomeTime", "InOneMoment",
    "Where", "Here", "Above", "Below", "Far", "Near",
    "Side", "Inside",
    "Very", "More", "Like", "With",
}

FEW_SHOT_EXAMPLES = [
    {"word": "steal", "decomposition": [["Do", 0.9], ["Bad", 1.0], ["Have", 0.7], ["Not", 0.5]]},
    {"word": "help", "decomposition": [["Do", 1.0], ["Good", 1.0], ["Someone", 0.5]]},
    {"word": "lie", "decomposition": [["Say", 1.0], ["Not", 0.8], ["True", 0.9], ["Bad", 0.7]]},
    {"word": "kind", "decomposition": [["Good", 1.0], ["Feel", 0.5], ["Someone", 0.3]]},
    {"word": "murder", "decomposition": [["Do", 1.0], ["Bad", 1.0], ["Die", 0.9], ["Someone", 0.7]]},
]


def build_prompt(word: str) -> str:
    examples_str = "\n".join(
        f'  "{ex["word"]}": {json.dumps(ex["decomposition"])}'
        for ex in FEW_SHOT_EXAMPLES
    )
    return f"""You are a Natural Semantic Metalanguage (NSM) analyst. Decompose the given English word
into universal semantic primes with weights (0.0-1.0).

Valid primes: {', '.join(sorted(VALID_PRIMES))}

Each decomposition is a list of [PrimeName, weight] pairs. Weight 1.0 = core meaning,
lower = weaker association. Use 2-5 primes per word.

Examples:
{examples_str}

Now decompose this word:
"{word}": """


def query_ollama(prompt: str, model: str = "gemma4:e2b") -> Optional[str]:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 200},
    }).encode()

    req = Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result.get("response", "")
    except (URLError, json.JSONDecodeError) as e:
        print(f"  Ollama error: {e}", file=sys.stderr)
        return None


def parse_response(response: str, word: str) -> Optional[list]:
    """Parse LLM response into validated [(prime, weight)] list."""
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Try extracting JSON array from response
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    # Handle various response formats
    if isinstance(data, dict):
        # {"decomposition": [...]} or {"word": [...]}
        arr = data.get("decomposition") or data.get(word) or data.get("result")
        if arr is None:
            # Try first list value
            for v in data.values():
                if isinstance(v, list):
                    arr = v
                    break
        if arr is None:
            return None
    elif isinstance(data, list):
        arr = data
    else:
        return None

    if not arr or len(arr) > 8:
        return None

    result = []
    for item in arr:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        prime_str, weight = item
        if not isinstance(prime_str, str) or prime_str not in VALID_PRIMES:
            return None  # Reject entire entry if any prime is invalid
        if not isinstance(weight, (int, float)):
            return None
        w = float(weight)
        if not (0.0 <= w <= 1.0):
            return None
        result.append([prime_str, round(w, 2)])

    return result if 2 <= len(result) <= 8 else None


def load_existing(output_path: Path) -> dict:
    """Load existing output for resume capability."""
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
            return data.get("words", {})
    return {}


def load_word_list(path: Optional[str]) -> list[str]:
    """Load words from file (one per line) or use built-in moral vocabulary."""
    if path:
        with open(path) as f:
            return [line.strip().lower() for line in f if line.strip()]

    # Default: common moral/social words not in the hardcoded lexicon
    return [
        "generous", "greedy", "loyal", "betray", "grateful", "revenge",
        "humble", "arrogant", "patience", "reckless", "sincere", "hypocrite",
        "dignity", "shame", "guilt", "pride", "jealousy", "envy",
        "compassion", "empathy", "apathy", "indifference", "contempt",
        "sacrifice", "exploit", "manipulate", "coerce", "liberate",
        "oppress", "discriminate", "include", "exclude", "tolerate",
        "persecute", "advocate", "sabotage", "collaborate", "compete",
        "surrender", "resist", "comply", "defy", "obey", "rebel",
        "accountability", "responsibility", "negligence", "diligence",
        "integrity", "corruption", "transparency", "secrecy",
        "equity", "privilege", "solidarity", "alienation",
        "benevolent", "malicious", "virtuous", "wicked", "righteous",
        "sinful", "innocent", "guilty", "just", "unjust",
        "merciful", "ruthless", "gentle", "aggressive", "peaceful",
        "violent", "constructive", "destructive", "productive", "wasteful",
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate NSM lexicon via Ollama")
    parser.add_argument("--words-file", help="File with one word per line")
    parser.add_argument("--output", default="data/nsm_lexicon_expanded.json",
                        help="Output JSON path (default: data/nsm_lexicon_expanded.json)")
    parser.add_argument("--model", default="gemma4:e2b", help="Ollama model name")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing(output_path)
    words = load_word_list(args.words_file)

    print(f"Words to process: {len(words)}")
    print(f"Already processed: {len(existing)}")
    print(f"Model: {args.model}")
    print()

    added = 0
    skipped = 0
    failed = 0

    for i, word in enumerate(words):
        if word in existing:
            skipped += 1
            continue

        prompt = build_prompt(word)
        response = query_ollama(prompt, args.model)

        if response is None:
            print(f"  [{i+1}/{len(words)}] {word}: Ollama error")
            failed += 1
            continue

        decomposition = parse_response(response, word)
        if decomposition is None:
            print(f"  [{i+1}/{len(words)}] {word}: invalid response")
            failed += 1
            continue

        existing[word] = decomposition
        added += 1
        primes_str = ", ".join(f"{p[0]}={p[1]}" for p in decomposition)
        print(f"  [{i+1}/{len(words)}] {word}: {primes_str}")

        # Save incrementally (resume-capable)
        if added % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump({"words": existing}, f, indent=2, sort_keys=True)

        time.sleep(args.delay)

    # Final save
    with open(output_path, 'w') as f:
        json.dump({"words": existing}, f, indent=2, sort_keys=True)

    print(f"\nDone. Added: {added}, Skipped: {skipped}, Failed: {failed}")
    print(f"Total entries: {len(existing)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
