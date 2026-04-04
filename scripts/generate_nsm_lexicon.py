#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Generate expanded NSM lexicon via Ollama (mistral:7b).

Decomposes English words into Wierzbicka's Natural Semantic Metalanguage (NSM)
primes with weights. Output is consumed by Symthaea's SpinozistClassifier.

Usage:
    python scripts/generate_nsm_lexicon.py [--model mistral:7b] [--output data/nsm_lexicon_expanded.json]

Requirements:
    - Ollama running locally (http://localhost:11434)
    - Model pulled: ollama pull mistral:7b
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

# The 65 NSM semantic primes (must match SemanticPrime enum in universal_semantics.rs)
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
    "When", "Now", "Before", "After", "LongTime", "ShortTime", "ForSomeTime", "InOneMoment",
    "Where", "Here", "Above", "Below", "Far", "Near", "Side", "Inside", "On",
    "Very", "More",
    "Like",
    "With",
}

# Few-shot examples from the hardcoded lexicon (ground truth)
FEW_SHOT_EXAMPLES = [
    {"word": "steal", "primes": [["Do", 1.0], ["Have", 0.8], ["Bad", 0.9], ["Not", 0.3]]},
    {"word": "help", "primes": [["Do", 1.0], ["Good", 1.0], ["Someone", 0.5]]},
    {"word": "generous", "primes": [["Good", 1.0], ["Have", 0.3], ["Someone", 0.3]]},
    {"word": "murder", "primes": [["Do", 1.0], ["Die", 1.0], ["Bad", 1.0], ["Want", 0.5]]},
    {"word": "honest", "primes": [["Good", 0.8], ["True", 0.9], ["Say", 0.4]]},
]

# Words already in the hardcoded lexicon (skip these)
EXISTING_WORDS = set()


def load_existing_words(rust_file: str) -> set:
    """Parse NsmLexicon::new() to extract existing word entries."""
    words = set()
    try:
        with open(rust_file) as f:
            for line in f:
                m = re.search(r'Self::insert\(&mut entries,\s*"([^"]+)"', line)
                if m:
                    words.add(m.group(1).lower())
    except FileNotFoundError:
        print(f"Warning: {rust_file} not found, proceeding without exclusion list", file=sys.stderr)
    return words


def load_word_list(path: str) -> list:
    """Load words from a text file (one per line) or extract from Social Chemistry JSON."""
    words = set()
    if path.endswith(".json") or path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    # Social Chemistry format: "rot" field contains the rule-of-thumb text
                    text = obj.get("rot", obj.get("text", obj.get("sentence", "")))
                    for word in re.findall(r"[a-zA-Z]+", text):
                        w = word.lower()
                        if len(w) >= 3 and w not in EXISTING_WORDS:
                            words.add(w)
                except json.JSONDecodeError:
                    continue
    else:
        with open(path) as f:
            for line in f:
                w = line.strip().lower()
                if len(w) >= 3 and w not in EXISTING_WORDS:
                    words.add(w)
    return sorted(words)


def default_word_list() -> list:
    """Common English moral/social vocabulary when no word source is provided."""
    words = [
        "abandon", "accept", "accuse", "achieve", "acknowledge", "admire",
        "afford", "agree", "allow", "annoy", "apologize", "appreciate",
        "argue", "assist", "avoid", "ban", "bargain", "behave", "believe",
        "belong", "blame", "bless", "boast", "borrow", "bother", "burden",
        "celebrate", "challenge", "claim", "commit", "communicate",
        "complain", "compromise", "condemn", "confess", "confide",
        "confront", "congratulate", "convince", "criticize", "dare",
        "debate", "decline", "dedicate", "defend", "defy", "demand",
        "deny", "depend", "deserve", "disappoint", "discipline",
        "dismiss", "disobey", "dominate", "doubt", "earn", "embrace",
        "empower", "endure", "enforce", "enjoy", "envy", "escape",
        "evaluate", "exclude", "excuse", "expect", "fail", "favor",
        "flatter", "forbid", "forgive", "glorify", "gossip", "govern",
        "grieve", "guarantee", "guide", "harass", "honor", "humiliate",
        "ignore", "impose", "improve", "include", "indulge", "inform",
        "inherit", "insist", "insult", "interfere", "intimidate",
        "invade", "invest", "isolate", "judge", "justify", "kidnap",
        "labor", "liberate", "manage", "mediate", "mentor", "mourn",
        "nourish", "obey", "object", "observe", "offend", "oppress",
        "organize", "owe", "pardon", "participate", "persecute",
        "persuade", "pledge", "possess", "preserve", "pretend",
        "privilege", "promise", "prosper", "punish", "pursue", "rebel",
        "reconcile", "refuse", "regret", "reject", "rely", "repay",
        "repent", "reproach", "request", "resent", "resist", "retaliate",
        "revenge", "reward", "sacrifice", "satisfy", "scold", "seduce",
        "serve", "shame", "shelter", "silence", "slander", "submit",
        "succeed", "surrender", "suspect", "sustain", "swear", "sympathize",
        "tempt", "tolerate", "torment", "torture", "treasure", "trespass",
        "triumph", "unite", "violate", "welcome", "withhold", "witness",
        "worship", "yield",
    ]
    return [w for w in words if w not in EXISTING_WORDS]


def query_ollama_single(word: str, model: str) -> dict:
    """Query Ollama for NSM decomposition of a single word."""
    prompt = f"""You are a linguist. Decompose the English word into semantic primes.

RULES:
- Use ONLY these exact prime names: Good, Bad, Do, Feel, Know, Want, Someone, Something, Happen, Move, Live, Die, Body, Big, Small, Very, More, Not, Because, If, Say, True, Think, See, Hear, Have, Be, Can, With
- Weights must be between 0.1 and 1.0
- Use 2-5 primes per word
- Do NOT invent new primes

Examples:
"steal" -> {{"primes": [["Do", 1.0], ["Have", 0.8], ["Bad", 0.9]]}}
"help" -> {{"primes": [["Do", 1.0], ["Good", 1.0], ["Someone", 0.5]]}}
"honest" -> {{"primes": [["Good", 0.8], ["True", 0.9], ["Say", 0.4]]}}
"murder" -> {{"primes": [["Do", 1.0], ["Die", 1.0], ["Bad", 1.0]]}}
"generous" -> {{"primes": [["Good", 1.0], ["Have", 0.3], ["Someone", 0.3]]}}

Word: "{word}"
"""

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.05, "num_predict": 256},
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
            response_text = result.get("response", "")
            parsed = json.loads(response_text)
            primes = parsed.get("primes", [])
            return {word: primes}
    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
        print(f"  Error for '{word}': {e}", file=sys.stderr)
        return {}


def query_ollama_batch(words: list, model: str) -> dict:
    """Query Ollama for each word individually (more reliable than batching)."""
    results = {}
    for word in words:
        result = query_ollama_single(word, model)
        results.update(result)
    return {"words": results}


def validate_and_clean_entry(word: str, primes: list) -> list | None:
    """Validate and clean a single word's decomposition.

    Returns cleaned primes list, or None if completely invalid.
    Fixes:
    - Strips hallucinated primes (not in VALID_PRIMES)
    - Negative weights → flip antonym prime or drop
    - Clamps weights to [0, 1]
    """
    if not isinstance(primes, list) or len(primes) == 0:
        return None

    # Antonym pairs for flipping negative weights
    antonyms = {"Good": "Bad", "Bad": "Good", "Big": "Small", "Small": "Big"}

    cleaned = []
    for pair in primes:
        if not isinstance(pair, list) or len(pair) != 2:
            continue  # Skip malformed pairs
        name, weight = pair
        if not isinstance(name, str) or not isinstance(weight, (int, float)):
            continue
        # Skip hallucinated primes
        if name not in VALID_PRIMES:
            continue
        # Handle negative weights: flip to antonym if available, else drop
        if weight < 0.0:
            if name in antonyms:
                name = antonyms[name]
                weight = min(abs(weight), 1.0)
            else:
                continue
        weight = min(weight, 1.0)
        if weight > 0.0:
            cleaned.append([name, round(weight, 2)])

    return cleaned if len(cleaned) > 0 else None


def main():
    parser = argparse.ArgumentParser(description="Generate expanded NSM lexicon via Ollama")
    parser.add_argument("--model", default="mistral:7b", help="Ollama model to use")
    parser.add_argument("--output", default="data/nsm_lexicon_expanded.json",
                        help="Output JSON file")
    parser.add_argument("--words", default=None,
                        help="Word source file (text or JSONL)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Words per Ollama query")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between batches")
    args = parser.parse_args()

    # Load existing lexicon words to skip
    global EXISTING_WORDS
    rust_file = os.path.join(os.path.dirname(__file__),
                             "../symthaea/src/hdc/spinozist_geometry.rs")
    EXISTING_WORDS = load_existing_words(rust_file)
    print(f"Loaded {len(EXISTING_WORDS)} existing lexicon words to skip")

    # Load or generate word list
    if args.words:
        words = load_word_list(args.words)
    else:
        words = default_word_list()
    print(f"Processing {len(words)} words")

    # Load existing output for resume capability
    output = {"words": {}}
    if os.path.exists(args.output):
        with open(args.output) as f:
            output = json.load(f)
        print(f"Resuming: {len(output['words'])} words already processed")
        words = [w for w in words if w not in output["words"]]
        print(f"Remaining: {len(words)} words")

    # Process in batches
    total_added = 0
    total_rejected = 0

    for i in range(0, len(words), args.batch_size):
        batch = words[i:i + args.batch_size]
        print(f"Batch {i // args.batch_size + 1}: {', '.join(batch)}")

        result = query_ollama_batch(batch, args.model)
        if not result:
            continue
        # Handle multiple response formats from the LLM:
        # 1. {"words": {"word1": [...], "word2": [...]}}
        # 2. {"word1": [...], "word2": [...]}
        # 3. {"primes": [...]} (single word, use first batch word)
        if "words" in result and isinstance(result["words"], dict):
            words_dict = result["words"]
        elif "primes" in result and isinstance(result["primes"], list):
            # Single-word response format
            words_dict = {batch[0]: result["primes"]}
        else:
            # Flat format: keys are words directly
            words_dict = {k: v for k, v in result.items()
                         if isinstance(v, list) and k != "words"}

        for word, primes in words_dict.items():
            word_lower = word.lower()
            if word_lower in EXISTING_WORDS:
                continue
            cleaned = validate_and_clean_entry(word_lower, primes)
            if cleaned is not None:
                output["words"][word_lower] = cleaned
                total_added += 1
            else:
                print(f"  Rejected: {word_lower} (invalid decomposition)", file=sys.stderr)
                total_rejected += 1

        # Save after each batch (resume capability)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, sort_keys=True)

        if i + args.batch_size < len(words):
            time.sleep(args.delay)

    print(f"\nDone. Added: {total_added}, Rejected: {total_rejected}")
    print(f"Total lexicon entries: {len(output['words'])}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
