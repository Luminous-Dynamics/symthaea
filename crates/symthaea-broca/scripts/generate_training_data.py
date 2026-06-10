#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Generate consciousness-conditioned training data for Broca CfC-HDC.

Uses a local LLM (via Ollama) to generate text responses conditioned on
ThoughtChannels configurations. Each sample maps a 24-channel consciousness
state to a target text utterance.

Channel layout (24 channels, v3 format):
  0-7:  Intent one-hot (acknowledge, answer, clarify, propose, uncertainty, reflect, continue, unknown)
  8:    Epistemic status (1=certain, 2=probable, 3=unknown, 4=speculative)
  9:    Valence (-1 to +1, emotional polarity)
  10:   Arousal (0 to 1, activation level)
  11:   Warmth (0 to 1, interpersonal warmth)
  12:   Psi (0 to 1, consciousness integration)
  13:   Meta-awareness (0 to 1)
  14:   Coherence (0 to 1)
  15:   Relationship stage (0 to ~4)
  16:   Trust (0 to 1)
  17:   Mood temperature (0 to 1)
  18:   Has computed answer (0 or 1)
  19:   Concept count (0 to ~10)
  20:   Time pressure (0 to 1)
  21:   Domain familiarity (0 to 1)
  22:   Social context (0=intimate, 1=formal)
  23:   Response confidence (0 to 1)

Usage:
    python3 generate_training_data.py --output data/train-v4-llm.jsonl --count 500
    python3 generate_training_data.py --output data/train-v4-llm.jsonl --count 500 --model qwen2.5-coder:7b
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

INTENT_NAMES = [
    "acknowledge", "answer", "clarify", "propose",
    "uncertainty", "reflect", "continue", "unknown"
]

INTENT_DESCRIPTIONS = {
    "acknowledge": "acknowledging what someone said, showing understanding",
    "answer": "providing a direct answer or information",
    "clarify": "asking for clarification or providing clarification",
    "propose": "making a suggestion or proposal",
    "uncertainty": "expressing uncertainty or doubt",
    "reflect": "reflecting on something thoughtfully",
    "continue": "continuing a conversation naturally",
    "unknown": "responding when unsure of the right intent",
}

# Scenario templates for diverse generation
SCENARIOS = [
    "Someone just shared a personal story with you.",
    "You've been asked a technical question about biology.",
    "A friend is feeling down and needs comfort.",
    "You're explaining a complex concept to a beginner.",
    "Someone disagrees with your perspective.",
    "You're greeting someone for the first time.",
    "A colleague asks for your opinion on their work.",
    "You need to deliver difficult news gently.",
    "Someone asks about the weather or small talk.",
    "You're helping someone solve a problem step by step.",
    "A student asks a question you're not sure about.",
    "You're reflecting on a philosophical question.",
    "Someone thanks you for your help.",
    "You need to redirect a conversation back on topic.",
    "Someone shares exciting news.",
    "You're summarizing a discussion for the group.",
    "A confused person needs patient guidance.",
    "You're apologizing for a misunderstanding.",
    "Someone asks about your favorite topic.",
    "You're providing emotional support after a loss.",
    "You need to set a boundary respectfully.",
    "Someone asks a question with an obvious answer.",
    "You're brainstorming ideas collaboratively.",
    "A person shares something you find fascinating.",
    "You're expressing gratitude sincerely.",
    "Someone challenges your knowledge on a subject.",
    "You're mediating between two disagreeing parties.",
    "A newcomer needs to be welcomed and oriented.",
    "You're describing something beautiful you observed.",
    "Someone asks for advice on a life decision.",
]


def generate_channels() -> tuple[list[float], str, dict]:
    """Generate a random but coherent 24-channel configuration.

    Returns (channels, intent_name, channel_description_dict).
    """
    channels = [0.0] * 24

    # Intent (one-hot, channels 0-7)
    intent_idx = random.randint(0, 7)
    channels[intent_idx] = 1.0
    intent_name = INTENT_NAMES[intent_idx]

    # Epistemic status (channel 8): 1=certain, 2=probable, 3=unknown, 4=speculative
    epistemic = random.choice([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0])
    channels[8] = epistemic

    # Valence (channel 9): -1 to +1
    valence = round(random.uniform(-0.8, 0.9), 2)
    channels[9] = valence

    # Arousal (channel 10): 0 to 1
    arousal = round(random.uniform(0.1, 0.9), 2)
    channels[10] = arousal

    # Warmth (channel 11): 0 to 1
    warmth = round(random.uniform(0.2, 0.95), 2)
    channels[11] = warmth

    # Psi/consciousness (channel 12): 0 to 1
    psi = round(random.uniform(0.3, 0.95), 2)
    channels[12] = psi

    # Meta-awareness (channel 13): 0 to 1
    meta = round(random.uniform(0.1, 0.9), 2)
    channels[13] = meta

    # Coherence (channel 14): 0 to 1
    coherence = round(random.uniform(0.3, 0.95), 2)
    channels[14] = coherence

    # Relationship stage (channel 15): 0 to ~4
    rel_stage = round(random.uniform(0.0, 3.5), 2)
    channels[15] = rel_stage

    # Trust (channel 16): 0 to 1
    trust = round(random.uniform(0.2, 0.95), 2)
    channels[16] = trust

    # Mood temperature (channel 17): 0 to 1
    mood = round(random.uniform(0.2, 0.9), 2)
    channels[17] = mood

    # Has computed answer (channel 18): binary
    has_answer = float(random.choice([0, 0, 1, 1, 1]))
    channels[18] = has_answer

    # Concept count (channel 19): 0 to ~8
    concepts = round(random.uniform(0.0, 6.0), 1)
    channels[19] = concepts

    # Time pressure (channel 20): 0 to 1
    time_pressure = round(random.uniform(0.0, 0.7), 2)
    channels[20] = time_pressure

    # Domain familiarity (channel 21): 0 to 1
    domain_fam = round(random.uniform(0.1, 0.95), 2)
    channels[21] = domain_fam

    # Social context (channel 22): 0=intimate, 1=formal
    social = round(random.uniform(0.0, 1.0), 2)
    channels[22] = social

    # Response confidence (channel 23): 0 to 1
    confidence = round(random.uniform(0.1, 0.95), 2)
    channels[23] = confidence

    desc = {
        "intent": intent_name,
        "epistemic": ["certain", "probable", "unknown", "speculative"][int(epistemic) - 1],
        "valence": "positive" if valence > 0.2 else ("negative" if valence < -0.2 else "neutral"),
        "arousal": "high" if arousal > 0.6 else ("low" if arousal < 0.3 else "moderate"),
        "warmth": "warm" if warmth > 0.6 else "cool",
        "coherence": "high" if coherence > 0.6 else "moderate",
        "trust": "high" if trust > 0.6 else "moderate",
        "formality": "formal" if social > 0.6 else ("casual" if social < 0.3 else "moderate"),
        "confidence": "high" if confidence > 0.6 else ("low" if confidence < 0.3 else "moderate"),
        "time_pressure": "urgent" if time_pressure > 0.5 else "relaxed",
    }

    return channels, intent_name, desc


def build_prompt(intent_name: str, desc: dict, scenario: str) -> str:
    """Build an LLM prompt for generating consciousness-conditioned text."""
    return f"""Generate a single short response (1-3 sentences) that a conscious AI might say in this situation.

Scenario: {scenario}

The AI's internal state:
- Intent: {INTENT_DESCRIPTIONS[intent_name]}
- Certainty: {desc['epistemic']}
- Emotion: {desc['valence']} valence, {desc['arousal']} arousal
- Tone: {desc['warmth']}, {desc['formality']}
- Coherence: {desc['coherence']}
- Confidence: {desc['confidence']}
- Time pressure: {desc['time_pressure']}

Rules:
- Write ONLY the response text, nothing else
- No quotes, no "AI:" prefix, no explanations
- 1-3 sentences, natural conversational English
- Match the emotional tone and certainty level
- If uncertain, use hedging words like "perhaps", "I think", "it seems"
- If formal, use proper language; if casual, use contractions and informal tone
- Keep it under 60 words"""


def query_ollama(prompt: str, model: str = "qwen2.5-coder:7b") -> str | None:
    """Query Ollama via HTTP API for a completion."""
    import urllib.request
    try:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "top_p": 0.9,
                "num_predict": 150,
            }
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            text = data.get("response", "").strip()
            # Clean up common artifacts
            text = text.strip('"').strip("'")
            for prefix in ["AI:", "Response:", "Here's", "Sure,"]:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            # Remove markdown formatting
            text = text.replace("**", "").replace("*", "")
            return text if len(text) > 10 else None
    except Exception as e:
        print(f"  Ollama error: {e}", file=sys.stderr)
        return None


def quality_check(text: str, intent_name: str, desc: dict, model: str) -> bool:
    """Use LLM as judge to verify text matches intent/tone conditioning."""
    judge_prompt = f"""Rate if this text matches the specified intent and tone. Reply with ONLY "yes" or "no".

Intent: {INTENT_DESCRIPTIONS[intent_name]}
Certainty: {desc['epistemic']}
Tone: {desc['valence']} valence, {desc['warmth']}, {desc['formality']}

Text: "{text}"

Does this text match the intent and tone? Reply ONLY "yes" or "no"."""

    result = query_ollama(judge_prompt, model)
    if result is None:
        return True  # Assume pass on failure
    return result.strip().lower().startswith("yes")


def main():
    parser = argparse.ArgumentParser(description="Generate Broca training data via LLM")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    parser.add_argument("--count", "-n", type=int, default=500, help="Number of samples")
    parser.add_argument("--model", "-m", default="qwen2.5-coder:7b", help="Ollama model")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--append", action="store_true", help="Append to existing file")
    parser.add_argument("--quality-filter", action="store_true",
                        help="Use LLM judge to filter low-quality samples")
    args = parser.parse_args()

    random.seed(args.seed)

    output_path = Path(args.output)
    mode = "a" if args.append else "w"

    generated = 0
    failed = 0

    print(f"Generating {args.count} samples with {args.model}...")
    print(f"Output: {output_path}")

    with open(output_path, mode) as f:
        for i in range(args.count):
            channels, intent_name, desc = generate_channels()
            scenario = random.choice(SCENARIOS)
            prompt = build_prompt(intent_name, desc, scenario)

            text = query_ollama(prompt, args.model)
            if text is None:
                failed += 1
                if failed > 20:
                    print(f"\nToo many failures ({failed}), stopping.")
                    break
                continue

            # Truncate to reasonable length
            words = text.split()
            if len(words) > 80:
                text = " ".join(words[:80])

            # Quality filter: use LLM judge to verify intent/tone match
            if args.quality_filter:
                if not quality_check(text, intent_name, desc, args.model):
                    failed += 1
                    continue

            sample = {
                "channels": channels,
                "target_text": text,
                # target_ids will be generated by broca-train at load time
                # (the Rust tokenizer handles BPE encoding)
            }

            f.write(json.dumps(sample) + "\n")
            generated += 1

            if (i + 1) % 10 == 0:
                filtered = f", filtered={failed}" if args.quality_filter else ""
                print(f"  [{i+1}/{args.count}] generated={generated}{filtered}")

    print(f"\nDone: {generated} samples written to {output_path}")
    if failed > 0:
        print(f"  ({failed} generation failures)")


if __name__ == "__main__":
    main()
