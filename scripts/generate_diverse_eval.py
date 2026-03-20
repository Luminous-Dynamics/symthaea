#!/usr/bin/env python3
"""Generate diverse evaluation dataset for Broca quality testing.

Creates samples with balanced coverage of:
- All 8 intents (acknowledge, answer, clarify, propose, uncertainty, reflect, continue, unknown)
- All 5 epistemic levels (0=Certain, 1=Probable, 2=Uncertain, 3=NoEvidence, 4=OutOfDomain)
- Varied emotional states (valence, arousal, warmth)
- 24-channel format

Usage:
    python generate_diverse_eval.py OUTPUT.jsonl [--samples-per-combo 3]
"""

import json
import random
import sys

INTENTS = [
    "acknowledge", "answer", "clarify", "propose",
    "uncertainty", "reflect", "continue", "unknown"
]

EPISTEMIC_LEVELS = {
    0.0: "Certain",
    1.0: "Probable",
    2.0: "Uncertain",
    3.0: "NoEvidence",
    4.0: "OutOfDomain",
}

# Representative target texts for each intent × epistemic combination
TEMPLATES = {
    "acknowledge": {
        0.0: "I understand completely. That's a clear and well-established point.",
        1.0: "That's a solid observation. The evidence supports your view.",
        2.0: "I hear what you're saying, though there may be different perspectives.",
        3.0: "I acknowledge your point, though I'm not sure of the underlying basis.",
        4.0: "I hear you, though this falls outside my area of knowledge.",
    },
    "answer": {
        0.0: "The answer is definitive: water boils at 100 degrees Celsius at sea level.",
        1.0: "Based on the available evidence, the answer is likely yes.",
        2.0: "The answer is somewhat uncertain, but current research suggests that it's possible.",
        3.0: "I can attempt an answer, but I should note there isn't strong evidence either way.",
        4.0: "This question falls outside my domain. I'd recommend consulting a specialist.",
    },
    "clarify": {
        0.0: "To clarify: the mechanism works through direct causal interaction.",
        1.0: "Let me clarify what I mean. The process involves several well-documented steps.",
        2.0: "Could you help me understand what you mean? I want to make sure I'm addressing the right question.",
        3.0: "I'd like to clarify, but I'm not entirely sure what framework would apply here.",
        4.0: "I'd need more context to clarify. This topic is beyond my current understanding.",
    },
    "propose": {
        0.0: "I propose we proceed with the established method, which has strong empirical support.",
        1.0: "Here's a suggestion: we could try the approach that has worked in similar cases.",
        2.0: "One possibility might be to explore a different angle, though I'm not certain it would work.",
        3.0: "I could suggest an approach, but I want to be transparent that this is speculative.",
        4.0: "I hesitate to propose anything specific here, as this is outside my expertise.",
    },
    "uncertainty": {
        0.0: "While there are some unknowns, the core mechanism is well understood.",
        1.0: "There is some uncertainty, but the general direction seems clear.",
        2.0: "I'm genuinely uncertain about this. The evidence is mixed and inconclusive.",
        3.0: "I honestly don't know. There isn't enough evidence to form a clear view.",
        4.0: "This is entirely outside what I can speak to with any confidence.",
    },
    "reflect": {
        0.0: "Reflecting on this, the pattern is clear and consistent with established theory.",
        1.0: "Looking at this more carefully, I think the general trend is meaningful.",
        2.0: "I find myself uncertain as I reflect on this. There seem to be competing explanations.",
        3.0: "The more I think about this, the less certain I become about any particular answer.",
        4.0: "I'm reflecting on how little I actually know about this domain.",
    },
    "continue": {
        0.0: "Continuing from where we left off, the next step follows logically.",
        1.0: "Building on that point, we can probably extend the analysis further.",
        2.0: "To continue, though I should note some uncertainty about the next steps.",
        3.0: "I can continue, but I want to acknowledge I'm moving into less familiar territory.",
        4.0: "I could continue, but I'd be venturing well outside my area of knowledge.",
    },
    "unknown": {
        0.0: "Thank you for sharing. The information is clear and straightforward.",
        1.0: "That's interesting. I think I follow the general idea.",
        2.0: "I'm processing what you've shared. Some aspects are clearer than others.",
        3.0: "I appreciate you sharing, though I'm not sure how to interpret this fully.",
        4.0: "I'm not sure what to make of this. It seems quite different from what I'm familiar with.",
    },
}


def make_channels(intent_idx: int, epistemic: float, rng: random.Random) -> list:
    """Create 24-channel thought vector."""
    channels = [0.0] * 24

    # Intent one-hot (channels 0-7)
    channels[intent_idx] = 1.0

    # Epistemic status (channel 8)
    channels[8] = epistemic

    # Emotional channels (9-11) — varied
    channels[9] = rng.uniform(-0.8, 0.8)   # valence
    channels[10] = rng.uniform(0.1, 0.9)   # arousal
    channels[11] = rng.uniform(0.2, 0.9)   # warmth

    # Consciousness channels (12-14)
    channels[12] = rng.uniform(0.3, 0.9)   # psi
    channels[13] = rng.uniform(0.2, 0.8)   # meta_awareness
    channels[14] = rng.uniform(0.3, 0.9)   # coherence

    # Relationship (15-16)
    channels[15] = rng.uniform(0.0, 5.0)   # relationship_stage
    channels[16] = rng.uniform(0.3, 0.9)   # trust

    # Mood temperature (17) — correlate with epistemic
    base_temp = 0.7 + epistemic * 0.15
    channels[17] = min(2.0, base_temp + rng.uniform(-0.1, 0.1))

    # Computed answer / concept count (18-19)
    channels[18] = 1.0 if epistemic < 2.0 else rng.choice([0.0, 1.0])
    channels[19] = rng.uniform(0.0, 8.0)

    # New channels (20-23)
    channels[20] = rng.uniform(0.0, 0.8)   # time_pressure
    channels[21] = max(0.0, 1.0 - epistemic * 0.2 + rng.uniform(-0.1, 0.1))  # domain_familiarity
    channels[22] = rng.uniform(0.0, 1.0)   # social_context
    channels[23] = max(0.0, 1.0 - epistemic * 0.2 + rng.uniform(-0.1, 0.1))  # response_confidence

    return [round(c, 6) for c in channels]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    output_path = sys.argv[1]
    samples_per_combo = 3
    if len(sys.argv) >= 3:
        samples_per_combo = int(sys.argv[2].replace("--samples-per-combo=", "").replace("--samples-per-combo", "").strip() or "3")

    rng = random.Random(42)
    samples = []

    for intent_idx, intent_name in enumerate(INTENTS):
        for epistemic, ep_name in EPISTEMIC_LEVELS.items():
            for _ in range(samples_per_combo):
                channels = make_channels(intent_idx, epistemic, rng)
                text = TEMPLATES[intent_name][epistemic]
                samples.append({
                    "channels": channels,
                    "target_text": text,
                    "target_ids": [],  # Will be populated by tokenizer
                })

    rng.shuffle(samples)

    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    total = len(samples)
    print(f"Generated {total} diverse eval samples")
    print(f"  {len(INTENTS)} intents × {len(EPISTEMIC_LEVELS)} epistemic levels × {samples_per_combo} samples")
    print(f"  Written to {output_path}")


if __name__ == "__main__":
    main()
