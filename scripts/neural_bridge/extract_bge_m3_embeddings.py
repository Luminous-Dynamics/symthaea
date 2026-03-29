#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
BGE-M3 Embedding Extraction

Extracts embeddings from BAAI/bge-m3 model for the expanded concept dataset.
These embeddings are used to train the HDC probe for Neural Bridge v2.

Usage:
    python extract_bge_m3_embeddings.py --output data/neural_bridge/bge_m3_embeddings.npz

Requires:
    pip install sentence-transformers numpy

    Or use the FlagEmbedding library:
    pip install FlagEmbedding numpy
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# Try FlagEmbedding (official BGE implementation) first
try:
    from FlagEmbedding import BGEM3FlagModel

    HAS_FLAG = True
except ImportError:
    HAS_FLAG = False

# Fallback to sentence-transformers
try:
    from sentence_transformers import SentenceTransformer

    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

if not HAS_FLAG and not HAS_SBERT:
    raise ImportError(
        "Neither FlagEmbedding nor sentence-transformers available. "
        "Install with: pip install FlagEmbedding  or  pip install sentence-transformers"
    )


# Expanded concept dataset (122 concepts from NSM + technical + scientific + philosophical + emotional + social)
EXPANDED_CONCEPTS = {
    # NSM Substantives (8)
    "I": [
        "I am here",
        "I think therefore I am",
        "I exist in this moment",
        "I am aware",
    ],
    "You": [
        "You are reading this",
        "You exist too",
        "You have thoughts",
        "You understand",
    ],
    "Someone": [
        "Someone is there",
        "Someone did this",
        "Someone exists",
        "Someone knows",
    ],
    "Something": [
        "Something happened",
        "Something exists",
        "Something is here",
        "Something strange",
    ],
    "People": [
        "People are many",
        "People live together",
        "People have feelings",
        "People communicate",
    ],
    "Body": [
        "My body moves",
        "The body is physical",
        "Body and mind connect",
        "Body needs rest",
    ],
    "This": [
        "This is it",
        "This moment matters",
        "This thing here",
        "This present instant",
    ],
    "Same": [
        "Same as before",
        "The same thing",
        "Same pattern repeats",
        "Same essence",
    ],
    "Other": [
        "Other than this",
        "The other side",
        "Other possibilities",
        "Other beings",
    ],
    # NSM Determiners (2)
    "One": [
        "One thing only",
        "One single entity",
        "One at a time",
        "One unique instance",
    ],
    "Two": [
        "Two things exist",
        "Two sides present",
        "Two alternatives",
        "Two companions",
    ],
    # NSM Quantifiers (4)
    "Some": [
        "Some things exist",
        "Some people know",
        "Some amount",
        "Some possibilities",
    ],
    "All": [
        "All things connected",
        "All beings matter",
        "All of everything",
        "All together",
    ],
    "Many": [
        "Many things happen",
        "Many people exist",
        "Many possibilities",
        "Many connections",
    ],
    "Much": ["Much to learn", "Much value here", "Much meaning", "Much importance"],
    # NSM Evaluators (3)
    "Good": [
        "Good is beneficial",
        "Good outcome desired",
        "Good action helps",
        "Good intention matters",
    ],
    "Bad": [
        "Bad harms others",
        "Bad outcome avoided",
        "Bad action hurts",
        "Bad intention damages",
    ],
    "Big": ["Big and vast", "Big importance", "Big magnitude", "Big significance"],
    "Small": [
        "Small and tiny",
        "Small detail",
        "Small magnitude",
        "Small but significant",
    ],
    # NSM Mental Predicates (7)
    "Think": [
        "I think deeply",
        "Think about meaning",
        "Think carefully",
        "Think before acting",
    ],
    "Know": [
        "I know this truth",
        "Know for certain",
        "Know through experience",
        "Know intuitively",
    ],
    "Want": [
        "I want something",
        "Want to achieve",
        "Want deeply",
        "Want to understand",
    ],
    "Feel": ["I feel emotion", "Feel the sensation", "Feel deeply", "Feel connected"],
    "See": ["I see clearly", "See the truth", "See with eyes", "See understanding"],
    "Hear": ["I hear sounds", "Hear the message", "Hear clearly", "Hear understanding"],
    # NSM Speech (2)
    "Say": [
        "I say words",
        "Say something meaningful",
        "Say with intention",
        "Say clearly",
    ],
    "Words": [
        "Words have meaning",
        "Words convey thoughts",
        "Words connect minds",
        "Words matter",
    ],
    # NSM Logical (1)
    "True": [
        "True is accurate",
        "True statement verified",
        "True understanding",
        "True knowledge",
    ],
    # NSM Actions (4)
    "Do": ["I do something", "Do the action", "Do with purpose", "Do intentionally"],
    "Happen": [
        "Something will happen",
        "Things happen naturally",
        "Happen unexpectedly",
        "Happen inevitably",
    ],
    "Move": [
        "I move forward",
        "Move through space",
        "Move with intention",
        "Move toward goal",
    ],
    "Touch": [
        "I touch gently",
        "Touch the surface",
        "Touch with care",
        "Touch to connect",
    ],
    # NSM Spatial (7)
    "Where": [
        "Where is it",
        "Where things are",
        "Where location matters",
        "Where to find",
    ],
    "Here": ["Here in this place", "Here and now", "Here present", "Here existing"],
    "Above": [
        "Above the ground",
        "Above in position",
        "Above in hierarchy",
        "Above beyond",
    ],
    "Below": [
        "Below the surface",
        "Below in position",
        "Below underneath",
        "Below hidden",
    ],
    "Far": ["Far away distant", "Far from here", "Far in distance", "Far reaching"],
    "Near": ["Near to me", "Near in proximity", "Near close by", "Near at hand"],
    "Side": ["On this side", "Side by side", "Side position", "Side aspect"],
    "Inside": [
        "Inside the boundary",
        "Inside contained",
        "Inside enclosed",
        "Inside interior",
    ],
    # NSM Temporal (6)
    "When": [
        "When it happens",
        "When the time comes",
        "When appropriate",
        "When ready",
    ],
    "Now": ["Now at this moment", "Now present time", "Now immediately", "Now is real"],
    "Before": [
        "Before this happened",
        "Before in time",
        "Before the event",
        "Before existing",
    ],
    "After": [
        "After this happens",
        "After the event",
        "After in time",
        "After completion",
    ],
    "Long_Time": [
        "A long time passes",
        "Long duration extends",
        "Long time span",
        "Long waiting",
    ],
    "Short_Time": [
        "A short time passes",
        "Brief duration",
        "Short time span",
        "Quick moment",
    ],
    "Moment": [
        "A single moment",
        "Moment in time",
        "Moment captured",
        "Moment experienced",
    ],
    # NSM Logical (5)
    "Not": ["Not this thing", "Not happening", "Not true", "Not present"],
    "Maybe": ["Maybe possible", "Maybe uncertain", "Maybe could be", "Maybe exists"],
    "Can": ["I can do this", "Can accomplish", "Can achieve", "Can realize"],
    "Because": [
        "Because of reason",
        "Because this happened",
        "Because cause",
        "Because explanation",
    ],
    "If": [
        "If this happens",
        "If condition met",
        "If possible",
        "If circumstances allow",
    ],
    # NSM Taxonomy (3)
    "Like": ["Like similar to", "Like comparison", "Like resembling", "Like analogous"],
    "Kind": ["A kind of thing", "Kind category", "Kind type", "Kind classification"],
    "Part": ["Part of whole", "Part belonging", "Part component", "Part element"],
    # NSM Existence (3)
    "There_Is": [
        "There is something",
        "There exists",
        "There is presence",
        "There is being",
    ],
    "Have": ["I have something", "Have possession", "Have ownership", "Have existence"],
    "Live": ["I live existence", "Live life", "Live experiencing", "Live being"],
    "Die": ["Die ending life", "Die cessation", "Die termination", "Die transition"],
    # NixOS/Technical Domain (12)
    "NixOS_Package": [
        "A NixOS package provides software",
        "Packages in Nix are reproducible",
        "Package derivation builds output",
        "Package manager handles dependencies",
    ],
    "NixOS_Service": [
        "A NixOS service runs daemon",
        "Service configuration in module",
        "Service manages process",
        "Systemd service integration",
    ],
    "Nix_Derivation": [
        "Derivation builds reproducibly",
        "Derivation defines build steps",
        "Derivation outputs stored",
        "Build derivation from expression",
    ],
    "Nix_Flake": [
        "Flake is project definition",
        "Flake provides inputs outputs",
        "Flake enables reproducibility",
        "Flake lock pins versions",
    ],
    "Nix_Store": [
        "Nix store holds outputs",
        "Store paths are immutable",
        "Store enables sharing",
        "Store deduplicates content",
    ],
    "NixOS_Configuration": [
        "Configuration defines system",
        "Configuration.nix is entry",
        "Configuration enables modules",
        "System configuration declarative",
    ],
    "Nix_Profile": [
        "Profile links user packages",
        "Profile provides environment",
        "Profile manages generations",
        "User profile customizes system",
    ],
    "NixOS_Module": [
        "Module extends configuration",
        "Module defines options",
        "Module enables services",
        "Modular system composition",
    ],
    "Nix_Expression": [
        "Expression evaluates to value",
        "Expression language functional",
        "Expression defines derivation",
        "Nix expression syntax",
    ],
    "Nix_Channel": [
        "Channel provides packages",
        "Channel updates fetch",
        "Channel stable unstable",
        "Package channel repository",
    ],
    "Nixpkgs": [
        "Nixpkgs collection packages",
        "Nixpkgs largest repository",
        "Nixpkgs community maintained",
        "Package set comprehensive",
    ],
    "Home_Manager": [
        "Home manager user config",
        "Home manages dotfiles",
        "Home declarative setup",
        "User environment management",
    ],
    # Computer Science Domain (12)
    "Algorithm": [
        "Algorithm solves problem",
        "Algorithm defines steps",
        "Algorithm has complexity",
        "Efficient algorithm optimization",
    ],
    "Data_Structure": [
        "Data structure organizes",
        "Structure enables operations",
        "Structure affects performance",
        "Data organization pattern",
    ],
    "Neural_Network": [
        "Neural network learns patterns",
        "Network layers connected",
        "Deep learning architecture",
        "Artificial neural computation",
    ],
    "Machine_Learning": [
        "Machine learning from data",
        "Learning improves performance",
        "Statistical pattern recognition",
        "Model training optimization",
    ],
    "Compiler": [
        "Compiler translates code",
        "Compilation generates output",
        "Compiler optimization passes",
        "Source to binary transformation",
    ],
    "Operating_System": [
        "Operating system manages resources",
        "OS provides abstraction",
        "Kernel handles hardware",
        "System software platform",
    ],
    "Database": [
        "Database stores data",
        "Database enables queries",
        "Relational data model",
        "Data persistence storage",
    ],
    "Cryptography": [
        "Cryptography secures data",
        "Encryption protects information",
        "Cryptographic protocol secure",
        "Security through mathematics",
    ],
    "Network_Protocol": [
        "Protocol defines communication",
        "Network protocol standard",
        "Protocol enables interoperability",
        "Data transmission rules",
    ],
    "Version_Control": [
        "Version control tracks changes",
        "Git manages history",
        "Repository stores versions",
        "Collaborative development tool",
    ],
    "API": [
        "API defines interface",
        "API enables integration",
        "Application programming interface",
        "Service communication contract",
    ],
    "Recursion": [
        "Recursion calls itself",
        "Recursive definition elegant",
        "Base case terminates",
        "Self-referential computation",
    ],
    # Scientific Domain (10)
    "Evolution": [
        "Evolution drives change",
        "Natural selection adapts",
        "Species evolve over time",
        "Biological evolution process",
    ],
    "Photosynthesis": [
        "Photosynthesis converts light",
        "Plants produce oxygen",
        "Energy from sunlight",
        "Carbon fixation process",
    ],
    "Mitochondria": [
        "Mitochondria power cells",
        "Cellular energy production",
        "Mitochondria ATP generation",
        "Powerhouse of cell",
    ],
    "DNA": [
        "DNA encodes genetics",
        "Genetic information stored",
        "DNA double helix structure",
        "Hereditary molecule blueprint",
    ],
    "Gravity": [
        "Gravity attracts mass",
        "Gravitational force universal",
        "Gravity curves spacetime",
        "Mass attracts mass",
    ],
    "Entropy": [
        "Entropy increases disorder",
        "Thermodynamic arrow time",
        "Entropy measures chaos",
        "System tends disorder",
    ],
    "Quantum_Superposition": [
        "Quantum superposition states",
        "Particle wave duality",
        "Quantum mechanics principle",
        "Multiple states simultaneously",
    ],
    "Electromagnetism": [
        "Electromagnetism unified force",
        "Light is electromagnetic",
        "Electric magnetic fields",
        "Electromagnetic radiation spectrum",
    ],
    "Chemical_Bond": [
        "Chemical bond connects atoms",
        "Covalent shares electrons",
        "Ionic transfers electrons",
        "Molecular bond formation",
    ],
    "Ecosystem": [
        "Ecosystem connects life",
        "Ecological relationships balance",
        "Ecosystem services vital",
        "Living systems interact",
    ],
    "Climate": [
        "Climate long term weather",
        "Climate change impacts",
        "Global climate patterns",
        "Climate system dynamics",
    ],
    # Philosophical Domain (10)
    "Consciousness": [
        "Consciousness is awareness",
        "Conscious experience subjective",
        "Consciousness emerges mind",
        "Phenomenal consciousness qualia",
    ],
    "Free_Will": [
        "Free will choice exists",
        "Determinism versus freedom",
        "Agency and autonomy",
        "Will determines action",
    ],
    "Ethics": [
        "Ethics guides behavior",
        "Moral philosophy values",
        "Ethical principles guide",
        "Right and wrong distinction",
    ],
    "Knowledge": [
        "Knowledge is justified belief",
        "Epistemology studies knowing",
        "Knowledge acquisition learning",
        "True belief understanding",
    ],
    "Reality": [
        "Reality exists objectively",
        "Perception shapes reality",
        "Ontology studies being",
        "What truly exists",
    ],
    "Mind": [
        "Mind thinks reasons feels",
        "Mental states cognitive",
        "Mind body problem",
        "Consciousness resides mind",
    ],
    "Truth": [
        "Truth corresponds reality",
        "Truth verification matters",
        "Objective truth exists",
        "Truth seeking essential",
    ],
    "Justice": [
        "Justice fairness equality",
        "Social justice matters",
        "Justice system law",
        "Fair treatment deserved",
    ],
    "Meaning": [
        "Meaning purpose significance",
        "Life has meaning",
        "Semantic meaning language",
        "Existential meaning sought",
    ],
    "Self": [
        "Self identity person",
        "Self awareness reflects",
        "Personal identity continues",
        "Self concept develops",
    ],
    # Emotional Domain (10)
    "Joy": [
        "Joy happiness delight",
        "Joy positive emotion",
        "Feeling of joy",
        "Joyful experience happiness",
    ],
    "Sadness": [
        "Sadness sorrow grief",
        "Sad feeling down",
        "Sadness accompanies loss",
        "Feeling sad melancholy",
    ],
    "Fear": [
        "Fear danger threat",
        "Fear protective emotion",
        "Feeling afraid scared",
        "Fear response survival",
    ],
    "Anger": [
        "Anger frustration rage",
        "Angry feeling upset",
        "Anger at injustice",
        "Feeling angry frustrated",
    ],
    "Love": [
        "Love deep affection",
        "Love connects beings",
        "Feeling love warmth",
        "Love care compassion",
    ],
    "Anxiety": [
        "Anxiety worry concern",
        "Anxious feeling uncertain",
        "Anxiety about future",
        "Feeling anxious worried",
    ],
    "Curiosity": [
        "Curiosity desire know",
        "Curious exploring learning",
        "Curiosity drives discovery",
        "Feeling curious interested",
    ],
    "Empathy": [
        "Empathy understanding others",
        "Empathy shares feelings",
        "Empathic connection deep",
        "Feeling empathy compassion",
    ],
    "Hope": [
        "Hope future positive",
        "Hopeful expectation better",
        "Hope motivates action",
        "Feeling hopeful optimistic",
    ],
    "Gratitude": [
        "Gratitude appreciation thankful",
        "Grateful feeling blessed",
        "Gratitude improves wellbeing",
        "Feeling grateful appreciative",
    ],
    # Social Domain (8)
    "Democracy": [
        "Democracy people govern",
        "Democratic participation voting",
        "Democracy freedom equality",
        "Government by people",
    ],
    "Freedom": [
        "Freedom liberty autonomy",
        "Free from oppression",
        "Freedom of choice",
        "Individual freedom rights",
    ],
    "Equality": [
        "Equality fair treatment",
        "Equal rights all",
        "Social equality justice",
        "Equality of opportunity",
    ],
    "Community": [
        "Community people together",
        "Community shared bonds",
        "Community support network",
        "Belonging to community",
    ],
    "Culture": [
        "Culture shared practices",
        "Cultural values traditions",
        "Culture shapes behavior",
        "Cultural heritage preserved",
    ],
    "Education": [
        "Education learning growth",
        "Education empowers individuals",
        "Educational development",
        "Learning through education",
    ],
    "Economy": [
        "Economy resources exchange",
        "Economic system market",
        "Economy affects wellbeing",
        "Economic activity production",
    ],
    "Law": [
        "Law rules society",
        "Legal system justice",
        "Law enforces order",
        "Rule of law principle",
    ],
}


def get_model():
    """Load BGE-M3 model."""
    if HAS_FLAG:
        print("Using FlagEmbedding (BGEM3FlagModel)...")
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        return model, "flag"
    else:
        print("Using sentence-transformers...")
        model = SentenceTransformer("BAAI/bge-m3")
        return model, "sbert"


def encode_concepts(model, model_type: str, concepts: Dict[str, List[str]]) -> tuple:
    """
    Encode all concepts using BGE-M3.

    Returns (embeddings, names) where embeddings is [n_concepts, 1024]
    """
    all_embeddings = []
    all_names = []

    print(f"\nEncoding {len(concepts)} concepts...")

    for concept_name, sentences in concepts.items():
        # Get embeddings for all sentences
        if model_type == "flag":
            result = model.encode(sentences, batch_size=4, max_length=512)
            # FlagEmbedding returns dict with 'dense_vecs'
            sentence_embeds = result["dense_vecs"]
        else:
            sentence_embeds = model.encode(sentences, convert_to_numpy=True)

        # Average the sentence embeddings for this concept
        mean_embedding = np.mean(sentence_embeds, axis=0).astype(np.float32)

        all_embeddings.append(mean_embedding)
        all_names.append(concept_name)

        print(f"  {concept_name}: {len(sentences)} sentences -> mean embedding")

    embeddings = np.stack(all_embeddings)
    print(f"\nFinal shape: {embeddings.shape}")

    return embeddings, all_names


def main():
    parser = argparse.ArgumentParser(
        description="Extract BGE-M3 embeddings for Neural Bridge v2 training"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/neural_bridge/bge_m3_embeddings.npz"),
        help="Output path for embeddings",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  BGE-M3 Embedding Extraction")
    print("=" * 60)
    print()

    # Load model
    model, model_type = get_model()

    # Encode concepts
    embeddings, names = encode_concepts(model, model_type, EXPANDED_CONCEPTS)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, embeddings=embeddings, names=np.array(names))

    print()
    print("=" * 60)
    print("  Extraction Complete!")
    print("=" * 60)
    print()
    print(f"Output: {args.output}")
    print(f"Concepts: {len(names)}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print()
    print("Sample embeddings (first 5):")
    for i in range(min(5, len(names))):
        norm = np.linalg.norm(embeddings[i])
        print(
            f"  {names[i]:20s}: norm={norm:.4f}, range=[{embeddings[i].min():.3f}, {embeddings[i].max():.3f}]"
        )
    print()
    print("Next step:")
    print(f"  python train_probe_bge_m3.py --embeddings {args.output}")


if __name__ == "__main__":
    main()
