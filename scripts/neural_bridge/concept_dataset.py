#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Concept Dataset for Neural Bridge Training

Contains concepts with multiple sentence descriptions,
aligned with Symthaea's SemanticPrime system.

Usage:
    from concept_dataset import get_all_concepts, get_concepts_by_domain
"""

from typing import Dict, List

# Core semantic concepts (aligned with Symthaea's SemanticPrime enum)
SEMANTIC_PRIME_CONCEPTS: List[Dict] = [
    {
        "name": "Action",
        "prime": "Action",
        "sentences": [
            "An action is something that is done or performed.",
            "Actions involve deliberate movement or activity.",
            "To act is to do something intentionally.",
            "Every action has a cause and an effect.",
        ],
    },
    {
        "name": "Agent",
        "prime": "Agent",
        "sentences": [
            "An agent is someone or something that acts.",
            "The agent of an action is the one who performs it.",
            "Agents have the capacity for intentional behavior.",
            "An agent makes choices and takes actions.",
        ],
    },
    {
        "name": "Patient",
        "prime": "Patient",
        "sentences": [
            "A patient is the entity affected by an action.",
            "The patient undergoes the action performed by the agent.",
            "In grammar, the patient receives the action of the verb.",
            "Patients are changed or affected by what happens to them.",
        ],
    },
    {
        "name": "Cause",
        "prime": "Cause",
        "sentences": [
            "A cause is what makes something happen.",
            "Causes precede their effects in time.",
            "To cause is to bring about a result.",
            "Every effect has at least one cause.",
        ],
    },
    {
        "name": "Time_Before",
        "prime": "Before",
        "sentences": [
            "Before means earlier in time than something else.",
            "What comes before precedes what comes after.",
            "The past is before the present.",
            "Causes come before their effects.",
        ],
    },
    {
        "name": "Time_After",
        "prime": "After",
        "sentences": [
            "After means later in time than something else.",
            "What comes after follows what came before.",
            "The future is after the present.",
            "Effects come after their causes.",
        ],
    },
    {
        "name": "Location",
        "prime": "Location",
        "sentences": [
            "A location is a place where something exists.",
            "Every physical object has a location in space.",
            "Location answers the question 'where?'",
            "Things can move from one location to another.",
        ],
    },
    {
        "name": "Truth",
        "prime": "Truth",
        "sentences": [
            "Truth is correspondence with reality.",
            "A true statement accurately describes the world.",
            "Truth is what is the case.",
            "To seek truth is to seek accurate knowledge.",
        ],
    },
    {
        "name": "Belief",
        "prime": "Belief",
        "sentences": [
            "A belief is something held to be true.",
            "Beliefs can be true or false.",
            "To believe is to accept something as true.",
            "Our beliefs shape how we see the world.",
        ],
    },
    {
        "name": "Desire",
        "prime": "Desire",
        "sentences": [
            "A desire is a wanting or wishing for something.",
            "Desires motivate action toward goals.",
            "To desire is to want something to be the case.",
            "Desires represent what we want.",
        ],
    },
    {
        "name": "Intention",
        "prime": "Intention",
        "sentences": [
            "An intention is a plan or purpose to do something.",
            "Intentions guide deliberate action.",
            "To intend is to plan to bring something about.",
            "Intentions connect desires to actions.",
        ],
    },
    {
        "name": "Valence_Positive",
        "prime": "Valence",
        "sentences": [
            "Positive valence indicates something good or desirable.",
            "Pleasant experiences have positive valence.",
            "Positive emotions feel good subjectively.",
            "We are drawn toward things with positive valence.",
        ],
    },
    {
        "name": "Valence_Negative",
        "prime": "Valence",
        "sentences": [
            "Negative valence indicates something bad or undesirable.",
            "Unpleasant experiences have negative valence.",
            "Negative emotions feel bad subjectively.",
            "We avoid things with negative valence.",
        ],
    },
    {
        "name": "Comparison_More",
        "prime": "More",
        "sentences": [
            "More means a greater amount or degree.",
            "Having more is having a larger quantity.",
            "More indicates increase or addition.",
            "Some things have more of a quality than others.",
        ],
    },
    {
        "name": "Comparison_Less",
        "prime": "Less",
        "sentences": [
            "Less means a smaller amount or degree.",
            "Having less is having a smaller quantity.",
            "Less indicates decrease or reduction.",
            "Some things have less of a quality than others.",
        ],
    },
]

# Domain-specific concepts (for NixOS domain)
NIXOS_CONCEPTS: List[Dict] = [
    {
        "name": "NixOS_Package",
        "domain": "nixos",
        "sentences": [
            "A NixOS package is a unit of software defined in Nix.",
            "Packages in NixOS are built reproducibly from derivations.",
            "Every package has a unique hash based on its inputs.",
            "Nix packages are stored in the Nix store.",
        ],
    },
    {
        "name": "NixOS_Service",
        "domain": "nixos",
        "sentences": [
            "A NixOS service is a background process managed by systemd.",
            "Services are enabled in configuration.nix.",
            "NixOS services are declaratively configured.",
            "System services start automatically at boot.",
        ],
    },
    {
        "name": "Nix_Derivation",
        "domain": "nixos",
        "sentences": [
            "A derivation is a build recipe in Nix.",
            "Derivations describe how to build software reproducibly.",
            "Each derivation has inputs, outputs, and a builder.",
            "Nix derivations are pure functions from inputs to outputs.",
        ],
    },
    {
        "name": "Nix_Flake",
        "domain": "nixos",
        "sentences": [
            "A Nix flake is a standardized way to package Nix projects.",
            "Flakes have inputs, outputs, and are locked for reproducibility.",
            "Flakes enable hermetic, reproducible builds.",
            "Every flake has a flake.nix file defining its structure.",
        ],
    },
    {
        "name": "Nix_Store",
        "domain": "nixos",
        "sentences": [
            "The Nix store is where all built packages are stored.",
            "Store paths contain a hash of all build inputs.",
            "The store is immutable - paths never change after creation.",
            "Multiple versions of packages can coexist in the store.",
        ],
    },
    {
        "name": "NixOS_Configuration",
        "domain": "nixos",
        "sentences": [
            "NixOS configuration is defined in configuration.nix.",
            "The entire system state is declared in one file.",
            "Configuration changes require a rebuild to take effect.",
            "NixOS configuration is declarative, not imperative.",
        ],
    },
    {
        "name": "Nix_Profile",
        "domain": "nixos",
        "sentences": [
            "A Nix profile is a user-specific collection of packages.",
            "Profiles allow per-user package installations.",
            "Profiles can be rolled back to previous generations.",
            "Each user can have their own independent profile.",
        ],
    },
    {
        "name": "NixOS_Module",
        "domain": "nixos",
        "sentences": [
            "A NixOS module is a reusable configuration fragment.",
            "Modules define options that can be configured.",
            "Modules are composed together to form the full config.",
            "Each module handles one aspect of system configuration.",
        ],
    },
]

# General knowledge concepts
GENERAL_CONCEPTS: List[Dict] = [
    {
        "name": "Mitochondria",
        "domain": "biology",
        "sentences": [
            "Mitochondria are the powerhouses of the cell.",
            "Mitochondria produce ATP through cellular respiration.",
            "These organelles have their own DNA.",
            "Mitochondria are found in nearly all eukaryotic cells.",
        ],
    },
    {
        "name": "Democracy",
        "domain": "politics",
        "sentences": [
            "Democracy is government by the people.",
            "In a democracy, citizens vote for their leaders.",
            "Democratic systems protect individual rights.",
            "Democracy means rule by the majority with minority protections.",
        ],
    },
    {
        "name": "Photosynthesis",
        "domain": "biology",
        "sentences": [
            "Photosynthesis converts sunlight into chemical energy.",
            "Plants use photosynthesis to produce glucose.",
            "Photosynthesis occurs in chloroplasts.",
            "This process releases oxygen as a byproduct.",
        ],
    },
    {
        "name": "Consciousness",
        "domain": "philosophy",
        "sentences": [
            "Consciousness is subjective experience and awareness.",
            "Being conscious means there is something it is like to be.",
            "Consciousness involves the integration of information.",
            "The hard problem asks why there is experience at all.",
        ],
    },
    {
        "name": "Gravity",
        "domain": "physics",
        "sentences": [
            "Gravity is the force that attracts objects with mass.",
            "Einstein described gravity as the curvature of spacetime.",
            "Gravity keeps planets in orbit around the sun.",
            "The strength of gravity depends on mass and distance.",
        ],
    },
    {
        "name": "Evolution",
        "domain": "biology",
        "sentences": [
            "Evolution is the change in species over generations.",
            "Natural selection drives evolutionary adaptation.",
            "Species evolve through random mutation and selection.",
            "All life on Earth shares common ancestors through evolution.",
        ],
    },
    {
        "name": "Algorithm",
        "domain": "computer_science",
        "sentences": [
            "An algorithm is a step-by-step procedure for solving a problem.",
            "Algorithms can be expressed in code or pseudocode.",
            "The efficiency of algorithms is measured by time and space complexity.",
            "Many algorithms are recursive, calling themselves to solve subproblems.",
        ],
    },
    {
        "name": "Neural_Network",
        "domain": "machine_learning",
        "sentences": [
            "A neural network is a computing system inspired by biological brains.",
            "Neural networks learn from data through training.",
            "Deep neural networks have many hidden layers.",
            "Neural networks are the foundation of modern AI.",
        ],
    },
    {
        "name": "Entropy",
        "domain": "physics",
        "sentences": [
            "Entropy is a measure of disorder in a system.",
            "The second law of thermodynamics states entropy tends to increase.",
            "Entropy determines the direction of time.",
            "Information entropy measures uncertainty in a message.",
        ],
    },
    {
        "name": "Justice",
        "domain": "philosophy",
        "sentences": [
            "Justice is giving each person what they are due.",
            "Justice involves fairness in the treatment of people.",
            "Legal justice is administered by courts and laws.",
            "Social justice concerns the fair distribution of resources.",
        ],
    },
]


def get_all_concepts() -> List[Dict]:
    """Return all concepts for training."""
    return SEMANTIC_PRIME_CONCEPTS + NIXOS_CONCEPTS + GENERAL_CONCEPTS


def get_concepts_by_domain(domain: str) -> List[Dict]:
    """Return concepts for a specific domain."""
    all_concepts = get_all_concepts()
    return [c for c in all_concepts if c.get("domain") == domain]


def get_semantic_primes() -> List[Dict]:
    """Return only the semantic prime concepts."""
    return SEMANTIC_PRIME_CONCEPTS


def get_nixos_concepts() -> List[Dict]:
    """Return NixOS-specific concepts."""
    return NIXOS_CONCEPTS


if __name__ == "__main__":
    concepts = get_all_concepts()
    print(f"Total concepts: {len(concepts)}")
    print()

    # Count by domain
    domains = {}
    for c in concepts:
        domain = c.get("domain", c.get("prime", "unknown"))
        domains[domain] = domains.get(domain, 0) + 1

    print("By domain/type:")
    for domain, count in sorted(domains.items()):
        print(f"  - {domain}: {count}")
    print()

    print("All concepts:")
    for c in concepts:
        print(f"  - {c['name']}: {len(c['sentences'])} sentences")
