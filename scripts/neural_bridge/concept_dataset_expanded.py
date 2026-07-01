#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Expanded Concept Dataset for Neural Bridge Training

Contains 120+ concepts organized into categories:
- Semantic Primes (Complete NSM set)
- NixOS/Nix Ecosystem
- Computer Science
- Natural Sciences
- Philosophy & Consciousness
- Emotions & Mental States
- Actions & Processes
- Social & Political Concepts

Each concept has 4-6 sentences for robust activation extraction.
"""

from typing import Dict, List

# =============================================================================
# SEMANTIC PRIMES - Complete Natural Semantic Metalanguage set
# =============================================================================

SEMANTIC_PRIMES = [
    # Substantives
    {
        "name": "I",
        "category": "substantive",
        "sentences": [
            "I am aware of my own existence and thoughts.",
            "I experience the world through my senses.",
            "I make decisions based on my values.",
            "I reflect on my past experiences.",
        ],
    },
    {
        "name": "You",
        "category": "substantive",
        "sentences": [
            "You are the person I am speaking to directly.",
            "You have your own unique perspective on things.",
            "You can understand what I am saying.",
            "You exist as a separate individual from me.",
        ],
    },
    {
        "name": "Someone",
        "category": "substantive",
        "sentences": [
            "Someone left this package at the door.",
            "Someone needs to take responsibility for this.",
            "Someone with expertise should examine this.",
            "Someone is calling your name.",
        ],
    },
    {
        "name": "Something",
        "category": "substantive",
        "sentences": [
            "Something strange happened yesterday.",
            "Something is blocking the entrance.",
            "Something must be done about this problem.",
            "There is something I need to tell you.",
        ],
    },
    {
        "name": "People",
        "category": "substantive",
        "sentences": [
            "People gather in communities for mutual support.",
            "People have different beliefs and customs.",
            "People communicate through language and gestures.",
            "People work together to achieve common goals.",
        ],
    },
    {
        "name": "Body",
        "category": "substantive",
        "sentences": [
            "The body requires nutrients and rest to function.",
            "Your body moves through physical space.",
            "The body has many interconnected systems.",
            "A healthy body supports a healthy mind.",
        ],
    },
    # Determiners
    {
        "name": "This",
        "category": "determiner",
        "sentences": [
            "This particular object is what I'm referring to.",
            "This moment right now is important.",
            "This idea deserves careful consideration.",
            "This is the answer we've been looking for.",
        ],
    },
    {
        "name": "Same",
        "category": "determiner",
        "sentences": [
            "These two things are the same in every way.",
            "We arrived at the same conclusion independently.",
            "The same rules apply to everyone equally.",
            "It happened in the same place as before.",
        ],
    },
    {
        "name": "Other",
        "category": "determiner",
        "sentences": [
            "The other option might work better.",
            "Other people have different opinions.",
            "We need to consider the other side.",
            "There must be some other explanation.",
        ],
    },
    # Quantifiers
    {
        "name": "One",
        "category": "quantifier",
        "sentences": [
            "There is only one correct answer.",
            "One person can make a difference.",
            "One at a time, please.",
            "This is one of many possibilities.",
        ],
    },
    {
        "name": "Two",
        "category": "quantifier",
        "sentences": [
            "Two objects sit side by side.",
            "It takes two people to have a conversation.",
            "Two plus two equals four.",
            "There are two sides to every story.",
        ],
    },
    {
        "name": "Some",
        "category": "quantifier",
        "sentences": [
            "Some things are better left unsaid.",
            "Some people prefer quiet environments.",
            "I need some time to think about this.",
            "Some progress is better than none.",
        ],
    },
    {
        "name": "All",
        "category": "quantifier",
        "sentences": [
            "All living things need water to survive.",
            "All the pieces fit together perfectly.",
            "We should treat all people with respect.",
            "All evidence points to this conclusion.",
        ],
    },
    {
        "name": "Many",
        "category": "quantifier",
        "sentences": [
            "Many different species live in this forest.",
            "Many attempts were made before success.",
            "There are many ways to solve this problem.",
            "Many people share this belief.",
        ],
    },
    {
        "name": "Much",
        "category": "quantifier",
        "sentences": [
            "There is much work to be done.",
            "I don't have much time left.",
            "Much has changed since then.",
            "How much does this cost?",
        ],
    },
    # Attributes
    {
        "name": "Good",
        "category": "attribute",
        "sentences": [
            "A good solution benefits everyone involved.",
            "Good actions lead to positive outcomes.",
            "This is good for your health.",
            "A good person acts with integrity.",
        ],
    },
    {
        "name": "Bad",
        "category": "attribute",
        "sentences": [
            "A bad decision can have lasting consequences.",
            "Bad weather forced us to cancel.",
            "This is bad for the environment.",
            "A bad habit is hard to break.",
        ],
    },
    {
        "name": "Big",
        "category": "attribute",
        "sentences": [
            "A big elephant walked across the plain.",
            "This is a big opportunity for growth.",
            "The big picture is what matters most.",
            "A big change requires careful planning.",
        ],
    },
    {
        "name": "Small",
        "category": "attribute",
        "sentences": [
            "A small seed can grow into a mighty tree.",
            "Small changes add up over time.",
            "The small details matter.",
            "Even a small contribution helps.",
        ],
    },
    # Mental Predicates
    {
        "name": "Think",
        "category": "mental",
        "sentences": [
            "I think about complex problems deeply.",
            "We think through our decisions carefully.",
            "Think before you speak.",
            "I think this approach will work.",
        ],
    },
    {
        "name": "Know",
        "category": "mental",
        "sentences": [
            "I know the answer to this question.",
            "We know from experience what works.",
            "To know something is to understand it.",
            "I know this to be true.",
        ],
    },
    {
        "name": "Want",
        "category": "mental",
        "sentences": [
            "I want to achieve my goals.",
            "People want different things in life.",
            "What do you want to accomplish?",
            "We want what is best for everyone.",
        ],
    },
    {
        "name": "Feel",
        "category": "mental",
        "sentences": [
            "I feel emotions deeply and authentically.",
            "We feel the texture with our hands.",
            "How do you feel about this?",
            "I feel that this is the right choice.",
        ],
    },
    {
        "name": "See",
        "category": "mental",
        "sentences": [
            "I see the world through my eyes.",
            "We see the evidence clearly.",
            "I see what you mean now.",
            "Can you see the difference?",
        ],
    },
    {
        "name": "Hear",
        "category": "mental",
        "sentences": [
            "I hear sounds with my ears.",
            "We hear the music playing softly.",
            "I hear what you're saying.",
            "Did you hear that noise?",
        ],
    },
    # Speech Predicates
    {
        "name": "Say",
        "category": "speech",
        "sentences": [
            "I say what I mean honestly.",
            "What did they say about it?",
            "Say the words clearly.",
            "I must say, this is impressive.",
        ],
    },
    {
        "name": "Words",
        "category": "speech",
        "sentences": [
            "Words carry meaning and power.",
            "Choose your words carefully.",
            "These words express my thoughts.",
            "Words alone cannot convey everything.",
        ],
    },
    {
        "name": "True",
        "category": "speech",
        "sentences": [
            "This statement is true and verifiable.",
            "A true friend supports you always.",
            "Is this claim true or false?",
            "The true meaning lies beneath the surface.",
        ],
    },
    # Actions & Events
    {
        "name": "Do",
        "category": "action",
        "sentences": [
            "I do my work with dedication.",
            "What should we do about this?",
            "Do the right thing.",
            "We do what we can.",
        ],
    },
    {
        "name": "Happen",
        "category": "action",
        "sentences": [
            "Events happen in sequence over time.",
            "What happened here yesterday?",
            "Things happen for a reason.",
            "This will happen eventually.",
        ],
    },
    {
        "name": "Move",
        "category": "action",
        "sentences": [
            "Objects move through space over time.",
            "We move forward with determination.",
            "Move carefully across the bridge.",
            "The project must move ahead.",
        ],
    },
    {
        "name": "Touch",
        "category": "action",
        "sentences": [
            "I touch objects to feel their texture.",
            "Don't touch the hot surface.",
            "Our hands touch gently.",
            "This story will touch your heart.",
        ],
    },
    # Location & Movement
    {
        "name": "Where",
        "category": "location",
        "sentences": [
            "Where is the meeting taking place?",
            "I know where this is heading.",
            "Where do we go from here?",
            "This is where I belong.",
        ],
    },
    {
        "name": "Here",
        "category": "location",
        "sentences": [
            "I am here right now.",
            "The answer is right here.",
            "Come here please.",
            "Here is where we start.",
        ],
    },
    {
        "name": "Above",
        "category": "location",
        "sentences": [
            "The sky is above us.",
            "Above all else, be honest.",
            "The temperature is above freezing.",
            "Rise above the challenges.",
        ],
    },
    {
        "name": "Below",
        "category": "location",
        "sentences": [
            "The roots grow below the surface.",
            "Below this line is restricted.",
            "The temperature dropped below zero.",
            "See the details below.",
        ],
    },
    {
        "name": "Far",
        "category": "location",
        "sentences": [
            "The destination is far from here.",
            "We have come far on this journey.",
            "Far away, I saw a light.",
            "This goes far beyond expectations.",
        ],
    },
    {
        "name": "Near",
        "category": "location",
        "sentences": [
            "The station is near the center.",
            "We are near the end of the project.",
            "Come near so I can see you.",
            "The answer is near at hand.",
        ],
    },
    {
        "name": "Side",
        "category": "location",
        "sentences": [
            "Stand on this side of the room.",
            "There are two sides to this argument.",
            "She sat by my side.",
            "We're on the same side.",
        ],
    },
    {
        "name": "Inside",
        "category": "location",
        "sentences": [
            "The contents are inside the box.",
            "Inside, everything was quiet.",
            "Look inside yourself for answers.",
            "The inside of the building is beautiful.",
        ],
    },
    # Time
    {
        "name": "When",
        "category": "time",
        "sentences": [
            "When did this happen exactly?",
            "I remember when we first met.",
            "When the time comes, we'll be ready.",
            "This is when we act.",
        ],
    },
    {
        "name": "Now",
        "category": "time",
        "sentences": [
            "The present moment is now.",
            "Now is the time to decide.",
            "I understand now what you meant.",
            "We must act now.",
        ],
    },
    {
        "name": "Before",
        "category": "time",
        "sentences": [
            "This happened before the other event.",
            "I have seen this before.",
            "Think before you act.",
            "Before long, it will be done.",
        ],
    },
    {
        "name": "After",
        "category": "time",
        "sentences": [
            "After the storm comes calm.",
            "What happens after this step?",
            "After consideration, I agree.",
            "We'll meet again after the event.",
        ],
    },
    {
        "name": "Long_Time",
        "category": "time",
        "sentences": [
            "This will take a long time to complete.",
            "A long time has passed since then.",
            "I waited a long time for this.",
            "It happened a long time ago.",
        ],
    },
    {
        "name": "Short_Time",
        "category": "time",
        "sentences": [
            "This will only take a short time.",
            "In a short time, we achieved much.",
            "Wait a short time before proceeding.",
            "A short time later, they arrived.",
        ],
    },
    {
        "name": "Moment",
        "category": "time",
        "sentences": [
            "Wait just a moment please.",
            "In that moment, everything changed.",
            "Seize the moment.",
            "A moment of clarity arrived.",
        ],
    },
    # Logical Concepts
    {
        "name": "Not",
        "category": "logical",
        "sentences": [
            "This is not what I expected.",
            "Not all statements are true.",
            "I am not certain about this.",
            "Not everyone agrees with this view.",
        ],
    },
    {
        "name": "Maybe",
        "category": "logical",
        "sentences": [
            "Maybe this approach will work.",
            "There is maybe a better solution.",
            "Maybe I was wrong about this.",
            "Maybe it will happen tomorrow.",
        ],
    },
    {
        "name": "Can",
        "category": "logical",
        "sentences": [
            "You can accomplish great things.",
            "Can this be done differently?",
            "We can improve with practice.",
            "I can see the solution now.",
        ],
    },
    {
        "name": "Because",
        "category": "logical",
        "sentences": [
            "This happened because of that cause.",
            "Because of you, we succeeded.",
            "I believe this because of the evidence.",
            "Because it matters, we must try.",
        ],
    },
    {
        "name": "If",
        "category": "logical",
        "sentences": [
            "If this is true, then that follows.",
            "What if we tried a different approach?",
            "If you work hard, you'll succeed.",
            "I wonder if this is correct.",
        ],
    },
    # Similarity & Difference
    {
        "name": "Like",
        "category": "similarity",
        "sentences": [
            "This is like that in many ways.",
            "I like this approach.",
            "Like attracts like.",
            "What is it like to experience this?",
        ],
    },
    {
        "name": "Kind",
        "category": "similarity",
        "sentences": [
            "What kind of thing is this?",
            "This is a different kind of problem.",
            "Be kind to others.",
            "All kinds of people came together.",
        ],
    },
    {
        "name": "Part",
        "category": "similarity",
        "sentences": [
            "This is part of a larger whole.",
            "Part of the solution is here.",
            "Every part plays a role.",
            "We are all part of nature.",
        ],
    },
    # Existence & Possession
    {
        "name": "There_Is",
        "category": "existence",
        "sentences": [
            "There is a solution to every problem.",
            "There is truth in what you say.",
            "There is no easy answer here.",
            "There is always hope.",
        ],
    },
    {
        "name": "Have",
        "category": "possession",
        "sentences": [
            "I have the resources needed.",
            "We have much to be grateful for.",
            "Do you have time to talk?",
            "I have an idea.",
        ],
    },
    # Life & Death
    {
        "name": "Live",
        "category": "life",
        "sentences": [
            "All living things live and grow.",
            "We live in an interconnected world.",
            "To live fully is to embrace life.",
            "Plants live through photosynthesis.",
        ],
    },
    {
        "name": "Die",
        "category": "life",
        "sentences": [
            "All living things eventually die.",
            "Ideas can live on or die out.",
            "The old ways must die for new ones to emerge.",
            "Stars are born and die over eons.",
        ],
    },
]

# =============================================================================
# NIXOS & NIX ECOSYSTEM
# =============================================================================

NIXOS_CONCEPTS = [
    {
        "name": "NixOS_Package",
        "category": "nix",
        "sentences": [
            "A NixOS package is a reproducible build of software.",
            "Packages in NixOS are defined declaratively in Nix expressions.",
            "Each package has a unique hash that ensures reproducibility.",
            "NixOS packages can be installed system-wide or per-user.",
        ],
    },
    {
        "name": "NixOS_Service",
        "category": "nix",
        "sentences": [
            "A NixOS service is configured declaratively in configuration.nix.",
            "Services in NixOS are managed by systemd.",
            "Enabling a service in NixOS is done through options.",
            "NixOS services can be customized with various settings.",
        ],
    },
    {
        "name": "Nix_Derivation",
        "category": "nix",
        "sentences": [
            "A derivation is a build action that produces an output.",
            "Derivations describe how to build software from source.",
            "Each derivation has inputs, outputs, and a builder.",
            "Nix derivations are content-addressed and immutable.",
        ],
    },
    {
        "name": "Nix_Flake",
        "category": "nix",
        "sentences": [
            "A flake is a self-contained Nix project with locked dependencies.",
            "Flakes provide reproducible, hermetic builds.",
            "Nix flakes have inputs and outputs defined in flake.nix.",
            "Flakes enable pure evaluation and better composability.",
        ],
    },
    {
        "name": "Nix_Store",
        "category": "nix",
        "sentences": [
            "The Nix store holds all packages and their dependencies.",
            "Store paths are content-addressed by their hash.",
            "The store at /nix/store contains immutable packages.",
            "Nix garbage collection removes unused store paths.",
        ],
    },
    {
        "name": "NixOS_Configuration",
        "category": "nix",
        "sentences": [
            "Configuration.nix defines the entire system state.",
            "NixOS configuration is declarative and reproducible.",
            "System configuration includes services, users, and packages.",
            "A single file can describe a complete operating system.",
        ],
    },
    {
        "name": "Nix_Profile",
        "category": "nix",
        "sentences": [
            "A profile is a user environment with installed packages.",
            "Profiles allow per-user package management.",
            "Switching profiles atomically changes the environment.",
            "Profile generations enable easy rollback.",
        ],
    },
    {
        "name": "NixOS_Module",
        "category": "nix",
        "sentences": [
            "A module is a reusable configuration component.",
            "Modules define options that can be set by users.",
            "NixOS modules compose to form the full configuration.",
            "Custom modules extend NixOS functionality.",
        ],
    },
    {
        "name": "Nix_Expression",
        "category": "nix",
        "sentences": [
            "Nix expressions are pure functional programs.",
            "Expressions evaluate to derivations or values.",
            "The Nix language is lazy and dynamically typed.",
            "Nix expressions describe software builds declaratively.",
        ],
    },
    {
        "name": "Nix_Channel",
        "category": "nix",
        "sentences": [
            "A channel is a named source of Nix packages.",
            "Channels point to specific versions of nixpkgs.",
            "Updating channels brings in new package versions.",
            "Channels are being replaced by flakes.",
        ],
    },
    {
        "name": "Nixpkgs",
        "category": "nix",
        "sentences": [
            "Nixpkgs is the largest repository of Nix packages.",
            "It contains over 80,000 packages and counting.",
            "Nixpkgs provides packages for multiple platforms.",
            "Contributing to nixpkgs follows standard PR workflow.",
        ],
    },
    {
        "name": "Home_Manager",
        "category": "nix",
        "sentences": [
            "Home Manager manages user dotfiles declaratively.",
            "It integrates with NixOS or standalone Nix.",
            "Home Manager configurations define user environments.",
            "Programs and services are configured per-user.",
        ],
    },
]

# =============================================================================
# COMPUTER SCIENCE
# =============================================================================

CS_CONCEPTS = [
    {
        "name": "Algorithm",
        "category": "cs",
        "sentences": [
            "An algorithm is a step-by-step procedure for computation.",
            "Algorithms have time and space complexity.",
            "Efficient algorithms solve problems with minimal resources.",
            "Algorithms can be expressed in code or pseudocode.",
        ],
    },
    {
        "name": "Data_Structure",
        "category": "cs",
        "sentences": [
            "A data structure organizes and stores data efficiently.",
            "Common data structures include arrays, trees, and graphs.",
            "Choosing the right structure affects performance.",
            "Data structures enable efficient data access patterns.",
        ],
    },
    {
        "name": "Neural_Network",
        "category": "cs",
        "sentences": [
            "A neural network learns patterns from training data.",
            "Networks have layers of interconnected neurons.",
            "Deep learning uses many hidden layers.",
            "Neural networks approximate complex functions.",
        ],
    },
    {
        "name": "Machine_Learning",
        "category": "cs",
        "sentences": [
            "Machine learning systems improve through experience.",
            "Models learn patterns from labeled training data.",
            "ML algorithms generalize from examples to new cases.",
            "Learning can be supervised, unsupervised, or reinforcement.",
        ],
    },
    {
        "name": "Compiler",
        "category": "cs",
        "sentences": [
            "A compiler translates source code to machine code.",
            "Compilers perform lexing, parsing, and code generation.",
            "Optimization passes improve generated code efficiency.",
            "Just-in-time compilers optimize at runtime.",
        ],
    },
    {
        "name": "Operating_System",
        "category": "cs",
        "sentences": [
            "An operating system manages hardware resources.",
            "It provides abstractions like processes and files.",
            "The kernel is the core of the operating system.",
            "OS scheduling determines which processes run.",
        ],
    },
    {
        "name": "Database",
        "category": "cs",
        "sentences": [
            "A database stores and organizes structured data.",
            "Queries retrieve data based on conditions.",
            "Transactions ensure data integrity.",
            "Databases support concurrent access.",
        ],
    },
    {
        "name": "Cryptography",
        "category": "cs",
        "sentences": [
            "Cryptography secures communication and data.",
            "Encryption transforms plaintext to ciphertext.",
            "Public key cryptography enables secure key exchange.",
            "Hash functions produce fixed-size digests.",
        ],
    },
    {
        "name": "Network_Protocol",
        "category": "cs",
        "sentences": [
            "Protocols define rules for communication.",
            "TCP provides reliable ordered delivery.",
            "HTTP transfers web resources.",
            "Protocol layers abstract complexity.",
        ],
    },
    {
        "name": "Version_Control",
        "category": "cs",
        "sentences": [
            "Version control tracks changes to files over time.",
            "Git is a distributed version control system.",
            "Branches enable parallel development.",
            "Commits record snapshots of project state.",
        ],
    },
    {
        "name": "API",
        "category": "cs",
        "sentences": [
            "An API defines how software components interact.",
            "APIs provide abstraction and encapsulation.",
            "REST APIs use HTTP for web services.",
            "Good APIs are well-documented and stable.",
        ],
    },
    {
        "name": "Recursion",
        "category": "cs",
        "sentences": [
            "Recursion is when a function calls itself.",
            "Base cases prevent infinite recursion.",
            "Recursive solutions break problems into subproblems.",
            "Many algorithms are naturally recursive.",
        ],
    },
]

# =============================================================================
# NATURAL SCIENCES
# =============================================================================

SCIENCE_CONCEPTS = [
    {
        "name": "Evolution",
        "category": "biology",
        "sentences": [
            "Evolution is descent with modification over generations.",
            "Natural selection favors adaptive traits.",
            "Species evolve through mutation and selection.",
            "Evolution explains the diversity of life.",
        ],
    },
    {
        "name": "Photosynthesis",
        "category": "biology",
        "sentences": [
            "Photosynthesis converts light energy to chemical energy.",
            "Plants use chlorophyll to capture sunlight.",
            "Carbon dioxide and water produce glucose and oxygen.",
            "Photosynthesis is the basis of most food chains.",
        ],
    },
    {
        "name": "Mitochondria",
        "category": "biology",
        "sentences": [
            "Mitochondria are the powerhouses of cells.",
            "They generate ATP through cellular respiration.",
            "Mitochondria have their own DNA.",
            "They evolved from ancient bacterial symbionts.",
        ],
    },
    {
        "name": "DNA",
        "category": "biology",
        "sentences": [
            "DNA stores genetic information in nucleotide sequences.",
            "The double helix structure encodes genes.",
            "DNA replication copies genetic material.",
            "Mutations in DNA drive evolution.",
        ],
    },
    {
        "name": "Gravity",
        "category": "physics",
        "sentences": [
            "Gravity attracts objects with mass toward each other.",
            "Einstein described gravity as spacetime curvature.",
            "Gravitational force depends on mass and distance.",
            "Gravity keeps planets in orbit around stars.",
        ],
    },
    {
        "name": "Entropy",
        "category": "physics",
        "sentences": [
            "Entropy measures disorder in a system.",
            "The second law says entropy tends to increase.",
            "Higher entropy means more possible microstates.",
            "Entropy limits the efficiency of heat engines.",
        ],
    },
    {
        "name": "Quantum_Superposition",
        "category": "physics",
        "sentences": [
            "Quantum systems exist in multiple states simultaneously.",
            "Measurement collapses the superposition to one state.",
            "Superposition enables quantum computing.",
            "Particles can be in many places at once.",
        ],
    },
    {
        "name": "Electromagnetism",
        "category": "physics",
        "sentences": [
            "Electromagnetism unifies electricity and magnetism.",
            "Light is an electromagnetic wave.",
            "Electric charges create electric fields.",
            "Moving charges produce magnetic fields.",
        ],
    },
    {
        "name": "Chemical_Bond",
        "category": "chemistry",
        "sentences": [
            "Chemical bonds hold atoms together in molecules.",
            "Covalent bonds share electrons between atoms.",
            "Ionic bonds transfer electrons between atoms.",
            "Bond strength determines molecular stability.",
        ],
    },
    {
        "name": "Ecosystem",
        "category": "ecology",
        "sentences": [
            "An ecosystem includes organisms and their environment.",
            "Energy flows through food webs.",
            "Ecosystems cycle nutrients through organisms.",
            "Biodiversity strengthens ecosystem resilience.",
        ],
    },
    {
        "name": "Climate",
        "category": "earth_science",
        "sentences": [
            "Climate is long-term weather patterns.",
            "Greenhouse gases trap heat in the atmosphere.",
            "Climate change affects global temperatures.",
            "Climate systems are complex and interconnected.",
        ],
    },
]

# =============================================================================
# PHILOSOPHY & CONSCIOUSNESS
# =============================================================================

PHILOSOPHY_CONCEPTS = [
    {
        "name": "Consciousness",
        "category": "philosophy",
        "sentences": [
            "Consciousness is subjective experience and awareness.",
            "The hard problem asks why experience exists.",
            "Consciousness emerges from complex neural activity.",
            "Qualia are the qualitative aspects of experience.",
        ],
    },
    {
        "name": "Free_Will",
        "category": "philosophy",
        "sentences": [
            "Free will is the ability to choose one's actions.",
            "Determinism questions whether choices are predetermined.",
            "Compatibilism reconciles free will with causation.",
            "Our sense of agency may be constructed.",
        ],
    },
    {
        "name": "Ethics",
        "category": "philosophy",
        "sentences": [
            "Ethics studies what is right and wrong.",
            "Moral philosophy examines how we should act.",
            "Ethical systems provide frameworks for decisions.",
            "Applied ethics addresses real-world dilemmas.",
        ],
    },
    {
        "name": "Knowledge",
        "category": "philosophy",
        "sentences": [
            "Knowledge is justified true belief.",
            "Epistemology studies how we know things.",
            "Evidence supports claims to knowledge.",
            "Skepticism questions our ability to know.",
        ],
    },
    {
        "name": "Reality",
        "category": "philosophy",
        "sentences": [
            "Reality is what exists independently of perception.",
            "Metaphysics examines the nature of reality.",
            "Physical reality may differ from appearance.",
            "Realism and idealism offer different views.",
        ],
    },
    {
        "name": "Mind",
        "category": "philosophy",
        "sentences": [
            "The mind encompasses thoughts, feelings, and consciousness.",
            "Mind-body dualism separates mental from physical.",
            "Physicalism says mind is entirely physical.",
            "Mental states have intentionality and meaning.",
        ],
    },
    {
        "name": "Truth",
        "category": "philosophy",
        "sentences": [
            "Truth is correspondence with reality.",
            "Coherence theories define truth by consistency.",
            "Pragmatism links truth to usefulness.",
            "Truth seeking is central to philosophy.",
        ],
    },
    {
        "name": "Justice",
        "category": "philosophy",
        "sentences": [
            "Justice is fairness in treatment and distribution.",
            "Distributive justice concerns fair allocation.",
            "Retributive justice concerns punishment.",
            "Justice requires impartial consideration.",
        ],
    },
    {
        "name": "Meaning",
        "category": "philosophy",
        "sentences": [
            "Meaning is what gives life purpose.",
            "Existentialism says we create our own meaning.",
            "Nihilism denies inherent meaning exists.",
            "Finding meaning is a fundamental human drive.",
        ],
    },
    {
        "name": "Self",
        "category": "philosophy",
        "sentences": [
            "The self is the subject of experience.",
            "Personal identity persists through time.",
            "The self may be a narrative construction.",
            "Buddhism questions the existence of a fixed self.",
        ],
    },
]

# =============================================================================
# EMOTIONS & MENTAL STATES
# =============================================================================

EMOTION_CONCEPTS = [
    {
        "name": "Joy",
        "category": "emotion",
        "sentences": [
            "Joy is a feeling of great happiness.",
            "Joy arises from positive experiences.",
            "Deep joy comes from meaningful connections.",
            "Joy can be shared and spread to others.",
        ],
    },
    {
        "name": "Sadness",
        "category": "emotion",
        "sentences": [
            "Sadness is feeling unhappy or sorrowful.",
            "Loss and disappointment cause sadness.",
            "Sadness helps us process difficult experiences.",
            "Expressing sadness can bring relief.",
        ],
    },
    {
        "name": "Fear",
        "category": "emotion",
        "sentences": [
            "Fear is a response to perceived danger.",
            "Fear triggers the fight-or-flight response.",
            "Healthy fear protects us from harm.",
            "Irrational fears can limit our lives.",
        ],
    },
    {
        "name": "Anger",
        "category": "emotion",
        "sentences": [
            "Anger is a strong feeling of displeasure.",
            "Anger arises when boundaries are crossed.",
            "Constructive anger can motivate change.",
            "Uncontrolled anger can be destructive.",
        ],
    },
    {
        "name": "Love",
        "category": "emotion",
        "sentences": [
            "Love is deep affection and care for others.",
            "Love creates bonds between people.",
            "Unconditional love accepts without conditions.",
            "Love is expressed through actions and words.",
        ],
    },
    {
        "name": "Anxiety",
        "category": "emotion",
        "sentences": [
            "Anxiety is worry about future events.",
            "Anxiety can motivate preparation.",
            "Chronic anxiety impairs daily functioning.",
            "Anxiety disorders require proper treatment.",
        ],
    },
    {
        "name": "Curiosity",
        "category": "emotion",
        "sentences": [
            "Curiosity is the desire to learn and explore.",
            "Curiosity drives discovery and innovation.",
            "Children are naturally curious about the world.",
            "Curiosity leads to deeper understanding.",
        ],
    },
    {
        "name": "Empathy",
        "category": "emotion",
        "sentences": [
            "Empathy is understanding others' feelings.",
            "Empathy helps us connect with others.",
            "Empathic listening validates experiences.",
            "Empathy is crucial for compassion.",
        ],
    },
    {
        "name": "Hope",
        "category": "emotion",
        "sentences": [
            "Hope is expectation that things will improve.",
            "Hope sustains us through difficult times.",
            "Hope motivates action toward goals.",
            "Without hope, despair can take hold.",
        ],
    },
    {
        "name": "Gratitude",
        "category": "emotion",
        "sentences": [
            "Gratitude is appreciation for what we have.",
            "Expressing gratitude improves well-being.",
            "Gratitude shifts focus to the positive.",
            "A grateful heart finds joy in small things.",
        ],
    },
]

# =============================================================================
# SOCIAL & POLITICAL CONCEPTS
# =============================================================================

SOCIAL_CONCEPTS = [
    {
        "name": "Democracy",
        "category": "political",
        "sentences": [
            "Democracy is government by the people.",
            "Citizens vote to elect representatives.",
            "Democratic systems protect individual rights.",
            "Democracy requires informed participation.",
        ],
    },
    {
        "name": "Freedom",
        "category": "political",
        "sentences": [
            "Freedom is the absence of coercion.",
            "Political freedom enables self-governance.",
            "Freedom of speech protects expression.",
            "Freedom must be balanced with responsibility.",
        ],
    },
    {
        "name": "Equality",
        "category": "political",
        "sentences": [
            "Equality means equal rights and opportunities.",
            "Legal equality treats everyone the same.",
            "Social equality reduces unfair hierarchies.",
            "Equality of outcome differs from opportunity.",
        ],
    },
    {
        "name": "Community",
        "category": "social",
        "sentences": [
            "A community is a group with shared interests.",
            "Communities provide support and belonging.",
            "Strong communities build social capital.",
            "Online communities connect distant people.",
        ],
    },
    {
        "name": "Culture",
        "category": "social",
        "sentences": [
            "Culture encompasses shared beliefs and practices.",
            "Cultural traditions are passed down generations.",
            "Different cultures have unique perspectives.",
            "Cultural exchange enriches societies.",
        ],
    },
    {
        "name": "Education",
        "category": "social",
        "sentences": [
            "Education transmits knowledge and skills.",
            "Education empowers individuals.",
            "Quality education reduces inequality.",
            "Lifelong learning adapts to change.",
        ],
    },
    {
        "name": "Economy",
        "category": "social",
        "sentences": [
            "The economy manages scarce resources.",
            "Economic systems allocate goods and services.",
            "Market economies use price signals.",
            "Economic growth can improve living standards.",
        ],
    },
    {
        "name": "Law",
        "category": "social",
        "sentences": [
            "Laws are rules enforced by authority.",
            "The rule of law ensures equal treatment.",
            "Laws evolve with society.",
            "Legal systems resolve disputes.",
        ],
    },
]


def get_all_concepts() -> List[Dict]:
    """Return all concepts combined into a single list."""
    all_concepts = []
    all_concepts.extend(SEMANTIC_PRIMES)
    all_concepts.extend(NIXOS_CONCEPTS)
    all_concepts.extend(CS_CONCEPTS)
    all_concepts.extend(SCIENCE_CONCEPTS)
    all_concepts.extend(PHILOSOPHY_CONCEPTS)
    all_concepts.extend(EMOTION_CONCEPTS)
    all_concepts.extend(SOCIAL_CONCEPTS)
    return all_concepts


def get_concepts_by_category() -> Dict[str, List[Dict]]:
    """Return concepts organized by category."""
    return {
        "semantic_primes": SEMANTIC_PRIMES,
        "nixos": NIXOS_CONCEPTS,
        "computer_science": CS_CONCEPTS,
        "natural_sciences": SCIENCE_CONCEPTS,
        "philosophy": PHILOSOPHY_CONCEPTS,
        "emotions": EMOTION_CONCEPTS,
        "social_political": SOCIAL_CONCEPTS,
    }


def print_summary():
    """Print dataset summary."""
    categories = get_concepts_by_category()
    total = 0

    print("=" * 60)
    print("  Expanded Concept Dataset Summary")
    print("=" * 60)
    print()

    for name, concepts in categories.items():
        print(f"  {name:25s}: {len(concepts):3d} concepts")
        total += len(concepts)

    print()
    print(f"  {'TOTAL':25s}: {total:3d} concepts")
    print()

    # Count total sentences
    all_concepts = get_all_concepts()
    total_sentences = sum(len(c["sentences"]) for c in all_concepts)
    print(f"  Total sentences: {total_sentences}")
    print("=" * 60)


if __name__ == "__main__":
    print_summary()
