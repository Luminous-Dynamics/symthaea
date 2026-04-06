// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hand-curated high-quality claims covering key knowledge domains.

use crate::RawClaim;
use plexus_common::EmpiricalLevel::*;

pub fn curated_claims() -> Vec<RawClaim> {
    RAW.iter().map(|(content, e, sources, tags)| RawClaim {
        content: content.to_string(),
        empirical_level: *e,
        sources: sources.iter().map(|s| s.to_string()).collect(),
        tags: tags.iter().map(|s| s.to_string()).collect(),
    }).collect()
}

const RAW: &[(&str, plexus_common::EmpiricalLevel, &[&str], &[&str])] = &[
    // ── Consciousness ──
    ("Integrated Information Theory proposes that consciousness corresponds to integrated information called Phi", E3, &["Tononi 2004"], &["consciousness", "iit"]),
    ("The hard problem of consciousness refers to explaining why subjective experience exists", E4, &["Chalmers 1995"], &["consciousness", "philosophy"]),
    ("Global Workspace Theory proposes consciousness arises from information broadcast across brain regions", E3, &["Baars 1988"], &["consciousness", "gwt"]),
    ("Neural correlates of consciousness are the minimal neuronal mechanisms sufficient for any specific conscious experience", E3, &["Koch 2004"], &["consciousness", "neuroscience"]),

    // ── Programming ──
    ("Rust is a memory-safe systems programming language that prevents data races at compile time", E4, &["rust-lang.org"], &["programming", "rust"]),
    ("Rust uses an ownership system with borrowing rules instead of garbage collection for memory management", E4, &["rust-lang.org"], &["programming", "rust"]),
    ("WebAssembly is a binary instruction format designed as a portable compilation target for high-level languages", E4, &["webassembly.org"], &["programming", "wasm"]),
    ("Git was created by Linus Torvalds in 2005 for Linux kernel development", E4, &["git-scm.com"], &["programming", "git"]),
    ("JavaScript was created in 10 days by Brendan Eich at Netscape in 1995", E4, &["MDN"], &["programming", "javascript"]),
    ("The first web browser was called WorldWideWeb later renamed Nexus created by Tim Berners-Lee", E4, &["CERN"], &["web", "browser"]),
    ("SQLite is the most widely deployed database engine in the world", E4, &["sqlite.org"], &["database", "sqlite"]),
    ("TCP/IP is the foundational protocol suite of the internet standardized in 1983", E4, &["IETF"], &["networking", "internet"]),

    // ── Physics ──
    ("Quantum entanglement allows particles to share states instantaneously regardless of distance", E4, &["Physics Review"], &["physics", "quantum"]),
    ("The speed of light in vacuum is approximately 299792458 meters per second", E4, &["NIST"], &["physics", "light"]),
    ("The Heisenberg uncertainty principle states position and momentum cannot both be precisely measured simultaneously", E4, &["Heisenberg 1927"], &["physics", "quantum"]),
    ("Dark matter constitutes approximately 27 percent of the total mass-energy of the universe", E3, &["Planck"], &["physics", "cosmology"]),
    ("The observable universe has a diameter of approximately 93 billion light-years", E3, &["NASA"], &["physics", "cosmology"]),
    ("General relativity predicts that gravity bends the path of light", E4, &["Einstein 1915"], &["physics", "relativity"]),

    // ── Biology ──
    ("DNA consists of two complementary strands forming a double helix structure", E4, &["Watson Crick 1953"], &["biology", "genetics"]),
    ("The human brain contains approximately 86 billion neurons", E4, &["Azevedo 2009"], &["biology", "neuroscience"]),
    ("CRISPR-Cas9 enables precise editing of DNA sequences in living organisms", E4, &["Doudna 2012"], &["biology", "genetics"]),
    ("The human genome contains approximately 3 billion base pairs and 20000 to 25000 protein-coding genes", E4, &["Human Genome Project"], &["biology", "genetics"]),
    ("Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight", E4, &["Biology"], &["biology", "photosynthesis"]),
    ("Neurons communicate through electrochemical signals called action potentials", E4, &["Neuroscience"], &["biology", "neuroscience"]),

    // ── Climate ──
    ("Ocean acidification is caused by absorption of atmospheric carbon dioxide by seawater lowering pH levels", E4, &["NOAA"], &["climate", "ocean"]),
    ("Global average sea level has risen approximately 8 to 9 inches since 1880", E4, &["NASA"], &["climate", "sea-level"]),
    ("The Arctic sea ice extent has declined approximately 13 percent per decade since 1979", E4, &["NASA"], &["climate", "arctic"]),
    ("Methane is approximately 80 times more potent than CO2 as a greenhouse gas over a 20 year period", E4, &["IPCC"], &["climate", "greenhouse"]),
    ("The ocean absorbs approximately 30 percent of anthropogenic CO2 emissions", E4, &["NOAA"], &["climate", "ocean"]),

    // ── Geography ──
    ("The Mariana Trench is the deepest known point in the ocean at approximately 36000 feet", E4, &["NOAA"], &["geography", "ocean"]),
    ("Mount Everest is the highest point on Earth at 8849 meters above sea level", E4, &["Nepal Survey"], &["geography", "mountains"]),
    ("Antarctica contains approximately 70 percent of the worlds fresh water frozen in its ice sheet", E4, &["NASA"], &["geography", "antarctica"]),
    ("The Amazon River carries more water than any other river system on Earth", E4, &["USGS"], &["geography", "rivers"]),

    // ── Energy ──
    ("Solar photovoltaic costs have declined approximately 90 percent since 2010", E4, &["IRENA"], &["energy", "solar"]),
    ("Nuclear fission produces energy by splitting heavy atomic nuclei like uranium-235", E4, &["Physics"], &["energy", "nuclear"]),

    // ── Decentralization ──
    ("Holochain is a framework for building peer-to-peer applications using agent-centric distributed hash tables", E3, &["holochain.org"], &["decentralization", "holochain"]),
    ("Distributed hash tables enable decentralized key-value storage across a peer-to-peer network", E4, &["Computer Science"], &["decentralization", "dht"]),
    ("Zero-knowledge proofs allow proving knowledge of information without revealing the information itself", E4, &["Goldwasser 1985"], &["cryptography", "zkp"]),
    ("Post-quantum cryptography develops algorithms resistant to quantum computer attacks", E3, &["NIST"], &["cryptography", "quantum"]),
    ("Public key cryptography uses mathematically related key pairs for encryption and digital signatures", E4, &["Diffie-Hellman 1976"], &["cryptography", "pki"]),

    // ── Mathematics ──
    ("The Pythagorean theorem states in a right triangle a squared plus b squared equals c squared", E4, &["Euclid"], &["mathematics", "geometry"]),
    ("Euler identity links five fundamental constants e to the power of i times pi plus 1 equals 0", E4, &["Euler 1748"], &["mathematics", "constants"]),
    ("Pi is an irrational number approximately equal to 3.14159265", E4, &["Mathematics"], &["mathematics", "constants"]),

    // ── Health ──
    ("Vaccines work by training the immune system to recognize specific pathogens", E4, &["WHO"], &["health", "vaccines"]),
    ("Antibiotics do not work against viral infections", E4, &["WHO"], &["health", "antibiotics"]),
    ("The WHO recommends adults engage in at least 150 minutes of moderate physical activity per week", E4, &["WHO"], &["health", "exercise"]),

    // ── Ecology ──
    ("Biodiversity loss is occurring at 100 to 1000 times the natural background extinction rate", E3, &["IPBES"], &["ecology", "biodiversity"]),
    ("Pollinators are responsible for approximately 75 percent of food crop production worldwide", E3, &["IPBES"], &["ecology", "pollinators"]),
    ("Forests cover approximately 31 percent of the global land area", E4, &["FAO"], &["ecology", "forests"]),

    // ── History ──
    ("The printing press was invented by Johannes Gutenberg around 1440", E4, &["Historical records"], &["history", "technology"]),
    ("The internet originated from ARPANET first connected in 1969", E4, &["DARPA"], &["history", "internet"]),
    ("The Universal Declaration of Human Rights was adopted by the United Nations in 1948", E4, &["UN"], &["history", "rights"]),
];
