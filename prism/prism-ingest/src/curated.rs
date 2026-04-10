// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hand-curated high-quality claims covering key knowledge domains.

use crate::RawClaim;
use prism_common::EmpiricalLevel::*;
use prism_common::{EmpiricalLevel, NormativeLevel, MaterialityLevel};

pub fn curated_claims() -> Vec<RawClaim> {
    RAW.iter().map(|(content, e, sources, tags)| {
        let (n, m) = infer_nm(*e, tags);
        RawClaim {
            content: content.to_string(),
            empirical_level: *e,
            normative_level: n,
            materiality_level: m,
            sources: sources.iter().map(|s| s.to_string()).collect(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
        }
    }).collect()
}

/// Infer N/M from E-level and topic tags (same heuristic as Knowledge mock data).
fn infer_nm(e: EmpiricalLevel, tags: &[&str]) -> (NormativeLevel, MaterialityLevel) {
    let n = match e {
        E4 => NormativeLevel::N3,
        E3 => NormativeLevel::N2,
        _ => NormativeLevel::N1,
    };
    let has_applied = tags.iter().any(|t|
        *t == "climate" || *t == "energy" || *t == "ecology" || *t == "health"
    );
    let has_abstract = tags.iter().any(|t|
        *t == "mathematics" || *t == "physics" || *t == "quantum"
    );
    let m = if has_applied { MaterialityLevel::M3 }
        else if has_abstract { MaterialityLevel::M1 }
        else { MaterialityLevel::M2 };
    (n, m)
}

const RAW: &[(&str, prism_common::EmpiricalLevel, &[&str], &[&str])] = &[
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

    // ── Medicine ──
    ("Penicillin was discovered by Alexander Fleming in 1928 from the mold Penicillium notatum", E4, &["Fleming 1929"], &["medicine", "history"]),
    ("The human heart beats approximately 100000 times per day pumping about 2000 gallons of blood", E4, &["AHA"], &["medicine", "anatomy"]),
    ("mRNA vaccines work by instructing cells to produce a protein that triggers an immune response", E4, &["NIH"], &["medicine", "vaccines"]),
    ("Antibiotic resistance is accelerated by overuse and misuse of antibiotics in humans and agriculture", E4, &["WHO"], &["medicine", "antimicrobial"]),
    ("The placebo effect can produce measurable physiological changes through expectation alone", E3, &["Benedetti 2005"], &["medicine", "neuroscience"]),

    // ── Computer Science ──
    ("P versus NP asks whether every problem whose solution can be verified quickly can also be solved quickly", E4, &["Cook 1971"], &["computer-science", "complexity"]),
    ("Machine learning models learn patterns from data rather than being explicitly programmed with rules", E4, &["Computer Science"], &["computer-science", "machine-learning"]),
    ("The halting problem proves that no general algorithm can determine if an arbitrary program will terminate", E4, &["Turing 1936"], &["computer-science", "computability"]),
    ("SHA-256 is a cryptographic hash function that produces a fixed 256-bit output from arbitrary input", E4, &["NIST"], &["computer-science", "cryptography"]),
    ("Distributed consensus algorithms like Raft and Paxos enable agreement among networked nodes despite failures", E4, &["Lamport 1998"], &["computer-science", "distributed"]),

    // ── Astronomy ──
    ("The Sun is approximately 4.6 billion years old and is classified as a G-type main-sequence star", E4, &["NASA"], &["astronomy", "sun"]),
    ("A black hole is a region of spacetime where gravity is so strong that nothing can escape from it", E4, &["Penrose 1965"], &["astronomy", "black-holes"]),
    ("The cosmic microwave background radiation is the oldest light in the universe originating 380000 years after the Big Bang", E4, &["Penzias Wilson 1965"], &["astronomy", "cosmology"]),
    ("Gravitational waves were first directly detected by LIGO on September 14 2015", E4, &["LIGO 2016"], &["astronomy", "gravitational-waves"]),
    ("Mars has the largest volcano in the solar system Olympus Mons standing about 72000 feet tall", E4, &["NASA"], &["astronomy", "mars"]),

    // ── Psychology ──
    ("Working memory has a limited capacity of approximately 4 chunks of information", E3, &["Cowan 2010"], &["psychology", "cognition"]),
    ("The Dunning-Kruger effect describes how people with limited competence tend to overestimate their ability", E3, &["Dunning Kruger 1999"], &["psychology", "cognition"]),
    ("Cognitive behavioral therapy is one of the most empirically supported treatments for depression and anxiety", E4, &["APA"], &["psychology", "therapy"]),
    ("Sleep deprivation impairs attention memory and decision-making within 24 hours", E4, &["Walker 2017"], &["psychology", "sleep"]),

    // ── Economics ──
    ("Gross Domestic Product measures the total value of goods and services produced within a country in a given period", E4, &["Economics"], &["economics", "gdp"]),
    ("The tragedy of the commons describes how individual self-interest can deplete shared resources", E4, &["Hardin 1968"], &["economics", "commons"]),
    ("Inflation is the rate at which the general level of prices for goods and services rises over time", E4, &["Economics"], &["economics", "inflation"]),

    // ── Materials Science ──
    ("Graphene is a single layer of carbon atoms arranged in a hexagonal lattice with extraordinary strength and conductivity", E4, &["Novoselov Geim 2004"], &["materials", "graphene"]),
    ("Superconductors carry electrical current with zero resistance below a critical temperature", E4, &["Physics"], &["materials", "superconductivity"]),

    // ── Agriculture ──
    ("The Green Revolution increased global food production through high-yield crop varieties and modern agriculture", E4, &["FAO"], &["agriculture", "food"]),
    ("Approximately one third of all food produced globally is lost or wasted each year", E4, &["FAO"], &["agriculture", "food-waste"]),
    ("Soil degradation affects approximately 33 percent of global soils threatening food security", E3, &["FAO"], &["agriculture", "soil"]),

    // ── Philosophy ──
    ("The Trolley Problem illustrates the tension between utilitarian and deontological ethical reasoning", E4, &["Foot 1967"], &["philosophy", "ethics"]),
    ("Epistemology is the branch of philosophy concerned with the nature scope and limits of knowledge", E4, &["Philosophy"], &["philosophy", "epistemology"]),
    ("The ship of Theseus paradox questions whether an object remains the same after all its parts are replaced", E4, &["Plutarch"], &["philosophy", "identity"]),

    // ── Oceanography ──
    ("The thermohaline circulation is a global ocean current system driven by differences in water density temperature and salinity", E4, &["Physical Oceanography"], &["oceanography", "currents"]),
    ("Coral reefs support approximately 25 percent of all marine species despite covering less than 1 percent of the ocean floor", E4, &["NOAA"], &["oceanography", "ecology"]),
];
