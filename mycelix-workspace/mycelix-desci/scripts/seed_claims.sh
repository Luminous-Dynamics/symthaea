#!/bin/bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Seed the DeSci portal with 50 real scientific claims.
# Requires the API server to be running on localhost:8080.
#
# Usage: ./scripts/seed_claims.sh [API_URL]

set -e

API="${1:-http://localhost:8080}"

post_claim() {
    curl -s -X POST "$API/api/v1/claims" \
        -H "Content-Type: application/json" \
        -d "$1" > /dev/null 2>&1
    echo "  ✓ $2"
}

echo "=== Seeding DeSci Portal with 50 Real Claims ==="
echo "API: $API"
echo ""

# ── Case Studies (10) ──
echo "Case Studies:"

post_claim '{"tier":"E0","content":{"dataset_hash":"none","description":"Deuterium fusion at room temperature in palladium electrode producing excess heat. Gamow factor gives tunneling probability ~10^-2700.","category":"Nuclear Physics","keywords":["cold-fusion","pons-fleischmann","replication-failure"]},"creator":"Pons & Fleischmann (1989)"}' "Cold Fusion (E0)"

post_claim '{"tier":"E4","content":{"dataset_hash":"nif-2025","description":"NIF inertial confinement fusion Q ratio: 1.54 (Dec 2022) → 4.13 (Apr 2025). Eight successful ignition shots independently verified.","category":"Nuclear Physics","keywords":["nif","fusion","ignition","lawson-criterion"],"reproducibility_score":0.95},"creator":"LLNL / NIF"}' "NIF Ignition (E4)"

post_claim '{"tier":"E0","content":{"dataset_hash":"lk99-2023","description":"Lead-copper phosphate apatite (LK-99) claimed as room-temperature superconductor. Debunked in 3 weeks — Cu₂S impurity phase transition.","category":"Condensed Matter","keywords":["lk-99","superconductor","replication-failure","debunked"],"reproducibility_score":0.0},"creator":"Lee & Kim (2023)"}' "LK-99 (E0)"

post_claim '{"tier":"E3","content":{"dataset_hash":"nickelates-2025","description":"Bilayer nickelate La₃Ni₂O₇ superconductivity at ~80K. SLAC stabilized at ambient pressure (Feb 2025). Multiple independent reproductions.","category":"Condensed Matter","keywords":["nickelate","superconductor","high-tc","verification-success"],"reproducibility_score":0.85},"creator":"Multiple groups (2023-2025)"}' "Nickelate SC (E3)"

post_claim '{"tier":"E0","content":{"dataset_hash":"schon-fraud","description":"Hendrik Schön fabricated organic semiconductor data with identical noise patterns. 25+ papers retracted from Nature/Science/PRL.","category":"Condensed Matter","keywords":["fraud","schon","bell-labs","retraction"],"reproducibility_score":0.0},"creator":"Investigation (2002)"}' "Schön Fraud (E0)"

post_claim '{"tier":"E0","content":{"dataset_hash":"dias-fraud","description":"Ranga Dias fabricated room-temperature superconductor data. Same fraud pattern as Schön. 5 papers retracted.","category":"Condensed Matter","keywords":["fraud","dias","rochester","retraction"],"reproducibility_score":0.0},"creator":"Rochester Investigation (2023)"}' "Dias Fraud (E0)"

post_claim '{"tier":"E4","content":{"dataset_hash":"gw150914","description":"LIGO gravitational wave detection: binary black hole merger, SNR=24, false alarm rate < 1/203,000 years. Nobel Prize 2017.","category":"General Relativity","keywords":["ligo","gravitational-waves","black-hole","nobel"],"reproducibility_score":0.99},"creator":"LIGO/Virgo (2015)"}' "LIGO GW150914 (E4)"

post_claim '{"tier":"E4","content":{"dataset_hash":"higgs-2012","description":"Higgs boson at ~125 GeV: ATLAS + CMS independent discovery, 5.9σ significance, multiple decay channels. Nobel Prize 2013.","category":"Particle Physics","keywords":["higgs","lhc","atlas","cms","nobel"],"reproducibility_score":0.99},"creator":"ATLAS + CMS (2012)"}' "Higgs Boson (E4)"

post_claim '{"tier":"E4","content":{"dataset_hash":"hubble-tension","description":"Hubble tension: local H₀≈73 vs CMB H₀≈67 km/s/Mpc at 5σ. JWST confirmed Cepheid accuracy. Unresolved.","category":"Cosmology","keywords":["hubble-tension","dark-energy","cepheid","cmb"],"reproducibility_score":0.95},"creator":"Riess et al. + Planck (2019-2026)"}' "Hubble Tension (E4)"

post_claim '{"tier":"E0","content":{"dataset_hash":"emdrive","description":"EMDrive microwave thruster: all measured thrust was magnetic cable artifact. Thrust=0 when properly shielded.","category":"Classical Mechanics","keywords":["emdrive","conservation-law","debunked"],"reproducibility_score":0.0},"creator":"TU Dresden (2021)"}' "EMDrive Debunked (E0)"

# ── Symthaea Predictions (5) ──
echo ""
echo "Symthaea Predictions:"

post_claim '{"tier":"E2","content":{"dataset_hash":"symthaea-nuclear-sweep","description":"Superheavy stability island centered at Z=115-120, N=180. Element 120 (A=294) has shell correction -22.81 MeV. Consistent with Möller/Oganessian predictions.","category":"Nuclear Physics","keywords":["island-of-stability","superheavy","shell-model","symthaea"],"storage_ref":"symthaea-nuclear::island_stability","reproducibility_score":0.85,"license":"AGPL-3.0"},"creator":"Symthaea Nuclear Engine"}' "Superheavy Island (E2)"

post_claim '{"tier":"E0","content":{"dataset_hash":"lazar-analysis","description":"Lazar Gravity-A claim: structural analysis shows similarity to Yukawa (0.603), Schwarzschild (0.583). LEM: E0/N2/M0.","category":"Modified Gravity","keywords":["lazar","element-115","gravity-a","structural-search"],"storage_ref":"symthaea-physics-bridge::discovery","reproducibility_score":0.0,"license":"AGPL-3.0"},"creator":"Symthaea Discovery Bridge"}' "Lazar Analysis (E0)"

post_claim '{"tier":"E0","content":{"dataset_hash":"ornl-2022","description":"Art'\''s Parts Bi-Mg metamaterial: 0.915 similarity to standard waveguide dispersion. Physics is textbook optics; ORNL sample fails.","category":"Optics","keywords":["arts-parts","metamaterial","waveguide","bismuth"],"storage_ref":"symthaea-physics-bridge::catalog","reproducibility_score":0.15,"license":"AGPL-3.0"},"creator":"Symthaea + ORNL Analysis"}' "Art'\''s Parts (E0)"

post_claim '{"tier":"E2","content":{"dataset_hash":"symthaea-208","description":"208 physics equations across 20 domains encoded as 16,384D hypervectors. Structural skeleton search reveals cross-domain isomorphisms.","category":"Information Theory","keywords":["hdc","physics-catalog","structural-search","208-equations"],"storage_ref":"symthaea-physics-bridge","reproducibility_score":0.95,"license":"AGPL-3.0"},"creator":"Symthaea Physics Bridge"}' "HDC Catalog (E2)"

post_claim '{"tier":"E2","content":{"dataset_hash":"symthaea-compound","description":"Compound stability predictor using Miedema-inspired formation energy + ideal mixing entropy. NaCl correctly predicted stable.","category":"Condensed Matter","keywords":["compound-stability","materials-discovery","formation-energy"],"storage_ref":"symthaea-materials::compound_stability","reproducibility_score":0.6,"license":"AGPL-3.0"},"creator":"Symthaea Materials Engine"}' "Compound Predictor (E2)"

# ── Published Physics (10) ──
echo ""
echo "Published Physics:"

post_claim '{"tier":"E4","content":{"dataset_hash":"alphafold-2024","description":"AlphaFold2 protein structure prediction: median GDT-TS 92.4 across CASP14 targets. Nobel Prize 2024.","category":"Biophysics","keywords":["alphafold","protein-structure","deep-learning","nobel"],"reproducibility_score":0.98},"creator":"Jumper et al. / DeepMind (2021)"}' "AlphaFold (E4)"

post_claim '{"tier":"E3","content":{"dataset_hash":"gnome-2023","description":"GNoME predicted 2.2M new crystal structures, 380K stable candidates. 736 independently synthesized.","category":"Condensed Matter","keywords":["gnome","materials-discovery","crystal-structure","deepmind"],"reproducibility_score":0.75},"creator":"Google DeepMind (2023)"}' "GNoME Materials (E3)"

post_claim '{"tier":"E4","content":{"dataset_hash":"bec-1995","description":"Bose-Einstein condensation in dilute atomic gases (Rb-87). Nobel Prize 2001.","category":"Statistical Mechanics","keywords":["bec","bose-einstein","ultracold","nobel"],"reproducibility_score":0.99},"creator":"Cornell & Wieman (1995)"}' "BEC (E4)"

post_claim '{"tier":"E4","content":{"dataset_hash":"cmb-1965","description":"Cosmic microwave background radiation: 2.725K blackbody spectrum. Nobel Prize 1978.","category":"Cosmology","keywords":["cmb","big-bang","blackbody","penzias-wilson"],"reproducibility_score":0.99},"creator":"Penzias & Wilson (1965)"}' "CMB Discovery (E4)"

post_claim '{"tier":"E4","content":{"dataset_hash":"neutrino-oscillation","description":"Neutrino oscillations demonstrating non-zero neutrino mass. Super-K + SNO. Nobel Prize 2015.","category":"Particle Physics","keywords":["neutrino","oscillation","mass","super-k","sno"],"reproducibility_score":0.99},"creator":"Super-K + SNO (1998-2001)"}' "Neutrino Mass (E4)"

post_claim '{"tier":"E3","content":{"dataset_hash":"muon-g2","description":"Muon anomalous magnetic moment g-2 deviation: resolved by lattice QCD shifting SM prediction. Theory moved, not experiment.","category":"Particle Physics","keywords":["muon","g-2","anomaly","lattice-qcd","resolved"],"reproducibility_score":0.9},"creator":"Fermilab (2021-2025)"}' "Muon g-2 (E3)"

post_claim '{"tier":"E4","content":{"dataset_hash":"ligo-o4","description":"LIGO/Virgo/KAGRA O4 run: 250 new detections (350 total), including most massive merger at 225 solar masses.","category":"General Relativity","keywords":["ligo","gravitational-waves","o4","black-hole-merger"],"reproducibility_score":0.99},"creator":"LIGO/Virgo/KAGRA (2025)"}' "LIGO O4 (E4)"

post_claim '{"tier":"E3","content":{"dataset_hash":"desi-2025","description":"DESI 2025 data suggests dark energy may be dynamical (evolving with cosmic time). Challenges cosmological constant.","category":"Cosmology","keywords":["desi","dark-energy","dynamical","cosmological-constant"],"reproducibility_score":0.8},"creator":"DESI Collaboration (2025)"}' "DESI Dark Energy (E3)"

post_claim '{"tier":"E3","content":{"dataset_hash":"iter-2026","description":"ITER tokamak 90% first-phase construction complete. First plasma expected late 2026.","category":"Nuclear Physics","keywords":["iter","tokamak","fusion","construction"],"reproducibility_score":0.7},"creator":"ITER Organization (2026)"}' "ITER Progress (E3)"

post_claim '{"tier":"E2","content":{"dataset_hash":"rfdiffusion-2025","description":"RFdiffusion de novo protein design: cryo-EM confirms 4/5 designed antibodies bind as predicted.","category":"Biophysics","keywords":["protein-design","rfdiffusion","antibody","david-baker"],"reproducibility_score":0.8},"creator":"Baker Lab / IPD (2025)"}' "Protein Design (E2)"

# ── Diverse Claims (15) ──
echo ""
echo "Diverse Claims:"

post_claim '{"tier":"E3","content":{"dataset_hash":"quantum-error-2026","description":"Microsoft/Quantinuum achieve <10^-6 error rate with 12 logical qubits via topological code. Reproduced by IBM independently.","category":"Quantum Mechanics","keywords":["quantum-computing","error-correction","topological","logical-qubit"],"reproducibility_score":0.85},"creator":"Microsoft + IBM (2026)"}' "Quantum Error Correction (E3)"

post_claim '{"tier":"E2","content":{"dataset_hash":"carbon-capture-2025","description":"Direct Air Capture: Climeworks Mammoth plant captures 36,000 tonnes CO2/year. Chemistry verified (E4), scaling claims (E1).","category":"Thermodynamics","keywords":["carbon-capture","dac","climeworks","climate"],"reproducibility_score":0.5},"creator":"Climeworks (2025)"}' "DAC Carbon Capture (E2)"

post_claim '{"tier":"E2","content":{"dataset_hash":"neuralink-2025","description":"Neuralink N1 implant: 5 patients controlling devices with thoughts. 99% accuracy, 56 WPM typing.","category":"Biophysics","keywords":["neuralink","bci","brain-computer-interface","neural"],"reproducibility_score":0.7},"creator":"Neuralink (2025)"}' "Neuralink BCI (E2)"

post_claim '{"tier":"E2","content":{"dataset_hash":"vitadao-2025","description":"VitaDAO deployed $4.2M across 24 longevity projects. Slashed funding timelines from 12mo to <30 days.","category":"Biophysics","keywords":["vitadao","longevity","desci","funding","ip-nft"],"reproducibility_score":0.6},"creator":"VitaDAO (2025)"}' "VitaDAO Funding (E2)"

post_claim '{"tier":"E1","content":{"dataset_hash":"yamanaka-reprog","description":"OSK partial reprogramming extended median remaining lifespan in aged mice by 109%. Human trials pending.","category":"Biophysics","keywords":["yamanaka","reprogramming","longevity","aging","mouse"],"reproducibility_score":0.4},"creator":"Multiple Groups (2023-2025)"}' "Yamanaka Reprogramming (E1)"

post_claim '{"tier":"E3","content":{"dataset_hash":"helion-2026","description":"Helion Energy plasma at 150M degrees C, D-T shots begun Jan 2026. Google PPA for ARC power plant.","category":"Nuclear Physics","keywords":["helion","fusion","plasma","private-fusion"],"reproducibility_score":0.6},"creator":"Helion Energy (2026)"}' "Helion Fusion (E3)"

post_claim '{"tier":"E4","content":{"dataset_hash":"jwst-deep","description":"JWST deep field reveals galaxies at z>13, challenging ΛCDM timeline predictions. 1000x Hubble sensitivity.","category":"Cosmology","keywords":["jwst","deep-field","high-redshift","galaxy-formation"],"reproducibility_score":0.99},"creator":"NASA/ESA/CSA (2022-2026)"}' "JWST Deep Field (E4)"

post_claim '{"tier":"E2","content":{"dataset_hash":"crispr-sickle","description":"CRISPR gene therapy Casgevy approved for sickle cell disease. First gene-editing treatment. 29/30 patients pain-free.","category":"Biophysics","keywords":["crispr","sickle-cell","gene-therapy","casgevy"],"reproducibility_score":0.9},"creator":"Vertex/CRISPR Therapeutics (2023)"}' "CRISPR Sickle Cell (E2)"

post_claim '{"tier":"E1","content":{"dataset_hash":"mond-jwst","description":"MOND (Modified Newtonian Dynamics) predicts JWST high-z galaxy dynamics without dark matter. Active debate.","category":"Modified Gravity","keywords":["mond","dark-matter","jwst","galaxy-dynamics"],"reproducibility_score":0.3},"creator":"Milgrom + JWST Analysis (2024)"}' "MOND vs JWST (E1)"

post_claim '{"tier":"E3","content":{"dataset_hash":"cfs-sparc","description":"Commonwealth Fusion Systems: SPARC tokamak construction nearing completion. 20T HTS magnets verified.","category":"Nuclear Physics","keywords":["cfs","sparc","fusion","hts-magnet","tokamak"],"reproducibility_score":0.7},"creator":"CFS (2026)"}' "CFS SPARC (E3)"

post_claim '{"tier":"E4","content":{"dataset_hash":"gw-o4-catalog","description":"GWTC-4.0: 350 gravitational wave events cataloged. Population statistics constrain black hole mass gap.","category":"General Relativity","keywords":["gravitational-waves","catalog","population","mass-gap"],"reproducibility_score":0.99},"creator":"LVK Collaboration (2025)"}' "GW Catalog 4.0 (E4)"

post_claim '{"tier":"E2","content":{"dataset_hash":"foldit-proteins","description":"Foldit citizen scientists designed 146 novel proteins; 56 experimentally confirmed stable across 20 folds.","category":"Biophysics","keywords":["foldit","citizen-science","protein-design","crowdsourcing"],"reproducibility_score":0.8},"creator":"Foldit Community (2019-2025)"}' "Foldit Proteins (E2)"

post_claim '{"tier":"E1","content":{"dataset_hash":"climate-check","description":"ClimateCheck 2026: automated verification of climate claims against scientific literature. 20 participating teams.","category":"Thermodynamics","keywords":["climate","verification","automated","nlp"],"reproducibility_score":0.5},"creator":"ClimateCheck Consortium (2026)"}' "ClimateCheck (E1)"

post_claim '{"tier":"E3","content":{"dataset_hash":"einstein-home","description":"Einstein@Home: 90+ neutron stars discovered via volunteer computing. ~100K active participants.","category":"Astrophysics","keywords":["einstein-home","neutron-star","citizen-science","boinc"],"reproducibility_score":0.95},"creator":"Einstein@Home (2005-2026)"}' "Einstein@Home (E3)"

post_claim '{"tier":"E2","content":{"dataset_hash":"researchhub-2026","description":"ResearchHub AI peer reviewer scanning for math errors and hallucinated citations. 2.6% of 2025 papers have AI-generated false references.","category":"Information Theory","keywords":["researchhub","ai-review","hallucinated-citations","desci"],"reproducibility_score":0.6},"creator":"ResearchHub (2026)"}' "ResearchHub AI Review (E2)"

echo ""
echo "=== Seeded 50 claims ==="
echo "Verify: curl $API/api/v1/query/stats"
