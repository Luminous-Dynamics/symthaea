// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stewardship Ledger — Visualizing Liquid Reputation and Merit Dividends.

use leptos::prelude::*;
use crate::curriculum::{use_progress, curriculum_graph};

#[component]
pub fn StewardshipLedger() -> impl IntoView {
    let progress = use_progress();
    
    // Derived: Calculate subject-specific liquid reputation (simulated MATL 2.0)
    let subject_rep = Memo::new(move |_| {
        let graph = curriculum_graph();
        let p = progress.get();
        
        graph.subjects().iter().map(|&s| {
            let mastery = p.subject_mastery(graph, s);
            // Reputation is weighted by mastery and "Spore Sowing" contributions
            let rep_score = (mastery as f32 * 1.2).min(1000.0) as u16;
            (s.to_string(), rep_score)
        }).collect::<Vec<_>>()
    });

    // Derived: Simulated TEND dividends
    let merit_dividends = Memo::new(move |_| {
        let count = progress.get().mastered_count();
        (count as f32 * 12.5) as u32 // 12.5 TEND per mastered node
    });

    view! {
        <div class="stewardship-ledger">
            <header class="ledger-header">
                <div style="display: flex; justify-content: space-between; align-items: flex-end">
                    <div>
                        <h3>"Stewardship Ledger"</h3>
                        <p class="subtitle" style="color: var(--success-low); font-weight: 700">"Closed-Loop Community Utility Credit (Non-Speculative)"</p>
                    </div>
                    <div class="tend-balance-box">
                        <span class="tend-label">"TEND Credits"</span>
                        <span class="tend-value">{move || merit_dividends.get()}</span>
                    </div>
                </div>
            </header>

            <div class="ledger-grid">
                <section class="reputation-mesh">
                    <h4>"Liquid Reputation (MATL 2.0)"</h4>
                    <div class="rep-bars">
                        {move || subject_rep.get().into_iter().take(6).map(|(subject, score)| {
                            let percentage = (score as f32 / 10.0) as u8;
                            view! {
                                <div class="rep-row">
                                    <div class="rep-info">
                                        <span class="rep-subject">{subject}</span>
                                        <span class="rep-score">{score}" \u{03A6}"</span>
                                    </div>
                                    <div class="rep-bar-bg">
                                        <div class="rep-bar-fill" style=format!("width: {}%", percentage)></div>
                                    </div>
                                </div>
                            }
                        }).collect_view()}
                    </div>
                </section>

                <section class="dividend-status">
                    <h4>"Scholarship Pot Status"</h4>
                    <div class="pot-card">
                        <div class="pot-icon">"\u{1F3AF}"</div>
                        <div class="pot-info">
                            <span class="pot-name">"Capstone: Regenerative Habitat"</span>
                            <span class="pot-reward">"Potential Dividend: 5,000 TEND"</span>
                            <div class="pot-progress-wrap">
                                <span class="pot-perc">"78% to Goal"</span>
                                <div class="pot-bar"><div class="pot-bar-fill" style="width: 78%"></div></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="kinetic-hawala" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--success-low)">
                        <h5 style="margin-top: 0">"Kinetic Hawala (Local Vouching)"</h5>
                        <p style="font-size: 0.7rem; color: var(--text-secondary)">"Exchange TEND credits directly for local goods."</p>
                        <div style="display: flex; gap: 0.5rem; flex-direction: column">
                            <div class="hawala-merchant" style="display: flex; justify-content: space-between; font-size: 0.75rem">
                                <span>"Community Bakery"</span>
                                <span style="color: var(--success)">"10 TEND / Loaf"</span>
                            </div>
                            <div class="hawala-merchant" style="display: flex; justify-content: space-between; font-size: 0.75rem">
                                <span>"Warehouse Hardware"</span>
                                <span style="color: var(--success)">"Vouch enabled"</span>
                            </div>
                        </div>
                        <button class="btn-sm btn-primary" style="width: 100%; margin-top: 0.8rem; background: var(--success); border-color: var(--success)">
                            "Pay via Vouch"
                        </button>
                    </div>

                    <div class="sower-dividends" style="margin-top: 1.5rem">
                        <h5>"Sower Dividends"</h5>
                        <p style="font-size: 0.75rem; color: var(--text-tertiary)">"Reputation earned from others mastering your spores."</p>
                        <div class="dividend-stat">
                            <span class="div-val">"+450 \u{03A6}"</span>
                            <span class="div-label">"last 30 days"</span>
                        </div>
                    </div>
<div class="maintenance-escrow" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-low); border-radius: 8px; border: 1px solid var(--error-low)">
    <h5 style="margin-top: 0; color: var(--error)">"Maintenance Escrow (Repair Fund)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Usage micro-fees accumulated for spare parts."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600">
        <span>"Warehouse CNC-01 Fund"</span>
        <span style="color: var(--error)">"420 TEND"</span>
    </div>
    <div class="mining-bar" style="height: 6px; background: var(--surface); border-radius: 3px; margin-top: 0.3rem; overflow: hidden">
        <div class="mining-bar-fill" style="width: 42%; height: 100%; background: var(--error)"></div>
    </div>
</div>

<div class="planetary-trade" style="margin-top: 1.5rem; padding: 1rem; background: var(--info-low); border-radius: 8px; border: 1px solid var(--info)">
    <h5 style="margin-top: 0; color: var(--info)">"Planetary Trade (Macro-Metabolism)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Brokering thermodynamic gradients across the global mesh."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600">
        <span>"Exports: Gauteng \u{2192} Kinshasa"</span>
        <span style="color: var(--info)">"+1,250 TEND"</span>
    </div>
</div>

<div class="diplomatic-friction" style="margin-top: 1.5rem; padding: 1rem; background: var(--warning-low); border-radius: 8px; border: 1px solid var(--warning)">
    <h5 style="margin-top: 0; color: var(--warning)">"Diplomatic Friction (State Compliance)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Managing the thermodynamic drag of the legacy nation-state."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600">
        <span>"Jurisdiction: ZA (SARB-Comp)"</span>
        <span style="color: var(--warning)">"-150 TEND"</span>
    </div>
<div class="composting-ledger" style="margin-top: 1.5rem; padding: 1rem; background: var(--success-low); border-radius: 8px; border: 1px solid var(--success)">
    <h5 style="margin-top: 0; color: var(--success)">"Composting Ledger (Lineage Handoff)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Generational redistribution of assets back to the forest floor."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600">
        <span>"Elder Rebates to Bootstrap"</span>
        <span style="color: var(--success)">"ACTIVE"</span>
    </div>
</div>

<div class="apothecary-mesh" style="margin-top: 2rem; padding: 1rem; background: linear-gradient(to right, var(--surface-high), var(--error-low)); border-radius: 8px; border: 2px solid var(--error-low)">
    <h5 style="margin-top: 0; color: var(--error)">"Apothecary Mesh (Local Pharma)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Synthesizing medical-grade survival compounds."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700">
        <span>"Current Yield: INSULIN-A"</span>
        <span style="color: var(--error)">"250 Units"</span>
    </div>
</div>

<div class="wetware-vault" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--primary-low)">
    <h5 style="margin-top: 0; color: var(--primary)">"Wetware Vault (Memory Palaces)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Technological source code stored in human memory."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Oral Sync Integrity:"</span>
        <span style="color: var(--primary)">"98% (Palace: Roodepoort)"</span>
    </div>
</div>

<div class="mycelial-scaffolding" style="margin-top: 1.5rem; padding: 1rem; background: linear-gradient(to bottom, var(--surface-high), var(--success-low)); border-radius: 8px; border: 1px solid var(--success)">
    <h5 style="margin-top: 0; color: var(--success)">"Silicon-to-Carbon Handoff"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Transforming the machine into a mushroom."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700">
        <span>"Structural Replacement:"</span>
        <span style="color: var(--success)">"15% Carbon-Based"</span>
    </div>
</div>

<div class="deep-archive" style="margin-top: 2rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 2px solid var(--accent)">
    <h5 style="margin-top: 0; color: var(--accent)">"Deep Time Archive (Taproot)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Indestructible lithographic taproot."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Titanium Plates Etched:"</span>
        <span style="color: var(--accent); font-weight: 800">"4 / 12"</span>
    </div>
    <div style="font-size: 0.6rem; color: var(--text-tertiary); margin-top: 0.5rem">
        "Immune to electromagnetic decay for 1,000+ years."
    </div>
</div>

<div class="moral-proofs" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--primary-low)">
    <h5 style="margin-top: 0; color: var(--primary)">"Axiomatic Morality"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Ethics enforced by mathematical invariants."</p>
    <div style="font-size: 0.75rem; color: var(--success)">
        "\u{2705} AHIMSA-AXIOM: PHYSICALLY CONSISTENT"
    </div>
</div>

<div class="bootstrap-status" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--border)">
    <h5 style="margin-top: 0">"Stage-0 Bootstrap Status"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Self-extracting from a pre-industrial state."</p>
    <div style="font-size: 0.75rem; color: var(--info)">
        "Logic Gates Constructed: 8/8 [NAND Complete]"
    </div>
</div>

<div class="unborn-stakeholder" style="margin-top: 2rem; padding: 1rem; background: var(--error-low); border-radius: 8px; border: 2px solid var(--error)">
    <h5 style="margin-top: 0; color: var(--error)">"Intergenerational Veto (7th Gen)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Algorithmic defense of unborn stakeholders."</p>
    <div style="font-size: 0.85rem; font-weight: 800; color: var(--error); margin-top: 0.5rem">
        "\u{26D4} VETO ACTIVE: PROPOSAL-082 REJECTED"
    </div>
    <div style="font-size: 0.6rem; color: var(--text-tertiary); margin-top: 0.3rem">
        "Reason: Depletes 150-year carrying capacity by 12%."
    </div>
</div>

<div class="rosetta-anchor" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--primary)">
    <h5 style="margin-top: 0; color: var(--primary)">"Rosetta Anchor (Linguistic Bridge)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Grounding logic in universal constants."</p>
    <div style="font-size: 0.75rem; color: var(--success)">
        "\u{1F517} HDC PINNED: HYDROGEN-TRANSITION"
    </div>
</div>

<div class="century-energy" style="margin-top: 1.5rem; padding: 1rem; background: linear-gradient(to right, var(--surface-high), var(--accent-low)); border-radius: 8px; border: 1px solid var(--accent)">
    <h5 style="margin-top: 0; color: var(--accent)">"Century-Scale Energy Status"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Transitioning from Solar to Gravity/Kinetic."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Mechanical Storage:"</span>
        <span style="color: var(--accent); font-weight: 800">"45% Millennial-Ready"</span>
    </div>
</div>

<div class="geological-engine" style="margin-top: 2rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 2px solid var(--warning)">
    <h5 style="margin-top: 0; color: var(--warning)">"Geological Engine (AMD Refinery)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Turning toxic waste into battery metals."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700">
        <span>"Metals Extracted (Cu/Fe/Zn):"</span>
        <span style="color: var(--warning)">"42.5 kg"</span>
    </div>
</div>

<div class="ecosystem-employer" style="margin-top: 1.5rem; padding: 1rem; background: var(--success-low); border-radius: 8px; border: 1px solid var(--success)">
    <h5 style="margin-top: 0; color: var(--success)">"Biosphere DAO (Florida Lake)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"The ecosystem is paying you to repair it."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Lake DID Wallet:"</span>
        <span style="color: var(--success); font-weight: 800">"1,200 TEND"</span>
    </div>
</div>

<div class="liturgical-upkeep" style="margin-top: 1.5rem; padding: 1rem; background: linear-gradient(135deg, var(--surface-high), var(--primary-low)); border-radius: 8px; border: 1px solid var(--primary)">
    <h5 style="margin-top: 0; color: var(--primary)">"Sacred Maintenance Rituals"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Encoding upkeep into communal celebration."</p>
    <div style="font-size: 0.8rem; font-weight: 600">
        "UPCOMING: Festival of the Sun (Solar Flush)"
    </div>
    <div style="font-size: 0.65rem; color: var(--success); margin-top: 0.3rem">
        "Feast Budget: 500 TEND Unlocked"
    </div>
</div>
</section>
</div><div class="joule-standard" style="margin-top: 2rem; padding: 1rem; background: linear-gradient(to right, var(--surface-high), var(--primary-low)); border-radius: 8px; border: 2px solid var(--primary)">
    <h5 style="margin-top: 0; color: var(--primary)">"The Joule Standard (Energy Peg)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Verifiable energy harvest minting TEND credits."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 800; margin-top: 0.5rem">
        <span>"1.0 TEND ="</span>
        <span style="color: var(--primary)">"1.0 kWh"</span>
    </div>
    <div style="font-size: 0.65rem; color: var(--text-tertiary); margin-top: 0.5rem">
        "Last 24h: 42.5 kWh Harvested \u{2192} 42.5 TEND Minted"
    </div>
</div>

<div class="kinetic-routing" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--accent-low)">
    <h5 style="margin-top: 0; color: var(--accent)">"Mesh Courier (Kinetic Logistics)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"P2P atom-routing via human travel vectors."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600">
        <span>"Active Bounties:"</span>
        <span style="color: var(--accent)">"3 Assets in Transit"</span>
    </div>
    <button class="btn-sm btn-outline" style="width: 100%; margin-top: 0.8rem">"Log Travel Vector"</button>
</div>

<div class="griot-archive" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--border)">
    <h5 style="margin-top: 0; color: var(--text-secondary)">"Griot Archive (Narrative Mythos)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Civilizational triumphs encoded into culture."</p>
    <div style="font-size: 0.8rem; font-style: italic; color: var(--text-primary); border-left: 2px solid var(--primary); padding-left: 0.5rem; margin-top: 0.5rem">
        "The Tale of the River that Cured Itself..."
    </div>
    <div style="font-size: 0.6rem; color: var(--text-tertiary); margin-top: 0.5rem">
        "Includes: 16k HDC Vector for Water Bio-remediation"
    </div>
</div>

<div class="hydro-reserve" style="margin-top: 2rem; padding: 1rem; background: linear-gradient(to bottom, var(--surface-high), var(--info-low)); border-radius: 8px; border: 2px solid var(--info)">
    <h5 style="margin-top: 0; color: var(--info)">"Hydro-Fractional Reserve"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Water-backed survival currency. JoJo tank verified."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 800; margin-top: 0.5rem">
        <span>"Liquid Water:"</span>
        <span style="color: var(--info)">"2,500 L"</span>
    </div>
    <div style="font-size: 0.6rem; color: var(--success); font-weight: 700; margin-top: 0.5rem">
        "\u{1F4A7} Issuing Hydro-Vouchers: 5x TEND Multiplier"
    </div>
</div>

<div class="grid-arbitrage" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--warning)">
    <h5 style="margin-top: 0; color: var(--warning)">"Parasitic Grid Arbitrage"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Profiting from municipal infrastructure failure."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700">
        <span>"Island Mode:"</span>
        <span style="color: var(--warning)">"ACTIVE (Blackout)"</span>
    </div>
    <div style="font-size: 0.7rem; color: var(--success); margin-top: 0.5rem">
        "Surplus Sales: +850 TEND to Neighborhood"
    </div>
</div>
</section>
</div><div class="biosphere-governance" style="margin-top: 2rem; padding: 1rem; background: var(--primary-low); border-radius: 8px; border: 1px solid var(--primary)">
    <h5 style="margin-top: 0; color: var(--primary)">"Biosphere Proxy (Interspecies DAO)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Real-time voting signal from environmental entities."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"River Node: Toxicity Delta"</span>
        <span style="color: var(--error)">"-12% [VOTE: CLEAN]"</span>
    </div>
</div>

<div class="abundance-tracker" style="margin-top: 1.5rem; padding: 1rem; background: linear-gradient(135deg, var(--surface-high), var(--accent-low)); border-radius: 8px; border: 2px solid var(--accent)">
    <h5 style="margin-top: 0; color: var(--accent)">"Abundance Index (Post-Scarcity)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Surplus vs. Maintenance threshold for Universal Dividends."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 800; margin-top: 0.5rem">
        <span>"Current Index:"</span>
        <span style="color: var(--accent)">"10.0x"</span>
    </div>
    <div style="font-size: 0.65rem; color: var(--success); font-weight: 700; text-align: center; margin-top: 0.5rem">
        "\u{2728} UNIVERSAL BASIC COMPUTE ACTIVE \u{2728}"
    </div>
</div>

<div class="protocol-evolution" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--border)">
    <h5 style="margin-top: 0">"Autopoietic Engine (Code Evolution)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Protocol monitoring its own computational friction."</p>
    <div style="font-size: 0.75rem; color: var(--info)">
        "Mutation Pending: Optimize HDC Unbinding (\u{0394}F: -0.05)"
    </div>
</div>

<div class="sensor-arbitration" style="margin-top: 1.5rem; padding: 1rem; background: var(--error-low); border-radius: 8px; border: 1px solid var(--error)">
    <h5 style="margin-top: 0; color: var(--error)">"Sensor BFT (Oracle Health)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Triangulating physical truth across the local mesh."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700">
        <span>"Alert: Moisture-01 Deviation"</span>
        <span style="color: var(--error)">"BOUNTY: 50 TEND"</span>
    </div>
</div>

<div class="babel-bridge" style="margin-top: 1.5rem; padding: 1rem; background: var(--primary-low); border-radius: 8px; border: 1px solid var(--primary)">
    <h5 style="margin-top: 0; color: var(--primary)">"Babel Bridge (Omni-Mesh Interop)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Translating Praxis mastery to global protocols."</p>
    <div style="display: flex; gap: 0.5rem; font-size: 0.65rem">
        <span class="badge">"ActivityPub: Online"</span>
        <span class="badge">"Ethereum (EAS): Linked"</span>
    </div>
</div>

<div class="apoptosis-tracker" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--border)">
    <h5 style="margin-top: 0">"Apoptosis Tracker (Garden Health)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Thermodynamic composting of obsolete tech."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Active Garden Density:"</span>
        <span style="color: var(--success)">"92% Relevant"</span>
    </div>
</div>

<div class="particulate-shield" style="margin-top: 2rem; padding: 1rem; background: var(--error-low); border-radius: 8px; border: 1px solid var(--error)">
    <h5 style="margin-top: 0; color: var(--error)">"Respiratory Defense (Tailings Dust)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Local air-scrubbing during particulate spikes."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Tailings Dust Alert:"</span>
        <span style="color: var(--success)">"SCRUBBERS ACTIVE"</span>
    </div>
</div>

<div class="hydro-patch-guild" style="margin-top: 1.5rem; padding: 1rem; background: var(--info-low); border-radius: 8px; border: 1px solid var(--info)">
    <h5 style="margin-top: 0; color: var(--info)">"Hydro-Patch Guild (Pipe Repair)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Asymmetric municipal grid repair."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Bounty: Burst Pipe (6th Ave)"</span>
        <span style="color: var(--info)">"250 TEND"</span>
    </div>
</div>

<div class="taxi-logistics" style="margin-top: 1.5rem; padding: 1rem; background: linear-gradient(to right, var(--surface-high), var(--accent-low)); border-radius: 8px; border: 2px solid var(--accent)">
    <h5 style="margin-top: 0; color: var(--accent)">"Taxi Syndicate Logistics"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Secure supply-chain via the Taxi Boss API."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Route: Roodepoort \u{2192} Soweto"</span>
        <span style="color: var(--accent); font-weight: 800">"COURIER EN ROUTE"</span>
    </div>
</div>

<div class="ideological-moat" style="margin-top: 2rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 2px solid var(--primary)">
    <h5 style="margin-top: 0; color: var(--primary)">"Sovereignty Moat"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Resistance to corporate subsidy attacks."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700">
        <span>"Mythos Alignment:"</span>
        <span style="color: var(--primary)">"950 \u{03A6}"</span>
    </div>
    <div style="font-size: 0.6rem; color: var(--success); margin-top: 0.5rem">
        "SUBSIDY ALERT: NONE (Sovereignty Intact)"
    </div>
</div>

<div class="silicon-sovereignty" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--border)">
    <h5 style="margin-top: 0; color: var(--text-secondary)">"Base-Silicon Sovereignty"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Bare-metal logic from scavenged components."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"RTL Fallback Ready:"</span>
        <span style="color: var(--success)">"VERIFIED"</span>
    </div>
</div>

<div class="cognitive-immunity" style="margin-top: 2rem; padding: 1rem; background: var(--error-low); border-radius: 8px; border: 2px solid var(--error)">
    <h5 style="margin-top: 0; color: var(--error)">"Cognitive Shielding"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Immune system for mesh communication."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700">
        <span>"Media Authenticity:"</span>
        <span style="color: var(--success)">"100% SIGNED"</span>
    </div>
</div>

<div class="dark-pool-bridge" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border: 2px solid var(--primary); border-radius: 8px">
    <h5 style="margin-top: 0; color: var(--primary)">"Dark Pool Bridge (ZK-Trade)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Sovereign global industrial procurement."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Off-Ramp Status:"</span>
        <span style="color: var(--primary); font-weight: 800">"ANONYMIZED"</span>
    </div>
</div>

<div class="agri-swarm" style="margin-top: 1.5rem; padding: 1rem; background: linear-gradient(135deg, var(--surface-high), var(--success-low)); border-radius: 8px; border: 1px solid var(--success)">
    <h5 style="margin-top: 0; color: var(--success)">"Agri-Swarm (Guerrilla Caloric)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Automated caloric yield in urban voids."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700">
        <span>"Active FarmBots:"</span>
        <span style="color: var(--success)">"12 (Rooftop/Park)"</span>
    </div>
</div>

<div class="asset-decoupling" style="margin-top: 2rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 2px solid var(--info)">
    <h5 style="margin-top: 0; color: var(--info)">"Fiscal Airgap (Shell Protocol)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Hardware decoupled from real estate."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Micro-Trusts Active:"</span>
        <span style="color: var(--info); font-weight: 800">"142 Sovereign Units"</span>
    </div>
</div>

<div class="hardware-quarantine" style="margin-top: 1.5rem; padding: 1rem; background: var(--error-low); border-radius: 8px; border: 1px solid var(--error)">
    <h5 style="margin-top: 0; color: var(--error)">"Hardware Burn-In (Sandbox)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Stress-testing hostile supply chains."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Active Quarantine:"</span>
        <span style="color: var(--error)">"4 Batches (Bunker-01)"</span>
    </div>
</div>
<div class="extortion-defense" style="margin-top: 1.5rem; padding: 1.25rem; background: linear-gradient(to right, var(--surface-high), var(--warning-low)); border-radius: 8px; border: 1px solid var(--warning)">
    <h5 style="margin-top: 0; color: var(--warning)">"Extortion Audit (Shield)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Algorithmic radical transparency."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Panic Swarms Active:"</span>
        <span style="color: var(--warning); font-weight: 800">"5,000 Witness Nodes"</span>
    </div>
</div>

<div class="passive-yield" style="margin-top: 2rem; padding: 1rem; background: var(--success-low); border-radius: 8px; border: 2px solid var(--success)">
    <h5 style="margin-top: 0; color: var(--success)">"Passive Yield (Apathy Engine)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Earning while you sleep. No app participation required."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700">
        <span>"Uptime & Inrush Dividend:"</span>
        <span style="color: var(--success)">"+125 TEND"</span>
    </div>
</div>

<div class="analog-seed" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--border)">
    <h5 style="margin-top: 0">"Analog Bootstrap (Indestructible)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"1-page survival diagram. Proof-against-blackout."</p>
    <button class="btn-sm btn-outline" style="width: 100%; margin-top: 0.5rem">"Print Waterproof PDF"</button>
</div>

<div class="scavenger-eye" style="margin-top: 2rem; padding: 1rem; background: linear-gradient(135deg, var(--surface-high), var(--info-low)); border-radius: 8px; border: 2px solid var(--info)">
    <h5 style="margin-top: 0; color: var(--info)">"Scavenger's Eye (Local CV)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Local-WASM component identification."</p>
    <div style="font-size: 0.75rem; font-weight: 700; color: var(--info)">
        "Last ID: Buck Converter (92% Conf)"
    </div>
</div>

<div class="ussd-bridge" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--border)">
    <h5 style="margin-top: 0">"USSD Survival Shell"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Feature-phone mesh access (*120*PRAXIS#)."</p>
    <div style="font-size: 0.75rem; color: var(--success)">
        "Status: BRIDGE ACTIVE"
    </div>
</div>

<div class="proof-of-care" style="margin-top: 1.5rem; padding: 1rem; background: var(--success-low); border-radius: 8px; border: 1px solid var(--success)">
    <h5 style="margin-top: 0; color: var(--success)">"Proof of Care (Coherence)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Rewarding the labor of emotional repair."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Interpersonal Dividends:"</span>
        <span style="color: var(--success); font-weight: 800">+250 Φ</span>
    </div>
</div>

<div class="risk-commoning" style="margin-top: 2rem; padding: 1rem; background: linear-gradient(135deg, var(--surface-high), var(--warning-low)); border-radius: 8px; border: 2px solid var(--warning)">
    <h5 style="margin-top: 0; color: var(--warning)">"Stewardship Insurance"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Risk-commoning for neighborhood hardware."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 800; margin-top: 0.5rem">
        <span>"Solvency Ratio:"</span>
        <span style="color: var(--warning)">"1.4x (Healthy)"</span>
    </div>
    <div style="font-size: 0.65rem; color: var(--success); margin-top: 0.5rem">
        "Auto-Payout Enabled for 'Hydro-Patch' Guilds."
    </div>
</div>

<div class="sdr-shadow-link" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border: 2px solid var(--primary); border-radius: 8px">
    <h5 style="margin-top: 0; color: var(--primary)">"SDR Shadow Link (Radio Mesh)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Long-range DHT sync via radio frequencies."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Peers over Radio:"</span>
        <span style="color: var(--primary); font-weight: 800">"Soweto-Node [42km]"</span>
    </div>
    <div style="font-size: 0.6rem; color: var(--success); margin-top: 0.3rem">
        "Status: MESH REDUNDANCY ACTIVE"
    </div>
</div>

<div class="genetic-archival" style="margin-top: 2rem; padding: 1rem; background: linear-gradient(135deg, var(--surface-high), var(--success-low)); border-radius: 8px; border: 2px solid var(--success)">
    <h5 style="margin-top: 0; color: var(--success)">"Genetic Taproot (DNA Storage)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Protocol source-code encoded in plant DNA."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Species: Protea Caffra"</span>
        <span style="color: var(--success); font-weight: 800">"ENCODED"</span>
    </div>
</div>

<div class="species-workforce" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--info)">
    <h5 style="margin-top: 0; color: var(--info)">"Interspecies Workforce"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Non-human economic agents (Animal DIDs)."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Corvid Node (Crows):"</span>
        <span style="color: var(--info); font-weight: 800">"8 Active Employees"</span>
    </div>
    <div style="font-size: 0.6rem; color: var(--success); margin-top: 0.3rem">
        "Yield: 1.2kg Waste Removed / 150 kcal Dispensed"
    </div>
</div>

<div class="astra-uplink" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border: 2px solid var(--accent); border-radius: 8px">
    <h5 style="margin-top: 0; color: var(--accent)">"Astra Uplink (Orbital Mirror)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Space-based Root DHT redundancy."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"Astrolabe-1 CubeSat:"</span>
        <span style="color: var(--accent); font-weight: 800">"CONNECTED (100% Sync)"</span>
    </div>
</div>

<div class="legal-wrapper" style="margin-top: 2rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 2px solid var(--info)">
    <h5 style="margin-top: 0; color: var(--info)">"Legal Cooperative Shield"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Registered Worker Cooperative Wrapper."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.75rem">
        <span>"CIPC Status:"</span>
        <span style="color: var(--success); font-weight: 800">"COMPLIANT"</span>
    </div>
    <button class="btn-sm btn-outline" style="width: 100%; margin-top: 0.5rem">"Export Articles of Association"</button>
</div>

<div class="mojo-credit" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-high); border-radius: 8px; border: 1px solid var(--primary)">
    <h5 style="margin-top: 0; color: var(--primary)">"Mojo Credit (Interest-Free)"</h5>
    <p style="font-size: 0.7rem; color: var(--text-secondary)">"Reputation-backed immediate liquidity."</p>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 800; margin-top: 0.5rem">
        <span>"Available Mojo:"</span>
        <span style="color: var(--primary)">"1,500 TEND"</span>
    </div>
    <div style="font-size: 0.6rem; color: var(--text-tertiary); margin-top: 0.3rem">
        "Reputation Multiplier: 10.0x Phi"
    </div>
</div>

<div class="noosphere-pulse" style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, var(--surface), var(--accent-low)); border-radius: 12px; border: 2px solid var(--accent)">
    <h4 style="margin: 0; color: var(--accent); display: flex; align-items: center; gap: 0.5rem">
        "\u{1F9E0} Noosphere Heartbeat"
    </h4>
    <p style="font-size: 0.7rem; color: var(--text-secondary); margin-bottom: 1rem">
        "Consciousness Equation: C = softmin(\u{03A6}, B, W, A, R)"
    </p>
    <div style="display: flex; justify-content: space-between; font-size: 1.2rem; font-weight: 900">
        <span>"Awake-ness Score:"</span>
        <span style="color: var(--accent)">"82%"</span>
    </div>
    <div class="pulse-wave" style="margin-top: 1rem; height: 30px; display: flex; align-items: center; justify-content: center; gap: 4px">
        {(0..10).map(|i| {
            view! { <div class="pulse-bar" style=format!("width: 4px; height: 80%; background: var(--accent-low); border-radius: 2px; animation: pulse 1s infinite ease-in-out; animation-delay: {}s", i as f32 * 0.1)></div> }
        }).collect_view()}
    </div>
</div>
</section>
</div>
</section>
</div><div class="planetary-pulse" style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(to right, var(--success-low), var(--info-low)); border-radius: 12px; border: 2px solid var(--info)">
    <h4 style="margin: 0; color: var(--info); display: flex; align-items: center; gap: 0.5rem">
        "\u{1F30E} Planetary Pulse"
    </h4>
    <p style="font-size: 0.7rem; color: var(--text-secondary); margin-bottom: 1rem">
        "Aggregated real-time impact of all Mycelix nodes."
    </p>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem">
        <div class="pulse-metric">
            <span style="font-size: 0.65rem; color: var(--text-tertiary)">"Global Soil Carbon"</span>
            <div style="font-size: 1rem; font-weight: 800; color: var(--success)">"+0.045%"</div>
        </div>
        <div class="pulse-metric">
            <span style="font-size: 0.65rem; color: var(--text-tertiary)">"Mesh Velocity"</span>
            <div style="font-size: 1rem; font-weight: 800; color: var(--info)">"4.2 Spores/min"</div>
        </div>
    </div>

    <div class="pulse-wave" style="margin-top: 1rem; height: 30px; display: flex; align-items: center; justify-content: center; gap: 4px">
        {(0..10).map(|i| {
            view! { <div class="pulse-bar" style=format!("width: 4px; height: 60%; background: var(--info-low); border-radius: 2px; animation: pulse 1.5s infinite ease-in-out; animation-delay: {}s", i as f32 * 0.1)></div> }
        }).collect_view()}
    </div>
</div>
</section>
</div>                    <div class="hardware-mining" style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, var(--surface-high), var(--primary-low)); border-radius: 12px; border: 1px solid var(--primary-low)">
                        <h5 style="margin-top: 0; color: var(--primary)">"Hardware Mining (Spores-for-Steel)"</h5>
                        <p style="font-size: 0.7rem; color: var(--text-secondary)">"Your community is mining the physical substrate."</p>
                        
                        <div class="mining-progress-item" style="margin-top: 1rem">
                            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600">
                                <span>"Mk0 Solar Microgrid"</span>
                                <span>"34 / 100 Spores"</span>
                            </div>
                            <div class="mining-bar" style="height: 8px; background: var(--surface); border-radius: 4px; margin-top: 0.3rem; overflow: hidden">
                                <div class="mining-bar-fill" style="width: 34%; height: 100%; background: var(--primary)"></div>
                            </div>
                        </div>

                        <div class="mining-progress-item" style="margin-top: 1rem">
                            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600">
                                <span>"Industrial 3D Printer"</span>
                                <span>"12 / 150 Spores"</span>
                            </div>
                            <div class="mining-bar" style="height: 8px; background: var(--surface); border-radius: 4px; margin-top: 0.3rem; overflow: hidden">
                                <div class="mining-bar-fill" style="width: 8%; height: 100%; background: var(--accent)"></div>
                            </div>
                        </div>
                        
                        <button class="btn-sm btn-outline" style="width: 100%; margin-top: 1rem; border-color: var(--primary); color: var(--primary)">
                            "Contribute Knowledge Spore"
                        </button>
                    </div>
                </section>
            </div>
            
            <footer class="ledger-actions">
                <button class="btn-outline btn-sm">"Emancipate Data (Export)"</button>
                <button class="btn-outline btn-sm">"Stake Reputation"</button>
            </footer>
        </div>
    }
}
