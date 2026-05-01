// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Mesh Infrastructure Context for Praxis.
//!
//! Provides reactive signals for local mesh connectivity, peer discovery,
//! and LoRa radio status. Integrates with the Symthaea Swarm Mesh.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

/// Status of the local mesh radio (e.g. LoRa HAT on Raspberry Pi)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MeshRadioStatus {
    #[default]
    Disabled,
    Initializing,
    Ready,
    Transmitting,
    Error,
}

/// A peer discovered over the local physical mesh (LoRa/B.A.T.M.A.N.)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshPeer {
    pub node_id: String,
    pub last_seen: u64,
    pub rssi: i16,      // Signal strength
    pub distance_est: f32, // Estimated distance in km
    pub is_pod_mate: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct MeshContext {
    pub radio_status: ReadSignal<MeshRadioStatus>,
    pub set_radio_status: WriteSignal<MeshRadioStatus>,
    pub active_peers: Memo<Vec<MeshPeer>>,
    pub mesh_certified_count: Memo<u32>,
}

/// Provide mesh context to the application
pub fn provide_mesh_context() {
    let (radio_status, set_radio_status) = signal(MeshRadioStatus::Ready); // Mock ready for now
    let (peers, _set_peers) = signal(vec![
        MeshPeer {
            node_id: "peer_alpha".to_string(),
            last_seen: 100,
            rssi: -85,
            distance_est: 2.5,
            is_pod_mate: true,
        },
        MeshPeer {
            node_id: "peer_beta".to_string(),
            last_seen: 200,
            rssi: -105,
            distance_est: 8.2,
            is_pod_mate: false,
        },
    ]);

    let active_peers = Memo::new(move |_| peers.get());
    let mesh_certified_count = Memo::new(move |_| 12); // Mock count

    provide_context(MeshContext {
        radio_status,
        set_radio_status,
        active_peers,
        mesh_certified_count,
    });
}

pub fn use_mesh() -> MeshContext {
    expect_context::<MeshContext>()
}

/// Component to show local mesh connectivity status
#[component]
pub fn MeshStatusBadge() -> impl IntoView {
    let ctx = use_mesh();
    let (show_peers, set_show_peers) = signal(false);
    
    view! {
        <div class="mesh-badge-container" style="position: relative">
            <div 
                class="mesh-badge" 
                title="Local Mycelial Mesh Status"
                on:click=move |_| set_show_peers.update(|v| *v = !*v)
                style="cursor: pointer; display: flex; align-items: center; gap: 0.5rem; padding: 0.25rem 0.75rem; background: var(--surface-low); border-radius: 20px; border: 1px solid var(--border)"
            >
                {move || match ctx.radio_status.get() {
                    MeshRadioStatus::Disabled => view! { <span class="mesh-icon disabled">"\u{1F4F6}"</span> }.into_any(),
                    MeshRadioStatus::Initializing => view! { <span class="mesh-icon loading">"\u{231B}"</span> }.into_any(),
                    MeshRadioStatus::Ready => view! { 
                        <div style="display: flex; align-items: center; gap: 0.35rem">
                            <span class="mesh-icon ready" style="color: var(--success)">"\u{1F4F6}"</span>
                            <span style="font-size: 0.75rem; font-weight: 700; color: var(--text)">
                                {move || ctx.active_peers.get().len()}
                            </span>
                        </div>
                    }.into_any(),
                    MeshRadioStatus::Transmitting => view! { <span class="mesh-icon pulse">"\u{1F4E1}"</span> }.into_any(),
                    MeshRadioStatus::Error => view! { <span class="mesh-icon error">"\u{26A0}"</span> }.into_any(),
                }}
            </div>

            // Nearby Peers Dropdown
            {move || if show_peers.get() {
                let peers = ctx.active_peers.get();
                view! {
                    <div class="mesh-peers-dropdown" style="position: absolute; top: 110%; right: 0; width: 220px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 1000; padding: 0.75rem">
                        <div style="font-size: 0.7rem; font-weight: 700; color: var(--text-tertiary); text-transform: uppercase; margin-bottom: 0.5rem">"Nearby via Swarm Mesh"</div>
                        {if peers.is_empty() {
                            view! { <div style="font-size: 0.8rem; color: var(--text-secondary); padding: 0.5rem 0">"No peers in range"</div> }.into_any()
                        } else {
                            view! {
                                <div style="display: flex; flex-direction: column; gap: 0.5rem">
                                    {peers.into_iter().map(|p| {
                                        let id = p.node_id.clone();
                                        let rssi = p.rssi;
                                        let dist = p.distance_est;
                                        let is_pod = p.is_pod_mate;
                                        view! {
                                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem; background: var(--surface-low); border-radius: 4px">
                                                <div style="display: flex; flex-direction: column">
                                                    <span style="font-size: 0.8rem; font-weight: 600">
                                                        {if is_pod { "\u{1F331} " } else { "" }} {id}
                                                    </span>
                                                    <span style="font-size: 0.65rem; color: var(--text-tertiary)">{dist} "km \u{00B7} " {rssi} "dBm"</span>
                                                </div>
                                                {if is_pod {
                                                    view! { <button style="font-size: 0.65rem; padding: 0.2rem 0.4rem; background: var(--primary); color: white; border: none; border-radius: 3px; cursor: pointer">"Endorse"</button> }.into_any()
                                                } else {
                                                    view! { <span style="font-size: 0.65rem; color: var(--text-tertiary)">"Connect"</span> }.into_any()
                                                }}
                                            </div>
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>
                            }.into_any()
                        }}
                    </div>
                }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }}
        </div>
    }
}
