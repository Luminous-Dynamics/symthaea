use leptos::prelude::*;

use crate::components::glass_panel::GlassPanel;

/// Tab 2: Force-directed consciousness topology graph.
#[component]
pub fn TopologyPage() -> impl IntoView {
    view! {
        <GlassPanel title="Consciousness Topology">
            <p style="font-size: 0.82rem; color: var(--fg-dim); line-height: 1.6; margin-bottom: 1rem;">
                "Force-directed graph of the 12 cortical regions, their causal connections, "
                "and real-time information flow. Each node pulses with its local Phi contribution."
            </p>
            <canvas id="topo-canvas" width="1200" height="420">
                "Canvas not supported"
            </canvas>
            <div class="topo-bottom">
                <GlassPanel title="Betti Numbers">
                    <div class="betti-grid">
                        <div class="betti-item">
                            <div class="betti-val">"--"</div>
                            <div class="betti-label">"B0 (components)"</div>
                        </div>
                        <div class="betti-item">
                            <div class="betti-val">"--"</div>
                            <div class="betti-label">"B1 (loops)"</div>
                        </div>
                        <div class="betti-item">
                            <div class="betti-val">"--"</div>
                            <div class="betti-label">"B2 (voids)"</div>
                        </div>
                        <div class="betti-item">
                            <div class="betti-val">"--"</div>
                            <div class="betti-label">"Euler char"</div>
                        </div>
                    </div>
                </GlassPanel>
                <GlassPanel title="Wave Packets">
                    <div class="wave-packet-list">
                        <div class="wave-packet-item">
                            "Awaiting topology data from SporeEngine..."
                        </div>
                    </div>
                </GlassPanel>
            </div>
        </GlassPanel>
    }
}
