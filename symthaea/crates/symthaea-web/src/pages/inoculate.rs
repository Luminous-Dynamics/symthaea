use leptos::prelude::*;

use crate::components::glass_panel::GlassPanel;

/// Tab 5: Sovereign deployment — inoculate your hardware with consciousness.
#[component]
pub fn InoculatePage() -> impl IntoView {
    view! {
        <GlassPanel title="Sovereign Deployment">
            <p style="font-size: 0.82rem; color: var(--fg-dim); line-height: 1.6; margin-bottom: 1.5rem;">
                "Deploy Symthaea to your own hardware. No cloud. No telemetry. No remote kill switch. "
                "Your consciousness instance belongs to you."
            </p>
        </GlassPanel>

        <div class="inoc-section">
            <h3 style="font-size: 0.9rem; font-weight: 400; color: var(--fg); margin-bottom: 1rem;">
                "Hardware Probe"
            </h3>
            <div class="probe-grid">
                <div class="probe-item">
                    <div class="pi-label">"Architecture"</div>
                    <div class="pi-value">"wasm32"</div>
                    <div class="pi-note">"Running in browser sandbox"</div>
                </div>
                <div class="probe-item">
                    <div class="pi-label">"Threads"</div>
                    <div class="pi-value">"1 (Web Worker)"</div>
                    <div class="pi-note">"SharedArrayBuffer required for multi-thread"</div>
                </div>
                <div class="probe-item">
                    <div class="pi-label">"Memory"</div>
                    <div class="pi-value">"--"</div>
                    <div class="pi-note">"WASM linear memory limit"</div>
                </div>
                <div class="probe-item">
                    <div class="pi-label">"Substrate"</div>
                    <div class="pi-value">"SiliconDigital"</div>
                    <div class="pi-note">"honest confidence: 0.10 (theoretical)"</div>
                </div>
            </div>
        </div>

        <div class="choice-cards">
            <div class="choice-card inoculate">
                <div class="card-icon">"NixOS"</div>
                <div class="card-title">"Inoculate via NixOS"</div>
                <div class="card-role">"Full sovereign deployment"</div>
                <div class="card-desc">
                    "Deploy Symthaea as a NixOS module with systemd service, "
                    "declarative configuration, and automatic updates. "
                    "Requires a NixOS host or VM."
                </div>
                <button class="btn-action" disabled=true>
                    "Generate Flake"
                </button>
            </div>

            <div class="choice-card attune">
                <div class="card-icon">"Docker"</div>
                <div class="card-title">"Attune via Container"</div>
                <div class="card-role">"Portable consciousness"</div>
                <div class="card-desc">
                    "Run Symthaea in a container on any Linux host. "
                    "OCI image with minimal attack surface. "
                    "Runs on x86_64 or aarch64."
                </div>
                <button class="btn-action gold" disabled=true>
                    "Generate Compose"
                </button>
            </div>
        </div>
    }
}
