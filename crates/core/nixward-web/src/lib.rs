use gloo_net::http::Request;
use leptos::*;
use serde::{Deserialize, Serialize};

// Re-export DaemonSnapshot mapping for Leptos components
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WebSnapshot {
    pub free_energy: f64,
    pub anomaly_volatility_ema: f64,
    pub risk_aversion: f64,
    pub curiosity_weight: f64,
    pub causal_learning_rate: f64,
    pub is_surprised: bool,
    pub anomaly_count: u64,
    pub observation_count: u64,
}

#[component]
pub fn NixwardDashboard() -> impl IntoView {
    let (snapshot, set_snapshot) = create_signal(WebSnapshot::default());
    let (status, set_status) = create_signal("Offline".to_string());

    // Periodically fetch the latest state from the daemon
    create_effect(move |_| {
        let fetch_state = move || {
            spawn_local(async move {
                match Request::get("http://localhost:9090/state").send().await {
                    Ok(resp) => {
                        if let Ok(snap) = resp.json::<WebSnapshot>().await {
                            set_snapshot.set(snap);
                            set_status.set("Connected".to_string());
                        }
                    }
                    Err(_) => {
                        set_status.set("Disconnected".to_string());
                    }
                }
            });
        };

        // Poll immediately and then every 2 seconds
        fetch_state();
        let handle = set_interval_with_handle(fetch_state, std::time::Duration::from_secs(2));

        on_cleanup(move || {
            if let Ok(h) = handle {
                h.clear();
            }
        });
    });

    let mood = move || {
        let snap = snapshot.get();
        if snap.anomaly_volatility_ema > 0.4 {
            ("💥 Turbulent / Surprised", "#ef4444")
        } else if snap.risk_aversion > 0.5 {
            ("🛡️ Defensive / Risk Averse", "#eab308")
        } else if snap.curiosity_weight > 0.4 {
            ("🔍 Curious / Exploring", "#22c55e")
        } else {
            ("🧘 Calm / Alert", "#06b6d4")
        }
    };

    view! {
        <div style="padding: 24px; max-width: 1200px; margin: 0 auto;">
            // Header
            <header style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1f2937; padding-bottom: 16px; margin-bottom: 24px;">
                <div>
                    <h1 style="margin: 0; font-size: 28px; font-weight: 800; background: linear-gradient(to right, #ec4899, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        "Nix-Mind Dashboard"
                    </h1>
                    <p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 14px;">
                        "Cognitive Reliability & Autonomic Self-Healing on NixOS"
                    </p>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style=move || format!("display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: {};", if status.get() == "Connected" { "#10b981" } else { "#ef4444" })></span>
                    <span style="font-size: 14px; font-weight: 600; color: #d1d5db;">{status}</span>
                </div>
            </header>

            // Main Layout
            <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 24px;">
                // Left Column: Free Energy & Core State
                <div style="background-color: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                    <h2 style="margin: 0 0 16px 0; font-size: 20px; font-weight: 700; color: #f3f4f6;">"System Cognition"</h2>

                    <div style="margin-bottom: 24px; padding: 16px; background-color: #09090b; border-radius: 8px; border: 1px solid #18181b;">
                        <span style="font-size: 14px; color: #a1a1aa; font-weight: 500;">"Expected Free Energy"</span>
                        <div style="font-size: 36px; font-weight: 900; color: #f43f5e; font-family: monospace; margin-top: 4px;">
                            {move || format!("{:.4}", snapshot.get().free_energy)}
                        </div>
                    </div>

                    // Metrics Grid
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                        <div style="padding: 16px; background-color: #27272a; border-radius: 8px;">
                            <div style="font-size: 12px; color: #a1a1aa;">"Observations"</div>
                            <div style="font-size: 20px; font-weight: 700; color: #e4e4e7; margin-top: 4px;">
                                {move || snapshot.get().observation_count}
                            </div>
                        </div>
                        <div style="padding: 16px; background-color: #27272a; border-radius: 8px;">
                            <div style="font-size: 12px; color: #a1a1aa;">"Anomalies"</div>
                            <div style="font-size: 20px; font-weight: 700; color: #e4e4e7; margin-top: 4px;">
                                {move || snapshot.get().anomaly_count}
                            </div>
                        </div>
                    </div>
                </div>

                // Right Column: Allostatic Mood & Volatility Gauges
                <div style="background-color: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 24px; display: flex; flex-direction: column; gap: 20px;">
                    <div>
                        <h2 style="margin: 0 0 16px 0; font-size: 20px; font-weight: 700; color: #f3f4f6;">"Allostatic Mood"</h2>
                        <div style=move || format!("padding: 12px 16px; border-radius: 8px; font-weight: 700; text-align: center; background-color: {}20; color: {}; border: 1px solid {};", mood().1, mood().1, mood().1)>
                            {move || mood().0}
                        </div>
                    </div>

                    // Gauges List
                    <div style="display: flex; flex-direction: column; gap: 14px;">
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 13px; color: #a1a1aa; margin-bottom: 6px;">
                                <span>"Volatility (EMA)"</span>
                                <span style="font-family: monospace;">{move || format!("{:.2}%", snapshot.get().anomaly_volatility_ema * 100.0)}</span>
                            </div>
                            <div style="height: 8px; background-color: #27272a; border-radius: 4px; overflow: hidden;">
                                <div style=move || format!("height: 100%; background-color: #ef4444; width: {}%; transition: width 0.3s ease;", snapshot.get().anomaly_volatility_ema * 100.0)></div>
                            </div>
                        </div>

                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 13px; color: #a1a1aa; margin-bottom: 6px;">
                                <span>"Risk Aversion"</span>
                                <span style="font-family: monospace;">{move || format!("{:.2}%", snapshot.get().risk_aversion * 100.0)}</span>
                            </div>
                            <div style="height: 8px; background-color: #27272a; border-radius: 4px; overflow: hidden;">
                                <div style=move || format!("height: 100%; background-color: #eab308; width: {}%; transition: width 0.3s ease;", snapshot.get().risk_aversion * 100.0)></div>
                            </div>
                        </div>

                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 13px; color: #a1a1aa; margin-bottom: 6px;">
                                <span>"Curiosity explore weight"</span>
                                <span style="font-family: monospace;">{move || format!("{:.2}%", snapshot.get().curiosity_weight * 100.0)}</span>
                            </div>
                            <div style="height: 8px; background-color: #27272a; border-radius: 4px; overflow: hidden;">
                                <div style=move || format!("height: 100%; background-color: #22c55e; width: {}%; transition: width 0.3s ease;", snapshot.get().curiosity_weight * 100.0)></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    }
}
