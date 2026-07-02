// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Craft Integration — Bridges Praxis learning with Craft doing.
//! Pulls vitality metrics and work history from the Mycelix Craft DNA.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

/// Vitality of a skill based on its usage in real-world Craft projects.
/// Uses Ebbinghaus forgetting curve combined with recent project activity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct SkillVitality {
    /// Vitality level (0-1000 permille)
    pub level_permille: u16,
    /// Days since last professional application in Craft
    pub days_since_last_applied: u32,
    /// Number of professional projects using this skill
    pub professional_project_count: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct CraftContext {
    /// Signal to get vitality for a specific skill node
    pub get_vitality: Callback<String, SkillVitality>,
}

/// Provide craft context to the application
pub fn provide_craft_context() {
    // Mock vitality logic for now — in production this calls the CraftBridge
    let get_vitality = Callback::new(move |node_id: String| {
        // Deterministic mock based on node_id for UI consistency
        let len = node_id.len() as u32;
        if len % 3 == 0 {
            SkillVitality {
                level_permille: 850,
                days_since_last_applied: 5,
                professional_project_count: 3,
            }
        } else if len % 3 == 1 {
            SkillVitality {
                level_permille: 420, // Fading
                days_since_last_applied: 120,
                professional_project_count: 1,
            }
        } else {
            SkillVitality {
                level_permille: 0, // Never applied in Craft
                days_since_last_applied: 0,
                professional_project_count: 0,
            }
        }
    });

    provide_context(CraftContext {
        get_vitality,
    });
}

pub fn use_craft() -> CraftContext {
    expect_context::<CraftContext>()
}

/// A "Fading Leaf" indicator for skill vitality
#[component]
pub fn VitalityIndicator(node_id: String) -> impl IntoView {
    let craft = use_craft();
    let vitality = move || (craft.get_vitality)(node_id.clone());
    
    view! {
        {move || {
            let v = vitality();
            if v.level_permille > 0 {
                let color = if v.level_permille > 700 { "var(--success)" }
                            else if v.level_permille > 300 { "var(--warning)" }
                            else { "var(--error)" };
                let icon = if v.level_permille > 700 { "\u{1F33F}" } // Fresh leaf
                           else { "\u{1F342}" }; // Falling leaf
                
                view! {
                    <span 
                        class="vitality-badge" 
                        title=format!("Craft Vitality: {}/1000 (Applied {} days ago)", v.level_permille, v.days_since_last_applied)
                        style=format!("color: {}; font-size: 0.8rem; display: inline-flex; align-items: center; gap: 0.2rem", color)
                    >
                        {icon}
                        <span style="font-weight: 700">{v.level_permille / 10}"%"</span>
                    </span>
                }.into_any()
            } else {
                view! { <span title="Not yet applied in professional Craft projects" style="opacity: 0.3; font-size: 0.8rem">"\u{1F331}"</span> }.into_any()
            }
        }}
    }
}
