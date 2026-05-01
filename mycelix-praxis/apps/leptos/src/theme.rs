// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Somatic Theme Engine — UI Adaptation based on Nervous System Regulation.

use leptos::prelude::*;
use crate::curriculum::{use_progress, ProgressStatus};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SomaticState {
    Regulated,
    HyperVigilant,
    Exhausted,
}

#[component]
pub fn SomaticThemeHandler<F, IV>(
    children: F,
) -> impl IntoView 
where 
    F: Fn() -> IV + 'static,
    IV: IntoView + 'static,
{
    let progress = use_progress();
    
    // Derived: Infer somatic state from recent activity and somatic milestones
    let somatic_state = Memo::new(move |_| {
        let p = progress.get();
        let mastered_somatic = p.nodes.values()
            .filter(|n| n.status == ProgressStatus::Mastered && n.mastery_permille > 900)
            .count();
            
        if mastered_somatic > 5 { SomaticState::Regulated }
        else { SomaticState::HyperVigilant }
    });

    let theme_class = move || match somatic_state.get() {
        SomaticState::Regulated => "soma-zen",
        SomaticState::HyperVigilant => "soma-alert",
        SomaticState::Exhausted => "soma-recovery",
    };

    view! {
        <div class=theme_class>
            {children()}
        </div>
    }
}
