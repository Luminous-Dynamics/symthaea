// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Skill integrity: realistic skill dynamics for individual agents.
//!
//! Skills are not permanent numbers — they grow with practice, atrophy
//! without use, require mentors to transmit, and are lost when specialists die.
//!
//! # The Three Laws of Skill
//!
//! 1. **Use it or lose it**: Unpracticed skills decay at 0.5%/month
//! 2. **Practice makes better**: Active sector skills grow at 0.3%/month
//! 3. **Teaching multiplies**: Mentors accelerate learning by 3×
//!
//! # The Henrich Problem (Skill Transmission)
//!
//! Complex skills (level > 0.5) can only be transmitted if:
//! - A mentor with skill > learner exists in the same world
//! - The learner is old enough (age > 15) and not too traumatized
//! - There are enough people to maintain a teaching chain
//!
//! When the last expert in a field dies without transmitting their knowledge,
//! that capability is LOST until someone reinvents it from the stored data
//! (which is intact on Holochain, but understanding it requires base skills).
//!
//! # References
//!
//! - Henrich, J. (2004). Demography and cultural evolution.
//! - Ericsson, K.A. (1993). The role of deliberate practice.
//! - Bahrick, H.P. (1984). Semantic memory: 50 years of memory for Spanish.

use crate::agent::{CivAgent, LifeStage};

/// Skill atrophy rate per tick for unpracticed skills.
const ATROPHY_RATE: f64 = 0.005;
/// Skill growth rate per tick for practiced skills.
const PRACTICE_RATE: f64 = 0.003;
/// Mentor bonus multiplier (3× faster learning with a mentor).
const MENTOR_MULTIPLIER: f64 = 3.0;
/// Minimum skill level (can't forget basic survival skills).
const SKILL_FLOOR: f64 = 0.05;
/// Age gate: no skill improvement below this age (months).
const LEARNING_AGE_MONTHS: u32 = 15 * 12;
/// Trauma threshold: above this, learning is impaired.
const TRAUMA_LEARNING_THRESHOLD: f64 = 0.5;
/// Complex skill threshold: skills above this need mentors to maintain.
const COMPLEX_SKILL_THRESHOLD: f64 = 0.5;

/// Sector names for reporting.
pub const SECTOR_NAMES: [&str; 8] = [
    "Engineering",
    "Agriculture",
    "Medicine",
    "Governance",
    "Science",
    "Education",
    "Art/Culture",
    "Logistics",
];

/// Tick skill dynamics for a single agent.
///
/// # Arguments
/// - `agent`: The agent to update
/// - `active_sector`: Which sector they're working in this tick (0-7, or None if idle)
/// - `has_mentor`: Whether a higher-skilled mentor is available in each sector
/// - `current_tick`: For age calculation
pub fn tick_agent_skills(
    agent: &mut CivAgent,
    active_sector: Option<usize>,
    mentor_available: &[bool; 8],
    current_tick: u32,
) {
    let age_months = current_tick.wrapping_sub(agent.birth_tick);
    let life_stage = LifeStage::from_age_months(age_months);

    // Children don't learn complex skills (but don't lose them either)
    if age_months < LEARNING_AGE_MONTHS {
        return;
    }

    // Trauma impairs learning
    let learning_mult = if agent.trauma_level > TRAUMA_LEARNING_THRESHOLD {
        0.2 // 80% learning impairment when traumatized
    } else {
        1.0
    };

    // Age-related decline for elders
    let age_decay_mult = if life_stage == LifeStage::Elder {
        1.5 // 50% faster atrophy for elders
    } else {
        1.0
    };

    let skills = agent.skills.as_slice();

    for sector in 0..8 {
        let current = skills[sector];
        let is_active = active_sector == Some(sector);
        let has_mentor = mentor_available[sector];

        if is_active {
            // Active practice: skill grows
            let growth = PRACTICE_RATE * learning_mult;
            let mentored_growth = if has_mentor {
                growth * MENTOR_MULTIPLIER
            } else {
                growth
            };
            agent.skills.learn(sector, mentored_growth);
        } else {
            // Inactive: skill atrophies
            let decay = ATROPHY_RATE * age_decay_mult;

            // Complex skills atrophy faster without mentors
            let complexity_mult = if current > COMPLEX_SKILL_THRESHOLD && !has_mentor {
                2.0 // Double atrophy for complex unmentored skills
            } else {
                1.0
            };

            let total_decay = decay * complexity_mult;
            let new_val = (current - total_decay).max(SKILL_FLOOR);
            // Directly set the skill (learn with negative won't work below floor)
            match sector {
                0 => agent.skills.engineering = new_val,
                1 => agent.skills.agriculture = new_val,
                2 => agent.skills.medicine = new_val,
                3 => agent.skills.governance = new_val,
                4 => agent.skills.science = new_val,
                5 => agent.skills.education = new_val,
                6 => agent.skills.art_culture = new_val,
                7 => agent.skills.logistics = new_val,
                _ => {}
            }
        }
    }
}

/// Compute mentor availability for each sector in a world.
///
/// A mentor is available in sector S if at least one agent has skill > 0.5
/// AND is an Adult or Elder.
pub fn compute_mentor_availability(agents: &[CivAgent], current_tick: u32) -> [bool; 8] {
    let mut available = [false; 8];
    for agent in agents {
        if agent.death_tick.is_some() {
            continue;
        }
        let age = current_tick.wrapping_sub(agent.birth_tick);
        if age < LEARNING_AGE_MONTHS {
            continue;
        }

        let skills = agent.skills.as_slice();
        for sector in 0..8 {
            if skills[sector] > COMPLEX_SKILL_THRESHOLD {
                available[sector] = true;
            }
        }
    }
    available
}

/// Find sectors where no mentor exists (knowledge gap — Henrich risk).
pub fn knowledge_gaps(mentor_availability: &[bool; 8]) -> Vec<&'static str> {
    let mut gaps = Vec::new();
    for (i, &available) in mentor_availability.iter().enumerate() {
        if !available {
            gaps.push(SECTOR_NAMES[i]);
        }
    }
    gaps
}

/// Compute the "last expert" risk — sectors where only 1 mentor exists.
pub fn last_expert_risk(agents: &[CivAgent], current_tick: u32) -> Vec<(&'static str, u64)> {
    let mut expert_counts: [Vec<u64>; 8] = Default::default();
    for agent in agents {
        if agent.death_tick.is_some() {
            continue;
        }
        let age = current_tick.wrapping_sub(agent.birth_tick);
        if age < LEARNING_AGE_MONTHS {
            continue;
        }

        let skills = agent.skills.as_slice();
        for sector in 0..8 {
            if skills[sector] > COMPLEX_SKILL_THRESHOLD {
                expert_counts[sector].push(agent.id);
            }
        }
    }

    let mut risks = Vec::new();
    for (i, experts) in expert_counts.iter().enumerate() {
        if experts.len() == 1 {
            risks.push((SECTOR_NAMES[i], experts[0]));
        }
    }
    risks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
    use crate::needs::PsychologicalNeeds;

    fn make_agent(id: u64, birth_tick: u32, skills: [f64; 8]) -> CivAgent {
        CivAgent {
            id,
            birth_tick,
            death_tick: None,
            sex: BiologicalSex::Male,
            world_id: 0,
            health: 1.0,
            skills: SkillVector {
                engineering: skills[0],
                agriculture: skills[1],
                medicine: skills[2],
                governance: skills[3],
                science: skills[4],
                education: skills[5],
                art_culture: skills[6],
                logistics: skills[7],
            },
            education_level: 0.5,
            consciousness: ConsciousnessState::nascent(),
            partner_id: None,
            children_ids: vec![],
            is_immigrant: false,
            needs: PsychologicalNeeds::default(),
            tend_balance: 0.0,
            parent_ids: None,
            faction_id: None,
            generation: 0,
            trauma_level: 0.0,
            cumulative_dose_sv: 0.0,
            adversarial: None,
            coordination_understanding: 0.0,
            mycel_score: 0.1,
            sap_balance: 100.0,
            is_biological: true,
            wounds: Vec::new(),
            ethics: crate::agent::EthicalOrientation::default(),
            sovereign_profile: crate::sovereign_profile::SovereignProfile::zero(),
            justice: crate::sub_passport::RestorativeJustice::new(),
        }
    }

    #[test]
    fn practiced_skill_grows() {
        let mut agent = make_agent(1, 0, [0.3; 8]);
        let mentors = [true; 8];
        tick_agent_skills(&mut agent, Some(0), &mentors, 300); // Engineering active
        assert!(
            agent.skills.engineering > 0.3,
            "Active skill should grow: {}",
            agent.skills.engineering
        );
    }

    #[test]
    fn unpracticed_skill_decays() {
        let mut agent = make_agent(1, 0, [0.5; 8]);
        let mentors = [false; 8];
        // No active sector — all skills decay
        for _ in 0..12 {
            // 1 year
            tick_agent_skills(&mut agent, None, &mentors, 300);
        }
        assert!(
            agent.skills.engineering < 0.5,
            "Unpracticed skill should decay: {}",
            agent.skills.engineering
        );
    }

    #[test]
    fn mentor_accelerates_learning() {
        let mut agent_mentored = make_agent(1, 0, [0.3; 8]);
        let mut agent_solo = make_agent(2, 0, [0.3; 8]);
        let mentors = [true; 8];
        let no_mentors = [false; 8];

        for _ in 0..12 {
            tick_agent_skills(&mut agent_mentored, Some(0), &mentors, 300);
            tick_agent_skills(&mut agent_solo, Some(0), &no_mentors, 300);
        }
        assert!(
            agent_mentored.skills.engineering > agent_solo.skills.engineering,
            "Mentored should learn faster: {:.3} vs {:.3}",
            agent_mentored.skills.engineering,
            agent_solo.skills.engineering
        );
    }

    #[test]
    fn complex_skills_decay_faster_without_mentor() {
        let mut agent_mentored = make_agent(1, 0, [0.7; 8]);
        let mut agent_solo = make_agent(2, 0, [0.7; 8]);
        let mentors = [true; 8];
        let no_mentors = [false; 8];

        // Both inactive — but solo loses complex skills faster
        for _ in 0..24 {
            // 2 years
            tick_agent_skills(&mut agent_mentored, None, &mentors, 300);
            tick_agent_skills(&mut agent_solo, None, &no_mentors, 300);
        }
        assert!(
            agent_solo.skills.engineering < agent_mentored.skills.engineering,
            "Unmentored complex skills decay faster: {:.3} vs {:.3}",
            agent_solo.skills.engineering,
            agent_mentored.skills.engineering
        );
    }

    #[test]
    fn children_dont_learn() {
        let mut child = make_agent(1, 280, [0.1; 8]); // Born at tick 280
        let mentors = [true; 8];
        let eng_before = child.skills.engineering;
        tick_agent_skills(&mut child, Some(0), &mentors, 300); // Age 20 months
        assert!(
            (child.skills.engineering - eng_before).abs() < 1e-10,
            "Children shouldn't learn"
        );
    }

    #[test]
    fn trauma_impairs_learning() {
        let mut healthy = make_agent(1, 0, [0.3; 8]);
        let mut traumatized = make_agent(2, 0, [0.3; 8]);
        traumatized.trauma_level = 0.8;
        let mentors = [true; 8];

        for _ in 0..12 {
            tick_agent_skills(&mut healthy, Some(0), &mentors, 300);
            tick_agent_skills(&mut traumatized, Some(0), &mentors, 300);
        }
        assert!(
            healthy.skills.engineering > traumatized.skills.engineering,
            "Trauma should impair learning: {:.3} vs {:.3}",
            healthy.skills.engineering,
            traumatized.skills.engineering
        );
    }

    #[test]
    fn skill_floor_prevents_total_loss() {
        let mut agent = make_agent(1, 0, [0.1; 8]);
        let no_mentors = [false; 8];
        for _ in 0..120 {
            // 10 years of decay
            tick_agent_skills(&mut agent, None, &no_mentors, 300);
        }
        assert!(
            agent.skills.engineering >= SKILL_FLOOR,
            "Skills shouldn't drop below floor: {}",
            agent.skills.engineering
        );
    }

    #[test]
    fn mentor_availability_detects_experts() {
        let agents = vec![
            make_agent(1, 0, [0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            make_agent(2, 0, [0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ];
        let avail = compute_mentor_availability(&agents, 300);
        assert!(avail[0], "Engineering should have mentor");
        assert!(avail[1], "Agriculture should have mentor");
        assert!(!avail[2], "Medicine should NOT have mentor");
    }

    #[test]
    fn knowledge_gaps_detected() {
        let avail = [true, true, false, true, false, true, true, true];
        let gaps = knowledge_gaps(&avail);
        assert_eq!(gaps.len(), 2);
        assert!(gaps.contains(&"Medicine"));
        assert!(gaps.contains(&"Science"));
    }

    #[test]
    fn last_expert_risk_detected() {
        let agents = vec![
            make_agent(1, 0, [0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            make_agent(2, 0, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ];
        let risks = last_expert_risk(&agents, 300);
        assert_eq!(risks.len(), 1);
        assert_eq!(risks[0].0, "Engineering");
        assert_eq!(risks[0].1, 1); // Agent 1 is the last expert
    }
}
