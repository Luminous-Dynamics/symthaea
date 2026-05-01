// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scavenging: find NSM primitive fragments in the maze.
//!
//! Scavenged items are data-drives containing Natural Semantic Metalanguage
//! primitives. Players verify and combine them to unlock Harmony lore.

use bevy::prelude::*;
use rand::Rng;

use crate::components::Player;

/// A scavengeable item in the world.
#[derive(Component)]
pub struct ScavengeItem {
    /// Which NSM primitive this fragment contains.
    pub primitive: NsmPrimitive,
    /// Epistemic level: 0=unverified, 1=preliminary, 2=tested, 3=replicated, 4=consensus
    pub epistemic_level: u8,
    /// Whether the player has picked this up.
    pub collected: bool,
}

/// Natural Semantic Metalanguage primitives — the building blocks of meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NsmPrimitive {
    // Substantives
    I,
    You,
    Someone,
    Something,
    People,
    Body,
    // Determiners
    This,
    Same,
    Other,
    // Quantifiers
    One,
    Two,
    Some,
    All,
    Much,
    // Evaluators
    Good,
    Bad,
    Big,
    Small,
    // Descriptors
    Think,
    Know,
    Want,
    Feel,
    See,
    Hear,
    // Speech
    Say,
    Words,
    True,
    // Actions & Events
    Do,
    Happen,
    Move,
    // Existence & Possession
    There,
    Be,
    Have,
    // Life & Death
    Live,
    Die,
    // Time
    When,
    Now,
    Before,
    After,
    Long,
    Short,
    // Space
    Where,
    Here,
    Above,
    Below,
    Far,
    Near,
    Side,
    Inside,
    // Logical
    Not,
    Maybe,
    Can,
    Because,
    If,
}

impl NsmPrimitive {
    /// Which Harmony this primitive contributes to when verified.
    pub fn harmony_index(&self) -> usize {
        match self {
            Self::I | Self::Body | Self::Be | Self::Live => 0, // ResonantCoherence (self-unity)
            Self::Someone | Self::People | Self::Feel | Self::Good => 1, // PanSentientFlourishing (care)
            Self::Know | Self::Think | Self::True | Self::Because => 2, // IntegralWisdom (understanding)
            Self::Do | Self::Happen | Self::Move | Self::Can => 3, // InfinitePlay (creative tension)
            Self::You | Self::Other | Self::Same | Self::Here => 4, // UniversalInterconnectedness
            Self::Have | Self::Some | Self::One | Self::Two => 5,  // SacredReciprocity
            Self::Before | Self::After | Self::Long | Self::Big => 6, // EvolutionaryProgression
            _ => 7, // SacredStillness (everything else → contemplation)
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::I => "I",
            Self::You => "YOU",
            Self::Someone => "SOMEONE",
            Self::Something => "SOMETHING",
            Self::People => "PEOPLE",
            Self::Body => "BODY",
            Self::This => "THIS",
            Self::Same => "SAME",
            Self::Other => "OTHER",
            Self::One => "ONE",
            Self::Two => "TWO",
            Self::Some => "SOME",
            Self::All => "ALL",
            Self::Much => "MUCH",
            Self::Good => "GOOD",
            Self::Bad => "BAD",
            Self::Big => "BIG",
            Self::Small => "SMALL",
            Self::Think => "THINK",
            Self::Know => "KNOW",
            Self::Want => "WANT",
            Self::Feel => "FEEL",
            Self::See => "SEE",
            Self::Hear => "HEAR",
            Self::Say => "SAY",
            Self::Words => "WORDS",
            Self::True => "TRUE",
            Self::Do => "DO",
            Self::Happen => "HAPPEN",
            Self::Move => "MOVE",
            Self::There => "THERE",
            Self::Be => "BE",
            Self::Have => "HAVE",
            Self::Live => "LIVE",
            Self::Die => "DIE",
            Self::When => "WHEN",
            Self::Now => "NOW",
            Self::Before => "BEFORE",
            Self::After => "AFTER",
            Self::Long => "LONG",
            Self::Short => "SHORT",
            Self::Where => "WHERE",
            Self::Here => "HERE",
            Self::Above => "ABOVE",
            Self::Below => "BELOW",
            Self::Far => "FAR",
            Self::Near => "NEAR",
            Self::Side => "SIDE",
            Self::Inside => "INSIDE",
            Self::Not => "NOT",
            Self::Maybe => "MAYBE",
            Self::Can => "CAN",
            Self::Because => "BECAUSE",
            Self::If => "IF",
        }
    }

    /// All 48 primitives.
    pub const ALL: [NsmPrimitive; 48] = [
        Self::I,
        Self::You,
        Self::Someone,
        Self::Something,
        Self::People,
        Self::Body,
        Self::This,
        Self::Same,
        Self::Other,
        Self::One,
        Self::Two,
        Self::Some,
        Self::All,
        Self::Much,
        Self::Good,
        Self::Bad,
        Self::Big,
        Self::Small,
        Self::Think,
        Self::Know,
        Self::Want,
        Self::Feel,
        Self::See,
        Self::Hear,
        Self::Say,
        Self::Words,
        Self::True,
        Self::Do,
        Self::Happen,
        Self::Move,
        Self::There,
        Self::Be,
        Self::Have,
        Self::Live,
        Self::Die,
        Self::When,
        Self::Now,
        Self::Before,
        Self::After,
        Self::Long,
        Self::Short,
        Self::Where,
        Self::Here,
        Self::Above,
        Self::Below,
        Self::Far,
        Self::Near,
        Self::Side,
    ];
}

/// Player's collected primitives.
#[derive(Resource, Default)]
pub struct CollectedPrimitives {
    pub primitives: Vec<NsmPrimitive>,
    /// Count of primitives per harmony [0..8]
    pub harmony_counts: [u32; 8],
}

impl CollectedPrimitives {
    pub fn add(&mut self, p: NsmPrimitive) {
        if !self.primitives.contains(&p) {
            self.harmony_counts[p.harmony_index()] += 1;
            self.primitives.push(p);
        }
    }

    pub fn total(&self) -> usize {
        self.primitives.len()
    }

    /// Check if a specific harmony has enough primitives to "unlock" its lore.
    pub fn harmony_unlocked(&self, harmony_idx: usize) -> bool {
        self.harmony_counts.get(harmony_idx).copied().unwrap_or(0) >= 3
    }
}

/// Spawn scavenge items in the dungeon at random walkable positions.
pub fn spawn_scavenge_items(
    mut commands: Commands,
    tiles: Query<(&Transform, &crate::components::Tile)>,
) {
    let mut rng = rand::thread_rng();
    let walkable_positions: Vec<Vec2> = tiles
        .iter()
        .filter(|(_, t)| t.walkable)
        .map(|(tf, _)| tf.translation.truncate())
        .collect();

    if walkable_positions.is_empty() {
        return;
    }

    // Place 8-12 items randomly
    let count = rng.gen_range(8..=12).min(walkable_positions.len());
    let mut used_indices = std::collections::HashSet::new();

    for i in 0..count {
        let mut pos_idx = rng.gen_range(0..walkable_positions.len());
        // Avoid placing two items on the same tile
        while used_indices.contains(&pos_idx) {
            pos_idx = (pos_idx + 1) % walkable_positions.len();
        }
        used_indices.insert(pos_idx);

        let pos = walkable_positions[pos_idx];
        let primitive = NsmPrimitive::ALL[i % NsmPrimitive::ALL.len()];

        commands.spawn((
            Sprite::from_color(Color::srgba(0.8, 0.6, 1.0, 0.7), Vec2::splat(12.0)),
            Transform::from_xyz(pos.x, pos.y, 1.5),
            ScavengeItem {
                primitive,
                epistemic_level: 0,
                collected: false,
            },
        ));
    }

    info!("Spawned {} scavenge items (NSM primitives)", count);
}

/// Player picks up nearby scavenge items.
pub fn scavenge_pickup_system(
    player: Query<&Transform, With<Player>>,
    mut items: Query<(&Transform, &mut ScavengeItem, &mut Sprite)>,
    mut collected: ResMut<CollectedPrimitives>,
    mut harmony: ResMut<super::harmonies::LocalHarmonyState>,
) {
    let Ok(player_tf) = player.single() else {
        return;
    };
    let player_pos = player_tf.translation.truncate();

    for (item_tf, mut item, mut sprite) in &mut items {
        if item.collected {
            continue;
        }

        let dist = player_pos.distance(item_tf.translation.truncate());
        if dist < 30.0 {
            item.collected = true;
            sprite.color = Color::srgba(0.0, 0.0, 0.0, 0.0); // hide

            collected.add(item.primitive);

            // Boost the corresponding harmony
            let idx = item.primitive.harmony_index();
            harmony.activations[idx] = (harmony.activations[idx] + 0.15).min(1.0);

            info!(
                "Collected NSM primitive: {} (+{}) [{}/48]",
                item.primitive.name(),
                super::harmonies::Harmony::ALL[idx].name(),
                collected.total(),
            );
        }
    }
}
