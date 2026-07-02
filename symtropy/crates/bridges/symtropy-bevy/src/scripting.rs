// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use bevy::prelude::*;
use rhai::{AST, Engine, Scope};
use symtropy_bevy_core::{BevyPhysics, PhysicsBody};

pub struct RoboticScriptingPlugin;

impl Plugin for RoboticScriptingPlugin {
    fn build(&self, app: &mut App) {
        let engine = Engine::new();
        // Register API here...

        app.insert_resource(RhaiEngine(engine))
            .add_systems(Update, run_scripts_system::<2>)
            .add_systems(Update, run_scripts_system::<3>)
            .add_systems(Update, run_scripts_system::<4>);
    }
}

#[derive(Resource)]
pub struct RhaiEngine(pub Engine);

#[derive(Component)]
pub struct ScriptComponent {
    pub ast: AST,
}

pub fn run_scripts_system<const D: usize>(
    engine: Res<RhaiEngine>,
    query: Query<(&PhysicsBody, &ScriptComponent)>,
    mut physics: ResMut<BevyPhysics<D>>,
) {
    for (body_comp, script) in query.iter() {
        if let Some(_body) = physics.world.body_mut(body_comp.handle) {
            let mut scope = Scope::new();
            // Inject body into scope...
            let _ = engine.0.run_ast_with_scope(&mut scope, &script.ast);
        }
    }
}
