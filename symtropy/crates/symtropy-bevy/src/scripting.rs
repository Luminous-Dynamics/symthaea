// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scripting integration for Symtropy using Rhai.

use crate::plugin::SymtropyPhysics;
use bevy::prelude::*;
use rhai::{Engine, Scope, AST};

/// Component for entities with attached Rhai scripts.
#[derive(Component, Debug, Clone, Default, Reflect)]
#[reflect(Component)]
pub struct ScriptComponent {
    pub script_path: String,
    #[reflect(ignore)]
    pub ast: Option<AST>,
}

pub struct RoboticScriptingPlugin;

impl Plugin for RoboticScriptingPlugin {
    fn build(&self, app: &mut App) {
        let mut engine = Engine::new();

        // Register core Symtropy types to Rhai
        engine.register_type_with_name::<f64>("f64");

        app.insert_resource(RhaiEngine(engine))
            .register_type::<ScriptComponent>()
            .add_systems(
                Update,
                (
                    run_scripts_system::<2>,
                    run_scripts_system::<3>,
                    run_scripts_system::<4>,
                ),
            );
    }
}

#[derive(Resource)]
pub struct RhaiEngine(pub Engine);

/// Execute attached Rhai scripts, providing access to Φ and physics state.
pub fn run_scripts_system<const D: usize>(
    physics: Res<SymtropyPhysics<D>>,
    engine: Res<RhaiEngine>,
    query: Query<(&crate::PhysicsBody, &ScriptComponent)>,
) {
    for (body, script) in query.iter() {
        if let Some(ast) = &script.ast {
            let mut scope = Scope::new();

            // Provide context to the script
            let phi = physics.field.phi(body.handle);
            scope.push("phi", phi);

            // Execute the script
            let _ = engine.0.run_ast_with_scope(&mut scope, ast);
        }
    }
}
