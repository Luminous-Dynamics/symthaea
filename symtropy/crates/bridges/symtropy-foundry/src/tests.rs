// SPDX-License-Identifier: AGPL-3.0-or-later

#[cfg(test)]
mod tests {
    use crate::*;
    use bevy::prelude::*;
    use bevy_camera::visibility::VisibilityRange;
    use symtropy_bevy_core::{BevyPhysics, PhysicsBody};
    use symtropy_math::Point;

    #[test]
    fn test_parse_lod_index() {
        assert_eq!(parse_lod_index("MyMesh_LOD0"), Some(0));
        assert_eq!(parse_lod_index("Tree_LOD1.001"), Some(1));
        assert_eq!(parse_lod_index("Rock_LOD2_Extra"), Some(2));
        assert_eq!(parse_lod_index("House"), None);
    }

    #[test]
    fn test_foundry_plugin_registers_system() {
        let mut app = App::new();
        app.add_plugins(FoundryPlugin::default());
    }

    #[test]
    fn test_collision_tag_processing() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin, FoundryPlugin::default()));

        app.insert_resource(BevyPhysics::<3>::default());

        let entity = app
            .world_mut()
            .spawn((
                Name::new("Wall_COLLISION"),
                GlobalTransform::from_translation(Vec3::new(1.0, 2.0, 3.0)),
            ))
            .id();

        app.update();

        let world = app.world();
        assert!(world.entity(entity).contains::<PhysicsBody>());
        assert!(world.entity(entity).contains::<FoundryProcessed>());
        let visibility = world.entity(entity).get::<Visibility>().unwrap();
        assert_eq!(*visibility, Visibility::Hidden);

        let physics = world.resource::<BevyPhysics<3>>();
        assert_eq!(physics.world.bodies.len(), 1);
        let body = &physics.world.bodies[0];
        assert_eq!(body.position(), Point::new([1.0, 2.0, 3.0]).0);
    }

    #[test]
    fn test_lod_tag_processing() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin, FoundryPlugin::default()));

        let entity = app
            .world_mut()
            .spawn((Name::new("Mesh_LOD0"), GlobalTransform::default()))
            .id();

        app.update();

        let world = app.world();
        assert!(world.entity(entity).contains::<VisibilityRange>());
        assert!(world.entity(entity).contains::<FoundryProcessed>());
        let range = world.entity(entity).get::<VisibilityRange>().unwrap();
        assert_eq!(range.start_margin.start, 0.0);
        assert_eq!(range.end_margin.end, 20.0);
    }
}
