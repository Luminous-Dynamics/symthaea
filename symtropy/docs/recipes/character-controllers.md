# Character Controllers

Character controllers in Symtropy use `PhysicsBody` components with `Kinematic` or `Dynamic` body types.

## Kinematic Controller
Best for platformers or top-down games where you want precise control over movement and don't want the character to be pushed around by forces.

```rust
fn move_character(
    mut query: Query<(&mut PhysicsBody, &mut Transform)>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    for (mut body, mut transform) in query.iter_mut() {
        if keys.pressed(KeyCode::KeyW) {
            transform.translation.y += 0.1;
        }
    }
}
```

## Dynamic Controller
Best for physics-heavy games where you want the character to interact naturally with the world.

```rust
fn jump(
    mut query: Query<&mut PhysicsBody>,
    keys: Res<ButtonInput<KeyCode>>,
    mut physics: ResMut<SymtropyPhysics<3>>,
) {
    for body in query.iter_mut() {
        if keys.just_pressed(KeyCode::Space) {
            physics.world.body_mut(body.handle).unwrap().apply_force(SVector::from([0.0, 500.0, 0.0]));
        }
    }
}
```
