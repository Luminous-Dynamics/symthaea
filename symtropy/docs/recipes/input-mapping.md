# Input Mapping

Symtropy uses Bevy's built-in `ButtonInput` and `Axis` resources for input mapping.

## Keyboard Mapping
```rust
fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut physics_query: Query<&mut PhysicsBody>,
) {
    if keys.just_pressed(KeyCode::KeyE) {
        // Interact logic
    }
}
```

## Gamepad Mapping
```rust
fn handle_gamepad(
    gamepads: Res<Gamepads>,
    axes: Res<Axis<GamepadAxis>>,
    buttons: Res<ButtonInput<GamepadButton>>,
) {
    for gamepad in gamepads.iter() {
        let left_stick_x = axes.get(GamepadAxis {
            gamepad,
            axis_type: GamepadAxisType::LeftStickX,
        }).unwrap();
        // Move logic
    }
}
```
