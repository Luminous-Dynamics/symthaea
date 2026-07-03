// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! # symtropy-bevy-core
//!
//! Permissively-licensed Bevy plugin for [`symtropy-physics`] N-dimensional physics.
//! Zero AGPL dependencies.
//!
//! Use [`BevyPhysicsPlugin`] for drop-in physics with no coupling, or
//! [`BevyPhysicsCorePlugin`] + [`step_physics`] with your own
//! [`PhysicsCallback`] resource to couple any per-body metric to physics.
//!
//! For Φ (integrated information) coupling, see the AGPL sibling crate
//! `symtropy-bevy` — same API, adds `ConsciousnessField` integration.

use bevy::{input::mouse::AccumulatedMouseMotion, prelude::*, window::CursorGrabMode};
use nalgebra::SVector;
use symtropy_physics::{CollisionEvent, PhysicsCallback, PhysicsWorld, body::BodyHandle};

// --- First-person input intent spine ---

/// Sensitivity defaults for first-person mouse look.
pub const DEFAULT_MOUSE_SENSITIVITY: Vec2 = Vec2::new(0.0022, 0.0018);

/// Component marking an entity as controlled by the reusable first-person spine.
#[derive(Component, Debug, Clone, Copy)]
pub struct FirstPersonController {
    pub walk_speed: f32,
    pub field_deck_speed: f32,
    pub sprint_multiplier: f32,
    pub crouch_multiplier: f32,
    pub mouse_sensitivity: Vec2,
}

impl Default for FirstPersonController {
    fn default() -> Self {
        Self {
            walk_speed: 5.0,
            field_deck_speed: 1.0,
            sprint_multiplier: 1.65,
            crouch_multiplier: 0.45,
            mouse_sensitivity: DEFAULT_MOUSE_SENSITIVITY,
        }
    }
}

/// Centralized keyboard bindings for the default first-person control scheme.
#[derive(Resource, Debug, Clone)]
pub struct InputBindings {
    pub move_forward: KeyCode,
    pub move_back: KeyCode,
    pub move_left: KeyCode,
    pub move_right: KeyCode,
    pub sprint: KeyCode,
    pub crouch: KeyCode,
    pub jump: KeyCode,
    pub interact: KeyCode,
    pub focus_inspect: KeyCode,
    pub toggle_field_deck: KeyCode,
    pub quick_tool: KeyCode,
    pub repair_tool: KeyCode,
    pub build_mode: KeyCode,
    pub chronicle_panel: KeyCode,
    pub basin_map: KeyCode,
    pub scan_visualization: KeyCode,
    pub previous_scan_mode: KeyCode,
    pub next_scan_mode: KeyCode,
    pub pause_or_release: KeyCode,
    pub controls_overlay: KeyCode,
    pub command_palette: KeyCode,
    pub dev_scenario_panel: KeyCode,
}

impl Default for InputBindings {
    fn default() -> Self {
        Self {
            move_forward: KeyCode::KeyW,
            move_back: KeyCode::KeyS,
            move_left: KeyCode::KeyA,
            move_right: KeyCode::KeyD,
            sprint: KeyCode::ShiftLeft,
            crouch: KeyCode::ControlLeft,
            jump: KeyCode::Space,
            interact: KeyCode::KeyE,
            focus_inspect: KeyCode::KeyF,
            toggle_field_deck: KeyCode::Tab,
            quick_tool: KeyCode::KeyQ,
            repair_tool: KeyCode::KeyR,
            build_mode: KeyCode::KeyB,
            chronicle_panel: KeyCode::KeyC,
            basin_map: KeyCode::KeyM,
            scan_visualization: KeyCode::KeyV,
            previous_scan_mode: KeyCode::BracketLeft,
            next_scan_mode: KeyCode::BracketRight,
            pause_or_release: KeyCode::Escape,
            controls_overlay: KeyCode::F1,
            command_palette: KeyCode::KeyK,
            dev_scenario_panel: KeyCode::F10,
        }
    }
}

/// Per-frame named input state. Gameplay systems should consume this instead
/// of reading raw keys directly.
#[derive(Resource, Default, Debug, Clone)]
pub struct IntentFrame {
    pub movement: Vec2,
    pub look_delta: Vec2,
    pub pressed: Vec<InputIntent>,
    pub just_pressed: Vec<InputIntent>,
}

impl IntentFrame {
    pub fn clear(&mut self) {
        self.movement = Vec2::ZERO;
        self.look_delta = Vec2::ZERO;
        self.pressed.clear();
        self.just_pressed.clear();
    }

    pub fn pressed(&self, intent: InputIntent) -> bool {
        self.pressed.contains(&intent)
    }

    pub fn just_pressed(&self, intent: InputIntent) -> bool {
        self.just_pressed.contains(&intent)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputIntent {
    MoveForward,
    MoveBack,
    MoveLeft,
    MoveRight,
    Sprint,
    Crouch,
    Jump,
    Interact,
    FocusInspect,
    OpenFieldDeck,
    CycleFieldDeckModePrev,
    CycleFieldDeckModeNext,
    EquipToolSlot(u8),
    QuickTool,
    RepairTool,
    BuildMode,
    OpenChroniclePanel,
    OpenBasinMap,
    ToggleScanVisualization,
    PauseOrRelease,
    OpenControlsOverlay,
    OpenCommandPalette,
    OpenDevScenarioPanel,
    PauseSimulation,
    StepSimulation,
    ResetScenario,
    CaptureReplay,
}

#[derive(Resource, Debug, Clone)]
pub struct ControlsState {
    pub mode: ControlMode,
    pub selected_tool_slot: u8,
    pub scan_mode: ScanMode,
    pub show_controls: bool,
    pub show_dev_panel: bool,
    pub mouse_captured: bool,
}

impl Default for ControlsState {
    fn default() -> Self {
        Self {
            mode: ControlMode::FirstPerson,
            selected_tool_slot: 1,
            scan_mode: ScanMode::Ecology,
            show_controls: false,
            show_dev_panel: false,
            mouse_captured: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlMode {
    FirstPerson,
    FieldDeck,
    Console,
    DevScenario,
    Pause,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanMode {
    Infrastructure,
    Ecology,
    MachineDiagnostics,
    CivicClaims,
    NullSignalCorruption,
    RepairPreview,
    ChronicleEvidence,
}

impl ScanMode {
    pub const ALL: [ScanMode; 7] = [
        ScanMode::Infrastructure,
        ScanMode::Ecology,
        ScanMode::MachineDiagnostics,
        ScanMode::CivicClaims,
        ScanMode::NullSignalCorruption,
        ScanMode::RepairPreview,
        ScanMode::ChronicleEvidence,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ScanMode::Infrastructure => "Infrastructure",
            ScanMode::Ecology => "Ecology",
            ScanMode::MachineDiagnostics => "Machine Diagnostics",
            ScanMode::CivicClaims => "Civic Claims",
            ScanMode::NullSignalCorruption => "Null / Signal Corruption",
            ScanMode::RepairPreview => "Repair Preview",
            ScanMode::ChronicleEvidence => "Chronicle Evidence",
        }
    }

    pub fn offset(self, delta: isize) -> Self {
        let current = Self::ALL.iter().position(|mode| *mode == self).unwrap_or(0) as isize;
        let len = Self::ALL.len() as isize;
        Self::ALL[((current + delta).rem_euclid(len)) as usize]
    }
}

// --- Resource wrapping the physics world ---

/// Bevy resource wrapping an N-dimensional [`PhysicsWorld`].
///
/// Access this in your systems to add bodies, step simulation manually,
/// or query state.
#[derive(Resource)]
pub struct BevyPhysics<const D: usize> {
    /// The N-dimensional rigid body world.
    pub world: PhysicsWorld<D>,
}

impl<const D: usize> Default for BevyPhysics<D> {
    fn default() -> Self {
        Self {
            world: PhysicsWorld::new(SVector::zeros()),
        }
    }
}

impl<const D: usize> BevyPhysics<D> {
    /// Create with custom gravity.
    pub fn with_gravity(gravity: [f64; D]) -> Self {
        Self {
            world: PhysicsWorld::new(SVector::from(gravity)),
        }
    }
}

// --- Linking component ---

/// Bevy component linking an entity to a physics body.
///
/// Attach to a Bevy entity with a `Transform`; `sync_transforms` writes the
/// physics body's position into it each `FixedUpdate`.
#[derive(Component)]
pub struct PhysicsBody {
    /// Handle to the body in the physics world.
    pub handle: BodyHandle,
    /// Visual radius for debug rendering.
    pub visual_radius: f32,
}

impl PhysicsBody {
    /// Create a new component for a given body handle and visual radius.
    pub fn new(handle: BodyHandle, visual_radius: f32) -> Self {
        Self {
            handle,
            visual_radius,
        }
    }
}

// --- Default no-coupling callback (resource flavor) ---

/// Bevy `Resource` form of the identity "no-coupling" callback.
///
/// Forces, impulses, and friction pass through unchanged. Use when you want
/// N-dimensional physics without any per-body state coupling. Implements
/// [`PhysicsCallback<D>`] for all `D`.
#[derive(Resource, Default)]
pub struct NoCouplingResource;

impl<const D: usize> PhysicsCallback<D> for NoCouplingResource {
    fn modulate_force(&self, _: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D> {
        *force
    }
    fn modulate_impulse(&self, impulse: f64, _: &SVector<f64, D>) -> f64 {
        impulse
    }
    fn friction_multiplier(&self, _: &SVector<f64, D>, _: BodyHandle) -> f64 {
        1.0
    }
    fn on_collision(&mut self, _: &CollisionEvent<D>) {}
    fn record_dissipation(&mut self, _: f64) {}
    fn record_work(&mut self, _: BodyHandle, _: f64) {}
    fn apply_trauma(&mut self, _: &CollisionEvent<D>) {}
}

// --- Systems ---

/// Translate raw keyboard/mouse state into named first-person intents.
pub fn input_intent_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    bindings: Res<InputBindings>,
    mut intents: ResMut<IntentFrame>,
) {
    intents.clear();
    intents.look_delta = mouse_motion.delta;

    if keyboard.pressed(bindings.move_forward) {
        intents.movement.y += 1.0;
        intents.pressed.push(InputIntent::MoveForward);
    }
    if keyboard.pressed(bindings.move_back) {
        intents.movement.y -= 1.0;
        intents.pressed.push(InputIntent::MoveBack);
    }
    if keyboard.pressed(bindings.move_left) {
        intents.movement.x -= 1.0;
        intents.pressed.push(InputIntent::MoveLeft);
    }
    if keyboard.pressed(bindings.move_right) {
        intents.movement.x += 1.0;
        intents.pressed.push(InputIntent::MoveRight);
    }
    if keyboard.pressed(bindings.sprint) || keyboard.pressed(KeyCode::ShiftRight) {
        intents.pressed.push(InputIntent::Sprint);
    }
    if keyboard.pressed(bindings.crouch) || keyboard.pressed(KeyCode::ControlRight) {
        intents.pressed.push(InputIntent::Crouch);
    }

    macro_rules! add_just {
        ($key:expr, $intent:expr) => {
            if keyboard.just_pressed($key) {
                intents.just_pressed.push($intent);
            }
        };
    }
    macro_rules! add_just_ctrl {
        ($key:expr, $intent:expr) => {
            if (keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight))
                && keyboard.just_pressed($key)
            {
                intents.just_pressed.push($intent);
            }
        };
    }

    add_just!(bindings.jump, InputIntent::Jump);
    add_just!(bindings.interact, InputIntent::Interact);
    add_just!(bindings.focus_inspect, InputIntent::FocusInspect);
    add_just!(bindings.toggle_field_deck, InputIntent::OpenFieldDeck);
    add_just!(bindings.quick_tool, InputIntent::QuickTool);
    add_just!(bindings.repair_tool, InputIntent::RepairTool);
    add_just!(bindings.build_mode, InputIntent::BuildMode);
    add_just!(bindings.chronicle_panel, InputIntent::OpenChroniclePanel);
    add_just!(bindings.basin_map, InputIntent::OpenBasinMap);
    add_just!(
        bindings.scan_visualization,
        InputIntent::ToggleScanVisualization
    );
    add_just!(
        bindings.previous_scan_mode,
        InputIntent::CycleFieldDeckModePrev
    );
    add_just!(bindings.next_scan_mode, InputIntent::CycleFieldDeckModeNext);
    add_just!(bindings.pause_or_release, InputIntent::PauseOrRelease);
    add_just!(bindings.controls_overlay, InputIntent::OpenControlsOverlay);
    add_just_ctrl!(bindings.command_palette, InputIntent::OpenCommandPalette);
    add_just!(
        bindings.dev_scenario_panel,
        InputIntent::OpenDevScenarioPanel
    );

    for (key, slot) in [
        (KeyCode::Digit1, 1),
        (KeyCode::Digit2, 2),
        (KeyCode::Digit3, 3),
        (KeyCode::Digit4, 4),
        (KeyCode::Digit5, 5),
        (KeyCode::Digit6, 6),
    ] {
        if keyboard.just_pressed(key) {
            intents.just_pressed.push(InputIntent::EquipToolSlot(slot));
        }
    }

    add_just!(KeyCode::KeyP, InputIntent::PauseSimulation);
    add_just!(KeyCode::Period, InputIntent::StepSimulation);
    add_just!(KeyCode::F8, InputIntent::ResetScenario);
    add_just!(KeyCode::F9, InputIntent::CaptureReplay);

    if mouse.just_pressed(MouseButton::Left) {
        intents.just_pressed.push(InputIntent::FocusInspect);
    }
}

/// Apply generic menu/tool/Field Deck mode changes from named intents.
pub fn control_mode_system(intents: Res<IntentFrame>, mut controls: ResMut<ControlsState>) {
    if intents.just_pressed(InputIntent::PauseOrRelease) {
        controls.mode = ControlMode::Pause;
        controls.mouse_captured = false;
        controls.show_dev_panel = false;
    }

    if intents.just_pressed(InputIntent::OpenControlsOverlay) {
        controls.show_controls = !controls.show_controls;
    }

    if intents.just_pressed(InputIntent::OpenDevScenarioPanel) {
        controls.show_dev_panel = !controls.show_dev_panel;
        controls.mode = if controls.show_dev_panel {
            controls.mouse_captured = false;
            ControlMode::DevScenario
        } else {
            controls.mouse_captured = true;
            ControlMode::FirstPerson
        };
    }

    if intents.just_pressed(InputIntent::OpenFieldDeck) {
        controls.show_dev_panel = false;
        match controls.mode {
            ControlMode::FieldDeck => {
                controls.mode = ControlMode::FirstPerson;
                controls.mouse_captured = true;
            }
            _ => {
                controls.mode = ControlMode::FieldDeck;
                controls.mouse_captured = false;
            }
        }
    }

    if intents.just_pressed(InputIntent::CycleFieldDeckModePrev) {
        controls.scan_mode = controls.scan_mode.offset(-1);
    }
    if intents.just_pressed(InputIntent::CycleFieldDeckModeNext) {
        controls.scan_mode = controls.scan_mode.offset(1);
    }

    for intent in &intents.just_pressed {
        match *intent {
            InputIntent::EquipToolSlot(slot) => controls.selected_tool_slot = slot,
            InputIntent::FocusInspect if controls.mode == ControlMode::Pause => {
                controls.mode = ControlMode::FirstPerson;
                controls.mouse_captured = true;
            }
            _ => {}
        }
    }
}

/// Sync the OS cursor with current first-person capture state.
pub fn cursor_capture_system(
    controls: Res<ControlsState>,
    mut query: Query<&mut bevy::window::CursorOptions>,
) {
    if !controls.is_changed() {
        return;
    }
    let Ok(mut cursor_options) = query.single_mut() else {
        return;
    };
    cursor_options.visible = !controls.mouse_captured;
    cursor_options.grab_mode = if controls.mouse_captured {
        CursorGrabMode::Locked
    } else {
        CursorGrabMode::None
    };
}

/// Move all entities carrying [`FirstPersonController`] from the current intent frame.
pub fn first_person_move_system(
    intents: Res<IntentFrame>,
    time: Res<Time>,
    controls: Res<ControlsState>,
    mut query: Query<(&mut Transform, &FirstPersonController)>,
) {
    for (mut transform, controller) in &mut query {
        let base_speed = match controls.mode {
            ControlMode::FirstPerson => controller.walk_speed,
            ControlMode::FieldDeck => controller.field_deck_speed,
            ControlMode::Console | ControlMode::DevScenario | ControlMode::Pause => 0.0,
        };

        if base_speed == 0.0 {
            continue;
        }

        let mut speed = base_speed;
        if intents.pressed(InputIntent::Sprint) && controls.mode == ControlMode::FirstPerson {
            speed *= controller.sprint_multiplier;
        }
        if intents.pressed(InputIntent::Crouch) {
            speed *= controller.crouch_multiplier;
        }

        let mut direction =
            *transform.forward() * intents.movement.y + *transform.right() * intents.movement.x;
        direction.y = 0.0;
        if direction.length_squared() > 0.0 {
            direction = direction.normalize();
            transform.translation += direction * speed * time.delta_secs();
        }
    }
}

/// Rotate first-person entities from accumulated mouse movement.
pub fn first_person_mouse_look_system(
    intents: Res<IntentFrame>,
    controls: Res<ControlsState>,
    mut query: Query<(&mut Transform, &FirstPersonController)>,
) {
    if !controls.mouse_captured || controls.mode != ControlMode::FirstPerson {
        return;
    }
    let delta = intents.look_delta;
    if delta == Vec2::ZERO {
        return;
    }

    for (mut transform, controller) in &mut query {
        let delta_yaw = -delta.x * controller.mouse_sensitivity.x;
        let delta_pitch = -delta.y * controller.mouse_sensitivity.y;
        let (yaw, pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
        let pitch_limit = std::f32::consts::FRAC_PI_2 - 0.01;
        transform.rotation = Quat::from_euler(
            EulerRot::YXZ,
            yaw + delta_yaw,
            (pitch + delta_pitch).clamp(-pitch_limit, pitch_limit),
            roll,
        );
    }
}

/// Reusable first-person input plugin. Add [`FirstPersonController`] to the
/// player/camera entity and read [`IntentFrame`] from game systems.
#[derive(Default)]
pub struct FirstPersonInputPlugin;

impl Plugin for FirstPersonInputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InputBindings>()
            .init_resource::<IntentFrame>()
            .init_resource::<ControlsState>()
            .add_systems(
                Update,
                (
                    input_intent_system,
                    control_mode_system,
                    cursor_capture_system,
                    first_person_move_system,
                    first_person_mouse_look_system,
                )
                    .chain(),
            );
    }
}

/// Step the physics world with a generic callback `Resource`.
///
/// Bring your own `C: PhysicsCallback<D> + Resource` to couple a custom metric
/// to physics, or use [`NoCouplingResource`] for uncoupled physics-only behavior.
pub fn step_physics<
    const D: usize,
    C: PhysicsCallback<D>
        + Resource
        + bevy::ecs::component::Component<Mutability = bevy::ecs::component::Mutable>,
>(
    mut physics: ResMut<BevyPhysics<D>>,
    mut cb: ResMut<C>,
    time: Res<Time<Fixed>>,
) {
    physics
        .world
        .step_with_callback(time.delta_secs_f64(), &mut *cb);
}

/// Sync physics body positions to Bevy `Transform`s.
///
/// 2D: writes `(x, y)` to `translation.x`/`.y`.
/// 3D: writes `(x, y, z)` to `translation`.
/// 4D: writes `(x, y, z)` to `translation` (w dropped — use `symtropy-render-bridge`
/// for cross-section projection).
pub fn sync_transforms<const D: usize>(
    physics: Res<BevyPhysics<D>>,
    mut query: Query<(&PhysicsBody, &mut Transform)>,
) {
    for (body_comp, mut transform) in &mut query {
        if let Some(body) = physics.world.body(body_comp.handle) {
            let pos = body.position();
            if D >= 1 {
                transform.translation.x = pos[0] as f32;
            }
            if D >= 2 {
                transform.translation.y = pos[1] as f32;
            }
            if D >= 3 {
                transform.translation.z = pos[2] as f32;
            }
        }
    }
}

// --- Plugins ---

/// Minimal Bevy plugin: registers [`BevyPhysics<D>`] + [`sync_transforms`] only.
///
/// Use this when you're supplying your own step system with a custom
/// [`PhysicsCallback`] resource.
pub struct BevyPhysicsCorePlugin<const D: usize> {
    /// Initial gravity.
    pub gravity: SVector<f64, D>,
}

impl<const D: usize> Default for BevyPhysicsCorePlugin<D> {
    fn default() -> Self {
        Self {
            gravity: SVector::zeros(),
        }
    }
}

impl<const D: usize> BevyPhysicsCorePlugin<D> {
    /// Create with the given gravity vector.
    pub fn with_gravity(gravity: [f64; D]) -> Self {
        Self {
            gravity: SVector::from(gravity),
        }
    }
}

/// Full Bevy plugin: registers [`BevyPhysics<D>`] + [`NoCouplingResource`] + a
/// default [`step_physics`] system + [`sync_transforms`].
///
/// Drop in for N-dimensional physics with no coupling. For custom couplings,
/// use [`BevyPhysicsCorePlugin`] and register your own step system.
pub struct BevyPhysicsPlugin<const D: usize> {
    /// Initial gravity.
    pub gravity: SVector<f64, D>,
}

impl<const D: usize> Default for BevyPhysicsPlugin<D> {
    fn default() -> Self {
        Self {
            gravity: SVector::zeros(),
        }
    }
}

impl<const D: usize> BevyPhysicsPlugin<D> {
    /// Create with the given gravity vector.
    pub fn with_gravity(gravity: [f64; D]) -> Self {
        Self {
            gravity: SVector::from(gravity),
        }
    }
}

// Per-dim Plugin impls — Bevy's Plugin trait isn't const-generic.

macro_rules! impl_core_plugin {
    ($d:literal) => {
        impl Plugin for BevyPhysicsCorePlugin<$d> {
            fn build(&self, app: &mut App) {
                app.insert_resource(BevyPhysics::<$d> {
                    world: PhysicsWorld::new(self.gravity),
                });
                app.add_systems(FixedUpdate, sync_transforms::<$d>);
            }
        }
    };
}

macro_rules! impl_full_plugin {
    ($d:literal) => {
        impl Plugin for BevyPhysicsPlugin<$d> {
            fn build(&self, app: &mut App) {
                app.insert_resource(BevyPhysics::<$d> {
                    world: PhysicsWorld::new(self.gravity),
                });
                app.insert_resource(NoCouplingResource);
                app.add_systems(
                    FixedUpdate,
                    (
                        step_physics::<$d, NoCouplingResource>,
                        sync_transforms::<$d>,
                    )
                        .chain(),
                );
            }
        }
    };
}

impl_core_plugin!(2);
impl_core_plugin!(3);
impl_core_plugin!(4);

impl_full_plugin!(2);
impl_full_plugin!(3);
impl_full_plugin!(4);

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use symtropy_math::Point;

    #[test]
    fn bevy_physics_default_has_zero_gravity_2d() {
        let p = BevyPhysics::<2>::default();
        let g = p.world.gravity;
        assert_eq!(g[0], 0.0);
        assert_eq!(g[1], 0.0);
    }

    #[test]
    fn bevy_physics_with_gravity_stores_gravity() {
        let p = BevyPhysics::<3>::with_gravity([0.0, -9.81, 0.0]);
        let g = p.world.gravity;
        assert!((g[1] - (-9.81)).abs() < 1e-9);
    }

    #[test]
    fn physics_body_new_stores_handle_and_radius() {
        let mut world = PhysicsWorld::<2>::new(SVector::zeros());
        let h = world.add_sphere(Point::new([0.0, 0.0]), 1.0, 1.0);
        let b = PhysicsBody::new(h, 0.5);
        assert_eq!(b.handle, h);
        assert!((b.visual_radius - 0.5).abs() < 1e-9);
    }

    #[test]
    fn default_input_bindings_match_first_person_spine() {
        let bindings = InputBindings::default();
        assert_eq!(bindings.move_forward, KeyCode::KeyW);
        assert_eq!(bindings.interact, KeyCode::KeyE);
        assert_eq!(bindings.toggle_field_deck, KeyCode::Tab);
        assert_eq!(bindings.dev_scenario_panel, KeyCode::F10);
    }

    #[test]
    fn scan_mode_cycles_deterministically() {
        assert_eq!(ScanMode::Ecology.offset(1), ScanMode::MachineDiagnostics);
        assert_eq!(
            ScanMode::Infrastructure.offset(-1),
            ScanMode::ChronicleEvidence
        );
    }

    #[test]
    fn intent_frame_tracks_named_intents() {
        let mut frame = IntentFrame::default();
        frame.pressed.push(InputIntent::Sprint);
        frame.just_pressed.push(InputIntent::Interact);

        assert!(frame.pressed(InputIntent::Sprint));
        assert!(frame.just_pressed(InputIntent::Interact));

        frame.clear();
        assert!(!frame.pressed(InputIntent::Sprint));
        assert!(!frame.just_pressed(InputIntent::Interact));
    }

    #[test]
    fn no_coupling_modulate_force_is_identity() {
        let cb = NoCouplingResource;
        let mut world = PhysicsWorld::<2>::new(SVector::zeros());
        let h = world.add_sphere(Point::new([0.0, 0.0]), 1.0, 1.0);
        let f_in = SVector::from([3.14, -2.71]);
        let f_out = <NoCouplingResource as PhysicsCallback<2>>::modulate_force(&cb, h, &f_in);
        assert_eq!(f_in, f_out);
    }

    #[test]
    fn no_coupling_friction_multiplier_is_one() {
        let cb = NoCouplingResource;
        let mut world = PhysicsWorld::<3>::new(SVector::zeros());
        let h = world.add_sphere(Point::new([0.0, 0.0, 0.0]), 1.0, 1.0);
        let point = SVector::from([0.0, 0.0, 0.0]);
        let mu = <NoCouplingResource as PhysicsCallback<3>>::friction_multiplier(&cb, &point, h);
        assert!((mu - 1.0).abs() < 1e-9);
    }

    #[test]
    fn core_plugin_2d_registers_resource() {
        let mut app = App::new();
        BevyPhysicsCorePlugin::<2>::default().build(&mut app);
        assert!(app.world().contains_resource::<BevyPhysics<2>>());
    }

    #[test]
    fn full_plugin_2d_registers_both_resources() {
        let mut app = App::new();
        BevyPhysicsPlugin::<2>::default().build(&mut app);
        assert!(app.world().contains_resource::<BevyPhysics<2>>());
        assert!(app.world().contains_resource::<NoCouplingResource>());
    }

    #[test]
    fn full_plugin_3d_registers_both_resources() {
        let mut app = App::new();
        BevyPhysicsPlugin::<3>::default().build(&mut app);
        assert!(app.world().contains_resource::<BevyPhysics<3>>());
        assert!(app.world().contains_resource::<NoCouplingResource>());
    }

    #[test]
    fn full_plugin_4d_registers_both_resources() {
        let mut app = App::new();
        BevyPhysicsPlugin::<4>::default().build(&mut app);
        assert!(app.world().contains_resource::<BevyPhysics<4>>());
        assert!(app.world().contains_resource::<NoCouplingResource>());
    }

    #[test]
    fn plugin_with_gravity_resource_accessible() {
        let mut app = App::new();
        BevyPhysicsPlugin::<2>::with_gravity([0.0, -9.81]).build(&mut app);
        let res = app.world().resource::<BevyPhysics<2>>();
        let g = res.world.gravity;
        assert!((g[1] - (-9.81)).abs() < 1e-9);
    }

    #[test]
    fn manual_step_with_no_coupling_gravity_pulls_body_down() {
        let mut p = BevyPhysics::<2>::with_gravity([0.0, -9.81]);
        let h = p.world.add_sphere(Point::new([0.0, 10.0]), 1.0, 1.0);
        let y0 = p.world.body(h).unwrap().position()[1];
        let mut cb = NoCouplingResource;
        for _ in 0..10 {
            p.world.step_with_callback(1.0 / 60.0, &mut cb);
        }
        let y1 = p.world.body(h).unwrap().position()[1];
        assert!(y1 < y0, "gravity should pull body down: y0={y0}, y1={y1}");
    }

    #[test]
    fn can_add_multiple_bodies() {
        let mut p = BevyPhysics::<3>::default();
        let h1 = p.world.add_sphere(Point::new([0.0, 0.0, 0.0]), 1.0, 1.0);
        let h2 = p.world.add_sphere(Point::new([5.0, 0.0, 0.0]), 1.0, 1.0);
        assert!(p.world.body(h1).is_some());
        assert!(p.world.body(h2).is_some());
    }
}
