// SPDX-License-Identifier: AGPL-3.0-or-later

use bevy::prelude::*;

pub struct LivingLedgerPlugin;

impl Plugin for LivingLedgerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_ledger_ui);
        app.add_systems(Update, update_ledger_ui);
    }
}

#[derive(Component)]
struct LedgerUiNode;

fn setup_ledger_ui(mut commands: Commands) {
    commands
        .spawn((
            Node {
                width: Val::Percent(30.0),
                height: Val::Percent(50.0),
                position_type: PositionType::Absolute,
                left: Val::Px(10.0),
                top: Val::Px(10.0),
                padding: UiRect::all(Val::Px(10.0)),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)),
            LedgerUiNode,
        ))
        .with_children(|parent| {
            parent.spawn(Text::new("Chronicle Ledger Stream:"));
        });
}

fn update_ledger_ui(_query: Query<&mut Text, With<LedgerUiNode>>) {
    // UI update logic here.
}
