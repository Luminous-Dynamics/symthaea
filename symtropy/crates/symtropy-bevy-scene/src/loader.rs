// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Asset loader for `.sym` scene files.
//!
//! A `.sym` file is a RON-serialized scene description that includes
//! both standard Bevy entities and Symtropy-specific coupling parameters.

use bevy::asset::{io::Reader, AssetLoader, LoadContext};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// A Symtropy scene description.
#[derive(Asset, TypePath, Serialize, Deserialize, Debug, Clone)]
pub struct SymtropyScene {
    /// Path to the underlying Bevy scene (`.scn.ron`).
    pub bevy_scene: String,
    /// Global Φ-gravity strength.
    pub phi_gravity: f64,
    /// Initial harmony sources to spawn.
    pub harmony_sources: Vec<HarmonySourceDef>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HarmonySourceDef {
    pub position: [f64; 3], // Simplified to 3D for the loader baseline
    pub activations: [f64; 9],
    pub radius: f64,
    pub strength: f64,
}

#[derive(Default, TypePath)]
pub struct SymtropySceneLoader;

impl AssetLoader for SymtropySceneLoader {
    type Asset = SymtropyScene;
    type Settings = ();
    type Error = Box<dyn std::error::Error + Send + Sync>;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let scene: SymtropyScene = ron::de::from_bytes(&bytes)?;
        Ok(scene)
    }

    fn extensions(&self) -> &[&str] {
        &["sym"]
    }
}
