// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DreamableAction implementation for ActionIR

use super::{ActionIR, DestructivenessLevel};
use std::collections::BTreeMap;
use symthaea_dream::DreamableAction;

impl DreamableAction for ActionIR {
    fn perturb(&self, seed: u64) -> Self {
        match self {
            ActionIR::RunCommand {
                program,
                args,
                env,
                working_dir,
            } => {
                let mut new_args = args.clone();
                if seed % 2 == 0 {
                    if new_args.is_empty() {
                        new_args.push("--help".to_string());
                    } else if program == "rm" {
                        return ActionIR::RunCommand {
                            program: "ls".to_string(),
                            args: new_args,
                            env: env.clone(),
                            working_dir: working_dir.clone(),
                        };
                    }
                }

                ActionIR::RunCommand {
                    program: program.clone(),
                    args: new_args,
                    env: env.clone(),
                    working_dir: working_dir.clone(),
                }
            }
            ActionIR::WriteFile {
                path,
                content,
                create_dirs,
            } => {
                let mut new_path = path.clone();
                if let Some(ext) = path.extension() {
                    let mut new_ext = ext.to_os_string();
                    new_ext.push(".dream");
                    new_path.set_extension(new_ext);
                } else {
                    new_path.set_extension("dream");
                }

                ActionIR::WriteFile {
                    path: new_path,
                    content: content.clone(),
                    create_dirs: *create_dirs,
                }
            }
            ActionIR::DeleteFile { path } => ActionIR::RunCommand {
                program: "ls".to_string(),
                args: vec!["-l".to_string(), path.to_string_lossy().to_string()],
                env: std::collections::BTreeMap::new(),
                working_dir: None,
            },
            ActionIR::Sequence(actions) => {
                let idx = (seed as usize) % actions.len().max(1);
                let mut new_actions = actions.clone();
                if !new_actions.is_empty() {
                    new_actions[idx] = new_actions[idx].perturb(seed.wrapping_add(1));
                }
                ActionIR::Sequence(new_actions)
            }
            _ => self.clone(),
        }
    }

    fn magnitude(&self) -> f32 {
        match self.destructiveness() {
            DestructivenessLevel::Destructive => 1.0,
            DestructivenessLevel::NeedsConfirmation => 0.7,
            DestructivenessLevel::Reversible => 0.3,
            DestructivenessLevel::ReadOnly => 0.05,
        }
    }
}