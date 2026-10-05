                    .active_inference
                    .process_input_with_causal_bias(best_goal, &root_causes);

                // Find the first action that is not on cooldown for this target
                let non_cooldown_action = plan.actions.iter().find(|sa| {
                    !self
                        .failed_action_cooldowns
                        .contains_key(&(target_name.clone(), sa.action.clone()))
                });

                if let Some(best_action) = non_cooldown_action.or_else(|| plan.actions.first()) {
                    if self.active_healing {
                        use nixward::action::executor::{NixOSCommand, NixOSExecutor, SafetyLevel, ServiceOperation};
                        let target_name_clone = target_name.clone();

                        // Try to generate a NixOS configuration AST hardening patch (Proposal 2)
                        let patch_tweak = self.generate_nixos_hardening_patch(&target_name_clone);

                        let (cmd, cmd_str, _is_patch) = if let Some((tweak, _, _)) = &patch_tweak {
                            let command_str =
                                format!("PATCH /etc/nixos/configuration.nix: {}", tweak);
                            (
                                NixOSCommand::RebuildSwitch {
                                    flake: None,
                                    extra_args: vec![],
                                },
                                command_str,
                                true,
                            )
                        } else {
                            let default_cmd = match &best_action.action {
                                ActionCategory::GarbageCollect => NixOSCommand::CollectGarbage {
                                    older_than_days: None,
                                    delete_all: false,
                                },
                                ActionCategory::Rebuild => NixOSCommand::RebuildSwitch {
                                    flake: None,
                                    extra_args: vec![],
                                },
                                ActionCategory::Rollback => NixOSCommand::EnvRollback,
                                ActionCategory::Enable => NixOSCommand::Service {
                                    operation: ServiceOperation::Enable {
                                        name: target_name_clone.clone(),
                                    },
                                },
                                ActionCategory::Disable => NixOSCommand::Service {
                                    operation: ServiceOperation::Disable {
                                        name: target_name_clone.clone(),
                                    },
                                },
                                _ => NixOSCommand::Service {
                                    operation: ServiceOperation::Restart {
                                        name: target_name_clone.clone(),
                                    },
                                },
                            };
                            let (bin, args) = default_cmd.to_command();
                            let command_str = format!("{} {}", bin, args.join(" "));
                            (default_cmd, command_str, false)
                        };

                        let safety = cmd.safety_level();
                        let is_modifying = safety != SafetyLevel::ReadOnly;

                        if is_modifying {