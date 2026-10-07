
                // SECURITY: Validate browser-supplied Nix config with pure-eval
                // (no network, no filesystem access, no builtins.exec)
                if !client_msg.configuration_nix.is_empty() {
                    use symthaea_spore::security::validate_nix_pure_eval;
                    if let Err(e) = validate_nix_pure_eval(&client_msg.configuration_nix) {
                        eprintln!(
                            "[{}] Nix pure-eval rejected browser config: {}",
                            peer_addr, e
                        );
                        let _ = ws_tx
                            .send(Message::Text(
                                RelayMessage::output(
                                    &format!("WARNING: Nix config validation: {}", e),
                                    "stderr",
                                )
                                .to_json(),
                            ))
                            .await;
                        // Continue anyway — pure-eval may reject valid NixOS modules
                        // that use impure features like <nixpkgs>. This is advisory, not blocking.
                    }
                }

                // Advisory pure evaluation and authoritative syntax parsing are deliberately
                // separate checks: module imports can make pure evaluation unavailable, but
                // malformed Nix syntax must never reach the disk-mutation script.
                if !client_msg.configuration_nix.is_empty() {
                    let config_path = format!("{}/configuration.nix", config_staging_dir);
                    match run_privileged_args("nix-instantiate", &["--parse", &config_path]).await {
                        Ok(result) if result.exit_status == 0 => {}
                        Ok(result) => {
                            let _ = ws_tx
                                .send(Message::Text(
                                    RelayMessage::error(&format!(
                                        "Nix configuration syntax validation failed: {}",
                                        result.stderr.chars().take(2000).collect::<String>()
                                    ))
                                    .to_json(),
                                ))
                                .await;
                            remove_transaction_artifact_dir(&transaction_dir);
                            continue;
                        }
                        Err(error) => {
                            let _ = ws_tx
                                .send(Message::Text(
                                    RelayMessage::error(&format!(
                                        "Nix configuration syntax validation could not be observed: {error}"
                                    ))
                                    .to_json(),
                                ))
                                .await;
                            remove_transaction_artifact_dir(&transaction_dir);
                            continue;
                        }
                    }
                }

                let mut script = generate_install_script(&client_msg, &transaction_dir);

                // Always: pre-install disk snapshot (instant, non-destructive)
                let snapshot = disk_snapshot(&disk);
                script = format!("{}\n{}", snapshot, script);

                // Patch configuration.nix with DE/GPU/locale — but only if the browser
                // didn't supply a full configuration.nix (which already has everything).
                if client_msg.configuration_nix.is_empty() {
                    let patch = system_config_patch(&client_msg);
                    if !patch.is_empty() {
                        if let Some(pos) = script.find("STAGE: Configuring swap") {
                            script.insert_str(pos, &patch);
                        } else if let Some(pos) = script.find("STAGE: Installing") {
                            script.insert_str(pos, &patch);
                        }
                    }
                }

                if client_msg.secure_boot {
                    if let Some(pos) = script.rfind("echo \"COMPLETE\"") {
                        script.insert_str(pos, secure_boot_postinstall());
                    } else {
                        script.push_str(secure_boot_postinstall());
                    }
                }
                if client_msg.tpm2_unlock {
                    if let Some(pos) = script.rfind("echo \"COMPLETE\"") {
                        script.insert_str(pos, tpm2_postinstall());