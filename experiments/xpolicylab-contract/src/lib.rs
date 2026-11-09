//! Dependency-free validator for the XPolicyLab policy-server boundary.
//!
//! This crate validates correlation and shape for policy-proposed action chunks.
//! It is deliberately not a motion planner, collision checker, controller,
//! safety authority, physical-observation verifier, or durable replay ledger.
//! Every accepted proposal must still pass through the existing Symtropy
//! authenticated admission and native controller authority path.
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextField {
    RequestId,
    EpisodeId,
    PolicyServerInstanceId,
    SchemaVersion,
    ActionSpace,
    ExpectedEnvironments,
    ActionChannels,
    ActionDimension,
    MaxActionsPerChunk,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    EmptyExpectedField(ContextField),
    DuplicateExpectedEnvironment(u32),
    ZeroActionDimension(String),
    EnvironmentCountMismatch {
        expected: usize,
        actual: usize,
    },
    EnvironmentOrderMismatch {
        position: usize,
        expected: u32,
        actual: u32,
    },
    RequestIdMismatch,
    EpisodeMismatch,
    PolicyServerInstanceMismatch,
    SchemaVersionMismatch,
    ActionSpaceMismatch,
    ChunkIndexMismatch {
        env_idx: u32,
        expected: u64,
        actual: u64,
    },
    EmptyActionChunk {
        env_idx: u32,
    },
    ActionChunkTooLarge {
        env_idx: u32,
        maximum: usize,
        actual: usize,
    },
    ChannelSetMismatch {
        env_idx: u32,
        step_index: usize,
    },
    ChannelDimensionMismatch {
        env_idx: u32,
        step_index: usize,
        channel: String,
        expected: usize,
        actual: usize,
    },
    NonFiniteAction {
        env_idx: u32,
        step_index: usize,
        channel: String,
        value_index: usize,
    },
    PolicyRpcTimeoutAmbiguous,
    PolicyServerRestarted,
}

/// The environment-side caller freezes the values below before issuing one
/// XPolicyLab model call. These identifiers are supplied by the Luminous
/// adapter; they are not assumed to be fields returned by the upstream model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedEnvironment {
    pub env_idx: u32,
    pub next_chunk_index: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedPolicyCall {
    pub request_id: String,
    pub episode_id: String,
    pub policy_server_instance_id: String,
    pub schema_version: String,
    /// A protocol mode such as "joint" or "ee"; this does not imply physical units.
    pub action_space: String,
    /// Ordered exactly as the env_idx_list passed to get_action_batch.
    pub environments: Vec<ExpectedEnvironment>,
    /// Per-channel dimensions for the exact robot/action configuration.
    pub channel_dimensions: BTreeMap<String, usize>,
    /// Bound policy-side output to a finite, preregistered chunk length.
    pub max_actions_per_chunk: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionStep {
    /// The native action dictionary represented as named numeric channels.
    pub channels: BTreeMap<String, Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentActionChunk {
    pub env_idx: u32,
    /// Adapter-assigned sequence identity; not an upstream model field.
    pub chunk_index: u64,
    /// Ordered action dictionaries returned by get_action().
    pub steps: Vec<ActionStep>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyActionResponse {
    /// Correlation envelope attached by the Luminous environment client.
    pub request_id: String,
    pub episode_id: String,
    /// A process-instance identity, not merely a friendly host/port name.
    pub policy_server_instance_id: String,
    pub schema_version: String,
    pub action_space: String,
    /// The client binds each output slot to the original env_idx_list position.
    pub environment_chunks: Vec<EnvironmentActionChunk>,
}

/// Transport outcomes from the upstream policy client. A timeout is kept
/// separate from a definite model rejection because the server may still be
/// processing the request. A server restart invalidates model-side episode state.
#[derive(Debug, Clone, Copy)]
pub enum PolicyCallResult<'a> {
    Response(&'a PolicyActionResponse),
    Timeout,
    ServerRestarted,
}

/// Validate one response before converting it into a policy proposal.
///
/// The exact environment order and each next chunk index are checked. All
/// channel names and dimensions must match the frozen action schema; no
/// implicit reshape, padding, truncation, channel substitution or reordering
/// is performed. Values must be finite, but range/units/physical limits remain
/// the responsibility of the downstream controller and its safety authority.
pub fn validate_action_response(
    expected: &ExpectedPolicyCall,
    response: &PolicyActionResponse,
) -> Result<(), ContractError> {
    validate_expected_context(expected)?;

    if response.request_id != expected.request_id {
        return Err(ContractError::RequestIdMismatch);
    }
    if response.episode_id != expected.episode_id {
        return Err(ContractError::EpisodeMismatch);
    }
    if response.policy_server_instance_id != expected.policy_server_instance_id {
        return Err(ContractError::PolicyServerInstanceMismatch);
    }
    if response.schema_version != expected.schema_version {
        return Err(ContractError::SchemaVersionMismatch);
    }
    if response.action_space != expected.action_space {
        return Err(ContractError::ActionSpaceMismatch);
    }

    if response.environment_chunks.len() != expected.environments.len() {
        return Err(ContractError::EnvironmentCountMismatch {
            expected: expected.environments.len(),
            actual: response.environment_chunks.len(),
        });
    }

    for (position, (expected_env, actual_chunk)) in expected
        .environments
        .iter()
        .zip(response.environment_chunks.iter())
        .enumerate()
    {
        if actual_chunk.env_idx != expected_env.env_idx {
            return Err(ContractError::EnvironmentOrderMismatch {
                position,
                expected: expected_env.env_idx,
                actual: actual_chunk.env_idx,
            });
        }
        if actual_chunk.chunk_index != expected_env.next_chunk_index {
            return Err(ContractError::ChunkIndexMismatch {
                env_idx: expected_env.env_idx,
                expected: expected_env.next_chunk_index,
                actual: actual_chunk.chunk_index,
            });
        }
        if actual_chunk.steps.is_empty() {
            return Err(ContractError::EmptyActionChunk {
                env_idx: expected_env.env_idx,
            });
        }
        if actual_chunk.steps.len() > expected.max_actions_per_chunk {
            return Err(ContractError::ActionChunkTooLarge {
                env_idx: expected_env.env_idx,
                maximum: expected.max_actions_per_chunk,
                actual: actual_chunk.steps.len(),
            });
        }

        for (step_index, step) in actual_chunk.steps.iter().enumerate() {
            if step.channels.len() != expected.channel_dimensions.len()
                || step.channels.keys().ne(expected.channel_dimensions.keys())
            {
                return Err(ContractError::ChannelSetMismatch {
                    env_idx: expected_env.env_idx,
                    step_index,
                });
            }

            for (channel, expected_dimension) in &expected.channel_dimensions {
                let values = step.channels.get(channel).ok_or(
                    ContractError::ChannelSetMismatch {
                        env_idx: expected_env.env_idx,
                        step_index,
                    },
                )?;
                if values.len() != *expected_dimension {
                    return Err(ContractError::ChannelDimensionMismatch {
                        env_idx: expected_env.env_idx,
                        step_index,
                        channel: channel.clone(),
                        expected: *expected_dimension,
                        actual: values.len(),
                    });
                }
                for (value_index, value) in values.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(ContractError::NonFiniteAction {
                            env_idx: expected_env.env_idx,
                            step_index,
                            channel: channel.clone(),
                            value_index,
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

/// Validate a completed policy call while preserving transport ambiguity.
/// A timeout or process restart never returns an action chunk to the caller.
pub fn validate_policy_call(
    expected: &ExpectedPolicyCall,
    result: PolicyCallResult<'_>,
) -> Result<(), ContractError> {
    match result {
        PolicyCallResult::Response(response) => validate_action_response(expected, response),
        PolicyCallResult::Timeout => Err(ContractError::PolicyRpcTimeoutAmbiguous),
        PolicyCallResult::ServerRestarted => Err(ContractError::PolicyServerRestarted),
    }
}

fn validate_expected_context(expected: &ExpectedPolicyCall) -> Result<(), ContractError> {
    for (value, field) in [
        (&expected.request_id, ContextField::RequestId),
        (&expected.episode_id, ContextField::EpisodeId),
        (
            &expected.policy_server_instance_id,
            ContextField::PolicyServerInstanceId,
        ),
        (&expected.schema_version, ContextField::SchemaVersion),
        (&expected.action_space, ContextField::ActionSpace),
    ] {
        if value.trim().is_empty() {
            return Err(ContractError::EmptyExpectedField(field));
        }
    }

    if expected.environments.is_empty() {
        return Err(ContractError::EmptyExpectedField(
            ContextField::ExpectedEnvironments,
        ));
    }
    let mut seen = BTreeSet::new();
    for environment in &expected.environments {
        if !seen.insert(environment.env_idx) {
            return Err(ContractError::DuplicateExpectedEnvironment(
                environment.env_idx,
            ));
        }
    }

    if expected.channel_dimensions.is_empty() {
        return Err(ContractError::EmptyExpectedField(
            ContextField::ActionChannels,
        ));
    }
    for (channel, dimension) in &expected.channel_dimensions {
        if channel.trim().is_empty() {
            return Err(ContractError::EmptyExpectedField(
                ContextField::ActionChannels,
            ));
        }
        if *dimension == 0 {
            return Err(ContractError::ZeroActionDimension(channel.clone()));
        }
    }
    if expected.max_actions_per_chunk == 0 {
        return Err(ContractError::EmptyExpectedField(
            ContextField::MaxActionsPerChunk,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected() -> ExpectedPolicyCall {
        ExpectedPolicyCall {
            request_id: "policy-call-01".into(),
            episode_id: "episode-01".into(),
            policy_server_instance_id: "policy-server-process-9".into(),
            schema_version: "xpl-v1.0".into(),
            action_space: "joint".into(),
            environments: vec![
                ExpectedEnvironment {
                    env_idx: 3,
                    next_chunk_index: 7,
                },
                ExpectedEnvironment {
                    env_idx: 8,
                    next_chunk_index: 4,
                },
            ],
            channel_dimensions: BTreeMap::from([
                ("left_arm_joint_state".into(), 7),
                ("left_ee_joint_state".into(), 1),
                ("right_arm_joint_state".into(), 7),
                ("right_ee_joint_state".into(), 1),
            ]),
            max_actions_per_chunk: 16,
        }
    }

    fn action_step() -> ActionStep {
        ActionStep {
            channels: BTreeMap::from([
                ("left_arm_joint_state".into(), vec![0.0; 7]),
                ("left_ee_joint_state".into(), vec![0.0; 1]),
                ("right_arm_joint_state".into(), vec![0.0; 7]),
                ("right_ee_joint_state".into(), vec![0.0; 1]),
            ]),
        }
    }

    fn response() -> PolicyActionResponse {
        PolicyActionResponse {
            request_id: "policy-call-01".into(),
            episode_id: "episode-01".into(),
            policy_server_instance_id: "policy-server-process-9".into(),
            schema_version: "xpl-v1.0".into(),
            action_space: "joint".into(),
            environment_chunks: vec![
                EnvironmentActionChunk {
                    env_idx: 3,
                    chunk_index: 7,
                    steps: vec![action_step(), action_step()],
                },
                EnvironmentActionChunk {
                    env_idx: 8,
                    chunk_index: 4,
                    steps: vec![action_step()],
                },
            ],
        }
    }

    #[test]
    fn accepts_one_complete_ordered_action_chunk_per_environment() {
        assert_eq!(validate_action_response(&expected(), &response()), Ok(()));
    }

    #[test]
    fn rejects_empty_expected_environment_set() {
        let mut context = expected();
        context.environments.clear();
        assert_eq!(
            validate_action_response(&context, &response()),
            Err(ContractError::EmptyExpectedField(
                ContextField::ExpectedEnvironments
            ))
        );
    }

    #[test]
    fn rejects_duplicate_expected_environment_indices() {
        let mut context = expected();
        context.environments[1].env_idx = 3;
        assert_eq!(
            validate_action_response(&context, &response()),
            Err(ContractError::DuplicateExpectedEnvironment(3))
        );
    }

    #[test]
    fn rejects_missing_environment_chunk() {
        let mut actual = response();
        actual.environment_chunks.pop();
        assert_eq!(
            validate_action_response(&expected(), &actual),
            Err(ContractError::EnvironmentCountMismatch {
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn rejects_reordered_environment_chunks() {
        let mut actual = response();
        actual.environment_chunks.swap(0, 1);
        assert_eq!(
            validate_action_response(&expected(), &actual),
            Err(ContractError::EnvironmentOrderMismatch {
                position: 0,
                expected: 3,
                actual: 8
            })
        );
    }

    #[test]
    fn rejects_request_episode_and_server_mismatch() {
        let mut actual = response();
        actual.request_id = "other-call".into();
        assert_eq!(
            validate_action_response(&expected(), &actual),
            Err(ContractError::RequestIdMismatch)
        );

        let mut actual = response();
        actual.episode_id = "prior-episode".into();
        assert_eq!(
            validate_action_response(&expected(), &actual),
            Err(ContractError::EpisodeMismatch)
        );

        let mut actual = response();
        actual.policy_server_instance_id = "restarted-process".into();
        assert_eq!(
            validate_action_response(&expected(), &actual),
            Err(ContractError::PolicyServerInstanceMismatch)
        );
    }

    #[test]
    fn rejects_schema_and_action_space_mismatch() {
        let mut actual = response();
        actual.schema_version = "unknown-schema".into();
        assert_eq!(
            validate_action_response(&expected(), &actual),
            Err(ContractError::SchemaVersionMismatch)
        );

        let mut actual = response();
        actual.action_space = "ee".into();
        assert_eq!(
            validate_action_response(&expected(), &actual),
            Err(ContractError::ActionSpaceMismatch)
        );
    }

    #[test]
    fn rejects_stale_or_skipped_chunk_index() {
        let mut actual = response();
        actual.environment_chunks[0].chunk_index = 6;
        assert_eq!(
            validate_action_response(&expected(), &actual),
            Err(ContractError::ChunkIndexMismatch {
                env_idx: 3,
                expected: 7,
                actual: 6
            })
        );
    }

    #[test]
    fn rejects_empty_and_overlong_chunks() {
        let mut actual = response();
        actual.environment_chunks[0].steps.clear();
        assert_eq!(
            validate_action_response(&expected(), &actual),
            Err(ContractError::EmptyActionChunk { env_idx: 3 })
        );

        let mut context = expected();
        context.max_actions_per_chunk = 1;
        assert_eq!(
            validate_action_response(&context, &response()),
            Err(ContractError::ActionChunkTooLarge {
                env_idx: 3,
                maximum: 1,
                actual: 2
            })
        );
    }

    #[test]
    fn rejects_missing_extra_or_renamed_action_channels() {
        let mut actual = response();
        actual.environment_chunks[0].steps[0]
            .channels
            .remove("right_ee_joint_state");
        assert!(matches!(
            validate_action_response(&expected(), &actual),
            Err(ContractError::ChannelSetMismatch { .. })
        ));

        let mut actual = response();
        actual.environment_chunks[0].steps[0]
            .channels
            .insert("unexpected".into(), vec![0.0]);
        assert!(matches!(
            validate_action_response(&expected(), &actual),
            Err(ContractError::ChannelSetMismatch { .. })
        ));
    }

    #[test]
    fn rejects_wrong_channel_dimension_and_non_finite_values() {
        let mut actual = response();
        actual.environment_chunks[0].steps[0]
            .channels
            .insert("left_arm_joint_state".into(), vec![0.0; 6]);
        assert_eq!(
            validate_action_response(&expected(), &actual),
            Err(ContractError::ChannelDimensionMismatch {
                env_idx: 3,
                step_index: 0,
                channel: "left_arm_joint_state".into(),
                expected: 7,
                actual: 6
            })
        );

        let mut actual = response();
        actual.environment_chunks[0].steps[0]
            .channels
            .get_mut("left_arm_joint_state")
            .unwrap()[2] = f64::NAN;
        assert_eq!(
            validate_action_response(&expected(), &actual),
            Err(ContractError::NonFiniteAction {
                env_idx: 3,
                step_index: 0,
                channel: "left_arm_joint_state".into(),
                value_index: 2
            })
        );
    }

    #[test]
    fn ambiguous_timeout_and_server_restart_never_validate_as_action_output() {
        assert_eq!(
            validate_policy_call(&expected(), PolicyCallResult::Timeout),
            Err(ContractError::PolicyRpcTimeoutAmbiguous)
        );
        assert_eq!(
            validate_policy_call(&expected(), PolicyCallResult::ServerRestarted),
            Err(ContractError::PolicyServerRestarted)
        );
        assert_eq!(
            validate_policy_call(&expected(), PolicyCallResult::Response(&response())),
            Ok(())
        );
    }

    #[test]
    fn rejects_invalid_expected_schema_before_reading_response() {
        let mut context = expected();
        context.channel_dimensions.insert("empty".into(), 0);
        assert_eq!(
            validate_action_response(&context, &response()),
            Err(ContractError::ZeroActionDimension("empty".into()))
        );
    }
}
