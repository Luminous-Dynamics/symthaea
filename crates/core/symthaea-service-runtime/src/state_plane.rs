// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-separated latest-wins state-plane adapter for the service runtime.
//!
//! Runtime activity and completed cognitive state intentionally occupy independent
//! latest-wins slots. A client may therefore observe `Processing` immediately while
//! the cognitive snapshot remains the last completed observation until the owner
//! finishes the current mutation.

use symthaea_interface_runtime::{
    LatestStatePublisher, LatestStateReceiver, LatestStateSample, StatePlaneError,
    latest_state_channel,
};

use crate::observed::ServiceRuntimePublisher;
use crate::{RuntimeActivity, ServiceRuntimeSnapshot};

/// Owner-side write capability for service activity and cognitive snapshots.
///
/// This type implements [`ServiceRuntimePublisher`] and should remain inside the
/// runtime owner after construction.
pub struct ServiceStatePlanePublisher {
    activity: LatestStatePublisher<RuntimeActivity>,
    snapshot: LatestStatePublisher<ServiceRuntimeSnapshot>,
}

impl ServiceRuntimePublisher for ServiceStatePlanePublisher {
    type Error = StatePlaneError;

    fn publish_activity(&mut self, activity: RuntimeActivity) -> Result<(), Self::Error> {
        self.activity.publish(activity).map(|_| ())
    }

    fn publish_snapshot(&mut self, snapshot: ServiceRuntimeSnapshot) -> Result<(), Self::Error> {
        self.snapshot.publish(snapshot).map(|_| ())
    }
}

/// Read-only subscription capability for the service state planes.
///
/// The hub internally retains publisher handles only so it can create independent
/// receivers. It exposes no publication method, preventing ordinary UI/status code
/// from acquiring the owner-side write capability through this API.
#[derive(Clone)]
pub struct ServiceStatePlaneHub {
    activity: LatestStatePublisher<RuntimeActivity>,
    snapshot: LatestStatePublisher<ServiceRuntimeSnapshot>,
}

impl ServiceStatePlaneHub {
    pub fn subscribe(&self) -> Result<ServiceStatePlaneSubscription, StatePlaneError> {
        Ok(ServiceStatePlaneSubscription {
            activity: self.activity.subscribe()?,
            snapshot: self.snapshot.subscribe()?,
        })
    }
}

/// Independent read cursor over runtime activity and completed cognitive state.
pub struct ServiceStatePlaneSubscription {
    activity: LatestStateReceiver<RuntimeActivity>,
    snapshot: LatestStateReceiver<ServiceRuntimeSnapshot>,
}

impl ServiceStatePlaneSubscription {
    pub fn activity_latest_if_changed(
        &mut self,
    ) -> Result<Option<LatestStateSample<RuntimeActivity>>, StatePlaneError> {
        self.activity.latest_if_changed()
    }

    pub fn snapshot_latest_if_changed(
        &mut self,
    ) -> Result<Option<LatestStateSample<ServiceRuntimeSnapshot>>, StatePlaneError> {
        self.snapshot.latest_if_changed()
    }

    pub fn peek_activity(
        &self,
    ) -> Result<Option<LatestStateSample<RuntimeActivity>>, StatePlaneError> {
        self.activity.peek_latest()
    }

    pub fn peek_snapshot(
        &self,
    ) -> Result<Option<LatestStateSample<ServiceRuntimeSnapshot>>, StatePlaneError> {
        self.snapshot.peek_latest()
    }
}

/// Construct initialized service state planes.
///
/// Both slots are populated before either capability is returned, so a subscriber
/// created through the returned hub starts from an explicit `Idle` activity and a
/// real startup snapshot rather than fabricated defaults.
pub fn service_state_planes(
    initial_snapshot: ServiceRuntimeSnapshot,
) -> Result<(ServiceStatePlanePublisher, ServiceStatePlaneHub), StatePlaneError> {
    let (activity, _initial_activity_receiver) = latest_state_channel();
    let (snapshot, _initial_snapshot_receiver) = latest_state_channel();

    snapshot.publish(initial_snapshot)?;
    activity.publish(RuntimeActivity::Idle)?;

    let hub = ServiceStatePlaneHub {
        activity: activity.clone(),
        snapshot: snapshot.clone(),
    };
    let publisher = ServiceStatePlanePublisher { activity, snapshot };
    Ok((publisher, hub))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use tokio::sync::{Semaphore, oneshot};

    use crate::observed::{
        ObservedServiceHandler, ServiceMutationExecutor, ServiceSnapshotter,
    };
    use crate::{
        CognitiveSummary, PartnershipSummary, ProcessOrigin, ServiceCommandKind,
        ServiceMutationCommand, SnapshotOrigin,
    };
    use symthaea_runtime_owner::{HandlerFuture, OwnerCommandContext, spawn_runtime_owner};

    fn snapshot_initialized(memories: usize) -> ServiceRuntimeSnapshot {
        ServiceRuntimeSnapshot::initialized(
            CognitiveSummary {
                consciousness_level: 0.2,
                self_loops: memories,
                graph_size: memories,
                complexity: 0.2,
                short_term_memories: memories,
                long_term_memories: 0,
            },
            PartnershipSummary {
                stage: "initial".into(),
                trust: 0.0,
                vulnerability: 0.0,
                reciprocity: 0.0,
                phi_dyad: 0.0,
                interactions: 0,
                trajectory_points: 0,
            },
        )
    }

    #[derive(Default)]
    struct TestEngine {
        values: Vec<String>,
    }

    struct BlockingExecutor {
        started: Option<oneshot::Sender<()>>,
        gate: Arc<Semaphore>,
    }

    impl ServiceMutationExecutor<TestEngine> for BlockingExecutor {
        type Reply = usize;

        fn execute<'a>(
            &'a mut self,
            engine: &'a mut TestEngine,
            _context: OwnerCommandContext,
            command: ServiceMutationCommand,
        ) -> HandlerFuture<'a, Self::Reply> {
            Box::pin(async move {
                if let Some(started) = self.started.take() {
                    let _ = started.send(());
                }
                let permit = self.gate.acquire().await.expect("test gate open");
                permit.forget();

                if let ServiceMutationCommand::ProcessText { content, .. } = command {
                    engine.values.push(content);
                }
                engine.values.len()
            })
        }
    }

    struct TestSnapshotter;

    impl ServiceSnapshotter<TestEngine> for TestSnapshotter {
        fn snapshot(
            &mut self,
            engine: &TestEngine,
            context: OwnerCommandContext,
        ) -> ServiceRuntimeSnapshot {
            ServiceRuntimeSnapshot::after_command(
                context,
                CognitiveSummary {
                    consciousness_level: 0.3,
                    self_loops: engine.values.len(),
                    graph_size: engine.values.len(),
                    complexity: 0.3,
                    short_term_memories: engine.values.len(),
                    long_term_memories: 0,
                },
                PartnershipSummary {
                    stage: "test".into(),
                    trust: 0.0,
                    vulnerability: 0.0,
                    reciprocity: 0.0,
                    phi_dyad: 0.0,
                    interactions: 0,
                    trajectory_points: 0,
                },
            )
        }
    }

    #[test]
    fn subscribers_start_from_explicit_idle_and_real_snapshot() {
        let (publisher, hub) = service_state_planes(snapshot_initialized(4)).unwrap();
        let mut subscriber = hub.subscribe().unwrap();

        let activity = subscriber
            .activity_latest_if_changed()
            .unwrap()
            .unwrap();
        let snapshot = subscriber
            .snapshot_latest_if_changed()
            .unwrap()
            .unwrap();

        assert_eq!(*activity.value, RuntimeActivity::Idle);
        assert_eq!(snapshot.value.origin(), SnapshotOrigin::Initialized);
        assert_eq!(snapshot.value.memory_count(), 4);
        drop(publisher);
    }

    #[tokio::test]
    async fn processing_advances_without_fabricating_a_new_cognitive_snapshot() {
        let (publisher, hub) = service_state_planes(snapshot_initialized(2)).unwrap();
        let mut subscriber = hub.subscribe().unwrap();

        // Establish the observer's baseline before the command begins.
        assert_eq!(
            *subscriber
                .activity_latest_if_changed()
                .unwrap()
                .unwrap()
                .value,
            RuntimeActivity::Idle
        );
        let baseline = subscriber
            .snapshot_latest_if_changed()
            .unwrap()
            .unwrap();
        assert_eq!(baseline.value.origin(), SnapshotOrigin::Initialized);
        assert_eq!(baseline.value.memory_count(), 2);

        let (started_tx, started_rx) = oneshot::channel();
        let gate = Arc::new(Semaphore::new(0));
        let executor = BlockingExecutor {
            started: Some(started_tx),
            gate: Arc::clone(&gate),
        };
        let handler = ObservedServiceHandler::new(executor, TestSnapshotter, publisher);
        let (owner, task) = spawn_runtime_owner(TestEngine::default(), handler, 2).unwrap();

        let ticket = owner
            .try_submit(ServiceMutationCommand::ProcessText {
                content: "hello".into(),
                origin: ProcessOrigin::ServiceQuery,
            })
            .unwrap();
        let expected = ticket.sequence();

        // The executor signals only after ObservedServiceHandler has published
        // owner-authenticated Processing activity.
        started_rx.await.unwrap();
        let activity = subscriber
            .activity_latest_if_changed()
            .unwrap()
            .unwrap();
        match activity.value.as_ref() {
            RuntimeActivity::Processing(processing) => {
                assert_eq!(processing.owner_command_seq(), expected);
                assert_eq!(processing.kind(), ServiceCommandKind::ProcessText);
            }
            other => panic!("expected processing activity, got {other:?}"),
        }

        // Cognition has not completed: the last completed snapshot remains the
        // baseline rather than inventing a newer measurement.
        assert!(subscriber.snapshot_latest_if_changed().unwrap().is_none());
        assert_eq!(
            subscriber.peek_snapshot().unwrap().unwrap().value.origin(),
            SnapshotOrigin::Initialized
        );

        gate.add_permits(1);
        let reply = ticket.resolve().await.unwrap();
        assert_eq!(reply.execution, 1);
        assert!(reply.observations_clean());

        let completed = subscriber
            .snapshot_latest_if_changed()
            .unwrap()
            .unwrap();
        assert_eq!(completed.value.origin(), SnapshotOrigin::AfterCommand(expected));
        assert_eq!(completed.value.memory_count(), 1);
        assert_eq!(
            *subscriber
                .activity_latest_if_changed()
                .unwrap()
                .unwrap()
                .value,
            RuntimeActivity::Idle
        );

        drop(owner);
        assert_eq!(task.await.unwrap().commands_completed, 1);
    }
}
