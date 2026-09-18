// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic owner-side wrapper that orders service activity and completed snapshots
//! around one mutable command execution.

use symthaea_runtime_owner::{
    HandlerFuture, OwnerCommandContext, OwnerCommandSeq, RuntimeCommandHandler,
};

use crate::{
    RuntimeActivity, ServiceCommandKind, ServiceMutationCommand, ServiceRuntimeSnapshot,
    SnapshotOrigin,
};

/// Stage at which an interface-state publication failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicationStage {
    StartActivity,
    Snapshot,
    FinishActivity,
}

/// Observation degradation is reported separately from cognitive execution.
#[derive(Debug)]
pub enum ObservationIssue<E> {
    Publication {
        stage: PublicationStage,
        error: E,
    },
    SnapshotCorrelation {
        expected: OwnerCommandSeq,
        observed: SnapshotOrigin,
    },
}

/// Result returned by the observed handler.
///
/// `execution` is the executor's domain result. `observation_issues` records state
/// publication/correlation failures without pretending the cognitive command did
/// not execute.
#[derive(Debug)]
pub struct ObservedServiceReply<R, E> {
    pub execution: R,
    pub observation_issues: Vec<ObservationIssue<E>>,
}

impl<R, E> ObservedServiceReply<R, E> {
    pub fn observations_clean(&self) -> bool {
        self.observation_issues.is_empty()
    }
}

/// Concrete command execution against the sole mutable engine.
pub trait ServiceMutationExecutor<E>: Send + 'static {
    type Reply: Send + 'static;

    fn execute<'a>(
        &'a mut self,
        engine: &'a mut E,
        context: OwnerCommandContext,
        command: ServiceMutationCommand,
    ) -> HandlerFuture<'a, Self::Reply>;
}

/// Extract one immutable completed snapshot after command execution returns.
pub trait ServiceSnapshotter<E>: Send + 'static {
    fn snapshot(
        &mut self,
        engine: &E,
        context: OwnerCommandContext,
    ) -> ServiceRuntimeSnapshot;
}

/// Publication target for fast activity and completed cognitive state.
///
/// The interface plane may implement this with latest-wins state channels. Errors
/// are never erased; the wrapper returns them alongside the execution result.
pub trait ServiceRuntimePublisher: Send + 'static {
    type Error: Send + 'static;

    fn publish_activity(&mut self, activity: RuntimeActivity) -> Result<(), Self::Error>;

    fn publish_snapshot(&mut self, snapshot: ServiceRuntimeSnapshot) -> Result<(), Self::Error>;
}

/// Composes command execution, snapshot extraction, and publication under the sole
/// mutable owner task.
pub struct ObservedServiceHandler<X, S, P> {
    executor: X,
    snapshotter: S,
    publisher: P,
}

impl<X, S, P> ObservedServiceHandler<X, S, P> {
    pub fn new(executor: X, snapshotter: S, publisher: P) -> Self {
        Self {
            executor,
            snapshotter,
            publisher,
        }
    }

    pub fn into_parts(self) -> (X, S, P) {
        (self.executor, self.snapshotter, self.publisher)
    }
}

impl<E, X, S, P> RuntimeCommandHandler<E, ServiceMutationCommand>
    for ObservedServiceHandler<X, S, P>
where
    E: Send + 'static,
    X: ServiceMutationExecutor<E>,
    S: ServiceSnapshotter<E>,
    P: ServiceRuntimePublisher,
{
    type Reply = ObservedServiceReply<X::Reply, P::Error>;

    fn handle<'a>(
        &'a mut self,
        engine: &'a mut E,
        context: OwnerCommandContext,
        command: ServiceMutationCommand,
    ) -> HandlerFuture<'a, Self::Reply> {
        let kind = command.kind();

        Box::pin(async move {
            let mut observation_issues = Vec::new();

            let start_activity = if kind == ServiceCommandKind::ShutdownPersist {
                RuntimeActivity::shutting_down(context)
            } else {
                RuntimeActivity::processing(context, kind)
            };
            if let Err(error) = self.publisher.publish_activity(start_activity) {
                observation_issues.push(ObservationIssue::Publication {
                    stage: PublicationStage::StartActivity,
                    error,
                });
            }

            let execution = self.executor.execute(engine, context, command).await;
            let snapshot = self.snapshotter.snapshot(engine, context);
            let expected = context.sequence();

            if snapshot.origin() == SnapshotOrigin::AfterCommand(expected) {
                if let Err(error) = self.publisher.publish_snapshot(snapshot) {
                    observation_issues.push(ObservationIssue::Publication {
                        stage: PublicationStage::Snapshot,
                        error,
                    });
                }
            } else {
                observation_issues.push(ObservationIssue::SnapshotCorrelation {
                    expected,
                    observed: snapshot.origin(),
                });
            }

            // Shutdown remains terminally visible rather than briefly claiming Idle
            // between persistence completion and process exit.
            if kind != ServiceCommandKind::ShutdownPersist
                && let Err(error) = self.publisher.publish_activity(RuntimeActivity::Idle)
            {
                observation_issues.push(ObservationIssue::Publication {
                    stage: PublicationStage::FinishActivity,
                    error,
                });
            }

            ObservedServiceReply {
                execution,
                observation_issues,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::sync::{Arc, Mutex};

    use crate::{CognitiveSummary, PartnershipSummary, ProcessOrigin};
    use symthaea_runtime_owner::spawn_runtime_owner;

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Record {
        Activity(RuntimeActivity),
        Snapshot(SnapshotOrigin),
    }

    #[derive(Clone)]
    struct RecordingPublisher {
        records: Arc<Mutex<Vec<Record>>>,
    }

    impl ServiceRuntimePublisher for RecordingPublisher {
        type Error = Infallible;

        fn publish_activity(&mut self, activity: RuntimeActivity) -> Result<(), Self::Error> {
            self.records
                .lock()
                .unwrap()
                .push(Record::Activity(activity));
            Ok(())
        }

        fn publish_snapshot(
            &mut self,
            snapshot: ServiceRuntimeSnapshot,
        ) -> Result<(), Self::Error> {
            self.records
                .lock()
                .unwrap()
                .push(Record::Snapshot(snapshot.origin()));
            Ok(())
        }
    }

    #[derive(Default)]
    struct TestEngine {
        values: Vec<String>,
    }

    struct TestExecutor;

    impl ServiceMutationExecutor<TestEngine> for TestExecutor {
        type Reply = usize;

        fn execute<'a>(
            &'a mut self,
            engine: &'a mut TestEngine,
            _context: OwnerCommandContext,
            command: ServiceMutationCommand,
        ) -> HandlerFuture<'a, Self::Reply> {
            Box::pin(async move {
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
                    consciousness_level: 0.1,
                    self_loops: engine.values.len(),
                    graph_size: engine.values.len(),
                    complexity: 0.1,
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

    #[tokio::test]
    async fn publication_order_completes_before_ticket_resolution() {
        let records = Arc::new(Mutex::new(Vec::new()));
        let publisher = RecordingPublisher {
            records: Arc::clone(&records),
        };
        let handler = ObservedServiceHandler::new(TestExecutor, TestSnapshotter, publisher);
        let (owner, task) = spawn_runtime_owner(TestEngine::default(), handler, 2).unwrap();

        let ticket = owner
            .try_submit(ServiceMutationCommand::ProcessText {
                content: "hello".into(),
                origin: ProcessOrigin::ServiceQuery,
            })
            .unwrap();
        let expected = ticket.sequence();
        let reply = ticket.resolve().await.unwrap();

        assert_eq!(reply.execution, 1);
        assert!(reply.observations_clean());

        let observed = records.lock().unwrap();
        assert_eq!(observed.len(), 3);
        match observed[0] {
            Record::Activity(RuntimeActivity::Processing(activity)) => {
                assert_eq!(activity.owner_command_seq(), expected);
                assert_eq!(activity.kind(), ServiceCommandKind::ProcessText);
            }
            _ => panic!("expected owner-bound processing activity first"),
        }
        assert_eq!(
            observed[1],
            Record::Snapshot(SnapshotOrigin::AfterCommand(expected))
        );
        assert_eq!(observed[2], Record::Activity(RuntimeActivity::Idle));
        drop(observed);

        drop(owner);
        assert_eq!(task.await.unwrap().commands_completed, 1);
    }

    struct BadSnapshotter;

    impl ServiceSnapshotter<TestEngine> for BadSnapshotter {
        fn snapshot(
            &mut self,
            _engine: &TestEngine,
            _context: OwnerCommandContext,
        ) -> ServiceRuntimeSnapshot {
            ServiceRuntimeSnapshot::initialized(
                CognitiveSummary {
                    consciousness_level: 0.0,
                    self_loops: 0,
                    graph_size: 0,
                    complexity: 0.0,
                    short_term_memories: 0,
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

    #[tokio::test]
    async fn mismatched_snapshot_is_not_published_as_post_command_state() {
        let records = Arc::new(Mutex::new(Vec::new()));
        let publisher = RecordingPublisher {
            records: Arc::clone(&records),
        };
        let handler = ObservedServiceHandler::new(TestExecutor, BadSnapshotter, publisher);
        let (owner, task) = spawn_runtime_owner(TestEngine::default(), handler, 1).unwrap();

        let ticket = owner
            .try_submit(ServiceMutationCommand::ProcessText {
                content: "hello".into(),
                origin: ProcessOrigin::ServiceQuery,
            })
            .unwrap();
        let expected = ticket.sequence();
        let reply = ticket.resolve().await.unwrap();

        assert_eq!(reply.observation_issues.len(), 1);
        assert!(matches!(
            &reply.observation_issues[0],
            ObservationIssue::SnapshotCorrelation {
                expected: seen,
                observed: SnapshotOrigin::Initialized,
            } if *seen == expected
        ));
        assert!(
            records
                .lock()
                .unwrap()
                .iter()
                .all(|record| !matches!(record, Record::Snapshot(_)))
        );

        drop(owner);
        assert_eq!(task.await.unwrap().commands_completed, 1);
    }
}
