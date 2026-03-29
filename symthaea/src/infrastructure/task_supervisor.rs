// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Task Supervisor — Supervised async tasks with pain channel
//!
//! Wraps `tokio::spawn` with panic catching and pain reporting.
//! When a supervised task panics, the supervisor:
//! 1. Catches the panic via `JoinHandle` (tokio catches panics automatically)
//! 2. Reports it through the pain channel as `InfrastructureError::AsyncTaskPanicked`
//! 3. Logs the panic info via tracing
//!
//! This means the organism *feels* when a background task dies,
//! which raises arousal, increases thermodynamic load, and triggers
//! cautious processing via the SomaticErrorBridge.

use super::somatic_error_bridge::{InfrastructureError, PainSender};
use std::future::Future;
use std::time::Duration;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

/// A supervised async task handle with metadata.
struct SupervisedTask {
    name: String,
    handle: JoinHandle<()>,
}

/// Supervisor for background async tasks.
///
/// All tasks spawned through this supervisor are monitored for panics.
/// Tokio catches panics in spawned tasks automatically — this supervisor
/// monitors the `JoinHandle` results and reports panics through the
/// pain channel. On shutdown, all tasks are given a grace period.
pub struct TaskSupervisor {
    tasks: Vec<SupervisedTask>,
    pain_tx: PainSender,
}

impl TaskSupervisor {
    /// Create a new task supervisor with the given pain channel.
    pub fn new(pain_tx: PainSender) -> Self {
        Self {
            tasks: Vec::new(),
            pain_tx,
        }
    }

    /// Spawn a supervised async task.
    ///
    /// The task runs in a separate tokio task. If it panics, tokio catches
    /// the panic and the supervisor wrapper reports it through the pain channel.
    pub fn spawn<F>(&mut self, name: &str, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let pain_tx = self.pain_tx.clone();
        let task_name = name.to_string();
        let task_name_monitor = task_name.clone();

        // Spawn the actual work
        let inner_handle = tokio::spawn(future);

        // Spawn a monitor that watches for panics
        let handle = tokio::spawn(async move {
            match inner_handle.await {
                Ok(()) => {} // Task completed normally
                Err(join_error) => {
                    if join_error.is_panic() {
                        let panic_info = join_error.into_panic();
                        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic_info.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "unknown panic".to_string()
                        };

                        error!(
                            task = task_name_monitor.as_str(),
                            panic_msg = msg.as_str(),
                            "Supervised task panicked"
                        );

                        // Report pain — the organism will feel this
                        let _ = pain_tx.send(InfrastructureError::AsyncTaskPanicked {
                            task_name: task_name_monitor,
                        });
                    } else {
                        warn!(
                            task = task_name_monitor.as_str(),
                            "Supervised task was cancelled"
                        );
                    }
                }
            }
        });

        self.tasks.push(SupervisedTask {
            name: task_name,
            handle,
        });
    }

    /// Number of tracked tasks (including completed ones).
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Graceful shutdown: wait for all tasks to complete within the timeout,
    /// then abort any remaining tasks.
    pub async fn shutdown(&mut self, timeout: Duration) {
        info!(
            task_count = self.tasks.len(),
            timeout_ms = timeout.as_millis() as u64,
            "Task supervisor shutting down"
        );

        for task in self.tasks.drain(..) {
            match tokio::time::timeout(timeout, task.handle).await {
                Ok(Ok(())) => {
                    info!(task = task.name.as_str(), "Task completed cleanly");
                }
                Ok(Err(e)) => {
                    error!(
                        task = task.name.as_str(),
                        error = %e,
                        "Task join error"
                    );
                }
                Err(_) => {
                    warn!(
                        task = task.name.as_str(),
                        "Task timed out on shutdown — aborting"
                    );
                }
            }
        }
    }

    /// Prune completed tasks from the tracked list to free memory.
    pub fn prune_completed(&mut self) {
        self.tasks.retain(|t| !t.handle.is_finished());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn make_supervisor() -> (TaskSupervisor, mpsc::Receiver<InfrastructureError>) {
        let (tx, rx) = mpsc::channel();
        (TaskSupervisor::new(tx), rx)
    }

    #[tokio::test]
    async fn test_spawn_normal_task() {
        let (mut supervisor, _rx) = make_supervisor();
        supervisor.spawn("test-task", async {
            // Normal task that completes successfully
        });
        assert_eq!(supervisor.task_count(), 1);
        supervisor.shutdown(Duration::from_secs(1)).await;
    }

    #[tokio::test]
    async fn test_spawn_panicking_task_reports_pain() {
        let (mut supervisor, rx) = make_supervisor();
        supervisor.spawn("panicking-task", async {
            panic!("intentional test panic");
        });

        // Give the task time to execute and the monitor to report
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Check that pain was reported
        let error = rx.try_recv().expect("Should have received pain signal");
        match error {
            InfrastructureError::AsyncTaskPanicked { task_name } => {
                assert_eq!(task_name, "panicking-task");
            }
            other => panic!("Expected AsyncTaskPanicked, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_shutdown_with_timeout() {
        let (mut supervisor, _rx) = make_supervisor();
        supervisor.spawn("long-task", async {
            tokio::time::sleep(Duration::from_secs(60)).await;
        });

        // Shutdown with short timeout — should not hang
        supervisor.shutdown(Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn test_prune_completed() {
        let (mut supervisor, _rx) = make_supervisor();
        supervisor.spawn("quick-task", async {});

        // Wait for task to complete
        tokio::time::sleep(Duration::from_millis(100)).await;

        supervisor.prune_completed();
        assert_eq!(
            supervisor.task_count(),
            0,
            "Completed task should be pruned"
        );
    }
}
