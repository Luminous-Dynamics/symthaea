// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Single-owner command runtime for mutable Symthaea engines.
//!
//! This crate establishes the ownership invariant needed by the unified interface
//! architecture without depending on the concrete `Symthaea` facade:
//!
//! > exactly one spawned task owns the mutable engine; interface clients receive
//! > bounded command handles, never `&mut` access to the engine itself.
//!
//! Admission is separated from completion. `try_submit` is non-blocking and fails
//! explicitly when the bounded mailbox is full or closed. An accepted command
//! returns a [`RuntimeCommandTicket`]; callers may await its result without holding
//! engine ownership or a service-wide mutex.
//!
//! Read-only UI/status traffic should eventually come from the runtime state plane,
//! not from commands that compete with cognition. This owner core intentionally
//! contains no snapshot/event implementation so those delivery semantics remain
//! separate from mutable-state authority.
//!
//! Dropping a ticket does **not** cancel an admitted command. Once accepted, the
//! owner executes it exactly once unless the owner task itself is aborted. Explicit
//! turn/utterance cancellation belongs in the command protocol above this crate.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

/// Boxed command-handler future borrowing the owner and engine for one command.
pub type HandlerFuture<'a, R> = Pin<Box<dyn Future<Output = R> + Send + 'a>>;

/// Runtime-local submission sequence.
///
/// This is transport/control-plane metadata, **not** `EventSeq` and not semantic
/// runtime ordering. Semantic events must still receive their authoritative cursor
/// from the runtime event plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnerCommandSeq(u64);

impl OwnerCommandSeq {
    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for OwnerCommandSeq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Handles one admitted command while holding the only mutable engine reference.
///
/// Implementations may await internally. Because the owner loop invokes this trait
/// sequentially, no second command can concurrently mutate the same engine.
pub trait RuntimeCommandHandler<E, C>: Send + 'static {
    type Reply: Send + 'static;

    fn handle<'a>(
        &'a mut self,
        engine: &'a mut E,
        command: C,
    ) -> HandlerFuture<'a, Self::Reply>
    where
        C: 'a;
}

/// Invalid owner construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeOwnerConfigError {
    ZeroMailboxCapacity,
}

impl fmt::Display for RuntimeOwnerConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMailboxCapacity => write!(f, "runtime owner mailbox capacity must be non-zero"),
        }
    }
}

impl std::error::Error for RuntimeOwnerConfigError {}

/// Non-blocking command admission failure. The rejected command is returned to the
/// caller so higher layers may retry, reject, or apply their own overload policy.
#[derive(Debug)]
pub enum OwnerSubmitError<C> {
    Full(C),
    Closed(C),
    SequenceExhausted(C),
}

impl<C> OwnerSubmitError<C> {
    pub fn into_command(self) -> C {
        match self {
            Self::Full(command) | Self::Closed(command) | Self::SequenceExhausted(command) => command,
        }
    }
}

impl<C> fmt::Display for OwnerSubmitError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full(_) => write!(f, "runtime owner mailbox is full"),
            Self::Closed(_) => write!(f, "runtime owner mailbox is closed"),
            Self::SequenceExhausted(_) => write!(f, "runtime owner submission sequence exhausted"),
        }
    }
}

impl<C: fmt::Debug> std::error::Error for OwnerSubmitError<C> {}

/// An accepted command whose completion may be awaited independently of admission.
pub struct RuntimeCommandTicket<R> {
    sequence: OwnerCommandSeq,
    receiver: oneshot::Receiver<R>,
}

impl<R> RuntimeCommandTicket<R> {
    pub fn sequence(&self) -> OwnerCommandSeq {
        self.sequence
    }

    pub async fn resolve(self) -> Result<R, OwnerCompletionError> {
        self.receiver.await.map_err(|_| OwnerCompletionError::OwnerStopped)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnerCompletionError {
    OwnerStopped,
}

impl fmt::Display for OwnerCompletionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OwnerStopped => write!(f, "runtime owner stopped before replying"),
        }
    }
}

impl std::error::Error for OwnerCompletionError {}

struct CommandEnvelope<C, R> {
    command: C,
    reply: oneshot::Sender<R>,
}

/// Cloneable control handle. Clones do not clone the engine; they all feed the one
/// bounded mailbox owned by the same runtime task.
pub struct RuntimeOwnerHandle<C, R> {
    sender: mpsc::Sender<CommandEnvelope<C, R>>,
    next_sequence: Arc<AtomicU64>,
}

impl<C, R> Clone for RuntimeOwnerHandle<C, R> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            next_sequence: Arc::clone(&self.next_sequence),
        }
    }
}

impl<C, R> RuntimeOwnerHandle<C, R> {
    pub fn mailbox_capacity(&self) -> usize {
        self.sender.max_capacity()
    }

    pub fn remaining_capacity(&self) -> usize {
        self.sender.capacity()
    }

    /// Admit a command only if mailbox capacity is immediately available.
    ///
    /// `try_reserve` is used before allocating a sequence number, so rejected
    /// full/closed submissions do not create artificial holes in this local
    /// control sequence.
    pub fn try_submit(&self, command: C) -> Result<RuntimeCommandTicket<R>, OwnerSubmitError<C>> {
        let permit = match self.sender.try_reserve() {
            Ok(permit) => permit,
            Err(mpsc::error::TrySendError::Full(())) => return Err(OwnerSubmitError::Full(command)),
            Err(mpsc::error::TrySendError::Closed(())) => {
                return Err(OwnerSubmitError::Closed(command));
            }
        };

        // Atomic value is the next sequence to issue. `u64::MAX` is deliberately
        // left unissued so overflow is detectable without wrapping to zero.
        let sequence = match self.next_sequence.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| current.checked_add(1),
        ) {
            Ok(current) => OwnerCommandSeq(current),
            Err(_) => return Err(OwnerSubmitError::SequenceExhausted(command)),
        };

        let (reply, receiver) = oneshot::channel();
        permit.send(CommandEnvelope { command, reply });
        Ok(RuntimeCommandTicket { sequence, receiver })
    }
}

/// Why the owner task exited normally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeOwnerExitReason {
    /// Every command handle was dropped, closing the bounded mailbox.
    MailboxClosed,
}

/// Normal owner-task completion report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeOwnerExit {
    pub reason: RuntimeOwnerExitReason,
    pub commands_completed: u64,
}

/// Spawn one task that exclusively owns `engine` and processes admitted commands
/// sequentially through `handler`.
pub fn spawn_runtime_owner<E, C, H>(
    engine: E,
    mut handler: H,
    mailbox_capacity: usize,
) -> Result<
    (
        RuntimeOwnerHandle<C, H::Reply>,
        JoinHandle<RuntimeOwnerExit>,
    ),
    RuntimeOwnerConfigError,
>
where
    E: Send + 'static,
    C: Send + 'static,
    H: RuntimeCommandHandler<E, C>,
{
    if mailbox_capacity == 0 {
        return Err(RuntimeOwnerConfigError::ZeroMailboxCapacity);
    }

    let (sender, mut receiver) = mpsc::channel::<CommandEnvelope<C, H::Reply>>(mailbox_capacity);
    let handle = RuntimeOwnerHandle {
        sender,
        next_sequence: Arc::new(AtomicU64::new(1)),
    };

    let task = tokio::spawn(async move {
        let mut engine = engine;
        let mut commands_completed = 0_u64;

        while let Some(envelope) = receiver.recv().await {
            let response = handler.handle(&mut engine, envelope.command).await;
            commands_completed = commands_completed.saturating_add(1);
            // Dropping a client ticket abandons the reply, not the command.
            let _ = envelope.reply.send(response);
        }

        RuntimeOwnerExit {
            reason: RuntimeOwnerExitReason::MailboxClosed,
            commands_completed,
        }
    });

    Ok((handle, task))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::{Semaphore, oneshot};

    struct AppendHandler;

    impl RuntimeCommandHandler<Vec<u64>, u64> for AppendHandler {
        type Reply = (usize, Vec<u64>);

        fn handle<'a>(
            &'a mut self,
            engine: &'a mut Vec<u64>,
            command: u64,
        ) -> HandlerFuture<'a, Self::Reply> {
            Box::pin(async move {
                engine.push(command);
                (engine.len(), engine.clone())
            })
        }
    }

    #[tokio::test]
    async fn commands_mutate_one_engine_sequentially() {
        let (handle, task) = spawn_runtime_owner(Vec::new(), AppendHandler, 4).unwrap();

        let first = handle.try_submit(10).unwrap();
        let second = handle.try_submit(20).unwrap();
        let third = handle.try_submit(30).unwrap();

        assert_eq!(first.sequence().get(), 1);
        assert_eq!(second.sequence().get(), 2);
        assert_eq!(third.sequence().get(), 3);

        assert_eq!(first.resolve().await.unwrap(), (1, vec![10]));
        assert_eq!(second.resolve().await.unwrap(), (2, vec![10, 20]));
        assert_eq!(third.resolve().await.unwrap(), (3, vec![10, 20, 30]));

        drop(handle);
        let exit = task.await.unwrap();
        assert_eq!(exit.reason, RuntimeOwnerExitReason::MailboxClosed);
        assert_eq!(exit.commands_completed, 3);
    }

    struct BlockingFirstHandler {
        started: Option<oneshot::Sender<()>>,
        gate: Arc<Semaphore>,
    }

    impl RuntimeCommandHandler<Vec<u64>, u64> for BlockingFirstHandler {
        type Reply = usize;

        fn handle<'a>(
            &'a mut self,
            engine: &'a mut Vec<u64>,
            command: u64,
        ) -> HandlerFuture<'a, Self::Reply> {
            Box::pin(async move {
                if let Some(started) = self.started.take() {
                    let _ = started.send(());
                    let permit = self.gate.acquire().await.expect("test gate open");
                    permit.forget();
                }
                engine.push(command);
                engine.len()
            })
        }
    }

    #[tokio::test]
    async fn full_mailbox_rejects_without_consuming_submission_sequence() {
        let (started_tx, started_rx) = oneshot::channel();
        let gate = Arc::new(Semaphore::new(0));
        let handler = BlockingFirstHandler {
            started: Some(started_tx),
            gate: Arc::clone(&gate),
        };
        let (handle, task) = spawn_runtime_owner(Vec::new(), handler, 1).unwrap();

        let first = handle.try_submit(1).unwrap();
        started_rx.await.unwrap();

        let second = handle.try_submit(2).unwrap();
        let rejected = match handle.try_submit(3) {
            Err(OwnerSubmitError::Full(command)) => command,
            Err(other) => panic!("expected full mailbox, got {other:?}"),
            Ok(_) => panic!("expected full mailbox rejection"),
        };
        assert_eq!(rejected, 3);

        gate.add_permits(1);
        assert_eq!(first.resolve().await.unwrap(), 1);
        assert_eq!(second.resolve().await.unwrap(), 2);

        let third = handle.try_submit(3).unwrap();
        assert_eq!(third.sequence().get(), 3);
        assert_eq!(third.resolve().await.unwrap(), 3);

        drop(handle);
        assert_eq!(task.await.unwrap().commands_completed, 3);
    }

    #[tokio::test]
    async fn dropping_ticket_does_not_cancel_admitted_command() {
        let (handle, task) = spawn_runtime_owner(Vec::new(), AppendHandler, 2).unwrap();
        let abandoned = handle.try_submit(7).unwrap();
        drop(abandoned);

        let observed = handle.try_submit(8).unwrap();
        assert_eq!(observed.resolve().await.unwrap(), (2, vec![7, 8]));

        drop(handle);
        assert_eq!(task.await.unwrap().commands_completed, 2);
    }

    #[test]
    fn zero_capacity_fails_closed_before_spawning() {
        let result = spawn_runtime_owner(Vec::<u64>::new(), AppendHandler, 0);
        assert!(matches!(result, Err(RuntimeOwnerConfigError::ZeroMailboxCapacity)));
    }
}
