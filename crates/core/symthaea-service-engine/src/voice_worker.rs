// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded dedicated worker for blocking voice presentation.
//!
//! Presentation jobs are serialized through a finite queue and executed by one
//! renderer-owning OS thread. Barge-in does **not** traverse that queue: callers use
//! the shared [`SessionVoicePresentationControl`] directly, so an interruption can
//! reach the active renderer while the worker is blocked inside synthesis/playback.

use std::fmt;
use std::sync::Arc;
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use symthaea_interface_types::{SessionId, TurnId};
use tokio::sync::oneshot;

use crate::voice_presenter::{
    CancellableRenderOutcome, CancellableVoiceRenderer, VoicePresentationDisposition,
};
use crate::voice_session::{
    VoiceCognitionOutcome, VoiceInterruptionReceipt, VoicePresentationToken, VoiceTurnIdentity,
};
use crate::voice_session_control::{
    SessionVoicePresentationControl, SessionVoiceRegistrationError, SessionVoiceStopOutcome,
    VoicePresentationKey,
};
use crate::wire::{ServiceWireOutcome, ServiceWireResponse};

const PLAYBACK_DRAIN_POLL: Duration = Duration::from_millis(2);
const PLAYBACK_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// One immutable presentation job correlated to its voice session/utterance/turn.
pub struct VoicePresentationJob {
    identity: VoiceTurnIdentity,
    turn_id: Option<TurnId>,
    presentation: VoicePresentationToken,
    outcome: ServiceWireOutcome,
}

impl VoicePresentationJob {
    pub fn from_cognition(outcome: VoiceCognitionOutcome) -> Self {
        Self {
            identity: outcome.identity,
            turn_id: outcome.turn_id,
            presentation: outcome.presentation,
            outcome: outcome.outcome,
        }
    }

    /// Explicit constructor for adapters/tests that already own the correlated
    /// voice components. Registration still verifies identity/token generation.
    pub fn from_parts(
        identity: VoiceTurnIdentity,
        turn_id: Option<TurnId>,
        presentation: VoicePresentationToken,
        outcome: ServiceWireOutcome,
    ) -> Self {
        Self {
            identity,
            turn_id,
            presentation,
            outcome,
        }
    }

    pub fn identity(&self) -> &VoiceTurnIdentity {
        &self.identity
    }

    pub fn turn_id(&self) -> Option<&TurnId> {
        self.turn_id.as_ref()
    }

    pub fn generation(&self) -> u64 {
        self.presentation.generation()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoicePresentationWorkerConfigError {
    ZeroCapacity,
}

impl fmt::Display for VoicePresentationWorkerConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => write!(f, "voice presentation worker capacity must be positive"),
        }
    }
}

impl std::error::Error for VoicePresentationWorkerConfigError {}

pub enum VoicePresentationSubmitError {
    Busy(VoicePresentationJob),
    Closed(VoicePresentationJob),
}

impl fmt::Debug for VoicePresentationSubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy(job) => f
                .debug_tuple("Busy")
                .field(&job.identity.session_id)
                .field(&job.generation())
                .finish(),
            Self::Closed(job) => f
                .debug_tuple("Closed")
                .field(&job.identity.session_id)
                .field(&job.generation())
                .finish(),
        }
    }
}

#[derive(Debug)]
pub enum VoicePresentationWorkerError<E> {
    Registration(SessionVoiceRegistrationError),
    Renderer(E),
    PlaybackDrainTimeout {
        session_id: SessionId,
        generation: u64,
    },
}

impl<E: fmt::Display> fmt::Display for VoicePresentationWorkerError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registration(error) => write!(f, "voice presentation registration failed: {error}"),
            Self::Renderer(error) => write!(f, "voice renderer failed: {error}"),
            Self::PlaybackDrainTimeout {
                session_id,
                generation,
            } => write!(
                f,
                "voice presentation {session_id} generation {generation} did not drain software playback within {:?}",
                PLAYBACK_DRAIN_TIMEOUT
            ),
        }
    }
}

impl<E> std::error::Error for VoicePresentationWorkerError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Registration(error) => Some(error),
            Self::Renderer(error) => Some(error),
            Self::PlaybackDrainTimeout { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum VoicePresentationTicketError<E> {
    Presentation(VoicePresentationWorkerError<E>),
    WorkerStopped,
}

impl<E: fmt::Display> fmt::Display for VoicePresentationTicketError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Presentation(error) => error.fmt(f),
            Self::WorkerStopped => write!(f, "voice presentation worker stopped before replying"),
        }
    }
}

impl<E> std::error::Error for VoicePresentationTicketError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Presentation(error) => Some(error),
            Self::WorkerStopped => None,
        }
    }
}

struct WorkerJob<E> {
    job: VoicePresentationJob,
    reply: oneshot::Sender<Result<VoicePresentationDisposition, VoicePresentationWorkerError<E>>>,
}

/// Completion observation for one admitted presentation.
///
/// Dropping this ticket does not cancel the admitted job. The renderer-owning worker
/// still executes it exactly once; barge-in is an explicit independent capability.
pub struct VoicePresentationTicket<E> {
    session_id: SessionId,
    generation: u64,
    turn_id: Option<TurnId>,
    reply: oneshot::Receiver<Result<VoicePresentationDisposition, VoicePresentationWorkerError<E>>>,
}

impl<E> VoicePresentationTicket<E> {
    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn turn_id(&self) -> Option<&TurnId> {
        self.turn_id.as_ref()
    }

    pub async fn resolve(
        self,
    ) -> Result<VoicePresentationDisposition, VoicePresentationTicketError<E>> {
        match self.reply.await {
            Ok(result) => result.map_err(VoicePresentationTicketError::Presentation),
            Err(_) => Err(VoicePresentationTicketError::WorkerStopped),
        }
    }
}

/// Cloneable bounded admission plus queue-independent interruption capability.
pub struct VoicePresentationWorkerHandle<E> {
    sender: SyncSender<WorkerJob<E>>,
    control: SessionVoicePresentationControl,
    capacity: usize,
}

impl<E> Clone for VoicePresentationWorkerHandle<E> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            control: self.control.clone(),
            capacity: self.capacity,
        }
    }
}

impl<E: Send + 'static> VoicePresentationWorkerHandle<E> {
    pub fn try_submit(
        &self,
        job: VoicePresentationJob,
    ) -> Result<VoicePresentationTicket<E>, VoicePresentationSubmitError> {
        let session_id = job.identity.session_id.clone();
        let generation = job.presentation.generation();
        let turn_id = job.turn_id.clone();
        let (reply_tx, reply_rx) = oneshot::channel();
        let worker_job = WorkerJob {
            job,
            reply: reply_tx,
        };

        match self.sender.try_send(worker_job) {
            Ok(()) => Ok(VoicePresentationTicket {
                session_id,
                generation,
                turn_id,
                reply: reply_rx,
            }),
            Err(TrySendError::Full(worker_job)) => {
                Err(VoicePresentationSubmitError::Busy(worker_job.job))
            }
            Err(TrySendError::Disconnected(worker_job)) => {
                Err(VoicePresentationSubmitError::Closed(worker_job.job))
            }
        }
    }

    /// Fast barge-in path. This never enters the bounded presentation queue.
    pub fn apply_interruption(
        &self,
        receipt: &VoiceInterruptionReceipt,
    ) -> SessionVoiceStopOutcome {
        self.control.apply_interruption(receipt)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn active_key(&self) -> Option<VoicePresentationKey> {
        self.control.active_key()
    }
}

pub struct VoicePresentationWorkerRuntime<E> {
    pub handle: VoicePresentationWorkerHandle<E>,
    pub task: JoinHandle<()>,
}

fn speakable_content(outcome: &ServiceWireOutcome) -> Option<&str> {
    match &outcome.response {
        ServiceWireResponse::QueryResponse { content, .. } if !content.trim().is_empty() => {
            Some(content)
        }
        _ => None,
    }
}

fn present_one<R>(
    renderer: &mut R,
    control: &SessionVoicePresentationControl,
    job: &VoicePresentationJob,
) -> Result<VoicePresentationDisposition, VoicePresentationWorkerError<R::Error>>
where
    R: CancellableVoiceRenderer,
{
    if job.presentation.is_cancelled() {
        return Ok(VoicePresentationDisposition::SuppressedCancelled);
    }
    let Some(content) = speakable_content(&job.outcome) else {
        return Ok(VoicePresentationDisposition::NoSpeakableResponse);
    };

    let generation = job.presentation.generation();
    let stop = renderer.stop_capability();
    let playback = Arc::clone(&stop);
    let _lease = match control.try_register(&job.identity, &job.presentation, stop) {
        Ok(lease) => lease,
        Err(SessionVoiceRegistrationError::Cancelled { .. }) => {
            return Ok(VoicePresentationDisposition::SuppressedCancelled);
        }
        Err(error) => return Err(VoicePresentationWorkerError::Registration(error)),
    };

    if job.presentation.is_cancelled() {
        playback.stop();
        return Ok(VoicePresentationDisposition::SuppressedCancelled);
    }

    let cancelled = || job.presentation.is_cancelled();
    match renderer
        .speak_cancellable(content, &cancelled)
        .map_err(VoicePresentationWorkerError::Renderer)?
    {
        CancellableRenderOutcome::Cancelled => {
            playback.stop();
            return Ok(VoicePresentationDisposition::CancelledDuringPresentation);
        }
        CancellableRenderOutcome::Completed => {}
    }

    let drain_started = Instant::now();
    while playback.is_speaking() {
        if job.presentation.is_cancelled() {
            playback.stop();
            return Ok(VoicePresentationDisposition::CancelledDuringPresentation);
        }
        if drain_started.elapsed() >= PLAYBACK_DRAIN_TIMEOUT {
            playback.stop();
            return Err(VoicePresentationWorkerError::PlaybackDrainTimeout {
                session_id: job.identity.session_id.clone(),
                generation,
            });
        }
        thread::sleep(PLAYBACK_DRAIN_POLL);
    }

    Ok(VoicePresentationDisposition::Presented)
}

pub fn spawn_voice_presentation_worker<R>(
    mut renderer: R,
    capacity: usize,
) -> Result<VoicePresentationWorkerRuntime<R::Error>, VoicePresentationWorkerConfigError>
where
    R: CancellableVoiceRenderer + Send + 'static,
    R::Error: Send + 'static,
{
    if capacity == 0 {
        return Err(VoicePresentationWorkerConfigError::ZeroCapacity);
    }

    let (sender, receiver) = sync_channel::<WorkerJob<R::Error>>(capacity);
    let control = SessionVoicePresentationControl::default();
    let worker_control = control.clone();
    let task = thread::Builder::new()
        .name("symthaea-voice-presenter".into())
        .spawn(move || {
            while let Ok(worker_job) = receiver.recv() {
                let result = present_one(&mut renderer, &worker_control, &worker_job.job);
                let _ = worker_job.reply.send(result);
            }
        })
        .expect("failed to spawn voice presentation worker");

    Ok(VoicePresentationWorkerRuntime {
        handle: VoicePresentationWorkerHandle {
            sender,
            control,
            capacity,
        },
        task,
    })
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;
    use crate::voice_control::VoiceStopCapability;
    use crate::voice_session::{VoiceIngress, VoiceSession};
    use crate::wire::ServiceWireDiagnostics;

    struct FakeStop {
        speaking: AtomicBool,
        stops: AtomicUsize,
    }

    impl FakeStop {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                speaking: AtomicBool::new(false),
                stops: AtomicUsize::new(0),
            })
        }
    }

    impl VoiceStopCapability for FakeStop {
        fn stop(&self) {
            self.stops.fetch_add(1, Ordering::SeqCst);
            self.speaking.store(false, Ordering::SeqCst);
        }

        fn is_speaking(&self) -> bool {
            self.speaking.load(Ordering::SeqCst)
        }
    }

    struct BlockingRenderer {
        stop: Arc<FakeStop>,
        entered: Arc<Barrier>,
        release: Arc<AtomicBool>,
        calls: Arc<AtomicUsize>,
    }

    impl CancellableVoiceRenderer for BlockingRenderer {
        type Error = Infallible;

        fn stop_capability(&self) -> Arc<dyn VoiceStopCapability> {
            let capability: Arc<dyn VoiceStopCapability> = self.stop.clone();
            capability
        }

        fn speak_cancellable(
            &mut self,
            _text: &str,
            cancelled: &dyn Fn() -> bool,
        ) -> Result<CancellableRenderOutcome, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.stop.speaking.store(true, Ordering::SeqCst);
            if !self.release.load(Ordering::Acquire) {
                self.entered.wait();
            }
            while !self.release.load(Ordering::Acquire)
                && !cancelled()
                && self.stop.is_speaking()
            {
                thread::yield_now();
            }
            let was_cancelled = cancelled() || !self.stop.is_speaking();
            self.stop.speaking.store(false, Ordering::SeqCst);
            Ok(if was_cancelled {
                CancellableRenderOutcome::Cancelled
            } else {
                CancellableRenderOutcome::Completed
            })
        }
    }

    fn query_outcome(content: &str) -> ServiceWireOutcome {
        ServiceWireOutcome {
            response: ServiceWireResponse::QueryResponse {
                content: content.to_string(),
                confidence: 0.9,
                safe: true,
                phi: 0.4,
                phi_dyad: 0.0,
                steps_to_emergence: 0,
                processing_time_ms: 1,
                creative_artifact: None,
            },
            diagnostics: ServiceWireDiagnostics::default(),
            bridge_telemetry: None,
        }
    }

    fn job(session: &VoiceSession, content: &str) -> VoicePresentationJob {
        let turn = session.begin_turn(VoiceIngress::LiveMicrophone).unwrap();
        VoicePresentationJob::from_parts(
            turn.identity().clone(),
            None,
            turn.presentation_token().clone(),
            query_outcome(content),
        )
    }

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[test]
    fn zero_capacity_is_rejected() {
        let result = spawn_voice_presentation_worker(
            BlockingRenderer {
                stop: FakeStop::new(),
                entered: Arc::new(Barrier::new(1)),
                release: Arc::new(AtomicBool::new(true)),
                calls: Arc::new(AtomicUsize::new(0)),
            },
            0,
        );
        assert!(matches!(result, Err(VoicePresentationWorkerConfigError::ZeroCapacity)));
    }

    #[test]
    fn bounded_queue_reports_busy_without_blocking() {
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(AtomicBool::new(false));
        let runtime = spawn_voice_presentation_worker(
            BlockingRenderer {
                stop: FakeStop::new(),
                entered: entered.clone(),
                release: release.clone(),
                calls: Arc::new(AtomicUsize::new(0)),
            },
            1,
        )
        .unwrap();
        let session = VoiceSession::with_session_id(SessionId::new("voice-session:q").unwrap());

        let first = runtime.handle.try_submit(job(&session, "one")).unwrap();
        entered.wait();
        let second = runtime.handle.try_submit(job(&session, "two")).unwrap();
        assert!(matches!(
            runtime.handle.try_submit(job(&session, "three")),
            Err(VoicePresentationSubmitError::Busy(_))
        ));

        release.store(true, Ordering::Release);
        drop(first);
        drop(second);
        drop(runtime.handle);
        runtime.task.join().unwrap();
    }

    #[test]
    fn barge_in_bypasses_worker_queue_and_stops_active_renderer() {
        let stop = FakeStop::new();
        let entered = Arc::new(Barrier::new(2));
        let runtime = spawn_voice_presentation_worker(
            BlockingRenderer {
                stop: stop.clone(),
                entered: entered.clone(),
                release: Arc::new(AtomicBool::new(false)),
                calls: Arc::new(AtomicUsize::new(0)),
            },
            1,
        )
        .unwrap();
        let session = VoiceSession::with_session_id(SessionId::new("voice-session:barge").unwrap());
        let ticket = runtime.handle.try_submit(job(&session, "long speech")).unwrap();
        entered.wait();

        let receipt = session.interrupt_presentation();
        assert_eq!(
            runtime.handle.apply_interruption(&receipt),
            SessionVoiceStopOutcome::StopRequested {
                session_id: SessionId::new("voice-session:barge").unwrap(),
                generation: 1,
                was_speaking: true,
            }
        );
        assert_eq!(stop.stops.load(Ordering::SeqCst), 1);
        assert_eq!(
            block_on(ticket.resolve()).unwrap(),
            VoicePresentationDisposition::CancelledDuringPresentation
        );

        drop(runtime.handle);
        runtime.task.join().unwrap();
    }

    #[test]
    fn dropping_ticket_does_not_cancel_admitted_presentation() {
        let calls = Arc::new(AtomicUsize::new(0));
        let runtime = spawn_voice_presentation_worker(
            BlockingRenderer {
                stop: FakeStop::new(),
                entered: Arc::new(Barrier::new(1)),
                release: Arc::new(AtomicBool::new(true)),
                calls: calls.clone(),
            },
            1,
        )
        .unwrap();
        let session = VoiceSession::with_session_id(SessionId::new("voice-session:drop").unwrap());
        drop(runtime.handle.try_submit(job(&session, "still speak")).unwrap());

        drop(runtime.handle);
        runtime.task.join().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
