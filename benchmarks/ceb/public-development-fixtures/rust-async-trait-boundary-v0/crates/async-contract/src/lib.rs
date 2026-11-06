//! Object-capable async service contract for the public CEB development fixture.

use std::future::Future;
use std::pin::Pin;

use ceb_async_boundary_domain_types::{DeliveryReceipt, Event, ServiceError};

/// Boxed future alias.
///
/// The development task is to qualify and strengthen the Send contract while
/// preserving the borrow lifetime and trait-object call shape.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// An asynchronous event sink with explicit associated result/error types.
pub trait EventSink: Send + Sync {
    type Receipt: Send;
    type Error: Send + Sync;

    fn submit<'a>(
        &'a self,
        event: Event,
    ) -> BoxFuture<'a, Result<Self::Receipt, Self::Error>>;
}

/// A concrete erased service shape used by the downstream adapter.
pub type ErasedEventSink =
    dyn EventSink<Receipt = DeliveryReceipt, Error = ServiceError> + Send + Sync;
