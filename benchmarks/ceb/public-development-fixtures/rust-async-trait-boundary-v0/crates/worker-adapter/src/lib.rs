//! Erased service consumer for the public CEB async-trait development fixture.

use ceb_async_boundary_contract::{BoxFuture, ErasedEventSink};
use ceb_async_boundary_domain_types::{DeliveryReceipt, Event, ServiceError};

/// Submit through the trait-object boundary while borrowing the service.
///
/// The task contract requires the returned future to be Send without imposing
/// an undocumented 'static bound on this borrow.
pub fn submit_erased<'a>(
    sink: &'a ErasedEventSink,
    event: Event,
) -> BoxFuture<'a, Result<DeliveryReceipt, ServiceError>> {
    sink.submit(event)
}
