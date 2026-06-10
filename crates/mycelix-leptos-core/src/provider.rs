// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Holochain context provider and reactive hooks.
//!
//! [`HolochainProvider`] injects a [`HolochainTransport`] into the Leptos
//! reactive context so that any descendant component can access it via
//! [`use_holochain`] without prop drilling.
//!
//! The transport is stored via `provide_context` which requires `Send + Sync`.
//! Since [`HolochainTransport::call_zome`] returns a non-`Send` future (it
//! uses browser WebSocket internally), [`use_zome_call`] uses
//! `Action::new_unsync` to dispatch calls on the current thread.

use leptos::prelude::*;
use mycelix_leptos_client::{ClientError, HolochainTransport};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Context provider that makes a [`HolochainTransport`] available to all
/// child components.
///
/// The transport type must be `Send + Sync` to satisfy Leptos's
/// `provide_context` requirements.
///
/// # Example
///
/// ```rust,no_run
/// use mycelix_leptos_core::HolochainProvider;
///
/// // In your root App component:
/// // view! {
/// //     <HolochainProvider transport=my_transport>
/// //         <Router />
/// //     </HolochainProvider>
/// // }
/// ```
#[component]
pub fn HolochainProvider<T: HolochainTransport + Clone + Send + Sync + 'static>(
    transport: T,
    children: Children,
) -> impl IntoView {
    provide_context(transport);
    children()
}

/// Retrieve the [`HolochainTransport`] from the nearest [`HolochainProvider`].
///
/// # Panics
///
/// Panics if called outside a `<HolochainProvider>`. This is intentional —
/// a missing provider is a programming error, not a runtime condition.
pub fn use_holochain<T: HolochainTransport + Clone + Send + Sync + 'static>() -> T {
    expect_context::<T>()
}

/// Return value from [`use_zome_call`], providing reactive access to
/// zome call state.
pub struct ZomeCallHandle<R> {
    /// The result of the most recent call, or `None` if no call has completed.
    pub data: ReadSignal<Option<Result<R, String>>>,
    /// Whether a call is currently in flight.
    pub loading: ReadSignal<bool>,
    /// Trigger function — call this to execute the zome call.
    ///
    /// Uses `Action::new_unsync` since `HolochainTransport::call_zome`
    /// returns a non-`Send` future (browser WebSocket).
    pub call: Action<(), Result<R, String>>,
}

/// Reactive helper for executing a zome call with automatic serialization.
///
/// Returns a [`ZomeCallHandle`] with signals for data, loading state, and
/// a trigger action. Uses `Action::new_unsync` internally because the
/// transport's futures are not `Send` (browser WASM constraint).
///
/// # Arguments
///
/// * `role_name` — Role from the hApp manifest (e.g. "governance")
/// * `zome_name` — Zome within the DNA (e.g. "agora")
/// * `fn_name` — Exported function name (e.g. "create_proposal")
/// * `payload` — The input payload to serialize and send
///
/// # Example
///
/// ```rust,no_run
/// use mycelix_leptos_core::use_zome_call;
///
/// // let handle = use_zome_call::<MyTransport, MyInput, MyOutput>(
/// //     "governance", "agora", "get_proposals", &()
/// // );
/// // handle.call.dispatch(());  // trigger the call
/// ```
pub fn use_zome_call<T, I, R>(
    role_name: &'static str,
    zome_name: &'static str,
    fn_name: &'static str,
    payload: I,
) -> ZomeCallHandle<R>
where
    T: HolochainTransport + Clone + Send + Sync + 'static,
    I: Serialize + Clone + 'static,
    R: DeserializeOwned + Clone + Send + Sync + 'static,
{
    let transport = use_holochain::<T>();
    let (data, set_data) = signal(None::<Result<R, String>>);
    let (loading, set_loading) = signal(false);

    // Use new_unsync because HolochainTransport::call_zome returns a
    // non-Send future (Pin<Box<dyn Future>>, no Send bound).
    let call = Action::new_unsync(move |_: &()| {
        let transport = transport.clone();
        let payload = payload.clone();
        async move {
            set_loading.set(true);
            let result: Result<R, String> = async {
                let encoded = mycelix_leptos_client::encode(&payload).map_err(|e| e.to_string())?;
                let response_bytes = transport
                    .call_zome(role_name, zome_name, fn_name, encoded)
                    .await
                    .map_err(|e: ClientError| e.to_string())?;
                let decoded: R =
                    mycelix_leptos_client::decode(&response_bytes).map_err(|e| e.to_string())?;
                Ok(decoded)
            }
            .await;
            set_loading.set(false);
            if let Ok(ref val) = result {
                set_data.set(Some(Ok(val.clone())));
            } else if let Err(ref e) = result {
                set_data.set(Some(Err(e.clone())));
            }
            result
        }
    });

    ZomeCallHandle {
        data,
        loading,
        call,
    }
}
