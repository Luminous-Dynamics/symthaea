// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Marks a zome function for type-safe bridge generation.
///
/// This macro:
/// 1. Passes through the original `#[hdk_extern]` or `pub fn` definition.
/// 2. Appends metadata about the function (name, inputs, outputs) to a 
///    build-time registry for the codegen to consume.
///
/// # Attributes
/// * `role` - The hApp role this zome belongs to (e.g., "civic", "finance").
/// * `zome` - The name of the zome (optional, defaults to crate name).
#[proc_macro_attribute]
pub fn mycelix_zome_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    
    // In a full implementation, this would append to a file in OUT_DIR or 
    // a project-local .mycelix/bridge-registry.json.
    // For now, we pass through and emit a dummy trait for metadata discovery.

    let gen = quote! {
        #input_fn

        /// Metadata marker for #fn_name
        #[allow(non_upper_case_globals)]
        const #fn_name: () = ();
    };
    
    gen.into()
}

/// Marks a zome function as requiring a specific 8D Sovereign Profile threshold.
///
/// Automatically injects the `require_civic` call into the function body.
#[proc_macro_attribute]
pub fn sovereign_gated(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input_fn = parse_macro_input!(item as ItemFn);
    let attr_args = attr.to_string(); // e.g., "proposal", "basic"
    
    let requirement = match attr_args.as_str() {
        "basic" => quote! { mycelix_bridge_common::civic_requirement_basic() },
        "proposal" => quote! { mycelix_bridge_common::civic_requirement_proposal() },
        "steward" => quote! { mycelix_bridge_common::civic_requirement_steward() },
        _ => quote! { mycelix_bridge_common::civic_requirement_basic() },
    };

    // Inject validation at start of function
    let block = &input_fn.block;
    let fn_name_str = input_fn.sig.ident.to_string();
    
    input_fn.block = Box::new(syn::parse_quote! {
        {
            mycelix_zome_helpers::require_civic("civic_bridge", &#requirement, #fn_name_str)?;
            #block
        }
    });

    quote! {
        #input_fn
    }.into()
}
