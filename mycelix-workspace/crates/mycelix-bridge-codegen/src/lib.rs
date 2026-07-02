// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::Result;
use quote::{format_ident, quote};
use std::path::Path;
use syn::{
    ItemFn, Meta, ReturnType, Type,
    visit::{self, Visit},
};
use walkdir::WalkDir;

/// Metadata for a single zome function.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZomeFnMetadata {
    pub name: String,
    pub input_type: String,
    pub output_type: String,
    pub role: String,
    pub zome: String,
}

/// Visitor to find and extract zome functions.
struct ZomeFnVisitor {
    fns: Vec<ZomeFnMetadata>,
    current_role: String,
    current_zome: String,
}

impl<'ast> Visit<'ast> for ZomeFnVisitor {
    fn visit_item_fn(&mut self, i: &'ast ItemFn) {
        let is_zome_fn = i.attrs.iter().any(|attr| {
            if let Meta::Path(path) = &attr.meta {
                let p = path.segments.last().unwrap().ident.to_string();
                p == "hdk_extern" || p == "mycelix_zome_fn"
            } else {
                false
            }
        });

        if is_zome_fn {
            let name = i.sig.ident.to_string();

            // Extract input type (assume first argument if present)
            let input_type = if let Some(syn::FnArg::Typed(pat)) = i.sig.inputs.first() {
                type_to_string(&pat.ty)
            } else {
                "()".to_string()
            };

            // Extract output type (unwrap ExternResult<T> -> T)
            let output_type = match &i.sig.output {
                ReturnType::Default => "()".to_string(),
                ReturnType::Type(_, ty) => match unwrap_extern_result(ty) {
                    Some(inner) => type_to_string(inner),
                    None => type_to_string(ty),
                },
            };

            self.fns.push(ZomeFnMetadata {
                name,
                input_type,
                output_type,
                role: self.current_role.clone(),
                zome: self.current_zome.clone(),
            });
        }

        visit::visit_item_fn(self, i);
    }
}

fn type_to_string(ty: &Type) -> String {
    quote!(#ty).to_string().replace(' ', "")
}

/// If `ty` is `ExternResult<T>`, return `Some(&T)`. Otherwise `None`.
///
/// Structural (via syn's parsed type), not string manipulation — a prior
/// version did `s.replace("ExternResult<", "").trim_end_matches('>')`,
/// which corrupts any nested-generic inner type (e.g. `ExternResult<Option<T>>`
/// -> the trailing `trim_end_matches('>')` strips both closing brackets,
/// producing the unparseable `Option<T` and panicking downstream in
/// `generate_leptos_bridge`'s `syn::parse_str::<Type>(..).unwrap()`).
fn unwrap_extern_result(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "ExternResult" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        syn::GenericArgument::Type(t) => Some(t),
        _ => None,
    })
}

pub fn scan_zome_dir(path: &Path, role: &str, zome: &str) -> Result<Vec<ZomeFnMetadata>> {
    let mut visitor = ZomeFnVisitor {
        fns: Vec::new(),
        current_role: role.to_string(),
        current_zome: zome.to_string(),
    };

    for entry in WalkDir::new(path) {
        let entry = entry?;
        if entry.path().extension().and_then(|s| s.to_str()) == Some("rs") {
            let content = std::fs::read_to_string(entry.path())?;
            let ast = syn::parse_file(&content)?;
            visitor.visit_file(&ast);
        }
    }

    Ok(visitor.fns)
}

pub fn generate_leptos_bridge(fns: &[ZomeFnMetadata]) -> String {
    let mut actions = Vec::new();

    for f in fns {
        let fn_name = format_ident!("{}", f.name);
        let fn_name_str = &f.name;
        let input_ty = syn::parse_str::<Type>(&f.input_type).unwrap();
        let output_ty = syn::parse_str::<Type>(&f.output_type).unwrap();
        let zome_name = &f.zome;

        actions.push(quote! {
            pub fn #fn_name() -> Action<#input_ty, Result<#output_ty, MycelixError>> {
                create_server_action(move |input: #input_ty| {
                    let zome_name = #zome_name;
                    let fn_name = #fn_name_str;
                    async move {
                        let client = use_holochain();
                        client.call_zome(zome_name, fn_name, input).await
                    }
                })
            }
        });
    }

    let expanded = quote! {
        use leptos::prelude::*;
        use mycelix_leptos_client::{use_holochain, MycelixError};

        #(#actions)*
    };

    expanded.to_string()
}
