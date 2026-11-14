//! Pure, read-only Rust syntax projection.
//!
//! This module parses one Rust source file into syntax-level declarations. It
//! never expands macros, follows modules, executes build scripts, resolves
//! names, evaluates cfg predicates, or claims compiler-resolved semantics.

use quote::ToTokens;
use serde::Serialize;
use syn::{
    visit::{self, Visit},
    Attribute, File, Visibility,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RustFileAnchor {
    pub repository_full_name: String,
    pub revision: String,
    pub tree_oid: String,
    pub source_path: String,
    pub blob_oid: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RustSyntaxFact {
    pub source_order: usize,
    pub kind: String,
    pub name: Option<String>,
    pub module_path: Vec<String>,
    pub visibility: String,
    /// Raw attribute token forms, including cfg/test/doc attributes.
    pub attributes: Vec<String>,
    /// A compact signature or syntax header, not a compiled/resolved symbol.
    pub declaration: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RustFileProjection {
    pub schema_version: String,
    pub authority: String,
    pub source: RustFileAnchor,
    pub file_attributes: Vec<String>,
    pub facts: Vec<RustSyntaxFact>,
    pub limitations: Vec<String>,
}

fn valid_git_oid(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_anchor(anchor: &RustFileAnchor) -> Result<(), String> {
    for (name, value) in [
        ("repository_full_name", anchor.repository_full_name.as_str()),
        ("source_path", anchor.source_path.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(format!("source anchor {name} must not be empty"));
        }
    }
    for (name, value) in [
        ("revision", anchor.revision.as_str()),
        ("tree_oid", anchor.tree_oid.as_str()),
        ("blob_oid", anchor.blob_oid.as_str()),
    ] {
        if !valid_git_oid(value) {
            return Err(format!(
                "source anchor {name} must be a 40- or 64-character hexadecimal Git object ID"
            ));
        }
    }
    Ok(())
}

fn visibility_text(visibility: &Visibility) -> String {
    match visibility {
        Visibility::Inherited => "private".to_owned(),
        other => other.to_token_stream().to_string(),
    }
}

fn attributes_text(attributes: &[Attribute]) -> Vec<String> {
    attributes
        .iter()
        .map(|attribute| attribute.meta.to_token_stream().to_string())
        .collect()
}

struct Extractor {
    module_path: Vec<String>,
    facts: Vec<RustSyntaxFact>,
}

impl Extractor {
    fn push(
        &mut self,
        kind: &str,
        name: Option<String>,
        visibility: Option<&Visibility>,
        attributes: &[Attribute],
        declaration: Option<String>,
    ) {
        self.facts.push(RustSyntaxFact {
            source_order: self.facts.len(),
            kind: kind.to_owned(),
            name,
            module_path: self.module_path.clone(),
            visibility: visibility.map(visibility_text).unwrap_or_else(|| "not_explicit".to_owned()),
            attributes: attributes_text(attributes),
            declaration,
        });
    }
}

impl<'ast> Visit<'ast> for Extractor {
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        self.push(
            if node.content.is_some() { "inline_module" } else { "module_declaration" },
            Some(node.ident.to_string()),
            Some(&node.vis),
            &node.attrs,
            Some(node.ident.to_token_stream().to_string()),
        );
        if node.content.is_some() {
            self.module_path.push(node.ident.to_string());
            visit::visit_item_mod(self, node);
            self.module_path.pop();
        }
    }

    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        self.push(
            "use_declaration",
            None,
            Some(&node.vis),
            &node.attrs,
            Some(node.tree.to_token_stream().to_string()),
        );
        visit::visit_item_use(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        self.push(
            "function",
            Some(node.sig.ident.to_string()),
            Some(&node.vis),
            &node.attrs,
            Some(node.sig.to_token_stream().to_string()),
        );
        visit::visit_item_fn(self, node);
    }

    fn visit_item_struct(&mut self, node: &'ast syn::ItemStruct) {
        self.push(
            "struct",
            Some(node.ident.to_string()),
            Some(&node.vis),
            &node.attrs,
            Some(format!("{} {}", node.ident, node.generics.to_token_stream())),
        );
        visit::visit_item_struct(self, node);
    }

    fn visit_item_enum(&mut self, node: &'ast syn::ItemEnum) {
        self.push(
            "enum",
            Some(node.ident.to_string()),
            Some(&node.vis),
            &node.attrs,
            Some(format!("{} {}", node.ident, node.generics.to_token_stream())),
        );
        visit::visit_item_enum(self, node);
    }

    fn visit_item_union(&mut self, node: &'ast syn::ItemUnion) {
        self.push(
            "union",
            Some(node.ident.to_string()),
            Some(&node.vis),
            &node.attrs,
            Some(format!("{} {}", node.ident, node.generics.to_token_stream())),
        );
        visit::visit_item_union(self, node);
    }

    fn visit_item_trait(&mut self, node: &'ast syn::ItemTrait) {
        self.push(
            "trait",
            Some(node.ident.to_string()),
            Some(&node.vis),
            &node.attrs,
            Some(format!("{} {}", node.ident, node.generics.to_token_stream())),
        );
        visit::visit_item_trait(self, node);
    }

    fn visit_item_type(&mut self, node: &'ast syn::ItemType) {
        self.push(
            "type_alias",
            Some(node.ident.to_string()),
            Some(&node.vis),
            &node.attrs,
            Some(format!("{} {}", node.ident, node.generics.to_token_stream())),
        );
        visit::visit_item_type(self, node);
    }

    fn visit_item_const(&mut self, node: &'ast syn::ItemConst) {
        self.push(
            "constant",
            Some(node.ident.to_string()),
            Some(&node.vis),
            &node.attrs,
            Some(format!("{}: {}", node.ident, node.ty.to_token_stream())),
        );
        visit::visit_item_const(self, node);
    }

    fn visit_item_static(&mut self, node: &'ast syn::ItemStatic) {
        self.push(
            "static",
            Some(node.ident.to_string()),
            Some(&node.vis),
            &node.attrs,
            Some(format!("{}: {}", node.ident, node.ty.to_token_stream())),
        );
        visit::visit_item_static(self, node);
    }

    fn visit_item_extern_crate(&mut self, node: &'ast syn::ItemExternCrate) {
        self.push(
            "extern_crate",
            Some(node.ident.to_string()),
            Some(&node.vis),
            &node.attrs,
            Some(node.to_token_stream().to_string()),
        );
        visit::visit_item_extern_crate(self, node);
    }

    fn visit_item_macro(&mut self, node: &'ast syn::ItemMacro) {
        self.push(
            "macro_syntax",
            node.ident.as_ref().map(ToString::to_string),
            None,
            &node.attrs,
            Some(node.mac.path.to_token_stream().to_string()),
        );
        // Intentionally do not visit or expand the macro token body.
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let declaration = match &node.trait_ {
            Some((_, trait_path, _)) => Some(format!(
                "{} for {}",
                trait_path.to_token_stream(),
                node.self_ty.to_token_stream()
            )),
            None => Some(node.self_ty.to_token_stream().to_string()),
        };
        self.push(
            "impl_block",
            None,
            None,
            &node.attrs,
            declaration,
        );
        visit::visit_item_impl(self, node);
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        self.push(
            "impl_method",
            Some(node.sig.ident.to_string()),
            Some(&node.vis),
            &node.attrs,
            Some(node.sig.to_token_stream().to_string()),
        );
        visit::visit_impl_item_fn(self, node);
    }

    fn visit_trait_item_fn(&mut self, node: &'ast syn::TraitItemFn) {
        self.push(
            "trait_method",
            Some(node.sig.ident.to_string()),
            None,
            &node.attrs,
            Some(node.sig.to_token_stream().to_string()),
        );
        visit::visit_trait_item_fn(self, node);
    }
}

/// Parse one Rust source file as a syntax-only, source-anchored projection.
///
/// This does not authenticate the supplied Git identity, follow module
/// declarations, expand macros, resolve imports, or evaluate cfg conditions.
pub fn project_rust_file(
    raw_source: &str,
    source: RustFileAnchor,
) -> Result<RustFileProjection, String> {
    validate_anchor(&source)?;
    let file: File = syn::parse_file(raw_source)
        .map_err(|error| format!("Rust source parse error: {error}"))?;

    let mut extractor = Extractor {
        module_path: Vec::new(),
        facts: Vec::new(),
    };
    for item in &file.items {
        extractor.visit_item(item);
    }

    let mut limitations = vec![
        "Syntax-extracted declarations are not compiler-resolved symbols or dependency edges.".to_owned(),
        "Out-of-line module declarations are recorded but their source files are not followed.".to_owned(),
        "Item-position macro invocations and macro definitions are recorded as opaque syntax; macros inside function bodies are not inventoried or expanded.".to_owned(),
        "cfg and cfg_attr expressions are preserved as attributes and are not evaluated into configuration worlds.".to_owned(),
        "Imports are recorded as syntax; names and target paths are not resolved. Field and variant-level inventories are not yet emitted.".to_owned(),
        "The supplied Git source anchor is recorded but is not authenticated by this parser.".to_owned(),
    ];
    limitations.sort();

    Ok(RustFileProjection {
        schema_version: "0.1.0".to_owned(),
        authority: "syntax_extracted".to_owned(),
        source,
        file_attributes: attributes_text(&file.attrs),
        facts: extractor.facts,
        limitations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor() -> RustFileAnchor {
        RustFileAnchor {
            repository_full_name: "example/repo".to_owned(),
            revision: "a".repeat(40),
            tree_oid: "b".repeat(40),
            source_path: "src/lib.rs".to_owned(),
            blob_oid: "c".repeat(40),
        }
    }

    #[test]
    fn parses_nested_modules_and_retains_module_paths() {
        let source = r#"
            pub mod api {
                pub struct Request;
                mod internal {
                    pub enum State { Ready, Failed }
                }
            }
        "#;
        let projection = project_rust_file(source, anchor()).expect("projection");
        let request = projection.facts.iter().find(|fact| fact.name.as_deref() == Some("Request")).expect("Request");
        assert_eq!(request.module_path, vec!["api".to_owned()]);
        let state = projection.facts.iter().find(|fact| fact.name.as_deref() == Some("State")).expect("State");
        assert_eq!(state.module_path, vec!["api".to_owned(), "internal".to_owned()]);
    }

    #[test]
    fn records_out_of_line_modules_without_following_them() {
        let projection = project_rust_file("pub mod worker;", anchor()).expect("projection");
        assert_eq!(projection.facts.len(), 1);
        assert_eq!(projection.facts[0].kind, "module_declaration");
        assert!(projection.limitations.iter().any(|value| value.contains("not followed")));
    }

    #[test]
    fn preserves_cfg_attributes_without_evaluating_them() {
        let source = r#"
            #[cfg(feature = "experimental")]
            pub fn candidate() {}
        "#;
        let projection = project_rust_file(source, anchor()).expect("projection");
        let function = projection.facts.iter().find(|fact| fact.name.as_deref() == Some("candidate")).expect("function");
        assert!(function.attributes.iter().any(|attribute| attribute.contains("cfg")));
        assert!(function.attributes.iter().any(|attribute| attribute.contains("experimental")));
    }

    #[test]
    fn records_use_declarations_as_unresolved_syntax() {
        let projection = project_rust_file("use crate::model::{A, B as Bee};", anchor()).expect("projection");
        let import = projection.facts.iter().find(|fact| fact.kind == "use_declaration").expect("import");
        assert!(import.declaration.as_deref().is_some_and(|value| value.contains("crate")));
        assert!(import.declaration.as_deref().is_some_and(|value| value.contains("Bee")));
        assert!(projection.limitations.iter().any(|value| value.contains("not resolved")));
    }

    #[test]
    fn records_test_attributes_and_function_signatures() {
        let source = r#"
            #[test]
            fn invariant_holds() {}
        "#;
        let projection = project_rust_file(source, anchor()).expect("projection");
        let test = projection.facts.iter().find(|fact| fact.name.as_deref() == Some("invariant_holds")).expect("test");
        assert!(test.attributes.iter().any(|attribute| attribute == "test"));
        assert!(test.declaration.as_deref().is_some_and(|value| value.contains("invariant_holds")));
    }

    #[test]
    fn records_macro_syntax_without_expansion() {
        let projection = project_rust_file("declare_api!();", anchor()).expect("projection");
        assert_eq!(projection.facts.len(), 1);
        assert_eq!(projection.facts[0].kind, "macro_syntax");
        assert_eq!(projection.facts[0].declaration.as_deref(), Some("declare_api"));
    }

    #[test]
    fn preserves_impl_methods_and_trait_methods() {
        let source = r#"
            pub trait Service { fn start(&self); }
            pub struct Worker;
            impl Service for Worker {
                fn start(&self) {}
            }
        "#;
        let projection = project_rust_file(source, anchor()).expect("projection");
        assert!(projection.facts.iter().any(|fact| fact.kind == "trait_method" && fact.name.as_deref() == Some("start")));
        assert!(projection.facts.iter().any(|fact| fact.kind == "impl_method" && fact.name.as_deref() == Some("start")));
        assert!(projection.facts.iter().any(|fact| fact.kind == "impl_block"));
    }

    #[test]
    fn rejects_invalid_source_identity() {
        let mut invalid = anchor();
        invalid.blob_oid = "Cargo.toml".to_owned();
        let error = project_rust_file("pub struct A;", invalid).expect_err("invalid identity");
        assert!(error.contains("blob_oid"));
    }

    #[test]
    fn rejects_invalid_rust_syntax() {
        let error = project_rust_file("pub fn broken( {", anchor()).expect_err("parse error");
        assert!(error.contains("Rust source parse error"));
    }
}
