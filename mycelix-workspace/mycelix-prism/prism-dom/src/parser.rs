// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HTML parser backed by `html5ever`.
//!
//! Implements `html5ever::TreeSink` to build a `DomTree` from raw HTML bytes.
//! Uses `RefCell` for interior mutability since `TreeSink` methods take `&self`.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;

use html5ever::tendril::*;
use html5ever::tree_builder::{ElementFlags, NodeOrText, QuirksMode, TreeSink};
use html5ever::{Attribute, ExpandedName, QualName};
use html5ever::{ParseOpts, parse_document};

use crate::node::{ElementData, NodeData, NodeId};
use crate::tree::DomTree;

/// Parse an HTML string into a `DomTree`.
pub fn parse_html(html: &str) -> DomTree {
    let mut tree = DomTree::new();
    let doc = tree.new_node(NodeData::Document);
    tree.set_root(doc);

    let sink = PrismSink {
        tree: RefCell::new(tree),
        doc,
    };

    let parser = parse_document(sink, ParseOpts::default());
    let sink = parser.one(html);
    sink.tree.into_inner()
}

/// TreeSink implementation that builds our arena-based `DomTree`.
/// Uses `RefCell` for interior mutability since TreeSink methods take `&self`.
struct PrismSink {
    tree: RefCell<DomTree>,
    doc: NodeId,
}

impl TreeSink for PrismSink {
    type Handle = NodeId;
    type Output = Self;
    type ElemName<'a> = ExpandedName<'a>;

    fn finish(self) -> Self::Output {
        self
    }

    fn parse_error(&self, _msg: Cow<'static, str>) {
        // Silently ignore parse errors (browsers are lenient)
    }

    fn get_document(&self) -> NodeId {
        self.doc
    }

    fn elem_name<'a>(&'a self, target: &'a NodeId) -> ExpandedName<'a> {
        // This is tricky: we need to return a reference into the RefCell.
        // Since html5ever only calls this during single-threaded parsing and
        // the node data won't move, we borrow and leak the guard for the
        // duration. In practice, html5ever drops the ExpandedName before
        // any mutation occurs.
        //
        // A safer alternative would be to store QualName outside the RefCell,
        // but that would require a separate arena. For now, we use a raw pointer
        // which is safe because the DomTree arena never reallocates existing nodes.
        let tree = self.tree.borrow();
        let node = tree.get(*target).expect("valid node");
        match &node.data {
            NodeData::Element(el) => {
                // SAFETY: The DomTree arena (Vec) may reallocate on push, but
                // elem_name is only called on already-existing nodes. The borrow
                // is dropped before any new nodes are created in the same call.
                let qn: *const QualName = &el.qual_name;
                unsafe { (*qn).expanded() }
            }
            _ => panic!("elem_name called on non-element"),
        }
    }

    fn create_element(
        &self,
        name: QualName,
        attrs: Vec<Attribute>,
        _flags: ElementFlags,
    ) -> NodeId {
        let attributes: HashMap<String, String> = attrs
            .into_iter()
            .map(|a| (a.name.local.to_string(), a.value.to_string()))
            .collect();

        self.tree
            .borrow_mut()
            .new_node(NodeData::Element(ElementData {
                tag: name.local.to_string(),
                qual_name: name,
                attributes,
            }))
    }

    fn create_comment(&self, text: StrTendril) -> NodeId {
        self.tree
            .borrow_mut()
            .new_node(NodeData::Comment(text.to_string()))
    }

    fn create_pi(&self, target: StrTendril, data: StrTendril) -> NodeId {
        self.tree
            .borrow_mut()
            .new_node(NodeData::ProcessingInstruction {
                target: target.to_string(),
                data: data.to_string(),
            })
    }

    fn append(&self, parent: &NodeId, child: NodeOrText<NodeId>) {
        let mut tree = self.tree.borrow_mut();
        match child {
            NodeOrText::AppendNode(node_id) => {
                tree.append_child(*parent, node_id);
            }
            NodeOrText::AppendText(text) => {
                // Try to merge with existing last text child
                let should_merge = tree
                    .get(*parent)
                    .and_then(|p| p.children.last().copied())
                    .and_then(|last| tree.get(last).map(|n| (last, n.is_text())))
                    .filter(|(_, is_text)| *is_text)
                    .map(|(id, _)| id);

                if let Some(last_id) = should_merge
                    && let NodeData::Text(ref mut existing) = tree.get_mut(last_id).unwrap().data
                {
                    existing.push_str(&text);
                    return;
                }
                let text_id = tree.new_node(NodeData::Text(text.to_string()));
                tree.append_child(*parent, text_id);
            }
        }
    }

    fn append_based_on_parent_node(
        &self,
        element: &NodeId,
        _prev_element: &NodeId,
        child: NodeOrText<NodeId>,
    ) {
        self.append(element, child);
    }

    fn append_doctype_to_document(
        &self,
        name: StrTendril,
        public_id: StrTendril,
        system_id: StrTendril,
    ) {
        let mut tree = self.tree.borrow_mut();
        let doctype = tree.new_node(NodeData::Doctype {
            name: name.to_string(),
            public_id: public_id.to_string(),
            system_id: system_id.to_string(),
        });
        let doc = self.doc;
        tree.append_child(doc, doctype);
    }

    fn get_template_contents(&self, target: &NodeId) -> NodeId {
        *target
    }

    fn same_node(&self, x: &NodeId, y: &NodeId) -> bool {
        x == y
    }

    fn set_quirks_mode(&self, _mode: QuirksMode) {
        // We don't track quirks mode for now
    }

    fn append_before_sibling(&self, sibling: &NodeId, new_node: NodeOrText<NodeId>) {
        let mut tree = self.tree.borrow_mut();
        let parent = tree
            .get(*sibling)
            .and_then(|n| n.parent)
            .expect("sibling must have parent");

        match new_node {
            NodeOrText::AppendNode(node_id) => {
                tree.insert_before(parent, node_id, *sibling);
            }
            NodeOrText::AppendText(text) => {
                let text_id = tree.new_node(NodeData::Text(text.to_string()));
                tree.insert_before(parent, text_id, *sibling);
            }
        }
    }

    fn add_attrs_if_missing(&self, target: &NodeId, attrs: Vec<Attribute>) {
        let mut tree = self.tree.borrow_mut();
        if let Some(node) = tree.get_mut(*target)
            && let NodeData::Element(ref mut el) = node.data
        {
            for attr in attrs {
                el.attributes
                    .entry(attr.name.local.to_string())
                    .or_insert_with(|| attr.value.to_string());
            }
        }
    }

    fn remove_from_parent(&self, target: &NodeId) {
        self.tree.borrow_mut().detach(*target);
    }

    fn reparent_children(&self, node: &NodeId, new_parent: &NodeId) {
        let mut tree = self.tree.borrow_mut();
        let children: Vec<NodeId> = tree
            .get(*node)
            .map(|n| n.children.clone())
            .unwrap_or_default();
        for child in children {
            tree.detach(child);
            tree.append_child(*new_parent, child);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_html() {
        let tree =
            parse_html("<html><head><title>Test</title></head><body><p>Hello</p></body></html>");
        assert!(!tree.is_empty());
        assert_eq!(tree.title(), Some("Test".to_string()));
        let text = tree.extract_visible_text();
        assert!(text.contains("Hello"));
    }

    #[test]
    fn parse_with_attributes() {
        let tree = parse_html(
            r#"<div id="main" class="container wide"><a href="https://example.com">Link</a></div>"#,
        );
        let divs = tree.find_by_tag("div");
        assert_eq!(divs.len(), 1);
        let el = tree.get(divs[0]).unwrap().as_element().unwrap();
        assert_eq!(el.id(), Some("main"));
        assert_eq!(el.classes(), vec!["container", "wide"]);
    }

    #[test]
    fn parse_strips_script_from_text() {
        let tree = parse_html(
            "<body><p>Visible</p><script>alert('hidden')</script><p>Also visible</p></body>",
        );
        let text = tree.extract_visible_text();
        assert!(text.contains("Visible"));
        assert!(text.contains("Also visible"));
        assert!(!text.contains("alert"));
    }

    #[test]
    fn parse_detects_password() {
        let tree = parse_html(r#"<form><input type="text" /><input type="password" /></form>"#);
        assert!(tree.has_password_field());
    }

    #[test]
    fn parse_no_password() {
        let tree = parse_html(r#"<form><input type="text" /><input type="email" /></form>"#);
        assert!(!tree.has_password_field());
    }

    #[test]
    fn parse_robots_noindex() {
        let tree = parse_html(
            r#"<html><head><meta name="robots" content="noindex"></head><body></body></html>"#,
        );
        assert_eq!(tree.robots_meta().unwrap(), "noindex");
    }

    #[test]
    fn parse_empty_html() {
        let tree = parse_html("");
        assert!(!tree.is_empty());
        assert_eq!(tree.extract_visible_text(), "");
    }

    #[test]
    fn parse_malformed_html() {
        let tree = parse_html("<p>Unclosed paragraph<div>And a div</p></div><b>Bold");
        let text = tree.extract_visible_text();
        assert!(text.contains("Unclosed paragraph"));
        assert!(text.contains("Bold"));
    }
}
