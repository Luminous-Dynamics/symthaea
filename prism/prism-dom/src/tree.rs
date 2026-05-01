// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Arena-based DOM tree.

use crate::node::{Node, NodeData, NodeId};

/// Arena-based DOM tree. All nodes are stored in a flat `Vec`; relationships
/// are expressed via `NodeId` indices. This avoids reference counting and
/// makes the tree trivially serializable.
#[derive(Debug)]
pub struct DomTree {
    nodes: Vec<Node>,
    /// The root document node (always index 0 when non-empty).
    root: Option<NodeId>,
}

impl DomTree {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: None,
        }
    }

    /// Allocate a new node in the arena and return its ID.
    pub fn new_node(&mut self, data: NodeData) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node::new(data));
        id
    }

    /// Set the root node of the document.
    pub fn set_root(&mut self, id: NodeId) {
        self.root = Some(id);
    }

    /// Get the root node ID.
    pub fn root(&self) -> Option<NodeId> {
        self.root
    }

    /// Get a reference to a node by ID.
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0)
    }

    /// Get a mutable reference to a node by ID.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id.0)
    }

    /// Append `child` as the last child of `parent`.
    pub fn append_child(&mut self, parent: NodeId, child: NodeId) {
        self.nodes[child.0].parent = Some(parent);
        self.nodes[parent.0].children.push(child);
    }

    /// Insert `child` before `sibling` in `parent`'s children.
    pub fn insert_before(&mut self, parent: NodeId, child: NodeId, sibling: NodeId) {
        self.nodes[child.0].parent = Some(parent);
        let children = &mut self.nodes[parent.0].children;
        if let Some(pos) = children.iter().position(|&id| id == sibling) {
            children.insert(pos, child);
        } else {
            children.push(child);
        }
    }

    /// Remove `child` from its parent's children list.
    pub fn detach(&mut self, child: NodeId) {
        if let Some(parent_id) = self.nodes[child.0].parent {
            self.nodes[parent_id.0].children.retain(|&id| id != child);
            self.nodes[child.0].parent = None;
        }
    }

    /// Total number of nodes in the tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Extract all visible text content from the tree.
    ///
    /// This walks the tree depth-first and collects text from all text nodes,
    /// skipping `<script>`, `<style>`, `<noscript>`, and hidden elements.
    /// Used by the reflex arc for content scanning.
    pub fn extract_visible_text(&self) -> String {
        let Some(root) = self.root else {
            return String::new();
        };
        let mut text = String::with_capacity(4096);
        self.collect_text(root, &mut text);
        text
    }

    fn collect_text(&self, node_id: NodeId, buf: &mut String) {
        let node = &self.nodes[node_id.0];
        match &node.data {
            NodeData::Text(t) => {
                if !t.trim().is_empty() {
                    if !buf.is_empty() {
                        buf.push(' ');
                    }
                    buf.push_str(t.trim());
                }
            }
            NodeData::Element(el) => {
                // Skip invisible elements
                let tag = el.tag.as_str();
                if matches!(tag, "script" | "style" | "noscript" | "svg" | "template") {
                    return;
                }
                for &child_id in &node.children {
                    self.collect_text(child_id, buf);
                }
            }
            NodeData::Document => {
                for &child_id in &node.children {
                    self.collect_text(child_id, buf);
                }
            }
            _ => {}
        }
    }

    /// Find all elements matching a tag name.
    pub fn find_by_tag(&self, tag: &str) -> Vec<NodeId> {
        let mut results = Vec::new();
        if let Some(root) = self.root {
            self.find_by_tag_recursive(root, tag, &mut results);
        }
        results
    }

    fn find_by_tag_recursive(&self, node_id: NodeId, tag: &str, results: &mut Vec<NodeId>) {
        let node = &self.nodes[node_id.0];
        if let NodeData::Element(el) = &node.data
            && el.tag == tag
        {
            results.push(node_id);
        }
        for &child_id in &node.children {
            self.find_by_tag_recursive(child_id, tag, results);
        }
    }

    /// Check if the document contains any password input fields.
    /// Used by the privacy zone classifier.
    pub fn has_password_field(&self) -> bool {
        let inputs = self.find_by_tag("input");
        inputs.iter().any(|&id| {
            self.get(id)
                .and_then(|n| n.as_element())
                .and_then(|el| el.attributes.get("type"))
                .is_some_and(|t| t.eq_ignore_ascii_case("password"))
        })
    }

    /// Extract the content of `<meta name="robots">` if present.
    pub fn robots_meta(&self) -> Option<String> {
        let metas = self.find_by_tag("meta");
        for &id in &metas {
            if let Some(el) = self.get(id).and_then(|n| n.as_element()) {
                let is_robots = el
                    .attributes
                    .get("name")
                    .is_some_and(|n| n.eq_ignore_ascii_case("robots"));
                if is_robots {
                    return el.attributes.get("content").cloned();
                }
            }
        }
        None
    }

    /// Extract the document title.
    pub fn title(&self) -> Option<String> {
        let titles = self.find_by_tag("title");
        titles.first().and_then(|&id| {
            let node = self.get(id)?;
            node.children
                .first()
                .and_then(|&child_id| self.get(child_id)?.as_text().map(|t| t.trim().to_string()))
        })
    }
}

impl Default for DomTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::ElementData;
    use html5ever::{QualName, namespace_url, ns};
    use std::collections::HashMap;

    fn make_element(tree: &mut DomTree, tag: &str) -> NodeId {
        tree.new_node(NodeData::Element(ElementData {
            tag: tag.to_string(),
            qual_name: QualName::new(None, ns!(html), tag.into()),
            attributes: HashMap::new(),
        }))
    }

    fn make_text(tree: &mut DomTree, text: &str) -> NodeId {
        tree.new_node(NodeData::Text(text.to_string()))
    }

    #[test]
    fn extract_visible_text_skips_script() {
        let mut tree = DomTree::new();
        let doc = tree.new_node(NodeData::Document);
        tree.set_root(doc);

        let body = make_element(&mut tree, "body");
        tree.append_child(doc, body);

        let p = make_element(&mut tree, "p");
        tree.append_child(body, p);
        let text = make_text(&mut tree, "Hello world");
        tree.append_child(p, text);

        let script = make_element(&mut tree, "script");
        tree.append_child(body, script);
        let js = make_text(&mut tree, "var x = 1;");
        tree.append_child(script, js);

        let text = tree.extract_visible_text();
        assert_eq!(text, "Hello world");
        assert!(!text.contains("var x"));
    }

    #[test]
    fn password_field_detection() {
        let mut tree = DomTree::new();
        let doc = tree.new_node(NodeData::Document);
        tree.set_root(doc);

        let form = make_element(&mut tree, "form");
        tree.append_child(doc, form);

        let mut attrs = HashMap::new();
        attrs.insert("type".to_string(), "password".to_string());
        let input = tree.new_node(NodeData::Element(ElementData {
            tag: "input".to_string(),
            qual_name: QualName::new(None, ns!(html), "input".into()),
            attributes: attrs,
        }));
        tree.append_child(form, input);

        assert!(tree.has_password_field());
    }

    #[test]
    fn robots_meta_extraction() {
        let mut tree = DomTree::new();
        let doc = tree.new_node(NodeData::Document);
        tree.set_root(doc);

        let head = make_element(&mut tree, "head");
        tree.append_child(doc, head);

        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), "robots".to_string());
        attrs.insert("content".to_string(), "noindex, nofollow".to_string());
        let meta = tree.new_node(NodeData::Element(ElementData {
            tag: "meta".to_string(),
            qual_name: QualName::new(None, ns!(html), "meta".into()),
            attributes: attrs,
        }));
        tree.append_child(head, meta);

        assert_eq!(tree.robots_meta().unwrap(), "noindex, nofollow");
    }

    #[test]
    fn title_extraction() {
        let mut tree = DomTree::new();
        let doc = tree.new_node(NodeData::Document);
        tree.set_root(doc);

        let head = make_element(&mut tree, "head");
        tree.append_child(doc, head);

        let title = make_element(&mut tree, "title");
        tree.append_child(head, title);
        let text = make_text(&mut tree, "  Prism  ");
        tree.append_child(title, text);

        assert_eq!(tree.title().unwrap(), "Prism");
    }
}
