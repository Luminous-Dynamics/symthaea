//! Code HDC Encoder — AST structure → Hypervectors
//!
//! The core innovation: encode AST structure into 16,384-D hypervectors
//! preserving compositional semantics. A function's HV encodes its name,
//! parameters, return type, and body structure — all queryable via unbinding.
//!
//! # Architecture
//!
//! ```text
//! CodeEntity (AST)
//!     ↓ encode_entity()
//! bind(role_hv, name_hv, position_hv)
//!     ↓
//! RealHV (16,384D)
//! ```
//!
//! # Key Insight
//!
//! A function's HV = `bind(FUNCTION, name) ⊕ bundle([params]) ⊕ bind(RETURNS, type)`.
//! You can query "what does this function take?" by unbinding with FUNCTION
//! and checking similarity to known patterns.

use std::collections::HashMap;

use symthaea_core::hdc::RealHV;

use crate::language::code_parser::{CodeEntity, EntityKind, ParsedCode, Relation, CodeRelation};

/// Encoder that converts AST structure into hypervectors
pub struct CodeHDEncoder {
    dim: usize,
    /// Role vectors: what kind of node
    role_vectors: HashMap<EntityKind, RealHV>,
    /// Relation vectors: how things connect
    relation_vectors: HashMap<Relation, RealHV>,
    /// Position vectors for encoding sequence order
    position_vectors: Vec<RealHV>,
    /// Maximum number of position vectors to pre-generate
    max_positions: usize,
}

impl CodeHDEncoder {
    /// Create a new encoder with the standard dimension
    pub fn new(dim: usize) -> Self {
        let mut encoder = Self {
            dim,
            role_vectors: HashMap::new(),
            relation_vectors: HashMap::new(),
            position_vectors: Vec::new(),
            max_positions: 256,
        };
        encoder.init_role_vectors();
        encoder.init_relation_vectors();
        encoder.init_position_vectors();
        encoder
    }

    /// Create with default 16,384-D dimension
    pub fn default_dim() -> Self {
        Self::new(symthaea_core::hdc::unified_hv::HDC_DIMENSION)
    }

    /// Get the encoding dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Initialize deterministic role vectors (one per EntityKind)
    fn init_role_vectors(&mut self) {
        let kinds = [
            EntityKind::Function,
            EntityKind::Method,
            EntityKind::Struct,
            EntityKind::Class,
            EntityKind::Enum,
            EntityKind::Interface,
            EntityKind::Trait,
            EntityKind::TraitImpl,
            EntityKind::Module,
            EntityKind::Import,
            EntityKind::Variable,
            EntityKind::Constant,
            EntityKind::TypeAlias,
            EntityKind::Macro,
            EntityKind::Comment,
            EntityKind::DocComment,
            EntityKind::Lifetime,
            EntityKind::GenericParam,
            EntityKind::UnsafeBlock,
            EntityKind::AsyncBlock,
            EntityKind::ClosureExpr,
            EntityKind::Decorator,
            EntityKind::Comprehension,
            EntityKind::WithStatement,
            EntityKind::MatchStatement,
            EntityKind::Binding,
            EntityKind::AttrSet,
            EntityKind::LetExpression,
            EntityKind::WithExpression,
            EntityKind::FlakeInput,
            EntityKind::FlakeOutput,
            EntityKind::Derivation,
            EntityKind::Other,
        ];

        for (i, kind) in kinds.iter().enumerate() {
            // Deterministic seed: role index * large prime
            let seed = (i as u64 + 1) * 2_654_435_761;
            self.role_vectors.insert(*kind, RealHV::random(self.dim, seed));
        }
    }

    /// Initialize relation vectors
    fn init_relation_vectors(&mut self) {
        let relations = [
            Relation::Calls,
            Relation::Imports,
            Relation::Implements,
            Relation::Extends,
            Relation::References,
            Relation::Contains,
            Relation::DependsOn,
            Relation::AssociatedWith,
        ];

        for (i, rel) in relations.iter().enumerate() {
            let seed = (i as u64 + 100) * 3_141_592_653;
            self.relation_vectors.insert(*rel, RealHV::random(self.dim, seed));
        }
    }

    /// Initialize position vectors for sequence encoding
    fn init_position_vectors(&mut self) {
        for i in 0..self.max_positions {
            let seed = (i as u64 + 1000) * 1_618_033_989;
            self.position_vectors.push(RealHV::random(self.dim, seed));
        }
    }

    // ========================================================================
    // Name Encoding
    // ========================================================================

    /// Encode a symbol name into a hypervector using character n-gram hashing
    pub fn encode_name(&self, name: &str) -> RealHV {
        if name.is_empty() {
            return RealHV::zero(self.dim);
        }

        let mut values = vec![0.0f32; self.dim];
        let name_lower = name.to_lowercase();

        // Character-level encoding with position sensitivity
        for (i, byte) in name_lower.bytes().enumerate() {
            let idx = ((byte as usize).wrapping_mul(31).wrapping_add(i.wrapping_mul(7))) % self.dim;
            values[idx] += 1.0;
        }

        // Bigram encoding for richer name representation
        let bytes: Vec<u8> = name_lower.bytes().collect();
        for window in bytes.windows(2) {
            let idx = ((window[0] as usize).wrapping_mul(127)
                .wrapping_add(window[1] as usize).wrapping_mul(131)) % self.dim;
            values[idx] += 0.5;
        }

        // Trigram encoding
        for window in bytes.windows(3) {
            let idx = ((window[0] as usize).wrapping_mul(251)
                .wrapping_add(window[1] as usize).wrapping_mul(257)
                .wrapping_add(window[2] as usize).wrapping_mul(263)) % self.dim;
            values[idx] += 0.25;
        }

        // Normalize
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut values {
                *v /= magnitude;
            }
        }

        RealHV::from_values(values)
    }

    // ========================================================================
    // Entity Encoding
    // ========================================================================

    /// Encode a single code entity: bind(role, name, position)
    pub fn encode_entity(&self, entity: &CodeEntity) -> RealHV {
        let role_hv = self.role_vectors.get(&entity.kind)
            .cloned()
            .unwrap_or_else(|| RealHV::zero(self.dim));
        let name_hv = self.encode_name(&entity.name);

        // bind(role, name)
        let bound = role_hv.bind(&name_hv);

        // If entity has annotations, encode them too
        let mut annotation_hvs = Vec::new();
        for (key, val) in &entity.annotations {
            let key_hv = self.encode_name(key);
            let val_hv = self.encode_name(val);
            annotation_hvs.push(key_hv.bind(&val_hv));
        }

        // Bundle children
        let child_hvs: Vec<RealHV> = entity.children.iter()
            .enumerate()
            .map(|(i, child)| {
                let child_hv = self.encode_entity(child);
                let pos_hv = self.position_vector(i);
                child_hv.bind(pos_hv)
            })
            .collect();

        // Combine: entity = bound ⊕ annotations ⊕ children
        let mut all: Vec<RealHV> = vec![bound.clone()];
        all.extend(annotation_hvs.into_iter());
        all.extend(child_hvs.into_iter());

        if all.len() == 1 {
            bound
        } else {
            RealHV::bundle(&all)
        }
    }

    /// Encode a function entity with full semantic structure:
    /// bind(FUNCTION, name) ⊕ bundle([param_hvs]) ⊕ bind(RETURNS, return_type)
    pub fn encode_function(&self, entity: &CodeEntity) -> RealHV {
        let base = self.encode_entity(entity);

        // Encode return type if present
        let return_hv = if let Some(ret_type) = entity.annotations.get("return_type") {
            let returns_role = self.encode_name("returns");
            let type_hv = self.encode_name(ret_type);
            returns_role.bind(&type_hv)
        } else {
            RealHV::zero(self.dim)
        };

        // Encode parameters if present
        let params_hv = if let Some(params) = entity.annotations.get("parameters") {
            self.encode_name(params)
        } else {
            RealHV::zero(self.dim)
        };

        // Encode visibility
        let vis_hv = if let Some(vis) = entity.annotations.get("visibility") {
            self.encode_name(vis)
        } else {
            RealHV::zero(self.dim)
        };

        RealHV::bundle(&[base, return_hv, params_hv, vis_hv])
    }

    // ========================================================================
    // Module-Level Encoding
    // ========================================================================

    /// Encode an entire parsed module as a single hypervector
    pub fn encode_module(&self, parsed: &ParsedCode) -> RealHV {
        if parsed.entities.is_empty() {
            return RealHV::zero(self.dim);
        }

        // Encode each entity with position binding
        let entity_hvs: Vec<RealHV> = parsed.entities.iter()
            .enumerate()
            .map(|(i, entity)| {
                let entity_hv = match entity.kind {
                    EntityKind::Function | EntityKind::Method => self.encode_function(entity),
                    _ => self.encode_entity(entity),
                };
                // Bind with position to preserve order
                let pos = self.position_vector(i);
                entity_hv.bind(pos)
            })
            .collect();

        // Bundle all entity HVs
        let module_hv = RealHV::bundle(&entity_hvs);

        // Encode relationships
        let relation_hv = self.encode_relations(&parsed.structure.relations);

        // Combine module entities + relations
        RealHV::bundle(&[module_hv, relation_hv])
    }

    /// Encode a set of relationships
    pub fn encode_relations(&self, relations: &[CodeRelation]) -> RealHV {
        if relations.is_empty() {
            return RealHV::zero(self.dim);
        }

        let hvs: Vec<RealHV> = relations.iter()
            .map(|rel| self.encode_relationship_triple(&rel.source, rel.relation, &rel.target))
            .collect();

        RealHV::bundle(&hvs)
    }

    /// Encode a single relationship triple: bind(source, relation, target)
    pub fn encode_relationship_triple(&self, source: &str, relation: Relation, target: &str) -> RealHV {
        let src_hv = self.encode_name(source);
        let rel_hv = self.relation_vectors.get(&relation)
            .cloned()
            .unwrap_or_else(|| RealHV::zero(self.dim));
        let tgt_hv = self.encode_name(target);

        // Triple: bind(bind(source, relation), target)
        src_hv.bind(&rel_hv).bind(&tgt_hv)
    }

    // ========================================================================
    // Diff Encoding
    // ========================================================================

    /// Encode a diff between two code snapshots
    pub fn encode_diff(&self, old: &ParsedCode, new: &ParsedCode) -> RealHV {
        let old_hv = self.encode_module(old);
        let new_hv = self.encode_module(new);

        // Diff in HDC space: new - old (element-wise subtraction)
        let mut diff_values = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            diff_values[i] = new_hv.values[i] - old_hv.values[i];
        }

        // Normalize
        let magnitude: f32 = diff_values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut diff_values {
                *v /= magnitude;
            }
        }

        RealHV::from_values(diff_values)
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Get position vector at index (wraps around if exceeds max)
    fn position_vector(&self, index: usize) -> &RealHV {
        &self.position_vectors[index % self.max_positions]
    }

    /// Get the role vector for an entity kind
    pub fn role_vector(&self, kind: EntityKind) -> Option<&RealHV> {
        self.role_vectors.get(&kind)
    }

    /// Get the relation vector for a relation type
    pub fn relation_vector(&self, relation: Relation) -> Option<&RealHV> {
        self.relation_vectors.get(&relation)
    }
}

impl Default for CodeHDEncoder {
    fn default() -> Self {
        Self::default_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::code_parser::Span;

    fn test_span() -> Span {
        Span {
            start_byte: 0, end_byte: 10,
            start_line: 0, start_col: 0,
            end_line: 0, end_col: 10,
        }
    }

    #[test]
    fn test_encode_name_deterministic() {
        let encoder = CodeHDEncoder::new(512);
        let hv1 = encoder.encode_name("hello");
        let hv2 = encoder.encode_name("hello");
        // Use approximate comparison due to floating-point precision
        assert!((hv1.similarity(&hv2) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_encode_name_similar_names() {
        let encoder = CodeHDEncoder::new(512);
        let hello = encoder.encode_name("hello_world");
        let hello2 = encoder.encode_name("hello_earth");
        let goodbye = encoder.encode_name("goodbye_moon");

        // Similar names should be more similar than dissimilar names
        let sim_similar = hello.similarity(&hello2);
        let sim_different = hello.similarity(&goodbye);
        assert!(sim_similar > sim_different,
            "Similar names should have higher similarity: {} vs {}", sim_similar, sim_different);
    }

    #[test]
    fn test_encode_entity() {
        let encoder = CodeHDEncoder::new(512);
        let entity = CodeEntity::new(EntityKind::Function, "my_func", test_span())
            .with_annotation("visibility", "pub");

        let hv = encoder.encode_entity(&entity);
        assert_eq!(hv.values.len(), 512);

        // Should be non-zero
        let magnitude: f32 = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(magnitude > 0.0);
    }

    #[test]
    fn test_encode_function_with_return_type() {
        let encoder = CodeHDEncoder::new(512);
        let entity = CodeEntity::new(EntityKind::Function, "calculate", test_span())
            .with_annotation("parameters", "(x: f64, y: f64)")
            .with_annotation("return_type", "f64");

        let hv = encoder.encode_function(&entity);
        assert_eq!(hv.values.len(), 512);
    }

    #[test]
    fn test_similar_functions_similar_hvs() {
        let encoder = CodeHDEncoder::new(512);

        let func1 = CodeEntity::new(EntityKind::Function, "add", test_span())
            .with_annotation("parameters", "(a: i32, b: i32)")
            .with_annotation("return_type", "i32");

        let func2 = CodeEntity::new(EntityKind::Function, "sum", test_span())
            .with_annotation("parameters", "(a: i32, b: i32)")
            .with_annotation("return_type", "i32");

        let func3 = CodeEntity::new(EntityKind::Struct, "Config", test_span());

        let hv1 = encoder.encode_function(&func1);
        let hv2 = encoder.encode_function(&func2);
        let hv3 = encoder.encode_entity(&func3);

        // Two functions should be more similar to each other than to a struct
        let sim_fns = hv1.similarity(&hv2);
        let sim_fn_struct = hv1.similarity(&hv3);
        assert!(sim_fns > sim_fn_struct,
            "Functions should be more similar: {} vs {}", sim_fns, sim_fn_struct);
    }

    #[test]
    fn test_encode_module() {
        let encoder = CodeHDEncoder::new(512);

        let mut parsed = ParsedCode::new("fn a() {} fn b() {}", "rust");
        parsed.entities.push(CodeEntity::new(EntityKind::Function, "a", test_span()));
        parsed.entities.push(CodeEntity::new(EntityKind::Function, "b", test_span()));

        let module_hv = encoder.encode_module(&parsed);
        assert_eq!(module_hv.values.len(), 512);

        let magnitude: f32 = module_hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(magnitude > 0.0);
    }

    #[test]
    fn test_encode_relationship_triple() {
        let encoder = CodeHDEncoder::new(512);

        let hv = encoder.encode_relationship_triple("Dog", Relation::Implements, "Animal");
        assert_eq!(hv.values.len(), 512);

        // Different relationships should produce different vectors
        let hv2 = encoder.encode_relationship_triple("Dog", Relation::Extends, "Animal");
        let sim = hv.similarity(&hv2);
        assert!(sim < 0.95, "Different relations should produce different vectors: sim={}", sim);
    }

    #[test]
    fn test_encode_diff() {
        let encoder = CodeHDEncoder::new(512);

        let mut old = ParsedCode::new("fn a() {}", "rust");
        old.entities.push(CodeEntity::new(EntityKind::Function, "a", test_span()));

        let mut new = ParsedCode::new("fn a() {} fn b() {}", "rust");
        new.entities.push(CodeEntity::new(EntityKind::Function, "a", test_span()));
        new.entities.push(CodeEntity::new(EntityKind::Function, "b", test_span()));

        let diff_hv = encoder.encode_diff(&old, &new);
        assert_eq!(diff_hv.values.len(), 512);

        // Diff should be non-zero (something changed)
        let magnitude: f32 = diff_hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(magnitude > 0.0);
    }

    #[test]
    fn test_role_vectors_orthogonal() {
        let encoder = CodeHDEncoder::new(512);

        let fn_role = encoder.role_vector(EntityKind::Function).unwrap();
        let struct_role = encoder.role_vector(EntityKind::Struct).unwrap();

        // Random high-dimensional vectors should be nearly orthogonal
        let sim = fn_role.similarity(struct_role).abs();
        assert!(sim < 0.2, "Role vectors should be nearly orthogonal: sim={}", sim);
    }
}
