# Connectome graph identity canonicalizer V0.1

**Protocol ID:** `CONNECTOME-STRUCTURAL-GRAPH-CANONICAL-V1`  
**Schema:** `research/connectomics/connectome-universe-structure-artifacts-v0.1.schema.json`  
**Use:** Synthetic directed-graph known-answer fixtures only.

## Input

A sidecar contains:
- `schema_version = "0.1.0"`;
- one or more `worlds`;
- each world has a unique non-empty `world_id` and a `structure`;
- each structure has exactly `nodes` and `edges`;
- each node has a unique non-empty `node_id` and object-valued `attributes`;
- each edge has a unique non-empty `edge_id`, non-empty `source` and `target`, and object-valued `attributes`.

An edge whose source or target is absent from the node set is invalid. Duplicate node IDs, edge IDs or world IDs are invalid. Unknown fields at schema-controlled object levels are invalid.

## Canonical representation

1. Parse the sidecar as JSON.
2. Validate all node and edge identifiers and references.
3. Sort nodes by `node_id` using Rust string ordering.
4. Sort edges by `edge_id` using Rust string ordering.
5. Preserve node IDs, edge IDs, direction, all node attributes and all edge attributes.
6. Construct exactly one JSON object with two keys: `nodes` and `edges`, each containing its sorted array.
7. Serialize with `serde_json::to_vec` under the pinned dependency closure in this crate. The crate uses the default `serde_json::Map` representation; the `preserve_order` feature is not enabled, so object keys serialize in the library's ordered-map order.
8. Prefix the serialized bytes with the UTF-8 bytes of `CONNECTOME-STRUCTURAL-GRAPH-CANONICAL-V1` followed by one NUL byte.
9. Compute SHA-256 over the prefix and canonical JSON bytes.

Equivalent expression:

```text
SHA256(
  UTF8("CONNECTOME-STRUCTURAL-GRAPH-CANONICAL-V1")
  || 0x00
  || serde_json::to_vec({ nodes: sorted_nodes, edges: sorted_edges })
)
```

## Exact meaning of a match

V0.1 is an **ID-preserving canonical representation digest**. It ignores input array order and JSON object key order, but preserves node IDs, edge IDs, direction and every attribute.

Therefore:
- same canonical bytes produce the same digest;
- reordered node/edge arrays produce the same digest;
- different node/edge IDs do not match, even if two graphs are isomorphic after relabelling;
- changed direction or changed attributes produce a different digest;
- no voxel, mesh, skeleton, coordinate-frame, physical transform, tolerance-based geometry or biological equivalence is inferred.

The term structural inclusion in the V0.1 report means equality under this strict ID-preserving representation, not graph-isomorphism equivalence. A future biological canonicalizer must be a separate version and must define its own neuron/segment lineage, split/merge ancestry, synapse attributes, physical coordinate transforms and permissible relabellings. Do not silently change V1 semantics.

A matching digest proves only representation identity under this canonicalizer. It does not prove anatomical correctness or that the candidate world is biologically plausible.
