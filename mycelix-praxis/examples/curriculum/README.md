# Curriculum Standards Mapping

Example data files that map academic standards to the Praxis knowledge graph.
These files can be loaded by a seed script to populate the DHT with structured
curriculum content.

## File Format

Each curriculum JSON file contains four sections:

### `metadata`
Framework identification, grade level, subject area, and domain listing.

### `nodes`
Array of knowledge graph nodes. Each node maps to the `KnowledgeNode` entry type
defined in `zomes/knowledge_zome/integrity/src/lib.rs`. Fields:

| Field | Type | Maps To |
|-------|------|---------|
| `id` | string | Local identifier (standard code, used by edges) |
| `title` | string | `KnowledgeNode.title` |
| `description` | string | `KnowledgeNode.description` |
| `node_type` | enum | `NodeType` (Concept, Skill, Topic, Course, Assessment, Project) |
| `difficulty` | enum | `DifficultyLevel` (Beginner, Intermediate, Advanced, Expert) |
| `domain` | string | `KnowledgeNode.domain` |
| `subdomain` | string | `KnowledgeNode.subdomain` |
| `tags` | array | `KnowledgeNode.tags` |
| `estimated_hours` | u32 | `KnowledgeNode.estimated_hours` |
| `grade_levels` | array | `KnowledgeNode.grade_levels` (GradeLevel enum variants) |
| `bloom_level` | enum | `BloomLevel` (Remember, Understand, Apply, Analyze, Evaluate, Create) |
| `subject_area` | enum | `SubjectArea` (Mathematics, Science, etc.) |
| `academic_standards` | array | `AcademicStandard` with framework, code, description, grade_level |

### `edges`
Array of prerequisite/relationship edges. Each maps to `LearningEdge`:

| Field | Type | Maps To |
|-------|------|---------|
| `from` | string | Source node `id` (resolved to `ActionHash` at load time) |
| `to` | string | Target node `id` (resolved to `ActionHash` at load time) |
| `edge_type` | enum | `EdgeType` (Requires, Recommends, RelatedTo, PartOf, etc.) |
| `strength_permille` | u16 | 0-1000 (0.0-1.0 as permille) |
| `rationale` | string | `LearningEdge.rationale` |

### `learning_paths` (optional)
Suggested orderings through the standards for instructional planning.

### `grade2_prerequisites` (optional)
Prior-grade standards that feed into this grade's content, useful for
cross-grade pathway planning.

## Loading Into the Knowledge Graph

A seed script would:

1. Parse the JSON file
2. Create each node via `create_knowledge_node` zome call
3. Collect the returned `ActionHash` for each node `id`
4. Create edges by resolving `from`/`to` ids to `ActionHash` values via
   `create_learning_edge` zome call
5. Optionally create `LearningPath` entries from the `learning_paths` section

Example pseudocode:

```typescript
import { AdminWebsocket, AppWebsocket } from '@holochain/client';

const data = JSON.parse(fs.readFileSync('grade3_math.json', 'utf8'));
const hashMap: Record<string, ActionHash> = {};

// Create nodes
for (const node of data.nodes) {
  const hash = await appWs.callZome({
    role_name: 'praxis',
    zome_name: 'knowledge_zome',
    fn_name: 'create_knowledge_node',
    payload: {
      title: node.title,
      description: node.description,
      node_type: node.node_type,
      difficulty: node.difficulty,
      domain: node.domain,
      subdomain: node.subdomain,
      tags: node.tags,
      estimated_hours: node.estimated_hours,
      grade_levels: node.grade_levels,
      bloom_level: node.bloom_level,
      subject_area: node.subject_area,
      academic_standards: node.academic_standards,
    },
  });
  hashMap[node.id] = hash;
}

// Create edges
for (const edge of data.edges) {
  await appWs.callZome({
    role_name: 'praxis',
    zome_name: 'knowledge_zome',
    fn_name: 'create_learning_edge',
    payload: {
      source_node: hashMap[edge.from],
      target_node: hashMap[edge.to],
      edge_type: edge.edge_type,
      strength_permille: edge.strength_permille,
      rationale: edge.rationale,
    },
  });
}
```

## Standards Source

Common Core State Standards for Mathematics:
https://www.thecorestandards.org/Math/

The standards text in these files is derived from the publicly available
Common Core State Standards, copyright 2010 National Governors Association
Center for Best Practices and Council of Chief State School Officers.

## Extending to Other Grades

To add Grade 4 or Grade 5 math:

1. Copy `grade3_math.json` as a template
2. Update `metadata` (grade_level, total_standards, domains)
3. Replace `nodes` with the target grade's standards
4. Map `edges` for within-grade prerequisites
5. Update `grade_levels` in each node (e.g., `"Grade4"`)
6. Add cross-grade edges where Grade 3 standards feed into Grade 4
   (e.g., 3.NF.A.3 -> 4.NF.A.1 for fraction equivalence progression)
7. Update `learning_paths` for the new grade's instructional sequences

The `grade2_prerequisites` section in Grade 3 shows the pattern for
documenting cross-grade dependencies. Grade 4 would have a corresponding
`grade3_prerequisites` section referencing the standards in this file.

### Grade progression for Common Core Math:
- **Grade 3**: 25 standards (5 domains: OA, NBT, NF, MD, G)
- **Grade 4**: 28 standards (5 domains: OA, NBT, NF, MD, G)
- **Grade 5**: 26 standards (5 domains: OA, NBT, NF, MD, G)

The domain structure remains consistent across grades, making it
straightforward to build vertical alignment edges between grade levels.
