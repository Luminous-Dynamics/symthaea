---

title: Infrastructure Loom Tech Node Data Schema
status: canonical-draft
milestone: seedworks-v0.1-to-v0.3
scope: data model, unlock logic, UI state, serialization, implementation
owner: design/engineering
depends_on:

* TECH_UNLOCK_TABLE_V0_1_TO_V0_3.md
* TECH_TREE_DEPENDENCY_SPINE.md
* INFRASTRUCTURE_LOOM_UI_UX_SPEC.md
* PUBLIC_WORKS_FABRICATION_BRANCH_V0_2.md
* ROBOTICS_PLATFORM_TECH_TREE_ADDENDUM.md
  recommended_path: docs/seedworks/04_engine/INFRASTRUCTURE_LOOM_TECH_NODE_SCHEMA.md

---

> **Code status (2026-07-02 review):** No corresponding implementation found in `symtropy/crates` or `symtropy/src`. Design/vision document only.

# Symtropy Engineering Spec: Infrastructure Loom Tech Node Data Schema

## Working Title

**Every Future Must Serialize**

## Core Thesis

The Infrastructure Loom must not be a hand-authored decorative map.

Every technology node should be a structured object that can be:

```text
displayed
filtered
queried
validated
locked
unlocked
corrupted
witnessed
recorded
tested
```

Core rule:

```text
A tech node is not a perk.
It is a claim that the world can now support a new capability.
```

---

# 1. Purpose

This document defines the shared data schema for the Infrastructure Loom.

It supports:

```text
Field Deck Loom View
Public Works Wall Terminal
Fabrication Bench recipes
Civic Kiosk permissions
Robot Dock authorization
Shared Tool Embassy xeno-translation
Chronicle event linking
NULL corruption overlays
```

The schema should work for:

```text
v0.1 survival repair nodes
v0.2 public works fabrication nodes
v0.3 robotics and xeno-translation nodes
v1.0 horizon nodes
```

Design rule:

```text
The UI should not invent tech-tree state.
It should render state produced by the simulation, Chronicle, and civic systems.
```

---

# 2. Core Concepts

Every Loom node has three identities.

## 2.1 Human-Readable Identity

Used by UI.

```text
Public Works Fabrication Bench
mk0-scout Cable-Crawler
Hybrid Filter Alpha
```

## 2.2 Stable Internal ID

Used by save files, tests, unlock logic, and Chronicle references.

```text
tech.public_works.fabrication_bench.v0_2
tech.robotics.mk0_scout.cable_crawler
tech.xeno.hybrid_filter_alpha
```

## 2.3 Device Bus Path

Used when the technology maps to an in-world system.

```text
/dev/sym/fabrication/public_works_bench_01
/dev/sym/robotics/mk0_scout_alpha
/dev/sym/hardware/hybrid_filter_alpha
```

Not every node needs a live Device Bus path.

For example:

```text
Rights Floor Warning
Regional Technician Passport
Atlas Gate Foreshadow
```

may exist as civic or roadmap nodes before they have device paths.

Design rule:

```text
Stable IDs are for code.
Device paths are for world-state.
Names are for humans.
```

---

# 3. Node Status Enum

Every node has one current status.

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TechNodeStatus {
    Playable,
    VisibleLocked,
    Foreshadowed,
    Stub,
    Roadmap,
    Deferred,
    RemovedFromScope,
    Corrupted,
    Deprecated,
}
```

## Status Meaning

```text
PLAYABLE:
The player can use this system directly.

VISIBLE_LOCKED:
The system exists in-world but is not yet usable.

FORESHADOWED:
The system is mentioned or glimpsed, but not interactable.

STUB:
Simplified implementation exists.

ROADMAP:
Designed for later versions but not present in current build.

DEFERRED:
Explicitly out of scope for the current milestone.

REMOVED_FROM_SCOPE:
Rejected for this milestone to protect focus.

CORRUPTED:
NULL mode, dead-authority, or data-integrity failure has compromised the node.

DEPRECATED:
Replaced by newer design but preserved for archive/history.
```

Design rule:

```text
The player should know whether a lock is physical, civic, technical, corrupted, or simply out of scope.
```

---

# 4. Milestone Enum

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeedworksMilestone {
    V0_1,
    V0_2,
    V0_3,
    V1_0Horizon,
    Expansion,
}
```

## Milestone Meaning

```text
v0.1:
The First Pipe

v0.2:
The First Workshop

v0.3:
The First Embassy

v1.0 Horizon:
Civilization-scale interlocking systems

Expansion:
Orbital, oceanic, interstellar, or post-v1.0 content
```

---

# 5. Discipline Enum

Each node may belong to one or more disciplines.

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TechDiscipline {
    ThermodynamicMaterialFabrication,
    ComputationalFieldArchitecture,
    SocioCivicLegitimacyChains,
    DeviceBusSubstrate,
    CargoAndLogistics,
    RoboticsAndAutomation,
    XenoTranslation,
    DeathAndReconstitution,
    GovernanceAndFactions,
    RegionalInfrastructure,
    InterstellarTransit,
}
```

Examples:

```text
Patch Conduit Mk0:
ThermodynamicMaterialFabrication
DeviceBusSubstrate

Proof-of-Repair Receipt:
SocioCivicLegitimacyChains
CargoAndLogistics

mk0-scout Cable-Crawler:
RoboticsAndAutomation
ComputationalFieldArchitecture
SocioCivicLegitimacyChains

Hybrid Filter Alpha:
XenoTranslation
ThermodynamicMaterialFabrication
SocioCivicLegitimacyChains
```

Design rule:

```text
The best nodes usually sit at discipline intersections.
```

---

# 6. Readiness Categories

Every node has six readiness categories.

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadinessSet {
    pub material: ReadinessMetric,
    pub power: ReadinessMetric,
    pub computation: ReadinessMetric,
    pub legitimacy: ReadinessMetric,
    pub maintenance: ReadinessMetric,
    pub consequence: ReadinessMetric,
}
```

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadinessMetric {
    pub percent: u8,
    pub state: ReadinessState,
    pub summary: String,
    pub blockers: Vec<String>,
}
```

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadinessState {
    Ready,
    Warning,
    Blocked,
    Unknown,
    Corrupted,
    NotApplicable,
}
```

Example:

```json
{
  "material": {
    "percent": 78,
    "state": "Warning",
    "summary": "Most materials present; motor service pack missing.",
    "blockers": ["robot_crawler_motor_service_pack"]
  },
  "power": {
    "percent": 71,
    "state": "Ready",
    "summary": "Dock voltage stable enough for supervised test.",
    "blockers": []
  },
  "computation": {
    "percent": 58,
    "state": "Warning",
    "summary": "Remote view available; route logging incomplete.",
    "blockers": ["route_manifest_logger"]
  },
  "legitimacy": {
    "percent": 49,
    "state": "Blocked",
    "summary": "Public inspection route not authorized.",
    "blockers": ["public_route_authorization"]
  },
  "maintenance": {
    "percent": 62,
    "state": "Warning",
    "summary": "Crawler dock damaged but serviceable.",
    "blockers": ["crawler_dock_repair"]
  },
  "consequence": {
    "percent": 55,
    "state": "Warning",
    "summary": "Machine witness dispute possible.",
    "blockers": ["machine_witness_policy"]
  }
}
```

Design rule:

```text
Readiness is not a progress bar.
It is an explanation of why the world is or is not ready.
```

---

# 7. Dependency Model

Each node can depend on other nodes, world events, resources, facilities, permissions, or Chronicle evidence.

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TechDependency {
    pub dependency_id: String,
    pub dependency_type: DependencyType,
    pub label: String,
    pub required_state: RequiredDependencyState,
    pub current_state: DependencyCurrentState,
    pub blocking: bool,
    pub visible_to_player: bool,
    pub field_deck_hint: Option<String>,
}
```

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyType {
    TechNode,
    DeviceBusNode,
    Material,
    Facility,
    PowerCondition,
    FieldDeckMode,
    ChronicleEvent,
    ProofOfRepair,
    CivicPermission,
    VoteOutcome,
    AccessClass,
    RobotPermissionEnvelope,
    XenoConsent,
    TranslationCalibration,
    FactionState,
    QuestOutcome,
}
```

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequiredDependencyState {
    Present,
    Active,
    Playable,
    Accepted,
    Certified,
    Authorized,
    Passed,
    Recognized,
    StableAboveThreshold,
    ForeshadowedOnly,
}
```

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyCurrentState {
    Missing,
    Present,
    Active,
    Failed,
    Partial,
    Disputed,
    Corrupted,
    Unknown,
    NotYetImplemented,
}
```

Design rule:

```text
A missing dependency should become a world action, not hidden spreadsheet logic.
```

---

# 8. Unlock Conditions

A node becomes playable when its unlock policy evaluates successfully.

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnlockPolicy {
    pub policy_id: String,
    pub required_all: Vec<String>,
    pub required_any: Vec<Vec<String>>,
    pub forbidden: Vec<String>,
    pub minimum_readiness: Option<MinimumReadiness>,
    pub allows_emergency_override: bool,
    pub override_risk: Option<String>,
}
```

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MinimumReadiness {
    pub material: Option<u8>,
    pub power: Option<u8>,
    pub computation: Option<u8>,
    pub legitimacy: Option<u8>,
    pub maintenance: Option<u8>,
    pub consequence: Option<u8>,
}
```

Example:

```json
{
  "policy_id": "unlock.public_works.fabrication_bench.v0_2",
  "required_all": [
    "chronicle.old_waterworks_outcome_recorded",
    "proof_of_repair.old_waterworks.accepted",
    "civic.firstlight_public_repair_charter.recognized"
  ],
  "required_any": [
    [
      "power.bench_voltage.stable_above_90",
      "civic.emergency_power_override.accepted"
    ]
  ],
  "forbidden": [
    "null.recipe_source_corrupted"
  ],
  "minimum_readiness": {
    "material": 40,
    "power": 70,
    "computation": 50,
    "legitimacy": 60,
    "maintenance": 40,
    "consequence": 50
  },
  "allows_emergency_override": true,
  "override_risk": "Bench can open under emergency authority, but produced parts cannot be certified."
}
```

Design rule:

```text
Emergency unlocks may permit action.
They should not grant full legitimacy.
```

---

# 9. Main Tech Node Struct

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TechNode {
    pub id: String,
    pub name: String,
    pub short_name: Option<String>,
    pub description: String,

    pub milestone: SeedworksMilestone,
    pub status: TechNodeStatus,
    pub disciplines: Vec<TechDiscipline>,
    pub dependency_layer: DependencyLayer,

    pub device_bus_path: Option<String>,
    pub parent_ids: Vec<String>,
    pub child_ids: Vec<String>,

    pub readiness: ReadinessSet,
    pub dependencies: Vec<TechDependency>,
    pub unlock_policy: UnlockPolicy,

    pub player_verbs: Vec<PlayerVerb>,
    pub facilities: Vec<String>,
    pub materials: Vec<String>,
    pub field_deck_modes: Vec<FieldDeckMode>,

    pub failure_modes: Vec<FailureModeRef>,
    pub chronicle_links: ChronicleLinkSet,

    pub ui: TechNodeUiMetadata,
    pub production_scope: ProductionScope,
}
```

---

# 10. Dependency Layer Enum

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyLayer {
    SurvivalRepair,
    FieldDeckDeviceBus,
    CargoAndMaterialLedgers,
    PowerAudioLaborSubstrate,
    PublicFabrication,
    RoboticsAndAutomation,
    CivicAndFactionInfrastructure,
    XenoTranslationLivingInfrastructure,
    RegionalInterstellarExpansion,
}
```

Design rule:

```text
The layer tells the player what kind of civilization problem the node belongs to.
```

---

# 11. Field Deck Mode Enum

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldDeckMode {
    Scan,
    Diag,
    Archive,
    Civic,
    Null,
    Witness,
    Repair,
}
```

Mode availability may vary by milestone.

```text
v0.1:
SCAN, DIAG, ARCHIVE, CIVIC playable.
NULL, WITNESS, REPAIR stubs.

v0.2:
REPAIR and WITNESS expanded.

v0.3:
Xeno-translation overlays appear through SCAN, DIAG, CIVIC, WITNESS.
```

---

# 12. Player Verb Enum

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlayerVerb {
    Walk,
    Scan,
    Read,
    Carry,
    Drop,
    Insert,
    Align,
    Seal,
    Initialize,
    Authorize,
    Commit,
    Fabricate,
    Calibrate,
    Test,
    Certify,
    Audit,
    Borrow,
    Return,
    Deploy,
    Recall,
    Witness,
    Vote,
    Negotiate,
    Translate,
    Quarantine,
}
```

Design rule:

```text
No unlock without a verb.
No verb without a possible failure.
```

---

# 13. Failure Mode Reference

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FailureModeRef {
    pub id: String,
    pub label: String,
    pub severity: FailureSeverity,
    pub visible_in_modes: Vec<FieldDeckMode>,
    pub consequence_summary: String,
}
```

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureSeverity {
    Minor,
    Major,
    Critical,
    Existential,
}
```

Examples:

```json
{
  "id": "failure.witness_rejected",
  "label": "WITNESS_REJECTED",
  "severity": "Major",
  "visible_in_modes": ["Civic", "Witness"],
  "consequence_summary": "Machine log is accurate but not accepted as sufficient testimony."
}
```

```json
{
  "id": "failure.translation_collapse",
  "label": "TRANSLATION_COLLAPSE",
  "severity": "Critical",
  "visible_in_modes": ["Diag", "Civic", "Null"],
  "consequence_summary": "Hybrid component no longer maps human control to alien metabolic state."
}
```

---

# 14. Chronicle Link Set

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChronicleLinkSet {
    pub unlocked_by_events: Vec<String>,
    pub can_emit_events: Vec<String>,
    pub related_events: Vec<String>,
    pub chronicle_line_preview: Option<String>,
}
```

Example:

```json
{
  "unlocked_by_events": [
    "event.old_waterworks_outcome_recorded",
    "event.proof_of_repair_issued"
  ],
  "can_emit_events": [
    "event.public_works_bench_reopened",
    "event.certified_seal_installed"
  ],
  "related_events": [
    "event.dead_authority_lock_inspected"
  ],
  "chronicle_line_preview": "The bench opened not because someone owned it, but because the repair had been witnessed."
}
```

Design rule:

```text
A node is not fully real until the Chronicle can say why it changed.
```

---

# 15. UI Metadata

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TechNodeUiMetadata {
    pub icon: String,
    pub branch_color_hint: BranchColorHint,
    pub layout_position: Option<LoomPosition>,
    pub compact_label: String,
    pub mode_copy: ModeCopySet,
    pub locked_explanation: Option<LockedExplanation>,
}
```

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchColorHint {
    Material,
    Power,
    Computation,
    Legitimacy,
    Maintenance,
    Consequence,
    Robotics,
    Xeno,
    Null,
    Archive,
}
```

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoomPosition {
    pub x: f32,
    pub y: f32,
    pub layer: i32,
}
```

Design rule:

```text
Layout hints are allowed, but dependency logic should not depend on screen position.
```

---

# 16. Mode Copy Set

Every node can expose different text depending on Field Deck mode.

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModeCopySet {
    pub scan: Option<String>,
    pub diag: Option<String>,
    pub archive: Option<String>,
    pub civic: Option<String>,
    pub null: Option<String>,
    pub witness: Option<String>,
    pub repair: Option<String>,
}
```

Example:

```json
{
  "scan": "Bench frame detected. Tool sockets intact. Ceramic seal tray missing.",
  "diag": "Bench power unstable. Transformer feed below certified fabrication threshold.",
  "archive": "Bench last opened under emergency repair authority in 2113.",
  "civic": "Proof-of-Repair required before public machine access.",
  "null": "Unsafe bypass claims bench can open without witness chain. Source unverified.",
  "witness": "Required evidence: Old Waterworks outcome, Archive Witness signature, accepted repair receipt.",
  "repair": "Bench cannot fabricate certified parts until calibration cycle passes."
}
```

Design rule:

```text
The same node should tell different truths under different instruments.
```

---

# 17. Locked Explanation

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LockedExplanation {
    pub what: String,
    pub why_locked: String,
    pub do_now: Vec<String>,
    pub evidence_missing: Vec<String>,
    pub facility_missing: Vec<String>,
    pub risk: Vec<String>,
    pub chronicle_unlock: Vec<String>,
}
```

Example:

```json
{
  "what": "A public machine for producing certified infrastructure parts.",
  "why_locked": "Proof-of-Repair has not yet been accepted by the Firstlight Public Repair Charter.",
  "do_now": [
    "Restore Old Waterworks.",
    "Commit repair outcome to Chronicle.",
    "Recover Archive Witness Cartridge signature."
  ],
  "evidence_missing": [
    "Archive Witness Cartridge signature",
    "Proof-of-Repair receipt"
  ],
  "facility_missing": [
    "Stable bench power above 90%"
  ],
  "risk": [
    "Dead authority lock may spoof recipe access."
  ],
  "chronicle_unlock": [
    "OldWaterworksOutcomeRecorded",
    "ProofOfRepairIssued"
  ]
}
```

Design rule:

```text
A locked node should become a quest, not a wall.
```

---

# 18. Production Scope

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProductionScope {
    pub implemented: bool,
    pub implementation_status: ImplementationStatus,
    pub current_build_flag: Option<String>,
    pub scope_warning: Option<String>,
    pub test_fixture_id: Option<String>,
}
```

```rust
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImplementationStatus {
    NotStarted,
    Stubbed,
    PlayablePrototype,
    VerticalSliceReady,
    ProductionReady,
    Deferred,
}
```

Example:

```json
{
  "implemented": false,
  "implementation_status": "Deferred",
  "current_build_flag": null,
  "scope_warning": "Foreshadow only in v0.1. Do not implement full fabrication before Old Waterworks loop is stable.",
  "test_fixture_id": null
}
```

Design rule:

```text
The data schema should protect production scope as much as player immersion.
```

---

# 19. Example Node: Public Works Fabrication Bench

```json
{
  "id": "tech.public_works.fabrication_bench.v0_2",
  "name": "Public Works Fabrication Bench",
  "short_name": "Fabrication Bench",
  "description": "A public machine for producing certified infrastructure parts after witnessed repair unlocks public trust.",
  "milestone": "V0_2",
  "status": "VisibleLocked",
  "disciplines": [
    "ThermodynamicMaterialFabrication",
    "SocioCivicLegitimacyChains"
  ],
  "dependency_layer": "PublicFabrication",
  "device_bus_path": "/dev/sym/fabrication/public_works_bench_01",
  "parent_ids": [
    "tech.proof_of_repair.old_waterworks.v0_1",
    "tech.chronicle.jsonl.v0_1"
  ],
  "child_ids": [
    "tech.fabrication.certified_seal_kit.v0_2",
    "tech.fabrication.pressure_test_rig.v0_2",
    "tech.robotics.mk0_scout.cable_crawler"
  ],
  "readiness": {
    "material": {
      "percent": 60,
      "state": "Warning",
      "summary": "Bench exists but calibration materials are missing.",
      "blockers": ["ceramic_seal_blank", "pressure_reference_sample"]
    },
    "power": {
      "percent": 84,
      "state": "Warning",
      "summary": "Bench power below certified threshold.",
      "blockers": ["bench_voltage_stable_above_90"]
    },
    "computation": {
      "percent": 65,
      "state": "Ready",
      "summary": "Field Deck can inspect and initialize bench.",
      "blockers": []
    },
    "legitimacy": {
      "percent": 40,
      "state": "Blocked",
      "summary": "Proof-of-Repair has not been accepted.",
      "blockers": ["proof_of_repair.old_waterworks.accepted"]
    },
    "maintenance": {
      "percent": 50,
      "state": "Warning",
      "summary": "Calibration cycle required.",
      "blockers": ["bench_calibration_passed"]
    },
    "consequence": {
      "percent": 70,
      "state": "Ready",
      "summary": "Bench reopening can emit Chronicle event.",
      "blockers": []
    }
  },
  "dependencies": [
    {
      "dependency_id": "proof_of_repair.old_waterworks.accepted",
      "dependency_type": "ProofOfRepair",
      "label": "Accepted Old Waterworks Proof-of-Repair",
      "required_state": "Accepted",
      "current_state": "Missing",
      "blocking": true,
      "visible_to_player": true,
      "field_deck_hint": "Restore Old Waterworks and commit repair outcome."
    }
  ],
  "unlock_policy": {
    "policy_id": "unlock.public_works.fabrication_bench.v0_2",
    "required_all": [
      "chronicle.old_waterworks_outcome_recorded",
      "proof_of_repair.old_waterworks.accepted"
    ],
    "required_any": [],
    "forbidden": [
      "null.recipe_source_corrupted"
    ],
    "minimum_readiness": {
      "material": 40,
      "power": 70,
      "computation": 50,
      "legitimacy": 60,
      "maintenance": 40,
      "consequence": 50
    },
    "allows_emergency_override": true,
    "override_risk": "Bench can open under emergency authority, but produced parts cannot be certified."
  },
  "player_verbs": [
    "Inspect",
    "LoadMaterial",
    "Fabricate",
    "Calibrate",
    "Certify",
    "Log"
  ],
  "facilities": [
    "/dev/sym/fabrication/public_works_bench_01"
  ],
  "materials": [
    "scrap_metal",
    "ceramic_seal_blank",
    "copper_conduit"
  ],
  "field_deck_modes": [
    "Scan",
    "Diag",
    "Civic",
    "Witness",
    "Null"
  ],
  "failure_modes": [
    {
      "id": "failure.material_unverified",
      "label": "MATERIAL_UNVERIFIED",
      "severity": "Major",
      "visible_in_modes": ["Diag", "Witness"],
      "consequence_summary": "Recipe may complete, but output cannot be certified."
    },
    {
      "id": "failure.null_recipe_contamination",
      "label": "NULL_RECIPE_CONTAMINATION",
      "severity": "Critical",
      "visible_in_modes": ["Null", "Diag"],
      "consequence_summary": "Recipe source may produce unsafe public works component."
    }
  ],
  "chronicle_links": {
    "unlocked_by_events": [
      "event.old_waterworks_outcome_recorded",
      "event.proof_of_repair_issued"
    ],
    "can_emit_events": [
      "event.public_works_bench_reopened"
    ],
    "related_events": [],
    "chronicle_line_preview": "The bench opened not because someone owned it, but because the repair had been witnessed."
  },
  "ui": {
    "icon": "fabrication_bench",
    "branch_color_hint": "Material",
    "layout_position": {
      "x": -0.4,
      "y": 0.1,
      "layer": 4
    },
    "compact_label": "Public Works Bench",
    "mode_copy": {
      "scan": "Bench frame detected. Tool sockets intact. Ceramic seal tray missing.",
      "diag": "Bench power unstable. Transformer feed below certified fabrication threshold.",
      "archive": "Bench last opened under emergency repair authority in 2113.",
      "civic": "Proof-of-Repair required before public machine access.",
      "null": "Unsafe bypass claims bench can open without witness chain. Source unverified.",
      "witness": "Required evidence: Old Waterworks outcome, Archive Witness signature, accepted repair receipt.",
      "repair": "Bench cannot fabricate certified parts until calibration cycle passes."
    },
    "locked_explanation": {
      "what": "A public machine for producing certified infrastructure parts.",
      "why_locked": "Proof-of-Repair has not yet been accepted by Firstlight Public Repair Charter.",
      "do_now": [
        "Restore Old Waterworks.",
        "Commit repair outcome to Chronicle.",
        "Recover Archive Witness Cartridge signature."
      ],
      "evidence_missing": [
        "Proof-of-Repair receipt"
      ],
      "facility_missing": [
        "Stable bench power above 90%"
      ],
      "risk": [
        "Dead authority lock may spoof recipe access."
      ],
      "chronicle_unlock": [
        "OldWaterworksOutcomeRecorded",
        "ProofOfRepairIssued"
      ]
    }
  },
  "production_scope": {
    "implemented": false,
    "implementation_status": "Stubbed",
    "current_build_flag": "loom_public_works_bench_stub",
    "scope_warning": "Visible-locked in v0.1. Playable in v0.2.",
    "test_fixture_id": "fixture.tech.public_works_bench.v0_2"
  }
}
```

---

# 20. Example Node: mk0-scout Cable-Crawler

```json
{
  "id": "tech.robotics.mk0_scout.cable_crawler",
  "name": "mk0-scout Cable-Crawler",
  "short_name": "Cable-Crawler",
  "description": "A supervised overhead inspection robot for route scouting, hazard marking, and visual witness support.",
  "milestone": "V0_2",
  "status": "VisibleLocked",
  "disciplines": [
    "RoboticsAndAutomation",
    "ComputationalFieldArchitecture",
    "SocioCivicLegitimacyChains"
  ],
  "dependency_layer": "RoboticsAndAutomation",
  "device_bus_path": "/dev/sym/robotics/mk0_scout_alpha",
  "parent_ids": [
    "tech.public_works.fabrication_bench.v0_2",
    "tech.fabrication.robot_crawler_motor_service_pack.v0_2"
  ],
  "child_ids": [
    "tech.civic.machine_testimony_review.v0_3"
  ],
  "readiness": {
    "material": {
      "percent": 78,
      "state": "Warning",
      "summary": "Crawler chassis present; motor service pack missing.",
      "blockers": ["robot_crawler_motor_service_pack"]
    },
    "power": {
      "percent": 71,
      "state": "Ready",
      "summary": "Dock voltage stable enough for supervised routine.",
      "blockers": []
    },
    "computation": {
      "percent": 58,
      "state": "Warning",
      "summary": "Remote view available; route logging incomplete.",
      "blockers": ["route_manifest_logger"]
    },
    "legitimacy": {
      "percent": 49,
      "state": "Blocked",
      "summary": "Public inspection route not authorized.",
      "blockers": ["public_route_authorization"]
    },
    "maintenance": {
      "percent": 62,
      "state": "Warning",
      "summary": "Crawler dock damaged but serviceable.",
      "blockers": ["crawler_dock_repair"]
    },
    "consequence": {
      "percent": 55,
      "state": "Warning",
      "summary": "Machine witness dispute possible.",
      "blockers": ["machine_witness_policy"]
    }
  },
  "dependencies": [
    {
      "dependency_id": "fabrication.robot_crawler_motor_service_pack.present",
      "dependency_type": "Material",
      "label": "Robot Crawler Motor Service Pack",
      "required_state": "Present",
      "current_state": "Missing",
      "blocking": true,
      "visible_to_player": true,
      "field_deck_hint": "Fabricate motor service pack at Public Works Fabrication Bench."
    },
    {
      "dependency_id": "civic.public_inspection_route.authorized",
      "dependency_type": "CivicPermission",
      "label": "Public inspection route authorization",
      "required_state": "Authorized",
      "current_state": "Disputed",
      "blocking": true,
      "visible_to_player": true,
      "field_deck_hint": "Submit route authorization request at Civic Kiosk."
    }
  ],
  "unlock_policy": {
    "policy_id": "unlock.robotics.mk0_scout.v0_2",
    "required_all": [
      "tech.public_works.fabrication_bench.playable",
      "fabrication.robot_crawler_motor_service_pack.present",
      "civic.public_inspection_route.authorized"
    ],
    "required_any": [],
    "forbidden": [
      "null.robot_command_echo_active"
    ],
    "minimum_readiness": {
      "material": 70,
      "power": 60,
      "computation": 55,
      "legitimacy": 60,
      "maintenance": 50,
      "consequence": 40
    },
    "allows_emergency_override": false,
    "override_risk": null
  },
  "player_verbs": [
    "Deploy",
    "Recall",
    "Scan",
    "Witness",
    "Audit"
  ],
  "facilities": [
    "/dev/sym/robotics/mk0_scout_alpha",
    "/dev/sym/fabrication/public_works_bench_01"
  ],
  "materials": [
    "robot_crawler_motor_service_pack",
    "battery_cell",
    "sensor_lens"
  ],
  "field_deck_modes": [
    "Scan",
    "Diag",
    "Civic",
    "Witness",
    "Null"
  ],
  "failure_modes": [
    {
      "id": "failure.witness_rejected",
      "label": "WITNESS_REJECTED",
      "severity": "Major",
      "visible_in_modes": ["Civic", "Witness"],
      "consequence_summary": "Machine visual log is accurate but not accepted as sufficient testimony."
    },
    {
      "id": "failure.route_boundary_denied",
      "label": "ROUTE_BOUNDARY_DENIED",
      "severity": "Minor",
      "visible_in_modes": ["Civic", "Diag"],
      "consequence_summary": "Crawler stops at edge of permission envelope."
    }
  ],
  "chronicle_links": {
    "unlocked_by_events": [
      "event.public_works_bench_reopened",
      "event.robot_route_authorized"
    ],
    "can_emit_events": [
      "event.machine_visual_log_submitted",
      "event.robot_witness_disputed"
    ],
    "related_events": [],
    "chronicle_line_preview": "The crawler saw what the operator could not safely reach, and the settlement argued whether a machine’s sight counted as witness."
  },
  "ui": {
    "icon": "robot_crawler",
    "branch_color_hint": "Robotics",
    "layout_position": {
      "x": 0.0,
      "y": 0.45,
      "layer": 5
    },
    "compact_label": "mk0-scout",
    "mode_copy": {
      "scan": "Crawler chassis present. Motor compartment open. Overhead cable route detected.",
      "diag": "Dock voltage stable. Route logger incomplete. Motor service pack missing.",
      "archive": "Inspection crawler rail installed during flood-control modernization.",
      "civic": "Public route authorization required. Machine witness policy unresolved.",
      "null": "Command echo risk detected in old route scheduler.",
      "witness": "Visual log may support but not replace human testimony.",
      "repair": "Install motor service pack before deployment."
    },
    "locked_explanation": {
      "what": "A supervised overhead inspection robot.",
      "why_locked": "Motor service pack and route authorization are missing.",
      "do_now": [
        "Fabricate motor service pack.",
        "Repair crawler dock.",
        "Submit public inspection route authorization."
      ],
      "evidence_missing": [
        "Route permission record",
        "Machine witness policy"
      ],
      "facility_missing": [
        "Crawler dock service pack"
      ],
      "risk": [
        "Crawler testimony may be rejected."
      ],
      "chronicle_unlock": [
        "RobotRouteAuthorized"
      ]
    }
  },
  "production_scope": {
    "implemented": false,
    "implementation_status": "NotStarted",
    "current_build_flag": null,
    "scope_warning": "Foreshadow in v0.1. First playable robotics candidate in v0.2.",
    "test_fixture_id": "fixture.tech.robotics.mk0_scout"
  }
}
```

---

# 21. Example Node: Hybrid Filter Alpha

```json
{
  "id": "tech.xeno.hybrid_filter_alpha",
  "name": "Hybrid Filter Alpha",
  "short_name": "Hybrid Filter",
  "description": "A xeno-hybrid water filtration component built through Tideborn metabolic exchange and Rights Forum licensing.",
  "milestone": "V0_3",
  "status": "Foreshadowed",
  "disciplines": [
    "XenoTranslation",
    "ThermodynamicMaterialFabrication",
    "SocioCivicLegitimacyChains"
  ],
  "dependency_layer": "XenoTranslationLivingInfrastructure",
  "device_bus_path": "/dev/sym/hardware/hybrid_filter_alpha",
  "parent_ids": [
    "tech.xeno.shared_tool_embassy",
    "tech.fabrication.biofilter_housing.v0_2"
  ],
  "child_ids": [
    "tech.xeno.translation_collapse",
    "tech.xeno.living_infrastructure_consent"
  ],
  "readiness": {
    "material": {
      "percent": 35,
      "state": "Blocked",
      "summary": "Biofilter housing prerequisite incomplete.",
      "blockers": ["biofilter_housing"]
    },
    "power": {
      "percent": 50,
      "state": "Warning",
      "summary": "Bio-electric conversion not calibrated.",
      "blockers": ["bio_electric_converter"]
    },
    "computation": {
      "percent": 30,
      "state": "Blocked",
      "summary": "Translation runtime unavailable.",
      "blockers": ["translation_pool"]
    },
    "legitimacy": {
      "percent": 10,
      "state": "Blocked",
      "summary": "Rights Forum license missing.",
      "blockers": ["multi_species_rights_forum_license"]
    },
    "maintenance": {
      "percent": 20,
      "state": "Blocked",
      "summary": "Metabolic maintenance protocol unavailable.",
      "blockers": ["metabolic_stabilizer"]
    },
    "consequence": {
      "percent": 80,
      "state": "Warning",
      "summary": "Hybrid failure can alter public water infrastructure.",
      "blockers": []
    }
  },
  "dependencies": [
    {
      "dependency_id": "xeno.tideborn_exchange.completed",
      "dependency_type": "XenoConsent",
      "label": "Tideborn Water-Civic exchange",
      "required_state": "Accepted",
      "current_state": "Missing",
      "blocking": true,
      "visible_to_player": true,
      "field_deck_hint": "Open Shared Tool Embassy and complete metabolic exchange."
    },
    {
      "dependency_id": "civic.rights_forum.hybrid_license",
      "dependency_type": "CivicPermission",
      "label": "Multi-Species Rights Forum license",
      "required_state": "Authorized",
      "current_state": "Missing",
      "blocking": true,
      "visible_to_player": true,
      "field_deck_hint": "Submit hybrid component for Rights Forum review."
    }
  ],
  "unlock_policy": {
    "policy_id": "unlock.xeno.hybrid_filter_alpha.v0_3",
    "required_all": [
      "tech.xeno.shared_tool_embassy.playable",
      "xeno.tideborn_exchange.completed",
      "civic.rights_forum.hybrid_license",
      "tech.fabrication.biofilter_housing.playable"
    ],
    "required_any": [],
    "forbidden": [
      "null.translation_drift_critical",
      "civic.consent_boundary_violated"
    ],
    "minimum_readiness": {
      "material": 70,
      "power": 60,
      "computation": 75,
      "legitimacy": 80,
      "maintenance": 70,
      "consequence": 50
    },
    "allows_emergency_override": false,
    "override_risk": null
  },
  "player_verbs": [
    "Handshake",
    "Calibrate",
    "Translate",
    "Stabilize",
    "License",
    "Quarantine"
  ],
  "facilities": [
    "/dev/sym/xeno/shared_tool_embassy",
    "/dev/sym/xeno/translation_pool",
    "/dev/sym/hardware/hybrid_filter_alpha"
  ],
  "materials": [
    "biofilter_housing",
    "tideborn_chemical_memory_block",
    "bio_electric_converter"
  ],
  "field_deck_modes": [
    "Scan",
    "Diag",
    "Civic",
    "Witness",
    "Null"
  ],
  "failure_modes": [
    {
      "id": "failure.translation_collapse",
      "label": "TRANSLATION_COLLAPSE",
      "severity": "Critical",
      "visible_in_modes": ["Diag", "Civic", "Null"],
      "consequence_summary": "Hybrid filter no longer maps human controls to alien metabolic state."
    },
    {
      "id": "failure.overgrowth_without_consent",
      "label": "OVERGROWTH_WITHOUT_CONSENT",
      "severity": "Critical",
      "visible_in_modes": ["Civic", "Diag", "Null"],
      "consequence_summary": "Living infrastructure grows beyond licensed maintenance boundary."
    }
  ],
  "chronicle_links": {
    "unlocked_by_events": [
      "event.shared_tool_embassy_opened",
      "event.tideborn_exchange_completed",
      "event.hybrid_filter_licensed"
    ],
    "can_emit_events": [
      "event.hybrid_filter_installed",
      "event.translation_collapse_contained",
      "event.living_infrastructure_consent_dispute"
    ],
    "related_events": [],
    "chronicle_line_preview": "The filter cleaned the water only because the settlement learned what the living membrane needed in return."
  },
  "ui": {
    "icon": "hybrid_filter",
    "branch_color_hint": "Xeno",
    "layout_position": {
      "x": 0.0,
      "y": 0.85,
      "layer": 7
    },
    "compact_label": "Hybrid Filter",
    "mode_copy": {
      "scan": "Hybrid containment geometry required. Human filter housing insufficient.",
      "diag": "Bio-electric conversion not calibrated. Translation runtime unavailable.",
      "archive": "No local precedent for Tideborn hybrid filtration.",
      "civic": "Rights Forum license required. Consent boundary must be defined before installation.",
      "null": "Shortcut claims alien membrane can be mounted without metabolic stabilizer. Unsafe.",
      "witness": "Required evidence: Tideborn exchange, Rights Forum license, calibration record.",
      "repair": "Cannot fabricate as normal recipe. Use Shared Tool Embassy translation pool."
    },
    "locked_explanation": {
      "what": "A xeno-hybrid water filtration component.",
      "why_locked": "Human infrastructure cannot yet host Tideborn metabolic logic safely.",
      "do_now": [
        "Build Biofilter Housing.",
        "Open Shared Tool Embassy.",
        "Complete Tideborn metabolic exchange.",
        "Obtain Rights Forum license."
      ],
      "evidence_missing": [
        "Tideborn consent",
        "Rights Forum license",
        "Translation calibration record"
      ],
      "facility_missing": [
        "Shared Tool Embassy",
        "Translation Pool",
        "Metabolic Stabilizer"
      ],
      "risk": [
        "Translation Collapse",
        "Overgrowth Without Consent",
        "Null Drift Amplification"
      ],
      "chronicle_unlock": [
        "SharedToolEmbassyOpened",
        "HybridFilterLicensed"
      ]
    }
  },
  "production_scope": {
    "implemented": false,
    "implementation_status": "Deferred",
    "current_build_flag": null,
    "scope_warning": "Foreshadow in v0.1 and v0.2. Playable no earlier than v0.3.",
    "test_fixture_id": "fixture.tech.xeno.hybrid_filter_alpha"
  }
}
```

---

# 22. Save / Runtime State Separation

The authored tech node should not be mutated directly.

Use two layers.

## Authored Definition

Static data.

```text
name
description
dependencies
unlock policy
mode copy
failure modes
UI metadata
```

## Runtime State

Dynamic data.

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TechNodeRuntimeState {
    pub node_id: String,
    pub status: TechNodeStatus,
    pub readiness: ReadinessSet,
    pub dependency_states: Vec<DependencyRuntimeState>,
    pub discovered: bool,
    pub pinned: bool,
    pub corrupted: bool,
    pub last_updated_tick: u64,
    pub unlocked_by_event: Option<String>,
}
```

Design rule:

```text
Definitions describe the possible future.
Runtime state describes what this settlement has actually earned.
```

---

# 23. Validation Rules

Every authored node must pass schema validation.

## Required Validation

```text
id is unique
name is non-empty
milestone is present
status is present
at least one discipline is present
dependency layer is present
readiness has all six categories
all dependency IDs resolve or are external-typed
all child IDs resolve or are roadmap-marked
all failure modes have severity
all playable nodes have at least one player verb
all visible-locked nodes have locked explanation
all xeno nodes have consent dependency
all robotics nodes have autonomy or permission metadata
all nodes with Device Bus paths use /dev/sym/*
```

## Forbidden

```text
PLAYABLE node with no player verbs
VISIBLE_LOCKED node with no explanation
robot node with no civic permission
xeno node with no consent requirement
fabrication node with no material dependency
Chronicle-producing node with no Chronicle link
NULL-corrupted node without visible warning
```

Design rule:

```text
Bad data should fail before bad futures reach the player.
```

---

# 24. v0.1 Minimal Node Set

The first implementation should only need these nodes:

```text
tech.field_deck.mk0
tech.field_deck.mode.scan
tech.field_deck.mode.diag
tech.field_deck.mode.archive
tech.field_deck.mode.civic
tech.repair.patch_conduit_mk0
tech.cargo.copper_conduit_segment
tech.archive.witness_cartridge
tech.device_bus.patch_conduit_alpha
tech.substrate.power.transformer_2_readout
tech.substrate.audio.pump_1_diagnostic
tech.chronicle.jsonl_v0
tech.proof_of_repair.old_waterworks
tech.public_works.fabrication_bench.v0_2
```

v0.1 requirement:

```text
The player completes Old Waterworks repair.
Proof-of-Repair activates.
Public Works Fabrication Bench changes state from FORESHADOWED to VISIBLE_LOCKED.
```

Design rule:

```text
The first Loom does not need many nodes.
It needs one node that changes because of what the player did.
```

---

# 25. Implementation Tickets

## L1 — Define Core Enums

Implement:

```text
TechNodeStatus
SeedworksMilestone
TechDiscipline
DependencyLayer
FieldDeckMode
PlayerVerb
```

Acceptance:

```text
enums serialize and deserialize
unit tests cover all variants
```

---

## L2 — Define TechNode Struct

Implement:

```text
TechNode
ReadinessSet
TechDependency
UnlockPolicy
ChronicleLinkSet
TechNodeUiMetadata
```

Acceptance:

```text
fixture node loads from JSON
schema rejects missing required fields
```

---

## L3 — Add v0.1 Fixture Nodes

Create fixtures for minimal v0.1 node set.

Acceptance:

```text
all fixtures validate
dependencies resolve
fabrication bench appears locked before Proof-of-Repair
```

---

## L4 — Runtime State Resolver

Implement a resolver that computes:

```text
current status
readiness summaries
dependency states
locked explanation
```

Acceptance:

```text
Old Waterworks completion activates Proof-of-Repair node
fabrication bench state changes accordingly
```

---

## L5 — Field Deck Mode Copy Resolver

Given node + mode, return mode-specific text.

Acceptance:

```text
SCAN, DIAG, CIVIC, NULL, WITNESS return different text for same node
fallback text exists
```

---

## L6 — Unlock Policy Evaluator

Implement simple required-all and required-any logic.

Acceptance:

```text
bench remains locked without Proof-of-Repair
bench unlock policy passes after required events
forbidden NULL state blocks unlock
```

---

## L7 — Chronicle Link Integration

Connect node state to Chronicle event IDs.

Acceptance:

```text
Proof-of-Repair node references ProofOfRepairIssued event
bench node can emit PublicWorksBenchReopened event
```

---

## L8 — UI Adapter

Create a view model for rendering.

```rust
pub struct TechNodeViewModel {
    pub id: String,
    pub label: String,
    pub status: TechNodeStatus,
    pub readiness: ReadinessSet,
    pub mode_text: String,
    pub dependency_lines: Vec<DependencyLineView>,
    pub actions: Vec<NodeAction>,
}
```

Acceptance:

```text
view model can render node detail panel
dependency line type is visible
locked explanation populates UI
```

---

# 26. Acceptance Test

The schema succeeds if:

```text
1. Designers can author a node without writing code.
2. Engineers can validate node data.
3. UI can render a node from schema.
4. Runtime can change node state after Chronicle events.
5. Locked nodes explain themselves.
6. NULL corruption can block or distort a node.
7. Robotics and xeno nodes require civic/consent dependencies.
8. v0.1 can ship with a tiny but meaningful Loom.
```

The schema fails if:

```text
nodes become abstract perk cards
unlocks happen without world evidence
UI state is hand-coded separately from node data
civic legitimacy is optional
Chronicle links are decorative only
robot/xeno nodes behave like ordinary recipes
```

---

# 27. Final Principles

```text
Every future must serialize.

A tech node is a claim.
A dependency is a responsibility.
A readiness bar is an explanation.
A lock is a quest.
An unlock is a public event.
A shortcut is a risk.
A Chronicle link is memory.

The schema should make it harder to create shallow technologies.

No node without a verb.
No verb without a failure.
No failure without a record.
No record without consequence.
```

Final line:

```text
The Loom did not store upgrades.
It stored the conditions under which the settlement was finally ready to become more capable.
```
