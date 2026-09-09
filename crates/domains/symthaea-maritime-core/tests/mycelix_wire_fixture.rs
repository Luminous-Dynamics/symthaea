// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::Deserialize;

const ROOT_FIXTURE: &str = include_str!("../fixtures/mycelix-maritime-evidence-v1-root.json");
const CHAINED_FIXTURE: &str =
    include_str!("../fixtures/mycelix-maritime-evidence-v1-chained.json");
const ROOT_DIGEST: &str = "2ec31adc7f4ca9200d6c3464171ca639b3075da93e4118bfd7b4688232dfdbc4";
const CHAINED_DIGEST: &str =
    "95b18b682028d2a463633f32ff4aecbdf415d988b83de60f22f5526166d6bdfe";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MaritimeEvidenceKindV1 {
    StateObservation,
    HealthObservation,
    AssuranceTransition,
    AuthorityTransition,
    PositionEvidenceReference,
    CommunicationsState,
    LogisticsEvent,
    MaintenanceEvent,
    RecoveryEvent,
}

impl MaritimeEvidenceKindV1 {
    fn label(self) -> &'static str {
        match self {
            Self::StateObservation => "state_observation",
            Self::HealthObservation => "health_observation",
            Self::AssuranceTransition => "assurance_transition",
            Self::AuthorityTransition => "authority_transition",
            Self::PositionEvidenceReference => "position_evidence_reference",
            Self::CommunicationsState => "communications_state",
            Self::LogisticsEvent => "logistics_event",
            Self::MaintenanceEvent => "maintenance_event",
            Self::RecoveryEvent => "recovery_event",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct MycelixMaritimeEvidenceV1 {
    schema_version: u8,
    platform_id: String,
    generation: u64,
    sequence: u64,
    observed_at_us: u64,
    kind: MaritimeEvidenceKindV1,
    payload_json: String,
    evidence_binding: String,
    #[serde(default)]
    position_evidence_refs: Vec<String>,
    previous_event_digest: Option<String>,
}

impl MycelixMaritimeEvidenceV1 {
    fn content_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"mycelix-maritime-evidence-v1\0");
        hash_bytes(&mut hasher, &[self.schema_version]);
        hash_bytes(&mut hasher, self.platform_id.as_bytes());
        hash_bytes(&mut hasher, &self.generation.to_le_bytes());
        hash_bytes(&mut hasher, &self.sequence.to_le_bytes());
        hash_bytes(&mut hasher, &self.observed_at_us.to_le_bytes());
        hash_bytes(&mut hasher, self.kind.label().as_bytes());
        hash_bytes(&mut hasher, self.payload_json.as_bytes());
        hash_bytes(&mut hasher, self.evidence_binding.as_bytes());
        for reference in &self.position_evidence_refs {
            hash_bytes(&mut hasher, reference.as_bytes());
        }
        match &self.previous_event_digest {
            Some(previous) => {
                hasher.update(&[1]);
                hash_bytes(&mut hasher, previous.as_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hasher.finalize().to_hex().to_string()
    }

    fn verify_successor(&self, next: &Self) -> Result<(), &'static str> {
        if self.platform_id != next.platform_id {
            return Err("platform changed");
        }
        if next.sequence != self.sequence.checked_add(1).ok_or("sequence overflow")? {
            return Err("sequence discontinuity");
        }
        let digest = self.content_digest();
        if next.previous_event_digest.as_deref() != Some(digest.as_str()) {
            return Err("predecessor binding mismatch");
        }
        Ok(())
    }
}

fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[test]
fn mycelix_v1_root_fixture_matches_cross_repo_wire_contract() {
    let envelope: MycelixMaritimeEvidenceV1 = serde_json::from_str(ROOT_FIXTURE).unwrap();

    assert_eq!(envelope.schema_version, 1);
    assert_eq!(envelope.platform_id, "auv-01");
    assert_eq!(envelope.generation, 7);
    assert_eq!(envelope.sequence, 42);
    assert_eq!(envelope.observed_at_us, 1_700_000_000_000_042);
    assert_eq!(envelope.kind, MaritimeEvidenceKindV1::HealthObservation);
    assert_eq!(
        envelope.payload_json,
        r#"{"severity":"healthy","envelope":"normal"}"#
    );
    assert_eq!(
        envelope.position_evidence_refs,
        vec!["mycelix-position:measurement:fixture-001".to_string()]
    );
    assert!(envelope.previous_event_digest.is_none());
    assert_eq!(envelope.content_digest(), ROOT_DIGEST);
}

#[test]
fn mycelix_v1_chained_fixture_independently_binds_exact_root() {
    let root: MycelixMaritimeEvidenceV1 = serde_json::from_str(ROOT_FIXTURE).unwrap();
    let chained: MycelixMaritimeEvidenceV1 = serde_json::from_str(CHAINED_FIXTURE).unwrap();

    assert_eq!(chained.sequence, 43);
    assert_eq!(chained.kind, MaritimeEvidenceKindV1::CommunicationsState);
    assert_eq!(
        chained.previous_event_digest.as_deref(),
        Some(ROOT_DIGEST)
    );
    assert_eq!(root.verify_successor(&chained), Ok(()));
    assert_eq!(chained.content_digest(), CHAINED_DIGEST);

    let mut tampered = root;
    tampered.payload_json = r#"{"severity":"unsafe"}"#.into();
    assert!(tampered.verify_successor(&chained).is_err());
}
