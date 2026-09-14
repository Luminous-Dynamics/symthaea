use symthaea_assurance_policy_lineage_anchor::PolicyLineageAnchor;
use symthaea_assurance_tpm2_tools_adapter::{
    CheckpointBindingContext, ToolExecution, Tpm2AdapterError, Tpm2ReadHierarchy,
    Tpm2ToolsAdapterPolicy, Tpm2ToolsExecutor, bind_observation_to_checkpoint,
    read_tpm2_nv_counter,
};
use symthaea_assurance_trust_store::{TrustStoreBackendKind, TrustStoreProfile};

const PUBLIC_TOOL: &[u8] = b"fake-tpm2-nvreadpublic-binary";
const READ_TOOL: &[u8] = b"fake-tpm2-nvread-binary";
const NV_NAME: &str = "000bdeadbeef";

fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

#[derive(Debug, Clone)]
struct FakeExecutor {
    public_stdout: Vec<u8>,
    read_stdout: Vec<u8>,
    public_stderr: Vec<u8>,
    read_stderr: Vec<u8>,
    public_exit: Option<i32>,
    read_exit: Option<i32>,
}

impl Default for FakeExecutor {
    fn default() -> Self {
        Self {
            public_stdout: public_yaml(NV_NAME, "ownerread|ownerwrite|nt=counter", 8)
                .into_bytes(),
            read_stdout: 42_u64.to_be_bytes().to_vec(),
            public_stderr: vec![],
            read_stderr: vec![],
            public_exit: Some(0),
            read_exit: Some(0),
        }
    }
}

impl Tpm2ToolsExecutor for FakeExecutor {
    fn executable_blake3(&self, executable: &str) -> Result<String, Tpm2AdapterError> {
        match executable {
            "/nix/store/public/bin/tpm2_nvreadpublic" => Ok(digest(PUBLIC_TOOL)),
            "/nix/store/read/bin/tpm2_nvread" => Ok(digest(READ_TOOL)),
            _ => Err(Tpm2AdapterError::Io(executable.into())),
        }
    }

    fn execute(
        &self,
        executable: &str,
        args: &[String],
    ) -> Result<ToolExecution, Tpm2AdapterError> {
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "-T" && pair[1] == "device:/dev/tpmrm0")
        );
        if executable.ends_with("tpm2_nvreadpublic") {
            Ok(ToolExecution {
                exit_code: self.public_exit,
                stdout: self.public_stdout.clone(),
                stderr: self.public_stderr.clone(),
            })
        } else {
            assert!(
                args.windows(2)
                    .any(|pair| pair[0] == "-s" && pair[1] == "8")
            );
            assert!(
                args.windows(2)
                    .any(|pair| pair[0] == "-C" && pair[1] == "o")
            );
            Ok(ToolExecution {
                exit_code: self.read_exit,
                stdout: self.read_stdout.clone(),
                stderr: self.read_stderr.clone(),
            })
        }
    }
}

fn public_yaml(name: &str, attributes: &str, size: usize) -> String {
    format!(
        "0x1500016:\n  name: {name}\n  hash algorithm:\n    friendly: sha256\n    value: 0xb\n  attributes:\n    friendly: {attributes}\n    value: 0x20040004\n  size: {size}\n  authorization policy:\n"
    )
}

fn policy() -> Tpm2ToolsAdapterPolicy {
    Tpm2ToolsAdapterPolicy {
        schema_version: "1".into(),
        adapter_id: "tpm2-tools:node-1".into(),
        logical_store_id: "policy-root".into(),
        trust_store_ref: "trust-store:tpm2:node-1".into(),
        counter_epoch: "epoch:tpm2:1".into(),
        nv_index: 0x0150_0016,
        expected_nv_name: NV_NAME.into(),
        tcti: "device:/dev/tpmrm0".into(),
        read_hierarchy: Tpm2ReadHierarchy::Owner,
        nvreadpublic_path: "/nix/store/public/bin/tpm2_nvreadpublic".into(),
        nvread_path: "/nix/store/read/bin/tpm2_nvread".into(),
        expected_nvreadpublic_blake3: digest(PUBLIC_TOOL),
        expected_nvread_blake3: digest(READ_TOOL),
        minimum_counter_value: 40,
        evidence_refs: vec!["review:tpm2-adapter-policy".into()],
    }
}

fn profile() -> TrustStoreProfile {
    TrustStoreProfile {
        schema_version: "1".into(),
        store_id: "policy-root".into(),
        trust_store_ref: "trust-store:tpm2:node-1".into(),
        backend_kind: TrustStoreBackendKind::HardwareMonotonicCounter,
        hardware_instance_ref: Some("tpm2:device:/dev/tpmrm0".into()),
        monotonic_counter_ref: "tpm2:nv:0x1500016".into(),
        initial_counter_epoch: "epoch:tpm2:1".into(),
        minimum_independent_recovery_approvals: 2,
        evidence_refs: vec!["review:trust-store-profile".into()],
    }
}

fn anchor() -> PolicyLineageAnchor {
    PolicyLineageAnchor {
        schema_version: "1".into(),
        anchor_id: "anchor:1".into(),
        anchor_revision: 1,
        manifest_id: "manifest:node-1".into(),
        deployment_id: "node-1".into(),
        tip_revision: 3,
        tip_manifest_digest: "blake3:manifest-tip".into(),
        predecessor_anchor_digest: None,
        recorded_at_ms: 9_000,
        trust_store_ref: "trust-store:tpm2:node-1".into(),
        independent_verification_ref: "verification:anchor".into(),
        evidence_refs: vec!["audit:anchor".into()],
    }
}

#[test]
fn exact_counter_read_produces_observation() {
    let observation = read_tpm2_nv_counter(&policy(), 10_000, &FakeExecutor::default()).unwrap();
    assert_eq!(observation.counter_value, 42);
    assert_eq!(observation.public_evidence.data_size, 8);
    assert_eq!(observation.public_evidence.nv_name, NV_NAME);
    assert!(observation.observation_digest().starts_with("blake3:"));
    assert!(!observation.grants_physical_authority());
}

#[test]
fn public_name_substitution_is_rejected() {
    let fake = FakeExecutor {
        public_stdout: public_yaml("000bfeedface", "ownerread|nt=counter", 8).into_bytes(),
        ..FakeExecutor::default()
    };
    assert_eq!(
        read_tpm2_nv_counter(&policy(), 10_000, &fake),
        Err(Tpm2AdapterError::NvNameMismatch)
    );
}

#[test]
fn non_counter_index_is_rejected() {
    let fake = FakeExecutor {
        public_stdout: public_yaml(NV_NAME, "ownerread|ownerwrite|ordinary", 8).into_bytes(),
        ..FakeExecutor::default()
    };
    assert_eq!(
        read_tpm2_nv_counter(&policy(), 10_000, &fake),
        Err(Tpm2AdapterError::NvIndexIsNotCounter)
    );
}

#[test]
fn wrong_size_and_short_read_are_rejected() {
    let wrong_size = FakeExecutor {
        public_stdout: public_yaml(NV_NAME, "ownerread|nt=counter", 32).into_bytes(),
        ..FakeExecutor::default()
    };
    assert_eq!(
        read_tpm2_nv_counter(&policy(), 10_000, &wrong_size),
        Err(Tpm2AdapterError::UnexpectedDataSize(32))
    );

    let short = FakeExecutor {
        read_stdout: vec![0; 7],
        ..FakeExecutor::default()
    };
    assert_eq!(
        read_tpm2_nv_counter(&policy(), 10_000, &short),
        Err(Tpm2AdapterError::UnexpectedCounterReadLength(7))
    );
}

#[test]
fn rollback_floor_is_enforced() {
    let fake = FakeExecutor {
        read_stdout: 39_u64.to_be_bytes().to_vec(),
        ..FakeExecutor::default()
    };
    assert_eq!(
        read_tpm2_nv_counter(&policy(), 10_000, &fake),
        Err(Tpm2AdapterError::CounterBelowProvisionedFloor)
    );
}

#[test]
fn stderr_is_not_silently_ignored() {
    let fake = FakeExecutor {
        read_stderr: b"warning".to_vec(),
        ..FakeExecutor::default()
    };
    assert_eq!(
        read_tpm2_nv_counter(&policy(), 10_000, &fake),
        Err(Tpm2AdapterError::ToolEmittedStderr)
    );
}

#[test]
fn executable_drift_is_rejected() {
    let mut policy = policy();
    policy.expected_nvread_blake3 = format!("blake3:{}", "0".repeat(64));
    assert_eq!(
        read_tpm2_nv_counter(&policy, 10_000, &FakeExecutor::default()),
        Err(Tpm2AdapterError::ExecutableDigestMismatch)
    );
}

#[test]
fn simulator_or_custom_tcti_is_not_admitted_as_hardware() {
    let mut policy = policy();
    policy.tcti = "mssim:host=localhost,port=2321".into();
    assert!(!policy.validate());
}

#[test]
fn observation_binds_to_existing_checkpoint_contract() {
    let policy = policy();
    let observation = read_tpm2_nv_counter(&policy, 10_000, &FakeExecutor::default()).unwrap();
    let context = CheckpointBindingContext {
        checkpoint_id: "checkpoint:tpm2:1".into(),
        store_revision: 1,
        predecessor_checkpoint_digest: None,
        attestation_ref: "attestation:tpm2:1".into(),
        independent_verification_ref: "verification:tpm2-adapter:1".into(),
        evidence_refs: vec!["audit:tpm2-checkpoint-binding".into()],
    };
    let checkpoint =
        bind_observation_to_checkpoint(&observation, &policy, &profile(), &anchor(), &context)
            .unwrap();
    assert_eq!(checkpoint.counter_value, 42);
    assert_eq!(checkpoint.anchor_digest, anchor().anchor_digest());
    assert!(
        checkpoint
            .evidence_refs
            .iter()
            .any(|value| value.starts_with("tpm2-observation:blake3:"))
    );
}

#[test]
fn different_anchor_store_cannot_be_bound() {
    let policy = policy();
    let observation = read_tpm2_nv_counter(&policy, 10_000, &FakeExecutor::default()).unwrap();
    let mut wrong_anchor = anchor();
    wrong_anchor.trust_store_ref = "trust-store:other".into();
    let context = CheckpointBindingContext {
        checkpoint_id: "checkpoint:tpm2:1".into(),
        store_revision: 1,
        predecessor_checkpoint_digest: None,
        attestation_ref: "attestation:tpm2:1".into(),
        independent_verification_ref: "verification:tpm2-adapter:1".into(),
        evidence_refs: vec!["audit:tpm2-checkpoint-binding".into()],
    };
    assert_eq!(
        bind_observation_to_checkpoint(
            &observation,
            &policy,
            &profile(),
            &wrong_anchor,
            &context,
        ),
        Err(Tpm2AdapterError::AnchorMismatch)
    );
}
