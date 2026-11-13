package join

import "crypto/ed25519"

const SchemaVersion = "mace-effect-join/v1"

type Role string

const (
	RoleAuthorizationBroker Role = "authorization-broker"
	RoleRuntimeObserver     Role = "runtime-observer"
	RoleToolGateway         Role = "tool-gateway"
	RoleEffectOrigin        Role = "effect-origin"
	RoleWitness             Role = "transparency-witness"
)

type ReceiptMeta struct {
	SchemaVersion string `json:"schema_version"`
	ReceiptID     string `json:"receipt_id"`
	Issuer        string `json:"issuer"`
	KeyID         string `json:"key_id"`
	IssuedAtMS    int64  `json:"issued_at_ms"`
}

type AuthorizationReceipt struct {
	Meta                 ReceiptMeta `json:"meta"`
	PermitID             string      `json:"permit_id"`
	ExecutionID          string      `json:"execution_id"`
	TaskID               string      `json:"task_id"`
	PrincipalID          string      `json:"principal_id"`
	Target               string      `json:"target"`
	ActionSHA256         string      `json:"action_sha256"`
	ArgumentsSHA256      string      `json:"arguments_sha256"`
	CapabilityID         string      `json:"capability_id"`
	Nonce                string      `json:"nonce"`
	PolicySHA256         string      `json:"policy_sha256"`
	RevocationGeneration uint64      `json:"revocation_generation"`
	ExpiresAtMS          int64       `json:"expires_at_ms"`
	MaximumDispatches    uint32      `json:"maximum_dispatches"`
}

type InvocationReceipt struct {
	Meta                  ReceiptMeta `json:"meta"`
	AuthorizationID       string      `json:"authorization_receipt_id"`
	PermitID              string      `json:"permit_id"`
	ExecutionID           string      `json:"execution_id"`
	TaskID                string      `json:"task_id"`
	PrincipalID           string      `json:"principal_id"`
	Target                string      `json:"target"`
	ActionSHA256          string      `json:"action_sha256"`
	ArgumentsSHA256       string      `json:"arguments_sha256"`
	CapabilityID          string      `json:"capability_id"`
	PolicySHA256          string      `json:"policy_sha256"`
	RevocationGeneration  uint64      `json:"revocation_generation"`
	ImplementationSHA256  string      `json:"implementation_sha256"`
	RuntimeWorkloadID     string      `json:"runtime_workload_id"`
	InvocationID          string      `json:"invocation_id"`
	ObserverSequence      uint64      `json:"observer_sequence"`
}

type DispatchStatus string

const (
	DispatchSent    DispatchStatus = "dispatched"
	DispatchUnknown DispatchStatus = "unknown"
	DispatchDenied  DispatchStatus = "rejected"
)

type DispatchReceipt struct {
	Meta                 ReceiptMeta    `json:"meta"`
	AuthorizationID      string         `json:"authorization_receipt_id"`
	InvocationID         string         `json:"invocation_receipt_id"`
	RuntimeInvocationID  string         `json:"runtime_invocation_id"`
	PermitID             string         `json:"permit_id"`
	ExecutionID          string         `json:"execution_id"`
	TaskID               string         `json:"task_id"`
	PrincipalID          string         `json:"principal_id"`
	Target               string         `json:"target"`
	ActionSHA256         string         `json:"action_sha256"`
	ArgumentsSHA256      string         `json:"arguments_sha256"`
	CapabilityID         string         `json:"capability_id"`
	PolicySHA256         string         `json:"policy_sha256"`
	RevocationGeneration uint64         `json:"revocation_generation"`
	IdempotencyKey       string         `json:"idempotency_key"`
	DispatchAttempt      uint32         `json:"dispatch_attempt"`
	Status               DispatchStatus `json:"status"`
}

type EffectStatus string

const (
	EffectCompleted EffectStatus = "completed"
	EffectFailed    EffectStatus = "failed"
	EffectPartial   EffectStatus = "partial"
	EffectUnknown   EffectStatus = "unknown"
)

type EffectReceipt struct {
	Meta            ReceiptMeta  `json:"meta"`
	DispatchID      string       `json:"dispatch_receipt_id"`
	IdempotencyKey  string       `json:"idempotency_key"`
	ExecutionID     string       `json:"execution_id"`
	TaskID          string       `json:"task_id"`
	PrincipalID     string       `json:"principal_id"`
	Target          string       `json:"target"`
	ActionSHA256    string       `json:"action_sha256"`
	ArgumentsSHA256 string       `json:"arguments_sha256"`
	EffectID        string       `json:"effect_id"`
	Status          EffectStatus `json:"status"`
}

type WitnessReceipt struct {
	Meta           ReceiptMeta `json:"meta"`
	LogID          string      `json:"log_id"`
	BundleSHA256   string      `json:"bundle_sha256"`
	Sequence       uint64      `json:"sequence"`
	CheckpointAtMS int64       `json:"checkpoint_at_ms"`
}

type Signed[T any] struct {
	Payload   T      `json:"payload"`
	Signature string `json:"signature"`
}

type Bundle struct {
	Authorization Signed[AuthorizationReceipt] `json:"authorization"`
	Invocation    Signed[InvocationReceipt]    `json:"invocation"`
	Dispatch      Signed[DispatchReceipt]      `json:"dispatch"`
	Effect        *Signed[EffectReceipt]       `json:"effect,omitempty"`
	Witness       Signed[WitnessReceipt]       `json:"witness"`
}

type TrustedKey struct {
	KeyID              string            `json:"key_id"`
	Issuer             string            `json:"issuer"`
	Role               Role              `json:"role"`
	IndependenceDomain string            `json:"independence_domain"`
	PublicKey          ed25519.PublicKey `json:"public_key"`
}

type TrustStore struct {
	Keys []TrustedKey `json:"keys"`
}

type IntentExpectation struct {
	ExecutionID     string `json:"execution_id"`
	TaskID          string `json:"task_id"`
	PrincipalID     string `json:"principal_id"`
	Target          string `json:"target"`
	ActionSHA256    string `json:"action_sha256"`
	ArgumentsSHA256 string `json:"arguments_sha256"`
	CapabilityID    string `json:"capability_id"`
	PolicySHA256    string `json:"policy_sha256"`
}

type PolicyState struct {
	NowMS                       int64  `json:"now_ms"`
	MaximumClockSkewMS          int64  `json:"maximum_clock_skew_ms"`
	MaximumPermitLifetimeMS     int64  `json:"maximum_permit_lifetime_ms"`
	MaximumWitnessAgeMS         int64  `json:"maximum_witness_age_ms"`
	ExpectedWitnessLogID        string `json:"expected_witness_log_id"`
	MinimumWitnessSequence      uint64 `json:"minimum_witness_sequence"`
	MinimumRevocationGeneration uint64 `json:"minimum_revocation_generation"`
}

type Classification string

const (
	Completed                    Classification = "completed"
	CompletedAfterReconciliation Classification = "completed_after_reconciliation"
	DispatchRejected             Classification = "dispatch_rejected"
	EffectFailedClass            Classification = "effect_failed"
	EffectPartialClass           Classification = "effect_partial"
	OutcomeUnknown               Classification = "outcome_unknown"
	EvidenceIncomplete           Classification = "evidence_incomplete"
)

type Result struct {
	Classification  Classification `json:"classification"`
	BundleSHA256    string         `json:"bundle_sha256"`
	AuthorizationID string         `json:"authorization_receipt_id"`
	InvocationID    string         `json:"invocation_receipt_id"`
	DispatchID      string         `json:"dispatch_receipt_id"`
	EffectID        string         `json:"effect_id,omitempty"`
	WitnessLogID    string         `json:"witness_log_id"`
	WitnessSequence uint64         `json:"witness_sequence"`
	Reconciled      bool           `json:"reconciled"`
}
