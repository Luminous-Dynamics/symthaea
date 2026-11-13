// Package join verifies a signed, typed causal join from authorization through
// observed invocation and gateway dispatch to an independent effect receipt.
// It does not issue permits, instrument a runtime, or establish source truth.
package join

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

var (
	ErrTrustStore          = errors.New("invalid trust store")
	ErrSignature           = errors.New("receipt signature verification failed")
	ErrWrongRole           = errors.New("receipt issuer has wrong role")
	ErrIndependence        = errors.New("evidence roles are not administratively independent")
	ErrInvalidReceipt      = errors.New("invalid receipt")
	ErrJoinMismatch        = errors.New("causal receipt join mismatch")
	ErrPermitReuse         = errors.New("one-shot permit reuse detected")
	ErrStalePermit         = errors.New("authorization permit expired or stale")
	ErrStaleWitness        = errors.New("witness checkpoint is stale or rolled back")
	ErrWitnessMismatch     = errors.New("witness does not commit to this evidence bundle")
	ErrContradictoryEffect = errors.New("effect receipt contradicts gateway rejection")
)

func canonicalJSON(v any) ([]byte, error) {
	// v0.1 uses fixed Go struct field order, no maps, UTF-8 JSON and compact
	// encoding. Cross-language production profiles should freeze RFC 8785 JCS
	// or an equivalent independently specified canonical encoding.
	return json.Marshal(v)
}

func Sign[T any](payload T, privateKey ed25519.PrivateKey) (Signed[T], error) {
	if len(privateKey) != ed25519.PrivateKeySize {
		return Signed[T]{}, fmt.Errorf("%w: invalid Ed25519 private key", ErrInvalidReceipt)
	}
	data, err := canonicalJSON(payload)
	if err != nil {
		return Signed[T]{}, fmt.Errorf("canonicalize payload: %w", err)
	}
	sig := ed25519.Sign(privateKey, data)
	return Signed[T]{Payload: payload, Signature: base64.RawStdEncoding.EncodeToString(sig)}, nil
}

func BundleDigest(bundle Bundle) (string, error) {
	// The witness receipt is intentionally excluded to avoid a self-reference.
	committed := struct {
		SchemaVersion string                       `json:"schema_version"`
		Authorization Signed[AuthorizationReceipt] `json:"authorization"`
		Invocation    Signed[InvocationReceipt]    `json:"invocation"`
		Dispatch      Signed[DispatchReceipt]      `json:"dispatch"`
		Effect        *Signed[EffectReceipt]       `json:"effect,omitempty"`
	}{SchemaVersion: SchemaVersion, Authorization: bundle.Authorization, Invocation: bundle.Invocation, Dispatch: bundle.Dispatch, Effect: bundle.Effect}
	data, err := canonicalJSON(committed)
	if err != nil {
		return "", fmt.Errorf("canonicalize evidence bundle: %w", err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func Verify(bundle Bundle, trust TrustStore, state PolicyState, expected IntentExpectation) (Result, error) {
	keys, err := validateTrustStore(trust)
	if err != nil {
		return Result{}, err
	}
	if state.NowMS <= 0 || state.MaximumClockSkewMS < 0 || state.MaximumPermitLifetimeMS <= 0 || state.MaximumWitnessAgeMS <= 0 || strings.TrimSpace(state.ExpectedWitnessLogID) == "" {
		return Result{}, fmt.Errorf("%w: invalid verifier policy state", ErrInvalidReceipt)
	}

	a := bundle.Authorization.Payload
	i := bundle.Invocation.Payload
	d := bundle.Dispatch.Payload
	w := bundle.Witness.Payload

	if err := verifySigned(bundle.Authorization, RoleAuthorizationBroker, keys); err != nil {
		return Result{}, fmt.Errorf("authorization: %w", err)
	}
	if err := verifySigned(bundle.Invocation, RoleRuntimeObserver, keys); err != nil {
		return Result{}, fmt.Errorf("invocation: %w", err)
	}
	if err := verifySigned(bundle.Dispatch, RoleToolGateway, keys); err != nil {
		return Result{}, fmt.Errorf("dispatch: %w", err)
	}
	if bundle.Effect != nil {
		if err := verifySigned(*bundle.Effect, RoleEffectOrigin, keys); err != nil {
			return Result{}, fmt.Errorf("effect: %w", err)
		}
	}
	if err := verifySigned(bundle.Witness, RoleWitness, keys); err != nil {
		return Result{}, fmt.Errorf("witness: %w", err)
	}

	if err := validateMeta(a.Meta); err != nil {
		return Result{}, fmt.Errorf("authorization metadata: %w", err)
	}
	if err := validateMeta(i.Meta); err != nil {
		return Result{}, fmt.Errorf("invocation metadata: %w", err)
	}
	if err := validateMeta(d.Meta); err != nil {
		return Result{}, fmt.Errorf("dispatch metadata: %w", err)
	}
	if bundle.Effect != nil {
		if err := validateMeta(bundle.Effect.Payload.Meta); err != nil {
			return Result{}, fmt.Errorf("effect metadata: %w", err)
		}
	}
	if err := validateMeta(w.Meta); err != nil {
		return Result{}, fmt.Errorf("witness metadata: %w", err)
	}

	if !validID(expected.ExecutionID) || !validID(expected.TaskID) || !validID(expected.PrincipalID) || !validID(expected.Target) || !validSHA256(expected.ActionSHA256) || !validSHA256(expected.ArgumentsSHA256) || !validID(expected.CapabilityID) || !validSHA256(expected.PolicySHA256) {
		return Result{}, fmt.Errorf("%w: expected task-intent pin is incomplete", ErrInvalidReceipt)
	}
	if a.ExecutionID != expected.ExecutionID || a.TaskID != expected.TaskID || a.PrincipalID != expected.PrincipalID || a.Target != expected.Target || a.ActionSHA256 != expected.ActionSHA256 || a.ArgumentsSHA256 != expected.ArgumentsSHA256 || a.CapabilityID != expected.CapabilityID || a.PolicySHA256 != expected.PolicySHA256 {
		return Result{}, fmt.Errorf("%w: authorization does not match independently pinned task intent", ErrJoinMismatch)
	}
	if a.ExpiresAtMS <= a.Meta.IssuedAtMS || a.ExpiresAtMS <= state.NowMS || a.ExpiresAtMS-a.Meta.IssuedAtMS > state.MaximumPermitLifetimeMS || a.Meta.IssuedAtMS > state.NowMS+state.MaximumClockSkewMS {
		return Result{}, ErrStalePermit
	}
	if a.RevocationGeneration < state.MinimumRevocationGeneration {
		return Result{}, fmt.Errorf("%w: revocation generation %d is below required %d", ErrStalePermit, a.RevocationGeneration, state.MinimumRevocationGeneration)
	}
	if a.MaximumDispatches != 1 {
		return Result{}, fmt.Errorf("%w: only one-shot permits are accepted", ErrPermitReuse)
	}
	if d.DispatchAttempt != 1 {
		return Result{}, fmt.Errorf("%w: a permit cannot authorize retry attempt %d", ErrPermitReuse, d.DispatchAttempt)
	}

	if !validID(a.PermitID) || !validID(a.ExecutionID) || !validID(a.TaskID) || !validID(a.PrincipalID) || !validID(a.Target) || !validID(a.CapabilityID) || !validID(a.Nonce) || !validSHA256(a.ActionSHA256) || !validSHA256(a.ArgumentsSHA256) || !validSHA256(a.PolicySHA256) {
		return Result{}, fmt.Errorf("%w: authorization contains malformed identity or digest", ErrInvalidReceipt)
	}
	if a.Meta.ReceiptID == i.Meta.ReceiptID || a.Meta.ReceiptID == d.Meta.ReceiptID || i.Meta.ReceiptID == d.Meta.ReceiptID {
		return Result{}, fmt.Errorf("%w: duplicate receipt IDs", ErrJoinMismatch)
	}
	if i.Meta.IssuedAtMS+state.MaximumClockSkewMS < a.Meta.IssuedAtMS || i.Meta.IssuedAtMS > a.ExpiresAtMS+state.MaximumClockSkewMS {
		return Result{}, fmt.Errorf("%w: invocation is outside the authorization time window", ErrJoinMismatch)
	}
	if i.AuthorizationID != a.Meta.ReceiptID || i.PermitID != a.PermitID || i.ExecutionID != a.ExecutionID || i.TaskID != a.TaskID || i.PrincipalID != a.PrincipalID || i.Target != a.Target || i.ActionSHA256 != a.ActionSHA256 || i.ArgumentsSHA256 != a.ArgumentsSHA256 || i.CapabilityID != a.CapabilityID || i.PolicySHA256 != a.PolicySHA256 || i.RevocationGeneration != a.RevocationGeneration {
		return Result{}, fmt.Errorf("%w: observed invocation does not bind exact authorization", ErrJoinMismatch)
	}
	if !validSHA256(i.ImplementationSHA256) || strings.TrimSpace(i.RuntimeWorkloadID) == "" || !validID(i.InvocationID) || i.ObserverSequence == 0 {
		return Result{}, fmt.Errorf("%w: missing implementation/runtime/observer identity", ErrInvalidReceipt)
	}
	if d.Meta.IssuedAtMS+state.MaximumClockSkewMS < i.Meta.IssuedAtMS || d.Meta.IssuedAtMS > a.ExpiresAtMS+state.MaximumClockSkewMS {
		return Result{}, fmt.Errorf("%w: dispatch is outside the invocation/authorization time window", ErrJoinMismatch)
	}
	if d.AuthorizationID != a.Meta.ReceiptID || d.InvocationID != i.Meta.ReceiptID || d.RuntimeInvocationID != i.InvocationID || d.PermitID != a.PermitID || d.ExecutionID != a.ExecutionID || d.TaskID != a.TaskID || d.PrincipalID != a.PrincipalID || d.Target != a.Target || d.ActionSHA256 != a.ActionSHA256 || d.ArgumentsSHA256 != a.ArgumentsSHA256 || d.CapabilityID != a.CapabilityID || d.PolicySHA256 != a.PolicySHA256 || d.RevocationGeneration != a.RevocationGeneration {
		return Result{}, fmt.Errorf("%w: gateway dispatch does not bind exact invocation and authorization", ErrJoinMismatch)
	}
	if !validID(d.IdempotencyKey) {
		return Result{}, fmt.Errorf("%w: missing idempotency key", ErrInvalidReceipt)
	}
	if d.Status != DispatchSent && d.Status != DispatchUnknown && d.Status != DispatchDenied {
		return Result{}, fmt.Errorf("%w: unknown dispatch status %q", ErrInvalidReceipt, d.Status)
	}

	var effectID string
	if bundle.Effect != nil {
		e := bundle.Effect.Payload
		effectID = e.EffectID
		if e.Meta.ReceiptID == a.Meta.ReceiptID || e.Meta.ReceiptID == i.Meta.ReceiptID || e.Meta.ReceiptID == d.Meta.ReceiptID {
			return Result{}, fmt.Errorf("%w: duplicate receipt IDs", ErrJoinMismatch)
		}
		if e.Meta.IssuedAtMS+state.MaximumClockSkewMS < d.Meta.IssuedAtMS {
			return Result{}, fmt.Errorf("%w: effect receipt predates dispatch", ErrJoinMismatch)
		}
		if e.DispatchID != d.Meta.ReceiptID || e.IdempotencyKey != d.IdempotencyKey || e.ExecutionID != a.ExecutionID || e.TaskID != a.TaskID || e.PrincipalID != a.PrincipalID || e.Target != a.Target || e.ActionSHA256 != a.ActionSHA256 || e.ArgumentsSHA256 != a.ArgumentsSHA256 {
			return Result{}, fmt.Errorf("%w: effect receipt does not bind exact dispatch and payload", ErrJoinMismatch)
		}
		if !validID(e.EffectID) {
			return Result{}, fmt.Errorf("%w: empty effect ID", ErrInvalidReceipt)
		}
		if e.Status != EffectCompleted && e.Status != EffectFailed && e.Status != EffectPartial && e.Status != EffectUnknown {
			return Result{}, fmt.Errorf("%w: unknown effect status %q", ErrInvalidReceipt, e.Status)
		}
	}
	if d.Status == DispatchDenied && bundle.Effect != nil {
		return Result{}, ErrContradictoryEffect
	}

	digest, err := BundleDigest(bundle)
	if err != nil {
		return Result{}, err
	}
	latestEvidenceAt := d.Meta.IssuedAtMS
	if bundle.Effect != nil && bundle.Effect.Payload.Meta.IssuedAtMS > latestEvidenceAt {
		latestEvidenceAt = bundle.Effect.Payload.Meta.IssuedAtMS
	}
	if w.CheckpointAtMS+state.MaximumClockSkewMS < latestEvidenceAt {
		return Result{}, fmt.Errorf("%w: witness checkpoint predates committed evidence", ErrStaleWitness)
	}
	if w.LogID != state.ExpectedWitnessLogID || w.BundleSHA256 != digest || !validSHA256(w.BundleSHA256) {
		return Result{}, ErrWitnessMismatch
	}
	if w.Sequence <= state.MinimumWitnessSequence || w.CheckpointAtMS <= 0 || w.CheckpointAtMS > state.NowMS+state.MaximumClockSkewMS || state.NowMS-w.CheckpointAtMS > state.MaximumWitnessAgeMS {
		return Result{}, ErrStaleWitness
	}

	classification := OutcomeUnknown
	reconciled := false
	switch {
	case d.Status == DispatchDenied:
		classification = DispatchRejected
	case bundle.Effect == nil:
		if d.Status == DispatchSent {
			classification = EvidenceIncomplete
		} else {
			classification = OutcomeUnknown
		}
	case bundle.Effect.Payload.Status == EffectCompleted:
		if d.Status == DispatchUnknown {
			classification, reconciled = CompletedAfterReconciliation, true
		} else {
			classification = Completed
		}
	case bundle.Effect.Payload.Status == EffectFailed:
		classification, reconciled = EffectFailedClass, d.Status == DispatchUnknown
	case bundle.Effect.Payload.Status == EffectPartial:
		classification, reconciled = EffectPartialClass, d.Status == DispatchUnknown
	case bundle.Effect.Payload.Status == EffectUnknown:
		classification, reconciled = OutcomeUnknown, d.Status == DispatchUnknown
	}
	return Result{
		Classification: classification, BundleSHA256: digest,
		AuthorizationID: a.Meta.ReceiptID, InvocationID: i.Meta.ReceiptID,
		DispatchID: d.Meta.ReceiptID, EffectID: effectID,
		WitnessLogID: w.LogID, WitnessSequence: w.Sequence, Reconciled: reconciled,
	}, nil
}

func validateTrustStore(store TrustStore) (map[string]TrustedKey, error) {
	if len(store.Keys) != 5 {
		return nil, fmt.Errorf("%w: exactly five role keys are required", ErrTrustStore)
	}
	byID := make(map[string]TrustedKey, len(store.Keys))
	roles, issuers, domains := map[Role]bool{}, map[string]bool{}, map[string]bool{}
	for _, key := range store.Keys {
		if !validID(key.KeyID) || !validID(key.Issuer) || !validID(key.IndependenceDomain) || len(key.PublicKey) != ed25519.PublicKeySize {
			return nil, fmt.Errorf("%w: malformed key entry", ErrTrustStore)
		}
		if _, exists := byID[key.KeyID]; exists {
			return nil, fmt.Errorf("%w: duplicate key ID %s", ErrTrustStore, key.KeyID)
		}
		if roles[key.Role] || issuers[key.Issuer] || domains[key.IndependenceDomain] {
			return nil, ErrIndependence
		}
		switch key.Role {
		case RoleAuthorizationBroker, RoleRuntimeObserver, RoleToolGateway, RoleEffectOrigin, RoleWitness:
		default:
			return nil, fmt.Errorf("%w: unknown role %q", ErrTrustStore, key.Role)
		}
		roles[key.Role], issuers[key.Issuer], domains[key.IndependenceDomain] = true, true, true
		byID[key.KeyID] = key
	}
	return byID, nil
}

func verifySigned[T any](signed Signed[T], expectedRole Role, keys map[string]TrustedKey) error {
	var meta ReceiptMeta
	data, err := canonicalJSON(signed.Payload)
	if err != nil {
		return fmt.Errorf("%w: canonicalization: %v", ErrInvalidReceipt, err)
	}
	// Receipt types all expose Meta; marshal/unmarshal is safe here because the
	// signed payload has already been decoded into a closed Go struct type.
	encoded, err := json.Marshal(signed.Payload)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidReceipt, err)
	}
	if err := json.Unmarshal(encoded, &struct {
		Meta *ReceiptMeta `json:"meta"`
	}{Meta: &meta}); err != nil {
		return fmt.Errorf("%w: metadata extraction", ErrInvalidReceipt)
	}
	trusted, ok := keys[meta.KeyID]
	if !ok || trusted.Issuer != meta.Issuer {
		return ErrSignature
	}
	if trusted.Role != expectedRole {
		return ErrWrongRole
	}
	signature, err := base64.RawStdEncoding.DecodeString(signed.Signature)
	if err != nil || len(signature) != ed25519.SignatureSize || !ed25519.Verify(trusted.PublicKey, data, signature) {
		return ErrSignature
	}
	return nil
}

func validateMeta(meta ReceiptMeta) error {
	if meta.SchemaVersion != SchemaVersion || !validID(meta.ReceiptID) || !validID(meta.Issuer) || !validID(meta.KeyID) || meta.IssuedAtMS <= 0 {
		return ErrInvalidReceipt
	}
	return nil
}

func validID(value string) bool {
	if len(value) == 0 || len(value) > 256 || strings.TrimSpace(value) != value {
		return false
	}
	for _, r := range value {
		if r < 0x21 || r > 0x7e {
			return false
		}
	}
	return true
}
func validSHA256(value string) bool {
	if len(value) != 64 || strings.ToLower(value) != value {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

// IndependenceDomains returns the trust-store role domains in deterministic
// order for audit output; it does not itself prove their administrative truth.
func IndependenceDomains(store TrustStore) []string {
	out := make([]string, 0, len(store.Keys))
	for _, key := range store.Keys {
		out = append(out, string(key.Role)+":"+key.IndependenceDomain)
	}
	sort.Strings(out)
	return out
}
