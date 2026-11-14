package join

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"testing"
)

type testKeys struct {
	trust TrustStore
	priv  map[Role]ed25519.PrivateKey
}

func digest(label string) string { h := sha256.Sum256([]byte(label)); return hex.EncodeToString(h[:]) }
func newTestKeys() testKeys {
	roles := []struct {
		role                Role
		issuer, key, domain string
	}{
		{RoleAuthorizationBroker, "auth-svc", "auth-key", "auth-domain"},
		{RoleRuntimeObserver, "observer-svc", "observer-key", "observer-domain"},
		{RoleToolGateway, "gateway-svc", "gateway-key", "gateway-domain"},
		{RoleEffectOrigin, "effect-ledger", "effect-key", "effect-domain"},
		{RoleWitness, "witness-svc", "witness-key", "witness-domain"},
	}
	out := testKeys{priv: map[Role]ed25519.PrivateKey{}}
	for idx, row := range roles {
		seed := make([]byte, ed25519.SeedSize)
		for i := range seed {
			seed[i] = byte((idx+1)*17 + i)
		}
		priv := ed25519.NewKeyFromSeed(seed)
		out.priv[row.role] = priv
		out.trust.Keys = append(out.trust.Keys, TrustedKey{KeyID: row.key, Issuer: row.issuer, Role: row.role, IndependenceDomain: row.domain, PublicKey: priv.Public().(ed25519.PublicKey)})
	}
	return out
}
func meta(id, issuer, key string, at int64) ReceiptMeta {
	return ReceiptMeta{SchemaVersion: SchemaVersion, ReceiptID: id, Issuer: issuer, KeyID: key, IssuedAtMS: at}
}

func validBundle(t interface {
	Helper()
	Fatal(...any)
}, k testKeys) Bundle {
	t.Helper()
	auth := AuthorizationReceipt{
		Meta: meta("auth-r1", "auth-svc", "auth-key", 1000), PermitID: "permit-1", ExecutionID: "exec-1", TaskID: "task-1", PrincipalID: "principal-1", Target: "ledger://account/42",
		ActionSHA256: digest("action"), ArgumentsSHA256: digest("args"), CapabilityID: "cap-1", Nonce: "nonce-1", PolicySHA256: digest("policy"), RevocationGeneration: 7, ExpiresAtMS: 3000, MaximumDispatches: 1,
	}
	as, err := Sign(auth, k.priv[RoleAuthorizationBroker])
	if err != nil { t.Fatal(err) }
	inv := InvocationReceipt{
		Meta: meta("obs-r1", "observer-svc", "observer-key", 1100), AuthorizationID: auth.Meta.ReceiptID, PermitID: auth.PermitID, ExecutionID: auth.ExecutionID, TaskID: auth.TaskID, PrincipalID: auth.PrincipalID,
		Target: auth.Target, ActionSHA256: auth.ActionSHA256, ArgumentsSHA256: auth.ArgumentsSHA256,
		CapabilityID: auth.CapabilityID, PolicySHA256: auth.PolicySHA256, RevocationGeneration: auth.RevocationGeneration,
		ImplementationSHA256: digest("actual-implementation"), RuntimeWorkloadID: "spiffe://example.org/agent/runtime-7", InvocationID: "invoke-1", ObserverSequence: 44,
	}
	is, err := Sign(inv, k.priv[RoleRuntimeObserver])
	if err != nil { t.Fatal(err) }
	dis := DispatchReceipt{
		Meta: meta("gateway-r1", "gateway-svc", "gateway-key", 1200), AuthorizationID: auth.Meta.ReceiptID, InvocationID: inv.Meta.ReceiptID, RuntimeInvocationID: inv.InvocationID, PermitID: auth.PermitID, ExecutionID: auth.ExecutionID, TaskID: auth.TaskID, PrincipalID: auth.PrincipalID,
		Target: auth.Target, ActionSHA256: auth.ActionSHA256, ArgumentsSHA256: auth.ArgumentsSHA256,
		CapabilityID: auth.CapabilityID, PolicySHA256: auth.PolicySHA256, RevocationGeneration: auth.RevocationGeneration,
		IdempotencyKey: "idem-1", DispatchAttempt: 1, Status: DispatchSent,
	}
	ds, err := Sign(dis, k.priv[RoleToolGateway])
	if err != nil { t.Fatal(err) }
	eff := EffectReceipt{
		Meta: meta("effect-r1", "effect-ledger", "effect-key", 1300), DispatchID: dis.Meta.ReceiptID, IdempotencyKey: dis.IdempotencyKey, ExecutionID: auth.ExecutionID, TaskID: auth.TaskID, PrincipalID: auth.PrincipalID,
		Target: auth.Target, ActionSHA256: auth.ActionSHA256, ArgumentsSHA256: auth.ArgumentsSHA256, EffectID: "effect-1", Status: EffectCompleted,
	}
	es, err := Sign(eff, k.priv[RoleEffectOrigin])
	if err != nil { t.Fatal(err) }
	b := Bundle{Authorization: as, Invocation: is, Dispatch: ds, Effect: &es}
	bundleDigest, err := BundleDigest(b)
	if err != nil { t.Fatal(err) }
	w := WitnessReceipt{Meta: meta("witness-r1", "witness-svc", "witness-key", 1400), LogID: "transparency-log-1", BundleSHA256: bundleDigest, Sequence: 12, CheckpointAtMS: 1450}
	ws, err := Sign(w, k.priv[RoleWitness])
	if err != nil { t.Fatal(err) }
	b.Witness = ws
	return b
}
func testExpectation(b Bundle) IntentExpectation {
	a := b.Authorization.Payload
	return IntentExpectation{ExecutionID: a.ExecutionID, TaskID: a.TaskID, PrincipalID: a.PrincipalID, Target: a.Target, ActionSHA256: a.ActionSHA256, ArgumentsSHA256: a.ArgumentsSHA256, CapabilityID: a.CapabilityID, PolicySHA256: a.PolicySHA256}
}
func testState() PolicyState {
	return PolicyState{NowMS: 1500, MaximumClockSkewMS: 100, MaximumPermitLifetimeMS: 5000, MaximumWitnessAgeMS: 1000, ExpectedWitnessLogID: "transparency-log-1", MinimumWitnessSequence: 11, MinimumRevocationGeneration: 7}
}
func resignBundle(t *testing.T, b *Bundle, k testKeys) {
	t.Helper()
	var err error
	b.Authorization, err = Sign(b.Authorization.Payload, k.priv[RoleAuthorizationBroker]); if err != nil { t.Fatal(err) }
	b.Invocation, err = Sign(b.Invocation.Payload, k.priv[RoleRuntimeObserver]); if err != nil { t.Fatal(err) }
	b.Dispatch, err = Sign(b.Dispatch.Payload, k.priv[RoleToolGateway]); if err != nil { t.Fatal(err) }
	if b.Effect != nil { *b.Effect, err = Sign(b.Effect.Payload, k.priv[RoleEffectOrigin]); if err != nil { t.Fatal(err) } }
	d, err := BundleDigest(*b); if err != nil { t.Fatal(err) }
	w := b.Witness.Payload; w.BundleSHA256 = d
	b.Witness, err = Sign(w, k.priv[RoleWitness]); if err != nil { t.Fatal(err) }
}

func TestValidChainCompletes(t *testing.T) {
	k:=newTestKeys(); b:=validBundle(t,k)
	r,err:=Verify(b,k.trust,testState(),testExpectation(b)); if err!=nil {t.Fatal(err)}
	if r.Classification!=Completed || r.Reconciled || r.EffectID!="effect-1" || r.WitnessSequence!=12 {t.Fatalf("unexpected result: %#v",r)}
}
func TestUnknownDispatchWithEffectReceiptIsReconciled(t *testing.T) {
	k:=newTestKeys(); b:=validBundle(t,k); b.Dispatch.Payload.Status=DispatchUnknown; resignBundle(t,&b,k)
	r,err:=Verify(b,k.trust,testState(),testExpectation(b)); if err!=nil {t.Fatal(err)}
	if r.Classification!=CompletedAfterReconciliation || !r.Reconciled {t.Fatalf("unexpected result: %#v",r)}
}
func TestMissingEffectNeverCompletes(t *testing.T) {
	k:=newTestKeys(); b:=validBundle(t,k); b.Effect=nil; resignBundle(t,&b,k)
	r,err:=Verify(b,k.trust,testState(),testExpectation(b)); if err!=nil {t.Fatal(err)}
	if r.Classification!=EvidenceIncomplete || r.EffectID!="" {t.Fatalf("missing terminal evidence became success: %#v",r)}
}
func TestEffectFailureAndPartialRemainNonSuccess(t *testing.T) {
	for _,status:=range []EffectStatus{EffectFailed,EffectPartial,EffectUnknown}{
		t.Run(string(status),func(t *testing.T){k:=newTestKeys();b:=validBundle(t,k);b.Effect.Payload.Status=status;resignBundle(t,&b,k)
			r,err:=Verify(b,k.trust,testState(),testExpectation(b));if err!=nil{t.Fatal(err)}
			want:=map[EffectStatus]Classification{EffectFailed:EffectFailedClass,EffectPartial:EffectPartialClass,EffectUnknown:OutcomeUnknown}[status]
			if r.Classification!=want{t.Fatalf("got %s want %s",r.Classification,want)}
		})
	}
}
func TestGatewayRejectionCannotHaveEffect(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k);b.Dispatch.Payload.Status=DispatchDenied;resignBundle(t,&b,k)
	_,err:=Verify(b,k.trust,testState(),testExpectation(b));if !errors.Is(err,ErrContradictoryEffect){t.Fatalf("got %v",err)}
}
func TestGatewayRejectionWithoutEffectIsRejectedDispatch(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k);b.Dispatch.Payload.Status=DispatchDenied;b.Effect=nil;resignBundle(t,&b,k)
	r,err:=Verify(b,k.trust,testState(),testExpectation(b));if err!=nil{t.Fatal(err)}
	if r.Classification!=DispatchRejected{t.Fatalf("got %s",r.Classification)}
}
func TestArgumentTargetAndParentSubstitutionReject(t *testing.T) {
	mutations:=map[string]func(*Bundle){
		"argument-digest":func(b *Bundle){b.Dispatch.Payload.ArgumentsSHA256=digest("mutated")},
		"target":func(b *Bundle){b.Dispatch.Payload.Target="ledger://account/99"},
		"parent-invocation":func(b *Bundle){b.Dispatch.Payload.InvocationID="other-observer-receipt"},
		"permit-id":func(b *Bundle){b.Invocation.Payload.PermitID="other-permit"},
		"runtime-implementation":func(b *Bundle){b.Invocation.Payload.ImplementationSHA256="bad"},
		"capability":func(b *Bundle){b.Dispatch.Payload.CapabilityID="other-capability"},
		"policy-generation":func(b *Bundle){b.Invocation.Payload.RevocationGeneration++},
		"effect-dispatch":func(b *Bundle){b.Effect.Payload.DispatchID="other-dispatch"},
	}
	for name,mutate:=range mutations{t.Run(name,func(t *testing.T){k:=newTestKeys();b:=validBundle(t,k);mutate(&b);resignBundle(t,&b,k)
		_,err:=Verify(b,k.trust,testState(),testExpectation(b));if err==nil{t.Fatal("mismatched chain accepted")}
	})}
}
func TestSignatureTamperingRejects(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k);b.Invocation.Signature="AA"+b.Invocation.Signature[2:]
	_,err:=Verify(b,k.trust,testState(),testExpectation(b));if !errors.Is(err,ErrSignature){t.Fatalf("got %v",err)}
}
func TestWrongIssuerRoleRejects(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k)
	b.Invocation.Payload.Meta.KeyID="gateway-key";b.Invocation.Payload.Meta.Issuer="gateway-svc";b.Invocation,_=Sign(b.Invocation.Payload,k.priv[RoleToolGateway])
	_,err:=Verify(b,k.trust,testState(),testExpectation(b));if !errors.Is(err,ErrWrongRole){t.Fatalf("got %v",err)}
}
func TestTrustDomainsMustBeIndependent(t *testing.T) {
	k:=newTestKeys();k.trust.Keys[1].IndependenceDomain=k.trust.Keys[0].IndependenceDomain;b:=validBundle(t,k)
	_,err:=Verify(b,k.trust,testState(),testExpectation(b));if !errors.Is(err,ErrIndependence){t.Fatalf("got %v",err)}
}
func TestExpiredAndRevokedPermitReject(t *testing.T) {
	for name,change:=range map[string]func(*PolicyState,*Bundle){
		"expired":func(s *PolicyState,b *Bundle){s.NowMS=3000},
		"revoked":func(s *PolicyState,b *Bundle){s.MinimumRevocationGeneration=8},
		"too-long":func(s *PolicyState,b *Bundle){b.Authorization.Payload.ExpiresAtMS=10000;resignBundleNoTest(b)},
	}{
		t.Run(name,func(t *testing.T){k:=newTestKeys();b:=validBundle(t,k);s:=testState();change(&s,&b)
			_,err:=Verify(b,k.trust,s,testExpectation(b))
			if name=="expired"||name=="revoked"{if !errors.Is(err,ErrStalePermit){t.Fatalf("got %v",err)}}else if err==nil{t.Fatal("oversized permit lifetime accepted")}
		})
	}
}
func resignBundleNoTest(b *Bundle) {
	seed:=make([]byte,ed25519.SeedSize);for i:=range seed{seed[i]=byte(17+i)}
	priv:=ed25519.NewKeyFromSeed(seed);b.Authorization,_=Sign(b.Authorization.Payload,priv)
}
func TestWitnessFreshnessSequenceAndBundleBinding(t *testing.T) {
	checks:=map[string]func(*PolicyState,*Bundle){
		"rollback-sequence":func(s *PolicyState,b *Bundle){s.MinimumWitnessSequence=12},
		"wrong-log":func(s *PolicyState,b *Bundle){s.ExpectedWitnessLogID="other-log"},
		"stale":func(s *PolicyState,b *Bundle){s.NowMS=5000},
		"wrong-bundle":func(s *PolicyState,b *Bundle){w:=b.Witness.Payload;w.BundleSHA256=digest("different");b.Witness,_=Sign(w,newTestKeys().priv[RoleWitness])},
	}
	for name,change:=range checks{t.Run(name,func(t *testing.T){k:=newTestKeys();b:=validBundle(t,k);s:=testState();change(&s,&b)
		_,err:=Verify(b,k.trust,s,testExpectation(b));if err==nil{t.Fatal("witness failure accepted")}
	})}
}
func TestOneShotPermitCannotAuthorizeRetry(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k);b.Dispatch.Payload.DispatchAttempt=2;resignBundle(t,&b,k)
	_,err:=Verify(b,k.trust,testState(),testExpectation(b));if !errors.Is(err,ErrPermitReuse){t.Fatalf("got %v",err)}
}
func TestEffectIdempotencyMismatchRejects(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k);b.Effect.Payload.IdempotencyKey="other-idem";resignBundle(t,&b,k)
	_,err:=Verify(b,k.trust,testState(),testExpectation(b));if !errors.Is(err,ErrJoinMismatch){t.Fatalf("got %v",err)}
}
func TestWitnessCannotPrecedeEffect(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k);w:=b.Witness.Payload;w.CheckpointAtMS=1100;b.Witness,_=Sign(w,k.priv[RoleWitness])
	_,err:=Verify(b,k.trust,testState(),testExpectation(b));if !errors.Is(err,ErrStaleWitness){t.Fatalf("got %v",err)}
}
func TestPermitDurationIsBounded(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k);b.Authorization.Payload.ExpiresAtMS=7000;resignBundle(t,&b,k);s:=testState();s.MaximumPermitLifetimeMS=1000
	_,err:=Verify(b,k.trust,s,testExpectation(b));if !errors.Is(err,ErrStalePermit){t.Fatalf("got %v",err)}
}
func TestDistinctReceiptIDsRequired(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k);b.Invocation.Payload.Meta.ReceiptID=b.Authorization.Payload.Meta.ReceiptID;resignBundle(t,&b,k)
	_,err:=Verify(b,k.trust,testState(),testExpectation(b));if err==nil{t.Fatal("duplicate receipt IDs accepted")}
}
func TestUnknownDispatchCanRemainUnreconciled(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k);b.Dispatch.Payload.Status=DispatchUnknown;b.Effect=nil;resignBundle(t,&b,k)
	r,err:=Verify(b,k.trust,testState(),testExpectation(b));if err!=nil{t.Fatal(err)}
	if r.Classification!=OutcomeUnknown{t.Fatalf("got %s",r.Classification)}
}
func TestBundleDigestChangesOnAnyReceiptMutation(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k);base,err:=BundleDigest(b);if err!=nil{t.Fatal(err)}
	b.Dispatch.Payload.Status=DispatchUnknown;next,err:=BundleDigest(b);if err!=nil{t.Fatal(err)}
	if base==next{t.Fatal("bundle digest failed to bind dispatch payload")}
}
func TestAllIndependentDomainsListed(t *testing.T) {
	k:=newTestKeys();domains:=IndependenceDomains(k.trust);if len(domains)!=5{t.Fatalf("got %d domains: %v",len(domains),domains)}
}
func TestValidReceiptMetaSchemaIsRequired(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k);b.Invocation.Payload.Meta.SchemaVersion="unknown";resignBundle(t,&b,k)
	_,err:=Verify(b,k.trust,testState(),testExpectation(b));if !errors.Is(err,ErrInvalidReceipt){t.Fatalf("got %v",err)}
}
func ExampleBundleDigest() {
	fmt.Println(len(digest("example")))
	// Output: 64
}
func TestAuthorizationMustMatchIndependentTaskIntentPin(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k);expected:=testExpectation(b);expected.ArgumentsSHA256=digest("different-expected-arguments")
	if _,err:=Verify(b,k.trust,testState(),expected);!errors.Is(err,ErrJoinMismatch){t.Fatalf("independently pinned intent mismatch accepted: %v",err)}
}
func TestGatewayMustBindActualRuntimeInvocationID(t *testing.T) {
	k:=newTestKeys();b:=validBundle(t,k);b.Dispatch.Payload.RuntimeInvocationID="other-invocation";resignBundle(t,&b,k)
	if _,err:=Verify(b,k.trust,testState(),testExpectation(b));!errors.Is(err,ErrJoinMismatch){t.Fatalf("runtime invocation mismatch accepted: %v",err)}
}
func BenchmarkVerifyValidChain(b *testing.B) {
	keys:=newTestKeys();bundle:=validBundle(b,keys);state:=testState();expected:=testExpectation(bundle);b.ReportAllocs();b.ResetTimer()
	for n:=0;n<b.N;n++{if _,err:=Verify(bundle,keys.trust,state,expected);err!=nil{b.Fatal(err)}}
}

func TestWitnessReceiptIDMustBeGloballyUnique(t *testing.T) {
	k:=newTestKeys(); b:=validBundle(t,k); w:=b.Witness.Payload
	w.Meta.ReceiptID=b.Authorization.Payload.Meta.ReceiptID
	b.Witness,_=Sign(w,k.priv[RoleWitness])
	if _,err:=Verify(b,k.trust,testState(),testExpectation(b)); !errors.Is(err,ErrJoinMismatch) {
		t.Fatalf("duplicate witness receipt ID accepted: %v",err)
	}
}
