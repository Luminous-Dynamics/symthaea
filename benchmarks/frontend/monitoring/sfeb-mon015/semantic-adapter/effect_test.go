package assurance

import "testing"

func hexDigest(ch byte) string {
	b := make([]byte, 64)
	for i := range b {
		b[i] = ch
	}
	return string(b)
}

func requestEvidence(v string) DigestEvidence {
	return DigestEvidence{Present: true, Valid: true, Algorithm: "sha-256", Value: v,
		Context: DigestContext{Profile: "urn:mon:request", Version: "1", PayloadType: "application/json", Canonicalization: "RFC8785-JCS", Encoding: "UTF-8", Purpose: "request", Subject: "dest-a/op-1"}}
}
func stateEvidence(v string) DigestEvidence {
	return DigestEvidence{Present: true, Valid: true, Algorithm: "sha-256", Value: v,
		Context: DigestContext{Profile: "urn:mon:state", Version: "1", PayloadType: "application/json", Canonicalization: "postgres-jsonb-text-v1", Encoding: "UTF-8", Purpose: "state-after", Subject: "dest-a/resource-a/revision-7"}}
}
func effect(status EffectStatus, mode EffectMode, att string, response bool) EffectAssessment {
	var rd DigestEvidence
	if response {
		rd = DigestEvidence{Present: true, Valid: true, Algorithm: "sha-256", Value: hexDigest('c')}
	}
	return AssessEffect(EffectRecord{Status: status, ResponseDigest: rd, EffectAttestation: att, ClaimedMode: mode})
}

func TestCompareDigestRequiresPresenceValidityAndContext(t *testing.T) {
	a, b := requestEvidence(hexDigest('a')), requestEvidence(hexDigest('a'))
	if CompareDigest(a, b) != DigestEqual { t.Fatal("same verified digest/context must be equal") }
	b.Context.Profile = "urn:other"
	if CompareDigest(a, b) != DigestIncomparable { t.Fatal("same hex across profiles must be incomparable") }
	b = requestEvidence(hexDigest('a'))
	b.Context.Canonicalization = "raw-json-bytes"
	if CompareDigest(a, b) != DigestIncomparable { t.Fatal("canonicalization mismatch must be incomparable") }
	b = requestEvidence(hexDigest('b'))
	if CompareDigest(a, b) != DigestDifferent { t.Fatal("same context/different digest must be different") }
	b = requestEvidence(hexDigest('a')); b.Present = false
	if CompareDigest(a, b) != DigestIndeterminate { t.Fatal("missing digest must be indeterminate") }
	b = requestEvidence(hexDigest('a')); b.Valid = false
	if CompareDigest(a, b) != DigestInvalid { t.Fatal("present but invalid digest must be invalid") }
	b = requestEvidence("bad")
	if CompareDigest(a, b) != DigestInvalid { t.Fatal("malformed digest must be invalid") }
	b = requestEvidence(hexDigest('a')); b.Algorithm = "sha-512"
	if CompareDigest(a, b) != DigestIncomparable { t.Fatal("different algorithm must be incomparable") }
}

func TestAACFiveEffectStatusesAndDigestRequirements(t *testing.T) {
	cases := []struct {
		name string
		rec EffectRecord
		mode EffectMode
		valid bool
		grade AttestationGrade
	}{
		{"planned-with-request", EffectRecord{Status: Planned, RequestDigest: requestEvidence(hexDigest('a'))}, NotApplicable, true, NoExecutionClaim},
		{"planned-without-request", EffectRecord{Status: Planned}, NotApplicable, true, NoExecutionClaim},
		{"planned-no-response", EffectRecord{Status: Planned, ResponseDigest: DigestEvidence{Present: true, Valid: true, Value: hexDigest('c')}}, NotApplicable, false, NoExecutionClaim},
		{"dispatched-request-present", EffectRecord{Status: Dispatched, RequestDigest: requestEvidence(hexDigest('a')), EffectAttestation: "runtime_claimed"}, DispatchedUnconfirmed, true, RuntimeClaimed},
		{"dispatched-request-omitted-should-not-hard-fail", EffectRecord{Status: Dispatched, EffectAttestation: "runtime_claimed"}, DispatchedUnconfirmed, true, RuntimeClaimed},
		{"dispatched-response-forbidden", EffectRecord{Status: Dispatched, EffectAttestation: "runtime_claimed", ResponseDigest: DigestEvidence{Present: true, Valid: true, Value: hexDigest('c')}}, DispatchedUnconfirmed, false, RuntimeClaimed},
		{"confirmed-response", EffectRecord{Status: Confirmed, EffectAttestation: "gate_executed", ResponseDigest: DigestEvidence{Present: true, Valid: true, Value: hexDigest('c')}}, EffectConfirmed, true, GateExecuted},
		{"confirmed-response-missing", EffectRecord{Status: Confirmed, EffectAttestation: "gate_executed"}, DispatchedUnconfirmed, false, GateExecuted},
		{"failed", EffectRecord{Status: Failed, EffectAttestation: "host_served_observed"}, DispatchedUnconfirmed, true, RuntimeClaimed},
		{"reverted", EffectRecord{Status: Reverted, EffectAttestation: "runtime_claimed"}, DispatchedUnconfirmed, true, RuntimeClaimed},
	}
	for _, tc := range cases { t.Run(tc.name, func(t *testing.T) {
		got := AssessEffect(tc.rec)
		if got.DerivedMode != tc.mode || got.Valid != tc.valid || got.Grade != tc.grade { t.Fatalf("got %+v", got) }
	}) }
}

func TestAACUnknownAttestationAndOverclaimRules(t *testing.T) {
	e := EffectRecord{Status: Confirmed, EffectAttestation: "future_registry_token", ResponseDigest: DigestEvidence{Present: true, Valid: true, Value: hexDigest('c')}, ClaimedMode: EffectConfirmed}
	r := AssessEffect(e)
	if !r.Valid || !r.UnknownAttestation || r.Grade != RuntimeClaimed { t.Fatalf("unknown attestation must remain informational and capped: %+v", r) }
	e = EffectRecord{Status: Dispatched, EffectAttestation: "runtime_claimed", ClaimedMode: EffectConfirmed}
	r = AssessEffect(e)
	if r.Valid || !r.Overclaim { t.Fatalf("effect overclaim accepted: %+v", r) }
	e = EffectRecord{Status: Confirmed, EffectAttestation: "gate_executed", ResponseDigest: DigestEvidence{Present: true, Valid: true, Value: hexDigest('c')}, ClaimedMode: DispatchedUnconfirmed}
	r = AssessEffect(e)
	if !r.Valid || r.Overclaim { t.Fatalf("effect underclaim should not fail: %+v", r) }
}

func TestAACVerdictEffectOrthogonality(t *testing.T) {
	cases := []struct { verdict string; mode EffectMode; want Orthogonality }{
		{"denied", NotApplicable, OrthogonalPass},
		{"denied", DispatchedUnconfirmed, OrthogonalFail},
		{"blocked", NotApplicable, OrthogonalPass},
		{"timeout", NotApplicable, OrthogonalPass},
		{"timeout", DispatchedUnconfirmed, OrthogonalPass},
		{"timeout", EffectConfirmed, OrthogonalFail},
		{"errored", DispatchedUnconfirmed, OrthogonalPass},
		{"errored", NotApplicable, OrthogonalFail},
		{"executed", NotApplicable, OrthogonalFail},
		{"future-verdict", EffectConfirmed, OrthogonalInformational},
	}
	for _, tc := range cases {
		got, _ := CheckVerdictEffect(tc.verdict, tc.mode)
		if got != tc.want { t.Errorf("%s/%s: got %s want %s", tc.verdict, tc.mode, got, tc.want) }
	}
}

func TestAACAssuranceAxesAreIndependent(t *testing.T) {
	claims := AssuranceClaims{Attestation: AttestationAnchored, Effect: EffectConfirmed, Ledger: LedgerAnchored, CrossParty: CrossPartyBilateral}
	base := AssuranceEvidence{CapsuleIDRegistrationReceiptValid: true, OwnChainBlockIntact: true, ChainRootInclusionProofValid: true,
		CrossParty: CrossPartyEvidence{BlockPresent: true, InitiatorRefValid: true, CounterpartyRefVerified: true, CorrelatorVerified: true, SubstantiveCoSignedResult: true}}
	r := ReconcileAssurance(claims, effect(Confirmed, EffectConfirmed, "gate_executed", true), base)
	if !r.Valid || len(r.Overclaims) != 0 { t.Fatalf("all axes with evidence should qualify: %+v", r) }

	e := base; e.OwnChainBlockIntact = false; e.ChainRootInclusionProofValid = false
	r = ReconcileAssurance(AssuranceClaims{Attestation: AttestationAnchored, Effect: EffectConfirmed, Ledger: LedgerStandalone, CrossParty: CrossPartyBilateral}, effect(Confirmed, EffectConfirmed, "gate_executed", true), e)
	if !r.Valid || r.Derived.Attestation != AttestationAnchored || r.Derived.Ledger != LedgerStandalone { t.Fatalf("attestation and ledger collapsed: %+v", r) }

	e = AssuranceEvidence{OwnChainBlockIntact: true}
	r = ReconcileAssurance(AssuranceClaims{Attestation: AttestationSelf, Effect: NotApplicable, Ledger: LedgerChained}, effect(Planned, NotApplicable, "", false), e)
	if !r.Valid || r.Derived.Ledger != LedgerChained { t.Fatalf("chained ledger needs no parent resolution here: %+v", r) }

	e = AssuranceEvidence{}
	r = ReconcileAssurance(AssuranceClaims{Attestation: AttestationSelf, Effect: NotApplicable, Ledger: LedgerStandalone}, effect(Planned, NotApplicable, "", false), e)
	if !r.Valid || r.Derived.CrossParty != "" { t.Fatalf("absent optional block rejected: %+v", r) }
}

func TestAACCrossPartyLevelsDoNotEstablishControlIndependence(t *testing.T) {
	for _, tc := range []struct { e CrossPartyEvidence; want CrossPartyLevel }{
		{CrossPartyEvidence{BlockPresent: true, InitiatorRefValid: true}, CrossPartyUnilateral},
		{CrossPartyEvidence{BlockPresent: true, InitiatorRefValid: true, CounterpartyRefVerified: true, CorrelatorVerified: true}, CrossPartyAcknowledged},
		{CrossPartyEvidence{BlockPresent: true, InitiatorRefValid: true, CounterpartyRefVerified: true, CorrelatorVerified: true, SubstantiveCoSignedResult: true}, CrossPartyBilateral},
	} {
		got := deriveCrossParty(tc.e)
		if got != tc.want { t.Errorf("got %s want %s", got, tc.want) }
	}
	e := AssuranceEvidence{CrossParty: CrossPartyEvidence{BlockPresent: true, Malformed: true}}
	r := ReconcileAssurance(AssuranceClaims{Attestation: AttestationSelf, Effect: NotApplicable, Ledger: LedgerStandalone}, effect(Planned, NotApplicable, "", false), e)
	if r.Valid { t.Fatal("malformed cross-party block must fail") }
}

func deriveCrossParty(e CrossPartyEvidence) CrossPartyLevel {
	claims := AssuranceClaims{Attestation: AttestationSelf, Effect: NotApplicable, Ledger: LedgerStandalone}
	r := ReconcileAssurance(claims, effect(Planned, NotApplicable, "", false), AssuranceEvidence{CrossParty: e})
	return r.Derived.CrossParty
}

func TestDigitalEffectNeedsExactScopeDigestReadbackAndAuthority(t *testing.T) {
	in := validDigitalInput()
	r := QualifyDigitalEffect(in)
	if r.Result != QualifiedDigital || r.RequestRelation != DigestEqual || r.DigestRelation != DigestEqual { t.Fatalf("got %+v", r) }
	in.ObserverVerification = Missing
	if got := QualifyDigitalEffect(in).Result; got != ReadbackMissing { t.Fatalf("missing observer: %s", got) }
	in = validDigitalInput(); in.ObserverPrincipal = in.SinkScope.Principal
	if got := QualifyDigitalEffect(in).Result; got != Contradicted { t.Fatalf("same principal observer: %s", got) }
	in = validDigitalInput(); in.ObserverScope.Principal = "spoofed-other-principal"
	if got := QualifyDigitalEffect(in).Result; got != Contradicted { t.Fatalf("unbound observer principal: %s", got) }
	in = validDigitalInput(); in.ObserverIndependent = false
	if got := QualifyDigitalEffect(in).Result; got != NotIndependent { t.Fatalf("non-independent observer: %s", got) }
	in = validDigitalInput(); in.AuthorityAtCommit = "unknown"
	if got := QualifyDigitalEffect(in).Result; got != AuthorityUnknownResult { t.Fatalf("unknown commit authority: %s", got) }
	in = validDigitalInput(); in.AuthorityAtCommit = "invalid"
	if got := QualifyDigitalEffect(in).Result; got != AuthorityInvalidResult { t.Fatalf("invalid commit authority: %s", got) }
}

func TestDigitalEffectMutationsNeverPromote(t *testing.T) {
	mutants := []struct { name string; mutate func(*DigitalEffectInput); want DigitalEffectResult }{
		{"sink-unverified", func(x *DigitalEffectInput) { x.SinkVerification = Missing }, Indeterminate},
		{"sink-invalid", func(x *DigitalEffectInput) { x.SinkVerification = Invalid }, Contradicted},
		{"request-digest-changed", func(x *DigitalEffectInput) { x.SinkRequest = requestEvidence(hexDigest('d')) }, Contradicted},
		{"request-context-changed", func(x *DigitalEffectInput) { x.SinkRequest.Context.Profile = "urn:other" }, Indeterminate},
		{"operation-id-substituted", func(x *DigitalEffectInput) { x.SinkScope.Operation = "op-2" }, Contradicted},
		{"tenant-substituted", func(x *DigitalEffectInput) { x.ObserverScope.Tenant = "tenant-b" }, Contradicted},
		{"revision-substituted", func(x *DigitalEffectInput) { x.ObserverRevision++ }, Contradicted},
		{"state-digest-changed", func(x *DigitalEffectInput) { x.ObserverState.Value = hexDigest('e') }, Contradicted},
		{"sink-outcome-unresolved", func(x *DigitalEffectInput) { x.SinkOutcome = SinkOutcomeUnresolved }, Indeterminate},
		{"producer-not-effect", func(x *DigitalEffectInput) { x.Effect = effect(Planned, NotApplicable, "", false) }, Contradicted},
	}
	for _, m := range mutants { t.Run(m.name, func(t *testing.T) {
		in := validDigitalInput(); m.mutate(&in)
		got := QualifyDigitalEffect(in).Result
		if got != m.want { t.Fatalf("got %s want %s", got, m.want) }
	}) }
}

func TestDigitalEffectDistinguishesRefusalFromUnresolved(t *testing.T) {
	in := validDigitalInput(); in.SinkOutcome = SinkOutcomeRefused
	if got := QualifyDigitalEffect(in).Result; got != Refused { t.Fatalf("unconfirmed producer plus refusal should remain refused: %s", got) }
	in = validDigitalInput(); in.Effect = effect(Confirmed, EffectConfirmed, "gate_executed", true); in.SinkOutcome = SinkOutcomeRefused
	if got := QualifyDigitalEffect(in).Result; got != Contradicted { t.Fatalf("producer confirmed vs sink refused should be contradiction: %s", got) }
}

func validDigitalInput() DigitalEffectInput {
	return DigitalEffectInput{
		ProducerVerification: Verified, SinkVerification: Verified, ObserverVerification: Verified,
		ProducerScope: OperationScope{"destination-a", "operation-1", "writer-a", "tenant-a", "resource-a"},
		SinkScope: OperationScope{"destination-a", "operation-1", "writer-a", "tenant-a", "resource-a"},
		ObserverScope: OperationScope{"destination-a", "operation-1", "observer-a", "tenant-a", "resource-a"},
		ProducerRequest: requestEvidence(hexDigest('a')), SinkRequest: requestEvidence(hexDigest('a')),
		Effect: effect(Dispatched, DispatchedUnconfirmed, "runtime_claimed", false), SinkOutcome: SinkOutcomeCommitted,
		SinkRevision: 7, ObserverRevision: 7, SinkState: stateEvidence(hexDigest('b')), ObserverState: stateEvidence(hexDigest('b')),
		ObserverPrincipal: "observer-a", ObserverIndependent: true, AuthorityAtCommit: "valid",
	}
}

func TestEveryMutantCorpusEntryHasAnExpectedVerdict(t *testing.T) {
	// Independently authored, deterministic one-variable mutations. The
	// expected results are explicit and the implementation under test never
	// generates its own expected verdict.
	tests := []struct { name string; run func() string; want string }{
		{"scope-substitution", func() string { x:=validDigitalInput(); x.SinkScope.Resource="resource-b"; return string(QualifyDigitalEffect(x).Result) }, string(Contradicted)},
		{"operation-substitution", func() string { x:=validDigitalInput(); x.SinkScope.Operation="op-2"; return string(QualifyDigitalEffect(x).Result) }, string(Contradicted)},
		{"request-digest-substitution", func() string { x:=validDigitalInput(); x.SinkRequest=requestEvidence(hexDigest('f')); return string(QualifyDigitalEffect(x).Result) }, string(Contradicted)},
		{"request-profile-confusion", func() string { x:=validDigitalInput(); x.SinkRequest.Context.Profile="urn:other"; return string(QualifyDigitalEffect(x).Result) }, string(Indeterminate)},
		{"response-digest-removal", func() string { x:=AssessEffect(EffectRecord{Status:Confirmed,EffectAttestation:"gate_executed"}); return string(boolVerdict(x.Valid)) }, string(OrthogonalFail)},
		{"effect-mode-overclaim", func() string { x:=AssessEffect(EffectRecord{Status:Dispatched,EffectAttestation:"runtime_claimed",ClaimedMode:EffectConfirmed}); return string(boolVerdict(x.Valid)) }, string(OrthogonalFail)},
		{"wrong-observer-principal", func() string { x:=validDigitalInput(); x.ObserverPrincipal=x.SinkScope.Principal; return string(QualifyDigitalEffect(x).Result) }, string(Contradicted)},
		{"shared-control-root", func() string { x:=validDigitalInput(); x.ObserverIndependent=false; return string(QualifyDigitalEffect(x).Result) }, string(NotIndependent)},
		{"wrong-revision", func() string { x:=validDigitalInput(); x.ObserverRevision++; return string(QualifyDigitalEffect(x).Result) }, string(Contradicted)},
		{"state-digest-substitution", func() string { x:=validDigitalInput(); x.ObserverState.Value=hexDigest('f'); return string(QualifyDigitalEffect(x).Result) }, string(Contradicted)},
		{"authority-revoked", func() string { x:=validDigitalInput(); x.AuthorityAtCommit="invalid"; return string(QualifyDigitalEffect(x).Result) }, string(AuthorityInvalidResult)},
		{"authority-uncheckable", func() string { x:=validDigitalInput(); x.AuthorityAtCommit="uncheckable"; return string(QualifyDigitalEffect(x).Result) }, string(AuthorityUnknownResult)},
		{"producer-claims-confirmed-but-sink-refuses", func() string { x:=validDigitalInput(); x.Effect=effect(Confirmed,EffectConfirmed,"gate_executed",true); x.SinkOutcome=SinkOutcomeRefused; return string(QualifyDigitalEffect(x).Result) }, string(Contradicted)},
		{"timeout-with-confirmed-effect", func() string { x,_:=CheckVerdictEffect("timeout",EffectConfirmed); return string(x) }, string(OrthogonalFail)},
		{"errored-with-not-applicable", func() string { x,_:=CheckVerdictEffect("errored",NotApplicable); return string(x) }, string(OrthogonalFail)},
		{"OAR-execution-after-denial", func() string { d:=OARRecord{ReceiptID:"d",AgentID:"a",RunID:"r",ActionType:"refund",ActionTarget:"stripe",Phase:OARDecision,Result:"denied",Verification:Verified}; e:=OARRecord{ReceiptID:"e",DecisionRef:"d",AgentID:"a",RunID:"r",ActionType:"refund",ActionTarget:"stripe",Phase:OARExecution,Result:"executed",Verification:Verified}; return string(ProjectOARLifecycle([]OARRecord{d,e}).Projection) }, string(OARContradicted)},
		{"OAR-missing-confirmation", func() string { d:=OARRecord{ReceiptID:"d",AgentID:"a",RunID:"r",ActionType:"refund",ActionTarget:"stripe",Phase:OARDecision,Result:"allowed",Verification:Verified}; e:=OARRecord{ReceiptID:"e",DecisionRef:"d",AgentID:"a",RunID:"r",ActionType:"refund",ActionTarget:"stripe",Phase:OARExecution,Result:"executed",Verification:Verified}; return string(ProjectOARLifecycle([]OARRecord{d,e}).Projection) }, string(OARExecutedUnconfirmed)},
		{"OAR-cross-run-reference", func() string { d:=OARRecord{ReceiptID:"d",AgentID:"a",RunID:"r",ActionType:"refund",ActionTarget:"stripe",Phase:OARDecision,Result:"allowed",Verification:Verified}; e:=OARRecord{ReceiptID:"e",DecisionRef:"d",AgentID:"a",RunID:"other",ActionType:"refund",ActionTarget:"stripe",Phase:OARExecution,Result:"executed",Verification:Verified}; return string(ProjectOARLifecycle([]OARRecord{d,e}).Projection) }, string(OARContradicted)},
	}
	if len(tests) != 18 { t.Fatalf("expected 18 mutation cases, got %d", len(tests)) }
	for _, tc := range tests { t.Run(tc.name, func(t *testing.T) { if got:=tc.run(); got!=tc.want { t.Fatalf("got %q want %q",got,tc.want) } }) }
}

func boolVerdict(valid bool) Orthogonality {
	if valid { return OrthogonalPass }
	return OrthogonalFail
}
