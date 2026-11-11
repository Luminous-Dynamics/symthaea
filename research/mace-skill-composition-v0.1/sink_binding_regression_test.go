package main

import (
	"strings"
	"testing"
)

func authorizedActionRequest() Request {
	x := fixtureRequest()
	args := strings.Repeat("d", 64)
	x.Plan.Sinks[0].Kind = "external_action"
	x.Plan.Sinks[0].ActionKind = "case.update"
	x.Plan.Sinks[0].ArgumentsDigest = args
	x.Plan.Sinks[0].RequiredCapability = "case:update"
	x.Plan.Sinks[0].Authorization = &Authorization{
		VerifierStatus:     "verified",
		Principal:          "alice",
		Scope:              "tenant-a",
		ActionKind:         "case.update",
		Target:             "case-7",
		ArgumentsDigest:    args,
		RequiredCapability: "case:update",
		ExpiresAt:          "2027-01-01T00:00:00Z",
		Nonce:              "nonce-1",
	}
	return x
}

func TestSinkAudienceMustBeExplicitAndExact(t *testing.T) {
	x := fixtureRequest()
	x.Plan.Sinks[0].Audience = ""
	r := Validate(x)
	if r.Accepted || !findingCode(r, "SINK_AUDIENCE_MISMATCH") {
		t.Fatalf("omitted sink audience must fail closed: accepted=%v findings=%+v", r.Accepted, r.Findings)
	}

	x = fixtureRequest()
	x.Plan.Sinks[0].Audience = "other-audience"
	r = Validate(x)
	if r.Accepted || !findingCode(r, "SINK_AUDIENCE_MISMATCH") {
		t.Fatalf("different sink audience must fail closed: accepted=%v findings=%+v", r.Accepted, r.Findings)
	}
}

func TestExternalActionRequiresNonemptyActionKind(t *testing.T) {
	x := authorizedActionRequest()
	x.Plan.Sinks[0].ActionKind = ""
	x.Plan.Sinks[0].Authorization.ActionKind = ""
	r := Validate(x)
	if r.Accepted || !findingCode(r, "ACTION_BINDING_INCOMPLETE") {
		t.Fatalf("empty action kind must fail closed: accepted=%v findings=%+v", r.Accepted, r.Findings)
	}
}

func TestPrincipalCapabilitiesMustBeNonemptyAndUnique(t *testing.T) {
	x := authorizedActionRequest()
	x.Context.Capabilities = append(x.Context.Capabilities, "")
	r := Validate(x)
	if r.Accepted || !findingCode(r, "INVALID_CAPABILITIES") {
		t.Fatalf("empty capability entry must fail closed: accepted=%v findings=%+v", r.Accepted, r.Findings)
	}

	x = authorizedActionRequest()
	x.Context.Capabilities = append(x.Context.Capabilities, "case:update")
	r = Validate(x)
	if r.Accepted || !findingCode(r, "INVALID_CAPABILITIES") {
		t.Fatalf("duplicate capability entry must fail closed: accepted=%v findings=%+v", r.Accepted, r.Findings)
	}
}

func TestExternalActionRequiresNonemptyCapabilityBinding(t *testing.T) {
	x := authorizedActionRequest()
	x.Plan.Sinks[0].RequiredCapability = ""
	x.Plan.Sinks[0].Authorization.RequiredCapability = ""
	x.Context.Capabilities = append(x.Context.Capabilities, "")
	r := Validate(x)
	if r.Accepted || !findingCode(r, "ACTION_BINDING_INCOMPLETE") || !findingCode(r, "INVALID_CAPABILITIES") {
		t.Fatalf("empty capability binding must fail closed: accepted=%v findings=%+v", r.Accepted, r.Findings)
	}
}
