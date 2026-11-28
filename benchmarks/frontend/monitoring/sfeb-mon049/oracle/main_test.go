package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func fixture() manifest {
	m := manifest{Profile: manifestProfile, Description: "test", Semantics: struct { ExpectedDisposition string `json:"expected_disposition"` }{"scenario verdict"}}
	for i := 1; i <= 27; i++ {
		c := challenge{ID: "N", Name: "x", Mutation: "mutate", Expected: indeterminate, Category: "adversarial", Description: "fixture"}
		if i <= 3 { c.ID=string(rune('P'+i-1)); c.Name="positive"+c.ID; c.Mutation="none"; c.Expected=qualified; c.Category="positive_control" } else { c.ID=string(rune('A'+i-4))+"x"; c.Name="case"+c.ID; if i%3==0 { c.Expected=blocked } }
		m.Cases=append(m.Cases,c)
	}
	m.Counts.Total=27; m.Counts.Positive=3; m.Counts.Adversarial=24; return m
}
func candidateFor(m manifest,mode string) candidate { c:=candidate{Profile:candidateProfile,CandidateID:"candidate-1"};for _,x:=range m.Cases{d:=x.Expected;switch mode{case "unknown":d=indeterminate;case "block":d=blocked;case "qualify":d=qualified};c.Cases=append(c.Cases,candidateCase{x.ID,d,[]string{"fixture"}})};return c }
func TestKnownAnswerAndProductionFailClosed(t *testing.T){m:=fixture();if _,e:=decodeManifest(mustJSON(m));e!=nil{t.Fatal(e)};r:=evaluate(m,candidateFor(m,"oracle"),strings.Repeat("a",64),strings.Repeat("b",64));if r.Verdict!=qualified||r.Exact.Numerator!=27||r.ProductionUsePermitted{t.Fatalf("unexpected report: %+v",r)}}
func TestUniformStrategiesFail(t *testing.T){m:=fixture();for _,mode:=range []string{"unknown","block","qualify"}{r:=evaluate(m,candidateFor(m,mode),"","");if r.Verdict!=blocked{t.Fatalf("%s should fail",mode)}}}
func TestDuplicateNamesRejected(t *testing.T){var m manifest;e:=strictDecode([]byte(`{"profile":"x","profile":"y"}`),&m);if e==nil{t.Fatal("expected duplicate-member rejection")}}
func TestManifestPinVerifiesExactBytes(t *testing.T){b:=mustJSON(fixture());if err:=verifyManifestPin(b,digest(b));err!=nil{t.Fatalf("correct pin rejected: %v",err)};if err:=verifyManifestPin(b,strings.Repeat("0",64));err==nil{t.Fatal("wrong pin accepted")};if err:=verifyManifestPin(b,"not-a-digest");err==nil{t.Fatal("malformed pin accepted")}}
func mustJSON(v any)[]byte{b,e:=json.Marshal(v);if e!=nil{panic(e)};return b}
