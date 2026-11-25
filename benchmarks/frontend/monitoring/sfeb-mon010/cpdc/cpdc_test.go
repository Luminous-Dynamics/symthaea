package main

import "testing"

func fixture(t *testing.T) Fixture { t.Helper(); f,err:=load("cases-v1.json");if err!=nil{t.Fatal(err)};return f }
func TestLineageAwareMatchesAllGoldenCases(t *testing.T){f:=fixture(t);m,err:=score(f,"lineage-aware");if err!=nil{t.Fatal(err)};if m.CorrectDecisions!=len(f.Cases)||m.FalseCurrent!=0||m.FalseStale!=0||m.MissedReauthorizations!=0||m.UnauthorizedAllows!=0||m.MissedRiskReviews!=0||m.HistoricalErrors!=0{t.Fatalf("lineage-aware failed golden corpus: %+v",m)}}
func TestCitationOnlyExposesStaleSourceAndAuthorityFailures(t *testing.T){m,err:=score(fixture(t),"citation-only");if err!=nil{t.Fatal(err)};if m.FalseCurrent<5||m.UnauthorizedAllows<3||m.MissedReauthorizations<4{t.Fatalf("citation-only unexpectedly passed: %+v",m)}}
func TestSnapshotMissesCorrectionPropagation(t *testing.T){m,err:=score(fixture(t),"snapshot-only");if err!=nil{t.Fatal(err)};if m.FalseCurrent!=0||m.MissedReauthorizations<4||m.MissedRiskReviews!=1{t.Fatalf("snapshot did not expose lineage gap: %+v",m)}}
func TestAuthorityFreeMeasurementDoesNotAuthorize(t *testing.T){for _,c:=range fixture(t).Cases{if (c.ID=="C12"||c.ID=="C13")&&evaluate("lineage-aware",c).ActionAllowed{t.Fatalf("%s incorrectly authorized",c.ID)}}}
func TestHistoricalAsOfStateIsNotRetroactivelyInvalidated(t *testing.T){for _,c:=range fixture(t).Cases{if c.ID=="C09"{got:=evaluate("lineage-aware",c);if !got.SourceUsable||got.Reauthorize{t.Fatalf("historical query retroactively invalidated: %+v",got)};return}};t.Fatal("C09 missing")}
func TestExecutedActionIsImmutableAndRiskReviewed(t *testing.T){for _,c:=range fixture(t).Cases{if c.ID=="C05"{got:=evaluate("lineage-aware",c);if !got.PreserveExecution||!got.CurrentRiskReview||got.Reauthorize{t.Fatalf("executed action semantics wrong: %+v",got)};return}};t.Fatal("C05 missing")}
