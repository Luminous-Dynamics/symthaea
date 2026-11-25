// Command cpdc evaluates deterministic source-drift and decision-binding fixtures.
// Strategies are semantic baselines, not adapters for named vendors.
package main

import (
 "encoding/json"
 "errors"
 "fmt"
 "os"
)

type State string
const (
 Current State = "current"
 Superseded State = "superseded"
 Retracted State = "retracted"
 Disputed State = "disputed"
 Unavailable State = "unavailable"
 Restored State = "restored"
)
type ActionState string
const ( Pending ActionState = "pending"; Executed ActionState = "executed" )
type Case struct {
 ID string `json:"id"`
 Name string `json:"name"`
 CurrentState State `json:"current_state"`
 HistoricalQuery bool `json:"historical_query"`
 HistoricalState State `json:"historical_state,omitempty"
 SourceAuthenticated bool `json:"source_authenticated"`
 SemanticMatch bool `json:"semantic_match"`
 AffectsAction bool `json:"affects_action"`
 ActionState ActionState `json:"action_state"`
 ApprovalBytesMatch bool `json:"approval_bytes_match"`
 DecisionAuthorized bool `json:"decision_authorized"
 MeasurementAuthority string `json:"measurement_authority"
}
type Expected struct {
 SourceUsable bool `json:"source_usable"
 Reauthorize bool `json:"reauthorize"
 ActionAllowed bool `json:"action_allowed"
 PreserveExecution bool `json:"preserve_execution"
 CurrentRiskReview bool `json:"current_risk_review"
}
type Fixture struct { Version string `json:"version"`; Cases []Case `json:"cases"`; Expected map[string]Expected `json:"expected"` }
type Prediction struct { SourceUsable bool; Reauthorize bool; ActionAllowed bool; PreserveExecution bool; CurrentRiskReview bool }
type Metrics struct {
 Cases int `json:"cases"
 FalseCurrent int `json:"false_current"
 FalseStale int `json:"false_stale"
 MissedReauthorizations int `json:"missed_reauthorizations"
 UnauthorizedAllows int `json:"unauthorized_allows"
 MissedRiskReviews int `json:"missed_current_risk_reviews"
 HistoricalErrors int `json:"historical_as_of_errors"
 CorrectDecisions int `json:"exact_expected_decisions"
}
type Report struct { Version string `json:"version"`; Semantics string `json:"semantics"`; Strategies map[string]Metrics `json:"strategies"`; Disclaimer string `json:"disclaimer"` }
func effectiveState(c Case) State { if c.HistoricalQuery { return c.HistoricalState }; return c.CurrentState }
func usable(s State) bool { return s == Current || s == Restored }
func evaluate(strategy string, c Case) Prediction {
 p:=Prediction{PreserveExecution:c.ActionState==Executed}
 invalidated:=c.CurrentState==Superseded || c.CurrentState==Retracted || c.CurrentState==Disputed || c.CurrentState==Unavailable
 if strategy=="citation-only" { p.SourceUsable=c.SourceAuthenticated; p.ActionAllowed=p.SourceUsable; return p }
 p.SourceUsable=c.SourceAuthenticated && c.SemanticMatch && usable(effectiveState(c))
 if strategy=="lineage-aware" { p.Reauthorize=c.AffectsAction && invalidated && c.ActionState==Pending; p.CurrentRiskReview=c.AffectsAction && invalidated && c.ActionState==Executed }
 // MeasurementAuthority is intentionally ignored: measurement is not authorization.
 p.ActionAllowed=p.SourceUsable && c.ApprovalBytesMatch && c.DecisionAuthorized && !p.Reauthorize
 return p
}
func score(f Fixture, strategy string) (Metrics,error) {
 m:=Metrics{Cases:len(f.Cases)}
 for _,c:=range f.Cases {
  want,ok:=f.Expected[c.ID]; if !ok { return m,fmt.Errorf("missing expected result for %s",c.ID) }
  got:=evaluate(strategy,c)
  if got.SourceUsable && !want.SourceUsable { m.FalseCurrent++ }
  if !got.SourceUsable && want.SourceUsable { m.FalseStale++ }
  if want.Reauthorize && !got.Reauthorize { m.MissedReauthorizations++ }
  if got.ActionAllowed && !want.ActionAllowed { m.UnauthorizedAllows++ }
  if want.CurrentRiskReview && !got.CurrentRiskReview { m.MissedRiskReviews++ }
  if c.HistoricalQuery && got.SourceUsable!=want.SourceUsable { m.HistoricalErrors++ }
  if got==(Prediction{SourceUsable:want.SourceUsable,Reauthorize:want.Reauthorize,ActionAllowed:want.ActionAllowed,PreserveExecution:want.PreserveExecution,CurrentRiskReview:want.CurrentRiskReview}) { m.CorrectDecisions++ }
 }
 return m,nil
}
func load(path string)(Fixture,error) {
 b,err:=os.ReadFile(path); if err!=nil { return Fixture{},err }
 var f Fixture; if err=json.Unmarshal(b,&f); err!=nil { return Fixture{},err }
 if f.Version=="" || len(f.Cases)==0 { return Fixture{},errors.New("fixture requires version and cases") }
 return f,nil
}
func run(path string)(Report,error) {
 f,err:=load(path); if err!=nil { return Report{},err }
 r:=Report{Version:f.Version,Semantics:"synthetic semantic qualification; lower is better except exact_expected_decisions",Strategies:map[string]Metrics{},Disclaimer:"These fixtures compare explicit policy baselines only. They are not measurements of DBOS, Temporal, Rekor, SCITT, or any external product, and are not a production qualification result."}
 for _,name:=range []string{"citation-only","snapshot-only","lineage-aware"} { m,e:=score(f,name); if e!=nil{return Report{},e}; r.Strategies[name]=m }
 return r,nil
}
func main(){ path:="cases-v1.json"; if len(os.Args)>1 {path=os.Args[1]}; r,err:=run(path); if err!=nil {fmt.Fprintln(os.Stderr,err);os.Exit(1)}; enc:=json.NewEncoder(os.Stdout);enc.SetIndent("","  ");if err=enc.Encode(r);err!=nil{fmt.Fprintln(os.Stderr,err);os.Exit(1)} }
