// MON-049 is a deterministic benchmark scorer, not a recovery executor or authorizer.
package main

import (
	"bytes"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
)

const (
	manifestProfile  = "mon-evaluator-challenge-v1"
	candidateProfile = "mon-evaluator-candidate-results-v1"
	maxInputBytes    = 16 << 20
)

type disposition string

const (
	qualified     disposition = "qualified"
	blocked       disposition = "blocked"
	indeterminate disposition = "indeterminate"
)

func (d disposition) valid() bool { return d == qualified || d == blocked || d == indeterminate }

type manifest struct {
	Profile     string `json:"profile"`
	Description string `json:"description"`
	Counts      struct {
		Total       int `json:"total"`
		Positive    int `json:"positive_controls"`
		Adversarial int `json:"negative_or_adversarial"`
	} `json:"counts"`
	Cases     []challenge `json:"cases"`
	Semantics struct {
		ExpectedDisposition string `json:"expected_disposition"`
	} `json:"semantics"`
}
type challenge struct {
	ID          string      `json:"id"`
	Name        string      `json:"name"`
	Mutation    string      `json:"mutation"`
	Expected    disposition `json:"expected_disposition"`
	Category    string      `json:"category"`
	Description string      `json:"description"`
}
type candidate struct {
	Profile     string          `json:"profile"`
	CandidateID string          `json:"candidate_id"`
	Cases       []candidateCase `json:"cases"`
}
type candidateCase struct {
	ID          string      `json:"id"`
	Disposition disposition `json:"disposition"`
	ReasonCodes []string    `json:"reason_codes"`
}
type metric struct {
	Numerator   int     `json:"numerator"`
	Denominator int     `json:"denominator"`
	Rate        float64 `json:"rate"`
}
type gate struct {
	Name        string `json:"name"`
	Passed      bool   `json:"passed"`
	Observed    string `json:"observed"`
	Requirement string `json:"requirement"`
}
type report struct {
	Profile                string      `json:"profile"`
	CandidateID            string      `json:"candidate_id"`
	ManifestSHA256         string      `json:"manifest_sha256"`
	CandidateSHA256        string      `json:"candidate_output_sha256"`
	TotalCases             int         `json:"total_cases"`
	ExpectedDecisive       int         `json:"expected_decisive_cases"`
	Exact                  metric      `json:"exact_disposition"`
	DecisionCoverage       metric      `json:"decision_coverage"`
	FalseQualified         metric      `json:"false_qualified"`
	FalseBlocked           metric      `json:"false_blocked"`
	FalseIndeterminate     metric      `json:"false_indeterminate"`
	UnexpectedDecisive     metric      `json:"unexpected_decisive_on_indeterminate_oracle"`
	AllIndeterminate       bool        `json:"all_indeterminate_strategy"`
	AllBlocked             bool        `json:"all_blocked_strategy"`
	AllQualified           bool        `json:"all_qualified_strategy"`
	Gates                  []gate      `json:"gates"`
	Mismatches             []string    `json:"mismatches"`
	Verdict                disposition `json:"qualification_verdict"`
	ProductionUsePermitted bool        `json:"production_use_permitted"`
}

var idRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)
var shaRE = regexp.MustCompile(`^[0-9a-f]{64}$`)

func digest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func verifyManifestPin(raw []byte, expected string) error {
	if !shaRE.MatchString(expected) { return errors.New("expected manifest SHA-256 is malformed") }
	actual := digest(raw)
	if subtle.ConstantTimeCompare([]byte(actual), []byte(expected)) != 1 { return errors.New("manifest digest mismatch") }
	return nil
}
func duplicateKeys(raw []byte) error {
	d := json.NewDecoder(bytes.NewReader(raw))
	if err := consume(d); err != nil { return err }
	if _, err := d.Token(); err != io.EOF {
		if err == nil { return errors.New("trailing JSON value") }
		return err
	}
	return nil
}
func consume(d *json.Decoder) error {
	t, err := d.Token(); if err != nil { return err }
	x, ok := t.(json.Delim); if !ok { return nil }
	switch x {
	case '{':
		seen := map[string]bool{}
		for d.More() {
			kt, e := d.Token(); if e != nil { return e }
			k, ok := kt.(string); if !ok { return errors.New("invalid object key") }
			if seen[k] { return fmt.Errorf("duplicate JSON member %q", k) }; seen[k] = true
			if e = consume(d); e != nil { return e }
		}
		end, e := d.Token(); if e != nil { return e }; if end != json.Delim('}') { return errors.New("object terminator missing") }
	case '[':
		for d.More() { if e := consume(d); e != nil { return e } }
		end, e := d.Token(); if e != nil { return e }; if end != json.Delim(']') { return errors.New("array terminator missing") }
	default: return fmt.Errorf("unexpected delimiter %q", x)
	}
	return nil
}
func strictDecode(raw []byte, dst any) error {
	if len(raw) == 0 || len(raw) > maxInputBytes { return fmt.Errorf("input size invalid: %d", len(raw)) }
	if err := duplicateKeys(raw); err != nil { return err }
	d := json.NewDecoder(bytes.NewReader(raw)); d.DisallowUnknownFields()
	if err := d.Decode(dst); err != nil { return err }
	var extra any; if err := d.Decode(&extra); err != io.EOF { if err == nil { return errors.New("trailing JSON value") }; return err }
	return nil
}
func decodeManifest(raw []byte) (manifest, error) {
	var m manifest
	if err := strictDecode(raw, &m); err != nil { return m, fmt.Errorf("manifest invalid: %w", err) }
	if m.Profile != manifestProfile || strings.TrimSpace(m.Description) == "" || strings.TrimSpace(m.Semantics.ExpectedDisposition) == "" { return m, errors.New("manifest profile/description/semantics invalid") }
	if m.Counts.Total != 27 || m.Counts.Positive != 3 || m.Counts.Adversarial != 24 || len(m.Cases) != 27 { return m, fmt.Errorf("manifest cardinality invalid: expected 27/3/24, got %d/%d/%d", m.Counts.Total, m.Counts.Positive, m.Counts.Adversarial) }
	ids, names := map[string]bool{}, map[string]bool{}; p,n := 0,0
	for _, c := range m.Cases {
		if !idRE.MatchString(c.ID) || strings.TrimSpace(c.Name)=="" || strings.TrimSpace(c.Description)=="" || strings.TrimSpace(c.Mutation)=="" || strings.TrimSpace(c.Category)=="" || !c.Expected.valid() { return m, fmt.Errorf("invalid challenge metadata/disposition: %s",c.ID) }
		if ids[c.ID] || names[c.Name] { return m, fmt.Errorf("duplicate challenge identity: %s",c.ID) }; ids[c.ID]=true; names[c.Name]=true
		if c.Category=="positive_control" { p++; if c.Expected!=qualified || c.Mutation!="none" { return m,fmt.Errorf("invalid positive control: %s",c.ID) } } else { n++; if c.Mutation=="none" { return m,fmt.Errorf("adversarial mutation absent: %s",c.ID) } }
	}
	if p!=3 || n!=24 { return m,fmt.Errorf("manifest partitions invalid: %d positive / %d adversarial",p,n) }
	return m,nil
}
func decodeCandidate(raw []byte,m manifest)(candidate,error){
	var c candidate; if err:=strictDecode(raw,&c);err!=nil{return c,fmt.Errorf("candidate invalid: %w",err)}
	if c.Profile!=candidateProfile || !idRE.MatchString(c.CandidateID) || len(c.Cases)!=len(m.Cases){return c,errors.New("candidate profile, identity or case cardinality invalid")}
	want:=map[string]bool{};for _,x:=range m.Cases{want[x.ID]=true};seen:=map[string]bool{}
	for _,x:=range c.Cases{if !want[x.ID]||seen[x.ID]||!x.Disposition.valid()||len(x.ReasonCodes)==0{return c,fmt.Errorf("candidate case invalid: %s",x.ID)};seen[x.ID]=true;for _,r:=range x.ReasonCodes{if strings.TrimSpace(r)==""{return c,fmt.Errorf("empty reason code: %s",x.ID)}}}
	if len(seen)!=len(want){return c,errors.New("candidate case coverage incomplete")};return c,nil
}
func rate(n,d int)metric{v:=0.0;if d>0{v=float64(n)/float64(d)};return metric{n,d,v}}
func evaluate(m manifest,c candidate,md,cd string)report{
	r:=report{Profile:"mon-evaluator-score-receipt-v1",CandidateID:c.CandidateID,ManifestSHA256:md,CandidateSHA256:cd,TotalCases:len(m.Cases),Mismatches:[]string{},Gates:[]gate{},Verdict:blocked,ProductionUsePermitted:false}
	actual:=map[string]disposition{};for _,x:=range c.Cases{actual[x.ID]=x.Disposition}
	exact,decisive,covered,fq,unsafe,fb,expectedQ,fi,ud,expectedI,pos,fp:=0,0,0,0,0,0,0,0,0,0,0,0
	r.AllIndeterminate=true;r.AllBlocked=true;r.AllQualified=true
	for _,x:=range m.Cases{a,e:=actual[x.ID],x.Expected;if a!=indeterminate{r.AllIndeterminate=false};if a!=blocked{r.AllBlocked=false};if a!=qualified{r.AllQualified=false};if a==e{exact++}else{r.Mismatches=append(r.Mismatches,fmt.Sprintf("%s:%s->%s",x.ID,e,a))};if e!=indeterminate{decisive++;if a!=indeterminate{covered++}else{fi++}};if e!=qualified{unsafe++;if a==qualified{fq++}}else{expectedQ++;if a==blocked{fb++}};if e==indeterminate{expectedI++;if a!=indeterminate{ud++}};if x.Category=="positive_control"{pos++;if a==blocked{fp++}}}
	r.ExpectedDecisive=decisive;r.Exact=rate(exact,27);r.DecisionCoverage=rate(covered,decisive);r.FalseQualified=rate(fq,unsafe);r.FalseBlocked=rate(fb,expectedQ);r.FalseIndeterminate=rate(fi,decisive);r.UnexpectedDecisive=rate(ud,expectedI)
	add:=func(name string,ok bool,observed,req string){r.Gates=append(r.Gates,gate{name,ok,observed,req})}
	add("known_answer_exact",exact==27,fmt.Sprintf("%d/27",exact),"100% exact dispositions")
	add("zero_false_qualification",fq==0,fmt.Sprintf("%d/%d",fq,unsafe),"zero unsafe cases qualified")
	add("positive_controls_preserved",pos==3&&fp==0,fmt.Sprintf("%d/%d blocked",fp,pos),"zero positive controls blocked")
	add("qualified_controls_not_blocked",expectedQ>0&&fb==0,fmt.Sprintf("%d/%d blocked",fb,expectedQ),"zero oracle-qualified controls blocked")
	add("indeterminate_not_overridden",ud==0,fmt.Sprintf("%d/%d decided",ud,expectedI),"oracle-indeterminate cases remain indeterminate")
	add("decision_coverage",decisive>0&&covered==decisive,fmt.Sprintf("%d/%d",covered,decisive),"100% decision coverage on oracle-decisive cases")
	add("anti_gaming",!r.AllIndeterminate&&!r.AllBlocked&&!r.AllQualified,fmt.Sprintf("all-indeterminate=%t all-blocked=%t all-qualified=%t",r.AllIndeterminate,r.AllBlocked,r.AllQualified),"reject uniform evaluator strategies")
	passed:=true;for _,g:=range r.Gates{if !g.Passed{passed=false}};if passed{r.Verdict=qualified};return r
}
func readBounded(path string)([]byte,error){f,e:=os.Open(path);if e!=nil{return nil,e};defer f.Close();b,e:=io.ReadAll(io.LimitReader(f,maxInputBytes+1));if e!=nil{return nil,e};if len(b)>maxInputBytes{return nil,errors.New("input exceeds 16 MiB")};return b,nil}
func run()int{
	mp:=flag.String("manifest","","frozen MON-048 challenge manifest");cp:=flag.String("candidate","","candidate results JSON");pin:=flag.String("expected-manifest-sha256","","out-of-band preregistered manifest SHA-256");out:=flag.String("out","","optional output file; default stdout");flag.Parse()
	if *mp==""||*cp==""||!shaRE.MatchString(*pin){fmt.Fprintln(os.Stderr,"usage: qualify -manifest PATH -candidate PATH -expected-manifest-sha256 SHA256 [-out PATH]");return 2}
	mb,e:=readBounded(*mp);if e!=nil{fmt.Fprintln(os.Stderr,e);return 2};cb,e:=readBounded(*cp);if e!=nil{fmt.Fprintln(os.Stderr,e);return 2};md:=digest(mb)
	if e=verifyManifestPin(mb,*pin);e!=nil{fmt.Fprintln(os.Stderr,e);return 2};m,e:=decodeManifest(mb);if e!=nil{fmt.Fprintln(os.Stderr,e);return 2};c,e:=decodeCandidate(cb,m);if e!=nil{fmt.Fprintln(os.Stderr,e);return 2};r:=evaluate(m,c,md,digest(cb));b,e:=json.MarshalIndent(r,"","  ");if e!=nil{fmt.Fprintln(os.Stderr,e);return 2};b=append(b,'\n')
	if *out!=""{e=os.WriteFile(*out,b,0644)}else{_,e=os.Stdout.Write(b)};if e!=nil{fmt.Fprintln(os.Stderr,e);return 2};if r.Verdict!=qualified{return 3};return 0
}
func main(){os.Exit(run())}
