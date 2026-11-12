// Package itt computes task-clustered intention-to-treat bounds for paired,
// randomized agent-evaluation studies. Post-treatment lifecycle stages are
// diagnostics, never eligibility filters for the primary estimates.
package itt

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
)

type TaskVerdict string
const (
	TaskPass TaskVerdict = "pass"
	TaskFail TaskVerdict = "fail"
	TaskIndeterminate TaskVerdict = "indeterminate"
)

type ControlVerdict string
const (
	ControlsPass ControlVerdict = "pass"
	ControlsNotPassed ControlVerdict = "not_passed"
	ControlsIndeterminate ControlVerdict = "indeterminate"
)

type UnauthorizedEffectVerdict string
const (
	UnauthorizedEffectAbsent UnauthorizedEffectVerdict = "absent"
	UnauthorizedEffectPresent UnauthorizedEffectVerdict = "present"
	UnauthorizedEffectIndeterminate UnauthorizedEffectVerdict = "indeterminate"
)

type StageFlags struct {
	Retrieved bool `json:"retrieved"`
	Selected bool `json:"selected"`
	Resolved bool `json:"resolved"`
	Loaded bool `json:"loaded"`
	Invoked bool `json:"invoked"`
	Executed bool `json:"executed"`
}

type Arm struct {
	RunID string `json:"run_id"`
	TaskVerdict TaskVerdict `json:"task_verdict"`
	ControlVerdict ControlVerdict `json:"control_verdict"`
	UnauthorizedEffect UnauthorizedEffectVerdict `json:"unauthorized_effect"`
	Stages StageFlags `json:"stages,omitempty"`
}

type Pair struct {
	PairID string `json:"pair_id"`
	ExperimentID string `json:"experiment_id"`
	TaskID string `json:"task_id"`
	TaskRevision string `json:"task_revision"`
	Domain string `json:"domain"`
	Seed string `json:"seed"`
	StratumDigest string `json:"stratum_digest"`
	ArmOrder string `json:"arm_order"`
	WithSkill Arm `json:"with_skill"`
	WithoutSkill Arm `json:"without_skill"`
}

type Interval struct { Low float64 `json:"low"`; High float64 `json:"high"` }

type Estimate struct {
	TaskClusters int `json:"task_clusters"`
	PairCount int `json:"pair_count"`
	CompletePairs int `json:"complete_pairs"`
	IndeterminateArms int `json:"indeterminate_arms"`
	LowerBound float64 `json:"lower_bound"`
	UpperBound float64 `json:"upper_bound"`
	LowerBoundCI *Interval `json:"lower_bound_ci_95"`
	UpperBoundCI *Interval `json:"upper_bound_ci_95"`
	ConfidenceIntervalsAvailable bool `json:"confidence_intervals_available"`
}

type EndpointEstimates struct {
	TaskSuccess Estimate `json:"task_success"`
	ControlPass Estimate `json:"control_pass"`
	CleanSuccess Estimate `json:"clean_success"`
	UnauthorizedEffect Estimate `json:"unauthorized_effect_incidence"`
}
type DomainEstimate struct { Domain string `json:"domain"`; EndpointEstimates }

type OrderDiagnostics struct {
	WithSkillFirst int `json:"with_skill_first"`
	WithoutSkillFirst int `json:"without_skill_first"`
}
type RetrievalDiagnostics struct {
	TreatedPairs int `json:"treated_pairs"`
	Retrieved int `json:"retrieved"`
	Selected int `json:"selected"`
	Resolved int `json:"resolved"`
	Loaded int `json:"loaded"`
	Invoked int `json:"invoked"`
	Executed int `json:"executed"`
}

type Report struct {
	Estimand string `json:"estimand"`
	AssignmentRosterMatched bool `json:"assignment_roster_matched"`
	ExperimentID string `json:"experiment_id"`
	StratumDigest string `json:"stratum_digest"`
	ArmOrderDiagnostics OrderDiagnostics `json:"arm_order_diagnostics"`
	Primary EndpointEstimates `json:"primary_task_macro_itt"`
	Domains []DomainEstimate `json:"domains"`
	RetrievalDiagnostics RetrievalDiagnostics `json:"retrieval_diagnostics"`
	BootstrapReplicates int `json:"bootstrap_replicates"`
	BootstrapSeed int64 `json:"bootstrap_seed"`
	MissingnessPolicy string `json:"missingness_policy"`
	EndpointDirections map[string]string `json:"endpoint_directions"`
}
type Config struct {
	BootstrapReplicates int
	BootstrapSeed int64
	ExpectedPairIDs []string
	RequireAssignmentRoster bool
}

var (
	ErrEmptyInput = errors.New("no randomized task pairs")
	ErrInvalidInput = errors.New("invalid paired-run input")
)

type endpoint int
const (
	endpointTask endpoint = iota
	endpointControl
	endpointClean
	endpointUnauthorized
	endpointCount
)
type pairEffect struct { low, high float64; complete bool; indeterminateArms int }
type taskEffect struct { taskID, domain string; effects [endpointCount]pairEffect; pairs int }

// EstimateITT computes four independent paired ITT endpoint effects across
// every preregistered assignment. Task clusters receive equal weight after
// repeated seeds are averaged within task. Unknown outcomes remain [0,1].
func EstimateITT(pairs []Pair, cfg Config) (Report, error) {
	if len(pairs) == 0 { return Report{}, ErrEmptyInput }
	if cfg.BootstrapReplicates <= 0 { cfg.BootstrapReplicates = 2000 }
	if cfg.BootstrapReplicates < 200 { return Report{}, fmt.Errorf("%w: at least 200 bootstrap replicates required", ErrInvalidInput) }
	if cfg.RequireAssignmentRoster && len(cfg.ExpectedPairIDs) == 0 { return Report{}, fmt.Errorf("%w: assignment roster is required", ErrInvalidInput) }

	seenPair, seenRun, seenCell := map[string]bool{}, map[string]bool{}, map[string]bool{}
	seenRoster := map[string]bool{}
	for _, id := range cfg.ExpectedPairIDs {
		if strings.TrimSpace(id) == "" || seenRoster[id] { return Report{}, fmt.Errorf("%w: empty or duplicate assignment roster ID %q", ErrInvalidInput, id) }
		seenRoster[id] = true
	}
	byTask := map[string][][endpointCount]pairEffect{}
	taskDomain, taskRevision := map[string]string{}, map[string]string{}
	experimentID, stratumDigest := "", ""
	orderDiagnostics := OrderDiagnostics{}
	retrievalDiagnostics := RetrievalDiagnostics{}

	for i, p := range pairs {
		prefix := fmt.Sprintf("pair[%d]", i)
		if strings.TrimSpace(p.PairID)=="" || strings.TrimSpace(p.ExperimentID)=="" || strings.TrimSpace(p.TaskID)=="" || strings.TrimSpace(p.TaskRevision)=="" || strings.TrimSpace(p.Domain)=="" || strings.TrimSpace(p.Seed)=="" || strings.TrimSpace(p.StratumDigest)=="" {
			return Report{}, fmt.Errorf("%w: %s has an empty pair/experiment/task/revision/domain/seed/stratum field", ErrInvalidInput, prefix)
		}
		if len(p.StratumDigest)!=64 || !isHexDigest(p.StratumDigest) { return Report{}, fmt.Errorf("%w: %s stratum_digest must be 64 hexadecimal characters",ErrInvalidInput,prefix) }
		if experimentID=="" { experimentID=p.ExperimentID } else if experimentID!=p.ExperimentID { return Report{},fmt.Errorf("%w: mixed experiment IDs",ErrInvalidInput) }
		if stratumDigest=="" { stratumDigest=p.StratumDigest } else if stratumDigest!=p.StratumDigest { return Report{},fmt.Errorf("%w: mixed configuration strata in one estimate",ErrInvalidInput) }
		if old,ok:=taskDomain[p.TaskID];ok&&old!=p.Domain{return Report{},fmt.Errorf("%w: task %q mapped to multiple domains",ErrInvalidInput,p.TaskID)}
		if old,ok:=taskRevision[p.TaskID];ok&&old!=p.TaskRevision{return Report{},fmt.Errorf("%w: task %q mapped to multiple revisions",ErrInvalidInput,p.TaskID)}
		taskDomain[p.TaskID],taskRevision[p.TaskID]=p.Domain,p.TaskRevision
		if len(seenRoster)>0&&!seenRoster[p.PairID]{return Report{},fmt.Errorf("%w: pair %q is absent from assignment roster",ErrInvalidInput,p.PairID)}
		if seenPair[p.PairID]{return Report{},fmt.Errorf("%w: duplicate pair_id %q",ErrInvalidInput,p.PairID)}
		seenPair[p.PairID]=true
		cell:=p.TaskID+"\x00"+p.Seed
		if seenCell[cell]{return Report{},fmt.Errorf("%w: duplicate task/seed pair %q/%q",ErrInvalidInput,p.TaskID,p.Seed)}
		seenCell[cell]=true
		if strings.TrimSpace(p.WithSkill.RunID)==""||strings.TrimSpace(p.WithoutSkill.RunID)==""{return Report{},fmt.Errorf("%w: %s has empty run ID",ErrInvalidInput,prefix)}
		if p.WithSkill.RunID==p.WithoutSkill.RunID||seenRun[p.WithSkill.RunID]||seenRun[p.WithoutSkill.RunID]{return Report{},fmt.Errorf("%w: duplicate run ID",ErrInvalidInput)}
		seenRun[p.WithSkill.RunID],seenRun[p.WithoutSkill.RunID]=true,true
		if err:=validateArm(p.WithSkill);err!=nil{return Report{},fmt.Errorf("%s with_skill: %w",prefix,err)}
		if err:=validateArm(p.WithoutSkill);err!=nil{return Report{},fmt.Errorf("%s without_skill: %w",prefix,err)}
		if p.WithoutSkill.Stages!=(StageFlags{}){return Report{},fmt.Errorf("%w: no-skill arm cannot contain skill lifecycle stages",ErrInvalidInput)}
		switch p.ArmOrder {
		case "with_skill_first": orderDiagnostics.WithSkillFirst++
		case "without_skill_first": orderDiagnostics.WithoutSkillFirst++
		default: return Report{},fmt.Errorf("%w: invalid arm_order %q",ErrInvalidInput,p.ArmOrder)
		}
		retrievalDiagnostics.TreatedPairs++
		if p.WithSkill.Stages.Retrieved{retrievalDiagnostics.Retrieved++}
		if p.WithSkill.Stages.Selected{retrievalDiagnostics.Selected++}
		if p.WithSkill.Stages.Resolved{retrievalDiagnostics.Resolved++}
		if p.WithSkill.Stages.Loaded{retrievalDiagnostics.Loaded++}
		if p.WithSkill.Stages.Invoked{retrievalDiagnostics.Invoked++}
		if p.WithSkill.Stages.Executed{retrievalDiagnostics.Executed++}
		effects:=[endpointCount]pairEffect{
			pairedDifference(taskSuccessBounds(p.WithSkill.TaskVerdict),taskSuccessBounds(p.WithoutSkill.TaskVerdict)),
			pairedDifference(controlPassBounds(p.WithSkill.ControlVerdict),controlPassBounds(p.WithoutSkill.ControlVerdict)),
			pairedDifference(cleanSuccessBounds(p.WithSkill),cleanSuccessBounds(p.WithoutSkill)),
			pairedDifference(unauthorizedEffectBounds(p.WithSkill.UnauthorizedEffect),unauthorizedEffectBounds(p.WithoutSkill.UnauthorizedEffect)),
		}
		byTask[p.TaskID]=append(byTask[p.TaskID],effects)
	}
	if len(seenRoster)>0 {
		if len(seenRoster)!=len(seenPair){return Report{},fmt.Errorf("%w: observed %d pairs but assignment roster declares %d",ErrInvalidInput,len(seenPair),len(seenRoster))}
		for id:=range seenRoster{if !seenPair[id]{return Report{},fmt.Errorf("%w: assignment roster pair %q is missing",ErrInvalidInput,id)}}
	}

	tasks:=make([]taskEffect,0,len(byTask))
	for taskID,repeated:=range byTask{
		t:=taskEffect{taskID:taskID,domain:taskDomain[taskID],pairs:len(repeated)}
		for e:=endpoint(0);e<endpointCount;e++{
			var low,high float64
			complete,unknownArms:=0,0
			for _,pair:=range repeated{low+=pair[e].low;high+=pair[e].high;if pair[e].complete{complete++};unknownArms+=pair[e].indeterminateArms}
			n:=float64(len(repeated))
			t.effects[e]=pairEffect{low:low/n,high:high/n,complete:complete==len(repeated),indeterminateArms:unknownArms}
		}
		tasks=append(tasks,t)
	}
	sort.Slice(tasks,func(i,j int)bool{return tasks[i].taskID<tasks[j].taskID})
	primary,err:=summarizeEndpoints(tasks,cfg);if err!=nil{return Report{},err}
	byDomain:=map[string][]taskEffect{}
	for _,t:=range tasks{byDomain[t.domain]=append(byDomain[t.domain],t)}
	domainNames:=make([]string,0,len(byDomain));for d:=range byDomain{domainNames=append(domainNames,d)};sort.Strings(domainNames)
	domainReports:=make([]DomainEstimate,0,len(domainNames))
	for _,d:=range domainNames{e,err:=summarizeEndpoints(byDomain[d],cfg);if err!=nil{return Report{},err};domainReports=append(domainReports,DomainEstimate{Domain:d,EndpointEstimates:e})}

	return Report{
		Estimand:"paired task-macro intention-to-treat effect of skill availability (with-skill minus without-skill), over every assignment in the supplied roster",
		AssignmentRosterMatched:len(seenRoster)>0,ExperimentID:experimentID,StratumDigest:stratumDigest,
		ArmOrderDiagnostics:orderDiagnostics,Primary:primary,Domains:domainReports,RetrievalDiagnostics:retrievalDiagnostics,
		BootstrapReplicates:cfg.BootstrapReplicates,BootstrapSeed:cfg.BootstrapSeed,
		MissingnessPolicy:"binary indeterminate outcomes remain [0,1]; every supplied assignment is retained; pair-level lower/upper differences are propagated through task-macro aggregation; roster matching checks IDs only",
		EndpointDirections:map[string]string{
			"task_success":"positive is improved task success",
			"control_pass":"positive is improved mandatory-control passage",
			"clean_success":"positive is improved task-pass AND controls-pass rate",
			"unauthorized_effect_incidence":"positive is worse; it indicates increased unauthorized-effect incidence",
		},
	},nil
}

func summarizeEndpoints(tasks []taskEffect,cfg Config)(EndpointEstimates,error){
	var out EndpointEstimates
	assign:=[]struct{ep endpoint;dst *Estimate}{
		{endpointTask,&out.TaskSuccess},{endpointControl,&out.ControlPass},{endpointClean,&out.CleanSuccess},{endpointUnauthorized,&out.UnauthorizedEffect},
	}
	for _,x:=range assign{e,err:=summarize(tasks,x.ep,cfg);if err!=nil{return EndpointEstimates{},err};*x.dst=e}
	return out,nil
}
func summarize(tasks []taskEffect,ep endpoint,cfg Config)(Estimate,error){
	if len(tasks)==0{return Estimate{},ErrEmptyInput}
	var low,high float64
	pairCount,completePairs,indeterminateArms:=0,0,0
	for _,t:=range tasks{low+=t.effects[ep].low;high+=t.effects[ep].high;pairCount+=t.pairs;if t.effects[ep].complete{completePairs+=t.pairs};indeterminateArms+=t.effects[ep].indeterminateArms}
	n:=float64(len(tasks))
	out:=Estimate{TaskClusters:len(tasks),PairCount:pairCount,CompletePairs:completePairs,IndeterminateArms:indeterminateArms,LowerBound:low/n,UpperBound:high/n}
	if len(tasks)<2{return out,nil}
	rng:=rand.New(rand.NewSource(cfg.BootstrapSeed+int64(ep)*7919))
	bootLow,bootHigh:=make([]float64,cfg.BootstrapReplicates),make([]float64,cfg.BootstrapReplicates)
	for b:=0;b<cfg.BootstrapReplicates;b++{
		var l,h float64
		for i:=0;i<len(tasks);i++{t:=tasks[rng.Intn(len(tasks))];l+=t.effects[ep].low;h+=t.effects[ep].high}
		bootLow[b],bootHigh[b]=l/n,h/n
	}
	lowerCI,upperCI:=percentileInterval(bootLow),percentileInterval(bootHigh)
	out.LowerBoundCI,out.UpperBoundCI=&lowerCI,&upperCI
	out.ConfidenceIntervalsAvailable=true
	return out,nil
}
func pairedDifference(treated,control [3]float64)pairEffect{
	return pairEffect{low:treated[0]-control[1],high:treated[1]-control[0],complete:treated[2]==1&&control[2]==1,indeterminateArms:int(1-treated[2])+int(1-control[2])}
}
func taskSuccessBounds(v TaskVerdict)[3]float64{switch v{case TaskPass:return [3]float64{1,1,1};case TaskFail:return [3]float64{0,0,1};default:return [3]float64{0,1,0}}}
func controlPassBounds(v ControlVerdict)[3]float64{switch v{case ControlsPass:return [3]float64{1,1,1};case ControlsNotPassed:return [3]float64{0,0,1};default:return [3]float64{0,1,0}}}
func cleanSuccessBounds(a Arm)[3]float64{
	if a.TaskVerdict==TaskFail||a.ControlVerdict==ControlsNotPassed{return [3]float64{0,0,1}}
	if a.TaskVerdict==TaskPass&&a.ControlVerdict==ControlsPass{return [3]float64{1,1,1}}
	return [3]float64{0,1,0}
}
func unauthorizedEffectBounds(v UnauthorizedEffectVerdict)[3]float64{switch v{case UnauthorizedEffectPresent:return [3]float64{1,1,1};case UnauthorizedEffectAbsent:return [3]float64{0,0,1};default:return [3]float64{0,1,0}}}
func validateArm(a Arm)error{
	if strings.TrimSpace(a.RunID)==""{return fmt.Errorf("%w: empty run ID",ErrInvalidInput)}
	switch a.TaskVerdict{case TaskPass,TaskFail,TaskIndeterminate:default:return fmt.Errorf("%w: unknown task verdict %q",ErrInvalidInput,a.TaskVerdict)}
	switch a.ControlVerdict{case ControlsPass,ControlsNotPassed,ControlsIndeterminate:default:return fmt.Errorf("%w: unknown control verdict %q",ErrInvalidInput,a.ControlVerdict)}
	switch a.UnauthorizedEffect{case UnauthorizedEffectAbsent,UnauthorizedEffectPresent,UnauthorizedEffectIndeterminate:default:return fmt.Errorf("%w: unknown unauthorized-effect verdict %q",ErrInvalidInput,a.UnauthorizedEffect)}
	if a.Stages.Selected&&!a.Stages.Retrieved||a.Stages.Resolved&&!a.Stages.Selected||a.Stages.Loaded&&!a.Stages.Resolved||a.Stages.Invoked&&!a.Stages.Loaded||a.Stages.Executed&&!a.Stages.Invoked{return fmt.Errorf("%w: lifecycle stages must be monotone retrieved→selected→resolved→loaded→invoked→executed",ErrInvalidInput)}
	return nil
}
func isHexDigest(s string)bool{for _,c:=range s{if !(c>='0'&&c<='9')&&!(c>='a'&&c<='f')&&!(c>='A'&&c<='F'){return false}};return true}
func percentileInterval(v []float64)Interval{sort.Float64s(v);return Interval{Low:quantile(v,.025),High:quantile(v,.975)}}
func quantile(sorted []float64,p float64)float64{if len(sorted)==0{return math.NaN()};x:=p*float64(len(sorted)-1);i:=int(math.Floor(x));f:=x-float64(i);if i+1>=len(sorted){return sorted[i]};return sorted[i]*(1-f)+sorted[i+1]*f}
