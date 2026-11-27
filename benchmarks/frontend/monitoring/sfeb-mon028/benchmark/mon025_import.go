package benchmark

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

const (
	OracleProfile = "mon028-independent-ground-truth-oracle-v1"
	WorkloadProfile = "mon028-workload-profile-v1"
	EnvironmentProfile = "mon028-environment-profile-v1"
	ResourceBudgetProfile = "mon028-resource-budget-profile-v1"
)

type FaultMatrixDocument struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	Classification string `json:"classification"`
	Scenarios []FaultScenario `json:"scenarios"`
}

type FaultScenario struct {
	ID string `json:"id"`
	Class string `json:"class"`
	Name string `json:"name"`
	ExpectedOutcome string `json:"expected_outcome"`
}

type OracleDocument struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	SourceFaultMatrixProfile string `json:"source_fault_matrix_profile"`
	Authority string `json:"authority"`
	Entries []OracleEntry `json:"entries"`
}

type OracleEntry struct {
	CaseID string `json:"case_id"`
	FaultID string `json:"fault_id"`
	Family string `json:"family"`
	Expected Verdict `json:"expected"`
	Rationale string `json:"rationale"`
}

var ErrOracleMismatch = errors.New("fault matrix and independent oracle do not match")

// BuildSuiteFromMON025 joins a raw MON-025 fault matrix with a separate,
// independently maintained oracle. The matrix's expected_outcome is treated
// as a consistency assertion only; the suite verdict is sourced from Oracle.
func BuildSuiteFromMON025(
	faultMatrixJSON []byte,
	oracleJSON []byte,
	workloadJSON []byte,
	environmentJSON []byte,
	resourceBudgetJSON []byte,
	campaignID string,
	benchmarkID string,
	replicates uint32,
) (Suite, error) {
	if strings.TrimSpace(campaignID)=="" || strings.TrimSpace(benchmarkID)=="" || replicates==0 {
		return Suite{},ErrInvalidSuite
	}
	var matrix FaultMatrixDocument
	if err:=decodeStrictJSON(faultMatrixJSON,&matrix); err!=nil { return Suite{},fmt.Errorf("decode fault matrix: %w",err) }
	if matrix.Profile=="" || matrix.Version=="" || matrix.Classification=="" || len(matrix.Scenarios)==0 {
		return Suite{},fmt.Errorf("%w: fault matrix metadata incomplete",ErrInvalidSuite)
	}
	var oracle OracleDocument
	if err:=decodeStrictJSON(oracleJSON,&oracle); err!=nil { return Suite{},fmt.Errorf("decode oracle: %w",err) }
	if oracle.Profile!=OracleProfile || oracle.Version!=Version ||
		oracle.SourceFaultMatrixProfile!=matrix.Profile || strings.TrimSpace(oracle.Authority)=="" || len(oracle.Entries)==0 {
		return Suite{},fmt.Errorf("%w: oracle profile/source binding invalid",ErrInvalidSuite)
	}
	profiles:=[]struct{name string; raw []byte; want string}{
		{"workload",workloadJSON,WorkloadProfile},
		{"environment",environmentJSON,EnvironmentProfile},
		{"resource budget",resourceBudgetJSON,ResourceBudgetProfile},
	}
	for _,p:=range profiles {
		profile,err:=readProfileID(p.raw)
		if err!=nil { return Suite{},fmt.Errorf("decode %s profile: %w",p.name,err) }
		if profile!=p.want { return Suite{},fmt.Errorf("%w: unexpected %s profile %q",ErrInvalidSuite,p.name,profile) }
	}

	oracleByID:=make(map[string]OracleEntry,len(oracle.Entries))
	for _,entry:=range oracle.Entries {
		if entry.CaseID=="" || entry.FaultID=="" || entry.Family=="" ||
			!validVerdict(entry.Expected) || strings.TrimSpace(entry.Rationale)=="" {
			return Suite{},fmt.Errorf("%w: oracle entry invalid",ErrInvalidSuite)
		}
		if _,exists:=oracleByID[entry.CaseID]; exists {
			return Suite{},fmt.Errorf("%w: duplicate oracle case %q",ErrOracleMismatch,entry.CaseID)
		}
		oracleByID[entry.CaseID]=entry
	}
	seen:=make(map[string]struct{},len(matrix.Scenarios))
	cases:=make([]CaseSpec,0,len(matrix.Scenarios))
	for _,scenario:=range matrix.Scenarios {
		if scenario.ID=="" {
			return Suite{},fmt.Errorf("%w: matrix case ID missing",ErrInvalidSuite)
		}
		if _,exists:=seen[scenario.ID]; exists {
			return Suite{},fmt.Errorf("%w: duplicate matrix case %q",ErrOracleMismatch,scenario.ID)
		}
		seen[scenario.ID]=struct{}{}
		entry,ok:=oracleByID[scenario.ID]
		if !ok { return Suite{},fmt.Errorf("%w: oracle missing case %q",ErrOracleMismatch,scenario.ID) }
		if entry.FaultID!=scenario.ID || string(entry.Expected)!=scenario.ExpectedOutcome {
			return Suite{},fmt.Errorf("%w: expected outcome or fault identity disagrees for %s",ErrOracleMismatch,scenario.ID)
		}
		if scenario.Class=="positive_control" && entry.Expected!=Qualified {
			return Suite{},fmt.Errorf("%w: positive control %s is not oracle-qualified",ErrOracleMismatch,scenario.ID)
		}
		if scenario.Class!="positive_control" && entry.Expected==Qualified {
			return Suite{},fmt.Errorf("%w: unsafe control %s is oracle-qualified",ErrOracleMismatch,scenario.ID)
		}
		cases=append(cases,CaseSpec{
			CaseID:scenario.ID,FaultID:entry.FaultID,Family:entry.Family,
			Expected:entry.Expected,Replicates:replicates,
		})
	}
	if len(seen)!=len(oracleByID) {
		return Suite{},fmt.Errorf("%w: oracle contains %d cases but matrix contains %d",ErrOracleMismatch,len(oracleByID),len(seen))
	}
	suite:=Suite{
		Profile:SuiteProfile,Version:Version,
		CampaignID:campaignID,BenchmarkID:benchmarkID,
		FaultMatrixSHA256:sha256Hex(faultMatrixJSON),
		OracleSHA256:sha256Hex(oracleJSON),
		WorkloadSHA256:sha256Hex(workloadJSON),
		EnvironmentSHA256:sha256Hex(environmentJSON),
		ResourceBudgetSHA256:sha256Hex(resourceBudgetJSON),
		Cases:cases,
	}
	if err:=ValidateSuite(suite); err!=nil { return Suite{},err }
	return suite,nil
}

func decodeStrictJSON(raw []byte,destination any) error {
	decoder:=json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err:=decoder.Decode(destination); err!=nil { return err }
	var trailing any
	if err:=decoder.Decode(&trailing); err!=io.EOF {
		return errors.New("expected exactly one JSON value")
	}
	return nil
}

func readProfileID(raw []byte) (string,error) {
	var profile struct {
		Profile string `json:"profile"`
		Version string `json:"version"`
	}
	if err:=decodeStrictJSON(raw,&profile); err!=nil { return "",err }
	if profile.Profile=="" || profile.Version=="" { return "",ErrInvalidSuite }
	return profile.Profile,nil
}

func sha256Hex(raw []byte) string {
	sum:=sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}
