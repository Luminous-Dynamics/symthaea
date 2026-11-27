package benchmark

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

type monoInputs struct {
	matrix []byte
	oracle []byte
	workload []byte
	environment []byte
	budget []byte
}

func readMON025Inputs(t *testing.T) monoInputs {
	t.Helper()
	read:=func(relative string) []byte {
		raw,err:=os.ReadFile(filepath.Join("..",relative))
		if err!=nil { t.Fatalf("read %s: %v",relative,err) }
		return raw
	}
	matrix,err:=os.ReadFile(filepath.Join("..","..","sfeb-mon025","FAULT-MATRIX-v1.json"))
	if err!=nil { t.Fatalf("read actual MON-025 fault matrix: %v",err) }
	return monoInputs{
		matrix:matrix,
		oracle:read("ORACLE-v1.json"),
		workload:read("WORKLOAD-example.json"),
		environment:read("ENVIRONMENT-example.json"),
		budget:read("RESOURCE-BUDGET-example.json"),
	}
}

func buildActualCorpus(t *testing.T,in monoInputs) Suite {
	t.Helper()
	suite,err:=BuildSuiteFromMON025(in.matrix,in.oracle,in.workload,in.environment,in.budget,
		"mon025-to-mon028-test-campaign","mon-recovery-e2e-v1",7)
	if err!=nil { t.Fatal(err) }
	return suite
}

func TestBuildSuiteFromActualMON025Corpus(t *testing.T) {
	in:=readMON025Inputs(t)
	suite:=buildActualCorpus(t,in)
	if len(suite.Cases)!=41 {
		t.Fatalf("expected all 41 MON-025 seeds, got %d",len(suite.Cases))
	}
	if suite.FaultMatrixSHA256!=sha256Hex(in.matrix) ||
		suite.OracleSHA256!=sha256Hex(in.oracle) ||
		suite.WorkloadSHA256!=sha256Hex(in.workload) ||
		suite.EnvironmentSHA256!=sha256Hex(in.environment) ||
		suite.ResourceBudgetSHA256!=sha256Hex(in.budget) {
		t.Fatal("generated suite did not bind all raw source profiles")
	}
	planned:=0
	qualified,blocked,indeterminate:=0,0,0
	families:=map[string]bool{}
	for _,c:=range suite.Cases {
		planned+=int(c.Replicates)
		families[c.Family]=true
		switch c.Expected {
		case Qualified: qualified++
		case Blocked: blocked++
		case Indeterminate: indeterminate++
		}
	}
	if planned!=41*7 || qualified!=4 || blocked!=20 || indeterminate!=17 {
		t.Fatalf("unexpected denominator/verdicts planned=%d qualified=%d blocked=%d indeterminate=%d",
			planned,qualified,blocked,indeterminate)
	}
	if len(families)<5 {
		t.Fatalf("expected multiple fault families, got %d",len(families))
	}
	if _,err:=ComputeSuiteDigest(suite); err!=nil { t.Fatal(err) }
}

func TestBuildSuiteRejectsMissingOracleCase(t *testing.T) {
	in:=readMON025Inputs(t)
	var oracle OracleDocument
	if err:=decodeStrictJSON(in.oracle,&oracle); err!=nil { t.Fatal(err) }
	oracle.Entries=oracle.Entries[:len(oracle.Entries)-1]
	in.oracle,_=json.Marshal(oracle)
	_,err:=BuildSuiteFromMON025(in.matrix,in.oracle,in.workload,in.environment,in.budget,
		"missing-oracle","mon-recovery-e2e-v1",1)
	if !errors.Is(err,ErrOracleMismatch) {
		t.Fatalf("missing oracle entry must fail, got %v",err)
	}
}

func TestBuildSuiteRejectsExpectedOutcomeDisagreement(t *testing.T) {
	in:=readMON025Inputs(t)
	var matrix FaultMatrixDocument
	if err:=decodeStrictJSON(in.matrix,&matrix); err!=nil { t.Fatal(err) }
	matrix.Scenarios[0].ExpectedOutcome="blocked"
	in.matrix,_=json.Marshal(matrix)
	_,err:=BuildSuiteFromMON025(in.matrix,in.oracle,in.workload,in.environment,in.budget,
		"oracle-disagreement","mon-recovery-e2e-v1",1)
	if !errors.Is(err,ErrOracleMismatch) {
		t.Fatalf("matrix/oracle disagreement must fail, got %v",err)
	}
}

func TestBuildSuiteRejectsProfileSubstitution(t *testing.T) {
	in:=readMON025Inputs(t)
	in.workload=in.environment
	_,err:=BuildSuiteFromMON025(in.matrix,in.oracle,in.workload,in.environment,in.budget,
		"profile-substitution","mon-recovery-e2e-v1",1)
	if !errors.Is(err,ErrInvalidSuite) {
		t.Fatalf("profile substitution must fail, got %v",err)
	}
}

func TestBuildSuiteRejectsTrailingJSONInOracle(t *testing.T) {
	in:=readMON025Inputs(t)
	in.oracle=append(in.oracle,[]byte("{}")...)
	_,err:=BuildSuiteFromMON025(in.matrix,in.oracle,in.workload,in.environment,in.budget,
		"trailing-oracle","mon-recovery-e2e-v1",1)
	if err==nil {
		t.Fatal("trailing JSON values must be rejected")
	}
}
