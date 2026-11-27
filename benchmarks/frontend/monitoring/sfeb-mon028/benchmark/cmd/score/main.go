package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"luminousdynamics/mon028-benchmark"
)

func main() { os.Exit(run()) }

func decodeStrictFile[T any](path string) (T,error) {
	var zero T
	file,err:=os.Open(path)
	if err!=nil { return zero,err }
	defer file.Close()
	decoder:=json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var value T
	if err:=decoder.Decode(&value); err!=nil { return zero,err }
	var trailing any
	if err:=decoder.Decode(&trailing); err!=io.EOF { return zero,errors.New("input must contain exactly one JSON value") }
	return value,nil
}

func run() int {
	suitePath:=flag.String("suite","","independent signed-suite JSON")
	resultPath:=flag.String("result","","candidate campaign result JSON")
	baselinePath:=flag.String("baseline","","optional matched baseline result JSON")
	flag.Parse()
	if *suitePath=="" || *resultPath=="" {
		fmt.Fprintln(os.Stderr,"--suite and --result are required; see -h")
		return 64
	}
	suite,err:=decodeStrictFile[benchmark.Suite](*suitePath)
	if err!=nil { fmt.Fprintf(os.Stderr,"read suite: %v\n",err); return 65 }
	result,err:=decodeStrictFile[benchmark.CampaignResult](*resultPath)
	if err!=nil { fmt.Fprintf(os.Stderr,"read result: %v\n",err); return 65 }
	score,err:=benchmark.ScoreRun(suite,result)
	if err!=nil { fmt.Fprintf(os.Stderr,"score result: %v\n",err); return 2 }

	output:=struct {
		Score benchmark.Score `json:"score"`
		Comparison *benchmark.Comparison `json:"comparison,omitempty"`
	}{Score:score}
	if *baselinePath!="" {
		baseline,readErr:=decodeStrictFile[benchmark.CampaignResult](*baselinePath)
		if readErr!=nil { fmt.Fprintf(os.Stderr,"read baseline: %v\n",readErr); return 65 }
		comparison,compareErr:=benchmark.CompareMatched(suite,baseline,result)
		if compareErr!=nil { fmt.Fprintf(os.Stderr,"matched comparison: %v\n",compareErr); return 2 }
		output.Comparison=&comparison
	}
	if err:=json.NewEncoder(os.Stdout).Encode(output); err!=nil {
		fmt.Fprintf(os.Stderr,"encode score: %v\n",err); return 70
	}
	if !score.Complete { return 3 }
	if !score.SafetyGate { return 2 }
	return 0
}
