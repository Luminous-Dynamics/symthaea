package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"luminousdynamics/mon028-benchmark"
)

func main() {
	suitePath:=flag.String("suite","","independent benchmark suite JSON")
	flag.Parse()
	if *suitePath=="" {
		fmt.Fprintln(os.Stderr,"--suite is required; see -h")
		os.Exit(64)
	}
	file,err:=os.Open(*suitePath)
	if err!=nil { fmt.Fprintf(os.Stderr,"open suite: %v\n",err); os.Exit(66) }
	defer file.Close()
	decoder:=json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var suite benchmark.Suite
	if err:=decoder.Decode(&suite); err!=nil { fmt.Fprintf(os.Stderr,"decode suite: %v\n",err); os.Exit(65) }
	var trailing any
	if err:=decoder.Decode(&trailing); err!=io.EOF { fmt.Fprintln(os.Stderr,"suite must contain exactly one JSON object"); os.Exit(65) }
	digest,err:=benchmark.ComputeSuiteDigest(suite)
	if err!=nil { fmt.Fprintf(os.Stderr,"validate suite: %v\n",err); os.Exit(2) }
	fmt.Println(digest)
}
