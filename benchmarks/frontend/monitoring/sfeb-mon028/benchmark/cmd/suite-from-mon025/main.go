package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"luminousdynamics/mon028-benchmark"
)

func main() { os.Exit(run()) }

func run() int {
	matrixPath:=flag.String("fault-matrix","","raw MON-025 FAULT-MATRIX-v1.json")
	oraclePath:=flag.String("oracle","","independently maintained MON-028 ORACLE-v1.json")
	workloadPath:=flag.String("workload-profile","","frozen workload profile JSON")
	environmentPath:=flag.String("environment-profile","","frozen environment profile JSON")
	budgetPath:=flag.String("resource-budget-profile","","enforced resource budget profile JSON")
	campaignID:=flag.String("campaign-id","","unique benchmark campaign ID")
	benchmarkID:=flag.String("benchmark-id","mon-recovery-e2e-v1","benchmark definition ID")
	replicates:=flag.Uint("replicates",30,"replicate count per fault case")
	outputPath:=flag.String("output","","generated suite JSON output path")
	flag.Parse()

	if *matrixPath=="" || *oraclePath=="" || *workloadPath=="" ||
		*environmentPath=="" || *budgetPath=="" || *campaignID=="" || *outputPath=="" ||
		*replicates==0 || uint64(*replicates)>uint64(^uint32(0)) {
		fmt.Fprintln(os.Stderr,"all input profiles, campaign ID, positive replicate count and --output are required; see -h")
		return 64
	}
	read:=func(name,path string) []byte {
		raw,err:=os.ReadFile(path)
		if err!=nil { fmt.Fprintf(os.Stderr,"read %s: %v\n",name,err); return nil }
		return raw
	}
	matrix:=read("fault matrix",*matrixPath)
	if matrix==nil { return 66 }
	oracle:=read("oracle",*oraclePath)
	if oracle==nil { return 66 }
	workload:=read("workload profile",*workloadPath)
	if workload==nil { return 66 }
	environment:=read("environment profile",*environmentPath)
	if environment==nil { return 66 }
	budget:=read("resource budget profile",*budgetPath)
	if budget==nil { return 66 }

	suite,err:=benchmark.BuildSuiteFromMON025(
		matrix,oracle,workload,environment,budget,
		*campaignID,*benchmarkID,uint32(*replicates),
	)
	if err!=nil {
		fmt.Fprintf(os.Stderr,"suite generation rejected: %v\n",err)
		return 2
	}
	digest,err:=benchmark.ComputeSuiteDigest(suite)
	if err!=nil { fmt.Fprintf(os.Stderr,"suite digest: %v\n",err); return 2 }
	encoded,err:=json.MarshalIndent(suite,"","  ")
	if err!=nil { fmt.Fprintf(os.Stderr,"encode suite: %v\n",err); return 70 }
	file,err:=os.OpenFile(*outputPath,os.O_WRONLY|os.O_CREATE|os.O_EXCL,0o600)
	if err!=nil { fmt.Fprintf(os.Stderr,"create suite output: %v\n",err); return 73 }
	if _,err:=file.Write(append(encoded,'\n')); err!=nil {
		_ = file.Close()
		_ = os.Remove(*outputPath)
		fmt.Fprintf(os.Stderr,"write suite output: %v\n",err)
		return 73
	}
	if err:=file.Sync(); err!=nil {
		_ = file.Close()
		_ = os.Remove(*outputPath)
		fmt.Fprintf(os.Stderr,"sync suite output: %v\n",err)
		return 73
	}
	if err:=file.Close(); err!=nil { _ = os.Remove(*outputPath); fmt.Fprintf(os.Stderr,"close suite output: %v\n",err); return 73 }
	fmt.Printf("suite_sha256=%s\n",digest)
	fmt.Printf("fault_cases=%d\n",len(suite.Cases))
	fmt.Printf("planned_observations=%d\n",len(suite.Cases)*int(*replicates))
	fmt.Println("The suite is unsigned. Package it through MON-027 and obtain a separate benchmark-authority signature before scoring.")
	return 0
}
