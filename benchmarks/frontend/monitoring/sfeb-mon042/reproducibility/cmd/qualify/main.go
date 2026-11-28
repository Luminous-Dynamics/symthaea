package main

import (
 "encoding/json"
 "errors"
 "flag"
 "fmt"
 "io"
 "os"
 reproducibility "luminousdynamics/mon042-reproducibility"
)
func main(){
 input:=flag.String("input","","MON-042 dependence-aware input JSON");flag.Parse()
 if *input==""{fail("--input is required",64)}
 f,err:=os.Open(*input);if err!=nil{fail("cannot open input: "+err.Error(),66)};defer f.Close()
 dec:=json.NewDecoder(io.LimitReader(f,(16<<20)+1));dec.DisallowUnknownFields()
 var in reproducibility.Input
 if err:=dec.Decode(&in);err!=nil{fail("invalid JSON: "+err.Error(),65)}
 var extra any;if err:=dec.Decode(&extra);!errors.Is(err,io.EOF){fail("input must contain exactly one JSON document",65)}
 out:=reproducibility.Evaluate(in)
 enc:=json.NewEncoder(os.Stdout);enc.SetIndent("","  ");if err:=enc.Encode(out);err!=nil{fail("cannot encode result: "+err.Error(),74)}
 switch out.StatisticalVerdict{case reproducibility.Qualified:os.Exit(0);case reproducibility.Blocked:os.Exit(2);default:os.Exit(3)}
}
func fail(s string,code int){fmt.Fprintln(os.Stderr,s);os.Exit(code)}
