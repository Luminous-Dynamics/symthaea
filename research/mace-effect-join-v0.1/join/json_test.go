package join

import "testing"

func TestDecodeStrictRejectsDuplicateKeys(t *testing.T) {
	var target struct { A string `json:"a"` }
	if err := DecodeStrict([]byte(`{"a":"first","a":"second"}`), &target); err == nil { t.Fatal("duplicate JSON object key accepted") }
}
func TestDecodeStrictRejectsUnknownFields(t *testing.T) {
	var target struct { A string `json:"a"` }
	if err := DecodeStrict([]byte(`{"a":"ok","extra":true}`), &target); err == nil { t.Fatal("unknown JSON field accepted") }
}
func TestDecodeStrictRejectsTrailingValues(t *testing.T) {
	var target struct { A string `json:"a"` }
	if err := DecodeStrict([]byte(`{"a":"ok"} {"a":"second"}`), &target); err == nil { t.Fatal("multiple top-level values accepted") }
}
func TestDecodeStrictAcceptsNestedDistinctKeys(t *testing.T) {
	var target struct { A map[string]any `json:"a"` }
	if err := DecodeStrict([]byte(`{"a":{"x":1,"y":{"x":2}}}`), &target); err != nil { t.Fatal(err) }
}
func TestDecodeStrictRejectsEmptyAndOversized(t *testing.T) {
	var target map[string]any
	if err := DecodeStrict(nil, &target); err == nil { t.Fatal("empty input accepted") }
	if err := DecodeStrict(make([]byte, MaximumRequestBytes+1), &target); err == nil { t.Fatal("oversized input accepted") }
}
func TestDecodeStrictRejectsCaseFoldFieldAliases(t *testing.T) {
	var target struct { A string `json:"a"` }
	if err := DecodeStrict([]byte(`{"a":"first","A":"second"}`), &target); err == nil { t.Fatal("case-fold-equivalent JSON object keys accepted") }
}
