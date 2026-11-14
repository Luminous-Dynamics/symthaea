package join

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

const MaximumRequestBytes = 8 * 1024 * 1024

// DecodeStrict rejects oversized input, repeated JSON object keys, unknown
// struct fields, malformed JSON and multiple top-level values. It prevents
// ambiguous encodings from producing different receipts in different tools.
func DecodeStrict(data []byte, target any) error {
	if len(data) == 0 || len(data) > MaximumRequestBytes {
		return fmt.Errorf("%w: request is empty or exceeds %d bytes", ErrInvalidReceipt, MaximumRequestBytes)
	}
	duplicateScanner := json.NewDecoder(bytes.NewReader(data))
	if err := scanValue(duplicateScanner); err != nil {
		return fmt.Errorf("%w: duplicate-key scan: %v", ErrInvalidReceipt, err)
	}
	if _, err := duplicateScanner.Token(); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("%w: multiple top-level JSON values", ErrInvalidReceipt)
		}
		return fmt.Errorf("%w: trailing JSON: %v", ErrInvalidReceipt, err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("%w: decode: %v", ErrInvalidReceipt, err)
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("%w: multiple top-level JSON values", ErrInvalidReceipt)
		}
		return fmt.Errorf("%w: trailing JSON: %v", ErrInvalidReceipt, err)
	}
	return nil
}

func scanValue(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]struct{})
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return err
			}
			key, ok := keyToken.(string)
			if !ok {
				return fmt.Errorf("object key is not a string")
			}
			normalizedKey := strings.ToLower(key)
			if _, exists := seen[normalizedKey]; exists {
				return fmt.Errorf("duplicate or case-fold-equivalent object key %q", key)
			}
			seen[normalizedKey] = struct{}{}
			if err := scanValue(decoder); err != nil {
				return err
			}
		}
		end, err := decoder.Token()
		if err != nil {
			return err
		}
		if end != json.Delim('}') {
			return fmt.Errorf("malformed object terminator")
		}
	case '[':
		for decoder.More() {
			if err := scanValue(decoder); err != nil {
				return err
			}
		}
		end, err := decoder.Token()
		if err != nil {
			return err
		}
		if end != json.Delim(']') {
			return fmt.Errorf("malformed array terminator")
		}
	default:
		return fmt.Errorf("unexpected delimiter %q", delim)
	}
	return nil
}
