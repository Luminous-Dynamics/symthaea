// Package targetverify is a second-language reference implementation for
// SYMPsi AIA target-mapping bytes. It is a test oracle, not a beacon verifier.
package targetverify

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
)

const domain = "sympsi.aia.target.v1\x00"

type Context struct {
	ProtocolID         string
	CampaignID         string
	TrialID            string
	ChainHash          [32]byte
	SchemeID           string
	ResponseCommitment [32]byte
	Round              uint64
	VerifiedRandomness [32]byte
}

// Digest constructs the exact domain-separated transcript used by the Rust
// implementation. "VerifiedRandomness" is an input from a separate verifier;
// this package does not authenticate a drand signature or establish chronology.
func Digest(context Context, counter uint32) ([32]byte, error) {
	var zero [32]byte
	if context.ProtocolID == "" || context.CampaignID == "" ||
		context.TrialID == "" || context.SchemeID == "" || context.Round == 0 {
		return zero, errors.New("protocol, campaign, trial, scheme and nonzero round are required")
	}

	payload := make([]byte, 0, 256)
	payload = append(payload, domain...)
	fields := [][]byte{
		[]byte(context.ProtocolID),
		[]byte(context.CampaignID),
		[]byte(context.TrialID),
		context.ChainHash[:],
		[]byte(context.SchemeID),
	}
	for _, field := range fields {
		var err error
		payload, err = appendLengthPrefixed(payload, field)
		if err != nil {
			return zero, err
		}
	}
	var roundBytes [8]byte
	binary.BigEndian.PutUint64(roundBytes[:], context.Round)
	payload = append(payload, roundBytes[:]...)

	payload, err := appendLengthPrefixed(payload, context.ResponseCommitment[:])
	if err != nil {
		return zero, err
	}
	payload, err = appendLengthPrefixed(payload, context.VerifiedRandomness[:])
	if err != nil {
		return zero, err
	}
	var counterBytes [4]byte
	binary.BigEndian.PutUint32(counterBytes[:], counter)
	payload = append(payload, counterBytes[:]...)
	return sha256.Sum256(payload), nil
}

func appendLengthPrefixed(output []byte, field []byte) ([]byte, error) {
	if uint64(len(field)) > uint64(^uint32(0)) {
		return nil, errors.New("field length exceeds uint32")
	}
	var length [4]byte
	binary.BigEndian.PutUint32(length[:], uint32(len(field)))
	output = append(output, length[:]...)
	output = append(output, field...)
	return output, nil
}

// TargetIndex performs unbiased rejection sampling for exactly four or five
// outcomes. It assumes context.VerifiedRandomness has already been verified
// by the separate, pinned beacon-verification layer.
func TargetIndex(choiceCount int, context Context) (int, error) {
	if choiceCount != 4 && choiceCount != 5 {
		return 0, fmt.Errorf("unsupported choice count: %d", choiceCount)
	}
	threshold := powMod256(choiceCount)
	for counter := uint32(0); ; counter++ {
		digest, err := Digest(context, counter)
		if err != nil {
			return 0, err
		}
		if !isBelowSmallInteger(digest, threshold) {
			return digestModulo(digest, choiceCount), nil
		}
		if counter == ^uint32(0) {
			return 0, errors.New("target mapping rejection counter exhausted")
		}
	}
}

func powMod256(modulus int) int {
	result := 1 % modulus
	for i := 0; i < 256; i++ {
		result = (result * 2) % modulus
	}
	return result
}

func isBelowSmallInteger(digest [32]byte, threshold int) bool {
	for _, value := range digest[:31] {
		if value != 0 {
			return false
		}
	}
	return int(digest[31]) < threshold
}

func digestModulo(digest [32]byte, modulus int) int {
	remainder := 0
	for _, value := range digest {
		remainder = (remainder*256 + int(value)) % modulus
	}
	return remainder
}
