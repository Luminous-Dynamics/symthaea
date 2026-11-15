package targetverify

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

func mustDecode32(t *testing.T, value string) [32]byte {
	t.Helper()
	var out [32]byte
	decoded, err := hex.DecodeString(value)
	if err != nil {
		t.Fatalf("decode hex: %v", err)
	}
	if len(decoded) != 32 {
		t.Fatalf("expected 32 bytes, got %d", len(decoded))
	}
	copy(out[:], decoded)
	return out
}

func quicknetExampleContext(t *testing.T) Context {
	t.Helper()
	chainHash := mustDecode32(t, "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971")
	randomness := mustDecode32(t, "fe290beca10872ef2fb164d2aa4442de4566183ec51c56ff3cd603d930e54fdd")
	responseCommitment := sha256.Sum256([]byte("sympsi-response-commitment-000042"))
	return Context{
		ProtocolID:         "sympsi-aia-01-v0.1",
		CampaignID:         "campaign-0001",
		TrialID:            "trial-000042",
		ChainHash:          chainHash,
		SchemeID:           "bls-unchained-g1-rfc9380",
		ResponseCommitment: responseCommitment,
		Round:              1000,
		VerifiedRandomness: randomness,
	}
}

func TestQuicknetKnownAnswerMatchesRustTranscript(t *testing.T) {
	context := quicknetExampleContext(t)
	digest, err := Digest(context, 0)
	if err != nil {
		t.Fatal(err)
	}
	const expectedDigest = "cd4ed4b3a8ab1da6185ea3a535bd6b87b71862f606a9d03cc5a388e8b854bde6"
	if got := hex.EncodeToString(digest[:]); got != expectedDigest {
		t.Fatalf("digest mismatch: got %s want %s", got, expectedDigest)
	}

	for _, choices := range []int{4, 5} {
		got, err := TargetIndex(choices, context)
		if err != nil {
			t.Fatalf("%d-way target: %v", choices, err)
		}
		if got != 2 {
			t.Fatalf("%d-way target mismatch: got %d want 2", choices, got)
		}
	}
}

func TestRoundAndResponseCommitmentAreDomainSeparated(t *testing.T) {
	context := quicknetExampleContext(t)
	first, err := Digest(context, 0)
	if err != nil {
		t.Fatal(err)
	}
	secondContext := context
	secondContext.Round++
	second, err := Digest(secondContext, 0)
	if err != nil {
		t.Fatal(err)
	}
	if first == second {
		t.Fatal("changing round did not change transcript digest")
	}

	thirdContext := context
	thirdContext.ResponseCommitment[0] ^= 0x01
	third, err := Digest(thirdContext, 0)
	if err != nil {
		t.Fatal(err)
	}
	if first == third {
		t.Fatal("changing response commitment did not change transcript digest")
	}
}

func TestRejectionSamplingBoundary(t *testing.T) {
	var zero [32]byte
	if isBelowSmallInteger(zero, powMod256(4)) {
		t.Fatal("four-way zero digest must be accepted because 2^256 mod 4 = 0")
	}
	if !isBelowSmallInteger(zero, powMod256(5)) {
		t.Fatal("five-way zero digest must be rejected because 2^256 mod 5 = 1")
	}
}

func TestUnsupportedChoiceCountRejected(t *testing.T) {
	context := quicknetExampleContext(t)
	if _, err := TargetIndex(3, context); err == nil {
		t.Fatal("unsupported three-way mapping unexpectedly passed")
	}
}
