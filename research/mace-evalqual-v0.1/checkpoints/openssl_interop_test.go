package checkpoints

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestPublishedCheckpointVectorVerifiesWithOpenSSL(t *testing.T) {
	openssl, err := exec.LookPath("openssl")
	if err != nil {
		t.Skip("OpenSSL CLI not installed; Go vector check remains active")
	}

	type vector struct {
		Checkpoint      Checkpoint `json:"checkpoint"`
		PublicKeyHex    string     `json:"public_key_hex"`
		SignatureHex    string     `json:"signature_hex"`
		SigningBytesHex string     `json:"signing_bytes_hex"`
	}
	wire, err := os.ReadFile("testdata/checkpoint-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var v vector
	if err := json.Unmarshal(wire, &v); err != nil {
		t.Fatal(err)
	}
	publicKey, err := hex.DecodeString(v.PublicKeyHex)
	if err != nil {
		t.Fatal(err)
	}
	signature, err := hex.DecodeString(v.SignatureHex)
	if err != nil {
		t.Fatal(err)
	}
	message, err := v.Checkpoint.SigningBytes()
	if err != nil {
		t.Fatal(err)
	}
	wantMessage, err := hex.DecodeString(v.SigningBytesHex)
	if err != nil {
		t.Fatal(err)
	}
	if string(message) != string(wantMessage) {
		t.Fatal("local signing preimage differs from published vector")
	}

	// Ed25519 SubjectPublicKeyInfo DER prefix from RFC 8410, section 10.3.
	spkiPrefix, err := hex.DecodeString("302a300506032b6570032100")
	if err != nil {
		t.Fatal(err)
	}
	publicKeyDER := append(spkiPrefix, publicKey...)
	dir := t.TempDir()
	publicKeyPath := filepath.Join(dir, "witness-public.der")
	messagePath := filepath.Join(dir, "signing-bytes.bin")
	signaturePath := filepath.Join(dir, "signature.bin")
	for path, data := range map[string][]byte{
		publicKeyPath: publicKeyDER,
		messagePath:   message,
		signaturePath: signature,
	} {
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}

	cmd := exec.Command(openssl, "pkeyutl", "-verify", "-pubin", "-inkey", publicKeyPath, "-keyform", "DER", "-rawin", "-in", messagePath, "-sigfile", signaturePath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("independent OpenSSL Ed25519 verification failed: %v: %s", err, output)
	}
}
