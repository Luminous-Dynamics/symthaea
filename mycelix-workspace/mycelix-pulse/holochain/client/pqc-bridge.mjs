import { AdminWebsocket, AppWebsocket } from '@holochain/client';
import { MlKem1024 } from 'mlkem';

window.HolochainClient = { AdminWebsocket, AppWebsocket };

// Expose PQC key exchange for Pulse encryption
window.PQCrypto = {
    // Generate a ML-KEM-1024 keypair
    async generateKeypair() {
        const kem = new MlKem1024();
        const [pk, sk] = await kem.generateKeyPair();
        return { publicKey: pk, secretKey: sk };
    },

    // Encapsulate a shared secret with recipient's public key
    async encapsulate(publicKey) {
        const kem = new MlKem1024();
        const [ciphertext, sharedSecret] = await kem.encap(publicKey);
        return { ciphertext, sharedSecret };
    },

    // Decapsulate using our secret key
    async decapsulate(ciphertext, secretKey) {
        const kem = new MlKem1024();
        const sharedSecret = await kem.decap(ciphertext, secretKey);
        return sharedSecret;
    },

    // Encrypt message with AES-256-GCM using shared secret
    async encrypt(plaintext, sharedSecret) {
        const key = await crypto.subtle.importKey(
            'raw', sharedSecret.slice(0, 32),
            { name: 'AES-GCM' }, false, ['encrypt']
        );
        const iv = crypto.getRandomValues(new Uint8Array(12));
        const encrypted = await crypto.subtle.encrypt(
            { name: 'AES-GCM', iv }, key,
            new TextEncoder().encode(plaintext)
        );
        return { ciphertext: new Uint8Array(encrypted), iv };
    },

    // Decrypt message with AES-256-GCM
    async decrypt(ciphertext, iv, sharedSecret) {
        const key = await crypto.subtle.importKey(
            'raw', sharedSecret.slice(0, 32),
            { name: 'AES-GCM' }, false, ['decrypt']
        );
        const decrypted = await crypto.subtle.decrypt(
            { name: 'AES-GCM', iv }, key, ciphertext
        );
        return new TextDecoder().decode(decrypted);
    },
};

console.log('[Pulse] PQC crypto (ML-KEM-1024 + AES-256-GCM) loaded');
