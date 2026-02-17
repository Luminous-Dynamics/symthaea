import { encrypt, decrypt } from '../encryption';

describe('Encryption Utils', () => {
  const originalText = 'sensitive-password-123';

  it('should encrypt and decrypt text correctly', () => {
    const encrypted = encrypt(originalText);
    const decrypted = decrypt(encrypted);

    expect(decrypted).toBe(originalText);
    expect(encrypted).not.toBe(originalText);
  });

  it('should produce different encrypted values for same input', () => {
    const encrypted1 = encrypt(originalText);
    const encrypted2 = encrypt(originalText);

    // Due to random salt and IV, encrypted values should be different
    expect(encrypted1).not.toBe(encrypted2);

    // But both should decrypt to the same value
    expect(decrypt(encrypted1)).toBe(originalText);
    expect(decrypt(encrypted2)).toBe(originalText);
  });

  it('should handle special characters', () => {
    const specialText = '!@#$%^&*()_+-=[]{}|;:,.<>?/~`';
    const encrypted = encrypt(specialText);
    const decrypted = decrypt(encrypted);

    expect(decrypted).toBe(specialText);
  });

  it('should handle unicode characters', () => {
    const unicodeText = '🔒 Secure Password 密碼 🔐';
    const encrypted = encrypt(unicodeText);
    const decrypted = decrypt(encrypted);

    expect(decrypted).toBe(unicodeText);
  });

  it('should handle empty string', () => {
    const encrypted = encrypt('');
    const decrypted = decrypt(encrypted);

    expect(decrypted).toBe('');
  });

  it('should throw error on invalid encrypted data', () => {
    expect(() => decrypt('invalid-base64-data')).toThrow();
  });
});
