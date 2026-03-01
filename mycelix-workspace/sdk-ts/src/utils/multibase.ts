/**
 * Minimal multibase/base58btc helpers for decoding tagged signatures/keys.
 *
 * Supports multibase prefix `z` (base58btc) only.
 */

const BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
const BASE58_MAP = (() => {
  const map = new Uint8Array(128);
  map.fill(0xff);
  for (let i = 0; i < BASE58_ALPHABET.length; i += 1) {
    map[BASE58_ALPHABET.charCodeAt(i)] = i;
  }
  return map;
})();

/**
 * Decode a base58btc-encoded string into bytes.
 */
export function decodeBase58Btc(input: string): Uint8Array {
  if (input.length === 0) {
    return new Uint8Array();
  }

  const bytes: number[] = [0];

  for (let i = 0; i < input.length; i += 1) {
    const code = input.charCodeAt(i);
    if (code > 127) {
      throw new Error(`Invalid base58 character code: ${code}`);
    }
    const value = BASE58_MAP[code];
    if (value === 0xff) {
      throw new Error(`Invalid base58 character: ${input[i]}`);
    }

    let carry = value;
    for (let j = 0; j < bytes.length; j += 1) {
      const total = bytes[j] * 58 + carry;
      bytes[j] = total & 0xff;
      carry = total >> 8;
    }
    while (carry > 0) {
      bytes.push(carry & 0xff);
      carry >>= 8;
    }
  }

  // Handle leading zeros (represented as leading '1's in base58)
  let leadingZeros = 0;
  while (leadingZeros < input.length && input[leadingZeros] === '1') {
    leadingZeros += 1;
  }

  const result = new Uint8Array(leadingZeros + bytes.length);
  for (let i = 0; i < bytes.length; i += 1) {
    result[result.length - 1 - i] = bytes[i];
  }

  return result;
}

/**
 * Decode a multibase string with `z` prefix (base58btc).
 */
export function decodeMultibaseBase58Btc(input: string): Uint8Array {
  if (!input.startsWith('z')) {
    throw new Error('Unsupported multibase prefix (expected "z")');
  }
  return decodeBase58Btc(input.slice(1));
}

/**
 * Extract the 2-byte multicodec prefix from a multibase string.
 */
export function multicodecPrefixFromMultibase(input: string): [number, number] {
  const bytes = decodeMultibaseBase58Btc(input);
  if (bytes.length < 2) {
    throw new Error('Multibase payload too short for multicodec prefix');
  }
  return [bytes[0], bytes[1]];
}
