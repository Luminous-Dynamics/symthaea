/**
 * MD5 hash implementation for Gravatar URLs
 * This is a proper MD5 implementation suitable for Gravatar
 */
export function md5(str: string): string {
  function rotateLeft(value: number, shift: number): number {
    return (value << shift) | (value >>> (32 - shift));
  }

  function addUnsigned(x: number, y: number): number {
    const lsw = (x & 0xFFFF) + (y & 0xFFFF);
    const msw = (x >> 16) + (y >> 16) + (lsw >> 16);
    return (msw << 16) | (lsw & 0xFFFF);
  }

  function f(x: number, y: number, z: number): number {
    return (x & y) | (~x & z);
  }

  function g(x: number, y: number, z: number): number {
    return (x & z) | (y & ~z);
  }

  function h(x: number, y: number, z: number): number {
    return x ^ y ^ z;
  }

  function i(x: number, y: number, z: number): number {
    return y ^ (x | ~z);
  }

  function ff(a: number, b: number, c: number, d: number, x: number, s: number, ac: number): number {
    a = addUnsigned(a, addUnsigned(addUnsigned(f(b, c, d), x), ac));
    return addUnsigned(rotateLeft(a, s), b);
  }

  function gg(a: number, b: number, c: number, d: number, x: number, s: number, ac: number): number {
    a = addUnsigned(a, addUnsigned(addUnsigned(g(b, c, d), x), ac));
    return addUnsigned(rotateLeft(a, s), b);
  }

  function hh(a: number, b: number, c: number, d: number, x: number, s: number, ac: number): number {
    a = addUnsigned(a, addUnsigned(addUnsigned(h(b, c, d), x), ac));
    return addUnsigned(rotateLeft(a, s), b);
  }

  function ii(a: number, b: number, c: number, d: number, x: number, s: number, ac: number): number {
    a = addUnsigned(a, addUnsigned(addUnsigned(i(b, c, d), x), ac));
    return addUnsigned(rotateLeft(a, s), b);
  }

  function convertToWordArray(str: string): number[] {
    const wordArray: number[] = [];
    const strLen = str.length;

    for (let i = 0; i < strLen; i++) {
      const byteIndex = i >> 2;
      wordArray[byteIndex] = wordArray[byteIndex] || 0;
      wordArray[byteIndex] |= (str.charCodeAt(i) & 0xFF) << ((i % 4) * 8);
    }

    return wordArray;
  }

  function wordToHex(value: number): string {
    let hex = '';
    for (let i = 0; i < 4; i++) {
      const byte = (value >>> (i * 8)) & 0xFF;
      hex += byte.toString(16).padStart(2, '0');
    }
    return hex;
  }

  const x = convertToWordArray(str);
  let a = 0x67452301;
  let b = 0xEFCDAB89;
  let c = 0x98BADCFE;
  let d = 0x10325476;

  const msgLen = str.length;
  x[msgLen >> 2] |= 0x80 << ((msgLen % 4) * 8);
  x[(((msgLen + 8) >> 6) << 4) + 14] = msgLen * 8;

  for (let i = 0; i < x.length; i += 16) {
    const aa = a, bb = b, cc = c, dd = d;

    a = ff(a, b, c, d, x[i + 0], 7, 0xD76AA478);
    d = ff(d, a, b, c, x[i + 1], 12, 0xE8C7B756);
    c = ff(c, d, a, b, x[i + 2], 17, 0x242070DB);
    b = ff(b, c, d, a, x[i + 3], 22, 0xC1BDCEEE);
    a = ff(a, b, c, d, x[i + 4], 7, 0xF57C0FAF);
    d = ff(d, a, b, c, x[i + 5], 12, 0x4787C62A);
    c = ff(c, d, a, b, x[i + 6], 17, 0xA8304613);
    b = ff(b, c, d, a, x[i + 7], 22, 0xFD469501);
    a = ff(a, b, c, d, x[i + 8], 7, 0x698098D8);
    d = ff(d, a, b, c, x[i + 9], 12, 0x8B44F7AF);
    c = ff(c, d, a, b, x[i + 10], 17, 0xFFFF5BB1);
    b = ff(b, c, d, a, x[i + 11], 22, 0x895CD7BE);
    a = ff(a, b, c, d, x[i + 12], 7, 0x6B901122);
    d = ff(d, a, b, c, x[i + 13], 12, 0xFD987193);
    c = ff(c, d, a, b, x[i + 14], 17, 0xA679438E);
    b = ff(b, c, d, a, x[i + 15], 22, 0x49B40821);

    a = gg(a, b, c, d, x[i + 1], 5, 0xF61E2562);
    d = gg(d, a, b, c, x[i + 6], 9, 0xC040B340);
    c = gg(c, d, a, b, x[i + 11], 14, 0x265E5A51);
    b = gg(b, c, d, a, x[i + 0], 20, 0xE9B6C7AA);
    a = gg(a, b, c, d, x[i + 5], 5, 0xD62F105D);
    d = gg(d, a, b, c, x[i + 10], 9, 0x02441453);
    c = gg(c, d, a, b, x[i + 15], 14, 0xD8A1E681);
    b = gg(b, c, d, a, x[i + 4], 20, 0xE7D3FBC8);
    a = gg(a, b, c, d, x[i + 9], 5, 0x21E1CDE6);
    d = gg(d, a, b, c, x[i + 14], 9, 0xC33707D6);
    c = gg(c, d, a, b, x[i + 3], 14, 0xF4D50D87);
    b = gg(b, c, d, a, x[i + 8], 20, 0x455A14ED);
    a = gg(a, b, c, d, x[i + 13], 5, 0xA9E3E905);
    d = gg(d, a, b, c, x[i + 2], 9, 0xFCEFA3F8);
    c = gg(c, d, a, b, x[i + 7], 14, 0x676F02D9);
    b = gg(b, c, d, a, x[i + 12], 20, 0x8D2A4C8A);

    a = hh(a, b, c, d, x[i + 5], 4, 0xFFFA3942);
    d = hh(d, a, b, c, x[i + 8], 11, 0x8771F681);
    c = hh(c, d, a, b, x[i + 11], 16, 0x6D9D6122);
    b = hh(b, c, d, a, x[i + 14], 23, 0xFDE5380C);
    a = hh(a, b, c, d, x[i + 1], 4, 0xA4BEEA44);
    d = hh(d, a, b, c, x[i + 4], 11, 0x4BDECFA9);
    c = hh(c, d, a, b, x[i + 7], 16, 0xF6BB4B60);
    b = hh(b, c, d, a, x[i + 10], 23, 0xBEBFBC70);
    a = hh(a, b, c, d, x[i + 13], 4, 0x289B7EC6);
    d = hh(d, a, b, c, x[i + 0], 11, 0xEAA127FA);
    c = hh(c, d, a, b, x[i + 3], 16, 0xD4EF3085);
    b = hh(b, c, d, a, x[i + 6], 23, 0x04881D05);
    a = hh(a, b, c, d, x[i + 9], 4, 0xD9D4D039);
    d = hh(d, a, b, c, x[i + 12], 11, 0xE6DB99E5);
    c = hh(c, d, a, b, x[i + 15], 16, 0x1FA27CF8);
    b = hh(b, c, d, a, x[i + 2], 23, 0xC4AC5665);

    a = ii(a, b, c, d, x[i + 0], 6, 0xF4292244);
    d = ii(d, a, b, c, x[i + 7], 10, 0x432AFF97);
    c = ii(c, d, a, b, x[i + 14], 15, 0xAB9423A7);
    b = ii(b, c, d, a, x[i + 5], 21, 0xFC93A039);
    a = ii(a, b, c, d, x[i + 12], 6, 0x655B59C3);
    d = ii(d, a, b, c, x[i + 3], 10, 0x8F0CCC92);
    c = ii(c, d, a, b, x[i + 10], 15, 0xFFEFF47D);
    b = ii(b, c, d, a, x[i + 1], 21, 0x85845DD1);
    a = ii(a, b, c, d, x[i + 8], 6, 0x6FA87E4F);
    d = ii(d, a, b, c, x[i + 15], 10, 0xFE2CE6E0);
    c = ii(c, d, a, b, x[i + 6], 15, 0xA3014314);
    b = ii(b, c, d, a, x[i + 13], 21, 0x4E0811A1);
    a = ii(a, b, c, d, x[i + 4], 6, 0xF7537E82);
    d = ii(d, a, b, c, x[i + 11], 10, 0xBD3AF235);
    c = ii(c, d, a, b, x[i + 2], 15, 0x2AD7D2BB);
    b = ii(b, c, d, a, x[i + 9], 21, 0xEB86D391);

    a = addUnsigned(a, aa);
    b = addUnsigned(b, bb);
    c = addUnsigned(c, cc);
    d = addUnsigned(d, dd);
  }

  return wordToHex(a) + wordToHex(b) + wordToHex(c) + wordToHex(d);
}
