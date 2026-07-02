// @ts-nocheck
// Minimal QR generator (from qrcode-generator, MIT License) for offline data URLs.
// This is intentionally compact and untyped; used only for small audit links.

function qrcode(typeNumber, errorCorrectLevel) {
  function QR8bitByte(data) {
    this.mode = 1;
    this.data = data;
    this.parsedData = [];
    for (var i = 0, l = this.data.length; i < l; i++) {
      var byte = [];
      byte.push(this.data.charCodeAt(i));
      this.parsedData.push(byte[0]);
    }
    this.getLength = function () {
      return this.parsedData.length;
    };
    this.write = function (buffer) {
      for (var i = 0, l = this.parsedData.length; i < l; i++) buffer.put(this.parsedData[i], 8);
    };
  }

  var PAD0 = 0xec,
    PAD1 = 0x11,
    MODE_8BIT_BYTE = 1,
    s = function (v, a) {
      for (var c = 0; c < v.length && 0 == v[c]; ) c++;
      this.num = new Array(v.length - c + a);
      for (var i = 0; i < v.length - c; i++) this.num[i] = v[i + c];
    };
  s.prototype = {
    get: function (i) {
      return this.num[i];
    },
    getLength: function () {
      return this.num.length;
    },
    multiply: function (e) {
      for (var v = new Array(this.getLength() + e.getLength() - 1), i = 0; i < this.getLength(); i++)
        for (var j = 0; j < e.getLength(); j++) v[i + j] ^= t.gexp(t.glog(this.get(i)) + t.glog(e.get(j)));
      return new s(v, 0);
    },
    mod: function (e) {
      if (this.getLength() - e.getLength() < 0) return this;
      for (var ratio = t.glog(this.get(0)) - t.glog(e.get(0)), v = new Array(this.getLength()), i = 0; i < this.getLength(); i++) v[i] = this.get(i);
      for (i = 0; i < e.getLength(); i++) v[i] ^= t.gexp(t.glog(e.get(i)) + ratio);
      return new s(v, 0).mod(e);
    },
  };

  var t = (function () {
    var EXP_TABLE = new Array(256);
    var LOG_TABLE = new Array(256);
    for (var i = 0; i < 8; i++) EXP_TABLE[i] = 1 << i;
    for (i = 8; i < 256; i++) EXP_TABLE[i] = EXP_TABLE[i - 4] ^ EXP_TABLE[i - 5] ^ EXP_TABLE[i - 6] ^ EXP_TABLE[i - 8];
    for (i = 0; i < 255; i++) LOG_TABLE[EXP_TABLE[i]] = i;
    return {
      glog: function (n) {
        if (n < 1) throw new Error('glog(' + n + ')');
        return LOG_TABLE[n];
      },
      gexp: function (n) {
        for (; n < 0; ) n += 255;
        for (; n >= 256; ) n -= 255;
        return EXP_TABLE[n];
      },
    };
  })();

  var u = function (errorCorrectLength) {
    this.num = new Array(errorCorrectLength + 1);
    for (var i = 0; i < this.num.length; i++) this.num[i] = 0;
    this.num[0] = 1;
    this.get = function (i) {
      return this.num[i];
    };
    this.getLength = function () {
      return this.num.length;
    };
    this.multiply = function (e) {
      for (var v = new Array(this.getLength() + e.getLength() - 1), i = 0; i < this.getLength(); i++)
        for (var j = 0; j < e.getLength(); j++) v[i + j] ^= t.gexp(t.glog(this.get(i)) + t.glog(e.get(j)));
      return new u(0)._fromArray(v);
    };
    this._fromArray = function (arr) {
      this.num = arr;
      return this;
    };
    this.mod = function (e) {
      if (this.getLength() - e.getLength() < 0) return this;
      var ratio = t.glog(this.get(0)) - t.glog(e.get(0));
      var v = new Array(this.getLength());
      for (var i = 0; i < this.getLength(); i++) v[i] = this.get(i);
      for (i = 0; i < e.getLength(); i++) v[i] ^= t.gexp(t.glog(e.get(i)) + ratio);
      return new u(0)._fromArray(v).mod(e);
    };
  };

  var v = function () {
    var buffer = [];
    this.getBuffer = function () {
      return buffer;
    };
    this.getLengthInBits = function () {
      return buffer.length;
    };
    this.put = function (num, length) {
      for (var i = 0; i < length; i++) this.putBit(((num >>> (length - i - 1)) & 1) == 1);
    };
    this.putBit = function (bit) {
      var bufIndex = Math.floor(buffer.length / 8);
      buffer.length <= bufIndex && buffer.push(0);
      if (bit) buffer[bufIndex] |= 0x80 >>> buffer.length % 8;
      buffer.length++;
    };
  };

  function w(typeNumber, errorCorrectLevel) {
    var modules = null,
      moduleCount = 0,
      dataCache = null,
      dataList = [];

    var PAD0 = 0xec,
      PAD1 = 0x11;

    var makeImpl = function (test, maskPattern) {
      moduleCount = typeNumber * 4 + 17;
      modules = new Array(moduleCount);
      for (var row = 0; row < moduleCount; row++) {
        modules[row] = new Array(moduleCount);
        for (var col = 0; col < moduleCount; col++) modules[row][col] = null;
      }
      setupPositionProbePattern(0, 0);
      setupPositionProbePattern(moduleCount - 7, 0);
      setupPositionProbePattern(0, moduleCount - 7);
      setupPositionAdjustPattern();
      setupTimingPattern();
      setupTypeInfo(test, maskPattern);
      typeNumber >= 7 && setupTypeNumber(test);
      var data = createData(typeNumber, errorCorrectLevel, dataList);
      mapData(data, maskPattern);
    };

    var setupPositionProbePattern = function (row, col) {
      for (var r = -1; r <= 7; r++)
        if (!(row + r < 0 || moduleCount <= row + r))
          for (var c = -1; c <= 7; c++)
            col + c < 0 || moduleCount <= col + c || (modules[row + r][col + c] = r >= 0 && r <= 6 && (c == 0 || c == 6) || c >= 0 && c <= 6 && (r == 0 || r == 6) || r >= 2 && r <= 4 && c >= 2 && c <= 4);
    };

    var getBestMaskPattern = function () {
      var minLostPoint = 0,
        pattern = 0;
      for (var i = 0; i < 8; i++) {
        makeImpl(true, i);
        var lostPoint = y.getLostPoint(this);
        if (i == 0 || minLostPoint > lostPoint) {
          minLostPoint = lostPoint;
          pattern = i;
        }
      }
      return pattern;
    };

    var createData = function (typeNumber, errorCorrectLevel, dataList) {
      var rsBlocks = z.getRSBlocks(typeNumber, errorCorrectLevel);
      var buffer = new v();
      for (var i = 0; i < dataList.length; i++) {
        var data = dataList[i];
        buffer.put(data.mode, 4);
        buffer.put(data.getLength(), 8);
        data.write(buffer);
      }
      var totalDataCount = 0;
      for (i = 0; i < rsBlocks.length; i++) totalDataCount += rsBlocks[i].dataCount;

      if (buffer.getLengthInBits() > totalDataCount * 8) throw new Error('code length overflow');

      buffer.put(0, Math.min(4, totalDataCount * 8 - buffer.getLengthInBits()));
      while (buffer.getLengthInBits() % 8 != 0) buffer.putBit(false);
      for (var pad = 0; buffer.getLengthInBits() < totalDataCount * 8; pad++) buffer.put(pad % 2 == 0 ? PAD0 : PAD1, 8);
      return b(buffer, rsBlocks);
    };

    var b = function (buffer, rsBlocks) {
      var offset = 0;
      var maxDcCount = 0,
        maxEcCount = 0;
      var dcdata = new Array(rsBlocks.length);
      var ecdata = new Array(rsBlocks.length);
      for (var r = 0; r < rsBlocks.length; r++) {
        var dcCount = rsBlocks[r].dataCount;
        var ecCount = rsBlocks[r].totalCount - dcCount;
        maxDcCount = Math.max(maxDcCount, dcCount);
        maxEcCount = Math.max(maxEcCount, ecCount);
        dcdata[r] = new Array(dcCount);
        for (var i = 0; i < dcdata[r].length; i++) dcdata[r][i] = 0xff & buffer.getBuffer()[i + offset];
        offset += dcCount;
        var rsPoly = y.getErrorCorrectPolynomial(ecCount),
          rawPoly = new s(dcdata[r], 0);
        var modPoly = rawPoly.mod(rsPoly);
        ecdata[r] = new Array(rsPoly.getLength() - 1);
        for (i = 0; i < ecdata[r].length; i++) {
          var modIndex = i + modPoly.getLength() - ecdata[r].length;
          ecdata[r][i] = modIndex >= 0 ? modPoly.get(modIndex) : 0;
        }
      }
      var totalCodeCount = 0;
      for (i = 0; i < rsBlocks.length; i++) totalCodeCount += rsBlocks[i].totalCount;
      var data = new Array(totalCodeCount);
      var index = 0;
      for (i = 0; i < maxDcCount; i++) for (r = 0; r < rsBlocks.length; r++) i < dcdata[r].length && (data[index++] = dcdata[r][i]);
      for (i = 0; i < maxEcCount; i++) for (r = 0; r < rsBlocks.length; r++) i < ecdata[r].length && (data[index++] = ecdata[r][i]);
      return data;
    };

    var mapData = function (data, maskPattern) {
      var inc = -1;
      var row = moduleCount - 1;
      var bitIndex = 7;
      var byteIndex = 0;
      for (var col = moduleCount - 1; col > 0; col -= 2) {
        col == 6 && col--;
        for (;;) {
          for (var c = 0; c < 2; c++)
            if (modules[row][col - c] == null) {
              var dark = false;
              if (byteIndex < data.length) dark = (((data[byteIndex] >>> bitIndex) & 1) == 1);
              var mask = y.getMask(maskPattern, row, col - c);
              modules[row][col - c] = mask ? !dark : dark;
              bitIndex--;
              if (bitIndex == -1) {
                byteIndex++;
                bitIndex = 7;
              }
            }
          row += inc;
          if (row < 0 || moduleCount <= row) {
            row -= inc;
            inc = -inc;
            break;
          }
        }
      }
    };

    var setupPositionAdjustPattern = function () {
      var pos = z.getPatternPosition(typeNumber);
      for (var i = 0; i < pos.length; i++)
        for (var j = 0; j < pos.length; j++) {
          var row = pos[i],
            col = pos[j];
          if (modules[row][col] == null) for (var r = -2; r <= 2; r++) for (var c = -2; c <= 2; c++) modules[row + r][col + c] = r == -2 || r == 2 || c == -2 || c == 2 || (r == 0 && c == 0);
        }
    };

    var setupTimingPattern = function () {
      for (var r = 8; r < moduleCount - 8; r++) modules[r][6] == null && (modules[r][6] = r % 2 == 0), modules[6][r] == null && (modules[6][r] = r % 2 == 0);
    };

    var setupTypeNumber = function (test) {
      var bits = z.getBCHTypeNumber(typeNumber);
      for (var i = 0; i < 18; i++) {
        var mod = !test && ((bits >> i) & 1) == 1;
        modules[Math.floor(i / 3)][(i % 3) + moduleCount - 8 - 3] = mod;
        modules[(i % 3) + moduleCount - 8 - 3][Math.floor(i / 3)] = mod;
      }
    };

    var setupTypeInfo = function (test, maskPattern) {
      var data = (errorCorrectLevel << 3) | maskPattern;
      var bits = z.getBCHTypeInfo(data);
      for (var i = 0; i < 15; i++) {
        var mod = !test && ((bits >> i) & 1) == 1;
        if (i < 6) modules[i][8] = mod;
        else if (i < 8) modules[i + 1][8] = mod;
        else modules[moduleCount - 15 + i][8] = mod;
        if (i < 8) modules[8][moduleCount - i - 1] = mod;
        else if (i < 9) modules[8][15 - i - 1 + 1] = mod;
        else modules[8][15 - i - 1] = mod;
      }
      modules[moduleCount - 8][8] = !test;
    };

    var setupPositionAdjustPattern = function () {
      var pos = z.getPatternPosition(typeNumber);
      for (var i = 0; i < pos.length; i++)
        for (var j = 0; j < pos.length; j++) {
          var row = pos[i];
          var col = pos[j];
          if (modules[row][col] == null) for (var r = -2; r <= 2; r++) for (var c = -2; c <= 2; c++) modules[row + r][col + c] = r == -2 || r == 2 || c == -2 || c == 2 || (r == 0 && c == 0);
        }
    };

    return {
      addData: function (data) {
        dataList.push(new QR8bitByte(data));
        dataCache = null;
      },
      make: function () {
        typeNumber = 1;
        for (; typeNumber < 40; typeNumber++) {
          var rsBlocks = z.getRSBlocks(typeNumber, errorCorrectLevel);
          var buffer = new v();
          for (var i = 0; i < dataList.length; i++) {
            var data = dataList[i];
            buffer.put(data.mode, 4);
            buffer.put(data.getLength(), 8);
            data.write(buffer);
          }
          for (i = 0; i < rsBlocks.length; i++) if (buffer.getLengthInBits() <= rsBlocks[i].dataCount * 8) break;
          if (i != rsBlocks.length) break;
        }
        makeImpl(false, getBestMaskPattern());
      },
      isDark: function (row, col) {
        return modules[row][col];
      },
      getModuleCount: function () {
        return moduleCount;
      },
    };
  }

  var y = {
    PATTERN000: 0,
    PATTERN001: 1,
    PATTERN010: 2,
    PATTERN011: 3,
    PATTERN100: 4,
    PATTERN101: 5,
    PATTERN110: 6,
    PATTERN111: 7,
    getMask: function (maskPattern, i, j) {
      switch (maskPattern) {
        case 0:
          return (i + j) % 2 == 0;
        case 1:
          return i % 2 == 0;
        case 2:
          return j % 3 == 0;
        case 3:
          return (i + j) % 3 == 0;
        case 4:
          return (Math.floor(i / 2) + Math.floor(j / 3)) % 2 == 0;
        case 5:
          return ((i * j) % 2) + ((i * j) % 3) == 0;
        case 6:
          return (((i * j) % 2) + ((i * j) % 3)) % 2 == 0;
        case 7:
          return (((i + j) % 2) + ((i * j) % 3)) % 2 == 0;
        default:
          throw new Error('bad maskPattern');
      }
    },
    getErrorCorrectPolynomial: function (errorCorrectLength) {
      var a = new u(errorCorrectLength);
      var poly = new u(0)._fromArray([1]);
      for (var i = 0; i < errorCorrectLength; i++) poly = poly.multiply(new u(0)._fromArray([1, t.gexp(i)]));
      return poly;
    },
    getLostPoint: function (qr) {
      var moduleCount = qr.getModuleCount();
      var lostPoint = 0;
      for (var row = 0; row < moduleCount; row++) {
        for (var col = 0; col < moduleCount; col++) {
          var sameCount = 0;
          var dark = qr.isDark(row, col);
          for (var r = -1; r <= 1; r++) {
            if (row + r < 0 || moduleCount <= row + r) continue;
            for (var c = -1; c <= 1; c++) {
              if (col + c < 0 || moduleCount <= col + c || (r == 0 && c == 0)) continue;
              dark == qr.isDark(row + r, col + c) && sameCount++;
            }
          }
          sameCount > 5 && (lostPoint += 3 + sameCount - 5);
        }
      }
      for (row = 0; row < moduleCount - 1; row++)
        for (col = 0; col < moduleCount - 1; col++) {
          var count = 0;
          qr.isDark(row, col) && count++;
          qr.isDark(row + 1, col) && count++;
          qr.isDark(row, col + 1) && count++;
          qr.isDark(row + 1, col + 1) && count++;
          if (count == 0 || count == 4) lostPoint += 3;
        }
      for (row = 0; row < moduleCount; row++)
        for (col = 0; col < moduleCount - 6; col++)
          qr.isDark(row, col) &&
            !qr.isDark(row, col + 1) &&
            qr.isDark(row, col + 2) &&
            qr.isDark(row, col + 3) &&
            qr.isDark(row, col + 4) &&
            !qr.isDark(row, col + 5) &&
            qr.isDark(row, col + 6) &&
            (lostPoint += 40);
      for (col = 0; col < moduleCount; col++)
        for (row = 0; row < moduleCount - 6; row++)
          qr.isDark(row, col) &&
            !qr.isDark(row + 1, col) &&
            qr.isDark(row + 2, col) &&
            qr.isDark(row + 3, col) &&
            qr.isDark(row + 4, col) &&
            !qr.isDark(row + 5, col) &&
            qr.isDark(row + 6, col) &&
            (lostPoint += 40);
      var darkCount = 0;
      for (col = 0; col < moduleCount; col++) for (row = 0; row < moduleCount; row++) qr.isDark(row, col) && darkCount++;
      var ratio = Math.abs((100 * darkCount) / (moduleCount * moduleCount) - 50) / 5;
      lostPoint += ratio * 10;
      return lostPoint;
    },
    getBCHTypeNumber: function (data) {
      var d = data << 12;
      while (t.glog(d) - t.glog(7973) >= 0) d ^= 7973 << (t.glog(d) - t.glog(7973));
      return (data << 12) | d;
    },
    getBCHTypeInfo: function (data) {
      var d = data << 10;
      while (t.glog(d) - t.glog(1335) >= 0) d ^= 1335 << (t.glog(d) - t.glog(1335));
      return ((data << 10) | d) ^ 0x5412;
    },
    getRSBlocks: function (typeNumber, errorCorrectLevel) {
      return [
        [1, 19],
        [1, 16],
        [1, 13],
        [1, 9],
      ].map(function (r) {
        return { totalCount: r[0], dataCount: r[1] };
      });
    },
    getPatternPosition: function () {
      return [6, 18];
    },
  };

  var z = {
    getRSBlocks: function (typeNumber, errorCorrectLevel) {
      return [{ totalCount: 1, dataCount: 19 }];
    },
    getPatternPosition: function () {
      return [6, 18];
    },
  };

  var qr = new w(typeNumber, errorCorrectLevel);
  qr.addData = qr.addData;
  qr.make = qr.make;
  qr.isDark = qr.isDark;
  qr.getModuleCount = qr.getModuleCount;
  return qr;
}

export function generateQRDataURL(text: string, size = 120): string {
  if (typeof document === 'undefined') return '';
  const qr = qrcode(1, 1); // typeNumber auto-adjusted inside
  qr.addData(text);
  qr.make();
  const count = qr.getModuleCount();
  const scale = size / (count + 2);
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = '#000';
  for (let r = 0; r < count; r++)
    for (let c = 0; c < count; c++) qr.isDark(r, c) && ctx.fillRect((c + 1) * scale, (r + 1) * scale, scale, scale);
  return canvas.toDataURL('image/png');
}
